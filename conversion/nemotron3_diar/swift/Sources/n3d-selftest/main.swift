// n3d-selftest — headless gate of the Swift host (NemotronDiarizer) against the Python side; the golden
// files come from ../../export_golden.py (_work/golden/). Parts, combinable in one call:
//
//   closed loop  --assets <dir> --bundle <.aimodel|.aimodelc> --wav <16 kHz mono> --mode ll|vll|ull|offline
//                [--golden <fixture>_<tag>_probs.f32le]   agreement@0.5 / max|Δp| / segments (bar 99.9 %);
//                                                        max|Δlogit| too when <..>_logits.f32le sits beside it
//                [--unit gpu|cpuOnly|ane|default] [--poison no-compress|no-pop]
//                [--chunks <golden dir>]                  feed the Python host's chunk rows instead of the
//                                                        Swift mel (separates the mel from the loop)
//                [--compare-logits <f32le>]               another run's logits (e.g. golden/pygpu/...)
//                [--dump-packed <dir>] [--dump-steps 0,1,54,78,128,last]  the packed rows [L, 512] per step
//                [--packed-golden <path prefix>]          compare them with <prefix><step>.f32le (bar 1e-3)
//                [--bench N] [--lock <path>]              warmup 5 + N graph calls, then a timed whole run
//                [--json <path>]
//   cache units  --assets <dir> --cache-golden <golden dir>   teacher-forced N3DSpeakerCache.update vs the
//                                                            NumPy host, bit for bit
//   mel          --assets <dir> --wav <path> --mel-golden <golden dir> [--fixture <name>]
//
// Exit 0 = every requested gate passed, 3 = a gate failed, 4 = an error.

import Darwin
import Foundation
import N3DGateSupport
import NemotronDiarizer

setvbuf(stdout, nil, _IONBF, 0)

func log(_ s: String) { print("[n3d] \(s)") }

// MARK: - cache units (teacher-forced)

func cacheCheck(assets: N3DAssets, golden: URL) throws -> (Bool, [String: Any]) {
    guard let units = try readJSON(golden.appendingPathComponent("cache/index.json")) as? [[String: Any]] else {
        throw SelfTestError.golden("cache/index.json is not a list")
    }
    let H = N3DSpeakerCache.hidden, S = N3DSpeakerCache.numSpeakers
    var pooledExact = 0, scoresExact = 0, scoresTotal = 0, afterExact = 0
    var failures: [String] = []
    for u in units {
        let prefix = u["prefix"] as! String
        func f(_ suffix: String) throws -> [Float] { try readF32(golden.appendingPathComponent("\(prefix)_\(suffix).f32le")) }
        let L = u["L"] as! Int, nCache = u["n_cache"] as! Int, nFifo = u["n_fifo"] as! Int, nChunk = u["n_chunk"] as! Int
        let rows = try f("rows"), logits = try f("logits"), probsBefore = try f("probs_before"), pooledRef = try f("pooled")
        var cache = N3DSpeakerCache(fifoLength: u["fifo_length"] as! Int, updatePeriod: u["update_period"] as! Int,
                                    embeds: Array(rows[0..<(nCache * H)]), probs: probsBefore,
                                    fifo: Array(rows[(nCache * H)..<((nCache + nFifo) * H)]),
                                    isCompressed: u["compressed_before"] as! Bool)
        let pooled = N3DSpeakerCache.poolProbs(logits: logits, frames: L)
        let pooledOK = pooled.map(\.bitPattern) == pooledRef.map(\.bitPattern)
        if pooledOK { pooledExact += 1 } else { failures.append("\(prefix): pooled probs differ (\(maxAbsDiff(pooled[...], pooledRef[...])))") }
        if u["compressed_now"] as! Bool {
            scoresTotal += 1
            // the probabilities compress() scored: stored + the popped FIFO rows' (from the NumPy pooled probs)
            let stored = (u["compressed_before"] as! Bool) ? probsBefore : Array(pooledRef[0..<(nCache * S)])
            let n = nFifo + nChunk
            let pop = cache.numPopped(n)
            let cacheP = stored + pooledRef[(nCache * S)..<((nCache + n) * S)].prefix(pop * S)
            let ref = try readF64(golden.appendingPathComponent("\(prefix)_scores.f64le"))
            let mine = cache.frameScores(probs: cacheP, frames: cacheP.count / S)
            if mine.map(\.bitPattern) == ref.map(\.bitPattern) { scoresExact += 1 } else {
                let diff = zip(mine, ref).filter { $0.bitPattern != $1.bitPattern }.count
                failures.append("\(prefix): float64 scores differ in \(diff) of \(ref.count)")
            }
        }
        cache.update(rows: rows, logits: logits, frames: L, silence: assets.silence, chunkFrames: nChunk)
        let eA = try f("embeds_after"), pA = try f("probs_after"), fA = try f("fifo_after")
        let same = cache.embeds.map(\.bitPattern) == eA.map(\.bitPattern) && cache.probs.map(\.bitPattern) == pA.map(\.bitPattern)
            && cache.fifo.map(\.bitPattern) == fA.map(\.bitPattern) && cache.isCompressed == (u["compressed_after"] as! Bool)
        if same { afterExact += 1 } else {
            failures.append("\(prefix): state after differs (cache \(cache.cacheFrames) vs \(eA.count / H), fifo \(cache.fifoFrames) vs \(fA.count / H))")
        }
    }
    let pass = failures.isEmpty
    log("cache units: \(units.count) teacher-forced updates (\(scoresTotal) compressions): pooled probs bit-exact \(pooledExact)/\(units.count), "
        + "float64 scores bit-exact \(scoresExact)/\(scoresTotal), state after bit-exact \(afterExact)/\(units.count) -> \(pass ? "PASS" : "FAIL")")
    for s in failures.prefix(10) { log("  \(s)") }
    return (pass, ["units": units.count, "compressions": scoresTotal, "pooled_exact": pooledExact,
                   "scores_exact": scoresExact, "state_after_exact": afterExact, "failures": failures, "pass": pass])
}

// MARK: - mel diagnostics

func melCheck(assets: N3DAssets, samples: [Float], fixture: String, golden: URL) throws -> (Bool, [String: Any]) {
    let mel = N3DMel(melFilters: assets.melFilters, hannWindow: assets.hannWindow)
    let embedder = N3DEmbedder(projection: assets.projection)
    let chunks = N3DMel.streamChunks(samples: samples.count, chunkFrames: 9, lookaheadFrames: 4)
    var rows: [[String: Any]] = []
    var worst = 0.0, over = 0, cells = 0, sameFrames = true
    func check(_ name: String, _ m: [Float], _ frames: Int, _ file: String) throws {
        let ref = try readF32(golden.appendingPathComponent(file))
        let refFrames = ref.count / N3DMel.nMels
        sameFrames = sameFrames && refFrames == frames
        let n = min(ref.count, m.count)
        let d = maxAbsDiff(m[0..<n], ref[0..<n])
        let o = zip(m[0..<n], ref[0..<n]).filter { abs(Double($0) - Double($1)) > 1e-4 }.count
        let e1 = embedder.embed(mel: m, frames: frames).embeds, e2 = embedder.embed(mel: ref, frames: refFrames).embeds
        let de = maxAbsDiff(e1[...], e2[...]).max
        worst = max(worst, d.max); over += o; cells += n
        rows.append(["part": name, "frames": frames, "frames_ref": refFrames, "max_abs": d.max, "unequal": d.unequal,
                     "over_1e-4": o, "cells": n, "embed_max_abs": de])
        log("mel \(name): frames \(frames)/\(refFrames), max|Δ| \(fmt(d.max)) (> 1e-4: \(o) of \(n)), bit-equal \(n - d.unequal)/\(n); embed max|Δ| \(fmt(de))")
    }
    for k in [0, 1, chunks.count - 1] {
        let c = chunks[k]
        let (m, F) = mel.logMel(samples[min(c.start, samples.count)..<min(c.end, samples.count)], center: c.isFirst)
        try check("\(fixture) ll chunk \(k)", m, F, "\(fixture)_ll_mel_chunk\(k).f32le")
    }
    let (m, F) = samples.withUnsafeBufferPointer { mel.logMel($0, center: true) }
    try check("\(fixture) offline", m, F, "\(fixture)_offline_mel.f32le")
    // the round-2 mel bar, here against the NumPy mirror: max <= 5e-4 and cells over 1e-4 < 0.1 %
    let pass = sameFrames && worst <= 5e-4 && Double(over) < 0.001 * Double(cells)
    log("mel vs NumPy mirror: max|Δ| \(fmt(worst)), > 1e-4 in \(over) of \(cells) cells -> \(pass ? "PASS" : "FAIL")")
    return (pass, ["parts": rows, "max_abs": worst, "over_1e-4": over, "cells": cells, "pass": pass])
}

// MARK: - closed loop

struct Capture {
    var rows: [Int: [Float]] = [:]
    var lastStep = -1
    var lastRows: [Float] = []
    var trace: [[Int]] = []          // step, L, cache, fifo, compressed after, cache after, fifo after
}

func closedLoop(args: Args, assets: N3DAssets, samples: [Float]) async throws -> (Bool, [String: Any]) {
    let tag = args["mode"] ?? "ll"
    let profile: N3DProfile
    if tag == "offline" { profile = .offline } else {
        guard let m = N3DStreamingMode(rawValue: tag) else { throw SelfTestError.usage("--mode ll|vll|ull|offline") }
        profile = .streamingProfile(mode: m)
    }
    guard let unit = N3DComputeUnits(rawValue: args["unit"] ?? "gpu") else {
        throw SelfTestError.usage("--unit gpu|cpuOnly|ane|default")
    }
    let bundle = args.url("bundle")!
    let diarizer = try await N3DDiarizer(assets: assets, computeUnits: unit, profile: profile, modelURL: bundle)
    log("loaded \(bundle.lastPathComponent) (\(unit.rawValue)) in \(String(format: "%.2f", diarizer.loadSeconds)) s; T=\(diarizer.graphLength), mode \(tag)")
    var j: [String: Any] = ["bundle": bundle.path, "unit": unit.rawValue, "mode": tag, "load_s": diarizer.loadSeconds]
    var pass = true
    if let p = args["poison"] {
        guard let poison = N3DSpeakerCache.Poison(rawValue: p) else { throw SelfTestError.usage("--poison no-compress|no-pop") }
        await diarizer.setPoison(poison)
        j["poison"] = p
        log("poison \(p) (negative control: the gate must fail)")
    }

    let nSteps = stepCount(samples: samples.count, profile: profile)
    var wanted = Set<Int>()
    var dumpList: [Int] = []
    if args.has("dump-packed") {
        let spec = args["dump-steps"] ?? "0,1,54,78,128,last"
        dumpList = spec.split(separator: ",").compactMap { $0 == "last" ? nSteps - 1 : Int($0) }.filter { $0 < nSteps }
        wanted.formUnion(dumpList)
    }
    let benchN = args.int("bench") ?? 0
    var benchSteps: [Int] = []
    if benchN > 0 {
        benchSteps = (0..<5).map { $0 % nSteps } + (0..<benchN).map { ($0 * 7) % nSteps }
        wanted.formUnion(benchSteps)
    }
    let box = Box(Capture())
    let wantedSteps = wanted
    let onStep: @Sendable (N3DStepInfo) -> Void = { info in
        box.mutate { c in
            if wantedSteps.contains(info.step) { c.rows[info.step] = info.rows }
            c.lastStep = info.step
            c.lastRows = info.rows
            c.trace.append([info.step, info.length, info.cacheFrames, info.fifoFrames, info.compressedAfter ? 1 : 0,
                            info.cacheFramesAfter, info.fifoFramesAfter])
        }
    }

    let out: N3DOutput
    if let chunkDir = args.url("chunks") {
        let key = args["key"] ?? args.url("golden")!.lastPathComponent.replacingOccurrences(of: "_probs.f32le", with: "")
        guard let index = try readJSON(chunkDir.appendingPathComponent("chunks/index.json")) as? [String: Any],
              let entry = index[key] as? [String: Any] else { throw SelfTestError.golden("chunks/index.json has no \(key)") }
        let all = try readF32(chunkDir.appendingPathComponent(entry["rows_file"] as! String))
        var steps: [N3DStepInput] = []
        var off = 0
        for s in entry["steps"] as! [[Int]] {
            let n = s[0] * N3DSpeakerCache.hidden
            steps.append(N3DStepInput(rows: Array(all[off..<(off + n)]), lookahead: s[1], emitFrames: s[2]))
            off += n
        }
        log("chunk rows from the Python host (\(key), \(steps.count) steps): the Swift mel is not used")
        j["chunks"] = key
        out = try await diarizer.process(steps: steps, frames: entry["n_frames"] as! Int, onStep: onStep)
    } else {
        out = try await diarizer.process(samples: samples, onStep: onStep)
    }
    let gms = out.graphSeconds.map { $0 * 1e3 }
    log("\(out.frames) frames (\(String(format: "%.2f", Double(samples.count) / 16000)) s audio), \(out.steps) steps, "
        + "\(out.compressions) compressions, near-ties \(out.nearTies); wall \(String(format: "%.3f", out.wallSeconds)) s "
        + "(mel+embed \(String(format: "%.3f", out.frontEndSeconds)) s), graph \(String(format: "%.2f", percentile(gms, 0.5))) ms/step median")
    j.merge(["frames": out.frames, "steps": out.steps, "compressions": out.compressions, "near_ties": out.nearTies,
             "wall_s": out.wallSeconds, "front_end_s": out.frontEndSeconds, "graph_ms_median": percentile(gms, 0.5),
             "graph_ms_p90": percentile(gms, 0.9), "trace": box.get.trace]) { $1 }

    if let golden = args.url("golden") {
        let refProbs = try readF32(golden)
        let logitsURL = golden.deletingLastPathComponent()
            .appendingPathComponent(golden.lastPathComponent.replacingOccurrences(of: "_probs.f32le", with: "_logits.f32le"))
        let refLogits = FileManager.default.fileExists(atPath: logitsURL.path) ? try readF32(logitsURL) : nil
        let cmp = Comparison(logits: out.logits, probs: out.probs, refProbs: refProbs, refLogits: refLogits,
                             tail: profile.kind == .offline ? 16 : 0)
        let ok = cmp.agreement >= 0.999
        let sg = cmp.segments
        log("vs \(golden.lastPathComponent): frames \(cmp.oursFrames)/\(cmp.refFrames) (compared \(cmp.compared), tail \(cmp.tail) left out), "
            + "agreement@0.5 \(pct(cmp.agreement)) (\(cmp.disagree) of \(cmp.elements) differ), max|Δp| \(fmt(cmp.maxAbsP))"
            + (cmp.maxAbsLogit.map { ", max|Δlogit| \(fmt($0))" } ?? "")
            + "; segments ref/ours/matched \(sg.nRef)/\(sg.nOurs)/\(sg.matched), max shift start/end \(sg.maxStartShift)/\(sg.maxEndShift), "
            + "structural \(sg.structural) -> \(ok ? "PASS" : "FAIL") (bar 99.9 %)")
        for c in sg.cases.prefix(6) { log("  structural: \(c)") }
        j["golden"] = golden.path
        j["comparison"] = cmp.json
        j["pass"] = ok
        pass = pass && ok
    }

    if let other = args.url("compare-logits") {
        let ref = try readF32(other)
        let n = min(ref.count, out.logits.count)
        let d = maxAbsDiff(out.logits[0..<n], ref[0..<n])
        var same = 0
        for i in 0..<n where (out.logits[i] > 0) == (ref[i] > 0) { same += 1 }
        log("vs \(other.lastPathComponent): frames \(out.frames)/\(ref.count / 8), max|Δlogit| \(fmt(d.max)), "
            + "bit-equal \(n - d.unequal)/\(n), decisions equal \(same)/\(n)")
        j["compare_logits"] = ["file": other.path, "frames_other": ref.count / 8, "max_abs_logit": d.max,
                               "unequal": d.unequal, "elements": n, "decisions_equal": same]
    }

    if let dumpDir = args.url("dump-packed") {
        try FileManager.default.createDirectory(at: dumpDir, withIntermediateDirectories: true)
        let cap = box.get
        var recs: [[String: Any]] = []
        var worst = 0.0
        var packedOK = true
        for k in dumpList {
            guard let rows = cap.rows[k] else { continue }
            try writeF32(rows, to: dumpDir.appendingPathComponent("packed_step\(k).f32le"))
            var rec: [String: Any] = ["step": k, "L": rows.count / N3DSpeakerCache.hidden]
            if let prefix = args["packed-golden"] {
                let ref = try readF32(URL(fileURLWithPath: ((prefix + "\(k).f32le") as NSString).expandingTildeInPath))
                let Lr = ref.count / N3DSpeakerCache.hidden
                if Lr == rows.count / N3DSpeakerCache.hidden {
                    let d = maxAbsDiff(rows[...], ref[...])
                    rec["L_ref"] = Lr
                    rec["max_abs"] = d.max
                    rec["bit_equal"] = rows.count - d.unequal
                    worst = max(worst, d.max)
                    packedOK = packedOK && d.max <= 1e-3
                    log("packed step \(k): L \(rows.count / 512)/\(Lr), max|Δ| \(fmt(d.max)), bit-equal \(rows.count - d.unequal)/\(rows.count)")
                } else {
                    rec["L_ref"] = Lr
                    packedOK = false
                    log("packed step \(k): L \(rows.count / 512) vs ref \(Lr) -> row count differs")
                }
            }
            recs.append(rec)
        }
        j["packed"] = recs
        if args["packed-golden"] != nil {
            log("packed parity (\(recs.count) steps): max|Δ| \(fmt(worst)) -> \(packedOK ? "PASS" : "FAIL") (bar 1e-3)")
            j["packed_pass"] = packedOK
            pass = pass && packedOK
        }
    }

    if benchN > 0 {
        var lockState = "not checked"
        var ownedLock: URL? = nil
        if let lock = args.url("lock") {
            if FileManager.default.fileExists(atPath: lock.path) {
                lockState = "contended (held by another session, left alone)"
            } else if FileManager.default.createFile(atPath: lock.path, contents: nil) {
                lockState = "taken by this run"
                ownedLock = lock
            }
        }
        defer { if let l = ownedLock { try? FileManager.default.removeItem(at: l) } }
        let T = diarizer.graphLength, H = N3DSpeakerCache.hidden
        let cap = box.get
        func input(_ k: Int) -> ([Float], [Float]) {
            let rows = cap.rows[k]!
            var p = rows
            p.append(contentsOf: repeatElement(0, count: T * H - rows.count))
            let L = rows.count / H
            return (p, (0..<T).map { $0 < L ? 1 : 0 })
        }
        let inputs = benchSteps.map(input)
        let load0 = loadAverage()
        for k in 0..<5 { _ = try await diarizer.runGraph(packed: inputs[k].0, valid: inputs[k].1) }
        var ts: [Double] = []
        for k in 0..<benchN {
            let t0 = ContinuousClock.now
            _ = try await diarizer.runGraph(packed: inputs[5 + k].0, valid: inputs[5 + k].1)
            let d = ContinuousClock.now - t0
            ts.append((Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18) * 1e3)
        }
        let again = try await diarizer.process(samples: samples)
        let gsum = again.graphSeconds.reduce(0, +)
        let audio = Double(samples.count) / 16000
        let b: [String: Any] = [
            "machine": sysctlString("machdep.cpu.brand_string"), "os": ProcessInfo.processInfo.operatingSystemVersionString,
            "gpu_lock": lockState, "load_avg_start": load0, "load_avg_end": loadAverage(),
            "calls": benchN, "warmup": 5, "ms_per_chunk_median": percentile(ts, 0.5), "ms_per_chunk_p90": percentile(ts, 0.9),
            "ms_per_chunk_min": ts.min() ?? .nan, "loop_steps": again.steps, "loop_wall_s": again.wallSeconds,
            "loop_rtf": again.wallSeconds / audio, "loop_graph_s": gsum, "loop_front_end_s": again.frontEndSeconds,
            "loop_host_s": again.wallSeconds - gsum, "audio_s": audio]
        log("bench (\(sysctlString("machdep.cpu.brand_string")), GPU lock: \(lockState)): graph \(String(format: "%.2f", percentile(ts, 0.5))) ms/chunk median, "
            + "p90 \(String(format: "%.2f", percentile(ts, 0.9))) (\(benchN) calls after 5 warmup) | whole \(String(format: "%.1f", audio)) s: "
            + "\(again.steps) steps, wall \(String(format: "%.3f", again.wallSeconds)) s, RTF \(String(format: "%.4f", again.wallSeconds / audio)), "
            + "graph \(String(format: "%.3f", gsum)) s, host \(String(format: "%.3f", again.wallSeconds - gsum)) s (mel+embed \(String(format: "%.3f", again.frontEndSeconds)) s)")
        j["bench"] = b
    }
    return (pass, j)
}

// MARK: - main

do {
    let args = try Args(CommandLine.arguments)
    guard let assetsURL = args.url("assets") else { throw SelfTestError.usage("--assets <dir> is required") }
    let assets = try N3DAssets(directory: assetsURL)
    var report: [String: Any] = ["assets": assetsURL.path, "argv": CommandLine.arguments]
    var ok = true
    var samples: [Float] = []
    if let wav = args.url("wav") {
        samples = try loadWav16kMono(wav)
        report["wav"] = wav.path
        report["samples"] = samples.count
    }
    if let g = args.url("cache-golden") {
        let (p, r) = try cacheCheck(assets: assets, golden: g)
        ok = ok && p
        report["cache"] = r
    }
    if let g = args.url("mel-golden") {
        guard let wav = args.url("wav") else { throw SelfTestError.usage("--mel-golden needs --wav") }
        let fixture = args["fixture"] ?? wav.deletingPathExtension().lastPathComponent.replacingOccurrences(of: "_16k", with: "")
        let (p, r) = try melCheck(assets: assets, samples: samples, fixture: fixture, golden: g)
        ok = ok && p
        report["mel"] = r
    }
    if args.has("bundle") {
        guard !samples.isEmpty || args.has("chunks") else { throw SelfTestError.usage("--bundle needs --wav") }
        let (p, r) = try await closedLoop(args: args, assets: assets, samples: samples)
        ok = ok && p
        report["closed_loop"] = r
    }
    report["pass"] = ok
    if let path = args.url("json") {
        let d = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try d.write(to: path)
    }
    log(ok ? "EXIT 0 (all requested gates PASS)" : "EXIT 3 (a gate FAILED)")
    exit(ok ? 0 : 3)
} catch {
    log("ERROR: \(error)")
    exit(4)
}
