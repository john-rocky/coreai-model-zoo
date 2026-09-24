// GateRunner — the iPhone half of the Nemotron-3-Diarization gate. The host (NemotronDiarizer), the golden
// files and the metrics (N3DGateSupport) are the ones n3d-selftest runs on the Mac
// (conversion/nemotron3_diar/swift), here driving the sideloaded AOT bundles. Added for the device: load
// times (first and second load in this process), a bench, the thermal state around every step, and the
// device / OS identity. Results: Documents/n3d_gate/result.json (rewritten after every stage, "status"
// running -> done) and result.log (one line per event).
//
// Assets: Library/Application Support/N3DAssets/ (../_stage.sh lays it out, ../_install.sh pushes it):
//   n3d_streaming_float16.h18p.aimodelc/       AOT h18p, preferred gpu            -> stage gpu_streaming
//   n3d_offline_float16.h18p.aimodelc/         AOT h18p, preferred gpu            -> stage gpu_offline
//   ane/n3d_streaming_float16.h18p.aimodelc/   AOT h18p, preferred neural-engine  -> stage ane_streaming (if staged)
//   embedder_projection.f32le  silence_embeds.f32le  mel_filters_128x257.f32le  hann_window_400.f32le  metadata.json
//   fixtures/<fixture>_16k.wav   golden/<fixture>_<tag>_{probs,logits}.f32le   golden/pygpu/<fixture>_<tag>_logits.f32le
//   MD5SUMS
//
// Launch environment (devicectl device process launch --environment-variables), all optional:
//   N3D_RUN_ID     echoed into result.json (../_run.sh waits for its own id)
//   N3D_STAGES     gpu,offline,ane (default all three; ane runs only when its bundle is staged)
//   N3D_MODES      streaming modes of the streaming stages (default ll; vll / ull use their golden files)
//   N3D_BENCH      graph calls timed after 5 warm-up calls (default 100; 0 = no bench)
//   N3D_GPU_UNIT   gpu | ane | default | cpuOnly (default gpu)     N3D_ANE_UNIT (default ane)
//   N3D_ASSETS     assets directory relative to the app's home (default Library/Application Support/N3DAssets)

import Foundation
import N3DGateSupport
import NemotronDiarizer

struct GateConfig: Sendable {
    static let bar = 0.999                                   // agreement@0.5, the Mac gate's bar
    static let fixtures = ["diarization_example", "test_multispk"]    // 97.6 s (compresses) and 21.5 s

    let runID: String
    let assets: URL
    let out: URL
    let stages: [String]
    let modes: [N3DStreamingMode]
    let benchCalls: Int
    let gpuUnit: N3DComputeUnits
    let aneUnit: N3DComputeUnits

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        let assets = env["N3D_ASSETS"].map { home.appendingPathComponent($0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("N3DAssets")
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = ISO8601DateFormatter().string(from: Date())
        func list(_ key: String, _ fallback: String) -> [String] {
            (env[key] ?? fallback).split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        }
        return GateConfig(
            runID: env["N3D_RUN_ID"] ?? stamp,
            assets: assets,
            out: docs.appendingPathComponent("n3d_gate"),
            stages: list("N3D_STAGES", "gpu,offline,ane"),
            modes: list("N3D_MODES", "ll").compactMap(N3DStreamingMode.init(rawValue:)),
            benchCalls: Int(env["N3D_BENCH"] ?? "") ?? 100,
            gpuUnit: N3DComputeUnits(rawValue: env["N3D_GPU_UNIT"] ?? "gpu") ?? .gpu,
            aneUnit: N3DComputeUnits(rawValue: env["N3D_ANE_UNIT"] ?? "ane") ?? .ane)
    }
}

/// The bundles each stage loads, relative to the assets directory.
enum StageBundles {
    #if os(iOS)
    static let streaming = "n3d_streaming_float16.h18p.aimodelc"
    static let offline = "n3d_offline_float16.h18p.aimodelc"
    static let ane = "ane/n3d_streaming_float16.h18p.aimodelc"
    #else
    // a Mac dry run of this file (the app is iOS-only): the macOS bundles in the same layout
    static let streaming = "n3d_streaming_float16.aimodel"
    static let offline = "n3d_offline_float16.aimodel"
    static let ane = "ane/n3d_streaming_float16.h16c.aimodelc"
    #endif
}

/// One bundle under test and the closed loops run on it.
struct StageSpec: Sendable {
    let key: String
    let bundle: URL
    let unit: N3DComputeUnits
    let runs: [(fixture: String, tag: String)]
    let optional: Bool                           // skipped (not failed) when the bundle is not staged

    var benchRun: (fixture: String, tag: String) { runs[0] }
}

actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0 = ContinuousClock.now
    private var log: FileHandle?
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []
    private var gpuLogits: [String: [Float]] = [:]           // "<fixture>_<tag>" -> logits of gpu_streaming

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
    }

    // MARK: - the run

    /// Every stage in order; true when every stage that ran passed.
    func run() async -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: config.out, withIntermediateDirectories: true)
        let resultURL = config.out.appendingPathComponent("result.json")
        let logURL = config.out.appendingPathComponent("result.log")
        try? fm.removeItem(at: resultURL)
        fm.createFile(atPath: logURL.path, contents: nil)
        log = try? FileHandle(forWritingTo: logURL)

        let launchIndex = bumpLaunchCount()
        let device = DeviceInfo.snapshot()
        report = ["app": "N3DGate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device, "bar_agreement": GateConfig.bar,
                  "config": ["assets": config.assets.path, "stages": config.stages, "modes": config.modes.map(\.rawValue),
                             "bench_calls": config.benchCalls, "gpu_unit": config.gpuUnit.rawValue,
                             "ane_unit": config.aneUnit.rawValue]]
        line("N3DGate run \(config.runID) (launch \(launchIndex) in this container)")
        line("device \(device["machine"] ?? "?"), \(device["os"] ?? "?") (build \(device["os_build"] ?? "?")), "
             + "thermal \(DeviceInfo.thermal()), low power \(device["low_power_mode"] ?? "?")")
        writeReport()

        // assets
        setStage("assets")
        let listed = checkAssets()
        let assets: N3DAssets
        var audio: [String: [Float]] = [:]
        do {
            assets = try N3DAssets(directory: config.assets)
            for f in GateConfig.fixtures {
                let wav = config.assets.appendingPathComponent("fixtures/\(f)_16k.wav")
                let s = try loadWav16kMono(wav)
                audio[f] = s
                line("fixture \(f): \(s.count) samples (\(String(format: "%.2f", Double(s.count) / 16000)) s)")
            }
        } catch {
            line("FATAL assets: \(error)")
            report["assets"] = listed
            return finish(ok: false, fatal: "assets: \(error)")
        }
        report["assets"] = listed

        // stages
        var allOK = true
        for spec in stageSpecs() {
            setStage(spec.key)
            let result = await runStage(spec, assets: assets, audio: audio)
            stageResults[spec.key] = result
            stageOrder.append(spec.key)
            if let p = result["pass"] as? Bool { allOK = allOK && p }
            writeReport()
        }
        return finish(ok: allOK, fatal: nil)
    }

    func stageSpecs() -> [StageSpec] {
        let a = config.assets
        let streamingRuns = config.modes.flatMap { m in GateConfig.fixtures.map { (fixture: $0, tag: m.rawValue) } }
        let offlineRuns = GateConfig.fixtures.map { (fixture: $0, tag: "offline") }
        var specs: [StageSpec] = []
        for s in config.stages {
            switch s {
            case "gpu":
                specs.append(StageSpec(key: "gpu_streaming", bundle: a.appendingPathComponent(StageBundles.streaming),
                                       unit: config.gpuUnit, runs: streamingRuns, optional: false))
            case "offline":
                specs.append(StageSpec(key: "gpu_offline", bundle: a.appendingPathComponent(StageBundles.offline),
                                       unit: config.gpuUnit, runs: offlineRuns, optional: false))
            case "ane":
                specs.append(StageSpec(key: "ane_streaming", bundle: a.appendingPathComponent(StageBundles.ane),
                                       unit: config.aneUnit, runs: streamingRuns, optional: true))
            default:
                line("unknown stage \(s) (N3D_STAGES takes gpu, offline, ane)")
            }
        }
        return specs
    }

    // MARK: - one stage: load twice, closed loops vs golden, bench

    func runStage(_ spec: StageSpec, assets: N3DAssets, audio: [String: [Float]]) async -> [String: Any] {
        var j: [String: Any] = ["bundle": spec.bundle.path.replacingOccurrences(of: config.assets.path + "/", with: ""),
                                "unit": spec.unit.rawValue]
        var thermal: [[String: Any]] = []
        func mark(_ at: String) { thermal.append(["at": at, "state": DeviceInfo.thermal(), "t_s": elapsed()]) }
        mark("start")
        guard !spec.runs.isEmpty else {
            j["skipped"] = "no runs (N3D_MODES)"
            return j
        }
        guard FileManager.default.fileExists(atPath: spec.bundle.path) else {
            let why = "bundle not staged: \(spec.bundle.lastPathComponent) in \(j["bundle"] ?? "")"
            line("\(spec.key): \(spec.optional ? "skipped" : "FAIL"), \(why)")
            j[spec.optional ? "skipped" : "error"] = why
            if !spec.optional { j["pass"] = false }
            return j
        }
        let (bytes, files) = DeviceInfo.tree(spec.bundle)
        j["bundle_mb"] = Double(bytes) / 1e6
        j["bundle_files"] = files
        line("\(spec.key): \(j["bundle"] ?? "") (\(String(format: "%.1f", Double(bytes) / 1e6)) MB, \(files) files), unit \(spec.unit.rawValue)")

        var pass = true
        do {
            let firstProfile = profile(spec.runs[0].tag)
            // load 1 = the first load of this bundle in this process (caches from earlier launches may remain)
            var first: N3DDiarizer? = try await N3DDiarizer(assets: assets, computeUnits: spec.unit, profile: firstProfile,
                                                            modelURL: spec.bundle)
            let load1 = first!.loadSeconds
            mark("after load 1")
            let firstCall = try await probeCall(first!)
            mark("after first call")
            first = nil
            let diarizer = try await N3DDiarizer(assets: assets, computeUnits: spec.unit, profile: firstProfile, modelURL: spec.bundle)
            let load2 = diarizer.loadSeconds
            mark("after load 2")
            j["load_first_s"] = load1
            j["first_call_ms"] = firstCall
            j["load_second_s"] = load2
            j["footprint_mb_after_load"] = DeviceInfo.footprintMB()
            line("\(spec.key): load \(String(format: "%.2f", load1)) s (first in this process), first call "
                 + "\(String(format: "%.1f", firstCall)) ms, load \(String(format: "%.2f", load2)) s (second); "
                 + "footprint \(String(format: "%.0f", DeviceInfo.footprintMB())) MB")

            var runs: [String: Any] = [:]
            var runOrder: [String] = []
            for (index, r) in spec.runs.enumerated() {
                let key = "\(r.fixture)_\(r.tag)"
                runOrder.append(key)
                guard let samples = audio[r.fixture] else { continue }
                let p = profile(r.tag)
                let doBench = index == 0 && config.benchCalls > 0
                var rec = try await closedLoop(spec, diarizer, key: key, samples: samples, profile: p, bench: doBench)
                mark("after \(key)")
                if let ok = rec["pass"] as? Bool { pass = pass && ok }
                if doBench, let rows = rec.removeValue(forKey: "_bench_rows") as? [Int: [Float]],
                   let steps = rec.removeValue(forKey: "_bench_steps") as? [Int] {
                    j["bench"] = try await bench(spec, diarizer, key: key, samples: samples, profile: p, rows: rows, steps: steps)
                    mark("after bench")
                }
                runs[key] = rec
            }
            j["runs"] = runs
            j["run_order"] = runOrder
            j["footprint_mb_end"] = DeviceInfo.footprintMB()
        } catch {
            line("\(spec.key): ERROR \(error)")
            j["error"] = "\(error)"
            pass = false
        }
        mark("end")
        j["thermal"] = thermal
        j["pass"] = pass
        line("\(spec.key): \(pass ? "PASS" : "FAIL") (bar agreement@0.5 >= \(GateConfig.bar * 100) % on every run)")
        return j
    }

    /// One call on a zero input (row 0 valid) right after the first load: where a lazy specialization would show.
    func probeCall(_ d: N3DDiarizer) async throws -> Double {
        let T = d.graphLength
        var valid = [Float](repeating: 0, count: T)
        valid[0] = 1
        let c0 = ContinuousClock.now
        _ = try await d.runGraph(packed: [Float](repeating: 0, count: T * N3DSpeakerCache.hidden), valid: valid)
        return Self.seconds(since: c0) * 1e3
    }

    func closedLoop(_ spec: StageSpec, _ d: N3DDiarizer, key: String, samples: [Float], profile p: N3DProfile,
                    bench: Bool) async throws -> [String: Any] {
        let nSteps = stepCount(samples: samples.count, profile: p)
        let benchSteps = bench ? (0..<5).map { $0 % nSteps } + (0..<config.benchCalls).map { ($0 * 7) % nSteps } : []
        let wanted = Set(benchSteps)
        let box = Box([Int: [Float]]())
        let out = try await d.process(samples: samples, profile: p) { info in
            if wanted.contains(info.step) { box.mutate { $0[info.step] = info.rows } }
        }
        let gms = out.graphSeconds.map { $0 * 1e3 }
        var rec: [String: Any] = [
            "frames": out.frames, "steps": out.steps, "compressions": out.compressions, "near_ties": out.nearTies,
            "wall_s": out.wallSeconds, "front_end_s": out.frontEndSeconds, "graph_ms_median": percentile(gms, 0.5),
            "graph_ms_p90": percentile(gms, 0.9), "graph_ms_first": gms.first ?? 0,
            "segments_ours": N3DDiarizer.segments(from: out.probs).map { [$0.speaker, $0.startFrame, $0.endFrame] }]
        line("\(spec.key) \(key): \(out.frames) frames, \(out.steps) steps, \(out.compressions) compressions, "
             + "near-ties \(out.nearTies); wall \(String(format: "%.3f", out.wallSeconds)) s (mel+embed "
             + "\(String(format: "%.3f", out.frontEndSeconds)) s), graph \(String(format: "%.2f", percentile(gms, 0.5))) ms/step median")

        let g = config.assets.appendingPathComponent("golden")
        let probsURL = g.appendingPathComponent("\(key)_probs.f32le")
        if FileManager.default.fileExists(atPath: probsURL.path) {
            let refProbs = try readF32(probsURL)
            let logitsURL = g.appendingPathComponent("\(key)_logits.f32le")
            let refLogits = FileManager.default.fileExists(atPath: logitsURL.path) ? try readF32(logitsURL) : nil
            let cmp = Comparison(logits: out.logits, probs: out.probs, refProbs: refProbs, refLogits: refLogits,
                                 tail: p.kind == .offline ? 16 : 0)
            let ok = cmp.agreement >= GateConfig.bar
            let sg = cmp.segments
            line("\(spec.key) \(key) vs transformers fp32: frames \(cmp.oursFrames)/\(cmp.refFrames) (compared \(cmp.compared), "
                 + "tail \(cmp.tail) left out), agreement@0.5 \(pct(cmp.agreement)) (\(cmp.disagree) of \(cmp.elements) differ), "
                 + "max|Δp| \(fmt(cmp.maxAbsP))" + (cmp.maxAbsLogit.map { ", max|Δlogit| \(fmt($0))" } ?? "")
                 + "; segments ref/ours/matched \(sg.nRef)/\(sg.nOurs)/\(sg.matched), max shift start/end "
                 + "\(sg.maxStartShift)/\(sg.maxEndShift), structural \(sg.structural) -> \(ok ? "PASS" : "FAIL")")
            for c in sg.cases.prefix(6) { line("  structural: \(c)") }
            rec["comparison"] = cmp.json
            rec["pass"] = ok
        } else {
            line("\(spec.key) \(key): no golden \(probsURL.lastPathComponent), loop timed only")
            rec["golden"] = "missing"
        }
        // the same fp16 bundle through host_loop.py on the Mac GPU (JIT): how far the phone's GPU is from the Mac's
        let macURL = g.appendingPathComponent("pygpu/\(key)_logits.f32le")
        if FileManager.default.fileExists(atPath: macURL.path) {
            rec["vs_mac_gpu"] = logitsDiff(out.logits, try readF32(macURL))
            line("\(spec.key) \(key) vs Mac GPU logits: \(Self.describe(rec["vs_mac_gpu"]))")
        }
        if spec.key == "gpu_streaming" {
            gpuLogits[key] = out.logits
        } else if spec.key == "ane_streaming", let gpu = gpuLogits[key] {
            // bit-equal to the GPU stage everywhere would mean the ANE bundle did not run on the ANE
            rec["vs_gpu_stage"] = logitsDiff(out.logits, gpu)
            line("\(spec.key) \(key) vs gpu_streaming logits: \(Self.describe(rec["vs_gpu_stage"]))")
        }
        if bench {
            rec["_bench_rows"] = box.get
            rec["_bench_steps"] = benchSteps
        }
        return rec
    }

    /// n3d-selftest's --bench on the phone: warm-up 5 + N graph calls on this run's own packed inputs (step
    /// (k * 7) mod steps), then one timed whole run.
    func bench(_ spec: StageSpec, _ d: N3DDiarizer, key: String, samples: [Float], profile p: N3DProfile,
               rows: [Int: [Float]], steps: [Int]) async throws -> [String: Any] {
        let T = d.graphLength, H = N3DSpeakerCache.hidden
        var inputs: [([Float], [Float])] = []
        for k in steps {
            guard let r = rows[k] else { throw SelfTestError.golden("bench step \(k) was not captured") }
            var packed = r
            packed.append(contentsOf: repeatElement(0, count: T * H - r.count))
            let L = r.count / H
            inputs.append((packed, (0..<T).map { $0 < L ? 1 : 0 }))
        }
        let thermalStart = DeviceInfo.thermal()
        for k in 0..<5 { _ = try await d.runGraph(packed: inputs[k].0, valid: inputs[k].1) }
        var ts: [Double] = []
        for k in 0..<(inputs.count - 5) {
            let c0 = ContinuousClock.now
            _ = try await d.runGraph(packed: inputs[5 + k].0, valid: inputs[5 + k].1)
            ts.append(Self.seconds(since: c0) * 1e3)
        }
        let again = try await d.process(samples: samples, profile: p)
        let gsum = again.graphSeconds.reduce(0, +)
        let audioS = Double(samples.count) / 16000
        let b: [String: Any] = [
            "on": key, "calls": ts.count, "warmup": 5, "ms_per_chunk_median": percentile(ts, 0.5),
            "ms_per_chunk_p90": percentile(ts, 0.9), "ms_per_chunk_min": ts.min() ?? 0, "ms_per_chunk_max": ts.max() ?? 0,
            "loop_steps": again.steps, "loop_wall_s": again.wallSeconds, "loop_rtf": again.wallSeconds / audioS,
            "loop_graph_s": gsum, "loop_front_end_s": again.frontEndSeconds, "loop_host_s": again.wallSeconds - gsum,
            "audio_s": audioS, "thermal_start": thermalStart, "thermal_end": DeviceInfo.thermal()]
        line("\(spec.key) bench on \(key): graph \(String(format: "%.2f", percentile(ts, 0.5))) ms/chunk median, p90 "
             + "\(String(format: "%.2f", percentile(ts, 0.9))) (\(ts.count) calls after 5 warm-up) | whole "
             + "\(String(format: "%.1f", audioS)) s: \(again.steps) steps, wall \(String(format: "%.3f", again.wallSeconds)) s, "
             + "RTF \(String(format: "%.4f", again.wallSeconds / audioS)), graph \(String(format: "%.3f", gsum)) s, host "
             + "\(String(format: "%.3f", again.wallSeconds - gsum)) s | thermal \(thermalStart) -> \(DeviceInfo.thermal())")
        return b
    }

    // MARK: - helpers

    func profile(_ tag: String) -> N3DProfile {
        tag == "offline" ? .offline : .streamingProfile(mode: N3DStreamingMode(rawValue: tag) ?? .lowLatency)
    }

    func logitsDiff(_ ours: [Float], _ other: [Float]) -> [String: Any] {
        let n = min(ours.count, other.count)
        let d = maxAbsDiff(ours[0..<n], other[0..<n])
        var same = 0
        for i in 0..<n where (ours[i] > 0) == (other[i] > 0) { same += 1 }
        return ["frames_ours": ours.count / 8, "frames_other": other.count / 8, "max_abs_logit": d.max,
                "bit_equal": n - d.unequal, "elements": n, "decisions_equal": same]
    }

    static func describe(_ v: Any?) -> String {
        guard let d = v as? [String: Any] else { return "-" }
        return "max|Δlogit| \(fmt(d["max_abs_logit"] as? Double ?? .nan)), bit-equal \(d["bit_equal"] ?? "?")/\(d["elements"] ?? "?"), "
            + "decisions equal \(d["decisions_equal"] ?? "?")/\(d["elements"] ?? "?")"
    }

    /// Files listed in MD5SUMS that are absent here (the host checks the md5 values after the push).
    func checkAssets() -> [String: Any] {
        let fm = FileManager.default
        var listed = 0
        var missing: [String] = []
        if let text = try? String(contentsOf: config.assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) {
            for row in text.split(separator: "\n") {
                let parts = row.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { continue }
                listed += 1
                let rel = parts[1].trimmingCharacters(in: .whitespaces)
                if !fm.fileExists(atPath: config.assets.appendingPathComponent(rel).path) { missing.append(rel) }
            }
        }
        line("assets \(config.assets.path): \(listed) files in MD5SUMS, \(missing.count) missing"
             + (missing.isEmpty ? "" : " (\(missing.prefix(5).joined(separator: ", "))\(missing.count > 5 ? ", ..." : ""))"))
        return ["dir": config.assets.path, "md5sums_listed": listed, "missing": missing]
    }

    func finish(ok: Bool, fatal: String?) -> Bool {
        report["status"] = fatal == nil ? "done" : "failed"
        if let f = fatal { report["fatal"] = f }
        report["pass"] = ok
        report["finished"] = Self.now()
        report["elapsed_s"] = elapsed()
        report["device_end"] = ["thermal": DeviceInfo.thermal(), "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled]
        let verdicts = stageOrder.map { k -> String in
            let s = stageResults[k] as? [String: Any] ?? [:]
            if s["skipped"] != nil { return "\(k)=skipped" }
            return "\(k)=\((s["pass"] as? Bool) == true ? "PASS" : "FAIL")"
        }
        report["summary"] = verdicts
        line("GATE_SUMMARY \(verdicts.joined(separator: " ")) VERDICT=\(ok ? "PASS" : "FAIL")"
             + (fatal.map { " (\($0))" } ?? ""))
        writeReport()
        // keep a copy per run next to the latest one
        let runs = config.out.appendingPathComponent("runs")
        try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let safe = config.runID.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
        try? FileManager.default.copyItem(at: config.out.appendingPathComponent("result.json"),
                                          to: runs.appendingPathComponent("\(safe).json"))
        line("DONE \(config.runID)")
        try? log?.close()
        log = nil
        setStage(ok ? "done: PASS" : "done: FAIL")
        return ok
    }

    func writeReport() {
        var r = report
        r["stages"] = stageResults
        r["stage_order"] = stageOrder
        r["updated"] = Self.now()
        let clean = Self.jsonSafe(r)
        guard JSONSerialization.isValidJSONObject(clean),
              let d = try? JSONSerialization.data(withJSONObject: clean, options: [.prettyPrinted, .sortedKeys]) else {
            line("ERROR result.json could not be serialized")
            return
        }
        let url = config.out.appendingPathComponent("result.json")
        let tmp = config.out.appendingPathComponent("result.json.tmp")
        do {
            try d.write(to: tmp)
            _ = try? FileManager.default.removeItem(at: url)
            try FileManager.default.moveItem(at: tmp, to: url)
        } catch {
            line("ERROR writing result.json: \(error)")
        }
    }

    /// JSONSerialization raises (it does not throw) on NaN / infinity: replace them with strings.
    static func jsonSafe(_ v: Any) -> Any {
        switch v {
        case let d as Double: return d.isFinite ? d : "\(d)"
        case let f as Float: return f.isFinite ? Double(f) : "\(f)"
        case let a as [Any]: return a.map(jsonSafe)
        case let m as [String: Any]: return m.mapValues(jsonSafe)
        default: return v
        }
    }

    func line(_ s: String) {
        let text = "[n3d] \(String(format: "%8.2f", elapsed())) \(s)"
        print(text)
        if let d = (text + "\n").data(using: .utf8) {
            try? log?.write(contentsOf: d)
            try? log?.synchronize()
        }
        emit(text)
    }

    func bumpLaunchCount() -> Int {
        let url = config.out.appendingPathComponent("launch_count")
        let n = (try? String(contentsOf: url, encoding: .utf8)).flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0
        try? "\(n + 1)\n".write(to: url, atomically: true, encoding: .utf8)
        return n + 1
    }

    func elapsed() -> Double { Self.seconds(since: t0) }

    static func seconds(since t: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - t
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    static func now() -> String { ISO8601DateFormatter().string(from: Date()) }
}
