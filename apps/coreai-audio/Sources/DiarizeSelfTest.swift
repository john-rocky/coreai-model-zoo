// Headless self-test for the Diarize path (DIARIZE_SELFTEST=1) — the Swift mirror of
// conversion/sortformer_diar/gate_e2e_engine.py + gate_long.py. Two gates on two clips (21.5 s demo,
// 64.5 s = demo×3 which exercises AOSC compress ~4×):
//   1. MEL gate:  Swift NeMo-mel(wav) vs the captured golden mel (cos).
//   2. LOOP gate: drive the SHIPPED fp16 .aimodel (Mac GPU) over the golden mel through the full
//                 streaming host loop, compare per-frame activity to NeMo forward_streaming
//                 (activity-agree @ 0.5 — the actual diarization decision). PASS = ≥99%.
// Plus an end-to-end line (Swift-mel -> loop) for information. Runs GUI-less (init()-launched).
// Then the 8-speaker gate ([8spk] lines, runNemotronDiarizeGate below). EXIT 0 only when both pass.
import Foundation
import N3DGateSupport
import NemotronDiarizer

func runDiarizeSelfTest() async {
    setvbuf(stdout, nil, _IONBF, 0)
    let logURL = URL(fileURLWithPath: ProcessInfo.processInfo.environment["DIAR_RESULT"] ?? "/tmp/diar_result.txt")
    try? "".write(to: logURL, atomically: true, encoding: .utf8)
    func log(_ s: String) {
        print("[DIAR] \(s)")
        if let h = try? FileHandle(forWritingTo: logURL) {
            h.seekToEndOfFile(); h.write(Data(("[DIAR] \(s)\n").utf8)); try? h.close()
        }
    }
    let sortformer = await runSortformerDiarizeGate(log: log)
    let nemotron = await runNemotronDiarizeGate(log: log)
    func verdict(_ code: Int32) -> String { code == 0 ? "PASS" : "exit \(code)" }
    log("SUMMARY 4spk \(verdict(sortformer)), 8spk \(verdict(nemotron))")
    let code = sortformer != 0 ? sortformer : nemotron
    log("EXIT \(code)")
    exit(code)
}

/// The 4-speaker Sortformer gates: 0 = PASS, 2 = assets missing, 3 = a loop gate below 99 %, 4 = error.
private func runSortformerDiarizeGate(log: (String) -> Void) async -> Int32 {
    guard let root = DiarizeAssets.root, let murl = DiarizeAssets.modelURL,
          let filters = DiarizeAssets.melFilters() else {
        log("FAIL: assets not found at \(DiarizeAssets.location.path)"); return 2
    }
    guard let wav = AudioLoader.load16kMono(root.appendingPathComponent("test_multispk_16k.wav")) else {
        log("FAIL: demo wav missing"); return 2
    }
    log("wav \(wav.count) samples (\(String(format: "%.1f", Double(wav.count) / 16000))s)")

    do {
        let t0 = ContinuousClock().now
        let diar = try await SortformerDiarizer(model: murl, melFilters: filters, computeUnits: .gpu)
        log(String(format: "loaded model in %.2fs", secs(since: t0)))

        var allPass = true
        // (clip label, samples, golden-mel file, golden-preds file)
        let clips: [(String, [Float], String, String)] = [
            ("demo 21.5s", wav, "golden_mel_128xT.f32", "golden_total_preds.f32"),
            ("long 64.5s", wav + wav + wav, "golden_long_mel_128xT.f32", "golden_long_total_preds.f32"),
        ]
        for (label, samples, melFile, predFile) in clips {
            guard let goldMel = DiarizeAssets.f32(melFile), let goldPreds = DiarizeAssets.f32(predFile) else {
                log("FAIL[\(label)]: golden files missing"); allPass = false; continue
            }
            let T = goldMel.count / 128
            let nOutGold = goldPreds.count / 4

            // 1. MEL gate — Swift mel vs golden mel (cos over the aligned [128, minT]).
            let mel = await diar.melForSelfTest(samples)
            let Ts = mel.count / 128
            let cosMel = melCos(mel, Ts, goldMel, T)
            log(String(format: "[\(label)] mel: Swift[128,%d] vs golden[128,%d]  cos %.6f", Ts, T, cosMel))

            // 2. LOOP gate — golden mel -> shipped graph -> host loop, activity-agree vs NeMo.
            let g0 = ContinuousClock().now
            let framesGolden = try await diar.framePreds(mel: goldMel, melFrames: T)
            let dt = secs(since: g0)
            let (agreeG, cosG) = agree(framesGolden, goldPreds, nOutGold)
            let loopPass = agreeG >= 0.99 && framesGolden.count == nOutGold
            log(String(format: "[\(label)] loop(golden mel): frames %d vs %d  cos %.6f  activity-agree %.2f%%  (%.2fs, %.1f× RT)  -> %@",
                       framesGolden.count, nOutGold, cosG, agreeG * 100, dt,
                       Double(samples.count) / 16000 / dt, loopPass ? "PASS" : "FAIL"))

            // end-to-end (Swift mel -> loop): informational — Swift mel ≈ golden, not bit-exact.
            let framesE2E = try await diar.framePreds(mel: mel, melFrames: Ts)
            let (agreeE, _) = agree(framesE2E, goldPreds, min(framesE2E.count, nOutGold))
            let segs = SortformerDiarizer.segments(from: framesE2E)
            log(String(format: "[\(label)] e2e(Swift mel): frames %d  activity-agree %.2f%%  -> %d speaker turns",
                       framesE2E.count, agreeE * 100, segs.count))
            for seg in segs.prefix(6) {
                log(String(format: "      spk%d  %.2f–%.2fs", seg.speaker, seg.startSec, seg.endSec))
            }
            allPass = allPass && loopPass
        }
        log(allPass ? "PASS" : "CHECK (a loop gate is below 99% activity-agree)")
        return allPass ? 0 : 3
    } catch { log("FAIL: \(error)"); return 4 }
}

/// The 8-speaker gate on the Transcribe tab's own path: fixture wav -> AudioLoader (as "Choose…") ->
/// NemotronDiarizerBridge (Swift mel -> low-latency loop on the GPU -> turns) vs the transformers fp32
/// probabilities. agreement@0.5 over every frame x 8 speakers with n3d-selftest's metric (N3DGateSupport),
/// bar 99.9 % and the same frame count; turns, segments and wall are reported. A 0.8 s clip and an empty
/// one (padded by the bridge) must give turns inside the clip; the 0.8 s clip also takes the first call
/// after the load, so the fixture walls are warm. On every clip the playback-synced run's chunk-at-a-time
/// loop must give bit-equal logits and the same turns.
///   golden: $N3D_GOLDEN (default <N3DAssets>/golden), <fixture>_ll_{probs,logits}.f32le of
///           conversion/nemotron3_diar/export_golden.py
///   wavs:   $N3D_FIXTURES (default <N3DAssets>/fixtures), <fixture>_16k.wav
/// 0 = PASS, 2 = assets / golden / wav missing, 3 = below the bar, 4 = error.
private func runNemotronDiarizeGate(log: (String) -> Void) async -> Int32 {
    guard let assets = NemotronDiarizerBridge.staged else {
        log("[8spk] FAIL: assets not found at \(NemotronDiarizerBridge.location.path)"); return 2
    }
    let env = ProcessInfo.processInfo.environment
    let golden = env["N3D_GOLDEN"].map { URL(fileURLWithPath: $0) } ?? assets.directory.appendingPathComponent("golden")
    let fixtures = env["N3D_FIXTURES"].map { URL(fileURLWithPath: $0) } ?? assets.directory.appendingPathComponent("fixtures")
    let bar = 0.999
    let S = NemotronDiarizerBridge.speakers
    do {
        let t0 = ContinuousClock().now
        let bridge = try await NemotronDiarizerBridge.load()
        log(String(format: "[8spk] loaded %@ in %.2fs (graph %.2fs; GPU, low latency 9+4, %d speakers)",
                   bridge.modelURL.lastPathComponent, secs(since: t0), bridge.loadSeconds, S))
        var missing = false, allPass = true

        let firstWav = fixtures.appendingPathComponent("test_multispk_16k.wav")
        guard let first = AudioLoader.load16kMono(firstWav) else {
            log("[8spk] FAIL: wav missing \(firstWav.path) (set N3D_FIXTURES)"); return 2
        }
        for n in [12_800, 0] {
            let s0 = ContinuousClock().now
            let clip = Array(first.prefix(n))
            let (turns, out) = try await bridge.diarize(clip)
            let lastFrame = n / N3DMel.hop
            let ok = turns.allSatisfy { $0.startFrame < $0.endFrame && $0.endFrame <= lastFrame }
            log(String(format: "[8spk short] %.2fs clip (padded to %.2fs): %d steps, turns %d, last turn end %.2fs <= %.2fs, %.2fs  -> %@",
                       Double(n) / 16000, Double(NemotronDiarizerBridge.minimumSamples) / 16000, out.steps,
                       turns.count, turns.last?.endSec ?? 0, Double(lastFrame) * NemotronDiarizerBridge.frameSec,
                       secs(since: s0), ok ? "OK" : "FAIL"))
            let stepwise = try await stepwiseCheck(bridge, clip, out, turns, label: "short \(String(format: "%.2fs", Double(n) / 16000))", log: log)
            allPass = allPass && ok && stepwise
        }

        for (label, fixture) in [("21.5s", "test_multispk"), ("97.6s", "diarization_example")] {
            let wavURL = fixtures.appendingPathComponent("\(fixture)_16k.wav")
            let probsURL = golden.appendingPathComponent("\(fixture)_ll_probs.f32le")
            guard let wav = AudioLoader.load16kMono(wavURL), let refProbs = try? readF32(probsURL) else {
                log("[8spk \(label)] FAIL: missing \(wavURL.path) or \(probsURL.path) (set N3D_FIXTURES / N3D_GOLDEN)")
                missing = true; continue
            }
            let refLogits = try? readF32(golden.appendingPathComponent("\(fixture)_ll_logits.f32le"))
            // the app decodes through AVAudioConverter; the gates read the PCM16 as soundfile does
            let pcm = (try? loadWav16kMono(wavURL)) ?? []
            let same = pcm.count == wav.count ? zip(pcm, wav).filter { $0.bitPattern == $1.bitPattern }.count : 0
            log("[8spk \(label)] wav \(wav.count) samples via AudioLoader; bit-equal to the PCM16 read in \(same) of \(pcm.count)")

            let w0 = ContinuousClock().now
            let (turns, out) = try await bridge.diarize(wav)
            let wall = secs(since: w0)
            let cmp = Comparison(logits: out.logits, probs: out.probs, refProbs: refProbs, refLogits: refLogits, tail: 0)
            let pass = cmp.agreement >= bar && cmp.oursFrames == cmp.refFrames
            let sg = cmp.segments
            log(String(format: "[8spk \(label)] frames %d vs %d  agreement@0.5 %@ (%d of %d differ)  max|Δp| %@%@  -> %@",
                       cmp.oursFrames, cmp.refFrames, pct(cmp.agreement), cmp.disagree, cmp.elements, fmt(cmp.maxAbsP),
                       cmp.maxAbsLogit.map { "  max|Δlogit| \(fmt($0))" } ?? "", pass ? "PASS" : "FAIL"))
            // the same bundle through conversion/nemotron3_diar/host_loop.py on this Mac's GPU (information)
            if let mac = try? readF32(golden.appendingPathComponent("pygpu/\(fixture)_ll_logits.f32le")),
               mac.count == out.logits.count {
                let d = maxAbsDiff(out.logits[...], mac[...])
                log("[8spk \(label)] vs host_loop.py on the Mac GPU: logits bit-equal in \(mac.count - d.unequal) of \(mac.count), max|Δlogit| \(fmt(d.max))")
            }
            let graphMs = percentile(out.graphSeconds.map { $0 * 1e3 }, 0.5)
            log(String(format: "[8spk \(label)] turns %d; segments ref/ours/matched %d/%d/%d (structural %d); wall %.3fs = %.1f× RT (mel+embed %.3fs, %d steps, graph %.2f ms/step median, %d compressions)",
                       turns.count, sg.nRef, sg.nOurs, sg.matched, sg.structural, wall, Double(wav.count) / 16000 / wall,
                       out.frontEndSeconds, out.steps, graphMs, out.compressions))
            for t in turns.prefix(6) {
                log(String(format: "      spk%d  %.2f–%.2fs", t.speaker, t.startSec, t.endSec))
            }
            let stepwise = try await stepwiseCheck(bridge, wav, out, turns, label: label, log: log)
            allPass = allPass && pass && stepwise
        }
        log(missing ? "[8spk] FAIL (golden or wav missing)" : allPass ? "[8spk] PASS" : "[8spk] CHECK (below the bar)")
        return missing ? 2 : allPass ? 0 : 3
    } catch { log("[8spk] FAIL: \(error)"); return 4 }
}

/// The playback-synced run's chunk-at-a-time loop (NemotronDiarizerStream) over the same clip, without waiting for
/// playback: its logits must be bit-equal to `diarize`'s (N3DDiarizer.process) and its turns the same.
private func stepwiseCheck(_ bridge: NemotronDiarizerBridge, _ samples: [Float], _ out: N3DOutput, _ turns: [SpeakerSegment],
                           label: String, log: (String) -> Void) async throws -> Bool {
    let stream = NemotronDiarizerStream(bridge: bridge, samples: samples)
    var probs: [Float] = []
    for _ in stream.chunks { probs.append(contentsOf: try await stream.step().probs) }
    let logits = await stream.logits
    let equal = logits.count == out.logits.count
        ? zip(logits, out.logits).filter { $0.bitPattern == $1.bitPattern }.count : 0
    let sameTurns = NemotronDiarizerBridge.turns(from: probs, frames: stream.clipFrames) == turns
    let pass = equal == out.logits.count && sameTurns
    log("[8spk \(label)] step by step (the playback-synced loop): \(stream.chunks.count) steps, logits bit-equal in \(equal) of \(out.logits.count), turns \(sameTurns ? "same" : "DIFFER")  -> \(pass ? "PASS" : "FAIL")")
    return pass
}

/// cos over the aligned [128, minT] region of two mel-major buffers.
private func melCos(_ a: [Float], _ ta: Int, _ b: [Float], _ tb: Int) -> Double {
    let t = min(ta, tb)
    var dot = 0.0, na = 0.0, nb = 0.0
    for m in 0..<128 {
        for i in 0..<t {
            let x = Double(a[m * ta + i]), y = Double(b[m * tb + i])
            dot += x * y; na += x * x; nb += y * y
        }
    }
    return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
}

/// activity-agree @ 0.5 + cos, comparing `frames[n][4]` to golden `[n*4]` frame-major over `n` frames.
private func agree(_ frames: [[Float]], _ gold: [Float], _ n: Int) -> (agree: Double, cos: Double) {
    var same = 0, tot = 0
    var dot = 0.0, na = 0.0, nb = 0.0
    for f in 0..<min(n, frames.count) {
        for s in 0..<4 {
            let g = gold[f * 4 + s], p = frames[f][s]
            if (p > 0.5) == (g > 0.5) { same += 1 }
            tot += 1
            dot += Double(p) * Double(g); na += Double(p) * Double(p); nb += Double(g) * Double(g)
        }
    }
    return (tot == 0 ? 0 : Double(same) / Double(tot), dot / (na.squareRoot() * nb.squareRoot() + 1e-12))
}

private func secs(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock().now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
}
