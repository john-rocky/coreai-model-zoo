// GateRunner — the decider-2b-vision gate on the DeciderVision library (VisionDecider: Pillow-order bicubic, the
// fixed-grid fp16w32 vision tower, the author's prompt, the int8mix_pf16 decoder in its chunk order, the letter
// softmax) with sideloaded JIT `.aimodel` assets, over the round-1 fixture (76 runs, 108 slots), on the iPhone or on the
// Mac. What the phone adds to the Mac run of the same library (round 8): the on-device JIT specialization, the cold and
// warm loads, footprint and headroom, thermal, and the time of one decision on the phone's own GPU.
// Results: <out>/result.json (rewritten after every stage and every few runs, "status" running -> done),
// result.log (one line per event), memory.tsv (every memory reading, written as taken); <out> = Documents/dv_gate on
// the iPhone. With DV_DUMP=1 (default) <out>/dump/ holds each run's full fp16 slot logits and tower output.
//
// Assets (<assets> = Library/Application Support/DeciderVisionAssets on the iPhone; DV_ASSETS on the Mac), laid out by
// ../_stage.sh and pushed by ../_install.sh:
//   decoder/          the decoder LanguageBundle: metadata.json, decider_2b_vision_decode_int8mix_pf16.aimodel (main
//                     S=1 + prefill S=16, int8 linears with layers 0/2/5 fp16), tokenizer/
//   towers/           decider_2b_vision_{g256,g448}_vision_fp16w32.aimodel (fp16 weights, fp32 compute): patches f32
//                     [4 G^2, 1536] -> image_embeds f32 [G^2, 2048]
//   fixtures/         rows.json, meta.json, images/, oracle_slim.json, mac_ref.json (Fixtures.swift)
//   MD5SUMS
//
// One tower at a time (the peak is the decoder plus one tower): the library fixes its towers when a VisionDecider is
// made, so each grid gets its own VisionDecider and the previous one is dropped first. Stages, in the order DV_STAGES
// gives (default all, in this order):
//   assets      MD5SUMS: every file present; md5 of every file up to 16 MB (the model files wait for "md5": reading
//               3.9 GB right before the first load would warm the file cache under the cold-load number)
//   load1       (a) the g256 tower alone (its cold specialization), dropped; (b) VisionDecider with no tower = the
//               tokenizer and the decoder (its cold specialization), dropped; (c) VisionDecider with the g256 tower
//               (decoder and tower now cached) = the decider of the g256 phase. Each step under the 100 ms memory
//               sampler, the Core AI cache sized between the steps. DV_LOAD_SPLIT=0: step (c) alone (everything cold).
//   warmup      one g256 decision (DV_WARMUP, default r01): the first call after the load
//   e2e_g256    the 35 image rows at g256: per run the ids, slots, letter logits, probabilities, argmax, full-vocabulary
//               top-1 against the oracle and the Mac, the sha256 of pixels / patches / tower output / slot logits, every
//               step's time; thermal, battery, footprint and headroom every 20 s
//   reset_g256  the phase's first run again: its slot logits bit-equal (the states are zeroed per row)
//   bench_g256  DV_BENCH_G256 (default r23, a 256x240 game frame): rest DV_BENCH_REST s (default 60), wait up to
//               DV_WAIT_NOMINAL s (default 300) for the thermal state nominal, then 1 warm-up + DV_BENCH_RUNS (default 5)
//               decisions back to back, each with its start offset (the 18 Pro's GPU slows after ~20 s of back-to-back
//               work)
//   load_g448   the g256 decider dropped; VisionDecider with the g448 tower: the decoder's warm reload, the g448 tower
//               cold
//   e2e_g448    the 35 image rows at g448
//   e2e_text    the 6 text rows (same decider)
//   reset_g448  the phase's first run (the first g448 run) again, bit-equal
//   bench_g448  DV_BENCH_G448 (default r23) at g448, then DV_BENCH_TEXT (default t06), each as bench_g256
//   load2       the decider dropped and VisionDecider with the g256 tower made again in this process (all warm), then
//               the warm-up row once more: bit-equal to its e2e run
//   md5         md5 of the files "assets" left for later (the model files)
//   load_aot    (not in the default list) DV_DECODER_AOT (a .aimodelc under the assets) loaded with
//               SpecializationOptions.default and one text decision (DV_BENCH_TEXT) on it
//
// Environment (devicectl device process launch --environment-variables on the iPhone; the shell on the Mac), all
// optional except DV_ASSETS on the Mac: DV_RUN_ID, DV_STAGES, DV_LIMIT (runs per arm), DV_WAIT_NOMINAL, DV_BENCH_RUNS,
// DV_BENCH_REST, DV_BENCH_G256, DV_BENCH_G448, DV_BENCH_TEXT, DV_WARMUP, DV_ASSETS, DV_OUT, DV_DECODER (bundle dir,
// default decoder), DV_TOWER_G256 / DV_TOWER_G448 (default the towers/ names), DV_DECODER_AOT, DV_LOAD_SPLIT (default 1),
// DV_DUMP (default 1), DV_MIN_FREE_GB (default 8), DV_EXIT_WHEN_DONE (Mac, default 1). Every DV_* variable is echoed into result.json (config.env).

import CoreAI
import CoreGraphics
import DeciderVision
import Foundation

struct GateConfig: Sendable {
    static let allStages = ["assets", "load1", "warmup", "e2e_g256", "reset_g256", "bench_g256", "load_g448", "e2e_g448",
                            "e2e_text", "reset_g448", "bench_g448", "load2", "md5"]

    let runID: String
    /// nil on a Mac without DV_ASSETS (the run stops with a fatal line)
    let assets: URL?
    let out: URL
    let stages: [String]
    let limit: Int
    let waitNominalSeconds: Double
    let benchRuns: Int
    let benchRest: Double
    let benchG256: String
    let benchG448: String
    let benchText: String
    let warmupRow: String
    let decoderPath: String
    let towerG256Path: String
    let towerG448Path: String
    let decoderAOTPath: String?
    let loadSplit: Bool
    /// load1 does not start below this much free space (a cold specialization that runs out of disk leaves partial
    /// caches behind: knowledge/pipelined-engine.md)
    let minFreeGB: Double
    let dump: Bool
    let exitWhenDone: Bool
    /// Files up to this many bytes are md5-checked in "assets"; the larger ones in "md5".
    let md5SmallLimit: Int
    let env: [String: String]

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        func path(_ s: String) -> URL { s.hasPrefix("/") ? URL(fileURLWithPath: s) : home.appendingPathComponent(s) }
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        #if os(iOS)
        let assets: URL? = env["DV_ASSETS"].map(path) ?? support.appendingPathComponent("DeciderVisionAssets")
        let out = env["DV_OUT"].map(path)
            ?? fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("dv_gate")
        let exitWhenDone = false
        #else
        let assets: URL? = env["DV_ASSETS"].map(path)
        let out = env["DV_OUT"].map(path) ?? support.appendingPathComponent("DeciderVisionGate/out")
        let exitWhenDone = env["DV_EXIT_WHEN_DONE"] != "0"
        #endif
        let stages = (env["DV_STAGES"].map { $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
            ?? allStages).filter { !$0.isEmpty }
        return GateConfig(
            runID: env["DV_RUN_ID"] ?? ISO8601DateFormatter().string(from: Date()),
            assets: assets, out: out, stages: stages,
            limit: max(0, Int(env["DV_LIMIT"] ?? "") ?? Int.max),
            waitNominalSeconds: max(0, Double(env["DV_WAIT_NOMINAL"] ?? "") ?? 300),
            benchRuns: max(1, Int(env["DV_BENCH_RUNS"] ?? "") ?? 5),
            benchRest: max(0, Double(env["DV_BENCH_REST"] ?? "") ?? 60),
            benchG256: env["DV_BENCH_G256"] ?? "r23",
            benchG448: env["DV_BENCH_G448"] ?? "r23",
            benchText: env["DV_BENCH_TEXT"] ?? "t06",
            warmupRow: env["DV_WARMUP"] ?? "r01",
            decoderPath: env["DV_DECODER"] ?? "decoder",
            towerG256Path: env["DV_TOWER_G256"] ?? "towers/decider_2b_vision_g256_vision_fp16w32.aimodel",
            towerG448Path: env["DV_TOWER_G448"] ?? "towers/decider_2b_vision_g448_vision_fp16w32.aimodel",
            decoderAOTPath: env["DV_DECODER_AOT"],
            loadSplit: env["DV_LOAD_SPLIT"] != "0",
            minFreeGB: Double(env["DV_MIN_FREE_GB"] ?? "") ?? 8,
            dump: env["DV_DUMP"] != "0",
            exitWhenDone: exitWhenDone,
            md5SmallLimit: 16 << 20,
            env: env.filter { $0.key.hasPrefix("DV_") })
    }

    var json: [String: Any] {
        ["assets": assets?.path ?? "(unset)", "out": out.path, "stages": stages, "limit": limit == Int.max ? -1 : limit,
         "wait_nominal_s": waitNominalSeconds, "bench_runs": benchRuns, "bench_rest_s": benchRest,
         "bench_rows": ["g256": benchG256, "g448": benchG448, "text": benchText], "warmup_row": warmupRow,
         "decoder": decoderPath, "tower_g256": towerG256Path, "tower_g448": towerG448Path,
         "decoder_aot": decoderAOTPath as Any, "load_split": loadSplit, "min_free_gb": minFreeGB, "dump": dump,
         "md5_small_limit_bytes": md5SmallLimit, "env": env]
    }
}

/// One scored run (for the arm summaries).
struct RunScore {
    let key: String
    let arm: String
    let slots: [Fixtures.SlotScore]
    let idsEqualOracle: Bool
    let slotsEqualOracle: Bool
    let idsEqualMac: Bool?
    let towerEqualMac: Bool?
    let slotLogitsEqualMac: Bool?
    let pixelsEqualMac: Bool?
    let decodedEqualMeta: Bool?
    let wallFromFile: Double
    let tower: Double?
    let decoder: Double
    let tStart: Double
}

actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0: ContinuousClock.Instant
    private let sink: LogSink
    private var memoryLog: MemoryLog?
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []
    private var fixtures: Fixtures?
    /// The decider in use and the grid of the one tower it holds (nil = no tower).
    private var decider: VisionDecider?
    private var deciderGrid: VisionDecider.Grid?
    /// Per run key: the scored run, its letter logits per slot (bit patterns) and, for the runs a later stage re-runs,
    /// the full slot logits.
    private var scores: [String: RunScore] = [:]
    private var letterBits: [String: [[UInt16]]] = [:]
    private var keptLogits: [String: [[Float16]]] = [:]
    private var phaseFirst: [String: String] = [:]        // phase ("g256" / "g448") -> its first e2e run key
    /// (path, md5) of the files "assets" left for "md5"
    private var deferredMD5: [(rel: String, sum: String)] = []
    private var assetsChecked = false

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
        let t0 = ContinuousClock.now
        self.t0 = t0
        sink = LogSink(t0: t0, emit: emit)
    }

    private var assets: URL { config.assets ?? URL(fileURLWithPath: "/nonexistent") }
    private var bundleURL: URL { assets.appendingPathComponent(config.decoderPath) }
    private func towerURL(_ g: VisionDecider.Grid) -> URL {
        assets.appendingPathComponent(g == .g256 ? config.towerG256Path : config.towerG448Path)
    }

    /// The Mac CLI's `--asset jit` options: the decoder GPU-preferred with frequent reshapes (its position ramp and
    /// the prefill width change per call), the towers GPU-preferred.
    static var decoderOptions: SpecializationOptions {
        var d = SpecializationOptions(preferredComputeUnitKind: .gpu)
        d.expectFrequentReshapes = true
        return d
    }
    static var towerOptions: SpecializationOptions { SpecializationOptions(preferredComputeUnitKind: .gpu) }

    static func describe(_ o: SpecializationOptions) -> String {
        if o == .default { return "SpecializationOptions.default" }
        let pref = o.preferredComputeUnitKind.map { "\($0)" } ?? "none"
        return "preferred \(pref), allowed \(o.allowedComputeUnitKinds.map { "\($0)" }.sorted()), expectFrequentReshapes \(o.expectFrequentReshapes)"
    }

    static var buildConfiguration: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }

    // MARK: - the run

    /// Every stage in order; true when every stage passed.
    func run() async -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: config.out, withIntermediateDirectories: true)
        let resultURL = config.out.appendingPathComponent("result.json")
        let logURL = config.out.appendingPathComponent("result.log")
        let memURL = config.out.appendingPathComponent("memory.tsv")
        try? fm.removeItem(at: resultURL)
        try? fm.removeItem(at: memURL)
        try? fm.removeItem(at: config.out.appendingPathComponent("dump"))
        fm.createFile(atPath: logURL.path, contents: nil)
        sink.open(logURL)
        memoryLog = MemoryLog(url: memURL, t0: t0)

        let launchIndex = bumpLaunchCount()
        var device = DeviceInfo.snapshot()
        let battery = await DeviceInfo.battery()
        device["battery_level"] = battery.level
        device["battery_state"] = battery.state
        device["power_source"] = DeviceInfo.powerSource(battery.state)
        device["footprint_mb"] = DeviceInfo.footprintMB()
        device["available_mb"] = DeviceInfo.availableMB()
        device["free_gb"] = DeviceInfo.freeGB(config.out)
        report = ["app": "DeciderVisionGate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device,
                  "model": "decider-2b-vision: decoder int8mix_pf16 (LanguageBundle, main S=1 + prefill S=16) + towers "
                      + "g256 / g448 fp16w32, JIT .aimodel, DeciderVision library (VisionDecider)",
                  "options": ["decoder": Self.describe(Self.decoderOptions), "towers": Self.describe(Self.towerOptions)],
                  "build": ["configuration": Self.buildConfiguration],
                  "bar": ["slots": "all equal the oracle's", "argmax": "all slots", "full_vocab_top1": "the oracle's argmax letter",
                          "max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002,
                          "mean_definition": "mean over runs of the run's mean |dp| over all its (slot, option) pairs "
                              + "(readout_gate_vision.py, round 4)", "reset": "bit-equal re-run per phase"],
                  "config": config.json]
        line("DeciderVisionGate run \(config.runID) (launch \(launchIndex) here), \(Self.buildConfiguration) build")
        line("device \(device["machine"] ?? "?") \(device["hw_model"] ?? "?"), \(device["os"] ?? "?") (build "
             + "\(device["os_build"] ?? "?")), Core AI arch \(device["coreai_architecture"] ?? "?"), thermal "
             + "\(DeviceInfo.thermal()), low power \(device["low_power_mode"] ?? "?"), battery "
             + "\(String(format: "%.0f", battery.level * 100)) % \(battery.state) (\(DeviceInfo.powerSource(battery.state))), "
             + "footprint \(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB, free "
             + "\(f1(DeviceInfo.freeGB(config.out))) GB")
        writeReport()

        guard config.assets != nil else {
            line("FATAL DV_ASSETS is not set (the Mac reads the stage directory from it)")
            return finish(ok: false, fatal: "DV_ASSETS not set")
        }
        #if os(macOS)
        // an iPhone AOT bundle (.h18p. / .h19p. ...) must never be loaded on a Mac
        for p in [config.decoderPath, config.towerG256Path, config.towerG448Path, config.decoderAOTPath ?? ""]
        where p.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil {
            line("FATAL refusing an iPhone AOT bundle on macOS: \(p)")
            return finish(ok: false, fatal: "iPhone AOT bundle on macOS: \(p)")
        }
        #endif
        do {
            let fx = try Fixtures(root: assets.appendingPathComponent("fixtures"))
            fixtures = fx
            report["fixtures"] = fx.files
            line("fixtures: \(fx.rows.count) rows (\(fx.runs(arm: "g256").count) image, \(fx.runs(arm: "text").count) text); "
                 + "oracle \(fx.oracle.count) runs, Mac reference \(fx.mac.count) runs")
        } catch {
            line("FATAL fixtures: \(error)")
            return finish(ok: false, fatal: "fixtures: \(error)")
        }

        var allOK = true
        for stage in config.stages {
            setStage(stage)
            let s0 = elapsed()
            var result: [String: Any]
            switch stage {
            case "assets": result = stageAssets()
            case "load1": result = await stageLoad1()
            case "warmup": result = await stageWarmup()
            case "e2e_g256": result = await stageE2E(arm: "g256")
            case "reset_g256": result = await stageReset(phase: "g256")
            case "bench_g256": result = await stageBench(rows: [(config.benchG256, "g256")])
            case "load_g448": result = await stageLoadGrid(.g448, key: "load_g448")
            case "e2e_g448": result = await stageE2E(arm: "g448")
            case "e2e_text": result = await stageE2E(arm: "text")
            case "reset_g448": result = await stageReset(phase: "g448")
            case "bench_g448": result = await stageBench(rows: [(config.benchG448, "g448"), (config.benchText, "text")])
            case "load2": result = await stageLoad2()
            case "md5": result = stageMD5()
            case "load_aot": result = await stageLoadAOT()
            default:
                line("unknown stage \(stage) (DV_STAGES takes \((GateConfig.allStages + ["load_aot"]).joined(separator: ", ")))")
                result = ["pass": false, "error": "unknown stage \(stage)"]
            }
            result["t_start_s"] = s0
            result["t_end_s"] = elapsed()
            stageResults[stage] = result
            stageOrder.append(stage)
            allOK = allOK && (result["pass"] as? Bool ?? false)
            report["e2e_summary"] = overallSummary()
            writeReport()
            if result["stop"] as? Bool == true {
                line("stopping after \(stage): the later stages need what it failed to provide")
                break
            }
        }
        if stageOrder.isEmpty {
            line("no stage ran")
            allOK = false
        }
        return finish(ok: allOK, fatal: nil)
    }

    // MARK: - assets

    func stageAssets() -> [String: Any] {
        let fm = FileManager.default
        let c0 = ContinuousClock.now
        var j: [String: Any] = ["dir": assets.path]
        guard let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) else {
            line("assets: no MD5SUMS in \(assets.path)")
            return ["dir": assets.path, "error": "no MD5SUMS", "pass": false, "stop": true]
        }
        var listed = 0, checked = 0, checkedBytes = 0, deferredBytes = 0
        var missing: [String] = [], mismatched: [String] = []
        var tops = Set<String>()
        deferredMD5 = []
        for row in text.split(separator: "\n") {
            let parts = row.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            listed += 1
            let sum = String(parts[0]), rel = parts[1].trimmingCharacters(in: .whitespaces)
            tops.insert(String(rel.split(separator: "/").first ?? ""))
            let url = assets.appendingPathComponent(rel)
            guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize else {
                missing.append(rel)
                continue
            }
            if size > config.md5SmallLimit {
                deferredMD5.append((rel, sum))
                deferredBytes += size
                continue
            }
            do {
                if try md5Hex(of: url) != sum { mismatched.append(rel) }
                checked += 1
                checkedBytes += size
            } catch {
                mismatched.append("\(rel) (\(error))")
            }
        }
        let names = ((try? fm.contentsOfDirectory(atPath: assets.path)) ?? []).sorted()
        let unlisted = names.filter { $0 != "MD5SUMS" && !tops.contains($0) }
        let dec = DeviceInfo.tree(bundleURL), t256 = DeviceInfo.tree(towerURL(.g256)), t448 = DeviceInfo.tree(towerURL(.g448))
        let required = [config.decoderPath + "/metadata.json", config.towerG256Path + "/main.mlirb",
                        config.towerG448Path + "/main.mlirb", "fixtures/rows.json"]
        let absent = required.filter { !fm.fileExists(atPath: assets.appendingPathComponent($0).path) }
        j["md5sums_listed"] = listed
        j["missing"] = missing
        j["md5_mismatch"] = mismatched
        j["md5_checked"] = checked
        j["md5_checked_bytes"] = checkedBytes
        j["md5_deferred"] = deferredMD5.map(\.rel)
        j["md5_deferred_bytes"] = deferredBytes
        j["unlisted_entries"] = unlisted
        j["required_absent"] = absent
        j["decoder_bytes"] = dec.bytes
        j["tower_g256_bytes"] = t256.bytes
        j["tower_g448_bytes"] = t448.bytes
        j["free_gb"] = DeviceInfo.freeGB(assets)
        j["seconds"] = seconds(since: c0)
        let ok = listed > 0 && missing.isEmpty && mismatched.isEmpty && absent.isEmpty
        assetsChecked = true
        line("assets \(assets.path): \(listed) files in MD5SUMS, \(missing.count) missing, md5 \(checked) checked "
             + "(\(mb(checkedBytes)) MB) with \(mismatched.count) different, \(deferredMD5.count) model files "
             + "(\(mb(deferredBytes)) MB) left for the md5 stage; decoder \(mb(dec.bytes)) MB, towers \(mb(t256.bytes)) / "
             + "\(mb(t448.bytes)) MB; unlisted \(unlisted); free \(f1(j["free_gb"] as? Double ?? -1)) GB")
        if !missing.isEmpty { line("assets: missing \(missing.prefix(8).joined(separator: ", "))") }
        if !mismatched.isEmpty { line("assets: md5 differs \(mismatched.prefix(8).joined(separator: ", "))") }
        if !absent.isEmpty { line("assets: required file absent \(absent.joined(separator: ", "))") }
        j["pass"] = ok
        if !ok { j["stop"] = true }
        return j
    }

    func stageMD5() -> [String: Any] {
        let c0 = ContinuousClock.now
        if !assetsChecked {
            if let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) {
                deferredMD5 = text.split(separator: "\n").compactMap { row in
                    let p = row.split(separator: " ", maxSplits: 1)
                    guard p.count == 2 else { return nil }
                    let rel = p[1].trimmingCharacters(in: .whitespaces)
                    let size = (try? assets.appendingPathComponent(rel).resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                    return size > config.md5SmallLimit ? (rel, String(p[0])) : nil
                }
            }
        }
        var mismatched: [String] = []
        var bytes = 0
        for (rel, sum) in deferredMD5 {
            let url = assets.appendingPathComponent(rel)
            do {
                if try md5Hex(of: url) != sum { mismatched.append(rel) }
                bytes += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            } catch {
                mismatched.append("\(rel) (\(error))")
            }
        }
        let s = seconds(since: c0)
        line("md5: \(deferredMD5.count) model files, \(mb(bytes)) MB in \(f1(s)) s, \(mismatched.count) different"
             + (mismatched.isEmpty ? "" : ": \(mismatched.joined(separator: ", "))"))
        return ["files": deferredMD5.map(\.rel), "bytes": bytes, "md5_mismatch": mismatched, "seconds": s,
                "pass": !deferredMD5.isEmpty && mismatched.isEmpty]
    }

    // MARK: - loads

    /// The Core AI cache of this app at `at` into `storage`; its bytes and files.
    private func cacheSnapshot(_ at: String, _ storage: inout [String: Any]) -> (bytes: Int, files: Int) {
        let s = DeviceInfo.storageSnapshot()
        storage[at] = s
        let c = s["coreai_cache"] as? [String: Any] ?? [:]
        return (c["bytes"] as? Int ?? 0, c["files"] as? Int ?? 0)
    }

    /// What a VisionDecider's init reported about its parts.
    private func loadRecord(_ d: VisionDecider, wall: Double) -> [String: Any] {
        var towers: [String: Double] = [:]
        for (g, t) in d.towers { towers["\(g)"] = t.loadSeconds }
        return ["wall_s": wall, "tokenizer_s": d.tokenizerLoadSeconds,
                "decoder_s": ["model": d.decoder.loadSeconds.model, "main": d.decoder.loadSeconds.main,
                              "prefill": d.decoder.loadSeconds.prefill as Any,
                              "total": d.decoder.loadSeconds.model + d.decoder.loadSeconds.main + (d.decoder.loadSeconds.prefill ?? 0)],
                "tower_s": towers, "function_names": d.decoder.functionNames, "prefill_chunk": d.decoder.chunk as Any]
    }

    private func loadLine(_ what: String, _ r: [String: Any], _ memory: [String: Any]) -> String {
        let dec = r["decoder_s"] as? [String: Any] ?? [:]
        let towers = (r["tower_s"] as? [String: Double] ?? [:]).sorted { $0.key < $1.key }
            .map { "tower \($0.key) \(f2($0.value)) s" }.joined(separator: ", ")
        return "\(what): wall \(f2(r["wall_s"] as? Double ?? -1)) s = tokenizer \(f2(r["tokenizer_s"] as? Double ?? -1)) s"
            + (towers.isEmpty ? "" : ", \(towers)")
            + ", decoder \(f2(dec["total"] as? Double ?? -1)) s (AIModel \(f2(dec["model"] as? Double ?? -1)), main "
            + "\(f2(dec["main"] as? Double ?? -1)), prefill \(f2(dec["prefill"] as? Double ?? -1))) | peak footprint "
            + "\(f1(memory["peak_footprint_mb"] as? Double ?? -1)) MB, least available "
            + "\(f1(memory["min_available_mb"] as? Double ?? -1)) MB, thermal \(DeviceInfo.thermal())"
    }

    /// Drops the decider in use (and its tower) and waits a moment for the memory to go.
    private func dropDecider(_ j: inout [String: Any], key: String) async {
        let had = decider != nil
        j["\(key)_footprint_mb_with_decider"] = DeviceInfo.footprintMB()
        decider = nil
        deciderGrid = nil
        try? await Task.sleep(for: .seconds(2))
        j["\(key)_dropped_decider"] = had
        j["\(key)_footprint_mb_after_drop"] = DeviceInfo.footprintMB()
        j["\(key)_available_mb_after_drop"] = DeviceInfo.availableMB()
    }

    /// A VisionDecider holding one tower (or none), under the memory sampler.
    private func makeDecider(_ grid: VisionDecider.Grid?, label: String, decoderAsset: URL? = nil,
                             decoderOptions: SpecializationOptions = GateRunner.decoderOptions)
        async -> (Result<VisionDecider, Error>, [String: Any], Double)
    {
        let bundle = bundleURL
        let towers: [VisionDecider.Grid: URL] = grid.map { [$0: towerURL($0)] } ?? [:]
        let towerOptions = Self.towerOptions
        return await sampled(label) {
            try await VisionDecider(bundle: bundle, decoder: decoderAsset, towers: towers,
                                    decoderOptions: decoderOptions, towerOptions: towerOptions)
        }
    }

    func stageLoad1() async -> [String: Any] {
        var j: [String: Any] = ["bundle": bundleURL.path, "tower_g256": towerURL(.g256).path, "split": config.loadSplit]
        var thermal: [[String: Any]] = []
        func mark(_ at: String) { thermal.append(["at": at, "state": DeviceInfo.thermal(), "t_s": elapsed()]) }
        var storage: [String: Any] = [:]
        mark("start")
        let dec = DeviceInfo.tree(bundleURL), tw = DeviceInfo.tree(towerURL(.g256))
        j["decoder_mb"] = Double(dec.bytes) / 1e6
        j["tower_g256_mb"] = Double(tw.bytes) / 1e6
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["footprint_mb_before"] = DeviceInfo.footprintMB()
        j["available_mb_before"] = DeviceInfo.availableMB()
        let b0 = await DeviceInfo.battery()
        j["battery_start"] = ["level": b0.level, "state": b0.state, "power": DeviceInfo.powerSource(b0.state)]
        line("load1: decoder bundle \(mb(dec.bytes)) MB, g256 tower \(mb(tw.bytes)) MB; Core AI cache \(mb(cache0.bytes)) MB "
             + "in \(cache0.files) files; footprint \(f1(DeviceInfo.footprintMB())) MB, available "
             + "\(f1(DeviceInfo.availableMB())) MB; battery \(String(format: "%.0f", b0.level * 100)) % \(b0.state) "
             + "(\(DeviceInfo.powerSource(b0.state))), thermal \(DeviceInfo.thermal())")
        let free = DeviceInfo.freeGB(assets)
        j["free_gb_before"] = free
        if free >= 0 && free < config.minFreeGB {
            line("load1: STOP free space \(f1(free)) GB < DV_MIN_FREE_GB \(f1(config.minFreeGB)) GB: the cold specializations "
                 + "(~7 GB of cache for the decoder and both towers) would run out of disk")
            j["error"] = "free space \(free) GB < \(config.minFreeGB) GB"
            j["pass"] = false
            j["stop"] = true
            return j
        }
        var step = "tower_g256_alone"
        do {
            if config.loadSplit {
                // (a) the g256 tower alone: its first specialization in this container
                j["step"] = step
                j["storage"] = storage
                writePartial("load1", j, thermal)
                let url = towerURL(.g256), options = Self.towerOptions
                let (tLoaded, tMemory, tWall) = await sampled("load1 tower g256 alone") {
                    try await VisionTower(contentsOf: url, grid: VisionDecider.Grid.g256.side, options: options)
                }
                var tower: VisionTower? = try tLoaded.get()
                j["tower_g256_alone"] = ["wall_s": tWall, "load_s": tower?.loadSeconds as Any, "memory": tMemory]
                tower = nil
                mark("after tower alone")
                let cache1 = cacheSnapshot("after_tower_alone", &storage)
                j["cache_bytes_after_tower_alone"] = cache1.bytes
                line("load1 (a) g256 tower alone: \(f2(tWall)) s, peak footprint \(f1(tMemory["peak_footprint_mb"] as? Double ?? -1)) "
                     + "MB, least available \(f1(tMemory["min_available_mb"] as? Double ?? -1)) MB; Core AI cache "
                     + "\(mb(cache1.bytes)) MB; thermal \(DeviceInfo.thermal())")

                // (b) the decoder (and the tokenizer), no tower: the decoder's first specialization
                step = "decoder_alone"
                j["step"] = step
                j["storage"] = storage
                writePartial("load1", j, thermal)
                let (dLoaded, dMemory, dWall) = await makeDecider(nil, label: "load1 decoder alone")
                var d: VisionDecider? = try dLoaded.get()
                var rec = loadRecord(d!, wall: dWall)
                rec["memory"] = dMemory
                j["decoder_alone"] = rec
                j["decoder_descriptors"] = d!.decoder.descriptors
                d = nil
                mark("after decoder alone")
                line("load1 (b) " + loadLine("decoder alone (cold)", rec, dMemory))
                try? await Task.sleep(for: .seconds(2))
                let cache2 = cacheSnapshot("after_decoder_alone", &storage)
                j["cache_bytes_after_decoder_alone"] = cache2.bytes
                j["footprint_mb_after_decoder_dropped"] = DeviceInfo.footprintMB()
                line("load1: Core AI cache \(mb(cache2.bytes)) MB after the decoder; footprint with it dropped "
                     + "\(f1(DeviceInfo.footprintMB())) MB")
            }

            // (c) the decider of the g256 phase: decoder + g256 tower (both cached when split, both cold when not)
            step = "decider_g256"
            j["step"] = step
            j["storage"] = storage
            writePartial("load1", j, thermal)
            let (loaded, memory, wall) = await makeDecider(.g256, label: "load1 decider g256")
            let d = try loaded.get()
            decider = d
            deciderGrid = .g256
            var rec = loadRecord(d, wall: wall)
            rec["memory"] = memory
            j["decider_g256"] = rec
            j["tower_descriptors"] = d.towers.reduce(into: [String: Any]()) { $0["\($1.key)"] = $1.value.descriptor }
            if j["decoder_descriptors"] == nil { j["decoder_descriptors"] = d.decoder.descriptors }
            mark("after decider")
            line("load1 (c) " + loadLine("decider g256 (\(config.loadSplit ? "decoder and tower cached" : "cold"))", rec, memory))
            let cache3 = cacheSnapshot("after_decider", &storage)
            j["cache_bytes_after_decider"] = cache3.bytes
            j["footprint_mb_after"] = DeviceInfo.footprintMB()
            j["available_mb_after"] = DeviceInfo.availableMB()
            step = "done"
            j["step"] = step
            j["pass"] = true
        } catch {
            line("load1: ERROR at \(step): \(error)")
            j["error"] = "\(error)"
            j["error_step"] = step
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
            j["stop"] = true
        }
        mark("end")
        let b1 = await DeviceInfo.battery()
        j["battery_end"] = ["level": b1.level, "state": b1.state, "power": DeviceInfo.powerSource(b1.state)]
        j["storage"] = storage
        j["thermal"] = thermal
        return j
    }

    /// Drop the decider in use, then a VisionDecider with `grid`'s tower (the decoder's reload is warm once load1 ran).
    func stageLoadGrid(_ grid: VisionDecider.Grid, key: String) async -> [String: Any] {
        var j: [String: Any] = ["grid": "\(grid)", "tower": towerURL(grid).path]
        var storage: [String: Any] = [:]
        j["previous_grid"] = deciderGrid.map { "\($0)" } as Any
        await dropDecider(&j, key: "before")
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["thermal_start"] = DeviceInfo.thermal()
        j["step"] = "decider_\(grid)"
        j["storage"] = storage
        writePartial(key, j, [])
        let (loaded, memory, wall) = await makeDecider(grid, label: "\(key) decider \(grid)")
        do {
            let d = try loaded.get()
            decider = d
            deciderGrid = grid
            var rec = loadRecord(d, wall: wall)
            rec["memory"] = memory
            j["decider"] = rec
            let cache1 = cacheSnapshot("after", &storage)
            j["cache_bytes_after"] = cache1.bytes
            j["footprint_mb_after"] = DeviceInfo.footprintMB()
            j["available_mb_after"] = DeviceInfo.availableMB()
            j["step"] = "done"
            j["pass"] = true
            line("\(key) " + loadLine("decider \(grid) (decoder reload, \(grid) tower)", rec, memory)
                 + "; Core AI cache \(mb(cache0.bytes)) -> \(mb(cache1.bytes)) MB")
        } catch {
            line("\(key): ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
            j["stop"] = true
        }
        let b = await DeviceInfo.battery()
        j["battery"] = ["level": b.level, "state": b.state, "power": DeviceInfo.powerSource(b.state)]
        j["storage"] = storage
        j["thermal_end"] = DeviceInfo.thermal()
        return j
    }

    /// Everything dropped and the g256 decider made again in this process (warm), then the warm-up row once more.
    func stageLoad2() async -> [String: Any] {
        var j = await stageLoadGrid(.g256, key: "load2")
        guard j["pass"] as? Bool == true, let d = decider, let fx = fixtures else { return j }
        let run = Fixtures.Run(id: config.warmupRow, arm: "g256")
        do {
            let (rec, logits, _) = try await decide(d, run: run, fx: fx, tStart: 0, dumpTag: nil)
            j["check_run"] = rec
            if let want = keptLogits[run.key] {
                let equal = Self.bitEqual(logits, want)
                j["check_bit_equal_e2e"] = equal
                line("load2 check \(run.key): slot logits bit-equal to its e2e run: \(equal)")
            }
        } catch {
            j["check_error"] = "\(error)"
            line("load2 check: ERROR \(error)")
        }
        return j
    }

    /// DV_DECODER_AOT (a .aimodelc under the assets) with SpecializationOptions.default, then one text decision.
    func stageLoadAOT() async -> [String: Any] {
        guard let rel = config.decoderAOTPath else { return ["pass": false, "error": "DV_DECODER_AOT not set"] }
        let url = assets.appendingPathComponent(rel)
        var j: [String: Any] = ["decoder": url.path, "bytes": DeviceInfo.tree(url).bytes]
        await dropDecider(&j, key: "before")
        j["step"] = "load"
        writePartial("load_aot", j, [])
        let (loaded, memory, wall) = await makeDecider(nil, label: "load_aot", decoderAsset: url, decoderOptions: .default)
        j["memory"] = memory
        j["wall_s"] = wall
        do {
            let d = try loaded.get()
            j["load"] = loadRecord(d, wall: wall)
            line("load_aot " + loadLine("AOT decoder \(url.lastPathComponent)", j["load"] as! [String: Any], memory))
            if let fx = fixtures {
                let run = Fixtures.Run(id: config.benchText, arm: "text")
                let (rec, _, _) = try await decide(d, run: run, fx: fx, tStart: 0, dumpTag: nil)
                j["check_run"] = rec
            }
            j["pass"] = true
        } catch {
            line("load_aot: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
        }
        return j
    }

    /// The decider holding `grid`'s tower (text rows ride any decider), made here (untimed) when no load stage made it.
    private func ensureDecider(_ arm: String, _ j: inout [String: Any]) async throws -> VisionDecider {
        if let d = decider, arm == "text" || deciderGrid.map({ "\($0)" }) == arm { return d }
        let grid: VisionDecider.Grid = arm == "g448" ? .g448 : .g256
        if decider != nil { await dropDecider(&j, key: "ensure") }
        let (loaded, memory, wall) = await makeDecider(grid, label: "decider made here \(grid)")
        let d = try loaded.get()
        decider = d
        deciderGrid = grid
        j["decider_made_here"] = loadRecord(d, wall: wall).merging(["memory": memory]) { _, n in n }
        line("decider \(grid) made here (no load stage before this one): \(f2(wall)) s")
        return d
    }

    // MARK: - decisions

    /// One run through `trace`, scored against the oracle and the Mac: (record, full slot logits, score).
    private func decide(_ d: VisionDecider, run: Fixtures.Run, fx: Fixtures, tStart: Double, dumpTag: String?)
        async throws -> ([String: Any], [[Float16]], RunScore)
    {
        guard let r = fx.rows[run.id] else { throw GateError.fixture("no row \(run.id)") }
        let questions = r.questions.map { DecisionQuestion(text: $0.text, options: $0.options) }
        var image: CGImage? = nil
        var fileSeconds = 0.0
        if run.arm != "text" {
            guard let name = r.image else { throw GateError.fixture("\(run.id) has no image") }
            let t = ContinuousClock.now
            image = try ImagePreprocess.loadCGImage(url: fx.imageURL(name))
            fileSeconds = seconds(since: t)
        }
        let grid: VisionDecider.Grid = run.arm == "g448" ? .g448 : .g256
        let tr = try await d.trace(image: image, context: r.context, questions: questions, grid: grid, keepLogits: true,
                                   useChunks: true)
        let wall = (tr.seconds["wall"] ?? 0) + fileSeconds
        let logits = tr.slotLogits ?? []
        let o = fx.oracle[run.key], m = fx.mac[run.key]
        var answers: [[String: Any]] = []
        var slotScores: [Fixtures.SlotScore] = []
        for (s, a) in tr.answers.enumerated() {
            let sc = Fixtures.scoreSlot(a, slot: s, oracle: o, mac: m)
            slotScores.append(sc)
            var aj: [String: Any] = ["t": a.slot, "letter_logits": a.letterLogits.map { Double(Float($0)) },
                                     "letter_logits_bits": a.letterLogits.map { Int($0.bitPattern) },
                                     "probs": a.probabilities, "argmax": a.argmax, "full_vocab_top1_id": a.fullVocabTop1,
                                     "full_vocab_top1_logit": Double(Float(a.fullVocabTop1Logit)),
                                     "full_vocab_top5_ids": a.fullVocabTop5, "finite": a.finite, "read_from": a.readFrom]
            aj.merge(sc.json) { _, n in n }
            answers.append(aj)
        }
        let calls = ["prefill": tr.pass.callIsPrefill.filter { $0 }.count, "main": tr.pass.callIsPrefill.filter { !$0 }.count,
                     "total": tr.pass.callIsPrefill.count]
        let flat = logits.flatMap { $0 }
        let slotSHA = sha256Hex(of: flat)
        let ids32 = tr.row.ids.map { Int32($0) }
        let procIDs = Fixtures.processorIDs(tr.row.ids)
        var j: [String: Any] = [
            "id": run.id, "arm": run.arm, "image": r.image as Any, "grid": tr.row.grid.map { [$0, $0] } as Any,
            "tokens": tr.row.ids.count, "ids": tr.row.ids, "ids_sha256": sha256Hex(of: ids32), "slots": tr.row.slots,
            "nopts": tr.row.optionCounts, "rope_shift_start": Int(tr.row.ropeShiftStart),
            "rope_shift_amount": Int(tr.row.ropeShiftAmount), "answers": answers, "calls": calls,
            "call_ms": tr.pass.callSeconds.map { $0 * 1e3 }, "call_is_prefill": tr.pass.callIsPrefill,
            "state_reset_ms": tr.pass.resetSeconds * 1e3, "seconds": tr.seconds, "image_file_decode_s": fileSeconds,
            "wall_from_file_s": wall, "slot_logits_sha256": slotSHA, "t_start_s": tStart,
            "footprint_mb": DeviceInfo.footprintMB(), "available_mb": DeviceInfo.availableMB(), "thermal": DeviceInfo.thermal(),
        ]
        var decodedEqualMeta: Bool? = nil, pixelsEqualMac: Bool? = nil
        if let p = tr.prepared {
            let decodedSHA = sha256Hex(of: p.decoded.pixels), resizedSHA = sha256Hex(of: p.resized.pixels)
            let patchesSHA = sha256Hex(of: p.patches)
            j["image_size_in"] = [p.decoded.width, p.decoded.height]
            j["decode_path"] = p.decoded.decodePath
            j["decoded_rgb_sha256"] = decodedSHA
            j["resized_rgb_sha256"] = resizedSHA
            j["patches_sha256"] = patchesSHA
            if let name = r.image, let want = fx.rgbSHA[name] { decodedEqualMeta = want == decodedSHA }
            if let m { pixelsEqualMac = m.resized_rgb_sha256 == resizedSHA && m.patches_sha256 == patchesSHA }
            j["decoded_rgb_sha256_equals_meta"] = decodedEqualMeta as Any
            j["pixels_and_patches_equal_mac"] = pixelsEqualMac as Any
        }
        var towerEqualMac: Bool? = nil
        if let e = tr.towerEmbeds {
            let sha = sha256Hex(of: e)
            j["tower_embeds_sha256"] = sha
            if let m { towerEqualMac = m.tower_embeds_sha256 == sha }
            j["tower_embeds_equal_mac"] = towerEqualMac as Any
        }
        let idsEqualOracle = o.map { $0.ids == procIDs } ?? false
        let slotsEqualOracle = o.map { $0.slot_idx == tr.row.slots } ?? false
        let idsEqualMac = m.map { $0.ids == tr.row.ids }
        let slotEqualMac = m?.slot_logits_sha256.map { $0 == slotSHA }
        j["ids_equal_oracle"] = idsEqualOracle
        j["slots_equal_oracle"] = slotsEqualOracle
        j["ids_equal_mac"] = idsEqualMac as Any
        j["slot_logits_equal_mac"] = slotEqualMac as Any
        if let m, let mw = m.wall_from_file_s { j["mac_wall_from_file_s"] = mw }
        if config.dump, let tag = dumpTag {
            let dir = config.out.appendingPathComponent("dump")
            try? FileManager.default.createDirectory(at: dir.appendingPathComponent("logits"), withIntermediateDirectories: true)
            try? FileManager.default.createDirectory(at: dir.appendingPathComponent("embeds"), withIntermediateDirectories: true)
            flat.withUnsafeBytes { raw in try? Data(raw).write(to: dir.appendingPathComponent("logits/\(tag).logits.f16")) }
            j["logits_file"] = "dump/logits/\(tag).logits.f16"
            if let e = tr.towerEmbeds {
                e.withUnsafeBytes { raw in try? Data(raw).write(to: dir.appendingPathComponent("embeds/\(tag).embeds.f32")) }
                j["embeds_file"] = "dump/embeds/\(tag).embeds.f32"
            }
        }
        let score = RunScore(key: run.key, arm: run.arm, slots: slotScores, idsEqualOracle: idsEqualOracle,
                             slotsEqualOracle: slotsEqualOracle, idsEqualMac: idsEqualMac, towerEqualMac: towerEqualMac,
                             slotLogitsEqualMac: slotEqualMac, pixelsEqualMac: pixelsEqualMac,
                             decodedEqualMeta: decodedEqualMeta, wallFromFile: wall, tower: tr.seconds["tower"],
                             decoder: tr.seconds["decoder"] ?? 0, tStart: tStart)
        return (j, logits, score)
    }

    private static func bitEqual(_ a: [[Float16]], _ b: [[Float16]]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { x, y in
            x.count == y.count && zip(x, y).allSatisfy { $0.bitPattern == $1.bitPattern }
        }
    }

    private func answerText(_ rec: [String: Any]) -> String {
        (rec["answers"] as? [[String: Any]] ?? []).map { a -> String in
            let p = (a["probs"] as? [Double] ?? []).map { String(format: "%.4f", $0) }.joined(separator: " ")
            let o = a["oracle"] as? [String: Any], m = a["mac"] as? [String: Any]
            return "[\(p)] -> \(PromptBuilder.letters[a["argmax"] as? Int ?? 0])"
                + (o.map { " |dp| \(String(format: "%.4f", $0["max_abs_dp"] as? Double ?? .nan))" } ?? "")
                + (m.map { " mac \(($0["letter_logits_bit_equal"] as? Bool) == true ? "=" : "~\(String(format: "%.4f", $0["max_abs_dp"] as? Double ?? .nan))")" } ?? "")
        }.joined(separator: " ")
    }

    /// [t s, thermal, battery level, battery state, footprint MB, available MB]
    private func timelinePoint(_ t: Double) async -> [Any] {
        let b = await DeviceInfo.battery()
        return [(t * 10).rounded() / 10, DeviceInfo.thermal(), b.level, b.state, DeviceInfo.footprintMB(), DeviceInfo.availableMB()]
    }

    func stageWarmup() async -> [String: Any] {
        var j: [String: Any] = ["row": config.warmupRow, "arm": "g256"]
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        do {
            let d = try await ensureDecider("g256", &j)
            j["thermal_start"] = DeviceInfo.thermal()
            let b = await DeviceInfo.battery()
            let run = Fixtures.Run(id: config.warmupRow, arm: "g256")
            let (result, memory, _) = await sampled("warmup") { try await self.decide(d, run: run, fx: fx, tStart: 0, dumpTag: nil) }
            let (rec, logits, score) = try result.get()
            keptLogits["warmup/\(run.key)"] = logits
            j["run"] = rec
            j["memory"] = memory
            j["battery"] = ["level": b.level, "state": b.state, "power": DeviceInfo.powerSource(b.state)]
            j["thermal_end"] = DeviceInfo.thermal()
            let ok = score.slots.allSatisfy(\.argmaxEqual) && score.idsEqualOracle
            j["pass"] = ok
            line("warmup \(run.key): wall \(f1(score.wallFromFile * 1e3)) ms (first decision after the load; tower "
                 + "\(f1((score.tower ?? 0) * 1e3)) ms, decoder \(f1(score.decoder * 1e3)) ms), thermal \(DeviceInfo.thermal()), "
                 + "battery \(String(format: "%.0f", b.level * 100)) % \(DeviceInfo.powerSource(b.state)) | \(answerText(rec))")
        } catch {
            line("warmup: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.describe(error)
            j["pass"] = false
            j["stop"] = decider == nil
        }
        return j
    }

    func stageE2E(arm: String) async -> [String: Any] {
        var j: [String: Any] = ["arm": arm]
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        let runs = Array(fx.runs(arm: arm).prefix(config.limit))
        j["runs_planned"] = runs.count
        let d: VisionDecider
        do {
            d = try await ensureDecider(arm, &j)
        } catch {
            line("e2e_\(arm): ERROR making the decider: \(error)")
            return ["error": "\(error)", "error_detail": Self.describe(error), "pass": false, "stop": true]
        }
        j["decider_grid"] = deciderGrid.map { "\($0)" } as Any
        j["thermal_start"] = DeviceInfo.thermal()
        line("e2e_\(arm): \(runs.count) runs on the \(deciderGrid.map { "\($0)" } ?? "tower-less") decider")
        var rows: [[String: Any]] = []
        var done: [RunScore] = []
        var errors = 0
        var timeline: [[Any]] = [await timelinePoint(0)]
        var nextTimeline = 20.0
        // the first e2e phase keeps every 100 ms reading of its first 60 s in the record (all of them go to memory.tsv)
        let sampler = MemorySampler("e2e_\(arm)", fullSeconds: arm == "g256" ? 60 : 0, timelineEvery: 20,
                                    memoryLog: memoryLog, progress: { [sink] in sink.line($0) })
        sampler.start()
        let e0 = ContinuousClock.now
        for (i, run) in runs.enumerated() {
            let tStart = seconds(since: e0)
            if tStart >= nextTimeline {
                timeline.append(await timelinePoint(tStart))
                while nextTimeline <= tStart { nextTimeline += 20 }
            }
            do {
                let (rec, logits, score) = try await decide(d, run: run, fx: fx, tStart: tStart,
                                                            dumpTag: "\(run.id)__\(run.arm)")
                // the text rows ride the g448 phase: its reset proof re-runs the phase's first run (the first g448 run)
                let phase = arm == "text" ? "g448" : arm
                if i == 0 && phaseFirst[phase] == nil { phaseFirst[phase] = run.key }
                if i == 0 || run.id == config.warmupRow { keptLogits[run.key] = logits }
                letterBits[run.key] = score.slots.indices.map { s in
                    ((rec["answers"] as? [[String: Any]])?[s]["letter_logits_bits"] as? [Int] ?? []).map { UInt16($0) }
                }
                scores[run.key] = score
                done.append(score)
                rows.append(rec)
                let macWall = (rec["mac_wall_from_file_s"] as? Double).map { " (Mac \(f1($0 * 1e3)))" } ?? ""
                line("e2e \(i + 1)/\(runs.count) \(run.key): \(rec["tokens"] ?? 0) tok, wall \(f1(score.wallFromFile * 1e3)) ms"
                     + macWall + " (tower \(f1((score.tower ?? 0) * 1e3)), decoder \(f1(score.decoder * 1e3))) | "
                     + answerText(rec) + (score.towerEqualMac == true ? " | tower = mac" : (score.towerEqualMac == false ? " | tower ≠ mac" : ""))
                     + (score.slotLogitsEqualMac == true ? ", logits = mac" : ""))
            } catch {
                errors += 1
                rows.append(["id": run.id, "arm": run.arm, "error": "\(error)", "error_detail": Self.describe(error)])
                line("e2e \(i + 1)/\(runs.count) \(run.key): ERROR \(error)")
            }
            if (i + 1) % 5 == 0 || i + 1 == runs.count {
                j["runs"] = rows
                j["summary"] = Self.summarize(done)
                j["timeline"] = timeline
                writePartial("e2e_\(arm)", j, [])
            }
            if errors >= 3 && done.isEmpty {
                line("e2e_\(arm): 3 errors and no run done: stopping the stage")
                break
            }
        }
        timeline.append(await timelinePoint(seconds(since: e0)))
        let memory = sampler.stop()
        let summary = Self.summarize(done)
        j["runs"] = rows
        j["summary"] = summary
        j["errors"] = errors
        j["memory"] = memory
        j["timeline"] = timeline
        j["timeline_columns"] = ["t_s", "thermal", "battery_level", "battery_state", "footprint_mb", "available_mb"]
        j["thermal_end"] = DeviceInfo.thermal()
        line("e2e_\(arm) every 20 s (t, thermal, battery, footprint): " + timeline.map {
            "\($0[0])s \($0[1]) \(String(format: "%.0f", ($0[2] as? Double ?? -1) * 100))% \($0[3]) \(String(format: "%.0f", $0[4] as? Double ?? -1)) MB"
        }.joined(separator: ", "))
        for l in Self.summaryLines("e2e_\(arm)", summary) { line(l) }
        j["pass"] = errors == 0 && done.count == runs.count && (summary["bar_pass"] as? Bool ?? false)
        return j
    }

    /// The phase's first e2e run again, after every other run of the phase went through the same states.
    func stageReset(phase: String) async -> [String: Any] {
        var j: [String: Any] = ["phase": phase]
        guard let fx = fixtures, let key = phaseFirst[phase], let want = keptLogits[key] else {
            return ["error": "no first run recorded for phase \(phase)", "pass": false]
        }
        let parts = key.split(separator: "/").map(String.init)
        let run = Fixtures.Run(id: parts[0], arm: parts[1])
        do {
            let d = try await ensureDecider(run.arm, &j)
            let (rec, logits, _) = try await decide(d, run: run, fx: fx, tStart: 0, dumpTag: nil)
            let equal = Self.bitEqual(logits, want)
            var maxDiff: Float = 0
            for (a, b) in zip(logits, want) { for (x, y) in zip(a, b) { maxDiff = max(maxDiff, abs(Float(x) - Float(y))) } }
            j["run"] = key
            j["bit_equal"] = equal
            j["slot_logits_max_abs_diff"] = Double(maxDiff)
            j["wall_from_file_s"] = rec["wall_from_file_s"] ?? 0
            j["thermal"] = DeviceInfo.thermal()
            j["pass"] = equal
            line("reset_\(phase) \(key) again: full slot logits bit-equal \(equal) (max |diff| \(maxDiff)), wall "
                 + "\(f1((rec["wall_from_file_s"] as? Double ?? 0) * 1e3)) ms")
        } catch {
            line("reset_\(phase): ERROR \(error)")
            j["error"] = "\(error)"
            j["pass"] = false
        }
        return j
    }

    /// Per row: rest, wait for nominal, 1 warm-up + benchRuns decisions back to back with their start offsets.
    func stageBench(rows benchRows: [(String, String)]) async -> [String: Any] {
        var j: [String: Any] = ["runs_timed": config.benchRuns, "rest_s": config.benchRest]
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        var out: [String: Any] = [:]
        var ok = true
        for (id, arm) in benchRows {
            var r: [String: Any] = ["row": id, "arm": arm]
            do {
                let d = try await ensureDecider(arm, &r)
                let run = Fixtures.Run(id: id, arm: arm)
                if config.benchRest > 0 {
                    line("bench \(run.key): resting \(Int(config.benchRest)) s before the burst (thermal \(DeviceInfo.thermal()))")
                    try? await Task.sleep(for: .seconds(config.benchRest))
                }
                if config.waitNominalSeconds > 0 { r["wait_nominal"] = await waitForNominal("bench \(run.key)") }
                let b0 = await DeviceInfo.battery()
                r["thermal_start"] = DeviceInfo.thermal()
                r["battery_start"] = ["level": b0.level, "state": b0.state, "power": DeviceInfo.powerSource(b0.state)]
                var decisions: [[String: Any]] = []
                var timed: [RunScore] = []
                var sameAsE2E = 0
                let c0 = ContinuousClock.now
                var lastEnd = 0.0
                for k in 0...config.benchRuns {                     // k = 0: the warm-up decision, not in the statistics
                    let tStart = seconds(since: c0)
                    let (rec, _, score) = try await decide(d, run: run, fx: fx, tStart: tStart, dumpTag: nil)
                    let tEnd = seconds(since: c0)
                    let b = await DeviceInfo.battery()
                    let bits = (rec["answers"] as? [[String: Any]] ?? []).map { ($0["letter_logits_bits"] as? [Int] ?? []).map { UInt16($0) } }
                    let equal = letterBits[run.key].map { $0 == bits }
                    if equal == true { sameAsE2E += 1 }
                    decisions.append(["k": k, "warmup": k == 0, "t_start_s": tStart, "t_end_s": tEnd,
                                      "wall_from_file_s": score.wallFromFile, "tower_s": score.tower as Any,
                                      "decoder_s": score.decoder, "seconds": rec["seconds"] ?? [:],
                                      "calls": rec["calls"] ?? [:], "thermal": DeviceInfo.thermal(),
                                      "battery_level": b.level, "battery_state": b.state,
                                      "letter_logits_equal_e2e": equal as Any,
                                      "footprint_mb": DeviceInfo.footprintMB()])
                    if k > 0 {
                        timed.append(score)
                        lastEnd = tEnd
                    }
                }
                let walls = timed.map { $0.wallFromFile * 1e3 }
                let towers = timed.compactMap { $0.tower.map { $0 * 1e3 } }
                let decs = timed.map { $0.decoder * 1e3 }
                let b1 = await DeviceInfo.battery()
                r["decisions"] = decisions
                r["summary"] = ["wall_ms_median": median(walls), "wall_ms_min": walls.min() ?? .nan, "wall_ms_max": walls.max() ?? .nan,
                                "tower_ms_median": towers.isEmpty ? Double.nan : median(towers),
                                "decoder_ms_median": median(decs), "timed_runs_end_s": lastEnd, "in_first_20s": lastEnd <= 20,
                                "letter_logits_equal_e2e": sameAsE2E, "decisions": config.benchRuns + 1]
                r["thermal_end"] = DeviceInfo.thermal()
                r["battery_end"] = ["level": b1.level, "state": b1.state, "power": DeviceInfo.powerSource(b1.state)]
                r["pass"] = true
                line("bench \(run.key) (1 warm-up + \(config.benchRuns) back to back, thermal \(r["thermal_start"] ?? "?") -> "
                     + "\(DeviceInfo.thermal()), battery \(String(format: "%.0f", b0.level * 100)) -> "
                     + "\(String(format: "%.0f", b1.level * 100)) % \(DeviceInfo.powerSource(b1.state))): wall median "
                     + "\(f1(median(walls))) ms (min \(f1(walls.min() ?? .nan)), max \(f1(walls.max() ?? .nan)))"
                     + (towers.isEmpty ? "" : ", tower \(f1(median(towers))) ms") + ", decoder \(f1(median(decs))) ms | timed runs "
                     + "end at \(f1(lastEnd)) s" + (lastEnd <= 20 ? " (inside the first 20 s)" : " (past 20 s)")
                     + " | letter logits = e2e \(sameAsE2E)/\(config.benchRuns + 1)")
            } catch {
                line("bench \(id)/\(arm): ERROR \(error)")
                r["error"] = "\(error)"
                r["error_detail"] = Self.describe(error)
                r["pass"] = false
                ok = false
            }
            out["\(id)/\(arm)"] = r
            j["rows"] = out
            writePartial(arm == "g256" ? "bench_g256" : "bench_g448", j, [])
        }
        j["rows"] = out
        j["pass"] = ok
        return j
    }

    // MARK: - summaries

    /// The bar over a set of scored runs, with the Mac comparison and the times.
    static func summarize(_ runs: [RunScore]) -> [String: Any] {
        let slots = runs.flatMap(\.slots)
        let allDeltas = slots.flatMap(\.deltas)
        let runMeans = runs.compactMap { r -> Double? in
            let d = r.slots.flatMap(\.deltas)
            return d.isEmpty ? nil : d.reduce(0, +) / Double(d.count)
        }
        let runSlotMeans = runs.compactMap { r -> Double? in
            let m = r.slots.compactMap { s -> Double? in s.deltas.isEmpty ? nil : s.deltas.reduce(0, +) / Double(s.deltas.count) }
            return m.isEmpty ? nil : m.reduce(0, +) / Double(m.count)
        }
        let maxDp = allDeltas.max() ?? .nan
        let meanR4 = runMeans.isEmpty ? Double.nan : runMeans.reduce(0, +) / Double(runMeans.count)
        let meanAlt = runSlotMeans.isEmpty ? Double.nan : runSlotMeans.reduce(0, +) / Double(runSlotMeans.count)
        let argmax = slots.filter(\.argmaxEqual).count
        let top1 = slots.filter(\.fullTop1IsOracleLetter).count
        let finite = slots.allSatisfy(\.finite)
        let idsOK = runs.filter(\.idsEqualOracle).count, slotsOK = runs.filter(\.slotsEqualOracle).count
        var worst: [String: Any] = [:]
        for r in runs {
            for (s, sc) in r.slots.enumerated() where (sc.deltas.max() ?? -1) >= maxDp {
                worst = ["run": r.key, "slot": s, "max_abs_dp": sc.deltas.max() ?? .nan]
            }
        }
        let macDp = slots.compactMap(\.macMaxAbsDp)
        let walls = runs.map { $0.wallFromFile * 1e3 }
        let early = runs.filter { $0.tStart < 20 }.map { $0.wallFromFile * 1e3 }
        let late = runs.filter { $0.tStart >= 20 }.map { $0.wallFromFile * 1e3 }
        var barPass = !runs.isEmpty && argmax == slots.count && top1 == slots.count && finite
        barPass = barPass && idsOK == runs.count && slotsOK == runs.count && maxDp <= 0.02 && meanR4 <= 0.002
        func count(_ xs: [Bool]) -> Int { xs.filter { $0 }.count }
        var mac: [String: Any] = ["slots_compared": macDp.count]
        mac["argmax_equal"] = count(slots.compactMap(\.macArgmaxEqual))
        mac["max_abs_dp"] = macDp.max() ?? Double.nan
        mac["median_abs_dp"] = median(macDp)
        mac["letter_logits_bit_equal"] = count(slots.compactMap(\.macLetterLogitsBitEqual))
        mac["ids_equal"] = count(runs.compactMap(\.idsEqualMac))
        mac["tower_embeds_equal_runs"] = count(runs.compactMap(\.towerEqualMac))
        mac["tower_runs"] = runs.compactMap(\.towerEqualMac).count
        mac["slot_logits_equal_runs"] = count(runs.compactMap(\.slotLogitsEqualMac))
        mac["pixels_and_patches_equal_runs"] = count(runs.compactMap(\.pixelsEqualMac))
        var wall: [String: Any] = ["median": median(walls), "min": walls.min() ?? Double.nan, "max": walls.max() ?? Double.nan]
        wall["median_first_20s"] = median(early)
        wall["runs_first_20s"] = early.count
        wall["median_after_20s"] = median(late)
        var s: [String: Any] = ["runs": runs.count, "slots": slots.count, "argmax_equal": argmax]
        s["full_vocab_top1_is_oracle_letter"] = top1
        s["finite_all"] = finite
        s["ids_equal_oracle"] = idsOK
        s["slots_equal_oracle"] = slotsOK
        s["max_abs_dp"] = maxDp
        s["worst"] = worst
        s["mean_of_run_mean_abs_dp"] = meanR4
        s["mean_of_run_slot_mean_abs_dp"] = meanAlt
        s["bar_pass"] = barPass
        s["mac"] = mac
        s["decoded_rgb_equal_meta_runs"] = count(runs.compactMap(\.decodedEqualMeta))
        s["wall_ms"] = wall
        return s
    }

    static func summaryLines(_ tag: String, _ s: [String: Any]) -> [String] {
        let m = s["mac"] as? [String: Any] ?? [:]
        let w = s["wall_ms"] as? [String: Any] ?? [:]
        func d(_ x: Any?) -> Double { x as? Double ?? .nan }
        return ["\(tag): \(s["runs"] ?? 0) runs, \(s["slots"] ?? 0) slots | argmax \(s["argmax_equal"] ?? 0), full-vocab top-1 "
                + "\(s["full_vocab_top1_is_oracle_letter"] ?? 0), ids \(s["ids_equal_oracle"] ?? 0), slots \(s["slots_equal_oracle"] ?? 0), "
                + "max|dp| \(f6(d(s["max_abs_dp"]))), mean \(f6(d(s["mean_of_run_mean_abs_dp"]))) (alt \(f6(d(s["mean_of_run_slot_mean_abs_dp"])))), "
                + "bar \((s["bar_pass"] as? Bool) == true ? "PASS" : "FAIL")",
                "\(tag) vs Mac Swift JIT: argmax \(m["argmax_equal"] ?? 0)/\(m["slots_compared"] ?? 0), |dp| max \(f6(d(m["max_abs_dp"]))) "
                + "median \(String(format: "%.2e", d(m["median_abs_dp"]))), letter logits bit-equal \(m["letter_logits_bit_equal"] ?? 0), "
                + "tower output equal \(m["tower_embeds_equal_runs"] ?? 0)/\(m["tower_runs"] ?? 0), slot logits equal "
                + "\(m["slot_logits_equal_runs"] ?? 0), ids equal \(m["ids_equal"] ?? 0) | wall ms median \(f1(d(w["median"]))) "
                + "(min \(f1(d(w["min"]))), max \(f1(d(w["max"])))), first 20 s \(f1(d(w["median_first_20s"]))), after "
                + "\(f1(d(w["median_after_20s"])))"]
    }

    /// The three e2e arms together (76 runs, 108 slots when complete) and the reset proofs.
    private func overallSummary() -> [String: Any] {
        var s = Self.summarize(Array(scores.values).sorted { $0.key < $1.key })
        var arms: [String: Any] = [:]
        for arm in ["g256", "g448", "text"] {
            let sel = scores.values.filter { $0.arm == arm }.sorted { $0.key < $1.key }
            if !sel.isEmpty { arms[arm] = Self.summarize(sel) }
        }
        s["per_arm"] = arms
        var resets: [String: Any] = [:]
        for p in ["g256", "g448"] {
            if let r = stageResults["reset_\(p)"] as? [String: Any] { resets[p] = r["bit_equal"] ?? false }
        }
        s["reset_bit_equal"] = resets
        return s
    }

    // MARK: - thermal, sampling, records

    /// Polls ProcessInfo.thermalState every 5 s until it is nominal, at most DV_WAIT_NOMINAL seconds.
    func waitForNominal(_ key: String) async -> [String: Any] {
        let cap = config.waitNominalSeconds
        let before = DeviceInfo.thermal()
        let c0 = ContinuousClock.now
        var polls = 0
        if before != "nominal" { line("\(key): thermal \(before); waiting for nominal (5 s steps, cap \(Int(cap)) s)") }
        while DeviceInfo.thermal() != "nominal" && seconds(since: c0) < cap {
            try? await Task.sleep(for: .seconds(5))
            polls += 1
        }
        let waited = seconds(since: c0)
        let after = DeviceInfo.thermal()
        if before != "nominal" {
            line("\(key): thermal \(after) after \(f1(waited)) s" + (after == "nominal" ? "" : " (cap reached: running anyway)"))
        }
        return ["cap_s": cap, "state_before": before, "state_after": after, "waited_s": waited, "polls": polls,
                "reached_nominal": after == "nominal"]
    }

    /// `body` under a MemorySampler (100 ms): its result or error, the sampler's record, the wall seconds.
    func sampled<T>(_ what: String, _ body: () async throws -> T) async -> (Result<T, Error>, [String: Any], Double) {
        let sampler = MemorySampler(what, memoryLog: memoryLog, progress: { [sink] in sink.line($0) })
        sampler.start()
        let c0 = ContinuousClock.now
        let result: Result<T, Error>
        do {
            result = .success(try await body())
        } catch {
            result = .failure(error)
        }
        let wall = seconds(since: c0)
        return (result, sampler.stop(), wall)
    }

    /// An error as result.json keeps it: the text, the Swift type and case, and the NSError bridge.
    static func describe(_ error: Error) -> [String: Any] {
        let ns = error as NSError
        return ["description": "\(error)", "reflecting": String(reflecting: error), "type": String(reflecting: type(of: error)),
                "ns_domain": ns.domain, "ns_code": ns.code,
                "ns_user_info": Dictionary(uniqueKeysWithValues: ns.userInfo.map { ($0.key, "\($0.value)") })]
    }

    /// The stage's record so far into result.json (a stage that is killed still leaves this much).
    func writePartial(_ key: String, _ j: [String: Any], _ thermal: [[String: Any]]) {
        var partial = j
        if !thermal.isEmpty { partial["thermal"] = thermal }
        partial["partial"] = true
        let order = stageOrder
        let previous = stageResults[key]
        stageResults[key] = partial
        stageOrder.append(key)
        writeReport()
        stageOrder = order
        stageResults[key] = previous
    }

    func finish(ok: Bool, fatal: String?) -> Bool {
        report["status"] = fatal == nil ? "done" : "failed"
        if let f = fatal { report["fatal"] = f }
        report["pass"] = ok
        report["finished"] = Self.now()
        report["elapsed_s"] = elapsed()
        report["device_end"] = ["thermal": DeviceInfo.thermal(), "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                                "footprint_mb": DeviceInfo.footprintMB(), "available_mb": DeviceInfo.availableMB()]
        report["e2e_summary"] = overallSummary()
        let verdicts = stageOrder.map { k -> String in
            let s = stageResults[k] as? [String: Any] ?? [:]
            return "\(k)=\((s["pass"] as? Bool) == true ? "PASS" : "FAIL")"
        }
        report["summary"] = verdicts
        if let s = report["e2e_summary"] as? [String: Any], (s["runs"] as? Int ?? 0) > 0 {
            for l in Self.summaryLines("all arms", s) { line(l) }
        }
        line("GATE_SUMMARY \(verdicts.joined(separator: " ")) VERDICT=\(ok ? "PASS" : "FAIL")" + (fatal.map { " (\($0))" } ?? ""))
        writeReport()
        let runs = config.out.appendingPathComponent("runs")
        try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let safe = config.runID.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
        try? FileManager.default.copyItem(at: config.out.appendingPathComponent("result.json"),
                                          to: runs.appendingPathComponent("\(safe).json"))
        line("DONE \(config.runID)")
        sink.close()
        memoryLog?.close()
        setStage(ok ? "done: PASS" : "done: FAIL")
        return ok
    }

    func writeReport() {
        var r = report
        r["stages"] = stageResults
        r["stage_order"] = stageOrder
        r["updated"] = Self.now()
        r["elapsed_s"] = elapsed()
        do {
            try writeJSON(r, to: config.out.appendingPathComponent("result.json"))
        } catch {
            line("ERROR writing result.json: \(error)")
        }
    }

    nonisolated func line(_ s: String) { sink.line(s) }

    func bumpLaunchCount() -> Int {
        let url = config.out.appendingPathComponent("launch_count")
        let n = (try? String(contentsOf: url, encoding: .utf8)).flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0
        try? "\(n + 1)\n".write(to: url, atomically: true, encoding: .utf8)
        return n + 1
    }

    func elapsed() -> Double { seconds(since: t0) }

    static func now() -> String { ISO8601DateFormatter().string(from: Date()) }
}
