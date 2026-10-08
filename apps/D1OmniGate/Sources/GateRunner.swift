// GateRunner — the d1-omni-600M device gate on the D1Omni library (the publisher's prompt rows, the decision graph of
// the row's bucket, the vision and audio graphs on media the host decodes here, the readout), over a subset of the
// fixture the port was gated on (78 rows: 60 text, 9 image, 9 audio; the 18 text rows of at most 64 positions again on
// L64), on the iPhone or on the Mac. What the phone adds to
// the Mac runs of the same library (rounds 8-10): the on-device JIT specialization of each `.aimodel` against the iPhone
// AOT `.aimodelc` (h19p), their loads, footprint and headroom, the probabilities on the phone's own GPU (and Neural
// Engine), thermal, and the time of a decision. The stage machinery, the thermal / battery records and result.json are
// apps/KevGate's (GateRunner.swift); the stages are d1-omni's.
// Results: <out>/result.json (rewritten when a stage starts, after every stage, every 10 rows and every bench series;
// "status" running -> done), result.log (one line per event, each with the thermal state and the battery), memory.tsv
// (every 100 ms from launch to exit, written as taken); <out> = Documents/d1omni_gate on the iPhone.
//
// Assets (<assets> = Library/Application Support/D1OmniAssets on the iPhone; D1_ASSETS on the Mac), laid out by
// ../_stage.sh and pushed by ../_install.sh:
//   jit/          the shipping bundles (strip_ship.py's macos-ship/ and macos-ship-small/, the same bytes as the iOS
//                 `ios/` folder): fp16-L64, fp16-L256 (with tokenizer/), fp16-L2048, fp16-L4096, vision-fp16
//                 (position_table.f32), audio-fp16-10s (mel_filters_128x257_f32.bin), each with metadata.json and its
//                 .aimodel (the JIT form)
//   aot/          <the same names>/<stem>.h19p.aimodelc: the iPhone 18 Pro's AOT (coreai-build --platform iOS
//                 --architecture h19p --preferred-compute gpu); D1_AOT_DIR points elsewhere (the Mac: its h16c AOT)
//   ane/          fp16-L256/ and audio-fp16-10s/ <stem>.h19p.aimodelc compiled with --preferred-compute neural-engine
//   fixtures/     subset.json, oracle_slim.json, mac_ref.json, bench.json, media/ (Fixtures.swift)
//   MD5SUMS
// An iPhone AOT bundle (.h18p. / .h19p.) is never opened on a Mac: the stages that would open one refuse it.
//
// Stages, in the order D1_STAGES gives (default assets,parity_jit,parity_aot,bench,long,ane,md5):
//   assets      MD5SUMS: every file present, md5 of every file up to 16 MB (the model files wait for md5: reading 8 GB
//               right before the first load would warm the file cache under the cold-load number), the free space
//   parity_jit  per group of graphs that live together (L64 + L256 + vision + audio, then L2048 + vision), a fresh
//               D1Omni on jit/ (each .aimodel specialized here, GPU preferred): the load of every graph (wall, AIModel,
//               main, the Core AI cache before / after, the memory), then the group's rows: per row the ids / markers /
//               bucket against the publisher's, p against the oracle (the bar) and logits / p against the Mac (bits,
//               max |d|); per image and clip every array of the media path against the Mac's (sha256); the rows of at
//               most 64 positions again on L64; the control. The group's instance is dropped before the next loads
//   parity_aot  the same on the AOT .aimodelc of each graph (D1_AOT_DIR, default aot/), and the rows against parity_jit's
//   bench       W1 / W2 (L64 and L256), W4, W5 (bench.json; D1_BENCH_WORKLOADS), per workload fresh instances of the forms
//               of D1_BENCH_FORMS (default jit,aot) x the workload's buckets, alternating per round in one process: wait
//               up to D1_WAIT_NOMINAL s (60) for the thermal state nominal, then series of at most max_series_s (18) s
//               with rest_s (10) s between them (the 18 Pro's GPU slows after 20 s of back-to-back calls): 3 warm-up
//               rounds in the first series, 1 in each later one, until `timed` (20) rounds are timed; a series stops
//               early when the thermal state turns serious, and the next one waits 60 s steps (at most 300 s) for it to
//               fall; every decision with its start offset, ms (media / inputs / graph / readout), thermal, battery and
//               footprint; the outputs checked on every call (JIT = AOT bits per bucket, the oracle)
//   long        the graph L4096 apart (its working set is 4-6 GB of footprint on the Mac, over the phone's default memory
//               limit): the 4 long rows on JIT, then on AOT, then W3 with the forms one after the other
//   ane         the Neural Engine graphs (D1_ANE = audio,decision; D1_ANE_SOURCE = aot and / or jit, tried in order: the
//               .aimodelc of D1_ANE_DIR compiled for the Neural Engine, or the .aimodel specialized here with the Neural
//               Engine preferred): their regions, load, then audio: aud_01..03's prefix on the ANE graph against the
//               GPU's, the 9 audio rows on the GPU decision graph against the oracle, W5 with the ANE audio graph against
//               W5 on the GPU in the same series; decision: the 56 bucket-256 text rows on the ANE decision graph (2 calls
//               each, the bar's values and the drift), W1 / W2 against the GPU in the same series
//   md5         md5 of the files "assets" left
//
// A stage writes its start into result.json before it runs. A launch that finds result.json still "running" records the
// stage the previous launch died in (a crash or a jetsam kill) and skips that stage if it is planned again
// (D1_RETRY_DIED=1 runs it anyway). D1_DEADLINE_S (seconds after the launch) skips a stage that cannot finish and stops
// a loop early (the record says so): ../_gate.sh sets it from what is left of the 30 min window.
//
// Environment (devicectl device process launch --environment-variables on the iPhone; the shell on the Mac), all
// optional except D1_ASSETS on the Mac: D1_RUN_ID, D1_STAGES, D1_ASSETS, D1_OUT, D1_JIT_DIR, D1_AOT_DIR, D1_ANE_DIR,
// D1_ANE, D1_ANE_SOURCE, D1_BENCH_FORMS, D1_BENCH_WORKLOADS, D1_BENCH_TIMED, D1_BENCH_WARMUP, D1_BENCH_SERIES_S, D1_BENCH_REST,
// D1_WAIT_NOMINAL, D1_SERIOUS_REST, D1_SERIOUS_CAP, D1_DEADLINE_S, D1_MIN_FREE_GB, D1_RETRY_DIED, D1_EXIT_WHEN_DONE
// (Mac, default 1). Every D1_* variable is echoed into result.json (config.env).

import CoreAI
import D1Omni
import Foundation

struct GateConfig: Sendable {
    static let defaultStages = ["assets", "parity_jit", "parity_aot", "bench", "long", "ane", "md5"]
    static let knownStages = ["assets", "parity_jit", "parity_aot", "bench", "long", "ane", "md5"]

    let runID: String
    /// nil on a Mac without D1_ASSETS (the run stops with a fatal line)
    let assets: URL?
    let out: URL
    let stages: [String]
    let jitDir: String
    let aotDir: String
    let aneDir: String
    let aneParts: [String]
    /// where the Neural Engine graph comes from, tried in order: "aot" (ane/'s .aimodelc compiled for the Neural
    /// Engine) and / or "jit" (jit/'s .aimodel, specialized here with the Neural Engine preferred)
    let aneSources: [String]
    let benchForms: [String]
    let benchWorkloads: [String]
    let benchTimed: Int?
    let benchWarmup: Int?
    let benchSeriesS: Double?
    let benchRest: Double?
    let waitNominal: Double
    let seriousRest: Double
    let seriousCap: Double
    let deadline: Double?
    let minFreeGB: Double
    let retryDied: Bool
    let exitWhenDone: Bool
    /// Files up to this many bytes are md5-checked in "assets"; the larger ones in "md5".
    let md5SmallLimit: Int
    let env: [String: String]

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        func path(_ s: String) -> URL { s.hasPrefix("/") ? URL(fileURLWithPath: s) : home.appendingPathComponent(s) }
        func list(_ k: String) -> [String] {
            (env[k] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        }
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        #if os(iOS)
        let assets: URL? = env["D1_ASSETS"].map(path) ?? support.appendingPathComponent("D1OmniAssets")
        let out = env["D1_OUT"].map(path)
            ?? fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("d1omni_gate")
        let exitWhenDone = false
        #else
        let assets: URL? = env["D1_ASSETS"].map(path)
        let out = env["D1_OUT"].map(path) ?? support.appendingPathComponent("D1OmniGate/out")
        let exitWhenDone = env["D1_EXIT_WHEN_DONE"] != "0"
        #endif
        let stages = list("D1_STAGES")
        let forms = list("D1_BENCH_FORMS")
        let workloads = list("D1_BENCH_WORKLOADS")
        let ane = list("D1_ANE")
        return GateConfig(
            runID: env["D1_RUN_ID"] ?? ISO8601DateFormatter().string(from: Date()),
            assets: assets, out: out, stages: stages.isEmpty ? defaultStages : stages,
            jitDir: env["D1_JIT_DIR"] ?? "jit", aotDir: env["D1_AOT_DIR"] ?? "aot", aneDir: env["D1_ANE_DIR"] ?? "ane",
            aneParts: ane.isEmpty ? ["audio", "decision"] : ane,
            aneSources: list("D1_ANE_SOURCE").isEmpty ? ["aot"] : list("D1_ANE_SOURCE"),
            benchForms: forms.isEmpty ? ["jit", "aot"] : forms,
            benchWorkloads: workloads.isEmpty ? ["W1", "W2", "W4", "W5"] : workloads,
            benchTimed: Int(env["D1_BENCH_TIMED"] ?? "").map { max(1, $0) },
            benchWarmup: Int(env["D1_BENCH_WARMUP"] ?? "").map { max(0, $0) },
            benchSeriesS: Double(env["D1_BENCH_SERIES_S"] ?? "").map { max(1, $0) },
            benchRest: Double(env["D1_BENCH_REST"] ?? "").map { max(0, $0) },
            waitNominal: max(0, Double(env["D1_WAIT_NOMINAL"] ?? "") ?? 60),
            seriousRest: max(1, Double(env["D1_SERIOUS_REST"] ?? "") ?? 60),
            seriousCap: max(0, Double(env["D1_SERIOUS_CAP"] ?? "") ?? 300),
            deadline: Double(env["D1_DEADLINE_S"] ?? ""),
            minFreeGB: Double(env["D1_MIN_FREE_GB"] ?? "") ?? 4,
            retryDied: env["D1_RETRY_DIED"] == "1",
            exitWhenDone: exitWhenDone,
            md5SmallLimit: 16 << 20,
            env: env.filter { $0.key.hasPrefix("D1_") })
    }

    var json: [String: Any] {
        ["assets": assets?.path ?? "(unset)", "out": out.path, "stages": stages, "jit_dir": jitDir, "aot_dir": aotDir,
         "ane_dir": aneDir, "ane_parts": aneParts, "ane_sources": aneSources, "bench_forms": benchForms,
         "bench_workloads": benchWorkloads,
         "bench_timed": benchTimed ?? -1, "bench_warmup": benchWarmup ?? -1, "bench_series_s": benchSeriesS ?? -1,
         "bench_rest_s": benchRest ?? -1, "wait_nominal_s": waitNominal, "serious_rest_s": seriousRest,
         "serious_cap_s": seriousCap, "deadline_s": deadline ?? -1, "min_free_gb": minFreeGB, "retry_died": retryDied,
         "md5_small_limit_bytes": md5SmallLimit, "env": env]
    }
}

/// One timed decision of the bench: the form's graphs and its prepared inputs.
struct BenchForm: @unchecked Sendable {
    /// "<kind>-L<bucket>" (jit-L64, aot-L256, ...), or a Neural Engine form's name
    let name: String
    /// the decision graph's bucket: forms of one bucket must agree bit for bit (JIT = AOT)
    let bucket: Int
    let d1: D1Omni
    let decision: DecisionGraph
    let vision: VisionGraph?
    let audio: AudioGraph?
}

actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0: ContinuousClock.Instant
    private let sink: LogSink
    private var monitor: MemoryMonitor?
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []
    private var fixtures: Fixtures?
    /// Per form ("jit", "aot", "jit-L64", "aot-L64"): the parity rows' probabilities and logits, and the GPU's audio
    /// prefix per clip (the Neural Engine stage compares with them).
    private var rowProbs: [String: [String: [Float]]] = [:]
    private var rowLogits: [String: [String: [Float]]] = [:]
    private var clipPrefix: [String: [String: [Float]]] = [:]
    /// (path, md5) of the files "assets" left for "md5"
    private var deferredMD5: [(rel: String, sum: String)] = []
    private var assetsChecked = false
    /// The stage the previous launch died in (result.json left "running"), if any.
    private var diedStage: String?
    /// While a stage made of parts runs ("long"): its key and the parts done so far (writePartial).
    private var partialParent: String?
    private var partialParentRecord: [String: Any] = [:]

    /// Neural Engine preferred (round 7's Python probe: from_preferred_compute_unit_kind(neural_engine)), never .default.
    static var aneOptions: SpecializationOptions { SpecializationOptions(preferredComputeUnitKind: .neuralEngine) }

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
        let t0 = ContinuousClock.now
        self.t0 = t0
        sink = LogSink(t0: t0, emit: emit)
    }

    private var assets: URL { config.assets ?? URL(fileURLWithPath: "/nonexistent") }
    private func assetURL(_ p: String) -> URL { p.hasPrefix("/") ? URL(fileURLWithPath: p) : assets.appendingPathComponent(p) }

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
        let previous = previousLaunch(resultURL)
        try? fm.removeItem(at: resultURL)
        fm.createFile(atPath: logURL.path, contents: nil)
        sink.open(logURL)
        let battery = await DeviceInfo.battery()
        BatteryCache.shared.set(battery)
        let mon = MemoryMonitor(url: memURL, t0: t0)
        mon.start()
        monitor = mon

        let launchIndex = bumpLaunchCount()
        var device = DeviceInfo.snapshot()
        device["battery_level"] = battery.level
        device["battery_state"] = battery.state
        device["power_source"] = DeviceInfo.powerSource(battery.state)
        device["footprint_mb"] = DeviceInfo.footprintMB()
        device["available_mb"] = DeviceInfo.availableMB()
        device["free_gb"] = DeviceInfo.freeGB(config.out)
        report = ["app": "D1OmniGate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device,
                  "model": "d1-omni-600M (LiquidAI, rev 414f8d64): the shipping fp16 graphs (strip_debug_info'd, L64 in round 12, "
                      + "the rest in round 10) — decision L64 / L256 / L2048 / L4096, vision (1 crop -> 256 rows), audio 10 s "
                      + "— on the D1Omni library",
                  "options": ["gpu": describe(DecisionGraph.gpuOptions), "ane": describe(Self.aneOptions)],
                  "build": ["configuration": Self.buildConfiguration],
                  "bar": ["name": "FACTS §7 (conversion/d1_omni/_metrics.py SHIP_BAR)",
                          "argmax": "every row whose oracle top-2 margin is above 0.02 (near ties counted apart)",
                          "max_abs_dp": 0.02, "mean_row_max_abs_dp": 0.002,
                          "ids": "every row's ids, markers and bucket = the publisher's", "finite": "finite logits",
                          "control": "each row against the next row of its class (type, K, mode): must FAIL"],
                  "config": config.json]
        if let p = previous { report["previous_launch"] = p }
        line("D1OmniGate run \(config.runID) (launch \(launchIndex) here), \(Self.buildConfiguration) build")
        line("device \(device["machine"] ?? "?") \(device["hw_model"] ?? "?"), \(device["os"] ?? "?") (build "
             + "\(device["os_build"] ?? "?")), Core AI arch \(device["coreai_architecture"] ?? "?"), low power "
             + "\(device["low_power_mode"] ?? "?"), power \(DeviceInfo.powerSource(battery.state)), footprint "
             + "\(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB, free "
             + "\(f1(DeviceInfo.freeGB(config.out))) GB, physical memory \(f2(device["physical_memory_gb"] as? Double ?? -1)) GB"
             + (config.deadline.map { ", deadline \(Int($0)) s after launch" } ?? ""))
        if let p = previous {
            line("previous launch: run \(p["run_id"] ?? "?") ended \(p["status"] ?? "?")"
                 + (diedStage.map { " — it died in stage \($0) (last result.json update \(p["updated"] ?? "?"))" } ?? ""))
        }
        writeReport()

        guard config.assets != nil else {
            line("FATAL D1_ASSETS is not set (the Mac reads the stage directory from it)")
            return finish(ok: false, fatal: "D1_ASSETS not set")
        }
        #if os(macOS)
        // an iPhone AOT bundle (.h18p. / .h19p. ...) must never be opened on a Mac
        var refuse: [String] = []
        if config.stages.contains("parity_aot") || (config.stages.contains("bench") && config.benchForms.contains("aot")) {
            refuse += iPhoneAssets(under: assetURL(config.aotDir))
        }
        if config.stages.contains("ane") { refuse += iPhoneAssets(under: assetURL(config.aneDir)) }
        if !refuse.isEmpty {
            line("FATAL refusing iPhone AOT bundles on macOS: \(refuse.prefix(4).joined(separator: ", "))")
            return finish(ok: false, fatal: "iPhone AOT bundle on macOS: \(refuse[0])")
        }
        #endif
        do {
            let fx = try Fixtures(root: assets.appendingPathComponent("fixtures"))
            fixtures = fx
            report["fixtures"] = fx.files
            let groups = Dictionary(grouping: fx.rows, by: \.group).mapValues(\.count)
            line("fixtures: \(fx.records.count) records, \(fx.rows.count) rows (\(groups.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }.joined(separator: ", "))); "
                 + "images \(fx.images.map(\.id)), clips \(fx.clips.map(\.id)); bench \(fx.bench.workloads.map(\.id))")
        } catch {
            line("FATAL fixtures: \(error)")
            return finish(ok: false, fatal: "fixtures: \(error)")
        }

        var allOK = true
        for stage in config.stages {
            setStage(stage)
            monitor?.set(stage)
            let s0 = elapsed()
            var result: [String: Any]
            let needs: [String: Double] = ["assets": 10, "parity_jit": 60, "parity_aot": 60, "bench": 60, "long": 90, "ane": 120,
                                           "md5": 30]
            let need = needs[stage] ?? 30
            if stage == diedStage && !config.retryDied {
                line("\(stage): SKIPPED — the previous launch died in this stage (D1_RETRY_DIED=1 runs it again)")
                result = ["skipped": true, "reason": "the previous launch died in this stage", "pass": false]
            } else if let left = remaining(), left < need {
                line("\(stage): SKIPPED — \(Int(left)) s left before the deadline, the stage needs about \(Int(need)) s")
                result = ["skipped": true, "deadline_skipped": true, "reason": "\(Int(left)) s left before the deadline",
                          "pass": false]
            } else {
                // the stage's start goes into result.json before it runs: a launch that dies here leaves it behind
                writePartial(stage, ["step": "start", "started_at_s": s0])
                switch stage {
                case "assets": result = stageAssets()
                case "parity_jit": result = await stageParity(kind: "jit", key: stage)
                case "parity_aot": result = await stageParity(kind: "aot", key: stage)
                case "bench": result = await stageBench(key: stage, workloads: config.benchWorkloads)
                case "long": result = await stageLong()
                case "ane": result = await stageANE()
                case "md5": result = stageMD5()
                default:
                    line("unknown stage \(stage) (D1_STAGES takes \(GateConfig.knownStages.joined(separator: ", ")))")
                    result = ["pass": false, "error": "unknown stage \(stage)"]
                }
            }
            result["t_start_s"] = s0
            result["t_end_s"] = elapsed()
            result["memory"] = monitor?.summary(stage) ?? [:]
            stageResults[stage] = result
            stageOrder.append(stage)
            allOK = allOK && (result["pass"] as? Bool ?? false)
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

    /// The previous launch's result.json, when it did not finish: its run id, status, the stage it was in, its last
    /// update. A copy is kept as runs/<run id>.died.json.
    private func previousLaunch(_ url: URL) -> [String: Any]? {
        guard let d = try? Data(contentsOf: url), let r = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else {
            return nil
        }
        let status = r["status"] as? String ?? "?"
        var p: [String: Any] = ["run_id": r["run_id"] ?? "?", "status": status, "updated": r["updated"] ?? "?",
                                "stage_order": r["stage_order"] ?? []]
        if status == "running" {
            let order = r["stage_order"] as? [String] ?? []
            let stages = r["stages"] as? [String: Any] ?? [:]
            if let last = order.last, (stages[last] as? [String: Any])?["partial"] as? Bool == true {
                diedStage = last
                p["died_in_stage"] = last
                p["died_stage_record"] = stages[last]
            }
            let runs = config.out.appendingPathComponent("runs")
            try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
            let safe = "\(r["run_id"] ?? "unknown")".replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
            try? FileManager.default.copyItem(at: url, to: runs.appendingPathComponent("\(safe).died.json"))
            p["copy"] = "runs/\(safe).died.json"
        }
        return p
    }

    /// The `.aimodelc` names under a directory that are an iPhone architecture's (".h18p.", ".h19p.", ...).
    private nonisolated func iPhoneAssets(under dir: URL) -> [String] {
        guard let e = FileManager.default.enumerator(atPath: dir.path) else { return [] }
        var out: [String] = []
        while let rel = e.nextObject() as? String {
            let name = (rel as NSString).lastPathComponent
            if name.hasSuffix(".aimodelc"), name.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil {
                out.append(rel)
                e.skipDescendants()
            }
        }
        return out
    }

    /// The one `.aimodelc` in a directory.
    private nonisolated func aimodelc(in dir: URL) throws -> URL {
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasSuffix(".aimodelc") }
        guard names.count == 1 else { throw GateError.assets("\(dir.path): \(names.count) .aimodelc (want 1)") }
        return dir.appendingPathComponent(names[0])
    }

    private func rel(_ u: URL) -> String {
        let a = assets.path + "/"
        return u.path.hasPrefix(a) ? String(u.path.dropFirst(a.count)) : u.path
    }

    private func remaining() -> Double? { config.deadline.map { $0 - elapsed() } }

    // MARK: - assets

    func stageAssets() -> [String: Any] {
        let fm = FileManager.default
        let c0 = ContinuousClock.now
        var j: [String: Any] = ["dir": assets.path]
        guard let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) else {
            line("assets: no MD5SUMS in \(assets.path)")
            return ["dir": assets.path, "error": "no MD5SUMS", "pass": false, "stop": true]
        }
        var listed = 0, checked = 0, checkedBytes = 0, deferredBytes = 0, listedBytes = 0
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
            listedBytes += size
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
        let required = ["fixtures/subset.json", "fixtures/oracle_slim.json", "fixtures/mac_ref.json", "fixtures/bench.json",
                        "\(config.jitDir)/fp16-L256/metadata.json", "\(config.jitDir)/fp16-L256/tokenizer/tokenizer.json"]
        let requiredURLs = required.map { (rel: $0, url: assetURL($0)) }
        let absent = requiredURLs.filter { !FileManager.default.fileExists(atPath: $0.url.path) }.map(\.rel)
        var dirs: [[String: Any]] = []
        for d in [config.jitDir, config.aotDir, config.aneDir] {
            let u = assetURL(d)
            for n in ((try? fm.contentsOfDirectory(atPath: u.path)) ?? []).sorted() {
                let t = DeviceInfo.tree(u.appendingPathComponent(n))
                dirs.append(["dir": "\(d)/\(n)", "bytes": t.bytes, "files": t.files])
            }
        }
        let free = DeviceInfo.freeGB(assets)
        j["md5sums_listed"] = listed
        j["listed_bytes"] = listedBytes
        j["missing"] = missing
        j["md5_mismatch"] = mismatched
        j["md5_checked"] = checked
        j["md5_checked_bytes"] = checkedBytes
        j["md5_deferred"] = deferredMD5.map(\.rel)
        j["md5_deferred_bytes"] = deferredBytes
        j["unlisted_entries"] = unlisted
        j["required_absent"] = absent
        j["graph_dirs"] = dirs
        j["free_gb"] = free
        j["min_free_gb"] = config.minFreeGB
        j["storage"] = DeviceInfo.storageSnapshot()
        j["seconds"] = seconds(since: c0)
        var ok = listed > 0 && missing.isEmpty && mismatched.isEmpty && absent.isEmpty
        assetsChecked = true
        line("assets \(assets.path): \(listed) files (\(mb(listedBytes)) MB) in MD5SUMS, \(missing.count) missing, md5 \(checked) "
             + "checked (\(mb(checkedBytes)) MB) with \(mismatched.count) different, \(deferredMD5.count) model files "
             + "(\(mb(deferredBytes)) MB) left for the md5 stage; unlisted \(unlisted); free \(f1(free)) GB")
        for d in dirs { line("  \(d["dir"] ?? "?"): \(mb(d["bytes"] as? Int ?? 0)) MB, \(d["files"] ?? 0) files") }
        if !missing.isEmpty { line("assets: missing \(missing.prefix(8).joined(separator: ", "))") }
        if !mismatched.isEmpty { line("assets: md5 differs \(mismatched.prefix(8).joined(separator: ", "))") }
        if !absent.isEmpty { line("assets: required file absent \(absent.joined(separator: ", "))") }
        if free >= 0 && free < config.minFreeGB {
            line("assets: STOP free space \(f1(free)) GB < D1_MIN_FREE_GB \(f1(config.minFreeGB)) GB")
            j["error"] = "free space \(free) GB < \(config.minFreeGB) GB"
            ok = false
        }
        j["pass"] = ok
        if !ok { j["stop"] = true }
        return j
    }

    func stageMD5() -> [String: Any] {
        let c0 = ContinuousClock.now
        if !assetsChecked, let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) {
            deferredMD5 = text.split(separator: "\n").compactMap { row in
                let p = row.split(separator: " ", maxSplits: 1)
                guard p.count == 2 else { return nil }
                let rel = p[1].trimmingCharacters(in: .whitespaces)
                let size = (try? assets.appendingPathComponent(rel).resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return size > config.md5SmallLimit ? (rel, String(p[0])) : nil
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
        line("md5: \(deferredMD5.count) files, \(mb(bytes)) MB in \(f1(s)) s, \(mismatched.count) different"
             + (mismatched.isEmpty ? "" : ": \(mismatched.joined(separator: ", "))"))
        return ["files": deferredMD5.map(\.rel), "bytes": bytes, "md5_mismatch": mismatched, "seconds": s,
                "pass": !deferredMD5.isEmpty && mismatched.isEmpty]
    }

    // MARK: - instances and loads

    /// The Core AI cache of this app: bytes and files.
    private func cacheBytes() -> (bytes: Int, files: Int) {
        let c = DeviceInfo.storageSnapshot()["coreai_cache"] as? [String: Any] ?? [:]
        return (c["bytes"] as? Int ?? 0, c["files"] as? Int ?? 0)
    }

    /// The decision buckets of the stage's jit/ folder (fp16-L<L>).
    private func stagedBuckets() -> [Int] { ((try? D1Omni.folders(macos: assetURL(config.jitDir))) ?? [:]).keys.sorted() }

    /// The D1Omni instance of a GPU form on some of the staged graphs: jit = each bundle's .aimodel, aot = D1_AOT_DIR's
    /// .aimodelc of each. `buckets` nil = every decision bucket; the vision / audio graphs only when asked. A form's
    /// graphs live as long as its instance: the stages make one per group of graphs and drop it after (the phone's
    /// default memory limit holds one large bucket's working set at a time, not all of them).
    private func makeInstance(_ kind: String, buckets: [Int]? = nil, vision: Bool = true, audio: Bool = true)
        async throws -> (D1Omni, [String: Any]) {
        let jitRoot = assetURL(config.jitDir)
        var folders = try D1Omni.folders(macos: jitRoot)
        if let b = buckets { folders = folders.filter { b.contains($0.key) } }
        guard !folders.isEmpty else { throw GateError.assets("no decision bucket of \(buckets ?? []) under \(jitRoot.path)") }
        let found = D1Omni.MediaFolders.find(macos: jitRoot)
        let visionFolder = vision ? found.vision : nil
        let audioFolders = audio ? found.audio : [:]
        var assetsByL: [Int: URL] = [:]
        var media: [String: URL] = [:]
        if kind == "aot" {
            let root = assetURL(config.aotDir)
            for L in folders.keys { assetsByL[L] = try aimodelc(in: root.appendingPathComponent("fp16-L\(L)")) }
            if visionFolder != nil { media["vision"] = try aimodelc(in: root.appendingPathComponent("vision-fp16")) }
            for sec in audioFolders.keys { media["audio-\(sec)"] = try aimodelc(in: root.appendingPathComponent("audio-fp16-\(sec)s")) }
        }
        let c0 = ContinuousClock.now
        // the tokenizer is staged with fp16-L256 only
        let d1 = try await D1Omni(folders: folders, assets: assetsByL,
                                  tokenizerFolder: jitRoot.appendingPathComponent("fp16-L256/tokenizer"),
                                  media: D1Omni.MediaFolders(vision: visionFolder, audio: audioFolders, assets: media))
        let rec: [String: Any] = ["kind": kind, "init_s": seconds(since: c0), "tokenizer_load_s": d1.tokenizer.loadSeconds,
                                  "buckets": d1.lengths, "vision": visionFolder.map { rel($0) } ?? "",
                                  "audio_buckets_s": audioFolders.keys.sorted(),
                                  "assets": assetsByL.sorted { $0.key < $1.key }.map { ["L": $0.key, "asset": rel($0.value)] },
                                  "media_assets": media.sorted { $0.key < $1.key }.map { ["graph": $0.key, "asset": rel($0.value)] }]
        return (d1, rec)
    }

    /// One graph's load, timed: wall, AIModel and main, the Core AI cache before / after, the memory during it.
    private func timedLoad(_ key: String, _ what: String,
                           _ body: () async throws -> (url: URL, kind: String, model: Double, function: Double,
                                                        descriptor: String)) async -> [String: Any] {
        let label = "\(key) load \(what)"
        monitor?.set(label)
        let c0 = cacheBytes()
        let fp0 = DeviceInfo.footprintMB()
        let t = ContinuousClock.now
        var r: [String: Any] = ["what": what, "cache_bytes_before": c0.bytes, "footprint_mb_before": fp0,
                                "thermal_start": DeviceInfo.thermal()]
        do {
            let g = try await body()
            let wall = seconds(since: t)
            let c1 = cacheBytes()
            r["asset"] = rel(g.url)
            r["asset_bytes"] = DeviceInfo.tree(g.url).bytes
            if let h = try? Data(contentsOf: g.url.appendingPathComponent("main.hash")) {
                r["asset_main_hash"] = h.map { String(format: "%02x", $0) }.joined()
            }
            r["kind"] = g.kind
            r["wall_s"] = wall
            r["model_s"] = g.model
            r["function_s"] = g.function
            r["descriptor"] = g.descriptor
            r["cache_bytes_after"] = c1.bytes
            r["cache_bytes_added"] = c1.bytes - c0.bytes
            r["footprint_mb_after"] = DeviceInfo.footprintMB()
            r["available_mb_after"] = DeviceInfo.availableMB()
            r["pass"] = true
            line("\(key) load \(what): \(g.kind) \(g.url.lastPathComponent) wall \(f2(wall)) s (AIModel \(f2(g.model)) s, main "
                 + "\(f2(g.function)) s), Core AI cache +\(mb(c1.bytes - c0.bytes)) MB, footprint \(f1(DeviceInfo.footprintMB())) MB, "
                 + "available \(f1(DeviceInfo.availableMB())) MB")
        } catch {
            r["wall_s"] = seconds(since: t)
            r["error"] = "\(error)"
            r["error_detail"] = Self.errorRecord(error)
            r["pass"] = false
            line("\(key) load \(what): ERROR \(error)")
        }
        r["memory"] = monitor?.summary(label) ?? [:]
        r["thermal_end"] = DeviceInfo.thermal()
        monitor?.set(key)
        return r
    }

    private func loadAll(_ d1: D1Omni, key: String) async -> [[String: Any]] {
        var loads: [[String: Any]] = []
        for L in d1.lengths {
            loads.append(await timedLoad(key, "decide L\(L)") {
                let g = try await d1.graph(L)
                return (g.url, g.kind, g.loadSeconds.model, g.loadSeconds.function, JSONWriter.compact(g.descriptor))
            })
        }
        if d1.media?.vision != nil {
            loads.append(await timedLoad(key, "vision") {
                let g = try await d1.vision().graph
                return (g.url, g.kind, g.loadSeconds.model, g.loadSeconds.function, JSONWriter.compact(g.descriptor))
            })
        }
        for sec in (d1.media?.audio.keys.sorted() ?? []) {
            loads.append(await timedLoad(key, "audio \(sec) s") {
                let g = try await d1.audio(AudioBucket(seconds: sec)).graph
                return (g.url, g.kind, g.loadSeconds.model, g.loadSeconds.function, JSONWriter.compact(g.descriptor))
            })
        }
        return loads
    }

    // MARK: - parity

    private func rowJSON(_ r: Fixtures.Row, _ s: Fixtures.RowScore, logits: [Float], probs: [Float], device: Row?,
                         seconds d: (inputs: Double, graph: Double, readout: Double)?, bucket: Int? = nil) -> [String: Any] {
        var j: [String: Any] = [
            "key": r.key, "group": r.group, "mode": r.mode.rawValue, "bucket": bucket ?? r.bucket, "positions": r.positions,
            "prefix_len": r.prefixLength, "ids_equal": s.idsEqual, "markers_equal": s.markersEqual,
            "bucket_equal": s.bucketEqual, "finite": s.finite, "logits": logits.map(Double.init),
            "logits_bits": s.logitsBits.map(Int.init), "probs": probs.map(Double.init), "probs_bits": s.probsBits.map(Int.init),
            "max_abs_dp": s.maxAbsDp, "argmax": s.argmax, "argmax_oracle": s.argmaxOracle, "argmax_equal": s.argmaxEqual,
            "near_tie": s.nearTie, "mac_logits_bit_equal": s.macLogitsBitEqual, "mac_probs_bit_equal": s.macProbsBitEqual,
            "mac_max_abs_dp": s.macMaxAbsDp, "mac_max_abs_dlogit": s.macMaxAbsDlogit, "thermal": DeviceInfo.thermal(),
        ]
        if let d {
            j["inputs_ms"] = d.inputs * 1e3
            j["graph_ms"] = d.graph * 1e3
            j["readout_ms"] = d.readout * 1e3
        }
        if !s.idsEqual, let device { j["device_ids"] = device.ids }
        if !s.markersEqual, let device { j["device_markers"] = device.markers }
        return j
    }

    /// The device's rows of a record in a mode, matched to the subset's rows by qid (code points).
    private func matchRows(_ d1: D1Omni, _ rec: Fixtures.Record, _ mode: Mode, prefix p: Int) throws -> [(Fixtures.Row, Row)] {
        let fx = fixtures!
        let rows = try d1.rows(state: rec.state, questions: rec.questions, mode: mode, prefixLength: p)
        return try fx.rows(of: rec.id, mode: mode).map { want in
            guard let got = rows.first(where: { $0.qid.unicodeScalars.elementsEqual(want.qid.unicodeScalars) }) else {
                throw GateError.fixture("\(rec.id): no question \(want.qid) in the device's rows")
            }
            return (want, got)
        }
    }

    /// One row on the graph of its round-10 bucket (or, `l64`, on L64), scored; its p and logits kept per form.
    private func decideScore(_ d1: D1Omni, _ want: Fixtures.Row, _ got: Row, prefix: [Float]?, kind: String, l64: Bool = false,
                             scores: inout [Fixtures.RowScore], out: inout [[String: Any]]) async throws {
        let fx = fixtures!
        let L = l64 ? 64 : want.bucket
        let dec = try await d1.decide(got, prefix: prefix, bucket: L)
        let s = fx.score(want, idsEqual: got.ids == want.ids, markersEqual: got.markers == want.markers,
                         bucketEqual: dec.bucket == L && got.positions == want.positions,
                         logits: dec.logits, probs: dec.probabilities, l64: l64)
        scores.append(s)
        let form = l64 ? "\(kind)-L64" : kind
        rowProbs[form, default: [:]][want.key] = dec.probabilities
        rowLogits[form, default: [:]][want.key] = dec.logits
        out.append(rowJSON(want, s, logits: dec.logits, probs: dec.probabilities, device: got, seconds: dec.seconds,
                           bucket: L))
    }

    /// "long": the graph L4096 apart (its working set on the Mac is 4-6 GB of footprint, over the phone's default memory
    /// limit of about 3.5 GB: a jetsam kill here must not take the other stages with it): the 4 long_3400 rows on JIT and
    /// on AOT, each instance dropped before the next, then W3 with the forms one after the other.
    func stageLong() async -> [String: Any] {
        var j: [String: Any] = [:]
        var ok = true
        partialParent = "long"
        defer {
            partialParent = nil
            partialParentRecord = [:]
        }
        for (kind, sub) in [("jit", "parity_jit"), ("aot", "parity_aot")] {
            let r = await stageParity(kind: kind, key: "long_\(sub)", groups: Self.longGroups)
            j[sub] = r
            partialParentRecord = j
            ok = ok && (r["pass"] as? Bool ?? false)
        }
        let b = await stageBench(key: "long_bench", workloads: ["W3"])
        j["bench"] = b
        ok = ok && (b["pass"] as? Bool ?? false)
        j["pass"] = ok
        return j
    }

    func stageParity(kind: String, key: String, groups: [(name: String, buckets: [Int], vision: Bool, audio: Bool)]? = nil)
        async -> [String: Any] {
        let groups = groups ?? Self.parityGroups
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        var j: [String: Any] = ["kind": kind, "thermal_start": DeviceInfo.thermal()]
        let b0 = await DeviceInfo.battery()
        j["battery_start"] = ["level": b0.level, "state": b0.state, "power": DeviceInfo.powerSource(b0.state)]
        j["free_gb_before"] = DeviceInfo.freeGB(assets)
        if let free = j["free_gb_before"] as? Double, free >= 0 && free < config.minFreeGB {
            line("\(key): STOP free space \(f1(free)) GB < D1_MIN_FREE_GB \(f1(config.minFreeGB)) GB")
            j["error"] = "free space \(free) GB < \(config.minFreeGB) GB"
            j["pass"] = false
            return j
        }
        var scores: [Fixtures.RowScore] = []
        var scoresL64: [Fixtures.RowScore] = []
        var out: [[String: Any]] = []
        var outL64: [[String: Any]] = []
        var images: [[String: Any]] = []
        var clips: [[String: Any]] = []
        var errors: [String] = []
        var loads: [[String: Any]] = []
        var groupsOut: [[String: Any]] = []
        var stopped: String? = nil
        let r0 = ContinuousClock.now
        let staged = stagedBuckets()
        // the groups of graphs that live together (Self.parityGroups); each is dropped before the next one loads
        for g in groups where stopped == nil {
            let buckets = g.buckets.filter { staged.contains($0) }
            if buckets.isEmpty { continue }
            let gkey = "\(key) \(g.name)"
            let d1: D1Omni
            do {
                monitor?.set("\(gkey) init")
                let (x, rec) = try await makeInstance(kind, buckets: buckets, vision: g.vision, audio: g.audio)
                d1 = x
                groupsOut.append(["group": g.name, "instance": rec])
                line("\(gkey): D1Omni on \(kind) assets: init \(f2(rec["init_s"] as? Double ?? -1)) s (tokenizer "
                     + "\(f2(d1.tokenizer.loadSeconds)) s), buckets \(d1.lengths), media \(rec["media_assets"] ?? [])")
            } catch {
                line("\(gkey): ERROR instance: \(error)")
                errors.append("\(g.name) instance: \(error)")
                continue
            }
            monitor?.set(gkey)
            writePartial(key, j.merging(["step": "loads \(g.name)", "loads": loads, "rows": out]) { $1 })
            let gl = await loadAll(d1, key: gkey)
            loads += gl.map { $0.merging(["group": g.name]) { $1 } }
            guard gl.allSatisfy({ $0["pass"] as? Bool == true }) else {
                errors.append("\(g.name): a graph did not load")
                continue
            }
            writePartial(key, j.merging(["step": "rows \(g.name)", "loads": loads, "rows": out]) { $1 })
            // text: the rows whose round-10 bucket is in this group, and (with L64) the rows of at most 64 positions
            for rec in fx.records where rec.kind == "text" {
                if let left = remaining(), left < 20 { stopped = "deadline (\(Int(left)) s left)"; break }
                let wants = fx.rows(of: rec.id, mode: .text)
                let partA = Set(wants.filter { buckets.contains($0.bucket) }.map(\.key))
                let partB = buckets.contains(64) ? Set(wants.filter { fx.l64Keys.contains($0.key) }.map(\.key)) : []
                if partA.isEmpty && partB.isEmpty { continue }
                do {
                    for (want, got) in try matchRows(d1, rec, .text, prefix: 0) {
                        if partA.contains(want.key) {
                            try await decideScore(d1, want, got, prefix: nil, kind: kind, scores: &scores, out: &out)
                        }
                        if partB.contains(want.key) {
                            try await decideScore(d1, want, got, prefix: nil, kind: kind, l64: true, scores: &scoresL64,
                                                  out: &outL64)
                        }
                    }
                } catch {
                    errors.append("\(rec.id): \(error)")
                    line("\(gkey) \(rec.id): ERROR \(error)")
                }
                if out.count % 10 < 4 { writePartial(key, j.merging(["step": "rows \(g.name)", "loads": loads, "rows": out]) { $1 }) }
            }
            line("\(gkey) text: \(scores.filter { $0.mode == "text" }.count) rows so far, L64 \(scoresL64.count) in "
                 + "\(f1(seconds(since: r0))) s")
            // images and clips whose rows' bucket is in this group
            let table: [Float]
            if g.vision {
                do { table = try d1.visionPositionTable() } catch {
                    table = []
                    errors.append("position table: \(error)")
                }
            } else {
                table = []
            }
            for rec in fx.records where rec.kind == "image" && stopped == nil && g.vision {
                guard let b = fx.rows(of: rec.id, mode: .image).first?.bucket, buckets.contains(b) else { continue }
                if let left = remaining(), left < 20 { stopped = "deadline (\(Int(left)) s left)"; break }
                guard let mref = fx.mac.images[rec.id], let file = rec.images.first else { continue }
                var im: [String: Any] = ["id": rec.id, "file": file]
                do {
                    let url = assets.appendingPathComponent("fixtures").appendingPathComponent(file)
                    var t = ContinuousClock.now
                    let rgb = try ImagePreprocess.decode(contentsOf: url)
                    im["decode_ms"] = seconds(since: t) * 1e3
                    im["decoder"] = rgb.decoder
                    im["px_wh"] = [rgb.width, rgb.height]
                    let rgbSHA = sha256Hex(of: rgb.pixels)
                    im["rgb_equal_mac"] = rgbSHA == mref.rgb
                    t = ContinuousClock.now
                    let crops = try ImagePreprocess.crops(rgb)
                    im["crops_ms"] = seconds(since: t) * 1e3
                    let vg = try await d1.vision()
                    var prefix: [Float] = []
                    var cropOut: [[String: Any]] = []
                    for (k, crop) in crops.enumerated() {
                        let t1 = ContinuousClock.now
                        let x = try ImagePreprocess.inputs(crop, table: table)
                        let inputsMs = seconds(since: t1) * 1e3
                        let t2 = ContinuousClock.now
                        let o = try await vg.output(x)
                        let callMs = seconds(since: t2) * 1e3
                        let o2 = try await vg.output(x)
                        prefix += o.prefix(x.tokens * DecisionGraph.hidden)
                        let mc = mref.crops.first { $0.k == k }
                        let outSHA = sha256Hex(of: o)
                        cropOut.append([
                            "k": k, "hw": [crop.height, crop.width], "tokens": x.tokens,
                            "crop_u8_equal_mac": sha256Hex(of: crop.values) == mc?.crop_u8,
                            "pixel_values_equal_mac": sha256Hex(of: x.pixelValues) == mc?.pixel_values,
                            "pos_embed_equal_mac": sha256Hex(of: x.posEmbed) == mc?.pos_embed,
                            "patch_mask_equal_mac": sha256Hex(of: x.patchMask) == mc?.patch_mask,
                            "unshuffle_index_equal_mac": sha256Hex(of: x.unshuffleIndex) == mc?.unshuffle_index,
                            "output_sha256": outSHA, "output_equal_mac": outSHA == mc?.output,
                            "output_max_abs_d_repeat": zip(o, o2).map { abs(Double($0) - Double($1)) }.max() ?? 0,
                            "inputs_ms": inputsMs, "call_ms": callMs,
                        ])
                    }
                    let p = prefix.count / DecisionGraph.hidden
                    let prefixSHA = sha256Hex(of: prefix)
                    im["crops"] = cropOut
                    im["crops_count_equal_mac"] = crops.count == mref.crops.count
                    im["prefix_rows"] = p
                    im["prefix_sha256"] = prefixSHA
                    im["prefix_equal_mac"] = prefixSHA == mref.prefix
                    let allEq = (im["rgb_equal_mac"] as? Bool == true) && (im["prefix_equal_mac"] as? Bool == true)
                        && cropOut.allSatisfy { c in ["crop_u8_equal_mac", "pixel_values_equal_mac", "pos_embed_equal_mac",
                                                      "patch_mask_equal_mac", "unshuffle_index_equal_mac", "output_equal_mac"]
                                                      .allSatisfy { c[$0] as? Bool == true } }
                    im["arrays_equal_mac"] = allEq
                    for (want, got) in try matchRows(d1, rec, .image, prefix: p) {
                        try await decideScore(d1, want, got, prefix: prefix, kind: kind, scores: &scores, out: &out)
                    }
                    line("\(key) \(rec.id): \(rgb.width)x\(rgb.height) (\(rgb.decoder)), \(crops.count) crops, P \(p), arrays = Mac "
                         + "\(allEq) (rgb \(im["rgb_equal_mac"] ?? "?"), prefix \(im["prefix_equal_mac"] ?? "?")), vision call "
                         + "\(cropOut.map { f1($0["call_ms"] as? Double ?? -1) }.joined(separator: "/")) ms")
                } catch {
                    im["error"] = "\(error)"
                    errors.append("\(rec.id): \(error)")
                    line("\(key) \(rec.id): ERROR \(error)")
                }
                images.append(im)
                writePartial(key, j.merging(["step": "rows \(g.name)", "loads": loads, "rows": out, "images": images]) { $1 })
            }
            for rec in fx.records where rec.kind == "audio" && stopped == nil && g.audio {
                guard let b = fx.rows(of: rec.id, mode: .audio).first?.bucket, buckets.contains(b) else { continue }
                if let left = remaining(), left < 20 { stopped = "deadline (\(Int(left)) s left)"; break }
                guard let mref = fx.mac.clips[rec.id], let file = rec.audio else { continue }
                var cl: [String: Any] = ["id": rec.id, "file": file]
                do {
                    let url = assets.appendingPathComponent("fixtures").appendingPathComponent(file)
                    var t = ContinuousClock.now
                    let samples = try AudioPreprocess.samples(contentsOf: url)
                    cl["decode_ms"] = seconds(since: t) * 1e3
                    t = ContinuousClock.now
                    let x = try d1.audioInputs(samples: samples)
                    cl["mel_masks_ms"] = seconds(since: t) * 1e3
                    let ag = try await d1.audio(x.bucket)
                    let t2 = ContinuousClock.now
                    let o = try await ag.output(x)
                    let callMs = seconds(since: t2) * 1e3
                    let o2 = try await ag.output(x)
                    let prefix = Array(o.prefix(x.prefixRows * DecisionGraph.hidden))
                    clipPrefix[kind, default: [:]][rec.id] = prefix
                    let eq: [String: Bool] = [
                        "samples": sha256Hex(of: samples) == mref.samples, "mel": sha256Hex(of: x.mel) == mref.mel,
                        "mask_f": sha256Hex(of: x.maskF) == mref.mask_f, "mask_f2": sha256Hex(of: x.maskF2) == mref.mask_f2,
                        "mask_f4": sha256Hex(of: x.maskF4) == mref.mask_f4, "mask_t": sha256Hex(of: x.maskT) == mref.mask_t,
                        "output": sha256Hex(of: o) == mref.output, "prefix": sha256Hex(of: prefix) == mref.prefix,
                    ]
                    cl["equal_mac"] = eq
                    cl["arrays_equal_mac"] = eq.values.allSatisfy { $0 }
                    cl["bucket_s"] = x.bucket.seconds
                    cl["frames"] = x.frames
                    cl["prefix_rows"] = x.prefixRows
                    cl["bucket_equal_mac"] = x.bucket.seconds == mref.bucket_s && x.frames == mref.frames && x.prefixRows == mref.prefix_rows
                    cl["call_ms"] = callMs
                    cl["output_sha256"] = sha256Hex(of: o)
                    cl["output_max_abs_d_repeat"] = zip(o, o2).map { abs(Double($0) - Double($1)) }.max() ?? 0
                    for (want, got) in try matchRows(d1, rec, .audio, prefix: x.prefixRows) {
                        try await decideScore(d1, want, got, prefix: prefix, kind: kind, scores: &scores, out: &out)
                    }
                    line("\(key) \(rec.id): \(samples.count) samples, bucket \(x.bucket.seconds) s, P \(x.prefixRows), arrays = Mac "
                         + "\(cl["arrays_equal_mac"] ?? "?") (\(eq.filter { !$0.value }.map(\.key).sorted())), audio call \(f1(callMs)) ms")
                } catch {
                    cl["error"] = "\(error)"
                    errors.append("\(rec.id): \(error)")
                    line("\(key) \(rec.id): ERROR \(error)")
                }
                clips.append(cl)
                writePartial(key, j.merging(["step": "rows \(g.name)", "loads": loads, "rows": out, "images": images,
                                             "clips": clips]) { $1 })
            }
            groupsOut[groupsOut.count - 1]["footprint_mb_end"] = DeviceInfo.footprintMB()
            groupsOut[groupsOut.count - 1]["memory"] = monitor?.summary(gkey) ?? [:]
            // d1 (this group's graphs) is released here, before the next group loads
        }
        // summaries
        let all = Fixtures.summarize(scores)
        var byGroup: [String: Any] = [:]
        for g in ["text", "image", "audio"] { byGroup[g] = Fixtures.summarize(scores.filter { $0.mode == g }) }
        // this call's rows only (the "long" stage runs after parity_jit / parity_aot in the same process)
        let doneA = Set(scores.map(\.key)), doneB = Set(scoresL64.map(\.key))
        let control = fx.control(fx.rows.filter { doneA.contains($0.key) }, rowProbs[kind] ?? [:])
        let l64Summary = Fixtures.summarize(scoresL64)
        j["groups"] = groupsOut
        j["loads"] = loads
        j["rows"] = out
        j["rows_l64"] = outL64
        j["images"] = images
        j["clips"] = clips
        j["summary"] = all
        j["summary_by_mode"] = byGroup
        j["summary_l64"] = l64Summary
        j["control"] = control
        j["errors"] = errors
        // the rows of this stage's groups: the round-10 bucket in a group (the long rows are the "long" stage's)
        let groupBuckets = Set(groups.flatMap { $0.buckets }.filter { staged.contains($0) })
        let plannedRows = fx.rows.filter { groupBuckets.contains($0.bucket) }.count
        j["rows_planned"] = plannedRows
        j["rows_done"] = scores.count
        j["rows_l64_planned"] = groupBuckets.contains(64) ? fx.l64Keys.count : 0
        j["rows_l64_done"] = scoresL64.count
        j["seconds_rows"] = seconds(since: r0)
        if let stopped { j["stopped"] = stopped }
        let mediaEqual = images.filter { $0["arrays_equal_mac"] as? Bool == true }.count
            + clips.filter { $0["arrays_equal_mac"] as? Bool == true }.count
        j["media_arrays_equal_mac"] = ["items": images.count + clips.count, "equal": mediaEqual]
        if kind == "aot" {
            for (a, b, name, done) in [("aot", "jit", "vs_jit", doneA), ("aot-L64", "jit-L64", "vs_jit_l64", doneB)] {
                guard let jp = rowProbs[b], let jl = rowLogits[b], !done.isEmpty else { continue }
                var eqP = 0, eqL = 0, n = 0
                var maxDp = 0.0
                for (k, p) in rowProbs[a] ?? [:] where done.contains(k) {
                    guard let q = jp[k], let la = rowLogits[a]?[k], let lj = jl[k] else { continue }
                    n += 1
                    if p.map(\.bitPattern) == q.map(\.bitPattern) { eqP += 1 }
                    if la.map(\.bitPattern) == lj.map(\.bitPattern) { eqL += 1 }
                    maxDp = max(maxDp, zip(p, q).map { abs(Double($0) - Double($1)) }.max() ?? 0)
                }
                j[name] = ["rows": n, "probs_bit_equal": eqP, "logits_bit_equal": eqL, "max_abs_dp": maxDp]
                line("\(key) \(a) vs \(b) (this process): logits bit-equal \(eqL)/\(n), p bit-equal \(eqP)/\(n), max|dp| \(f6(maxDp))")
            }
        }
        if !scoresL64.isEmpty { line("\(key) L64: " + l64Summary.line) }
        let b1 = await DeviceInfo.battery()
        j["battery_end"] = ["level": b1.level, "state": b1.state, "power": DeviceInfo.powerSource(b1.state)]
        j["thermal_end"] = DeviceInfo.thermal()
        j["free_gb_after"] = DeviceInfo.freeGB(assets)
        j["storage_after"] = DeviceInfo.storageSnapshot()
        line("\(key): " + all.line + " | control \(control["status"] ?? "?") (must FAIL) | media arrays = Mac "
             + "\(mediaEqual)/\(images.count + clips.count)")
        for g in ["text", "image", "audio"] { line("\(key) \(g): " + ((byGroup[g] as? [String: Any]) ?? [:]).line) }
        let l64OK = !groupBuckets.contains(64)
            || (scoresL64.count == fx.l64Keys.count && (l64Summary["bar_pass"] as? Bool ?? false))
        // a control with no pair of rows of one class (the 4 long rows) proves nothing either way
        let controlOK = (control["paired_rows"] as? Int ?? 0) == 0 || (control["caught"] as? Bool ?? false)
        j["pass"] = errors.isEmpty && stopped == nil && scores.count == plannedRows && (all["bar_pass"] as? Bool ?? false)
            && controlOK && l64OK
        return j
    }

    /// The parity groups: the graphs that live together in one instance, dropped before the next group loads. L4096 is
    /// the "long" stage's (longGroups).
    static let parityGroups: [(name: String, buckets: [Int], vision: Bool, audio: Bool)] = [
        ("short", [64, 256], true, true), ("L2048", [2048], true, false),
    ]
    static let longGroups: [(name: String, buckets: [Int], vision: Bool, audio: Bool)] = [("L4096", [4096], false, false)]

    // MARK: - bench

    /// What one workload needs, prepared once before its series: rows, bucket, the media graph's inputs.
    private struct Prepared: @unchecked Sendable {
        let workload: Fixtures.Workload
        let rows: [Row]
        let keys: [String]
        let bucket: Int
        let crops: [CropInputs]
        let audio: AudioInputs?
        let record: Fixtures.Record
    }

    private func prepare(_ w: Fixtures.Workload, _ d1: D1Omni) async throws -> Prepared {
        guard let fx = fixtures, let rec = fx.byID[w.record] else { throw GateError.fixture("no record \(w.record)") }
        let mode = Mode(rawValue: w.mode) ?? .text
        var crops: [CropInputs] = []
        var audio: AudioInputs? = nil
        var p = 0
        if mode == .image, let f = rec.images.first {
            let rgb = try ImagePreprocess.decode(contentsOf: assets.appendingPathComponent("fixtures").appendingPathComponent(f))
            crops = try d1.imageInputs([rgb])
            p = crops.reduce(0) { $0 + $1.tokens }
        } else if mode == .audio, let f = rec.audio {
            let x = try d1.audioInputs(samples: try AudioPreprocess.samples(contentsOf: assets.appendingPathComponent("fixtures").appendingPathComponent(f)))
            audio = x
            p = x.prefixRows
        }
        let all = try d1.rows(state: rec.state, questions: rec.questions, mode: mode, prefixLength: p)
        let rows = try w.qids.map { q in
            guard let r = all.first(where: { $0.qid.unicodeScalars.elementsEqual(q.unicodeScalars) }) else {
                throw GateError.fixture("\(w.record): no question \(q)")
            }
            return r
        }
        let L = try rows.map { try d1.bucket(for: $0) }.max() ?? 256
        return Prepared(workload: w, rows: rows, keys: rows.map { "\(w.record)/\($0.qid)/\(mode.rawValue)" }, bucket: L,
                        crops: crops, audio: audio, record: rec)
    }

    /// One decision of a workload on one form: media graph (if any) -> decision graph per row -> readout.
    private nonisolated func decide(_ p: Prepared, _ f: BenchForm)
        async throws -> (ms: Double, media: Double, inputs: Double, graph: Double, readout: Double, logits: [[Float]],
                         probs: [[Float]], prefixSHA: String?) {
        let t = ContinuousClock.now
        var prefix: [Float]? = nil
        if let a = p.audio, let ag = f.audio {
            prefix = Array(try await ag.output(a).prefix(a.prefixRows * DecisionGraph.hidden))
        } else if !p.crops.isEmpty, let vg = f.vision {
            var pf: [Float] = []
            for c in p.crops { pf += try await vg.output(c).prefix(c.tokens * DecisionGraph.hidden) }
            prefix = pf
        }
        let tm = ContinuousClock.now
        var logits: [[Float]] = [], probs: [[Float]] = []
        var pi = 0.0, pg = 0.0, pr = 0.0
        for row in p.rows {
            let a = ContinuousClock.now
            let x = try GraphInputs(row: row, length: f.bucket)
            let b = ContinuousClock.now
            let s = try await f.decision.scores(x, prefix: prefix)
            let c = ContinuousClock.now
            let z = x.markers.map { s[$0] }
            probs.append(Readout.probabilities(logits: z, question: row.question, calibrate: row.calibrate, config: f.d1.config))
            let e = ContinuousClock.now
            logits.append(z)
            pi += D1Clock.seconds(a, b)
            pg += D1Clock.seconds(b, c)
            pr += D1Clock.seconds(c, e)
        }
        let ms = seconds(since: t) * 1e3
        return (ms, D1Clock.seconds(t, tm) * 1e3, pi * 1e3, pg * 1e3, pr * 1e3, logits, probs, prefix.map { sha256Hex(of: $0) })
    }

    /// Series of rounds (each round = one decision per form, the order alternating), timed per the bench rules.
    private func series(_ key: String, _ p: Prepared, forms: [BenchForm], timed: Int, warmFirst: Int, warmLater: Int,
                        maxSeriesS: Double, restS: Double) async -> [String: Any] {
        let fx = fixtures!
        var decisions: [[String: Any]] = []
        var seriesOut: [[String: Any]] = []
        var samples: [String: [Double]] = [:], warm: [String: [Double]] = [:]
        var parts: [String: [(Double, Double, Double, Double)]] = [:]
        var firstBits: [String: [[UInt32]]] = [:]
        var firstPrefix: [String: String] = [:]
        var sameEveryCall = true, prefixSame = true, formsEqual = true
        var worstDp = 0.0
        var argmaxOK = true
        var timedDone = 0, round = 0, index = 0
        var stopped: String? = nil
        let b0 = ContinuousClock.now
        while timedDone < timed {
            if index > 0 && restS > 0 {
                line("\(key): resting \(f1(restS)) s before series \(index + 1)")
                try? await Task.sleep(for: .seconds(restS))
            }
            // a serious thermal state: rest in steps until it falls (at most D1_SERIOUS_CAP s per wait)
            var seriousWait = 0.0
            while ["serious", "critical"].contains(DeviceInfo.thermal()) && seriousWait < config.seriousCap {
                line("\(key): thermal \(DeviceInfo.thermal()) before series \(index + 1): resting \(Int(config.seriousRest)) s")
                try? await Task.sleep(for: .seconds(config.seriousRest))
                seriousWait += config.seriousRest
            }
            if let left = remaining(), left < maxSeriesS + 25 {
                stopped = "deadline (\(Int(left)) s left)"
                break
            }
            let warmN = index == 0 ? warmFirst : warmLater
            let bs = await DeviceInfo.battery()
            BatteryCache.shared.set(bs)
            let s0 = ContinuousClock.now
            let thermalStart = DeviceInfo.thermal()
            var r = 0, lastRound = 0.0, timedHere = 0
            var cut: String? = nil
            while r < warmN || timedDone < timed {
                let elapsedS = seconds(since: s0)
                // at least one timed round per series (a series that only warms up would never finish the bench)
                if r >= warmN && timedHere > 0 && elapsedS + lastRound > maxSeriesS { cut = "time"; break }
                if ["serious", "critical"].contains(DeviceInfo.thermal()) { cut = "thermal \(DeviceInfo.thermal())"; break }
                let order = round % 2 == 0 ? forms : Array(forms.reversed())
                let isWarm = r < warmN
                let rs = ContinuousClock.now
                var bitsThisRound: [String: [[UInt32]]] = [:]
                for f in order {
                    let tStart = seconds(since: s0)
                    do {
                        let d = try await decide(p, f)
                        let bits = d.logits.map { $0.map(\.bitPattern) }
                        bitsThisRound[f.name] = bits
                        if let fb = firstBits[f.name] { if fb != bits { sameEveryCall = false } } else { firstBits[f.name] = bits }
                        if let ph = d.prefixSHA {
                            if let fp = firstPrefix[f.name] { if fp != ph { prefixSame = false } } else { firstPrefix[f.name] = ph }
                        }
                        for (k, pr) in zip(p.keys, d.probs) {
                            guard let o = fx.oracle[k] else { continue }
                            let pd = pr.map(Double.init)
                            worstDp = max(worstDp, zip(pd, o.probs).map { abs($0 - $1) }.max() ?? .infinity)
                            if Fixtures.firstArgmax(pd) != o.argmax_index && !o.near_tie { argmaxOK = false }
                        }
                        let b = BatteryCache.shared.get()
                        decisions.append([
                            "form": f.name, "round": round, "series": index, "warmup": isWarm,
                            "t_series_s": tStart, "t_bench_s": seconds(since: b0), "ms": d.ms, "media_ms": d.media,
                            "inputs_ms": d.inputs, "graph_ms": d.graph, "readout_ms": d.readout,
                            "thermal": DeviceInfo.thermal(), "battery_level": b.level, "battery_state": b.state,
                            "footprint_mb": DeviceInfo.footprintMB(),
                        ])
                        if isWarm { warm[f.name, default: []].append(d.ms) } else {
                            samples[f.name, default: []].append(d.ms)
                            parts[f.name, default: []].append((d.media, d.inputs, d.graph, d.readout))
                        }
                    } catch {
                        stopped = "\(f.name): \(error)"
                        line("\(key) \(f.name): ERROR \(error)")
                        break
                    }
                }
                if stopped != nil { break }
                if forms.count > 1, let a = bitsThisRound[forms[0].name] {
                    // forms of one bucket (JIT and AOT of the same graph) must agree; other buckets are other graphs
                    _ = a
                    for f in forms {
                        guard let first = forms.first(where: { $0.bucket == f.bucket && $0.name != f.name }),
                              let x = bitsThisRound[f.name], let y = bitsThisRound[first.name] else { continue }
                        if x != y { formsEqual = false }
                    }
                }
                lastRound = seconds(since: rs)
                r += 1
                round += 1
                if !isWarm {
                    timedDone += 1
                    timedHere += 1
                }
            }
            let be = await DeviceInfo.battery()
            BatteryCache.shared.set(be)
            seriesOut.append(["index": index, "warmup_rounds": min(r, warmN), "timed_rounds": timedHere,
                              "duration_s": seconds(since: s0), "t_bench_start_s": D1Clock.seconds(b0, s0),
                              "thermal_start": thermalStart, "thermal_end": DeviceInfo.thermal(),
                              "battery_start": ["level": bs.level, "state": bs.state, "power": DeviceInfo.powerSource(bs.state)],
                              "battery_end": ["level": be.level, "state": be.state, "power": DeviceInfo.powerSource(be.state)],
                              "serious_wait_s": seriousWait, "cut": cut ?? "done"])
            line("\(key) series \(index + 1): \(timedHere) timed rounds (+\(min(r, warmN)) warm-up) in \(f1(seconds(since: s0))) s, "
                 + "thermal \(thermalStart) -> \(DeviceInfo.thermal()), battery \(String(format: "%.0f", be.level * 100)) % "
                 + "\(DeviceInfo.powerSource(be.state))" + (cut.map { " (cut: \($0))" } ?? ""))
            index += 1
            if stopped != nil { break }
            if r == 0 { stopped = "a series ran no round"; break }
        }
        var formsOut: [String: Any] = [:]
        for f in forms {
            let xs = samples[f.name] ?? []
            let ps = parts[f.name] ?? []
            func med(_ k: KeyPath<(Double, Double, Double, Double), Double>) -> Double { median(ps.map { $0[keyPath: k] }) }
            let thermals = decisions.filter { $0["form"] as? String == f.name && $0["warmup"] as? Bool == false }
                .compactMap { $0["thermal"] as? String }
            formsOut[f.name] = ["n": xs.count, "median_ms": median(xs), "p10_ms": percentile(xs, 0.1),
                                "p90_ms": percentile(xs, 0.9), "mean_ms": xs.isEmpty ? Double.nan : xs.reduce(0, +) / Double(xs.count),
                                "min_ms": xs.min() ?? Double.nan, "max_ms": xs.max() ?? Double.nan, "samples_ms": xs,
                                "warmup_ms": warm[f.name] ?? [],
                                "parts_median_ms": ["media_ms": med(\.0), "inputs_ms": med(\.1), "graph_ms": med(\.2),
                                                    "readout_ms": med(\.3)],
                                "thermal_of_timed": Dictionary(grouping: thermals, by: { $0 }).mapValues(\.count)]
        }
        var j: [String: Any] = ["forms": formsOut, "series": seriesOut, "decisions": decisions,
                                "output_check": ["bits_same_every_call_per_form": sameEveryCall,
                                                 "prefix_same_every_call_per_form": prefixSame,
                                                 "forms_bit_equal_every_round": formsEqual,
                                                 "max_abs_dp_vs_oracle": worstDp, "argmax_equal_every_call": argmaxOK,
                                                 "status": sameEveryCall && prefixSame && argmaxOK && worstDp <= 0.02 ? "PASS" : "FAIL"],
                                "timed_target": timed, "timed_done": timedDone, "rounds": round]
        if let stopped { j["stopped"] = stopped }
        j["first_call_bits"] = firstBits.mapValues { sha256Hex(of: $0.flatMap { $0 }) }
        return j
    }

    private func benchForm(_ name: String, _ d1: D1Omni, bucket L: Int, mode: Mode, audioBucket: AudioBucket?) async throws -> BenchForm {
        let g = try await d1.graph(L)
        var vg: VisionGraph? = nil
        var ag: AudioGraph? = nil
        if mode == .image { vg = try await d1.vision() }
        if mode == .audio, let b = audioBucket { ag = try await d1.audio(b) }
        return BenchForm(name: name, bucket: L, d1: d1, decision: g, vision: vg, audio: ag)
    }

    private func waitForNominal(_ key: String) async -> [String: Any] {
        let cap = config.waitNominal
        let before = DeviceInfo.thermal()
        let c0 = ContinuousClock.now
        if before != "nominal" { line("\(key): thermal \(before); waiting for nominal (5 s steps, cap \(Int(cap)) s)") }
        while DeviceInfo.thermal() != "nominal" && seconds(since: c0) < cap {
            try? await Task.sleep(for: .seconds(5))
        }
        let after = DeviceInfo.thermal()
        if before != "nominal" {
            line("\(key): thermal \(after) after \(f1(seconds(since: c0))) s" + (after == "nominal" ? "" : " (cap reached: running anyway)"))
        }
        return ["cap_s": cap, "state_before": before, "state_after": after, "waited_s": seconds(since: c0),
                "reached_nominal": after == "nominal"]
    }

    /// The preprocessing of a workload, timed apart (5 runs after 1 dropped): tokenize; decode, crops and patches; mel.
    private func preprocessTimes(_ p: Prepared, _ d1: D1Omni) -> [String: Any] {
        func time(_ body: () throws -> Void) -> [Double] {
            var v: [Double] = []
            for i in 0..<6 {
                let t = ContinuousClock.now
                try? body()
                if i > 0 { v.append(seconds(since: t) * 1e3) }
            }
            return v
        }
        let rec = p.record
        let mode = Mode(rawValue: p.workload.mode) ?? .text
        let fixturesDir = assets.appendingPathComponent("fixtures")
        var out: [String: Any] = [:]
        let pl = mode == .text ? 0 : (p.audio?.prefixRows ?? p.crops.reduce(0) { $0 + $1.tokens })
        out["tokenize_ms"] = median(time { _ = try d1.rows(state: rec.state, questions: rec.questions, mode: mode, prefixLength: pl) })
        if mode == .image, let f = rec.images.first {
            let u = fixturesDir.appendingPathComponent(f)
            out["decode_ms"] = median(time { _ = try ImagePreprocess.decode(contentsOf: u) })
            if let rgb = try? ImagePreprocess.decode(contentsOf: u) {
                out["crops_resize_ms"] = median(time { _ = try ImagePreprocess.crops(rgb) })
                out["patches_positions_ms"] = median(time { _ = try d1.imageInputs([rgb]) })
            }
        } else if mode == .audio, let f = rec.audio {
            let u = fixturesDir.appendingPathComponent(f)
            out["wav_ms"] = median(time { _ = try AudioPreprocess.samples(contentsOf: u) })
            if let s = try? AudioPreprocess.samples(contentsOf: u) {
                out["mel_masks_ms"] = median(time { _ = try d1.audioInputs(samples: s) })
            }
        }
        return out
    }

    /// One workload's forms on fresh instances: every kind of D1_BENCH_FORMS x every bucket the workload times, with
    /// the media graph it needs; the loads recorded.
    private func benchForms(_ w: Fixtures.Workload, kinds: [String], key: String) async throws
        -> (forms: [BenchForm], prepared: Prepared, instances: [[String: Any]], loads: [[String: Any]]) {
        let mode = Mode(rawValue: w.mode) ?? .text
        let staged = stagedBuckets()
        let buckets = w.buckets.filter { staged.contains($0) }
        guard !buckets.isEmpty else { throw GateError.assets("\(w.id): none of the buckets \(w.buckets) is staged") }
        var forms: [BenchForm] = []
        var recs: [[String: Any]] = []
        var loads: [[String: Any]] = []
        var prepared: Prepared? = nil
        for kind in kinds {
            let (d1, rec) = try await makeInstance(kind, buckets: buckets, vision: mode == .image, audio: mode == .audio)
            recs.append(rec)
            loads += await loadAll(d1, key: "\(key) \(kind)")
            let p = try await prepare(w, d1)
            if prepared == nil { prepared = p }
            for L in buckets {
                forms.append(try await benchForm("\(kind)-L\(L)", d1, bucket: L, mode: mode, audioBucket: p.audio?.bucket))
            }
        }
        return (forms, prepared!, recs, loads)
    }

    func stageBench(key stageKey: String, workloads: [String]) async -> [String: Any] {
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        var j: [String: Any] = ["forms": config.benchForms, "workloads_planned": workloads]
        monitor?.set(stageKey)
        let timed = config.benchTimed ?? fx.bench.series.timed
        let warmFirst = config.benchWarmup ?? fx.bench.series.warmup_first
        let warmLater = min(warmFirst, fx.bench.series.warmup_later)
        let maxS = config.benchSeriesS ?? fx.bench.series.max_series_s
        let rest = config.benchRest ?? fx.bench.series.rest_s
        j["rules"] = ["timed": timed, "warmup_first": warmFirst, "warmup_later": warmLater, "max_series_s": maxS, "rest_s": rest,
                      "wait_nominal_cap_s": config.waitNominal, "serious_rest_s": config.seriousRest,
                      "serious_cap_s": config.seriousCap]
        var items: [[String: Any]] = []
        var ok = true
        let order = fx.bench.order ?? fx.bench.workloads.map(\.id)
        let planned = order.compactMap { id in fx.bench.workloads.first { $0.id == id } }
            .filter { workloads.contains($0.id) }
        for w in planned {
            if let left = remaining(), left < maxS + 30 {
                line("bench \(w.id): SKIPPED — \(Int(left)) s left before the deadline")
                items.append(["workload": w.id, "skipped": true, "reason": "deadline"])
                ok = false
                continue
            }
            let key = "bench \(w.id)"
            monitor?.set(key)
            var it: [String: Any] = ["workload": w.id, "what": w.what, "record": w.record, "qids": w.qids, "mode": w.mode,
                                     "buckets": w.buckets, "interleave": w.interleave ?? true]
            do {
                // interleaved: every form loaded at once, alternating per round; otherwise one kind after the other
                let blocks: [[String]] = (w.interleave ?? true) ? [config.benchForms] : config.benchForms.map { [$0] }
                var merged: [String: Any] = [:]
                var formsOut: [String: Any] = [:]
                var blockRecs: [[String: Any]] = []
                var firstBits: [String: String] = [:]
                var blockOK = true
                for kinds in blocks {
                    let (forms, p, recs, loads) = try await benchForms(w, kinds: kinds, key: key)
                    it["positions"] = p.rows.map(\.positions)
                    it["prefix_rows"] = p.rows.first?.prefixLength ?? 0
                    it["crops"] = p.crops.count
                    it["audio_bucket_s"] = p.audio?.bucket.seconds ?? 0
                    let wait = await waitForNominal(key)
                    let s = await series(key, p, forms: forms, timed: timed, warmFirst: warmFirst, warmLater: warmLater,
                                         maxSeriesS: maxS, restS: rest)
                    for (k, v) in s["forms"] as? [String: Any] ?? [:] { formsOut[k] = v }
                    if let fb = s["first_call_bits"] as? [String: String] { firstBits.merge(fb) { a, _ in a } }
                    blockRecs.append(["kinds": kinds, "instances": recs, "loads": loads, "wait_nominal": wait,
                                      "series": s["series"] ?? [], "decisions": s["decisions"] ?? [],
                                      "output_check": s["output_check"] ?? [:], "stopped": s["stopped"] ?? NSNull()])
                    if merged.isEmpty { merged["preprocess_ms"] = preprocessTimes(p, forms[0].d1) }
                    let oc = s["output_check"] as? [String: Any] ?? [:]
                    blockOK = blockOK && (oc["status"] as? String == "PASS") && s["stopped"] == nil
                    // the forms of this block are released here
                }
                // not interleaved: JIT = AOT on the first call of each block, per bucket
                var crossEqual: Bool? = nil
                if blocks.count > 1 {
                    crossEqual = true
                    for L in w.buckets {
                        let byKind = config.benchForms.compactMap { firstBits["\($0)-L\(L)"] }
                        if byKind.count > 1 && Set(byKind).count > 1 { crossEqual = false }
                    }
                }
                it["forms"] = formsOut
                it["blocks"] = blockRecs
                it["preprocess_ms"] = merged["preprocess_ms"] ?? [:]
                let checks = blockRecs.compactMap { $0["output_check"] as? [String: Any] }
                it["output_check"] = [
                    "status": checks.allSatisfy { $0["status"] as? String == "PASS" } ? "PASS" : "FAIL",
                    "forms_bit_equal_every_round": checks.allSatisfy { $0["forms_bit_equal_every_round"] as? Bool ?? false },
                    "max_abs_dp_vs_oracle": checks.compactMap { $0["max_abs_dp_vs_oracle"] as? Double }.max() ?? Double.nan,
                    "first_calls_bit_equal_across_blocks": crossEqual as Any? ?? NSNull(),
                ]
                let oc = it["output_check"] as? [String: Any] ?? [:]
                line("\(key) (\(w.what), buckets \(w.buckets)): " + formsOut.keys.sorted().map { k in
                    let m = formsOut[k] as? [String: Any] ?? [:]
                    return "\(k) median \(f2(m["median_ms"] as? Double ?? .nan)) ms (p10 \(f2(m["p10_ms"] as? Double ?? .nan)), p90 "
                        + "\(f2(m["p90_ms"] as? Double ?? .nan)), n \(m["n"] ?? 0))"
                }.joined(separator: ", ") + " | outputs \(oc["status"] ?? "?") (JIT = AOT \(oc["forms_bit_equal_every_round"] ?? "?")"
                    + (crossEqual.map { ", first calls across blocks \($0)" } ?? "") + ", max|dp| "
                    + "\(f6(oc["max_abs_dp_vs_oracle"] as? Double ?? .nan)))")
                ok = ok && blockOK && (crossEqual ?? true)
            } catch {
                line("\(key): ERROR \(error)")
                it["error"] = "\(error)"
                it["error_detail"] = Self.errorRecord(error)
                ok = false
            }
            items.append(it)
            j["items"] = items
            writePartial(stageKey, j)
            monitor?.set(stageKey)
        }
        j["items"] = items
        j["pass"] = ok && !items.isEmpty
        return j
    }

    // MARK: - Neural Engine

    /// A Neural Engine graph's file: "aot" = ane/<name>/'s .aimodelc (compiled for the Neural Engine), "jit" = the
    /// bundle's .aimodel (jit/<name>/, metadata.json assets.main), which the runtime specializes here.
    private func aneAsset(_ source: String, _ name: String) throws -> URL {
        if source == "jit" {
            let folder = assetURL(config.jitDir).appendingPathComponent(name)
            let meta = try JSONParser.parse(Data(contentsOf: folder.appendingPathComponent("metadata.json")))
            guard let main = meta["assets"]?["main"]?.string else { throw GateError.assets("\(folder.path): no assets.main") }
            return folder.appendingPathComponent(main)
        }
        return try aimodelc(in: assetURL(config.aneDir).appendingPathComponent(name))
    }

    /// One way of loading a Neural Engine graph: its file and the regions the file holds (an AOT compiled for the Neural
    /// Engine names them; a .aimodel has none: its regions form at specialization, in the Core AI cache).
    private func aneAttempt(_ source: String, _ url: URL) -> [String: Any] {
        let r = DeviceInfo.aneRegions(url)
        return ["source": source, "asset": rel(url), "asset_bytes": DeviceInfo.tree(url).bytes, "ane_regions": r.regions,
                "ane_region_entries": r.entries]
    }

    /// The Neural Engine region entries the load added to this app's Core AI cache (a JIT specialization for the Neural
    /// Engine writes them there).
    private func aneCacheDelta(_ before: (regions: Int, entries: Int)) -> [String: Any] {
        let after = DeviceInfo.aneRegions(DeviceInfo.coreAICacheDir())
        return ["cache_ane_region_names_before": before.regions, "cache_ane_region_names_after": after.regions,
                "cache_ane_region_entries_added": after.entries - before.entries]
    }

    func stageANE() async -> [String: Any] {
        guard let fx = fixtures else { return ["error": "no fixtures", "pass": false] }
        // the GPU side: the JIT decision graph L256 and the audio graph (the rows' graph and the W1 / W2 / W5 GPU forms)
        let gpuKind = "jit"
        let gpu: D1Omni
        var j: [String: Any] = ["parts": config.aneParts, "gpu_form": gpuKind, "options": describe(Self.aneOptions)]
        do {
            let (x, rec) = try await makeInstance(gpuKind, buckets: [256], vision: false, audio: true)
            gpu = x
            j["gpu_instance"] = rec
            j["gpu_loads"] = await loadAll(gpu, key: "ane gpu")
        } catch {
            line("ane: ERROR the GPU instance: \(error)")
            return ["error": "GPU instance: \(error)", "pass": false]
        }
        var ok = true
        let timed = config.benchTimed ?? fx.bench.series.timed
        let warmFirst = config.benchWarmup ?? fx.bench.series.warmup_first
        let warmLater = min(warmFirst, fx.bench.series.warmup_later)
        let maxS = config.benchSeriesS ?? fx.bench.series.max_series_s
        let rest = config.benchRest ?? fx.bench.series.rest_s

        if config.aneParts.contains("audio") {
            var a: [String: Any] = [:]
            do {
                var ag: AudioGraph? = nil
                let bucket = AudioBucket(seconds: 10)
                var attempts: [[String: Any]] = []
                for source in config.aneSources where ag == nil {
                    let url = try aneAsset(source, "audio-fp16-10s")
                    var at = aneAttempt(source, url)
                    writePartial("ane", j.merging(["audio": a.merging(["attempts": attempts + [at]]) { $1 }, "step": "audio load"]) { $1 })
                    let c0 = DeviceInfo.aneRegions(DeviceInfo.coreAICacheDir())
                    at["load"] = await timedLoad("ane", "audio 10 s (Neural Engine, \(source))") {
                        let g = try await AudioGraph(contentsOf: url, bucket: bucket, options: Self.aneOptions)
                        ag = g
                        return (g.graph.url, g.graph.kind, g.graph.loadSeconds.model, g.graph.loadSeconds.function,
                                JSONWriter.compact(g.graph.descriptor))
                    }
                    at.merge(aneCacheDelta(c0)) { $1 }
                    attempts.append(at)
                }
                a["attempts"] = attempts
                a.merge(attempts.last ?? [:]) { $1 }
                guard let ag else { throw GateError.assets("the Neural Engine audio graph did not load (\(config.aneSources))") }
                var clipsOut: [[String: Any]] = []
                var scores: [Fixtures.RowScore] = []
                var rowsOut: [[String: Any]] = []
                var maxDpGPU = 0.0, nDpGPU = 0
                for rec in fx.records where rec.kind == "audio" {
                    guard let file = rec.audio else { continue }
                    let samples = try AudioPreprocess.samples(contentsOf: assets.appendingPathComponent("fixtures").appendingPathComponent(file))
                    let x = try gpu.audioInputs(samples: samples)
                    let t = ContinuousClock.now
                    let o = try await ag.output(x)
                    let callMs = seconds(since: t) * 1e3
                    let o2 = try await ag.output(x)
                    let prefix = Array(o.prefix(x.prefixRows * DecisionGraph.hidden))
                    var gp = clipPrefix[gpuKind]?[rec.id]
                    if gp == nil { gp = Array(try await gpu.audio(x.bucket).output(x).prefix(x.prefixRows * DecisionGraph.hidden)) }
                    let d = zip(prefix, gp!).map { abs(Double($0) - Double($1)) }
                    let mref = fx.mac.clips[rec.id]
                    clipsOut.append(["id": rec.id, "prefix_rows": x.prefixRows, "call_ms": callMs,
                                     "prefix_sha256": sha256Hex(of: prefix), "prefix_equal_mac_gpu": sha256Hex(of: prefix) == mref?.prefix,
                                     "prefix_bit_equal_gpu": prefix.map(\.bitPattern) == gp!.map(\.bitPattern),
                                     "prefix_max_abs_d_gpu": d.max() ?? .nan,
                                     "prefix_mean_abs_d_gpu": d.isEmpty ? Double.nan : d.reduce(0, +) / Double(d.count),
                                     "prefix_finite": prefix.allSatisfy(\.isFinite),
                                     "output_max_abs_d_repeat": zip(o, o2).map { abs(Double($0) - Double($1)) }.max() ?? 0])
                    for (want, got) in try matchRows(gpu, rec, .audio, prefix: x.prefixRows) {
                        let dec = try await gpu.decide(got, prefix: prefix)
                        let s = fx.score(want, idsEqual: got.ids == want.ids, markersEqual: got.markers == want.markers,
                                         bucketEqual: dec.bucket == want.bucket, logits: dec.logits, probs: dec.probabilities)
                        scores.append(s)
                        var rj = rowJSON(want, s, logits: dec.logits, probs: dec.probabilities, device: got, seconds: dec.seconds)
                        if let q = rowProbs[gpuKind]?[want.key] {
                            let dg = zip(dec.probabilities, q).map { abs(Double($0) - Double($1)) }.max() ?? .nan
                            rj["max_abs_dp_gpu"] = dg
                            maxDpGPU = max(maxDpGPU, dg)
                            nDpGPU += 1
                        }
                        rowsOut.append(rj)
                    }
                    line("ane audio \(rec.id): call \(f1(callMs)) ms, prefix vs GPU max|d| \(f6(d.max() ?? .nan)), repeat max|d| "
                         + "\(f6(clipsOut.last?["output_max_abs_d_repeat"] as? Double ?? .nan))")
                }
                let sum = Fixtures.summarize(scores)
                a["clips"] = clipsOut
                a["rows"] = rowsOut
                a["summary"] = sum
                a["rows_max_abs_dp_gpu"] = nDpGPU > 0 ? maxDpGPU as Any : NSNull()
                a["rows_compared_gpu"] = nDpGPU
                line("ane audio rows (ANE audio graph + GPU \(gpuKind) decision): " + sum.line + " | vs GPU rows "
                     + gpuRowsDiff(maxDpGPU, nDpGPU))
                writePartial("ane", j.merging(["audio": a, "step": "audio ms"]) { $1 })
                // W5 with the ANE audio graph against W5 on the GPU, the same series
                if let w5 = fx.bench.workloads.first(where: { $0.id == "W5" }) {
                    let p = try await prepare(w5, gpu)
                    let gpuForm = try await benchForm("gpu-\(gpuKind)", gpu, bucket: p.bucket, mode: .audio, audioBucket: p.audio?.bucket)
                    let aneForm = BenchForm(name: "ane-audio", bucket: gpuForm.bucket, d1: gpu, decision: gpuForm.decision,
                                            vision: nil, audio: ag)
                    a["wait_nominal"] = await waitForNominal("ane W5")
                    let s = await series("ane W5", p, forms: [aneForm, gpuForm], timed: timed, warmFirst: warmFirst,
                                         warmLater: warmLater, maxSeriesS: maxS, restS: rest)
                    var s2 = s
                    if var oc = s2["output_check"] as? [String: Any] {
                        // the two forms differ by design (another unit): only the per-form checks and the oracle hold
                        oc["forms_bit_equal_every_round_expected"] = false
                        s2["output_check"] = oc
                    }
                    a["w5"] = s2
                    let fo = s["forms"] as? [String: Any] ?? [:]
                    let ane = fo["ane-audio"] as? [String: Any] ?? [:], g = fo["gpu-\(gpuKind)"] as? [String: Any] ?? [:]
                    line("ane W5: ANE audio median \(f2(ane["median_ms"] as? Double ?? .nan)) ms (audio graph "
                         + "\(f2((ane["parts_median_ms"] as? [String: Any])?["media_ms"] as? Double ?? .nan)) ms) vs GPU "
                         + "\(f2(g["median_ms"] as? Double ?? .nan)) ms (audio graph "
                         + "\(f2((g["parts_median_ms"] as? [String: Any])?["media_ms"] as? Double ?? .nan)) ms)")
                }
                a["pass"] = (sum["bar_pass"] as? Bool ?? false)
            } catch {
                line("ane audio: ERROR \(error)")
                a["error"] = "\(error)"
                a["error_detail"] = Self.errorRecord(error)
                a["pass"] = false
            }
            ok = ok && (a["pass"] as? Bool ?? false)
            j["audio"] = a
            writePartial("ane", j)
        }

        if config.aneParts.contains("decision") {
            if let left = remaining(), left < 90 {
                line("ane decision: SKIPPED — \(Int(left)) s left before the deadline")
                j["decision"] = ["skipped": true, "reason": "deadline"]
            } else {
                var d: [String: Any] = [:]
                do {
                    var dg: DecisionGraph? = nil
                    var attempts: [[String: Any]] = []
                    for source in config.aneSources where dg == nil {
                        let url = try aneAsset(source, "fp16-L256")
                        var at = aneAttempt(source, url)
                        writePartial("ane", j.merging(["decision": d.merging(["attempts": attempts + [at]]) { $1 },
                                                       "step": "decision load"]) { $1 })
                        let c0 = DeviceInfo.aneRegions(DeviceInfo.coreAICacheDir())
                        at["load"] = await timedLoad("ane", "decide L256 (Neural Engine, \(source))") {
                            let g = try await DecisionGraph(contentsOf: url, length: 256, options: Self.aneOptions)
                            dg = g
                            return (g.url, g.kind, g.loadSeconds.model, g.loadSeconds.function, JSONWriter.compact(g.descriptor))
                        }
                        at.merge(aneCacheDelta(c0)) { $1 }
                        attempts.append(at)
                    }
                    d["attempts"] = attempts
                    d.merge(attempts.last ?? [:]) { $1 }
                    guard let dg else { throw GateError.assets("the Neural Engine decision graph did not load (\(config.aneSources))") }
                    var scores: [Fixtures.RowScore] = []
                    var rowsOut: [[String: Any]] = []
                    var maxDrift = 0.0, maxDpGPU = 0.0, nDpGPU = 0
                    for rec in fx.records where rec.kind == "text" {
                        for (want, got) in try matchRows(gpu, rec, .text, prefix: 0) where want.bucket == 256 {
                            let x = try GraphInputs(row: got, length: 256)
                            let t = ContinuousClock.now
                            let sc = try await dg.scores(x)
                            let callMs = seconds(since: t) * 1e3
                            let sc2 = try await dg.scores(x)
                            let z = x.markers.map { sc[$0] }, z2 = x.markers.map { sc2[$0] }
                            let drift = zip(z, z2).map { abs(Double($0) - Double($1)) }.max() ?? 0
                            maxDrift = max(maxDrift, drift)
                            let p = Readout.probabilities(logits: z, question: got.question, calibrate: got.calibrate, config: gpu.config)
                            let s = fx.score(want, idsEqual: got.ids == want.ids, markersEqual: got.markers == want.markers,
                                             bucketEqual: true, logits: z, probs: p)
                            scores.append(s)
                            var rj = rowJSON(want, s, logits: z, probs: p, device: got, seconds: nil)
                            rj["call_ms"] = callMs
                            rj["drift_marker_logits"] = drift
                            if let q = rowProbs[gpuKind]?[want.key] {
                                let dpg = zip(p, q).map { abs(Double($0) - Double($1)) }.max() ?? .nan
                                rj["max_abs_dp_gpu"] = dpg
                                maxDpGPU = max(maxDpGPU, dpg.isNaN ? 0 : dpg)
                                nDpGPU += 1
                            }
                            rowsOut.append(rj)
                        }
                    }
                    let sum = Fixtures.summarize(scores)
                    d["rows"] = rowsOut
                    d["summary"] = sum
                    d["max_drift_marker_logits"] = maxDrift
                    d["rows_max_abs_dp_gpu"] = nDpGPU > 0 ? maxDpGPU as Any : NSNull()
                    d["rows_compared_gpu"] = nDpGPU
                    line("ane decision L256 (\(scores.count) text rows, 2 calls each): " + sum.line + " | drift \(f6(maxDrift)), "
                         + "vs GPU rows \(gpuRowsDiff(maxDpGPU, nDpGPU)) (expected FAIL: the Mac's ANE decision graph failed the bar)")
                    writePartial("ane", j.merging(["decision": d, "step": "decision ms"]) { $1 })
                    var wl: [String: Any] = [:]
                    for wid in ["W1", "W2"] {
                        guard let w = fx.bench.workloads.first(where: { $0.id == wid }) else { continue }
                        if let left = remaining(), left < maxS + 30 { wl[wid] = ["skipped": true, "reason": "deadline"]; continue }
                        let p = try await prepare(w, gpu)
                        let gpuForm = try await benchForm("gpu-\(gpuKind)", gpu, bucket: p.bucket, mode: .text, audioBucket: nil)
                        let aneForm = BenchForm(name: "ane-decision", bucket: 256, d1: gpu, decision: dg, vision: nil, audio: nil)
                        _ = await waitForNominal("ane \(wid)")
                        var s = await series("ane \(wid)", p, forms: [aneForm, gpuForm], timed: timed, warmFirst: warmFirst,
                                             warmLater: warmLater, maxSeriesS: maxS, restS: rest)
                        if var oc = s["output_check"] as? [String: Any] {
                            oc["forms_bit_equal_every_round_expected"] = false
                            s["output_check"] = oc
                        }
                        wl[wid] = s
                        let fo = s["forms"] as? [String: Any] ?? [:]
                        let ane = fo["ane-decision"] as? [String: Any] ?? [:], g = fo["gpu-\(gpuKind)"] as? [String: Any] ?? [:]
                        line("ane \(wid): ANE decision median \(f2(ane["median_ms"] as? Double ?? .nan)) ms vs GPU \(gpuKind) "
                             + "\(f2(g["median_ms"] as? Double ?? .nan)) ms")
                    }
                    d["bench"] = wl
                    d["pass"] = true   // measured: the bar's verdict is recorded, not required (the Mac's ANE form failed it)
                } catch {
                    line("ane decision: ERROR \(error)")
                    d["error"] = "\(error)"
                    d["error_detail"] = Self.errorRecord(error)
                    d["pass"] = false
                }
                ok = ok && (d["pass"] as? Bool ?? false)
                j["decision"] = d
            }
        }
        j["pass"] = ok
        return j
    }

    // MARK: - records

    /// An error as result.json keeps it: the text, the Swift type and case, and the NSError bridge.
    static func errorRecord(_ error: Error) -> [String: Any] {
        let ns = error as NSError
        return ["description": "\(error)", "reflecting": String(reflecting: error), "type": String(reflecting: type(of: error)),
                "ns_domain": ns.domain, "ns_code": ns.code,
                "ns_user_info": Dictionary(uniqueKeysWithValues: ns.userInfo.map { ($0.key, "\($0.value)") })]
    }

    /// The stage's record so far into result.json (a stage that is killed still leaves this much). Inside a stage made of
    /// parts ("long"), a part's record goes under the stage's key, so a launch that dies there names the stage.
    func writePartial(_ key: String, _ j: [String: Any]) {
        var partial = j
        partial["partial"] = true
        var k = key
        if let parent = partialParent, parent != key {
            var whole = (stageResults[parent] as? [String: Any]) ?? partialParentRecord
            whole[key] = j
            whole["running"] = key
            whole["partial"] = true
            partial = whole
            k = parent
        }
        let order = stageOrder
        let previous = stageResults[k]
        stageResults[k] = partial
        stageOrder.append(k)
        writeReport()
        stageOrder = order
        stageResults[k] = previous
    }

    func finish(ok: Bool, fatal: String?) -> Bool {
        report["status"] = fatal == nil ? "done" : "failed"
        if let f = fatal { report["fatal"] = f }
        report["pass"] = ok
        report["finished"] = Self.now()
        report["elapsed_s"] = elapsed()
        report["device_end"] = ["thermal": DeviceInfo.thermal(), "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                                "footprint_mb": DeviceInfo.footprintMB(), "available_mb": DeviceInfo.availableMB(),
                                "free_gb": DeviceInfo.freeGB(config.out), "battery_level": BatteryCache.shared.get().level,
                                "battery_state": BatteryCache.shared.get().state, "storage": DeviceInfo.storageSnapshot()]
        let verdicts = stageOrder.map { k -> String in
            let s = stageResults[k] as? [String: Any] ?? [:]
            return "\(k)=\((s["skipped"] as? Bool) == true ? "SKIPPED" : ((s["pass"] as? Bool) == true ? "PASS" : "FAIL"))"
        }
        report["summary"] = verdicts
        line("GATE_SUMMARY \(verdicts.joined(separator: " ")) VERDICT=\(ok ? "PASS" : "FAIL")" + (fatal.map { " (\($0))" } ?? ""))
        writeReport()
        let runs = config.out.appendingPathComponent("runs")
        try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let safe = config.runID.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
        try? FileManager.default.copyItem(at: config.out.appendingPathComponent("result.json"),
                                          to: runs.appendingPathComponent("\(safe).json"))
        line("DONE \(config.runID)")
        monitor?.stop()
        sink.close()
        setStage(ok ? "done: PASS" : "done: FAIL")
        return ok
    }

    func writeReport() {
        var r = report
        r["stages"] = stageResults
        r["stage_order"] = stageOrder
        r["updated"] = Self.now()
        r["elapsed_s"] = elapsed()
        r["memory_tsv_lines"] = monitor?.lines ?? 0
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

enum D1Clock {
    static func seconds(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
        let d = b - a
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}

extension Dictionary where Key == String, Value == Any {
    /// One line of a Fixtures.summarize record.
    var line: String {
        let m = self["mac"] as? [String: Any] ?? [:]
        func d(_ k: String) -> Double { self[k] as? Double ?? .nan }
        return "\(self["rows"] ?? 0) rows | argmax non-near-tie \(self["argmax_equal_non_near_tie"] ?? 0)/\(self["non_near_tie"] ?? 0), "
            + "near-tie \(self["argmax_equal_near_tie"] ?? 0)/\(self["near_tie"] ?? 0), ids \(self["ids_markers_bucket_equal"] ?? 0), "
            + "max|dp| \(f6(d("max_abs_dp"))) (\(self["max_abs_dp_row"] ?? "")), mean \(f6(d("mean_row_max_abs_dp"))), bar "
            + "\((self["bar_pass"] as? Bool) == true ? "PASS" : "FAIL") | vs Mac: logits bits \(m["logits_bit_equal"] ?? 0), p bits "
            + "\(m["probs_bit_equal"] ?? 0)/\(m["rows"] ?? 0), max|dp| \(f6(m["max_abs_dp"] as? Double ?? .nan)), max|dlogit| "
            + "\(f6(m["max_abs_dlogit"] as? Double ?? .nan))"
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
