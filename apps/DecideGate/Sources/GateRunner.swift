// GateRunner — the iPhone half of the GLiNER2.5-Decide gate. The graph host (DecideGraph), the fixtures and the
// metrics (DecideGateSupport) are the ones decide-selftest runs on the Mac (conversion/gliner25_decide/swift),
// here driving sideloaded bundles: the AOT bundles of the phone's architecture, or (DECIDE_BUNDLE_KIND) the JIT
// bundle the phone specializes itself, or the JIT bundle with another architecture's AOT files beside it. Added for
// the device: load times (first and second load in this process), the first call, the app footprint, a bench, the
// thermal state around every step, and the device / OS identity; around each load and the first call, the memory
// every 100 ms (peak footprint, least os_proc_available_memory) and the container's Core AI cache
// (Library/Caches/coreai-cache) before and after. Results: Documents/decide_gate/result.json (rewritten after every
// stage, "status" running -> done) and result.log (one line per event).
//
// Assets: Library/Application Support/DecideAssets/ (../_stage.sh lays it out, ../_install.sh pushes it):
//   gliner25-decide_float16_s256_m32.<arch>.aimodelc/  AOT, preferred gpu        -> stage <unit>_s256 (kind aot)
//   gliner25-decide_float16_s512_m32.<arch>.aimodelc/  AOT, preferred gpu        -> stage <unit>_s512 (kind aot)
//     <arch> = this phone's Core AI architecture (AIModel.deviceArchitectureName): h19p on the iPhone 18 Pro
//     (iPhone19,2), h18p on the iPhone 17 Pro; a bundle compiled for another one does not load
//     (incompatibleCompiledAssetArchitecture)
//   gliner25-decide_float16_s<S>_m32.aimodel/          JIT (main.mlirb), as macos/ ships it   (kind jit)
//   gliner25-decide_float16_s<S>_m32.mixed.aimodel/    the JIT files (main.mlirb, main.hash, metadata.json) with
//                                                      another architecture's AOT files beside them
//                                                      (main-h18p.mlirb, main-h18p-delegates/, stats.json): the
//                                                      shape of GLiNER2-PII-CoreAI's ios/ bundle   (kind mixed)
//   fixtures/{readme21,fast_decisions_s256,fast_decisions_long}.json   gliner2 2.0.0 fp32 oracle cases
//   golden/pygpu_s256.json  golden/pygpu_s512.json    case id -> the Mac GPU logits of the same fp16 bundles
//   MD5SUMS
//
// Launch environment (devicectl device process launch --environment-variables), all optional:
//   DECIDE_RUN_ID    echoed into result.json (../_run.sh waits for its own id)
//   DECIDE_STAGES    s256,s512 (default both)
//   DECIDE_BENCH     graph calls timed after 5 warm-up calls (default 100; 0 = no bench)
//   DECIDE_UNIT      gpu | default | cpuOnly (default gpu)
//   DECIDE_ASSETS    assets directory relative to the app's home (default Library/Application Support/DecideAssets)
//   DECIDE_WAIT_NOMINAL  seconds: before each stage's bench, poll the thermal state every 5 s until it is nominal,
//                        at most this long (default 0 = no wait); the wait and the states go into result.json
//   DECIDE_BENCH_FIRST   1 = the bench right after load 2, before the case loop (default: after the case loop)
//   DECIDE_BUNDLE_KIND   aot | jit | mixed (default aot): which of the bundles above each stage loads

import DecideGateSupport
import DecideGraph
import Foundation

struct GateConfig: Sendable {
    let runID: String
    let assets: URL
    let out: URL
    let stages: [String]
    let benchCalls: Int
    let unit: DecideComputeUnits
    let waitNominalSeconds: Double
    let benchFirst: Bool
    let bundleKindName: String
    /// nil when DECIDE_BUNDLE_KIND names no kind (the run stops with a fatal line)
    let bundleKind: BundleKind?

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        let assets = env["DECIDE_ASSETS"].map { home.appendingPathComponent($0) }
            ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("DecideAssets")
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let stamp = ISO8601DateFormatter().string(from: Date())
        let stages = (env["DECIDE_STAGES"] ?? "s256,s512").split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        let kind = env["DECIDE_BUNDLE_KIND"] ?? BundleKind.aot.rawValue
        return GateConfig(
            runID: env["DECIDE_RUN_ID"] ?? stamp,
            assets: assets,
            out: docs.appendingPathComponent("decide_gate"),
            stages: stages,
            benchCalls: Int(env["DECIDE_BENCH"] ?? "") ?? 100,
            unit: DecideComputeUnits(rawValue: env["DECIDE_UNIT"] ?? "gpu") ?? .gpu,
            waitNominalSeconds: max(0, Double(env["DECIDE_WAIT_NOMINAL"] ?? "") ?? 0),
            benchFirst: env["DECIDE_BENCH_FIRST"] == "1",
            bundleKindName: kind,
            bundleKind: BundleKind(rawValue: kind))
    }
}

/// Which bundle a stage loads (DECIDE_BUNDLE_KIND).
enum BundleKind: String, Sendable, CaseIterable {
    /// <name>.<arch>.aimodelc: compiled ahead of time for this phone's architecture (the shipped iOS form)
    case aot
    /// <name>.aimodel: main.mlirb only; the phone specializes it at the first load and caches the result
    case jit
    /// <name>.mixed.aimodel: the JIT files plus another architecture's AOT files (GLiNER2-PII-CoreAI's ios/ shape)
    case mixed
}

/// One bundle under test and the fixture sets it runs.
struct StageSpec: Sendable {
    let key: String                              // <unit>_<tag>, e.g. gpu_s256
    let tag: String
    let bundle: URL
    let sets: [String]
    let golden: URL

    /// The fixture sets per shape: every case must fit S (the S=512 bundle also takes the S=256 sets).
    static let setsByTag = ["s256": ["readme21", "fast_decisions_s256"],
                            "s512": ["readme21", "fast_decisions_s256", "fast_decisions_long"]]
    static let allSets = ["readme21", "fast_decisions_s256", "fast_decisions_long"]

    static func bundleName(_ tag: String, kind: BundleKind) -> String {
        let stem = "gliner25-decide_float16_\(tag)_m32"
        switch kind {
        case .jit: return "\(stem).aimodel"
        case .mixed: return "\(stem).mixed.aimodel"
        case .aot:
            #if os(iOS)
            return "\(stem).\(DecideGraph.deviceArchitecture).aimodelc"
            #else
            return "\(stem).aimodel"                        // a Mac dry run of this file (the app is iOS-only)
            #endif
        }
    }
}

actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0: ContinuousClock.Instant
    private let sink: LogSink
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
        let t0 = ContinuousClock.now
        self.t0 = t0
        sink = LogSink(t0: t0, emit: emit)
    }

    // MARK: - the run

    /// Every stage in order; true when every stage passed.
    func run() async -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: config.out, withIntermediateDirectories: true)
        let resultURL = config.out.appendingPathComponent("result.json")
        let logURL = config.out.appendingPathComponent("result.log")
        try? fm.removeItem(at: resultURL)
        fm.createFile(atPath: logURL.path, contents: nil)
        sink.open(logURL)

        let launchIndex = bumpLaunchCount()
        var device = DeviceInfo.snapshot()
        device["coreai_architecture"] = DecideGraph.deviceArchitecture
        let battery = await DeviceInfo.battery()
        device["battery_level"] = battery.level
        device["battery_state"] = battery.state
        report = ["app": "DecideGate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device, "model": DecideModel.id, "model_rev": DecideModel.revision,
                  "gate": "decisions 100 % equal to the gliner2 2.0.0 fp32 oracle on every task, every row finite",
                  "config": ["assets": config.assets.path, "stages": config.stages, "bench_calls": config.benchCalls,
                             "unit": config.unit.rawValue, "wait_nominal_s": config.waitNominalSeconds,
                             "bench_first": config.benchFirst, "bundle_kind": config.bundleKindName]]
        line("DecideGate run \(config.runID) (launch \(launchIndex) in this container)")
        line("device \(device["machine"] ?? "?"), \(device["os"] ?? "?") (build \(device["os_build"] ?? "?")), Core AI arch "
             + "\(DecideGraph.deviceArchitecture), thermal \(DeviceInfo.thermal()), low power \(device["low_power_mode"] ?? "?"), "
             + "battery \(String(format: "%.0f", battery.level * 100)) % \(battery.state)")
        writeReport()
        guard let kind = config.bundleKind else {
            line("FATAL DECIDE_BUNDLE_KIND \(config.bundleKindName): not one of "
                 + BundleKind.allCases.map(\.rawValue).joined(separator: ", "))
            return finish(ok: false, fatal: "unknown DECIDE_BUNDLE_KIND \(config.bundleKindName)")
        }

        // assets: the MD5SUMS listing (bundles it does not list are left over from an earlier stage: removed), then
        // every fixture set
        setStage("assets")
        var assets = checkAssets()
        assets["removed_unlisted_bundles"] = removeUnlistedBundles()
        assets["free_gb_after"] = DeviceInfo.freeGB(config.assets)
        report["assets"] = assets
        var fixtures: [String: [DecideCase]] = [:]
        var fixtureFiles: [[String: Any]] = []
        do {
            for set in StageSpec.allSets {
                let (cases, files) = try loadFixtures([config.assets.appendingPathComponent("fixtures/\(set).json")])
                fixtures[set] = cases
                fixtureFiles.append(contentsOf: files)
                line("fixture \(set): \(cases.count) cases, \(cases.reduce(0) { $0 + $1.taskResults.count }) tasks, "
                     + "longest \(cases.map(\.seqLen).max() ?? 0) tokens")
            }
        } catch {
            line("FATAL fixtures: \(error)")
            return finish(ok: false, fatal: "fixtures: \(error)")
        }
        report["fixtures"] = fixtureFiles

        var allOK = true
        for spec in stageSpecs(kind) {
            setStage(spec.key)
            let result = await runStage(spec, fixtures: fixtures)
            stageResults[spec.key] = result
            stageOrder.append(spec.key)
            allOK = allOK && (result["pass"] as? Bool ?? false)
            writeReport()
        }
        if stageOrder.isEmpty {
            line("no stage to run (DECIDE_STAGES takes s256, s512)")
            allOK = false
        }
        return finish(ok: allOK, fatal: nil)
    }

    func stageSpecs(_ kind: BundleKind) -> [StageSpec] {
        config.stages.compactMap { tag in
            guard let sets = StageSpec.setsByTag[tag] else {
                line("unknown stage \(tag) (DECIDE_STAGES takes s256, s512)")
                return nil
            }
            return StageSpec(key: "\(config.unit.rawValue)_\(tag)", tag: tag,
                             bundle: config.assets.appendingPathComponent(StageSpec.bundleName(tag, kind: kind)), sets: sets,
                             golden: config.assets.appendingPathComponent("golden/pygpu_\(tag).json"))
        }
    }

    // MARK: - one stage: load twice, every fixture case vs the oracle and the Mac GPU, bench

    func runStage(_ spec: StageSpec, fixtures: [String: [DecideCase]]) async -> [String: Any] {
        var j: [String: Any] = ["bundle": spec.bundle.lastPathComponent, "bundle_kind": config.bundleKindName,
                                "unit": config.unit.rawValue, "fixtures": spec.sets]
        var thermal: [[String: Any]] = []
        func mark(_ at: String) { thermal.append(["at": at, "state": DeviceInfo.thermal(), "t_s": elapsed()]) }
        mark("start")
        let batteryStart = await DeviceInfo.battery()
        j["battery_start"] = ["level": batteryStart.level, "state": batteryStart.state]
        guard FileManager.default.fileExists(atPath: spec.bundle.path) else {
            line("\(spec.key): FAIL, bundle not staged: \(spec.bundle.lastPathComponent)")
            j["error"] = "bundle not staged: \(spec.bundle.lastPathComponent)"
            j["pass"] = false
            return j
        }
        let (bytes, files) = DeviceInfo.tree(spec.bundle)
        j["bundle_bytes"] = bytes
        j["bundle_mb"] = Double(bytes) / 1e6
        j["bundle_files"] = files
        let cases = spec.sets.flatMap { fixtures[$0] ?? [] }
        line("\(spec.key): \(spec.bundle.lastPathComponent) (\(String(format: "%.1f", Double(bytes) / 1e6)) MB, \(files) files), "
             + "unit \(config.unit.rawValue), \(cases.count) cases from \(spec.sets.joined(separator: " + "))")

        // the container's caches at each step: "before_load_1", "after_load_1", "after_first_call", "after_load_2", "end"
        var storage: [String: Any] = [:]
        func snapshot(_ at: String) -> (bytes: Int, files: Int) {
            let s = DeviceInfo.storageSnapshot()
            storage[at] = s
            j["storage"] = storage
            let c = s["coreai_cache"] as? [String: Any] ?? [:]
            return (c["bytes"] as? Int ?? 0, c["files"] as? Int ?? 0)
        }
        func mb(_ bytes: Int) -> String { String(format: "%.1f", Double(bytes) / 1e6) }

        var pass = false
        do {
            guard !cases.isEmpty else { throw GateError.fixture("no cases for \(spec.key)") }
            let cache0 = snapshot("before_load_1")
            let footprint0 = DeviceInfo.footprintMB(), available0 = DeviceInfo.availableMB()
            j["cache_bytes_before_load"] = cache0.bytes
            j["cache_files_before_load"] = cache0.files
            j["footprint_mb_before_load"] = footprint0
            j["available_mb_before_load"] = available0
            line("\(spec.key): load 1 (\(config.bundleKindName) bundle) starting; coreai-cache \(mb(cache0.bytes)) MB in "
                 + "\(cache0.files) files; footprint \(String(format: "%.0f", footprint0)) MB, available "
                 + "\(String(format: "%.0f", available0)) MB")
            writePartial(spec.key, j, thermal)                   // a load that is killed still leaves this much
            // load 1 = the first load of this bundle in this process (caches from earlier launches may remain)
            let (loaded1, memory1, wall1) = await sampled("\(spec.key) load 1") {
                try await DecideGraph(contentsOf: spec.bundle, computeUnits: self.config.unit)
            }
            j["load_first_memory"] = memory1
            j["load_first_peak_footprint_mb"] = memory1["peak_footprint_mb"]
            j["load_first_min_available_mb"] = memory1["min_available_mb"]
            j["load_first_wall_s"] = wall1
            j["available_mb_after_load"] = DeviceInfo.availableMB()
            mark("after load 1")
            let cache1 = snapshot("after_load_1")
            j["cache_bytes_after_load"] = cache1.bytes
            j["cache_files_after_load"] = cache1.files
            line("\(spec.key): load 1 \(loaded1.isSuccess ? "done" : "FAILED") after \(String(format: "%.2f", wall1)) s; peak "
                 + "footprint \(String(format: "%.0f", memory1["peak_footprint_mb"] as? Double ?? -1)) MB, least available "
                 + "\(String(format: "%.0f", memory1["min_available_mb"] as? Double ?? -1)) MB; coreai-cache "
                 + "\(mb(cache1.bytes)) MB in \(cache1.files) files")
            var first: DecideGraph? = try loaded1.get()
            let load1 = first!.loadSeconds
            let g0 = try graphInputs(cases[0], S: first!.seqLength, MMAX: first!.maxLabels)
            let (called, memoryCall, callWall) = await sampled("\(spec.key) first call") { [first] in
                try await first!.run(inputIds: g0.inputIds, mask: g0.attentionMask, labelIdx: g0.labelIdx)
            }
            _ = try called.get()
            let firstCall = callWall * 1e3
            j["first_call_memory"] = memoryCall
            j["first_call_peak_footprint_mb"] = memoryCall["peak_footprint_mb"]
            mark("after first call")
            let cacheCall = snapshot("after_first_call")
            j["cache_bytes_after_first_call"] = cacheCall.bytes
            first = nil                                            // load 2 is a new model, the first one gone
            let (loaded2, memory2, _) = await sampled("\(spec.key) load 2") {
                try await DecideGraph(contentsOf: spec.bundle, computeUnits: self.config.unit)
            }
            j["load_second_memory"] = memory2
            j["load_second_peak_footprint_mb"] = memory2["peak_footprint_mb"]
            let graph = try loaded2.get()
            let load2 = graph.loadSeconds
            mark("after load 2")
            let cache2 = snapshot("after_load_2")
            j["cache_bytes_after_load_2"] = cache2.bytes
            let footprint = DeviceInfo.footprintMB()
            j["load_first_s"] = load1
            j["first_call_ms"] = firstCall
            j["first_call_case"] = cases[0].id
            j["load_second_s"] = load2
            j["footprint_mb_after_load"] = footprint
            j["S"] = graph.seqLength
            j["MMAX"] = graph.maxLabels
            j["contract"] = graph.contractDescription
            line("\(spec.key): load \(String(format: "%.2f", load1)) s (first in this process), first call "
                 + "\(String(format: "%.1f", firstCall)) ms (\(cases[0].id)), load \(String(format: "%.2f", load2)) s (second); "
                 + "footprint \(String(format: "%.0f", footprint)) MB; \(graph.contractDescription)")
            if config.benchFirst && config.benchCalls > 0 {
                mark("before bench")
                j["bench"] = try await bench(spec.key, graph, cases, position: "after load 2, before the cases")
                mark("after bench")
            }

            var mac: [String: [Double]] = [:]
            do {
                mac = try loadMacGPULogits(spec.golden)
            } catch {
                line("\(spec.key): no Mac GPU logits (\(error)); the vs-Mac comparison is skipped")
                j["golden_error"] = "\(error)"
            }
            var rows: [CaseRecord] = []
            for (i, c) in cases.enumerated() {
                var r = try await runCase(graph, c, index: i)
                if let m = mac[c.id] { r.vsMacGPU = logitsDiff(r, c, other: m) }
                rows.append(r)
                if !r.decisionsEqual || !r.finite || i < 3 || i % 50 == 0 {
                    line("\(spec.key) " + caseLine(i, r))
                    diffLines(r).forEach { line($0) }
                }
            }
            mark("after cases")
            let s = summarize(rows)
            let v = summarizeMacGPU(rows)
            pass = gatePasses(rows)
            j["summary"] = s
            j["vs_mac_gpu"] = v
            j["cases"] = rows.map(\.json)
            line("\(spec.key): \(s["decisions_equal_tasks"] ?? 0)/\(s["tasks"] ?? 0) decisions equal "
                 + "(\(s["cases_with_changed_decision"] ?? 0) cases changed), max|Δlogit| \(fmt(s["max_abs_dlogit"] as? Double ?? .nan)), "
                 + "max|Δprob| \(fmt(s["max_abs_dprob"] as? Double ?? .nan)), min cos \(String(format: "%.7f", s["min_cos"] as? Double ?? .nan)), "
                 + "non-finite \(s["nonfinite_cases"] ?? 0); call \(String(format: "%.2f", s["warm_call_ms_median"] as? Double ?? .nan)) ms "
                 + "median over the cases")
            line("\(spec.key) vs Mac GPU logits (same bundle): \(v["cases"] ?? 0) cases (\(v["cases_without_mac_row"] ?? 0) without a row), "
                 + "max|Δlogit| \(fmt(v["max_abs_dlogit"] as? Double ?? .nan)), bit-equal \(v["bit_equal"] ?? 0)/\(v["elements"] ?? 0), "
                 + "decisions equal \(v["decisions_equal_tasks"] ?? 0)/\(v["tasks"] ?? 0)")
            writePartial(spec.key, j, thermal)

            if !config.benchFirst && config.benchCalls > 0 {
                mark("before bench")
                j["bench"] = try await bench(spec.key, graph, cases, position: "after the cases")
                mark("after bench")
            }
            j["footprint_mb_end"] = DeviceInfo.footprintMB()
        } catch {
            line("\(spec.key): ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.describe(error)
            pass = false
        }
        mark("end")
        j["cache_bytes_end"] = snapshot("end").bytes
        let batteryEnd = await DeviceInfo.battery()
        j["battery_end"] = ["level": batteryEnd.level, "state": batteryEnd.state]
        j["thermal"] = thermal
        j["pass"] = pass
        line("\(spec.key): \(pass ? "PASS" : "FAIL") (decisions 100 % equal to the oracle, every row finite)")
        return j
    }

    /// DECIDE_WAIT_NOMINAL (when set), then warm-up 5 + DECIDE_BENCH calls on fixture inputs.
    func bench(_ key: String, _ graph: DecideGraph, _ cases: [DecideCase], position: String) async throws -> [String: Any] {
        let wait = config.waitNominalSeconds > 0 ? await waitForNominal(key) : nil
        let thermalStart = DeviceInfo.thermal()
        let ts = try await benchCalls(graph, cases, calls: config.benchCalls)
        var b = benchJSON(ts)
        b["thermal_start"] = thermalStart
        b["thermal_end"] = DeviceInfo.thermal()
        b["position"] = position
        if let w = wait { b["wait_nominal"] = w }
        line("\(key) bench (\(position)): \(String(format: "%.2f", b["ms_median"] as? Double ?? .nan)) ms median, p90 "
             + "\(String(format: "%.2f", b["ms_p90"] as? Double ?? .nan)), min \(String(format: "%.2f", b["ms_min"] as? Double ?? .nan)), "
             + "max \(String(format: "%.2f", b["ms_max"] as? Double ?? .nan)) (\(ts.count) calls after 5 warm-up) | thermal "
             + "\(thermalStart) -> \(DeviceInfo.thermal())")
        return b
    }

    /// Polls ProcessInfo.thermalState every 5 s until it is nominal, at most DECIDE_WAIT_NOMINAL seconds.
    func waitForNominal(_ key: String) async -> [String: Any] {
        let cap = config.waitNominalSeconds
        let before = DeviceInfo.thermal()
        let t0 = ContinuousClock.now
        var polls = 0
        if before != "nominal" { line("\(key): thermal \(before) before the bench; waiting for nominal (5 s steps, cap \(Int(cap)) s)") }
        while DeviceInfo.thermal() != "nominal" && seconds(since: t0) < cap {
            try? await Task.sleep(for: .seconds(5))
            polls += 1
        }
        let waited = seconds(since: t0)
        let after = DeviceInfo.thermal()
        if before != "nominal" {
            line("\(key): thermal \(after) after \(String(format: "%.1f", waited)) s"
                 + (after == "nominal" ? "" : " (cap reached: the bench runs anyway)"))
        }
        return ["cap_s": cap, "state_before": before, "state_after": after, "waited_s": waited, "polls": polls,
                "reached_nominal": after == "nominal"]
    }

    /// `body` under a MemorySampler (100 ms): its result or error, the sampler's record, the wall seconds.
    func sampled<T>(_ what: String, _ body: () async throws -> T) async -> (Result<T, Error>, [String: Any], Double) {
        let sampler = MemorySampler(what, progress: { [sink] in sink.line($0) })
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

    // MARK: - helpers

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

    /// Deletes gliner25-decide_*.aimodelc / .aimodel directories in the assets directory that MD5SUMS does not list (a
    /// push adds to the container and never removes, so a restage for another architecture leaves the old bundles).
    func removeUnlistedBundles() -> [String] {
        let fm = FileManager.default
        guard let text = try? String(contentsOf: config.assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) else { return [] }
        let listed = Set(text.split(separator: "\n").compactMap { row -> String? in
            let parts = row.split(separator: " ", maxSplits: 1)
            return parts.count == 2 ? String(parts[1].split(separator: "/").first ?? "") : nil
        })
        guard !listed.isEmpty, let names = try? fm.contentsOfDirectory(atPath: config.assets.path) else { return [] }
        var removed: [String] = []
        for name in names.sorted() where name.hasPrefix("gliner25-decide_") && (name.hasSuffix(".aimodelc") || name.hasSuffix(".aimodel"))
            && !listed.contains(name) {
            let (bytes, _) = DeviceInfo.tree(config.assets.appendingPathComponent(name))
            if (try? fm.removeItem(at: config.assets.appendingPathComponent(name))) != nil {
                removed.append(name)
                line("removed \(name) (\(String(format: "%.1f", Double(bytes) / 1e6)) MB): not in MD5SUMS")
            }
        }
        return removed
    }

    /// The stage's record so far into result.json (a crash in the bench still leaves the case results).
    func writePartial(_ key: String, _ j: [String: Any], _ thermal: [[String: Any]]) {
        var partial = j
        partial["thermal"] = thermal
        partial["partial"] = true
        let order = stageOrder
        stageResults[key] = partial
        stageOrder.append(key)
        writeReport()
        stageOrder = order
        stageResults[key] = nil
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
            return "\(k)=\((s["pass"] as? Bool) == true ? "PASS" : "FAIL")"
        }
        report["summary"] = verdicts
        line("GATE_SUMMARY \(verdicts.joined(separator: " ")) VERDICT=\(ok ? "PASS" : "FAIL")" + (fatal.map { " (\($0))" } ?? ""))
        writeReport()
        // keep a copy per run next to the latest one
        let runs = config.out.appendingPathComponent("runs")
        try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let safe = config.runID.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
        try? FileManager.default.copyItem(at: config.out.appendingPathComponent("result.json"),
                                          to: runs.appendingPathComponent("\(safe).json"))
        line("DONE \(config.runID)")
        sink.close()
        setStage(ok ? "done: PASS" : "done: FAIL")
        return ok
    }

    func writeReport() {
        var r = report
        r["stages"] = stageResults
        r["stage_order"] = stageOrder
        r["updated"] = Self.now()
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

/// result.log, stdout and the screen, one line per event, from the runner or from a sampler's thread.
final class LogSink: @unchecked Sendable {
    private let lock = NSLock()
    private let t0: ContinuousClock.Instant
    private let emit: @Sendable (String) -> Void
    private var handle: FileHandle?

    init(t0: ContinuousClock.Instant, emit: @escaping @Sendable (String) -> Void) {
        self.t0 = t0
        self.emit = emit
    }

    func open(_ url: URL) { lock.withLock { handle = try? FileHandle(forWritingTo: url) } }

    func close() {
        lock.withLock {
            try? handle?.close()
            handle = nil
        }
    }

    func line(_ s: String) {
        let text = "[decide] \(String(format: "%8.2f", seconds(since: t0))) \(s)"
        print(text)
        lock.withLock {
            if let d = (text + "\n").data(using: .utf8) {
                try? handle?.write(contentsOf: d)
                try? handle?.synchronize()
            }
        }
        emit(text)
    }
}

extension Result {
    var isSuccess: Bool {
        if case .success = self { return true }
        return false
    }
}
