// GateRunner — the iPhone half of the GLiNER2.5-Decide gate. The graph host (DecideGraph), the fixtures and the
// metrics (DecideGateSupport) are the ones decide-selftest runs on the Mac (conversion/gliner25_decide/swift),
// here driving the sideloaded AOT h18p bundles. Added for the device: load times (first and second load in this
// process), the first call, the app footprint, a bench, the thermal state around every step, and the device / OS
// identity. Results: Documents/decide_gate/result.json (rewritten after every stage, "status" running -> done)
// and result.log (one line per event).
//
// Assets: Library/Application Support/DecideAssets/ (../_stage.sh lays it out, ../_install.sh pushes it):
//   gliner25-decide_float16_s256_m32.<arch>.aimodelc/  AOT, preferred gpu        -> stage <unit>_s256
//   gliner25-decide_float16_s512_m32.<arch>.aimodelc/  AOT, preferred gpu        -> stage <unit>_s512
//     <arch> = this phone's Core AI architecture (AIModel.deviceArchitectureName): h19p on the iPhone 18 Pro
//     (iPhone19,2), h18p on the iPhone 17 Pro; a bundle compiled for another one does not load
//     (incompatibleCompiledAssetArchitecture)
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
        return GateConfig(
            runID: env["DECIDE_RUN_ID"] ?? stamp,
            assets: assets,
            out: docs.appendingPathComponent("decide_gate"),
            stages: stages,
            benchCalls: Int(env["DECIDE_BENCH"] ?? "") ?? 100,
            unit: DecideComputeUnits(rawValue: env["DECIDE_UNIT"] ?? "gpu") ?? .gpu)
    }
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

    static func bundleName(_ tag: String) -> String {
        #if os(iOS)
        "gliner25-decide_float16_\(tag)_m32.\(DecideGraph.deviceArchitecture).aimodelc"
        #else
        "gliner25-decide_float16_\(tag)_m32.aimodel"       // a Mac dry run of this file (the app is iOS-only)
        #endif
    }
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

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
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
        log = try? FileHandle(forWritingTo: logURL)

        let launchIndex = bumpLaunchCount()
        var device = DeviceInfo.snapshot()
        device["coreai_architecture"] = DecideGraph.deviceArchitecture
        report = ["app": "DecideGate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device, "model": DecideModel.id, "model_rev": DecideModel.revision,
                  "gate": "decisions 100 % equal to the gliner2 2.0.0 fp32 oracle on every task, every row finite",
                  "config": ["assets": config.assets.path, "stages": config.stages, "bench_calls": config.benchCalls,
                             "unit": config.unit.rawValue]]
        line("DecideGate run \(config.runID) (launch \(launchIndex) in this container)")
        line("device \(device["machine"] ?? "?"), \(device["os"] ?? "?") (build \(device["os_build"] ?? "?")), Core AI arch "
             + "\(DecideGraph.deviceArchitecture), thermal \(DeviceInfo.thermal()), low power \(device["low_power_mode"] ?? "?")")
        writeReport()

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
        for spec in stageSpecs() {
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

    func stageSpecs() -> [StageSpec] {
        config.stages.compactMap { tag in
            guard let sets = StageSpec.setsByTag[tag] else {
                line("unknown stage \(tag) (DECIDE_STAGES takes s256, s512)")
                return nil
            }
            return StageSpec(key: "\(config.unit.rawValue)_\(tag)", tag: tag,
                             bundle: config.assets.appendingPathComponent(StageSpec.bundleName(tag)), sets: sets,
                             golden: config.assets.appendingPathComponent("golden/pygpu_\(tag).json"))
        }
    }

    // MARK: - one stage: load twice, every fixture case vs the oracle and the Mac GPU, bench

    func runStage(_ spec: StageSpec, fixtures: [String: [DecideCase]]) async -> [String: Any] {
        var j: [String: Any] = ["bundle": spec.bundle.lastPathComponent, "unit": config.unit.rawValue, "fixtures": spec.sets]
        var thermal: [[String: Any]] = []
        func mark(_ at: String) { thermal.append(["at": at, "state": DeviceInfo.thermal(), "t_s": elapsed()]) }
        mark("start")
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

        var pass = false
        do {
            guard !cases.isEmpty else { throw GateError.fixture("no cases for \(spec.key)") }
            // load 1 = the first load of this bundle in this process (caches from earlier launches may remain)
            var first: DecideGraph? = try await DecideGraph(contentsOf: spec.bundle, computeUnits: config.unit)
            let load1 = first!.loadSeconds
            mark("after load 1")
            let g0 = try graphInputs(cases[0], S: first!.seqLength, MMAX: first!.maxLabels)
            let c0 = ContinuousClock.now
            _ = try await first!.run(inputIds: g0.inputIds, mask: g0.attentionMask, labelIdx: g0.labelIdx)
            let firstCall = seconds(since: c0) * 1e3
            mark("after first call")
            first = nil
            let graph = try await DecideGraph(contentsOf: spec.bundle, computeUnits: config.unit)
            let load2 = graph.loadSeconds
            mark("after load 2")
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

            if config.benchCalls > 0 {
                let thermalStart = DeviceInfo.thermal()
                let ts = try await benchCalls(graph, cases, calls: config.benchCalls)
                var b = benchJSON(ts)
                b["thermal_start"] = thermalStart
                b["thermal_end"] = DeviceInfo.thermal()
                j["bench"] = b
                mark("after bench")
                line("\(spec.key) bench: \(String(format: "%.2f", b["ms_median"] as? Double ?? .nan)) ms median, p90 "
                     + "\(String(format: "%.2f", b["ms_p90"] as? Double ?? .nan)), min \(String(format: "%.2f", b["ms_min"] as? Double ?? .nan)), "
                     + "max \(String(format: "%.2f", b["ms_max"] as? Double ?? .nan)) (\(ts.count) calls after 5 warm-up) | thermal "
                     + "\(thermalStart) -> \(DeviceInfo.thermal())")
            }
            j["footprint_mb_end"] = DeviceInfo.footprintMB()
        } catch {
            line("\(spec.key): ERROR \(error)")
            j["error"] = "\(error)"
            pass = false
        }
        mark("end")
        j["thermal"] = thermal
        j["pass"] = pass
        line("\(spec.key): \(pass ? "PASS" : "FAIL") (decisions 100 % equal to the oracle, every row finite)")
        return j
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

    /// Deletes gliner25-decide_*.aimodelc directories in the assets directory that MD5SUMS does not list (a push
    /// adds to the container and never removes, so a restage for another architecture leaves the old bundles).
    func removeUnlistedBundles() -> [String] {
        let fm = FileManager.default
        guard let text = try? String(contentsOf: config.assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) else { return [] }
        let listed = Set(text.split(separator: "\n").compactMap { row -> String? in
            let parts = row.split(separator: " ", maxSplits: 1)
            return parts.count == 2 ? String(parts[1].split(separator: "/").first ?? "") : nil
        })
        guard !listed.isEmpty, let names = try? fm.contentsOfDirectory(atPath: config.assets.path) else { return [] }
        var removed: [String] = []
        for name in names.sorted() where name.hasPrefix("gliner25-decide_") && name.hasSuffix(".aimodelc") && !listed.contains(name) {
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
        do {
            try writeJSON(r, to: config.out.appendingPathComponent("result.json"))
        } catch {
            line("ERROR writing result.json: \(error)")
        }
    }

    func line(_ s: String) {
        let text = "[decide] \(String(format: "%8.2f", elapsed())) \(s)"
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

    func elapsed() -> Double { seconds(since: t0) }

    static func now() -> String { ISO8601DateFormatter().string(from: Date()) }
}
