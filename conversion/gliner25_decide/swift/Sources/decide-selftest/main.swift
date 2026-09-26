// decide-selftest — headless gate of the Swift host (DecideGraph + DecideGateSupport) on this Mac, against the
// gliner2 2.0.0 fp32 oracle fixtures (../../../gliner25_decide_oracle.py) and the Python engine gate of the same
// bundle (../../../export_gliner25_decide.py, results/gate_s<S>_gpu.json). Parts, combinable in one call:
//
//   --bundle <.aimodel|.aimodelc> --fixtures <json> [<json> ...]
//        every fixture case, one call each: decisions vs the oracle (single label: softmax argmax; multi label:
//        sigmoid >= cls_threshold, none -> argmax), max|Δlogit|, max|Δprob|, as the Python GATE-3 prints them
//   [--unit gpu|cpuOnly|default]        default gpu (for an AOT .aimodelc, default keeps its compiled placement)
//   [--reference <reference_sNNN.json>] graphInputs of that case vs the export's arrays, bit for bit
//   [--pygpu <gate_sNNN_gpu.json>]      the Python Mac GPU logits of the same bundle: max|Δ| per case; and the
//                                       host rule re-run on those logits vs the Python per-task records
//   [--poison]                          label_idx + 1 (negative control: the gate must FAIL)
//   [--bench N]                         warm-up 5 + N calls on fixture inputs
//   [--json <path>]
//
// Exit 0 = every requested gate passed, 3 = a gate failed, 4 = an error. An .h18p. bundle is refused on a Mac.

import Darwin
import DecideGateSupport
import DecideGraph
import Foundation

setvbuf(stdout, nil, _IONBF, 0)

func log(_ s: String) { print("[decide] \(s)") }

// MARK: - --reference: the Swift graphInputs vs the export's arrays

/// reference_s<S>.json (export_gliner25_decide.write_side_files); JSONDecoder keeps every double exact.
struct ReferenceFile: Decodable {
    struct Slice: Decodable { let task: String; let start: Int; let end: Int }
    struct Oracle: Decodable { let task: String; let logits: [Double] }
    let caseID: String
    let S: Int
    let MMAX: Int
    let inputIDs: [Int]
    let attentionMask: [Int]
    let labelIdx: [Int]
    let taskSlices: [Slice]
    let oracleFP32: [Oracle]

    enum CodingKeys: String, CodingKey {
        case S, MMAX
        case caseID = "case_id", inputIDs = "input_ids", attentionMask = "attention_mask", labelIdx = "label_idx"
        case taskSlices = "task_slices", oracleFP32 = "oracle_fp32"
    }
}

func referenceCheck(_ url: URL, cases: [DecideCase], graph: DecideGraph) throws -> (Bool, [String: Any]) {
    guard let data = try? Data(contentsOf: url) else { throw GateError.golden("missing \(url.path)") }
    let r = try JSONDecoder().decode(ReferenceFile.self, from: data)
    let (id, S, MMAX) = (r.caseID, r.S, r.MMAX)
    guard S == graph.seqLength, MMAX == graph.maxLabels else {
        throw GateError.usage("\(url.lastPathComponent) is for S=\(S) MMAX=\(MMAX), the bundle takes S=\(graph.seqLength) MMAX=\(graph.maxLabels)")
    }
    guard let c = cases.first(where: { $0.id == id }) else { throw GateError.golden("reference case \(id) is not in the fixtures") }
    let g = try graphInputs(c, S: S, MMAX: MMAX)
    func firstDiff(_ a: [Int32], _ b: [Int]) -> Int? {
        if a.count != b.count { return min(a.count, b.count) }
        return a.indices.first { Int(a[$0]) != b[$0] }
    }
    let dIds = firstDiff(g.inputIds, r.inputIDs), dMask = firstDiff(g.attentionMask, r.attentionMask)
    let dLabel = firstDiff(g.labelIdx, r.labelIdx)
    let slicesEqual = r.taskSlices.map { $0.start..<$0.end } == g.slices
        && r.taskSlices.map(\.task) == c.taskResults.map(\.task)
    let oracleEqual = r.oracleFP32.count == c.taskResults.count
        && zip(r.oracleFP32, c.taskResults).allSatisfy { $0.task == $1.task && $0.logits.map(\.bitPattern) == $1.logits.map(\.bitPattern) }
    let ok = dIds == nil && dMask == nil && dLabel == nil && slicesEqual && oracleEqual
    log("reference \(url.lastPathComponent) (\(id), len \(c.seqLen)): input_ids \(dIds.map { "differ at \($0)" } ?? "bit-equal \(S)/\(S)"), "
        + "attention_mask \(dMask.map { "differ at \($0)" } ?? "bit-equal \(S)/\(S)"), "
        + "label_idx \(dLabel.map { "differ at \($0)" } ?? "bit-equal \(MMAX)/\(MMAX)"), task slices \(slicesEqual ? "equal" : "DIFFER") "
        + "\(g.slices.map { "[\($0.lowerBound),\($0.upperBound))" }.joined(separator: " ")), oracle logits in fixture "
        + "\(oracleEqual ? "bit-equal" : "DIFFER") -> \(ok ? "PASS" : "FAIL")")
    return (ok, ["file": url.path, "case_id": id, "seq_len": c.seqLen, "S": S, "MMAX": MMAX,
                 "input_ids_first_diff": dIds ?? -1, "attention_mask_first_diff": dMask ?? -1,
                 "label_idx_first_diff": dLabel ?? -1, "task_slices_equal": slicesEqual,
                 "oracle_logits_bit_equal": oracleEqual, "pass": ok,
                 "swift_label_idx": g.labelIdx.map { Int($0) }, "task_slices": g.slices.map { [$0.lowerBound, $0.upperBound] }])
}

// MARK: - --pygpu: the host rule re-run on the Python engine's own logits

/// compareCase on the Python Mac GPU logits, field by field against the Python per-task records of the same
/// cases: the decision rule and the metrics of this port, apart from the engine.
func hostRuleCheck(_ url: URL, cases: [DecideCase], S: Int, MMAX: Int) throws -> (Bool, [String: Any]) {
    let rows = try loadMacGPUCases(url)
    guard rows.contains(where: { $0.tasks != nil }) else {
        log("host rule: skipped (\(url.lastPathComponent) has logits only, no per-task records)")
        return (true, ["file": url.path, "skipped": "no per-task records"])
    }
    let byID = Dictionary(uniqueKeysWithValues: cases.map { ($0.id, $0) })
    var tasks = 0, decisionsSame = 0, flagsSame = 0, missing = 0
    var bitSame = ["max_abs_dlogit": 0, "max_abs_dprob": 0, "oracle_margin": 0]
    var worst = ["max_abs_dlogit": 0.0, "max_abs_dprob": 0.0, "oracle_margin": 0.0]
    for row in rows {
        guard let c = byID[row.id], let py = row.tasks else { missing += 1; continue }
        let g = try graphInputs(c, S: S, MMAX: MMAX)
        let ours = try compareCase(c, logitsRow: row.logits, slices: g.slices)
        guard ours.count == py.count else { missing += 1; continue }
        for (o, p) in zip(ours, py) {
            tasks += 1
            if p.decision == o.decision { decisionsSame += 1 }
            if p.decisionEqual == o.decisionEqual { flagsSame += 1 }
            for (key, pv, v) in [("max_abs_dlogit", p.maxAbsDLogit, o.maxAbsDLogit), ("max_abs_dprob", p.maxAbsDProb, o.maxAbsDProb),
                                 ("oracle_margin", p.oracleMargin, o.oracleMargin)] {
                if pv.bitPattern == v.bitPattern { bitSame[key]! += 1 }
                worst[key] = max(worst[key]!, abs(pv - v))
            }
        }
    }
    let ok = missing == 0 && tasks > 0 && decisionsSame == tasks && flagsSame == tasks
    log("host rule on the Python engine's logits (\(rows.count) cases, \(tasks) tasks): decisions identical \(decisionsSame)/\(tasks), "
        + "decision_equal identical \(flagsSame)/\(tasks); bit-equal max_abs_dlogit \(bitSame["max_abs_dlogit"]!)/\(tasks) "
        + "(max |d| \(fmt(worst["max_abs_dlogit"]!))), max_abs_dprob \(bitSame["max_abs_dprob"]!)/\(tasks) "
        + "(max |d| \(fmt(worst["max_abs_dprob"]!))), oracle_margin \(bitSame["oracle_margin"]!)/\(tasks) -> \(ok ? "PASS" : "FAIL")")
    return (ok, ["file": url.path, "cases": rows.count, "cases_missing": missing, "tasks": tasks,
                 "decisions_identical": decisionsSame, "decision_equal_identical": flagsSame,
                 "bit_equal": bitSame, "max_abs_diff": worst, "pass": ok])
}

// MARK: - main

do {
    let args = try Args(CommandLine.arguments)
    guard let bundle = args.url("bundle") else { throw GateError.usage("--bundle <.aimodel|.aimodelc> is required") }
    let fixtureURLs = args.urls("fixtures")
    guard !fixtureURLs.isEmpty else { throw GateError.usage("--fixtures <json> [<json> ...] is required") }
    guard let unit = DecideComputeUnits(rawValue: args["unit"] ?? "gpu") else { throw GateError.usage("--unit gpu|cpuOnly|default") }
    let poison = args.has("poison")
    let benchN = args.int("bench") ?? 0
    let started = ISO8601DateFormatter().string(from: Date())
    var report: [String: Any] = [
        "kind": "gliner25-decide-swift-selftest", "argv": CommandLine.arguments, "started": started,
        "machine": sysctlString("machdep.cpu.brand_string"), "hw_model": sysctlString("hw.model"),
        "os": ProcessInfo.processInfo.operatingSystemVersionString, "os_build": sysctlString("kern.osversion"),
        "coreai_architecture": DecideGraph.deviceArchitecture,
        "timing_note": "contended: the Mac GPU is shared with other sessions; call_ms is wall time per call, not a benchmark"]

    let (cases, files) = try loadFixtures(fixtureURLs)
    log("fixtures: \(files.map { "\($0["set"] ?? "?") \($0["cases"] ?? 0) cases / \($0["tasks"] ?? 0) tasks" }.joined(separator: ", ")); "
        + "\(cases.count) cases, \(cases.reduce(0) { $0 + $1.taskResults.count }) tasks, longest \(cases.map(\.seqLen).max() ?? 0) tokens")
    report["fixtures"] = files

    let graph = try await DecideGraph(contentsOf: bundle, computeUnits: unit)
    log("loaded \(bundle.lastPathComponent) (\(unit.rawValue)) in \(String(format: "%.2f", graph.loadSeconds)) s: \(graph.contractDescription)")
    report["bundle"] = bundle.path
    report["unit"] = unit.rawValue
    report["load_s"] = graph.loadSeconds
    report["S"] = graph.seqLength
    report["MMAX"] = graph.maxLabels
    report["contract"] = graph.contractDescription
    var ok = true

    if let ref = args.url("reference") {
        let (p, r) = try referenceCheck(ref, cases: cases, graph: graph)
        ok = ok && p
        report["reference"] = r
    }

    if poison { log("poison: label_idx + 1 (negative control: the gate must FAIL)") }
    var rows: [CaseRecord] = []
    for (i, c) in cases.enumerated() {
        let r = try await runCase(graph, c, index: i, poison: poison)
        rows.append(r)
        if !r.decisionsEqual || i < 3 || i % 50 == 0 {
            log(caseLine(i, r))
            if !poison { diffLines(r).forEach(log) }
        }
    }

    var pyOK = true
    if let py = args.url("pygpu") {
        let mac = try loadMacGPULogits(py)
        for i in rows.indices {
            if let m = mac[rows[i].id] { rows[i].vsMacGPU = logitsDiff(rows[i], cases[i], other: m) }
        }
        let s = summarizeMacGPU(rows)
        log("vs Python Mac GPU logits (\(py.lastPathComponent), same bundle): \(s["cases"] ?? 0) cases "
            + "(\(s["cases_without_mac_row"] ?? 0) without a row), max|Δlogit| \(fmt(s["max_abs_dlogit"] as? Double ?? .nan)), "
            + "bit-equal \(s["bit_equal"] ?? 0)/\(s["elements"] ?? 0), decisions equal \(s["decisions_equal_tasks"] ?? 0)/\(s["tasks"] ?? 0)")
        var j = s
        j["file"] = py.path
        if !poison {
            let (p, h) = try hostRuleCheck(py, cases: cases, S: graph.seqLength, MMAX: graph.maxLabels)
            pyOK = p
            j["host_rule"] = h
        }
        report["vs_python_mac_gpu"] = j
    }

    let summary = summarize(rows)
    let pass = gatePasses(rows)
    log("summary: \(summary["decisions_equal_tasks"] ?? 0)/\(summary["tasks"] ?? 0) decisions equal "
        + "(\(summary["cases_with_changed_decision"] ?? 0) cases changed), max|Δlogit| \(fmt(summary["max_abs_dlogit"] as? Double ?? .nan)), "
        + "max|Δprob| \(fmt(summary["max_abs_dprob"] as? Double ?? .nan)), min cos \(String(format: "%.7f", summary["min_cos"] as? Double ?? .nan)), "
        + "non-finite \(summary["nonfinite_cases"] ?? 0), first call \(String(format: "%.1f", summary["first_call_ms"] as? Double ?? .nan)) ms, "
        + "warm median \(String(format: "%.2f", summary["warm_call_ms_median"] as? Double ?? .nan)) ms / p90 "
        + "\(String(format: "%.2f", summary["warm_call_ms_p90"] as? Double ?? .nan)) ms (contended)")
    if poison {
        log("poison: \(pass ? "the gate still PASSES — the comparator cannot go red" : "the gate goes red as it must")")
    } else {
        log("gate (decisions 100 % equal, every row finite) -> \(pass ? "PASS" : "FAIL")")
    }
    ok = ok && pass && pyOK
    report["poison"] = poison
    report["summary"] = summary
    report["pass_decisions"] = pass
    report["cases"] = rows.map(\.json)

    if benchN > 0 {
        let load0 = loadAverage()
        let ts = try await benchCalls(graph, cases, calls: benchN)
        var b = benchJSON(ts)
        b["load_avg_start"] = load0
        b["load_avg_end"] = loadAverage()
        log("bench: \(String(format: "%.2f", b["ms_median"] as? Double ?? .nan)) ms median, p90 "
            + "\(String(format: "%.2f", b["ms_p90"] as? Double ?? .nan)), min \(String(format: "%.2f", b["ms_min"] as? Double ?? .nan)), "
            + "max \(String(format: "%.2f", b["ms_max"] as? Double ?? .nan)) (\(benchN) calls after 5 warm-up; contended GPU, "
            + "load avg \(load0))")
        report["bench"] = b
    }
    report["pass"] = ok
    report["finished"] = ISO8601DateFormatter().string(from: Date())
    if let path = args.url("json") {
        try writeJSON(report, to: path)
        log("wrote \(path.path)")
    }
    log(ok ? "EXIT 0 (all requested gates PASS)" : "EXIT 3 (a gate FAILED)")
    exit(ok ? 0 : 3)
} catch {
    log("ERROR: \(error)")
    exit(4)
}
