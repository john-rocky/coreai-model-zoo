// DecideGateSupport — what the Mac self-test (decide-selftest) and the iPhone gate app (apps/DecideGate) share:
// the oracle fixture reader, a 1:1 port of ../../export_gliner25_decide.py's host side (graph_inputs, probs_of,
// decide, compare_case, cos, summarize), one engine call per fixture case, the Mac-GPU logits reader, timing
// statistics and JSON output. One copy, so a number from the phone means what the same number from the Mac means.
//
// The fixtures are ../../gliner25_decide_oracle.py's JSON (gliner2 2.0.0, fp32, CPU): each case carries the
// oracle's unpadded input_ids, the [P] / [L] marker positions per task, and per task the labels, the fp32 logits,
// the probabilities and the decision.

import Darwin
import DecideGraph
import Foundation

public enum GateError: Error, CustomStringConvertible {
    case usage(String)
    case fixture(String)
    case input(String)
    case golden(String)

    public var description: String {
        switch self {
        case .usage(let s): return "usage: \(s)"
        case .fixture(let s): return "fixture: \(s)"
        case .input(let s): return "graph input: \(s)"
        case .golden(let s): return "golden: \(s)"
        }
    }
}

/// The model the fixtures and bundles come from, and the marker ids of its vocabulary (tokenizer_config.json).
public enum DecideModel {
    public static let id = "fastino/GLiNER2.5-Decide"
    public static let revision = "7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6"
    public static let markerP = 128003                   // [P]
    public static let markerL = 128007                   // [L]
    public static let padID: Int32 = 0                   // [PAD]
}

// MARK: - fixtures

public struct DecideTaskResult: Decodable, Sendable {
    public let task: String
    public let labels: [String]
    public let multiLabel: Bool
    public let clsThreshold: Double
    public let classAct: String
    public let logits: [Double]
    public let probs: [Double]
    /// The oracle's decision as a list (the JSON stores a single-label decision as one string).
    public let decision: [String]

    enum CodingKeys: String, CodingKey {
        case task, labels, logits, probs, decision
        case multiLabel = "multi_label", clsThreshold = "cls_threshold", classAct = "class_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        task = try c.decode(String.self, forKey: .task)
        labels = try c.decode([String].self, forKey: .labels)
        multiLabel = try c.decode(Bool.self, forKey: .multiLabel)
        clsThreshold = try c.decode(Double.self, forKey: .clsThreshold)
        classAct = try c.decode(String.self, forKey: .classAct)
        logits = try c.decode([Double].self, forKey: .logits)
        probs = try c.decode([Double].self, forKey: .probs)
        // compare_case: ref_dec = tr["decision"] if tr["multi_label"] else [tr["decision"]]
        decision = multiLabel ? try c.decode([String].self, forKey: .decision) : [try c.decode(String.self, forKey: .decision)]
    }
}

public struct DecideCase: Decodable, Sendable {
    public let id: String
    public let seqLen: Int
    public let inputIds: [Int]
    public let schemaSpecialIndices: [[Int]]
    public let taskResults: [DecideTaskResult]

    enum CodingKeys: String, CodingKey {
        case id
        case seqLen = "seq_len", inputIds = "input_ids", schemaSpecialIndices = "schema_special_indices"
        case taskResults = "task_results"
    }

    public var labelCount: Int { taskResults.reduce(0) { $0 + $1.labels.count } }
}

public struct FixtureHeader: Decodable, Sendable {
    public let set: String
    public let created: String
    public let modelRev: String

    enum CodingKeys: String, CodingKey { case set, created, modelRev = "model_rev" }
}

struct FixtureFile: Decodable {
    let header: FixtureHeader
    let cases: [DecideCase]
}

/// Every case of the fixture files, in file order (export_gliner25_decide.load_fixtures).
public func loadFixtures(_ urls: [URL]) throws -> (cases: [DecideCase], files: [[String: Any]]) {
    var cases: [DecideCase] = []
    var files: [[String: Any]] = []
    var seen = Set<String>()
    for url in urls {
        guard let data = try? Data(contentsOf: url) else { throw GateError.fixture("missing \(url.path)") }
        let f = try JSONDecoder().decode(FixtureFile.self, from: data)
        guard f.header.modelRev == DecideModel.revision else {
            throw GateError.fixture("\(url.lastPathComponent) is for model rev \(f.header.modelRev), not \(DecideModel.revision)")
        }
        for c in f.cases {
            guard seen.insert(c.id).inserted else { throw GateError.fixture("case \(c.id) appears twice") }
        }
        cases.append(contentsOf: f.cases)
        files.append(["path": url.path, "set": f.header.set, "created": f.header.created, "cases": f.cases.count,
                      "tasks": f.cases.reduce(0) { $0 + $1.taskResults.count }])
    }
    return (cases, files)
}

// MARK: - graph inputs (graph_inputs)

public struct GraphInputs: Sendable {
    public let inputIds: [Int32]
    public let attentionMask: [Int32]
    public let labelIdx: [Int32]
    /// Each task's [start, end) in labelIdx and in the logits row, in task order.
    public let slices: [Range<Int>]
}

/// input_ids padded with [PAD] (0) to S, attention_mask 1 on the real tokens, label_idx = the [L] positions of
/// every task concatenated in task order, the unused slots repeating the first [L] position. `poison` adds 1 to
/// every label_idx slot (each label's first sub-word instead of its [L] marker): the comparator must go red.
public func graphInputs(_ c: DecideCase, S: Int, MMAX: Int, poison: Bool = false) throws -> GraphInputs {
    let ids = c.inputIds
    let n = ids.count
    guard n <= S else { throw GateError.input("\(c.id): \(n) tokens > S=\(S)") }
    var inputIds = [Int32](repeating: DecideModel.padID, count: S)
    var mask = [Int32](repeating: 0, count: S)
    for i in 0..<n {
        inputIds[i] = Int32(ids[i])
        mask[i] = 1
    }
    var lPos: [Int] = []
    var slices: [Range<Int>] = []
    for ssi in c.schemaSpecialIndices {
        guard let p = ssi.first, (0..<n).contains(p), ids[p] == DecideModel.markerP else {
            throw GateError.input("\(c.id): \(ssi) does not start at a [P] marker")
        }
        guard ssi.dropFirst().allSatisfy({ (0..<n).contains($0) && ids[$0] == DecideModel.markerL }) else {
            throw GateError.input("\(c.id): \(ssi) has a position that is not an [L] marker")
        }
        slices.append(lPos.count..<(lPos.count + ssi.count - 1))
        lPos.append(contentsOf: ssi.dropFirst())
    }
    guard !lPos.isEmpty, lPos.count <= MMAX else { throw GateError.input("\(c.id): \(lPos.count) labels, MMAX=\(MMAX)") }
    var labelIdx = [Int32](repeating: Int32(lPos[0]), count: MMAX)
    for (k, p) in lPos.enumerated() { labelIdx[k] = Int32(p) }
    if poison { labelIdx = labelIdx.map { $0 + 1 } }
    return GraphInputs(inputIds: inputIds, attentionMask: mask, labelIdx: labelIdx, slices: slices)
}

// MARK: - decision rule (probs_of / decide), in NumPy's float64 order

/// numpy.add.reduce over a contiguous float64 vector: pairwise_sum (below 8 elements a plain loop from -0.0,
/// up to 128 eight accumulators, above that halves), so a softmax matches the Python host bit for bit.
func pairwiseSum(_ a: ArraySlice<Double>) -> Double {
    let n = a.count, b = a.startIndex
    if n < 8 {
        var res = -0.0
        for x in a { res += x }
        return res
    }
    if n <= 128 {
        var r = Array(a[b..<(b + 8)])
        var i = 8
        while i < n - (n % 8) {
            for j in 0..<8 { r[j] += a[b + i + j] }
            i += 8
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n {
            res += a[b + i]
            i += 1
        }
        return res
    }
    var n2 = n / 2
    n2 -= n2 % 8
    return pairwiseSum(a[b..<(b + n2)]) + pairwiseSum(a[(b + n2)...])
}

/// probs_of: sigmoid for class_act "sigmoid", or "auto" on a multi-label task; softmax otherwise.
public func probsOf(_ logits: [Double], multi: Bool, act: String) -> [Double] {
    if act == "sigmoid" || (act == "auto" && multi) {
        return logits.map { 1.0 / (1.0 + exp(-$0)) }
    }
    let m = nanMax(logits)
    let e = logits.map { exp($0 - m) }
    let s = pairwiseSum(e[...])
    return e.map { $0 / s }
}

/// numpy.argmax: the first maximum; the first NaN if there is one.
public func argmax(_ xs: [Double]) -> Int {
    guard !xs.isEmpty else { return 0 }
    if let k = xs.firstIndex(where: { $0.isNaN }) { return k }
    var best = 0
    for k in 1..<xs.count where xs[k] > xs[best] { best = k }
    return best
}

/// decide: multi-label = every label with prob >= threshold (none -> argmax); single-label = argmax.
public func decide(_ probs: [Double], multi: Bool, threshold: Double) -> [Int] {
    if multi {
        let hits = probs.indices.filter { probs[$0] >= threshold }
        return hits.isEmpty ? [argmax(probs)] : hits
    }
    return [argmax(probs)]
}

/// numpy.max: NaN if any element is NaN.
public func nanMax(_ xs: [Double]) -> Double {
    var m = -Double.infinity
    for x in xs {
        if x.isNaN { return .nan }
        m = max(m, x)
    }
    return m
}

// MARK: - comparison (compare_case / cos / summarize)

public struct TaskComparison: Sendable {
    public let task: String
    public let multiLabel: Bool
    public let nLabels: Int
    public let decisionEqual: Bool
    public let decision: [String]
    public let oracleDecision: [String]
    public let maxAbsDLogit: Double
    public let maxAbsDProb: Double
    public let oracleMargin: Double

    public var json: [String: Any] {
        ["task": task, "multi_label": multiLabel, "n_labels": nLabels, "decision_equal": decisionEqual,
         "decision": decision, "oracle_decision": oracleDecision, "max_abs_dlogit": maxAbsDLogit,
         "max_abs_dprob": maxAbsDProb, "oracle_margin": oracleMargin]
    }
}

/// compare_case: one [MMAX] logits row against the oracle's fp32 task results, task by task.
public func compareCase(_ c: DecideCase, logitsRow: [Double], slices: [Range<Int>]) throws -> [TaskComparison] {
    guard slices.count == c.taskResults.count else { throw GateError.input("\(c.id): \(slices.count) slices") }
    var out: [TaskComparison] = []
    for (tr, s) in zip(c.taskResults, slices) {
        guard s.upperBound <= logitsRow.count else { throw GateError.input("\(c.id): slice \(s) past the row") }
        let got = Array(logitsRow[s])
        guard got.count == tr.logits.count, tr.probs.count == tr.logits.count else {
            throw GateError.fixture("\(c.id) \(tr.task): \(got.count) logits vs \(tr.logits.count) in the oracle")
        }
        let p = probsOf(got, multi: tr.multiLabel, act: tr.classAct)
        let dec = decide(p, multi: tr.multiLabel, threshold: tr.clsThreshold)
        var refIdx: [Int] = []
        for label in tr.decision {
            guard let k = tr.labels.firstIndex(of: label) else {
                throw GateError.fixture("\(c.id) \(tr.task): oracle decision \(label) is not a label")
            }
            refIdx.append(k)
        }
        let margin: Double
        if tr.multiLabel {
            margin = tr.probs.map { abs($0 - tr.clsThreshold) }.min() ?? .infinity
        } else {
            let top = tr.logits.sorted(by: >)
            margin = top.count > 1 ? top[0] - top[1] : .infinity
        }
        out.append(TaskComparison(
            task: tr.task, multiLabel: tr.multiLabel, nLabels: s.count, decisionEqual: dec == refIdx,
            decision: dec.map { tr.labels[$0] }, oracleDecision: tr.decision,
            maxAbsDLogit: nanMax(zip(got, tr.logits).map { abs($0 - $1) }),
            maxAbsDProb: nanMax(zip(p, tr.probs).map { abs($0 - $1) }),
            oracleMargin: margin))
    }
    return out
}

/// cos (auxiliary, not a gate).
public func cosine(_ a: [Double], _ b: [Double]) -> Double {
    var ab = 0.0, aa = 0.0, bb = 0.0
    for (x, y) in zip(a, b) {
        ab += x * y
        aa += x * x
        bb += y * y
    }
    return ab / (aa.squareRoot() * bb.squareRoot())
}

/// One engine call on one fixture case (engine_gate's row).
public struct CaseRecord: Sendable {
    public let id: String
    public let seqLen: Int
    public let nLabels: Int
    public let callMs: Double
    public let firstCall: Bool
    public let finite: Bool
    public let allZero: Bool
    public let tasks: [TaskComparison]
    public let cos: Double
    /// The first nLabels slots of the row (the rest repeat the first [L] and are never read).
    public let logits: [Float]
    public let slices: [Range<Int>]
    public var vsMacGPU: LogitsDiff? = nil

    public var decisionsEqual: Bool { tasks.allSatisfy(\.decisionEqual) }
    public var maxAbsDLogit: Double { nanMax(tasks.map(\.maxAbsDLogit)) }
    public var maxAbsDProb: Double { nanMax(tasks.map(\.maxAbsDProb)) }

    public var json: [String: Any] {
        var j: [String: Any] = ["id": id, "seq_len": seqLen, "n_labels": nLabels, "call_ms": callMs,
                                "first_call": firstCall, "finite": finite, "all_zero": allZero,
                                "tasks": tasks.map(\.json), "cos": cos, "logits": logits.map { Double($0) }]
        if let v = vsMacGPU { j["vs_mac_gpu"] = v.json }
        return j
    }
}

/// Builds the inputs, runs the graph once and compares the row with the oracle.
public func runCase(_ graph: DecideGraph, _ c: DecideCase, index: Int, poison: Bool = false) async throws -> CaseRecord {
    let g = try graphInputs(c, S: graph.seqLength, MMAX: graph.maxLabels, poison: poison)
    let t0 = ContinuousClock.now
    let row = try await graph.run(inputIds: g.inputIds, mask: g.attentionMask, labelIdx: g.labelIdx)
    let ms = seconds(since: t0) * 1e3
    let n = g.slices.last?.upperBound ?? 0
    let row64 = row.map { Double($0) }
    let refAll = c.taskResults.flatMap(\.logits)
    return CaseRecord(id: c.id, seqLen: c.seqLen, nLabels: n, callMs: ms, firstCall: index == 0,
                      finite: row.allSatisfy(\.isFinite), allZero: row[0..<n].allSatisfy { $0 == 0 },
                      tasks: try compareCase(c, logitsRow: row64, slices: g.slices),
                      cos: cosine(Array(row64[0..<n]), refAll), logits: Array(row[0..<n]), slices: g.slices)
}

/// summarize + the engine fields of engine_gate (the same keys as the Python GATE-3 summary).
public func summarize(_ rows: [CaseRecord]) -> [String: Any] {
    let tasks = rows.flatMap(\.tasks)
    let equal = tasks.filter(\.decisionEqual).count
    let ms = rows.map(\.callMs)
    let warm = rows.filter { !$0.firstCall }.map(\.callMs)
    let w = warm.isEmpty ? ms : warm
    return [
        "cases": rows.count, "tasks": tasks.count, "decisions_equal_tasks": equal,
        "decisions_equal_pct": tasks.isEmpty ? 0 : (100_000.0 * Double(equal) / Double(tasks.count)).rounded() / 1000,
        "cases_with_changed_decision": rows.filter { !$0.decisionsEqual }.count,
        "max_abs_dlogit": nanMax(tasks.map(\.maxAbsDLogit)), "max_abs_dprob": nanMax(tasks.map(\.maxAbsDProb)),
        "min_cos": rows.map(\.cos).min() ?? .nan,
        "nonfinite_cases": rows.filter { !$0.finite }.count, "all_zero_cases": rows.filter(\.allZero).count,
        "first_call_ms": ms.first ?? .nan, "warm_call_ms_median": percentile(w, 0.5),
        "warm_call_ms_p90": percentile(w, 0.9), "warm_call_ms_max": w.max() ?? .nan,
    ]
}

/// The gate: every task's decision equals the oracle's and every row is finite.
public func gatePasses(_ rows: [CaseRecord]) -> Bool {
    !rows.isEmpty && rows.allSatisfy { $0.decisionsEqual && $0.finite }
}

/// The log line engine_gate prints for a case.
public func caseLine(_ i: Int, _ r: CaseRecord) -> String {
    let id = r.id.count < 34 ? r.id + String(repeating: " ", count: 34 - r.id.count) : r.id
    return String(format: "[%3d] ", i) + (r.decisionsEqual ? "OK   " : "DIFF ") + id
        + String(format: " len=%3d max|dlogit|=%.4f max|dprob|=%.5f %7.1f ms", r.seqLen, r.maxAbsDLogit, r.maxAbsDProb, r.callMs)
        + (r.vsMacGPU.map { String(format: " | vs Mac GPU max|d| %.4g", $0.maxAbs) } ?? "")
}

/// The detail lines of a case whose decision differs.
public func diffLines(_ r: CaseRecord) -> [String] {
    r.tasks.filter { !$0.decisionEqual }.map {
        "      task \($0.task): engine \($0.decision) vs oracle \($0.oracleDecision) (oracle margin \(String(format: "%.5f", $0.oracleMargin)))"
    }
}

// MARK: - the Mac GPU's logits for the same bundle (Python engine_gate)

public struct LogitsDiff: Sendable {
    public let maxAbs: Double
    public let bitEqual: Int
    public let elements: Int
    public let decisionsEqual: Int
    public let tasks: Int

    public var json: [String: Any] {
        ["max_abs_dlogit": maxAbs, "bit_equal": bitEqual, "elements": elements, "decisions_equal_tasks": decisionsEqual,
         "tasks": tasks]
    }
}

/// One case of the Python engine gate (engine.clean.cases[*]); `tasks` is absent in the staged golden files.
public struct PyEngineCase: Decodable, Sendable {
    public struct Task: Decodable, Sendable {
        public let task: String
        public let decision: [String]
        public let decisionEqual: Bool
        public let maxAbsDLogit: Double
        public let maxAbsDProb: Double
        public let oracleMargin: Double

        enum CodingKeys: String, CodingKey {
            case task, decision
            case decisionEqual = "decision_equal", maxAbsDLogit = "max_abs_dlogit", maxAbsDProb = "max_abs_dprob"
            case oracleMargin = "oracle_margin"
        }
    }

    public let id: String
    public let logits: [Double]
    public let tasks: [Task]?
}

struct MacGPUFile: Decodable {
    struct Pass: Decodable { let cases: [PyEngineCase] }
    struct Engine: Decodable { let clean: Pass }
    let engine: Engine?
    let logits: [String: [Double]]?
}

/// The Mac GPU cases (logits = the first n_labels slots). Reads either the Python gate JSON
/// (results/gate_s<S>_gpu.json: engine.clean.cases[*]) or the staged golden file ({"logits": {id: [...]}}).
/// JSONDecoder, not JSONSerialization: the latter does not round every decimal to the nearest double.
public func loadMacGPUCases(_ url: URL) throws -> [PyEngineCase] {
    guard let data = try? Data(contentsOf: url) else { throw GateError.golden("missing \(url.path)") }
    let f = try JSONDecoder().decode(MacGPUFile.self, from: data)
    if let cases = f.engine?.clean.cases { return cases }
    guard let m = f.logits else { throw GateError.golden("\(url.lastPathComponent): no engine.clean.cases and no logits map") }
    return m.keys.sorted().map { PyEngineCase(id: $0, logits: m[$0]!, tasks: nil) }
}

/// case id -> the Mac GPU logits (loadMacGPUCases).
public func loadMacGPULogits(_ url: URL) throws -> [String: [Double]] {
    Dictionary(try loadMacGPUCases(url).map { ($0.id, $0.logits) }, uniquingKeysWith: { a, _ in a })
}

/// Ours vs another run's logits of the same case: max |d|, bit-equal elements (as float32), and whether the two
/// rows give the same decision task by task.
public func logitsDiff(_ r: CaseRecord, _ c: DecideCase, other: [Double]) -> LogitsDiff {
    let n = min(r.logits.count, other.count)
    var diffs: [Double] = []
    var same = 0
    for i in 0..<n {
        diffs.append(abs(Double(r.logits[i]) - other[i]))
        if Float(other[i]).bitPattern == r.logits[i].bitPattern { same += 1 }
    }
    var equal = 0
    for (tr, s) in zip(c.taskResults, r.slices) where s.upperBound <= n {
        let a = decide(probsOf(r.logits[s].map { Double($0) }, multi: tr.multiLabel, act: tr.classAct),
                       multi: tr.multiLabel, threshold: tr.clsThreshold)
        let b = decide(probsOf(Array(other[s]), multi: tr.multiLabel, act: tr.classAct),
                       multi: tr.multiLabel, threshold: tr.clsThreshold)
        if a == b { equal += 1 }
    }
    // a length mismatch must not pass silently
    let m = r.logits.count == other.count ? nanMax(diffs) : Double.nan
    return LogitsDiff(maxAbs: m, bitEqual: same, elements: n, decisionsEqual: equal, tasks: c.taskResults.count)
}

/// The vs-Mac-GPU totals over the cases that have a Mac row.
public func summarizeMacGPU(_ rows: [CaseRecord]) -> [String: Any] {
    let d = rows.compactMap(\.vsMacGPU)
    return ["cases": d.count, "cases_without_mac_row": rows.count - d.count,
            "max_abs_dlogit": d.isEmpty ? Double.nan : nanMax(d.map(\.maxAbs)),
            "bit_equal": d.reduce(0) { $0 + $1.bitEqual }, "elements": d.reduce(0) { $0 + $1.elements },
            "decisions_equal_tasks": d.reduce(0) { $0 + $1.decisionsEqual }, "tasks": d.reduce(0) { $0 + $1.tasks }]
}

// MARK: - bench

/// warm-up calls on cases 0..<warmup, then `calls` timed calls on case (7 k) mod n; ms per call.
public func benchCalls(_ graph: DecideGraph, _ cases: [DecideCase], calls: Int, warmup: Int = 5) async throws -> [Double] {
    guard !cases.isEmpty else { return [] }
    let order = (0..<warmup).map { $0 % cases.count } + (0..<calls).map { ($0 * 7) % cases.count }
    let inputs = try order.map { try graphInputs(cases[$0], S: graph.seqLength, MMAX: graph.maxLabels) }
    for k in 0..<warmup {
        _ = try await graph.run(inputIds: inputs[k].inputIds, mask: inputs[k].attentionMask, labelIdx: inputs[k].labelIdx)
    }
    var ts: [Double] = []
    for k in warmup..<inputs.count {
        let t0 = ContinuousClock.now
        _ = try await graph.run(inputIds: inputs[k].inputIds, mask: inputs[k].attentionMask, labelIdx: inputs[k].labelIdx)
        ts.append(seconds(since: t0) * 1e3)
    }
    return ts
}

public func benchJSON(_ ts: [Double], warmup: Int = 5) -> [String: Any] {
    ["calls": ts.count, "warmup": warmup, "ms_median": percentile(ts, 0.5), "ms_p90": percentile(ts, 0.9),
     "ms_min": ts.min() ?? .nan, "ms_max": ts.max() ?? .nan, "inputs": "fixture cases, (7 k) mod n"]
}

// MARK: - helpers

public func percentile(_ xs: [Double], _ q: Double) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    let pos = q * Double(s.count - 1)
    let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))           // numpy.percentile (linear)
}

public func seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

public func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

public func fmt(_ x: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)e", x) }

/// JSONSerialization raises (it does not throw) on NaN / infinity: replace them with strings.
public func jsonSafe(_ v: Any) -> Any {
    switch v {
    case let d as Double: return d.isFinite ? d : "\(d)"
    case let f as Float: return f.isFinite ? Double(f) : "\(f)"
    case let a as [Any]: return a.map(jsonSafe)
    case let m as [String: Any]: return m.mapValues(jsonSafe)
    default: return v
    }
}

/// Pretty, key-sorted JSON (NaN / infinity as strings), written through a temporary file.
public func writeJSON(_ object: [String: Any], to url: URL) throws {
    let clean = jsonSafe(object)
    guard JSONSerialization.isValidJSONObject(clean) else { throw GateError.usage("not serializable: \(url.lastPathComponent)") }
    let d = try JSONSerialization.data(withJSONObject: clean, options: [.prettyPrinted, .sortedKeys])
    let tmp = url.appendingPathExtension("tmp")
    try d.write(to: tmp)
    _ = try? FileManager.default.removeItem(at: url)
    try FileManager.default.moveItem(at: tmp, to: url)
}
