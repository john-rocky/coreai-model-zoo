// Readout — the option probabilities, the answers and the response (conversion/d1/host.py §5–6: `option_logits`,
// `probs_from_logits`, `py_sum`, `answer`, `response`; the provider's prompt.readout and api.answer):
//
//   z[id]   = h_slot . E[id] in float64 (the fp16 / fp32 hidden row and the fp32 option rows widened exactly), through
//             NumPy's own BLAS call for `E @ h` (D1BLAS: Accelerate's cblas_dgemv, cblas_ddot for one id, on
//             page-aligned copies as NumPy's arrays are), so the logits equal host.py's bit for bit on the same machine
//   score_k = max over group k of z (Python's max: a later equal value never replaces the first)
//   p       = exp(score - max(score)) / Python 3.12 sum() of them, in Double (the C library's exp, as math.exp)
//   noul    {"type": "noul", "noul": p[0]}
//   choice  {"type": "choice", "choice": labels[argmax], "confidence": p[argmax], "probabilities": {label: p}}
//   score   {"type": "score", "score": sum(i * p_i), "confidence": p[argmax], "probabilities": {"i": p_i},
//            "legend": {"i": level_i}}
//   argmax  the first index of the largest p; the floats are not rounded (JSON as Python's repr)
//   body    {"answers": {name: answer} in request order, "usage": {"input_tokens": n, "output_tokens": 0}}
// The provider's readout takes the log-softmax over the whole vocabulary first; a constant per row cancels in the
// softmax over options (round 1 measured the difference: max |dp| 1.75e-7 on random fp32 vocabularies).

import D1BLAS
import Foundation

/// The bundle's option table (head/option_rows): ascending ids and, once a bundle exists, their fp32 rows.
public struct D1OptionTable: Sendable {
    public let ids: [Int]
    public let hidden: Int
    /// row-major [ids.count, hidden], as Double (fp32 widened exactly); empty for an id list alone
    public let rows: [Double]
    let index: [Int: Int]

    public var idSet: Set<Int> { Set(ids) }

    public init(ids: [Int], hidden: Int = 0, rows: [Double] = []) throws {
        guard ids == ids.sorted(), Set(ids).count == ids.count else {
            throw D1Error.contract("option table ids are not distinct and ascending")
        }
        guard rows.isEmpty || rows.count == ids.count * hidden else {
            throw D1Error.contract("option table rows: \(rows.count) values for \(ids.count) x \(hidden)")
        }
        self.ids = ids
        self.hidden = hidden
        self.rows = rows
        index = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($0.element, $0.offset) })
    }

    /// An option_rows.json (or the round-3a stand-in option_ids.json): its `ids`.
    public static func ids(json url: URL) throws -> D1OptionTable {
        let j = try JSONParser.parse(Data(contentsOf: url))
        guard let ids = j["ids"]?.array?.compactMap(\.intValue), !ids.isEmpty else {
            throw D1Error.bundle("\(url.lastPathComponent): no `ids`")
        }
        return try D1OptionTable(ids: ids)
    }

    /// The rows of `ids`, gathered row-major as Double.
    func gather(_ want: [Int]) throws -> [Double] {
        guard !rows.isEmpty else { throw D1Error.contract("the option table holds ids only (no rows)") }
        var out = [Double]()
        out.reserveCapacity(want.count * hidden)
        for i in want {
            guard let k = index[i] else { throw D1Error.contract("id \(i) is not in the option table") }
            out += rows[(k * hidden)..<((k + 1) * hidden)]
        }
        return out
    }
}

public enum D1Readout {
    /// Python 3.12's `sum()` of floats (bltinmodule.c): the first value added to int 0, then left to right with
    /// Neumaier's compensation, the compensation added at the end when it is nonzero and finite (host.py `py_sum`).
    public static func pySum(_ values: [Double]) -> Double {
        guard let first = values.first else { return 0 }
        var f = 0.0 + first
        var c = 0.0
        for x in values.dropFirst() {
            let t = f + x
            if abs(f) >= abs(x) {
                c += (f - t) + x
            } else {
                c += (x - t) + f
            }
            f = t
        }
        if c != 0 && c.isFinite { f += c }
        return f
    }

    /// Every id the readout reads, in first-seen order (`host.group_ids`).
    public static func groupIDs(_ groups: [[Int]]) -> [Int] {
        var out: [Int] = []
        for g in groups { for i in g where !out.contains(i) { out.append(i) } }
        return out
    }

    /// `host.option_logits`: z = E[ids] @ h in float64, NumPy's call (D1BLAS `d1_matvec`); `rows` row-major
    /// [ids.count, h.count].
    public static func optionLogits(hidden h: [Double], rows: [Double], count: Int) -> [Double] {
        precondition(rows.count == count * h.count, "optionLogits: \(rows.count) values for \(count) x \(h.count)")
        var y = [Double](repeating: 0, count: count)
        rows.withUnsafeBufferPointer { a in
            h.withUnsafeBufferPointer { x in
                y.withUnsafeMutableBufferPointer { out in
                    let rc = d1_matvec(a.baseAddress!, count, h.count, x.baseAddress!, out.baseAddress!)
                    precondition(rc == 0, "d1_matvec: could not allocate its aligned copies")
                }
            }
        }
        return y
    }

    /// `host.probs_from_logits`: group max, then the softmax in double.
    public static func probabilities(logits z: [Int: Double], groups: [[Int]]) -> [Double] {
        let scores = groups.map { g -> Double in
            var m = z[g[0]]!
            for i in g.dropFirst() where z[i]! > m { m = z[i]! }
            return m
        }
        var m = scores[0]
        for s in scores.dropFirst() where s > m { m = s }
        let exps = scores.map { Foundation.exp($0 - m) }
        let total = pySum(exps)
        return exps.map { $0 / total }
    }

    /// `host.readout`: the slot's hidden row and the table's rows -> the option probabilities (and the logits).
    public static func readout(hidden: [Double], table: D1OptionTable, groups: [[Int]]) throws -> (logits: [Int: Double], p: [Double]) {
        let ids = groupIDs(groups)
        let z = optionLogits(hidden: hidden, rows: try table.gather(ids), count: ids.count)
        let logits = Dictionary(uniqueKeysWithValues: zip(ids, z))
        return (logits, probabilities(logits: logits, groups: groups))
    }

    /// The first index of the largest value (Python's max over range(n) with a key).
    static func firstArgmax(_ p: [Double]) -> Int {
        var best = 0
        for i in 1..<p.count where p[i] > p[best] { best = i }
        return best
    }

    /// `api.answer` for one question.
    public static func answer(_ q: D1Question, _ p: [Double]) -> JSONValue {
        switch q.kind {
        case .noul:
            return .obj([("type", .string("noul")), ("noul", .double(p[0]))])
        case .choice:
            let best = firstArgmax(p)
            return .obj([("type", .string("choice")), ("choice", .string(q.labels[best])), ("confidence", .double(p[best])),
                         ("probabilities", .obj(zip(q.labels, p).map { ($0, .double($1)) }))])
        case .score:
            let best = firstArgmax(p)
            let score = pySum(p.enumerated().map { Double($0.offset) * $0.element })
            return .obj([("type", .string("score")), ("score", .double(score)), ("confidence", .double(p[best])),
                         ("probabilities", .obj(p.enumerated().map { (String($0.offset), .double($0.element)) })),
                         ("legend", .obj(q.levels.enumerated().map { (String($0.offset), .string($0.element)) }))])
        }
    }

    /// `host.response`: the answers in request order and the usage.
    public static func response(_ questions: [D1Question], probabilities: [[Double]], inputTokens: Int) -> JSONValue {
        .obj([("answers", .obj(zip(questions, probabilities).map { ($0.name, answer($0, $1)) })),
              ("usage", .obj([("input_tokens", .int(inputTokens)), ("output_tokens", .int(0))]))])
    }
}
