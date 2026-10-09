// Readout — the scores at the markers -> the publisher's probabilities -> the answers and the response
// (conversion/d1_omni/host.py §3–4: `probabilities_from_logits` = modeling_d1 `_forward`, prompt.py `answer`,
// `response`):
//
//   z = logits[:K] in float32; a text row: z / float32(T); p = softmax(z) in float32 as NumPy computes it
//   (exp = libm expf, the sum = NumPy's add.reduce: 0 + a pairwise sum, 8 partial sums from 8 values on);
//   a noul is scored as [false, true] and reported as [yes, no] (p reversed)
//   noul   {"type": "noul", "noul": p_yes}
//   choice {"type": "choice", "choice": the first argmax's name, "confidence": p[best], "probabilities": {name: p}}
//   score  {"type": "score", "score": sum_i i * p_i (Python 3.12's sum), "confidence": p[best],
//           "probabilities": {"i": p_i}, "legend": {"i": _criterion(level)}}
//   body   {"answers": {name: answer}, "usage": {"input_tokens": sum of the rows' positions, "output_tokens": 0}}
//
// The probabilities are float32 values; the answers carry them as Python floats (the same doubles).

import Darwin
import Foundation

public enum Readout {
    /// NumPy's float32 `add.reduce` of a contiguous vector: the identity 0 plus `pairwise_sum` (loops_utils.h.src):
    /// below 8 values a left-to-right sum from -0.0; up to 128 values eight partial sums (r[j] += a[i + j]) combined as
    /// ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), then the remainder left to right; above, the two halves.
    /// (Round 8 checked the order against numpy 2.3.5 on 240,000 random vectors of 2..17 values: every bit equal.)
    public static func numpySum(_ a: [Float]) -> Float {
        func pairwise(_ start: Int, _ n: Int) -> Float {
            if n < 8 {
                var res: Float = -0.0
                for i in 0..<n { res += a[start + i] }
                return res
            } else if n <= 128 {
                var r = Array(a[start..<(start + 8)])
                var i = 8
                while i < n - (n % 8) {
                    for j in 0..<8 { r[j] += a[start + i + j] }
                    i += 8
                }
                var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
                while i < n {
                    res += a[start + i]
                    i += 1
                }
                return res
            }
            var n2 = n / 2
            n2 -= n2 % 8
            return pairwise(start, n2) + pairwise(start + n2, n - n2)
        }
        return Float(0) + pairwise(0, a.count)
    }

    /// host.py `softmax32`: exp(z - max(z)) / sum, every step in float32 (np.exp on float32 is libm's expf here).
    public static func softmax32(_ z: [Float]) -> [Float] {
        guard var m = z.first else { return [] }
        for v in z.dropFirst() where v > m { m = v }
        let e = z.map { expf($0 - m) }
        let s = numpySum(e)
        return e.map { $0 / s }
    }

    /// host.py `probabilities_from_logits`: the first K marker logits, / T on a text row (T as float32), the float32
    /// softmax, a noul reversed to [yes, no].
    public static func probabilities(logits: [Float], question q: Question, calibrate: Bool,
                                     config: D1Config = .publisher) -> [Float] {
        var z = Array(logits.prefix(q.options))
        if calibrate {
            let t = Float(Prompt.temperature(q, config: config))
            z = z.map { $0 / t }
        }
        let p = softmax32(z)
        return q.type == .noul ? Array(p.reversed()) : p
    }

    /// The first index of the largest value (`max(range(n), key=probs.__getitem__)`: a later equal value never wins).
    static func firstArgmax(_ p: [Double]) -> Int {
        var best = 0
        for i in 1..<p.count where p[i] > p[best] { best = i }
        return best
    }

    /// prompt.py `answer` (`probs` already in the reported order).
    public static func answer(_ q: Question, probs pf: [Float]) -> JSONValue {
        let p = pf.map(Double.init)
        if q.type == .noul {
            return .obj([("type", .string("noul")), ("noul", .double(p[0]))])
        }
        let best = firstArgmax(p)
        if q.type == .choice {
            let names = q.names
            return .obj([("type", .string("choice")), ("choice", .string(names[best])), ("confidence", .double(p[best])),
                         ("probabilities", .obj(zip(names, p).map { ($0, .double($1)) }))])
        }
        let levels = q.criteria?.array ?? []
        return .obj([("type", .string("score")),
                     ("score", .double(PythonFormat.pySum(p.enumerated().map { Double($0.offset) * $0.element }))),
                     ("confidence", .double(p[best])),
                     ("probabilities", .obj(p.enumerated().map { (String($0.offset), .double($0.element)) })),
                     ("legend", .obj(levels.enumerated().map { (String($0.offset), .string(Prompt.criterion($0.element))) }))])
    }

    /// host.py `response`: system_one's body for one request's rows (named questions) and their probabilities.
    public static func response(rows: [Row], probabilities: [[Float]]) -> JSONValue {
        .obj([("answers", .obj(zip(rows, probabilities).map { ($0.qid, answer($0.question, probs: $1)) })),
              ("usage", .obj([("input_tokens", .int(rows.reduce(0) { $0 + $1.positions })), ("output_tokens", .int(0))]))])
    }
}
