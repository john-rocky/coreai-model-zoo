// N3DGateSupport — what the Mac self-test (n3d-selftest) and the iPhone gate app (apps/N3DGate) share:
// WAV input, float files, JSON, the comparison metrics of ../../gate_closed_loop.py (compare_output /
// segment_diff), timing statistics. One copy, so a number from the phone means what the same number
// from the Mac means.

import Darwin
import Foundation
import NemotronDiarizer

public enum SelfTestError: Error, CustomStringConvertible {
    case usage(String)
    case wav(String)
    case golden(String)

    public var description: String {
        switch self {
        case .usage(let s): return "usage: \(s)"
        case .wav(let s): return "wav: \(s)"
        case .golden(let s): return "golden: \(s)"
        }
    }
}

/// 16 kHz mono PCM16 or float32 WAV -> [Float] (PCM16 / 32768, like soundfile's float32 read).
public func loadWav16kMono(_ url: URL) throws -> [Float] {
    let data = try Data(contentsOf: url)
    func u32(_ o: Int) -> UInt32 { data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self)) } }
    func u16(_ o: Int) -> UInt16 { data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt16.self)) } }
    guard data.count >= 12, String(decoding: data[0..<4], as: UTF8.self) == "RIFF",
          String(decoding: data[8..<12], as: UTF8.self) == "WAVE" else { throw SelfTestError.wav("not RIFF/WAVE") }
    var off = 12
    var format = 0, channels = 0, rate = 0, bits = 0
    while off + 8 <= data.count {
        let id = String(decoding: data[off..<(off + 4)], as: UTF8.self)
        let size = Int(u32(off + 4))
        let body = off + 8
        if id == "fmt " {
            format = Int(u16(body)); channels = Int(u16(body + 2)); rate = Int(u32(body + 4)); bits = Int(u16(body + 14))
            if format == 0xFFFE && size >= 26 { format = Int(u16(body + 24)) }       // WAVE_FORMAT_EXTENSIBLE
        } else if id == "data" {
            guard channels == 1, rate == N3DMel.sampleRate else {
                throw SelfTestError.wav("need 16 kHz mono, got \(rate) Hz x \(channels)")
            }
            let n = min(size, data.count - body)
            if format == 1 && bits == 16 {
                return data.withUnsafeBytes { raw in
                    (0..<(n / 2)).map { Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: body + 2 * $0, as: Int16.self))) / 32768 }
                }
            }
            if format == 3 && bits == 32 {
                return data.withUnsafeBytes { raw in
                    (0..<(n / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: body + 4 * $0, as: UInt32.self))) }
                }
            }
            throw SelfTestError.wav("unsupported format \(format) / \(bits) bit")
        }
        off = body + size + (size & 1)
    }
    throw SelfTestError.wav("no data chunk")
}

public func readF32(_ url: URL) throws -> [Float] {
    guard let d = try? Data(contentsOf: url) else { throw SelfTestError.golden("missing \(url.path)") }
    return N3DAssets.readF32LE(d)
}

public func readF64(_ url: URL) throws -> [Double] {
    guard let d = try? Data(contentsOf: url) else { throw SelfTestError.golden("missing \(url.path)") }
    return d.withUnsafeBytes { raw in
        (0..<(d.count / 8)).map { Double(bitPattern: UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 8, as: UInt64.self))) }
    }
}

public func writeF32(_ values: [Float], to url: URL) throws {
    var d = Data(capacity: values.count * 4)
    for v in values { withUnsafeBytes(of: v.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
    try d.write(to: url)
}

public func readJSON(_ url: URL) throws -> Any {
    guard let d = try? Data(contentsOf: url) else { throw SelfTestError.golden("missing \(url.path)") }
    return try JSONSerialization.jsonObject(with: d)
}

public func maxAbsDiff(_ a: ArraySlice<Float>, _ b: ArraySlice<Float>) -> (max: Double, unequal: Int) {
    var m = 0.0, u = 0
    for (x, y) in zip(a, b) {
        if x.bitPattern != y.bitPattern { u += 1 }
        m = max(m, abs(Double(x) - Double(y)))
    }
    return (m, u)
}

@inline(__always) public func sigmoid64(_ x: Float) -> Double { 1.0 / (1.0 + exp(-Double(x))) }

/// gate_closed_loop.segment_diff: pair segments speaker by speaker by overlap; one-to-one overlaps are
/// matched (boundary shifts in frames), anything else (split, merge, no counterpart) is structural.
public struct SegmentDiff: Sendable {
    public var nRef = 0, nOurs = 0, matched = 0, maxStartShift = 0, maxEndShift = 0, shifted = 0, structural = 0
    public var cases: [String] = []

    public init(ours: [N3DSegment], ref: [N3DSegment]) {
        nRef = ref.count
        nOurs = ours.count
        let speakers = Set(ours.map(\.speaker) + ref.map(\.speaker)).sorted()
        func overlaps(_ a: N3DSegment, _ b: N3DSegment) -> Bool { min(a.endFrame, b.endFrame) > max(a.startFrame, b.startFrame) }
        for spk in speakers {
            let r = ref.filter { $0.speaker == spk }, o = ours.filter { $0.speaker == spk }
            var used = Set<Int>()
            for rs in r {
                let cand = o.indices.filter { overlaps(rs, o[$0]) }
                if cand.count == 1 && r.filter({ overlaps(o[cand[0]], $0) }).count == 1 {
                    let os = o[cand[0]]
                    used.insert(cand[0])
                    let ds = os.startFrame - rs.startFrame, de = os.endFrame - rs.endFrame
                    matched += 1
                    maxStartShift = max(maxStartShift, abs(ds))
                    maxEndShift = max(maxEndShift, abs(de))
                    if ds != 0 || de != 0 { shifted += 1 }
                } else {
                    structural += 1
                    used.formUnion(cand)
                    cases.append("spk\(spk) ref [\(rs.startFrame),\(rs.endFrame)) ours \(cand.map { "[\(o[$0].startFrame),\(o[$0].endFrame))" })")
                }
            }
            for j in o.indices where !used.contains(j) {
                structural += 1
                cases.append("spk\(spk) ref none ours [\(o[j].startFrame),\(o[j].endFrame))")
            }
        }
    }

    public var json: [String: Any] {
        ["n_ref": nRef, "n_ours": nOurs, "matched": matched, "max_start_shift": maxStartShift,
         "max_end_shift": maxEndShift, "shifted": shifted, "structural": structural, "structural_cases": cases]
    }
}

/// gate_closed_loop.compare_output against the golden probabilities (and logits when given).
public struct Comparison: Sendable {
    public let compared: Int, oursFrames: Int, refFrames: Int, tail: Int
    public let agreement: Double, disagree: Int, elements: Int
    public let maxAbsP: Double
    public let maxAbsLogit: Double?
    public let segments: SegmentDiff

    public init(logits: [Float], probs: [Float], refProbs: [Float], refLogits: [Float]?, tail: Int) {
        let S = N3DSpeakerCache.numSpeakers
        oursFrames = logits.count / S
        refFrames = refProbs.count / S
        self.tail = tail
        let n = max(0, min(oursFrames, refFrames) - tail)
        compared = n
        var same = 0, dp = 0.0, dl = 0.0
        for i in 0..<(n * S) {
            let pa = sigmoid64(logits[i]), pb = Double(refProbs[i])
            if (pa > 0.5) == (pb > 0.5) { same += 1 }
            dp = max(dp, abs(pa - pb))
            if let rl = refLogits { dl = max(dl, abs(Double(logits[i]) - Double(rl[i]))) }
        }
        elements = n * S
        disagree = elements - same
        agreement = elements == 0 ? 0 : Double(same) / Double(elements)
        maxAbsP = dp
        maxAbsLogit = refLogits == nil ? nil : dl
        segments = SegmentDiff(ours: N3DDiarizer.segments(from: Array(probs[0..<(n * S)])),
                               ref: N3DDiarizer.segments(from: Array(refProbs[0..<(n * S)])))
    }

    public var json: [String: Any] {
        var j: [String: Any] = ["frames_ours": oursFrames, "frames_ref": refFrames, "compared": compared,
                                "tail_excluded": tail, "agreement": agreement, "disagree": disagree,
                                "elements": elements, "max_abs_p": maxAbsP, "segments": segments.json]
        if let l = maxAbsLogit { j["max_abs_logit"] = l }
        return j
    }
}

/// How many graph steps `N3DDiarizer.process(samples:)` takes for a recording of `samples` samples.
public func stepCount(samples: Int, profile: N3DProfile) -> Int {
    switch profile.kind {
    case .streaming:
        return N3DMel.streamChunks(samples: samples, chunkFrames: profile.chunkFrames, lookaheadFrames: profile.lookaheadFrames).count
    case .offline:
        let rows = (N3DMel.validFrames(samples: samples, center: true) + 7) / 8
        return (rows + profile.chunkFrames - 1) / profile.chunkFrames
    }
}

public func percentile(_ xs: [Double], _ q: Double) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    let pos = q * Double(s.count - 1)
    let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))           // numpy.percentile (linear)
}

public func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

public func fmt(_ x: Double, _ digits: Int = 3) -> String { String(format: "%.\(digits)e", x) }
public func pct(_ x: Double) -> String { String(format: "%.4f %%", x * 100) }

/// Collects values from the loop's @Sendable step callback (called synchronously inside the actor).
public final class Box<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: T
    public init(_ v: T) { value = v }
    public func mutate(_ f: (inout T) -> Void) { lock.lock(); f(&value); lock.unlock() }
    public var get: T { lock.lock(); defer { lock.unlock() }; return value }
}
