// N3DSpeakerCache — the Arrival-Order Speaker Cache + FIFO of ../host_loop.py (SpeakerCache), itself a
// line-by-line copy of transformers' Nemotron3DiarizationSpeakerCache.update / _compress (@ 4b28d51):
//
//   update(rows, logits, nChunk):
//     probs = avg_pool8(sigmoid(logits[:L*8]))                          [L, 8] encoder-rate
//     fifo += the step's nChunk chunk rows (look-ahead rows are fed again next step)
//     if len(fifo) > fifoLength: pop = min(max(updatePeriod, len(fifo) - fifoLength), len(fifo))
//       fifoProbs = probs[nCache ..< nCache + len(fifo)]
//       stored = the cache's saved probs if it is compressed, else probs[..< nCache]
//       cache += fifo[..< pop] with (stored, fifoProbs[..< pop]); fifo = fifo[pop...]
//       if len(cache) > 264: compress -> 264 rows, isCompressed = true
//   compress(embeds [N, 512], probs [N, 8]):
//     scores = log(max(p, .25)) - log(max(1 - p, .25)) + Σ_k log(max(1 - p_k, .25)) - log .5
//     non-speech (p <= .5) -> -inf; a speaker with >= minPositiveScores (16) positive frames also loses
//     its non-positive speech frames (-inf)
//     scores[264...] += 0.05; per speaker the top 24 += 2 log 2, then the top 48 += log 2
//     one +inf row appended (the silence row N = silence_embeds, zero probs)
//     top 264 of the speaker-major flat scores [8 * (N + 1)]; -inf picks -> sentinel; ascending sort;
//     frame = N for the sentinel else min(idx % (N + 1), N); gather embeds / probs
//
// Bit-for-bit with the NumPy host (measured on this Mac, export venv NumPy 2.3.5): sigmoid is float32
// 1 / (1 + expf(-x)); the 8x pool sums the sub-frames left to right and divides by 8; the scores are
// float64 from the float32 probabilities with libm log, the per-frame Σ_k in NumPy's pairwise order
// ((0+1)+(2+3))+((4+5)+(6+7)); top-k ties go to the lower index (stable argsort).

import Foundation

public struct N3DSpeakerCache: Sendable {
    public static let length = 264
    public static let numSpeakers = 8
    public static let hidden = 512
    public static let subsampling = 8
    public static let silenceFramesPerSpeaker = 1
    public static let predictionScoreThreshold = 0.25
    public static let latestFramesScoreBoost = 0.05
    public static let minPositiveScoresRate = 0.5, strongBoostRate = 0.75, weakBoostRate = 1.5
    /// Top-k boundary gaps below this many float32 ulp are counted in `nearTies` (diagnostic).
    public static let nearTieULP = 4.0

    static let logHalf = log(0.5)
    static let strongBoost = -2.0 * log(0.5)
    static let weakBoost = -log(0.5)

    /// Negative controls for the self-test (host_loop.py --poison).
    public enum Poison: String, Sendable {
        /// An overflowing cache is cut to its first 264 rows instead of compressed.
        case noCompress = "no-compress"
        /// Rows leaving the FIFO are dropped instead of moved to the cache.
        case noPop = "no-pop"
    }

    public let fifoLength: Int
    public let updatePeriod: Int
    public let minPositiveScores: Int
    public let numStrongBoosted: Int
    public let numWeakBoosted: Int
    public var poison: Poison?

    /// [cacheFrames, 512] cache rows, [cacheFrames, 8] their saved probabilities, [fifoFrames, 512] FIFO rows.
    public private(set) var embeds: [Float] = []
    public private(set) var probs: [Float] = []
    public private(set) var fifo: [Float] = []
    public private(set) var isCompressed = false
    public private(set) var compressions = 0
    public private(set) var nearTies = 0

    public var cacheFrames: Int { embeds.count / Self.hidden }
    public var fifoFrames: Int { fifo.count / Self.hidden }

    public init(fifoLength: Int, updatePeriod: Int, poison: Poison? = nil) {
        self.fifoLength = fifoLength
        self.updatePeriod = updatePeriod
        self.poison = poison
        let budget = Self.length / Self.numSpeakers - Self.silenceFramesPerSpeaker      // 32 rows per speaker
        minPositiveScores = Int((Double(budget) * Self.minPositiveScoresRate).rounded(.down))   // 16
        numStrongBoosted = Int((Double(budget) * Self.strongBoostRate).rounded(.down))          // 24
        numWeakBoosted = Int((Double(budget) * Self.weakBoostRate).rounded(.down))              // 48
    }

    /// A cache resumed from a saved state (the teacher-forced self-test builds it from the Python run).
    public init(fifoLength: Int, updatePeriod: Int, embeds: [Float], probs: [Float], fifo: [Float],
                isCompressed: Bool) {
        self.init(fifoLength: fifoLength, updatePeriod: updatePeriod)
        precondition(embeds.count % Self.hidden == 0 && fifo.count % Self.hidden == 0)
        self.embeds = embeds
        self.probs = probs
        self.fifo = fifo
        self.isCompressed = isCompressed
    }

    /// [cache | FIFO] rows, the head of the next step's packed input.
    public var rows: [Float] { embeds + fifo }

    public func numPopped(_ n: Int) -> Int {
        if n <= fifoLength { return 0 }
        return min(max(updatePeriod, n - fifoLength), n)
    }

    /// One step's update. `rows` = the step's packed rows [L, 512] ([cache | FIFO | chunk]), `logits` =
    /// the graph's [L*8, 8] real rows, `chunkFrames` = the chunk's rows without look-ahead.
    public mutating func update(rows: [Float], logits: [Float], frames L: Int, silence: [Float], chunkFrames: Int) {
        let H = Self.hidden, S = Self.numSpeakers
        let nCache = cacheFrames, nFifo = fifoFrames
        let pooled = Self.poolProbs(logits: logits, frames: L)                  // [L, 8]
        let start = nCache + nFifo
        var newFifo = fifo
        newFifo.append(contentsOf: rows[(start * H)..<((start + chunkFrames) * H)])
        let n = newFifo.count / H
        let pop = numPopped(n)
        if pop > 0 {
            let fifoProbs = pooled[(nCache * S)..<((nCache + n) * S)]
            let stored = isCompressed ? probs : Array(pooled[0..<(nCache * S)])
            var cacheE: [Float], cacheP: [Float]
            if poison == .noPop {
                cacheE = embeds
                cacheP = stored
            } else {
                cacheE = embeds + newFifo[0..<(pop * H)]
                cacheP = stored + fifoProbs.prefix(pop * S)
            }
            newFifo = Array(newFifo[(pop * H)...])
            if cacheE.count / H > Self.length {
                if poison == .noCompress {
                    cacheE = Array(cacheE[0..<(Self.length * H)])
                    cacheP = Array(cacheP[0..<(Self.length * S)])
                } else {
                    (cacheE, cacheP) = compress(embeds: cacheE, probs: cacheP, silence: silence)
                }
                isCompressed = true
                compressions += 1
            }
            embeds = cacheE
            probs = cacheP
        }
        fifo = newFifo
    }

    /// avg_pool1d(sigmoid(logits), 8, 8): [L*8, 8] logits -> [L, 8] probabilities, float32.
    public static func poolProbs(logits: [Float], frames L: Int) -> [Float] {
        let S = numSpeakers, sub = subsampling
        var out = [Float](repeating: 0, count: L * S)
        for r in 0..<L {
            for s in 0..<S {
                var acc = sigmoid(logits[(r * sub) * S + s])
                for k in 1..<sub { acc = acc + sigmoid(logits[(r * sub + k) * S + s]) }
                out[r * S + s] = acc / Float(sub)
            }
        }
        return out
    }

    @inline(__always) public static func sigmoid(_ x: Float) -> Float {
        Float(1) / (Float(1) + expf(-x))
    }

    /// float64 scores [N, 8] from the float32 probabilities (host_loop.SpeakerCache.frame_scores).
    public func frameScores(probs p: [Float], frames N: Int) -> [Double] {
        let S = Self.numSpeakers
        var scores = [Double](repeating: 0, count: N * S)
        var lp = [Double](repeating: 0, count: S), lc = [Double](repeating: 0, count: S)
        for f in 0..<N {
            for s in 0..<S {
                let v = Double(p[f * S + s])
                lp[s] = log(max(v, Self.predictionScoreThreshold))
                lc[s] = log(max(1.0 - v, Self.predictionScoreThreshold))
            }
            let sumC = ((lc[0] + lc[1]) + (lc[2] + lc[3])) + ((lc[4] + lc[5]) + (lc[6] + lc[7]))
            for s in 0..<S { scores[f * S + s] = ((lp[s] - lc[s]) + sumC) - Self.logHalf }
        }
        // non-speech -> -inf
        for i in 0..<scores.count where !(p[i] > 0.5) { scores[i] = -.infinity }
        // a speaker with enough positive frames loses its non-positive speech frames
        var positives = [Int](repeating: 0, count: S)
        for f in 0..<N { for s in 0..<S where scores[f * S + s] > 0 { positives[s] += 1 } }
        for f in 0..<N {
            for s in 0..<S {
                let i = f * S + s
                let isPos = scores[i] > 0, isSpeech = p[i] > 0.5
                if !isPos && isSpeech && positives[s] >= minPositiveScores { scores[i] = -.infinity }
            }
        }
        return scores
    }

    /// Indices of the k largest values, ties to the lower index (NumPy's stable argsort of -values).
    static func topKDescending(_ values: [Double], _ k: Int) -> [Int] {
        let order = values.indices.sorted { a, b in
            let va = -values[a], vb = -values[b]
            return va < vb || (va == vb && a < b)
        }
        return Array(order.prefix(k))
    }

    /// A top-k boundary within `nearTieULP` float32 ulp (host_loop.SpeakerCache._note_tie), counted.
    mutating func noteTie(_ values: [Double], _ k: Int) {
        guard k < values.count else { return }
        let sorted = values.sorted(by: >)
        let kth = sorted[k - 1], next = sorted[k]
        guard kth.isFinite && next.isFinite else { return }
        let ulp = Double(Float(abs(next)).ulp)
        if kth - next < Self.nearTieULP * ulp { nearTies += 1 }
    }

    mutating func boost(_ scores: inout [Double], frames N: Int, count k: Int, add: Double) {
        let S = Self.numSpeakers
        for s in 0..<S {
            let column = (0..<N).map { scores[$0 * S + s] }
            noteTie(column, k)
            for f in Self.topKDescending(column, k) { scores[f * S + s] = scores[f * S + s] + add }
        }
    }

    mutating func compress(embeds e: [Float], probs p: [Float], silence: [Float]) -> ([Float], [Float]) {
        let H = Self.hidden, S = Self.numSpeakers, K = Self.length
        let N = p.count / S
        var scores = frameScores(probs: p, frames: N)
        for f in K..<N { for s in 0..<S { scores[f * S + s] = scores[f * S + s] + Self.latestFramesScoreBoost } }
        boost(&scores, frames: N, count: numStrongBoosted, add: Self.strongBoost)
        boost(&scores, frames: N, count: numWeakBoosted, add: Self.weakBoost)
        let nScored = N + Self.silenceFramesPerSpeaker
        let sentinel = nScored * S
        var flat = [Double](repeating: .infinity, count: S * nScored)          // row N = +inf
        for s in 0..<S { for f in 0..<N { flat[s * nScored + f] = scores[f * S + s] } }
        noteTie(flat, K)
        var idx = Self.topKDescending(flat, K)
        for i in idx.indices where flat[idx[i]] == -.infinity { idx[i] = sentinel }
        idx.sort()
        var outE = [Float](repeating: 0, count: K * H)
        var outP = [Float](repeating: 0, count: K * S)
        for (i, j) in idx.enumerated() {
            let frame = j == sentinel ? N : min(j % nScored, N)
            if frame == N {
                outE.replaceSubrange((i * H)..<((i + 1) * H), with: silence)
            } else {
                outE.replaceSubrange((i * H)..<((i + 1) * H), with: e[(frame * H)..<((frame + 1) * H)])
                outP.replaceSubrange((i * S)..<((i + 1) * S), with: p[(frame * S)..<((frame + 1) * S)])
            }
        }
        return (outE, outP)
    }
}
