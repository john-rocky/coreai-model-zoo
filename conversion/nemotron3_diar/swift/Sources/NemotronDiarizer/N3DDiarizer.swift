// N3DDiarizer — ../host_loop.py `run()` in Swift: audio -> per-step [cache | FIFO | chunk] rows,
// left-packed into the fixed-T graph input with `valid`, -> logits -> emitted frames + speaker-cache
// update. Same step sequence as the Python host (packing, valid, emit range, cache update order).
//
//   let assets = try N3DAssets(directory: bundleDir)
//   let diarizer = try await N3DDiarizer(assets: assets, computeUnits: .gpu,
//                                        profile: .streamingProfile(mode: .lowLatency))
//   let out = try await diarizer.process(samples: pcm16kMono)      // out.logits / out.probs [frames, 8]
//   let turns = N3DDiarizer.segments(from: out.probs)             // who spoke when, overlaps kept

import Foundation

/// The processor's streaming modes: (chunk, look-ahead) in encoder frames of 80 ms.
public enum N3DStreamingMode: String, Sendable, CaseIterable {
    case lowLatency = "ll"            // 9 + 4
    case veryLowLatency = "vll"       // 6 + 2
    case ultraLowLatency = "ull"      // 3 + 1

    public var chunkFrames: Int {
        switch self { case .lowLatency: 9; case .veryLowLatency: 6; case .ultraLowLatency: 3 }
    }
    public var lookaheadFrames: Int {
        switch self { case .lowLatency: 4; case .veryLowLatency: 2; case .ultraLowLatency: 1 }
    }
}

/// A host profile: how audio becomes steps, and the FIFO policy. Its graph length T must match the
/// loaded bundle (541 for the three streaming modes, 684 offline).
public struct N3DProfile: Sendable, Equatable {
    public enum Kind: String, Sendable { case streaming, offline }

    public let kind: Kind
    public let mode: N3DStreamingMode?
    public let chunkFrames: Int
    public let lookaheadFrames: Int
    public let fifoLength: Int
    public let updatePeriod: Int
    public let graphLength: Int

    /// Per-chunk mel from each chunk's own audio slice (the model card's inputs_generator);
    /// FIFO 264, update period 222, T = 541.
    public static func streamingProfile(mode: N3DStreamingMode) -> N3DProfile {
        N3DProfile(kind: .streaming, mode: mode, chunkFrames: mode.chunkFrames, lookaheadFrames: mode.lookaheadFrames,
                   fifoLength: 264, updatePeriod: 222, graphLength: 541)
    }

    /// Whole-recording mel (center=true, one pass), chunks of 340 rows + up to 40 look-ahead rows;
    /// FIFO 40, update period 300, T = 684; output cut to the recording's mel frame count.
    public static let offline = N3DProfile(kind: .offline, mode: nil, chunkFrames: 340, lookaheadFrames: 40,
                                           fifoLength: 40, updatePeriod: 300, graphLength: 684)
}

/// One step's chunk rows (incl. look-ahead) and how many 10 ms frames it emits.
public struct N3DStepInput: Sendable {
    public let rows: [Float]          // [rowCount, 512]
    public let lookahead: Int
    public let emitFrames: Int

    public init(rows: [Float], lookahead: Int, emitFrames: Int) {
        self.rows = rows
        self.lookahead = lookahead
        self.emitFrames = emitFrames
    }

    public var rowCount: Int { rows.count / N3DSpeakerCache.hidden }
}

/// What one step did (for self-tests and tracing). `rows` is the packed input's real part [L, 512].
public struct N3DStepInfo: Sendable {
    public let step: Int
    public let rows: [Float]
    public let length: Int
    public let cacheFrames: Int
    public let fifoFrames: Int
    public let chunkFrames: Int
    public let lookahead: Int
    public let compressedAfter: Bool
    public let cacheFramesAfter: Int
    public let fifoFramesAfter: Int
    public let graphSeconds: Double
}

public struct N3DOutput: Sendable {
    /// Emitted 10 ms frames.
    public let frames: Int
    /// [frames, 8] speaker logits.
    public let logits: [Float]
    /// [frames, 8] float32 sigmoid of the logits.
    public let probs: [Float]
    public let steps: Int
    public let compressions: Int
    /// Top-k boundary gaps under 4 float32 ulp met in the compressions (decisions inside rounding).
    public let nearTies: Int
    /// Per-step graph call time (input copy + run + output read).
    public let graphSeconds: [Double]
    /// Mel + embed time, then the whole call.
    public let frontEndSeconds: Double
    public let wallSeconds: Double
}

/// One speaker turn: `speaker` is active over frames [startFrame, endFrame) (10 ms each).
public struct N3DSegment: Sendable, Hashable {
    public let speaker: Int
    public let startFrame: Int
    public let endFrame: Int

    public init(speaker: Int, startFrame: Int, endFrame: Int) {
        self.speaker = speaker
        self.startFrame = startFrame
        self.endFrame = endFrame
    }

    public var start: Double { Double(startFrame) / 100 }
    public var end: Double { Double(endFrame) / 100 }
}

public actor N3DDiarizer {
    public static let frameSeconds = 0.01

    public nonisolated let profile: N3DProfile
    public nonisolated let computeUnits: N3DComputeUnits
    public nonisolated let modelURL: URL
    public nonisolated let loadSeconds: Double
    let graph: N3DGraph
    let mel: N3DMel
    let embedder: N3DEmbedder
    let silence: [Float]
    public private(set) var poison: N3DSpeakerCache.Poison?

    public static func streamingProfile(mode: N3DStreamingMode) -> N3DProfile { .streamingProfile(mode: mode) }
    public static var offline: N3DProfile { .offline }

    /// Loads the graph for `profile` (from `modelURL`, else the assets directory's bundle for the
    /// profile kind) and the host constants.
    public init(assets: N3DAssets, computeUnits: N3DComputeUnits = .gpu,
                profile: N3DProfile = .streamingProfile(mode: .lowLatency), modelURL: URL? = nil) async throws {
        guard let url = modelURL ?? assets.modelURL(for: profile.kind) else {
            throw N3DError.missingFile("n3d_\(profile.kind.rawValue)_float16 bundle in \(assets.directory.path)")
        }
        let graph = try await N3DGraph(contentsOf: url, computeUnits: computeUnits)
        guard graph.length == profile.graphLength else {
            throw N3DError.contract("bundle T=\(graph.length), \(profile.kind.rawValue) profile needs T=\(profile.graphLength)")
        }
        self.graph = graph
        self.profile = profile
        self.computeUnits = computeUnits
        self.modelURL = url
        loadSeconds = graph.loadSeconds
        mel = N3DMel(melFilters: assets.melFilters, hannWindow: assets.hannWindow)
        embedder = N3DEmbedder(projection: assets.projection)
        silence = assets.silence
    }

    public func setPoison(_ poison: N3DSpeakerCache.Poison?) { self.poison = poison }

    /// Diarize a whole 16 kHz mono recording. `profile` may switch between the streaming modes (same T).
    public func process(samples: [Float], profile: N3DProfile? = nil,
                        onStep: (@Sendable (N3DStepInfo) -> Void)? = nil) async throws -> N3DOutput {
        let p = profile ?? self.profile
        let t0 = ContinuousClock.now
        let (steps, nFrames) = try buildSteps(samples, p)
        let front = Self.seconds(since: t0)
        return try await run(steps: steps, frames: nFrames, profile: p, onStep: onStep, started: t0, frontEnd: front)
    }

    /// Drive the loop over precomputed chunk rows (the self-test feeds the Python host's rows here to
    /// separate the mel from the loop). `frames` = the output length the last step is cut to.
    public func process(steps: [N3DStepInput], frames: Int, profile: N3DProfile? = nil,
                        onStep: (@Sendable (N3DStepInfo) -> Void)? = nil) async throws -> N3DOutput {
        try await run(steps: steps, frames: frames, profile: profile ?? self.profile, onStep: onStep,
                      started: ContinuousClock.now, frontEnd: 0)
    }

    /// One graph call on a full packed input [T * 512] / valid [T] (timing).
    public func runGraph(packed: [Float], valid: [Float]) async throws -> [Float] {
        try await graph.run(packed: packed, valid: valid)
    }

    public nonisolated var graphLength: Int { profile.graphLength }

    // MARK: steps (host_loop.stream_steps / offline_steps / build_steps)

    func buildSteps(_ samples: [Float], _ p: N3DProfile) throws -> ([N3DStepInput], Int) {
        let sub = N3DMel.stack
        var steps: [N3DStepInput] = []
        var nFrames = 0
        switch p.kind {
        case .streaming:
            let chunks = N3DMel.streamChunks(samples: samples.count, chunkFrames: p.chunkFrames,
                                             lookaheadFrames: p.lookaheadFrames)
            for chunk in chunks {
                let end = min(chunk.end, samples.count), start = min(chunk.start, end)
                let (m, F) = mel.logMel(samples[start..<end], center: chunk.isFirst)
                let (emb, _) = embedder.embed(mel: m, frames: F)
                if chunk.isLast {
                    steps.append(N3DStepInput(rows: emb, lookahead: 0, emitFrames: F))
                    nFrames += F
                } else {
                    guard F == (p.chunkFrames + p.lookaheadFrames) * sub else {
                        throw N3DError.audioTooShort(samples: samples.count, minimum: chunks[0].end)
                    }
                    steps.append(N3DStepInput(rows: emb, lookahead: p.lookaheadFrames, emitFrames: p.chunkFrames * sub))
                    nFrames += p.chunkFrames * sub
                }
            }
        case .offline:
            let (m, F) = samples.withUnsafeBufferPointer { mel.logMel($0, center: true) }
            guard F > 0 else { throw N3DError.audioTooShort(samples: samples.count, minimum: N3DMel.hop) }
            let (emb, R) = embedder.embed(mel: m, frames: F)
            let H = N3DSpeakerCache.hidden
            var s = 0
            while s < R {
                let e = min(s + p.chunkFrames, R)
                let la = min(p.lookaheadFrames, R - e)
                steps.append(N3DStepInput(rows: Array(emb[(s * H)..<((e + la) * H)]), lookahead: la, emitFrames: (e - s) * sub))
                s += p.chunkFrames
            }
            nFrames = F
        }
        return (steps, nFrames)
    }

    // MARK: the loop (host_loop.run)

    func run(steps: [N3DStepInput], frames nFrames: Int, profile p: N3DProfile,
             onStep: (@Sendable (N3DStepInfo) -> Void)?, started t0: ContinuousClock.Instant,
             frontEnd: Double) async throws -> N3DOutput {
        guard p.graphLength == graph.length else {
            throw N3DError.contract("bundle T=\(graph.length), profile needs T=\(p.graphLength)")
        }
        let T = graph.length, H = N3DSpeakerCache.hidden, S = N3DSpeakerCache.numSpeakers
        let sub = N3DSpeakerCache.subsampling
        var cache = N3DSpeakerCache(fifoLength: p.fifoLength, updatePeriod: p.updatePeriod, poison: poison)
        var packed = [Float](repeating: 0, count: T * H)
        var valid = [Float](repeating: 0, count: T)
        var out: [Float] = []
        out.reserveCapacity(nFrames * S)
        var graphTimes: [Double] = []
        for (i, step) in steps.enumerated() {
            let nCache = cache.cacheFrames, nFifo = cache.fifoFrames
            let rows = cache.rows + step.rows
            let L = rows.count / H
            guard L <= T else { throw N3DError.contract("step \(i): \(L) rows > T=\(T)") }
            packed.replaceSubrange(0..<rows.count, with: rows)
            for j in rows.count..<packed.count { packed[j] = 0 }
            for j in 0..<T { valid[j] = j < L ? 1 : 0 }
            let g0 = ContinuousClock.now
            let full = try await graph.run(packed: packed, valid: valid)
            let dt = Self.seconds(since: g0)
            graphTimes.append(dt)
            let logits = Array(full[0..<(L * sub * S)])
            let nChunk = step.rowCount - step.lookahead
            cache.update(rows: rows, logits: logits, frames: L, silence: silence, chunkFrames: nChunk)
            let s = (nCache + nFifo) * sub
            let e = s + min(nChunk * sub, step.emitFrames)
            out.append(contentsOf: logits[(s * S)..<(e * S)])
            onStep?(N3DStepInfo(step: i, rows: rows, length: L, cacheFrames: nCache, fifoFrames: nFifo,
                                chunkFrames: nChunk, lookahead: step.lookahead, compressedAfter: cache.isCompressed,
                                cacheFramesAfter: cache.cacheFrames, fifoFramesAfter: cache.fifoFrames, graphSeconds: dt))
        }
        if out.count > nFrames * S { out.removeLast(out.count - nFrames * S) }
        let probs = out.map { N3DSpeakerCache.sigmoid($0) }
        return N3DOutput(frames: out.count / S, logits: out, probs: probs, steps: steps.count,
                         compressions: cache.compressions, nearTies: cache.nearTies, graphSeconds: graphTimes,
                         frontEndSeconds: frontEnd, wallSeconds: Self.seconds(since: t0))
    }

    static func seconds(since t: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - t
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    // MARK: segmentation (transformers Nemotron3DiarizationProcessor.extract_speaker_dict)

    /// Per speaker, every run of frames with probability > threshold is a segment [start, end); segments
    /// are sorted by (start, speaker) and overlapping speech stays overlapping.
    public static func segments(from probs: [Float], threshold: Float = 0.5) -> [N3DSegment] {
        let S = N3DSpeakerCache.numSpeakers
        let n = probs.count / S
        var segs: [N3DSegment] = []
        for spk in 0..<S {
            var f = 0
            while f < n {
                guard probs[f * S + spk] > threshold else { f += 1; continue }
                let a = f
                while f < n && probs[f * S + spk] > threshold { f += 1 }
                segs.append(N3DSegment(speaker: spk, startFrame: a, endFrame: f))
            }
        }
        segs.sort { ($0.startFrame, $0.speaker) < ($1.startFrame, $1.speaker) }
        return segs
    }
}
