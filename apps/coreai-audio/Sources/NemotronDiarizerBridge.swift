// NemotronDiarizerBridge — the Transcribe tab's 8-speaker diarizer: NVIDIA Nemotron-3-Diarization (streaming
// Sortformer, OpenMDW-1.1) on Core AI through the NemotronDiarizer package (conversion/nemotron3_diar/swift:
// Swift log-mel -> streaming loop -> speaker cache, gated on the Mac and iPhone GPU). This layer only finds
// the staged bundle, fixes the streaming mode (low latency: 9 + 4 encoder frames, 0.72 s chunks) and turns the
// per-frame probabilities into the app's SpeakerSegment turns with the 4-speaker Sortformer path's rule, in
// 10 ms frames. The package's own `N3DDiarizer.segments(from:)` (every speaker's runs, overlaps kept) is left
// as it is; the ASR needs one speaker per slice.
//
// Assets = an N3DAssets directory (the four .f32le host constants, metadata.json, the graph bundles):
//   macOS  Sources/N3DAssets                         dev symlink -> conversion/nemotron3_diar/_work/ship
//   iOS    Library/Application Support/N3DAssets/    sideload; the AOT n3d_streaming_float16.h18p.aimodelc
import Foundation
import NemotronDiarizer

struct NemotronDiarizerBridge: Sendable {
    static let speakers = N3DSpeakerCache.numSpeakers        // 8
    static let frameSec = N3DDiarizer.frameSeconds           // 0.01
    static let mode = N3DStreamingMode.lowLatency
    // The turn rule, in one place: a frame goes to its most probable speaker when that probability is above
    // `activityThreshold` (overlapped speech goes to the stronger voice); same-speaker frames in a row form a
    // turn, and one speaker's turns at most `bridgeFrames` apart merge. 48 frames of 10 ms = 0.48 s = the
    // Sortformer path's 6 frames of 80 ms.
    static let activityThreshold: Float = 0.5
    static let bridgeFrames = 48

    /// The first streaming chunk's length (16,680 samples = 1.04 s in low latency); a shorter clip is padded.
    static let minimumSamples = N3DMel.streamChunks(samples: 0, chunkFrames: mode.chunkFrames,
                                                     lookaheadFrames: mode.lookaheadFrames)[0].end

    static var location: URL {
        #if os(macOS)
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("N3DAssets")
        #else
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("N3DAssets")
        #endif
    }

    /// The staged assets, read once per process. Nil when the directory is missing, a host constant does not
    /// load, or there is no streaming graph for this platform; the Transcribe tab then uses the Sortformer.
    static let staged: N3DAssets? = {
        guard let assets = try? N3DAssets(directory: location), assets.modelURL(for: .streaming) != nil else {
            return nil
        }
        return assets
    }()

    let diarizer: N3DDiarizer
    // The host front end and the silence row again, for NemotronDiarizerStream (the package keeps its own private).
    let mel: N3DMel
    let embedder: N3DEmbedder
    let silence: [Float]

    /// Loads the staged streaming graph on the GPU (the configuration the Mac and iPhone gates passed).
    static func load() async throws -> NemotronDiarizerBridge {
        guard let assets = staged else {
            throw NemotronDiarizeError(message: "Nemotron-3 diarizer not staged at \(location.path)")
        }
        do {
            let diarizer = try await N3DDiarizer(assets: assets, computeUnits: .gpu,
                                                 profile: .streamingProfile(mode: mode))
            return NemotronDiarizerBridge(diarizer: diarizer,
                                          mel: N3DMel(melFilters: assets.melFilters, hannWindow: assets.hannWindow),
                                          embedder: N3DEmbedder(projection: assets.projection), silence: assets.silence)
        } catch let error as N3DError {
            throw NemotronDiarizeError(message: "Nemotron-3 diarizer: \(error)")
        }
    }

    var modelURL: URL { diarizer.modelURL }
    var loadSeconds: Double { diarizer.loadSeconds }

    /// 16 kHz mono -> the app's turns and the model output they come from. A clip shorter than the first
    /// chunk is padded with silence, and its turns stop at the clip's last whole 10 ms frame.
    func diarize(_ samples: [Float]) async throws -> (turns: [SpeakerSegment], output: N3DOutput) {
        var input = samples
        if input.count < Self.minimumSamples {
            input.append(contentsOf: repeatElement(0, count: Self.minimumSamples - input.count))
        }
        let output: N3DOutput
        do {
            output = try await diarizer.process(samples: input)
        } catch let error as N3DError {
            throw NemotronDiarizeError(message: "Nemotron-3 diarizer: \(error)")
        }
        let frames = input.count == samples.count
            ? output.frames : min(output.frames, samples.count / N3DMel.hop)
        return (Self.turns(from: output.probs, frames: frames), output)
    }

    /// A `diarize` output's probabilities cut to the clip ([frames, 8]; a padded clip's frames stop at its end).
    static func clipProbs(_ output: N3DOutput, samples: Int) -> (probs: [Float], frames: Int) {
        let frames = min(output.frames, samples / N3DMel.hop)
        return (Array(output.probs[0..<(frames * speakers)]), frames)
    }

    /// Frame-major probabilities [frames, 8] -> turns (the rule above).
    static func turns(from probs: [Float], frames: Int) -> [SpeakerSegment] {
        var label = [Int](repeating: -1, count: frames)            // -1 = nobody above the threshold
        for f in 0..<frames {
            var best = -1, bestP = activityThreshold
            for s in 0..<speakers where probs[f * speakers + s] > bestP {
                best = s
                bestP = probs[f * speakers + s]
            }
            label[f] = best
        }
        var runs: [SpeakerSegment] = []
        var i = 0
        while i < frames {
            let s = label[i]
            if s < 0 { i += 1; continue }
            var j = i + 1
            while j < frames && label[j] == s { j += 1 }
            runs.append(SpeakerSegment(speaker: s, startFrame: i, endFrame: j, frameSec: frameSec))
            i = j
        }
        var turns: [SpeakerSegment] = []
        for run in runs {
            if let last = turns.last, last.speaker == run.speaker, run.startFrame - last.endFrame <= bridgeFrames {
                turns[turns.count - 1] = SpeakerSegment(speaker: run.speaker, startFrame: last.startFrame,
                                                        endFrame: run.endFrame, frameSec: frameSec)
            } else {
                turns.append(run)
            }
        }
        return turns
    }
}

/// The low-latency loop one chunk at a time, for the Transcribe tab's playback-synced run: chunk k goes through the
/// graph once its audio, look-ahead included, has played (`readsUpTo(step:)`). The chunks, mel, packed rows and
/// speaker-cache update are those of `N3DDiarizer.process(samples:)` (DIARIZE_SELFTEST checks that the logits are
/// bit-equal); the graph call is the package's `runGraph`. An actor, so the host work stays off the main thread.
actor NemotronDiarizerStream {
    /// One chunk's emitted frames, cut to the clip.
    struct Step: Sendable {
        let firstFrame: Int
        let frames: Int
        let probs: [Float]              // [frames, 8]
        let graphSeconds: Double
        let hostSeconds: Double         // mel + embed + packing + cache update
    }

    nonisolated let chunks: [N3DChunk]
    /// The clip's own samples and 10 ms frames (a clip shorter than the first chunk is padded like `diarize`).
    nonisolated let clipSamples: Int
    nonisolated let clipFrames: Int

    private let bridge: NemotronDiarizerBridge
    private let samples: [Float]
    private let profile: N3DProfile
    private var cache: N3DSpeakerCache
    private var packed: [Float]
    private var valid: [Float]
    private var next = 0
    /// Every emitted frame's logits so far, padding included [frames, 8] (what `process` returns).
    private(set) var logits: [Float] = []

    init(bridge: NemotronDiarizerBridge, samples: [Float]) {
        var input = samples
        if input.count < NemotronDiarizerBridge.minimumSamples {
            input.append(contentsOf: repeatElement(0, count: NemotronDiarizerBridge.minimumSamples - input.count))
        }
        let p = bridge.diarizer.profile
        let chunks = N3DMel.streamChunks(samples: input.count, chunkFrames: p.chunkFrames, lookaheadFrames: p.lookaheadFrames)
        // what the loop emits: chunk * 8 frames per chunk, the last chunk's own mel frames
        let last = chunks[chunks.count - 1]
        let lastFrames = max(0, N3DMel.validFrames(samples: last.end - min(last.start, last.end), center: last.isFirst))
        let emitted = (chunks.count - 1) * p.chunkFrames * N3DMel.stack + lastFrames
        self.bridge = bridge
        self.samples = input
        self.profile = p
        self.chunks = chunks
        clipSamples = samples.count
        clipFrames = input.count == samples.count ? emitted : min(emitted, samples.count / N3DMel.hop)
        cache = N3DSpeakerCache(fifoLength: p.fifoLength, updatePeriod: p.updatePeriod)
        packed = [Float](repeating: 0, count: p.graphLength * N3DSpeakerCache.hidden)
        valid = [Float](repeating: 0, count: p.graphLength)
    }

    /// The clip sample chunk `k` reads up to: it can run once playback has passed it.
    nonisolated func readsUpTo(step k: Int) -> Int { min(chunks[k].end, clipSamples) }

    /// Runs the next chunk (`host_loop.run`'s step body).
    func step() async throws -> Step {
        precondition(next < chunks.count, "every chunk has run")
        let t0 = ContinuousClock.now
        let chunk = chunks[next]
        let H = N3DSpeakerCache.hidden, S = N3DSpeakerCache.numSpeakers, sub = N3DSpeakerCache.subsampling
        let end = min(chunk.end, samples.count), start = min(chunk.start, end)
        let (m, F) = bridge.mel.logMel(samples[start..<end], center: chunk.isFirst)
        let (emb, _) = bridge.embedder.embed(mel: m, frames: F)
        let lookahead = chunk.isLast ? 0 : profile.lookaheadFrames
        let emitFrames = chunk.isLast ? F : profile.chunkFrames * N3DMel.stack
        guard chunk.isLast || F == (profile.chunkFrames + profile.lookaheadFrames) * N3DMel.stack else {
            throw NemotronDiarizeError(message: "Nemotron-3 diarizer: chunk \(next) has \(F) mel frames")
        }
        let nCache = cache.cacheFrames, nFifo = cache.fifoFrames
        let rows = cache.rows + emb
        let L = rows.count / H
        guard L <= profile.graphLength else {
            throw NemotronDiarizeError(message: "Nemotron-3 diarizer: step \(next): \(L) rows > T=\(profile.graphLength)")
        }
        packed.replaceSubrange(0..<rows.count, with: rows)
        for j in rows.count..<packed.count { packed[j] = 0 }
        for j in 0..<profile.graphLength { valid[j] = j < L ? 1 : 0 }
        var host = Self.seconds(since: t0)

        let g0 = ContinuousClock.now
        let full: [Float]
        do {
            full = try await bridge.diarizer.runGraph(packed: packed, valid: valid)
        } catch let error as N3DError {
            throw NemotronDiarizeError(message: "Nemotron-3 diarizer: \(error)")
        }
        let graph = Self.seconds(since: g0)

        let h0 = ContinuousClock.now
        let stepLogits = Array(full[0..<(L * sub * S)])
        let nChunk = emb.count / H - lookahead
        cache.update(rows: rows, logits: stepLogits, frames: L, silence: bridge.silence, chunkFrames: nChunk)
        let s = (nCache + nFifo) * sub
        let e = s + min(nChunk * sub, emitFrames)
        let first = logits.count / S
        logits.append(contentsOf: stepLogits[(s * S)..<(e * S)])
        next += 1
        let kept = max(0, min(first + e - s, clipFrames) - first)
        let probs = stepLogits[(s * S)..<((s + kept) * S)].map { N3DSpeakerCache.sigmoid($0) }
        host += Self.seconds(since: h0)
        return Step(firstFrame: first, frames: kept, probs: probs, graphSeconds: graph, hostSeconds: host)
    }

    private static func seconds(since t: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - t
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}

/// A package error with a readable message for the Transcribe status line (N3DError is not a LocalizedError).
struct NemotronDiarizeError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}
