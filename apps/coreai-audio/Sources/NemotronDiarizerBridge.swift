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

    /// Loads the staged streaming graph on the GPU (the configuration the Mac and iPhone gates passed).
    static func load() async throws -> NemotronDiarizerBridge {
        guard let assets = staged else {
            throw NemotronDiarizeError(message: "Nemotron-3 diarizer not staged at \(location.path)")
        }
        do {
            let diarizer = try await N3DDiarizer(assets: assets, computeUnits: .gpu,
                                                 profile: .streamingProfile(mode: mode))
            return NemotronDiarizerBridge(diarizer: diarizer)
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

/// A package error with a readable message for the Transcribe status line (N3DError is not a LocalizedError).
struct NemotronDiarizeError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}
