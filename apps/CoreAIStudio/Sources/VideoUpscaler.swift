// VideoUpscaler — AVFoundation frame-by-frame video super-resolution pipeline. Mirrors
// VideoInterpolator's AVAssetReader/AVAssetWriter/audio-passthrough/progress/cancellation
// structure (see that file), but is deliberately model-agnostic: it depends on the
// `FrameUpscaler` protocol below, not on UpscaleEngine (AdcSR) concretely, so a future
// PiperSR-based conformer can be swapped in at the call site without touching this file again.
//
// Divergence from VideoInterpolator: RIFE needs frame PAIRS (buffer-the-whole-clip is the
// simplest correct thing there), but upscaling is a per-frame, embarrassingly-independent
// operation — each decoded frame is upscaled and written immediately, one at a time, with no
// need to hold the whole clip in memory first. The output frame size also isn't known upfront
// (a FrameUpscaler's scale factor is opaque to this file by design), so the writer/adaptor are
// configured lazily, once the first frame's upscaled size is known.

import AVFoundation
import CoreGraphics
import CoreImage
import Foundation

/// Model-agnostic per-frame upscaler. UpscaleEngine (AdcSR) conforms to this in
/// UpscaleEngine.swift; a future PiperSR engine can conform the same way.
protocol FrameUpscaler {
    func upscale(_ image: CGImage) async throws -> CGImage
}

@MainActor
final class VideoUpscaler: ObservableObject {
    struct SourceInfo: Equatable {
        let width: Int
        let height: Int
        let fps: Double
        let duration: Double
        let frameCount: Int
    }

    enum Status: Equatable {
        case idle
        case analyzing
        case working(progress: Double)
        case done(URL)
        case error(String)

        var isBusy: Bool {
            switch self {
            case .analyzing, .working: return true
            default: return false
            }
        }
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var sourceInfo: SourceInfo?

    private let upscaler: FrameUpscaler
    private var work: Task<Void, Never>?

    init(upscaler: FrameUpscaler) {
        self.upscaler = upscaler
    }

    func loadSource(_ url: URL) {
        work?.cancel()
        status = .analyzing
        work = Task {
            do {
                sourceInfo = try await Self.analyze(url: url)
                status = .idle
            } catch is CancellationError {
            } catch {
                status = .error("\(error)")
            }
        }
    }

    func run(sourceURL: URL, outputURL: URL) {
        work?.cancel()
        status = .working(progress: 0)
        work = Task {
            do {
                try await self.process(sourceURL: sourceURL, outputURL: outputURL) { p in
                    self.status = .working(progress: p)
                }
                try Task.checkCancellation()
                status = .done(outputURL)
            } catch is CancellationError {
            } catch {
                status = .error("\(error)")
            }
        }
    }

    func cancel() {
        work?.cancel()
        if status.isBusy { status = .idle }
    }

    // MARK: - Analysis

    private static func analyze(url: URL) async throws -> SourceInfo {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw err("no video track")
        }
        let size = try await track.load(.naturalSize)
        let fps = try await track.load(.nominalFrameRate)
        let duration = try await asset.load(.duration)
        let seconds = CMTimeGetSeconds(duration)
        let frameCount = Int((seconds * Double(fps)).rounded())
        return SourceInfo(
            width: Int(size.width), height: Int(size.height),
            fps: Double(fps), duration: seconds, frameCount: frameCount)
    }

    // MARK: - Pipeline

    private func process(sourceURL: URL, outputURL: URL, progress: @escaping (Double) -> Void) async throws {
        let asset = AVURLAsset(url: sourceURL)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw Self.err("no video track")
        }
        let nominalFPS = Double(try await videoTrack.load(.nominalFrameRate))
        let duration = try await asset.load(.duration)
        let seconds = CMTimeGetSeconds(duration)
        let estimatedFrameCount = max(1, Int((seconds * nominalFPS).rounded()))

        if FileManager.default.fileExists(atPath: outputURL.path) {
            try FileManager.default.removeItem(at: outputURL)
        }

        let reader = try AVAssetReader(asset: asset)
        let videoOutputSettings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        let videoReaderOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: videoOutputSettings)
        videoReaderOutput.alwaysCopiesSampleData = false
        reader.add(videoReaderOutput)

        let audioTrack = try await asset.loadTracks(withMediaType: .audio).first
        var audioReaderOutput: AVAssetReaderTrackOutput?
        var audioFormatHint: CMFormatDescription?
        if let audioTrack {
            let out = AVAssetReaderTrackOutput(track: audioTrack, outputSettings: nil)
            reader.add(out)
            audioReaderOutput = out
            // A nil-outputSettings (passthrough) AVAssetWriterInput needs a source format hint
            // on this SDK, or writer.add(_:) throws NSInvalidArgumentException at runtime
            // (✓ VERIFIED: reproduced via the e2e harness before this fix was added).
            audioFormatHint = try await audioTrack.load(.formatDescriptions).first
        }

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)

        guard reader.startReading() else { throw Self.err("reader failed: \(reader.error?.localizedDescription ?? "unknown")") }

        // Output frame size is unknown until the first upscale (the FrameUpscaler's scale
        // factor is opaque here by design) — the writer/adaptor/writer.startWriting() are all
        // set up lazily, once we've upscaled frame 0.
        var writerInput: AVAssetWriterInput?
        var adaptor: AVAssetWriterInputPixelBufferAdaptor?
        var audioWriterInput: AVAssetWriterInput?
        var writerStarted = false
        var outW = 0, outH = 0

        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(nominalFPS.rounded()))
        var presentationTime = CMTime.zero
        var written = 0

        let ciContext = CIContext()

        func startWriterIfNeeded(sampleWidth: Int, sampleHeight: Int) throws {
            guard !writerStarted else { return }
            outW = sampleWidth
            outH = sampleHeight

            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: outW,
                AVVideoHeightKey: outH,
            ]
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            input.expectsMediaDataInRealTime = false
            let pba = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: [
                    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                    kCVPixelBufferWidthKey as String: outW,
                    kCVPixelBufferHeightKey as String: outH,
                ])
            writer.add(input)
            writerInput = input
            adaptor = pba

            if audioReaderOutput != nil {
                let aInput = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: audioFormatHint)
                aInput.expectsMediaDataInRealTime = false
                writer.add(aInput)
                audioWriterInput = aInput
            }

            guard writer.startWriting() else { throw Self.err("writer failed: \(writer.error?.localizedDescription ?? "unknown")") }
            writer.startSession(atSourceTime: .zero)
            writerStarted = true
        }

        func append(_ image: CGImage) async throws {
            guard let writerInput, let adaptor else { throw Self.err("writer not started") }
            while !writerInput.isReadyForMoreMediaData {
                try Task.checkCancellation()
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            guard let buffer = Self.makePixelBuffer(from: image, width: outW, height: outH) else {
                throw Self.err("pixel buffer creation failed")
            }
            adaptor.append(buffer, withPresentationTime: presentationTime)
            presentationTime = presentationTime + frameDuration
            written += 1
            progress(Double(written) / Double(estimatedFrameCount))
        }

        // Decode -> upscale -> write, one frame at a time (see file header for why this
        // diverges from VideoInterpolator's buffer-the-whole-clip approach).
        while let sample = videoReaderOutput.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cg = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            let upscaled = try await upscaler.upscale(cg)
            try Task.checkCancellation()
            try startWriterIfNeeded(sampleWidth: upscaled.width, sampleHeight: upscaled.height)
            try await append(upscaled)
        }
        guard writerStarted, let writerInput else { throw Self.err("source clip has no frames") }
        writerInput.markAsFinished()

        if let audioReaderOutput, let audioWriterInput {
            while let sample = audioReaderOutput.copyNextSampleBuffer() {
                try Task.checkCancellation()
                while !audioWriterInput.isReadyForMoreMediaData {
                    try await Task.sleep(nanoseconds: 5_000_000)
                }
                audioWriterInput.append(sample)
            }
            audioWriterInput.markAsFinished()
        }

        await writer.finishWriting()
        if writer.status == .failed {
            throw Self.err("writer failed: \(writer.error?.localizedDescription ?? "unknown")")
        }
    }

    nonisolated private static func makePixelBuffer(from image: CGImage, width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
        ]
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return buffer
    }

    private static func err(_ msg: String) -> NSError {
        NSError(domain: "VideoUpscaler", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
