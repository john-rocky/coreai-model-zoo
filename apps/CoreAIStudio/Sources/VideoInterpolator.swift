// VideoInterpolator — SVP/HandBrake-style N-way FPS multiplier built on
// FrameInterpolationEngine: decode a source clip, RIFE-interpolate (count = multiplier - 1)
// frames between every adjacent pair, re-encode at multiplier x the source FPS, original audio
// passed through unmodified.
//
// v1 buffers the whole source clip's frames in memory before interpolating — correct and
// simple for the short clips this feature targets (HandBrake/SVP-length previews and clips, not
// hour-long footage); a streaming/windowed variant is a natural follow-up but isn't needed for
// this to work end-to-end.

import AVFoundation
import CoreGraphics
import CoreImage
import Foundation

@MainActor
final class VideoInterpolator: ObservableObject {
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

    private let engine: FrameInterpolationEngine
    private var work: Task<Void, Never>?

    init(engine: FrameInterpolationEngine) {
        self.engine = engine
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

    func run(sourceURL: URL, multiplier: Int, outputURL: URL) {
        guard engine.isReady, multiplier >= 2 else { return }
        work?.cancel()
        status = .working(progress: 0)
        work = Task {
            do {
                try await self.process(sourceURL: sourceURL, multiplier: multiplier, outputURL: outputURL) { p in
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

    private func process(sourceURL: URL, multiplier: Int, outputURL: URL, progress: @escaping (Double) -> Void) async throws {
        let asset = AVURLAsset(url: sourceURL)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw Self.err("no video track")
        }
        let naturalSize = try await videoTrack.load(.naturalSize)
        let nominalFPS = Double(try await videoTrack.load(.nominalFrameRate))
        let w = Int(naturalSize.width), h = Int(naturalSize.height)

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
            // on this SDK, or writer.add(_:) throws NSInvalidArgumentException at runtime.
            audioFormatHint = try await audioTrack.load(.formatDescriptions).first
        }

        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let videoSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
        writerInput.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
            ])
        writer.add(writerInput)

        var audioWriterInput: AVAssetWriterInput?
        if audioReaderOutput != nil {
            let input = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: audioFormatHint)
            input.expectsMediaDataInRealTime = false
            writer.add(input)
            audioWriterInput = input
        }

        guard reader.startReading() else { throw Self.err("reader failed: \(reader.error?.localizedDescription ?? "unknown")") }
        guard writer.startWriting() else { throw Self.err("writer failed: \(writer.error?.localizedDescription ?? "unknown")") }
        writer.startSession(atSourceTime: .zero)

        // Decode the source clip to CGImages up front — see file header for why this is fine
        // for v1's target clip lengths.
        let ciContext = CIContext()
        var frames: [CGImage] = []
        while let sample = videoReaderOutput.copyNextSampleBuffer() {
            try Task.checkCancellation()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            guard let cg = ciContext.createCGImage(ciImage, from: ciImage.extent) else { continue }
            frames.append(cg)
        }
        guard frames.count >= 2 else { throw Self.err("source clip has fewer than 2 frames") }

        let interiorPerGap = multiplier - 1
        let outFrameCount = (frames.count - 1) * multiplier + 1
        var written = 0
        let outFPS = nominalFPS * Double(multiplier)
        let frameDuration = CMTime(value: 1, timescale: CMTimeScale(outFPS.rounded()))
        var presentationTime = CMTime.zero

        func append(_ image: CGImage) async throws {
            while !writerInput.isReadyForMoreMediaData {
                try Task.checkCancellation()
                try await Task.sleep(nanoseconds: 5_000_000)
            }
            guard let buffer = Self.makePixelBuffer(from: image, width: w, height: h) else {
                throw Self.err("pixel buffer creation failed")
            }
            adaptor.append(buffer, withPresentationTime: presentationTime)
            presentationTime = presentationTime + frameDuration
            written += 1
            progress(Double(written) / Double(max(outFrameCount, 1)))
        }

        for i in 0..<(frames.count - 1) {
            try Task.checkCancellation()
            try await append(frames[i])
            if interiorPerGap > 0 {
                let mids = try await engine.interpolate(frames[i], frames[i + 1], count: interiorPerGap)
                for m in mids {
                    try Task.checkCancellation()
                    try await append(m)
                }
            }
        }
        if let last = frames.last {
            try await append(last)
        }
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
        NSError(domain: "VideoInterpolator", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
