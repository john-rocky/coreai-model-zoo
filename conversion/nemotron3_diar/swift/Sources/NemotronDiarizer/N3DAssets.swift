// N3DAssets — the host-side files of a Nemotron-3-Diarization bundle directory (../export_n3d.py):
// four raw little-endian float32 constants (C order) and metadata.json, next to the graph bundles
// (`n3d_<profile>_float16.aimodel`, and on iPhone the AOT `n3d_<profile>_float16.h18p.aimodelc`).
// The sha256 values in metadata.json are informational; nothing here checks them.

import Foundation

public enum N3DError: Error, CustomStringConvertible, Sendable {
    case missingFile(String)
    case badSize(file: String, expectedFloats: Int, gotBytes: Int)
    case functionNotFound(String)
    case contract(String)
    case iosBundleOnMac(String)
    case audioTooShort(samples: Int, minimum: Int)
    case graphOutput(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file \(p)"
        case .badSize(let f, let n, let b): return "\(f): expected \(n) float32 (\(n * 4) bytes), got \(b) bytes"
        case .functionNotFound(let n): return "function '\(n)' not in the bundle"
        case .contract(let s): return "graph contract: \(s)"
        case .iosBundleOnMac(let p): return "refusing an iOS (h18p) bundle on macOS: \(p)"
        case .audioTooShort(let n, let m):
            return "audio of \(n) samples is shorter than the first streaming chunk (\(m) samples)"
        case .graphOutput(let s): return "graph output: \(s)"
        }
    }
}

public struct N3DAssets: Sendable {
    public static let hidden = 512
    public static let stackedWidth = N3DMel.nMels * N3DMel.stack      // 1024

    public let directory: URL
    /// [512, 1024] `model.audio_tower.embedder.projection.weight` (Linear 1024 -> 512, no bias).
    public let projection: [Float]
    /// [512] `silence_embeds`: the speaker cache's silence row.
    public let silence: [Float]
    /// [128, 257] librosa slaney filterbank (bit-identical to the transformers feature extractor's).
    public let melFilters: [Float]
    /// [400] `torch.hann_window(400, periodic=False)`.
    public let hannWindow: [Float]
    /// metadata.json as stored (nil if absent).
    public let metadata: Data?

    public init(directory: URL) throws {
        self.directory = directory
        projection = try Self.readF32LE(directory.appendingPathComponent("embedder_projection.f32le"),
                                        count: Self.hidden * Self.stackedWidth)
        silence = try Self.readF32LE(directory.appendingPathComponent("silence_embeds.f32le"), count: Self.hidden)
        melFilters = try Self.readF32LE(directory.appendingPathComponent("mel_filters_128x257.f32le"),
                                        count: N3DMel.nMels * N3DMel.nFreq)
        hannWindow = try Self.readF32LE(directory.appendingPathComponent("hann_window_400.f32le"),
                                        count: N3DMel.winLength)
        metadata = try? Data(contentsOf: directory.appendingPathComponent("metadata.json"))
    }

    /// A raw little-endian float32 file of exactly `count` values.
    public static func readF32LE(_ url: URL, count: Int) throws -> [Float] {
        guard let data = try? Data(contentsOf: url) else { throw N3DError.missingFile(url.path) }
        guard data.count == count * 4 else {
            throw N3DError.badSize(file: url.lastPathComponent, expectedFloats: count, gotBytes: data.count)
        }
        return readF32LE(data)
    }

    /// Every float32 of a little-endian buffer.
    public static func readF32LE(_ data: Data) -> [Float] {
        let n = data.count / 4
        return data.withUnsafeBytes { raw in
            (0..<n).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    /// The graph for a profile kind in this directory. iOS: the AOT `.h18p.aimodelc` when present, else
    /// the `.aimodel`. macOS: only the `.aimodel` (an iOS bundle is never picked on a Mac).
    public func modelURL(for kind: N3DProfile.Kind) -> URL? {
        let base = "n3d_\(kind.rawValue)_float16"
        var names = ["\(base).aimodel"]
        #if os(iOS)
        names.insert("\(base).h18p.aimodelc", at: 0)
        #endif
        for name in names {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        return nil
    }
}
