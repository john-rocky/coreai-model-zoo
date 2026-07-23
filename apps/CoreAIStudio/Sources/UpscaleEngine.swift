// UpscaleEngine — downloads/loads the AdcSR x4 super-resolution Core AI bundle and runs it
// natively (no coreai-kit — see project.yml). Host pipeline (tiling, feather-blend, global
// color-match) is a direct Swift port of apps/CoreAIStudio/reference/adcsr_host_reference.py,
// which is itself gated (self-consistency: coverage, finiteness, color-match correctness) and
// run against the real bundle before this file was written — see that script's docstring for
// why the exact tiling parameters here are an authored contract, not a reverse-engineered
// clone of the closed-source CoreAIKitVision `SuperResolver`.
//
// Graph contract (conversion/export_adcsr.py, ✓ VERIFIED by reading the export code): lr
// [1,3,128,128] in [-1,1] -> sr [1,3,512,512], no in-graph normalization — the host does
// rgb*2-1 in, clamp((sr+1)/2) out. Compute unit: GPU (a large diffusion-derived graph — the
// Mac-native "large model / max throughput" tier, per knowledge/compute-units-and-authoring.md;
// this is also the OTHER half of the app's "use the whole chip" story alongside RIFE's ANE
// split — see conversion/rife_compute_router.py).

import CoreAI
import CoreGraphics
import Foundation

@MainActor
final class UpscaleEngine: ObservableObject {
    struct ModelOption: Identifiable, Hashable {
        var id: String { repoId }
        let repoId: String
        let bundleDirName: String
        let title: String
        let aimodelName: String
    }

    static let catalog: [ModelOption] = [
        ModelOption(
            repoId: "mlboydaisuke/AdcSR-CoreAI",
            bundleDirName: "adcsr-CoreAI",
            title: "AdcSR ×4",
            aimodelName: "adcsr_x4_float32.aimodel")
    ]

    enum Status: Equatable {
        case idle, downloading, loading, ready, upscaling
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model loaded"
            case .downloading: return "Downloading AdcSR (~1.7 GB)…"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .upscaling: return "Upscaling ×4 on GPU…"
            case .error(let m): return "Error: \(m)"
            }
        }

        var isBusy: Bool {
            switch self {
            case .downloading, .loading, .upscaling: return true
            default: return false
            }
        }
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var sourceImage: CGImage?
    @Published private(set) var resultImage: CGImage?
    @Published private(set) var upscaleSeconds: Double?
    @Published private(set) var modelTitle = ""

    let downloader = ModelDownloader()

    private var fn: InferenceFunction?
    private var work: Task<Void, Never>?

    // Tiling contract — see this file's header + adcsr_host_reference.py.
    // nonisolated: read from the nonisolated static run/tiling functions below (a @MainActor
    // class's static members are actor-isolated by default; these are plain constants, safe
    // to free from that).
    nonisolated private static let tile = 128
    nonisolated private static let scale = 4
    nonisolated private static let srTile = tile * scale
    nonisolated private static let maxInputSide = 512
    nonisolated private static let overlap = 16
    nonisolated private static let stride = tile - overlap

    var canUpscale: Bool {
        if case .ready = status { return true }
        return false
    }

    func setImage(_ cg: CGImage) {
        sourceImage = cg
        resultImage = nil
        upscaleSeconds = nil
        if case .error = status { status = fn != nil ? .ready : .idle }
    }

    // MARK: - Loading

    func loadFromHub(_ option: ModelOption = UpscaleEngine.catalog[0]) {
        work?.cancel()
        modelTitle = option.title
        status = .downloading
        work = Task {
            do {
                let dest = try Self.bundleDestination(for: option)
                await downloader.fetch(
                    repo: "https://huggingface.co/\(option.repoId)",
                    items: [.init(remote: option.aimodelName, local: option.aimodelName)],
                    into: dest)
                try Task.checkCancellation()
                if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                try await loadBundle(at: dest.appendingPathComponent(option.aimodelName))
            } catch is CancellationError {
            } catch {
                status = .error("\(error)")
            }
        }
    }

    func loadLocal(_ url: URL) {
        work?.cancel()
        modelTitle = url.lastPathComponent
        work = Task {
            do { try await loadBundle(at: url) }
            catch { status = .error("\(error)") }
        }
    }

    private func loadBundle(at url: URL) async throws {
        status = .loading
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let opts = ComputeRouter.specializationOptions(for: .gpu)
        let model = try await AIModel(contentsOf: url, options: opts)
        guard let f = try model.loadFunction(named: "main") else {
            throw Self.err("bundle has no 'main' function")
        }
        try Task.checkCancellation()
        fn = f
        status = .ready
    }

    // MARK: - Upscale

    func upscale() {
        guard let fn, let source = sourceImage, !status.isBusy else { return }
        status = .upscaling
        resultImage = nil
        work = Task {
            do {
                let t0 = Date()
                let out = try await Self.runUpscale(fn: fn, image: source)
                try Task.checkCancellation()
                resultImage = out
                upscaleSeconds = Date().timeIntervalSince(t0)
                status = .ready
            } catch is CancellationError {
            } catch {
                status = .error("\(error)")
            }
        }
    }

    func cancel() {
        work?.cancel()
        if status.isBusy { status = fn != nil ? .ready : .idle }
    }

    // MARK: - Core algorithm (nonisolated: runs off the main actor; InferenceFunction/AIModel
    // are Sendable structs, so no @unchecked Sendable box is needed here, unlike engines built
    // on higher-level non-Sendable runtime types).

    nonisolated private static func runUpscale(fn: InferenceFunction, image: CGImage) async throws -> CGImage {
        let capped = try capToMaxSide(image, maxSide: maxInputSide)
        let source = try RGBBuffer(cgImage: capped)
        let xs = tileOrigins(size: source.width, tile: tile, stride: stride)
        let ys = tileOrigins(size: source.height, tile: tile, stride: stride)

        var canvas = RGBBuffer(width: source.width * scale, height: source.height * scale)
        var weightSum = [Float](repeating: 0, count: canvas.width * canvas.height)

        for (iy, oy) in ys.enumerated() {
            for (ix, ox) in xs.enumerated() {
                let lrTile = source.cropped(x: ox, y: oy, width: tile, height: tile)
                let srTileBuf = try await runTile(fn: fn, lrTile: lrTile)
                let feather = featherWeight(
                    tilePx: srTile, overlapPx: overlap, scale: scale,
                    hasLeft: ix > 0, hasRight: ix < xs.count - 1,
                    hasTop: iy > 0, hasBottom: iy < ys.count - 1)
                canvas.accumulate(srTileBuf, weight: feather, at: ox * scale, y: oy * scale, into: &weightSum)
            }
        }
        canvas.normalize(by: weightSum)
        canvas.colorMatch(to: source)  // GLOBAL, once, after stitching — see file header
        return try canvas.toCGImage()
    }

    nonisolated private static func runTile(fn: InferenceFunction, lrTile: RGBBuffer) async throws -> RGBBuffer {
        let x = lrTile.toNDArray(scale: 2.0, offset: -1.0)  // [0,1] -> [-1,1]
        var outputs = try await fn.run(inputs: ["lr": x])
        guard let value = outputs.remove("sr"), let sr = value.ndArray else {
            throw NSError(domain: "UpscaleEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "no 'sr' output"])
        }
        return try RGBBuffer(ndArray: sr, scale: 0.5, offset: 0.5)  // [-1,1] -> [0,1] (clamped in colorMatch/toCGImage)
    }

    nonisolated private static func capToMaxSide(_ image: CGImage, maxSide: Int) throws -> CGImage {
        let w = image.width, h = image.height
        let longSide = max(w, h)
        var scaleFactor: Double = 1.0
        if longSide > maxSide {
            scaleFactor = Double(maxSide) / Double(longSide)
        } else if min(w, h) < tile {
            scaleFactor = Double(tile) / Double(min(w, h))
        }
        if scaleFactor == 1.0 { return image }
        let newW = max(tile, Int((Double(w) * scaleFactor).rounded()))
        let newH = max(tile, Int((Double(h) * scaleFactor).rounded()))
        return try resizeImage(image, toWidth: newW, height: newH)
    }

    // MARK: - Storage / errors

    private static func bundleDestination(for option: ModelOption) throws -> URL {
        let docs = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent(option.bundleDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func err(_ msg: String) -> NSError {
        NSError(domain: "UpscaleEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}

// MARK: - FrameUpscaler conformance (VideoUpscaler.swift) — exposes the already-loaded `fn`
// through the model-agnostic per-frame protocol, so VideoUpscaler never depends on UpscaleEngine
// concretely. Declared in this file (not an extension elsewhere) because it needs access to the
// private `fn` property.

extension UpscaleEngine: FrameUpscaler {
    @MainActor
    func upscale(_ image: CGImage) async throws -> CGImage {
        guard let fn else { throw Self.err("no model loaded") }
        return try await Self.runUpscale(fn: fn, image: image)
    }
}
