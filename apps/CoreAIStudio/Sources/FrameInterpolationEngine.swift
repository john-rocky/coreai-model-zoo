// FrameInterpolationEngine — downloads/loads the RIFE v4.26 Core AI bundle(s) and runs frame
// interpolation natively. Not published to Hugging Face yet (session that authored the
// conversion pipeline was conversion-first; see zoo/rife-v4.26.md) — sideload-first design via
// `loadLocal`, with an HF catalog entry ready for once it ships.
//
// Compute routing: tries the SPLIT bundle first (flow-estimation on ANE, warp+merge on GPU —
// conversion/rife_compute_router.py, backed by real Tier-1/Tier-2 measurements). Falls back to
// the MONOLITH bundle on GPU if the split load throws — `knowledge/swift-runtime.md` documents
// a real failure mode where a raw ANE-default load on a single-`main` vision graph can crash
// (`Program load failure`); this fallback absorbs that risk without giving up the ANE path
// when it works. `--split-warp` in conversion/export_rife.py names the three files this engine
// expects in one directory: `<stem>.aimodel` (monolith), `<stem>_flow.aimodel`,
// `<stem>_warpmerge.aimodel`.
//
// Static-shape host contract (v1 — see zoo/rife-v4.26.md "Deferred: app integration"): the
// graph's H,W are fixed at export time (this session exported 384x640). Real frames of any
// size are resized to the graph's shape, run, and the result resized back — the same
// resize-to-graph / resize-back pattern Depth Anything 3 ships with. A per-resolution catalog +
// nearest-fit is the noted future optimization, not built here.

import CoreAI
import CoreGraphics
import Foundation

@MainActor
final class FrameInterpolationEngine: ObservableObject {
    struct ModelOption: Identifiable, Hashable {
        var id: String { repoId }
        let repoId: String
        let bundleDirName: String
        let title: String
        let stem: String
    }

    // Placeholder repo — flip once the bundle in zoo/rife-v4.26.md is published; loadLocal
    // works today against this session's exported bundles.
    static let catalog: [ModelOption] = [
        ModelOption(
            repoId: "mlboydaisuke/RIFE-v4.26-CoreAI", bundleDirName: "rife-CoreAI",
            title: "RIFE v4.26", stem: "rife-v4.26_384x640_float32")
    ]

    enum BundlePath: Equatable { case split, monolith }

    enum Status: Equatable {
        case idle, downloading, loading, ready, interpolating
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model loaded"
            case .downloading: return "Downloading RIFE…"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .interpolating: return "Interpolating…"
            case .error(let m): return "Error: \(m)"
            }
        }

        var isBusy: Bool {
            switch self {
            case .downloading, .loading, .interpolating: return true
            default: return false
            }
        }
    }

    @Published private(set) var status: Status = .idle
    @Published private(set) var loadedPath: BundlePath?
    @Published private(set) var graphWidth = 0
    @Published private(set) var graphHeight = 0
    @Published private(set) var modelTitle = ""

    let downloader = ModelDownloader()

    private var flowFn: InferenceFunction?
    private var warpFn: InferenceFunction?
    private var monolithFn: InferenceFunction?
    private var work: Task<Void, Never>?

    var isReady: Bool { if case .ready = status { return true }; return false }

    // MARK: - Loading

    func loadFromHub(_ option: ModelOption = FrameInterpolationEngine.catalog[0], mode: ComputeMode = .interactive) {
        work?.cancel()
        modelTitle = option.title
        status = .downloading
        work = Task {
            do {
                let dest = try Self.bundleDestination(for: option)
                let items = [
                    ModelDownloader.Item(remote: "\(option.stem).aimodel", local: "\(option.stem).aimodel"),
                    ModelDownloader.Item(remote: "\(option.stem)_flow.aimodel", local: "\(option.stem)_flow.aimodel"),
                    ModelDownloader.Item(remote: "\(option.stem)_warpmerge.aimodel", local: "\(option.stem)_warpmerge.aimodel"),
                ]
                await downloader.fetch(repo: "https://huggingface.co/\(option.repoId)", items: items, into: dest)
                try Task.checkCancellation()
                if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                try await loadBundle(at: dest, stem: option.stem, mode: mode)
            } catch is CancellationError {
            } catch {
                status = .error("\(error)")
            }
        }
    }

    /// `dir` must contain `<stem>.aimodel`, `<stem>_flow.aimodel`, `<stem>_warpmerge.aimodel`
    /// (export_rife.py --split-warp's naming convention).
    func loadLocal(_ dir: URL, stem: String, mode: ComputeMode = .interactive) {
        work?.cancel()
        modelTitle = stem
        work = Task {
            do { try await loadBundle(at: dir, stem: stem, mode: mode) }
            catch { status = .error("\(error)") }
        }
    }

    private func loadBundle(at dir: URL, stem: String, mode: ComputeMode) async throws {
        status = .loading
        let accessed = dir.startAccessingSecurityScopedResource()
        defer { if accessed { dir.stopAccessingSecurityScopedResource() } }

        let splitPlan = ComputeRouter.route(bundleKind: .split, mode: mode)
        do {
            let flowURL = dir.appendingPathComponent("\(stem)_flow.aimodel")
            let warpURL = dir.appendingPathComponent("\(stem)_warpmerge.aimodel")
            let flowModel = try await AIModel(
                contentsOf: flowURL, options: ComputeRouter.specializationOptions(for: splitPlan.functions["flow"]!))
            let warpModel = try await AIModel(
                contentsOf: warpURL, options: ComputeRouter.specializationOptions(for: splitPlan.functions["warpmerge"]!))
            guard let ffn = try flowModel.loadFunction(named: "main"),
                  let wfn = try warpModel.loadFunction(named: "main") else {
                throw Self.err("split bundle missing 'main' function")
            }
            try Task.checkCancellation()
            flowFn = ffn; warpFn = wfn; monolithFn = nil; loadedPath = .split
            try setGraphShape(from: ffn.descriptor)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            // Graceful fallback — see file header.
            let monoPlan = ComputeRouter.route(bundleKind: .monolith, mode: mode)
            let monoURL = dir.appendingPathComponent("\(stem).aimodel")
            let monoModel = try await AIModel(
                contentsOf: monoURL, options: ComputeRouter.specializationOptions(for: monoPlan.functions["main"]!))
            guard let mfn = try monoModel.loadFunction(named: "main") else {
                throw Self.err("monolith bundle missing 'main' function")
            }
            try Task.checkCancellation()
            flowFn = nil; warpFn = nil; monolithFn = mfn; loadedPath = .monolith
            try setGraphShape(from: mfn.descriptor)
        }
        status = .ready
    }

    private func setGraphShape(from descriptor: InferenceFunctionDescriptor) throws {
        guard case .ndArray(let d) = descriptor.inputDescriptor(of: "img0") else {
            throw Self.err("could not read 'img0' descriptor")
        }
        graphHeight = d.shape[2]
        graphWidth = d.shape[3]
    }

    func cancel() {
        work?.cancel()
        if status.isBusy { status = isReady || flowFn != nil || monolithFn != nil ? .ready : .idle }
    }

    // MARK: - Interpolation

    /// The true midpoint (t=0.5) or an arbitrary tween (t in (0,1)).
    func interpolate(_ a: CGImage, _ b: CGImage, at t: Double) async throws -> CGImage {
        status = .interpolating
        defer { status = .ready }
        return try await Self.runOne(
            flowFn: flowFn, warpFn: warpFn, monolithFn: monolithFn,
            img0: a, img1: b, timestep: t, graphW: graphWidth, graphH: graphHeight)
    }

    /// N-way: `count` frames strictly between `a` and `b`, evenly spaced (timestep = k/(count+1)
    /// for k = 1...count) — the SVP-style N-times FPS building block.
    func interpolate(_ a: CGImage, _ b: CGImage, count: Int) async throws -> [CGImage] {
        guard count > 0 else { return [] }
        status = .interpolating
        defer { status = .ready }
        var out: [CGImage] = []
        out.reserveCapacity(count)
        for k in 1...count {
            let t = Double(k) / Double(count + 1)
            let mid = try await Self.runOne(
                flowFn: flowFn, warpFn: warpFn, monolithFn: monolithFn,
                img0: a, img1: b, timestep: t, graphW: graphWidth, graphH: graphHeight)
            out.append(mid)
        }
        return out
    }

    private static func runOne(
        flowFn: InferenceFunction?, warpFn: InferenceFunction?, monolithFn: InferenceFunction?,
        img0: CGImage, img1: CGImage, timestep: Double, graphW: Int, graphH: Int
    ) async throws -> CGImage {
        guard graphW > 0, graphH > 0 else { throw err("no bundle loaded") }
        let origW = img0.width, origH = img0.height
        let a = try resizeImage(img0, toWidth: graphW, height: graphH)
        let b = try resizeImage(img1, toWidth: graphW, height: graphH)
        let x0 = try ndArray(from: a)
        let x1 = try ndArray(from: b)

        var t = NDArray(shape: [1, 1, 1, 1], scalarType: .float32)
        let tView = t.mutableView(as: Float.self)
        tView.withUnsafeMutablePointer { ptr, _, _ in ptr[0] = Float(timestep) }

        let midArr: NDArray
        if let flowFn, let warpFn {
            var flowOut = try await flowFn.run(inputs: ["img0": x0, "img1": x1, "timestep": t])
            var warpIn: [String: NDArray] = ["img0": x0, "img1": x1]
            for name in ["f0", "f1", "flow", "mask", "feat", "t_full"] {
                guard let v = flowOut.remove(name), let nd = v.ndArray else {
                    throw err("split flow output missing '\(name)'")
                }
                warpIn[name] = nd
            }
            var warpOut = try await warpFn.run(inputs: warpIn)
            guard let v = warpOut.remove("mid"), let nd = v.ndArray else { throw err("no 'mid' output") }
            midArr = nd
        } else if let monolithFn {
            var out = try await monolithFn.run(inputs: ["img0": x0, "img1": x1, "timestep": t])
            guard let v = out.remove("mid"), let nd = v.ndArray else { throw err("no 'mid' output") }
            midArr = nd
        } else {
            throw err("no function loaded")
        }

        let midImg = try cgImage(from: midArr)
        return try resizeImage(midImg, toWidth: origW, height: origH)
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
        NSError(domain: "FrameInterpolationEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
