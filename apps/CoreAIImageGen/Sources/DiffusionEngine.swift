// DiffusionEngine — downloads / loads a Core AI diffusion bundle and generates images.
//
// FLUX.2 runs on Apple's stock CoreAIDiffusionPipeline: `FlowTransformerPipeline` from
// apple/coreai-models at the commit project.yml pins. It reads whatever bundle that pipeline reads
// (FLUX.2 and Sana Sprint, picked from metadata.json), so such a `coreai.diffusion.export` folder drops
// in via "Local…". GLM-Image and Z-Image-Turbo run bespoke host loops on the Core AI runtime
// (GlmImagePipeline below, ZImagePipeline.swift).
//
// The hosted catalog is macOS-only: FLUX.2 klein 4B's peak footprint exceeds the iOS per-process
// memory limit (the int4 export traced ~0.4 GB over on a 12 GB iPhone 17 Pro, 2026-06-15 — the text
// encoder is not released before the transformer runs). The iOS app loads bundles via "Local…".
//
// Model delivery: a FLUX.2 folder comes down with swift-transformers' HubApi, one file at a time,
// pinned to the Hub revision in the catalog, so the app gets exactly the files it was measured with.
// GLM-Image and Z-Image use the shared AppShared/ModelDownloader (range-chunked parallel download
// with cross-launch resume and atomic bundle placement) for the `.aimodel` directory bundles + the
// tokenizer, plus a couple of direct GETs for the tiny root files the HF tree API can't enumerate.

import CoreAI
import CoreAIDiffusionPipeline
import CoreAIShared
import CoreGraphics
import Foundation
import Hub
import ImageIO
import Tokenizers
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#endif

/// Thread-safe cancel flag — readable from the pipeline's (possibly off-main)
/// progress callback without touching @MainActor state.
final class CancellationToken: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    private var steps = 0
    var isCancelled: Bool { lock.withLock { flag } }
    func cancel() { lock.withLock { flag = true } }
    /// Denoising steps finished so far (written by the FLUX.2 progress callback).
    var stepsDone: Int { lock.withLock { steps } }
    func stepFinished(_ step: Int) { lock.withLock { steps = step } }
}

/// One FLUX.2 folder of a Hugging Face repo, pinned to a revision: the files this app reads from
/// it, with each file's size at that revision. HubApi fetches them one at a time, so an interrupted
/// download keeps the files that finished and the next try fetches only the rest. The folder counts
/// as on disk only when every size matches, so the runtime never opens a partial bundle.
struct FluxFolder: Hashable, Sendable {
    struct File: Hashable, Sendable {
        let path: String
        let bytes: Int64
    }

    let repo: String
    let revision: String
    let folder: String
    let files: [File]

    var totalBytes: Int64 { files.reduce(0) { $0 + $1.bytes } }

    /// The folder the pipeline opens: <base>/models/<repo>/<folder>.
    func root(base: URL) -> URL {
        HubApi(downloadBase: base).localRepoLocation(HubApi.Repo(id: repo)).appending(path: folder)
    }

    func isOnDisk(base: URL) -> Bool { files.allSatisfy { size(of: $0.path, base: base) == $0.bytes } }

    func bytesOnDisk(base: URL) -> Int64 {
        files.reduce(0) { $0 + min(size(of: $1.path, base: base) ?? 0, $1.bytes) }
    }

    private func size(of path: String, base: URL) -> Int64? {
        let attributes = try? FileManager.default.attributesOfItem(atPath: root(base: base).appending(path: path).path)
        return (attributes?[.size] as? NSNumber)?.int64Value
    }

    /// Downloads the files that are not on disk yet. `progress` gets the bytes done so far.
    func download(base: URL, progress: @escaping @Sendable (Int64) -> Void) async throws {
        // cache: nil = one copy of each file, under `base` (no second copy in the Hub cache).
        // useOfflineMode: false = no network gives the network's own error, not a missing-file one.
        let hub = HubApi(downloadBase: base, cache: nil, useOfflineMode: false)
        var done: Int64 = 0
        for file in files {
            if size(of: file.path, base: base) != file.bytes {
                let before = done
                try await hub.snapshot(from: HubApi.Repo(id: repo), revision: revision,
                                       matching: ["\(folder)/\(file.path)"]) { @Sendable fraction in
                    progress(before + Int64(fraction.fractionCompleted * Double(file.bytes)))
                }
                try Task.checkCancellation()
                guard size(of: file.path, base: base) == file.bytes else {
                    throw NSError(domain: "DiffusionEngine", code: 1, userInfo: [
                        NSLocalizedDescriptionKey: "\(file.path) did not download completely."])
                }
            }
            done += file.bytes
            progress(done)
        }
    }
}

extension FluxFolder {
    static let repo = "mlboydaisuke/FLUX.2-klein-4B-CoreAI"
    /// The Hub commit that holds the 2026-10-10 folders, exported with the apple/coreai-models
    /// commit project.yml pins.
    static let revision = "039db98cbf1247680ea5e76048f87b3ce061e6d6"

    /// The tokenizer and the small root files, the same bytes in both folders.
    private static let shared: [File] = [
        File(path: "tokenizer/chat_template.jinja", bytes: 4_168),
        File(path: "tokenizer/config.json", bytes: 375),
        File(path: "tokenizer/tokenizer.json", bytes: 11_422_650),
        File(path: "tokenizer/tokenizer_config.json", bytes: 375),
        File(path: "vae_bn_mean.npy", bytes: 640),
        File(path: "vae_bn_var.npy", bytes: 640),
    ]

    /// int8 per-block 32 (text encoder + transformer): text-to-image and Edit. The transformer
    /// carries the image-to-image entrypoints, and the VAE encoder turns the reference into tokens.
    /// The folder's half-size VAEs (512 and tiled decoding) are left out: this app decodes at 1024.
    static let int8 = FluxFolder(
        repo: repo, revision: revision, folder: "macos-int8",
        files: [File(path: "metadata.json", bytes: 1_549)] + shared + [
            File(path: "VAEDecoder.aimodel/main.hash", bytes: 32),
            File(path: "VAEDecoder.aimodel/metadata.json", bytes: 395),
            File(path: "VAEDecoder.aimodel/main.mlirb", bytes: 99_294_687),
            File(path: "VAEEncoder.aimodel/main.hash", bytes: 32),
            File(path: "VAEEncoder.aimodel/metadata.json", bytes: 395),
            File(path: "VAEEncoder.aimodel/main.mlirb", bytes: 68_896_263),
            File(path: "TextEncoder.aimodel/main.hash", bytes: 32),
            File(path: "TextEncoder.aimodel/metadata.json", bytes: 396),
            File(path: "TextEncoder.aimodel/main.mlirb", bytes: 3_309_658_432),
            File(path: "Transformer.aimodel/main.hash", bytes: 32),
            File(path: "Transformer.aimodel/metadata.json", bytes: 396),
            File(path: "Transformer.aimodel/main.mlirb", bytes: 4_119_647_410),
        ])

    /// fp16: text-to-image only — the folder has no image-to-image transformer, so its VAE encoder
    /// is left out too (with it, the pipeline would load it for nothing).
    static let fp16 = FluxFolder(
        repo: repo, revision: revision, folder: "macos-fp16",
        files: [File(path: "metadata.json", bytes: 974)] + shared + [
            File(path: "VAEDecoder.aimodel/main.hash", bytes: 32),
            File(path: "VAEDecoder.aimodel/metadata.json", bytes: 395),
            File(path: "VAEDecoder.aimodel/main.mlirb", bytes: 99_294_680),
            File(path: "TextEncoder.aimodel/main.hash", bytes: 32),
            File(path: "TextEncoder.aimodel/metadata.json", bytes: 396),
            File(path: "TextEncoder.aimodel/main.mlirb", bytes: 6_228_947_553),
            File(path: "Transformer.aimodel/main.hash", bytes: 32),
            File(path: "Transformer.aimodel/metadata.json", bytes: 396),
            File(path: "Transformer.aimodel/main.mlirb", bytes: 7_751_312_967),
        ])
}

@MainActor
final class DiffusionEngine: ObservableObject {

    /// A Core AI-converted diffusion bundle published on the Hugging Face Hub.
    struct ModelOption: Identifiable, Hashable {
        // Several options share a repo (FLUX int8/fp16, GLM 512/1024, Z-Image 512/1024), so the repo
        // cannot identify a row — ForEach needs unique ids.
        var id: String { title }
        /// Short name, also the value of `-model` in hands-off runs.
        let key: String
        let repoId: String           // "org/name" on the Hub
        let bundleDirName: String    // local folder name under Documents/
        let title: String
        let defaultSteps: Int
        let defaultGuidance: Float
        // FLUX.2: one folder of the Hub repo at a pinned revision, run by the stock pipeline.
        var flux: FluxFolder? = nil
        // GLM-Image (AR+diffusion hybrid) uses the bespoke GlmImagePipeline, not the high-level
        // diffusion pipeline. Its bundle is a folder of AR/DiT/VAE .aimodelc + tokenizer + ehs.f32.
        // One Hub repo hosts both resolutions (the AR is shared); glmSize picks the DiT/VAE pair.
        var isGLM: Bool = false
        var glmSize: Int = 1024
        // Z-Image-Turbo: bespoke host loop (negated CFG, host-prepped tokens + RoPE).
        // One DiT graph serves every side, so the option carries the side to render at.
        var isZImage: Bool = false
        var zSide: Int = 512
    }

    // Hosted catalog — macOS only. FLUX.2 klein 4B overruns the iOS memory limit, so the
    // iOS app ships with an empty catalog and loads bundles via "Local…".
    static let catalog: [ModelOption] = {
        #if os(macOS)
        return [
            // FLUX.2 klein 4B (Black Forest Labs, Apache-2.0): two folders of one Hub repo. int8 is
            // the default and also runs Edit; fp16 is text-to-image only.
            ModelOption(
                key: "int8",
                repoId: FluxFolder.repo,
                bundleDirName: "FLUX.2-klein-4B-CoreAI",
                title: "FLUX.2 klein 4B int8 (7.6 GB · text-to-image, Edit)",
                defaultSteps: 4, defaultGuidance: 1.0,
                flux: .int8),
            ModelOption(
                key: "fp16",
                repoId: FluxFolder.repo,
                bundleDirName: "FLUX.2-klein-4B-CoreAI",
                title: "FLUX.2 klein 4B fp16 (14.1 GB · text-to-image)",
                defaultSteps: 4, defaultGuidance: 1.0,
                flux: .fp16),
            // GLM-Image (zai-org, MIT) — AR + flow-matching diffusion hybrid. 1024 is native
            // quality; 512 is a faster variant (same weights, smaller static graph). Both live in
            // ONE Hub repo (the 9.6 GB AR is shared); "Download & Load" loads a bundle already
            // present under Documents/ or fetches it from the Hub otherwise.
            ModelOption(
                key: "glm1024",
                repoId: "mlboydaisuke/GLM-Image-CoreAI",
                bundleDirName: "GLM-Image-1024",
                title: "GLM-Image 1024 (AR+diffusion)",
                defaultSteps: 20, defaultGuidance: 1.5,
                isGLM: true, glmSize: 1024),
            ModelOption(
                key: "glm512",
                repoId: "mlboydaisuke/GLM-Image-CoreAI",
                bundleDirName: "GLM-Image-512",
                title: "GLM-Image 512 (AR+diffusion)",
                defaultSteps: 20, defaultGuidance: 1.5,
                isGLM: true, glmSize: 512),
            // Z-Image-Turbo (Tongyi-MAI, Apache-2.0). ONE bf16 DiT graph covers every side —
            // the image-token and caption axes are dynamic — so 512 and 1024 share weights and
            // differ only in what the host feeds them.
            ModelOption(
                key: "zimage512",
                repoId: "mlboydaisuke/Z-Image-Turbo-CoreAI",
                bundleDirName: "Z-Image-Turbo",
                title: "Z-Image-Turbo 512",
                defaultSteps: 8, defaultGuidance: 1.0,
                isZImage: true, zSide: 512),
            ModelOption(
                key: "zimage1024",
                repoId: "mlboydaisuke/Z-Image-Turbo-CoreAI",
                bundleDirName: "Z-Image-Turbo",
                title: "Z-Image-Turbo 1024",
                defaultSteps: 8, defaultGuidance: 1.0,
                isZImage: true, zSide: 1024),
        ]
        #else
        return []
        #endif
    }()

    // GLM bundle contents on the Hub. The repo hosts both resolutions; the size-specific
    // DiT/VAE download to size-agnostic local names so the pipeline's resolver finds them.
    private static func glmItems(size: Int) -> [ModelDownloader.Item] {
        [
            .init(remote: "glm_image_ar.aimodelc", local: "glm_image_ar.aimodelc"),
            .init(remote: "glm_image_dit_\(size).aimodelc", local: "glm_image_dit.aimodelc"),
            .init(remote: "glm_image_vae_\(size).aimodel", local: "glm_image_vae.aimodel"),
            .init(remote: "tokenizer", local: "tokenizer"),
        ]
    }
    private static let glmRootFiles = ["ehs.f32"]

    /// Z-Image bundle contents on the Hub. One repo, one DiT (dynamic axes) and per-side VAEs;
    /// the ~2.4 MB glue (RoPE tables + t_embedder graph) keeps the host from re-implementing
    /// the RoPE and the timestep MLP, and the ids-input encoder keeps the 778 MB embedding
    /// matrix inside its graph.
    private static func zimageItems(side: Int) -> [ModelDownloader.Item] {
        [
            .init(remote: "zimage_dit_512_cap32_full_native_bf16_dyncap_dynimg_iofp32.aimodel",
                  local: "zimage_dit.aimodel"),
            .init(remote: "zimage_encoder_seq64_full_bf16_ids_iofp32.aimodel", local: "zimage_encoder.aimodel"),
            .init(remote: "zimage_vae_\(side)_fp32.aimodel", local: "zimage_vae_\(side).aimodel"),
            .init(remote: "glue/zimage_t_embedder_fp32/zimage_t_embedder_fp32.aimodel",
                  local: "zimage_t_embedder.aimodel"),
            .init(remote: "tokenizer", local: "tokenizer"),
        ]
    }
    private static let zimageRootFiles = ["glue/rope_axis0.f32", "glue/rope_axis1.f32",
                                          "glue/rope_axis2.f32", "glue/rope_meta.json"]

    enum Status: Equatable {
        case idle
        case downloading
        case loading
        case ready
        case generating(step: Int, total: Int)
        case stopping
        case error(String)

        var label: String {
            switch self {
            case .idle: return "No model loaded"
            case .downloading: return "Downloading model…"
            case .loading: return "Loading model…"
            case .ready: return "Ready"
            case .generating(let s, let t): return "Generating… step \(s)/\(t)"
            case .stopping: return "Stopping after this step…"
            case .error(let m): return "Error: \(m)"
            }
        }

        var isBusy: Bool {
            switch self {
            case .downloading, .loading, .generating, .stopping: return true
            default: return false
            }
        }
    }

    /// Bytes of a FLUX.2 folder download (HubApi), for the progress bar.
    struct HubProgress: Equatable {
        var done: Int64
        var total: Int64
    }

    @Published var status: Status = .idle
    @Published var image: CGImage?
    @Published var exportURL: URL?
    @Published var savedURL: URL?
    @Published var modelTitle: String = "—"
    @Published var loadSeconds: Double?
    @Published var generateSeconds: Double?
    @Published var imageSize: String = ""
    /// A sentence about the last Generate that did not give an image (e.g. after Stop).
    @Published var notice: String?
    /// Set while a FLUX.2 folder downloads; GLM / Z-Image report through `downloader`.
    @Published var hubDownload: HubProgress?
    /// True when the loaded FLUX.2 bundle can run Edit: it has a VAE encoder and a traced
    /// image-to-image graph (macos-int8 does; macos-fp16 does not).
    @Published var supportsEdit: Bool = false
    /// Native square side the loaded model generates at (e.g. 1024). Used to letterbox an Edit
    /// reference into the square the runtime expects, then crop the result back.
    private var modelSide: Int = 1024

    /// GLM-Image (AR+diffusion hybrid) runs a bespoke low-level pipeline (GlmImagePipeline) instead
    /// of the high-level CoreAIDiffusionPipeline auto-detect path. Set when such a bundle is loaded.
    @Published var isGLM = false
    private var glm: GlmImagePipeline?
    private var zimage: ZImagePipeline?
    private var isZImage = false
    private var zSide = 512

    /// Shared range-chunked downloader (atomic placement + cross-launch resume).
    let downloader = ModelDownloader()

    private var pipeline: FlowTransformerPipeline?
    private var work: Task<Void, Never>?
    private var cancelToken = CancellationToken()

    var canGenerate: Bool { if case .ready = status { return true }; return false }

    // macOS runs the full 1024 components and keeps every model loaded between images; iOS runs
    // the 512 / half components and loads each stage on demand, releasing it after, to keep the
    // peak footprint down.
    #if os(iOS)
    private static let fluxMode: DecodeResolution = .half
    private static let lazyModelLoading = true
    #else
    private static let fluxMode: DecodeResolution = .auto
    private static let lazyModelLoading = false
    #endif
    /// Edit attends to the reference at the output's own token grid (64×64 tokens at 1024).
    private static let referenceGrid: ReferenceGrid = .full

    // MARK: - Loading

    /// Download a converted bundle from the Hugging Face Hub (cached after the first
    /// run, resumable across launches) and load it.
    func loadFromHub(_ option: ModelOption) {
        work?.cancel()
        image = nil; exportURL = nil; loadSeconds = nil; generateSeconds = nil; notice = nil
        modelTitle = option.title
        // Not `.downloading` yet — a bundle already staged under Documents/ never hits the Hub,
        // and flashing a download bar at it is a lie. `.loading` is busy, so the button is
        // disabled either way; the fetch below promotes the status if it actually runs.
        status = .loading
        // Keep the screen awake for the multi-GB download: if the device auto-locks the app
        // gets suspended and the (foreground) URLSession transfer stalls. Re-enabled when the
        // whole flow finishes (see the defer below).
        Self.setIdleTimerDisabled(true)

        // The fine-grained download progress (fraction / byte detail) is read straight off
        // `downloader` / `hubDownload` by the view; this Task only sequences the phases and
        // surfaces a terminal error.
        work = Task {
            defer { Self.setIdleTimerDisabled(false) }
            do {
                self.zSide = option.zSide
                let dest = try Self.bundleDestination(for: option)
                if let flux = option.flux {
                    if !flux.isOnDisk(base: dest) {
                        self.hubDownload = HubProgress(done: flux.bytesOnDisk(base: dest), total: flux.totalBytes)
                        self.status = .downloading
                        try await flux.download(base: dest) { bytes in
                            Task { @MainActor in self.showHubDownload(bytes) }
                        }
                        self.hubDownload = nil
                    }
                    try await self.loadPipeline(at: flux.root(base: dest))
                    return
                }
                let items: [ModelDownloader.Item]
                if option.isGLM { items = Self.glmItems(size: option.glmSize) }
                else { items = Self.zimageItems(side: option.zSide) }
                let roots = option.isGLM ? Self.glmRootFiles : Self.zimageRootFiles
                // Load a bundle already present under Documents/ without re-downloading; otherwise
                // fetch it from the Hub. (GLM: the folder is "complete" once the AR bundle + ehs.f32
                // are present.)
                let alreadyLocal = (option.isGLM && Self.glmBundleComplete(at: dest))
                    || (option.isZImage && Self.zimageBundleComplete(at: dest))
                if !alreadyLocal {
                    self.status = .downloading
                    await downloader.fetch(
                        repo: "https://huggingface.co/\(option.repoId)",
                        items: items, into: dest)
                    try Task.checkCancellation()   // cancelled mid-download → clean exit, not .error
                    if case .failed(let msg) = downloader.phase { throw Self.err(msg) }
                    try await Self.fetchRootFiles(repoId: option.repoId, names: roots, into: dest)
                }
                try await self.loadPipeline(at: dest)
            } catch is CancellationError {
                // user cancelled or started another action — cancel() owns the resulting state
            } catch {
                self.hubDownload = nil
                // A cancelled HubApi transfer surfaces as a URL error; cancel() owns that state too.
                if !Task.isCancelled { self.status = .error(error.localizedDescription) }
            }
        }
        holdActivity("Downloading and loading \(option.title)")
    }

    /// Preselect a model whose bundle is already on disk, so opening the app and hitting
    /// "Download & Load" cannot kick off a multi-GB transfer for a model the user never chose.
    /// Falls back to the first entry. (FLUX checks every file's size; GLM and Z-Image their key files.)
    static var defaultSelection: ModelOption? {
        let docs = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                appropriateFor: nil, create: false)
        return catalog.first(where: { opt in
            guard let dir = docs?.appendingPathComponent(opt.bundleDirName, isDirectory: true)
            else { return false }
            if let flux = opt.flux { return flux.isOnDisk(base: dir) }
            return (opt.isGLM && glmBundleComplete(at: dir))
                || (opt.isZImage && zimageBundleComplete(at: dir))
        }) ?? catalog.first
    }

    /// True when a GLM bundle folder already holds everything the pipeline needs locally.
    /// A staged Z-Image bundle is complete once the DiT and the RoPE glue are present.
    private static func zimageBundleComplete(at dir: URL) -> Bool {
        ZImagePipeline.looksLikeZImage(dir)
            && FileManager.default.fileExists(atPath: dir.appendingPathComponent("rope_meta.json").path)
    }

    private static func glmBundleComplete(at dir: URL) -> Bool {
        GlmImagePipeline.looksLikeGLM(dir)
            && FileManager.default.fileExists(atPath: dir.appendingPathComponent("ehs.f32").path)
    }

    private static func setIdleTimerDisabled(_ disabled: Bool) {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = disabled
        #endif
    }

    /// Load a bundle already exported to a local folder.
    func loadLocal(_ url: URL) {
        work?.cancel()
        image = nil; exportURL = nil; loadSeconds = nil; generateSeconds = nil; notice = nil
        modelTitle = url.lastPathComponent
        status = .loading
        work = Task {
            do { try await self.loadPipeline(at: url) }
            catch is CancellationError {}
            catch { self.status = .error(error.localizedDescription) }
        }
        holdActivity("Loading \(url.lastPathComponent)")
    }

    private func loadPipeline(at url: URL) async throws {
        status = .loading
        supportsEdit = false
        // Let go of the loaded model first, so two never sit in memory at once.
        pipeline = nil
        glm = nil
        zimage = nil
        isGLM = false
        isZImage = false

        // GLM-Image bundle (AR+diffusion hybrid) — bespoke low-level pipeline, not the auto-detect path.
        if GlmImagePipeline.looksLikeGLM(url) {
            let glmStart = ContinuousClock.now
            let pipe = GlmImagePipeline(dir: url)
            try await pipe.load()
            self.glm = pipe
            self.isGLM = true
            self.loadSeconds = Self.seconds(since: glmStart)
            self.imageSize = "\(pipe.side)×\(pipe.side)"
            self.modelSide = pipe.side
            self.status = .ready
            return
        }
        // Z-Image bundle — bespoke host loop as well.
        if ZImagePipeline.looksLikeZImage(url) {
            let zStart = ContinuousClock.now
            let pipe = ZImagePipeline(dir: url)
            try await pipe.load()
            self.zimage = pipe
            self.isZImage = true
            self.loadSeconds = Self.seconds(since: zStart)
            self.imageSize = "\(zSide)×\(zSide)"
            self.modelSide = zSide
            self.status = .ready
            return
        }

        // Everything else goes to Apple's FlowTransformerPipeline. The most common wrong pick gets
        // a sentence of its own instead of the runtime's error.
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("metadata.json").path) else {
            throw Self.err("“\(url.lastPathComponent)” is not a bundle this app can load: it has no metadata.json "
                + "(FLUX.2, Sana Sprint) and no GLM-Image or Z-Image graphs. Choose the folder that holds "
                + "metadata.json, tokenizer/ and the .aimodel folders.")
        }
        let start = ContinuousClock.now
        // Reads metadata.json and the tokenizer and picks the components for the best resolution the
        // folder has (1024×1024 when Transformer and VAEDecoder are there).
        let built = try await FlowTransformerPipeline(from: url, mode: Self.fluxMode)
        if !Self.lazyModelLoading {
            // Load every model text-to-image runs now. On the first launch Core AI also specializes
            // each one for this Mac and caches the result; later launches read the cache.
            try await built.loadResources()
        }
        try Task.checkCancellation()
        self.pipeline = built
        self.loadSeconds = Self.seconds(since: start)
        let size = built.defaultImageSize
        self.imageSize = "\(size.width)×\(size.height)"
        self.modelSide = size.width
        self.supportsEdit = built.encoder != nil && built.img2imgRoutes[Self.referenceGrid] != nil
        self.status = .ready
    }

    /// Documents/<bundleDirName> — a clean, dedicated folder that holds the placed
    /// bundles + root files. (Kept separate from any HF cache so a partial transfer
    /// can never leave a half-bundle the runtime would choke on.)
    private static func bundleDestination(for option: ModelOption) throws -> URL {
        let docs = try FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
        let dir = docs.appendingPathComponent(option.bundleDirName, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Fetch the tiny root files with a plain resolve GET (skipping any already present).
    /// `names` are repo paths and may be nested (Z-Image keeps its RoPE tables under `glue/`);
    /// they always land flat at the bundle root, where the pipelines look for them.
    private static func fetchRootFiles(repoId: String, names: [String], into dest: URL) async throws {
        let fm = FileManager.default
        for name in names {
            let target = dest.appendingPathComponent((name as NSString).lastPathComponent)
            if fm.fileExists(atPath: target.path) { continue }
            guard let url = URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(name)") else {
                throw err("bad file url for \(name)")
            }
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
                throw err("\(name): HTTP \((resp as? HTTPURLResponse)?.statusCode ?? -1)")
            }
            try data.write(to: target, options: .atomic)
        }
    }

    private func showHubDownload(_ bytes: Int64) {
        guard var progress = hubDownload, bytes > progress.done else { return }
        // ~500 updates over a multi-GB download, not one per network packet.
        let step = max(progress.total / 500, 1)
        guard bytes / step != progress.done / step || bytes >= progress.total else { return }
        progress.done = bytes
        hubDownload = progress
    }

    // MARK: - Generation

    /// Generate an image from the prompt (text-to-image).
    func generate(prompt: String, negativePrompt: String, steps: Int, guidance: Float, seed: UInt32) {
        if isGLM {
            generateGLM(prompt: prompt, steps: steps, guidance: guidance, seed: seed)
            holdActivity("Generating an image")
            return
        }
        if isZImage {
            generateZImage(prompt: prompt, negativePrompt: negativePrompt,
                           steps: steps, guidance: guidance, seed: seed)
            holdActivity("Generating an image")
            return
        }
        guard let pipeline, canGenerate else { return }
        // FLUX.2 klein is guidance-distilled: one pass per step, and the model ignores the guidance
        // value. Above 1.0 the pipeline adds a second, unconditional pass per step and mixes the
        // two (classifier-free guidance).
        let config = PipelineConfiguration(
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            stepCount: steps,
            guidanceScale: guidance,
            guidanceMode: guidance > 1 ? .manual : .distilled,
            lazyModelLoading: Self.lazyModelLoading)
        runFlux(pipeline, config, steps: steps, cropTo: nil)
    }

    /// FLUX.2 in-context edit on the stock pipeline (reference-token image-to-image): the reference's
    /// latent tokens are appended after the noise tokens, marked T=10 on the RoPE time axis, and the
    /// output is denoised from pure noise while attending to them — the instruction edits the picture
    /// and the subject is kept. The reference is letterboxed into the model's square (no stretching)
    /// and the result is cropped back to its aspect ratio.
    func edit(referenceImage: CGImage, instruction: String, steps: Int, guidance: Float, seed: UInt32) {
        guard let pipeline, supportsEdit, canGenerate else { return }
        let config = PipelineConfiguration(
            prompt: instruction,
            seed: seed,
            stepCount: steps,
            guidanceScale: guidance,
            startingImage: Self.fitToSquare(referenceImage, side: modelSide),
            referenceGrid: Self.referenceGrid,
            guidanceMode: guidance > 1 ? .manual : .distilled,
            lazyModelLoading: Self.lazyModelLoading)
        runFlux(pipeline, config, steps: steps, cropTo: (w: referenceImage.width, h: referenceImage.height))
    }

    private func runFlux(_ pipeline: FlowTransformerPipeline, _ config: PipelineConfiguration,
                         steps: Int, cropTo source: (w: Int, h: Int)?) {
        work?.cancel()
        savedURL = nil
        notice = nil
        let token = CancellationToken()
        cancelToken = token
        status = .generating(step: 0, total: steps)
        work = Task {
            do {
                let start = ContinuousClock.now
                let result = try await pipeline.generateImages(configuration: config) { @Sendable progress in
                    let s = progress.step, t = progress.totalSteps
                    token.stepFinished(s)
                    Task { @MainActor in
                        if case .generating = self.status { self.status = .generating(step: s, total: t) }
                    }
                    return !token.isCancelled
                }
                // Stop: the pipeline finishes the step it is on and still decodes; that unfinished
                // image is dropped. (A Stop during the decode keeps the finished image.)
                if token.isCancelled, token.stepsDone < steps {
                    self.notice = "Stopped after step \(token.stepsDone) of \(steps). The unfinished image was not kept."
                    self.status = .ready
                    return
                }
                self.generateSeconds = Self.seconds(since: start)
                var cg = result.images.first
                // Crop the square result back to the reference's aspect ratio (Edit only).
                if let out = cg, let s = source, s.w != s.h {
                    cg = Self.cropToAspect(out, aspectW: s.w, aspectH: s.h)
                }
                self.image = cg
                if let cg { self.imageSize = "\(cg.width)×\(cg.height)" }
                self.exportURL = Self.writeTempPNG(cg)
                self.status = .ready
            } catch {
                self.status = token.isCancelled ? .ready : .error(error.localizedDescription)
            }
        }
        holdActivity("Generating an image")
    }

    /// GLM-Image generation (text→image only). Drives the bespoke AR→DiT→VAE pipeline. The UI's
    /// Steps (default 20 — quality reference; ~12 is a faster preview) and Guidance (1.5) are
    /// honored via a dynamically computed flow-match schedule.
    /// Z-Image-Turbo: negated CFG + FlowMatchEuler over one dynamic-shape DiT graph.
    private func generateZImage(prompt: String, negativePrompt: String,
                                steps: Int, guidance: Float, seed: UInt32) {
        guard let zimage, canGenerate else { return }
        work?.cancel()
        savedURL = nil
        let token = CancellationToken()
        cancelToken = token
        status = .generating(step: 0, total: steps)
        let side = zSide
        work = Task {
            do {
                let start = ContinuousClock.now
                let cg = try await zimage.generate(
                    prompt: prompt, negativePrompt: negativePrompt, side: side,
                    steps: steps, guidance: guidance, seed: UInt64(seed)) { @Sendable step, total in
                        Task { @MainActor in self.status = .generating(step: step, total: total) }
                        return !token.isCancelled
                    }
                if token.isCancelled { self.status = .ready; return }
                self.generateSeconds = Self.seconds(since: start)
                self.image = cg
                self.imageSize = "\(cg.width)×\(cg.height)"
                self.exportURL = Self.writeTempPNG(cg)
                self.status = .ready
            } catch is CancellationError {
                self.status = .ready
            } catch {
                self.status = .error(error.localizedDescription)
            }
        }
    }

    private func generateGLM(prompt: String, steps: Int, guidance: Float, seed: UInt32) {
        guard let glm, canGenerate else { return }
        work?.cancel()
        savedURL = nil
        let token = CancellationToken()
        cancelToken = token
        status = .generating(step: 0, total: steps)
        work = Task {
            do {
                let start = ContinuousClock.now
                let cg = try await glm.generate(prompt: prompt, seed: UInt64(seed),
                                                steps: steps, guidance: guidance) { @Sendable step, total in
                    Task { @MainActor in self.status = .generating(step: step, total: total) }
                    return !token.isCancelled
                }
                if token.isCancelled { self.status = .ready; return }
                self.generateSeconds = Self.seconds(since: start)
                self.image = cg
                self.imageSize = "\(cg.width)×\(cg.height)"
                self.exportURL = Self.writeTempPNG(cg)
                self.status = .ready
            } catch is CancellationError {
                self.status = .ready
            } catch {
                self.status = .error("\(error)")
            }
        }
    }

    /// Cancel the in-flight work. A FLUX.2 generation finishes the step it is on and then returns
    /// (.stopping until it does); GLM / Z-Image generation falls back to the loaded model (.ready);
    /// a cancelled download/load drops to .idle (no model yet).
    func cancel() {
        cancelToken.cancel()
        Self.setIdleTimerDisabled(false)
        switch status {
        case .generating where pipeline != nil:
            status = .stopping
        case .generating:
            work?.cancel()
            status = .ready
        case .downloading, .loading:
            work?.cancel()
            hubDownload = nil
            status = .idle
        default: break
        }
    }

    var isDownloadingOrLoading: Bool {
        switch status { case .downloading, .loading: return true; default: return false }
    }

    /// Marks the current work as user-initiated for as long as it runs. Without it macOS App-Naps
    /// the app once its window is not visible (screen locked or covered): a model download fell to
    /// ~0.1 MB/s while curl pulled 5–6 MB/s on the same Mac, and inference slows down too.
    private func holdActivity(_ reason: String) {
        guard let job = work else { return }
        Task {
            let activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: reason)
            await job.value
            ProcessInfo.processInfo.endActivity(activity)
        }
    }

    // MARK: - Saving

    /// Copy the just-generated PNG into the user's Downloads (macOS) / a share-ready temp
    /// (iOS) and return the destination. Returns nil if there's nothing to save.
    @discardableResult
    func saveImageToDownloads() -> URL? {
        guard let src = exportURL else { return nil }
        let fm = FileManager.default
        let name = "coreai-\(modelTitle.replacingOccurrences(of: " ", with: "-"))-\(Int(Date().timeIntervalSince1970)).png"
        #if os(macOS)
        let dir = (try? fm.url(for: .downloadsDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        #else
        let dir = (try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true))
            ?? fm.temporaryDirectory
        #endif
        let target = dir.appendingPathComponent(name)
        do {
            if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
            try fm.copyItem(at: src, to: target)
            savedURL = target
            return target
        } catch { return nil }
    }

    // MARK: - Helpers

    /// Letterbox an image into `side`×`side` without stretching: an aspect-fill (cover) copy
    /// fills the square as background, the whole aspect-fit image is drawn centered on top. The
    /// model then edits a coherent full-bleed square; the padded margins are cropped off after.
    private static func fitToSquare(_ image: CGImage, side: Int) -> CGImage {
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                data: nil, width: side, height: side, bitsPerComponent: 8,
                bytesPerRow: 4 * side, space: cs,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return image }
        ctx.interpolationQuality = .high
        let w = CGFloat(image.width), h = CGFloat(image.height), s = CGFloat(side)
        // Cover background (fills the square, cropping the overflow).
        let cover = max(s / w, s / h)
        let cw = w * cover, ch = h * cover
        ctx.draw(image, in: CGRect(x: (s - cw) / 2, y: (s - ch) / 2, width: cw, height: ch))
        // Fit foreground (whole image, centered).
        let fit = min(s / w, s / h)
        let fw = w * fit, fh = h * fit
        ctx.draw(image, in: CGRect(x: (s - fw) / 2, y: (s - fh) / 2, width: fw, height: fh))
        return ctx.makeImage() ?? image
    }

    /// Crop a square result back to the source aspect ratio (the centered aspect-fit region).
    private static func cropToAspect(_ square: CGImage, aspectW: Int, aspectH: Int) -> CGImage {
        let s = CGFloat(min(square.width, square.height))
        let fit = min(s / CGFloat(aspectW), s / CGFloat(aspectH))
        let fw = (CGFloat(aspectW) * fit).rounded()
        let fh = (CGFloat(aspectH) * fit).rounded()
        let x = ((CGFloat(square.width) - fw) / 2).rounded()
        let y = ((CGFloat(square.height) - fh) / 2).rounded()
        return square.cropping(to: CGRect(x: x, y: y, width: fw, height: fh)) ?? square
    }

    /// Write a CGImage to a temp PNG for sharing/saving via SwiftUI ShareLink.
    private static func writeTempPNG(_ cg: CGImage?) -> URL? {
        guard let cg else { return nil }
        let name = "coreai-image-\(UInt32.random(in: 0 ... .max)).png"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        return CGImageDestinationFinalize(dest) ? url : nil
    }

    private static func seconds(since start: ContinuousClock.Instant) -> Double {
        let d = ContinuousClock.now - start
        let (secs, atto) = d.components
        return Double(secs) + Double(atto) / 1e18
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "DiffusionEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}

// MARK: - GLM-Image (AR + diffusion hybrid) pipeline

/// GLM-Image (zai-org, MIT, 16B) can't use the high-level CoreAIDiffusionPipeline auto-detect path:
/// it's an autoregressive GLM-4-9B visual-token generator feeding a flow-matching DiT. This drives
/// the three Core AI bundles + a host loop directly — a faithful port of ondevice/GlmImageRunner,
/// which was verified numerically exact vs the Python reference (PSNR 69 dB, pixel-identical with
/// matched prior+noise) and byte-exact on tokenization.
///
/// A GLM bundle folder holds:  glm_image_ar…aimodelc · glm_image_dit…aimodelc · glm_image_vae…aimodel
///                             · a tokenizer dir (tokenizer.json) · ehs.f32 (empty-glyph text embed [1472])
@MainActor
final class GlmImagePipeline {
    // AR / mrope
    private let nLayers = 40, nKV = 2, headDim = 128, kvSeq = 2048, rot = 64, visionVocab = 16512
    private let mrope = [8, 12, 12]
    private let theta = 10000.0
    private lazy var inv: [Double] = (0..<rot / 2).map { 1.0 / pow(theta, Double(2 * $0) / Double(rot)) }

    // Per-size config (glyph-free T2I). GLM-Image's 512 subset vs its native 1024, selected at
    // load time from the DiT graph's latent shape. 1024 emits a 32×32 large prior grid
    // (→ 2×-upsampled 64×64 = 4096 tokens, 4× the 512 path) at 2× spatial resolution = much
    // sharper. Suffix = <sop>H/32 W/32<eop> coarse + <sop>16 16<eop> fine + IMG_START; verified
    // byte-exact vs the HF GlmImageProcessor.
    struct SizeCfg { let side: Int; let suffix: [Int]; let grids: [(Int, Int)] }
    static let cfg512 = SizeCfg(
        side: 512,
        suffix: [167845, 115829, 16732, 115829, 167846, 167845, 115829, 16732, 115829, 167846, 16384],
        grids: [(16, 16), (16, 16)])
    static let cfg1024 = SizeCfg(
        side: 1024,
        suffix: [167845, 117687, 16732, 117687, 167846, 167845, 115829, 16732, 115829, 167846, 16384],
        grids: [(32, 32), (16, 16)])

    /// Flow-match schedule for any step count (verified vs the diffusers scheduler, max diff 5e-7):
    /// raw integer timesteps trunc(linspace(1000,1,steps+1)) condition the DiT (adaLN — must stay
    /// UNSHIFTED); the latent euler steps use the mu-shifted sigmas σ' = μ/(μ + (1/σ − 1)) with the
    /// resolution shift μ = 0.75·side/256 + 0.25 (1.75 @512, 3.25 @1024).
    static func schedule(side: Int, steps: Int) -> (rawTs: [Float], sigmas: [Float]) {
        let mu = 0.75 * Double(side) / 256.0 + 0.25
        var rawTs: [Float] = []
        var sigmas: [Float] = []
        for i in 0..<steps {
            let t = Double(Int(1000.0 - 999.0 * Double(i) / Double(steps)))   // trunc(linspace)
            rawTs.append(Float(t))
            sigmas.append(Float(mu / (mu + (1000.0 / t - 1.0))))
        }
        sigmas.append(0)
        return (rawTs, sigmas)
    }

    private let dir: URL
    private var arFn: InferenceFunction?, arD: InferenceFunctionDescriptor?
    private var ditFn: InferenceFunction?, ditD: InferenceFunctionDescriptor?
    private var vaeFn: InferenceFunction?, vaeD: InferenceFunctionDescriptor?
    private var tokenizer: (any Tokenizer)?
    private var ehs: [Float] = []
    private var cfg = GlmImagePipeline.cfg512   // replaced from the loaded DiT's latent shape in load()
    var side: Int { cfg.side }

    init(dir: URL) { self.dir = dir }

    /// True if `url` looks like a GLM-Image bundle (has an `…ar….aimodelc`).
    static func looksLikeGLM(_ url: URL) -> Bool { resolve(in: url, contains: "ar", ext: "aimodelc") != nil }

    // MARK: Loading

    func load() async throws {
        guard let arURL = Self.resolve(in: dir, contains: "ar", ext: "aimodelc"),
              let ditURL = Self.resolve(in: dir, contains: "dit", ext: "aimodelc"),
              let vaeURL = Self.resolve(in: dir, contains: "vae", ext: "aimodel"),
              let tokURL = Self.resolveTokenizer(in: dir),
              let ehsURL = Self.resolveFile(in: dir, suffix: ".f32")
        else { throw Self.err("GLM bundle incomplete (need ar/dit .aimodelc, vae .aimodel, tokenizer, ehs.f32)") }

        (arFn, arD) = try await Self.loadModel(arURL, cpu: false)
        (ditFn, ditD) = try await Self.loadModel(ditURL, cpu: false)
        // Pick 512/1024 config from the DiT's latent side (hidden_states = [1,16,lat,lat], lat = size/8).
        if let dd = ditD {
            cfg = (descriptor(dd, "hidden_states", .input).shape[2] * 8 >= 1024) ? Self.cfg1024 : Self.cfg512
        }
        (vaeFn, vaeD) = try await Self.loadModel(vaeURL, cpu: true) // fp16 VAE overflows → fp32 CPU
        tokenizer = try await AutoTokenizer.from(modelFolder: tokURL)
        let data = try Data(contentsOf: ehsURL)
        ehs = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard ehs.count == 1472 else { throw Self.err("ehs \(ehs.count) != 1472") }
    }

    // MARK: Generation

    /// prompt → 512×512 image. `progress(step, total)` returns false to cancel (checked each DiT step).
    func generate(prompt: String, seed: UInt64, steps: Int = 20, guidance: Float = 1.5,
                  progress: @escaping @Sendable (Int, Int) -> Bool) async throws -> CGImage {
        let steps = max(4, min(steps, 100))   // sane clamp for the UI stepper
        guard let tokenizer, let arFn, let arD, let ditFn, let ditD, let vaeFn, let vaeD else {
            throw Self.err("GLM pipeline not loaded")
        }
        // 1) AR inputs (tokenize + size-specific suffix + host 3D positions).
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: false) + cfg.suffix
        let L = ids.count
        let (decT, decH, decW) = decodePositions(startPos: L)
        let gs = cfg.grids.map { $0.0 * $0.1 }
        let maxNew = gs.reduce(0, +) + 1, offset = gs.dropFirst().reduce(0, +)
        let (th, tw) = cfg.grids[0]

        // 2) AR prefill + sampled decode → prior[1024].
        var rng = RNG(s: seed)
        var key = alloc(arD, "keyCache", stateShape(arD, "keyCache", dyn: kvSeq), .state)
        var val = alloc(arD, "valueCache", stateShape(arD, "valueCache", dyn: kvSeq), .state)
        fillF(&key, [Float](repeating: 0, count: key.shape.reduce(1, *)))
        fillF(&val, [Float](repeating: 0, count: val.shape.reduce(1, *)))

        func arStep(token: Int, t: Int, h: Int, w: Int, idx: Int) async throws -> [Float] {
            let (cs, sn) = arCosSin(t: t, h: h, w: w)
            var idA = alloc(arD, "input_ids", [1, 1], .input); fillI(&idA, [Int32(token)])
            var posA = alloc(arD, "position_ids", [1, idx + 1], .input); fillI(&posA, (0...idx).map { Int32($0) })
            var cosA = alloc(arD, "cos", [1, 1, rot], .input); fillF(&cosA, cs)
            var sinA = alloc(arD, "sin", [1, 1, rot], .input); fillF(&sinA, sn)
            var lg = alloc(arD, "logits", [1, 1, visionVocab], .output)
            var st = InferenceFunction.MutableViews(); st.insert(&key, for: "keyCache"); st.insert(&val, for: "valueCache")
            var ov = InferenceFunction.MutableViews(); ov.insert(&lg, for: "logits")
            _ = try await arFn.run(inputs: ["input_ids": idA, "position_ids": posA, "cos": cosA, "sin": sinA],
                                   states: consume st, outputViews: consume ov)
            return flattenAsFloat(lg)
        }

        var logits = [Float]()
        for i in 0..<L { logits = try await arStep(token: ids[i], t: i, h: i, w: i, idx: i) }  // prefill = ramp
        var gen = [sampleToken(logits, &rng)]
        for k in 0..<(maxNew - 1) {
            logits = try await arStep(token: gen[gen.count - 1], t: decT[k], h: decH[k], w: decW[k], idx: L + k)
            gen.append(sampleToken(logits, &rng))
        }
        // 2nd (large) block → th×tw → 2× nearest upsample → prior[1024].
        let block = Array(gen[offset..<(offset + th * tw)])
        var prior = [Int32](); prior.reserveCapacity(th * tw * 4)
        for r in 0..<(2 * th) { for c in 0..<(2 * tw) { prior.append(Int32(block[(r / 2) * tw + (c / 2)])) } }

        // 3) DiT 20-step CFG flow-match euler.
        let latS = side / 8, latN = 16 * latS * latS
        let (cos, sin) = ditRope(latH: latS, latW: latS)
        let tgt: [Float] = [Float(side), Float(side)], crop: [Float] = [0, 0]
        var lat = gaussian(latN, seed: seed)

        func dit(_ lat: [Float], ts: Float, scale: Float) async throws -> [Float] {
            var h = alloc(ditD, "hidden_states", [1, 16, latS, latS], .input); fillF(&h, lat)
            var e = alloc(ditD, "encoder_hidden_states", [1, 1, ehs.count], .input); fillF(&e, ehs)
            var pt = alloc(ditD, "prior_token_id", [1, prior.count], .input); fillI(&pt, prior)
            var ps = alloc(ditD, "prior_scale", [1, 1, 1], .input); fillF(&ps, [scale])
            var t = alloc(ditD, "timestep", [1], .input); fillF(&t, [ts])
            var tg = alloc(ditD, "target_size", [1, 2], .input); fillF(&tg, tgt)
            var cr = alloc(ditD, "crop_coords", [1, 2], .input); fillF(&cr, crop)
            var co = alloc(ditD, "cos", [latS / 2 * latS / 2, 128], .input); fillF(&co, cos)
            var si = alloc(ditD, "sin", [latS / 2 * latS / 2, 128], .input); fillF(&si, sin)
            var noise = alloc(ditD, "noise", [1, 16, latS, latS], .output)
            var ov = InferenceFunction.MutableViews(); ov.insert(&noise, for: "noise")
            _ = try await ditFn.run(
                inputs: ["hidden_states": h, "encoder_hidden_states": e, "prior_token_id": pt, "prior_scale": ps,
                         "timestep": t, "target_size": tg, "crop_coords": cr, "cos": co, "sin": si],
                states: InferenceFunction.MutableViews(), outputViews: consume ov)
            return flattenAsFloat(noise)
        }

        // Schedule: RAW integer timesteps condition the DiT (adaLN — must stay unshifted; feeding
        // the mu-shifted value skews the time conditioning every step → resolution-dependent color
        // drift). The latent euler steps use the mu-shifted sigmas. See `schedule(side:steps:)`.
        let (rawTs, sigmas) = Self.schedule(side: side, steps: steps)
        for i in 0..<steps {
            if !progress(i, steps) { throw CancellationError() }
            let ts = rawTs[i] - 1
            let nc = try await dit(lat, ts: ts, scale: 1)
            let nu = try await dit(lat, ts: ts, scale: 0)
            let dsig = sigmas[i + 1] - sigmas[i]
            for k in 0..<latN { lat[k] += dsig * (nu[k] + guidance * (nc[k] - nu[k])) }
        }

        // 4) VAE decode (fp32 CPU) → image[1,3,H,W].
        var z = alloc(vaeD, "z", [1, 16, latS, latS], .input); fillF(&z, lat)
        var img = alloc(vaeD, "image", [1, 3, side, side], .output)
        var ov = InferenceFunction.MutableViews(); ov.insert(&img, for: "image")
        _ = try await vaeFn.run(inputs: ["z": z], states: InferenceFunction.MutableViews(), outputViews: consume ov)
        guard let cg = Self.makeCGImage(rgb: flattenAsFloat(img), side: side) else { throw Self.err("VAE→CGImage failed") }
        return cg
    }

    // MARK: Host math (ported from GlmImageRunner)

    /// decode 3D-mrope positions: grids processed in reverse; t = block base, h += row, w += col, + end.
    private func decodePositions(startPos: Int) -> (t: [Int], h: [Int], w: [Int]) {
        var tl = [Int](), hl = [Int](), wl = [Int](); var dp = startPos
        for i in 1...cfg.grids.count {
            let (h, w) = cfg.grids[cfg.grids.count - i]
            for r in 0..<h { for c in 0..<w { tl.append(dp); hl.append(dp + r); wl.append(dp + c) } }
            dp += max(h, w)
        }
        tl.append(dp); hl.append(dp); wl.append(dp)
        return (tl, hl, wl)
    }

    /// AR mrope cos/sin for one 3D position → [64] each. channel k: axis = k<8 ? t : k<20 ? h : w.
    private func arCosSin(t: Int, h: Int, w: Int) -> (cos: [Float], sin: [Float]) {
        var cs = [Float](repeating: 0, count: rot), sn = cs
        for k in 0..<(rot / 2) {
            let axis = k < mrope[0] ? t : (k < mrope[0] + mrope[1] ? h : w)
            let ang = Double(axis) * inv[k]
            let c = Float(Foundation.cos(ang)), s = Float(Foundation.sin(ang))
            cs[k] = c; cs[k + rot / 2] = c; sn[k] = s; sn[k + rot / 2] = s
        }
        return (cs, sn)
    }

    /// DiT rope cos/sin per patch → [hw][128] flattened; j<32:r·inv, 32..63:c·inv, 64..95:r·inv, 96..127:c·inv.
    private func ditRope(latH: Int, latW: Int) -> (cos: [Float], sin: [Float]) {
        let h = latH / 2, w = latW / 2, dim = 128
        var cs = [Float](repeating: 0, count: h * w * dim), sn = cs
        for r in 0..<h {
            for c in 0..<w {
                let base = (r * w + c) * dim
                for j in 0..<dim {
                    let (axis, fi): (Int, Int) = j < 32 ? (r, j) : j < 64 ? (c, j - 32) : j < 96 ? (r, j - 64) : (c, j - 96)
                    let ang = Double(axis) * inv[fi]
                    cs[base + j] = Float(Foundation.cos(ang)); sn[base + j] = Float(Foundation.sin(ang))
                }
            }
        }
        return (cs, sn)
    }

    /// temp 0.9 / top-p 0.75 (GLM-Image generation_config). Greedy collapses visual tokens to flat
    /// regions; sampling restores detail. Inclusive top-p prefix + always the top token.
    private func sampleToken(_ logits: [Float], _ rng: inout RNG, temp: Double = 0.9, topP: Double = 0.75) -> Int {
        let n = logits.count
        var mx = -Double.infinity
        for v in logits { let d = Double(v) / temp; if d > mx { mx = d } }
        var p = [Double](repeating: 0, count: n); var sum = 0.0
        for i in 0..<n { let e = Foundation.exp(Double(logits[i]) / temp - mx); p[i] = e; sum += e }
        for i in 0..<n { p[i] /= sum }
        let order = Array(0..<n).sorted { p[$0] > p[$1] }
        var kept = [Int](); var csum = 0.0
        for (rank, idx) in order.enumerated() { csum += p[idx]; if rank == 0 || csum <= topP { kept.append(idx) } else { break } }
        var kp = kept.map { p[$0] }; let ks = kp.reduce(0, +)
        for i in 0..<kp.count { kp[i] /= ks }
        let r = rng.next(); var acc = 0.0
        for i in 0..<kp.count { acc += kp[i]; if r <= acc { return kept[i] } }
        return kept.last!
    }

    /// Box–Muller over SplitMix64 (host noise; deterministic per seed).
    private func gaussian(_ n: Int, seed: UInt64) -> [Float] {
        var rng = RNG(s: seed &+ 0xD1B54A32D192ED03)
        var out = [Float](repeating: 0, count: n); var i = 0
        while i < n {
            let u1 = max(rng.next(), 1e-12), u2 = rng.next()
            let r = (-2 * Foundation.log(u1)).squareRoot()
            out[i] = Float(r * Foundation.cos(2 * Double.pi * u2)); i += 1
            if i < n { out[i] = Float(r * Foundation.sin(2 * Double.pi * u2)); i += 1 }
        }
        return out
    }

    struct RNG {
        var s: UInt64
        mutating func next() -> Double {
            s &+= 0x9E3779B97F4A7C15
            var z = s
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            z ^= z >> 31
            return Double(z >> 11) / Double(1 << 53)
        }
    }

    // MARK: NDArray helpers

    private enum IOKind { case input, state, output }
    private func descriptor(_ d: InferenceFunctionDescriptor, _ name: String, _ k: IOKind) -> NDArrayDescriptor {
        let io = switch k {
        case IOKind.input: d.inputDescriptor(of: name)
        case .state: d.stateDescriptor(of: name)
        case .output: d.outputDescriptor(of: name)
        }
        guard case .ndArray(let nd)? = io else { fatalError("\(name) not ndarray") }
        return nd
    }
    private func alloc(_ d: InferenceFunctionDescriptor, _ name: String, _ shape: [Int], _ k: IOKind) -> NDArray {
        NDArray(descriptor: descriptor(d, name, k).resolvingDynamicDimensions(shape))
    }
    private func stateShape(_ d: InferenceFunctionDescriptor, _ name: String, dyn: Int) -> [Int] {
        descriptor(d, name, .state).shape.map { $0 < 0 ? dyn : $0 }
    }
    private func fillF(_ a: inout NDArray, _ v: [Float]) {
        switch a.scalarType {
        case .float16: fillNDArray(&a, as: Float16.self, with: v.map { Float16($0) })
        case .float32: fillNDArray(&a, as: Float.self, with: v)
        default: fatalError("fillF on \(a.scalarType)")
        }
    }
    private func fillI(_ a: inout NDArray, _ v: [Int32]) { fillNDArray(&a, as: Int32.self, with: v) }

    // MARK: Static helpers

    private static func loadModel(_ url: URL, cpu: Bool) async throws -> (InferenceFunction, InferenceFunctionDescriptor) {
        // AOT-compiled .aimodelc must load with its baked delegate (.default). Forcing a compute
        // unit (preferredComputeUnitKind:) re-specializes the graph via JIT, which wedges large
        // OS27 graphs → failedToSpecialize. (.default has preferredComputeUnitKind == nil; the
        // AR/DiT were AOT-compiled for GPU h16c, the VAE runs CPU-only in fp32.)
        let opts = cpu ? SpecializationOptions.cpuOnly : SpecializationOptions.default
        // Load the REAL path, not a symlink: AIModel(contentsOf:) does not follow a symlinked
        // bundle to its arch-specific AOT delegate subdir (main-h16c-delegates), so a symlinked
        // bundle fails with `failedToSpecialize`. resolvingSymlinksInPath fixes it.
        let m = try await AIModel(contentsOf: url.resolvingSymlinksInPath(), options: opts)
        guard let d = m.functionDescriptor(for: "main"), let f = try m.loadFunction(named: "main") else {
            throw Self.err("no 'main' function in \(url.lastPathComponent)")
        }
        return (f, d)
    }

    /// Image[3,H,W] (~[-1,1]) → clip(v/2+0.5) → RGBA8 → CGImage.
    private static func makeCGImage(rgb: [Float], side: Int) -> CGImage? {
        let plane = side * side
        var px = [UInt8](repeating: 255, count: plane * 4)
        for i in 0..<plane {
            for c in 0..<3 {
                let v = min(max(rgb[c * plane + i] / 2 + 0.5, 0), 1)
                px[i * 4 + c] = UInt8((v * 255).rounded())
            }
        }
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        return px.withUnsafeMutableBytes { buf in
            CGContext(data: buf.baseAddress, width: side, height: side, bitsPerComponent: 8,
                      bytesPerRow: 4 * side, space: cs,
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
        }
    }

    private static func resolve(in dir: URL, contains: String, ext: String) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return items.first { $0.pathExtension == ext && $0.lastPathComponent.lowercased().contains(contains) }
    }
    private static func resolveFile(in dir: URL, suffix: String) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return items.first { $0.lastPathComponent.lowercased().hasSuffix(suffix) }
    }
    /// A subdir named tokenizer/processor holding tokenizer.json, else the folder itself if it has one.
    private static func resolveTokenizer(in dir: URL) -> URL? {
        let fm = FileManager.default
        for name in ["tokenizer", "processor"] {
            let sub = dir.appendingPathComponent(name, isDirectory: true)
            if fm.fileExists(atPath: sub.appendingPathComponent("tokenizer.json").path) { return sub }
        }
        return fm.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) ? dir : nil
    }

    private static func err(_ msg: String) -> Error {
        NSError(domain: "GlmImagePipeline", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
