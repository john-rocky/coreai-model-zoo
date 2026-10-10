import CoreGraphics
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

/// Generation mode. Text→Image (pure prompt) and Edit (FLUX.2 reference-token image-to-image:
/// the reference is kept, the prompt is the edit instruction).
private enum GenMode: String, CaseIterable, Identifiable {
    case textToImage = "Text → Image"
    case edit = "Edit"
    var id: String { rawValue }
    var actionLabel: String { self == .edit ? "Apply Edit" : "Generate" }
    var actionIcon: String { self == .edit ? "wand.and.rays" : "sparkles" }
}

struct ContentView: View {
    @StateObject private var engine = DiffusionEngine()

    // Optional: the iOS build ships an empty hosted catalog (load via "Local…").
    @State private var selectedModel = DiffusionEngine.defaultSelection
    @State private var prompt = "a watercolor painting of a red fox reading a book by candlelight, cozy, detailed"
    @State private var negativePrompt = ""
    @State private var steps = DiffusionEngine.catalog.first?.defaultSteps ?? 4
    @State private var guidance = DiffusionEngine.catalog.first?.defaultGuidance ?? 1.0
    @State private var seedText = "42"
    @State private var showingFolderImporter = false

    // Edit: the reference image the instruction edits.
    @State private var mode: GenMode = .textToImage
    @State private var inputImage: CGImage?
    @State private var showingImageImporter = false

    /// Modes the loaded model supports: Edit needs a VAE encoder and a traced image-to-image
    /// graph (FLUX.2 int8 has both). Text→Image is always available.
    private var availableModes: [GenMode] {
        engine.supportsEdit ? [.textToImage, .edit] : [.textToImage]
    }

    var body: some View {
        #if os(macOS)
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    group("Model") { modelControls }
                    if availableModes.count > 1 { group("Mode") { modeControls } }
                    group(mode == .edit ? "Instruction" : "Prompt") { promptControls(maxWidth: nil) }
                    group("Settings") { settingsControls }
                    generateButton
                }
                .padding(18)
            }
            .frame(minWidth: 320, idealWidth: 350, maxWidth: 440)
            .fileImporter(isPresented: $showingFolderImporter, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { engine.loadLocal(url) }
            }
            .fileImporter(isPresented: $showingImageImporter, allowedContentTypes: [.image]) { result in
                if case .success(let url) = result { loadInputImage(from: url) }
            }
            .onChange(of: engine.supportsEdit) { _, _ in
                if !availableModes.contains(mode) { mode = .textToImage }
            }
            canvas.frame(minWidth: 460)
        }
        .task { await runAutoplay() }
        #else
        NavigationStack {
            // GeometryReader gives a CONCRETE content width. A `TextField(axis: .vertical)`
            // reports its full single-line intrinsic width upward, which inside a Form/List
            // blows the whole column wider than the screen (content overflows both edges).
            // Capping the text fields at this measured width — not `.infinity` — keeps the
            // reported width bounded, so everything stays inside the device.
            GeometryReader { geo in
                let contentWidth = geo.size.width - 32
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        canvas
                            .frame(maxWidth: .infinity)
                            .frame(height: 230)
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                        card("Model") { modelControls }
                        if availableModes.count > 1 { card("Mode") { modeControls } }
                        card(mode == .edit ? "Instruction" : "Prompt") { promptControls(maxWidth: contentWidth) }
                        card("Settings") { settingsControls }
                    }
                    .padding(16)
                    .frame(width: geo.size.width, alignment: .leading)
                }
                .scrollDismissesKeyboard(.interactively)
                .safeAreaInset(edge: .bottom) {
                    generateButton
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.bar)
                }
            }
            .navigationTitle("CoreAI Image Gen")
            .navigationBarTitleDisplayMode(.inline)
            .fileImporter(isPresented: $showingFolderImporter, allowedContentTypes: [.folder]) { result in
                if case .success(let url) = result { engine.loadLocal(url) }
            }
            .fileImporter(isPresented: $showingImageImporter, allowedContentTypes: [.image]) { result in
                if case .success(let url) = result { loadInputImage(from: url) }
            }
            .onChange(of: engine.supportsEdit) { _, _ in
                if !availableModes.contains(mode) { mode = .textToImage }
            }
        }
        #endif
    }

    // MARK: - Control content (shared; wrapped in `group`/`card` per platform)

    @ViewBuilder private var modelControls: some View {
        if !DiffusionEngine.catalog.isEmpty {
            Picker("Model", selection: $selectedModel) {
                ForEach(DiffusionEngine.catalog) { Text($0.title).tag(Optional($0)) }
            }
            #if os(macOS)
            .labelsHidden()
            #else
            .pickerStyle(.menu)
            #endif
            .disabled(engine.status.isBusy)
        }

        HStack {
            if let model = selectedModel {
                Button { downloadAndLoad(model) } label: {
                    Label("Download & Load", systemImage: "arrow.down.circle")
                }
                .disabled(engine.status.isBusy)
            }
            // Local… stays enabled even mid-download — tapping it cancels the transfer first.
            Button {
                if engine.isDownloadingOrLoading { engine.cancel() }
                showingFolderImporter = true
            } label: {
                Label("Local…", systemImage: "folder")
            }
        }

        if engine.isDownloadingOrLoading {
            Button(role: .destructive) { engine.cancel() } label: {
                Label("Cancel download", systemImage: "xmark.circle")
            }
        }

        statusLine
    }

    // Mode switch + (in Edit) the reference picker. Shown only when the loaded model can edit.
    @ViewBuilder private var modeControls: some View {
        Picker("Mode", selection: $mode) {
            ForEach(availableModes) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .disabled(engine.status.isBusy)

        if mode == .edit { referenceImageControls }
    }

    @ViewBuilder private var referenceImageControls: some View {
        // Reference preview — full width at its true aspect ratio, tap to (re)choose.
        Group {
            if let cg = inputImage {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .frame(maxHeight: 240)
            } else {
                RoundedRectangle(cornerRadius: 10)
                    .fill(Color(white: 0.12))
                    .frame(height: 120)
                    .overlay {
                        VStack(spacing: 6) {
                            Image(systemName: "photo.badge.plus").font(.title2)
                            Text("Choose a reference image").font(.caption)
                        }
                        .foregroundStyle(.secondary)
                    }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .contentShape(Rectangle())
        .onTapGesture { if !engine.status.isBusy { showingImageImporter = true } }

        Button { showingImageImporter = true } label: {
            Label(inputImage == nil ? "Choose Image…" : "Change Image…", systemImage: "photo")
                .frame(maxWidth: .infinity)
        }
        .disabled(engine.status.isBusy)

        Text("The reference is kept; the instruction edits it (e.g. \"add a red hat\", \"make it night\").")
            .font(.caption2).foregroundStyle(.secondary)
    }

    @ViewBuilder private func promptControls(maxWidth: CGFloat?) -> some View {
        TextField("Prompt", text: $prompt, axis: .vertical)
            .lineLimit(2...6)
            .frame(maxWidth: maxWidth ?? .infinity, alignment: .leading)
        TextField("Negative prompt (optional)", text: $negativePrompt, axis: .vertical)
            .lineLimit(1...3)
            .foregroundStyle(.secondary)
            .frame(maxWidth: maxWidth ?? .infinity, alignment: .leading)
    }

    @ViewBuilder private var settingsControls: some View {
        Stepper("Steps: \(steps)", value: $steps, in: 1...50)
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Guidance")
                Spacer()
                Text(String(format: "%.1f", guidance)).monospacedDigit().foregroundStyle(.secondary)
            }
            Slider(value: $guidance, in: 0...10)
        }
        HStack {
            Text("Seed")
            TextField("seed", text: $seedText)
                .multilineTextAlignment(.trailing)
                .monospacedDigit()
                #if os(iOS)
                .keyboardType(.numberPad)
                #endif
            Button { seedText = String(UInt32.random(in: 0 ... .max)) } label: {
                Image(systemName: "die.face.5")
            }
            .buttonStyle(.borderless)
        }
    }

    private var statusLine: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                if engine.status.isBusy { ProgressView().controlSize(.small) }
                Text(engine.status.label)
                    .font(.caption).foregroundStyle(statusColor).lineLimit(2)
                Spacer()
            }
            if let notice = engine.notice {
                Text(notice).font(.caption).foregroundStyle(.secondary).lineLimit(3)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var statusColor: Color {
        if case .error = engine.status { return .red }
        if case .ready = engine.status { return .green }
        return .secondary
    }

    private var generateButton: some View {
        Group {
            switch engine.status {
            case .generating:
                Button(role: .destructive) { engine.cancel() } label: {
                    Label("Stop", systemImage: "stop.fill").frame(maxWidth: .infinity)
                }
            case .stopping:
                Button {} label: {
                    Label("Stopping…", systemImage: "stop.fill").frame(maxWidth: .infinity)
                }
                .disabled(true)
            default:
                Button { pressGenerate() } label: {
                    Label(mode.actionLabel, systemImage: mode.actionIcon).frame(maxWidth: .infinity)
                }
                .disabled(!canPressGenerate)
            }
        }
        .controlSize(.large)
        .buttonStyle(.borderedProminent)
    }

    /// The Download & Load button.
    private func downloadAndLoad(_ model: DiffusionEngine.ModelOption) {
        steps = model.defaultSteps
        guidance = model.defaultGuidance
        engine.loadFromHub(model)
    }

    /// Generate / Apply Edit is pressable only with a loaded model, a prompt, and (in Edit) a reference.
    private var canPressGenerate: Bool {
        engine.canGenerate
            && !prompt.trimmingCharacters(in: .whitespaces).isEmpty
            && (mode != .edit || (inputImage != nil && engine.supportsEdit))
    }

    /// The Generate / Apply Edit button.
    private func pressGenerate() {
        let seedValue = UInt32(seedText) ?? 42
        if mode == .edit, let ref = inputImage {
            engine.edit(referenceImage: ref, instruction: prompt, steps: steps, guidance: guidance, seed: seedValue)
        } else {
            engine.generate(
                prompt: prompt, negativePrompt: negativePrompt,
                steps: steps, guidance: guidance, seed: seedValue)
        }
    }

    // MARK: - Canvas (shared)

    private var canvas: some View {
        ZStack {
            Color(white: 0.09)
            if let cg = engine.image {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .interpolation(.high)
                    .aspectRatio(contentMode: .fit)
                    .padding(12)
            } else {
                placeholder
            }

            if case .generating(let s, let t) = engine.status {
                progressOverlay(value: Double(s), total: Double(max(t, 1)))
            } else if case .downloading = engine.status {
                VStack {
                    Spacer()
                    Group {
                        if let hub = engine.hubDownload {
                            HubDownloadBar(progress: hub)
                        } else {
                            DownloadBar(downloader: engine.downloader)
                        }
                    }
                    .padding(12)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
                    .padding(20)
                }
            }
        }
        .overlay(alignment: .topTrailing) {
            HStack(spacing: 12) {
                if let secs = engine.generateSeconds, engine.image != nil {
                    Text("\(engine.imageSize) · \(String(format: "%.1fs", secs))")
                        .font(.caption).foregroundStyle(.white.opacity(0.7))
                }
                if let url = engine.exportURL {
                    Button {
                        if let saved = engine.saveImageToDownloads() { reveal(saved) }
                    } label: {
                        Image(systemName: engine.savedURL != nil
                              ? "checkmark.circle.fill" : "square.and.arrow.down")
                    }
                    .help("Save the image to Downloads")
                    ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                }
            }
            .buttonStyle(.borderless)
            .padding(10)
        }
    }

    /// Decode a user-picked image file into a CGImage for the Edit reference.
    /// Any resolution or aspect ratio works: the engine letterboxes it into the model's square.
    private func loadInputImage(from url: URL) {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return }
        inputImage = cg
    }

    /// Reveal a saved file in Finder (macOS); no-op elsewhere.
    private func reveal(_ url: URL) {
        #if os(macOS)
        NSWorkspace.shared.activateFileViewerSelecting([url])
        #endif
    }

    private func progressOverlay(value: Double, total: Double) -> some View {
        VStack {
            Spacer()
            ProgressView(value: value, total: total) {
                Text(engine.status.label).font(.caption)
            }
            .padding(12)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
            .padding(20)
        }
    }

    private var placeholder: some View {
        VStack(spacing: 10) {
            Image(systemName: "photo.artframe")
                .font(.system(size: 46)).foregroundStyle(.tertiary)
            Text(placeholderText)
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
    }

    private var placeholderText: String {
        switch engine.status {
        case .idle:
            return DiffusionEngine.catalog.isEmpty
                ? "Tap Local… to load a Core AI diffusion bundle (FLUX.2, Sana Sprint, GLM-Image or Z-Image)."
                : "Pick a model and tap Download & Load to begin."
        case .downloading: return "Downloading the converted bundle from Hugging Face — a few GB, cached after the first run."
        case .loading: return "Loading the model into the Core AI runtime…"
        case .ready: return "Enter a prompt and tap Generate."
        case .error(let m): return m
        case .generating, .stopping: return ""
        }
    }

    @ViewBuilder
    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
    }

    #if os(iOS)
    @ViewBuilder
    private func card<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground),
                    in: RoundedRectangle(cornerRadius: 14))
    }
    #endif
}

/// Download progress, observing the shared downloader directly so the bar and byte
/// counter advance as chunks land (the engine only owns the high-level phase).
private struct DownloadBar: View {
    @ObservedObject var downloader: ModelDownloader

    var body: some View {
        VStack(spacing: 6) {
            ProgressView(value: downloader.fraction)
            Text(downloader.detail.isEmpty ? "starting…" : downloader.detail)
                .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
        }
    }
}

/// FLUX.2 folder download progress (HubApi, one file at a time).
private struct HubDownloadBar: View {
    let progress: DiffusionEngine.HubProgress

    var body: some View {
        VStack(spacing: 6) {
            ProgressView(value: Double(progress.done), total: Double(max(progress.total, 1)))
            Text("\(Self.bytes(progress.done)) / \(Self.bytes(progress.total))")
                .font(.caption2).monospacedDigit().foregroundStyle(.secondary)
        }
    }

    private static func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }
}

#if os(macOS)
// MARK: - Hands-off runs (checks, timings, screenshots)

extension ContentView {
    /// The `-autoplay 1` sequence (launch arguments: Autoplay.swift). It presses the same controls a
    /// person would, through the same functions; nothing here is a separate code path.
    fileprivate func runAutoplay() async {
        guard Autoplay.claim() else { return }
        var report = RunReport()
        var outcome = "DONE"
        do {
            try await Autoplay.pause()
            if let path = LaunchOptions.local {
                Telemetry.line("TAP Local… folder=\(path)")
                engine.loadLocal(URL(filePath: (path as NSString).expandingTildeInPath))
            } else {
                let key = LaunchOptions.model ?? DiffusionEngine.catalog.first?.key ?? ""
                guard let option = DiffusionEngine.catalog.first(where: { $0.key == key }) else {
                    throw AutoplayError("-model \(key) is not one of: "
                        + DiffusionEngine.catalog.map(\.key).joined(separator: ", "))
                }
                Telemetry.line("PICK model=\(key)")
                selectedModel = option
                try await Autoplay.pause()
                Telemetry.line("TAP Download & Load")
                downloadAndLoad(option)
            }
            try await Autoplay.until("the download and the load", timeout: 3600) { !engine.status.isBusy }
            report.loads.append(.init(
                title: engine.modelTitle, status: engine.status.label, seconds: engine.loadSeconds,
                image_size: engine.imageSize, edit: engine.supportsEdit))
            Telemetry.line("LOADED status=\(engine.status.label) seconds=\(engine.loadSeconds ?? -1) size=\(engine.imageSize) edit=\(engine.supportsEdit)")
            guard engine.canGenerate else {
                try await Autoplay.pause()
                return Autoplay.finish(report, outcome: "FAILED")
            }

            if let path = LaunchOptions.reference {
                let url = URL(filePath: (path as NSString).expandingTildeInPath)
                guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                    throw AutoplayError("cannot read the reference image \(path)")
                }
                Telemetry.line("TAP Edit, reference=\(path) \(cg.width)x\(cg.height)")
                mode = .edit
                inputImage = cg
                try await Autoplay.pause()
            }
            if let text = LaunchOptions.prompt { prompt = text }
            if let value = LaunchOptions.steps { steps = value }
            if let value = LaunchOptions.guidance { guidance = value }
            if let value = LaunchOptions.seed { seedText = value }
            Telemetry.line("TYPED mode=\(mode.rawValue) prompt=\(prompt) steps=\(steps) guidance=\(guidance) seed=\(seedText)")

            for run in 1...max(LaunchOptions.runs, 1) {
                try await Autoplay.pause()
                var record = RunReport.Generation(
                    mode: mode.rawValue, prompt: prompt, steps: steps, guidance: guidance, seed: seedText,
                    reference: mode == .edit ? LaunchOptions.reference : nil)
                guard canPressGenerate else {
                    record.error = "\(mode.actionLabel) is disabled (prompt, reference or model missing)"
                    Telemetry.line("NOT PRESSED \(record.error!)")
                    report.generations.append(record)
                    continue
                }
                Telemetry.line("TAP \(mode.actionLabel) run=\(run)")
                pressGenerate()
                if let stopStep = LaunchOptions.stopAfterStep {
                    try await Autoplay.until("step \(stopStep)", timeout: 900) {
                        if case .generating(let step, _) = engine.status { return step >= stopStep }
                        return !engine.status.isBusy
                    }
                    Telemetry.line("TAP Stop")
                    engine.cancel()
                }
                try await Autoplay.until("the image", timeout: 1800) { !engine.status.isBusy }
                record.notice = engine.notice
                if case .error(let message) = engine.status { record.error = message }
                if engine.notice == nil, record.error == nil, let image = engine.image {
                    record.seconds = engine.generateSeconds
                    Autoplay.record(image, png: engine.exportURL, run: run, into: &record)
                }
                Telemetry.line("RESULT run=\(run) seconds=\(record.seconds ?? -1) notice=\(record.notice ?? "-") error=\(record.error ?? "-") png_sha256=\(record.png_sha256 ?? "-")")
                report.generations.append(record)
            }
            try await Autoplay.pause()
        } catch {
            Telemetry.line("ERROR \(error)")
            outcome = "ERROR"
        }
        Autoplay.finish(report, outcome: outcome)
    }
}
#endif
