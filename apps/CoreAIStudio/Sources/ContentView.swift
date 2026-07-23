// ContentView.swift — CoreAIStudio's UI: a HandBrake-style batch conversion queue built on
// ConversionQueue, sequencing Upscale (AdcSR ×4) and Interpolate (RIFE ×2/×4) video jobs.
// Model loading (Download/Load Local for each engine) is still an app-level concern — the
// queue itself doesn't load models — so those controls live in a small "Models" section above
// the queue.

import AVKit
import SwiftUI
import UniformTypeIdentifiers

// MARK: - Root

struct ContentView: View {
    @StateObject private var upscaleEngine: UpscaleEngine
    @StateObject private var interpEngine: FrameInterpolationEngine
    @StateObject private var videoUpscaler: VideoUpscaler
    @StateObject private var videoInterp: VideoInterpolator
    @StateObject private var queue: ConversionQueue

    init() {
        let upscale = UpscaleEngine()
        let interp = FrameInterpolationEngine()
        let vUpscaler = VideoUpscaler(upscaler: upscale)
        let vInterp = VideoInterpolator(engine: interp)
        _upscaleEngine = StateObject(wrappedValue: upscale)
        _interpEngine = StateObject(wrappedValue: interp)
        _videoUpscaler = StateObject(wrappedValue: vUpscaler)
        _videoInterp = StateObject(wrappedValue: vInterp)
        _queue = StateObject(wrappedValue: ConversionQueue(videoUpscaler: vUpscaler, videoInterpolator: vInterp))
    }

    var body: some View {
        QueueView(
            upscaleEngine: upscaleEngine,
            interpEngine: interpEngine,
            videoUpscaler: videoUpscaler,
            videoInterp: videoInterp,
            queue: queue)
    }
}

// MARK: - Shared bits

struct StatusLine: View {
    let text: String
    let busy: Bool
    var body: some View {
        HStack(spacing: 8) {
            if busy { ProgressView().controlSize(.small) }
            Text(text).font(.footnote).foregroundStyle(.secondary)
        }
    }
}

// MARK: - Queue

struct QueueView: View {
    @ObservedObject var upscaleEngine: UpscaleEngine
    @ObservedObject var interpEngine: FrameInterpolationEngine
    @ObservedObject var videoUpscaler: VideoUpscaler
    @ObservedObject var videoInterp: VideoInterpolator
    @ObservedObject var queue: ConversionQueue

    @State private var showImporter = false
    @State private var isTargeted = false
    @State private var pendingKindChoice: PendingKindChoice = .upscale

    /// Local Hashable stand-in for QueueItemKind (which is only Equatable) so it can back a
    /// Picker selection; mapped to the real QueueItemKind at add-time via `pendingKind`.
    private enum PendingKindChoice: Hashable {
        case upscale
        case interpolate2x
        case interpolate4x
    }

    private var pendingKind: QueueItemKind {
        switch pendingKindChoice {
        case .upscale: return .upscaleVideo
        case .interpolate2x: return .interpolateVideo(multiplier: 2)
        case .interpolate4x: return .interpolateVideo(multiplier: 4)
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("CoreAIStudio").font(.title2.bold())
                Text("Upscale (AdcSR ×4) and Interpolate (RIFE ×2/×4) — queue video jobs and run them in order.")
                    .font(.subheadline).foregroundStyle(.secondary)

                modelsSection

                Divider()

                addSection

                Divider()

                queueSection

                controlsSection
            }
            .padding()
            .frame(maxWidth: 900, alignment: .leading)
        }
    }

    // MARK: Models

    private var modelsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Models").font(.headline)

            HStack {
                Text("Upscale (AdcSR)").frame(width: 130, alignment: .leading)
                Button(upscaleEngine.canUpscale ? "Model loaded" : "Download AdcSR (~1.7 GB)") {
                    upscaleEngine.loadFromHub()
                }
                .disabled(upscaleEngine.status.isBusy || upscaleEngine.canUpscale)
                if case .downloading = upscaleEngine.status {
                    ProgressView(value: upscaleEngine.downloader.fraction)
                        .frame(width: 140)
                }
                StatusLine(text: upscaleEngine.status.label, busy: upscaleEngine.status.isBusy)
            }

            HStack {
                Text("Interpolate (RIFE)").frame(width: 130, alignment: .leading)
                Button(interpEngine.isReady ? "Model loaded" : "Download RIFE") {
                    interpEngine.loadFromHub()
                }
                .disabled(interpEngine.status.isBusy || interpEngine.isReady)
                Button("Load Local…") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    panel.canChooseFiles = false
                    panel.prompt = "Load"
                    panel.begin { response in
                        guard response == .OK, let dir = panel.url else { return }
                        // Matches export_rife.py's naming convention (see
                        // FrameInterpolationEngine's header comment).
                        interpEngine.loadLocal(dir, stem: "rife-v4.26_384x640_float32")
                    }
                }
                .disabled(interpEngine.status.isBusy || interpEngine.isReady)
                if case .downloading = interpEngine.status {
                    ProgressView(value: interpEngine.downloader.fraction)
                        .frame(width: 140)
                }
                StatusLine(text: interpEngine.status.label, busy: interpEngine.status.isBusy)
            }
        }
    }

    // MARK: Add

    private var addSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Add video").font(.headline)

            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.gray.opacity(isTargeted ? 0.2 : 0.08))
                    .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.secondary.opacity(0.3)))
                VStack(spacing: 6) {
                    Image(systemName: "film.stack").font(.title2)
                    Text("Drop a video, or click to choose").font(.caption)
                }
                .foregroundStyle(.secondary)
            }
            .frame(minHeight: 90)
            .onTapGesture { showImporter = true }
            .onDrop(of: [.fileURL], isTargeted: $isTargeted) { providers in
                guard let provider = providers.first else { return false }
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in queue.add(sourceURL: url, kind: pendingKind) }
                }
                return true
            }
            .fileImporter(isPresented: $showImporter, allowedContentTypes: [.movie]) { result in
                if case .success(let url) = result { queue.add(sourceURL: url, kind: pendingKind) }
            }

            HStack {
                Text("Kind for next add")
                Picker("", selection: $pendingKindChoice) {
                    Text("Upscale").tag(PendingKindChoice.upscale)
                    Text("Interpolate ×2").tag(PendingKindChoice.interpolate2x)
                    Text("Interpolate ×4").tag(PendingKindChoice.interpolate4x)
                }
                .pickerStyle(.segmented)
                .frame(width: 340)
            }
        }
    }

    // MARK: Queue list

    private var queueSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Queue (\(queue.items.count))").font(.headline)

            if queue.items.isEmpty {
                Text("No items yet — add a video above.")
                    .font(.callout).foregroundStyle(.secondary)
                    .padding(.vertical, 8)
            } else {
                List {
                    ForEach(queue.items) { item in
                        QueueRow(item: item, onRemove: { queue.remove(id: item.id) })
                    }
                    .onMove(perform: queue.move)
                    .onDelete { offsets in
                        // Descending order: removing ascending indices in a forward loop would
                        // shift later indices after each removal (a real hazard here since
                        // macOS List supports multi-row selection + delete, not just
                        // single-row swipe).
                        for index in offsets.sorted(by: >) { queue.remove(at: index) }
                    }
                }
                .listStyle(.inset)
                .frame(height: min(CGFloat(queue.items.count) * 56 + 16, 420))
            }
        }
    }

    // MARK: Controls

    /// Simple overall readiness check: are the model(s) needed for the kinds currently in the
    /// queue loaded? This is intentionally coarse (queue-wide, not per-item) — a stricter
    /// per-item model-readiness gate (e.g. disabling just the items whose engine isn't ready
    /// yet) is a reasonable future improvement, but isn't needed to fix the "why is Start
    /// grayed out" problem: each item's own status text already explains itself once running,
    /// and the hint below explains the Start button.
    private var modelsReadyForQueue: Bool {
        let needsUpscale = queue.items.contains { if case .upscaleVideo = $0.kind { return true }; return false }
        let needsInterp = queue.items.contains { if case .interpolateVideo = $0.kind { return true }; return false }
        if needsUpscale && !upscaleEngine.canUpscale { return false }
        if needsInterp && !interpEngine.isReady { return false }
        return true
    }

    private var startDisabledReason: String? {
        if queue.items.isEmpty { return "Add at least one video to the queue." }
        if queue.isRunning { return "Queue is already running." }
        if !modelsReadyForQueue {
            var missing: [String] = []
            let needsUpscale = queue.items.contains { if case .upscaleVideo = $0.kind { return true }; return false }
            let needsInterp = queue.items.contains { if case .interpolateVideo = $0.kind { return true }; return false }
            if needsUpscale && !upscaleEngine.canUpscale { missing.append("AdcSR (Upscale)") }
            if needsInterp && !interpEngine.isReady { missing.append("RIFE (Interpolate)") }
            return "Load the required model(s) first: \(missing.joined(separator: ", "))."
        }
        return nil
    }

    private var controlsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Button("Start Queue") {
                    Task { await queue.start() }
                }
                .buttonStyle(.borderedProminent)
                .disabled(queue.items.isEmpty || queue.isRunning || !modelsReadyForQueue)

                if queue.isRunning {
                    Button("Cancel Current") { queue.cancelCurrent() }
                }

                Button("Clear Completed") { queue.clearCompleted() }
                    .disabled(queue.items.isEmpty)
            }
            if let reason = startDisabledReason {
                Text(reason).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(.top, 4)
    }
}

private struct QueueRow: View {
    let item: QueueItem
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(item.displayName).font(.body)
                Text(item.kind.displayName).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            statusView
            Button {
                onRemove()
            } label: {
                Image(systemName: "xmark.circle")
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var statusView: some View {
        switch item.status {
        case .pending:
            Text("Pending").font(.caption).foregroundStyle(.secondary)
        case .running(let progress):
            HStack(spacing: 6) {
                ProgressView(value: progress).frame(width: 120)
                Text("\(Int(progress * 100))%").font(.caption).monospacedDigit()
            }
        case .done(let outputURL):
            HStack(spacing: 6) {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([outputURL])
                }
                .buttonStyle(.link)
                .font(.caption)
            }
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(.red)
                .lineLimit(1)
                .frame(maxWidth: 220, alignment: .trailing)
        }
    }
}
