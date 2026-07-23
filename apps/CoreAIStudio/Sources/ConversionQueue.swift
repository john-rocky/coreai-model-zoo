// ConversionQueue — a HandBrake-style batch conversion queue: a coordination/data-model layer
// that orchestrates VideoUpscaler (upscale jobs) and VideoInterpolator (interpolate jobs, see
// that file for the RIFE/multiplier pipeline it wraps). No Core AI / AVFoundation logic lives
// here — this file only sequences jobs and bridges each engine's own @Published `status` into a
// per-item QueueItemStatus that a future queue UI (a separate work unit) can observe.
//
// Output naming/location matches VideoView.convert()'s existing save-panel convention in
// ContentView.swift (`_<multiplier>x.mp4` next to the chosen name) — since the queue has no save
// panel to prompt per item, outputs are placed next to the source file with a suffix before the
// extension: `_2x`/`_4x` for interpolation (mirroring `_\(multiplier)x`), `_upscaled` for
// upscale jobs.

import Foundation

enum QueueItemKind: Equatable {
    case upscaleVideo
    case interpolateVideo(multiplier: Int)  // 2 or 4, matches VideoInterpolator's multiplier concept

    /// Suffix inserted before the source's extension when deriving a default output URL.
    var outputSuffix: String {
        switch self {
        case .upscaleVideo: return "_upscaled"
        case .interpolateVideo(let multiplier): return "_\(multiplier)x"
        }
    }

    var displayName: String {
        switch self {
        case .upscaleVideo: return "Upscale"
        case .interpolateVideo(let multiplier): return "Interpolate ×\(multiplier)"
        }
    }
}

enum QueueItemStatus: Equatable {
    case pending
    case running(progress: Double)
    case done(outputURL: URL)
    case failed(String)

    var isBusy: Bool {
        if case .running = self { return true }
        return false
    }
}

struct QueueItem: Identifiable {
    let id: UUID
    var sourceURL: URL
    var kind: QueueItemKind
    var status: QueueItemStatus
    /// Where the output will be (or was) written. Filled in with a derived default at `add`
    /// time; a future UI could let the user override it before the item runs.
    var outputURL: URL

    var displayName: String { sourceURL.lastPathComponent }

    init(id: UUID = UUID(), sourceURL: URL, kind: QueueItemKind, outputURL: URL? = nil, status: QueueItemStatus = .pending) {
        self.id = id
        self.sourceURL = sourceURL
        self.kind = kind
        self.status = status
        self.outputURL = outputURL ?? QueueItem.defaultOutputURL(sourceURL: sourceURL, kind: kind)
    }

    static func defaultOutputURL(sourceURL: URL, kind: QueueItemKind) -> URL {
        let base = sourceURL.deletingPathExtension().lastPathComponent + kind.outputSuffix
        return sourceURL.deletingLastPathComponent()
            .appendingPathComponent(base)
            .appendingPathExtension("mp4")
    }
}

@MainActor
final class ConversionQueue: ObservableObject {
    @Published private(set) var items: [QueueItem] = []
    @Published private(set) var isRunning: Bool = false

    private let videoUpscaler: VideoUpscaler
    private let videoInterpolator: VideoInterpolator

    /// Set by `cancelCurrent()`, observed by the polling loop in `bridgeAndAwait` — the engines
    /// themselves reset their own @Published status back to `.idle` on cancel (see
    /// VideoInterpolator.cancel()), so polling status alone can't distinguish "cancelled" from
    /// "never started"; this flag disambiguates.
    private var cancelRequested = false

    init(videoUpscaler: VideoUpscaler, videoInterpolator: VideoInterpolator) {
        self.videoUpscaler = videoUpscaler
        self.videoInterpolator = videoInterpolator
    }

    // MARK: - Editing

    func add(sourceURL: URL, kind: QueueItemKind) {
        items.append(QueueItem(sourceURL: sourceURL, kind: kind))
    }

    func remove(at index: Int) {
        guard items.indices.contains(index) else { return }
        if isRunning, items[index].status.isBusy {
            cancelCurrent()
        }
        items.remove(at: index)
    }

    func remove(id: UUID) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        remove(at: index)
    }

    func move(from source: IndexSet, to destination: Int) {
        items.move(fromOffsets: source, toOffset: destination)
    }

    func clearCompleted() {
        items.removeAll { item in
            if case .done = item.status { return true }
            return false
        }
    }

    // MARK: - Running

    /// Processes items sequentially, one at a time, updating each item's status in place as it
    /// runs. If one item fails, the queue moves on to the next rather than aborting entirely.
    /// Looks items up by id (not index) at each step so concurrent `move`/`remove` calls from a
    /// queue-editing UI can't desync this loop from a shifting array.
    func start() async {
        guard !isRunning else { return }
        isRunning = true
        defer { isRunning = false }

        while let id = items.first(where: { if case .pending = $0.status { return true }; return false })?.id {
            await runItem(id: id)
        }
    }

    func cancelCurrent() {
        guard isRunning else { return }
        cancelRequested = true
        videoUpscaler.cancel()
        videoInterpolator.cancel()
    }

    // MARK: - Per-item execution

    private func runItem(id: UUID) async {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        items[index].status = .running(progress: 0)
        let item = items[index]

        switch item.kind {
        case .upscaleVideo:
            await runUpscale(item: item, id: id)
        case .interpolateVideo(let multiplier):
            await runInterpolate(item: item, id: id, multiplier: multiplier)
        }
    }

    private func runUpscale(item: QueueItem, id: UUID) async {
        videoUpscaler.loadSource(item.sourceURL)
        videoUpscaler.run(sourceURL: item.sourceURL, outputURL: item.outputURL)
        await bridgeAndAwait(id: id) { [videoUpscaler] in
            switch videoUpscaler.status {
            case .idle, .analyzing:
                return nil
            case .working(let progress):
                return .running(progress: progress)
            case .done(let url):
                return .done(outputURL: url)
            case .error(let message):
                return .failed(message)
            }
        }
    }

    private func runInterpolate(item: QueueItem, id: UUID, multiplier: Int) async {
        videoInterpolator.loadSource(item.sourceURL)
        videoInterpolator.run(sourceURL: item.sourceURL, multiplier: multiplier, outputURL: item.outputURL)
        await bridgeAndAwait(id: id) { [videoInterpolator] in
            switch videoInterpolator.status {
            case .idle, .analyzing:
                return nil
            case .working(let progress):
                return .running(progress: progress)
            case .done(let url):
                return .done(outputURL: url)
            case .error(let message):
                return .failed(message)
            }
        }
    }

    /// Polls `poll()` until it reports a terminal QueueItemStatus (`.done`/`.failed`), writing
    /// each intermediate status into the item (looked up by id) as it goes. `poll` returning nil
    /// means the underlying engine hasn't started producing progress yet (still
    /// idle/analyzing). If the item is removed from the queue mid-run, the loop exits quietly
    /// (there's nothing left to update) rather than looping forever.
    ///
    /// `sawRunning` guards against a real hazard in both engines' `run(...)`: it's a fire-and-
    /// forget call that silently no-ops if its own readiness guard fails (e.g. VideoInterpolator
    /// requires `engine.isReady`), *without* touching `status` — so immediately after such a
    /// no-op, `status` is still whatever the *previous* job on this engine left behind (idle, or
    /// even a stale `.done`/`.error` from the last item). Without this guard we'd either hang
    /// forever polling a stationary `.idle`, or worse, misattribute a stale terminal result to
    /// this item. We only accept a terminal status once we've first observed `.running` for this
    /// job; if that never happens within `startGrace`, we fail this item outright instead of
    /// blocking the rest of the queue.
    private func bridgeAndAwait(id: UUID, poll: () -> QueueItemStatus?) async {
        cancelRequested = false
        var sawRunning = false
        var waitedWithoutStart: UInt64 = 0
        let startGraceNanos: UInt64 = 5_000_000_000
        let pollIntervalNanos: UInt64 = 50_000_000

        while true {
            guard let index = items.firstIndex(where: { $0.id == id }) else { return }
            if cancelRequested {
                items[index].status = .failed("Cancelled")
                cancelRequested = false
                return
            }
            if let status = poll() {
                if case .running = status { sawRunning = true }
                if sawRunning {
                    items[index].status = status
                    switch status {
                    case .done, .failed: return
                    default: break
                    }
                }
            }
            if !sawRunning {
                waitedWithoutStart += pollIntervalNanos
                if waitedWithoutStart >= startGraceNanos {
                    items[index].status = .failed("Job never started — engine not ready?")
                    return
                }
            }
            try? await Task.sleep(nanoseconds: pollIntervalNanos)
        }
    }
}
