// DiarizeLiveView — the 8-speaker Diarize run as a full-screen live screen, drawn like the "Play live" view of the
// LiteRT Nemotron-3-Diarization demo (LiveView.kt in the zoo's nemotron3diar Android app). Top to bottom: a state
// badge (READY / LIVE / DONE) with the playback clock (0.1 s steps) and the number of speakers so far; the latency
// line; the waveform up to the playhead, colored by the speaker the model has given each 10 ms frame (grey until the
// chunk that covers it has run); eight speaker lanes, numbered by first appearance; the playhead; the time axis over
// the whole clip; a small per-chunk line. Positions and sizes are LiveView.kt's, in units of width / 1080 (its
// 1080 x 2340 phone); the bands above and below stay empty for captions. A DIARIZE_DEMO launch shows only this screen;
// Diarize + Choose… shows it full screen until Close.
import NemotronDiarizer
import SwiftUI
#if os(iOS)
import UIKit
#endif

/// What the live screen draws besides the lanes (DiarizeTimeline): the phase, the clip's 10 ms peaks, the speaker of
/// every frame the diarizer has emitted, and the latency and per-chunk lines.
@MainActor
final class DiarizeLive: ObservableObject {
    enum Phase { case ready, live, done }

    /// A chunk's first frame waits for the chunk and its look-ahead: (9 + 4) encoder frames of 80 ms = 1.04 s.
    static let designLatency: Double = {
        let mode = NemotronDiarizerBridge.mode
        return Double((mode.chunkFrames + mode.lookaheadFrames) * N3DMel.stack * N3DMel.hop) / Double(N3DMel.sampleRate)
    }()
    static let engine = "fp16 · Core AI GPU"
    /// DIARIZE_DEMO_TRANSCRIPT=1: once the clip is done, "who said what" under the lanes.
    nonisolated static let showsTranscript: Bool = {
        let v = ProcessInfo.processInfo.environment["DIARIZE_DEMO_TRANSCRIPT"] ?? ""
        return !v.isEmpty && v != "0"
    }()

    @Published private(set) var phase = Phase.ready
    @Published private(set) var latencyText = DiarizeLive.latencyLine(nil)
    @Published private(set) var stepText = DiarizeLive.engine
    /// The clip's peak |sample| per 10 ms frame (at least one frame) and the largest of them.
    private(set) var peaks: [Float] = [0]
    private(set) var peakScale: Float = 1
    private(set) var duration = 0.0
    /// Per emitted frame: the model speaker slot with the highest probability above the threshold, else -1.
    private(set) var dominant: [Int8] = []
    private var chunkSeconds: [Double] = []
    private var position: (@MainActor () -> Double)?

    /// Where the playhead is: nowhere before playback, the player's position while it plays, the end once done.
    var seconds: Double {
        switch phase {
        case .ready: return 0
        case .live: return position?() ?? 0
        case .done: return duration
        }
    }

    /// A new clip: READY, its peaks for the waveform, nothing emitted yet.
    func setClip(_ samples: [Float]) {
        let hop = N3DMel.hop
        let count = max(1, (samples.count + hop - 1) / hop)
        var peaks = [Float](repeating: 0, count: count)
        samples.withUnsafeBufferPointer { x in
            for f in 0..<count {
                var p: Float = 0
                for i in (f * hop)..<min(x.count, (f + 1) * hop) { p = max(p, abs(x[i])) }
                peaks[f] = p
            }
        }
        self.peaks = peaks
        peakScale = max(1e-4, peaks.max() ?? 1)
        duration = Double(samples.count) / Double(N3DMel.sampleRate)
        dominant = []
        chunkSeconds = []
        position = nil
        latencyText = Self.latencyLine(nil)
        stepText = Self.engine
        phase = .ready
    }

    /// Playback has started; the clock and the playhead follow `position` (seconds).
    func start(position: @escaping @MainActor () -> Double) {
        self.position = position
        phase = .live
    }

    /// The next emitted frames ([count, 8] probabilities), after DiarizeTimeline has taken them.
    func append(probs: [Float], frames count: Int) {
        let S = NemotronDiarizerBridge.speakers
        for f in 0..<count {
            var best = -1, bestP = NemotronDiarizerBridge.activityThreshold
            for s in 0..<S where probs[f * S + s] > bestP {
                best = s
                bestP = probs[f * S + s]
            }
            dominant.append(Int8(best))
        }
    }

    /// Chunk `index` (from 0) took `seconds` on the device, host work included.
    func chunkDone(_ index: Int, seconds: Double) {
        chunkSeconds.append(seconds)
        let s = chunkSeconds.sorted(), n = s.count
        latencyText = Self.latencyLine(n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2)
        stepText = String(format: "chunk %d · %.0f ms · %@", index, seconds * 1000, Self.engine)
    }

    func finish() {
        position = nil
        phase = .done
    }

    /// A failed run, the way LiveView.kt shows one: in the per-chunk line.
    func fail(_ message: String) {
        stepText = "FAIL: \(message)"
    }

    static func latencyLine(_ onDevice: Double?) -> String {
        String(format: "latency %.2f s + on-device ", designLatency)
            + (onDevice.map { String(format: "%.2f s", $0) } ?? "…")
    }
}

/// LiveView.kt's colors; the eight speaker colors are TimelineView.kt's, by lane.
enum DiarizeLiveStyle {
    static let background = Color(rgb: 0x0E1116)
    static let lane = Color(rgb: 0x1B2028)
    static let pending = Color(rgb: 0x6B7280)
    static let dimLabel = Color(rgb: 0x3C424C)
    static let axis = Color(rgb: 0x8A919C)
    static let latency = Color(rgb: 0xB8BEC6)
    static let speakers: [Color] = [0x4285F4, 0xEA4335, 0xFBBC05, 0x34A853, 0xAB47BC, 0x00ACC1, 0xFF7043, 0x9E9D24]
        .map { Color(rgb: $0) }

    static func badge(_ phase: DiarizeLive.Phase) -> (text: String, color: Color) {
        switch phase {
        case .ready: return ("READY", Color(rgb: 0x5F6368))
        case .live: return ("● LIVE", Color(rgb: 0xE53935))
        case .done: return ("DONE", Color(rgb: 0x2E7D32))
        }
    }

    /// The color of lane `number` (from 1).
    static func speaker(_ number: Int) -> Color { speakers[(max(number, 1) - 1) % speakers.count] }
}

extension Color {
    /// 0xRRGGBB, sRGB.
    init(rgb: UInt32) {
        self.init(.sRGB, red: Double((rgb >> 16) & 0xFF) / 255, green: Double((rgb >> 8) & 0xFF) / 255,
                  blue: Double(rgb & 0xFF) / 255)
    }
}

/// LiveView.kt's coordinates (a 1080 x 2340 phone, in units of width / 1080) on this screen. A taller or shorter
/// screen grows or shrinks the empty bands above and below the content; one too short for the content itself
/// shrinks the unit instead.
struct DiarizeLiveLayout {
    static let phoneHeight: CGFloat = 2340
    static let contentTop: CGFloat = 250          // the badge's top edge is at 256
    static let contentBottom: CGFloat = 2010      // the per-chunk line's baseline is at 1996

    let u: CGFloat
    let dx: CGFloat
    let dy: CGFloat

    init(size: CGSize) {
        let content = Self.contentBottom - Self.contentTop
        let bands = Self.phoneHeight - content
        if size.height >= content * size.width / 1080 {
            u = size.width / 1080
            dx = 0
            dy = (size.height - Self.phoneHeight * u) * Self.contentTop / bands
        } else {
            u = size.height / content
            dx = (size.width - 1080 * u) / 2
            dy = -Self.contentTop * u
        }
    }

    func x(_ v: CGFloat) -> CGFloat { dx + v * u }
    func y(_ v: CGFloat) -> CGFloat { dy + v * u }

    // LiveView.kt's rows
    var headerY: CGFloat { y(300) }
    var waveTop: CGFloat { y(470) }
    var laneTop: CGFloat { y(850) }
    var lanesBottom: CGFloat { laneTop + 7 * 130 * u + 108 * u }
    var axisY: CGFloat { lanesBottom + 52 * u }
    var stepLineY: CGFloat { axisY + 76 * u }
    var left: CGFloat { x(172) }
    var right: CGFloat { x(1080 - 44) }
}

/// The live screen: redrawn every display frame while the clip plays.
struct DiarizeLiveView: View {
    @ObservedObject var model: TranscribeModel
    @ObservedObject var live: DiarizeLive
    /// The manual run's full-screen sheet: a small Close once the clip is done.
    var onClose: (() -> Void)?

    init(model: TranscribeModel, onClose: (() -> Void)? = nil) {
        self.model = model
        live = model.diarizeLive
        self.onClose = onClose
    }

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 60, paused: live.phase != .live)) { _ in
            DiarizeLiveScreen(model: model, seconds: live.seconds, onClose: onClose)
        }
        .ignoresSafeArea()
        #if os(iOS)
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
        .onAppear {
            DiarizeOrientation.portraitOnly(true)
            UIApplication.shared.isIdleTimerDisabled = true   // the screen stays on, as LiveView.kt's activity keeps it
        }
        .onDisappear {
            guard onClose != nil else { return }
            DiarizeOrientation.portraitOnly(false)
            UIApplication.shared.isIdleTimerDisabled = false
        }
        #endif
    }
}

/// The live screen at one playhead position (the off-screen snapshots render this directly).
struct DiarizeLiveScreen: View {
    @ObservedObject var model: TranscribeModel
    @ObservedObject var live: DiarizeLive
    @ObservedObject var timeline: DiarizeTimeline
    let seconds: Double
    var onClose: (() -> Void)?
    /// False for ImageRenderer, which cannot draw a scroll view: the last transcript lines, stacked.
    var scrolls = true
    /// Pixels per point of the waveform's columns (nil: the display's).
    var pixelScale: CGFloat?
    @Environment(\.displayScale) private var displayScale

    init(model: TranscribeModel, seconds: Double, onClose: (() -> Void)? = nil, scrolls: Bool = true,
         pixelScale: CGFloat? = nil) {
        self.model = model
        live = model.diarizeLive
        timeline = model.timeline
        self.seconds = seconds
        self.onClose = onClose
        self.scrolls = scrolls
        self.pixelScale = pixelScale
    }

    var body: some View {
        GeometryReader { geo in
            let layout = DiarizeLiveLayout(size: geo.size)
            ZStack(alignment: .topLeading) {
                Canvas { context, size in draw(in: &context, size: size) }
                if DiarizeLive.showsTranscript && live.phase == .done {
                    transcript(layout, size: geo.size)
                }
                if let onClose, live.phase == .done {
                    Button(action: onClose) {
                        Text("Close")
                            .font(.system(size: 32 * layout.u, weight: .semibold))
                            .foregroundStyle(DiarizeLiveStyle.latency)
                            .padding(.horizontal, 30 * layout.u)
                            .padding(.vertical, 12 * layout.u)
                            .background(DiarizeLiveStyle.lane, in: Capsule())
                    }
                    .buttonStyle(.plain)
                    .position(x: geo.size.width / 2, y: geo.size.height - 120 * layout.u)   // above the home indicator
                }
            }
        }
        .background(DiarizeLiveStyle.background)
    }

    // MARK: - Drawing (LiveView.onDraw)

    private func draw(in context: inout GraphicsContext, size: CGSize) {
        let L = DiarizeLiveLayout(size: size)
        let u = L.u
        let x0 = L.left, x1 = L.right, w = x1 - x0
        let span = CGFloat(live.peaks.count)
        func xOf(_ frame: CGFloat) -> CGFloat { x0 + frame / span * w }

        // header: state badge, clock, speakers; the latency line under it
        let headerY = L.headerY
        let (badge, badgeColor) = DiarizeLiveStyle.badge(live.phase)
        let badgeText = text(context, badge, 36 * u, bold: true, .white)
        let bw = measure(badgeText).width + 44 * u
        let pill = CGRect(x: L.x(44), y: headerY - 44 * u, width: bw, height: 64 * u)
        context.fill(Path(roundedRect: pill, cornerRadius: 32 * u), with: .color(badgeColor))
        put(&context, badgeText, x: pill.minX + 22 * u, baseline: headerY + 1 * u)

        let tenths = Int(seconds * 10)
        let clock = live.phase == .ready ? "–.– s" : String(format: "%d.%d s", tenths / 10, tenths % 10)
        put(&context, text(context, clock, 72 * u, .white), x: pill.maxX + 32 * u, baseline: headerY + 14 * u)

        let n = timeline.speakerCount
        put(&context, text(context, n == 1 ? "1 speaker" : "\(n) speakers", 44 * u, bold: true, .white),
            x: x1, baseline: headerY + 4 * u, trailing: true)
        put(&context, text(context, live.latencyText, 36 * u, DiarizeLiveStyle.latency),
            x: L.x(44), baseline: headerY + 96 * u)

        // waveform up to the playhead: one bar per pixel column, in the color of the lane the model gave it
        let lanes = timeline.lanes
        var laneOfSlot = [Int](repeating: -1, count: NemotronDiarizerBridge.speakers)
        for (i, lane) in lanes.enumerated() where lane.speaker < laneOfSlot.count { laneOfSlot[lane.speaker] = i }
        let waveTop = L.waveTop, waveH = 300 * u, mid = waveTop + waveH / 2
        let headFrame = seconds / NemotronDiarizerBridge.frameSec
        let scale = pixelScale ?? displayScale
        let cols = max(1, Int(w * scale))
        let colW = w / CGFloat(cols), barW = max(1 / scale, colW * 0.8)
        let peaks = live.peaks, dominant = live.dominant
        let emitted = dominant.count
        var bars = [Path](repeating: Path(), count: DiarizeLiveStyle.speakers.count + 1)   // + grey
        for col in 0..<cols {
            let f0 = Int(CGFloat(col) * span / CGFloat(cols))
            let f1 = max(f0 + 1, Int(CGFloat(col + 1) * span / CGFloat(cols)))
            if Double(f0) >= headFrame { break }
            var p: Float = 0
            for f in f0..<min(f1, peaks.count) { p = max(p, peaks[f]) }
            let h = max(2 * u, CGFloat((p / live.peakScale).squareRoot()) * waveH / 2 * 0.96)
            var color = DiarizeLiveStyle.speakers.count
            if f0 < emitted {
                for f in f0..<min(f1, emitted) where dominant[f] >= 0 {
                    let lane = laneOfSlot[Int(dominant[f])]
                    if lane >= 0 { color = lane % DiarizeLiveStyle.speakers.count }
                    break
                }
            }
            bars[color].addRect(CGRect(x: x0 + CGFloat(col) * colW, y: mid - h, width: barW, height: 2 * h))
        }
        for (i, path) in bars.enumerated() where !path.isEmpty {
            context.fill(path, with: .color(i < DiarizeLiveStyle.speakers.count ? DiarizeLiveStyle.speakers[i]
                                                                                : DiarizeLiveStyle.pending))
        }

        // eight speaker lanes: a dark track, the label, the active stretches as full-height bars
        let rowH = 130 * u, laneH = 108 * u
        for i in 0..<DiarizeLiveStyle.speakers.count {
            let y = L.laneTop + CGFloat(i) * rowH
            context.fill(Path(roundedRect: CGRect(x: x0, y: y, width: w, height: laneH), cornerRadius: 10 * u),
                         with: .color(DiarizeLiveStyle.lane))
            let active = i < lanes.count
            put(&context, text(context, "SPK \(i + 1)", 34 * u, bold: true,
                               active ? DiarizeLiveStyle.speakers[i] : DiarizeLiveStyle.dimLabel),
                x: L.x(44), baseline: y + laneH / 2 + 12 * u)
            guard active else { continue }
            var runs = Path()
            for r in lanes[i].runs {
                let a = xOf(CGFloat(r.lowerBound)), b = xOf(CGFloat(r.upperBound))
                runs.addRect(CGRect(x: a, y: y, width: b - a, height: laneH))
            }
            context.fill(runs, with: .color(DiarizeLiveStyle.speakers[i]))
        }
        let lanesBottom = L.lanesBottom

        // playhead across the waveform and the lanes
        if live.phase != .ready {
            let px = xOf(min(CGFloat(headFrame), span))
            context.fill(Path(CGRect(x: px - 1.5 * u, y: waveTop - 12 * u, width: 3 * u,
                                     height: lanesBottom - waveTop + 24 * u)), with: .color(.white))
            context.fill(Path(ellipseIn: CGRect(x: px - 9 * u, y: waveTop - 21 * u, width: 18 * u, height: 18 * u)),
                         with: .color(.white))
        }

        // time axis: a tick every 10 s, "0 s" at the first, the clip's end on the right
        if live.duration > 0 {
            let clipSeconds = span * CGFloat(NemotronDiarizerBridge.frameSec)
            var t = 0
            while CGFloat(t) < clipSeconds - 4 {
                let x = xOf(CGFloat(t) / CGFloat(NemotronDiarizerBridge.frameSec))
                context.fill(Path(CGRect(x: x - 1 * u, y: lanesBottom + 14 * u, width: 2 * u, height: 12 * u)),
                             with: .color(DiarizeLiveStyle.axis))
                if t == 0 {
                    put(&context, text(context, "0 s", 26 * u, DiarizeLiveStyle.axis), x: x, baseline: L.axisY)
                }
                t += 10
            }
            put(&context, text(context, String(format: "%.0f s", clipSeconds), 26 * u, DiarizeLiveStyle.axis),
                x: x1, baseline: L.axisY, trailing: true)
        }

        // per-chunk line
        put(&context, text(context, live.stepText, 28 * u, DiarizeLiveStyle.axis), x: L.x(44), baseline: L.stepLineY)
    }

    private func text(_ context: GraphicsContext, _ string: String, _ size: CGFloat, bold: Bool = false,
                      _ color: Color) -> GraphicsContext.ResolvedText {
        context.resolve(Text(string).font(.system(size: size, weight: bold ? .bold : .regular).monospacedDigit())
            .foregroundStyle(color))
    }

    private func measure(_ text: GraphicsContext.ResolvedText) -> CGSize {
        text.measure(in: CGSize(width: 100_000, height: 100_000))
    }

    /// Draws `text` with its baseline at `y`, from `x` rightwards (or ending at `x` when `trailing`).
    private func put(_ context: inout GraphicsContext, _ text: GraphicsContext.ResolvedText, x: CGFloat,
                     baseline y: CGFloat, trailing: Bool = false) {
        let s = measure(text)
        let top = y - text.firstBaseline(in: s)
        context.draw(text, in: CGRect(x: trailing ? x - s.width : x, y: top, width: s.width, height: s.height))
    }

    // MARK: - Transcript (DIARIZE_DEMO_TRANSCRIPT=1)

    /// "Who said what" in the band under the per-chunk line: a SPK chip in the lane's color, then the text.
    private func transcript(_ L: DiarizeLiveLayout, size: CGSize) -> some View {
        let u = L.u
        let top = L.stepLineY + 40 * u
        let bottom = size.height - (onClose == nil ? 100 : 170) * u
        let width = L.right - L.x(44), height = max(0, bottom - top)
        let lines = model.diarizedLines
        return Group {
            if scrolls {
                ScrollViewReader { proxy in
                    ScrollView { rows(lines, u: u) }
                        .onChange(of: lines.count) {
                            guard let last = lines.last else { return }
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                }
            } else {
                // the newest lines that fit, like the scroll view once it has followed them
                rows(lines.suffix(8), u: u)
                    .frame(width: width, height: height, alignment: .bottom)
                    .clipped()
            }
        }
        .frame(width: width, height: height)
        .offset(x: L.x(44), y: top)
    }

    private func rows<C: RandomAccessCollection>(_ lines: C, u: CGFloat) -> some View where C.Element == DiarizedLine {
        LazyVStack(alignment: .leading, spacing: 14 * u) {
            ForEach(lines) { line in
                HStack(alignment: .firstTextBaseline, spacing: 16 * u) {
                    Text("SPK \(line.speaker)")
                        .font(.system(size: 28 * u, weight: .bold).monospacedDigit())
                        .foregroundStyle(DiarizeLiveStyle.speaker(line.speaker))
                        .padding(.horizontal, 12 * u)
                        .padding(.vertical, 4 * u)
                        .background(DiarizeLiveStyle.lane, in: RoundedRectangle(cornerRadius: 8 * u))
                    Text(line.text)
                        .font(.system(size: 34 * u))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .id(line.id)
            }
        }
    }
}

#if os(iOS)
/// The live screen is portrait only: the app's orientation mask while it shows (from launch in a DIARIZE_DEMO run).
final class DiarizeOrientation: NSObject, UIApplicationDelegate {
    private static var portrait = DiarizeDemoOptions.current != nil

    func application(_ application: UIApplication,
                     supportedInterfaceOrientationsFor window: UIWindow?) -> UIInterfaceOrientationMask {
        Self.portrait ? .portrait : .all   // .all: each screen's own default (iPhone: not upside down)
    }

    static func portraitOnly(_ on: Bool) {
        guard portrait != on else { return }
        portrait = on
        for case let scene as UIWindowScene in UIApplication.shared.connectedScenes {
            for window in scene.windows { window.rootViewController?.setNeedsUpdateOfSupportedInterfaceOrientations() }
            if on { scene.requestGeometryUpdate(.iOS(interfaceOrientations: .portrait)) }
        }
    }
}
#endif
