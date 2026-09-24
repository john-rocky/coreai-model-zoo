// DiarizeTimelineView — "who spoke when" for the Transcribe tab's 8-speaker Diarize: one lane per speaker in
// order of first appearance (Speaker 1, 2, … in 8 fixed colors), each lane's stretches above the activity
// threshold as bars, time across the whole clip, and a playhead while the clip plays. During the
// playback-synced run the bars grow chunk by chunk as the diarizer emits them, about a second behind the
// audio (0.72 s chunks + 0.32 s look-ahead); after a mic recording they appear all at once.
import SwiftUI

/// The lanes the timeline draws, fed by the diarizer's per-frame probabilities (frame-major [frames, 8]).
@MainActor
final class DiarizeTimeline: ObservableObject {
    struct Lane: Identifiable, Equatable {
        let speaker: Int                // the model's speaker slot, 0..<8
        let number: Int                 // 1-based, in order of first appearance: "Speaker 1"
        var runs: [Range<Int>]          // 10 ms frames above the threshold
        var id: Int { speaker }
    }

    /// Eight fixed colors; the transcript labels use the same ones.
    static let colors: [Color] = [.blue, .orange, .green, .purple, .pink, .teal, .red, .brown]
    static func color(_ number: Int) -> Color { colors[(max(number, 1) - 1) % colors.count] }

    @Published private(set) var lanes: [Lane] = []
    @Published private(set) var duration: Double = 0
    /// True while the clip plays and the diarizer streams (playhead + "behind" caption).
    @Published private(set) var live = false
    private(set) var frames = 0
    /// Where playback is, in seconds; read by the view on every display frame while `live`.
    private(set) var playhead: (@MainActor () -> Double)?

    let frameSec = NemotronDiarizerBridge.frameSec

    /// Speakers found so far (the lanes shown).
    var speakerCount: Int { lanes.count }

    /// The lane number of a model speaker slot, if it has spoken.
    func number(forSpeaker speaker: Int) -> Int? { lanes.first { $0.speaker == speaker }?.number }

    /// Empty lanes over a clip of `duration` seconds, growing as `append` delivers frames.
    func start(duration: Double, playhead: (@MainActor () -> Double)?) {
        lanes = []
        frames = 0
        self.duration = duration
        self.playhead = playhead
        live = playhead != nil
    }

    func finish() {
        live = false
        playhead = nil
    }

    /// The whole clip at once (a mic recording diarized after the fact).
    func show(probs: [Float], frames count: Int, duration: Double) {
        start(duration: duration, playhead: nil)
        append(probs: probs, frames: count)
    }

    /// The next `count` frames: extend each speaker's bars; a speaker's first active frame opens its lane.
    func append(probs: [Float], frames count: Int) {
        let S = NemotronDiarizerBridge.speakers, threshold = NemotronDiarizerBridge.activityThreshold
        let f0 = frames
        var found: [(speaker: Int, runs: [Range<Int>])] = []
        for s in 0..<S {
            var runs: [Range<Int>] = []
            var f = 0
            while f < count {
                guard probs[f * S + s] > threshold else { f += 1; continue }
                let a = f
                while f < count && probs[f * S + s] > threshold { f += 1 }
                runs.append((f0 + a)..<(f0 + f))
            }
            if !runs.isEmpty { found.append((s, runs)) }
        }
        var next = lanes
        // newcomers get the next numbers in the order they started talking (then by slot)
        let newcomers = found.filter { f in !next.contains { $0.speaker == f.speaker } }
            .sorted { ($0.runs[0].lowerBound, $0.speaker) < ($1.runs[0].lowerBound, $1.speaker) }
        for n in newcomers { next.append(Lane(speaker: n.speaker, number: next.count + 1, runs: [])) }
        for (s, runs) in found {
            guard let i = next.firstIndex(where: { $0.speaker == s }) else { continue }
            for run in runs {
                if let last = next[i].runs.last, last.upperBound == run.lowerBound {
                    next[i].runs[next[i].runs.count - 1] = last.lowerBound..<run.upperBound
                } else {
                    next[i].runs.append(run)
                }
            }
        }
        frames = f0 + count
        if next != lanes { lanes = next }
    }
}

/// The whole 8-speaker Diarize result: the timeline, one "Speaker N: text" line per turn (the label in the lane's
/// color, following the newest line), and the summary line. `scrolls: false` stacks the last lines without a scroll
/// view, for the off-screen snapshots (ImageRenderer cannot draw a scroll view).
struct DiarizeResultView: View {
    @ObservedObject var model: TranscribeModel
    var scrolls = true

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            DiarizeTimelineView(timeline: model.timeline)
            if scrolls {
                ScrollViewReader { proxy in
                    ScrollView { lines(model.diarizedLines).padding(10) }
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                        .onChange(of: model.diarizedLines.count) {
                            guard let last = model.diarizedLines.last else { return }
                            withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                        }
                }
                .frame(minHeight: 110)
            } else {
                lines(model.diarizedLines.suffix(8))
                    .padding(10)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
            }
            if !model.diarizeSummary.isEmpty {
                Text(model.diarizeSummary).font(.footnote).foregroundStyle(.secondary)
            }
        }
    }

    private func lines<C: RandomAccessCollection>(_ items: C) -> some View where C.Element == DiarizedLine {
        LazyVStack(alignment: .leading, spacing: 8) {
            if items.isEmpty {
                Text(model.busy ? "Who said what appears here once the clip ends." : " ")
                    .foregroundStyle(.secondary)
            }
            ForEach(items) { line in
                let label = Text("Speaker \(line.speaker):").bold().foregroundStyle(DiarizeTimeline.color(line.speaker))
                Text("\(label) \(line.text)")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .id(line.id)
            }
        }
        .textSelection(.enabled)
    }
}

struct DiarizeTimelineView: View {
    @ObservedObject var timeline: DiarizeTimeline

    private let laneHeight: CGFloat = 20
    private let laneGap: CGFloat = 5
    private let labelWidth: CGFloat = 74

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Who spoke when").font(.subheadline.weight(.semibold))
                Spacer()
                if timeline.live {
                    Text("streaming, 1.0 s behind").font(.caption).foregroundStyle(.secondary)
                }
            }
            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: laneGap) {
                    ForEach(timeline.lanes) { lane in
                        Text("Speaker \(lane.number)")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(DiarizeTimeline.color(lane.number))
                            .frame(height: laneHeight)
                    }
                }
                .frame(width: labelWidth, alignment: .leading)
                VStack(spacing: 3) {
                    TimelineView(.animation(minimumInterval: 1.0 / 30, paused: !timeline.live)) { _ in
                        let position = timeline.playhead?()
                        Canvas { context, size in drawLanes(in: &context, size: size, playhead: position) }
                    }
                    .frame(height: tracksHeight)
                    Canvas { context, size in drawAxis(in: &context, size: size) }
                        .frame(height: 14)
                }
            }
            .animation(.easeOut(duration: 0.25), value: timeline.lanes.count)
        }
    }

    /// At least one lane of track, so the playhead shows before anyone has spoken.
    private var tracksHeight: CGFloat {
        let rows = CGFloat(max(timeline.lanes.count, 1))
        return rows * laneHeight + (rows - 1) * laneGap
    }

    private func drawLanes(in context: inout GraphicsContext, size: CGSize, playhead: Double?) {
        let rows = max(timeline.lanes.count, 1)
        for i in 0..<rows {
            let track = CGRect(x: 0, y: CGFloat(i) * (laneHeight + laneGap), width: size.width, height: laneHeight)
            context.fill(Path(roundedRect: track, cornerRadius: 4), with: .color(.secondary.opacity(0.12)))
        }
        guard timeline.duration > 0 else { return }
        let scale = size.width / timeline.duration
        for (i, lane) in timeline.lanes.enumerated() {
            let y = CGFloat(i) * (laneHeight + laneGap) + 3
            let color = DiarizeTimeline.color(lane.number)
            for run in lane.runs {
                let x = Double(run.lowerBound) * timeline.frameSec * scale
                let w = max(Double(run.count) * timeline.frameSec * scale, 1)
                context.fill(Path(roundedRect: CGRect(x: x, y: y, width: w, height: laneHeight - 6), cornerRadius: 2),
                             with: .color(color))
            }
        }
        if let playhead {
            let x = min(max(playhead * scale, 1), size.width - 1)
            context.fill(Path(CGRect(x: x - 1, y: 0, width: 2, height: size.height)), with: .color(.primary.opacity(0.75)))
        }
    }

    /// m:ss marks every 5 / 10 / 15 / 30 / 60 s (at most six).
    private func drawAxis(in context: inout GraphicsContext, size: CGSize) {
        let d = timeline.duration
        guard d > 0 else { return }
        let step = [1.0, 2, 5, 10, 15, 30, 60, 120, 300, 600].first { d / $0 <= 5 } ?? 1200
        var t = 0.0
        while t <= d + 1e-6 {
            let x = t / d * size.width
            let seconds = Int(t.rounded())
            let label = context.resolve(Text(String(format: "%d:%02d", seconds / 60, seconds % 60))
                .font(.caption2).foregroundStyle(.secondary))
            let anchor: UnitPoint = t == 0 ? .topLeading : (x > size.width - 16 ? .topTrailing : .top)
            context.draw(label, at: CGPoint(x: x, y: 0), anchor: anchor)
            t += step
        }
    }
}
