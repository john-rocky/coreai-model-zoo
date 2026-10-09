// StoryView — the video's layout (`-autoplay story`): one message to a screen, large enough to read at a phone feed's
// width. The phone's screen is 9:19.5; every page is drawn inside a 9:16 band at its center (402 x 715 at the phone's
// point size), so a 9:16 crop of the recording loses nothing. The pages, in the order Autoplay shows them: a title
// card (how many messages, how many questions, where they are answered); a few messages one at a time (the message,
// then its three answers one after another with their probabilities, then the seconds the app timed for that message);
// the whole inbox's count and seconds; the order log's first question with its answer and the log's seconds; the end
// card. Every word comes from the samples (DemoSet.display) or the response, every number from the response or the app's
// clock (DemoModel); Autoplay sets the pace and records what each page showed (Story.shown, written by RunLog).

import SwiftUI

@MainActor
@Observable
final class Story {
    enum Page: Hashable {
        case blank, open, item(String), scale, log, end

        var name: String {
            switch self {
            case .blank: return "blank"
            case .open: return "open"
            case .item: return "item"
            case .scale: return "scale"
            case .log: return "log"
            case .end: return "end"
            }
        }
    }

    /// One page as it was shown: when it came up, when its last part appeared, and its words in screen order.
    struct Shown {
        let page: String
        let id: String?
        let at: Date
        var full: Date?
        var words: [(String, String)]
    }

    var page: Page = .blank
    /// how many of the current page's answers are on screen
    var revealed = 0
    /// the current page's seconds line is on screen
    var secondsShown = false
    private(set) var shown: [Shown] = []

    /// The title card comes up at once (its first frame is the clip's first, whole); later pages cross-fade.
    func show(_ p: Page, id: String? = nil, words: [(String, String)] = []) {
        revealed = 0
        secondsShown = false
        if p == .open { page = p } else { withAnimation(.easeInOut(duration: 0.3)) { page = p } }
        shown.append(Shown(page: p.name, id: id, at: Date(), full: nil, words: words))
    }

    /// The current page is complete: its words as the screen now shows them.
    func full(words: [(String, String)]) {
        guard !shown.isEmpty else { return }
        shown[shown.count - 1].full = Date()
        shown[shown.count - 1].words += words
    }
}

/// The story's words, shared by the pages and the run log, so what the log says the screen showed is what it showed.
@MainActor
enum StoryWords {
    static let labels = ["refund": "Refund?", "team": "Team:", "urgency": "Urgency:", "damage": "Damage:",
                         "boxes": "Boxes:", "opened": "Opened?"]

    static func label(_ q: String, _ demo: DemoSet?) -> String { labels[q] ?? demo?.display(q)?.label ?? q }

    /// a yes / no question reads Yes or No; the others keep the samples' words for the option or level
    static func answer(_ a: Answer) -> String {
        switch a.option {
        case "yes": return "Yes"
        case "no": return "No"
        default: return a.text
        }
    }

    static func p(_ a: Answer) -> String { String(format: "%.2f", a.p) }
    static func seconds(_ s: Double) -> String { String(format: "%.2f s", s) }
    static let onDevice = " · on device"
    static func forQuestions(_ n: Int) -> String { " for \(n) questions" }

    static func open(_ model: DemoModel) -> [(String, String)] {
        let n = model.inbox.count
        let q = model.inbox.first?.questions.count ?? 3
        var w: [(String, String)] = [("count", "\(n)"), ("what", "support messages"), ("questions", "\(q) questions each"),
                                     ("where", "answered on the \(DeviceInfo.kind)")]
        if model.offline { w.append(("offline", "offline")) }
        return w
    }

    static func item(_ model: DemoModel, _ id: String) -> [(String, String)] {
        guard let item = model.inbox.first(where: { $0.id == id }), let r = model.results[id] else { return [] }
        var w: [(String, String)] = [("message", item.text)]
        for (k, a) in r.answers.enumerated() {
            w += [("label_\(k + 1)", label(a.question, model.demo)), ("answer_\(k + 1)", answer(a)), ("p_\(k + 1)", p(a))]
        }
        w += [("seconds", seconds(r.seconds)), ("seconds_tail", onDevice)]
        return w
    }

    static func scale(_ model: DemoModel) -> [(String, String)] {
        [("messages", "\(model.doneCount) messages"), ("answers", "\(model.answerCount) answers"),
         ("total", model.inboxTotal.map(seconds) ?? ""), ("where", "on device")]
    }

    static func log(_ model: DemoModel) -> [(String, String)] {
        guard let demo = model.demo, let r = model.logResult, let a = r.answers.first else { return [] }
        return [("title", "\(demo.samples.log.entries)-entry order log"),
                ("question", demo.display(a.question)?.label ?? a.question), ("answer", a.text), ("p", p(a)),
                ("seconds", seconds(r.seconds)), ("seconds_tail", forQuestions(r.answers.count))]
    }
}

struct StoryView: View {
    let model: DemoModel

    var body: some View {
        GeometryReader { geo in
            let w = min(geo.size.width, geo.size.height * 9 / 16)
            ZStack {
                Style.background
                page(w / 402)
                    .id(model.story.page)
                    .transition(.opacity)
                    .frame(width: w, height: w * 16 / 9)
                    .position(x: geo.size.width / 2, y: geo.size.height / 2)
            }
        }
        .ignoresSafeArea()
    }

    @ViewBuilder private func page(_ u: CGFloat) -> some View {
        switch model.story.page {
        case .blank: Style.background
        case .open: OpenPage(model: model, u: u)
        case .item(let id): ItemPage(model: model, id: id, u: u)
        case .scale: ScalePage(model: model, u: u)
        case .log: LogPage(model: model, u: u)
        case .end: EndCardView()
        }
    }
}

/// The title card: how many messages, how many questions each, where they are answered.
private struct OpenPage: View {
    let model: DemoModel
    let u: CGFloat

    var body: some View {
        let w = Dictionary(StoryWords.open(model), uniquingKeysWith: { a, _ in a })
        VStack(alignment: .leading, spacing: 10 * u) {
            Text(w["count"] ?? "")
                .font(.system(size: 150 * u, weight: .bold).monospacedDigit())
                .foregroundStyle(Style.blue)
                .padding(.bottom, -14 * u)
            Text(w["what"] ?? "")
                .font(.system(size: 40 * u, weight: .bold))
                .foregroundStyle(.white)
            Text(w["questions"] ?? "")
                .font(.system(size: 40 * u, weight: .bold))
                .foregroundStyle(.white)
            VStack(alignment: .leading, spacing: 4 * u) {
                Text(w["where"] ?? "")
                if let off = w["offline"] { Text(off) }
            }
            .font(.system(size: 32 * u, weight: .semibold))
            .foregroundStyle(Style.latency)
            .padding(.top, 14 * u)
        }
        .lineLimit(1)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .padding(.horizontal, 24 * u)
    }
}

/// One message: the message (its text, or its picture with the text over the picture's top), its three answers one
/// after another, the seconds the app timed for it.
private struct ItemPage: View {
    let model: DemoModel
    let id: String
    let u: CGFloat

    var body: some View {
        let item = model.inbox.first { $0.id == id }
        let result = model.results[id]
        let story = model.story
        VStack(spacing: 0) {
            Spacer(minLength: 0)
            if let item {
                if let image = model.demo?.images[id] { PictureMessage(image: image, text: item.text, u: u) }
                else { TextMessage(text: item.text, u: u) }
            }
            VStack(spacing: 12 * u) {
                ForEach(Array((item?.questions ?? []).enumerated()), id: \.offset) { k, q in
                    let a = result.flatMap { k < $0.answers.count ? $0.answers[k] : nil }
                    StoryAnswer(label: StoryWords.label(q, model.demo), answer: a, u: u)
                        .opacity(a != nil && k < story.revealed ? 1 : 0)
                        .offset(y: a != nil && k < story.revealed ? 0 : 10 * u)
                        .animation(.easeOut(duration: 0.25), value: story.revealed)
                }
            }
            .padding(.top, 14 * u)
            SecondsLine(seconds: result.map { StoryWords.seconds($0.seconds) } ?? " ", tail: StoryWords.onDevice, u: u)
                .opacity(result != nil && story.secondsShown ? 1 : 0)
                .animation(.easeOut(duration: 0.25), value: story.secondsShown)
                .padding(.top, 14 * u)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12 * u)
    }
}

/// A text message: the customer's words, large, on a card.
private struct TextMessage: View {
    let text: String
    let u: CGFloat

    var body: some View {
        Text(text)
            .font(.system(size: 44 * u, weight: .medium))
            .foregroundStyle(.white)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 12 * u)
            .padding(.vertical, 16 * u)
            .background(Style.lane, in: RoundedRectangle(cornerRadius: 22 * u))
    }
}

/// A message with a picture: the picture the width of the page, the customer's words over its top (the wall in every
/// picture), darkened so they read.
private struct PictureMessage: View {
    let image: CGImage
    let text: String
    let u: CGFloat

    var body: some View {
        let side = 370 * u
        ZStack(alignment: .top) {
            Image(decorative: image, scale: 1)
                .resizable()
                .interpolation(.high)
                .scaledToFill()
                .frame(width: side, height: side)
                .clipped()
            LinearGradient(colors: [.black.opacity(0.78), .black.opacity(0.5), .black.opacity(0)],
                           startPoint: .top, endPoint: .bottom)
                .frame(height: side * 0.44)
            Text(text)
                .font(.system(size: 44 * u, weight: .semibold))
                .foregroundStyle(.white)
                .shadow(color: .black.opacity(0.55), radius: 3 * u)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 18 * u)
                .padding(.top, 10 * u)
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: 22 * u))
    }
}

/// One answer: the question in short and the answer large on one line with its probability at the right, a thick bar
/// of that probability under them. A line too long for the page (a long answer) puts the probability beside the bar.
private struct StoryAnswer: View {
    let label: String
    let answer: Answer?
    let u: CGFloat

    var body: some View {
        let color = answer.map { Style.answer($0.question, $0.option) } ?? Style.pending
        ViewThatFits(in: .horizontal) {
            VStack(spacing: 6 * u) {
                HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                    words
                    probability(color)
                }
                bar(color)
            }
            VStack(spacing: 6 * u) {
                words
                HStack(alignment: .center, spacing: 10 * u) {
                    bar(color)
                    probability(color)
                }
            }
        }
    }

    private var words: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
            Text(label)
                .font(.system(size: 30 * u, weight: .semibold))
                .foregroundStyle(Style.latency)
            Text(answer.map(StoryWords.answer) ?? " ")
                .font(.system(size: 48 * u, weight: .bold))
                .foregroundStyle(.white)
        }
        .lineLimit(1)
        .fixedSize()
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func probability(_ color: Color) -> some View {
        Text(answer.map(StoryWords.p) ?? " ")
            .font(.system(size: 30 * u, weight: .semibold).monospacedDigit())
            .foregroundStyle(color)
            .lineLimit(1)
            .fixedSize()
    }

    private func bar(_ color: Color) -> some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(Style.lane)
                Capsule().fill(color).frame(width: max(12 * u, g.size.width * (answer?.p ?? 0)))
            }
        }
        .frame(height: 12 * u)
    }
}

/// The seconds the app timed, large, and what they cover.
private struct SecondsLine: View {
    let seconds: String
    let tail: String
    let u: CGFloat

    var body: some View {
        let s = Text(seconds).font(.system(size: 46 * u, weight: .bold).monospacedDigit()).foregroundStyle(.white)
        let t = Text(tail).font(.system(size: 30 * u, weight: .semibold)).foregroundStyle(Style.latency)
        return Text("\(s)\(t)")
            .lineLimit(1)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// The whole inbox: how many messages and answers, the seconds of the run that decided all of them.
private struct ScalePage: View {
    let model: DemoModel
    let u: CGFloat

    var body: some View {
        let w = Dictionary(StoryWords.scale(model), uniquingKeysWith: { a, _ in a })
        VStack(alignment: .leading, spacing: 8 * u) {
            Text(w["messages"] ?? "")
            Text(w["answers"] ?? "")
            Text(w["total"] ?? "")
                .font(.system(size: 112 * u, weight: .bold).monospacedDigit())
                .foregroundStyle(Style.blue)
                .padding(.top, 6 * u)
            Text(w["where"] ?? "")
                .font(.system(size: 34 * u, weight: .semibold))
                .foregroundStyle(Style.latency)
        }
        .font(.system(size: 50 * u, weight: .bold))
        .foregroundStyle(.white)
        .lineLimit(1)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .padding(.horizontal, 26 * u)
    }
}

/// The order log: its size, its first question, the answer with its probability, the seconds for its questions.
private struct LogPage: View {
    let model: DemoModel
    let u: CGFloat

    var body: some View {
        let w = Dictionary(StoryWords.log(model), uniquingKeysWith: { a, _ in a })
        let story = model.story
        let a = model.logResult?.answers.first
        let color = a.map { Style.answer($0.question, $0.option) } ?? Style.pending
        VStack(alignment: .leading, spacing: 0) {
            Spacer(minLength: 0)
            Text(w["title"] ?? "")
                .font(.system(size: 40 * u, weight: .bold))
                .foregroundStyle(Style.latency)
                .lineLimit(1)
            Text(w["question"] ?? "")
                .font(.system(size: 40 * u, weight: .semibold))
                .foregroundStyle(.white)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 34 * u)
            VStack(alignment: .leading, spacing: 10 * u) {
                Text(w["answer"] ?? " ")
                    .font(.system(size: 56 * u, weight: .bold))
                    .foregroundStyle(.white)
                    .lineLimit(1)
                HStack(alignment: .center, spacing: 14 * u) {
                    GeometryReader { g in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Style.lane)
                            Capsule().fill(color).frame(width: max(14 * u, g.size.width * (a?.p ?? 0)))
                        }
                    }
                    .frame(height: 14 * u)
                    Text(w["p"] ?? " ")
                        .font(.system(size: 40 * u, weight: .semibold).monospacedDigit())
                        .foregroundStyle(color)
                        .lineLimit(1)
                }
            }
            .opacity(a != nil && story.revealed > 0 ? 1 : 0)
            .offset(y: a != nil && story.revealed > 0 ? 0 : 10 * u)
            .animation(.easeOut(duration: 0.25), value: story.revealed)
            .padding(.top, 26 * u)
            SecondsLine(seconds: w["seconds"] ?? " ", tail: w["seconds_tail"] ?? "", u: u)
                .opacity(a != nil && story.secondsShown ? 1 : 0)
                .animation(.easeOut(duration: 0.25), value: story.secondsShown)
                .padding(.top, 40 * u)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 22 * u)
    }
}
