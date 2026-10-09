// LogView — the second beat: a day's order log (55 entries, the port's fixture `long_34k`, about 3,400 tokens of JSON
// as the model reads it) and three audit questions about it, answered in one request with Ask. Top to bottom: the pill
// (READY / ● READING / DONE) with the clock and the answers done; "Order log" and the samples pill; the log, one line
// an entry (scrolled through before Ask when autoplayed); the three questions with their answers and probabilities;
// the DONE line with the app's seconds; the footer.

import SwiftUI

struct LogView: View {
    let model: DemoModel
    let manual: Bool

    var body: some View {
        Column { u in
            VStack(alignment: .leading, spacing: 0) {
                Header(phase: model.logPhase, running: "READING",
                       clock: { model.clock(start: model.logStart, total: model.logResult?.seconds, phase: model.logPhase, now: $0) },
                       count: "\(model.logResult?.answers.count ?? 0) / \(model.demo?.samples.log.questions.count ?? 3)", u: u)
                    .padding(.top, 8 * u)
                HStack(alignment: .center, spacing: 8 * u) {
                    Text("\(model.demo?.samples.log.title ?? "Order log") · \(model.demo?.samples.log.entries ?? 0) entries, one day")
                        .font(.system(size: 13.5 * u, weight: .semibold))
                        .foregroundStyle(Style.latency)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    Spacer(minLength: 0)
                    SamplesPill(offline: model.offline, u: u)
                }
                .padding(.top, 8 * u)
                LogLines(lines: model.demo?.logEntries ?? [], target: model.logScroll, u: u)
                    .frame(height: 280 * u)
                    .padding(.top, 10 * u)
                VStack(spacing: 6 * u) {
                    ForEach(Array((model.demo?.samples.log.questions ?? []).enumerated()), id: \.offset) { k, q in
                        QuestionRow(question: model.demo?.display(q)?.label ?? q, answer: model.logResult?.answers[k],
                                    running: model.logPhase == .running, hero: k == 0, u: u)
                    }
                }
                .padding(.top, 12 * u)
                .frame(maxHeight: .infinity, alignment: .top)
                ZStack {
                    if model.logPhase == .ready || (manual && model.logPhase == .done) {
                        button(u)
                    } else {
                        Text(model.logPhase == .failed ? (model.failure ?? "") : model.logDoneLine)
                            .font(.system(size: 13.5 * u, weight: .semibold).monospacedDigit())
                            .foregroundStyle(model.logPhase == .failed ? Style.red : .white)
                            .lineLimit(1)
                            .minimumScaleFactor(0.6)
                            .opacity(model.logPhase == .done || model.logPhase == .failed ? 1 : 0)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .frame(height: 46 * u)
                .padding(.top, 6 * u)
                HStack {
                    Text(model.footer)
                        .font(.system(size: 10.4 * u).monospacedDigit())
                        .foregroundStyle(Style.axis)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    Spacer(minLength: 0)
                    if manual && model.logPhase != .running {
                        Button("‹ Inbox") { model.screen = .inbox }
                            .font(.system(size: 12 * u, weight: .semibold))
                            .buttonStyle(.plain)
                            .foregroundStyle(Style.latency)
                    }
                }
                .padding(.top, 6 * u)
                .padding(.bottom, 6 * u)
            }
            .padding(.horizontal, 16 * u)
        }
    }

    private func button(_ u: CGFloat) -> some View {
        let primary = model.logPhase == .ready
        return Button {
            Task { await model.ask() }
        } label: {
            Text(primary ? "Ask" : "Ask again")
                .font(.system(size: (primary ? 17 : 15) * u, weight: .semibold))
                .foregroundStyle(primary ? .white : Style.latency)
                .padding(.horizontal, (primary ? 44 : 24) * u)
                .padding(.vertical, (primary ? 11 : 9) * u)
                .background(primary ? Style.action : Style.lane, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// The log, one monospaced line an entry; `target` is the line kept in view (autoplay scrolls it).
struct LogLines: View {
    let lines: [String]
    let target: Int
    let u: CGFloat

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                LazyVStack(alignment: .leading, spacing: 1.5 * u) {
                    ForEach(Array(lines.enumerated()), id: \.offset) { k, line in
                        Text(line)
                            .font(.system(size: 11 * u, design: .monospaced))
                            .foregroundStyle(Style.latency)
                            .lineLimit(1)
                            .minimumScaleFactor(0.6)
                            .id(k)
                    }
                }
                .padding(10 * u)
            }
            .scrollIndicators(.visible)
            .background(Style.lane, in: RoundedRectangle(cornerRadius: 10 * u))
            // the panel's lower edge fades out, so a line cut by the edge does not read as a broken line
            .overlay(alignment: .bottom) {
                LinearGradient(colors: [Style.lane.opacity(0), Style.lane], startPoint: .top, endPoint: .bottom)
                    .frame(height: 30 * u)
                    .clipShape(UnevenRoundedRectangle(bottomLeadingRadius: 10 * u, bottomTrailingRadius: 10 * u))
                    .allowsHitTesting(false)
            }
            .onChange(of: target) { _, k in
                withAnimation(.linear(duration: 0.05)) { proxy.scrollTo(k, anchor: .bottom) }
            }
        }
    }
}

/// One audit question in full, then its answer, probability and bar.
struct QuestionRow: View {
    let question: String
    let answer: Answer?
    let running: Bool
    /// the beat's first question, drawn large
    var hero = false
    let u: CGFloat

    var body: some View {
        let color = answer.map { Style.answer($0.question, $0.option) } ?? Style.pending
        VStack(alignment: .leading, spacing: 5 * u) {
            Text(question)
                .font(.system(size: (hero ? 16 : 14) * u, weight: .semibold))
                .foregroundStyle(hero ? .white : Style.latency)
                .lineLimit(2)
                .minimumScaleFactor(0.8)
            HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                Text(answer?.text ?? (running ? "reading…" : " "))
                    .font(.system(size: (hero ? 30 : 18) * u, weight: .bold))
                    .foregroundStyle(answer == nil ? Style.pending : .white)
                Spacer(minLength: 4 * u)
                if let answer {
                    Text(String(format: "%.2f", answer.p))
                        .font(.system(size: (hero ? 24 : 16) * u, weight: .semibold).monospacedDigit())
                        .foregroundStyle(color)
                }
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3 * u).fill(Style.background)
                    RoundedRectangle(cornerRadius: 3 * u).fill(color).frame(width: g.size.width * (answer?.p ?? 0))
                }
            }
            .frame(height: (hero ? 9 : 6) * u)
        }
        .padding(.vertical, (hero ? 11 : 8) * u)
        .padding(.horizontal, 12 * u)
        .background(Style.lane, in: RoundedRectangle(cornerRadius: 10 * u))
        .overlay(RoundedRectangle(cornerRadius: 10 * u).stroke(hero && answer != nil ? Style.blue : .clear, lineWidth: 1.5 * u))
        .animation(.easeOut(duration: 0.2), value: answer != nil)
    }
}

/// The last two seconds: what ran, where to get it.
struct EndCardView: View {
    var body: some View {
        Column { u in
            VStack(spacing: 14 * u) {
                Spacer()
                Text("d1-3B")
                    .font(.system(size: 44 * u, weight: .bold))
                    .foregroundStyle(.white)
                Text("Liquid AI's decision model, on Core AI")
                    .font(.system(size: 17 * u, weight: .semibold))
                    .foregroundStyle(Style.latency)
                Text("on device · no text generated · a probability for every option")
                    .font(.system(size: 13 * u))
                    .foregroundStyle(Style.axis)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 30 * u)
                VStack(spacing: 8 * u) {
                    Text("huggingface.co/mlboydaisuke/d1-3B-CoreAI")
                    Text("github.com/john-rocky/coreai-model-zoo")
                }
                .font(.system(size: 15 * u, weight: .semibold).monospaced())
                .foregroundStyle(Color(rgb: 0x8AB4F8))
                .padding(.top, 18 * u)
                Spacer()
            }
            .frame(maxWidth: .infinity)
        }
    }
}
