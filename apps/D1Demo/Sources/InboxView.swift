// InboxView — the support inbox, one screen, dark and full height on a 9:16 column. Top to bottom: the pill (READY /
// ● TRIAGING / DONE) with the clock from the press and the messages done; "Support inbox" and the samples pill; the
// spotlight (the message being decided, large: its text or its picture and note, its three answers with their
// probabilities, its seconds; after DONE a picture message opened by a tap, else the summary of the whole inbox); the
// twelve messages as rows that fill with three answer chips; the DONE line (messages, answers, the seconds the app
// timed); a small footer. Every number is the model's or the app's clock, from DemoModel.

import SwiftUI

struct InboxView: View {
    let model: DemoModel
    let manual: Bool

    var body: some View {
        Column { u in
            VStack(alignment: .leading, spacing: 0) {
                Header(phase: model.inboxPhase, running: "TRIAGING",
                       clock: { model.clock(start: model.inboxStart, total: model.inboxTotal, phase: model.inboxPhase, now: $0) },
                       count: "\(model.doneCount) / \(max(model.inbox.count, 12))", u: u)
                    .padding(.top, 8 * u)
                HStack(alignment: .center, spacing: 8 * u) {
                    Text("Support inbox · 3 questions a message")
                        .font(.system(size: 13.5 * u, weight: .semibold))
                        .foregroundStyle(Style.latency)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    Spacer(minLength: 0)
                    SamplesPill(offline: model.offline, u: u)
                }
                .padding(.top, 8 * u)
                Spotlight(model: model, u: u)
                    .frame(height: 246 * u)
                    .padding(.top, 10 * u)
                MessageList(model: model, u: u)
                    .padding(.top, 10 * u)
                    .frame(maxHeight: .infinity, alignment: .top)
                // one fixed bar under the list: the button at READY, the DONE line after, so nothing moves and the
                // button never covers a message
                ZStack {
                    if model.inboxPhase == .ready || (manual && model.inboxPhase == .done) {
                        button(u)
                    } else if model.inboxPhase == .loading {
                        Text(model.detail)
                            .font(.system(size: 14 * u, weight: .semibold))
                            .foregroundStyle(Style.latency)
                    } else {
                        Text(model.inboxPhase == .failed ? (model.failure ?? "") : model.inboxDoneLine)
                            .font(.system(size: 13.5 * u, weight: .semibold).monospacedDigit())
                            .foregroundStyle(model.inboxPhase == .failed ? Style.red : .white)
                            .lineLimit(1)
                            .minimumScaleFactor(0.6)
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
                    if manual && model.inboxPhase == .done {
                        Button("Order log ›") { model.screen = .log }
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
        let primary = model.inboxPhase == .ready
        return Button {
            Task { await model.triage() }
        } label: {
            Text(primary ? "Triage inbox" : "Triage again")
                .font(.system(size: (primary ? 17 : 15) * u, weight: .semibold))
                .foregroundStyle(primary ? .white : Style.latency)
                .padding(.horizontal, (primary ? 34 : 24) * u)
                .padding(.vertical, (primary ? 11 : 9) * u)
                .background(primary ? Style.action : Style.lane, in: Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// The large card: one message and its answers, or the summary of the inbox after DONE.
struct Spotlight: View {
    let model: DemoModel
    let u: CGFloat

    var body: some View {
        Group {
            if let id = model.spotlight, let item = model.inbox.first(where: { $0.id == id }) {
                message(item)
            } else if model.inboxPhase == .done {
                summary
            } else {
                Color.clear
            }
        }
        .padding(12 * u)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Style.lane.opacity(0.55), in: RoundedRectangle(cornerRadius: 12 * u))
        .overlay(RoundedRectangle(cornerRadius: 12 * u).stroke(Style.dim, lineWidth: 1 * u))
    }

    private func message(_ item: Samples.Item) -> some View {
        let result = model.results[item.id]
        let number = (model.inbox.firstIndex { $0.id == item.id } ?? 0) + 1
        return VStack(alignment: .leading, spacing: 6 * u) {
            HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                Text(String(format: "#%02d", number))
                    .font(.system(size: 12 * u, weight: .bold).monospacedDigit())
                    .foregroundStyle(Style.axis)
                Text(item.kind == "picture" ? "message with a picture" : "message")
                    .font(.system(size: 12 * u, weight: .semibold))
                    .foregroundStyle(Style.axis)
                Spacer(minLength: 0)
                if let result {
                    Text(DemoModel.secs(result.seconds))
                        .font(.system(size: 13 * u, weight: .semibold).monospacedDigit())
                        .foregroundStyle(Style.latency)
                } else if model.inboxPhase == .running {
                    Text("deciding…")
                        .font(.system(size: 13 * u, weight: .semibold))
                        .foregroundStyle(Style.pending)
                }
            }
            if let image = model.demo?.images[item.id] {
                HStack(alignment: .top, spacing: 12 * u) {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 88 * u, height: 88 * u)
                        .clipShape(RoundedRectangle(cornerRadius: 8 * u))
                    Text(item.text)
                        .font(.system(size: 17 * u, weight: .medium))
                        .foregroundStyle(.white)
                        .lineLimit(4)
                        .minimumScaleFactor(0.8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 88 * u)
            } else {
                Text(item.text)
                    .font(.system(size: 18 * u, weight: .medium))
                    .foregroundStyle(.white)
                    .lineLimit(4, reservesSpace: true)
                    .minimumScaleFactor(0.8)
                    .frame(maxWidth: .infinity, minHeight: 88 * u, alignment: .topLeading)
            }
            VStack(spacing: 4 * u) {
                ForEach(Array(item.questions.enumerated()), id: \.offset) { k, q in
                    AnswerRow(label: model.demo?.display(q)?.label ?? q, answer: result?.answers[k],
                              hint: model.demo?.optionsHint(q), u: u)
                }
            }
        }
    }

    private var summary: some View {
        let s = model.summary
        return VStack(alignment: .leading, spacing: 12 * u) {
            Text("\(model.doneCount) messages triaged")
                .font(.system(size: 20 * u, weight: .bold))
                .foregroundStyle(.white)
            HStack(spacing: 8 * u) {
                ForEach(s.teams, id: \.0) { team, n in
                    VStack(spacing: 2 * u) {
                        Text("\(n)")
                            .font(.system(size: 24 * u, weight: .bold).monospacedDigit())
                        Text(team)
                            .font(.system(size: 12 * u, weight: .semibold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.7)
                    }
                    .foregroundStyle(Style.answer("team", team.lowercased()))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8 * u)
                    .background(Style.lane, in: RoundedRectangle(cornerRadius: 8 * u))
                }
            }
            line("Refund asked", "\(s.refunds)", Style.purple, "Urgent now", "\(s.urgent)", Color(rgb: 0xE53935))
            line("Pictures: damaged", "\(s.damaged) of \(s.pictures)", Style.amber, "opened", "\(s.opened)", Style.red)
        }
    }

    private func line(_ a: String, _ av: String, _ ac: Color, _ b: String, _ bv: String, _ bc: Color) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6 * u) {
            Text(a).foregroundStyle(Style.latency)
            Text(av).foregroundStyle(ac).fontWeight(.bold)
            Text("·").foregroundStyle(Style.axis)
            Text(b).foregroundStyle(Style.latency)
            Text(bv).foregroundStyle(bc).fontWeight(.bold)
        }
        .font(.system(size: 15 * u, weight: .semibold).monospacedDigit())
        .lineLimit(1)
        .minimumScaleFactor(0.7)
    }
}

/// The twelve messages: number, a photo's thumbnail or a message glyph, the text in one line, the answer chips.
struct MessageList: View {
    let model: DemoModel
    let u: CGFloat

    var body: some View {
        GeometryReader { g in
            let n = CGFloat(max(model.inbox.count, 1))
            let rowH = min(34 * u, (g.size.height - (n - 1) * 2 * u) / n)
            VStack(spacing: 2 * u) {
                ForEach(Array(model.inbox.enumerated()), id: \.element.id) { k, item in
                    row(k, item, height: rowH)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            if model.inboxPhase == .done { model.spotlight = model.spotlight == item.id ? nil : item.id }
                        }
                }
            }
        }
    }

    private func row(_ k: Int, _ item: Samples.Item, height: CGFloat) -> some View {
        let result = model.results[item.id]
        let current = model.inboxPhase == .running && model.spotlight == item.id
        return HStack(alignment: .center, spacing: 7 * u) {
            Text(String(format: "%02d", k + 1))
                .font(.system(size: 10.5 * u, weight: .medium).monospacedDigit())
                .foregroundStyle(result == nil && !current ? Style.dim : Style.axis)
                .frame(width: 15 * u, alignment: .leading)
            Group {
                if let image = model.demo?.images[item.id] {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFill()
                } else {
                    Image(systemName: "envelope.fill")
                        .resizable()
                        .scaledToFit()
                        .padding(5 * u)
                        .foregroundStyle(result == nil ? Style.dim : Style.axis)
                }
            }
            .frame(width: min(24 * u, height - 4 * u), height: min(24 * u, height - 4 * u))
            .clipShape(RoundedRectangle(cornerRadius: 4 * u))
            Text(item.text)
                .font(.system(size: 12 * u))
                .foregroundStyle(result == nil ? Style.pending : .white)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
                .layoutPriority(0)
            HStack(spacing: 4 * u) {
                if let result {
                    ForEach(Array(result.answers.enumerated()), id: \.offset) { _, a in
                        Chip(answer: a, u: u).transition(.opacity.combined(with: .offset(x: 8 * u)))
                    }
                }
            }
            .fixedSize()
            .layoutPriority(1)
            .animation(.easeOut(duration: 0.18), value: result != nil)
        }
        .padding(.horizontal, 6 * u)
        .frame(height: height)
        .background(current || (model.inboxPhase == .done && model.spotlight == item.id) ? Style.laneHi : .clear,
                    in: RoundedRectangle(cornerRadius: 6 * u))
    }
}
