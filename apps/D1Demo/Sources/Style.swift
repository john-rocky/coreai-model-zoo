// Style — the colors and the pill of the zoo's accepted decision demos (kit Examples/Decide RoomCheckView and
// Examples/TextClassify InboxScreen): background #0E1116, rows #1B2028, the state pill READY grey / ● running red /
// DONE green, sizes in units of the 9:16 column's width / 402 so a phone recording reads at feed width.

import SwiftUI

enum Style {
    static let background = Color(rgb: 0x0E1116)
    static let lane = Color(rgb: 0x1B2028)
    static let laneHi = Color(rgb: 0x252B35)
    static let pending = Color(rgb: 0x6B7280)
    static let dim = Color(rgb: 0x3C424C)
    static let axis = Color(rgb: 0x8A919C)
    static let latency = Color(rgb: 0xB8BEC6)
    static let action = Color(rgb: 0x4285F4)

    static let blue = Color(rgb: 0x4285F4)
    static let green = Color(rgb: 0x34A853)
    static let amber = Color(rgb: 0xFBBC05)
    static let red = Color(rgb: 0xEA4335)
    static let orange = Color(rgb: 0xFF7043)
    static let purple = Color(rgb: 0xAB47BC)
    static let teal = Color(rgb: 0x00ACC1)

    static func pill(_ phase: Phase, running: String) -> (text: String, color: Color) {
        switch phase {
        case .loading: return ("LOADING", Color(rgb: 0x5F6368))
        case .ready: return ("READY", Color(rgb: 0x5F6368))
        case .running: return ("● \(running)", Color(rgb: 0xE53935))
        case .done: return ("DONE", Color(rgb: 0x2E7D32))
        case .failed: return ("FAILED", Color(rgb: 0xE53935))
        }
    }

    /// The color of one answer: by question and the option it picked.
    static func answer(_ question: String, _ option: String) -> Color {
        switch (question, option) {
        case ("team", "billing"): return blue
        case ("team", "shipping"): return green
        case ("team", "technical"): return amber
        case ("team", "fraud"): return red
        case ("refund", "yes"): return purple
        case ("urgency", "2"): return Color(rgb: 0xE53935)
        case ("urgency", "1"): return orange
        case ("damage", "0"): return green
        case ("damage", "1"): return amber
        case ("damage", "2"): return red
        case ("opened", "yes"): return red
        case ("boxes", _): return teal
        case ("refunded_twice", _): return blue
        case ("big_refunds_approved", "no"): return red
        case ("big_refunds_approved", "yes"): return green
        case ("cancellations", _): return teal
        default: return axis
        }
    }
}

enum Phase: Equatable { case loading, ready, running, done, failed }

extension Color {
    /// 0xRRGGBB, sRGB.
    init(rgb: UInt32) {
        self.init(
            .sRGB, red: Double((rgb >> 16) & 0xFF) / 255, green: Double((rgb >> 8) & 0xFF) / 255,
            blue: Double(rgb & 0xFF) / 255)
    }
}

/// The 9:16 column every screen draws in: `u` is a point at 402 wide; a wider window gets bands at the sides.
struct Column<Content: View>: View {
    @ViewBuilder let content: (CGFloat) -> Content

    var body: some View {
        GeometryReader { geo in
            let width = min(geo.size.width, geo.size.height * 9 / 16)
            content(width / 402)
                .frame(width: width, height: geo.size.height)
                .frame(maxWidth: .infinity)
        }
        .background(Style.background.ignoresSafeArea())
    }
}

/// The pill, the clock (tenths while running, the run's recorded total at DONE) and a count on the right.
struct Header: View {
    let phase: Phase
    let running: String
    let clock: (Date) -> String
    let count: String
    let u: CGFloat

    var body: some View {
        TimelineView(.animation(minimumInterval: 0.1, paused: phase != .running)) { context in
            let (text, color) = Style.pill(phase, running: running)
            HStack(alignment: .firstTextBaseline, spacing: 12 * u) {
                Text(text)
                    .font(.system(size: 13.4 * u, weight: .bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8 * u)
                    .padding(.vertical, 5 * u)
                    .background(color, in: Capsule())
                Text(clock(context.date))
                    .font(.system(size: 26.8 * u).monospacedDigit())
                    .foregroundStyle(.white)
                Spacer(minLength: 0)
                Text(count)
                    .font(.system(size: 16.4 * u, weight: .bold).monospacedDigit())
                    .foregroundStyle(.white)
            }
        }
    }
}

/// "in-app samples · offline": the samples ship with the app; "offline" only while the phone has no Wi-Fi and no
/// cellular path (NWPathMonitor), so the pill never claims what the phone does not show.
struct SamplesPill: View {
    let offline: Bool
    let u: CGFloat

    var body: some View {
        Text(offline ? "in-app samples · offline" : "in-app samples")
            .font(.system(size: 11.5 * u, weight: .semibold))
            .foregroundStyle(Style.latency)
            .padding(.horizontal, 8 * u)
            .padding(.vertical, 3.5 * u)
            .background(Style.lane, in: Capsule())
            .overlay(Capsule().stroke(Style.dim, lineWidth: 1 * u))
    }
}

/// One answer as a row: the question in short, the answer and its probability, a bar of that probability.
struct AnswerRow: View {
    let label: String
    let answer: Answer?
    var hint: String? = nil
    let u: CGFloat
    var large = false

    var body: some View {
        let color = answer.map { Style.answer($0.question, $0.option) } ?? Style.pending
        VStack(alignment: .leading, spacing: 4 * u) {
            HStack(alignment: .firstTextBaseline, spacing: 8 * u) {
                Text(label)
                    .font(.system(size: (large ? 15 : 13.5) * u, weight: .semibold))
                    .foregroundStyle(answer == nil ? Style.pending : Style.latency)
                    .lineLimit(1)
                    .minimumScaleFactor(0.7)
                Spacer(minLength: 4 * u)
                if answer == nil, let hint {
                    Text(hint)
                        .font(.system(size: 12 * u, weight: .medium))
                        .foregroundStyle(Style.pending)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                }
                if let answer {
                    Text(answer.text)
                        .font(.system(size: (large ? 16 : 14) * u, weight: .bold))
                        .foregroundStyle(.white)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                    Text(String(format: "%.2f", answer.p))
                        .font(.system(size: (large ? 15 : 13.5) * u, weight: .semibold).monospacedDigit())
                        .foregroundStyle(color)
                        .frame(width: 40 * u, alignment: .trailing)
                }
            }
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3 * u).fill(Style.background)
                    RoundedRectangle(cornerRadius: 3 * u).fill(color)
                        .frame(width: g.size.width * (answer?.p ?? 0))
                }
            }
            .frame(height: 6 * u)
        }
        .padding(.vertical, 5 * u)
        .padding(.horizontal, 10 * u)
        .background(Style.lane, in: RoundedRectangle(cornerRadius: 8 * u))
    }
}

/// A small colored answer: the list rows' chips.
struct Chip: View {
    let answer: Answer
    let u: CGFloat

    var body: some View {
        Text(answer.chip)
            .font(.system(size: 10.4 * u, weight: .bold))
            .foregroundStyle(Style.answer(answer.question, answer.option))
            .lineLimit(1)
            .padding(.horizontal, 6 * u)
            .padding(.vertical, 2.5 * u)
            .background(Style.lane, in: RoundedRectangle(cornerRadius: 6 * u))
    }
}
