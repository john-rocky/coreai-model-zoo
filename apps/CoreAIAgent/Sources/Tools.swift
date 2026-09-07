import EventKit
import Foundation
import FoundationModels
import UIKit

// Three tools that touch the phone. The model never sees these implementations —
// only the name, description and @Generable argument schema that FoundationModels
// derives from them; the ZooFMProvider dialect renders those into MiniCPM5's
// `<tools>` block and parses its `<function …>` calls back into typed arguments.

/// Shared EventKit store + access requests (iOS 17+ full-access API).
enum Calendar_ {
    // EKEventStore is documented thread-safe for reads/saves; Swift 6 cannot see that.
    nonisolated(unsafe) static let store = EKEventStore()

    static func ensureEventsAccess() async throws {
        if EKEventStore.authorizationStatus(for: .event) != .fullAccess {
            let ok = try await store.requestFullAccessToEvents()
            if !ok { throw ToolError.denied("calendar") }
        }
    }

    static func ensureRemindersAccess() async throws {
        if EKEventStore.authorizationStatus(for: .reminder) != .fullAccess {
            let ok = try await store.requestFullAccessToReminders()
            if !ok { throw ToolError.denied("reminders") }
        }
    }

    static func day(_ which: String) -> Date? {
        let cal = Calendar.current
        let today = cal.startOfDay(for: Date())
        switch which.lowercased() {
        case "today": return today
        case "tomorrow": return cal.date(byAdding: .day, value: 1, to: today)
        default:
            let f = DateFormatter()
            f.dateFormat = "yyyy-MM-dd"
            return f.date(from: which).map { cal.startOfDay(for: $0) }
        }
    }
}

enum ToolError: LocalizedError {
    case denied(String)
    case badInput(String)
    var errorDescription: String? {
        switch self {
        case .denied(let what): return "Access to \(what) was not granted."
        case .badInput(let what): return "Could not understand \(what)."
        }
    }
}

/// Reads the calendar for one day. Returns a compact one-line-per-event list —
/// the model reasons over this text, so it is short and unambiguous.
struct CalendarEventsTool: Tool {
    let name = "get_calendar_events"
    let description = "List the events on the user's calendar for one day."

    @Generable
    struct Arguments {
        @Guide(description: "Which day: \"today\", \"tomorrow\" or a date as YYYY-MM-DD")
        var day: String
    }

    func call(arguments: Arguments) async throws -> String {
        try await Calendar_.ensureEventsAccess()
        guard let start = Calendar_.day(arguments.day) else {
            throw ToolError.badInput("day '\(arguments.day)'")
        }
        let end = Calendar.current.date(byAdding: .day, value: 1, to: start)!
        let predicate = Calendar_.store.predicateForEvents(withStart: start, end: end, calendars: nil)
        let events = Calendar_.store.events(matching: predicate).sorted { $0.startDate < $1.startDate }
        let f = DateFormatter()
        f.dateFormat = "HH:mm"
        AgentLog.shared.toolExecuted(name, summary: "\(events.count) events on \(arguments.day)")
        if events.isEmpty { return "No events on \(arguments.day)." }
        let lines = events.map { e in
            e.isAllDay
                ? "all day: \(e.title ?? "(untitled)")"
                : "\(f.string(from: e.startDate))-\(f.string(from: e.endDate)) \(e.title ?? "(untitled)")"
        }
        return "Events on \(arguments.day):\n" + lines.joined(separator: "\n")
    }
}

/// Creates a reminder with an alarm at a given day + time. The one tool that
/// changes something on the phone — the demo's proof that this is not chat.
struct CreateReminderTool: Tool {
    let name = "create_reminder"
    let description = "Create a reminder that alerts the user at a given day and time."

    @Generable
    struct Arguments {
        @Guide(description: "Short reminder text")
        var title: String
        @Guide(description: "Which day: \"today\", \"tomorrow\" or a date as YYYY-MM-DD")
        var day: String
        @Guide(description: "Alert time as HH:MM in 24-hour format")
        var time: String
    }

    func call(arguments: Arguments) async throws -> String {
        try await Calendar_.ensureRemindersAccess()
        guard let dayStart = Calendar_.day(arguments.day) else {
            throw ToolError.badInput("day '\(arguments.day)'")
        }
        let parts = arguments.time.split(separator: ":").compactMap { Int($0) }
        guard parts.count == 2, (0..<24).contains(parts[0]), (0..<60).contains(parts[1]) else {
            throw ToolError.badInput("time '\(arguments.time)'")
        }
        var comps = Calendar.current.dateComponents([.year, .month, .day], from: dayStart)
        comps.hour = parts[0]
        comps.minute = parts[1]
        let due = Calendar.current.date(from: comps)!
        let reminder = EKReminder(eventStore: Calendar_.store)
        reminder.title = arguments.title
        reminder.calendar = Calendar_.store.defaultCalendarForNewReminders()
        reminder.dueDateComponents = comps
        reminder.addAlarm(EKAlarm(absoluteDate: due))
        try Calendar_.store.save(reminder, commit: true)
        let f = DateFormatter()
        f.dateFormat = "EEE HH:mm"
        AgentLog.shared.toolExecuted(name, summary: "'\(arguments.title)' at \(f.string(from: due))")
        return "Reminder \"\(arguments.title)\" set for \(arguments.day) at \(arguments.time)."
    }
}

/// Battery and storage — a tool with no arguments and no permission prompt.
struct DeviceStatusTool: Tool {
    let name = "get_device_status"
    let description = "Get the phone's battery level, charging state and free storage."

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> String {
        let device = await MainActor.run { () -> (Float, UIDevice.BatteryState) in
            UIDevice.current.isBatteryMonitoringEnabled = true
            return (UIDevice.current.batteryLevel, UIDevice.current.batteryState)
        }
        let level = device.0 < 0 ? "unknown" : "\(Int(device.0 * 100))%"
        let state: String
        switch device.1 {
        case .charging: state = "charging"
        case .full: state = "full"
        case .unplugged: state = "on battery"
        default: state = "unknown"
        }
        var free = "unknown"
        if let values = try? URL(fileURLWithPath: NSHomeDirectory())
            .resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
            let bytes = values.volumeAvailableCapacityForImportantUsage
        {
            free = String(format: "%.1f GB", Double(bytes) / 1e9)
        }
        AgentLog.shared.toolExecuted(name, summary: "battery \(level), \(free) free")
        return "Battery \(level), \(state). Free storage: \(free)."
    }
}

/// What the UI shows while a tool runs (the transcript only carries the result
/// after the call returns). Tools call it from nonisolated contexts; the handler
/// runs on the main actor.
@MainActor
final class AgentLog {
    nonisolated static let shared = AgentLog()
    var onToolExecuted: ((String, String) -> Void)?
    nonisolated init() {}
    nonisolated func toolExecuted(_ name: String, summary: String) {
        Task { @MainActor in self.onToolExecuted?(name, summary) }
    }
}
