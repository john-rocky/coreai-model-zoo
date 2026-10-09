// Autoplay — runs the demo hands-off for a recording (the zoo's accepted demos' pattern, kit Examples/Decide):
//
//   devicectl device process launch --device <id> com.daisukemajima.d1demo -- -autoplay all -trigger t-<epoch>.trigger \
//       -delay 1.5 -log 1 -runid <id>
//
// loads (the model, the samples, the hidden warm-up), shows the inbox at READY, waits until the trigger file exists in
// Documents (a recorder places it once the capture is rolling), waits `delay`, then presses the screen's own buttons:
// Triage inbox; after DONE it opens each picture message in turn (`detail` s each, as a tap on its row would) and the
// summary (`hold` s); then the order log at READY, scrolled through (`scroll` s), Ask; after DONE (`logHold` s) the end
// card (`end` s). `-autoplay inbox` / `-autoplay log` run one screen. With `-log 1` every step is a line of
// Documents/d1demo/autoplay.log and the run's numbers go to Documents/d1demo/run-<id>.json (`RunLog`): what the screen
// showed, the seconds it came from, every response body. Nothing else changes: the screens are the same code with the
// same buttons; this only presses them. On the Mac `-out <dir>` replaces Documents/d1demo.
//
// `-autoplay story` is the video's layout (StoryView): a blank screen at READY; after the trigger and `delay`, the title
// card comes up and the run starts behind it, the same run as Triage inbox (all twelve messages back to back, each its
// own `decide`) and then Ask on the order log. Over that run the pages show the `-items` messages one at a time
// (`msg` s the message alone, then its answers `step` s apart, then its seconds, held `itemHold` s), the inbox's count
// and total (`scaleHold` s), the log's first question (`logMsg` s, then its answer and the log's seconds, held
// `logHold` s), the end card (`end` s). A page waits for its result if the run has not reached it yet. The seconds on
// every page are the run's own (the clock around each `decide`); only the pace of the pages is set here. The run log
// keeps every page: when it came up, when its last part appeared, its words.

import D1
import Foundation
import ImageIO
import Observation
import SwiftUI

@MainActor
@Observable
final class Autoplay {
    enum Mode: String { case inbox, log, all, story }

    let mode: Mode?
    let delay: Double
    let trigger: String?
    let log: Bool
    let runID: String
    let detail: Double
    let hold: Double
    let scroll: Double
    let logHold: Double
    let end: Double
    let thermalWatch: Double
    let open: Double
    let msg: Double
    let step: Double
    let itemHold: Double
    let scaleHold: Double
    let logMsg: Double
    let storyItems: [String]
    /// `-render <dir>`: each story page, once complete, drawn at the phone's size into <dir> (a Mac run's look at them)
    let render: URL?
    private var snaps = 0
    let out: URL
    private var fired = false

    init() {
        let d = UserDefaults.standard  // -key value command-line pairs land here
        mode = d.string(forKey: "autoplay").flatMap(Mode.init(rawValue:))
        func num(_ k: String, _ v: Double) -> Double { d.object(forKey: k) == nil ? v : d.double(forKey: k) }
        delay = num("delay", 1.5)
        detail = num("detail", 1.3)
        hold = num("hold", 1.8)
        scroll = num("scroll", 1.5)
        logHold = num("logHold", 2.5)
        end = num("end", 2.5)
        thermalWatch = num("thermalWatch", 900)
        open = num("open", 2.0)
        msg = num("msg", 1.0)
        step = num("step", 0.4)
        itemHold = num("itemHold", 1.4)
        scaleHold = num("scaleHold", 3.0)
        logMsg = num("logMsg", 1.2)
        storyItems = (d.string(forKey: "items") ?? "m01,m09,m04,m12,m05").split(separator: ",").map(String.init)
        render = d.string(forKey: "render").map { URL(filePath: ($0 as NSString).expandingTildeInPath) }
        // a relative trigger path lives in Documents, where `devicectl device copy to` can put it
        trigger = d.string(forKey: "trigger").map { $0.hasPrefix("/") ? $0 : URL.documentsDirectory.appending(path: $0).path }
        log = d.bool(forKey: "log")
        runID = d.string(forKey: "runid") ?? ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
        out = d.string(forKey: "out").map { URL(filePath: ($0 as NSString).expandingTildeInPath) }
            ?? URL.documentsDirectory.appending(path: "d1demo")
    }

    /// The whole autoplayed run, once.
    func run(_ model: DemoModel) async {
        guard let mode, !fired else { return }
        fired = true
        let record = RunLog(model: model, autoplay: self)
        if mode == .log { model.screen = .log }
        if mode == .story { model.screen = .story }
        await model.load()
        guard model.failure == nil else {
            write("FAILED load: \(model.failure ?? "")")
            record.save(stage: "failed")
            await watchThermal()
            return
        }
        write("READY load \(String(format: "%.2f", model.loadSeconds ?? 0)) s, warm-up "
            + model.warmups.map { "\($0.id) \(String(format: "%.3f", $0.seconds)) s" }.joined(separator: ", ")
            + " | offline \(model.offline) \(model.network["interfaces"] ?? "") | free "
            + (DeviceInfo.freeGB.map { String(format: "%.1f GB", $0) } ?? "-") + " | thermal \(DeviceInfo.thermal)")
        record.save(stage: "ready")
        if let trigger {
            while !FileManager.default.fileExists(atPath: trigger) { try? await Task.sleep(for: .milliseconds(100)) }
            write("TRIGGER")
        }
        try? await Task.sleep(for: .seconds(delay))
        if mode == .story {
            await story(model, record)
            return
        }
        if mode == .inbox || mode == .all {
            write("TRIAGE")
            await model.triage()
            guard model.inboxPhase == .done else { return fail(record, "triage") }
            write("INBOX_DONE " + String(format: "%.4f", model.inboxTotal ?? 0) + " | " + model.inboxDoneLine)
            record.save(stage: "inbox")
            for item in model.inbox where item.kind == "picture" {
                model.spotlight = item.id
                write("DETAIL \(item.id)")
                try? await Task.sleep(for: .seconds(detail))
            }
            model.spotlight = nil
            write("SUMMARY")
            try? await Task.sleep(for: .seconds(hold))
        }
        if mode == .log || mode == .all {
            model.screen = .log
            write("LOG_READY")
            try? await Task.sleep(for: .seconds(0.8))
            let n = model.demo?.logEntries.count ?? 0
            for k in stride(from: 0, to: n, by: 2) {
                model.logScroll = k
                try? await Task.sleep(for: .seconds(scroll / Double(max(1, n / 2))))
            }
            model.logScroll = n - 1
            try? await Task.sleep(for: .seconds(0.4))
            model.logScroll = 0
            write("ASK")
            await model.ask()
            guard model.logPhase == .done else { return fail(record, "ask") }
            write("LOG_DONE " + String(format: "%.4f", model.logResult?.seconds ?? 0) + " | " + model.logDoneLine)
            record.save(stage: "log")
            try? await Task.sleep(for: .seconds(logHold))
        }
        if mode == .all {
            model.screen = .end
            write("END")
            try? await Task.sleep(for: .seconds(end))
        }
        record.save(stage: "done")
        write("DONE \(record.url.lastPathComponent)")
        await watchThermal()
    }

    /// The video's pages over one run (see the header). Every page goes into the run log as it was shown.
    private func story(_ model: DemoModel, _ record: RunLog) async {
        let story = model.story
        // the pages keep to one timeline from the title card on (each step `s` after the last one's planned time), so
        // the small lateness of every sleep does not add up over the clip; a page that waited for its result starts the
        // timeline again from the moment it got it
        let clock = ContinuousClock()
        var next = clock.now
        func wait(_ s: Double) async {
            next = next + .seconds(s)
            try? await Task.sleep(until: next, tolerance: .milliseconds(2), clock: clock)
        }
        func until(_ done: @MainActor () -> Bool) async {
            guard !done() else { return }
            while !done() { try? await Task.sleep(for: .milliseconds(10)) }
            next = clock.now
        }
        story.show(.open, words: StoryWords.open(model))
        story.full(words: [])
        write("STORY_OPEN")
        snap(model, "open")
        // the run behind the pages: Triage inbox, then Ask, as the buttons would
        let run = Task { @MainActor [self] in
            write("TRIAGE")
            await model.triage()
            guard model.inboxPhase == .done else { return }
            write("INBOX_DONE " + String(format: "%.4f", model.inboxTotal ?? 0) + " | " + model.inboxDoneLine)
            record.save(stage: "inbox")
            write("ASK")
            await model.ask()
            guard model.logPhase == .done else { return }
            write("LOG_DONE " + String(format: "%.4f", model.logResult?.seconds ?? 0) + " | " + model.logDoneLine)
            record.save(stage: "log")
        }
        await wait(open)
        for id in storyItems {
            guard model.inbox.contains(where: { $0.id == id }) else { continue }
            story.show(.item(id), id: id)
            write("STORY_ITEM \(id)")
            let first = id == storyItems.first { item in model.demo?.images[item] != nil }
            if first { snap(model, "\(id)_message") }
            await wait(msg)
            await until { model.results[id] != nil || model.inboxPhase != .running }
            guard let r = model.results[id] else { return fail(record, "story \(id)") }
            for k in 1...r.answers.count {
                story.revealed = k
                write("STORY_ANSWER \(id) \(k)")
                if first && k < r.answers.count { snap(model, "\(id)_answer\(k)") }
                await wait(step)
            }
            story.secondsShown = true
            story.full(words: StoryWords.item(model, id))
            write("STORY_FULL item \(id) " + StoryWords.seconds(r.seconds))
            snap(model, id)
            await wait(itemHold)
        }
        await until { model.inboxPhase != .running }
        guard model.inboxPhase == .done else { return fail(record, "story inbox") }
        story.show(.scale)
        story.full(words: StoryWords.scale(model))
        write("STORY_FULL scale " + StoryWords.seconds(model.inboxTotal ?? 0))
        snap(model, "scale")
        await wait(scaleHold)
        await run.value
        guard model.logPhase == .done, model.logResult != nil else { return fail(record, "story log") }
        story.show(.log)
        write("STORY_LOG")
        await wait(logMsg)
        story.revealed = 1
        write("STORY_ANSWER log 1")
        await wait(step)
        story.secondsShown = true
        story.full(words: StoryWords.log(model))
        write("STORY_FULL log " + StoryWords.seconds(model.logResult?.seconds ?? 0))
        snap(model, "log")
        await wait(logHold)
        story.show(.end)
        story.full(words: [])
        write("END")
        snap(model, "end")
        await wait(end)
        record.save(stage: "done")
        write("DONE \(record.url.lastPathComponent)")
        await watchThermal()
    }

    /// The story's page as it is now, drawn at the phone's size (402 x 874 points at 3x: 1206 x 2622 px, the phone's
    /// screenshot) into `-render <dir>` as NN_<name>.png. Only with `-render`: a Mac run's look at the pages (a locked
    /// Mac has no window to capture). Drawing takes the main actor for a moment; the run's decisions are not on it.
    private func snap(_ model: DemoModel, _ name: String) {
        guard let render else { return }
        snaps += 1
        let r = ImageRenderer(content: StoryView(model: model).frame(width: 402, height: 874)
            .environment(\.colorScheme, .dark))
        r.scale = 3
        let url = render.appending(path: String(format: "%02d_", snaps) + name + ".png")
        try? FileManager.default.createDirectory(at: render, withIntermediateDirectories: true)
        guard let cg = r.cgImage, let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
        else { return write("RENDER failed \(name)") }
        CGImageDestinationAddImage(dest, cg, nil)
        write(CGImageDestinationFinalize(dest) ? "RENDER \(url.lastPathComponent)" : "RENDER failed \(name)")
    }

    /// After the run: the thermal state every 10 s for `thermalWatch` s (a shared phone goes back cool, the script that
    /// holds it waits for "THERMAL nominal" before it lets go).
    private func watchThermal() async {
        let end = Date().addingTimeInterval(thermalWatch)
        while Date() < end {
            write("THERMAL \(DeviceInfo.thermal)")
            try? await Task.sleep(for: .seconds(10))
        }
    }

    private func fail(_ record: RunLog, _ what: String) {
        write("FAILED \(what): \(record.model.failure ?? "")")
        record.save(stage: "failed")
        Task { await watchThermal() }
    }

    /// One timestamped line of Documents/d1demo/autoplay.log (only with `-log 1`).
    func write(_ line: String) {
        guard log else { return }
        try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let url = out.appending(path: "autoplay.log")
        let stamp = ISO8601DateFormatter.withFractions.string(from: Date())
        let text = "\(stamp) \(runID) \(line)\n"
        if let h = try? FileHandle(forWritingTo: url) {
            _ = try? h.seekToEnd()
            try? h.write(contentsOf: Data(text.utf8))
            try? h.close()
        } else {
            try? text.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

extension ISO8601DateFormatter {
    @MainActor static let withFractions: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
}

/// The run's record, rewritten at every stage: the device, the network path, the load, the warm-ups, every inbox item
/// (its seconds, the words on screen, the response body as `json.dumps(indent=2)` writes it) and the log's, the DONE
/// lines as shown. Python's float repr throughout (PythonFormat), so a Mac script reads back the same doubles.
@MainActor
struct RunLog {
    let model: DemoModel
    let autoplay: Autoplay
    var url: URL { autoplay.out.appending(path: "run-\(autoplay.runID).json") }

    func save(stage: String) {
        guard autoplay.log else { return }
        func s(_ v: String) -> JSONValue { .string(v) }
        func d(_ v: Double) -> JSONValue { .double(v) }
        func obj(_ m: [(String, JSONValue)]) -> JSONValue { .object(m.map { JSONMember($0.0, $0.1) }) }
        func answers(_ a: [Answer]) -> JSONValue {
            .array(a.map { obj([("question", s($0.question)), ("option", s($0.option)), ("text", s($0.text)),
                                ("chip", s($0.chip)), ("p", d($0.p)), ("p_shown", s(String(format: "%.2f", $0.p)))]) })
        }
        var items: [JSONValue] = []
        for item in model.inbox {
            guard let r = model.results[item.id] else { continue }
            items.append(obj([("id", s(item.id)), ("kind", s(item.kind)), ("seconds", d(r.seconds)),
                              ("seconds_shown", s(DemoModel.secs(r.seconds))), ("answers", answers(r.answers)),
                              ("response", s(r.response))]))
        }
        var top: [(String, JSONValue)] = [
            ("schema", s("d1demo-run/1")), ("run_id", s(autoplay.runID)), ("stage", s(stage)),
            ("written", s(ISO8601DateFormatter.withFractions.string(from: Date()))),
            ("device", obj([("identifier", s(DeviceInfo.identifier)), ("name", s(DeviceInfo.name)),
                            ("os", s(ProcessInfo.processInfo.operatingSystemVersionString)),
                            ("thermal", s(DeviceInfo.thermal)),
                            ("free_gb", DeviceInfo.freeGB.map(d) ?? .null),
                            ("battery", obj(DeviceInfo.battery.sorted { $0.key < $1.key }.map { ($0.key, s($0.value)) }))])),
            ("network", obj(model.network.sorted { $0.key < $1.key }.map { ($0.key, s($0.value)) })),
            ("pill", s(model.offline ? "in-app samples · offline" : "in-app samples")),
            ("asset", s(model.config.asset.rawValue)),
            ("load_seconds", model.loadSeconds.map(d) ?? .null),
            ("warmup", .array(model.warmups.map { obj([("id", s($0.id)), ("seconds", d($0.seconds))]) })),
            ("autoplay", obj([("mode", s(autoplay.mode?.rawValue ?? "")), ("delay", d(autoplay.delay)),
                              ("detail", d(autoplay.detail)), ("hold", d(autoplay.hold)), ("scroll", d(autoplay.scroll)),
                              ("logHold", d(autoplay.logHold)), ("end", d(autoplay.end)), ("open", d(autoplay.open)),
                              ("msg", d(autoplay.msg)), ("step", d(autoplay.step)), ("itemHold", d(autoplay.itemHold)),
                              ("scaleHold", d(autoplay.scaleHold)), ("logMsg", d(autoplay.logMsg)),
                              ("items", .array(autoplay.storyItems.map(s)))])),
            ("inbox", obj([("items", .array(items)), ("total_seconds", model.inboxTotal.map(d) ?? .null),
                           ("done_line", s(model.inboxDoneLine)),
                           ("clock_shown", s(model.clock(start: model.inboxStart, total: model.inboxTotal,
                                                          phase: model.inboxPhase, now: Date())))])),
        ]
        if let r = model.logResult {
            top.append(("log", obj([("seconds", d(r.seconds)), ("seconds_shown", s(DemoModel.secs(r.seconds))),
                                    ("answers", answers(r.answers)), ("response", s(r.response)),
                                    ("done_line", s(model.logDoneLine)),
                                    ("clock_shown", s(model.clock(start: model.logStart, total: r.seconds,
                                                                   phase: model.logPhase, now: Date())))])))
        }
        if autoplay.mode == .story {
            let f = ISO8601DateFormatter.withFractions
            top.append(("story", .array(model.story.shown.map { p in
                obj([("page", s(p.page)), ("id", p.id.map(s) ?? .null), ("at", s(f.string(from: p.at))),
                     ("full", p.full.map { s(f.string(from: $0)) } ?? .null), ("words", obj(p.words.map { ($0.0, s($0.1)) }))])
            })))
        }
        if let f = model.failure { top.append(("failure", s(f))) }
        try? FileManager.default.createDirectory(at: autoplay.out, withIntermediateDirectories: true)
        try? PythonFormat.dumps(obj(top), indent: 1, asciiOnly: false).write(to: url, atomically: true, encoding: .utf8)
    }
}
