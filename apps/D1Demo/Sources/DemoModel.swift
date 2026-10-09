// DemoModel — the demo's state and its two actions. Load: the decider (`D1Decider`, the iPhone's int8mlp decoder and the
// tower, each `.aimodel` specialized here: `loadGraph(asset: .jit)`, the card's "Use it"), the samples, then one hidden
// warm-up decision per request shape (a text, a picture, the long log; none of them shown). Triage: the 12 inbox
// messages one after another, each its own `decide(requestJSON:images:shared: true)`. Ask: the order log's three
// questions in one request. Every second on screen is the app's own ContinuousClock around those `decide` calls (the
// pictures' decoding and the tower included, the load not); the run log (`RunLog`) keeps the same numbers and the
// response bodies. The video's pages (StoryView) read the same results; `story` is where they are.

import CoreGraphics
import D1
import Foundation
import Network
import Observation
#if os(iOS)
import UIKit
#endif

struct ItemResult: Sendable {
    let seconds: Double
    let answers: [Answer]
    let response: String
}

@MainActor
@Observable
final class DemoModel {
    enum Screen { case inbox, log, end, story }

    var screen: Screen = .inbox
    var inboxPhase: Phase = .loading
    var logPhase: Phase = .loading
    var detail = "loading the model"
    private(set) var demo: DemoSet?
    var results: [String: ItemResult] = [:]
    /// the item the spotlight shows: while triaging the one being decided; after, the one opened (nil = the summary)
    var spotlight: String?
    var inboxStart: Date?
    var inboxTotal: Double?
    var logStart: Date?
    var logResult: ItemResult?
    var logScroll = 0
    var offline = false
    var network: [String: String] = [:]
    var loadSeconds: Double?
    var warmups: [(id: String, seconds: Double)] = []
    var failure: String?
    /// the video's pages (`-autoplay story`)
    let story = Story()

    let config: DemoConfig
    private var decider: D1Decider?
    private let monitor = NWPathMonitor()

    init(config: DemoConfig) {
        self.config = config
        monitor.pathUpdateHandler = { [weak self] path in
            // offline = no Wi-Fi and no cellular path (a USB link to the Mac is neither)
            let kinds = path.availableInterfaces.map { "\($0.type)" }
            let online = path.status == .satisfied && path.availableInterfaces.contains { $0.type == .wifi || $0.type == .cellular }
            let status = "\(path.status)"
            Task { @MainActor in
                self?.offline = !online
                self?.network = ["status": status, "interfaces": kinds.joined(separator: ","), "offline": "\(!online)"]
            }
        }
        monitor.start(queue: DispatchQueue(label: "d1demo.path"))
    }

    var inbox: [Samples.Item] { demo?.samples.inbox ?? [] }
    var doneCount: Int { inbox.filter { results[$0.id] != nil }.count }
    var answerCount: Int { inbox.reduce(0) { $0 + (results[$1.id]?.answers.count ?? 0) } }

    // ------------------------------------------------------------------------------------------------------ load
    func load() async {
        guard decider == nil, failure == nil else { return }
        do {
            let paths = try config.paths()
            demo = try DemoSet(root: paths.demo)
            spotlight = inbox.first?.id
            detail = config.asset == .jit ? "specializing the model on this device" : "loading the model"
            let t0 = ContinuousClock.now
            let d1 = try await Self.open(decoder: paths.decoder, tower: paths.tower, asset: config.asset)
            loadSeconds = Self.seconds(t0.duration(to: .now))
            decider = d1
            detail = "warming up"
            for id in demo?.samples.warmup ?? [] {
                guard let data = demo?.requests[id] else { continue }
                let (_, s) = try await Self.timedDecide(d1, data, demo?.pictures[id] ?? [])
                warmups.append((id, s))
            }
            inboxPhase = .ready
            logPhase = .ready
            detail = ""
        } catch {
            failure = "\(error)"
            detail = failure ?? ""
            inboxPhase = .failed
            logPhase = .failed
        }
    }

    @concurrent nonisolated static func open(decoder: URL, tower: URL, asset: D1Graph.Asset) async throws -> D1Decider {
        let d1 = try await D1Decider(bundle: decoder)
        try await d1.loadGraph(asset: asset, tower: tower, towerAsset: asset)
        return d1
    }

    /// One decision, timed where it runs: the app's ContinuousClock around `decide` alone.
    @concurrent nonisolated static func timedDecide(_ d1: D1Decider, _ data: Data, _ images: [URL]) async throws
        -> (JSONValue, Double)
    {
        let t0 = ContinuousClock.now
        let body = try await d1.decide(requestJSON: data, images: images, shared: true)
        return (body, seconds(t0.duration(to: .now)))
    }

    nonisolated static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    // ---------------------------------------------------------------------------------------------------- triage
    func triage() async {
        guard inboxPhase == .ready || inboxPhase == .done, let d1 = decider, let demo else { return }
        results = [:]
        inboxTotal = nil
        inboxPhase = .running
        inboxStart = Date()
        var total = 0.0
        do {
            for item in demo.samples.inbox {
                spotlight = item.id
                guard let data = demo.requests[item.id] else { throw DemoError.sample("no request for \(item.id)") }
                let (body, s) = try await Self.timedDecide(d1, data, demo.pictures[item.id] ?? [])
                total += s
                results[item.id] = ItemResult(seconds: s, answers: try demo.answers(body, questions: item.questions),
                                              response: PythonFormat.dumps(body, indent: 2, asciiOnly: false))
            }
            inboxTotal = total
            inboxPhase = .done
        } catch {
            failure = "\(error)"
            inboxPhase = .failed
        }
    }

    // ------------------------------------------------------------------------------------------------------- ask
    func ask() async {
        guard logPhase == .ready || logPhase == .done, let d1 = decider, let demo else { return }
        logResult = nil
        logPhase = .running
        logStart = Date()
        do {
            let log = demo.samples.log
            guard let data = demo.requests[log.id] else { throw DemoError.sample("no request for the log") }
            let (body, s) = try await Self.timedDecide(d1, data, [])
            logResult = ItemResult(seconds: s, answers: try demo.answers(body, questions: log.questions),
                                   response: PythonFormat.dumps(body, indent: 2, asciiOnly: false))
            logPhase = .done
        } catch {
            failure = "\(error)"
            logPhase = .failed
        }
    }

    // ---------------------------------------------------------------------------------------------------- words
    /// The DONE line of the inbox, the one the run log keeps beside the seconds it came from.
    var inboxDoneLine: String {
        guard let t = inboxTotal else { return " " }
        return "\(doneCount) messages · \(answerCount) answers · " + Self.secs(t) + " · on device"
    }

    var logDoneLine: String {
        guard let r = logResult, let demo else { return " " }
        return "\(r.answers.count) answers · \(demo.samples.log.entries) entries · " + Self.secs(r.seconds) + " · on device"
    }

    nonisolated static func secs(_ s: Double) -> String { String(format: "%.2f s", s) }

    /// tenths while running; at DONE the run's recorded total, rounded like the DONE line
    func clock(start: Date?, total: Double?, phase: Phase, now: Date) -> String {
        if phase == .done, let total { return String(format: "%.1f s", total) }
        guard phase == .running, let start else { return "0.0 s" }
        let tenths = Int(now.timeIntervalSince(start) * 10)
        return "\(tenths / 10).\(tenths % 10) s"
    }

    var footer: String { "d1-3B (Liquid AI) · int8 MLP · Core AI GPU · " + DeviceInfo.name }

    /// The summary at DONE: the routing the answers add up to.
    var summary: (teams: [(String, Int)], refunds: Int, urgent: Int, damaged: Int, opened: Int, pictures: Int) {
        var teams: [String: Int] = [:]
        var refunds = 0, urgent = 0, damaged = 0, opened = 0, pictures = 0
        for item in inbox {
            guard let r = results[item.id] else { continue }
            if item.kind == "picture" { pictures += 1 }
            for a in r.answers {
                switch a.question {
                case "team": teams[a.text, default: 0] += 1
                case "refund": if a.option == "yes" { refunds += 1 }
                case "urgency": if a.option == "2" { urgent += 1 }
                case "damage": if a.option != "0" { damaged += 1 }
                case "opened": if a.option == "yes" { opened += 1 }
                default: break
                }
            }
        }
        let order = ["Billing", "Shipping", "Technical", "Fraud"]
        let t = order.compactMap { k in teams[k].map { (k, $0) } } + teams.filter { !order.contains($0.key) }.map { ($0.key, $0.value) }
        return (t, refunds, urgent, damaged, opened, pictures)
    }
}

/// Where the assets are and how to load them: the iPhone reads Library/Application Support/D1Assets (pushed by
/// _install.sh) and specializes each `.aimodel` (`.jit`); the Mac takes -assets / -decoder / -tower and -asset.
struct DemoConfig {
    let assets: URL?
    let decoder: URL?
    let tower: URL?
    let asset: D1Graph.Asset

    static func fromDefaults() -> DemoConfig {
        let d = UserDefaults.standard
        func url(_ k: String) -> URL? { d.string(forKey: k).map { URL(filePath: ($0 as NSString).expandingTildeInPath) } }
        let asset = D1Graph.Asset(rawValue: d.string(forKey: "asset") ?? "") ?? .jit
        return DemoConfig(assets: url("assets"), decoder: url("decoder"), tower: url("tower"), asset: asset)
    }

    func paths() throws -> (decoder: URL, tower: URL, demo: URL) {
        let root = assets ?? URL.applicationSupportDirectory.appending(path: "D1Assets")
        let p = (decoder ?? root.appending(path: "decoder"), tower ?? root.appending(path: "tower"), root.appending(path: "demo"))
        for (what, u) in [("decoder", p.0), ("tower", p.1), ("demo", p.2)] where !FileManager.default.fileExists(atPath: u.path) {
            throw DemoError.assets("no \(what) at \(u.path)")
        }
        return p
    }
}

enum DeviceInfo {
    static var identifier: String {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
    }

    /// What the device is called on the video's pages.
    static var kind: String {
        #if os(iOS)
        return "iPhone"
        #else
        return "Mac"
        #endif
    }

    /// The marketing name of the phone the card measured, else the identifier.
    static var name: String {
        #if os(iOS)
        return identifier == "iPhone19,2" ? "iPhone 18 Pro" : identifier
        #else
        return "Mac"
        #endif
    }

    /// The volume's free space for important use, in GB (devicectl does not report it).
    static var freeGB: Double? {
        let v = try? URL.homeDirectory.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage.map { Double($0) / 1e9 }
    }

    static var thermal: String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    @MainActor static var battery: [String: String] {
        #if os(iOS)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let s: String
        switch UIDevice.current.batteryState {
        case .charging: s = "charging"
        case .full: s = "full"
        case .unplugged: s = "unplugged"
        default: s = "unknown"
        }
        return ["level": String(format: "%.2f", UIDevice.current.batteryLevel), "state": s]
        #else
        return [:]
        #endif
    }
}
