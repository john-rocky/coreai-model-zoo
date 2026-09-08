import Foundation
import FoundationModels
import Network
import Observation
import ZooFMProvider

/// One turn of the agent as the UI shows it: the pieces of a FoundationModels
/// transcript, in order, plus timing. Entries are appended live while the
/// model streams, then reconciled with `session.transcript` at the end of the
/// turn (tool calls / results only appear there).
struct TurnEntry: Identifiable, Equatable {
    enum Kind: Equatable {
        case prompt
        case reasoning
        case toolCall(name: String, arguments: String)
        case toolResult(name: String)
        case response
        case error
    }
    let id = UUID()
    var kind: Kind
    var text: String
}

struct TurnStats: Equatable {
    /// Seconds from send to the first answer text (covers the tool round trip: two prefills).
    var firstTextSeconds = 0.0
    var promptTokens = 0
    var cachedTokens = 0
    var outputTokens = 0
    var seconds = 0.0
    var toolCalls = 0
}

enum ModelSource {
    static let repo = "mlboydaisuke/MiniCPM5-2B-CoreAI"
    static let remotePath = "int8"
    static let localName = "minicpm5_2b_int8"
    static var modelsDir: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models")
    }
    static var bundleDir: URL { modelsDir.appendingPathComponent(localName) }
    static var isPresent: Bool {
        FileManager.default.fileExists(atPath: bundleDir.appendingPathComponent("metadata.json").path)
    }
}

@MainActor
@Observable
final class AgentModel {
    enum Phase: Equatable {
        case needsDownload, downloading, loading, ready, generating, failed(String)
    }

    var phase: Phase = ModelSource.isPresent ? .loading : .needsDownload
    var entries: [TurnEntry] = []
    var stats: [TurnStats] = []
    /// Off by default on the phone: the trace costs a second prefill's worth of tokens per turn.
    var thinking = false
    var online = true
    var loadSeconds = 0.0
    #if os(iOS)
    let downloader = ModelDownloader()
    #else
    let downloader = NoDownloader()   // macOS verification build: seed Documents/models by hand
    #endif

    private var session: LanguageModelSession?
    private var model: ZooLanguageModel?
    private let monitor = NWPathMonitor()

    init() {
        monitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor in self?.online = path.status == .satisfied }
        }
        monitor.start(queue: DispatchQueue(label: "net"))
        AgentLog.shared.onToolExecuted = { [weak self] name, summary in
            self?.entries.append(TurnEntry(kind: .toolResult(name: name), text: summary))
        }
    }

    // MARK: model

    func download() async {
        #if os(iOS)
        phase = .downloading
        await downloader.fetch(
            repo: "https://huggingface.co/" + ModelSource.repo,
            items: [ModelDownloader.Item(remote: ModelSource.remotePath, local: ModelSource.localName)],
            into: ModelSource.modelsDir)
        if case .failed(let why) = downloader.phase {
            phase = .failed("download: \(why)")
            return
        }
        guard ModelSource.isPresent else {
            phase = .failed("download finished but the bundle is missing")
            return
        }
        phase = .loading
        await load()
        #else
        phase = .failed("macOS build: copy the int8 bundle to \(ModelSource.bundleDir.path)")
        #endif
    }

    func load() async {
        phase = .loading
        let t0 = Date()
        do {
            let dialect = MiniCPMDialect(thinking: thinking ? .model : .off)
            let m = try await ZooLanguageModel(resourcesAt: ModelSource.bundleDir, dialect: dialect)
            model = m
            session = makeSession(m)
            loadSeconds = Date().timeIntervalSince(t0)
            phase = .ready
        } catch {
            phase = .failed("load: \(error)")
        }
    }

    /// Thinking is a dialect option, so toggling it means a fresh session.
    func setThinking(_ on: Bool) async {
        thinking = on
        entries.removeAll()
        stats.removeAll()
        usageSeen = (0, 0, 0)
        await load()
    }

    func newConversation() {
        guard let model else { return }
        entries.removeAll()
        stats.removeAll()
        usageSeen = (0, 0, 0)
        session = makeSession(model)
    }

    private func makeSession(_ model: ZooLanguageModel) -> LanguageModelSession {
        LanguageModelSession(
            model: model,
            tools: [CalendarEventsTool(), CreateReminderTool(), DeviceStatusTool()],
            instructions: """
                You are an assistant running on the user's iPhone. Use the tools to read the \
                calendar, create reminders and check the device. Answer briefly.
                """)
    }

    // MARK: self-test (AGENT_SELFTEST=1): download if needed, load, run the presets, log to stderr

    func selfTest(prompts: [String]) async {
        func log(_ line: String) { FileHandle.standardError.write(Data((line + "\n").utf8)) }
        log("[selftest] start phase=\(phase) online=\(online) thinking=\(thinking)")
        if phase == .needsDownload { await download() }
        if phase == .loading { await load() }
        guard phase == .ready else { log("[selftest] ERROR not ready: \(phase)"); return }
        log(String(format: "[selftest] loaded in %.1f s", loadSeconds))
        for prompt in prompts {
            let before = entries.count
            await send(prompt)
            for entry in entries.dropFirst(before) {
                switch entry.kind {
                case .prompt: log("[selftest] > \(entry.text)")
                case .reasoning: log("[selftest] think(\(entry.text.count) chars)")
                case .toolCall(let name, let args): log("[selftest] tool \(name) \(args)")
                case .toolResult(let name): log("[selftest] result \(name): \(entry.text.replacingOccurrences(of: "\n", with: " | "))")
                case .response: log("[selftest] < \(entry.text.replacingOccurrences(of: "\n", with: " "))")
                case .error: log("[selftest] ERROR \(entry.text)")
                }
            }
            if let t = stats.last {
                log(String(format: "[selftest] turn %.1f s (first text %.1f s) prompt=%d cached=%d out=%d tools=%d",
                           t.seconds, t.firstTextSeconds, t.promptTokens, t.cachedTokens, t.outputTokens, t.toolCalls))
            }
        }
        log("[selftest] DONE")
    }

    // MARK: turns

    func send(_ prompt: String) async {
        guard let session, phase == .ready else { return }
        phase = .generating
        entries.append(TurnEntry(kind: .prompt, text: prompt))
        let before = session.transcript.count
        let t0 = Date()
        var turn = TurnStats()
        do {
            // Short cap on purpose: the pipelined engine keeps decoding to the cap
            // after EOS (each respond pays the whole cap), and iOS caps the growing KV
            // at 1024 tokens. Without the trace the model answers in 30–60 tokens.
            // 120 without the trace: a three-event calendar answer measured 80 tokens. With the
            // trace the model needs ~400, which does not fit three turns under the iOS 1024 KV
            // cap — so the trace stays a macOS-only option.
            let options = GenerationOptions(maximumResponseTokens: thinking ? 400 : 120)
            let stream = session.streamResponse(to: prompt, options: options)
            var responseIndex: Int?
            var firstText: Date?
            for try await partial in stream {
                let text = partial.content
                if firstText == nil, !text.isEmpty {
                    firstText = Date()
                    turn.firstTextSeconds = firstText!.timeIntervalSince(t0)
                }
                if let i = responseIndex {
                    entries[i].text = text
                } else if !text.isEmpty {
                    entries.append(TurnEntry(kind: .response, text: text))
                    responseIndex = entries.count - 1
                }
            }
            turn.seconds = Date().timeIntervalSince(t0)
            reconcile(from: before, session: session, into: &turn)
        } catch {
            turn.seconds = Date().timeIntervalSince(t0)
            reconcile(from: before, session: session, into: &turn)
            entries.append(TurnEntry(kind: .error, text: "\(error)"))
        }
        stats.append(turn)
        phase = .ready
    }

    /// Insert the transcript entries the stream does not surface (reasoning,
    /// tool calls, tool outputs) in their real order, and read usage.
    private func reconcile(from start: Int, session: LanguageModelSession, into turn: inout TurnStats) {
        let new = Array(session.transcript.dropFirst(start))
        // Drop the live-only entries of this turn and rebuild from the transcript.
        if let promptIdx = entries.lastIndex(where: { $0.kind == .prompt }) {
            entries.removeSubrange((promptIdx + 1)...)
        }
        for entry in new {
            switch entry {
            case .prompt:
                break  // already shown
            case .reasoning(let r):
                let text = r.segments.compactMap { seg -> String? in
                    if case .text(let t) = seg { return t.content } else { return nil }
                }.joined()
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    entries.append(TurnEntry(kind: .reasoning, text: text))
                }
            case .toolCalls(let calls):
                for call in calls {
                    turn.toolCalls += 1
                    entries.append(
                        TurnEntry(
                            kind: .toolCall(name: call.toolName, arguments: call.arguments.jsonString),
                            text: ""))
                }
            case .toolOutput(let output):
                let text = output.segments.compactMap { seg -> String? in
                    if case .text(let t) = seg { return t.content } else { return nil }
                }.joined()
                entries.append(TurnEntry(kind: .toolResult(name: output.toolName), text: text))
            case .response(let r):
                let text = r.segments.compactMap { seg -> String? in
                    if case .text(let t) = seg { return t.content } else { return nil }
                }.joined()
                if !text.isEmpty {
                    entries.append(TurnEntry(kind: .response, text: text))
                }
            default:
                break
            }
        }
        // session.usage accumulates across turns (and across the two respond calls of a
        // tool round trip); show this turn's share.
        let usage = session.usage
        turn.promptTokens = usage.input.totalTokenCount - usageSeen.input
        turn.cachedTokens = usage.input.cachedTokenCount - usageSeen.cached
        turn.outputTokens = usage.output.totalTokenCount - usageSeen.output
        usageSeen = (usage.input.totalTokenCount, usage.input.cachedTokenCount, usage.output.totalTokenCount)
    }

    private var usageSeen: (input: Int, cached: Int, output: Int) = (0, 0, 0)
}

#if !os(iOS)
/// Stand-in for the shared ModelDownloader (which needs UIKit background tasks).
@MainActor
final class NoDownloader: ObservableObject {
    enum Phase: Equatable { case idle, failed(String), done }
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var fraction: Double = 0
    @Published private(set) var detail = ""
}
#endif
