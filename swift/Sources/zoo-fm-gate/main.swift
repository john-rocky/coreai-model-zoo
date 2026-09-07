// zoo-fm-gate — verification harness for ZooFMProvider (macOS).
//
// One scenario per process (a finished turn's background drain keeps the GPU
// busy past the last respond; separate processes keep timings clean):
//
//   swift run -c release zoo-fm-gate <bundle-dir> chat        # plain-chat regress, streamed
//   swift run -c release zoo-fm-gate <bundle-dir> tools       # 2-tool sequential scenario
//   swift run -c release zoo-fm-gate <bundle-dir> multiturn   # 3-turn KV-reuse measurement
//
// ZOO_FM_DEBUG=1 prints the executor's KV fast-path / reset decisions.

import Foundation
import FoundationModels
import ZooFMProvider

struct WeatherTool: Tool {
    let name = "get_weather"
    let description = "Get the current weather for a city."

    @Generable
    struct Arguments {
        @Guide(description: "Name of the city, in English")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_weather(city: \(arguments.city))")
        ToolCallLog.shared.record(name)
        return "Sunny, 24 degrees Celsius in \(arguments.city)."
    }
}

struct LocalTimeTool: Tool {
    let name = "get_local_time"
    let description = "Get the current local time in a city."

    @Generable
    struct Arguments {
        @Guide(description: "Name of the city, in English")
        var city: String
    }

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_local_time(city: \(arguments.city))")
        ToolCallLog.shared.record(name)
        return "It is 8:30 PM in \(arguments.city)."
    }
}

struct FavoriteCityTool: Tool {
    let name = "get_favorite_city"
    let description = "Get the user's favorite city. Takes no input."

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_favorite_city()")
        ToolCallLog.shared.record(name)
        return "The user's favorite city is Sapporo."
    }
}

// MARK: - apps/CoreAIAgent mirror (fixed data; same names/descriptions/schemas)

struct MockCalendarEventsTool: Tool {
    let name = "get_calendar_events"
    let description = "List the events on the user's calendar for one day."

    @Generable
    struct Arguments {
        @Guide(description: "Which day: \"today\", \"tomorrow\" or a date as YYYY-MM-DD")
        var day: String
    }

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_calendar_events(day: \(arguments.day))")
        ToolCallLog.shared.record(name)
        return "Events on \(arguments.day):\n10:00-10:30 Standup\n13:00-14:00 Lunch with Ken\n16:00-17:00 Design review"
    }
}

struct MockCreateReminderTool: Tool {
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
        print("  [tool] create_reminder(title: \(arguments.title), day: \(arguments.day), time: \(arguments.time))")
        ToolCallLog.shared.record("\(name)@\(arguments.day) \(arguments.time)")
        return "Reminder \"\(arguments.title)\" set for \(arguments.day) at \(arguments.time)."
    }
}

struct MockDeviceStatusTool: Tool {
    let name = "get_device_status"
    let description = "Get the phone's battery level, charging state and free storage."

    @Generable
    struct Arguments {}

    func call(arguments: Arguments) async throws -> String {
        print("  [tool] get_device_status()")
        ToolCallLog.shared.record(name)
        return "Battery 63%, on battery. Free storage: 4.2 GB."
    }
}

/// Records which tools the framework actually executed.
final class ToolCallLog: @unchecked Sendable {
    static let shared = ToolCallLog()
    private let lock = NSLock()
    private var calls: [String] = []

    func record(_ name: String) {
        lock.lock()
        calls.append(name)
        lock.unlock()
    }
    var recorded: [String] {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }
}

/// Captures the tool definitions the framework hands the executor, so the
/// mock gate can re-render the finished transcript through each dialect.
final class ToolDefLog: @unchecked Sendable {
    static let shared = ToolDefLog()
    private let lock = NSLock()
    private var defs: [Transcript.ToolDefinition] = []

    func record(_ definitions: [Transcript.ToolDefinition]) {
        lock.lock()
        if defs.isEmpty { defs = definitions }
        lock.unlock()
    }
    var recorded: [Transcript.ToolDefinition] {
        lock.lock()
        defer { lock.unlock() }
        return defs
    }
}

func dumpTranscript(_ transcript: Transcript) {
    print("[transcript]")
    for entry in transcript {
        switch entry {
        case .instructions: print("  - instructions")
        case .prompt: print("  - prompt")
        case .toolCalls(let calls):
            let rendered = calls.map { "\($0.toolName)(\($0.arguments.jsonString))" }
            print("  - toolCalls [\(rendered.joined(separator: ", "))]")
        case .toolOutput(let output):
            print("  - toolOutput \(output.toolName)")
        case .response(let r):
            let text = r.segments.compactMap {
                if case .text(let t) = $0 { return t.content } else { return nil }
            }.joined()
            print("  - response (\(text.count) chars)")
        case .reasoning:
            print("  - reasoning")
        default:
            print("  - other")
        }
    }
}

func dumpUsage(_ usage: LanguageModelSession.Usage, label: String) {
    print(
        "[usage:\(label)] input=\(usage.input.totalTokenCount)"
            + " (cached \(usage.input.cachedTokenCount))"
            + " output=\(usage.output.totalTokenCount)"
            + " (reasoning \(usage.output.reasoningTokenCount))")
}

// MARK: - Mock model (no engine) — verifies FRAMEWORK semantics CPU-only:
// how channel events map to transcript entries, multi-call coalescing, the
// tool round trip, and usage aggregation.

struct MockLanguageModel: LanguageModel {
    typealias Executor = MockExecutor
    var capabilities: LanguageModelCapabilities {
        LanguageModelCapabilities([.toolCalling, .reasoning])
    }
    var executorConfiguration: MockExecutor.Configuration { .init() }
}

struct MockExecutor: LanguageModelExecutor {
    typealias Model = MockLanguageModel
    struct Configuration: Hashable, Sendable {}
    init(configuration: Configuration) throws {}

    nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: MockLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Mirrors ZooExecutor's event policy: usage once per turn, attached
        // to the kind of entry the turn produced. (The first version of this
        // mock sent WWDC-339-style upfront usage on `.response` — on the
        // 27.0 beta that materializes an EMPTY Response entry on tool turns,
        // which is exactly why ZooExecutor doesn't do it.)
        ToolDefLog.shared.record(request.enabledToolDefinitions)
        let sawToolOutput = request.transcript.contains { entry in
            if case .toolOutput = entry { return true } else { return false }
        }
        if !sawToolOutput {
            // First round: think, then TWO tool calls in one turn.
            await channel.send(.reasoning(action: .appendText("I need weather and time.", tokenCount: 5)))
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: UUID().uuidString, name: "get_weather",
                        action: .appendArguments(#"{"city": "Tokyo"}"#, tokenCount: 4))))
            await channel.send(
                .toolCalls(
                    action: .toolCall(
                        id: UUID().uuidString, name: "get_local_time",
                        action: .appendArguments(#"{"city": "Tokyo"}"#, tokenCount: 4))))
            await channel.send(
                .toolCalls(action: .updateMetadata(["modelID": "mock", "requestID": request.id.uuidString])))
            await channel.send(
                .toolCalls(
                    action: .updateUsage(
                        input: .init(totalTokenCount: 100, cachedTokenCount: 0),
                        output: .init(totalTokenCount: 30, reasoningTokenCount: 5))))
        } else {
            await channel.send(
                .response(action: .appendText("Sunny, 24°C, and it is 8:30 PM in Tokyo.", tokenCount: 12)))
            await channel.send(
                .response(action: .updateMetadata(["modelID": "mock", "requestID": request.id.uuidString])))
            await channel.send(
                .response(
                    action: .updateUsage(
                        input: .init(totalTokenCount: 150, cachedTokenCount: 100),
                        output: .init(totalTokenCount: 12, reasoningTokenCount: 0))))
        }
        await Task.yield()
    }
}

@main
struct ZooFMGate {
    static func main() async {
        do {
            try await run()
        } catch {
            FileHandle.standardError.write(Data("FATAL: \(error)\n".utf8))
            exit(1)
        }
    }

    static func run() async throws {
        let args = CommandLine.arguments
        if args.count == 2, args[1] == "selftest" {
            try selftest()
            return
        }
        if args.count == 2, args[1] == "mock" {
            try await mock()
            return
        }
        guard args.count >= 3 else {
            print(
                "usage: zoo-fm-gate <bundle-dir> <chat|tools|multiturn>"
                    + " | zoo-fm-gate <selftest|mock>")
            exit(2)
        }
        let bundleURL = URL(fileURLWithPath: args[1])
        let scenario = args[2]

        let t0 = Date()
        // ZOO_FM_NOTHINK=1: MiniCPM5 without its reasoning trace (the dialect's
        // enable_thinking=false prefix) — the shape a 1024-token iOS budget wants.
        let dialect: (any PromptDialect)? =
            ProcessInfo.processInfo.environment["ZOO_FM_NOTHINK"] == "1"
            ? MiniCPMDialect(thinking: .off) : nil
        let model = try await ZooLanguageModel(resourcesAt: bundleURL, dialect: dialect)
        print(String(format: "[load] %.2f s", Date().timeIntervalSince(t0)))

        switch scenario {
        case "chat": try await chat(model)
        case "tools": try await tools(model)
        case "toolchain": try await toolchain(model)
        case "agent":
            // Optional 4th arg = maximumResponseTokens (the app uses 220).
            let cap = args.count >= 4 ? Int(args[3]) : 220
            try await agent(model, maxTokens: cap)
        case "multiturn":
            // Optional 4th arg = maximumResponseTokens. Capped turns end by
            // token exhaustion, not EOS — the one case without post-EOS
            // overshoot in the cache, where the append-only fast path can hit.
            let cap = args.count >= 4 ? Int(args[3]) : nil
            try await multiturn(model, maxTokens: cap)
        default:
            print("unknown scenario: \(scenario)")
            exit(2)
        }
    }

    // CPU-only: a real LanguageModelSession over a mock executor (no engine).
    // Verifies the framework-side semantics this provider relies on.
    static func mock() async throws {
        let session = LanguageModelSession(
            model: MockLanguageModel(),
            tools: [WeatherTool(), LocalTimeTool()],
            instructions: "You are a helpful assistant."
        )
        let response = try await session.respond(to: "Weather and time in Tokyo?")
        print("[response] \(response.content)")
        dumpUsage(response.usage, label: "response")
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)

        var failures: [String] = []
        let executed = ToolCallLog.shared.recorded
        if !executed.contains("get_weather") { failures.append("get_weather not executed") }
        if !executed.contains("get_local_time") { failures.append("get_local_time not executed") }

        var toolCallsEntries = 0
        var callsInFirstEntry = 0
        var emptyResponses = 0
        var reasoningEntries = 0
        for entry in session.transcript {
            switch entry {
            case .toolCalls(let calls):
                toolCallsEntries += 1
                if toolCallsEntries == 1 { callsInFirstEntry = calls.count }
            case .response(let r):
                let text = r.segments.compactMap {
                    if case .text(let t) = $0 { return t.content } else { return nil }
                }.joined()
                if text.isEmpty { emptyResponses += 1 }
            case .reasoning: reasoningEntries += 1
            default: break
            }
        }
        print(
            "[mock] toolCallsEntries=\(toolCallsEntries) callsInFirst=\(callsInFirstEntry)"
                + " emptyResponseEntries=\(emptyResponses) reasoningEntries=\(reasoningEntries)")
        if toolCallsEntries != 1 || callsInFirstEntry != 2 {
            failures.append(
                "expected ONE toolCalls entry with TWO calls, got \(toolCallsEntries)/\(callsInFirstEntry)")
        }
        if emptyResponses > 0 {
            failures.append("upfront metadata/usage materialized \(emptyResponses) empty response entries")
        }
        if response.content.isEmpty { failures.append("final response empty") }

        // Dialect renders of the REAL framework-built transcript (the round
        // trip above produced instructions/prompt/reasoning/toolCalls/
        // toolOutput/response entries — exactly what `render` must replay).
        let defs = ToolDefLog.shared.recorded
        if defs.count != 2 { failures.append("tool definitions not captured (\(defs.count))") }

        let hermesPrompt = HermesDialect().render(
            transcript: session.transcript, tools: defs, requireToolCall: false)
        for needle in [
            "<tools>",
            #"{"type": "function", "function": {"name": "get_weather""#,
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": ",
            "<tool_response>",
            "<|im_start|>assistant\n",
        ] {
            if !hermesPrompt.contains(needle) {
                failures.append("hermes render missing: \(needle.prefix(60))")
            }
        }

        let lfmPrompt = LFMDialect().render(
            transcript: session.transcript, tools: defs, requireToolCall: false)
        for needle in [
            "List of tools: [{\"name\": \"get_weather\"",
            "<|tool_call_start|>[get_weather(city=\"Tokyo\"), get_local_time(city=\"Tokyo\")]<|tool_call_end|>",
            "<|im_start|>tool\n<|tool_response_start|>",
            "<|tool_response_end|><|im_end|>",
        ] {
            if !lfmPrompt.contains(needle) {
                failures.append("lfm render missing: \(needle.prefix(60))")
            }
        }
        if lfmPrompt.contains("<tools>") {
            failures.append("lfm render leaked the hermes <tools> block")
        }
        // Reasoning entries must not be replayed by either dialect.
        if hermesPrompt.contains("I need weather and time.")
            || lfmPrompt.contains("I need weather and time.")
        {
            failures.append("reasoning entry leaked into a render")
        }

        if failures.isEmpty {
            print("GATE PASS: mock (framework semantics + dialect renders)")
        } else {
            for failure in failures { print("GATE FAIL: \(failure)") }
            exit(1)
        }
    }

    // CPU-only: the streaming tag parser and tool-call JSON parse, fed
    // deliberately awkward chunkings (tags straddling deltas, multi-call
    // turns, partial-marker holds at flush).
    static func selftest() throws {
        func collect(_ chunks: [String]) -> [StreamTagParser.Event] {
            var parser = StreamTagParser()
            var events: [StreamTagParser.Event] = []
            for chunk in chunks { events += parser.consume(chunk) }
            events += parser.flush()
            return events
        }
        func merged(_ events: [StreamTagParser.Event]) -> (
            text: String, reasoning: String, payloads: [String]
        ) {
            var text = "", reasoning = "", payloads: [String] = []
            for event in events {
                switch event {
                case .text(let t): text += t
                case .reasoning(let r): reasoning += r
                case .toolCallPayload(let p): payloads.append(p)
                }
            }
            return (text, reasoning, payloads)
        }
        var failures: [String] = []
        func expect(_ condition: Bool, _ label: String) {
            if !condition { failures.append(label) }
        }

        // 1) think + tool call, tags straddling chunk boundaries
        let r1 = merged(
            collect([
                "<th", "ink>I should call", " the tool.</th", "ink>",
                "<tool", "_call>\n{\"name\": \"get_weather\", \"argum",
                "ents\": {\"city\": \"Tokyo\"}}\n</tool_c", "all>",
            ]))
        expect(r1.text.isEmpty, "1: no plain text expected, got '\(r1.text)'")
        expect(r1.reasoning == "I should call the tool.", "1: reasoning mismatch '\(r1.reasoning)'")
        expect(r1.payloads.count == 1, "1: expected 1 payload, got \(r1.payloads.count)")

        // 2) two tool calls in one turn, text in between
        let r2 = merged(
            collect([
                "Let me check.", "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>",
                "\n<tool_call>{\"name\":\"b\",\"argum", "ents\":{\"x\":1}}</tool_call>",
            ]))
        expect(r2.payloads.count == 2, "2: expected 2 payloads, got \(r2.payloads.count)")
        expect(r2.text.contains("Let me check."), "2: leading text lost")

        // 3) plain text with a partial-marker hold released at flush
        let r3 = merged(collect(["The answer is 1 <", "2 and 3."]))
        expect(r3.text == "The answer is 1 <2 and 3.", "3: hold-back text lost: '\(r3.text)'")

        // 4) Hermes parse: well-formed, argument-less, multi-call array,
        //    malformed
        let hermes = HermesDialect()
        let c4 = try hermes.parseToolCalls(
            #"{"name": "get_weather", "arguments": {"city": "Tokyo"}}"#, tools: [])
        expect(c4.count == 1 && c4[0].name == "get_weather", "4: name mismatch")
        expect(c4[0].argumentsJSON.contains("Tokyo"), "4: args mismatch")
        let c5 = try hermes.parseToolCalls(#"{"name": "ping"}"#, tools: [])
        expect(c5[0].argumentsJSON == "{}", "5: empty args mismatch: \(c5[0].argumentsJSON)")
        let c5b = try hermes.parseToolCalls(
            #"[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]"#,
            tools: [])
        expect(
            c5b.count == 2 && c5b[1].name == "b" && c5b[1].argumentsJSON == #"{"x":1}"#,
            "5b: hermes array form mismatch")
        do {
            _ = try hermes.parseToolCalls("not json at all", tools: [])
            expect(false, "6: malformed payload did not throw")
        } catch is ZooFMProviderError {
            // expected
        }

        // 5) unterminated tool call flushes its partial payload
        let r6 = merged(collect(["<tool_call>{\"name\": \"trunc\""]))
        expect(r6.payloads.count == 1, "7: partial payload not flushed")

        // ---- LFM dialect ----

        func collectLFM(_ chunks: [String]) -> [StreamTagParser.Event] {
            let lfm = LFMDialect()
            var parser = StreamTagParser(
                toolOpen: lfm.toolCallOpen, toolClose: lfm.toolCallClose)
            var events: [StreamTagParser.Event] = []
            for chunk in chunks { events += parser.consume(chunk) }
            events += parser.flush()
            return events
        }
        let lfm = LFMDialect()

        // 6) LFM special-token markers straddling chunk boundaries, with
        //    trailing visible text after the call block (the model card's
        //    "call first, then text" shape)
        let r8 = merged(
            collectLFM([
                "<|tool_call_st", "art|>[get_weather(city=\"To",
                "kyo\")]<|tool_call_e", "nd|>Checking now.",
            ]))
        expect(r8.payloads == ["[get_weather(city=\"Tokyo\")]"], "8: LFM payload mismatch \(r8.payloads)")
        expect(r8.text == "Checking now.", "8: trailing text lost: '\(r8.text)'")

        // 7) pythonic parse: canonical two-call block (the S6c emission)
        let p1 = try lfm.parseToolCalls(
            #"[get_weather(city="Tokyo"), get_local_time(city="Tokyo")]"#, tools: [])
        expect(p1.count == 2, "9: expected 2 calls, got \(p1.count)")
        expect(
            p1[0].name == "get_weather" && p1[0].argumentsJSON == #"{"city":"Tokyo"}"#,
            "9: first call mismatch \(p1)")
        expect(p1[1].name == "get_local_time", "9: second call mismatch")

        // 8) tolerance: single quotes, bare values with spaces, no brackets
        let p2 = try lfm.parseToolCalls(#"[navigate(to='Tokyo Station', mode=walking)]"#, tools: [])
        expect(
            p2[0].argumentsJSON == #"{"mode":"walking","to":"Tokyo Station"}"#,
            "10: tolerant values mismatch \(p2[0].argumentsJSON)")
        let p3 = try lfm.parseToolCalls(#"get_weather(city="Tokyo")"#, tools: [])
        expect(p3.count == 1, "11: bracketless call not parsed")

        // 9) tolerance: Python literals, nested containers, floats
        let p4 = try lfm.parseToolCalls(
            #"[configure(flag=True, n=3, ratio=1.5, note=None, opts={"a": [1, 2], "b": 'x'})]"#,
            tools: [])
        expect(
            p4[0].argumentsJSON
                == #"{"flag":true,"n":3,"note":null,"opts":{"a":[1,2],"b":"x"},"ratio":1.5}"#,
            "12: literal/nesting mismatch \(p4[0].argumentsJSON)")

        // 10) positional argument maps onto a single-parameter tool (schema
        //     hint path, exercised directly on the parser)
        var positional = PythonicCallParser(
            #"get_weather("Tokyo")"#, parameterHints: ["get_weather": ["city"]])
        let p5 = positional.parseCalls()
        expect(
            p5.count == 1 && p5[0].argumentsJSON == #"{"city":"Tokyo"}"#,
            "13: positional mapping failed \(p5)")

        // 11) salvage: an unparseable call is skipped, the rest executes
        let p6 = try lfm.parseToolCalls(
            #"[get_w&eather(city), get_weather(city="Tokyo")]"#, tools: [])
        expect(
            p6.count == 1 && p6[0].argumentsJSON == #"{"city":"Tokyo"}"#,
            "14: salvage failed \(p6)")

        // 12) truncated tail (stream cut mid-string) still yields the call
        let p7 = try lfm.parseToolCalls(#"[get_weather(city="Tok"#, tools: [])
        expect(
            p7.count == 1 && p7[0].argumentsJSON == #"{"city":"Tok"}"#,
            "15: truncated tail mismatch \(p7)")

        // 13) nothing salvageable throws
        do {
            _ = try lfm.parseToolCalls("no call here at all!!", tools: [])
            expect(false, "16: garbage payload did not throw")
        } catch is ZooFMProviderError {
            // expected
        }

        // 14) stray/mismatched closers must TERMINATE (no infinite loop) and
        //     still salvage the well-formed part. Each of these pinned a CPU
        //     core before the arg/list/dict loop terminators landed.
        let p8 = try lfm.parseToolCalls(#"get_weather(city="Tokyo"})"#, tools: [])
        expect(
            p8.count == 1 && p8[0].argumentsJSON == #"{"city":"Tokyo"}"#,
            "17: stray `}` after args \(p8)")
        let p9 = try lfm.parseToolCalls(#"[get_weather(city="Tokyo"]"#, tools: [])
        expect(
            p9.count == 1 && p9[0].argumentsJSON == #"{"city":"Tokyo"}"#,
            "18: stray `]` for `)` \(p9)")
        let p10 = try lfm.parseToolCalls(#"[configure(items=[1, 2)]"#, tools: [])
        expect(
            p10.count == 1 && p10[0].argumentsJSON == #"{"items":[1,2]}"#,
            "19: stray `)` inside list arg \(p10)")
        let p11 = try lfm.parseToolCalls(##"[configure(opts={"a": 1])]"##, tools: [])
        expect(
            p11.count == 1 && p11[0].argumentsJSON == #"{"opts":{"a":1}}"#,
            "20: stray `]` inside dict arg \(p11)")

        // 15) a bareword VALUE containing `[` is scanned to the next structural
        //     delimiter (parseValue treats only a LEADING `[` as a list), so it
        //     neither spins nor mis-parses as a list.
        let p12 = try lfm.parseToolCalls(#"set(expr=arr[i)"#, tools: [])
        expect(
            p12.count == 1 && p12[0].argumentsJSON == #"{"expr":"arr[i"}"#,
            "21: bareword containing `[` \(p12)")

        // 16) regression for the terminator fix: parseBareword must NOT consume
        //     the `)`, so an empty value keeps its terminator and `fn(a=)` still
        //     parses (a="").
        let p13 = try lfm.parseToolCalls(#"fn(a=)"#, tools: [])
        expect(
            p13.count == 1 && p13[0].argumentsJSON == #"{"a":""}"#,
            "22: empty value keeps its `)` terminator \(p13)")

        // 17) jsonEscaped: FM tool descriptions are natural language, so a quote
        //     or newline must be escaped to keep the <tools> / List-of-tools JSON
        //     valid and one line per tool (the bug behind the shared
        //     toolDescriptorJSON helper).
        expect(jsonEscaped(#"a "quote""#) == #"a \"quote\""#, "23: jsonEscaped quote")
        expect(jsonEscaped("line\nbreak") == #"line\nbreak"#, "24: jsonEscaped newline")
        expect(jsonEscaped("tab\tend") == #"tab\tend"#, "25: jsonEscaped tab")
        expect(jsonEscaped(#"back\slash"#) == #"back\\slash"#, "26: jsonEscaped backslash")

        if failures.isEmpty {
            print("GATE PASS: selftest (parser + hermes + lfm dialects)")
        } else {
            for failure in failures { print("GATE FAIL: \(failure)") }
            exit(1)
        }
    }

    // Gate (b): plain chat — no tools attached, so the rendered system prompt
    // carries no tools block. Streams via streamResponse to prove textDelta
    // immediacy (first-delta latency ≪ turn time).
    static func chat(_ model: ZooLanguageModel) async throws {
        let session = LanguageModelSession(
            model: model, instructions: "You are a helpful assistant.")

        // Unlike Apple's adapter (wrong prewarm signature → silent no-op),
        // ZooExecutor's prewarm actually runs — this should take visible time
        // and make turn 1's first delta faster.
        let tw = Date()
        session.prewarm()
        print(String(format: "[prewarm] %.2f s", Date().timeIntervalSince(tw)))

        for prompt in ["Why is the sky blue? Answer in two sentences.", "And at sunset?"] {
            print("[prompt] \(prompt)")
            let t = Date()
            var firstDelta: TimeInterval?
            var previous = ""
            let stream = session.streamResponse(to: prompt)
            for try await snapshot in stream {
                let content = snapshot.content
                if firstDelta == nil, !content.isEmpty {
                    firstDelta = Date().timeIntervalSince(t)
                }
                let delta = String(content.dropFirst(previous.count))
                FileHandle.standardOutput.write(Data(delta.utf8))
                previous = content
            }
            print()
            let dt = Date().timeIntervalSince(t)
            print(
                String(
                    format: "[turn] %.2f s (first delta %.2f s, %d chars)",
                    dt, firstDelta ?? -1, previous.count))
            guard !previous.isEmpty else {
                print("GATE FAIL: empty response")
                exit(1)
            }
        }
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)
        print("GATE PASS: chat")
    }

    // Gate (a): two tools, a prompt needing both. The model may call them
    // one per respond round (weather → toolOutput → time → toolOutput →
    // answer) or both in one turn; the gate requires that BOTH executed and
    // the final answer is grounded on both results.
    static func tools(_ model: ZooLanguageModel) async throws {
        let session = LanguageModelSession(
            model: model,
            tools: [WeatherTool(), LocalTimeTool()],
            instructions: "You are a helpful assistant."
        )

        let prompt = "What is the weather in Tokyo right now, and what time is it there?"
        print("[prompt] \(prompt)")
        let t = Date()
        let response = try await session.respond(to: prompt)
        let dt = Date().timeIntervalSince(t)
        print("[response] \(response.content)")
        print(String(format: "[turn] %.2f s", dt))
        dumpUsage(response.usage, label: "response")
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)

        let executed = Set(ToolCallLog.shared.recorded)
        let answer = response.content.lowercased()
        var failures: [String] = []
        if !executed.contains("get_weather") { failures.append("get_weather never executed") }
        if !executed.contains("get_local_time") { failures.append("get_local_time never executed") }
        if !answer.contains("24") { failures.append("answer not grounded on weather result") }
        if !(answer.contains("8:30") || answer.contains("20:30")) {
            failures.append("answer not grounded on time result")
        }
        if failures.isEmpty {
            print("GATE PASS: tools (executed: \(ToolCallLog.shared.recorded.joined(separator: " → ")))")
        } else {
            print("GATE FAIL: \(failures.joined(separator: "; "))")
            exit(1)
        }
    }

    // Sequential tool chain: the second call DEPENDS on the first's result
    // (city unknown until get_favorite_city returns), so the model must call
    // a tool on the respond AFTER a toolOutput — the cross-round path.
    static func toolchain(_ model: ZooLanguageModel) async throws {
        let session = LanguageModelSession(
            model: model,
            tools: [FavoriteCityTool(), WeatherTool()],
            instructions: "You are a helpful assistant."
        )
        let prompt = "What is the weather in my favorite city right now?"
        print("[prompt] \(prompt)")
        let t = Date()
        let response = try await session.respond(to: prompt)
        let dt = Date().timeIntervalSince(t)
        print("[response] \(response.content)")
        print(String(format: "[turn] %.2f s", dt))
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)

        let executed = ToolCallLog.shared.recorded
        let answer = response.content.lowercased()
        var failures: [String] = []
        if executed.first != "get_favorite_city" {
            failures.append("expected get_favorite_city first, got \(executed)")
        }
        if !executed.contains("get_weather") { failures.append("get_weather never executed") }
        if !answer.contains("sapporo") { failures.append("answer not grounded on favorite city") }
        if !answer.contains("24") { failures.append("answer not grounded on weather result") }
        if failures.isEmpty {
            print("GATE PASS: toolchain (executed: \(executed.joined(separator: " → ")))")
        } else {
            print("GATE FAIL: \(failures.joined(separator: "; "))")
            exit(1)
        }
    }

    // apps/CoreAIAgent's flow with fixed data: turn 1 reads tomorrow's calendar, turn 2
    // asks for a reminder relative to the first event (the model must carry the
    // tool result across turns and do the time arithmetic), turn 3 a no-argument tool.
    static func agent(_ model: ZooLanguageModel, maxTokens: Int?) async throws {
        let session = LanguageModelSession(
            model: model,
            tools: [MockCalendarEventsTool(), MockCreateReminderTool(), MockDeviceStatusTool()],
            instructions: """
                You are an assistant running on the user's iPhone. Use the tools to read the \
                calendar, create reminders and check the device. Answer briefly.
                """)
        let options = maxTokens.map { GenerationOptions(maximumResponseTokens: $0) } ?? GenerationOptions()
        var failures: [String] = []
        let prompts = [
            "What's on my calendar tomorrow?",
            "Remind me 15 minutes before the first one.",
            "How much battery and storage do I have left?",
        ]
        for prompt in prompts {
            print("[prompt] \(prompt)")
            let t = Date()
            let response = try await session.respond(to: prompt, options: options)
            let dt = Date().timeIntervalSince(t)
            print("[response] \(response.content)")
            print(String(format: "[turn] %.2f s", dt))
            dumpUsage(response.usage, label: "response")
        }
        dumpUsage(session.usage, label: "session")
        dumpTranscript(session.transcript)
        let executed = ToolCallLog.shared.recorded
        print("[executed] \(executed.joined(separator: " → "))")
        if executed.first != "get_calendar_events" { failures.append("turn 1 did not read the calendar first") }
        if !executed.contains("create_reminder@tomorrow 09:45") {
            failures.append("turn 2 did not create the reminder at tomorrow 09:45 (got \(executed))")
        }
        if !executed.contains("get_device_status") { failures.append("turn 3 did not call get_device_status") }
        if failures.isEmpty {
            print("GATE PASS: agent")
        } else {
            print("GATE FAIL: \(failures.joined(separator: "; "))")
            exit(1)
        }
    }

    // Stretch harness: 3 short turns; with ZOO_FM_DEBUG=1 the executor logs
    // whether each turn reused the cache (fast path) or reset. Reports
    // per-turn latency and cachedTokenCount so the multi-turn tax is visible.
    static func multiturn(_ model: ZooLanguageModel, maxTokens: Int?) async throws {
        let session = LanguageModelSession(
            model: model,
            instructions: maxTokens != nil
                ? "You are a helpful assistant. Answer thoroughly and at length."
                : "You are a terse assistant. Answer in one short sentence.")
        let options =
            maxTokens.map { GenerationOptions(maximumResponseTokens: $0) } ?? GenerationOptions()
        if let maxTokens {
            print("[multiturn] maximumResponseTokens=\(maxTokens) (turns end by cap, not EOS)")
        }
        // With a cap, prompts must want MORE tokens than the cap so turns end
        // by exhaustion (no post-EOS overshoot in the cache → the append-only
        // fast path can hit). Without one, terse prompts measure the
        // EOS-ended (overshoot) regime.
        let prompts =
            maxTokens != nil
            ? [
                "Describe the planet Mars in detail.",
                "Now describe Jupiter the same way.",
                "Compare the two planets you described.",
            ]
            : [
                "Name one planet.",
                "Name another one.",
                "Which of those two is bigger?",
            ]
        for (index, prompt) in prompts.enumerated() {
            print("[prompt \(index + 1)] \(prompt)")
            let t = Date()
            let response = try await session.respond(to: prompt, options: options)
            let dt = Date().timeIntervalSince(t)
            print("[response] \(response.content)")
            print(String(format: "[turn %d] %.2f s", index + 1, dt))
            dumpUsage(response.usage, label: "turn\(index + 1)")
        }
        dumpTranscript(session.transcript)
        print("GATE PASS: multiturn (latencies above; see ZOO_FM_DEBUG for reuse decisions)")
    }
}
