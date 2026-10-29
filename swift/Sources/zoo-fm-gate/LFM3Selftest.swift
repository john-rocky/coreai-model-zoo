// LFM3Selftest.swift — `zoo-fm-gate lfm3-selftest`, CPU-only, no bundle:
//
// * LFM3Dialect renders = transformers 5.20.2 `apply_chat_template` of LiquidAI/LFM3-1.7B's
//   chat_template.jinja (sha256 f45005c5…) on the same conversations (the goldens below are that output,
//   split at turn boundaries)
// * the XML call parser (`<function=` / `<parameter=`), values typed against the tool schema
// * the stream parser's think-prefill start: a prompt ending in an open `<think>` makes the first text
//   reasoning (LFM3 with thinking on); a closed `<think></think>` or no marker starts in text as before

import Foundation
import FoundationModels
import ZooFMProvider

private struct LFM3LookupSecretTool: Tool {
    let name = "lookup_secret"
    let description = "Returns the secret value stored under a key in the user's vault."

    @Generable
    struct Arguments {
        @Guide(description: "Vault key")
        var key: String
    }

    func call(arguments: Arguments) async throws -> String { "48213" }
}

private struct LFM3CreateEventTool: Tool {
    let name = "create_event"
    let description = "Creates a calendar event."

    @Generable
    struct Arguments {
        @Guide(description: "Event title")
        var title: String
        @Guide(description: "Start hour, 0-23")
        var start_hour: Int
        @Guide(description: "Length in hours")
        var duration_hours: Double
        @Guide(description: "Whole-day event")
        var all_day: Bool
        @Guide(description: "Attendee names")
        var attendees: [String]
    }

    func call(arguments: Arguments) async throws -> String { "created" }
}

// The system turn both tool goldens share: the instructions, a blank line, one json.dumps line per tool.
private let lfm3ToolsSystem =
    "<|im_start|>system\nYou are a helpful assistant.\n\n<tools>\n"
    + "{\"name\": \"lookup_secret\", \"description\": \"Returns the secret value stored under a key in the user's vault.\", \"parameters\": {\"type\": \"object\", \"title\": \"Arguments\", \"properties\": {\"key\": {\"type\": \"string\", \"description\": \"Vault key\"}}, \"required\": [\"key\"], \"additionalProperties\": false, \"x-order\": [\"key\"]}}\n"
    + "{\"name\": \"create_event\", \"description\": \"Creates a calendar event.\", \"parameters\": {\"type\": \"object\", \"title\": \"Arguments\", \"properties\": {\"title\": {\"type\": \"string\", \"description\": \"Event title\"}, \"start_hour\": {\"type\": \"integer\", \"description\": \"Start hour, 0-23\"}, \"duration_hours\": {\"type\": \"number\", \"description\": \"Length in hours\"}, \"all_day\": {\"type\": \"boolean\", \"description\": \"Whole-day event\"}, \"attendees\": {\"type\": \"array\", \"description\": \"Attendee names\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"title\", \"start_hour\", \"duration_hours\", \"all_day\", \"attendees\"], \"additionalProperties\": false, \"x-order\": [\"title\", \"start_hour\", \"duration_hours\", \"all_day\", \"attendees\"]}}\n"
    + "</tools><|im_end|>\n"

private let lfm3ToolQuestion =
    "Use the lookup_secret tool to read the value stored under the key \"alpha\", then tell me that value."
private let lfm3TwoCallQuestion =
    "Add a half-hour standup at 9 with Ann and Bo, then read the secret under the key \"beta\"."

/// transformers' renders of the conversations built in `lfm3Selftest` (ids: 359 / 367 / 429 / 105 / 22).
private let lfm3Goldens: [String: String] = [
    "tool_roundtrip_off":
        "<|startoftext|>" + lfm3ToolsSystem
        + "<|im_start|>user\n\(lfm3ToolQuestion)<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think><tool_call>\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\n48213\n</tool_response><|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>",
    "tool_roundtrip_on":
        "<|startoftext|>" + lfm3ToolsSystem
        + "<|im_start|>user\n\(lfm3ToolQuestion)<|im_end|>\n"
        + "<|im_start|>assistant\n<think>I should call the tool with key alpha.</think><tool_call>\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\n48213\n</tool_response><|im_end|>\n"
        + "<|im_start|>assistant\n<think>",
    "tool_twocalls_off":
        "<|startoftext|>" + lfm3ToolsSystem
        + "<|im_start|>user\n\(lfm3TwoCallQuestion)<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think><tool_call>\n<function=create_event>\n<parameter=title>\nStandup\n</parameter>\n<parameter=start_hour>\n9\n</parameter>\n<parameter=duration_hours>\n0.5\n</parameter>\n<parameter=all_day>\nfalse\n</parameter>\n<parameter=attendees>\n[\"Ann\", \"Bo\"]\n</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n<function=lookup_secret>\n<parameter=key>\nbeta\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\ncreated\n</tool_response><|im_end|>\n"
        + "<|im_start|>tool\n<tool_response>\n77120\n</tool_response><|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>",
    "multiturn3_on":
        "<|startoftext|>"
        + "<|im_start|>user\nRemember this code word for later: TANGERINE-7319. Reply with one short sentence.<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>Nice to meet you. I will remember it.<|im_end|>\n"
        + "<|im_start|>user\nWhat is 12 + 30? Answer briefly.<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>12 + 30 = 42.<|im_end|>\n"
        + "<|im_start|>user\nWhat was the code word I gave you at the start? Answer with the code word only.<|im_end|>\n"
        + "<|im_start|>assistant\n<think>",
    "q1_off":
        "<|startoftext|>"
        + "<|im_start|>user\nWhat is 17 + 25? Answer briefly.<|im_end|>\n"
        + "<|im_start|>assistant\n<think></think>",
]

extension ZooFMGate {
    static func lfm3Selftest() throws {
        var failures: [String] = []
        func expect(_ condition: Bool, _ label: String) {
            if !condition { failures.append(label) }
        }
        func text(_ s: String) -> [Transcript.Segment] { [.text(.init(content: s))] }
        func call(_ id: String, _ name: String, _ json: String) throws -> Transcript.ToolCall {
            Transcript.ToolCall(id: id, toolName: name, arguments: try GeneratedContent(json: json))
        }

        // ---- 1) render = transformers' apply_chat_template ----
        let tools = [
            Transcript.ToolDefinition(tool: LFM3LookupSecretTool()),
            Transcript.ToolDefinition(tool: LFM3CreateEventTool()),
        ]
        let system = Transcript.Entry.instructions(
            .init(segments: text("You are a helpful assistant."), toolDefinitions: tools))
        func roundTrip(thinking: Bool) throws -> Transcript {
            var entries: [Transcript.Entry] = [system, .prompt(.init(segments: text(lfm3ToolQuestion)))]
            if thinking { entries.append(.reasoning(.init(segments: text("I should call the tool with key alpha.")))) }
            entries.append(.toolCalls(.init([try call("c1", "lookup_secret", #"{"key":"alpha"}"#)])))
            entries.append(.toolOutput(.init(id: "c1", toolName: "lookup_secret", segments: text("48213"))))
            return Transcript(entries: entries)
        }
        let twoCalls = Transcript(entries: [
            system,
            .prompt(.init(segments: text(lfm3TwoCallQuestion))),
            .toolCalls(.init([
                // Keys out of declaration order on purpose: the replay follows the schema's x-order.
                try call("c1", "create_event",
                         #"{"attendees":["Ann","Bo"],"all_day":false,"title":"Standup","duration_hours":0.5,"start_hour":9}"#),
                try call("c2", "lookup_secret", #"{"key":"beta"}"#),
            ])),
            .toolOutput(.init(id: "c1", toolName: "create_event", segments: text("created"))),
            .toolOutput(.init(id: "c2", toolName: "lookup_secret", segments: text("77120"))),
        ])
        let multiTurn = Transcript(entries: [
            .prompt(.init(segments: text(
                "Remember this code word for later: TANGERINE-7319. Reply with one short sentence."))),
            .reasoning(.init(segments: text("The user gave a code word to remember."))),
            .response(.init(segments: text("Nice to meet you. I will remember it."))),
            .prompt(.init(segments: text("What is 12 + 30? Answer briefly."))),
            .reasoning(.init(segments: text("Simple addition: 12 plus 30 is 42."))),
            .response(.init(segments: text("12 + 30 = 42."))),
            .prompt(.init(segments: text(
                "What was the code word I gave you at the start? Answer with the code word only."))),
        ])
        let oneTurn = Transcript(entries: [.prompt(.init(segments: text("What is 17 + 25? Answer briefly.")))])
        let renders: [(String, String)] = [
            ("tool_roundtrip_off", LFM3Dialect(thinking: .off).render(
                transcript: try roundTrip(thinking: false), tools: tools, requireToolCall: false)),
            ("tool_roundtrip_on", LFM3Dialect().render(
                transcript: try roundTrip(thinking: true), tools: tools, requireToolCall: false)),
            ("tool_twocalls_off", LFM3Dialect(thinking: .off).render(
                transcript: twoCalls, tools: tools, requireToolCall: false)),
            ("multiturn3_on", LFM3Dialect().render(transcript: multiTurn, tools: [], requireToolCall: false)),
            ("q1_off", LFM3Dialect(thinking: .off).render(transcript: oneTurn, tools: [], requireToolCall: false)),
        ]
        for (name, rendered) in renders {
            let golden = lfm3Goldens[name]!
            if rendered != golden {
                let i = zip(rendered, golden).prefix { $0 == $1 }.count
                failures.append(
                    "render \(name): differs at char \(i): got '\(rendered.dropFirst(i).prefix(60))' "
                        + "want '\(golden.dropFirst(i).prefix(60))'")
            }
        }
        // A re-render is byte-stable (FoundationModels' own schema / arguments encoding is not).
        let again = LFM3Dialect(thinking: .off).render(transcript: twoCalls, tools: tools, requireToolCall: false)
        expect(again == lfm3Goldens["tool_twocalls_off"], "render: second render differs")

        // ---- 2) XML call parse ----
        let lfm3 = LFM3Dialect()
        let p1 = try lfm3.parseToolCalls(
            "\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n", tools: tools)
        expect(p1 == [ParsedToolCall(name: "lookup_secret", argumentsJSON: #"{"key":"alpha"}"#)], "parse 1 arg: \(p1)")
        let p2 = try lfm3.parseToolCalls(
            "\n<function=create_event>\n<parameter=title>\nStandup\n</parameter>\n<parameter=start_hour>\n9\n</parameter>\n"
                + "<parameter=duration_hours>\n0.5\n</parameter>\n<parameter=all_day>\nfalse\n</parameter>\n"
                + "<parameter=attendees>\n[\"Ann\", \"Bo\"]\n</parameter>\n</function>\n", tools: tools)
        expect(
            p2 == [ParsedToolCall(
                name: "create_event",
                argumentsJSON: #"{"all_day":false,"attendees":["Ann","Bo"],"duration_hours":0.5,"start_hour":9,"title":"Standup"}"#)],
            "parse typed args: \(p2)")
        let p3 = try lfm3.parseToolCalls(
            "\n<function=lookup_secret>\n<parameter=key>\nalpha\n</parameter>\n</function>\n"
                + "<function=lookup_secret>\n<parameter=key>\nbeta\n</parameter>\n</function>\n", tools: tools)
        expect(p3.map(\.argumentsJSON) == [#"{"key":"alpha"}"#, #"{"key":"beta"}"#], "parse 2 calls in a block: \(p3)")
        // No schema for this tool: a JSON literal is its value, anything else (or a JSON string) is text.
        let p4 = try lfm3.parseToolCalls(
            "<function=configure>\n<parameter=n>\n3\n</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n"
                + "<parameter=flag>\ntrue\n</parameter>\n<parameter=none>\nnull\n</parameter>\n"
                + "<parameter=list>\n[1, 2]\n</parameter>\n<parameter=opts>\n{\"a\": {\"b\": [true]}}\n</parameter>\n"
                + "<parameter=word>\nTokyo Station\n</parameter>\n<parameter=quoted>\n\"x\"\n</parameter>\n</function>",
            tools: [])
        expect(
            p4.first?.argumentsJSON
                == #"{"flag":true,"list":[1,2],"n":3,"none":null,"opts":{"a":{"b":[true]}},"quoted":"\"x\"","ratio":1.5,"word":"Tokyo Station"}"#,
            "parse untyped JSON values: \(p4)")
        // Whitespace: only one newline after the opening tag and one before </parameter> are layout; a string
        // keeps the rest, a typed value is trimmed.
        let p5 = try lfm3.parseToolCalls(
            "\n\n  <function=create_event>\n\n<parameter=title>\n\n Two words \n\n</parameter>\n"
                + "<parameter=start_hour> 7 </parameter>\n\n</function>\n\n", tools: tools)
        expect(p5.first?.argumentsJSON == #"{"start_hour":7,"title":"\n Two words \n"}"#, "parse whitespace: \(p5)")
        // A string-typed value that reads as JSON stays a string.
        let p6 = try lfm3.parseToolCalls(
            "<function=create_event>\n<parameter=title>\n123\n</parameter>\n</function>", tools: tools)
        expect(p6.first?.argumentsJSON == #"{"title":"123"}"#, "parse string-typed number: \(p6)")
        // A truncated block (no </parameter>, no </function>) keeps what arrived.
        let p7 = try lfm3.parseToolCalls("\n<function=lookup_secret>\n<parameter=key>\nalp", tools: tools)
        expect(p7.first?.argumentsJSON == #"{"key":"alp"}"#, "parse truncated: \(p7)")
        do {
            _ = try lfm3.parseToolCalls("{\"name\": \"lookup_secret\"}", tools: tools)
            expect(false, "parse: a block without <function= did not throw")
        } catch is ZooFMProviderError {
            // expected
        }
        // The dialect's own replay parses back to the same call (round trip through the stream markers).
        var marker = StreamTagParser(toolOpen: lfm3.toolCallOpen, toolClose: lfm3.toolCallClose)
        var payloads: [String] = []
        for chunk in ["<tool_", "call>\n<function=create_event>\n<parameter=title>\nStandup\n</parameter>\n",
                      "<parameter=start_hour>\n9\n</parameter>\n<parameter=duration_hours>\n0.5\n</parameter>\n",
                      "<parameter=all_day>\nfalse\n</parameter>\n<parameter=attendees>\n[\"Ann\", \"Bo\"]\n</parameter>\n",
                      "</function>\n</tool_c", "all>\n\n<tool_call>\n<function=lookup_secret>\n<parameter=key>\nbeta\n",
                      "</parameter>\n</function>\n</tool_call>"] {
            for case .toolCallPayload(let p) in marker.consume(chunk) { payloads.append(p) }
        }
        let p8 = try payloads.flatMap { try lfm3.parseToolCalls($0, tools: tools) }
        expect(
            p8.map(\.argumentsJSON) == [
                #"{"all_day":false,"attendees":["Ann","Bo"],"duration_hours":0.5,"start_hour":9,"title":"Standup"}"#,
                #"{"key":"beta"}"#,
            ], "parse streamed blocks: \(p8)")

        // ---- 3) think-prefill start of the stream parser ----
        func split(_ chunks: [String], startsInReasoning: Bool) -> (text: String, reasoning: String, payloads: Int) {
            var parser = StreamTagParser(startsInReasoning: startsInReasoning)
            var text = "", reasoning = "", payloads = 0
            for event in chunks.flatMap({ parser.consume($0) }) + parser.flush() {
                switch event {
                case .text(let t): text += t
                case .reasoning(let r): reasoning += r
                case .toolCallPayload: payloads += 1
                }
            }
            return (text, reasoning, payloads)
        }
        let onPrompt = lfm3Goldens["tool_roundtrip_on"]!
        let offPrompt = lfm3Goldens["tool_roundtrip_off"]!
        expect(StreamTagParser.promptOpensThinking(onPrompt), "prefill: `…<think>` not open")
        expect(StreamTagParser.promptOpensThinking("…assistant\n<think>\n"), "prefill: `…<think>\\n` not open")
        expect(!StreamTagParser.promptOpensThinking(offPrompt), "prefill: `…<think></think>` counted open")
        expect(!StreamTagParser.promptOpensThinking("…assistant\n<think>\n\n</think>\n\n"), "prefill: closed + newlines counted open")
        expect(!StreamTagParser.promptOpensThinking("…<|im_start|>assistant\n"), "prefill: no marker counted open")
        let s1 = split(["思考は", "ここまで。</think>答えは 42。"], startsInReasoning: true)
        expect(s1.reasoning == "思考はここまで。" && s1.text == "答えは 42。", "stream: reasoning then answer \(s1)")
        let s2 = split(["17 と 25 を足す", "</th", "ink>", "42"], startsInReasoning: true)
        expect(s2.reasoning == "17 と 25 を足す" && s2.text == "42", "stream: </think> split across chunks \(s2)")
        let s3 = split(["考えが", "終わらない"], startsInReasoning: true)
        expect(s3.reasoning == "考えが終わらない" && s3.text.isEmpty, "stream: no </think> before the end \(s3)")
        let s4 = split(["42"], startsInReasoning: StreamTagParser.promptOpensThinking(offPrompt))
        expect(s4.text == "42" && s4.reasoning.isEmpty, "stream: thinking off answers in text \(s4)")
        let s5 = split(["key alpha</think>", "<tool_call>\n<function=lookup_secret>\n</function>\n</tool_call>"],
                       startsInReasoning: true)
        expect(s5.reasoning == "key alpha" && s5.payloads == 1, "stream: reasoning then a call \(s5)")
        let s6 = split(["<think>still parsed</think>after"], startsInReasoning: false)
        expect(s6.reasoning == "still parsed" && s6.text == "after", "stream: text-start parser unchanged \(s6)")

        if failures.isEmpty {
            print("GATE PASS: lfm3-selftest (render = transformers, XML calls, think-prefill stream)")
        } else {
            for failure in failures { print("GATE FAIL: \(failure)") }
            exit(1)
        }
    }
}
