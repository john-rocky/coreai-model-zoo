import Foundation
import FoundationModels
import Tokenizers

/// A model family's tool-calling "dialect": how tool definitions are
/// advertised in the prompt, how the model marks a tool-call block in its
/// output stream, how that block's body parses into calls, and how past
/// calls/results are replayed into the next prompt.
///
/// Why this exists: in-context format instructions do NOT override a model's
/// training prior. LFM2.5, prompted with the Hermes `<tool_call>` JSON
/// instruction, still emits its native
/// `<|tool_call_start|>[fn(arg="x")]<|tool_call_end|>` pythonic form
/// (verified on-device, gate S6c). Tool calling therefore has to speak each
/// model's native dialect; plain chat is unaffected.
///
/// Known dialects of the current zoo lineup:
/// * qwen3.5 — Hermes: `<tools>` JSON block, `<tool_call>{json}</tool_call>`
///   calls, `<tool_response>` results, ChatML framing → `HermesDialect`.
/// * LFM2.5 — `List of tools: [json…]` in system, pythonic calls between
///   `<|tool_call_start|>`/`<|tool_call_end|>` special tokens, results in
///   `<|tool_response_start|>` wrappers, ChatML framing → `LFMDialect`.
/// * granite-4.0 — Hermes tool syntax verbatim, but `<|start_of_role|>`
///   framing (per its chat template); a future dialect, not yet implemented.
/// * gemma4 — fully custom non-JSON format (`<|tool_call>call:name{…}`,
///   `<|"|>` quote token); recorded, not implemented.
/// * MiniCPM5 — Hermes-style `<tools>` JSON block + the template's guidelines,
///   calls as `<function name="…"><param name="…">…</param></function>` XML,
///   `<tool_response>` results, ChatML framing → `MiniCPMDialect`.
public protocol PromptDialect: Sendable {
    /// Short identifier for logs and debugging.
    var name: String { get }

    /// Marker pair delimiting one tool-call block in the model's output
    /// stream. The stream parser withholds the block's body and hands it to
    /// `parseToolCalls` when the close marker arrives.
    var toolCallOpen: String { get }
    var toolCallClose: String { get }

    /// Render the transcript + tool definitions into the model's prompt
    /// string (the dialect owns the whole render, framing included —
    /// dialects differ in framing, not just tool syntax).
    func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String

    /// Parse the body of ONE complete tool-call block. A block may contain
    /// several calls (LFM emits `[fn1(…), fn2(…)]` in one block). `tools`
    /// provides schema hints for tolerant parsing (e.g. mapping a positional
    /// argument onto a single-parameter tool). Throws
    /// `ZooFMProviderError.malformedToolCall` when nothing in the block is
    /// salvageable.
    func parseToolCalls(
        _ payload: String,
        tools: [Transcript.ToolDefinition]
    ) throws -> [ParsedToolCall]
}

/// One tool call recovered from a model's output block, arguments normalized
/// to a JSON object string (the form `.appendArguments` expects).
public struct ParsedToolCall: Sendable, Equatable {
    public let name: String
    public let argumentsJSON: String

    public init(name: String, argumentsJSON: String) {
        self.name = name
        self.argumentsJSON = argumentsJSON
    }
}

/// Picks the dialect for a bundle by probing its tokenizer vocab: models
/// trained with the LFM tool special tokens get the LFM dialect, everything
/// else defaults to Hermes (the de-facto open-model standard).
///
/// SCOPE: the probe only distinguishes LFM (special tokens present) from
/// Hermes (everything else). granite-4.0 and gemma4 use Hermes-INCOMPATIBLE
/// framing/syntax (see the family notes on `PromptDialect`) and would be
/// mis-rendered as Hermes by this fallback — load those with an explicit
/// `dialect:` once their dialects exist. The fallback is correct for today's
/// shipped lineup (qwen → Hermes, LFM → LFM); the call site
/// (`ZooLanguageModel.init`) documents the same caveat.
public func defaultDialect(probing tokenizer: any Tokenizer) -> any PromptDialect {
    // A vocab probe must exclude the unknown-token id: some tokenizer models
    // answer `convertTokenToId` with the unk id instead of nil (MiniCPM5's does,
    // which routed it to LFM before this check existed).
    func has(_ token: String) -> Bool {
        guard let id = tokenizer.convertTokenToId(token) else { return false }
        return id != tokenizer.unknownTokenId
    }
    if has("<|tool_call_start|>"), has("<|tool_call_end|>") {
        return LFMDialect()
    }
    // MiniCPM5 carries its XML call syntax as single vocab tokens (`<function`,
    // `<param`, `</param>`, `</function>`); a Hermes-tuned vocab tokenizes those as
    // fragments. (MiniCPM5 also owns `<tool_call>` tokens, so that probe cannot
    // separate the two.)
    if ["<function", "<param", "</param>", "</function>"].allSatisfy(has) {
        return MiniCPMDialect()
    }
    return HermesDialect()
}

// MARK: - Shared rendering helpers

extension PromptDialect {
    /// Flattens a transcript entry's segments to prompt text (structured
    /// segments render as their JSON).
    package func segmentsText(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text): return text.content
            case .structure(let structure): return structure.content.jsonString
            default: return nil
            }
        }.joined(separator: "\n")
    }

    /// JSON-encodes a tool's GenerationSchema (`{}` when encoding fails —
    /// an empty schema is still a valid prompt).
    package func schemaJSON(_ tool: Transcript.ToolDefinition) -> String {
        if let data = try? JSONEncoder().encode(tool.parameters),
            let s = String(data: data, encoding: .utf8)
        {
            return s
        }
        return "{}"
    }

    /// Property names of a tool's parameter schema, for positional-argument
    /// mapping. nil when the schema doesn't decode to an object.
    package func parameterNames(_ tool: Transcript.ToolDefinition) -> [String]? {
        guard
            let data = try? JSONEncoder().encode(tool.parameters),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let properties = object["properties"] as? [String: Any]
        else { return nil }
        return Array(properties.keys)
    }

    /// The `{"name", "description", "parameters"}` tool descriptor that both
    /// the Hermes `<tools>` block and LFM's `List of tools:` line embed (Hermes
    /// additionally wraps it in `{"type":"function","function":…}`). Name and
    /// description are JSON-escaped: FM tool descriptions are natural language,
    /// so a `"` or newline would otherwise emit invalid JSON inside the block
    /// and break the one-line-per-tool layout. Shared so the two dialects can't
    /// drift on escaping again.
    package func toolDescriptorJSON(_ tool: Transcript.ToolDefinition) -> String {
        #"{"name": "\#(jsonEscaped(tool.name))", "description": "\#(jsonEscaped(tool.description))", "parameters": \#(schemaJSON(tool))}"#
    }

    /// ChatML turn framing shared by Hermes and LFM
    /// (`<|im_start|>role\n…<|im_end|>`). The two dialects frame
    /// instructions/prompt/response turns identically and differ only in the
    /// system block and how tool calls/results render, so those are injected:
    /// `head` builds the system block from the final system text (return "" for
    /// no block), and `toolEntry` renders the dialect-specific `.toolCalls` /
    /// `.toolOutput` entries (returning nil for any entry it doesn't handle,
    /// e.g. reasoning, which is never replayed). granite/gemma have different
    /// framing and do not use this.
    package func renderChatML(
        transcript: Transcript,
        defaultSystem: String,
        head: (_ system: String) -> String,
        toolEntry: (Transcript.Entry) -> (role: String, content: String)?
    ) -> String {
        var system = defaultSystem
        var body = ""

        func append(role: String, _ content: String) {
            body += "<|im_start|>\(role)\n\(content)<|im_end|>\n"
        }

        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                system = segmentsText(instructions.segments)
            case .prompt(let prompt):
                append(role: "user", segmentsText(prompt.segments))
            case .response(let response):
                append(role: "assistant", segmentsText(response.segments))
            default:
                if let rendered = toolEntry(entry) {
                    append(role: rendered.role, rendered.content)
                }
                // else: reasoning (and any other entry) is not replayed.
            }
        }

        return head(system) + body + "<|im_start|>assistant\n"
    }
}

/// Escapes a string for embedding inside a JSON string literal.
package func jsonEscaped(_ s: String) -> String {
    var out = ""
    out.reserveCapacity(s.count)
    for scalar in s.unicodeScalars {
        switch scalar {
        case "\"": out += "\\\""
        case "\\": out += "\\\\"
        case "\n": out += "\\n"
        case "\r": out += "\\r"
        case "\t": out += "\\t"
        default:
            if scalar.value < 0x20 {
                out += String(format: "\\u%04x", scalar.value)
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
    }
    return out
}
