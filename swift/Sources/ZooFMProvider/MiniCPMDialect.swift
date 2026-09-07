import Foundation
import FoundationModels

/// The MiniCPM5 tool-calling dialect (OpenBMB, `openbmb/MiniCPM5-1B` / `-2B`):
/// ChatML framing like Hermes, but calls are XML, not JSON —
///
/// * tools advertised in the system message inside `<tools>…</tools>`, one
///   `{"type":"function","function":{…}}` JSON line each, followed by the
///   template's "Tool usage guidelines" paragraph
/// * calls emitted as `<function name="fn"><param name="p">value</param>…</function>`
///   (a `param-value` that contains `<`, `&` or a newline is wrapped in
///   `<![CDATA[…]]>`); several calls are several consecutive `<function>` blocks
/// * past calls replayed in that same XML as an assistant turn; tool results
///   as user-role `<tool_response>\n…\n</tool_response>`
/// * reasoning (`<think>…</think>`) is emitted by the model unprompted;
///   `thinking: .off` renders the template's `enable_thinking=false` prefix
///   (`<think>\n\n</think>\n\n`) so the answer starts directly — the shape a
///   1024-token iOS context budget wants
///
/// Verbatim from the model's `chat_template.jinja` (the guidelines text
/// included — an in-context format instruction is only as good as its
/// agreement with the training prior). The vocab carries single tokens for
/// `<function`, `<param`, `</param>`, `</function>` and `<tool_response>`,
/// which is what `defaultDialect(probing:)` keys on.
public struct MiniCPMDialect: PromptDialect {
    public enum Thinking: Sendable {
        /// Leave it to the model (the template's default: no prefix).
        case model
        /// Render `<think>\n\n</think>\n\n` after the assistant header
        /// (`enable_thinking=false`): the model answers without a trace.
        case off
    }

    public let name = "minicpm"
    public let toolCallOpen = "<function"
    public let toolCallClose = "</function>"
    public let thinking: Thinking

    public init(thinking: Thinking = .model) {
        self.thinking = thinking
    }

    public func render(
        transcript: Transcript,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool = false
    ) -> String {
        let rendered = renderChatML(
            transcript: transcript,
            defaultSystem: "",
            head: { system in
                let content = self.systemContent(
                    base: system, tools: tools, requireToolCall: requireToolCall)
                return content.isEmpty ? "" : "<|im_start|>system\n" + content + "<|im_end|>\n"
            },
            toolEntry: { entry in
                switch entry {
                case .toolCalls(let calls):
                    let rendered = calls.map { call in
                        self.functionXML(name: call.toolName, argumentsJSON: call.arguments.jsonString)
                    }.joined(separator: "\n")
                    return ("assistant", rendered)
                case .toolOutput(let output):
                    return (
                        "user",
                        "<tool_response>\n\(self.segmentsText(output.segments))\n</tool_response>")
                default:
                    return nil  // reasoning entries are not replayed into history
                }
            })
        switch thinking {
        case .model: return rendered
        case .off: return rendered + "<think>\n\n</think>\n\n"
        }
    }

    /// The template's tool block: `# Tools` + `<tools>` JSON lines + the
    /// guidelines paragraph. No tools — no block (a plain-chat prompt is
    /// byte-identical to one rendered without tool support; with no system
    /// text either, no system turn at all — the template omits it).
    private func systemContent(
        base: String,
        tools: [Transcript.ToolDefinition],
        requireToolCall: Bool
    ) -> String {
        guard !tools.isEmpty else { return base }
        let lines = tools.map { tool in
            #"{"type": "function", "function": \#(toolDescriptorJSON(tool))}"#
        }
        let definitions = """
            # Tools

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            \(lines.joined(separator: "\n"))
            </tools>

            Tool usage guidelines:
            - You may call zero or more functions. If no function calls are needed, just answer normally and do not include any <function ... </function>.
            - When calling a function, return an XML object within <function ... </function> using:
            <function name="function-name"><param name="param-name">param-value</param></function>
            - param-value may be multi-line. If it contains <, & or newline characters, wrap it in a CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param>
            """
        let requirement = requireToolCall
            ? "\n- You MUST respond with a function call; do not answer directly."
            : ""
        return base.isEmpty ? definitions + requirement : base + "\n\n" + definitions + requirement
    }

    /// Renders one call in the template's XML form (`<param>` per argument;
    /// strings that would break the XML go into CDATA, non-strings render as
    /// their JSON literal — the template writes them raw).
    func functionXML(name: String, argumentsJSON: String) -> String {
        var out = "<function name=\"\(name)\">"
        if let data = argumentsJSON.data(using: .utf8),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        {
            for key in object.keys.sorted() {
                let value = object[key]!
                out += "<param name=\"\(key)\">"
                if let s = value as? String {
                    if s.contains("<") || s.contains("&") || s.contains("\n") {
                        out += "<![CDATA[\(s)]]>"
                    } else {
                        out += s
                    }
                } else if let d = try? JSONSerialization.data(
                    withJSONObject: value, options: [.fragmentsAllowed, .sortedKeys]),
                    let s = String(data: d, encoding: .utf8)
                {
                    out += s
                } else {
                    out += "\(value)"
                }
                out += "</param>"
            }
        }
        return out + "</function>"
    }

    /// Parses the body of ONE `<function …</function>` block: the stream
    /// parser strips the `<function` / `</function>` markers, so the payload
    /// is ` name="fn">` followed by the `<param>` elements. Values are typed
    /// against the tool's parameter schema (number / integer / boolean /
    /// array / object literals are passed through as JSON; everything else is
    /// a JSON string), so the framework's schema decode sees the types it
    /// declared.
    public func parseToolCalls(
        _ payload: String,
        tools: [Transcript.ToolDefinition]
    ) throws -> [ParsedToolCall] {
        let body = payload.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let nameMatch = body.firstMatch(of: /^\s*name\s*=\s*"([^"]+)"\s*>/) else {
            throw ZooFMProviderError.malformedToolCall(payload: body)
        }
        let name = String(nameMatch.1)
        let rest = String(body[nameMatch.range.upperBound...])
        let types = parameterTypes(tools.first { $0.name == name })
        var arguments: [String: Any] = [:]
        let paramPattern =
            /<param\s+name\s*=\s*"([^"]+)"\s*>(?:<!\[CDATA\[(.*?)\]\]>|(.*?))<\/param>/
                .dotMatchesNewlines()
        for match in rest.matches(of: paramPattern) {
            let key = String(match.1)
            let raw = match.2.map(String.init) ?? match.3.map(String.init) ?? ""
            arguments[key] = typedValue(raw, declared: types[key])
        }
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: arguments, options: [.fragmentsAllowed, .sortedKeys]),
            let json = String(data: data, encoding: .utf8)
        else {
            throw ZooFMProviderError.malformedToolCall(payload: body)
        }
        return [ParsedToolCall(name: name, argumentsJSON: json)]
    }

    /// `property name -> schema type` for a tool (nil types when unknown).
    private func parameterTypes(_ tool: Transcript.ToolDefinition?) -> [String: String] {
        guard let tool,
            let data = try? JSONEncoder().encode(tool.parameters),
            let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let properties = object["properties"] as? [String: Any]
        else { return [:] }
        var types: [String: String] = [:]
        for (key, value) in properties {
            if let dict = value as? [String: Any], let type = dict["type"] as? String {
                types[key] = type
            }
        }
        return types
    }

    /// Coerces a raw `<param>` text to the declared JSON type; falls back to
    /// the string when the literal does not parse (the framework then reports
    /// the schema mismatch, which is the honest failure).
    private func typedValue(_ raw: String, declared: String?) -> Any {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        switch declared {
        case "integer":
            return Int(text) ?? (Double(text).map { Int($0) } ?? text)
        case "number":
            return Double(text) ?? text
        case "boolean":
            switch text.lowercased() {
            case "true", "yes", "1": return true
            case "false", "no", "0": return false
            default: return text
            }
        case "array", "object":
            if let data = text.data(using: .utf8),
                let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
            {
                return parsed
            }
            return text
        default:
            return raw
        }
    }
}
