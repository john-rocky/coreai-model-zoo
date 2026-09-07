import CoreAILanguageModels
import Foundation
import FoundationModels
import Tokenizers

/// A zoo LanguageBundle behind FoundationModels' `LanguageModel` protocol,
/// adding the capabilities Apple's `CoreAILanguageModel` adapter does not
/// implement: tool calling, usage events, and append-only KV transcript reuse.
///
/// Built on the same public pieces as Apple's adapter — an `InferenceEngine`
/// from `CoreAIRunner.makeInferenceEngine()` and a `Tokenizer` from
/// `LanguageBundle.loadTokenizer()` — so any bundle that runs in the chat
/// apps runs here. Prompting goes through a per-model `PromptDialect`
/// (auto-picked by probing the tokenizer vocab: LFM tool special tokens →
/// `LFMDialect`, otherwise the Qwen/Hermes ChatML dialect); the bundle
/// tokenizer's Jinja template is not relied on.
///
/// ```swift
/// let model = try await ZooLanguageModel(resourcesAt: bundleDir)
/// let session = LanguageModelSession(model: model, tools: [WeatherTool()])
/// let answer = try await session.respond(to: "Weather in Tokyo?")
/// ```
///
/// Guided generation is deliberately NOT declared: zoo pipelined bundles
/// sample on-GPU (`supportsLogits == false`), so schema requests throw
/// `LanguageModelError.unsupportedCapability`.
///
/// One model instance assumes serial use (one `LanguageModelSession` at a
/// time) — the underlying engine traps on concurrent generate calls, same as
/// Apple's adapter.
public struct ZooLanguageModel: LanguageModel {
    public typealias Executor = ZooExecutor

    let engine: any InferenceEngine
    let tokenizer: any Tokenizer
    let modelID: String
    /// Chain-of-thought marker pair, probed from the tokenizer vocab at init
    /// (nil when the model has no reasoning markup).
    let thinkingMarkers: ThinkingMarkers?
    /// The model family's tool-calling dialect (rendering + stream markers +
    /// call parsing). Auto-probed from the tokenizer vocab unless overridden.
    let dialect: any PromptDialect

    public var capabilities: LanguageModelCapabilities {
        var capabilities: [LanguageModelCapabilities.Capability] = [.toolCalling]
        if thinkingMarkers != nil {
            capabilities.append(.reasoning)
        }
        return LanguageModelCapabilities(capabilities)
    }

    public var executorConfiguration: ZooExecutor.Configuration {
        ZooExecutor.Configuration(
            engine: engine,
            tokenizer: tokenizer,
            modelID: modelID,
            thinkingMarkers: thinkingMarkers,
            dialect: dialect
        )
    }

    /// Loads a LanguageBundle directory (`metadata.json` + `.aimodel` +
    /// `tokenizer/`) and auto-picks the engine, exactly like Apple's
    /// `CoreAILanguageModel(resourcesAt:)`. Sets `COREAI_CHUNK_THRESHOLD=1`
    /// (required by decode-only S=1 bundles, and harmless otherwise) before
    /// engine creation.
    ///
    /// - Parameter dialect: tool-calling dialect override; nil probes the
    ///   tokenizer vocab (`defaultDialect(probing:)`). The probe only knows
    ///   LFM vs Hermes, and falls back to Hermes for everything else — pass an
    ///   explicit `dialect:` for granite-4.0 / gemma4 bundles (Hermes-
    ///   incompatible framing) once their dialects exist.
    public init(resourcesAt url: URL, dialect: (any PromptDialect)? = nil) async throws {
        setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        let bundle = try LanguageBundle(at: url)
        let runner = CoreAIRunner(from: bundle)
        let engine = try await runner.makeInferenceEngine()
        let tokenizer = try await bundle.loadTokenizer()
        self.init(
            engine: engine,
            tokenizer: tokenizer,
            modelID: url.standardizedFileURL.path,
            dialect: dialect
        )
    }

    /// Wraps an already-created engine + tokenizer. `modelID` keys the
    /// session's executor cache — use a bundle-unique string (the bundle
    /// path); two models with the same ID share one executor and its engine
    /// state.
    public init(
        engine: any InferenceEngine, tokenizer: any Tokenizer, modelID: String,
        dialect: (any PromptDialect)? = nil
    ) {
        self.engine = engine
        self.tokenizer = tokenizer
        self.modelID = modelID
        self.thinkingMarkers = ThinkingMarkers(probing: tokenizer)
        self.dialect = dialect ?? defaultDialect(probing: tokenizer)
    }
}

/// Open/close chain-of-thought markers, verified to exist in the tokenizer
/// vocab (same probe as Apple's `CoreAIExecutor.detectThinkingMarkers`, but
/// nil when absent so `capabilities.reasoning` is only declared on models
/// that actually think).
struct ThinkingMarkers: Hashable, Sendable {
    let open: String
    let close: String

    init?(probing tokenizer: any Tokenizer) {
        let candidates: [(open: String, close: String)] = [
            ("<think>", "</think>"),
            ("<|reasoning_start|>", "<|reasoning_end|>"),
        ]
        for pair in candidates {
            if tokenizer.convertTokenToId(pair.open) != nil,
                tokenizer.convertTokenToId(pair.close) != nil
            {
                self.open = pair.open
                self.close = pair.close
                return
            }
        }
        return nil
    }
}

/// Failures specific to this provider. WWDC26 339: when the built-in
/// `LanguageModelError` cases don't cover a situation, define your own type.
public enum ZooFMProviderError: Error, LocalizedError {
    /// The model emitted a `<tool_call>` block whose body is not a JSON
    /// object with a `name` string.
    case malformedToolCall(payload: String)

    public var errorDescription: String? {
        switch self {
        case .malformedToolCall(let payload):
            return "Model emitted an unparseable tool call: \(payload.prefix(200))"
        }
    }
}
