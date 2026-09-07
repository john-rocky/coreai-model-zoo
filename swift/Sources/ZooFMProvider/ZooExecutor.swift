import CoreAILanguageModels
import Foundation
import FoundationModels
import Synchronization
import Tokenizers

/// Executor for `ZooLanguageModel`: streaming generation with incremental
/// tool-call/thinking marker detection (markers come from the model's
/// `PromptDialect`), usage events, and an append-only KV fast path.
///
/// ## Events
/// Text streams out as `.response` events the moment each token decodes
/// cleanly; reasoning streams as `.reasoning`; tool calls are emitted as one
/// `.toolCalls` event per call (consecutive events coalesce into a single
/// transcript entry) once their JSON body is complete. Metadata + usage are
/// sent once per turn at the end, attached to the kind of entry the turn
/// produced — a usage event of a kind the turn never emits content for would
/// materialize an empty transcript entry (verified on the 27.0 beta).
///
/// ## KV reuse and the over-generation pump
/// The engine preserves KV between `generate()` calls and continues from its
/// current position, so when the new transcript's rendered tokens extend the
/// tokens already in the cache, only the suffix is prefilled
/// (`Usage.Input.cachedTokenCount` reports the reuse).
///
/// One engine reality shapes this design: breaking out of the token stream
/// (at EOS) does NOT stop the pipelined engine — it keeps generating to
/// `maxTokens` in the background, and those post-EOS tokens land in the KV
/// cache. `respond` therefore pumps the engine stream through a relay task
/// that keeps draining (and recording) tokens after the consumer stops, and
/// the next `respond` settles that bookkeeping before deciding whether the
/// cache still matches the transcript. Post-EOS tokens usually diverge from
/// the next rendered prompt, in which case the executor falls back to
/// reset + full re-prefill — correctness first.
public struct ZooExecutor: LanguageModelExecutor {
    public typealias Model = ZooLanguageModel

    public struct Configuration: Hashable, Sendable {
        let engine: any InferenceEngine
        let tokenizer: any Tokenizer
        let modelID: String
        let thinkingMarkers: ThinkingMarkers?
        let dialect: any PromptDialect

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.modelID == rhs.modelID
        }
        public func hash(into hasher: inout Hasher) {
            hasher.combine(modelID)
        }
    }

    private let engine: any InferenceEngine
    private let tokenizer: any Tokenizer
    private let modelID: String
    private let thinkingMarkers: ThinkingMarkers?
    private let dialect: any PromptDialect
    private let state = TurnState()

    public init(configuration: Configuration) throws {
        self.engine = configuration.engine
        self.tokenizer = configuration.tokenizer
        self.modelID = configuration.modelID
        self.thinkingMarkers = configuration.thinkingMarkers
        self.dialect = configuration.dialect
        zooFMDebug("executor for \(configuration.modelID): dialect=\(configuration.dialect.name)")
    }

    // MARK: - Prewarm

    /// NOTE: the signature must be exactly `prewarm(model:transcript:)` — the
    /// protocol ships a default no-op, so a near-miss compiles and is silently
    /// never called.
    ///
    /// Warm = one real 1-token generate + reset (compiles the sampler graph
    /// and touches the weights). `engine.warmup()` is deliberately not used:
    /// its default query length builds an S=256 step that decode-only S=1
    /// bundles reject.
    ///
    /// The warm work registers itself as the in-flight turn, so a respond
    /// that races prewarm awaits it in `settle()` instead of contending for
    /// the engine. It returns `[]`: the trailing reset leaves nothing in the
    /// cache, which is exactly what settle computes from an empty turn.
    public func prewarm(model: ZooLanguageModel, transcript: Transcript) {
        let engine = self.engine
        let tokenizer = self.tokenizer
        let state = self.state
        let semaphore = DispatchSemaphore(value: 0)
        // Block via a GCD thread — semaphore.wait() on a cooperative thread
        // while Task{} needs one would starve (same pattern as Apple's
        // adapter).
        DispatchQueue(label: "zoo.fm.prewarm").async {
            // Snapshot the in-flight pump BEFORE creating the warm task — the
            // warm task must not settle() itself once registered (self-await).
            let previous = state.takePump()
            let warm = Task<[Int32], any Error> {
                defer { semaphore.signal() }
                if let previous {
                    _ = try? await previous.value  // engine free before generate
                }
                let seed = tokenizer.encode(text: "Hi").first.map(Int32.init) ?? 1
                let stream = try await engine.generate(
                    with: [seed],
                    samplingConfiguration: .greedy,
                    inferenceOptions: InferenceOptions(maxTokens: 1))
                for try await _ in stream {}
                try await engine.reset()
                return []
            }
            state.beginTurn(base: [], fed: [], pump: warm)
        }
        semaphore.wait()
    }

    // MARK: - respond

    public nonisolated(nonsending) func respond(
        to request: LanguageModelExecutorGenerationRequest,
        model: ZooLanguageModel,
        streamingInto channel: LanguageModelExecutorGenerationChannel
    ) async throws {
        // Pipelined zoo bundles sample on-GPU — no logits, no constrained
        // decoding. Approximate-or-throw rule: there is no honest
        // approximation of a schema, so throw.
        if request.schema != nil {
            throw LanguageModelError.unsupportedCapability(
                .init(
                    capability: .guidedGeneration,
                    debugDescription:
                        "GPU-pipelined zoo bundles sample on-device and expose no logits; "
                        + "guided generation needs a sequential engine."))
        }

        // 1) Settle the previous turn: await its background drain and learn
        //    exactly which tokens are in the KV cache (nil = unknown → reset).
        let kvTokens = await state.settle()

        // 2) Render the transcript. toolCallingMode (WWDC26 242): disallowed
        //    drops the tools block entirely; required adds a must-call
        //    instruction (prompt-level approximation — local models have no
        //    grammar enforcement).
        let mode = request.generationOptions.toolCallingMode?.kind ?? .allowed
        let tools = mode == .disallowed ? [] : request.enabledToolDefinitions
        let promptText = dialect.render(
            transcript: request.transcript,
            tools: tools,
            requireToolCall: mode == .required)
        let promptTokens = tokenizer.encode(text: promptText).map(Int32.init)

        // 3) Append-only KV fast path: skip reset and feed only the suffix
        //    when the rendered prompt extends what's already in the cache.
        let fed: [Int32]
        let kvBase: [Int32]
        if let kv = kvTokens, kv.isEmpty {
            fed = promptTokens
            kvBase = []
        } else if let kv = kvTokens, promptTokens.count > kv.count,
            promptTokens.starts(with: kv)
        {
            fed = Array(promptTokens[kv.count...])
            kvBase = kv
            zooFMDebug("KV fast path: reusing \(kv.count) tokens, prefilling \(fed.count)")
        } else {
            if let kv = kvTokens {
                zooFMDebug(
                    "KV diverged (cache \(kv.count) tokens vs prompt \(promptTokens.count)) — reset")
            }
            try await engine.reset()
            fed = promptTokens
            kvBase = []
        }
        let cachedCount = kvBase.count

        // (WWDC 339 suggests metadata + usage upfront, but a usage-only
        // .response event materializes an EMPTY Response transcript entry
        // when the turn ends in tool calls — verified against the 27.0 beta
        // framework. Until kind-agnostic usage exists, both are sent once at
        // end of turn, attached to the entry kind the turn produced.)

        // 4) Generate through a relay: the pump owns the engine stream and
        //    keeps draining it (recording every token) after this function
        //    stops consuming at EOS — see the type comment.
        let maxTokens = request.generationOptions.maximumResponseTokens ?? 512
        let sampling = request.generationOptions.temperature.map {
            SamplingConfiguration(temperature: $0)
        } ?? .greedy

        let (relay, relayContinuation) = AsyncThrowingStream<Int32, any Error>.makeStream()
        let engine = self.engine
        let pump = Task {
            var ids: [Int32] = []
            do {
                let stream = try await engine.generate(
                    with: fed,
                    samplingConfiguration: sampling,
                    inferenceOptions: InferenceOptions(maxTokens: maxTokens))
                for try await output in stream {
                    ids.append(output.tokenId)
                    relayContinuation.yield(output.tokenId)
                }
                relayContinuation.finish()
            } catch {
                relayContinuation.finish(throwing: error)
                throw error
            }
            return ids
        }
        state.beginTurn(base: kvBase, fed: fed, pump: pump)

        // 5) Stream: incremental UTF-8-safe detok (U+FFFD hold + one retained
        //    context token, same contract as Apple's adapter) feeding the tag
        //    parser; text goes out the moment it decodes.
        var parser = StreamTagParser(
            thinkOpen: thinkingMarkers?.open ?? "<think>",
            thinkClose: thinkingMarkers?.close ?? "</think>",
            toolOpen: dialect.toolCallOpen,
            toolClose: dialect.toolCallClose)
        var pendingTokens: [Int] = []
        var previousDecodedText = ""
        var generatedCount = 0
        var reasoningEventCount = 0
        var sentResponseText = false
        var toolPayloads: [String] = []
        // Whitespace-only text held back pending the next event (see dispatch).
        var pendingWhitespace = ""
        let eosTokenId = tokenizer.eosTokenId

        func dispatch(_ events: [StreamTagParser.Event]) async {
            for event in events {
                switch event {
                case .text(let text):
                    // Hold whitespace-only text: if a tool-call block follows, it
                    // is inter-block padding (e.g. the `\n\n` between `</think>`
                    // and the tool-call marker) that would otherwise materialize a
                    // whitespace-only Response transcript entry. Real text flushes
                    // any held whitespace ahead of it; held whitespace that is
                    // never followed by real text (trailing, or right before a tool
                    // call) is dropped.
                    if text.allSatisfy(\.isWhitespace) {
                        pendingWhitespace += text
                    } else {
                        let combined = pendingWhitespace + text
                        pendingWhitespace = ""
                        sentResponseText = true
                        await channel.send(
                            .response(action: .appendText(combined, tokenCount: 1)))
                    }
                case .reasoning(let text):
                    reasoningEventCount += 1
                    await channel.send(.reasoning(action: .appendText(text, tokenCount: 1)))
                case .toolCallPayload(let payload):
                    pendingWhitespace = ""  // drop inter-block whitespace before a call
                    toolPayloads.append(payload)
                }
            }
        }

        do {
            for try await tokenId in relay {
                if let eos = eosTokenId, Int(tokenId) == eos { break }
                // Count AFTER the EOS check so the never-emitted EOS sentinel is
                // not folded into Usage.Output (matches Apple's reference adapter).
                generatedCount += 1

                pendingTokens.append(Int(tokenId))
                let decodedText = tokenizer.decode(tokens: pendingTokens)
                let common = decodedText.commonPrefix(with: previousDecodedText)
                let delta = String(decodedText.dropFirst(common.count))
                // Check U+FFFD on the full decode, not the delta: consecutive
                // partial tokens can decode to identical replacement strings,
                // making the delta empty and hiding the incomplete state.
                if decodedText.unicodeScalars.contains(where: { $0 == "\u{FFFD}" }) {
                    previousDecodedText = decodedText
                    continue
                }
                await dispatch(parser.consume(delta))
                // Retain one token of context: SentencePiece-style tokenizers
                // need a predecessor to infer leading spaces, and one token
                // bounds the re-decode to O(1) per step.
                if let last = pendingTokens.last {
                    pendingTokens = [last]
                    previousDecodedText = tokenizer.decode(tokens: [last])
                }
            }
        } catch InferenceRuntimeError.contextLengthExceeded(let position, let maxContext) {
            throw LanguageModelError.contextSizeExceeded(
                .init(
                    contextSize: maxContext,
                    tokenCount: position,
                    debugDescription: "Transcript no longer fits the model context."))
        }
        // Flush content the parser held back for a possible marker match —
        // without this, text at the EOS boundary is lost.
        await dispatch(parser.flush())

        // 6) Tool calls: each payload parses through the model's dialect
        //    (one block may contain several calls — LFM emits
        //    `[fn1(…), fn2(…)]`); every call gets a minted id, and
        //    consecutive .toolCalls events form one transcript entry, so a
        //    multi-call turn lands as a single ToolCalls entry.
        if !toolPayloads.isEmpty {
            let calls = try toolPayloads.flatMap {
                try dialect.parseToolCalls($0, tools: tools)
            }
            for call in calls {
                await channel.send(
                    .toolCalls(
                        action: .toolCall(
                            id: UUID().uuidString,
                            name: call.name,
                            action: .appendArguments(
                                call.argumentsJSON,
                                tokenCount: max(1, call.argumentsJSON.count / 4)))))
            }
        }

        // 7) Metadata + usage once per turn, attached to the kind of entry
        //    this turn actually produced (see the note above step 4).
        // Beta 5: metadata values are `ConvertibleToGeneratedContent` (String conforms).
        let metadata: [String: any ConvertibleToGeneratedContent] = [
            "modelID": modelID,
            "requestID": request.id.uuidString,
        ]
        let usageInput = LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: promptTokens.count, cachedTokenCount: cachedCount)
        let usageOutput = LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: generatedCount, reasoningTokenCount: reasoningEventCount)
        if !toolPayloads.isEmpty {
            await channel.send(.toolCalls(action: .updateMetadata(metadata)))
            await channel.send(
                .toolCalls(action: .updateUsage(input: usageInput, output: usageOutput)))
        } else if sentResponseText || reasoningEventCount == 0 {
            await channel.send(.response(action: .updateMetadata(metadata)))
            await channel.send(
                .response(action: .updateUsage(input: usageInput, output: usageOutput)))
        } else {
            await channel.send(.reasoning(action: .updateMetadata(metadata)))
            await channel.send(
                .reasoning(action: .updateUsage(input: usageInput, output: usageOutput)))
        }

        await Task.yield()
    }
}

// MARK: - Turn state

/// Mutex-protected bookkeeping shared by all copies of one executor: which
/// token ids the engine's KV cache holds, and the pump task still draining
/// the previous turn's stream.
final class TurnState: Sendable {
    private struct Turn: Sendable {
        /// Tokens known to be consumed into the KV cache. nil = unknown —
        /// the initial state (a fresh executor can't know what an earlier
        /// session left in the engine) and the state after an engine error;
        /// either way the next respond resets first.
        var kvTokens: [Int32]? = nil
        var pendingBase: [Int32] = []
        var pendingFed: [Int32] = []
        var pump: Task<[Int32], any Error>?
    }

    private let turn = Mutex<Turn>(Turn())

    func beginTurn(base: [Int32], fed: [Int32], pump: Task<[Int32], any Error>) {
        turn.withLock {
            $0.pendingBase = base
            $0.pendingFed = fed
            $0.pump = pump
        }
    }

    /// Detaches the in-flight pump without folding its result (prewarm's
    /// reset discards the cache anyway).
    func takePump() -> Task<[Int32], any Error>? {
        turn.withLock {
            let pump = $0.pump
            $0.pump = nil
            return pump
        }
    }

    /// Awaits the previous turn's pump and folds its result into `kvTokens`.
    /// The last yielded token was sampled but never consumed by a subsequent
    /// step, so it is NOT in the cache — hence `dropLast()`.
    func settle() async -> [Int32]? {
        let (pump, base, fed) = turn.withLock {
            let values = ($0.pump, $0.pendingBase, $0.pendingFed)
            $0.pump = nil
            return values
        }
        guard let pump else {
            return turn.withLock { $0.kvTokens }
        }
        do {
            let ids = try await pump.value
            let kv = base + fed + ids.dropLast()
            turn.withLock { $0.kvTokens = kv }
            zooFMDebug("settled turn: \(ids.count) tokens streamed, cache now \(kv.count) tokens")
            return kv
        } catch {
            turn.withLock { $0.kvTokens = nil }
            zooFMDebug("settle: previous turn failed (\(error)) — cache unknown")
            return nil
        }
    }

}

// MARK: - Debug logging

let zooFMDebugEnabled = ProcessInfo.processInfo.environment["ZOO_FM_DEBUG"] != nil

func zooFMDebug(_ message: @autoclosure () -> String) {
    guard zooFMDebugEnabled else { return }
    FileHandle.standardError.write(Data("[zoo-fm] \(message())\n".utf8))
}
