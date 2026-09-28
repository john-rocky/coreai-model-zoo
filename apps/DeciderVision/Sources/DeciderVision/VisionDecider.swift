// VisionDecider — image + context + questions -> the probability of every option, per question:
//
//   let decider = try await VisionDecider(bundle: bundleDir, towers: [.g256: towerURL])
//   let probs = try await decider.decide(image: cgImage, context: "…",
//                                        questions: [DecisionQuestion(text: "How many circles?", options: ["2", "3"])],
//                                        grid: .g256)
//
// The bundle directory is the LanguageBundle (metadata.json, <name>.aimodel, tokenizer/); `decoder` overrides the
// asset (an AOT `.aimodelc` of the same graph). At each slot the letters A.. (ids 32..) over the question's options
// are read and turned into probabilities with a softmax at T = 1 in float64.

import CoreAI
import CoreGraphics
import Foundation

@available(macOS 27, iOS 27, *)
public final class VisionDecider: @unchecked Sendable {
    public enum Grid: Int, Sendable, CaseIterable, CustomStringConvertible {
        case g256 = 8
        case g448 = 14

        public var side: Int { rawValue }
        public var tile: Int { ImagePreprocess.tileSide(grid: rawValue) }
        public var description: String { "g\(tile)" }

        public init?(tile: Int) {
            guard let g = Grid.allCases.first(where: { $0.tile == tile }) else { return nil }
            self = g
        }
    }

    /// What the bundle's metadata.json says about the read-out.
    public struct Metadata: Sendable {
        public let name: String
        public let asset: String
        public let vocab: Int
        public let maxContext: Int
        public let prefillChunk: Int?
        public let letterIDs: [Int]
        public let temperature: Double

        public init(bundle: URL) throws {
            let url = bundle.appendingPathComponent("metadata.json")
            guard let data = try? Data(contentsOf: url),
                  let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let lang = j["language"] as? [String: Any], let assets = j["assets"] as? [String: Any],
                  let asset = assets["main"] as? String, let vocab = lang["vocab_size"] as? Int,
                  let ctx = lang["max_context_length"] as? Int
            else { throw DeciderVisionError.bundle("\(url.path): no language / assets block") }
            let decision = j["decision"] as? [String: Any] ?? [:]
            name = j["name"] as? String ?? bundle.lastPathComponent
            self.asset = asset
            self.vocab = vocab
            maxContext = ctx
            prefillChunk = lang["prefill_chunk"] as? Int
            letterIDs = decision["letter_ids"] as? [Int] ?? Array(32...41)
            temperature = decision["temperature"] as? Double ?? 1.0
        }
    }

    /// One question's answer at its slot.
    public struct Answer: Sendable {
        public let slot: Int
        public let letterLogits: [Float16]
        public let probabilities: [Double]
        public let argmax: Int
        /// Full-vocabulary argmax and its logit, and the five best ids (stable order).
        public let fullVocabTop1: Int
        public let fullVocabTop1Logit: Float16
        public let fullVocabTop5: [Int]
        public let finite: Bool
        public let readFrom: String
    }

    /// Everything one decision did, for gates and timing.
    public struct Trace: Sendable {
        public let row: DecisionRow
        public let answers: [Answer]
        public let prepared: ImagePreprocess.Prepared?
        public let towerEmbeds: [Float]?
        public let pass: DecisionDecoder.Pass
        /// Full-vocab fp16 logits per slot (kept only when asked).
        public let slotLogits: [[Float16]]?
        public let seconds: [String: Double]
    }

    public let metadata: Metadata
    public let prompt: PromptBuilder
    public let decoder: DecisionDecoder
    public private(set) var towers: [Grid: VisionTower]
    public let tokenizerLoadSeconds: Double

    public init(bundle: URL, decoder decoderURL: URL? = nil, towers towerURLs: [Grid: URL],
                decoderOptions: SpecializationOptions = .default, towerOptions: SpecializationOptions = .default) async throws
    {
        let meta = try Metadata(bundle: bundle)
        let t0 = ContinuousClock.now
        let builder = try await PromptBuilder.load(tokenizerFolder: bundle.appendingPathComponent("tokenizer"))
        tokenizerLoadSeconds = secondsSince(t0)
        for (i, id) in meta.letterIDs.enumerated() {
            let got = builder.tokenizer.encode(text: PromptBuilder.letters[i], addSpecialTokens: false)
            guard got == [id] else {
                throw DeciderVisionError.bundle("letter \(PromptBuilder.letters[i]) encodes to \(got), metadata says \(id)")
            }
        }
        var loaded: [Grid: VisionTower] = [:]
        for (g, url) in towerURLs.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            loaded[g] = try await VisionTower(contentsOf: url, grid: g.side, options: towerOptions)
        }
        let asset = decoderURL ?? bundle.appendingPathComponent(meta.asset)
        decoder = try await DecisionDecoder(contentsOf: asset, vocab: meta.vocab, maxContext: meta.maxContext,
                                            prefillChunk: meta.prefillChunk, options: decoderOptions)
        metadata = meta
        prompt = builder
        towers = loaded
    }

    /// Per question, the probability of each option (in the order given).
    public func decide(image: CGImage?, context: String, questions: [DecisionQuestion], grid: Grid = .g256)
        async throws -> [[Double]]
    {
        try await trace(image: image, context: context, questions: questions, grid: grid).answers.map(\.probabilities)
    }

    /// The whole decision with its intermediate values and times. `image == nil` = a text-only row.
    public func trace(image: CGImage?, context: String, questions: [DecisionQuestion], grid: Grid = .g256,
                      keepLogits: Bool = false, useChunks: Bool = true) async throws -> Trace
    {
        let t0 = ContinuousClock.now
        var seconds: [String: Double] = [:]
        var prepared: ImagePreprocess.Prepared? = nil
        var emb: [Float]? = nil
        if let image {
            guard let tower = towers[grid] else { throw DeciderVisionError.bundle("no \(grid) tower loaded") }
            let p = try ImagePreprocess.prepare(image, grid: grid.side)
            seconds["decode_rgb"] = p.decodeSeconds
            seconds["resize"] = p.resizeSeconds
            seconds["patches"] = p.patchSeconds
            let t = ContinuousClock.now
            emb = try await tower.encode(patches: p.patches)
            seconds["tower"] = secondsSince(t)
            prepared = p
        }
        let t1 = ContinuousClock.now
        let row = try prompt.build(context: context, questions: questions, grid: image == nil ? nil : grid.side)
        seconds["tokenize"] = secondsSince(t1)
        let t2 = ContinuousClock.now
        let inputs = try decoder.staticInputs(towerEmbeds: emb, grid: row.grid, start: row.ropeShiftStart,
                                              amount: row.ropeShiftAmount)
        seconds["static_inputs"] = secondsSince(t2)
        let pass = try await decoder.run(ids: row.ids, slots: row.slots, inputs: inputs, useChunks: useChunks)
        seconds["decoder"] = pass.seconds
        let t3 = ContinuousClock.now
        var answers: [Answer] = []
        for (k, slot) in pass.slots.enumerated() {
            answers.append(Self.answer(slot, options: row.optionCounts[k], letterIDs: metadata.letterIDs,
                                       temperature: metadata.temperature))
        }
        seconds["readout"] = secondsSince(t3)
        seconds["wall"] = secondsSince(t0)
        return Trace(row: row, answers: answers, prepared: prepared, towerEmbeds: emb, pass: pass,
                     slotLogits: keepLogits ? pass.slots.map(\.logits) : nil, seconds: seconds)
    }

    static func answer(_ slot: DecisionDecoder.Slot, options n: Int, letterIDs: [Int], temperature: Double) -> Answer {
        let lg = slot.logits
        let letters = letterIDs.prefix(n).map { lg[$0] }
        let p = softmax(letters.map { Double($0) / temperature })
        // numpy.argsort(-x, kind="stable")[:5] over float32: the largest first, ties by the lower id
        var top: [(Float, Int)] = []
        for (i, v) in lg.enumerated() {
            let f = Float(v)
            if top.count < 5 || f > top[top.count - 1].0 {
                var j = top.count
                while j > 0 && f > top[j - 1].0 { j -= 1 }
                top.insert((f, i), at: j)
                if top.count > 5 { top.removeLast() }
            }
        }
        let full = argmax(lg)
        return Answer(slot: slot.position, letterLogits: Array(letters), probabilities: p, argmax: argmax(p),
                      fullVocabTop1: full, fullVocabTop1Logit: lg[full], fullVocabTop5: top.map(\.1),
                      finite: lg.allSatisfy { $0.isFinite }, readFrom: slot.readFrom)
    }
}
