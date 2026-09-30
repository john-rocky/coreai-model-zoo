// JuliaDecisions.swift — Julia-1 on Core AI from Swift: the reference host (text in, the publisher's
// typed answers out). The Swift port of `_julia_host.py`, which is the publisher's own julia/data.py
// `sequence()` and julia/typed.py `predict_typed`, step for step:
//
//   row      [CLS 2] enc("{type} question: {instructions}") [SEP 1] ([MASK 4] enc(" " + option)[..<48])...
//            [SEP 1] enc(state) [SEP 1]; every piece encoded alone without special tokens (an empty piece is
//            no tokens); strict: a row that needs any cut, or text holding the literal "<mask>", is refused
//   options  the criteria text as given — choice: the descriptions in order (the answer is the caller's id),
//            score: the rubric in order, noul: [false, true] descriptions, or the literal "false" / "true"
//   answer   softmax of the raw marker logits at T = 1 (the checkpoint has no calibration); choice = the
//            winning id, score = sum(i * p_i) (not rounded), noul = p[true]
//
// The options are where Julia differs from laya (coreai-kit's EncoderPrompt renders `label: description`,
// `level i: …` and `false: …` / `true: …`): a laya host gives Julia other rows and other answers.
//
// Dependencies: the system CoreAI framework (macOS 27 / iOS 27) and a tokenizer for the bundle's
// tokenizer/tokenizer.json, passed in as `encode` — with swift-transformers:
//
//   let tokenizer = try await AutoTokenizer.from(modelFolder: folder.appendingPathComponent("tokenizer"))
//   let julia = try await JuliaDecisions(folder: folder) { text in
//       tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
//   }
//   let answers = try await julia.predict(state: "I was charged twice for the same order.", questions: [
//       ("team", .choice("Which team should handle this request?", [
//           ("billing", "Billing and payment disputes"), ("shipping", "Shipping and delivery"),
//           ("access", "Account access and login")])),
//   ])
//   print(answers["team"]!.choice!)   // "billing"
//
// A structured state is text here: serialize it the way Python's json.dumps(state, ensure_ascii=False)
// writes it (", " and ": " separators, keys in insertion order) — the publisher's rows are built from that.

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public final class JuliaDecisions: @unchecked Sendable {
    public enum Question: Sendable {
        /// Instructions and the (id, description) pairs, in order.
        case choice(String, [(String, String)])
        /// Instructions and the rubric, in order; the answer is the expected zero-based index.
        case score(String, [String])
        /// Instructions and the false / true descriptions; nil = the literal words "false" / "true".
        case noul(String, falseText: String?, trueText: String?)
    }

    public struct Answer: Sendable {
        public let type: String
        /// Answer keys in option order: the choice ids, "0"…"n-1" for a score, "false" / "true" for a noul.
        public let keys: [String]
        /// Full softmax probabilities at T = 1, in `keys` order (no display rounding).
        public let probabilities: [Double]
        public let choice: String?
        public let score: Double?
        public let noul: Double?
        public let maxProbability: Double?
    }

    public enum JuliaError: Error {
        case notJulia(String), badQuestion(String), reservedMarker, optionTooLong, headTooLong, stateTooLong
    }

    public static let cls: Int32 = 2, sep: Int32 = 1, pad: Int32 = 0, mask: Int32 = 4
    public static let maskText = "<mask>", optionTokens = 48

    public let window: Int
    public let headLength: Int
    private let encodePiece: @Sendable (String) -> [Int32]
    private let function: InferenceFunction
    private let descriptor: InferenceFunctionDescriptor

    /// Loads one variant folder (the .aimodel, tokenizer/, metadata.json) on `options`
    /// (the GPU preference is what ships; `.cpuOnly` is the parity option).
    public init(
        folder: URL, options: SpecializationOptions = SpecializationOptions(preferredComputeUnitKind: .gpu),
        encode: @escaping @Sendable (String) -> [Int32]
    ) async throws {
        let metadata = try JSONSerialization.jsonObject(
            with: Data(contentsOf: folder.appendingPathComponent("metadata.json"))) as? [String: Any] ?? [:]
        guard let decision = metadata["decision"] as? [String: Any], decision["layout"] as? String == "julia",
            let window = decision["window"] as? Int, let headLength = decision["head_max_len"] as? Int,
            let asset = (metadata["assets"] as? [String: String])?["main"]
        else { throw JuliaError.notJulia(folder.path) }
        let model = try await AIModel(contentsOf: folder.appendingPathComponent(asset), options: options)
        guard let descriptor = model.functionDescriptor(for: "main"), let function = try model.loadFunction(named: "main")
        else { throw JuliaError.notJulia(folder.path) }
        self.window = window
        self.headLength = headLength
        self.encodePiece = encode
        self.function = function
        self.descriptor = descriptor
    }

    // MARK: - The publisher's sequence()

    private func encode(_ text: String) -> [Int32] { text.isEmpty ? [] : encodePiece(text) }

    private static func clean(_ text: String) -> String {
        text.replacingOccurrences(of: maskText, with: " ", options: .literal)
    }

    /// The row of one question (strict): token ids and the marker position of each option.
    public func row(state: String, instructions: String, type: String, options: [String]) throws -> (ids: [Int32], markers: [Int]) {
        guard (2...20).contains(options.count), options.allSatisfy({ !$0.isEmpty }), type != "noul" || options.count == 2
        else { throw JuliaError.badQuestion("2–20 nonempty options; a noul has exactly [false, true]") }
        if ([state, instructions] + options).contains(where: { $0.range(of: Self.maskText, options: .literal) != nil }) {
            throw JuliaError.reservedMarker
        }
        let head = encode("\(type) question: " + Self.clean(instructions))
        let optionIDs = options.map { encode(" " + Self.clean($0)) }
        guard optionIDs.allSatisfy({ $0.count <= Self.optionTokens }) else { throw JuliaError.optionTooLong }
        let budget = headLength - optionIDs.reduce(0) { $0 + $1.count + 1 }
        // Strict: the publisher's squeeze (budget < 16) always cuts an option, so it is a refusal here.
        guard budget >= 16, head.count <= budget else { throw JuliaError.headTooLong }
        var ids: [Int32] = [Self.cls] + head + [Self.sep]
        var markers: [Int] = []
        for option in optionIDs {
            markers.append(ids.count)
            ids += [Self.mask] + option
        }
        ids.append(Self.sep)
        let stateIDs = encode(Self.clean(state))
        guard stateIDs.count <= window - ids.count - 1 else { throw JuliaError.stateTooLong }
        return (ids + stateIDs + [Self.sep], markers)
    }

    // MARK: - The graph

    /// The raw option logits of one question, in option order (the publisher's engine.logits).
    public func logits(state: String, instructions: String, type: String, options: [String]) async throws -> [Float] {
        let (ids, markers) = try row(state: state, instructions: instructions, type: type, options: options)
        var inputIDs = [Int32](repeating: Self.pad, count: window)
        var attention = [Int32](repeating: 0, count: window)
        inputIDs.replaceSubrange(0..<ids.count, with: ids)
        attention.replaceSubrange(0..<ids.count, with: [Int32](repeating: 1, count: ids.count))
        var onehot = [Float](repeating: 0, count: 3)
        onehot[["choice", "score", "noul"].firstIndex(of: type)!] = 1
        var inputs: [String: NDArray] = [:]
        inputs["input_ids"] = try array("input_ids", inputIDs)
        inputs["attention_mask"] = try array("attention_mask", attention)
        inputs["qtype_onehot"] = try array("qtype_onehot", onehot)
        var outputs = try await function.run(inputs: inputs)
        guard let tokenLogits = outputs.remove("token_logits")?.ndArray else { throw JuliaError.notJulia("no token_logits") }
        let all = tokenLogits.view(as: Float.self).withUnsafePointer { pointer, _, _ in
            Array(UnsafeBufferPointer(start: pointer, count: window))
        }
        return markers.map { all[$0] }
    }

    /// A fresh input array of the graph's declared descriptor (static shape), filled with `values`.
    private func array<T: BitwiseCopyable>(_ name: String, _ values: [T]) throws -> NDArray {
        guard case .ndArray(let declared) = descriptor.inputDescriptor(of: name) else { throw JuliaError.notJulia(name) }
        var array = NDArray(descriptor: declared)
        var view = array.mutableView(as: T.self)
        view.copyElements(fromContentsOf: values)
        return array
    }

    // MARK: - The publisher's predict_typed()

    /// One answer per named question, each question its own `main` call, as the publisher's
    /// `predict(state=..., questions=...)` answers them.
    public func predict(state: String, questions: [(String, Question)]) async throws -> [String: Answer] {
        var answers: [String: Answer] = [:]
        for (id, question) in questions {
            let (type, instructions, keys, options): (String, String, [String], [String])
            switch question {
            case .choice(let text, let pairs):
                (type, instructions, keys, options) = ("choice", text, pairs.map(\.0), pairs.map(\.1))
            case .score(let text, let rubric):
                (type, instructions, keys, options) = ("score", text, rubric.indices.map(String.init), rubric)
            case .noul(let text, let no, let yes):
                guard (no == nil) == (yes == nil) else { throw JuliaError.badQuestion("noul criteria need both sides") }
                (type, instructions, keys, options) = ("noul", text, ["false", "true"], [no ?? "false", yes ?? "true"])
            }
            let z = try await logits(state: state, instructions: instructions, type: type, options: options).map(Double.init)
            let top = z.max()!
            let e = z.map { exp($0 - top) }
            let total = e.reduce(0, +)
            let p = e.map { $0 / total }
            let best = p.indices.max { p[$0] < p[$1] }!
            answers[id] = Answer(
                type: type, keys: keys, probabilities: p,
                choice: type == "choice" ? keys[best] : nil,
                score: type == "score" ? p.enumerated().reduce(0) { $0 + Double($1.offset) * $1.element } : nil,
                noul: type == "noul" ? p[1] : nil,
                maxProbability: type == "noul" ? nil : p.max())
        }
        return answers
    }
}
