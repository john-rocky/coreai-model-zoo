// PromptBuilder — (context, questions) -> the decoder's id row, the author's contract (conversion/decider_vision/host.py
// `build_ids`, from the checkpoint's `decider/prompt.py build()` and `decider/vision.py prepare()`):
//
//   text = encode("Context:\n" + context)[:1536]
//        + per question: encode("\n\nQuestion{ k}: {text}\nOptions:" + "\n({letter}) {option}"... + "\nAnswer{ k}: (")
//   text = encode(decode(text))                           prepare() hands the decoded string to the processor
//   ids  = [<|vision_start|>] + [V + k for k in 0..<H*W] + [<|vision_end|>] + text      (image rows)
//        = text                                                                          (text-only rows)
//
// " k" is the 1-based question number, written only when there are several questions. An id >= V (248320) is row
// id - V of the tower output. The slots are the author's rule: " (" (318) right after ":" (25) with "Answer" (15666)
// among the 5 tokens before it, one per question. rope_shift_start = 1 + H*W (the <|vision_end|> index), amount =
// H*W - max(H, W); a text-only row gets start 1 << 30, amount 0.

import Foundation
import Tokenizers

@available(macOS 27, iOS 27, *)
public struct DecisionQuestion: Sendable, Equatable, Codable {
    public var text: String
    public var options: [String]

    public init(text: String, options: [String]) {
        self.text = text
        self.options = options
    }
}

/// One built row: what the decoder reads and where the answers are.
@available(macOS 27, iOS 27, *)
public struct DecisionRow: Sendable, Equatable {
    /// Decoder ids; the image block is V + k.
    public let ids: [Int]
    /// Positions of the answer slots, ascending, one per question.
    public let slots: [Int]
    public let ropeShiftStart: Int32
    public let ropeShiftAmount: Int32
    /// Merged grid side of the image block (8 or 14), nil for a text-only row.
    public let grid: Int?
    /// Options per question.
    public let optionCounts: [Int]
}

@available(macOS 27, iOS 27, *)
public struct PromptBuilder: Sendable {
    public static let vocab = 248_320
    public static let visionStart = 248_053
    public static let imagePad = 248_056
    public static let visionEnd = 248_054
    public static let slotToken = 318
    public static let colon = 25
    public static let answer = 15_666
    public static let letters: [String] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    public static let maxOptions = 10
    public static let maxContextTokens = 1536
    public static let noShift: Int32 = 1 << 30

    public let tokenizer: any Tokenizer

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    public static func load(tokenizerFolder: URL) async throws -> PromptBuilder {
        PromptBuilder(tokenizer: try await AutoTokenizer.from(modelFolder: tokenizerFolder))
    }

    func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// The author's `build()` with no shuffling (before prepare()'s decode / re-encode).
    public func textIDs(context: String, questions: [DecisionQuestion]) throws -> [Int] {
        var ids = Array(encode("Context:\n" + context).prefix(Self.maxContextTokens))
        let multi = questions.count > 1
        for (k, q) in questions.enumerated() {
            guard q.options.count <= Self.maxOptions else {
                // The author samples down to 10 at training time keeping the gold option; a host has no gold.
                throw DeciderVisionError.prompt("question \(k): \(q.options.count) options (at most \(Self.maxOptions))")
            }
            guard !q.options.isEmpty else { throw DeciderVisionError.prompt("question \(k): no options") }
            let num = multi ? " \(k + 1)" : ""
            var block = "\n\nQuestion\(num): \(q.text)\nOptions:"
            for (j, o) in q.options.enumerated() { block += "\n(\(Self.letters[j])) \(o)" }
            block += "\nAnswer\(num): ("
            ids += encode(block)
        }
        return ids
    }

    /// The decoder row. `grid` = merged grid side of the tower that made the image rows (8 / 14), nil for text only.
    public func build(context: String, questions: [DecisionQuestion], grid: Int?) throws -> DecisionRow {
        guard !questions.isEmpty else { throw DeciderVisionError.prompt("no questions") }
        let built = try textIDs(context: context, questions: questions)
        let text = encode(tokenizer.decode(tokens: built, skipSpecialTokens: false))
        let ids: [Int]
        let start: Int32, amount: Int32
        if let g = grid {
            let n = g * g
            ids = [Self.visionStart] + (0..<n).map { Self.vocab + $0 } + [Self.visionEnd] + text
            start = Int32(1 + n)
            amount = Int32(n - g)
        } else {
            ids = text
            start = Self.noShift
            amount = 0
        }
        let slots = Self.findSlots(ids)
        guard slots.count == questions.count else {
            throw DeciderVisionError.prompt("\(slots.count) slots for \(questions.count) questions")
        }
        return DecisionRow(ids: ids, slots: slots, ropeShiftStart: start, ropeShiftAmount: amount, grid: grid,
                           optionCounts: questions.map(\.options.count))
    }

    /// The author's slot rule on the final row.
    public static func findSlots(_ ids: [Int]) -> [Int] {
        guard ids.count > 2 else { return [] }
        return (2..<ids.count).filter { i in
            ids[i] == slotToken && ids[i - 1] == colon && ids[max(0, i - 5)..<i].contains(answer)
        }
    }

    /// V + k back to <|image_pad|>: the processor's form of the same row.
    public static func processorIDs(_ ids: [Int]) -> [Int] { ids.map { $0 >= vocab ? imagePad : $0 } }
}
