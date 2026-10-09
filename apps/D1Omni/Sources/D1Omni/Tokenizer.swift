// Tokenizer — the publisher's `enc` (prompt.py: `tok(escape(s), add_special_tokens=False)["input_ids"]`) on
// swift-transformers' tokenizer of the bundle's tokenizer/ (the snapshot's tokenizer.json + tokenizer_config.json):
//
//   escape(s)  every `<|name|>` (name in [A-Za-z0-9_]+) rewritten as `<¦name¦>`, on code points (Python's re.sub), so
//              caller text never produces a delimiter or the marker
//   enc(s)     the ids of escape(s) without the post-processor's `<|startoftext|>`; "" -> no ids
//
// tokenizer.json is a byte-level BPE (no normalizer; a Split pre-tokenizer with a regex, then ByteLevel) with 509 added
// tokens, all special. 504 are `<|...|>` names, which escape() rewrites; the other five — `<image>`, `Mathias`,
// `python`, `<think>`, `</think>` — are matched in caller text as single tokens even after escape(), as the
// publisher's tokenizer does (round 8 measured both sides on the same strings).
//
// At load the nine ids the row builder uses are checked by token text against the bundle's metadata.json
// (decision.token_ids), and each token text must encode to its id alone.

import Foundation
import Tokenizers

/// The nine ids of host.py's TOKEN_IDS, read from the tokenizer by token text.
public struct D1TokenIDs: Sendable, Equatable {
    public let pad: Int
    public let bos: Int
    public let imEnd: Int
    public let mask: Int
    public let state: Int
    public let q: Int
    public let opt: Int
    public let optEnd: Int
    public let decide: Int

    /// (token text, the row builder's name); prompt.py DELIM / MARKER and the bos / pad tokens.
    public static let tokens: [(token: String, name: String)] = [
        ("<|pad|>", "pad"), ("<|startoftext|>", "bos"), ("<|im_end|>", "im_end"), ("<|mask|>", "marker"),
        ("<|reserved_7|>", "state"), ("<|reserved_8|>", "q"), ("<|reserved_9|>", "opt"), ("<|reserved_10|>", "opt_end"),
        ("<|reserved_11|>", "decide"),
    ]

    public var byToken: [String: Int] {
        ["<|pad|>": pad, "<|startoftext|>": bos, "<|im_end|>": imEnd, "<|mask|>": mask, "<|reserved_7|>": state,
         "<|reserved_8|>": q, "<|reserved_9|>": opt, "<|reserved_10|>": optEnd, "<|reserved_11|>": decide]
    }
}

public struct D1Tokenizer: Sendable {
    public let tokenizer: any Tokenizer
    public let ids: D1TokenIDs
    /// reading tokenizer.json and building the tokenizer
    public let loadSeconds: Double

    /// The tokenizer of `folder` (tokenizer.json + tokenizer_config.json); its nine ids must equal `expected`
    /// (metadata.json's decision.token_ids, by token text).
    public static func load(folder: URL, expected: [String: Int]) async throws -> D1Tokenizer {
        let t0 = ContinuousClock.now
        let tok = try await AutoTokenizer.from(modelFolder: folder)
        return try D1Tokenizer(tokenizer: tok, expected: expected, loadSeconds: secondsSince(t0))
    }

    public init(tokenizer: any Tokenizer, expected: [String: Int], loadSeconds: Double = 0) throws {
        self.tokenizer = tokenizer
        self.loadSeconds = loadSeconds
        var found: [String: Int] = [:]
        var bad: [String] = []
        for (token, _) in D1TokenIDs.tokens {
            guard let id = tokenizer.convertTokenToId(token) else {
                bad.append("\(token) is not in the tokenizer")
                continue
            }
            let encoded = tokenizer.encode(text: token, addSpecialTokens: false)
            if encoded != [id] { bad.append("\(token) encodes to \(encoded), not [\(id)]") }
            if let want = expected[token], want != id { bad.append("\(token) is \(id), metadata.json says \(want)") }
            if expected[token] == nil { bad.append("metadata.json has no id for \(token)") }
            found[token] = id
        }
        guard bad.isEmpty, let pad = found["<|pad|>"], let bos = found["<|startoftext|>"], let imEnd = found["<|im_end|>"],
              let mask = found["<|mask|>"], let state = found["<|reserved_7|>"], let q = found["<|reserved_8|>"],
              let opt = found["<|reserved_9|>"], let optEnd = found["<|reserved_10|>"], let decide = found["<|reserved_11|>"]
        else { throw D1OmniError.contract("tokenizer: \(bad.joined(separator: "; "))") }
        ids = D1TokenIDs(pad: pad, bos: bos, imEnd: imEnd, mask: mask, state: state, q: q, opt: opt, optEnd: optEnd,
                         decide: decide)
    }

    /// prompt.py `escape`: `<|name|>` -> `<¦name¦>` (`re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)`), on code
    /// points.
    public static func escape(_ text: String) -> String {
        let s = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        func isWord(_ c: Unicode.Scalar) -> Bool {
            switch c.value {
            case 0x30...0x39, 0x41...0x5A, 0x61...0x7A, 0x5F: return true
            default: return false
            }
        }
        var i = 0
        while i < s.count {
            if s[i] == "<", i + 1 < s.count, s[i + 1] == "|" {
                var j = i + 2
                while j < s.count, isWord(s[j]) { j += 1 }
                if j > i + 2, j + 1 < s.count, s[j] == "|", s[j + 1] == ">" {
                    out.append("<")
                    out.append("\u{00A6}")
                    out.append(contentsOf: s[(i + 2)..<j])
                    out.append("\u{00A6}")
                    out.append(">")
                    i = j + 2
                    continue
                }
            }
            out.append(s[i])
            i += 1
        }
        return String(out)
    }

    /// `tok(text, add_special_tokens=False)["input_ids"]`: no `<|startoftext|>`; the empty text has no ids.
    public func plain(_ text: String) -> [Int] {
        text.unicodeScalars.isEmpty ? [] : tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// prompt.py's `enc` inside `encode`: the ids of escape(s).
    public func enc(_ s: String) -> [Int] { plain(Self.escape(s)) }
}
