// Encoder — token ids, option aliases, readout groups and rows (conversion/d1/host.py §3–4; the provider's prompt.py
// `option_codes`, `aliases`, `readout_ids` and runner.py `_request` / `_logz_ids` at LiquidAI/d1-3B@da1fe36a), on the
// checkpoint's tokenizer.json through swift-transformers.
//
// The ids are HF `tokenizers`' (0.23.2) `encode(text, add_special_tokens=False)` of this tokenizer.json:
//   1. the 124 added tokens are cut out of the text first, leftmost and longest first, on the raw text (all have
//      lstrip / rstrip / single_word / normalized false), each one id — so "<|startoftext|>" in any text is 124894;
//   2. every other section is cut by the pre-tokenizer's Split regex (behavior Isolated) on code points;
//   3. each piece, byte-level mapped (GPT-2's bytes_to_unicode), is one id when the whole piece is a vocabulary entry
//      (the BPE model's `ignore_merges: true`), else the BPE merges.
// swift-transformers (resolved 2026-10-08: 1.3.4, c21fdcde3903; `from: "1.3.3"`) loads the file and runs the merges,
// but differs from `tokenizers` at steps 2 and 3: its Split matches through `String.range(of:options:
// .regularExpression)`, which snaps matches to grapheme clusters (open upstream: huggingface/swift-transformers PR
// #398), and its BPE ignores `ignore_merges` (open upstream: PR #397; on this tokenizer 2,459 vocabulary entries are
// not rebuilt by the merges, one fixture row is 116 ids in `tokenizers` and 118 through the merges). Its own ids
// (`plainTokens`) equal `tokenizers`' on 74 of the fixture's 393 rows. So this host does steps 1–3 itself and hands
// swift-transformers one piece at a time, through a second tokenizer whose pre-tokenizer is ByteLevel alone (one piece
// = one BPE word). When #397 and #398 are in a release the layer can go: `d1 rows --plain` (gate_swift.py ids,
// `plain_swift_transformers_rows`) then shows swift-transformers' own ids equal on every row.
//
// The pipeline is checked at load against exactly what this file reproduces (no normalizer; Sequence[Split(the regex
// below, Isolated, not inverted), ByteLevel(no prefix space, no regex)]; a BPE model without byte fallback or
// dropout; added tokens matched raw) and so are the special tokens' ids: a different tokenizer fails at load, not in
// an id.
//
//   aliases   each option code in order takes its own id when encode(code) is one id not taken yet, else the first
//             single-id untaken entry of A..Z, 00..99, a..z, #0..#199, AA..ZZ; the option line prints the taken code
//   groups    choice [alias id] + [encode(" " + code)[0]] when that is one other id; noul the single-id forms of
//             yes / Yes / YES and of no / No / NO; score [encode(str(i))[0]] for i < K
//   row       ids = encode(prefix + suffix), slot = len(ids) - 1
//   request   one question: its row (input_tokens = len(row)); several: the Tree of `_request`, trunk = encode(prefix),
//             branch = encode(suffix) apart (input_tokens = len(trunk) + Σ len(branch)); `shared` = `_logz_ids`' common
//             prefix of the rows (each row keeping its last id)
//   table     a request any of whose readout ids is outside the option table (head/option_rows) is refused whole

import Foundation
import Hub
import Tokenizers

public final class D1Tokenizer: @unchecked Sendable {
    /// The Split regex of the checkpoint's tokenizer.json (LFM2), the only one this encoder is written for.
    public static let splitPattern =
        #"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s"#

    /// The special tokens the host relies on (round 1 / 2b contract): token text -> id.
    public static let specialTokens: [(String, Int)] = {
        var t: [(String, Int)] = [("<|startoftext|>", 124894), ("<|im_start|>", 124899), ("<|im_end|>", 124900),
                                  ("<|pad|>", 124893), ("<image>", 124907), ("<|image_start|>", 125009),
                                  ("<|image_end|>", 125010), ("<|img_thumbnail|>", 125008)]
        for r in 1...10 {
            for c in 1...10 { t.append(("<|img_row_\(r)_col_\(c)|>", 124908 + 10 * (r - 1) + (c - 1))) }
        }
        return t
    }()

    /// The fallback pool of `prompt._FALLBACK_POOL`.
    public static let fallbackPool: [String] = {
        let up = (65...90).map { String(UnicodeScalar(UInt8($0))) }
        let low = (97...122).map { String(UnicodeScalar(UInt8($0))) }
        return up + (0..<100).map { $0 < 10 ? "0\($0)" : String($0) } + low + (0..<200).map { "#\($0)" }
            + up.flatMap { a in up.map { a + $0 } }
    }()

    /// GPT-2's bytes_to_unicode: byte -> its one-character string.
    static let byteTable: [String] = {
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var t = [String](repeating: "", count: 256)
        for (b, c) in zip(bs, cs) { t[b] = String(UnicodeScalar(UInt32(c))!) }
        return t
    }()

    /// swift-transformers on tokenizer.json with the pre-tokenizer replaced by ByteLevel alone: one piece -> its BPE ids.
    let bpe: PreTrainedTokenizer
    /// swift-transformers as loaded (`plainTokens`, for comparison); nil unless asked for at load.
    public let plain: PreTrainedTokenizer?
    let splitRegex: NSRegularExpression
    let addedRegex: NSRegularExpression
    public let addedIDs: [String: Int]
    public let ignoreMerges: Bool
    public let tokenizerClass: String?
    public let tokenizerJSONSHA256: String?
    private let lock = NSLock()
    private var singles: [String: Int?] = [:]

    /// The tokenizer of `folder` (tokenizer.json + tokenizer_config.json: a bundle's tokenizer/ or the HF snapshot).
    public static func load(folder: URL, plain: Bool = false) async throws -> D1Tokenizer {
        let config = LanguageModelConfigurationFromHub(modelFolder: folder)
        guard let tokenizerConfig = try await config.tokenizerConfig else {
            throw D1Error.contract("\(folder.path): no tokenizer_config.json")
        }
        let data = try await config.tokenizerData
        return try D1Tokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: data, plain: plain,
                               sha256: try? D1Files.sha256(folder.appendingPathComponent("tokenizer.json")))
    }

    public init(tokenizerConfig: Config, tokenizerData: Config, plain: Bool, sha256: String?) throws {
        var bad: [String] = []
        if !tokenizerData["normalizer"].isNull() { bad.append("a normalizer is set") }
        let pre = tokenizerData["preTokenizer"]
        let steps = pre["pretokenizers"].array(or: [])
        if pre["type"].string() != "Sequence" || steps.count != 2 {
            bad.append("pre_tokenizer is not Sequence[Split, ByteLevel]")
        } else {
            let s = steps[0], b = steps[1]
            if s["type"].string() != "Split" || s["pattern"]["Regex"].string() != Self.splitPattern
                || s["behavior"].string() != "Isolated" || s["invert"].boolean(or: false) {
                bad.append("the Split step is not the LFM2 regex, Isolated, not inverted")
            }
            if b["type"].string() != "ByteLevel" || b["addPrefixSpace"].boolean(or: true) || b["useRegex"].boolean(or: true) {
                bad.append("the ByteLevel step adds a prefix space or splits with its own regex")
            }
        }
        let model = tokenizerData["model"]
        if model["type"].string() != "BPE" { bad.append("the model is not BPE") }
        if model["byteFallback"].boolean(or: false) { bad.append("byte_fallback is set") }
        if !model["dropout"].isNull() { bad.append("BPE dropout is set") }
        var added: [String: Int] = [:]
        for t in tokenizerData["addedTokens"].array(or: []) {
            guard let id = t["id"].integer(), let content = t["content"].string() else { continue }
            if t["lstrip"].boolean(or: false) || t["rstrip"].boolean(or: false) || t["singleWord"].boolean(or: false)
                || t["normalized"].boolean(or: false) {
                bad.append("added token \(content) has lstrip / rstrip / single_word / normalized set")
            }
            added[content] = id
        }
        guard bad.isEmpty else { throw D1Error.contract("tokenizer.json: " + bad.joined(separator: "; ")) }
        ignoreMerges = model["ignoreMerges"].boolean(or: false)
        addedIDs = added
        tokenizerClass = tokenizerConfig["tokenizerClass"].string()
        tokenizerJSONSHA256 = sha256

        // the added tokens, longest first (code points), as one alternation: leftmost, then longest
        let sorted = added.keys.sorted { a, b in
            a.unicodeScalars.count != b.unicodeScalars.count ? a.unicodeScalars.count > b.unicodeScalars.count : a < b
        }
        addedRegex = try NSRegularExpression(pattern: sorted.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|"))
        splitRegex = try NSRegularExpression(pattern: Self.splitPattern)

        guard var dict = tokenizerData.dictionary() else { throw D1Error.contract("tokenizer.json is not an object") }
        guard let preKey = dict.keys.first(where: { $0.string == "pre_tokenizer" || $0.string == "preTokenizer" }) else {
            throw D1Error.contract("tokenizer.json has no pre_tokenizer")
        }
        dict[preKey] = Config(["type": Config("ByteLevel"), "add_prefix_space": Config(false), "trim_offsets": Config(true),
                               "use_regex": Config(false)] as [BinaryDistinctString: Config])
        bpe = try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: Config(dict), strict: true)
        self.plain = plain ? try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: true) : nil

        // the special tokens: in the added vocabulary with the contract's id, and encoded alone to that id
        for (token, id) in Self.specialTokens {
            if added[token] != id { bad.append("\(token) is \(added[token].map(String.init) ?? "absent"), not \(id)") }
            let e = encode(token)
            if e != [id] { bad.append("encode(\(token)) = \(e), not [\(id)]") }
        }
        guard bad.isEmpty else { throw D1Error.contract("tokenizer: " + bad.joined(separator: "; ")) }
    }

    // MARK: - encode

    /// `tokenizers` `encode(text, add_special_tokens=False)` of this tokenizer.json (steps 1–3 above).
    public func encode(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        var ids: [Int] = []
        let ns = text as NSString
        var pos = 0
        for m in addedRegex.matches(in: text, range: NSRange(location: 0, length: ns.length)) where m.range.length > 0 {
            if m.range.location > pos {
                encodeSection(ns.substring(with: NSRange(location: pos, length: m.range.location - pos)), into: &ids)
            }
            ids.append(addedIDs[ns.substring(with: m.range)]!)
            pos = m.range.location + m.range.length
        }
        if pos < ns.length { encodeSection(ns.substring(from: pos), into: &ids) }
        return ids
    }

    /// The Split regex's pieces of a section (Isolated: every match and every gap between matches is a piece), cut
    /// on UTF-16 offsets as ICU matches code points.
    public func pieces(_ section: String) -> [String] {
        let ns = section as NSString
        var out: [String] = []
        var pos = 0
        for m in splitRegex.matches(in: section, range: NSRange(location: 0, length: ns.length)) where m.range.length > 0 {
            if m.range.location > pos { out.append(ns.substring(with: NSRange(location: pos, length: m.range.location - pos))) }
            out.append(ns.substring(with: m.range))
            pos = m.range.location + m.range.length
        }
        if pos < ns.length { out.append(ns.substring(from: pos)) }
        return out
    }

    /// A piece's byte-level form (the BPE word).
    public static func byteLevel(_ piece: String) -> String {
        var s = ""
        s.reserveCapacity(piece.utf8.count)
        for b in piece.utf8 { s += byteTable[Int(b)] }
        return s
    }

    private func encodeSection(_ section: String, into ids: inout [Int]) {
        for p in pieces(section) {
            if ignoreMerges {
                let w = Self.byteLevel(p)
                if addedIDs[w] == nil, let id = bpe.convertTokenToId(w) {
                    ids.append(id)
                    continue
                }
            }
            ids += bpe.encode(text: p, addSpecialTokens: false)
        }
    }

    /// swift-transformers' own `encode(text:addSpecialTokens: false)` (comparison only; needs `plain: true` at load).
    public func plainTokens(_ text: String) -> [Int]? {
        guard let plain else { return nil }
        return text.isEmpty ? [] : plain.encode(text: text, addSpecialTokens: false)
    }

    /// The id of a text that encodes to exactly one id (`prompt.aliases` / `_ids`), memoised.
    public func singleToken(_ text: String) -> Int? {
        lock.lock()
        if let hit = singles[text] {
            lock.unlock()
            return hit
        }
        lock.unlock()
        let e = encode(text)
        let v: Int? = e.count == 1 ? e[0] : nil
        lock.lock()
        singles[text] = v
        lock.unlock()
        return v
    }

    // MARK: - aliases, groups

    /// `prompt.aliases`: a distinct single-id code per label, [(code, id)].
    public func aliases(_ labels: [String]) throws -> [(code: String, id: Int)] {
        var used = Set<Int>()
        var out: [(code: String, id: Int)] = []
        func take(_ raw: String) -> Bool {
            guard let i = singleToken(raw), !used.contains(i) else { return false }
            out.append((raw, i))
            used.insert(i)
            return true
        }
        let codes = D1Text.optionCodes(labels)
        for code in codes where !take(code) {
            if !Self.fallbackPool.contains(where: take) {
                throw D1Error.request("no single-token alias left for \(codes.count) options")
            }
        }
        return out
    }

    /// `prompt._ids`: the single-id forms of `texts`, deduplicated in order.
    func forms(_ texts: [String]) -> [Int] {
        var out: [Int] = []
        for t in texts {
            if let i = singleToken(t), !out.contains(i) { out.append(i) }
        }
        return out
    }

    /// `prompt.readout_ids`: one id group per option, max-pooled.
    public func readoutGroups(_ q: D1Question, aliases: [(code: String, id: Int)]?) throws -> [[Int]] {
        switch q.kind {
        case .noul:
            let yes = forms(["yes", "Yes", "YES"]), no = forms(["no", "No", "NO"])
            guard !yes.isEmpty, !no.isEmpty else { throw D1Error.request("tokenizer has no single-token yes/no") }
            return [yes, no]
        case .score:
            let groups = q.levels.indices.map { forms([String($0)]) }
            if groups.contains(where: \.isEmpty) {
                throw D1Error.request("score with \(q.levels.count) levels needs single-token digits")
            }
            return groups
        case .choice:
            let al = try aliases ?? self.aliases(q.labels)
            return al.map { a in [a.id] + forms([" " + a.code]).filter { $0 != a.id } }
        }
    }

    // MARK: - rows

    /// `host.build_question`: one validated question -> its row.
    public func row(state: JSONValue, _ q: D1Question) throws -> D1Row {
        let al = q.kind == .choice ? try aliases(q.labels) : nil
        let codes = al?.map(\.code)
        let prefix = D1Text.prefix(state), suffix = D1Text.suffix(q, codes: codes)
        let ids = encode(prefix + suffix)
        return D1Row(name: q.name, kind: q.kind, text: prefix + suffix, suffix: suffix, ids: ids, codes: codes,
                     aliasIDs: al?.map(\.id), groups: try readoutGroups(q, aliases: al), keys: q.keys,
                     levels: q.kind == .score ? q.levels : nil)
    }

    /// `host.build_request`: the request's rows, the option-table check (`table` = the bundle's option-row ids) and the
    /// path: one question = its row; several = the Tree (trunk = encode(prefix), each branch = encode(suffix)).
    public func rows(_ request: D1Request, table: Set<Int>?) throws -> D1Rows {
        let rows = try request.questions.map { try row(state: request.state, $0) }
        if let table { try Self.optionTableCheck(rows, table: table) }
        let shared = Self.sharedPrefix(rows.map(\.ids))
        guard rows.count > 1 else {
            return D1Rows(rows: rows, trunk: nil, branches: nil, shared: shared, inputTokens: rows[0].ids.count)
        }
        let trunk = encode(D1Text.prefix(request.state))
        let branches = rows.map { encode($0.suffix) }
        return D1Rows(rows: rows, trunk: trunk, branches: branches, shared: shared,
                      inputTokens: trunk.count + branches.reduce(0) { $0 + $1.count })
    }

    /// `host.shared_prefix` (`runner._logz_ids`): the rows' common prefix, each row keeping at least its last id.
    public static func sharedPrefix(_ rows: [[Int]]) -> Int {
        guard rows.count >= 2, let shortest = rows.map(\.count).min() else { return 0 }
        var shared = 0
        while shared < shortest - 1, Set(rows.map { $0[shared] }).count == 1 { shared += 1 }
        return shared
    }

    /// `host.option_table_check`: the first readout id outside the table refuses the request.
    public static func optionTableCheck(_ rows: [D1Row], table: Set<Int>) throws {
        for r in rows {
            for (key, g) in zip(r.keys, r.groups) {
                for i in g where !table.contains(i) {
                    throw D1Error.request("questions.\(r.name): the token id \(i) of label \(PythonFormat.reprString(key)) is "
                        + "not in the option table")
                }
            }
        }
    }

    /// `host.graph_context_check`: a row of `length` ids fits a static-S graph (its padded end <= maxContext - 1).
    public static func graphContextCheck(length: Int, chunk: Int, maxContext: Int) throws {
        let padded = (length + chunk - 1) / chunk * chunk
        if padded > maxContext - 1 {
            throw D1Error.graphLimit("a row of \(length) tokens runs \(padded) padded positions, over the graph's "
                + "\(maxContext - 1) (rows of at most \((maxContext - 1) / chunk * chunk) tokens at S = \(chunk))")
        }
    }
}

/// One question's row.
public struct D1Row: Sendable {
    public let name: String
    public let kind: D1Question.Kind
    /// prefix + suffix
    public let text: String
    public let suffix: String
    public let ids: [Int]
    /// the aliases' codes and ids (choice only)
    public let codes: [String]?
    public let aliasIDs: [Int]?
    public let groups: [[Int]]
    public let keys: [String]
    /// score: the levels (the legend)
    public let levels: [String]?
    /// the answer slot: the row's last id
    public var slot: Int { ids.count - 1 }
}

/// A request's rows and the path the provider runs it on.
public struct D1Rows: Sendable {
    public let rows: [D1Row]
    /// several questions: the Tree's trunk and branches (encoded apart); one question: nil
    public let trunk: [Int]?
    public let branches: [[Int]]?
    /// `_logz_ids`' common prefix of the rows
    public let shared: Int
    public let inputTokens: Int

    public var path: String { trunk == nil ? "row" : "tree" }
    /// whether trunk + branch equals the question's row ids (a pre-token boundary at the prefix's end)
    public var equalsRow: [Bool]? {
        guard let trunk, let branches else { return nil }
        return zip(rows, branches).map { trunk + $0.1 == $0.0.ids }
    }
}
