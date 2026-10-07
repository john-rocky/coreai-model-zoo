// D1Decider — a System One request -> the System One response, in the order host.py runs it:
//
//   let d1 = try await D1Decider(bundle: bundleDir)
//   let rows = try d1.rows(requestJSON: data)          // request checks -> rows -> option table -> graph limit
//   let response = try await d1.decide(requestJSON: data)
//
//   request ──D1Request (host.py's checks)──> D1Text (prefix / suffix) ──D1Tokenizer──> one row per question,
//           aliases, readout groups, keys; the Tree's trunk / branches and input_tokens
//   graph   (round 3c+) per row ceil(T / S) calls of the static-S decoder from zero states -> the slot's hidden row
//   readout D1Readout (float64, NumPy's BLAS call) on head/option_rows -> p -> the answers -> the response body
//
// Round 3a: everything but the graph. `decide` throws `D1Error.graphNotWired` after the rows; `readout` takes a slot's
// hidden row from the caller (a file, a test) and runs the rest. Everything model-specific comes from the bundle:
// metadata.json (`language.prefill_chunk` S, `max_context_length`, `decision.prompt.special`, `decision.option_table`,
// `vision.image_token`), tokenizer/, head/option_rows.{json,safetensors}. A bundle that differs from the contract fails
// at load, not in a probability.

import Foundation

public final class D1Decider: @unchecked Sendable {
    /// What the bundle's metadata.json says.
    public struct Metadata: Sendable {
        public let name: String
        public let asset: String
        public let vocab: Int
        public let maxContext: Int
        public let chunk: Int
        /// decision.prompt.special: token -> id
        public let special: [String: Int]
        public let optionFiles: [String]
        public let optionCount: Int
        public let imageTokenID: Int?

        public init(bundle: URL) throws {
            let url = bundle.appendingPathComponent("metadata.json")
            let j = try JSONParser.parse(Data(contentsOf: url))
            guard j["kind"]?.string == "decision-backbone" else {
                throw D1Error.bundle("\(url.path): kind is not decision-backbone")
            }
            guard let lang = j["language"], let asset = j["assets"]?["main"]?.string,
                  let vocab = lang["vocab_size"]?.intValue, let ctx = lang["max_context_length"]?.intValue,
                  let chunk = lang["prefill_chunk"]?.intValue, let decision = j["decision"],
                  let special = decision["prompt"]?["special"]?.members,
                  let table = decision["option_table"], let files = table["files"]?.array?.compactMap(\.string),
                  let n = table["n"]?.intValue
            else { throw D1Error.bundle("\(url.path): no language / assets / decision.prompt.special / decision.option_table") }
            var sp: [String: Int] = [:]
            for m in special {
                guard let token = m.value["token"]?.string, let id = m.value["id"]?.intValue else {
                    throw D1Error.bundle("\(url.path): decision.prompt.special.\(m.key) is not {token, id}")
                }
                sp[token] = id
            }
            name = j["name"]?.string ?? bundle.lastPathComponent
            self.asset = asset
            self.vocab = vocab
            maxContext = ctx
            self.chunk = chunk
            self.special = sp
            optionFiles = files
            optionCount = n
            imageTokenID = j["vision"]?["image_token"]?["id"]?.intValue
        }
    }

    public let bundle: URL
    public let metadata: Metadata
    public let tokenizer: D1Tokenizer
    public let table: D1OptionTable

    public init(bundle: URL) async throws {
        self.bundle = bundle
        metadata = try Metadata(bundle: bundle)
        tokenizer = try await D1Tokenizer.load(folder: bundle.appendingPathComponent("tokenizer"))
        table = try Self.loadTable(bundle.appendingPathComponent("head"), count: metadata.optionCount)
        var bad: [String] = []
        for (token, id) in metadata.special.sorted(by: { $0.value < $1.value }) where tokenizer.addedIDs[token] != id {
            bad.append("\(token): metadata \(id), tokenizer \(tokenizer.addedIDs[token].map(String.init) ?? "absent")")
        }
        if let image = metadata.imageTokenID, image != D1Vision.imageID {
            bad.append("vision.image_token.id \(image), the host's \(D1Vision.imageID)")
        }
        guard bad.isEmpty else { throw D1Error.contract("metadata.json vs the tokenizer: " + bad.joined(separator: "; ")) }
    }

    /// head/option_rows.json + option_rows.safetensors: the same ascending ids in both, rows [n, hidden] fp32.
    public static func loadTable(_ head: URL, count: Int) throws -> D1OptionTable {
        let js = try JSONParser.parse(Data(contentsOf: head.appendingPathComponent("option_rows.json")))
        guard let ids = js["ids"]?.array?.compactMap(\.intValue), let file = js["file"]?.string,
              let hidden = js["hidden"]?.intValue
        else { throw D1Error.bundle("option_rows.json: no ids / file / hidden") }
        let t = try Safetensors.read(head.appendingPathComponent(file))
        guard case .i32(let shapeIDs, let stIDs)? = t["ids"], case .f32(let shapeRows, let rows)? = t["rows"] else {
            throw D1Error.bundle("\(file): no int32 `ids` / fp32 `rows`")
        }
        guard shapeIDs == [ids.count], stIDs.map(Int.init) == ids, shapeRows == [ids.count, hidden], ids.count == count else {
            throw D1Error.contract("option table: json \(ids.count) ids, safetensors ids \(shapeIDs) rows \(shapeRows), "
                + "metadata n \(count), hidden \(hidden)")
        }
        return try D1OptionTable(ids: ids, hidden: hidden, rows: rows.map(Double.init))
    }

    /// The request's rows within the contract: host.py's checks, the option table, the graph's row limit.
    public func rows(requestJSON data: Data) throws -> D1Rows {
        let request = try D1Request(data: data)
        let rows = try tokenizer.rows(request, table: table.idSet)
        for r in rows.rows {
            try D1Tokenizer.graphContextCheck(length: r.ids.count, chunk: metadata.chunk, maxContext: metadata.maxContext)
        }
        return rows
    }

    /// Round 3a: the rows, then `graphNotWired` (the decoder is called from round 3c on).
    public func decide(requestJSON data: Data) async throws -> JSONValue {
        _ = try rows(requestJSON: data)
        throw D1Error.graphNotWired
    }

    /// The readout and the response from the slots' hidden rows (one per question, `hidden` wide, in request order).
    public func response(requestJSON data: Data, slotHidden: [[Double]]) throws -> JSONValue {
        let request = try D1Request(data: data)
        let rows = try tokenizer.rows(request, table: table.idSet)
        guard slotHidden.count == rows.rows.count, slotHidden.allSatisfy({ $0.count == table.hidden }) else {
            throw D1Error.contract("\(slotHidden.count) hidden rows for \(rows.rows.count) questions (\(table.hidden) wide)")
        }
        let p = try zip(rows.rows, slotHidden).map { try D1Readout.readout(hidden: $1, table: table, groups: $0.groups).p }
        return D1Readout.response(request.questions, probabilities: p, inputTokens: rows.inputTokens)
    }
}

/// A .safetensors file's F32 / I32 tensors (an 8-byte little-endian header length, the JSON header, the raw data).
enum Safetensors {
    enum Tensor {
        case f32([Int], [Float])
        case i32([Int], [Int32])
    }

    static func read(_ url: URL) throws -> [String: Tensor] {
        let raw = try Data(contentsOf: url)
        guard raw.count >= 8 else { throw D1Error.bundle("\(url.lastPathComponent): shorter than its header length") }
        let n = raw.prefix(8).enumerated().reduce(0) { $0 | (Int($1.element) << (8 * $1.offset)) }
        guard 8 + n <= raw.count else { throw D1Error.bundle("\(url.lastPathComponent): header length \(n)") }
        let header = try JSONParser.parse(raw.subdata(in: 8..<(8 + n)))
        let base = 8 + n
        var out: [String: Tensor] = [:]
        for m in header.members ?? [] where m.key != "__metadata__" {
            guard let dtype = m.value["dtype"]?.string, let shape = m.value["shape"]?.array?.compactMap(\.intValue),
                  let off = m.value["data_offsets"]?.array?.compactMap(\.intValue), off.count == 2
            else { throw D1Error.bundle("\(url.lastPathComponent): \(m.key) has no dtype / shape / data_offsets") }
            let count = shape.reduce(1, *)
            guard off[1] - off[0] == count * 4, base + off[1] <= raw.count, dtype == "F32" || dtype == "I32" else {
                throw D1Error.bundle("\(url.lastPathComponent): \(m.key) is \(dtype) \(shape) at \(off)")
            }
            var bits = [UInt32](repeating: 0, count: count)
            bits.withUnsafeMutableBytes { dst in
                raw.withUnsafeBytes { src in
                    dst.copyMemory(from: UnsafeRawBufferPointer(rebasing: src[(base + off[0])..<(base + off[1])]))
                }
            }
            bits = bits.map { UInt32(littleEndian: $0) }
            out[m.key] = dtype == "F32" ? .f32(shape, bits.map { Float(bitPattern: $0) }) : .i32(shape, bits.map { Int32(bitPattern: $0) })
        }
        return out
    }
}
