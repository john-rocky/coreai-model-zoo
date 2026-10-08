// D1Decider — a System One request -> the System One response, in the order host.py runs it:
//
//   let d1 = try await D1Decider(bundle: bundleDir)
//   let rows = try d1.rows(requestJSON: data)          // request checks -> rows -> option table -> graph limit
//   try await d1.loadGraph(tower: towerBundleDir)      // the decoder's AOT asset (and the tower's, for pictures)
//   let response = try await d1.decide(requestJSON: data, images: [pictureURL])
//
//   request ──D1Request (host.py's checks)──> D1Text (prefix / suffix) ──D1Tokenizer──> one row per question,
//           aliases, readout groups, keys; the Tree's trunk / branches and input_tokens
//   graph   (round 3c) D1Graph: per row ceil(T / S) calls of the static-S decoder (Decoder.swift) from zero states, or
//           the state's first floor(Ls / S) * S ids once and every row's rest from a copy of the states (shared); the
//           pictures' crops through the tower (Tower.swift) into the image rows first -> the slot's hidden row
//   readout D1Readout (float64, NumPy's BLAS call) on head/option_rows -> p -> the answers -> the response body
//
// Round 3a: everything but the graph; round 3c wires it (`loadGraph`, then `decide` / `trace` / `prepare`; a decider
// loaded without a graph still stops with `D1Error.graphNotWired` after the rows). `response(requestJSON:slotHidden:)`
// takes slot hidden rows from the caller (a file, a test) and runs the rest. Everything model-specific comes from the bundle:
// metadata.json (`language.prefill_chunk` S, `max_context_length`, `decision.prompt.special`, `decision.option_table`,
// `vision.image_token`), tokenizer/, head/option_rows.{json,safetensors}. A bundle that differs from the contract fails
// at load, not in a probability.

import CoreAI
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
    /// The decoder (and tower) once `loadGraph` ran; nil = the text side only (`decide` stops at the rows).
    public internal(set) var graph: D1Graph?

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

    /// The response for one request (its JSON text) and its pictures (files, in the order the request names them: the
    /// k-th file is the k-th `<image>` of the prompt); without a loaded graph, the rows and then `graphNotWired`. A
    /// request the host refuses throws `D1Error.request` / `.graphLimit` with host.py's text, before any graph call.
    public func decide(requestJSON data: Data, images: [URL] = [], shared: Bool = false) async throws -> JSONValue {
        guard graph != nil else {
            _ = try rows(requestJSON: data)
            throw D1Error.graphNotWired
        }
        let request = try D1Request(data: data)
        let pictures = images.isEmpty ? nil : try self.pictures(files: images)
        return try await trace(request: request, mode: shared ? .shared : .direct, pictures: pictures).response
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

// MARK: - The graph (round 3c)
//
// conversion/d1/decide.py is the specification this part copies (its `D1.build` / `decide` / `prepare` /
// `decide_prepared`), and gate_swift.py compares the two on the same AOT assets: the rows' ids, the hidden rows (sha256
// of the fp16 bytes), p (bits) and the response's bytes.
//
//   plan     D1Tokenizer.rows (host.py's checks; the option table on a model bundle) -> per question the row's ids (with
//            pictures: vision_host.prompt_ids of prefix-with-markers + suffix, <image> -> 128,000 + k), the graph's ids
//            (a toy bundle folds them), the groups the readout reads (given, the toy's folded, or the row's), the option
//            table on those groups, the row limit, the image-token limit, input_tokens, and Ls
//   Ls       the state's stable tokens (decide.py `stable_prefix`): the prefix's ids minus the ids of its last pre-token,
//            the one piece a question's text can change (":\n" then "\n..." is the one token ":\n\n"); 0 when a row does
//            not start with them. shared / prepared run k = floor(Ls / S) * S of them once.
//   pictures each file (D1Pixels: decode, cap_pixels, plan, crops) -> every crop's four inputs (D1TowerInputs, the tower
//            bundle's position table); the pixel work comes first, as decide.py does it, then the plan's checks in
//            decide.py's order (the option table, every row's padded length, the image tokens against N with the
//            metadata's `vision.limits.refusal_text`): a refused request makes no graph call, the tower's included
//   images   every crop's four inputs through the tower, crops and pictures in order -> image rows fp16

/// The decoder's side of a bundle's metadata.json: the graph's contract and what the host feeds it.
public struct D1GraphMetadata: Sendable {
    public let contract: D1GraphContract
    /// language.prefill_chunk S (input_ids [1, S])
    public let chunk: Int
    public let maxContext: Int
    /// d (hidden [1, S, d])
    public let hidden: Int
    /// N (image_embeds [N, d])
    public let imageRows: Int
    /// language.vocab_size: the V of the extension ids V + k the graph reads
    public let vocab: Int
    /// the pad id the graph reads: <|pad|> (124893), or a toy's folded id
    public let padID: Int
    /// a toy bundle (metadata `toy`, rounds 2a / 3b): every real id folded id % vocab, an extension id 128,000 + k ->
    /// vocab + k, the readout groups folded the same way unless they are given
    public let toyFold: Int?
    /// the refusal of a request whose pictures need more than N image rows: metadata `vision.limits.refusal_text` with
    /// `<n>` for the request's image tokens ("images: <n> image tokens over the graph's N image rows"); a bundle without
    /// a vision block (the toys) gets the same text
    public let imageRefusal: String

    public init(bundle: URL, special: [String: Int]) throws {
        let url = bundle.appendingPathComponent("metadata.json")
        let j = try JSONParser.parse(Data(contentsOf: url))
        guard let lang = j["language"], let c = lang["contract"], let s = lang["prefill_chunk"]?.intValue,
              let ctx = lang["max_context_length"]?.intValue, let v = lang["vocab_size"]?.intValue
        else { throw D1Error.bundle("\(url.path): no language.contract / prefill_chunk / max_context_length / vocab_size") }
        let contract = try D1GraphContract(c, what: "\(url.lastPathComponent) language.contract")
        guard contract.chunk == s, contract.inputs["input_ids"] == D1TensorSpec(shape: [1, s], dtype: "int32"),
              contract.inputs["position_ids"] == D1TensorSpec(shape: [1, -1], dtype: "int32"),
              contract.outputs["hidden"] == D1TensorSpec(shape: [1, s, contract.hidden], dtype: "float16"),
              contract.inputs["image_embeds"] == D1TensorSpec(shape: [contract.imageRows, contract.hidden], dtype: "float16")
        else {
            throw D1Error.contract("\(url.lastPathComponent) language.contract: not input_ids [1, \(s)] int32, position_ids "
                + "[1, -1] int32, image_embeds [N, d] float16 -> hidden [1, \(s), d] float16")
        }
        guard let pad = special["<|pad|>"] else { throw D1Error.bundle("\(url.path): no <|pad|> in decision.prompt.special") }
        if let toy = j["toy"] {
            guard toy["fold"]?.string == "id % \(v)", let folded = toy["pad_folded"]?.intValue, folded == pad % v else {
                throw D1Error.contract("\(url.lastPathComponent) toy: fold \(toy["fold"]?.string ?? "absent"), pad_folded "
                    + "\(toy["pad_folded"]?.intValue.map(String.init) ?? "absent") is not id % \(v)")
            }
            toyFold = v
            padID = folded
        } else {
            guard v == D1Vision.extensionBase else {
                throw D1Error.contract("\(url.lastPathComponent): vocab_size \(v) is not the extension base \(D1Vision.extensionBase)")
            }
            toyFold = nil
            padID = pad
        }
        let n = contract.imageRows
        if let vision = j["vision"] {
            guard let text = vision["limits"]?["refusal_text"]?.string, text.components(separatedBy: "<n>").count == 2,
                  text.replacingOccurrences(of: "<n>", with: "\(n + 1)")
                  == D1Tokenizer.imageRowsRefusal(tokens: n + 1, imageRows: n),
                  vision["n_image_tokens"]?.intValue == n
            else {
                throw D1Error.contract("\(url.lastPathComponent) vision: n_image_tokens \(vision["n_image_tokens"]?.intValue.map(String.init) ?? "absent"), "
                    + "limits.refusal_text \(vision["limits"]?["refusal_text"]?.string ?? "absent") is not the host's for N = \(n)")
            }
            imageRefusal = text
        } else {
            imageRefusal = D1Tokenizer.imageRowsRefusal(tokens: nil, imageRows: n)
        }
        self.contract = contract
        chunk = s
        maxContext = ctx
        hidden = contract.hidden
        imageRows = n
        vocab = v
    }
}

/// The loaded graphs: the decoder (and the tower, for pictures) and how they were loaded.
public final class D1Graph: @unchecked Sendable {
    public enum Asset: String, Sendable { case aot, jit }

    public let metadata: D1GraphMetadata
    public let decoder: D1Decoder
    public let tower: D1Tower?
    public let asset: Asset
    public let assetURL: URL
    /// the decoder's and the tower's AIModel + loadFunction, the state allocation, in seconds
    public let loadSeconds: Double
    /// the warm-up call's seconds (nil when not asked for)
    public let warmUpSeconds: Double?

    init(metadata: D1GraphMetadata, decoder: D1Decoder, tower: D1Tower?, asset: Asset, assetURL: URL, loadSeconds: Double,
         warmUpSeconds: Double?) {
        self.metadata = metadata
        self.decoder = decoder
        self.tower = tower
        self.asset = asset
        self.assetURL = assetURL
        self.loadSeconds = loadSeconds
        self.warmUpSeconds = warmUpSeconds
    }
}

/// One question's row as the graph runs it.
public struct D1GraphRow: Sendable {
    /// the text row (Encoder.swift): name, kind, text, suffix, codes, the row's own groups, keys
    public let row: D1Row
    /// the row's ids (with pictures: the processor's, <image> -> 128,000 + k)
    public let ids: [Int]
    /// the ids in the graph's vocabulary (a toy folds them)
    public let graphIDs: [Int]
    public let slot: Int
    /// the groups the readout reads (the row's; a toy's given or folded ones)
    public let readGroups: [[Int]]
}

/// A request's rows for the graph.
public struct D1Plan: Sendable {
    public let request: D1Request
    public let rows: [D1GraphRow]
    public let inputTokens: Int
    /// Ls: the state's stable tokens every row starts with
    public let stateTokens: Int
    /// the prefix's ids (the trunk)
    public let trunkLength: Int
    public let imageTokens: Int
}

/// A state run once and kept: its first k = floor(Ls / S) * S ids and the three states after them (text requests).
public final class D1Prepared: @unchecked Sendable {
    public let state: JSONValue
    /// the kept ids (k of them, real ids)
    public let ids: [Int]
    public let k: Int
    public let stateTokens: Int
    /// the kept ids' hidden rows [k * d] fp16
    public let hidden: [Float16]
    public let callSeconds: [Double]
    public let seconds: Double
    let states: [String: NDArray]

    init(state: JSONValue, ids: [Int], k: Int, stateTokens: Int, hidden: [Float16], callSeconds: [Double], seconds: Double,
         states: [String: NDArray]) {
        self.state = state
        self.ids = ids
        self.k = k
        self.stateTokens = stateTokens
        self.hidden = hidden
        self.callSeconds = callSeconds
        self.seconds = seconds
        self.states = states
    }
}

/// Everything one decision did, for gates and timing.
public struct D1Trace: Sendable {
    public let plan: D1Plan
    /// "direct", "shared" or "prepared"
    public let mode: String
    /// the ids run once for every row (0 = every row ran whole)
    public let sharedK: Int
    /// per row, the hidden rows [T * d] fp16 the readout read from (shared / prepared: the kept rows, then the row's)
    public let hidden: [[Float16]]
    public let logits: [[Int: Double]]
    public let probabilities: [[Double]]
    public let response: JSONValue
    /// every graph call's seconds, in order
    public let callSeconds: [Double]
    /// state zeroing / restoring
    public let resetSeconds: Double
    /// per crop, the tower's whole output (256 * d)
    public let towerOutputs: [[Float]]
    public let towerSeconds: [Double]
    public let imageRows: Int
    /// prepared: rows that did not start with the prepared ids (run whole from zero states)
    public let rowsRunWhole: Int
    /// plan (rows, checks), images (tower + image rows written), graph (states + calls), readout (p + response), wall
    public let seconds: [String: Double]
}

extension D1Decider {
    public enum Mode: String, Sendable { case direct, shared }

    /// Loads the decoder graph and wires `decide`: `asset` .aot = `<bundles>_aotc/<name>.h16c.aimodelc` with
    /// SpecializationOptions.default, .jit = the bundle's `.aimodel` specialized here with D1Paths.decoderJITOptions
    /// (`assetURL` / `options` override either); `tower` = a tower bundle for requests with pictures (`towerAsset`
    /// likewise). Both are checked against their metadata.json at load. `warmUp` runs one call before any request.
    @discardableResult
    public func loadGraph(asset: D1Graph.Asset = .aot, assetURL: URL? = nil, options: SpecializationOptions? = nil,
                          tower: URL? = nil, towerAsset: D1Graph.Asset = .aot, warmUp: Bool = false) async throws -> D1Graph {
        let gm = try D1GraphMetadata(bundle: bundle, special: metadata.special)
        let url = assetURL ?? (asset == .aot ? D1Paths.aot(bundle: bundle, name: metadata.name)
                                             : bundle.appendingPathComponent(metadata.asset))
        let opts = options ?? (url.pathExtension == "aimodelc" ? .default : D1Paths.decoderJITOptions)
        let t0 = ContinuousClock.now
        let dec = try await D1Decoder(contentsOf: url, contract: gm.contract, maxContext: gm.maxContext, padID: gm.padID,
                                      options: opts)
        var tw: D1Tower? = nil
        if let tower {
            var towerURL: URL? = nil
            if towerAsset == .jit { towerURL = try D1Tower.modelAsset(bundle: tower) }
            let t = try await D1Tower(bundle: tower, asset: towerURL)
            guard t.width == gm.hidden else { throw D1Error.contract("the tower's width \(t.width) != the decoder's \(gm.hidden)") }
            tw = t
        }
        let load = d1Seconds(since: t0)
        var w: Double? = nil
        if warmUp { w = try await dec.warmUp() }
        let g = D1Graph(metadata: gm, decoder: dec, tower: tw, asset: url.pathExtension == "aimodelc" ? .aot : .jit,
                        assetURL: url, loadSeconds: load, warmUpSeconds: w)
        graph = g
        return g
    }

    public func requireGraph() throws -> D1Graph {
        guard let graph else { throw D1Error.graphNotWired }
        return graph
    }

    /// A request's picture files (text order) -> every crop's four inputs, made here with the loaded tower bundle's
    /// position table (`D1TowerInputs.pictures(files:table:)`).
    public func pictures(files: [URL]) throws -> D1TowerInputs {
        guard let tower = try requireGraph().tower else {
            throw D1Error.contract("a request with pictures needs a tower (loadGraph(tower:))")
        }
        return try D1TowerInputs.pictures(files: files, table: tower.positionTable)
    }

    /// Real ids -> the graph's (a toy folds them: id % V, an extension id 128,000 + k -> V + k).
    public func graphIDs(_ ids: [Int]) throws -> [Int] {
        guard let v = try requireGraph().metadata.toyFold else { return ids }
        return ids.map { $0 >= D1Vision.extensionBase ? v + ($0 - D1Vision.extensionBase) : $0 % v }
    }

    /// decide.py `stable_prefix`: of a prefix's ids, those every row starting with this prefix also starts with — all but
    /// the ids of the prefix's last pre-token (the text after its last added token, cut by the Split regex). 0 when the
    /// last piece's ids are not the prefix's last ids (no sharing: still exact).
    public func stablePrefix(_ text: String, _ ids: [Int]) -> Int {
        var cut = text.startIndex
        for t in tokenizer.addedIDs.keys {
            if let r = text.range(of: t, options: [.literal, .backwards]), r.upperBound > cut { cut = r.upperBound }
        }
        let tail = String(text[cut...])
        guard !tail.isEmpty, let last = tokenizer.pieces(tail).last else { return ids.count }
        let lastIDs = tokenizer.encode(last)
        let n = ids.count - lastIDs.count
        guard n >= 0, Array(ids[n...]) == lastIDs else { return 0 }
        return n
    }

    /// The graph's rows of a request (decide.py `D1.build`). `groups` = {question name: groups} replaces the groups the
    /// readout reads (a toy's: round 2a's toy oracle); `pictures` = the request's pictures with their crops' inputs.
    public func plan(request: D1Request, groups: [String: [[Int]]]? = nil, pictures: D1TowerInputs? = nil) throws -> D1Plan {
        let gm = try requireGraph().metadata
        let text = try tokenizer.rows(request, table: gm.toyFold == nil ? table.idSet : nil)
        let plans = pictures?.pictures.map(\.plan) ?? []
        let prefix = D1Text.prefix(request.state, images: String(repeating: D1Vision.imageToken, count: plans.count))
        var rows: [D1GraphRow] = []
        for r in text.rows {
            let ids = plans.isEmpty ? r.ids
                : D1Vision.extensionIDs(try D1Vision.promptIDs(tokenizer, text: prefix + r.suffix, plans: plans))
            let read = groups?[r.name] ?? (gm.toyFold.map { v in r.groups.map { $0.map { $0 % v } } } ?? r.groups)
            rows.append(D1GraphRow(row: r, ids: ids, graphIDs: try graphIDs(ids), slot: ids.count - 1, readGroups: read))
        }
        try D1Tokenizer.optionTableCheck(rows.map {
            D1Row(name: $0.row.name, kind: $0.row.kind, text: $0.row.text, suffix: $0.row.suffix, ids: $0.row.ids,
                  codes: $0.row.codes, aliasIDs: $0.row.aliasIDs, groups: $0.readGroups, keys: $0.row.keys, levels: $0.row.levels)
        }, table: table.idSet)
        for r in rows { try D1Tokenizer.graphContextCheck(length: r.ids.count, chunk: gm.chunk, maxContext: gm.maxContext) }
        let imageTokens = D1Vision.imageTokenCount(plans)
        try D1Tokenizer.imageRowsCheck(tokens: imageTokens, imageRows: gm.imageRows, refusal: gm.imageRefusal)
        let trunk = plans.isEmpty ? tokenizer.encode(prefix)
            : D1Vision.extensionIDs(try D1Vision.promptIDs(tokenizer, text: prefix, plans: plans))
        var stable = stablePrefix(prefix, trunk)
        if rows.contains(where: { Array($0.ids.prefix(stable)) != Array(trunk.prefix(stable)) }) { stable = 0 }
        let inputTokens = plans.isEmpty ? text.inputTokens
            : try D1Vision.requestIDs(tokenizer, request,
                                      pictures: pictures!.pictures.map { (width: $0.width, height: $0.height) }).inputTokens
        return D1Plan(request: request, rows: rows, inputTokens: inputTokens, stateTokens: stable, trunkLength: trunk.count,
                      imageTokens: imageTokens)
    }

    /// The whole decision with its intermediate values and times. `zeroImages` (a gate's control) runs the pictures'
    /// rows with the image rows left zero, no tower call.
    public func trace(request: D1Request, mode: Mode = .direct, groups: [String: [[Int]]]? = nil,
                      pictures: D1TowerInputs? = nil, zeroImages: Bool = false) async throws -> D1Trace {
        let g = try requireGraph()
        let t0 = ContinuousClock.now
        let p = try plan(request: request, groups: groups, pictures: pictures)
        let t1 = ContinuousClock.now
        var towerOut: [[Float]] = []
        var towerSecs: [Double] = []
        var imageRows: [Float16]? = nil
        if let pictures, !pictures.pictures.isEmpty, !zeroImages {
            guard let tower = g.tower else { throw D1Error.contract("a request with pictures needs a tower (loadGraph(tower:))") }
            let r = try await tower.imageRows(pictures.pictures.flatMap(\.crops))
            imageRows = r.rows
            towerOut = r.outputs
            towerSecs = r.seconds
        }
        try g.decoder.setImageRows(imageRows)
        let t2 = ContinuousClock.now
        let S = g.metadata.chunk
        let k = mode == .shared ? (p.stateTokens / S) * S : 0
        var hidden: [[Float16]] = []
        var calls: [Double] = []
        var reset = 0.0
        if k > 0 {
            let prefix = Array(p.rows[0].graphIDs[0..<k])
            guard p.rows.allSatisfy({ Array($0.graphIDs.prefix(k)) == prefix }) else {
                throw D1Error.contract("the rows do not share their first \(k) ids")
            }
            let (pre, tails) = try await g.decoder.runShared(prefix: prefix, tails: p.rows.map { Array($0.graphIDs[k...]) })
            calls += pre.callSeconds
            reset += pre.resetSeconds
            for t in tails {
                hidden.append(pre.hidden + t.hidden)
                calls += t.callSeconds
                reset += t.resetSeconds
            }
        } else {
            for r in p.rows {
                let pass = try await g.decoder.run(ids: r.graphIDs)
                hidden.append(pass.hidden)
                calls += pass.callSeconds
                reset += pass.resetSeconds
            }
        }
        return try answer(plan: p, hidden: hidden, mode: mode.rawValue, k: k, calls: calls, reset: reset, towerOutputs: towerOut,
                          towerSeconds: towerSecs, rowsRunWhole: 0, times: (t0, t1, t2, ContinuousClock.now))
    }

    /// The state run once (its first k = floor(Ls / S) * S ids) and kept, for `decide(prepared:)` later (text only).
    public func prepare(state: JSONValue) async throws -> D1Prepared {
        let g = try requireGraph()
        let t0 = ContinuousClock.now
        let prefix = D1Text.prefix(state)
        let trunk = tokenizer.encode(prefix)
        let stable = stablePrefix(prefix, trunk)
        let k = (stable / g.metadata.chunk) * g.metadata.chunk
        try g.decoder.setImageRows(nil)
        let ids = Array(trunk[0..<k])
        let (pass, states) = try await g.decoder.prepare(prefix: try graphIDs(ids))
        return D1Prepared(state: state, ids: ids, k: k, stateTokens: stable, hidden: pass.hidden, callSeconds: pass.callSeconds,
                          seconds: d1Seconds(since: t0), states: states)
    }

    /// The response to {state: the prepared state, questions} (the questions object's JSON text), on the kept states.
    public func decide(prepared: D1Prepared, questionsJSON: Data) async throws -> JSONValue {
        try await trace(prepared: prepared, questions: try JSONParser.parse(questionsJSON)).response
    }

    /// The decision on a prepared state: every row's ids from k on, from a copy of the kept states (the shared path's
    /// second half); a row that does not start with the kept ids runs whole from zero states (counted).
    public func trace(prepared: D1Prepared, questions: JSONValue, groups: [String: [[Int]]]? = nil) async throws -> D1Trace {
        let g = try requireGraph()
        let t0 = ContinuousClock.now
        let p = try plan(request: try D1Request(json: .obj([("state", prepared.state), ("questions", questions)])), groups: groups)
        let t1 = ContinuousClock.now
        try g.decoder.setImageRows(nil)
        let k = prepared.k
        let tails: [[Int]?] = p.rows.map { Array($0.ids.prefix(k)) == prepared.ids ? Array($0.graphIDs[k...]) : nil }
        let t2 = ContinuousClock.now
        let passes = try await g.decoder.runPrepared(states: prepared.states, from: k, tails: tails, whole: p.rows.map(\.graphIDs))
        let hidden = zip(tails, passes).map { $0.0 != nil ? prepared.hidden + $0.1.hidden : $0.1.hidden }
        return try answer(plan: p, hidden: hidden, mode: "prepared", k: k, calls: passes.flatMap(\.callSeconds),
                          reset: passes.reduce(0) { $0 + $1.resetSeconds }, towerOutputs: [], towerSeconds: [],
                          rowsRunWhole: tails.filter { $0 == nil }.count, times: (t0, t1, t2, ContinuousClock.now))
    }

    /// The readout, the answers and the response from a decision's hidden rows (times: start, after the plan, after the
    /// images, after the graph).
    func answer(plan p: D1Plan, hidden: [[Float16]], mode: String, k: Int, calls: [Double], reset: Double,
                towerOutputs: [[Float]], towerSeconds: [Double], rowsRunWhole: Int,
                times: (ContinuousClock.Instant, ContinuousClock.Instant, ContinuousClock.Instant, ContinuousClock.Instant))
        throws -> D1Trace
    {
        let (t0, t1, t2, t3) = times
        let d = try requireGraph().metadata.hidden
        var logits: [[Int: Double]] = []
        var probs: [[Double]] = []
        for (r, h) in zip(p.rows, hidden) {
            guard h.count == r.ids.count * d else { throw D1Error.contract("\(h.count) hidden values for \(r.ids.count) x \(d)") }
            let out = try D1Readout.readout(hidden: h[(r.slot * d)..<((r.slot + 1) * d)].map { Double($0) }, table: table,
                                            groups: r.readGroups)
            logits.append(out.logits)
            probs.append(out.p)
        }
        let response = D1Readout.response(p.request.questions, probabilities: probs, inputTokens: p.inputTokens)
        let t4 = ContinuousClock.now
        func s(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
            let x = b - a
            return Double(x.components.seconds) + Double(x.components.attoseconds) * 1e-18
        }
        return D1Trace(plan: p, mode: mode, sharedK: k, hidden: hidden, logits: logits, probabilities: probs, response: response,
                       callSeconds: calls, resetSeconds: reset, towerOutputs: towerOutputs, towerSeconds: towerSeconds,
                       imageRows: p.imageTokens, rowsRunWhole: rowsRunWhole,
                       seconds: ["plan": s(t0, t1), "images": s(t1, t2), "graph": s(t2, t3), "readout": s(t3, t4), "wall": s(t0, t4)])
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
