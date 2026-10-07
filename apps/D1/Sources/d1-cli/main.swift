// d1 — the Mac CLI over the D1 library (round 3a: the text side, no graph).
//
//   d1 render-test --in values.json --tokenizer <dir> --out out.json
//       state blocks, the request checks, prefix / per-question suffix / codes / keys / legend, Python's repr / str /
//       round(x, 4) of doubles, json.dumps (indent 2, ensure_ascii off; and the defaults), str.strip / isalpha / repr
//   d1 rows --records records.json --tokenizer <dir> [--option-ids option_ids.json] --out out.json [--plain]
//       every question's row (validated one by one, as test_host.py does) and, when every question is accepted, the
//       request's path / trunk / branches / shared / input_tokens with the option-table check; per-row encode ms;
//       --plain: swift-transformers' own ids of every row beside the host's
//   d1 encode-test --texts texts.json --tokenizer <dir> --out out.json [--plain]
//       ids (and the Split pieces) of given texts; --plain: swift-transformers' own ids too
//   d1 image-plan --sizes sizes.json --out out.json
//       per (w, h): the plan of the picture as given and after cap_pixels (rows, cols, crops, grids, tokens, the run)
//   d1 image-rows --records image_records.json --tokenizer <dir> --out out.json
//       per request with pictures (w, h only): the processor's ids, extension ids, branches, input_tokens
//   d1 readout-test --in cases.json --head <dir> --hidden hidden.f32 --out out.json
//       z and p (bits) of each (hidden row, readout groups) on an option table read as a bundle reads it
//   d1 answers-test --in cases.json --out out.json
//       json.dumps(answers) and json.dumps(response, indent=2, ensure_ascii=False) from given probabilities
//   d1 bundle-check --bundle <dir> --out out.json
//       D1Decider's load contract on a bundle (metadata, tokenizer, head/option_rows), then `decide` (graph not wired)
//
// Round 3c: the graph (Decoder.swift, Tower.swift; the decoder asset --asset aot = <bundles>_aotc/<name>.h16c.aimodelc,
// jit = the bundle's .aimodel specialized here; a tower bundle with --tower, a crop's four inputs from --tower-inputs).
//   d1 decide --bundle <dir> [--asset aot|jit] [--tower <dir> --tower-inputs <dir>] --request req.json [--shared]
//          [--groups groups.json] --out resp.json [--trace trace.json] [--reps N] [--warm]
//       one request {state, questions[, images: [paths]]} -> the response (json.dumps indent 2); the trace: per row the
//       ids, the slot, the hidden rows' sha256, the logits and p (bits), each call's ms; --reps N decides it N more times
//   d1 fixture --bundle <dir> --records records.json [--groups groups.json] --arms direct[,shared] [--asset aot|jit]
//          [--tower <dir> --tower-inputs <dir> [--zero-image-control]] [--dump-hidden <dir>] [--warm] --out pass.json
//       every record from its raw request: direct (and shared) rows, hidden sha256, p bits, the response's bytes; a refused
//       request's text and each of its questions alone; the first record again at the end (the state reset). A record
//       may name pictures of the --tower-inputs manifest ("pictures": [ids]); --zero-image-control runs those again with
//       the image rows left zero. groups.json = {record id: {question: groups}} (a toy's readout groups)
//   d1 prepare-test --bundle <dir> --records records.json [--groups groups.json] [--asset aot|jit] --out out.json
//       every accepted request of 2+ questions: shared, then prepare(state) + decide(prepared) with all its questions and
//       with each alone: hidden rows, p and the response against shared's

import CryptoKit
import D1
import Darwin
import Foundation

struct Args {
    /// the options without a value
    static let flagNames: Set<String> = ["--plain", "--shared", "--warm", "--zero-image-control"]
    var command = ""
    var values: [String: String] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) throws {
        guard argv.count > 1 else { throw CLIError.usage("no command") }
        command = argv[1]
        var i = 2
        while i < argv.count {
            let a = argv[i]
            guard a.hasPrefix("--") else { throw CLIError.usage("unexpected \(a)") }
            if Self.flagNames.contains(a) {
                flags.insert(a)
                i += 1
                continue
            }
            guard i + 1 < argv.count else { throw CLIError.usage("\(a) needs a value") }
            values[a] = argv[i + 1]
            i += 2
        }
    }

    func one(_ k: String) -> String? { values[k] }
    func need(_ k: String) throws -> String {
        guard let v = values[k] else { throw CLIError.usage("missing \(k)") }
        return v
    }
}

enum CLIError: Error, CustomStringConvertible {
    case usage(String)
    case failed(String)
    var description: String {
        switch self {
        case .usage(let s): return "usage: \(s)"
        case .failed(let s): return s
        }
    }
}

func url(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL }
func readJSON(_ u: URL) throws -> JSONValue { try JSONParser.parse(Data(contentsOf: u)) }
func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

func seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

let binarySHA256: String? = Bundle.main.executableURL.flatMap { try? sha256(Data(contentsOf: $0)) }

func environment() -> JSONValue {
    #if DEBUG
    let config = "Debug"
    #else
    let config = "Release"
    #endif
    return .obj([("os", .string(ProcessInfo.processInfo.operatingSystemVersionString)),
                 ("os_build", .string(sysctlString("kern.osversion"))), ("chip", .string(sysctlString("machdep.cpu.brand_string"))),
                 ("build_configuration", .string(config)), ("argv", .strings(CommandLine.arguments)),
                 ("binary_sha256", .optional(binarySHA256))])
}

func log(_ s: String) {
    print(s)
    fflush(stdout)
}

func bits(_ x: Double) -> JSONValue { .string(String(x.bitPattern, radix: 16)) }

func errorText(_ e: Error) -> String { (e as? D1Error)?.message ?? "\(e)" }

func loadTokenizer(_ args: Args) async throws -> (D1Tokenizer, Double) {
    let t0 = ContinuousClock.now
    let tok = try await D1Tokenizer.load(folder: url(try args.need("--tokenizer")), plain: args.flags.contains("--plain"))
    return (tok, seconds(since: t0))
}

func tokenizerJSON(_ tok: D1Tokenizer, loadSeconds: Double) -> JSONValue {
    .obj([("class", .optional(tok.tokenizerClass)), ("tokenizer_json_sha256", .optional(tok.tokenizerJSONSHA256)),
          ("ignore_merges", .bool(tok.ignoreMerges)), ("added_tokens", .int(tok.addedIDs.count)),
          ("special_tokens_checked", .int(D1Tokenizer.specialTokens.count)), ("load_s", .double(loadSeconds)),
          ("plain_loaded", .bool(tok.plain != nil))])
}

func rowJSON(_ r: D1Row) -> JSONValue {
    .obj([("name", .string(r.name)), ("type", .string(r.kind.rawValue)), ("text", .string(r.text)), ("row_ids", .ints(r.ids)),
          ("row_len", .int(r.ids.count)), ("slot", .int(r.slot)), ("codes", r.codes.map { .strings($0) } ?? .null),
          ("alias_ids", r.aliasIDs.map { .ints($0) } ?? .null), ("groups", .array(r.groups.map { .ints($0) })),
          ("keys", .strings(r.keys))])
}

// MARK: - render-test

func renderTest(_ args: Args) async throws {
    let j = try readJSON(url(try args.need("--in")))
    let (tok, loadS) = try await loadTokenizer(args)
    func parsed(_ v: JSONValue) -> Result<JSONValue, Error> { Result { try JSONParser.parse(v.string ?? "") } }
    let states: [JSONValue] = (j["states"]?.array ?? []).map {
        switch parsed($0) {
        case .success(let s): return .string(D1Text.stateBlock(s))
        case .failure(let e): return .string("ERROR \(errorText(e))")
        }
    }
    let requests: [JSONValue] = (j["requests"]?.array ?? []).map { v in
        guard case .success(let req) = parsed(v) else { return .obj([("parse_error", .bool(true))]) }
        var out: [(String, JSONValue)] = []
        do {
            _ = try D1Request(json: req)
            out.append(("accept", .bool(true)))
        } catch {
            out += [("accept", .bool(false)), ("error", .string(errorText(error)))]
        }
        if let st = req["state"] { out.append(("prefix", .string(D1Text.prefix(st)))) }
        var qs: [JSONValue] = []
        for m in req["questions"]?.members ?? [] {
            do {
                let q = try D1Request.validateQuestion(name: m.key, m.value)
                let codes = q.kind == .choice ? try tok.aliases(q.labels).map(\.code) : nil
                qs.append(.obj([("name", .string(q.name)), ("type", .string(q.kind.rawValue)),
                                ("question_block", .string(D1Text.questionBlock(q, codes: codes))),
                                ("suffix", .string(D1Text.suffix(q, codes: codes))), ("codes", codes.map { .strings($0) } ?? .null),
                                ("keys", .strings(q.keys)), ("legend", q.kind == .score ? .strings(q.levels) : .null)]))
            } catch {
                qs.append(.obj([("name", .string(m.key)), ("error", .string(errorText(error)))]))
            }
        }
        out.append(("questions", .array(qs)))
        return .obj(out)
    }
    let doubles = (j["doubles"]?.array ?? []).map { $0.double ?? .nan }
    let dumpsIn = (j["dumps"]?.array ?? []).map(parsed)
    let strings = (j["strings"]?.array ?? []).compactMap(\.string)
    let labelLists = (j["label_lists"]?.array ?? []).map { ($0.array ?? []).compactMap(\.string) }
    func dumpsOut(_ f: (JSONValue) -> String) -> JSONValue {
        .array(dumpsIn.map { r in
            switch r {
            case .success(let v): return .string(f(v))
            case .failure(let e): return .string("ERROR \(errorText(e))")
            }
        })
    }
    let out: JSONValue = .obj([
        ("tokenizer", tokenizerJSON(tok, loadSeconds: loadS)),
        ("state_blocks", .array(states)), ("requests", .array(requests)),
        ("repr", .strings(doubles.map(PythonFormat.floatRepr))), ("str", .strings(doubles.map(PythonFormat.floatStr))),
        ("round4", .strings(doubles.map { PythonFormat.floatRepr(PythonFormat.pyRound($0, 4)) })),
        ("dumps_indent2", dumpsOut { PythonFormat.dumps($0, indent: 2, asciiOnly: false) }),
        ("dumps_default", dumpsOut { PythonFormat.dumps($0) }),
        ("repr_str", .strings(strings.map(PythonFormat.reprString))), ("strip", .strings(strings.map(PythonFormat.strip))),
        ("isalpha", .array(strings.map { .bool(PythonFormat.isAlpha($0)) })),
        ("option_codes", .array(labelLists.map { .strings(D1Text.optionCodes($0)) })),
        ("environment", environment()),
    ])
    try JSONWriter.write(out, to: url(try args.need("--out")), pretty: false)
    log("render-test: \(states.count) states, \(requests.count) requests, \(doubles.count) doubles, \(dumpsIn.count) dumps, "
        + "\(strings.count) strings, \(labelLists.count) label lists")
}

// MARK: - rows

func rowsCommand(_ args: Args) async throws {
    let (tok, loadS) = try await loadTokenizer(args)
    var table: D1OptionTable? = nil
    if let t = args.one("--option-ids") { table = try D1OptionTable.ids(json: url(t)) }
    let recordsURL = url(try args.need("--records"))
    guard let records = try readJSON(recordsURL)["records"]?.array else { throw CLIError.failed("no records") }
    var out: [JSONValue] = []
    var rowMs: [Double] = []
    let tAll = ContinuousClock.now
    for rec in records {
        let id = rec["id"]?.string ?? ""
        let req = rec["request"] ?? .null
        let state = req["state"] ?? .null
        var qs: [JSONValue] = []
        var refused: [(String, JSONValue)] = []
        for m in req["questions"]?.members ?? [] {
            do {
                let q = try D1Request.validateQuestion(name: m.key, m.value)
                let t = ContinuousClock.now
                let r = try tok.row(state: state, q)
                let ms = seconds(since: t) * 1e3
                rowMs.append(ms)
                var j = rowJSON(r)
                if case .object(var mm) = j {
                    mm.append(JSONMember("ms", .double(ms)))
                    if let plain = tok.plainTokens(r.text) { mm.append(JSONMember("plain_row_ids", .ints(plain))) }
                    j = .object(mm)
                }
                qs.append(j)
            } catch {
                refused.append((m.key, .string(errorText(error))))
            }
        }
        var j: [(String, JSONValue)] = [("id", .string(id)), ("questions", .array(qs)), ("refused", .obj(refused))]
        do {   // the whole request, as host.build_request: the checks, every row, the option table, the Tree
            let request = try D1Request(json: req)
            let rows = try tok.rows(request, table: table?.idSet)
            j += [("path", .string(rows.path)), ("input_tokens", .int(rows.inputTokens)), ("shared", .int(rows.shared))]
            if let trunk = rows.trunk, let branches = rows.branches, let eq = rows.equalsRow {
                j += [("trunk_ids", .ints(trunk)), ("branch_ids", .array(branches.map { .ints($0) })),
                      ("trunk_len", .int(trunk.count)), ("branch_lens", .ints(branches.map(\.count))),
                      ("equals_row", .array(eq.map { .bool($0) }))]
            }
        } catch {
            j.append(("request_error", .string(errorText(error))))
        }
        out.append(.obj(j))
    }
    let sorted = rowMs.sorted()
    func q(_ p: Double) -> Double { sorted.isEmpty ? .nan : sorted[min(sorted.count - 1, Int((p * Double(sorted.count - 1)).rounded()))] }
    let doc: JSONValue = .obj([("schema", .string("d1-swift-rows/1")), ("records_json", .string(recordsURL.path)),
                               ("tokenizer", tokenizerJSON(tok, loadSeconds: loadS)),
                               ("option_table", table.map { .obj([("ids", .int($0.ids.count))]) } ?? .null),
                               ("row_ms", .obj([("n", .int(rowMs.count)), ("p50", .double(q(0.5))), ("p99", .double(q(0.99))),
                                                ("max", .double(sorted.last ?? .nan)), ("total", .double(rowMs.reduce(0, +)))])),
                               ("rows_s_total", .double(seconds(since: tAll))), ("environment", environment()),
                               ("records", .array(out))])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("rows: \(out.count) records, \(rowMs.count) rows, encode ms p50 \(String(format: "%.3f", q(0.5))) max "
        + "\(String(format: "%.3f", sorted.last ?? .nan))")
}

// MARK: - encode-test

func encodeTest(_ args: Args) async throws {
    let (tok, loadS) = try await loadTokenizer(args)
    let texts = (try readJSON(url(try args.need("--texts")))["texts"]?.array ?? []).map { $0.string ?? "" }
    var ids: [JSONValue] = [], plain: [JSONValue] = [], pieces: [JSONValue] = [], ms: [Double] = []
    for t in texts {
        let t0 = ContinuousClock.now
        let e = tok.encode(t)
        ms.append(seconds(since: t0) * 1e3)
        ids.append(.ints(e))
        if let p = tok.plainTokens(t) { plain.append(.ints(p)) }
        pieces.append(.strings(tok.pieces(t)))
    }
    let doc: JSONValue = .obj([("tokenizer", tokenizerJSON(tok, loadSeconds: loadS)), ("ids", .array(ids)),
                               ("plain_ids", tok.plain == nil ? .null : .array(plain)), ("pieces", .array(pieces)),
                               ("ms", .array(ms.map { .double($0) })), ("environment", environment())])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("encode-test: \(texts.count) texts, total \(String(format: "%.1f", ms.reduce(0, +))) ms")
}

// MARK: - image-plan / image-rows

func planJSON(_ p: D1Vision.Plan) -> JSONValue {
    let run = D1Vision.imageTokens(p)
    return .obj([("size", .ints([p.width, p.height])), ("rows", .int(p.rows)), ("cols", .int(p.cols)),
                 ("n_crops", .int(p.crops.count)),
                 ("crops", .array(p.crops.map { c in
                     .obj([("kind", .string(c.kind.rawValue)), ("grid", .ints([c.grid.h, c.grid.w])), ("tokens", .int(c.tokens)),
                           ("row", .int(c.row)), ("col", .int(c.col))])
                 })),
                 ("tokens", .int(p.tokens)), ("max_patches", .int(p.crops.map { $0.grid.h * $0.grid.w }.max() ?? 0)),
                 ("run_len", .int(run.count)), ("run_head", .ints(Array(run.prefix(3)))), ("run_tail", .ints(Array(run.suffix(3)))),
                 ("run_sha256", .string(sha256(Data(run.map(String.init).joined(separator: ",").utf8))))])
}

func imagePlan(_ args: Args) throws {
    let sizes = (try readJSON(url(try args.need("--sizes")))["sizes"]?.array ?? []).map { ($0.array ?? []).compactMap(\.intValue) }
    let out: [JSONValue] = sizes.map { s in
        let (w, h) = (s[0], s[1])
        let c = D1Vision.capSize(width: w, height: h)
        return .obj([("w", .int(w)), ("h", .int(h)), ("direct", planJSON(D1Vision.plan(height: h, width: w))),
                     ("capped", planJSON(D1Vision.plan(height: c.height, width: c.width)))])
    }
    let doc: JSONValue = .obj([("target_ratios", .array(D1Vision.targetRatios.map { .ints([$0.cols, $0.rows]) })),
                               ("rows", .array(out)), ("environment", environment())])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("image-plan: \(out.count) sizes")
}

func imageRows(_ args: Args) async throws {
    let (tok, loadS) = try await loadTokenizer(args)
    let recs = try readJSON(url(try args.need("--records")))["records"]?.array ?? []
    var out: [JSONValue] = []
    for r in recs {
        let id = r["id"]?.string ?? ""
        let pics = (r["pictures"]?.array ?? []).map { p -> (width: Int, height: Int) in
            let a = (p.array ?? []).compactMap(\.intValue)
            return (a[0], a[1])
        }
        do {
            let request = try D1Request(json: .obj([("state", r["state"] ?? .null), ("questions", r["questions"] ?? .null)]))
            let res = try D1Vision.requestIDs(tok, request, pictures: pics)
            out.append(.obj([("id", .string(id)), ("text", .string(res.text)), ("ids", .ints(res.ids)),
                             ("extension_ids", .ints(D1Vision.extensionIDs(res.ids))),
                             ("branch_ids", .array(res.branches.map { .ints($0) })), ("input_tokens", .int(res.inputTokens)),
                             ("n_image_tokens", .int(D1Vision.imageTokenCount(res.plans))),
                             ("plans", .array(res.plans.map(planJSON)))]))
        } catch {
            out.append(.obj([("id", .string(id)), ("error", .string(errorText(error)))]))
        }
    }
    let doc: JSONValue = .obj([("tokenizer", tokenizerJSON(tok, loadSeconds: loadS)), ("records", .array(out)),
                               ("environment", environment())])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("image-rows: \(out.count) records")
}

// MARK: - readout-test / answers-test

func readoutTest(_ args: Args) throws {
    let j = try readJSON(url(try args.need("--in")))
    let table = try D1Decider.loadTable(url(try args.need("--head")), count: j["table_n"]?.intValue ?? -1)
    let raw = try Data(contentsOf: url(try args.need("--hidden")))
    let floats: [Float] = raw.withUnsafeBytes { b in (0..<(raw.count / 4)).map { Float(bitPattern: UInt32(littleEndian: b.load(fromByteOffset: $0 * 4, as: UInt32.self))) } }
    let d = table.hidden
    var out: [JSONValue] = []
    let t0 = ContinuousClock.now
    for c in j["cases"]?.array ?? [] {
        let k = c["hidden_index"]?.intValue ?? 0
        let h = floats[(k * d)..<((k + 1) * d)].map(Double.init)
        let groups = (c["groups"]?.array ?? []).map { ($0.array ?? []).compactMap(\.intValue) }
        let r = try D1Readout.readout(hidden: h, table: table, groups: groups)
        let ids = D1Readout.groupIDs(groups)
        out.append(.obj([("ids", .ints(ids)), ("logit_bits", .array(ids.map { bits(r.logits[$0]!) })),
                         ("p_bits", .array(r.p.map(bits))), ("p", .strings(r.p.map(PythonFormat.floatRepr)))]))
    }
    let doc: JSONValue = .obj([("table", .obj([("n", .int(table.ids.count)), ("hidden", .int(d))])), ("cases", .array(out)),
                               ("seconds", .double(seconds(since: t0))), ("environment", environment())])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("readout-test: \(out.count) cases")
}

func answersTest(_ args: Args) throws {
    let j = try readJSON(url(try args.need("--in")))
    var out: [JSONValue] = []
    for c in j["cases"]?.array ?? [] {
        do {
            let request = try D1Request(json: try JSONParser.parse(c["request"]?.string ?? ""))
            let probs = (c["probs"]?.array ?? []).map { ($0.array ?? []).map { $0.double ?? .nan } }
            let resp = D1Readout.response(request.questions, probabilities: probs, inputTokens: c["input_tokens"]?.intValue ?? 0)
            out.append(.obj([("answers_dumps", .string(PythonFormat.dumps(resp["answers"]!))),
                             ("response_dumps_indent2", .string(PythonFormat.dumps(resp, indent: 2, asciiOnly: false)))]))
        } catch {
            out.append(.obj([("error", .string(errorText(error)))]))
        }
    }
    try JSONWriter.write(.obj([("cases", .array(out)), ("environment", environment())]), to: url(try args.need("--out")), pretty: false)
    log("answers-test: \(out.count) cases")
}

// MARK: - bundle-check

func bundleCheck(_ args: Args) async throws {
    let bundle = url(try args.need("--bundle"))
    var out: [(String, JSONValue)] = [("bundle", .string(bundle.path))]
    let t0 = ContinuousClock.now
    let d1 = try await D1Decider(bundle: bundle)
    let m = d1.metadata
    out += [("load_s", .double(seconds(since: t0))),
            ("metadata", .obj([("name", .string(m.name)), ("asset", .string(m.asset)), ("vocab", .int(m.vocab)),
                               ("max_context", .int(m.maxContext)), ("chunk", .int(m.chunk)), ("special", .int(m.special.count)),
                               ("option_n", .int(m.optionCount))])),
            ("table", .obj([("n", .int(d1.table.ids.count)), ("hidden", .int(d1.table.hidden)),
                            ("ids_sha256", .string(sha256(Data(d1.table.ids.map(String.init).joined(separator: ",").utf8))))])),
            ("tokenizer", tokenizerJSON(d1.tokenizer, loadSeconds: 0))]
    // a noul request (its yes / no ids must be in the bundle's option table) and a score request (digits)
    for (name, text) in [("noul", #"{"state": "s", "questions": {"a": {"type": "noul", "instructions": "Is it?"}}}"#),
                         ("score", #"{"state": "s", "questions": {"a": {"type": "score", "instructions": "How?", "criteria": ["low", "high"]}}}"#)] {
        do {
            _ = try await d1.decide(requestJSON: Data(text.utf8))
            out.append(("decide_\(name)", .string("returned")))
        } catch {
            out.append(("decide_\(name)", .string(errorText(error))))
        }
    }
    try JSONWriter.write(.obj(out + [("environment", environment())]), to: url(try args.need("--out")), pretty: false)
    log("bundle-check: \(m.name), table \(d1.table.ids.count) x \(d1.table.hidden)")
}

// MARK: - decide / fixture / prepare-test (round 3c: the graph)

func hex(_ x: Double) -> JSONValue { .string(String(x.bitPattern, radix: 16)) }

/// Two fp16 arrays with the same bytes (bit for bit: -0 and 0 differ, a NaN equals itself).
func bitsEqual(_ a: [Float16], _ b: [Float16]) -> Bool {
    a.count == b.count && a.withUnsafeBytes { x in b.withUnsafeBytes { y in x.elementsEqual(y) } }
}

/// The runtime's cache directory of this process (`~/Library/Caches/coreai-cache/<OS build>/d1`): each entry and its bytes.
func cacheListing() -> JSONValue {
    let dir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Caches/coreai-cache/\(sysctlString("kern.osversion"))/d1")
    let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path))?.sorted() ?? []
    return .obj([("dir", .string(dir.path)), ("entries", .array(names.map { n in
        var bytes = 0
        if let e = FileManager.default.enumerator(at: dir.appendingPathComponent(n), includingPropertiesForKeys: [.fileSizeKey]) {
            for case let f as URL in e { bytes += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        }
        return .obj([("name", .string(n)), ("bytes", .int(bytes))])
    }))])
}

func assetKind(_ args: Args) throws -> D1Graph.Asset {
    guard let a = D1Graph.Asset(rawValue: args.one("--asset") ?? "aot") else { throw CLIError.usage("--asset aot|jit") }
    return a
}

/// The decider with its graph, and what the load did.
func loadGraphDecider(_ args: Args) async throws -> (D1Decider, D1Graph, JSONValue) {
    let cacheBefore = cacheListing()
    let t0 = ContinuousClock.now
    let d1 = try await D1Decider(bundle: url(try args.need("--bundle")))
    let tText = seconds(since: t0)
    let g = try await d1.loadGraph(asset: try assetKind(args), tower: args.one("--tower").map(url),
                                   warmUp: args.flags.contains("--warm"))
    let m = g.metadata
    let info: JSONValue = .obj([
        ("bundle", .string(d1.bundle.path)), ("name", .string(d1.metadata.name)), ("asset", .string(g.asset.rawValue)),
        ("asset_path", .string(g.assetURL.path)), ("options", .string(d1Describe(g.decoder.options))),
        ("text_load_s", .double(tText)), ("graph_load_s", .double(g.loadSeconds)),
        ("decoder_model_s", .double(g.decoder.loadSeconds.model)), ("decoder_function_s", .double(g.decoder.loadSeconds.function)),
        ("state_allocation_s", .double(g.decoder.allocationSeconds)), ("warm_up_s", g.warmUpSeconds.map { .double($0) } ?? .null),
        ("functions", .strings(g.decoder.functionNames)), ("descriptor", g.decoder.descriptor),
        ("chunk", .int(m.chunk)), ("max_context", .int(m.maxContext)), ("hidden", .int(m.hidden)), ("image_rows", .int(m.imageRows)),
        ("vocab", .int(m.vocab)), ("pad_id", .int(m.padID)), ("toy_fold", m.toyFold.map { .int($0) } ?? .null),
        ("tower", g.tower.map { t in .obj([("bundle", .string(t.bundle.path)), ("asset_path", .string(t.url.path)),
                                           ("options", .string(d1Describe(t.options))), ("load_s", .double(t.loadSeconds)),
                                           ("descriptor", t.descriptor)]) } ?? .null),
        ("coreai_cache_before_load", cacheBefore), ("coreai_cache_after_load", cacheListing()),
    ])
    return (d1, g, info)
}

/// {question: groups} of a JSON object (a toy's readout groups).
func groupsMap(_ v: JSONValue?) -> [String: [[Int]]]? {
    guard let m = v?.members else { return nil }
    return Dictionary(uniqueKeysWithValues: m.map { ($0.key, ($0.value.array ?? []).map { ($0.array ?? []).compactMap(\.intValue) }) })
}

/// One row of a trace as the gate reads it (decide.py `row_record`'s fields).
func graphRowJSON(_ r: D1GraphRow, _ h: [Float16], _ p: [Double], _ z: [Int: Double], d: Int) -> [(String, JSONValue)] {
    let gids = D1Readout.groupIDs(r.readGroups)
    return [("name", .string(r.row.name)), ("type", .string(r.row.kind.rawValue)), ("T", .int(r.ids.count)), ("slot", .int(r.slot)),
            ("row_ids_sha256", .string(sha256(Data(PythonFormat.dumps(.ints(r.ids)).utf8)))),
            ("graph_ids_sha256", .string(sha256(Data(PythonFormat.dumps(.ints(r.graphIDs)).utf8)))),
            ("read_groups", .array(r.readGroups.map { .ints($0) })), ("hidden_sha256", .string(D1Decoder.sha256(h))),
            ("slot_hidden_sha256", .string(D1Decoder.sha256(Array(h[(r.slot * d)..<((r.slot + 1) * d)])))),
            ("finite", .bool(h.allSatisfy(\.isFinite))), ("all_zero", .bool(h.allSatisfy { $0 == 0 })),
            ("p", .array(p.map { .double($0) })), ("p_bits", .array(p.map(hex))),
            ("logit_ids", .ints(gids)), ("logit_bits", .array(gids.map { hex(z[$0]!) }))]
}

func traceRowsJSON(_ t: D1Trace, d: Int) -> JSONValue {
    .array(zip(t.plan.rows.indices, t.plan.rows).map { k, r in
        .obj(graphRowJSON(r, t.hidden[k], t.probabilities[k], t.logits[k], d: d))
    })
}

func traceTimesJSON(_ t: D1Trace) -> JSONValue {
    .obj([("mode", .string(t.mode)), ("shared_k", .int(t.sharedK)), ("calls", .int(t.callSeconds.count)),
          ("call_ms", .array(t.callSeconds.map { .double($0 * 1e3) })), ("reset_s", .double(t.resetSeconds)),
          ("seconds", .obj(t.seconds.sorted { $0.key < $1.key }.map { ($0.key, JSONValue.double($0.value)) })),
          ("tower_ms", .array(t.towerSeconds.map { .double($0 * 1e3) })), ("rows_run_whole", .int(t.rowsRunWhole))])
}

func writeHidden(_ h: [Float16], to u: URL) throws {
    try h.withUnsafeBytes { Data($0) }.write(to: u)
}

/// A request object without its `images` member, and the pictures it names (the files' stems = the manifest's ids).
func splitImages(_ j: JSONValue) -> (request: JSONValue, pictures: [String]?) {
    guard let m = j.members, let imgs = m.first(where: { $0.key == "images" })?.value.array else { return (j, nil) }
    let ids = imgs.compactMap(\.string).map { (($0 as NSString).lastPathComponent as NSString).deletingPathExtension }
    return (.object(m.filter { $0.key != "images" }), ids)
}

func decideCommand(_ args: Args) async throws {
    let (d1, g, info) = try await loadGraphDecider(args)
    let (reqJSON, picIDs) = splitImages(try readJSON(url(try args.need("--request"))))
    let pictures = try picIDs.map { ids -> D1TowerInputs in
        guard let dir = args.one("--tower-inputs") else { throw CLIError.usage("a request with images needs --tower-inputs") }
        return try D1TowerInputs.read(url(dir), ids: ids)
    }
    let groups = try args.one("--groups").map { groupsMap(try readJSON(url($0))) } ?? nil
    let mode: D1Decider.Mode = args.flags.contains("--shared") ? .shared : .direct
    var body: JSONValue
    var trace: JSONValue = .null
    var reps: [JSONValue] = []
    do {
        let request = try D1Request(json: reqJSON)
        let t = try await d1.trace(request: request, mode: mode, groups: groups, pictures: pictures)
        body = t.response
        trace = .obj([("rows", traceRowsJSON(t, d: g.metadata.hidden)), ("times", traceTimesJSON(t)),
                      ("input_tokens", .int(t.plan.inputTokens)), ("state_tokens", .int(t.plan.stateTokens)),
                      ("image_rows", .int(t.imageRows))])
        for _ in 0..<(Int(args.one("--reps") ?? "0") ?? 0) {
            reps.append(traceTimesJSON(try await d1.trace(request: request, mode: mode, groups: groups, pictures: pictures)))
        }
    } catch let e as D1Error {
        switch e {
        case .request, .graphLimit, .json: body = .obj([("error", .string(e.message))])
        default: throw e
        }
    }
    try Data(PythonFormat.dumps(body, indent: 2, asciiOnly: false).utf8).write(to: url(try args.need("--out")))
    if let tp = args.one("--trace") {
        try JSONWriter.write(.obj([("graph", info), ("trace", trace), ("reps", .array(reps)), ("environment", environment())]),
                             to: url(tp))
    }
    log(PythonFormat.dumps(body, asciiOnly: false))
}

/// One fixture record: direct (and shared) on the request; a refused request's text and its questions alone.
func fixtureRecord(_ d1: D1Decider, _ rec: JSONValue, groups: [String: [[Int]]]?, towerInputs: URL?, arms: Set<String>,
                   zeroControl: Bool, dump: URL?) async throws -> (json: JSONValue, hidden: [[Float16]])
{
    let id = rec["id"]?.string ?? ""
    let reqJSON = rec["request"] ?? .null
    let d = try d1.requireGraph().metadata.hidden
    var j: [(String, JSONValue)] = [("id", .string(id))]
    var pictures: D1TowerInputs? = nil
    if let ids = rec["pictures"]?.array?.compactMap(\.string) {
        guard let dir = towerInputs else { throw CLIError.usage("record \(id) names pictures: --tower-inputs") }
        pictures = try D1TowerInputs.read(dir, ids: ids)
    }
    do {
        let request = try D1Request(json: reqJSON)
        let t = try await d1.trace(request: request, mode: .direct, groups: groups, pictures: pictures)
        j += [("accepted", .bool(true)), ("rows", traceRowsJSON(t, d: d)), ("input_tokens", .int(t.plan.inputTokens)),
              ("state_tokens", .int(t.plan.stateTokens)), ("trunk_len", .int(t.plan.trunkLength)), ("image_rows", .int(t.imageRows)),
              ("response_indent2", .string(PythonFormat.dumps(t.response, indent: 2, asciiOnly: false))),
              ("answers_dumps", .string(PythonFormat.dumps(t.response["answers"]!))), ("direct", traceTimesJSON(t))]
        if !t.towerOutputs.isEmpty {
            j.append(("tower_outputs_sha256", .strings(t.towerOutputs.map { o in
                o.withUnsafeBytes { sha256(Data($0)) }
            })))
        }
        if let dump {
            for (k, h) in t.hidden.enumerated() { try writeHidden(h, to: dump.appendingPathComponent("\(id)__\(k).f16")) }
        }
        if arms.contains("shared") {
            let s = try await d1.trace(request: request, mode: .shared, groups: groups, pictures: pictures)
            j.append(("shared", .obj([
                ("k", .int(s.sharedK)), ("times", traceTimesJSON(s)),
                ("hidden_bit_equal_direct", .array(zip(s.hidden, t.hidden).map { .bool(bitsEqual($0.0, $0.1)) })),
                ("hidden_sha256", .strings(s.hidden.map(D1Decoder.sha256))),
                ("p_bits", .array(s.probabilities.map { .array($0.map(hex)) })),
                ("response_indent2_equal_direct", .bool(PythonFormat.dumps(s.response, indent: 2, asciiOnly: false)
                    == PythonFormat.dumps(t.response, indent: 2, asciiOnly: false))),
            ])))
        }
        if zeroControl, pictures != nil {
            let z = try await d1.trace(request: request, mode: .direct, groups: groups, pictures: pictures, zeroImages: true)
            j.append(("zero_images", .obj([("hidden_sha256", .strings(z.hidden.map(D1Decoder.sha256)))])))
            if let dump {
                for (k, h) in z.hidden.enumerated() { try writeHidden(h, to: dump.appendingPathComponent("\(id)__\(k)__zero.f16")) }
            }
        }
        return (.obj(j), t.hidden)
    } catch let e as D1Error {
        switch e {
        case .request, .graphLimit, .json: break
        default: throw e
        }
        j += [("accepted", .bool(false)), ("error", .string(e.message))]
        var sub: [JSONValue] = []
        var hidden: [[Float16]] = []
        for m in reqJSON["questions"]?.members ?? [] {
            guard (try? D1Request.validateQuestion(name: m.key, m.value)) != nil else { continue }
            let one = try D1Request(json: .obj([("state", reqJSON["state"] ?? .null), ("questions", .object([m]))]))
            let t = try await d1.trace(request: one, mode: .direct, groups: groups)
            sub.append(.obj(graphRowJSON(t.plan.rows[0], t.hidden[0], t.probabilities[0], t.logits[0], d: d)
                            + [("direct", traceTimesJSON(t))]))
            hidden.append(t.hidden[0])
            if let dump { try writeHidden(t.hidden[0], to: dump.appendingPathComponent("\(id)__\(m.key).f16")) }
        }
        j.append(("sub", .array(sub)))
        return (.obj(j), hidden)
    }
}

func fixtureCommand(_ args: Args) async throws {
    let arms = Set((args.one("--arms") ?? "direct").split(separator: ",").map(String.init))
    let records = try readJSON(url(try args.need("--records")))["records"]?.array ?? []
    guard !records.isEmpty else { throw CLIError.failed("no records") }
    let groupsAll = try args.one("--groups").map { try readJSON(url($0)) }
    let dump = args.one("--dump-hidden").map(url)
    if let dump { try FileManager.default.createDirectory(at: dump, withIntermediateDirectories: true) }
    let (d1, _, info) = try await loadGraphDecider(args)
    var out: [JSONValue] = []
    var first: [[Float16]] = []
    var resetCheck: JSONValue = .null
    let tAll = ContinuousClock.now
    for (i, rec) in (records + [records[0]]).enumerated() {
        let id = rec["id"]?.string ?? ""
        let again = i == records.count
        let t0 = ContinuousClock.now
        let (j, h) = try await fixtureRecord(d1, rec, groups: groupsMap(groupsAll?[id]), towerInputs: args.one("--tower-inputs").map(url),
                                             arms: again ? ["direct"] : arms, zeroControl: !again && args.flags.contains("--zero-image-control"),
                                             dump: again ? nil : dump)
        if again {
            let same = h.count == first.count && zip(h, first).allSatisfy { bitsEqual($0.0, $0.1) }
            resetCheck = .obj([("record", .string(id)), ("bit_equal", .bool(same)), ("rows", .int(h.count))])
            log("reset re-run \(id): bit-equal \(same)")
            continue
        }
        if i == 0 { first = h }
        var jj = j
        if case .object(var m) = jj {
            m.append(JSONMember("wall_s", .double(seconds(since: t0))))
            jj = .object(m)
        }
        out.append(jj)
        if i % 40 == 0 { log("  \(i + 1)/\(records.count) \(id)") }
    }
    let doc: JSONValue = .obj([("schema", .string("d1-swift-fixture/1")), ("records_json", .string(try args.need("--records"))),
                               ("arms", .strings(arms.sorted())), ("graph", info), ("reset_check", resetCheck),
                               ("records", .array(out)), ("runs_wall_s", .double(seconds(since: tAll))),
                               ("coreai_cache_end", cacheListing()), ("environment", environment())])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("fixture: \(out.count) records, arms \(arms.sorted()), \(String(format: "%.1f", seconds(since: tAll))) s")
}

func prepareTest(_ args: Args) async throws {
    let records = try readJSON(url(try args.need("--records")))["records"]?.array ?? []
    let groupsAll = try args.one("--groups").map { try readJSON(url($0)) }
    let (d1, _, info) = try await loadGraphDecider(args)
    var out: [JSONValue] = []
    for rec in records {
        let id = rec["id"]?.string ?? ""
        let reqJSON = rec["request"] ?? .null
        guard let request = try? D1Request(json: reqJSON), request.questions.count > 1 else { continue }
        let groups = groupsMap(groupsAll?[id])
        let s: D1Trace
        do {
            s = try await d1.trace(request: request, mode: .shared, groups: groups)
        } catch let e as D1Error {
            out.append(.obj([("id", .string(id)), ("refused", .string(e.message))]))
            continue
        }
        let pr = try await d1.prepare(state: request.state)
        let p = try await d1.trace(prepared: pr, questions: reqJSON["questions"]!, groups: groups)
        var singles: [Bool] = []
        for (k, m) in (reqJSON["questions"]?.members ?? []).enumerated() {
            let one = try await d1.trace(prepared: pr, questions: .object([m]), groups: groups)
            singles.append(bitsEqual(one.hidden[0], s.hidden[k]))
        }
        out.append(.obj([
            ("id", .string(id)), ("rows", .int(s.hidden.count)), ("state_tokens", .int(s.plan.stateTokens)),
            ("k_shared", .int(s.sharedK)), ("k_prepared", .int(pr.k)), ("prepare_calls", .int(pr.callSeconds.count)),
            ("prepare_s", .double(pr.seconds)), ("prepared", traceTimesJSON(p)), ("shared", traceTimesJSON(s)),
            ("hidden_bit_equal_shared", .array(zip(p.hidden, s.hidden).map { .bool(bitsEqual($0.0, $0.1)) })),
            ("p_bit_equal_shared", .array(zip(p.probabilities, s.probabilities).map { .bool($0.0.map(\.bitPattern) == $0.1.map(\.bitPattern)) })),
            ("single_hidden_bit_equal_shared", .array(singles.map { .bool($0) })),
            ("response_indent2_equal_shared", .bool(PythonFormat.dumps(p.response, indent: 2, asciiOnly: false)
                == PythonFormat.dumps(s.response, indent: 2, asciiOnly: false))),
            ("hidden_sha256", .strings(p.hidden.map(D1Decoder.sha256))),
        ]))
        log("  \(id): prepared = shared \(zip(p.hidden, s.hidden).allSatisfy { bitsEqual($0.0, $0.1) }), singles \(singles)")
    }
    try JSONWriter.write(.obj([("schema", .string("d1-swift-prepare-test/1")), ("graph", info), ("records", .array(out)),
                               ("environment", environment())]), to: url(try args.need("--out")), pretty: false)
    log("prepare-test: \(out.count) records")
}

// MARK: - main

do {
    let args = try Args(CommandLine.arguments)
    switch args.command {
    case "render-test": try await renderTest(args)
    case "rows": try await rowsCommand(args)
    case "encode-test": try await encodeTest(args)
    case "image-plan": try imagePlan(args)
    case "image-rows": try await imageRows(args)
    case "readout-test": try readoutTest(args)
    case "answers-test": try answersTest(args)
    case "bundle-check": try await bundleCheck(args)
    case "decide": try await decideCommand(args)
    case "fixture": try await fixtureCommand(args)
    case "prepare-test": try await prepareTest(args)
    default: throw CLIError.usage("commands: render-test, rows, encode-test, image-plan, image-rows, readout-test, answers-test, "
        + "bundle-check, decide, fixture, prepare-test")
    }
} catch {
    FileHandle.standardError.write(Data("d1: \(error)\n".utf8))
    exit(1)
}
