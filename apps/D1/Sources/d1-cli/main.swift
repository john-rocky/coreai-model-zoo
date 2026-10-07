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

import CryptoKit
import D1
import Darwin
import Foundation

struct Args {
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
            if a == "--plain" {
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
    default: throw CLIError.usage("commands: render-test, rows, encode-test, image-plan, image-rows, readout-test, answers-test, bundle-check")
    }
} catch {
    FileHandle.standardError.write(Data("d1: \(error)\n".utf8))
    exit(1)
}
