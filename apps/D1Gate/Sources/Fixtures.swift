// Fixtures — what the gate reads from the assets directory's fixtures/ (../_stage.sh writes it from the lane's working
// directory ~/code/coreai/_d1_3b) and the per-row verdict against the two references. Every file is read with the D1
// library's JSONParser (number literals kept; the probabilities parse exactly):
//   requests.json     the records the gate answers, in order, each {id, set ("fixture" | "image"), source, request,
//                     images?}: the round-1 fixture (361 text records, 393 questions the provider answers; own_email_03
//                     is refused by the host whole and its two valid questions run alone), then the 12 picture records
//                     (24 questions; `images` are files under fixtures/, card_cats' URL is left out = skipped)
//   oracle_slim.json  per record the provider's fp32 oracle (L/oracle/records_oracle.json, records_oracle_images.json):
//                     per question name, type, keys, row_ids (the processor's: <image> = 124907), groups, probs (the
//                     question's own row), api_probs (the API path: the Tree for several questions), argmax, top2_margin,
//                     near_tie; and the names the provider refused
//   mac_ref.json      per record, per question, the Mac's read-out of the same decoder asset (round 5c's readout gate,
//                     AOT h16c, int8mlp pf16; the picture records: round 8's decide.py run with the tower fp16w32 AOT):
//                     the sha256 of the hidden rows (fp16 [T, d] as stored) and p as float64 bit patterns (hex)
//   red_arms.json     round 4's five red arms (fixtures/red_arms_r4.json): per arm its perturbed requests, each with
//                     its base record, the compared questions and the provider's probabilities of the perturbed row
//   bench.json        the timed items (round 7's: one_question, three_shared, three_direct, state_3_4k, image_384px) and
//                     the AOT asset's subset
//
// The bar (FACTS §7, readout_gate.py BAR, unchanged): argmax = the oracle's on every question whose oracle top-2 margin
// is above 0.02 (the near-ties counted apart); max |dp| <= 0.02 over every option of every question; the mean over runs
// (one run = one question) of the run's mean |dp| <= 0.002; and, as gate_swift.py asks, every row's ids (the picture
// slots mapped back to <image>), readout groups and keys equal the oracle's, every hidden value finite, no row all zero.
// A run is scored on the float64 p the library returns.

import D1
import Foundation

struct Fixtures {
    struct Record: Sendable {
        let id: String
        /// "fixture" (text) or "image"
        let set: String
        let source: String
        /// the request as written (number literals kept)
        let json: JSONValue
        /// picture files, relative to fixtures/ (image records)
        let images: [String]

        var questionNames: [String] { json["questions"]?.members?.map(\.key) ?? [] }
    }

    struct OracleQuestion: Sendable {
        let name: String
        let type: String
        let keys: [String]
        let rowIDs: [Int]
        let groups: [[Int]]
        let probs: [Double]
        let apiProbs: [Double]?
        let argmax: String
        let top2Margin: Double
        let nearTie: Bool
    }

    struct OracleRecord: Sendable {
        let set: String
        let questions: [OracleQuestion]
        let refused: [String]

        func question(_ name: String) -> OracleQuestion? { questions.first { $0.name == name } }
    }

    struct MacRow: Sendable {
        let pBits: [String]
        let hiddenSHA: String
    }

    struct BenchItem: Sendable {
        let name: String
        /// "text" or "image"
        let kind: String
        let record: String
        /// the questions kept (request order); nil = all
        let questions: [String]?
        let shared: Bool
        let reps: Int
        /// seconds of rest between two timed decisions (an item whose decision runs past ~20 s on the 18 Pro's GPU)
        let repRest: Double
    }

    struct RedRequest: Sendable {
        let name: String
        let json: JSONValue
        /// the fixture record the perturbed rows are compared with (the same question names)
        let base: String
        /// the compared questions (nil = every question of the request)
        let only: [String]?
        /// the provider's probabilities of the perturbed rows, by question
        let oracle: [String: [Double]]
    }

    struct RedArm: Sendable {
        let id: String
        let kind: String
        let requests: [RedRequest]
    }

    /// One question's run, scored.
    struct RowScore: Sendable {
        /// "<record>:<question>"
        let key: String
        let set: String
        let idsEqualOracle: Bool
        let argmax: Int
        let argmaxOracle: Int
        let nearTie: Bool
        let deltas: [Double]
        let apiDeltas: [Double]?
        let finite: Bool
        let allZero: Bool
        /// float64 bit patterns, hex (Python's format(bits, "x"))
        let pBits: [String]
        let hiddenSHA: String
        let macHiddenEqual: Bool?
        let macPBitEqual: Bool?
        let macMaxAbsDp: Double?
        let macArgmaxEqual: Bool?

        var argmaxEqual: Bool { argmax == argmaxOracle }
        var maxAbsDp: Double { deltas.max() ?? .nan }
        var meanAbsDp: Double { deltas.isEmpty ? .nan : deltas.reduce(0, +) / Double(deltas.count) }
        var apiMaxAbsDp: Double? { apiDeltas?.max() }
    }

    let root: URL
    let records: [Record]
    let byID: [String: Record]
    let oracle: [String: OracleRecord]
    let mac: [String: [String: MacRow]]
    let red: [RedArm]
    let bench: [BenchItem]
    let benchAOT: [BenchItem]
    let skipped: [String]
    let files: [String: Any]

    init(root: URL, oracleOverride: URL?) throws {
        self.root = root
        func load(_ url: URL) throws -> (Data, JSONValue) {
            do {
                let d = try Data(contentsOf: url)
                return (d, try JSONParser.parse(d))
            } catch {
                throw GateError.fixture("\(url.lastPathComponent): \(error)")
            }
        }
        var f: [String: Any] = [:]
        func note(_ key: String, _ url: URL, _ d: Data, _ extra: [String: Any]) {
            f[key] = ["path": url.path, "bytes": d.count, "sha256": sha256Hex(d)].merging(extra) { a, _ in a }
        }

        // requests.json
        let reqURL = root.appendingPathComponent("requests.json")
        let (reqData, reqDoc) = try load(reqURL)
        guard let list = reqDoc["records"]?.array else { throw GateError.fixture("requests.json: no records") }
        var recs: [Record] = []
        for (i, r) in list.enumerated() {
            guard let id = r["id"]?.string, let set = r["set"]?.string, let req = r["request"], req.members != nil else {
                throw GateError.fixture("requests.json record \(i): no id / set / request object")
            }
            let images = try Self.strings(r["images"] ?? .array([]), "requests.json \(id) images")
            recs.append(Record(id: id, set: set, source: r["source"]?.string ?? "", json: req, images: images))
        }
        records = recs
        byID = Dictionary(uniqueKeysWithValues: recs.map { ($0.id, $0) })
        skipped = (reqDoc["skipped"]?.array ?? []).compactMap { $0["id"]?.string }
        note("requests_json", reqURL, reqData, ["records": recs.count, "skipped": skipped])

        // oracle_slim.json
        let oracleURL = oracleOverride ?? root.appendingPathComponent("oracle_slim.json")
        let (oracleData, oracleDoc) = try load(oracleURL)
        guard let om = oracleDoc["records"]?.members else { throw GateError.fixture("oracle_slim.json: no records") }
        var orc: [String: OracleRecord] = [:]
        for m in om {
            var qs: [OracleQuestion] = []
            for q in m.value["questions"]?.array ?? [] {
                let what = "oracle_slim.json \(m.key)"
                guard let name = q["name"]?.string, let type = q["type"]?.string, let argmax = q["argmax"]?.string,
                      let margin = q["top2_margin"]?.double, let near = q["near_tie"]?.boolValue
                else { throw GateError.fixture("\(what): a question without name / type / argmax / top2_margin / near_tie") }
                var api: [Double]? = nil
                if let v = q["api_probs"], !v.isNull { api = try Self.doubles(v, "\(what) \(name) api_probs") }
                qs.append(OracleQuestion(name: name, type: type, keys: try Self.strings(q["keys"], "\(what) \(name) keys"),
                                         rowIDs: try Self.ints(q["row_ids"], "\(what) \(name) row_ids"),
                                         groups: try (q["groups"]?.array ?? []).map { try Self.ints($0, "\(what) \(name) groups") },
                                         probs: try Self.doubles(q["probs"], "\(what) \(name) probs"), apiProbs: api,
                                         argmax: argmax, top2Margin: margin, nearTie: near))
            }
            orc[m.key] = OracleRecord(set: m.value["set"]?.string ?? "", questions: qs,
                                      refused: (try? Self.strings(m.value["refused"], "refused")) ?? [])
        }
        oracle = orc
        note("oracle_slim_json", oracleURL, oracleData, ["records": orc.count, "override": oracleOverride != nil,
                                                          "questions": orc.values.reduce(0) { $0 + $1.questions.count }])

        // mac_ref.json
        let macURL = root.appendingPathComponent("mac_ref.json")
        let (macData, macDoc) = try load(macURL)
        var mr: [String: [String: MacRow]] = [:]
        for m in macDoc["records"]?.members ?? [] {
            var rows: [String: MacRow] = [:]
            for q in m.value["rows"]?.members ?? [] {
                guard let sha = q.value["hidden_sha256"]?.string else { throw GateError.fixture("mac_ref.json \(m.key) \(q.key)") }
                rows[q.key] = MacRow(pBits: try Self.strings(q.value["p_bits"], "mac_ref.json \(m.key) \(q.key) p_bits"), hiddenSHA: sha)
            }
            mr[m.key] = rows
        }
        mac = mr
        note("mac_ref_json", macURL, macData, ["records": mr.count, "rows": mr.values.reduce(0) { $0 + $1.count }])

        // red_arms.json
        let redURL = root.appendingPathComponent("red_arms.json")
        let (redData, redDoc) = try load(redURL)
        var arms: [RedArm] = []
        for a in redDoc["arms"]?.array ?? [] {
            guard let id = a["id"]?.string, let kind = a["kind"]?.string else { throw GateError.fixture("red_arms.json: an arm without id / kind") }
            var rqs: [RedRequest] = []
            for r in a["requests"]?.array ?? [] {
                guard let name = r["name"]?.string, let req = r["request"], req.members != nil, let base = r["base"]?.string else {
                    throw GateError.fixture("red_arms.json \(id): a request without name / request / base")
                }
                let only: [String]? = (r["only"].map { $0.isNull } ?? true) ? nil : try Self.strings(r["only"], "red \(id) only")
                var o: [String: [Double]] = [:]
                for m in r["oracle"]?.members ?? [] { o[m.key] = try Self.doubles(m.value, "red \(id) oracle \(m.key)") }
                rqs.append(RedRequest(name: name, json: req, base: base, only: only, oracle: o))
            }
            arms.append(RedArm(id: id, kind: kind, requests: rqs))
        }
        red = arms
        note("red_arms_json", redURL, redData, ["arms": arms.map(\.id)])

        // bench.json
        let benchURL = root.appendingPathComponent("bench.json")
        let (benchData, benchDoc) = try load(benchURL)
        func items(_ v: JSONValue?) throws -> [BenchItem] {
            try (v?.array ?? []).map { it in
                guard let name = it["name"]?.string, let kind = it["kind"]?.string, let record = it["record"]?.string,
                      let reps = it["reps"]?.intValue
                else { throw GateError.fixture("bench.json: an item without name / kind / record / reps") }
                let qs: [String]? = (it["questions"].map { $0.isNull } ?? true) ? nil : try Self.strings(it["questions"], "bench \(name)")
                return BenchItem(name: name, kind: kind, record: record, questions: qs, shared: it["shared"]?.boolValue ?? false,
                                 reps: reps, repRest: it["rep_rest_s"]?.double ?? 0)
            }
        }
        bench = try items(benchDoc["bench"])
        benchAOT = try items(benchDoc["bench_aot"])
        note("bench_json", benchURL, benchData, ["bench": bench.map(\.name), "bench_aot": benchAOT.map(\.name)])
        files = f

        // every record has its oracle record; every oracle question is one the request names
        for r in recs {
            guard let o = oracle[r.id] else { throw GateError.fixture("oracle_slim.json: no record \(r.id)") }
            let names = Set(r.questionNames)
            for q in o.questions where !names.contains(q.name) {
                throw GateError.fixture("\(r.id): the oracle's question \(q.name) is not in the request")
            }
        }
        for it in bench + benchAOT where byID[it.record] == nil { throw GateError.fixture("bench.json \(it.name): no record \(it.record)") }
        for a in red { for r in a.requests where oracle[r.base] == nil { throw GateError.fixture("red \(a.id): no base \(r.base)") } }
    }

    func recordsOf(set: String) -> [Record] { records.filter { $0.set == set } }

    // MARK: - JSON helpers (strict: a value of another type is an error, not a default)

    static func strings(_ v: JSONValue?, _ what: String) throws -> [String] {
        guard let a = v?.array else { throw GateError.fixture("\(what): not an array") }
        let s = a.compactMap(\.string)
        guard s.count == a.count else { throw GateError.fixture("\(what): not every element is a string") }
        return s
    }

    static func ints(_ v: JSONValue?, _ what: String) throws -> [Int] {
        guard let a = v?.array else { throw GateError.fixture("\(what): not an array") }
        let s = a.compactMap(\.intValue)
        guard s.count == a.count else { throw GateError.fixture("\(what): not every element is an int") }
        return s
    }

    static func doubles(_ v: JSONValue?, _ what: String) throws -> [Double] {
        guard let a = v?.array else { throw GateError.fixture("\(what): not an array") }
        let s = a.compactMap(\.double)
        guard s.count == a.count else { throw GateError.fixture("\(what): not every element is a number") }
        return s
    }

    // MARK: - scoring

    /// numpy's argmax: the first index of the largest value.
    static func firstArgmax(_ p: [Double]) -> Int {
        var best = 0
        for i in p.indices.dropFirst() where p[i] > p[best] { best = i }
        return best
    }

    static func hexBits(_ x: Double) -> String { String(x.bitPattern, radix: 16) }

    static func fromBits(_ s: String) -> Double { Double(bitPattern: UInt64(s, radix: 16) ?? 0) }

    /// A graph row against the oracle question and the Mac reference: the row's ids (picture slots V + k mapped back to
    /// <image>), the groups the readout read and the keys must be the oracle's.
    static func score(key: String, set: String, row: D1GraphRow, hidden: [Float16], p: [Double], oracle: OracleQuestion,
                      mac: MacRow?) -> RowScore {
        let processor = row.ids.map { $0 >= D1Vision.extensionBase ? D1Vision.imageID : $0 }
        let idsEqual = processor == oracle.rowIDs && row.readGroups == oracle.groups && row.row.keys == oracle.keys
            && p.count == oracle.probs.count
        return stored(key: key, set: set, pBits: p.map(hexBits), hiddenSHA: sha256Hex(of: hidden), idsEqual: idsEqual,
                      finite: hidden.allSatisfy(\.isFinite), allZero: hidden.allSatisfy { $0 == 0 }, oracle: oracle, mac: mac)
    }

    /// A row from its recorded values (this launch's, or an earlier launch's p_ref file).
    static func stored(key: String, set: String, pBits: [String], hiddenSHA: String, idsEqual: Bool, finite: Bool,
                       allZero: Bool, oracle: OracleQuestion, mac: MacRow?) -> RowScore {
        let p = pBits.map(fromBits)
        let deltas = zip(p, oracle.probs).map { abs($0 - $1) }
        let api = oracle.apiProbs.map { a in zip(p, a).map { abs($0 - $1) } }
        var macHidden: Bool? = nil, macBits: Bool? = nil, macDp: Double? = nil, macArg: Bool? = nil
        if let m = mac {
            macHidden = m.hiddenSHA == hiddenSHA
            macBits = m.pBits == pBits
            let pm = m.pBits.map(fromBits)
            if pm.count == p.count {
                macDp = zip(p, pm).map { abs($0 - $1) }.max() ?? 0
                macArg = firstArgmax(p) == firstArgmax(pm)
            }
        }
        return RowScore(key: key, set: set, idsEqualOracle: idsEqual && p.count == oracle.probs.count, argmax: firstArgmax(p),
                        argmaxOracle: oracle.keys.firstIndex(of: oracle.argmax) ?? -1, nearTie: oracle.nearTie, deltas: deltas,
                        apiDeltas: api, finite: finite, allZero: allZero, pBits: pBits, hiddenSHA: hiddenSHA,
                        macHiddenEqual: macHidden, macPBitEqual: macBits, macMaxAbsDp: macDp, macArgmaxEqual: macArg)
    }

    /// The bar over a set of scored rows, with the Mac comparison and the API path's |dp| (when the oracle has it).
    static func summarize(_ rows: [RowScore]) -> [String: Any] {
        let far = rows.filter { !$0.nearTie }, near = rows.filter(\.nearTie)
        let maxDp = rows.map(\.maxAbsDp).max() ?? .nan
        let mean = rows.isEmpty ? Double.nan : rows.map(\.meanAbsDp).reduce(0, +) / Double(rows.count)
        let worst = rows.max { $0.maxAbsDp < $1.maxAbsDp }
        let idsOK = rows.filter(\.idsEqualOracle).count
        let finite = rows.allSatisfy(\.finite), zeros = rows.filter(\.allZero).count
        let argFar = far.filter(\.argmaxEqual).count, argNear = near.filter(\.argmaxEqual).count
        let pass = !rows.isEmpty && argFar == far.count && maxDp <= 0.02 && mean <= 0.002 && idsOK == rows.count
            && finite && zeros == 0
        let macDp = rows.compactMap(\.macMaxAbsDp)
        func count(_ xs: [Bool?]) -> Int { xs.compactMap { $0 }.filter { $0 }.count }
        var mac: [String: Any] = ["rows_compared": macDp.count]
        mac["hidden_sha256_equal"] = count(rows.map(\.macHiddenEqual))
        mac["p_bit_equal"] = count(rows.map(\.macPBitEqual))
        mac["argmax_equal"] = count(rows.map(\.macArgmaxEqual))
        mac["max_abs_dp"] = macDp.max() ?? Double.nan
        mac["median_abs_dp"] = median(macDp)
        let api = rows.compactMap(\.apiMaxAbsDp)
        var s: [String: Any] = ["questions": rows.count, "questions_non_near_tie": far.count,
                                "argmax_equal_non_near_tie": argFar, "near_tie_questions": near.count,
                                "argmax_equal_near_tie": argNear]
        s["max_abs_dp"] = maxDp
        s["worst_row"] = worst?.key ?? ""
        s["mean_of_run_mean_abs_dp"] = mean
        s["ids_equal_oracle"] = idsOK
        s["finite_all"] = finite
        s["all_zero_rows"] = zeros
        s["bar_pass"] = pass
        s["mac"] = mac
        s["api_path"] = ["rows": api.count, "max_abs_dp": api.max() ?? Double.nan]
        return s
    }

    static func summaryLine(_ tag: String, _ s: [String: Any]) -> String {
        let m = s["mac"] as? [String: Any] ?? [:]
        func d(_ x: Any?) -> Double { x as? Double ?? .nan }
        return "\(tag): \(s["questions"] ?? 0) questions | argmax non-near-tie \(s["argmax_equal_non_near_tie"] ?? 0)/"
            + "\(s["questions_non_near_tie"] ?? 0), near-tie \(s["argmax_equal_near_tie"] ?? 0)/\(s["near_tie_questions"] ?? 0), "
            + "ids \(s["ids_equal_oracle"] ?? 0), max|dp| \(f6(d(s["max_abs_dp"]))) (\(s["worst_row"] ?? "")), mean "
            + "\(f6(d(s["mean_of_run_mean_abs_dp"]))), bar \((s["bar_pass"] as? Bool) == true ? "PASS" : "FAIL") | vs Mac: "
            + "hidden sha256 \(m["hidden_sha256_equal"] ?? 0)/\(m["rows_compared"] ?? 0), p bits \(m["p_bit_equal"] ?? 0), "
            + "argmax \(m["argmax_equal"] ?? 0), max|dp| \(f6(d(m["max_abs_dp"])))"
    }

    /// The request with only the named questions (request order), as timing.py's sub_request; nil = the whole request.
    static func subRequest(_ request: JSONValue, names: [String]?) -> JSONValue {
        guard let names, case .object(let m) = request else { return request }
        return .object(m.map { member in
            guard member.key == "questions", case .object(let qs) = member.value else { return member }
            return JSONMember("questions", .object(qs.filter { names.contains($0.key) }))
        })
    }
}
