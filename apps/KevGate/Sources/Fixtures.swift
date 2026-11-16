// Fixtures — what the gate reads from the assets directory's fixtures/ (../_stage.sh writes it from the lane's working
// directory ~/code/coreai/_kev) and the per-row verdict against the two references:
//   requests.json        the records the gate answers, in order, each {id, set ("fixture" | "heldout"), source, request}:
//                        the round-1 fixture (384 records, 434 questions), then the held-out set (130 records / questions)
//   oracle_slim.json     per record, per question, the author's fp32 oracle (L/oracle/records_oracle.json,
//                        L/oracle/heldout/records_oracle.json): row_ids, decide, opts, keys, probs, argmax, top2_margin,
//                        near_tie
//   mac_ref.json         per record, per question, the Mac's Swift read-out of the same bundle (round 7: `kev fixture
//                        --shared`, AOT h16c, Apple M4 Max; L/swift/gate/{fixture,heldout}_kev-0.8b.json): the sha256 of
//                        the hidden rows the head read (fp16, row-major), p (Float bit patterns), the shared prefix's p
//                        bits, and the record's answers as json.dumps writes them
//   bench.json           the timed items (conversion/kev/timing.py's items): `bench` (single rows, then multi-question
//                        requests direct and shared) and `bench_aot` (the AOT asset's subset)
//   oracle_slim_4b.json / mac_ref_4b.json   the same two references for Kev-4B, the one record load_4b decides
//
// The bar (conversion/kev/readout_gate.py BAR, unchanged): per question, argmax = the oracle's on every
// question whose oracle top-2 margin is above 0.02 (the near-ties are counted apart); max |dp| <= 0.02 over every option
// of every question; the mean over runs (one run = one question) of the run's mean |dp| <= 0.002; and here, as the
// Swift gate (gate_swift.py G1) also asks: every row's ids, <decide> / </opt> indices and keys equal the oracle's, and
// every hidden value finite. A run is scored on the p the library returns (float64 head, rounded to Float once).

import Foundation
import Kev

struct Fixtures {
    struct Record: Sendable {
        let id: String
        let set: String
        let source: String
        /// the request as written (number literals kept): KevRequest(json:) and the bench's sub-requests read it
        let json: JSONValue
        let request: KevRequest
    }

    struct OracleQuestion: Decodable, Sendable {
        let qid: String
        let type: String
        let keys: [String]
        let row_ids: [Int]
        let decide: Int
        let opts: [Int]
        let probs: [Double]
        let argmax: String
        let top2_margin: Double
        let near_tie: Bool
    }

    struct OracleRecord: Decodable, Sendable {
        let set: String
        let questions: [OracleQuestion]
    }

    struct OracleFile: Decodable {
        let records: [String: OracleRecord]
    }

    struct MacRow: Decodable, Sendable {
        let hidden_sha256: String
        let p_bits: [UInt32]
        let shared_p_bits: [UInt32]?
    }

    struct MacRecord: Decodable, Sendable {
        let set: String
        let answers_json: String
        let rows: [MacRow]
    }

    struct MacFile: Decodable {
        let records: [String: MacRecord]
    }

    struct BenchItem: Decodable, Sendable {
        let name: String
        let record: String
        /// the questions kept, in request order (timing.py's sub_request)
        let keep: [Int]
        /// "direct" and / or "shared"
        let modes: [String]
        let reps: Int
    }

    struct BenchFile: Decodable {
        let bench: [BenchItem]
        let bench_aot: [BenchItem]
    }

    /// One question's run, scored.
    struct RowScore: Sendable {
        let key: String
        let set: String
        let idsEqualOracle: Bool
        let argmax: Int
        let argmaxOracle: Int
        let nearTie: Bool
        let deltas: [Double]
        let finite: Bool
        let allZero: Bool
        let pBits: [UInt32]
        let hiddenSHA: String
        let macHiddenEqual: Bool?
        let macPBitEqual: Bool?
        let macMaxAbsDp: Double?
        let macArgmaxEqual: Bool?

        var argmaxEqual: Bool { argmax == argmaxOracle }
        var maxAbsDp: Double { deltas.max() ?? .nan }
        var meanAbsDp: Double { deltas.isEmpty ? .nan : deltas.reduce(0, +) / Double(deltas.count) }
    }

    let root: URL
    let records: [Record]
    let byID: [String: Record]
    let oracle: [String: OracleRecord]
    let mac: [String: MacRecord]
    let bench: BenchFile
    let oracle4B: [String: OracleRecord]
    let mac4B: [String: MacRecord]
    let files: [String: Any]

    init(root: URL, oracleOverride: URL?) throws {
        self.root = root
        func load(_ url: URL) throws -> Data {
            do { return try Data(contentsOf: url) } catch {
                throw GateError.fixture("\(url.lastPathComponent): \(error.localizedDescription)")
            }
        }
        let reqURL = root.appendingPathComponent("requests.json")
        let reqData = try load(reqURL)
        guard let list = try JSONParser.parse(reqData)["records"]?.array else {
            throw GateError.fixture("requests.json: no records")
        }
        var recs: [Record] = []
        for (i, r) in list.enumerated() {
            guard let id = r["id"]?.string, let set = r["set"]?.string, let req = r["request"] else {
                throw GateError.fixture("requests.json record \(i): no id / set / request")
            }
            do {
                recs.append(Record(id: id, set: set, source: r["source"]?.string ?? "", json: req,
                                   request: try KevRequest(json: req)))
            } catch {
                throw GateError.fixture("requests.json \(id): \(error)")
            }
        }
        records = recs
        byID = Dictionary(uniqueKeysWithValues: recs.map { ($0.id, $0) })
        let oracleURL = oracleOverride ?? root.appendingPathComponent("oracle_slim.json")
        let oracleData = try load(oracleURL)
        oracle = try JSONDecoder().decode(OracleFile.self, from: oracleData).records
        let macURL = root.appendingPathComponent("mac_ref.json")
        let macData = try load(macURL)
        mac = try JSONDecoder().decode(MacFile.self, from: macData).records
        let benchURL = root.appendingPathComponent("bench.json")
        let benchData = try load(benchURL)
        bench = try JSONDecoder().decode(BenchFile.self, from: benchData)
        var f: [String: Any] = [
            "requests_json": ["path": reqURL.path, "bytes": reqData.count, "sha256": sha256Hex(reqData), "records": recs.count],
            "oracle_slim_json": ["path": oracleURL.path, "bytes": oracleData.count, "sha256": sha256Hex(oracleData),
                                 "records": oracle.count, "override": oracleOverride != nil],
            "mac_ref_json": ["path": macURL.path, "bytes": macData.count, "sha256": sha256Hex(macData), "records": mac.count],
            "bench_json": ["path": benchURL.path, "sha256": sha256Hex(benchData), "bench": bench.bench.map(\.name),
                           "bench_aot": bench.bench_aot.map(\.name)],
        ]
        let o4 = root.appendingPathComponent("oracle_slim_4b.json"), m4 = root.appendingPathComponent("mac_ref_4b.json")
        if let d = try? Data(contentsOf: o4), let m = try? Data(contentsOf: m4) {
            oracle4B = try JSONDecoder().decode(OracleFile.self, from: d).records
            mac4B = try JSONDecoder().decode(MacFile.self, from: m).records
            f["oracle_slim_4b_json"] = ["sha256": sha256Hex(d), "records": oracle4B.count]
            f["mac_ref_4b_json"] = ["sha256": sha256Hex(m), "records": mac4B.count]
        } else {
            oracle4B = [:]
            mac4B = [:]
        }
        // every record has its oracle questions, and the Mac reference is per record too
        for r in recs {
            guard let o = oracle[r.id] else { throw GateError.fixture("oracle_slim.json: no record \(r.id)") }
            guard o.questions.count == r.request.questions.count else {
                throw GateError.fixture("\(r.id): \(r.request.questions.count) questions, the oracle \(o.questions.count)")
            }
        }
        files = f
    }

    func recordsOf(set: String) -> [Record] { records.filter { $0.set == set } }

    /// numpy's argmax: the first index of the largest value.
    static func firstArgmax(_ p: [Double]) -> Int {
        var best = 0
        for i in p.indices.dropFirst() where p[i] > p[best] { best = i }
        return best
    }

    /// Row k of a record against the oracle and the Mac reference (`mac` = nil when the set has none).
    static func score(key: String, set: String, row: KevRow, hidden: [Float16], probs: [Float],
                      oracle: OracleQuestion, mac: MacRow?) -> RowScore {
        let p = probs.map(Double.init)
        let deltas = zip(p, oracle.probs).map { abs($0 - $1) }
        let idsEqual = row.ids == oracle.row_ids && row.decide == oracle.decide && row.opts == oracle.opts
            && row.keys == oracle.keys && p.count == oracle.probs.count
        let bits = probs.map(\.bitPattern)
        let sha = sha256Hex(of: hidden)
        var macHidden: Bool? = nil, macBits: Bool? = nil, macDp: Double? = nil, macArg: Bool? = nil
        if let m = mac {
            macHidden = m.hidden_sha256 == sha
            macBits = m.p_bits == bits
            let pm = m.p_bits.map { Double(Float(bitPattern: $0)) }
            if pm.count == p.count {
                macDp = zip(p, pm).map { abs($0 - $1) }.max() ?? 0
                macArg = firstArgmax(p) == firstArgmax(pm)
            }
        }
        return RowScore(key: key, set: set, idsEqualOracle: idsEqual, argmax: firstArgmax(p),
                        argmaxOracle: oracle.keys.firstIndex(of: oracle.argmax) ?? -1, nearTie: oracle.near_tie,
                        deltas: deltas, finite: hidden.allSatisfy { $0.isFinite }, allZero: hidden.allSatisfy { $0 == 0 },
                        pBits: bits, hiddenSHA: sha, macHiddenEqual: macHidden, macPBitEqual: macBits, macMaxAbsDp: macDp,
                        macArgmaxEqual: macArg)
    }

    /// The bar over a set of scored rows, with the Mac comparison.
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

    /// The request with the questions at `keep` (request order), as timing.py's sub_request.
    static func subRequest(_ request: JSONValue, keep: [Int]) -> JSONValue {
        guard case .object(let m) = request else { return request }
        return .object(m.map { member in
            guard member.key == "questions", case .object(let qs) = member.value else { return member }
            return JSONMember("questions", .object(qs.enumerated().filter { keep.contains($0.offset) }.map(\.element)))
        })
    }
}
