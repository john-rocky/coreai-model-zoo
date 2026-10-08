// Fixtures — what the gate reads from the assets directory's fixtures/ (../_fixtures.py writes it from the lane's files)
// and each row's verdict against the two references:
//   subset.json       the records the gate answers (their requests as written: the key order and the number literals the
//                     host's state text depends on) and the 78 rows it scores: 60 text rows (buckets 256 / 4096), 9 image
//                     rows (img_01..03: buckets 256 / 2048) and 9 audio rows (aud_01..03: the 10 s clip, bucket 256), each
//                     with the publisher's ids, markers, bucket, positions and P; the images and clips (media/)
//   oracle_slim.json  per row the publisher's fp32 oracle: probabilities (reported order), argmax index, top-2 margin,
//                     near tie
//   mac_ref.json      per row the Mac's Swift run of the same graphs (round 10, macos-ship: JIT = AOT h16c = the Python
//                     runtime) as float32 bits of the marker logits and the probabilities, and rows_l64: the 18 rows of
//                     at most 64 positions on L64 (round 12's Swift run); per image and clip the sha256 of every array of
//                     the host's media path (decoded RGB, crops and their four vision inputs, samples, mel, masks, each
//                     media graph output, the prefix)
//   bench.json        the timed workloads W1..W5 and the series rules
//
// The bar (FACTS §7, conversion/d1_omni/_metrics.py SHIP_BAR, unchanged): argmax equal on every row whose oracle top-2
// margin is above 0.02 (near ties counted apart); max |dp| <= 0.02 over every option of every row; the mean over rows of
// each row's max |dp| <= 0.002; here also every row's ids, markers and bucket equal to the publisher's, and finite
// logits. The control (_metrics.wrong_pairing) judges each row against the oracle of the next row of its class (type,
// K, mode) and must FAIL, or the instrument cannot go red.

import D1Omni
import Foundation

struct Fixtures {
    struct Row: Sendable {
        let key: String
        let group: String
        let id: String
        let qid: String
        let mode: Mode
        let type: String
        let K: Int
        let bucket: Int
        let positions: Int
        let prefixLength: Int
        let ids: [Int]
        let markers: [Int]
    }

    struct Record: Sendable {
        let id: String
        let kind: String
        let source: String
        let state: JSONValue?
        let questions: JSONValue
        let images: [String]
        let audio: String?
    }

    struct Media: Decodable, Sendable {
        let id: String
        let file: String
        let file_sha256: String
        let prefix_rows: Int
    }

    struct OracleRow: Decodable, Sendable {
        let type: String
        let K: Int
        let probs: [Double]
        let argmax_index: Int
        let top2_margin: Double?
        let near_tie: Bool
    }

    struct OracleFile: Decodable {
        let rows: [String: OracleRow]
    }

    struct MacRow: Decodable, Sendable {
        let logits_bits: [UInt32]
        let probs_bits: [UInt32]
    }

    struct MacCrop: Decodable, Sendable {
        let k: Int
        let crop_u8: String
        let pixel_values: String
        let pos_embed: String
        let patch_mask: String
        let unshuffle_index: String
        let output: String
    }

    struct MacImage: Decodable, Sendable {
        let rgb: String
        let prefix: String
        let crops: [MacCrop]
        let prefix_rows: Int
    }

    struct MacClip: Decodable, Sendable {
        let samples: String
        let mel: String
        let mask_f: String
        let mask_f2: String
        let mask_f4: String
        let mask_t: String
        let output: String
        let prefix: String
        let bucket_s: Int
        let frames: Int
        let prefix_rows: Int
    }

    struct MacFile: Decodable {
        let rows: [String: MacRow]
        /// the rows of at most 64 positions on L64 (round 12's Swift run)
        let rows_l64: [String: MacRow]
        let images: [String: MacImage]
        let clips: [String: MacClip]
    }

    struct Workload: Decodable, Sendable {
        let id: String
        let what: String
        let record: String
        let qids: [String]
        let mode: String
        /// the decision buckets timed (W1 / W2: L64 and L256)
        let buckets: [Int]
        /// false: the forms run one after the other, not alternating (W3: two L4096 working sets at once)
        let interleave: Bool?
    }

    struct Series: Decodable, Sendable {
        let warmup_first: Int
        let warmup_later: Int
        let timed: Int
        let max_series_s: Double
        let rest_s: Double
    }

    struct BenchFile: Decodable {
        let workloads: [Workload]
        let series: Series
        let order: [String]?
    }

    let root: URL
    let records: [Record]
    let rows: [Row]
    /// the keys of the text rows of at most 64 positions, run again on L64
    let l64Keys: Set<String>
    let images: [Media]
    let clips: [Media]
    let oracle: [String: OracleRow]
    let mac: MacFile
    let bench: BenchFile
    let files: [String: Any]

    var byID: [String: Record] { Dictionary(uniqueKeysWithValues: records.map { ($0.id, $0) }) }

    func rows(of id: String, mode: Mode) -> [Row] { rows.filter { $0.id == id && $0.mode == mode } }

    init(root: URL) throws {
        self.root = root
        func load(_ name: String) throws -> Data {
            do { return try Data(contentsOf: root.appendingPathComponent(name)) } catch {
                throw GateError.fixture("\(name): \(error.localizedDescription)")
            }
        }
        let subsetData = try load("subset.json")
        let subset = try JSONParser.parse(subsetData)
        guard let recs = subset["records"]?.array, let rws = subset["rows"]?.array else {
            throw GateError.fixture("subset.json: no records / rows")
        }
        records = try recs.map { r in
            guard let id = r["id"]?.string, let kind = r["kind"]?.string, let req = r["request"],
                  let qs = req["questions"] else { throw GateError.fixture("subset.json: a record without id / kind / request") }
            return Record(id: id, kind: kind, source: r["source"]?.string ?? "", state: req["state"], questions: qs,
                          images: r["images"]?.array?.compactMap(\.string) ?? [], audio: r["audio"]?.string)
        }
        rows = try rws.map { r in
            guard let key = r["key"]?.string, let id = r["id"]?.string, let qid = r["qid"]?.string,
                  let m = r["mode"]?.string, let mode = Mode(rawValue: m), let bucket = r["bucket"]?.intValue,
                  let positions = r["positions"]?.intValue, let ids = r["ids"]?.array?.compactMap(\.intValue),
                  let markers = r["markers"]?.array?.compactMap(\.intValue) else {
                throw GateError.fixture("subset.json: a row without key / id / qid / mode / bucket / ids / markers")
            }
            return Row(key: key, group: r["group"]?.string ?? "", id: id, qid: qid, mode: mode, type: r["type"]?.string ?? "",
                       K: r["K"]?.intValue ?? 0, bucket: bucket, positions: positions,
                       prefixLength: r["prefix_len"]?.intValue ?? 0, ids: ids, markers: markers)
        }
        l64Keys = Set(subset["l64"]?["keys"]?.array?.compactMap(\.string) ?? [])
        let dec = JSONDecoder()
        images = try dec.decode([Media].self, from: try JSONSerialization.data(withJSONObject:
            (try JSONSerialization.jsonObject(with: subsetData) as? [String: Any])?["images"] ?? []))
        clips = try dec.decode([Media].self, from: try JSONSerialization.data(withJSONObject:
            (try JSONSerialization.jsonObject(with: subsetData) as? [String: Any])?["clips"] ?? []))
        let oracleData = try load("oracle_slim.json")
        oracle = try dec.decode(OracleFile.self, from: oracleData).rows
        let macData = try load("mac_ref.json")
        mac = try dec.decode(MacFile.self, from: macData)
        let benchData = try load("bench.json")
        bench = try dec.decode(BenchFile.self, from: benchData)
        for r in rows {
            guard oracle[r.key] != nil else { throw GateError.fixture("oracle_slim.json: no row \(r.key)") }
            guard mac.rows[r.key] != nil else { throw GateError.fixture("mac_ref.json: no row \(r.key)") }
        }
        for k in l64Keys where mac.rows_l64[k] == nil || oracle[k] == nil {
            throw GateError.fixture("mac_ref.json rows_l64 / oracle_slim.json: no row \(k)")
        }
        files = [
            "subset_json": ["bytes": subsetData.count, "sha256": sha256Hex(subsetData), "records": records.count,
                            "rows": rows.count],
            "oracle_slim_json": ["bytes": oracleData.count, "sha256": sha256Hex(oracleData), "rows": oracle.count],
            "mac_ref_json": ["bytes": macData.count, "sha256": sha256Hex(macData), "rows": mac.rows.count,
                             "rows_l64": mac.rows_l64.count, "images": mac.images.count, "clips": mac.clips.count],
            "bench_json": ["sha256": sha256Hex(benchData), "workloads": bench.workloads.map(\.id)],
        ]
    }

    /// One row's run, scored against the oracle and the Mac.
    struct RowScore: Sendable {
        let key: String
        let group: String
        let mode: String
        let type: String
        let K: Int
        let idsEqual: Bool
        let markersEqual: Bool
        let bucketEqual: Bool
        let finite: Bool
        let logitsBits: [UInt32]
        let probsBits: [UInt32]
        let deltas: [Double]
        let argmax: Int
        let argmaxOracle: Int
        let nearTie: Bool
        let topMargin: Double?
        let macLogitsBitEqual: Bool
        let macProbsBitEqual: Bool
        let macMaxAbsDp: Double
        let macMaxAbsDlogit: Double

        var argmaxEqual: Bool { argmax == argmaxOracle }
        var maxAbsDp: Double { deltas.max() ?? .infinity }
    }

    /// numpy's argmax: the first index of the largest value.
    static func firstArgmax(_ p: [Double]) -> Int {
        var best = 0
        for i in p.indices.dropFirst() where p[i] > p[best] { best = i }
        return best
    }

    /// `l64`: the row on L64, against the Mac's L64 run (mac_ref rows_l64).
    func score(_ row: Row, idsEqual: Bool, markersEqual: Bool, bucketEqual: Bool, logits: [Float], probs: [Float],
               oracleOverride: OracleRow? = nil, l64: Bool = false) -> RowScore {
        let o = oracleOverride ?? oracle[row.key]!
        let m = l64 ? mac.rows_l64[row.key]! : mac.rows[row.key]!
        let p = probs.map(Double.init)
        let finite = logits.allSatisfy(\.isFinite) && probs.allSatisfy(\.isFinite) && probs.count == o.probs.count
        let deltas = finite ? zip(p, o.probs).map { abs($0 - $1) } : [Double.infinity]
        let lb = logits.map(\.bitPattern), pb = probs.map(\.bitPattern)
        let mp = m.probs_bits.map { Double(Float(bitPattern: $0)) }, ml = m.logits_bits.map { Double(Float(bitPattern: $0)) }
        let macDp = mp.count == p.count ? (zip(p, mp).map { abs($0 - $1) }.max() ?? 0) : .infinity
        let macDl = ml.count == logits.count ? (zip(logits.map(Double.init), ml).map { abs($0 - $1) }.max() ?? 0) : .infinity
        return RowScore(key: row.key, group: row.group, mode: row.mode.rawValue, type: row.type, K: row.K, idsEqual: idsEqual,
                        markersEqual: markersEqual, bucketEqual: bucketEqual, finite: finite, logitsBits: lb, probsBits: pb,
                        deltas: deltas, argmax: Self.firstArgmax(p), argmaxOracle: o.argmax_index, nearTie: o.near_tie,
                        topMargin: o.top2_margin, macLogitsBitEqual: lb == m.logits_bits, macProbsBitEqual: pb == m.probs_bits,
                        macMaxAbsDp: macDp, macMaxAbsDlogit: macDl)
    }

    /// FACTS §7 over a set of scored rows, with the Mac comparison.
    static func summarize(_ rows: [RowScore]) -> [String: Any] {
        let clear = rows.filter { !$0.nearTie }, ties = rows.filter(\.nearTie)
        let finite = rows.filter(\.finite).count
        let maxDp = rows.map(\.maxAbsDp).max() ?? .nan
        let worst = rows.max { $0.maxAbsDp < $1.maxAbsDp }
        let mean = rows.isEmpty || finite != rows.count ? Double.infinity : rows.map(\.maxAbsDp).reduce(0, +) / Double(rows.count)
        let ids = rows.filter { $0.idsEqual && $0.markersEqual && $0.bucketEqual }.count
        let argClear = clear.filter(\.argmaxEqual).count
        let pass = !rows.isEmpty && argClear == clear.count && maxDp <= 0.02 && mean <= 0.002 && finite == rows.count
            && ids == rows.count
        var mac: [String: Any] = ["rows": rows.count]
        mac["logits_bit_equal"] = rows.filter(\.macLogitsBitEqual).count
        mac["probs_bit_equal"] = rows.filter(\.macProbsBitEqual).count
        mac["max_abs_dp"] = rows.map(\.macMaxAbsDp).max() ?? Double.nan
        mac["max_abs_dlogit"] = rows.map(\.macMaxAbsDlogit).max() ?? Double.nan
        let worstMac = rows.max { $0.macMaxAbsDp < $1.macMaxAbsDp }
        mac["max_abs_dp_row"] = worstMac?.key ?? ""
        var s: [String: Any] = ["rows": rows.count, "finite_rows": finite, "non_near_tie": clear.count,
                                "argmax_equal_non_near_tie": argClear, "near_tie": ties.count,
                                "argmax_equal_near_tie": ties.filter(\.argmaxEqual).count,
                                "argmax_flips": rows.filter { !$0.argmaxEqual }.map(\.key)]
        s["max_abs_dp"] = maxDp
        s["max_abs_dp_row"] = worst?.key ?? ""
        s["mean_row_max_abs_dp"] = mean
        s["ids_markers_bucket_equal"] = ids
        s["bar_pass"] = pass
        s["mac"] = mac
        return s
    }

    /// _metrics.wrong_pairing: each row's probabilities against the oracle of the next row of its class (type, K, mode);
    /// the bar must FAIL.
    func control(_ rows: [Row], _ probs: [String: [Float]]) -> [String: Any] {
        var classes: [String: [Row]] = [:]
        var order: [String] = []
        for r in rows {
            let c = "\(r.type)/\(r.K)/\(r.mode.rawValue)"
            if classes[c] == nil { order.append(c) }
            classes[c, default: []].append(r)
        }
        var judged: [RowScore] = []
        var unpaired: [String] = []
        for c in order {
            let members = classes[c]!
            if members.count < 2 {
                unpaired += members.map(\.key)
                continue
            }
            for (j, r) in members.enumerated() {
                guard let p = probs[r.key] else { continue }
                let donor = oracle[members[(j + 1) % members.count].key]!
                judged.append(score(r, idsEqual: true, markersEqual: true, bucketEqual: true, logits: [], probs: p,
                                    oracleOverride: donor))
            }
        }
        let s = Self.summarize(judged)
        let status = (s["bar_pass"] as? Bool) == true ? "PASS" : "FAIL"
        return ["status": status, "must_be": "FAIL", "caught": status == "FAIL", "paired_rows": judged.count,
                "unpaired_rows": unpaired, "rows_over_dp_bar": judged.filter { $0.maxAbsDp > 0.02 }.count,
                "max_abs_dp": s["max_abs_dp"] ?? Double.nan, "mean_row_max_abs_dp": s["mean_row_max_abs_dp"] ?? Double.nan]
    }
}
