// Fixtures — what the gate reads from the assets directory's fixtures/ (../_stage.sh writes it from the port's working
// directory ~/code/coreai/_decider2bv) and the per-run verdict against the two references:
//   rows.json        the round-1 fixture: 41 rows (35 image rows, 6 text rows), context + questions + options
//   meta.json        the fixture images: path, the decoded RGB's sha256 (Pillow convert("RGB"))
//   images/*.png     the 29 images
//   oracle_slim.json per run (g256 / g448 / text: 76 runs, 108 slots) the author's fp32 oracle: ids (processor form,
//                    <|image_pad|> in the image block), slot positions, option counts, probabilities, argmax
//   mac_ref.json     per run the Mac's Swift JIT read-out of the same assets (round 8, decider-vision fixture --asset
//                    jit): ids, slots, letter logits, probabilities, argmax, full-vocabulary top-1, and the sha256 of the
//                    decoded / resized pixels, of the patches, of the tower output and of the full fp16 slot logits
// The run list is rows.json's order, as the Mac CLI runs it: every image row at g256 and g448, every text row once.

import DeciderVision
import Foundation

struct Fixtures {
    static let letter0 = 32
    static let vocab = PromptBuilder.vocab
    static let imagePad = PromptBuilder.imagePad

    struct Question: Decodable {
        let text: String
        let options: [String]
    }

    struct Row: Decodable {
        let id: String
        let image: String?
        let context: String
        let questions: [Question]
        let form: String?
    }

    struct OracleRun: Decodable {
        let ids: [Int]
        let slot_idx: [Int]
        let nopts: [Int]
        let probs: [[Double]]
        let argmax: [Int]
    }

    struct MacAnswer: Decodable {
        let letter_logits: [Double]
        let probs: [Double]
        let argmax: Int
        let full_vocab_top1_id: Int
        let read_from: String?
    }

    struct MacRun: Decodable {
        let ids: [Int]
        let slots: [Int]
        let answers: [MacAnswer]
        let tower_embeds_sha256: String?
        let resized_rgb_sha256: String?
        let patches_sha256: String?
        let decoded_rgb_sha256: String?
        let slot_logits_sha256: String?
        let wall_from_file_s: Double?
    }

    struct Run: Sendable {
        let id: String
        let arm: String            // g256 / g448 / text
        var key: String { "\(id)/\(arm)" }
    }

    let root: URL
    let rows: [String: Row]
    let rowOrder: [String]
    let rgbSHA: [String: String]
    let oracle: [String: OracleRun]
    let mac: [String: MacRun]
    let files: [String: Any]

    init(root: URL) throws {
        self.root = root
        func load(_ name: String) throws -> Data {
            do { return try Data(contentsOf: root.appendingPathComponent(name)) } catch {
                throw GateError.fixture("\(name): \(error.localizedDescription)")
            }
        }
        struct RowsFile: Decodable { let rows: [Row] }
        struct MetaFile: Decodable {
            struct Image: Decodable { let name: String; let path: String; let rgb_sha256: String }
            let images: [Image]
        }
        struct OracleFile: Decodable { let runs: [String: OracleRun] }
        struct MacFile: Decodable { let runs: [String: MacRun] }
        let rowsData = try load("rows.json")
        let rowList = try JSONDecoder().decode(RowsFile.self, from: rowsData).rows
        rows = Dictionary(uniqueKeysWithValues: rowList.map { ($0.id, $0) })
        rowOrder = rowList.map(\.id)
        let meta = try JSONDecoder().decode(MetaFile.self, from: try load("meta.json"))
        rgbSHA = Dictionary(uniqueKeysWithValues: meta.images.map { ($0.name, $0.rgb_sha256) })
        let oracleData = try load("oracle_slim.json"), macData = try load("mac_ref.json")
        oracle = try JSONDecoder().decode(OracleFile.self, from: oracleData).runs
        mac = try JSONDecoder().decode(MacFile.self, from: macData).runs
        files = ["rows_json_sha256": sha256Hex(rowsData), "oracle_slim_sha256": sha256Hex(oracleData),
                 "mac_ref_sha256": sha256Hex(macData), "rows": rowList.count,
                 "image_rows": rowList.filter { $0.image != nil }.count, "oracle_runs": oracle.count, "mac_runs": mac.count]
    }

    /// The runs of one arm in rows.json's order.
    func runs(arm: String) -> [Run] {
        rowOrder.compactMap { id in
            guard let r = rows[id] else { return nil }
            if arm == "text" { return r.image == nil ? Run(id: id, arm: arm) : nil }
            return r.image != nil ? Run(id: id, arm: arm) : nil
        }
    }

    func imageURL(_ name: String) -> URL { root.appendingPathComponent("images/\(name).png") }

    // MARK: - verdicts

    /// One slot scored against the oracle and the Mac.
    struct SlotScore {
        let json: [String: Any]
        let deltas: [Double]           // |p - p_oracle| per option
        let argmaxEqual: Bool
        let fullTop1IsOracleLetter: Bool
        let finite: Bool
        let macArgmaxEqual: Bool?
        let macMaxAbsDp: Double?
        let macLetterLogitsBitEqual: Bool?
    }

    static func scoreSlot(_ a: VisionDecider.Answer, slot s: Int, oracle o: OracleRun?, mac m: MacRun?) -> SlotScore {
        var j: [String: Any] = [:]
        var deltas: [Double] = []
        var argmaxEqual = false, top1 = false
        if let o, s < o.probs.count {
            let po = Array(o.probs[s].prefix(a.probabilities.count))
            deltas = zip(a.probabilities, po).map { abs($0 - $1) }
            argmaxEqual = a.argmax == o.argmax[s]
            top1 = a.fullVocabTop1 == letter0 + o.argmax[s]
            j["oracle"] = ["probs": po, "argmax": o.argmax[s], "max_abs_dp": deltas.max() ?? .nan,
                           "mean_abs_dp": deltas.isEmpty ? Double.nan : deltas.reduce(0, +) / Double(deltas.count),
                           "argmax_equal": argmaxEqual, "full_vocab_top1_is_oracle_letter": top1,
                           "top2_margin": po.count > 1 ? (po.sorted().last! - po.sorted().dropLast().last!) : Double.nan]
        }
        var macArgmax: Bool? = nil, macDp: Double? = nil, macBits: Bool? = nil
        if let m, s < m.answers.count {
            let ma = m.answers[s]
            macArgmax = ma.argmax == a.argmax
            let d = zip(a.probabilities, ma.probs).map { abs($0 - $1) }
            macDp = d.max() ?? .nan
            macBits = ma.letter_logits.count == a.letterLogits.count
                && zip(ma.letter_logits, a.letterLogits).allSatisfy { Float16($0).bitPattern == $1.bitPattern }
            j["mac"] = ["probs": ma.probs, "argmax": ma.argmax, "argmax_equal": macArgmax!, "max_abs_dp": macDp!,
                        "letter_logits_bit_equal": macBits!, "full_vocab_top1_equal": ma.full_vocab_top1_id == a.fullVocabTop1,
                        "read_from_equal": ma.read_from.map { $0 == a.readFrom } as Any]
        }
        return SlotScore(json: j, deltas: deltas, argmaxEqual: argmaxEqual, fullTop1IsOracleLetter: top1, finite: a.finite,
                         macArgmaxEqual: macArgmax, macMaxAbsDp: macDp, macLetterLogitsBitEqual: macBits)
    }

    /// The row's ids in the processor's form (V + k -> <|image_pad|>), as the oracle stores them.
    static func processorIDs(_ ids: [Int]) -> [Int] { ids.map { $0 >= vocab ? imagePad : $0 } }
}
