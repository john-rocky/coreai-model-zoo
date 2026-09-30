// main.swift — the Swift host (JuliaDecisions.swift) against the fixture rows, on a Mac bundle.
//
//   swiftc -O -target arm64-apple-macos27.0 ../JuliaDecisions.swift main.swift -o julia-parity
//   ./julia-parity <variant folder> <pieces.json> [gpu|cpu_only]
//
// pieces.json (swift_pieces_julia.py) holds, per fixture row, the request and the publisher's token ids for
// every text piece the builder encodes (the head, each " " + option, the state), taken from the checkpoint's
// tokenizer.json by the `tokenizers` library: the Swift builder encodes through that table, so this checks the
// row assembly, the graph call and the readout, not a Swift tokenizer. Gate: ids and markers identical to the
// publisher's on every strict row; argmax identical and max |dp| <= 1e-3 at T = 1 vs the publisher's logits.

import CoreAI
import Foundation

struct Piece: Decodable { let text: String; let ids: [Int32] }
struct Row: Decodable {
    let row_id: String
    let type: String
    let question: String
    let options: [String]
    let state: String
    let ids: [Int32]
    let markers: [Int]
    let publisher_logits: [Double]
}
struct Pieces: Decodable { let pieces: [Piece]; let rows: [Row] }

@available(macOS 27, *)
func run() async throws -> Int32 {
    let args = CommandLine.arguments
    guard args.count >= 3 else {
        print("usage: julia-parity <variant folder> <pieces.json> [gpu|cpu_only]")
        return 2
    }
    let folder = URL(fileURLWithPath: args[1])
    let data = try JSONDecoder().decode(Pieces.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
    var table: [String: [Int32]] = [:]
    for piece in data.pieces { table[piece.text] = piece.ids }
    let lookup = table
    let options: SpecializationOptions = args.count > 3 && args[3] == "cpu_only"
        ? .cpuOnly : SpecializationOptions(preferredComputeUnitKind: .gpu)
    let julia = try await JuliaDecisions(folder: folder, options: options) { text in
        guard let ids = lookup[text] else { fatalError("piece not in the table: \(text.prefix(60))") }
        return ids
    }
    var sameRows = 0, argmax = 0, worst = 0.0, checked = 0
    let started = Date()
    for row in data.rows {
        let built = try julia.row(state: row.state, instructions: row.question, type: row.type, options: row.options)
        if built.ids == row.ids && built.markers == row.markers { sameRows += 1 }
        let z = try await julia.logits(state: row.state, instructions: row.question, type: row.type, options: row.options)
            .map(Double.init)
        func softmax(_ v: [Double]) -> [Double] {
            let top = v.max()!
            let e = v.map { exp($0 - top) }
            let s = e.reduce(0, +)
            return e.map { $0 / s }
        }
        let p = softmax(z), q = softmax(row.publisher_logits)
        let error = zip(p, q).map { abs($0 - $1) }.max()!
        worst = max(worst, error)
        if p.indices.max(by: { p[$0] < p[$1] }) == q.indices.max(by: { q[$0] < q[$1] }) { argmax += 1 }
        checked += 1
    }
    let seconds = Date().timeIntervalSince(started)
    let pass = sameRows == checked && argmax == checked && worst <= 1e-3
    let summary: [String: Any] = [
        "status": pass ? "PASS" : "FAIL", "rows": checked, "ids_and_markers_identical": sameRows,
        "argmax_identical": argmax, "max_probability_error": worst, "seconds": seconds,
        "folder": folder.lastPathComponent, "compute": args.count > 3 ? args[3] : "gpu",
    ]
    print(String(data: try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]), encoding: .utf8)!)
    return pass ? 0 : 1
}

if #available(macOS 27, *) {
    let code = try await run()
    exit(code)
}
