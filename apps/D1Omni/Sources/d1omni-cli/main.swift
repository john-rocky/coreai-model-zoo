// d1omni — the Mac CLI over the D1Omni library (conversion/d1_omni/gate_swift.py runs it and checks its output).
//
//   d1omni rows     --bundle-dir <macos> --fixtures records.json --out rows.json [--oracle records_ref.json ...]
//   d1omni tokenize --bundle-dir <macos> --in strings.json --out ids.json
//   d1omni parity   --bundle-dir <macos> --fixtures records.json --reference swift_ref.json --out parity.json
//                   [--asset aot|jit] [--aot-root <compiled>] [--repeats 2]
//   d1omni ask      --bundle-dir <macos> [--asset aot|jit] [--aot-root <compiled>] --state "<text>" | --state-json '<json>'
//                   --question '<question json>' | --questions '<{name: question} json>' [--out resp.json]
//   d1omni time     --bundle-dir <macos> --aot-root <compiled> --fixtures records.json --reference swift_ref.json
//                   --out t.json [--workloads W1,W2,W3] [--rounds 35] [--warmup 5] [--cold] [--label run1]
//                   [--media-reference <swift_ref_media dir> --media-root <work>]   (W4 / W5)
//   d1omni parity-media --bundle-dir <macos> --fixtures records.json --media-reference <swift_ref_media dir>
//                   --media-root <work> --out parity.json [--asset aot|jit] [--aot-root <compiled>] [--repeats 2]
//   d1omni ask ... --image <file> [--image <file> ...] | --audio <file.wav>
//
// <macos>: the folder of the decision buckets (`decide-fp16-L<L>/` or `fp16-L<L>/`, each with metadata.json, the
// .aimodel and tokenizer/). --asset jit (default for ask) loads each bucket's .aimodel, which the runtime specializes
// here; --asset aot loads the Mac's AOT .aimodelc of the same bundle, found under --aot-root (`*/<name>.h16c.aimodelc`
// whose provenance/aot-manifest.json says preferred_compute gpu, architecture h16c, and names the bucket's folder as
// its source). Both load with SpecializationOptions(preferredComputeUnitKind: .gpu), never .default.
//
// rows is the host half alone (tokenizer and prompt, no graph): every record of the fixtures in text mode and, when
// it has media, in its media mode (P from the media's size in its provenance), each row's ids, markers, the encode
// details encode_rows.py records, and the sha256 of the six graph inputs at the row's bucket; with --oracle (the
// publisher's run of the same fixtures) also each oracle row's probabilities from its raw marker logits through the
// Swift readout and each request's response from the oracle's probabilities (json.dumps text).
// parity runs every row of the reference (gate_swift.py pyref: the Python runtime on the same AOT) through the graph
// of its bucket, --repeats times, and writes the marker logits and probabilities (and their bits) beside the
// reference's. time is one timing process: the AOT and the JIT graph of every bucket a workload needs, loaded in this
// process (--cold: this process's cache entries for them removed first, so the first loads specialize), the two
// alternated per round (forward on even rounds, reversed on odd), --warmup rounds dropped; one decision = graph inputs
// -> call(s) -> scores copied -> marker gather + temperature + float32 softmax (tokenizing apart); a media workload's
// decision first runs its media graph on inputs prepared beforehand and cuts the P prefix rows (W4: the vision graph of
// img_01's one crop, W5: the 10 s audio graph of aud_01), its preprocessing (decode, crops + resize, patches +
// positions, mel + masks, tokenize) timed apart in the same process.
// parity-media (round 9) runs the Swift media path on every image and clip of the Python host's dump
// (conversion/d1_omni/host_dump_media.py): the decoded RGB, each crop, the four vision inputs, the prefix; the samples,
// the mel, the four masks, the prefix — each against the dump (bit-equal or max |d|) — the same graph on the dump's
// own inputs (arm B: the runtime call alone), then every media row through the decision graph of its bucket with the
// Swift prefix, and a control row with the next item's prefix (cut or repeated to the row's P).

import CoreAI
import CryptoKit
import D1Omni
import Darwin
import Foundation

// MARK: - Arguments

struct Args {
    var command = ""
    var values: [String: [String]] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) throws {
        guard argv.count > 1 else { throw CLIError.usage("no command") }
        command = argv[1]
        var i = 2
        let flagNames: Set<String> = ["--cold"]
        while i < argv.count {
            let a = argv[i]
            guard a.hasPrefix("--") else { throw CLIError.usage("unexpected \(a)") }
            if flagNames.contains(a) {
                flags.insert(a)
                i += 1
                continue
            }
            guard i + 1 < argv.count else { throw CLIError.usage("\(a) needs a value") }
            values[a, default: []].append(argv[i + 1])
            i += 2
        }
    }

    func one(_ k: String) -> String? { values[k]?.last }
    func all(_ k: String) -> [String] { values[k] ?? [] }
    func need(_ k: String) throws -> String {
        guard let v = one(k) else { throw CLIError.usage("missing \(k)") }
        return v
    }
    func int(_ k: String, _ d: Int) -> Int { one(k).flatMap(Int.init) ?? d }
    func list(_ k: String) -> [String] {
        (one(k) ?? "").split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
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

func url(_ path: String) -> URL {
    URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
}

// MARK: - Output helpers

func hexDigest<D: Digest>(_ d: D) -> String { d.map { String(format: "%02x", $0) }.joined() }
func sha256(_ data: Data) -> String { hexDigest(SHA256.hash(data: data)) }
func sha256<T>(of values: [T]) -> String { values.withUnsafeBytes { hexDigest(SHA256.hash(data: $0)) } }
/// sha256 of json.dumps(ids) ("[1, 17, ...]"): encode_rows.py's ids_sha256.
func idsSHA256(_ ids: [Int]) -> String { sha256(Data(("[" + ids.map(String.init).joined(separator: ", ") + "]").utf8)) }

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

/// The process's physical footprint and its lifetime peak, in bytes.
func footprint() -> (now: Int, peak: Int) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? (Int(info.phys_footprint), Int(info.ledger_phys_footprint_peak)) : (-1, -1)
}

func footprintJSON() -> JSONValue {
    let f = footprint()
    return .obj([("phys_footprint_bytes", .int(f.now)), ("phys_footprint_peak_bytes", .int(f.peak))])
}

let binarySHA256: String? = Bundle.main.executableURL.flatMap { try? sha256(Data(contentsOf: $0)) }

func now() -> String {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime]
    f.timeZone = .current
    return f.string(from: Date())
}

func environment() -> JSONValue {
    #if DEBUG
    let config = "Debug"
    #else
    let config = "Release"
    #endif
    return .obj([("os", .string(ProcessInfo.processInfo.operatingSystemVersionString)),
                 ("os_build", .string(sysctlString("kern.osversion"))),
                 ("chip", .string(sysctlString("machdep.cpu.brand_string"))), ("model", .string(sysctlString("hw.model"))),
                 ("build_configuration", .string(config)), ("pid", .int(Int(getpid()))),
                 ("process_name", .string(ProcessInfo.processInfo.processName)),
                 ("device_architecture", .string(AIModel.deviceArchitectureName)),
                 ("argv", .strings(CommandLine.arguments)), ("binary_sha256", .optional(binarySHA256)), ("at", .string(now()))])
}

func directoryBytes(_ dir: URL) -> Int {
    var bytes = 0
    if let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileAllocatedSizeKey, .isRegularFileKey]) {
        for case let f as URL in e {
            if let v = try? f.resourceValues(forKeys: [.fileAllocatedSizeKey, .isRegularFileKey]), v.isRegularFile == true {
                bytes += v.fileAllocatedSize ?? 0
            }
        }
    }
    return bytes
}

/// The runtime's specialization cache of this process (~/Library/Caches/coreai-cache/<os build>/<process name>).
func cacheDir() -> URL {
    FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Caches/coreai-cache/\(sysctlString("kern.osversion"))/\(ProcessInfo.processInfo.processName)")
}

func cacheState() -> JSONValue {
    let entries = ((try? FileManager.default.contentsOfDirectory(atPath: cacheDir().path)) ?? []).sorted()
    return .obj([("path", .string(cacheDir().path)), ("entries", .strings(entries)),
                 ("entry_bytes", .obj(entries.map { ($0, .int(directoryBytes(cacheDir().appendingPathComponent($0)))) }))])
}

/// The hash file of an asset (`main.hash` of a .aimodel / .aimodelc, hex): the name of the cache entry its load makes.
func mainHash(_ asset: URL) -> String? {
    (try? Data(contentsOf: asset.appendingPathComponent("main.hash"))).map { $0.map { String(format: "%02x", $0) }.joined() }
}

func log(_ s: String) {
    let f = DateFormatter()
    f.dateFormat = "HH:mm:ss"
    FileHandle.standardError.write(Data("[\(f.string(from: Date()))] \(s)\n".utf8))
}

func readJSON(_ u: URL) throws -> JSONValue { try JSONParser.parse(Data(contentsOf: u)) }

func seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

func floats(_ xs: [Float]) -> JSONValue { .array(xs.map { .double(Double($0)) }) }
func bits(_ xs: [Float]) -> JSONValue { .ints(xs.map { Int($0.bitPattern) }) }

func stats(_ v: [Double]) -> JSONValue {
    guard !v.isEmpty else { return .null }
    let s = v.sorted()
    // NumPy's percentile (linear interpolation), as timing_run.py computes it
    func pct(_ q: Double) -> Double {
        let pos = q / 100 * Double(s.count - 1)
        let lo = Int(pos.rounded(.down)), hi = min(lo + 1, s.count - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
    }
    return .obj([("n", .int(v.count)), ("median", .double(pct(50))), ("p10", .double(pct(10))), ("p90", .double(pct(90))),
                 ("mean", .double(v.reduce(0, +) / Double(v.count))), ("min", .double(s.first!)), ("max", .double(s.last!))])
}

// MARK: - Bundles and assets

/// The Mac AOT `.aimodelc` of a bucket folder under `root`: `*/<stem>.h16c.aimodelc` whose aot-manifest says gpu, h16c,
/// and names `folder` as its source.
func findAOT(root: URL, folder: URL, asset: String) throws -> URL {
    let stem = (asset as NSString).deletingPathExtension
    var found: [URL] = []
    for dir in try FileManager.default.contentsOfDirectory(atPath: root.path).sorted() {
        let d = root.appendingPathComponent(dir)
        let c = d.appendingPathComponent("\(stem).h16c.aimodelc")
        let m = d.appendingPathComponent("provenance/aot-manifest.json")
        guard FileManager.default.fileExists(atPath: c.path), let j = try? readJSON(m) else { continue }
        guard j["preferred_compute"]?.string == "gpu", j["architecture"]?.string == "h16c", j["status"]?.string == "COMPILED",
              let src = j["source"]?["folder"]?.string, url(src).path == folder.path else { continue }
        found.append(c)
    }
    guard found.count == 1 else {
        throw CLIError.failed("\(found.count) AOT assets for \(folder.lastPathComponent) under \(root.path): \(found.map(\.path))")
    }
    return found[0]
}

struct Setup {
    let folders: [Int: URL]
    let kind: String
    let assets: [Int: URL]

    init(_ args: Args, defaultKind: String = "aot") throws {
        folders = try D1Omni.folders(macos: url(try args.need("--bundle-dir")))
        kind = args.one("--asset") ?? defaultKind
        var a: [Int: URL] = [:]
        if kind == "aot" {
            let root = url(try args.need("--aot-root"))
            for (L, f) in folders {
                let meta = try readJSON(f.appendingPathComponent("metadata.json"))
                guard let main = meta["assets"]?["main"]?.string else { throw CLIError.failed("\(f.path): no assets.main") }
                a[L] = try findAOT(root: root, folder: f, asset: main)
            }
        } else if kind != "jit" {
            throw CLIError.usage("--asset aot|jit")
        }
        assets = a
    }

    func load(preload: Bool = false) async throws -> D1Omni {
        try await D1Omni(folders: folders, assets: assets, preload: preload)
    }

    var json: JSONValue {
        .obj([("kind", .string(kind)),
              ("folders", .obj(folders.sorted { $0.key < $1.key }.map { (String($0.key), .string($0.value.path)) })),
              ("assets", .obj(assets.sorted { $0.key < $1.key }.map { (String($0.key), .string($0.value.path)) }))])
    }
}

// MARK: - Fixtures

/// A record's media mode and prefix length, from its provenance (encode_rows.py `media_of`): an image's px
/// [width, height] (image_info.px or px), a clip's frames (clip.frames or audio_info.frames).
func mediaOf(_ record: JSONValue) throws -> (mode: Mode, prefix: Int, from: JSONValue)? {
    guard let media = record["media"], !media.isNull else { return nil }
    let prov = record["provenance"] ?? .null
    if media.has("images") {
        guard let px = (prov["image_info"]?["px"] ?? prov["px"])?.array, px.count == 2, let w = px[0].intValue,
              let h = px[1].intValue else { throw CLIError.failed("\(record["id"]?.string ?? "?"): no image px") }
        return (.image, try MediaLength.imagePrefixLength([(w, h)]), .obj([("px", .ints([w, h]))]))
    }
    guard let frames = (prov["clip"]?["frames"] ?? prov["audio_info"]?["frames"])?.intValue else {
        throw CLIError.failed("\(record["id"]?.string ?? "?"): audio without a sample count")
    }
    return (.audio, MediaLength.audioPrefixLength(samples: frames), .obj([("samples", .int(frames))]))
}

func records(_ fixtures: URL) throws -> [JSONValue] {
    guard let r = try readJSON(fixtures)["records"]?.array else { throw CLIError.failed("\(fixtures.path): no records") }
    return r
}

/// The rows of one record in one mode.
func recordRows(_ d1: D1Omni, _ record: JSONValue, _ mode: Mode) throws -> [(row: Row, detail: EncodeDetail)] {
    guard let req = record["request"], let qs = req["questions"] else { throw CLIError.failed("record without a request") }
    let prefix = mode == .text ? 0 : (try mediaOf(record)?.prefix ?? 0)
    return try Prompt.rowsWithDetail(d1.tokenizer, state: req["state"], questions: qs, mode: mode, prefixLength: prefix,
                                     config: d1.config)
}

// MARK: - rows

func graphHashes(_ x: GraphInputs) -> JSONValue {
    .obj([("input_ids", .string(sha256(of: x.inputIDs))), ("pad_mask", .string(sha256(of: x.padMask))),
          ("prefix_mask", .string(sha256(of: x.prefixMask))), ("keep_right", .string(sha256(of: x.keepRight))),
          ("qtype_onehot", .string(sha256(of: x.qtype)))])
}

func rowsCommand(_ args: Args) async throws {
    let setup = try Setup(args, defaultKind: "jit")
    let t0 = ContinuousClock.now
    let d1 = try await setup.load()
    let loadS = seconds(since: t0)
    let fixtures = url(try args.need("--fixtures"))
    var out: [JSONValue] = []
    var requests: [JSONValue] = []
    var errors: [JSONValue] = []
    var encodeMs: [Double] = []
    let tAll = ContinuousClock.now
    for record in try records(fixtures) {
        let id = record["id"]?.string ?? "?"
        var modes: [(Mode, JSONValue?)] = [(.text, nil)]
        if let m = try mediaOf(record) { modes.append((m.mode, m.from)) }
        for (mode, from) in modes {
            let t = ContinuousClock.now
            let rows: [(row: Row, detail: EncodeDetail)]
            do {
                rows = try recordRows(d1, record, mode)
            } catch {
                errors.append(.obj([("id", .string(id)), ("mode", .string(mode.rawValue)), ("error", .string("\(error)"))]))
                continue
            }
            encodeMs.append(seconds(since: t) * 1e3)
            for (r, det) in rows {
                let L = GraphInputs.bucket(positions: r.positions, buckets: d1.lengths)
                var o: [(String, JSONValue)] = [
                    ("id", .string(id)), ("qid", .string(r.qid)), ("mode", .string(mode.rawValue)),
                    ("type", .string(r.question.type.rawValue)), ("K", .int(r.question.options)), ("prefix", .int(r.prefixLength)),
                    ("n_ids", .int(r.ids.count)), ("positions", .int(r.positions)), ("bucket", L.map { .int($0) } ?? .null),
                    ("max_len", .int(r.maxLen)), ("markers", .ints(r.markers)), ("q_delim_pos", .int(det.qDelimPos)),
                    ("ids_sha256", .string(idsSHA256(r.ids))), ("ids", .ints(r.ids)),
                    ("state_tokens", .int(det.stateTokens)), ("state_room", .int(det.stateRoom)),
                    ("state_cut", .int(max(0, det.stateTokens - det.stateRoom))),
                    ("instructions_tokens", .int(det.instructionsTokens)),
                    ("instructions_cut", .int(max(0, det.instructionsTokens + 1 - max(16, det.budget)))),
                    ("budget", .int(det.budget)), ("per_option", .int(det.perOption)),
                    ("option_cut", .ints(det.optionTokens.map { max(0, $0 - det.perOption) })),
                    ("row_cut", .bool(r.ids.count == r.maxLen)), ("calibrate", .bool(r.calibrate)),
                    ("temperature_key", .string(Prompt.temperatureKey(r.question))),
                    ("T", .double(Prompt.temperature(r.question, config: d1.config))),
                ]
                if let from { o.append(("media_from", from)) }
                if let L {
                    let x = try GraphInputs(row: r, length: L)
                    o.append(("graph", .obj([("L", .int(L)), ("markers", .ints(x.markers)), ("sha256", graphHashes(x))])))
                }
                out.append(.obj(o))
            }
            requests.append(.obj([("id", .string(id)), ("mode", .string(mode.rawValue)), ("rows", .int(rows.count)),
                                  ("usage_input_tokens", .int(rows.reduce(0) { $0 + $1.row.positions }))]))
        }
    }
    let totalS = seconds(since: tAll)
    // the oracle's rows through the Swift readout, its responses through Swift's answer()
    var oracle: [JSONValue] = []
    for path in args.all("--oracle") {
        let o = try readJSON(url(path))
        let fixturesByID = Dictionary(try records(fixtures).compactMap { r in r["id"]?.string.map { ($0, r) } },
                                      uniquingKeysWith: { a, _ in a })
        for rec in o["records"]?.array ?? [] {
            guard let id = rec["id"]?.string, let modeName = rec["mode"]?.string, let mode = Mode(rawValue: modeName),
                  let record = fixturesByID[id] else { continue }
            let rows = try recordRows(d1, record, mode).map(\.row)
            var probsFromLogits: [JSONValue] = []
            var oracleProbs: [[Float]] = []
            for q in rec["questions"]?.array ?? [] {
                guard let qid = q["qid"]?.string, let row = rows.first(where: { $0.qid.unicodeScalars.elementsEqual(qid.unicodeScalars) }),
                      let raw = q["logits_raw"]?.array?.compactMap({ $0.double }), let probs = q["probs"]?.array?.compactMap({ $0.double })
                else { throw CLIError.failed("\(path): \(id)/\(modeName): a question the Swift rows do not have") }
                let p = Readout.probabilities(logits: raw.map(Float.init), question: row.question, calibrate: row.calibrate,
                                              config: d1.config)
                probsFromLogits.append(.obj([("qid", .string(qid)), ("ids_equal", .bool((q["ids"]?.array?.compactMap { $0.intValue } ?? []) == row.ids)),
                                             ("markers_equal", .bool((q["markers"]?.array?.compactMap { $0.intValue } ?? []) == row.markers)),
                                             ("probs", floats(p)), ("probs_bits", bits(p))]))
                oracleProbs.append(probs.map(Float.init))
            }
            let ordered = rows   // the request's order; the oracle lists every question of the request
            guard ordered.count == oracleProbs.count else {
                throw CLIError.failed("\(path): \(id)/\(modeName): \(ordered.count) rows, \(oracleProbs.count) oracle questions")
            }
            let response = Readout.response(rows: ordered, probabilities: oracleProbs)
            oracle.append(.obj([("oracle", .string(path)), ("id", .string(id)), ("mode", .string(modeName)),
                                ("questions", .array(probsFromLogits)),
                                ("response_json", .string(PythonFormat.dumps(response)))]))
        }
    }
    let doc: JSONValue = .obj([
        ("schema", .string("d1omni-rows/1")), ("fixtures", .string(fixtures.path)),
        ("fixtures_sha256", .string(sha256(try Data(contentsOf: fixtures)))),
        ("tokenizer_folder", .string(d1.buckets[d1.lengths[0]]!.folder.appendingPathComponent("tokenizer").path)),
        ("token_ids", .obj(d1.tokenizer.ids.byToken.sorted { $0.value < $1.value }.map { ($0.key, .int($0.value)) })),
        ("buckets", .ints(d1.lengths)),
        ("load_seconds", .double(loadS)), ("tokenizer_load_seconds", .double(d1.tokenizer.loadSeconds)),
        ("encode_seconds_total", .double(totalS)), ("encode_ms_per_request", stats(encodeMs)),
        ("rows", .array(out)), ("requests", .array(requests)), ("errors", .array(errors)), ("oracle", .array(oracle)),
        ("environment", environment()), ("footprint", footprintJSON()),
    ])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("rows: \(out.count) rows, \(requests.count) requests, \(errors.count) errors, \(oracle.count) oracle requests; "
        + "tokenizer load \(String(format: "%.2f", d1.tokenizer.loadSeconds)) s, encode \(String(format: "%.2f", totalS)) s")
}

// MARK: - tokenize

func tokenizeCommand(_ args: Args) async throws {
    let setup = try Setup(args, defaultKind: "jit")
    let d1 = try await setup.load()
    let input = try readJSON(url(try args.need("--in")))
    let strings = input["strings"]?.array?.compactMap(\.string) ?? []
    let out: [JSONValue] = strings.map { s in
        .obj([("text", .string(s)), ("plain", .ints(d1.tokenizer.plain(s))), ("escaped", .string(D1Tokenizer.escape(s))),
              ("enc", .ints(d1.tokenizer.enc(s)))])
    }
    try JSONWriter.write(.obj([("schema", .string("d1omni-tokenize/1")), ("items", .array(out)), ("environment", environment())]),
                         to: url(try args.need("--out")), pretty: false)
    log("tokenize: \(out.count) strings")
}

// MARK: - parity

func parityCommand(_ args: Args) async throws {
    let setup = try Setup(args)
    let repeats = max(1, args.int("--repeats", 2))
    let fixtures = url(try args.need("--fixtures"))
    let reference = try readJSON(url(try args.need("--reference")))
    let refDir = url(try args.need("--reference")).deletingLastPathComponent()
    let cacheBefore = cacheState()
    let fp0 = footprintJSON()
    let t0 = ContinuousClock.now
    let d1 = try await setup.load()
    let initS = seconds(since: t0)
    let byID = Dictionary(try records(fixtures).compactMap { r in r["id"]?.string.map { ($0, r) } }, uniquingKeysWith: { a, _ in a })
    var rowsCache: [String: [Row]] = [:]
    var loads: [(String, JSONValue)] = []
    var out: [JSONValue] = []
    let refRows = reference["rows"]?.array ?? []
    // bucket by bucket (each graph loads once, its load timed)
    let order = refRows.indices.sorted { (refRows[$0]["bucket"]?.intValue ?? 0, $0) < (refRows[$1]["bucket"]?.intValue ?? 0, $1) }
    var worstPy = 0.0, worstPyP = 0.0, bitEqual = 0, worstDrift = 0.0
    for i in order {
        let ref = refRows[i]
        guard let id = ref["id"]?.string, let modeName = ref["mode"]?.string, let mode = Mode(rawValue: modeName),
              let qid = ref["qid"]?.string, let L = ref["bucket"]?.intValue, let record = byID[id] else {
            throw CLIError.failed("reference row \(i): id / mode / qid / bucket, or no fixture record")
        }
        let key = "\(id)\u{0}\(modeName)"
        if rowsCache[key] == nil { rowsCache[key] = try recordRows(d1, record, mode).map(\.row) }
        guard let row = rowsCache[key]!.first(where: { $0.qid.unicodeScalars.elementsEqual(qid.unicodeScalars) }) else {
            throw CLIError.failed("\(id)/\(modeName): no question \(qid)")
        }
        let refIDs = ref["ids"]?.array?.compactMap { $0.intValue } ?? []
        guard row.ids == refIDs, try d1.bucket(for: row) == L else {
            throw CLIError.failed("\(id)/\(qid)/\(modeName): the Swift row differs from the reference's (ids or bucket)")
        }
        if d1.loaded[L] == nil {
            let tl = ContinuousClock.now
            let g = try await d1.graph(L)
            loads.append((String(L), .obj([("asset", .string(g.url.path)), ("kind", .string(g.kind)),
                                           ("model_seconds", .double(g.loadSeconds.model)),
                                           ("function_seconds", .double(g.loadSeconds.function)),
                                           ("wall_seconds", .double(seconds(since: tl))), ("main_hash", .optional(mainHash(g.url))),
                                           ("options", .string(describe(g.options))), ("descriptor", g.descriptor),
                                           ("footprint_after", footprintJSON())])))
            log("loaded L\(L) \(g.kind) in \(String(format: "%.2f", seconds(since: tl))) s")
        }
        var prefix: [Float]? = nil
        if let pf = ref["prefix_file"]?.string {
            let data = try Data(contentsOf: refDir.appendingPathComponent(pf))
            prefix = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        var decisions: [D1Omni.Decision] = []
        for _ in 0..<repeats { decisions.append(try await d1.decide(row, prefix: prefix, bucket: L)) }
        let first = decisions[0]
        let drift = decisions.dropFirst().map { d in zip(d.logits, first.logits).map { abs(Double($0) - Double($1)) }.max() ?? 0 }.max() ?? 0
        let pyLogits = ref["logits"]?.array?.compactMap { $0.double }.map(Float.init) ?? []
        let pyProbs = ref["probs"]?.array?.compactMap { $0.double }.map(Float.init) ?? []
        let dLogit = zip(first.logits, pyLogits).map { abs(Double($0) - Double($1)) }.max() ?? .nan
        let dP = zip(first.probabilities, pyProbs).map { abs(Double($0) - Double($1)) }.max() ?? .nan
        let same = first.logits.map(\.bitPattern) == pyLogits.map(\.bitPattern) && first.probabilities.map(\.bitPattern) == pyProbs.map(\.bitPattern)
        worstPy = max(worstPy, dLogit)
        worstPyP = max(worstPyP, dP)
        worstDrift = max(worstDrift, drift)
        if same { bitEqual += 1 }
        out.append(.obj([("id", .string(id)), ("qid", .string(qid)), ("mode", .string(modeName)), ("bucket", .int(L)),
                         ("positions", .int(row.positions)), ("prefix", .int(row.prefixLength)),
                         ("logits", floats(first.logits)), ("logits_bits", bits(first.logits)),
                         ("probs", floats(first.probabilities)), ("probs_bits", bits(first.probabilities)),
                         ("python_max_abs_dlogit", .double(dLogit)), ("python_max_abs_dp", .double(dP)),
                         ("python_bit_equal", .bool(same)), ("repeats", .int(repeats)), ("drift_marker_logits", .double(drift)),
                         ("call_ms", .doubles(decisions.map { $0.seconds.graph * 1e3 }))]))
    }
    let doc: JSONValue = .obj([
        ("schema", .string("d1omni-parity/1")), ("setup", setup.json), ("fixtures", .string(fixtures.path)),
        ("reference", .string(url(try args.need("--reference")).path)), ("repeats", .int(repeats)),
        ("init_seconds", .double(initS)), ("tokenizer_load_seconds", .double(d1.tokenizer.loadSeconds)),
        ("loads", .obj(loads)),
        ("summary", .obj([("rows", .int(out.count)), ("python_bit_equal_rows", .int(bitEqual)),
                          ("python_max_abs_dlogit", .double(worstPy)), ("python_max_abs_dp", .double(worstPyP)),
                          ("max_drift_marker_logits", .double(worstDrift))])),
        ("rows", .array(out)), ("footprint_before", fp0), ("footprint_end", footprintJSON()),
        ("cache_before", cacheBefore), ("cache_after", cacheState()), ("environment", environment()),
    ])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("parity (\(setup.kind)): \(out.count) rows, Python bit-equal \(bitEqual), max |dlogit| vs Python \(worstPy), drift \(worstDrift)")
}

// MARK: - ask

func askCommand(_ args: Args) async throws {
    let setup = try Setup(args, defaultKind: "jit")
    let images = args.all("--image").map(url), audio = args.one("--audio").map(url)
    let media = images.isEmpty && audio == nil ? nil : try mediaSetup(args, kind: setup.kind)
    let d1 = try await D1Omni(folders: setup.folders, assets: setup.assets, media: media)
    let state: JSONValue? = try args.one("--state-json").map { try JSONParser.parse($0) } ?? args.one("--state").map { .string($0) }
    let questions: JSONValue
    if let q = args.one("--questions") {
        questions = try JSONParser.parse(q)
    } else {
        questions = .obj([("question", try JSONParser.parse(try args.need("--question")))])
    }
    guard images.isEmpty || audio == nil else { throw CLIError.usage("--image or --audio, not both") }
    let t = ContinuousClock.now
    let response: JSONValue
    if !images.isEmpty {
        response = try await d1.systemOne(state: state, questions: questions, images: images)
    } else if let audio {
        response = try await d1.systemOne(state: state, questions: questions, audio: audio)
    } else {
        response = try await d1.systemOne(state: state, questions: questions)
    }
    let ms = seconds(since: t) * 1e3
    if let out = args.one("--out") {
        try JSONWriter.write(.obj([("response", response), ("ms_with_load_of_used_buckets", .double(ms)), ("setup", setup.json),
                                   ("environment", environment())]), to: url(out))
    }
    print(PythonFormat.dumps(response, asciiOnly: false))
}

// MARK: - time

struct Workload {
    let id: String
    let what: String
    let record: String
    let qids: [String]
}

let workloads: [String: Workload] = [
    "W1": Workload(id: "W1", what: "one question, short state", record: "card_text", qids: ["refund"]),
    "W2": Workload(id: "W2", what: "three questions in one pass (3 calls)", record: "card_text", qids: ["refund", "team", "urgency"]),
    "W3": Workload(id: "W3", what: "one question over a 3.4k-token state", record: "long_3400", qids: ["tension_10"]),
]

func timeCommand(_ args: Args) async throws {
    let folders = try D1Omni.folders(macos: url(try args.need("--bundle-dir")))
    let root = url(try args.need("--aot-root"))
    let fixtures = url(try args.need("--fixtures"))
    let reference = try args.one("--reference").map { try readJSON(url($0)) } ?? .obj([])
    let rounds = args.int("--rounds", 35), warmup = args.int("--warmup", 5)
    let wantedAll = args.list("--workloads").isEmpty ? ["W1", "W2", "W3"] : args.list("--workloads")
    let wanted = wantedAll.filter { workloads[$0] != nil }
    let wantedMedia = wantedAll.filter { mediaWorkloads[$0] != nil }
    let started = now()
    let cacheBefore = cacheState()
    let fp0 = footprintJSON()
    // the library instance: tokenizer + rows + the media folders' tables (its own graphs are not used here)
    let tInit = ContinuousClock.now
    let mediaFolders = D1Omni.MediaFolders.find(macos: url(try args.need("--bundle-dir")))
    let d1 = try await D1Omni(folders: folders, media: mediaFolders)
    let initS = seconds(since: tInit)
    let byID = Dictionary(try records(fixtures).compactMap { r in r["id"]?.string.map { ($0, r) } }, uniquingKeysWith: { a, _ in a })
    // the media workloads: rows, inputs prepared once, the bucket
    var mediaCases: [MediaCase] = []
    for w in wantedMedia {
        mediaCases.append(try MediaCase(mediaWorkloads[w]!, d1: d1, byID: byID, root: url(try args.need("--media-root"))))
    }
    // rows and the buckets they need
    var cases: [(Workload, [Row], Int)] = []
    for w in wantedAll where workloads[w] == nil && mediaWorkloads[w] == nil { throw CLIError.usage("unknown workload \(w)") }
    for w in wanted {
        guard let wl = workloads[w], let record = byID[wl.record] else { throw CLIError.usage("unknown workload \(w)") }
        let all = try recordRows(d1, record, .text).map(\.row)
        let rows = try wl.qids.map { q in
            guard let r = all.first(where: { $0.qid == q }) else { throw CLIError.failed("\(wl.record): no question \(q)") }
            return r
        }
        cases.append((wl, rows, try rows.map { try d1.bucket(for: $0) }.max()!))
    }
    // load the AOT and the JIT graph of every bucket needed (--cold: this process's cache entries for them removed
    // first, so each first load specializes / registers here)
    var graphs: [String: DecisionGraph] = [:]
    var loads: [(String, JSONValue)] = []
    var evicted: [JSONValue] = []
    func evict(_ asset: URL) throws {
        if args.flags.contains("--cold"), let h = mainHash(asset) {
            let entry = cacheDir().appendingPathComponent(h)
            if FileManager.default.fileExists(atPath: entry.path) {
                let bytes = directoryBytes(entry)
                try FileManager.default.removeItem(at: entry)
                evicted.append(.obj([("entry", .string(entry.path)), ("bytes", .int(bytes))]))
            }
        }
    }
    // the media graphs of the media workloads: AOT and JIT, loaded twice each like the decision graphs
    var mediaGraphs: [String: MediaGraph] = [:]
    for c in mediaCases {
        let folder = c.mediaFolder
        let jit = folder.appendingPathComponent(try readJSON(folder.appendingPathComponent("metadata.json"))["assets"]?["main"]?.string ?? "")
        let aot = try findAOT(root: root, folder: folder, asset: jit.lastPathComponent)
        for (kind, asset) in [("aot", aot), ("jit", jit)] {
            let fid = "\(c.mediaForm)-swift-\(kind)"
            if mediaGraphs[fid] != nil { continue }
            try evict(asset)
            var each: [JSONValue] = []
            var g: MediaGraph? = nil
            for pass in 0..<2 {
                let t = ContinuousClock.now
                let x = try await c.load(asset)
                let wall = seconds(since: t)
                each.append(.obj([("pass", .int(pass)), ("model_seconds", .double(x.loadSeconds.model)),
                                  ("function_seconds", .double(x.loadSeconds.function)), ("wall_seconds", .double(wall)),
                                  ("footprint_after", footprintJSON())]))
                if pass == 0 { g = x }
            }
            mediaGraphs[fid] = g!
            loads.append((fid, .obj([("asset", .string(asset.path)), ("kind", .string(kind)), ("main_hash", .optional(mainHash(asset))),
                                     ("options", .string(describe(g!.options))), ("loads", .array(each))])))
            log("\(fid): load \(String(format: "%.2f", each[0]["wall_seconds"]?.double ?? 0)) s, again "
                + "\(String(format: "%.3f", each[1]["wall_seconds"]?.double ?? 0)) s")
        }
    }
    for L in Set(cases.map(\.2) + mediaCases.map(\.bucket)).sorted() {
        let folder = folders[L]!
        let meta = try readJSON(folder.appendingPathComponent("metadata.json"))
        let jit = folder.appendingPathComponent(meta["assets"]?["main"]?.string ?? "")
        let aot = try findAOT(root: root, folder: folder, asset: jit.lastPathComponent)
        for (kind, asset) in [("aot", aot), ("jit", jit)] {
            try evict(asset)
            var each: [JSONValue] = []
            var g: DecisionGraph? = nil
            for pass in 0..<2 {   // the first load (cold with --cold), then a second one in the same process
                let t = ContinuousClock.now
                let x = try await DecisionGraph(contentsOf: asset, length: L)
                let wall = seconds(since: t)
                each.append(.obj([("pass", .int(pass)), ("model_seconds", .double(x.loadSeconds.model)),
                                  ("function_seconds", .double(x.loadSeconds.function)), ("wall_seconds", .double(wall)),
                                  ("footprint_after", footprintJSON())]))
                if pass == 0 { g = x }
            }
            let fid = "dec-fp16-L\(L)-swift-\(kind)"
            graphs[fid] = g!
            loads.append((fid, .obj([("asset", .string(asset.path)), ("kind", .string(kind)), ("main_hash", .optional(mainHash(asset))),
                                     ("options", .string(describe(g!.options))), ("loads", .array(each))])))
            log("\(fid): load \(String(format: "%.2f", each[0]["wall_seconds"]?.double ?? 0)) s, again "
                + "\(String(format: "%.3f", each[1]["wall_seconds"]?.double ?? 0)) s")
        }
    }
    let fpLoaded = footprintJSON()
    // the oracle rows (the reference's oracle_probs) for the output check
    var oracleP: [String: [Double]] = [:]
    var nearTie: [String: Bool] = [:]
    for r in reference["rows"]?.array ?? [] where r["mode"]?.string == "text" {
        if let id = r["id"]?.string, let q = r["qid"]?.string {
            oracleP["\(id)/\(q)"] = r["oracle_probs"]?.array?.compactMap { $0.double }
            nearTie["\(id)/\(q)"] = r["near_tie"]?.boolValue
        }
    }
    var results: [(String, JSONValue)] = []
    for (wl, rows, L) in cases {
        let forms = ["dec-fp16-L\(L)-swift-aot", "dec-fp16-L\(L)-swift-jit"]
        var samples: [String: [Double]] = [:], warm: [String: [Double]] = [:]
        var parts: [String: [(Double, Double, Double)]] = [:]
        var firstLogits: [[Float]]? = nil
        var allEqual = true
        var worstDp = 0.0
        var argmaxOK = true
        let tw = ContinuousClock.now
        for r in 0..<rounds {
            for fid in (r % 2 == 0 ? forms : forms.reversed()) {
                let g = graphs[fid]!
                let t = ContinuousClock.now
                var logits: [[Float]] = []
                var probs: [[Float]] = []
                var pi = 0.0, pg = 0.0, pr = 0.0
                for row in rows {
                    let a = ContinuousClock.now
                    let x = try GraphInputs(row: row, length: L)
                    let b = ContinuousClock.now
                    let s = try await g.scores(x)
                    let c = ContinuousClock.now
                    let z = x.markers.map { s[$0] }
                    probs.append(Readout.probabilities(logits: z, question: row.question, calibrate: row.calibrate, config: d1.config))
                    let e = ContinuousClock.now
                    logits.append(z)
                    pi += D1OmniTime.seconds(a, b)
                    pg += D1OmniTime.seconds(b, c)
                    pr += D1OmniTime.seconds(c, e)
                }
                let ms = seconds(since: t) * 1e3
                if r < warmup { warm[fid, default: []].append(ms) } else {
                    samples[fid, default: []].append(ms)
                    parts[fid, default: []].append((pi * 1e3, pg * 1e3, pr * 1e3))
                }
                if let f = firstLogits {
                    if zip(f, logits).contains(where: { $0.map(\.bitPattern) != $1.map(\.bitPattern) }) { allEqual = false }
                } else {
                    firstLogits = logits
                }
                for (row, p) in zip(rows, probs) {
                    guard let o = oracleP["\(wl.record)/\(row.qid)"] else { continue }
                    let dp = zip(p, o).map { abs(Double($0) - $1) }.max() ?? .nan
                    worstDp = max(worstDp, dp)
                    let am = p.indices.max { p[$0] < p[$1] || (p[$0] == p[$1] && $0 > $1) } ?? 0
                    let ao = o.indices.max { o[$0] < o[$1] || (o[$0] == o[$1] && $0 > $1) } ?? 0
                    if am != ao && nearTie["\(wl.record)/\(row.qid)"] != true { argmaxOK = false }
                }
            }
        }
        let loopS = seconds(since: tw)
        // tokenizing, apart
        var tok: [Double] = []
        let record = byID[wl.record]!
        for i in 0..<(warmup + rounds - warmup) {
            let t = ContinuousClock.now
            _ = try recordRows(d1, record, .text)
            if i >= warmup { tok.append(seconds(since: t) * 1e3) }
        }
        var formsOut: [(String, JSONValue)] = []
        for fid in forms {
            let ps = parts[fid] ?? []
            func med(_ k: KeyPath<(Double, Double, Double), Double>) -> Double {
                let v = ps.map { $0[keyPath: k] }.sorted()
                return v.isEmpty ? .nan : (v.count % 2 == 1 ? v[v.count / 2] : (v[v.count / 2 - 1] + v[v.count / 2]) / 2)
            }
            formsOut.append((fid, .obj([("stats_ms", stats(samples[fid] ?? [])), ("samples_ms", .doubles(samples[fid] ?? [])),
                                        ("warmup_ms", .doubles(warm[fid] ?? [])),
                                        ("parts_median_ms", .obj([("inputs_ms", .double(med(\.0))), ("graph_ms", .double(med(\.1))),
                                                                  ("readout_ms", .double(med(\.2)))]))])))
            let st = stats(samples[fid] ?? [])
            log("\(wl.id) \(fid): median \(String(format: "%.2f", st["median"]?.double ?? 0)) ms "
                + "(p10 \(String(format: "%.2f", st["p10"]?.double ?? 0)), p90 \(String(format: "%.2f", st["p90"]?.double ?? 0)))")
        }
        results.append((wl.id, .obj([
            ("what", .string(wl.what)), ("record", .string(wl.record)), ("qids", .strings(wl.qids)), ("bucket", .int(L)),
            ("positions", .ints(rows.map(\.positions))), ("forms", .obj(formsOut)), ("form_order_even_rounds", .strings(forms)),
            ("output_check", .obj([("jit_and_aot_logits_bit_equal_every_call", .bool(allEqual)),
                                   ("max_abs_dp_vs_oracle", .double(worstDp)), ("argmax_equal_every_call", .bool(argmaxOK)),
                                   ("first_call_logits", .array((firstLogits ?? []).map { floats($0) })),
                                   ("status", .string(allEqual && argmaxOK && worstDp <= 0.02 ? "PASS" : "FAIL"))])),
            ("tokenize_ms", stats(tok)), ("loop_seconds", .double(loopS)), ("footprint_after", footprintJSON()),
        ])))
    }
    // the media workloads: the oracle rows of the Python host's media dump
    var mediaOracle: [String: (probs: [Double], nearTie: Bool)] = [:]
    if !mediaCases.isEmpty {
        let index = try readJSON(url(try args.need("--media-reference")).appendingPathComponent("index.json"))
        for r in index["rows"]?.array ?? [] {
            if let id = r["id"]?.string, let q = r["qid"]?.string, let p = r["oracle_probs"]?.array?.compactMap({ $0.double }) {
                mediaOracle["\(id)/\(q)"] = (p, r["near_tie"]?.boolValue ?? false)
            }
        }
    }
    for c in mediaCases {
        let L = c.bucket
        let forms = ["aot", "jit"].map { (kind: $0, id: "\(c.mediaForm)+dec-fp16-L\(L)-swift-\($0)") }
        var samples: [String: [Double]] = [:], warm: [String: [Double]] = [:]
        var parts: [String: [(Double, Double, Double, Double)]] = [:]
        var firstLogits: [[Float]]? = nil
        var firstPrefixHash: String? = nil
        var allEqual = true, prefixEqual = true
        var worstDp = 0.0
        var argmaxOK = true
        let tw = ContinuousClock.now
        for r in 0..<rounds {
            for f in (r % 2 == 0 ? forms : forms.reversed()) {
                let mg = mediaGraphs["\(c.mediaForm)-swift-\(f.kind)"]!, g = graphs["dec-fp16-L\(L)-swift-\(f.kind)"]!
                let t = ContinuousClock.now
                var prefix: [Float] = []
                for (inputs, keep) in zip(c.calls, c.keep) {
                    let out = try await mg.run(inputs)
                    prefix += out.prefix(keep * DecisionGraph.hidden)
                }
                let tm = ContinuousClock.now
                var logits: [[Float]] = []
                var probs: [[Float]] = []
                var pi = 0.0, pg = 0.0, pr = 0.0
                for row in c.rows {
                    let a = ContinuousClock.now
                    let x = try GraphInputs(row: row, length: L)
                    let b = ContinuousClock.now
                    let s = try await g.scores(x, prefix: prefix)
                    let cc = ContinuousClock.now
                    let z = x.markers.map { s[$0] }
                    probs.append(Readout.probabilities(logits: z, question: row.question, calibrate: row.calibrate, config: d1.config))
                    let e = ContinuousClock.now
                    logits.append(z)
                    pi += D1OmniTime.seconds(a, b)
                    pg += D1OmniTime.seconds(b, cc)
                    pr += D1OmniTime.seconds(cc, e)
                }
                let ms = seconds(since: t) * 1e3
                let media = D1OmniTime.seconds(t, tm) * 1e3
                if r < warmup { warm[f.id, default: []].append(ms) } else {
                    samples[f.id, default: []].append(ms)
                    parts[f.id, default: []].append((media, pi * 1e3, pg * 1e3, pr * 1e3))
                }
                let ph = sha256(of: prefix)
                if let h = firstPrefixHash { if h != ph { prefixEqual = false } } else { firstPrefixHash = ph }
                if let fl = firstLogits {
                    if zip(fl, logits).contains(where: { $0.map(\.bitPattern) != $1.map(\.bitPattern) }) { allEqual = false }
                } else {
                    firstLogits = logits
                }
                for (row, p) in zip(c.rows, probs) {
                    guard let o = mediaOracle["\(c.wl.record)/\(row.qid)"] else { throw CLIError.failed("\(c.wl.record)/\(row.qid): no oracle row") }
                    let dp = zip(p, o.probs).map { abs(Double($0) - $1) }.max() ?? .nan
                    worstDp = max(worstDp, dp)
                    let am = p.indices.max { p[$0] < p[$1] || (p[$0] == p[$1] && $0 > $1) } ?? 0
                    let ao = o.probs.indices.max { o.probs[$0] < o.probs[$1] || (o.probs[$0] == o.probs[$1] && $0 > $1) } ?? 0
                    if am != ao && !o.nearTie { argmaxOK = false }
                }
            }
        }
        let loopS = seconds(since: tw)
        // the preprocessing, apart (warm-up dropped): decode, crops / mel, patches, tokenize
        var pre: [(String, JSONValue)] = []
        for (name, step) in c.preprocess(d1) {
            var v: [Double] = []
            for i in 0..<rounds {
                let t = ContinuousClock.now
                try step()
                if i >= warmup { v.append(seconds(since: t) * 1e3) }
            }
            pre.append((name, stats(v)))
        }
        var formsOut: [(String, JSONValue)] = []
        for f in forms {
            let ps = parts[f.id] ?? []
            func med(_ k: KeyPath<(Double, Double, Double, Double), Double>) -> Double {
                let v = ps.map { $0[keyPath: k] }.sorted()
                return v.isEmpty ? .nan : (v.count % 2 == 1 ? v[v.count / 2] : (v[v.count / 2 - 1] + v[v.count / 2]) / 2)
            }
            formsOut.append((f.id, .obj([("stats_ms", stats(samples[f.id] ?? [])), ("samples_ms", .doubles(samples[f.id] ?? [])),
                                         ("warmup_ms", .doubles(warm[f.id] ?? [])),
                                         ("parts_median_ms", .obj([("media_ms", .double(med(\.0))), ("inputs_ms", .double(med(\.1))),
                                                                   ("graph_ms", .double(med(\.2))), ("readout_ms", .double(med(\.3)))]))])))
            let st = stats(samples[f.id] ?? [])
            log("\(c.wl.id) \(f.id): median \(String(format: "%.2f", st["median"]?.double ?? 0)) ms "
                + "(p10 \(String(format: "%.2f", st["p10"]?.double ?? 0)), p90 \(String(format: "%.2f", st["p90"]?.double ?? 0)))")
        }
        results.append((c.wl.id, .obj([
            ("what", .string(c.wl.what)), ("record", .string(c.wl.record)), ("mode", .string(c.wl.mode.rawValue)),
            ("qids", .strings(c.wl.qids)), ("file", .string(c.file.path)), ("bucket", .int(L)),
            ("media", c.describe), ("positions", .ints(c.rows.map(\.positions))), ("prefix", .ints(c.rows.map(\.prefixLength))),
            ("forms", .obj(formsOut)), ("form_order_even_rounds", .strings(forms.map(\.id))),
            ("output_check", .obj([("jit_and_aot_logits_bit_equal_every_call", .bool(allEqual)),
                                   ("jit_and_aot_prefix_bit_equal_every_call", .bool(prefixEqual)),
                                   ("max_abs_dp_vs_oracle", .double(worstDp)), ("argmax_equal_every_call", .bool(argmaxOK)),
                                   ("first_call_logits", .array((firstLogits ?? []).map { floats($0) })),
                                   ("status", .string(allEqual && prefixEqual && argmaxOK && worstDp <= 0.02 ? "PASS" : "FAIL"))])),
            ("preprocess_ms", .obj(pre)), ("loop_seconds", .double(loopS)), ("footprint_after", footprintJSON()),
        ])))
    }
    let doc: JSONValue = .obj([
        ("schema", .string("d1omni-time/1")), ("label", .optional(args.one("--label"))), ("started", .string(started)),
        ("finished", .string(now())), ("rounds", .int(rounds)), ("warmup", .int(warmup)), ("cold", .bool(args.flags.contains("--cold"))),
        ("init_seconds", .double(initS)), ("tokenizer_load_seconds", .double(d1.tokenizer.loadSeconds)),
        ("evicted_cache_entries", .array(evicted)), ("loads", .obj(loads)), ("workloads", .obj(results)),
        ("footprint_start", fp0), ("footprint_after_loads", fpLoaded), ("footprint_end", footprintJSON()),
        ("cache_before", cacheBefore), ("cache_after", cacheState()), ("environment", environment()),
        ("one_decision", .string("graph inputs (GraphInputs: ids, masks, qtype; prefix_embeds = the graph's zero array) -> "
            + "InferenceFunction.run -> scores copied (ND.read) -> marker gather + temperature + float32 softmax; a W2 decision "
            + "is its 3 rows; tokenizing timed apart. W4 / W5: first the media graph on inputs prepared beforehand (NDArrays made "
            + "in the call), its P rows cut and concatenated, then the decision as above with that prefix; decode, crops, "
            + "patches + positions, mel + masks and tokenize timed apart")),
    ])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("time: \(url(try args.need("--out")).path)")
}

struct MediaWorkload {
    let id: String
    let what: String
    let record: String
    let qids: [String]
    let mode: Mode
}

let mediaWorkloads: [String: MediaWorkload] = [
    "W4": MediaWorkload(id: "W4", what: "one question about a 384 px image (1 crop)", record: "img_01", qids: ["room"], mode: .image),
    "W5": MediaWorkload(id: "W5", what: "one question about 10 s of audio", record: "aud_01", qids: ["topic"], mode: .audio),
]

/// One media workload: its file, its rows, the media graph's inputs prepared once (one entry per call) and the rows
/// each call keeps, the decision bucket.
struct MediaCase {
    let wl: MediaWorkload
    let file: URL
    let rows: [Row]
    let bucket: Int
    let mediaFolder: URL
    let mediaForm: String
    let calls: [[String: MediaGraph.Input]]
    let keep: [Int]
    let audioBucket: AudioBucket?
    let describe: JSONValue
    let record: JSONValue

    init(_ wl: MediaWorkload, d1: D1Omni, byID: [String: JSONValue], root: URL) throws {
        guard let record = byID[wl.record], let req = record["request"], let qs = req["questions"], let media = record["media"] else {
            throw CLIError.failed("\(wl.record): no media record")
        }
        self.wl = wl
        self.record = record
        let all: [Row]
        if wl.mode == .image {
            guard let rel = media["images"]?.array?.first?.string, let folder = d1.media?.vision else { throw CLIError.failed("\(wl.record): no image / vision folder") }
            file = root.appendingPathComponent(rel)
            let rgb = try ImagePreprocess.decode(contentsOf: file)
            let crops = try d1.imageInputs([rgb])
            calls = crops.map { ["pixel_values": .float($0.pixelValues), "pos_embed": .float($0.posEmbed),
                                 "patch_mask": .float($0.patchMask), "unshuffle_index": .int32($0.unshuffleIndex)] }
            keep = crops.map(\.tokens)
            mediaFolder = folder
            mediaForm = "vis-fp16"
            audioBucket = nil
            let p = try MediaLength.imagePrefixLength([(rgb.width, rgb.height)])
            all = try d1.rows(state: req["state"], questions: qs, mode: .image, prefixLength: p)
            describe = .obj([("px_wh", .ints([rgb.width, rgb.height])), ("crops", .int(crops.count)), ("tokens", .ints(keep))])
        } else {
            guard let rel = media["audio"]?.string else { throw CLIError.failed("\(wl.record): no audio") }
            file = root.appendingPathComponent(rel)
            let samples = try AudioPreprocess.samples(contentsOf: file)
            let x = try d1.audioInputs(samples: samples)
            guard let folder = d1.media?.audio[x.bucket.seconds] else { throw CLIError.failed("no audio folder for \(x.bucket.seconds) s") }
            calls = [["mel": .float(x.mel), "mask_f": .float(x.maskF), "mask_f2": .float(x.maskF2), "mask_f4": .float(x.maskF4),
                      "mask_t": .float(x.maskT)]]
            keep = [x.prefixRows]
            mediaFolder = folder
            mediaForm = "aud-fp16-\(x.bucket.seconds)s"
            audioBucket = x.bucket
            all = try d1.rows(state: req["state"], questions: qs, mode: .audio, prefixLength: x.prefixRows)
            describe = .obj([("samples", .int(samples.count)), ("seconds", .double(Double(samples.count) / 16000)),
                             ("bucket_s", .int(x.bucket.seconds)), ("prefix_rows", .int(x.prefixRows))])
        }
        rows = try wl.qids.map { q in
            guard let r = all.first(where: { $0.qid == q }) else { throw CLIError.failed("\(wl.record): no question \(q)") }
            return r
        }
        bucket = try rows.map { try d1.bucket(for: $0) }.max()!
    }

    func load(_ asset: URL) async throws -> MediaGraph {
        if let b = audioBucket { return try await AudioGraph(contentsOf: asset, bucket: b).graph }
        return try await VisionGraph(contentsOf: asset).graph
    }

    /// What a host does before its first media call, step by step (timed apart).
    func preprocess(_ d1: D1Omni) -> [(String, () throws -> Void)] {
        let req = record["request"]!
        if wl.mode == .image {
            let rgb = try? ImagePreprocess.decode(contentsOf: file)
            let crops = rgb.flatMap { try? ImagePreprocess.crops($0) } ?? []
            let table = (try? d1.visionPositionTable()) ?? []
            let p = rgb.flatMap { try? MediaLength.imagePrefixLength([($0.width, $0.height)]) } ?? 0
            return [("file_decode_png", { _ = try ImagePreprocess.decode(contentsOf: file) }),
                    ("crops_resize", { _ = try ImagePreprocess.crops(rgb!) }),
                    ("patches_positions", { _ = try crops.map { try ImagePreprocess.inputs($0, table: table) } }),
                    ("tokenize", { _ = try d1.rows(state: req["state"], questions: req["questions"]!, mode: .image, prefixLength: p) })]
        }
        let samples = (try? AudioPreprocess.samples(contentsOf: file)) ?? []
        return [("file_decode_wav", { _ = try AudioPreprocess.samples(contentsOf: file) }),
                ("mel_and_masks", { _ = try d1.audioInputs(samples: samples) }),
                ("tokenize", { _ = try d1.rows(state: req["state"], questions: req["questions"]!, mode: .audio,
                                               prefixLength: MediaLength.audioPrefixLength(samples: samples.count)) })]
    }
}

enum D1OmniTime {
    static func seconds(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
        let d = b - a
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}

// MARK: - .npy (the Python host's dump)

/// A C-order little-endian .npy array (format 1.0 / 2.0 / 3.0): '<f4', '<i4', '<i2', '|u1'.
struct NPY {
    let descr: String
    let shape: [Int]
    let payload: Data

    init(_ u: URL) throws {
        let d = try Data(contentsOf: u)
        let b = [UInt8](d.prefix(12))
        guard b.count >= 10, b[0] == 0x93, String(decoding: b[1..<6], as: UTF8.self) == "NUMPY" else {
            throw CLIError.failed("\(u.path): not a .npy file")
        }
        let headerLength: Int, start: Int
        if b[6] == 1 {
            headerLength = Int(b[8]) | Int(b[9]) << 8
            start = 10
        } else {
            headerLength = Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24
            start = 12
        }
        let header = String(decoding: d[start..<(start + headerLength)], as: UTF8.self)
        func field(_ key: String) -> String? {
            guard let r = header.range(of: "'\(key)': ") else { return nil }
            return String(header[r.upperBound...])
        }
        guard let ds = field("descr"), let q1 = ds.firstIndex(of: "'"), let q2 = ds[ds.index(after: q1)...].firstIndex(of: "'"),
              let fo = field("fortran_order"), fo.hasPrefix("False"), let sh = field("shape"),
              let p1 = sh.firstIndex(of: "("), let p2 = sh.firstIndex(of: ")") else {
            throw CLIError.failed("\(u.path): header \(header)")
        }
        descr = String(ds[ds.index(after: q1)..<q2])
        shape = sh[sh.index(after: p1)..<p2].split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        payload = d.subdata(in: (start + headerLength)..<d.count)
        let item = ["<f4": 4, "<i4": 4, "<i2": 2, "|u1": 1][descr] ?? 0
        guard item > 0, payload.count == shape.reduce(1, *) * item else {
            throw CLIError.failed("\(u.path): \(descr) \(shape) with \(payload.count) bytes")
        }
    }

    var count: Int { shape.reduce(1, *) }
    func floats() -> [Float] { payload.withUnsafeBytes { r in (0..<count).map { Float(bitPattern: r.loadUnaligned(fromByteOffset: 4 * $0, as: UInt32.self)) } } }
    func int32s() -> [Int32] { payload.withUnsafeBytes { r in (0..<count).map { r.loadUnaligned(fromByteOffset: 4 * $0, as: Int32.self) } } }
    func int16s() -> [Int16] { payload.withUnsafeBytes { r in (0..<count).map { r.loadUnaligned(fromByteOffset: 2 * $0, as: Int16.self) } } }
    func uint8s() -> [UInt8] { [UInt8](payload) }
}

/// Float arrays against a reference: bit-equal, the elements that differ, max |d| (float64), the largest ulp step.
func compareFloats(_ a: [Float], _ b: [Float]) -> JSONValue {
    guard a.count == b.count else { return .obj([("bit_equal", .bool(false)), ("count", .ints([a.count, b.count]))]) }
    var diff = 0, maxAbs = 0.0, maxUlp: Int64 = 0, at = -1
    for i in 0..<a.count where a[i].bitPattern != b[i].bitPattern {
        diff += 1
        let d = abs(Double(a[i]) - Double(b[i]))
        if d > maxAbs { maxAbs = d; at = i }
        func ordered(_ x: Float) -> Int64 { let v = Int64(Int32(bitPattern: x.bitPattern)); return v < 0 ? Int64(Int32.min) - v : v }
        maxUlp = max(maxUlp, abs(ordered(a[i]) - ordered(b[i])))
    }
    return .obj([("bit_equal", .bool(diff == 0)), ("elems", .int(a.count)), ("diff_elems", .int(diff)),
                 ("max_abs", .double(maxAbs)), ("max_abs_at", .int(at)), ("max_ulp", .int(Int(maxUlp)))])
}

func compareBytes<T: FixedWidthInteger>(_ a: [T], _ b: [T]) -> JSONValue {
    guard a.count == b.count else { return .obj([("bit_equal", .bool(false)), ("count", .ints([a.count, b.count]))]) }
    var diff = 0, maxD = 0
    for i in 0..<a.count where a[i] != b[i] {
        diff += 1
        maxD = max(maxD, abs(Int(a[i]) - Int(b[i])))
    }
    return .obj([("bit_equal", .bool(diff == 0)), ("elems", .int(a.count)), ("diff_elems", .int(diff)), ("max_diff", .int(maxD))])
}

/// np.resize of a [rows * 1024] prefix to `p` rows: the flattened rows repeated (or cut) to fill.
func resizedPrefix(_ x: [Float], rows p: Int) -> [Float] {
    let n = p * DecisionGraph.hidden
    return (0..<n).map { x[$0 % x.count] }
}

// MARK: - media setup (round 9)

/// The media folders under --bundle-dir and, for --asset aot, the Mac AOT of each media bundle under --aot-root.
func mediaSetup(_ args: Args, kind: String) throws -> D1Omni.MediaFolders {
    let dir = url(try args.need("--bundle-dir"))
    let found = D1Omni.MediaFolders.find(macos: dir)
    guard kind == "aot" else { return found }
    let root = url(try args.need("--aot-root"))
    var assets: [String: URL] = [:]
    if let v = found.vision {
        let main = try readJSON(v.appendingPathComponent("metadata.json"))["assets"]?["main"]?.string ?? ""
        assets["vision"] = try findAOT(root: root, folder: v, asset: main)
    }
    for (sec, f) in found.audio {
        let main = try readJSON(f.appendingPathComponent("metadata.json"))["assets"]?["main"]?.string ?? ""
        assets["audio-\(sec)"] = try findAOT(root: root, folder: f, asset: main)
    }
    return D1Omni.MediaFolders(vision: found.vision, audio: found.audio, assets: assets)
}

func graphLoadJSON(_ g: MediaGraph, wall: Double) -> JSONValue {
    .obj([("asset", .string(g.url.path)), ("kind", .string(g.kind)), ("model_seconds", .double(g.loadSeconds.model)),
          ("function_seconds", .double(g.loadSeconds.function)), ("wall_seconds", .double(wall)),
          ("main_hash", .optional(mainHash(g.url))), ("options", .string(describe(g.options))), ("descriptor", g.descriptor),
          ("footprint_after", footprintJSON())])
}

// MARK: - parity-media

func parityMediaCommand(_ args: Args) async throws {
    let setup = try Setup(args)
    let media = try mediaSetup(args, kind: setup.kind)
    let repeats = max(1, args.int("--repeats", 2))
    let refDir = url(try args.need("--media-reference"))
    let root = url(try args.need("--media-root"))
    let fixtures = url(try args.need("--fixtures"))
    let index = try readJSON(refDir.appendingPathComponent("index.json"))
    let byID = Dictionary(try records(fixtures).compactMap { r in r["id"]?.string.map { ($0, r) } }, uniquingKeysWith: { a, _ in a })
    let cacheBefore = cacheState()
    let fp0 = footprintJSON()
    let t0 = ContinuousClock.now
    let d1 = try await D1Omni(folders: setup.folders, assets: setup.assets, media: media)
    let initS = seconds(since: t0)
    func npy(_ f: JSONValue?) throws -> NPY {
        guard let name = f?["file"]?.string else { throw CLIError.failed("index.json: a file entry without a name") }
        return try NPY(refDir.appendingPathComponent(name))
    }
    // the Python rows: (id, mode, qid, arm) -> row
    var pyRows: [String: JSONValue] = [:]
    for r in index["rows"]?.array ?? [] {
        if let id = r["id"]?.string, let m = r["mode"]?.string, let q = r["qid"]?.string, let a = r["arm"]?.string {
            pyRows["\(id)/\(m)/\(q)/\(a)"] = r
        }
    }
    var loads: [(String, JSONValue)] = []
    func loaded(_ key: String, _ g: MediaGraph, _ tl: ContinuousClock.Instant) {
        if !loads.contains(where: { $0.0 == key }) {
            loads.append((key, graphLoadJSON(g, wall: seconds(since: tl))))
            log("loaded \(key) \(g.kind) in \(String(format: "%.2f", seconds(since: tl))) s")
        }
    }
    var outRows: [JSONValue] = []
    var worst = (rowsBit: 0, rows: 0, drift: 0.0)

    /// The rows of one media record, decided on `prefix`, against the Python rows of `arm`; plus the control prefix.
    func decideRows(_ id: String, _ mode: Mode, _ prefix: [Float], prefixRows p: Int, pyArm: String, control: [Float],
                    controlDonor: String) async throws {
        guard let record = byID[id], let req = record["request"], let qs = req["questions"] else {
            throw CLIError.failed("\(id): no fixture record")
        }
        let rows = try d1.rows(state: req["state"], questions: qs, mode: mode, prefixLength: p)
        for row in rows {
            guard let py = pyRows["\(id)/\(mode.rawValue)/\(row.qid)/\(pyArm)"] else { throw CLIError.failed("\(id)/\(row.qid): no Python row") }
            let pyIDs = py["ids"]?.array?.compactMap { $0.intValue } ?? []
            guard row.ids == pyIDs, row.prefixLength == py["prefix_len"]?.intValue else {
                throw CLIError.failed("\(id)/\(row.qid)/\(mode.rawValue): the Swift row differs from the Python row (ids or P)")
            }
            let L = try d1.bucket(for: row)
            if d1.loaded[L] == nil {
                let tl = ContinuousClock.now
                let g = try await d1.graph(L)
                loads.append(("decide_L\(L)", .obj([("asset", .string(g.url.path)), ("kind", .string(g.kind)),
                                                     ("model_seconds", .double(g.loadSeconds.model)),
                                                     ("function_seconds", .double(g.loadSeconds.function)),
                                                     ("wall_seconds", .double(seconds(since: tl))), ("main_hash", .optional(mainHash(g.url))),
                                                     ("options", .string(describe(g.options))), ("footprint_after", footprintJSON())])))
                log("loaded L\(L) \(g.kind) in \(String(format: "%.2f", seconds(since: tl))) s")
            }
            var decisions: [D1Omni.Decision] = []
            for _ in 0..<repeats { decisions.append(try await d1.decide(row, prefix: prefix, bucket: L)) }
            let first = decisions[0]
            let drift = decisions.dropFirst().map { d in zip(d.logits, first.logits).map { abs(Double($0) - Double($1)) }.max() ?? 0 }.max() ?? 0
            let pyLogits = py["logits"]?.array?.compactMap { $0.double }.map(Float.init) ?? []
            let pyProbs = py["probs"]?.array?.compactMap { $0.double }.map(Float.init) ?? []
            let same = first.logits.map(\.bitPattern) == pyLogits.map(\.bitPattern) && first.probabilities.map(\.bitPattern) == pyProbs.map(\.bitPattern)
            worst.rows += 1
            worst.rowsBit += same ? 1 : 0
            worst.drift = max(worst.drift, drift)
            let ctl = try await d1.decide(row, prefix: control, bucket: L)
            outRows.append(.obj([
                ("id", .string(id)), ("qid", .string(row.qid)), ("mode", .string(mode.rawValue)), ("bucket", .int(L)),
                ("positions", .int(row.positions)), ("prefix", .int(row.prefixLength)), ("python_arm", .string(pyArm)),
                ("logits", floats(first.logits)), ("logits_bits", bits(first.logits)),
                ("probs", floats(first.probabilities)), ("probs_bits", bits(first.probabilities)),
                ("python_bit_equal", .bool(same)),
                ("python_max_abs_dlogit", .double(zip(first.logits, pyLogits).map { abs(Double($0) - Double($1)) }.max() ?? .nan)),
                ("python_max_abs_dp", .double(zip(first.probabilities, pyProbs).map { abs(Double($0) - Double($1)) }.max() ?? .nan)),
                ("repeats", .int(repeats)), ("drift_marker_logits", .double(drift)),
                ("control_donor", .string(controlDonor)), ("control_logits", floats(ctl.logits)),
                ("call_ms", .doubles(decisions.map { $0.seconds.graph * 1e3 }))]))
        }
    }

    // ---------------------------------------------------------------- images
    let images = index["images"]?.array ?? []
    var imageOut: [JSONValue] = []
    var imagePrefixes: [String: (prefix: [Float], rows: Int)] = [:]
    var imageOrder: [String] = []
    for img in images {
        guard let id = img["id"]?.string, let file = img["file"]?.string else { throw CLIError.failed("index.json: an image without id / file") }
        let path = root.appendingPathComponent(file)
        var times: [(String, JSONValue)] = []
        var t = ContinuousClock.now
        let rgb = try ImagePreprocess.decode(contentsOf: path)
        times.append(("decode_ms", .double(seconds(since: t) * 1e3)))
        let refRGB = try npy(img["files"]?["rgb"])
        let rgbCheck = compareBytes(rgb.pixels, refRGB.uint8s())
        t = ContinuousClock.now
        let crops = try ImagePreprocess.crops(rgb)
        times.append(("crops_resize_ms", .double(seconds(since: t) * 1e3)))
        t = ContinuousClock.now
        let table = try d1.visionPositionTable()
        let inputs = try crops.map { try ImagePreprocess.inputs($0, table: table) }
        times.append(("patches_positions_ms", .double(seconds(since: t) * 1e3)))
        let refCrops = img["crops"]?.array ?? []
        guard refCrops.count == crops.count else { throw CLIError.failed("\(id): \(crops.count) crops, the dump has \(refCrops.count)") }
        let tl = ContinuousClock.now
        let vg = try await d1.vision()
        loaded("vision", vg.graph, tl)
        var cropOut: [JSONValue] = []
        var prefix: [Float] = []
        var drift = 0.0
        for (k, (x, ref)) in zip(inputs, refCrops).enumerated() {
            let f = ref["files"]
            let refCrop = try npy(f?["crop"])
            let cropCheck = compareBytes(crops[k].values, refCrop.uint8s())
            let refPV = try npy(f?["pixel_values"]), refPos = try npy(f?["pos_embed"])
            let refMask = try npy(f?["patch_mask"]), refIdx = try npy(f?["unshuffle_index"])
            var outs: [[Float]] = []
            for _ in 0..<repeats { outs.append(try await vg.output(x)) }
            drift = max(drift, zip(outs.last!, outs[0]).map { abs(Double($0) - Double($1)) }.max() ?? 0)
            prefix += outs[0].prefix(x.tokens * DecisionGraph.hidden)
            // arm B: the dump's own inputs through the same graph (the runtime call alone)
            let refInputs = CropInputs(grid: x.grid, tokens: x.tokens, pixelValues: refPV.floats(), posEmbed: refPos.floats(),
                                       patchMask: refMask.floats(), unshuffleIndex: refIdx.int32s())
            let armB = try await vg.output(refInputs)
            cropOut.append(.obj([
                ("k", .int(k)), ("size_hw", .ints([crops[k].height, crops[k].width])), ("grid", .ints([x.grid.rows, x.grid.columns])),
                ("tokens", .int(x.tokens)), ("crop_u8", cropCheck),
                ("pixel_values", compareFloats(x.pixelValues, refPV.floats())), ("pos_embed", compareFloats(x.posEmbed, refPos.floats())),
                ("patch_mask", compareFloats(x.patchMask, refMask.floats())),
                ("unshuffle_index", compareBytes(x.unshuffleIndex, refIdx.int32s())),
                ("graph_output_sha256", .string(sha256(of: outs[0]))),
                ("python_graph_output_sha256", .optional(f?["graph_output_sha256"]?.string)),
                ("arm_b_graph_output_sha256", .string(sha256(of: armB))),
                ("arm_b_equal_python", .bool(sha256(of: armB) == f?["graph_output_sha256"]?.string))]))
        }
        let refPrefix = try npy(img["files"]?["prefix"])
        let p = prefix.count / DecisionGraph.hidden
        imagePrefixes[id] = (prefix, p)
        imageOrder.append(id)
        imageOut.append(.obj([("id", .string(id)), ("file", .string(file)), ("px_wh", .ints([rgb.width, rgb.height])),
                              ("rgb", rgbCheck), ("crops", .array(cropOut)), ("prefix_rows", .int(p)),
                              ("prefix", compareFloats(prefix, refPrefix.floats())), ("vision_drift", .double(drift)),
                              ("times", .obj(times))]))
        log("\(id): rgb \(rgbCheck["bit_equal"]?.boolValue == true ? "bit-equal" : "DIFFERS"), \(crops.count) crops, P \(p), prefix "
            + "\(compareFloats(prefix, refPrefix.floats())["bit_equal"]?.boolValue == true ? "bit-equal" : "differs")")
    }
    for (i, id) in imageOrder.enumerated() {
        let donor = imageOrder[(i + 1) % imageOrder.count]
        let (prefix, p) = imagePrefixes[id]!
        try await decideRows(id, .image, prefix, prefixRows: p, pyArm: "python",
                             control: resizedPrefix(imagePrefixes[donor]!.prefix, rows: p), controlDonor: donor)
    }

    // ---------------------------------------------------------------- audio
    let clips = index["clips"]?.array ?? []
    var clipOut: [JSONValue] = []
    var clipPrefixes: [String: (prefix: [Float], rows: Int)] = [:]
    var clipOrder: [String] = []
    for clip in clips {
        guard let id = clip["id"]?.string, let file = clip["file"]?.string else { throw CLIError.failed("index.json: a clip without id / file") }
        let path = root.appendingPathComponent(file)
        var times: [(String, JSONValue)] = []
        var t = ContinuousClock.now
        let refSamples = try npy(clip["files"]?["samples"]).int16s()
        let samples: [Int16]
        let source: String
        if path.pathExtension.lowercased() == "wav" {
            samples = try AudioPreprocess.samples(contentsOf: path)
            source = "wav (Swift RIFF parser)"
        } else {
            samples = refSamples   // flac: the dump's int16 samples (soundfile), no Swift flac decoder
            source = "the dump's samples.npy (soundfile int16; no Swift flac decoder)"
        }
        times.append(("decode_ms", .double(seconds(since: t) * 1e3)))
        t = ContinuousClock.now
        let x = try d1.audioInputs(samples: samples)
        times.append(("mel_masks_ms", .double(seconds(since: t) * 1e3)))
        let f = clip["files"]
        let refMel = try npy(f?["mel_numpy"]).floats(), refMelTorch = try npy(f?["mel_torch"]).floats()
        let refMasks = try ["mask_f", "mask_f2", "mask_f4", "mask_t"].map { try npy(f?[$0]).floats() }
        let tl = ContinuousClock.now
        let ag = try await d1.audio(x.bucket)
        loaded("audio_\(x.bucket.seconds)s", ag.graph, tl)
        var outs: [[Float]] = []
        for _ in 0..<repeats { outs.append(try await ag.output(x)) }
        let drift = zip(outs.last!, outs[0]).map { abs(Double($0) - Double($1)) }.max() ?? 0
        let prefix = Array(outs[0].prefix(x.prefixRows * DecisionGraph.hidden))
        // arm B: the dump's NumPy-mel inputs through the same graph
        let refInputs = AudioInputs(bucket: x.bucket, mel: refMel, maskF: refMasks[0], maskF2: refMasks[1], maskF4: refMasks[2],
                                    maskT: refMasks[3], samples: x.samples, frames: x.frames, columns: x.columns, prefixRows: x.prefixRows)
        let armB = try await ag.output(refInputs)
        let refPrefix = try npy(f?["prefix_numpy"]).floats(), refPrefixTorch = try npy(f?["prefix_torch"]).floats()
        clipPrefixes[id] = (prefix, x.prefixRows)
        clipOrder.append(id)
        let melCheck = compareFloats(x.mel, refMel)
        clipOut.append(.obj([
            ("id", .string(id)), ("file", .string(file)), ("samples_from", .string(source)), ("samples", compareBytes(samples, refSamples)),
            ("bucket_s", .int(x.bucket.seconds)), ("frames", .int(x.frames)), ("prefix_rows", .int(x.prefixRows)),
            ("mel_vs_numpy", melCheck), ("mel_vs_torch", compareFloats(x.mel, refMelTorch)),
            ("mask_f", compareFloats(x.maskF, refMasks[0])), ("mask_f2", compareFloats(x.maskF2, refMasks[1])),
            ("mask_f4", compareFloats(x.maskF4, refMasks[2])), ("mask_t", compareFloats(x.maskT, refMasks[3])),
            ("prefix_vs_numpy", compareFloats(prefix, refPrefix)), ("prefix_vs_torch", compareFloats(prefix, refPrefixTorch)),
            ("graph_output_sha256", .string(sha256(of: outs[0]))),
            ("python_graph_output_numpy_sha256", .optional(f?["graph_output_numpy_sha256"]?.string)),
            ("arm_b_equal_python", .bool(sha256(of: armB) == f?["graph_output_numpy_sha256"]?.string)),
            ("audio_drift", .double(drift)), ("times", .obj(times))]))
        log("\(id): samples \(compareBytes(samples, refSamples)["bit_equal"]?.boolValue == true ? "bit-equal" : "DIFFER"), bucket "
            + "\(x.bucket.seconds) s, mel diff \(melCheck["diff_elems"]?.intValue ?? -1) (max \(melCheck["max_abs"]?.double ?? -1)), P \(x.prefixRows)")
    }
    for (i, id) in clipOrder.enumerated() {
        let donor = clipOrder[(i + 1) % clipOrder.count]
        let (prefix, p) = clipPrefixes[id]!
        try await decideRows(id, .audio, prefix, prefixRows: p, pyArm: "python_numpy_mel",
                             control: resizedPrefix(clipPrefixes[donor]!.prefix, rows: p), controlDonor: donor)
    }
    let doc: JSONValue = .obj([
        ("schema", .string("d1omni-parity-media/1")), ("setup", setup.json),
        ("media_assets", .obj(media.assets.sorted { $0.key < $1.key }.map { ($0.key, .string($0.value.path)) })),
        ("media_reference", .string(refDir.path)), ("fixtures", .string(fixtures.path)), ("repeats", .int(repeats)),
        ("init_seconds", .double(initS)), ("loads", .obj(loads)),
        ("summary", .obj([("images", .int(imageOut.count)), ("clips", .int(clipOut.count)), ("rows", .int(worst.rows)),
                          ("rows_bit_equal_python", .int(worst.rowsBit)), ("max_drift_marker_logits", .double(worst.drift))])),
        ("images", .array(imageOut)), ("clips", .array(clipOut)), ("rows", .array(outRows)),
        ("footprint_before", fp0), ("footprint_end", footprintJSON()), ("cache_before", cacheBefore), ("cache_after", cacheState()),
        ("environment", environment()),
    ])
    try JSONWriter.write(doc, to: url(try args.need("--out")), pretty: false)
    log("parity-media (\(setup.kind)): \(imageOut.count) images, \(clipOut.count) clips, \(worst.rows) rows, Python bit-equal \(worst.rowsBit)")
}

// MARK: - main

let argv = CommandLine.arguments
do {
    let args = try Args(argv)
    switch args.command {
    case "rows": try await rowsCommand(args)
    case "tokenize": try await tokenizeCommand(args)
    case "parity": try await parityCommand(args)
    case "ask": try await askCommand(args)
    case "time": try await timeCommand(args)
    case "parity-media": try await parityMediaCommand(args)
    default: throw CLIError.usage("command must be rows | tokenize | parity | parity-media | ask | time")
    }
} catch {
    FileHandle.standardError.write(Data("d1omni: \(error)\n".utf8))
    exit(1)
}
