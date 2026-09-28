// decider-vision — the Mac CLI over the DeciderVision library.
//
//   decider-vision ask --bundle <dir> [--tower <aimodel(c)> --grid 256|448 --image x.png] --context "…"
//       --question "…" --options "a|b|c" [--question … --options …] [--asset aot|jit] [--decoder <asset>] [--json out]
//   decider-vision fixture --bundle <dir> --tower-g256 <aimodel(c)> --tower-g448 <aimodel(c)> --rows rows.json
//       --images <dir> --out out.json [--asset aot|jit] [--decoder <asset>] [--meta meta.json] [--dump-dir <dir>]
//       [--runs r01:g256,t02:text] [--reload] [--no-dump] [--s1] [--label text]
//   decider-vision make-image --out x.png [--size 256] [--circles 3]
//   decider-vision preprocess --images <dir> --out-dir <dir>       (the host half alone: tiles + patches' sha256)
//
// --asset aot (default): the decoder is the bundle's AOT asset, <bundle>/../../bundles_aotc/<name>.h16c.aimodelc
// (or --decoder), loaded with SpecializationOptions.default as the Python gates load it. --asset jit: the bundle's
// .aimodel specialized here with the exporter's AOT settings (GPU preferred, frequent reshapes; the towers GPU
// preferred) — the options the zoo's engines load a dynamic language model with.
//
// fixture runs every row of rows.json: an image row at g256 and g448, a text row once (the round-1 fixture: 70 + 6
// runs), then re-runs the first run (the states are zeroed per row: its slot logits must repeat bit for bit). Per run
// it writes the ids, slots, letter logits, probabilities, full-vocabulary top-1, the sha256 of the decoded and resized
// pixels, of the patches and of the tower output, and every step's time; into --dump-dir the resized RGB8 tiles
// (<image>__g256.rgb), the tower outputs (<id>__<arm>.embeds.f32) and the slot logits (<id>__<arm>.logits.f16);
// --no-dump skips those files (timed passes), --reload drops and loads everything again at the end (warm load times),
// --s1 reads with "main" only (S = 1 order).

import CoreAI
import CoreGraphics
import CryptoKit
import DeciderVision
import Darwin
import Foundation
import ImageIO
import UniformTypeIdentifiers

// MARK: - Arguments

struct Args {
    var command = ""
    var values: [String: [String]] = [:]
    var flags: Set<String> = []

    init(_ argv: [String]) throws {
        guard argv.count > 1 else { throw CLIError.usage("no command") }
        command = argv[1]
        var i = 2
        let flagNames: Set<String> = ["--reload", "--s1", "--no-dump"]
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
    func need(_ k: String) throws -> String {
        guard let v = one(k) else { throw CLIError.usage("missing \(k)") }
        return v
    }
    func all(_ k: String) -> [String] { values[k] ?? [] }
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

func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

func sha256<T>(of values: [T]) -> String {
    values.withUnsafeBytes { raw in SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined() }
}

func write<T>(_ values: [T], to u: URL) throws {
    try values.withUnsafeBytes { raw in try Data(raw).write(to: u) }
}

func jsonSafe(_ v: Any) -> Any {
    switch v {
    case let d as Double: return d.isFinite ? d : "\(d)"
    case let f as Float: return f.isFinite ? Double(f) : "\(f)"
    case let a as [Any]: return a.map(jsonSafe)
    case let m as [String: Any]: return m.mapValues(jsonSafe)
    default: return v
    }
}

func writeJSON(_ object: [String: Any], to u: URL) throws {
    let d = try JSONSerialization.data(withJSONObject: jsonSafe(object), options: [.prettyPrinted, .sortedKeys])
    let tmp = u.appendingPathExtension("tmp")
    try d.write(to: tmp)
    _ = try? FileManager.default.removeItem(at: u)
    try FileManager.default.moveItem(at: tmp, to: u)
}

func sysctlString(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "?" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
}

/// The process's physical footprint (what jetsam counts on the phone), in bytes.
func footprint() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}

func environment() -> [String: Any] {
    #if DEBUG
    let config = "Debug"
    #else
    let config = "Release"
    #endif
    return ["os": ProcessInfo.processInfo.operatingSystemVersionString, "os_build": sysctlString("kern.osversion"),
            "chip": sysctlString("machdep.cpu.brand_string"), "model": sysctlString("hw.model"),
            "build_configuration": config, "pid": Int(getpid()),
            "device_architecture": AIModel.deviceArchitectureName,
            "argv": CommandLine.arguments]
}

func ms(_ s: Double) -> String { String(format: "%.1f", s * 1e3) }

// MARK: - Assets

struct Assets {
    let bundle: URL
    let decoder: URL
    let kind: String
    let decoderOptions: SpecializationOptions
    let towerOptions: SpecializationOptions

    init(_ args: Args) throws {
        bundle = url(try args.need("--bundle"))
        let meta = try VisionDecider.Metadata(bundle: bundle)
        kind = args.one("--asset") ?? "aot"
        switch kind {
        case "aot":
            let stem = (meta.asset as NSString).deletingPathExtension
            decoder = args.one("--decoder").map(url)
                ?? bundle.deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("bundles_aotc/\(stem).h16c.aimodelc")
            decoderOptions = .default
            towerOptions = .default
        case "jit":
            decoder = args.one("--decoder").map(url) ?? bundle.appendingPathComponent(meta.asset)
            var d = SpecializationOptions(preferredComputeUnitKind: .gpu)
            d.expectFrequentReshapes = true
            decoderOptions = d
            towerOptions = SpecializationOptions(preferredComputeUnitKind: .gpu)
        default:
            throw CLIError.usage("--asset aot|jit")
        }
        guard FileManager.default.fileExists(atPath: decoder.path) else { throw CLIError.failed("no decoder asset at \(decoder.path)") }
    }

    var json: [String: Any] {
        ["bundle": bundle.path, "decoder": decoder.path, "asset": kind,
         "decoder_options": describe(decoderOptions), "tower_options": describe(towerOptions)]
    }
}

func describe(_ o: SpecializationOptions) -> String {
    if o == .default { return "SpecializationOptions.default" }
    let pref = o.preferredComputeUnitKind.map { "\($0)" } ?? "none"
    return "preferred \(pref), allowed \(o.allowedComputeUnitKinds.map { "\($0)" }.sorted()), expectFrequentReshapes \(o.expectFrequentReshapes)"
}

// MARK: - Records

func answerJSON(_ a: VisionDecider.Answer) -> [String: Any] {
    ["t": a.slot, "letter_logits": a.letterLogits.map { Double(Float($0)) }, "probs": a.probabilities,
     "argmax": a.argmax, "full_vocab_top1_id": a.fullVocabTop1, "full_vocab_top1_logit": Double(Float(a.fullVocabTop1Logit)),
     "full_vocab_top5_ids": a.fullVocabTop5, "finite": a.finite, "read_from": a.readFrom]
}

func traceJSON(_ tr: VisionDecider.Trace) -> [String: Any] {
    var calls = ["prefill": tr.pass.callIsPrefill.filter { $0 }.count, "main": tr.pass.callIsPrefill.filter { !$0 }.count]
    calls["total"] = tr.pass.callIsPrefill.count
    let pre = zip(tr.pass.callIsPrefill, tr.pass.callSeconds).filter { $0.0 }.map { $0.1 * 1e3 }
    let mainMs = zip(tr.pass.callIsPrefill, tr.pass.callSeconds).filter { !$0.0 }.map { $0.1 * 1e3 }
    var j: [String: Any] = [
        "ids": tr.row.ids, "slots": tr.row.slots, "tokens": tr.row.ids.count, "nopts": tr.row.optionCounts,
        "rope_shift_start": Int(tr.row.ropeShiftStart), "rope_shift_amount": Int(tr.row.ropeShiftAmount),
        "grid": tr.row.grid.map { [$0, $0] } as Any, "answers": tr.answers.map(answerJSON), "calls": calls,
        "call_ms": tr.pass.callSeconds.map { $0 * 1e3 }, "call_is_prefill": tr.pass.callIsPrefill,
        "prefill_call_ms": pre, "main_call_ms": mainMs, "state_reset_ms": tr.pass.resetSeconds * 1e3,
        "seconds": tr.seconds,
    ]
    if let p = tr.prepared {
        j["image_size_in"] = [p.decoded.width, p.decoded.height]
        j["decode_path"] = p.decoded.decodePath
        j["decoded_rgb_sha256"] = sha256(of: p.decoded.pixels)
        j["resized_rgb_sha256"] = sha256(of: p.resized.pixels)
        j["patches_sha256"] = sha256(of: p.patches)
        j["patches_shape"] = [p.patches.count / ImagePreprocess.patchVector, ImagePreprocess.patchVector]
    }
    if let e = tr.towerEmbeds { j["tower_embeds_sha256"] = sha256(of: e) }
    return j
}

// MARK: - ask

func ask(_ args: Args) async throws {
    let assets = try Assets(args)
    let qs = args.all("--question"), os = args.all("--options")
    guard !qs.isEmpty, qs.count == os.count else { throw CLIError.usage("one --options per --question") }
    let questions = zip(qs, os).map { DecisionQuestion(text: $0, options: $1.split(separator: "|").map(String.init)) }
    var towers: [VisionDecider.Grid: URL] = [:]
    var grid = VisionDecider.Grid.g256
    var image: CGImage? = nil
    var imagePath: String? = nil
    if let p = args.one("--image") {
        guard let t = args.one("--tower") else { throw CLIError.usage("--image needs --tower and --grid") }
        guard let g = Int(args.one("--grid") ?? "256").flatMap(VisionDecider.Grid.init(tile:)) else {
            throw CLIError.usage("--grid 256|448")
        }
        grid = g
        towers[g] = url(t)
        image = try ImagePreprocess.loadCGImage(url: url(p))
        imagePath = url(p).path
    }
    let t0 = ContinuousClock.now
    let decider = try await VisionDecider(bundle: assets.bundle, decoder: assets.decoder, towers: towers,
                                          decoderOptions: assets.decoderOptions, towerOptions: assets.towerOptions)
    let load = secondsNow(since: t0)
    let context = try args.need("--context")
    let tr = try await decider.trace(image: image, context: context, questions: questions, grid: grid)
    print("decoder \(assets.decoder.lastPathComponent) (\(assets.kind)), load \(String(format: "%.2f", load)) s")
    if let p = tr.prepared {
        print("image \(imagePath ?? "") \(p.decoded.width)x\(p.decoded.height) (\(p.decoded.decodePath)) -> \(grid) tile, tower \(ms(tr.seconds["tower"] ?? 0)) ms")
    }
    print("ids \(tr.row.ids.count), slots \(tr.row.slots), calls prefill \(tr.pass.callIsPrefill.filter { $0 }.count) / main \(tr.pass.callIsPrefill.filter { !$0 }.count), decision wall \(ms(tr.seconds["wall"] ?? 0)) ms")
    for (k, a) in tr.answers.enumerated() {
        print("Q\(k + 1) \(questions[k].text)")
        for (j, o) in questions[k].options.enumerated() {
            print(String(format: "  (%@) %-24@ %.6f%@", PromptBuilder.letters[j], o, a.probabilities[j], j == a.argmax ? "  <-" : ""))
        }
        print("  full-vocab top-1 id \(a.fullVocabTop1) (letter ids \(decider.metadata.letterIDs.prefix(questions[k].options.count).map { $0 }))")
    }
    if let out = args.one("--json") {
        var j = traceJSON(tr)
        j["questions"] = questions.map { ["text": $0.text, "options": $0.options] }
        j["context"] = context
        j["image"] = imagePath as Any
        j["assets"] = assets.json
        j["environment"] = environment()
        j["tower"] = towers[grid]?.path as Any
        try writeJSON(j, to: url(out))
        print("json: \(url(out).path)")
    }
}

// MARK: - fixture

struct FixtureRun {
    let id: String
    let arm: String
    let row: [String: Any]
}

func fixture(_ args: Args) async throws {
    let assets = try Assets(args)
    let rowsURL = url(try args.need("--rows"))
    let imagesDir = url(try args.need("--images"))
    let outURL = url(try args.need("--out"))
    let dump = url(args.one("--dump-dir") ?? outURL.deletingPathExtension().path + "_dump")
    let fm = FileManager.default
    for d in ["resized", "embeds", "logits"] {
        try fm.createDirectory(at: dump.appendingPathComponent(d), withIntermediateDirectories: true)
    }
    let rowsData = try Data(contentsOf: rowsURL)
    guard let rj = try JSONSerialization.jsonObject(with: rowsData) as? [String: Any], let rows = rj["rows"] as? [[String: Any]] else {
        throw CLIError.failed("\(rowsURL.path): no rows")
    }
    var rgbSHA: [String: String] = [:]
    if let m = args.one("--meta"), let mj = try JSONSerialization.jsonObject(with: Data(contentsOf: url(m))) as? [String: Any],
       let ims = mj["images"] as? [[String: Any]] {
        for im in ims { if let n = im["name"] as? String, let s = im["rgb_sha256"] as? String { rgbSHA[n] = s } }
    }
    var runs: [FixtureRun] = []
    for r in rows {
        guard let id = r["id"] as? String else { continue }
        if r["image"] is String {
            runs.append(FixtureRun(id: id, arm: "g256", row: r))
            runs.append(FixtureRun(id: id, arm: "g448", row: r))
        } else {
            runs.append(FixtureRun(id: id, arm: "text", row: r))
        }
    }
    if let sel = args.one("--runs") {
        let want = Set(sel.split(separator: ",").map(String.init))
        runs = runs.filter { want.contains("\($0.id):\($0.arm)") }
    }
    let towerPaths: [VisionDecider.Grid: URL] = [.g256: url(try args.need("--tower-g256")), .g448: url(try args.need("--tower-g448"))]
    let log = { (s: String) in print("[\(String(format: "%7.2f", Date().timeIntervalSince1970.truncatingRemainder(dividingBy: 100000)))] \(s)"); fflush(stdout) }

    var report: [String: Any] = ["schema": "decider-vision-swift-fixture/1", "label": args.one("--label") ?? "",
                                 "assets": assets.json, "environment": environment(),
                                 "rows_json": rowsURL.path, "rows_json_sha256": sha256(rowsData),
                                 "images_dir": imagesDir.path, "dump_dir": dump.path,
                                 "towers": towerPaths.reduce(into: [String: String]()) { $0["\($1.key)"] = $1.value.path },
                                 "started": ISO8601DateFormatter().string(from: Date())]
    report["footprint_start_bytes"] = footprint()
    log("loading: decoder \(assets.decoder.path) (\(assets.kind)), towers g256 / g448")
    let tLoad = ContinuousClock.now
    var decider: VisionDecider? = try await VisionDecider(
        bundle: assets.bundle, decoder: assets.decoder, towers: towerPaths,
        decoderOptions: assets.decoderOptions, towerOptions: assets.towerOptions)
    let loadWall = secondsNow(since: tLoad)
    guard let d0 = decider else { return }
    func loadRecord(_ d: VisionDecider, wall: Double) -> [String: Any] {
        ["wall_s": wall, "tokenizer_s": d.tokenizerLoadSeconds,
         "decoder_s": ["model": d.decoder.loadSeconds.model, "main": d.decoder.loadSeconds.main,
                       "prefill": d.decoder.loadSeconds.prefill as Any],
         "tower_s": d.towers.reduce(into: [String: Double]()) { $0["\($1.key)"] = $1.value.loadSeconds }]
    }
    report["load_first"] = loadRecord(d0, wall: loadWall)
    report["function_names"] = d0.decoder.functionNames
    report["decoder_descriptors"] = d0.decoder.descriptors
    report["tower_descriptors"] = d0.towers.reduce(into: [String: Any]()) { $0["\($1.key)"] = $1.value.descriptor }
    report["prefill_chunk"] = d0.decoder.chunk as Any
    report["footprint_after_load_bytes"] = footprint()
    log("loaded in \(String(format: "%.2f", loadWall)) s (tokenizer \(String(format: "%.2f", d0.tokenizerLoadSeconds)) s, decoder model \(String(format: "%.2f", d0.decoder.loadSeconds.model)) s)")

    var images: [String: CGImage] = [:]
    var imageFileSeconds: [String: Double] = [:]
    var written = Set<String>()
    var out: [[String: Any]] = []
    var firstLogits: [[Float16]]? = nil
    let useChunks = !args.flags.contains("--s1")
    let dumping = !args.flags.contains("--no-dump")
    report["dumping"] = dumping

    func runOne(_ run: FixtureRun, record: Bool) async throws -> ([String: Any], [[Float16]]) {
        let r = run.row
        let context = r["context"] as? String ?? ""
        let qs = (r["questions"] as? [[String: Any]] ?? []).map {
            DecisionQuestion(text: $0["text"] as? String ?? "", options: $0["options"] as? [String] ?? [])
        }
        var image: CGImage? = nil
        var fileSeconds = 0.0
        let name = r["image"] as? String
        if let name {
            // the file decode is timed on every run (a decision starts from the file), the CGImage is not reused
            let t = ContinuousClock.now
            let img = try ImagePreprocess.loadCGImage(url: imagesDir.appendingPathComponent("\(name).png"))
            fileSeconds = secondsNow(since: t)
            image = img
            images[name] = img
            imageFileSeconds[name] = fileSeconds
        }
        let grid: VisionDecider.Grid = run.arm == "g448" ? .g448 : .g256
        let tr = try await d0.trace(image: image, context: context, questions: qs, grid: grid, keepLogits: true,
                                    useChunks: useChunks)
        var j = traceJSON(tr)
        j["id"] = run.id
        j["arm"] = run.arm
        j["image"] = name as Any
        j["image_file_decode_s"] = fileSeconds
        j["wall_from_file_s"] = (tr.seconds["wall"] ?? 0) + fileSeconds
        j["footprint_bytes"] = footprint()
        if record, dumping, let p = tr.prepared, let name {
            if let want = rgbSHA[name] { j["decoded_rgb_sha256_equals_meta"] = (want == sha256(of: p.decoded.pixels)) }
            let key = "\(name)__\(run.arm)"
            if !written.contains(key) {
                try write(p.resized.pixels, to: dump.appendingPathComponent("resized/\(key).rgb"))
                written.insert(key)
            }
            j["resized_file"] = "resized/\(key).rgb"
            if let e = tr.towerEmbeds {
                try write(e, to: dump.appendingPathComponent("embeds/\(run.id)__\(run.arm).embeds.f32"))
                j["embeds_file"] = "embeds/\(run.id)__\(run.arm).embeds.f32"
            }
        }
        let logits = tr.slotLogits ?? []
        if record, dumping {
            try write(logits.flatMap { $0 }, to: dump.appendingPathComponent("logits/\(run.id)__\(run.arm).logits.f16"))
            j["logits_file"] = "logits/\(run.id)__\(run.arm).logits.f16"
        }
        return (j, logits)
    }

    let tRuns = ContinuousClock.now
    for (i, run) in runs.enumerated() {
        let (j, logits) = try await runOne(run, record: true)
        if i == 0 { firstLogits = logits }
        out.append(j)
        let ans = (j["answers"] as? [[String: Any]] ?? []).map { a -> String in
            let p = (a["probs"] as? [Double] ?? []).map { String(format: "%.4f", $0) }.joined(separator: " ")
            return "[\(p)] -> \(PromptBuilder.letters[a["argmax"] as? Int ?? 0])"
        }.joined(separator: " ")
        let secs = j["seconds"] as? [String: Double] ?? [:]
        log("\(i + 1)/\(runs.count) \(run.id)/\(run.arm): \(j["tokens"] ?? 0) tok, calls \(j["calls"] ?? [:]), wall \(ms(j["wall_from_file_s"] as? Double ?? 0)) ms (tower \(ms(secs["tower"] ?? 0)), decoder \(ms(secs["decoder"] ?? 0))) \(ans)")
    }
    report["runs_wall_s"] = secondsNow(since: tRuns)
    report["runs"] = out
    report["footprint_after_runs_bytes"] = footprint()

    // reset check: the first run again, after every other row went through the same states
    if let first = runs.first, let want = firstLogits {
        let (j, got) = try await runOne(first, record: false)
        let equal = got.count == want.count && zip(got, want).allSatisfy { a, b in
            a.count == b.count && zip(a, b).allSatisfy { $0.bitPattern == $1.bitPattern }
        }
        var maxDiff: Float = 0
        for (a, b) in zip(got, want) { for (x, y) in zip(a, b) { maxDiff = max(maxDiff, abs(Float(x) - Float(y))) } }
        report["reset_check"] = ["run": "\(first.id)/\(first.arm)", "bit_equal": equal, "slot_logits_max_abs_diff": Double(maxDiff),
                                 "wall_from_file_s": j["wall_from_file_s"] ?? 0]
        log("reset re-run \(first.id)/\(first.arm): slot logits bit-equal \(equal)")
    }

    if args.flags.contains("--reload") {
        decider = nil
        let t = ContinuousClock.now
        let d1 = try await VisionDecider(bundle: assets.bundle, decoder: assets.decoder, towers: towerPaths,
                                         decoderOptions: assets.decoderOptions, towerOptions: assets.towerOptions)
        report["load_reload"] = loadRecord(d1, wall: secondsNow(since: t))
        log("reload in \(String(format: "%.2f", secondsNow(since: t))) s")
    }
    report["finished"] = ISO8601DateFormatter().string(from: Date())
    try writeJSON(report, to: outURL)
    log("json: \(outURL.path)")
}

func secondsNow(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

// MARK: - preprocess

/// Every image in --images at both grids, no model: the resized tiles and the patches' sha256 (the host half alone).
func preprocess(_ args: Args) throws {
    let dir = url(try args.need("--images"))
    let outDir = url(try args.need("--out-dir"))
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let names = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".png") }.sorted()
    var recs: [[String: Any]] = []
    for f in names {
        let name = (f as NSString).deletingPathExtension
        let img = try ImagePreprocess.loadCGImage(url: dir.appendingPathComponent(f))
        for g in VisionDecider.Grid.allCases {
            let p = try ImagePreprocess.prepare(img, grid: g.side)
            try write(p.resized.pixels, to: outDir.appendingPathComponent("\(name)__\(g).rgb"))
            recs.append(["image": name, "grid": "\(g)", "size_in": [p.decoded.width, p.decoded.height],
                         "decode_path": p.decoded.decodePath, "decoded_rgb_sha256": sha256(of: p.decoded.pixels),
                         "resized_rgb_sha256": sha256(of: p.resized.pixels), "patches_sha256": sha256(of: p.patches),
                         "resized_file": "\(name)__\(g).rgb",
                         "ms": ["decode_rgb": p.decodeSeconds * 1e3, "resize": p.resizeSeconds * 1e3, "patches": p.patchSeconds * 1e3]])
        }
    }
    try writeJSON(["images": recs, "environment": environment()], to: outDir.appendingPathComponent("preprocess.json"))
    print("\(recs.count) tiles -> \(outDir.path)")
}

// MARK: - make-image

/// White square with N red discs (no antialiasing): a fixture-free image for the manual check.
func makeImage(_ args: Args) throws {
    let out = url(try args.need("--out"))
    let side = Int(args.one("--size") ?? "256") ?? 256
    let n = Int(args.one("--circles") ?? "3") ?? 3
    guard let space = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side * 4,
                              space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw CLIError.failed("no context") }
    ctx.setShouldAntialias(false)
    ctx.setFillColor(CGColor(colorSpace: space, components: [1, 1, 1, 1])!)
    ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
    ctx.setFillColor(CGColor(colorSpace: space, components: [1, 0, 0, 1])!)
    let s = Double(side)
    let r = s * 0.11
    for k in 0..<n {
        // centers on a gentle diagonal zig-zag so no two discs touch
        let cx = s * (Double(k) + 0.5) / Double(n)
        let cy = s * (k % 2 == 0 ? 0.36 : 0.64)
        ctx.fillEllipse(in: CGRect(x: cx - r, y: cy - r, width: 2 * r, height: 2 * r))
    }
    guard let img = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(out as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw CLIError.failed("cannot write \(out.path)") }
    CGImageDestinationAddImage(dest, img, nil)
    guard CGImageDestinationFinalize(dest) else { throw CLIError.failed("cannot write \(out.path)") }
    print("wrote \(out.path) (\(side)x\(side), \(n) red discs)")
}

// MARK: - main

do {
    let args = try Args(CommandLine.arguments)
    switch args.command {
    case "ask": try await ask(args)
    case "fixture": try await fixture(args)
    case "make-image": try makeImage(args)
    case "preprocess": try preprocess(args)
    default: throw CLIError.usage("commands: ask, fixture, preprocess, make-image")
    }
} catch {
    FileHandle.standardError.write(Data("decider-vision: \(error)\n".utf8))
    exit(1)
}
