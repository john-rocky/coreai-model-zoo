// d1-pixels-test — the Mac CLI that writes what D1Pixels (Sources/D1/ImagePixels.swift) makes of picture files, for
// conversion/d1/gate_swift_pixels.py to hold against vision_host.py. No graph, no weights.
//
//   d1-pixels-test --images <dir> [<dir> ...] --out <dir> [--table <safetensors or raw f32>] [--stages]
//                  [--resize torch|pillow|float] [--positions fused|unfused]
//       every .png / .jpg / .jpeg / .tif / .tiff in the dirs (by name): decode (EXIF orientation applied), cap, plan,
//       crops, and per crop the tower's four inputs as raw little-endian files
//         <out>/<file name>/c<k>_patches.f32 [1024, 768], c<k>_pos_table.f32 [1024, d] (with --table),
//         c<k>_key_bias.f32 [1024], c<k>_unshuffle_idx.i32 [256, 4]
//       --stages adds decoded.u8 [h, w, 3], capped.u8 (when the cap resized), c<k>_crop.u8 [H, W, 3] (after the
//       resize) and c<k>_normalized.f32 [H, W, 3]. <out>/index.json lists every picture, crop and file with sha256.
//       --resize pillow | float and --positions unfused are the gate's negative controls (the processor's resize is
//       torch's integer kernel, its position table torch's fused float32 one).
//   d1-pixels-test --decode-only --images <dir> [<dir> ...] --out <dir>
//       decoded.u8 per picture and how it was read (layout, orientation, exact)
//   d1-pixels-test --pos-only --table <file> --grids 24x24,16x24,... --out <dir> [--positions fused|unfused]
//       the table resized to each grid: <out>/pos_<h>x<w>.f32 [h * w, d]

import CryptoKit
import D1
import Foundation

enum CLIError: Error, CustomStringConvertible {
    case usage(String)
    var description: String {
        switch self {
        case .usage(let s): return "usage: \(s)"
        }
    }
}

struct Options {
    var images: [String] = []
    var out = ""
    var table: String?
    var grids: [(h: Int, w: Int)] = []
    var stages = false
    var decodeOnly = false
    var posOnly = false
    var resize = "torch"
    var fused = true

    init(_ argv: [String]) throws {
        var i = 1
        func value(_ flag: String) throws -> String {
            guard i + 1 < argv.count, !argv[i + 1].hasPrefix("--") else { throw CLIError.usage("\(flag) needs a value") }
            i += 1
            return argv[i]
        }
        while i < argv.count {
            let a = argv[i]
            switch a {
            case "--images":
                while i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
                    images.append(argv[i + 1])
                    i += 1
                }
                guard !images.isEmpty else { throw CLIError.usage("--images needs at least one dir") }
            case "--out": out = try value(a)
            case "--table": table = try value(a)
            case "--grids":
                grids = try value(a).split(separator: ",").map { g in
                    let p = g.split(separator: "x").compactMap { Int($0) }
                    guard p.count == 2, p[0] > 0, p[1] > 0 else { throw CLIError.usage("--grids wants HxW,HxW,..: \(g)") }
                    return (h: p[0], w: p[1])
                }
            case "--stages": stages = true
            case "--decode-only": decodeOnly = true
            case "--pos-only": posOnly = true
            case "--resize":
                resize = try value(a)
                guard ["torch", "pillow", "float"].contains(resize) else { throw CLIError.usage("--resize torch|pillow|float") }
            case "--positions":
                let v = try value(a)
                guard v == "fused" || v == "unfused" else { throw CLIError.usage("--positions fused|unfused") }
                fused = v == "fused"
            default:
                throw CLIError.usage("unexpected \(a)")
            }
            i += 1
        }
        guard !out.isEmpty else { throw CLIError.usage("missing --out") }
        if posOnly {
            guard table != nil, !grids.isEmpty else { throw CLIError.usage("--pos-only needs --table and --grids") }
        } else {
            guard !images.isEmpty else { throw CLIError.usage("missing --images") }
        }
    }
}

func url(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL }
func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

/// Writes `values` as raw little-endian bytes (the host's byte order on arm64) and returns {path, sha256, bytes}.
func write<T>(_ values: [T], _ dir: URL, _ name: String, root: URL) throws -> [String: Any] {
    let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
    let u = dir.appendingPathComponent(name)
    try data.write(to: u)
    return ["path": String(u.path.dropFirst(root.path.count + 1)), "sha256": sha256(data), "bytes": data.count]
}

func pictureFiles(_ dirs: [String]) throws -> [URL] {
    let exts: Set<String> = ["png", "jpg", "jpeg", "tif", "tiff"]
    var out: [URL] = []
    for d in dirs {
        let names = try FileManager.default.contentsOfDirectory(atPath: url(d).path).sorted()
        out += names.filter { exts.contains(($0 as NSString).pathExtension.lowercased()) }.map { url(d).appendingPathComponent($0) }
    }
    let ids = out.map(\.lastPathComponent)
    guard Set(ids).count == ids.count else { throw CLIError.usage("two pictures share a file name across the dirs") }
    return out
}

// MARK: the negative control's resize: bicubic in float64, no int16 weights, no uint8 between the passes

func cubic(_ t: Double) -> Double {
    let a = -0.5, x = abs(t)
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
    return 0
}

func floatPass(_ src: [Double], rows: Int, cols: Int, alongRows: Bool, out n: Int) -> [Double] {
    let inSize = alongRows ? rows : cols
    let scale = Double(inSize) / Double(n)
    let support = scale >= 1 ? 2 * scale : 2
    let inv = scale >= 1 ? 1 / scale : 1
    let orows = alongRows ? n : rows, ocols = alongRows ? cols : n
    var dst = [Double](repeating: 0, count: orows * ocols * 3)
    for i in 0..<n {
        let center = scale * (Double(i) + 0.5)
        let lo = max(Int(center - support + 0.5), 0), hi = min(Int(center + support + 0.5), inSize)
        var ws = (lo..<hi).map { cubic((Double($0) - center + 0.5) * inv) }
        let total = ws.reduce(0, +)
        if total != 0 { ws = ws.map { $0 / total } }
        for r in 0..<(alongRows ? cols : rows) {
            for c in 0..<3 {
                var acc = 0.0
                for (j, w) in ws.enumerated() {
                    acc += w * (alongRows ? src[((lo + j) * cols + r) * 3 + c] : src[(r * cols + lo + j) * 3 + c])
                }
                dst[(alongRows ? (i * ocols + r) : (r * ocols + i)) * 3 + c] = acc
            }
        }
    }
    return dst
}

func resizeFloat(_ image: D1Pixels.RGB, width: Int, height: Int) -> D1Pixels.RGB {
    if image.width == width && image.height == height { return image }
    var x = image.bytes.map(Double.init)
    var w = image.width
    let h = image.height
    if w != width {
        x = floatPass(x, rows: h, cols: w, alongRows: false, out: width)
        w = width
    }
    if h != height { x = floatPass(x, rows: h, cols: w, alongRows: true, out: height) }
    return D1Pixels.RGB(width: width, height: height, bytes: x.map { UInt8(min(max($0.rounded(.toNearestOrEven), 0), 255)) })
}

// MARK: main

func run(_ o: Options) throws {
    let root = url(o.out)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    var index: [String: Any] = ["tool": "d1-pixels-test", "argv": Array(CommandLine.arguments.dropFirst())]
    var table: D1Pixels.PositionTable?
    if let t = o.table {
        table = try D1Pixels.PositionTable.load(url: url(t))
        index["table"] = ["path": url(t).path, "sha256": sha256(try Data(contentsOf: url(t))), "dim": table!.dim]
    }
    if o.posOnly {
        var grids: [[String: Any]] = []
        for g in o.grids {
            let file = try write(table!.resized(height: g.h, width: g.w, fused: o.fused), root, "pos_\(g.h)x\(g.w).f32",
                                 root: root)
            grids.append(["grid": [g.h, g.w], "file": file])
        }
        index["positions"] = o.fused ? "fused" : "unfused"
        index["grids"] = grids
        try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
            .write(to: root.appendingPathComponent("index.json"))
        print("pos-only: \(o.grids.count) grids -> \(root.path)")
        return
    }
    let resize: (D1Pixels.RGB, Int, Int) -> D1Pixels.RGB
    switch o.resize {
    case "pillow": resize = { D1Pixels.resizePillowBicubic($0, width: $1, height: $2) }
    case "float": resize = { resizeFloat($0, width: $1, height: $2) }
    default: resize = { D1Pixels.resizeTorchU8Bicubic($0, width: $1, height: $2) }
    }
    index["resize"] = o.resize
    index["positions"] = o.fused ? "fused" : "unfused"
    var pictures: [[String: Any]] = []
    for file in try pictureFiles(o.images) {
        let id = file.lastPathComponent
        let dir = root.appendingPathComponent(id)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let t0 = ContinuousClock.now
        let decoded = try D1Pixels.decode(url: file)
        var entry: [String: Any] = [
            "id": id, "path": file.path, "file_sha256": sha256(try Data(contentsOf: file)), "type": decoded.type,
            "layout": decoded.layout, "exact": decoded.exact, "orientation": decoded.orientation,
            "stored": [decoded.storedWidth, decoded.storedHeight], "decoded": [decoded.rgb.width, decoded.rgb.height],
        ]
        var files: [String: Any] = [:]
        if o.stages || o.decodeOnly { files["decoded"] = try write(decoded.rgb.bytes, dir, "decoded.u8", root: root) }
        if o.decodeOnly {
            entry["files"] = files
            pictures.append(entry)
            print("\(id): \(decoded.layout) orientation \(decoded.orientation) \(decoded.rgb.width)x\(decoded.rgb.height)")
            continue
        }
        let pic = D1Pixels.picture(decoded, resize: resize)
        let inputs = try D1Pixels.towerInputs(pic, table: table, fused: o.fused)
        if o.stages && (pic.capped.width != decoded.rgb.width || pic.capped.height != decoded.rgb.height) {
            files["capped"] = try write(pic.capped.bytes, dir, "capped.u8", root: root)
        }
        entry["capped"] = [pic.capped.width, pic.capped.height]
        entry["plan"] = ["rows": pic.plan.rows, "cols": pic.plan.cols, "thumb": [pic.plan.thumbHeight, pic.plan.thumbWidth],
                         "tokens": pic.plan.tokens]
        var crops: [[String: Any]] = []
        for (k, (c, ti)) in zip(pic.plan.crops, inputs).enumerated() {
            let crop = pic.crops[k]
            var cf: [String: Any] = [
                "patches": try write(ti.patches, dir, "c\(k)_patches.f32", root: root),
                "key_bias": try write(ti.keyBias, dir, "c\(k)_key_bias.f32", root: root),
                "unshuffle_idx": try write(ti.unshuffleIndex, dir, "c\(k)_unshuffle_idx.i32", root: root),
            ]
            if table != nil { cf["pos_table"] = try write(ti.posTable, dir, "c\(k)_pos_table.f32", root: root) }
            if o.stages {
                cf["crop"] = try write(crop.bytes, dir, "c\(k)_crop.u8", root: root)
                cf["normalized"] = try write(crop.bytes.map { D1Pixels.normalizeTable[Int($0)] }, dir,
                                             "c\(k)_normalized.f32", root: root)
            }
            crops.append(["k": k, "kind": c.kind.rawValue, "row": c.row, "col": c.col, "size": [c.height, c.width],
                          "grid": [ti.gridHeight, ti.gridWidth], "n_patches": ti.patchCount, "n_tokens": ti.tokenCount,
                          "files": cf])
        }
        let d = ContinuousClock.now - t0
        entry["seconds"] = Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
        entry["crops"] = crops
        entry["files"] = files
        pictures.append(entry)
        print("\(id): \(decoded.layout) o\(decoded.orientation) \(decoded.rgb.width)x\(decoded.rgb.height) -> "
              + "\(pic.capped.width)x\(pic.capped.height), \(crops.count) crops")
    }
    index["pictures"] = pictures
    try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
        .write(to: root.appendingPathComponent("index.json"))
    print("\(pictures.count) pictures -> \(root.path)")
}

do {
    try run(try Options(CommandLine.arguments))
} catch {
    FileHandle.standardError.write(Data("d1-pixels-test: \(error)\n".utf8))
    exit(1)
}
