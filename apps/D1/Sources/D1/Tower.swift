// Tower — the d1 vision tower on the low-level runtime: the tower bundle of conversion/d1/export_vision.py (SigLIP2 +
// the projector in its exact form, the crop's grid given as inputs), one stateless function `main` called once per
// crop. The type is apps/ClefFlash's VisionTower (zoo d1-3b 8e82d36); the contract is the tower bundle's metadata.json
// `graph`, checked against the loaded function at load.
//
//   inputs   patches [1024, 768], pos_table [1024, d_v], key_bias [1024] (float32, or float16 for an fp16 tower: the
//            host's float32 values are cast here, as gate_tower.py casts them), unshuffle_idx [256, 4] i32
//   output   image_embeds [256, d] (float32, or float16): rows 0 ..< h w / 4 are the crop's image tokens in its merged
//            grid's row-major order, the rest padding
//
// A request's image rows (conversion/d1/host.py's companion vision_host.py §6, metadata `vision.rows`): every crop's
// first h w / 4 rows, crops in order (per picture its tiles row-major, then the thumbnail), pictures in text order,
// cast to float16 (round to nearest even, NumPy's astype) -> the decoder's image_embeds rows 0 ..< n.
//
// A crop's four inputs come from the picture file (`D1TowerInputs.pictures(files:table:)`: ImagePixels.swift's decode,
// cap_pixels, crops, the torch uint8 bicubic resize, patches, the position table of the tower bundle's
// `host/position_embedding.safetensors` resized per crop, the unshuffle index = vision_host.tower_inputs, the order
// decide.py and readout_gate_vision.py run) or, to split an error between the pixels and the graphs, from raw files
// (`D1TowerInputs.read`: what vision_host.tower_inputs wrote, little-endian float32 / int32).

import CoreAI
import Foundation

public final class D1Tower: @unchecked Sendable {
    /// One crop's four inputs, row-major, in float32 / int32 (the host's types).
    public struct CropInputs: Sendable {
        public let patches: [Float]
        public let posTable: [Float]
        public let keyBias: [Float]
        public let unshuffle: [Int32]
        /// (h, w) patches and the crop's image tokens h w / 4
        public let grid: (h: Int, w: Int)
        public let tokens: Int

        public init(patches: [Float], posTable: [Float], keyBias: [Float], unshuffle: [Int32], grid: (h: Int, w: Int),
                    tokens: Int) {
            self.patches = patches
            self.posTable = posTable
            self.keyBias = keyBias
            self.unshuffle = unshuffle
            self.grid = grid
            self.tokens = tokens
        }
    }

    public static let inputNames = ["patches", "pos_table", "key_bias", "unshuffle_idx"]

    public let bundle: URL
    public let url: URL
    public let name: String
    public let contract: D1GraphContract
    /// d (image_embeds [256, d]: the decoder's hidden width)
    public let width: Int
    public let maxTokens: Int
    /// the bundle's `host/position_embedding.safetensors` [256, d_v] (metadata.json `host_files`), resized per crop
    public let positionTable: D1Pixels.PositionTable
    public let loadSeconds: Double
    public let descriptor: JSONValue
    public let options: SpecializationOptions
    /// crops encoded since the load (a gate reads it around a request the host refuses: no call)
    public private(set) var callCount = 0
    private let function: InferenceFunction
    private let inputDescriptors: [String: NDArrayDescriptor]

    public static let positionTableFile = "host/position_embedding.safetensors"

    /// `bundle` = a tower bundle directory; `asset` = its `.aimodelc` (AOT) or `.aimodel` (JIT), nil = the AOT asset
    /// beside it (`<bundles>_aotc/<name>.h16c.aimodelc`). The contract's input shapes are checked against the host's
    /// (patches [1024, 768], pos_table [1024, d_v] with d_v the position table's width, key_bias [1024]).
    public init(bundle: URL, asset: URL? = nil, options: SpecializationOptions? = nil) async throws {
        let meta = try JSONParser.parse(Data(contentsOf: bundle.appendingPathComponent("metadata.json")))
        guard meta["kind"]?.string == "vision-tower", let name = meta["name"]?.string, let g = meta["graph"],
              meta["assets"]?["main"]?.string != nil
        else { throw D1Error.bundle("\(bundle.path)/metadata.json: not a vision-tower bundle with name / graph / assets.main") }
        let contract = try D1GraphContract(g, what: "\(name) metadata.json graph")
        guard Set(contract.inputs.keys) == Set(Self.inputNames), let out = contract.outputs["image_embeds"], out.shape.count == 2,
              contract.inputs["unshuffle_idx"] == D1TensorSpec(shape: [out.shape[0], 4], dtype: "int32")
        else { throw D1Error.contract("\(name): the graph is not patches / pos_table / key_bias / unshuffle_idx -> image_embeds") }
        let table = try D1Pixels.PositionTable.load(url: bundle.appendingPathComponent(Self.positionTableFile))
        let p = D1Pixels.maxPatches
        guard out.shape[0] == D1Pixels.maxTokens, contract.inputs["patches"]?.shape == [p, D1Pixels.patchDim],
              contract.inputs["pos_table"]?.shape == [p, table.dim], contract.inputs["key_bias"]?.shape == [p]
        else {
            throw D1Error.contract("\(name): inputs \(contract.inputs.sorted { $0.key < $1.key }.map { "\($0.key) \($0.value)" }) "
                + "-> image_embeds \(out) are not the host's (patches [\(p), \(D1Pixels.patchDim)], pos_table [\(p), "
                + "\(table.dim)] = \(Self.positionTableFile), key_bias [\(p)], \(D1Pixels.maxTokens) rows)")
        }
        let url = asset ?? D1Paths.aot(bundle: bundle, name: name)
        let opts = options ?? (url.pathExtension == "aimodelc" ? .default : D1Paths.towerJITOptions)
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: opts)
        guard let fd = model.functionDescriptor(for: contract.function), let fn = try model.loadFunction(named: contract.function)
        else { throw D1Error.contract("tower \(url.lastPathComponent): no \"\(contract.function)\" (functions \(model.functionNames))") }
        loadSeconds = d1Seconds(since: t0)
        try D1Contract.check(fd, what: "tower \(url.lastPathComponent)", inputs: contract.inputs, outputs: contract.outputs,
                             states: contract.states)
        self.bundle = bundle
        self.url = url
        self.name = name
        self.contract = contract
        width = out.shape[1]
        maxTokens = out.shape[0]
        positionTable = table
        descriptor = D1Contract.describe(fd)
        self.options = opts
        function = fn
        inputDescriptors = Dictionary(uniqueKeysWithValues: Self.inputNames.map { ($0, ND.descriptor(fd.inputDescriptor(of: $0))!) })
    }

    /// A tower bundle's `.aimodel` (metadata.json `assets.main`), the asset the runtime specializes here (JIT).
    public static func modelAsset(bundle: URL) throws -> URL {
        let meta = try JSONParser.parse(Data(contentsOf: bundle.appendingPathComponent("metadata.json")))
        guard let main = meta["assets"]?["main"]?.string else { throw D1Error.bundle("\(bundle.path)/metadata.json: no assets.main") }
        return bundle.appendingPathComponent(main)
    }

    /// One crop -> image_embeds [256 * d] row-major as Float (an fp16 tower's values widened exactly).
    public func encode(_ c: CropInputs) async throws -> [Float] {
        var inputs: [String: NDArray] = [:]
        for (n, v) in [("patches", c.patches), ("pos_table", c.posTable), ("key_bias", c.keyBias)] {
            let d = inputDescriptors[n]!
            guard v.count == d.shape.reduce(1, *) else {
                throw D1Error.contract("tower input \(n): \(v.count) values for \(d.shape)")
            }
            inputs[n] = contract.inputs[n]!.dtype == "float16" ? ND.make(v.map { Float16($0) }, d) : ND.make(v, d)
        }
        let ud = inputDescriptors["unshuffle_idx"]!
        guard c.unshuffle.count == ud.shape.reduce(1, *) else {
            throw D1Error.contract("tower input unshuffle_idx: \(c.unshuffle.count) values for \(ud.shape)")
        }
        inputs["unshuffle_idx"] = ND.make(c.unshuffle, ud)
        callCount += 1
        var outputs = try await function.run(inputs: inputs)
        guard let array = outputs.remove("image_embeds")?.ndArray else {
            throw D1Error.contract("tower: no image_embeds in the outputs")
        }
        let emb: [Float] = contract.outputs["image_embeds"]!.dtype == "float16"
            ? ND.read(array, as: Float16.self).map { Float($0) } : ND.read(array, as: Float.self)
        guard emb.count == maxTokens * width else {
            throw D1Error.contract("tower: \(emb.count) values for \(maxTokens) x \(width)")
        }
        return emb
    }

    /// Every crop in order -> the decoder's image rows (each crop's first `tokens` rows, float16) and each crop's whole
    /// output (for a gate).
    public func imageRows(_ crops: [CropInputs]) async throws -> (rows: [Float16], outputs: [[Float]], seconds: [Double]) {
        var rows: [Float16] = []
        var outs: [[Float]] = []
        var secs: [Double] = []
        for c in crops {
            guard c.tokens <= maxTokens else { throw D1Error.contract("a crop of \(c.tokens) tokens over the tower's \(maxTokens)") }
            let t = ContinuousClock.now
            let e = try await encode(c)
            secs.append(d1Seconds(since: t))
            rows += e[0..<(c.tokens * width)].map { Float16($0) }
            outs.append(e)
        }
        return (rows, outs, secs)
    }
}

/// The four tower inputs of every crop of a request's pictures, in text order: made from the picture files here
/// (`pictures(files:table:)`, ImagePixels.swift) or read from raw files (`read`, `--tower-inputs <dir>`):
/// `<dir>/manifest.json` = {"pictures": [{"id", "size": [w, h] (as the picture arrives), "crops": [{"crop", "kind",
/// "grid": [h, w], "n_tokens", "patches", "pos_table", "key_bias", "unshuffle_idx"}]}]} with each input a raw
/// little-endian file (float32; unshuffle_idx int32) beside it. The plan Swift makes from the picture's size
/// (`D1Vision.plan`) must give the same crops.
public struct D1TowerInputs: Sendable {
    public struct Picture: Sendable {
        public let id: String
        /// the picture as it arrives (after its EXIF orientation, before cap_pixels)
        public let width: Int
        public let height: Int
        public let plan: D1Vision.Plan
        public let crops: [D1Tower.CropInputs]
    }

    public let pictures: [Picture]

    /// Picture files, in text order -> every crop's four inputs (vision_host.py §1–8 through `D1Pixels`: decode with
    /// the EXIF orientation, cap_pixels, the plan, each crop resized and cut, patches / key_bias / unshuffle_idx, the
    /// position table resized to the crop's grid once per distinct grid). A picture's id is its file name without the
    /// extension.
    public static func pictures(files: [URL], table: D1Pixels.PositionTable) throws -> D1TowerInputs {
        var out: [Picture] = []
        for url in files {
            let pic = try D1Pixels.picture(url: url)
            let w = pic.decoded.rgb.width, h = pic.decoded.rgb.height
            let check = D1Vision.plan(pictureWidth: w, pictureHeight: h)
            guard check.crops == pic.plan.crops else {
                throw D1Error.contract("\(url.lastPathComponent): the plan of the capped pixels differs from the plan of \(w) x \(h)")
            }
            let inputs = try D1Pixels.towerInputs(pic, table: table)
            let crops = zip(pic.plan.crops, inputs).map { c, t in
                D1Tower.CropInputs(patches: t.patches, posTable: t.posTable, keyBias: t.keyBias, unshuffle: t.unshuffleIndex,
                                   grid: (t.gridHeight, t.gridWidth), tokens: c.tokens)
            }
            out.append(Picture(id: url.deletingPathExtension().lastPathComponent, width: w, height: h, plan: pic.plan,
                               crops: crops))
        }
        return D1TowerInputs(pictures: out)
    }

    public static func read(_ dir: URL, ids: [String]? = nil) throws -> D1TowerInputs {
        let m = try JSONParser.parse(Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        var out: [Picture] = []
        var byID: [String: JSONValue] = [:]
        for p in m["pictures"]?.array ?? [] {
            guard let id = p["id"]?.string else { throw D1Error.bundle("\(dir.path)/manifest.json: a picture without an id") }
            byID[id] = p
        }
        let order = try (ids ?? (m["pictures"]?.array ?? []).compactMap { $0["id"]?.string }).map { id -> JSONValue in
            guard let p = byID[id] else { throw D1Error.bundle("\(dir.path)/manifest.json: no picture \(id)") }
            return p
        }
        for p in order {
            guard let id = p["id"]?.string, let size = p["size"]?.array?.compactMap(\.intValue), size.count == 2,
                  let crops = p["crops"]?.array
            else { throw D1Error.bundle("\(dir.path)/manifest.json: a picture without id / size / crops") }
            let plan = D1Vision.plan(pictureWidth: size[0], pictureHeight: size[1])
            guard plan.crops.count == crops.count else {
                throw D1Error.contract("\(id): \(crops.count) crops in the manifest, the plan of \(size[0]) x \(size[1]) has "
                    + "\(plan.crops.count)")
            }
            var cs: [D1Tower.CropInputs] = []
            for (c, pc) in zip(crops, plan.crops) {
                guard let grid = c["grid"]?.array?.compactMap(\.intValue), grid.count == 2, let n = c["n_tokens"]?.intValue,
                      grid[0] == pc.grid.h, grid[1] == pc.grid.w, n == pc.tokens, c["kind"]?.string == pc.kind.rawValue
                else { throw D1Error.contract("\(id) \(c["crop"]?.string ?? "?"): the manifest's crop differs from the plan's") }
                func file(_ k: String) throws -> Data {
                    guard let f = c[k]?.string else { throw D1Error.bundle("\(id): no \(k) file") }
                    return try Data(contentsOf: dir.appendingPathComponent(f))
                }
                cs.append(D1Tower.CropInputs(patches: Self.floats(try file("patches")), posTable: Self.floats(try file("pos_table")),
                                             keyBias: Self.floats(try file("key_bias")), unshuffle: Self.int32s(try file("unshuffle_idx")),
                                             grid: (grid[0], grid[1]), tokens: n))
            }
            out.append(Picture(id: id, width: size[0], height: size[1], plan: plan, crops: cs))
        }
        return D1TowerInputs(pictures: out)
    }

    static func floats(_ d: Data) -> [Float] {
        d.withUnsafeBytes { b in (0..<(d.count / 4)).map { Float(bitPattern: UInt32(littleEndian: b.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) } }
    }

    static func int32s(_ d: Data) -> [Int32] {
        d.withUnsafeBytes { b in (0..<(d.count / 4)).map { Int32(littleEndian: b.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self)) } }
    }
}

/// Where a bundle's AOT asset lives and how a `.aimodel` is specialized here.
public enum D1Paths {
    /// `<bundles>/<name>` -> `<bundles>_aotc/<name>.h16c.aimodelc` (the lane's layout, readout_gate.py's rule).
    public static func aot(bundle: URL, name: String) -> URL {
        let parent = bundle.deletingLastPathComponent()
        return parent.deletingLastPathComponent().appendingPathComponent("\(parent.lastPathComponent)_aotc")
            .appendingPathComponent("\(name).h16c.aimodelc")
    }

    /// The decoder `.aimodel`'s specialization: GPU preferred with frequent reshapes, the exporter's AOT flags
    /// (`coreai-build compile --preferred-compute gpu --expect-frequent-reshapes`; apps/Kev, apps/ClefFlash).
    public static var decoderJITOptions: SpecializationOptions {
        var o = SpecializationOptions(preferredComputeUnitKind: .gpu)
        o.expectFrequentReshapes = true
        return o
    }

    /// The tower `.aimodel`'s: GPU preferred (export_vision.py compiles it without --expect-frequent-reshapes).
    public static var towerJITOptions: SpecializationOptions { SpecializationOptions(preferredComputeUnitKind: .gpu) }
}
