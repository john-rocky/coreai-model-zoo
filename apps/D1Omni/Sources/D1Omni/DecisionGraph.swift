// DecisionGraph — one bucket's decision graph on the system CoreAI runtime (AIModel + loadFunction + run), and the
// graph's six inputs for a row (conversion/d1_omni/host.py §2 `graph_inputs`, `bucket_for`):
//
//   input_ids     [1,L] int32    0 on [0, P), the row's ids on [P, P+n), 0 (<|pad|>) after
//   prefix_embeds [1,L,1024] f32 the media prefix on [0, P), 0.0 elsewhere
//   pad_mask      [1,L] f32      1.0 on [0, P+n)
//   prefix_mask   [1,L] f32      1.0 on [0, P)
//   keep_right    [1,L] f32      0.0 at P-1 when P > 0, 1.0 elsewhere
//   qtype_onehot  [1,3] f32      choice / score / noul
//   -> scores     [1,L] f32      read at P + markers
//
// The asset is the bucket's AOT `.aimodelc` or its `.aimodel`, which the runtime specializes here (JIT). Either way
// the options are explicit: GPU preferred, no expectFrequentReshapes (the graphs are static; the AOT compile used
// `--preferred-compute gpu` without `--expect-frequent-reshapes`). `main`'s descriptor is checked against the contract
// at load. A text row's prefix_embeds is one zero array per graph, made at load and reused (never written).

import CoreAI
import Foundation

/// The graph's inputs for one row at bucket length L (host.py `graph_inputs`), as flat row-major arrays.
public struct GraphInputs: Sendable {
    public let length: Int
    public let inputIDs: [Int32]
    public let padMask: [Float]
    public let prefixMask: [Float]
    public let keepRight: [Float]
    public let qtype: [Float]
    /// P + markers: the positions of the scores the host reads
    public let markers: [Int]
    public let prefixLength: Int

    public static let inputNames = ["input_ids", "prefix_embeds", "pad_mask", "prefix_mask", "keep_right", "qtype_onehot"]

    public init(row: Row, length L: Int) throws {
        let p = row.prefixLength, n = row.ids.count
        guard p + n <= L else { throw D1OmniError.graphLimit("a row of \(p) + \(n) positions does not fit length \(L)") }
        var ids = [Int32](repeating: 0, count: L)
        for i in 0..<n { ids[p + i] = Int32(row.ids[i]) }
        var pad = [Float](repeating: 0, count: L)
        for i in 0..<(p + n) { pad[i] = 1 }
        var prefix = [Float](repeating: 0, count: L)
        for i in 0..<p { prefix[i] = 1 }
        var keep = [Float](repeating: 1, count: L)
        if p > 0 { keep[p - 1] = 0 }
        var q = [Float](repeating: 0, count: 3)
        q[row.question.type.index] = 1
        length = L
        inputIDs = ids
        padMask = pad
        prefixMask = prefix
        keepRight = keep
        qtype = q
        markers = row.markers.map { p + $0 }
        prefixLength = p
    }

    /// host.py `bucket_for`: the smallest bucket that holds the positions, nil if none does.
    public static func bucket(positions: Int, buckets: [Int]) -> Int? { buckets.sorted().first { positions <= $0 } }
}

public final class DecisionGraph: @unchecked Sendable {
    public let length: Int
    public let url: URL
    /// "aot" (.aimodelc) or "jit" (.aimodel specialized here)
    public let kind: String
    public let options: SpecializationOptions
    public let functionNames: [String]
    public let descriptor: JSONValue
    /// AIModel(contentsOf:) and loadFunction(named: "main"), seconds
    public let loadSeconds: (model: Double, function: Double)
    public static let hidden = 1024

    private let model: AIModel
    private let main: InferenceFunction
    private let descriptors: [String: NDArrayDescriptor]
    private let zeroPrefix: NDArray

    /// GPU preferred, no frequent reshapes: the AOT compile's flags, for either asset (never `.default`, which can fall
    /// back to the CPU and hide the accelerator's numbers).
    public static var gpuOptions: SpecializationOptions { SpecializationOptions(preferredComputeUnitKind: .gpu) }

    public init(contentsOf url: URL, length L: Int, options: SpecializationOptions = DecisionGraph.gpuOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = secondsSince(t0)
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: "main"), let fn = try model.loadFunction(named: "main") else {
            throw D1OmniError.contract("\(url.lastPathComponent): no function \"main\" (functions \(model.functionNames))")
        }
        let tMain = secondsSince(t1)
        let want: [String: TensorSpec] = [
            "input_ids": TensorSpec(shape: [1, L], type: .int32),
            "prefix_embeds": TensorSpec(shape: [1, L, Self.hidden], type: .float32),
            "pad_mask": TensorSpec(shape: [1, L], type: .float32), "prefix_mask": TensorSpec(shape: [1, L], type: .float32),
            "keep_right": TensorSpec(shape: [1, L], type: .float32), "qtype_onehot": TensorSpec(shape: [1, 3], type: .float32),
        ]
        var bad: [String] = []
        if Set(md.inputNames) != Set(want.keys) { bad.append("inputs \(md.inputNames.sorted())") }
        for (name, spec) in want where TensorSpec.of(md.inputDescriptor(of: name)) != spec {
            bad.append("\(name) \(TensorSpec.of(md.inputDescriptor(of: name))?.description ?? "missing") != \(spec)")
        }
        if md.outputNames != ["scores"] || TensorSpec.of(md.outputDescriptor(of: "scores")) != TensorSpec(shape: [1, L], type: .float32) {
            bad.append("outputs \(md.outputNames) \(TensorSpec.of(md.outputDescriptor(of: "scores"))?.description ?? "") != scores [1, \(L)] float32")
        }
        if !md.stateNames.isEmpty { bad.append("states \(md.stateNames)") }
        if !bad.isEmpty { throw D1OmniError.contract("\(url.lastPathComponent) main: \(bad.joined(separator: "; "))") }
        var d: [String: NDArrayDescriptor] = [:]
        for name in GraphInputs.inputNames { d[name] = ND.descriptor(md.inputDescriptor(of: name))! }
        self.length = L
        self.url = url
        self.kind = url.pathExtension == "aimodelc" ? "aot" : "jit"
        self.options = options
        self.functionNames = model.functionNames
        self.descriptor = describe(md)
        self.loadSeconds = (tModel, tMain)
        self.model = model
        self.main = fn
        self.descriptors = d
        self.zeroPrefix = ND.zeros(d["prefix_embeds"]!)
    }

    /// One call: the inputs (and the media prefix [P * 1024] row-major when P > 0) -> scores [L].
    public func scores(_ x: GraphInputs, prefix: [Float]? = nil) async throws -> [Float] {
        guard x.length == length else { throw D1OmniError.contract("inputs for L = \(x.length) on the L = \(length) graph") }
        let prefixArray: NDArray
        if x.prefixLength > 0 {
            guard let prefix, prefix.count == x.prefixLength * Self.hidden else {
                throw D1OmniError.request("a row with a \(x.prefixLength)-position prefix needs \(x.prefixLength) x \(Self.hidden) values")
            }
            var full = [Float](repeating: 0, count: length * Self.hidden)
            full.replaceSubrange(0..<prefix.count, with: prefix)
            prefixArray = ND.make(full, descriptors["prefix_embeds"]!)
        } else {
            prefixArray = zeroPrefix
        }
        let inputs: [String: NDArray] = [
            "input_ids": ND.make(x.inputIDs, descriptors["input_ids"]!),
            "prefix_embeds": prefixArray,
            "pad_mask": ND.make(x.padMask, descriptors["pad_mask"]!),
            "prefix_mask": ND.make(x.prefixMask, descriptors["prefix_mask"]!),
            "keep_right": ND.make(x.keepRight, descriptors["keep_right"]!),
            "qtype_onehot": ND.make(x.qtype, descriptors["qtype_onehot"]!),
        ]
        var outputs = try await main.run(inputs: inputs)
        guard let array = outputs.remove("scores")?.ndArray else { throw D1OmniError.contract("no scores in the outputs") }
        let s = ND.read(array, as: Float.self)
        guard s.count == length else { throw D1OmniError.contract("\(s.count) scores for L = \(length)") }
        return s
    }
}
