// MediaGraphs — the vision and audio graphs on the system CoreAI runtime (AIModel + loadFunction("main") + run), each
// with the contract of its bundle's metadata.json checked at load:
//
//   vision  pixel_values [1,1024,768] f32, pos_embed [1,1024,768] f32, patch_mask [1,1024] f32,
//           unshuffle_index [256,4] int32                                   -> prefix [1,256,1024] f32
//   audio   mel [1,128,F] f32, mask_f [1,F], mask_f2 [1,F2], mask_f4 [1,F4], mask_t [1,T] f32
//                                                                           -> prefix [1,T,1024] f32
//           one bundle per clip bucket: sec 5 / 10 / 20 / 30 -> F 501 / 1001 / 2001 / 3001, T 63 / 126 / 251 / 376
//
// As with the decision graph, the asset is the bundle's `.aimodel` (the runtime specializes it at load: JIT) or the
// Mac's AOT `.aimodelc` of the same bundle, always with SpecializationOptions(preferredComputeUnitKind: .gpu).

import CoreAI
import Foundation

/// One loaded media graph: `main`'s inputs checked against `contract`, one call at a time.
public final class MediaGraph: @unchecked Sendable {
    public enum Input: Sendable {
        case float([Float])
        case int32([Int32])
    }

    public let url: URL
    /// "aot" (.aimodelc) or "jit" (.aimodel specialized here)
    public let kind: String
    public let options: SpecializationOptions
    public let descriptor: JSONValue
    /// AIModel(contentsOf:) and loadFunction(named: "main"), seconds
    public let loadSeconds: (model: Double, function: Double)
    public let output: String
    public let outputShape: [Int]

    private let model: AIModel
    private let main: InferenceFunction
    private let descriptors: [String: NDArrayDescriptor]
    private let specs: [String: TensorSpec]

    init(contentsOf url: URL, inputs want: [String: TensorSpec], output: (String, TensorSpec),
         options: SpecializationOptions) async throws {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = secondsSince(t0)
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: "main"), let fn = try model.loadFunction(named: "main") else {
            throw D1OmniError.contract("\(url.lastPathComponent): no function \"main\" (functions \(model.functionNames))")
        }
        let tMain = secondsSince(t1)
        var bad: [String] = []
        if Set(md.inputNames) != Set(want.keys) { bad.append("inputs \(md.inputNames.sorted()) != \(want.keys.sorted())") }
        for (name, spec) in want where TensorSpec.of(md.inputDescriptor(of: name)) != spec {
            bad.append("\(name) \(TensorSpec.of(md.inputDescriptor(of: name))?.description ?? "missing") != \(spec)")
        }
        if md.outputNames != [output.0] || TensorSpec.of(md.outputDescriptor(of: output.0)) != output.1 {
            bad.append("outputs \(md.outputNames) \(TensorSpec.of(md.outputDescriptor(of: output.0))?.description ?? "") != \(output.0) \(output.1)")
        }
        if !md.stateNames.isEmpty { bad.append("states \(md.stateNames)") }
        if !bad.isEmpty { throw D1OmniError.contract("\(url.lastPathComponent) main: \(bad.joined(separator: "; "))") }
        var d: [String: NDArrayDescriptor] = [:]
        for name in want.keys { d[name] = ND.descriptor(md.inputDescriptor(of: name))! }
        self.url = url
        self.kind = url.pathExtension == "aimodelc" ? "aot" : "jit"
        self.options = options
        self.descriptor = describe(md)
        self.loadSeconds = (tModel, tMain)
        self.output = output.0
        self.outputShape = output.1.shape
        self.model = model
        self.main = fn
        self.descriptors = d
        self.specs = want
    }

    /// One call -> the output, flat row-major float32.
    public func run(_ inputs: [String: Input]) async throws -> [Float] {
        var arrays: [String: NDArray] = [:]
        for (name, spec) in specs {
            guard let x = inputs[name] else { throw D1OmniError.request("missing input \(name)") }
            let count = spec.shape.reduce(1, *)
            switch x {
            case .float(let v):
                guard spec.type == .float32, v.count == count else { throw D1OmniError.request("\(name): \(v.count) float32 for \(spec)") }
                arrays[name] = ND.make(v, descriptors[name]!)
            case .int32(let v):
                guard spec.type == .int32, v.count == count else { throw D1OmniError.request("\(name): \(v.count) int32 for \(spec)") }
                arrays[name] = ND.make(v, descriptors[name]!)
            }
        }
        var outputs = try await main.run(inputs: arrays)
        guard let array = outputs.remove(output)?.ndArray else { throw D1OmniError.contract("no \(output) in the outputs") }
        let out = ND.read(array, as: Float.self)
        guard out.count == outputShape.reduce(1, *) else { throw D1OmniError.contract("\(out.count) values of \(output) for \(outputShape)") }
        return out
    }
}

/// The vision graph: one crop -> its prefix rows.
public final class VisionGraph: @unchecked Sendable {
    public static let names = ["pixel_values", "pos_embed", "patch_mask", "unshuffle_index"]
    public let graph: MediaGraph

    public init(contentsOf url: URL, options: SpecializationOptions = DecisionGraph.gpuOptions) async throws {
        let p = ImagePreprocess.maxPatches, d = ImagePreprocess.patchDim
        graph = try await MediaGraph(
            contentsOf: url,
            inputs: ["pixel_values": TensorSpec(shape: [1, p, d], type: .float32),
                     "pos_embed": TensorSpec(shape: [1, p, ImagePreprocess.hidden], type: .float32),
                     "patch_mask": TensorSpec(shape: [1, p], type: .float32),
                     "unshuffle_index": TensorSpec(shape: [ImagePreprocess.maxTokens, 4], type: .int32)],
            output: ("prefix", TensorSpec(shape: [1, ImagePreprocess.maxTokens, DecisionGraph.hidden], type: .float32)),
            options: options)
    }

    /// The graph's whole output [256 * 1024] for one crop's inputs.
    public func output(_ x: CropInputs) async throws -> [Float] {
        try await graph.run(["pixel_values": .float(x.pixelValues), "pos_embed": .float(x.posEmbed),
                             "patch_mask": .float(x.patchMask), "unshuffle_index": .int32(x.unshuffleIndex)])
    }

    /// The crop's prefix: the first (ph / 2)(pw / 2) rows of the output, [tokens * 1024].
    public func prefix(_ x: CropInputs) async throws -> [Float] {
        Array(try await output(x).prefix(x.tokens * DecisionGraph.hidden))
    }
}

/// The audio graph of one clip bucket: one clip's mel and masks -> its prefix rows.
public final class AudioGraph: @unchecked Sendable {
    public static let names = ["mel", "mask_f", "mask_f2", "mask_f4", "mask_t"]
    public let graph: MediaGraph
    public let bucket: AudioBucket

    public init(contentsOf url: URL, bucket: AudioBucket, options: SpecializationOptions = DecisionGraph.gpuOptions) async throws {
        self.bucket = bucket
        graph = try await MediaGraph(
            contentsOf: url,
            inputs: ["mel": TensorSpec(shape: [1, AudioPreprocess.features, bucket.F], type: .float32),
                     "mask_f": TensorSpec(shape: [1, bucket.F], type: .float32),
                     "mask_f2": TensorSpec(shape: [1, bucket.F2], type: .float32),
                     "mask_f4": TensorSpec(shape: [1, bucket.F4], type: .float32),
                     "mask_t": TensorSpec(shape: [1, bucket.T], type: .float32)],
            output: ("prefix", TensorSpec(shape: [1, bucket.T, DecisionGraph.hidden], type: .float32)),
            options: options)
    }

    /// The graph's whole output [T * 1024] for one clip's inputs (at this bucket).
    public func output(_ x: AudioInputs) async throws -> [Float] {
        guard x.bucket == bucket else { throw D1OmniError.request("inputs of the \(x.bucket.seconds) s bucket on the \(bucket.seconds) s graph") }
        return try await graph.run(["mel": .float(x.mel), "mask_f": .float(x.maskF), "mask_f2": .float(x.maskF2),
                                    "mask_f4": .float(x.maskF4), "mask_t": .float(x.maskT)])
    }

    /// The clip's prefix: the first P rows of the output, [P * 1024].
    public func prefix(_ x: AudioInputs) async throws -> [Float] {
        Array(try await output(x).prefix(x.prefixRows * DecisionGraph.hidden))
    }
}
