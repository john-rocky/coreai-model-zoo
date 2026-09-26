// DecideGraph — the exported GLiNER2.5-Decide graph (function `main`) on the system CoreAI framework:
//   input_ids [1, S] i32 + attention_mask [1, S] i32 + label_idx [1, MMAX] i32 -> logits [1, MMAX] f32
// S and MMAX come from the bundle's descriptor (the shipped bundles: S = 256 or 512, MMAX = 32). The graph is
// DeBERTa-v3-large -> the [L] rows picked by label_idx -> the 1024-2048-1 label head; softmax / sigmoid and the
// threshold run on the host (DecideGateSupport). The load / call pattern follows N3DGraph
// (conversion/nemotron3_diar/swift); no dependency on it.

import CoreAI
import Foundation

public enum DecideComputeUnits: String, Sendable, CaseIterable {
    /// Preferred GPU over the full allowed set (the ship configuration).
    case gpu
    /// The CPU alone (the parity configuration, not a performance one).
    case cpuOnly
    /// `SpecializationOptions.default` (an AOT `.aimodelc` keeps its compiled placement).
    case `default`

    public var specializationOptions: SpecializationOptions {
        switch self {
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .cpuOnly: return .cpuOnly
        case .default: return .default
        }
    }
}

public enum DecideError: Error, CustomStringConvertible, Sendable {
    case missingFile(String)
    case functionNotFound(String)
    case contract(String)
    case iosBundleOnMac(String)
    case graphOutput(String)

    public var description: String {
        switch self {
        case .missingFile(let p): return "missing file \(p)"
        case .functionNotFound(let n): return "function '\(n)' not in the bundle"
        case .contract(let s): return "graph contract: \(s)"
        case .iosBundleOnMac(let p): return "refusing an iPhone AOT bundle on macOS: \(p)"
        case .graphOutput(let s): return "graph output: \(s)"
        }
    }
}

public struct DecideGraph: Sendable {
    public static let functionName = "main"
    public static let inputNames = ["input_ids", "attention_mask", "label_idx"]
    /// The architecture Core AI specializes for on this device (e.g. h16c on an M4 Mac).
    public static var deviceArchitecture: String { AIModel.deviceArchitectureName }

    let model: AIModel                          // held for the function's lifetime (a function outliving its
                                                // model returns garbage: knowledge/conversion-guide.md)
    let function: InferenceFunction
    let idsDescriptor: NDArrayDescriptor
    let maskDescriptor: NDArrayDescriptor
    let labelDescriptor: NDArrayDescriptor
    public let url: URL
    public let computeUnits: DecideComputeUnits
    /// S: tokens per call (input_ids / attention_mask length).
    public let seqLength: Int
    /// MMAX: label slots per call (label_idx / logits length).
    public let maxLabels: Int
    public let loadSeconds: Double
    /// "input_ids int32 [1, 256], ..." as the bundle declares them.
    public let contractDescription: String

    public init(contentsOf url: URL, computeUnits: DecideComputeUnits) async throws {
        #if os(macOS)
        // an iPhone AOT bundle (.h18p. / .h19p. ...) must never be loaded on a Mac (it wedges the GPU stack until a reboot)
        if url.path.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil { throw DecideError.iosBundleOnMac(url.path) }
        #endif
        guard FileManager.default.fileExists(atPath: url.path) else { throw DecideError.missingFile(url.path) }
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: computeUnits.specializationOptions)
        guard let descriptor = model.functionDescriptor(for: Self.functionName) else {
            throw DecideError.functionNotFound(Self.functionName)
        }
        guard case .ndArray(let ids) = descriptor.inputDescriptor(of: "input_ids"),
              case .ndArray(let mask) = descriptor.inputDescriptor(of: "attention_mask"),
              case .ndArray(let labels) = descriptor.inputDescriptor(of: "label_idx") else {
            throw DecideError.contract("inputs \(descriptor.inputNames), expected \(Self.inputNames)")
        }
        guard descriptor.outputNames.contains("logits") else {
            throw DecideError.contract("outputs \(descriptor.outputNames), expected logits")
        }
        guard ids.shape.count == 2, ids.shape[0] == 1, mask.shape == ids.shape,
              labels.shape.count == 2, labels.shape[0] == 1 else {
            throw DecideError.contract("input_ids \(ids.shape) / attention_mask \(mask.shape) / label_idx \(labels.shape)")
        }
        var outText = "logits ?"
        if case .ndArray(let od) = descriptor.outputDescriptor(of: "logits") {
            guard od.shape == [1, labels.shape[1]] else { throw DecideError.contract("logits \(od.shape)") }
            outText = "logits \(od.scalarType) \(od.shape)"
        }
        guard let function = try model.loadFunction(named: Self.functionName) else {
            throw DecideError.functionNotFound(Self.functionName)
        }
        self.model = model
        self.function = function
        self.url = url
        self.computeUnits = computeUnits
        idsDescriptor = ids
        maskDescriptor = mask
        labelDescriptor = labels
        seqLength = ids.shape[1]
        maxLabels = labels.shape[1]
        contractDescription = "input_ids \(ids.scalarType) \(ids.shape), attention_mask \(mask.scalarType) \(mask.shape), "
            + "label_idx \(labels.scalarType) \(labels.shape) -> \(outText)"
        let d = ContinuousClock.now - t0
        loadSeconds = Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    /// One call: input_ids / mask [S], labelIdx [MMAX] -> logits [MMAX] (float32).
    public func run(inputIds: [Int32], mask: [Int32], labelIdx: [Int32]) async throws -> [Float] {
        guard inputIds.count == seqLength, mask.count == seqLength, labelIdx.count == maxLabels else {
            throw DecideError.contract("got \(inputIds.count) ids / \(mask.count) mask / \(labelIdx.count) label_idx, "
                                       + "the bundle takes S=\(seqLength), MMAX=\(maxLabels)")
        }
        let ids = try Self.makeArray(idsDescriptor, inputIds)
        let m = try Self.makeArray(maskDescriptor, mask)
        let l = try Self.makeArray(labelDescriptor, labelIdx)
        var outputs = try await function.run(inputs: ["input_ids": ids, "attention_mask": m, "label_idx": l])
        guard let logits = outputs.remove("logits")?.ndArray else { throw DecideError.graphOutput("no logits") }
        let out = try Self.readFloats(logits)
        guard out.count == maxLabels else { throw DecideError.graphOutput("logits has \(out.count) values") }
        return out
    }

    static func makeArray(_ descriptor: NDArrayDescriptor, _ values: [Int32]) throws -> NDArray {
        var array = NDArray(descriptor: descriptor)
        switch descriptor.scalarType {
        case .int32:
            var view = array.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: values)
        case .int64:
            var view = array.mutableView(as: Int64.self)
            view.copyElements(fromContentsOf: values.map { Int64($0) })
        default:
            throw DecideError.contract("input scalar type \(descriptor.scalarType)")
        }
        return array
    }

    /// Row-major values of an output array, honouring its strides (element units).
    static func readFloats(_ array: NDArray) throws -> [Float] {
        let shape = array.shape, strides = array.strides
        switch array.scalarType {
        case .float32:
            return array.view(as: Float.self).withUnsafePointer { ptr, _, _ in gather(ptr, shape, strides) { $0 } }
        case .float16:
            return array.view(as: Float16.self).withUnsafePointer { ptr, _, _ in gather(ptr, shape, strides) { Float($0) } }
        default:
            throw DecideError.graphOutput("scalar type \(array.scalarType)")
        }
    }

    static func gather<T>(_ ptr: UnsafePointer<T>, _ shape: [Int], _ strides: [Int], _ convert: (T) -> Float) -> [Float] {
        let count = shape.reduce(1, *)
        var dense = [Int](repeating: 1, count: shape.count)
        for d in stride(from: shape.count - 2, through: 0, by: -1) { dense[d] = dense[d + 1] * shape[d + 1] }
        if strides == dense || strides.isEmpty {
            return (0..<count).map { convert(ptr[$0]) }
        }
        var out = [Float](repeating: 0, count: count)
        for flat in 0..<count {
            var rem = flat, offset = 0
            for d in 0..<shape.count {
                let i = rem / dense[d]
                rem -= i * dense[d]
                offset += i * strides[d]
            }
            out[flat] = convert(ptr[offset])
        }
        return out
    }
}
