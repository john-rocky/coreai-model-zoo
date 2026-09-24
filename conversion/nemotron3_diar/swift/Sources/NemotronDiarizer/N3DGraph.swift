// N3DGraph — the exported fixed-T graph (function `main`) on the system CoreAI framework:
//   packed [1, T, 512] f32 + valid [1, T] f32 -> logits [1, T*8, 8] f32
// (both the float16 and float32 bundles have float32 I/O; float16 inputs are converted if a bundle
// declares them). The load / call pattern follows CoreAIKit's GraphModel; no dependency on it.

import CoreAI
import Foundation

public enum N3DComputeUnits: String, Sendable, CaseIterable {
    /// Preferred GPU over the full allowed set (the ship configuration on Mac).
    case gpu
    /// Preferred Neural Engine.
    case ane
    /// The CPU alone (the parity configuration for the float32 bundle).
    case cpuOnly
    /// `SpecializationOptions.default` (an AOT `.aimodelc` keeps its compiled placement).
    case `default`

    public var specializationOptions: SpecializationOptions {
        switch self {
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .ane: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .cpuOnly: return .cpuOnly
        case .default: return .default
        }
    }
}

struct N3DGraph: Sendable {
    static let functionName = "main"
    let function: InferenceFunction
    let packedDescriptor: NDArrayDescriptor
    let validDescriptor: NDArrayDescriptor
    let length: Int                                    // T
    let loadSeconds: Double

    init(contentsOf url: URL, computeUnits: N3DComputeUnits) async throws {
        #if os(macOS)
        // an iOS AOT bundle must never be loaded on a Mac (it wedges the GPU stack until a reboot)
        if url.lastPathComponent.contains(".h18p.") { throw N3DError.iosBundleOnMac(url.path) }
        #endif
        guard FileManager.default.fileExists(atPath: url.path) else { throw N3DError.missingFile(url.path) }
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: computeUnits.specializationOptions)
        guard let descriptor = model.functionDescriptor(for: Self.functionName) else {
            throw N3DError.functionNotFound(Self.functionName)
        }
        guard case .ndArray(let pd) = descriptor.inputDescriptor(of: "packed"),
              case .ndArray(let vd) = descriptor.inputDescriptor(of: "valid") else {
            throw N3DError.contract("inputs \(descriptor.inputNames), expected packed + valid")
        }
        guard descriptor.outputNames.contains("logits") else {
            throw N3DError.contract("outputs \(descriptor.outputNames), expected logits")
        }
        guard pd.shape.count == 3, pd.shape[0] == 1, pd.shape[2] == N3DSpeakerCache.hidden,
              vd.shape == [1, pd.shape[1]] else {
            throw N3DError.contract("packed \(pd.shape) / valid \(vd.shape)")
        }
        if case .ndArray(let od) = descriptor.outputDescriptor(of: "logits") {
            guard od.shape == [1, pd.shape[1] * N3DSpeakerCache.subsampling, N3DSpeakerCache.numSpeakers] else {
                throw N3DError.contract("logits \(od.shape)")
            }
        }
        guard let function = try model.loadFunction(named: Self.functionName) else {
            throw N3DError.functionNotFound(Self.functionName)
        }
        self.function = function
        packedDescriptor = pd
        validDescriptor = vd
        length = pd.shape[1]
        let d = ContinuousClock.now - t0
        loadSeconds = Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    /// One call: packed [T * 512], valid [T] -> logits [T * 8 * 8] (row-major, float32).
    func run(packed: [Float], valid: [Float]) async throws -> [Float] {
        let p = try Self.makeArray(packedDescriptor, packed)
        let v = try Self.makeArray(validDescriptor, valid)
        var outputs = try await function.run(inputs: ["packed": p, "valid": v])
        guard let logits = outputs.remove("logits")?.ndArray else { throw N3DError.graphOutput("no logits") }
        let out = try Self.readFloats(logits)
        guard out.count == length * N3DSpeakerCache.subsampling * N3DSpeakerCache.numSpeakers else {
            throw N3DError.graphOutput("logits has \(out.count) values")
        }
        return out
    }

    static func makeArray(_ descriptor: NDArrayDescriptor, _ values: [Float]) throws -> NDArray {
        var array = NDArray(descriptor: descriptor)
        switch descriptor.scalarType {
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: values)
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: values.map { Float16($0) })
        default:
            throw N3DError.contract("input scalar type \(descriptor.scalarType)")
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
            throw N3DError.graphOutput("scalar type \(array.scalarType)")
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
