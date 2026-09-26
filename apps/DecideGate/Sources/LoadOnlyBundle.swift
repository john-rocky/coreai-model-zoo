// LoadOnlyBundle — any exported graph on the system CoreAI framework, for the load-only mode (DECIDE_LOAD_ONLY):
// AIModel(contentsOf:options:) with preferred gpu, the function `main` as the bundle declares it, and one call on
// zero inputs. No model's contract is assumed (DecideGraph checks GLiNER2.5-Decide's): the inputs are read from the
// function descriptor, and a call is made only when every input is a static-shape array of a type filled here.

import CoreAI
import Foundation

enum LoadOnlyError: Error, CustomStringConvertible, Sendable {
    case bundle(String)
    case missingFile(String)
    case iosBundleOnMac(String)
    case functionNotFound(String, available: [String])
    case input(String)

    var description: String {
        switch self {
        case .bundle(let s): return "bundle: \(s)"
        case .missingFile(let p): return "missing file \(p)"
        case .iosBundleOnMac(let p): return "refusing an iPhone AOT bundle on macOS: \(p)"
        case .functionNotFound(let n, let names): return "function '\(n)' not in the bundle (functions: \(names))"
        case .input(let s): return "input: \(s)"
        }
    }
}

/// One output of a call: what the graph returned, and how many of its values are NaN / inf (float outputs).
struct LoadOnlyOutput: Sendable {
    let name: String
    let scalarType: String
    let shape: [Int]
    let nonFinite: Int?

    var json: [String: Any] {
        var j: [String: Any] = ["name": name, "scalar_type": scalarType, "shape": shape]
        if let n = nonFinite { j["nonfinite"] = n }
        return j
    }
}

struct LoadOnlyCall: Sendable {
    /// function.run alone (the zero inputs are made before it, the outputs read after it)
    let runSeconds: Double
    let outputs: [LoadOnlyOutput]
}

struct LoadOnlyBundle: Sendable {
    static let functionName = "main"
    static let zeroFilled: Set<NDArray.ScalarType> = [.int32, .int64, .float32, .float16]

    let model: AIModel                          // held for the function's lifetime
    let function: InferenceFunction
    let descriptor: InferenceFunctionDescriptor
    let functionNames: [String]
    /// AIModel(contentsOf:options:) (the specialization, JIT or AOT, happens here)
    let modelSeconds: Double
    /// functionDescriptor(for:) + loadFunction(named:)
    let functionSeconds: Double
    /// the whole load, as DecideGraph.loadSeconds counts it
    var loadSeconds: Double { modelSeconds + functionSeconds }

    init(contentsOf url: URL) async throws {
        #if os(macOS)
        // an iPhone AOT bundle (.h18p. / .h19p. ...) must never be loaded on a Mac (it wedges the GPU stack until a reboot)
        if url.path.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil { throw LoadOnlyError.iosBundleOnMac(url.path) }
        #endif
        guard FileManager.default.fileExists(atPath: url.path) else { throw LoadOnlyError.missingFile(url.path) }
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        let t1 = ContinuousClock.now
        guard let descriptor = model.functionDescriptor(for: Self.functionName),
              let function = try model.loadFunction(named: Self.functionName) else {
            throw LoadOnlyError.functionNotFound(Self.functionName, available: model.functionNames)
        }
        let t2 = ContinuousClock.now
        self.model = model
        self.function = function
        self.descriptor = descriptor
        functionNames = model.functionNames
        modelSeconds = Self.seconds(t1 - t0)
        functionSeconds = Self.seconds(t2 - t1)
    }

    /// Why no call on zero inputs is made (nil: one is): states, an input that is not an array, a dynamic or empty
    /// dimension, or a scalar type not filled here.
    var callSkipReason: String? {
        if !descriptor.stateNames.isEmpty {
            return "states \(descriptor.stateNames.joined(separator: ", ")): not driven in the load-only mode"
        }
        for name in descriptor.inputNames {
            guard let value = descriptor.inputDescriptor(of: name) else { return "input \(name) has no descriptor" }
            switch value {
            case .ndArray(let d):
                if d.hasDynamicShape || d.shape.contains(where: { $0 <= 0 }) { return "dynamic shape \(name) \(d.shape)" }
                if !Self.zeroFilled.contains(d.scalarType) { return "input \(name) is \(d.scalarType): no zero fill for it" }
            case .image:
                return "input \(name) is an image"
            @unknown default:
                return "input \(name): unknown descriptor kind"
            }
        }
        return nil
    }

    /// One call with every input all zeros; the outputs' names, types, shapes and non-finite counts.
    func runZeros() async throws -> LoadOnlyCall {
        var inputs: [String: NDArray] = [:]
        for name in descriptor.inputNames {
            guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else { throw LoadOnlyError.input("\(name) is not an array") }
            inputs[name] = try Self.zeros(d, name)
        }
        let t0 = ContinuousClock.now
        var outputs = try await function.run(inputs: inputs)
        let runSeconds = Self.seconds(ContinuousClock.now - t0)
        var records: [LoadOnlyOutput] = []
        for name in Array(outputs.names) {
            guard let array = outputs.remove(name)?.ndArray else {
                records.append(LoadOnlyOutput(name: name, scalarType: "not an array", shape: [], nonFinite: nil))
                continue
            }
            records.append(LoadOnlyOutput(name: name, scalarType: "\(array.scalarType)", shape: array.shape,
                                          nonFinite: Self.nonFiniteCount(array)))
        }
        return LoadOnlyCall(runSeconds: runSeconds, outputs: records)
    }

    /// "name type [shape], ... | states ... -> name type [shape], ..." as the bundle declares them.
    var contractDescription: String {
        let ins = descriptor.inputNames.map { Self.text($0, descriptor.inputDescriptor(of: $0)) }
        let states = descriptor.stateNames.map { Self.text($0, descriptor.stateDescriptor(of: $0)) }
        let outs = descriptor.outputNames.map { Self.text($0, descriptor.outputDescriptor(of: $0)) }
        return ins.joined(separator: ", ") + (states.isEmpty ? "" : " | states " + states.joined(separator: ", "))
            + " -> " + outs.joined(separator: ", ")
    }

    /// The same, one record per value: {name, kind, scalar_type, shape}.
    var contractJSON: [String: Any] {
        ["function": descriptor.name, "functions": functionNames,
         "inputs": descriptor.inputNames.map { Self.record($0, descriptor.inputDescriptor(of: $0)) },
         "states": descriptor.stateNames.map { Self.record($0, descriptor.stateDescriptor(of: $0)) },
         "outputs": descriptor.outputNames.map { Self.record($0, descriptor.outputDescriptor(of: $0)) }]
    }

    static func text(_ name: String, _ value: InferenceValue.Descriptor?) -> String {
        guard let value else { return "\(name) ?" }
        switch value {
        case .ndArray(let d): return "\(name) \(d.scalarType) \(d.shape)"
        case .image(let i): return "\(name) image \(i.width)x\(i.height)"
        @unknown default: return "\(name) ?"
        }
    }

    static func record(_ name: String, _ value: InferenceValue.Descriptor?) -> [String: Any] {
        guard let value else { return ["name": name, "kind": "none"] }
        switch value {
        case .ndArray(let d):
            return ["name": name, "kind": "ndArray", "scalar_type": "\(d.scalarType)", "shape": d.shape,
                    "dynamic": d.hasDynamicShape]
        case .image(let i):
            return ["name": name, "kind": "image", "width": i.width, "height": i.height, "pixel_format": Int(i.pixelFormatType)]
        @unknown default:
            return ["name": name, "kind": "unknown"]
        }
    }

    static func zeros(_ d: NDArrayDescriptor, _ name: String) throws -> NDArray {
        var array = NDArray(descriptor: d)
        let n = d.shape.reduce(1, *)
        switch d.scalarType {
        case .int32:
            var view = array.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: repeatElement(Int32(0), count: n))
        case .int64:
            var view = array.mutableView(as: Int64.self)
            view.copyElements(fromContentsOf: repeatElement(Int64(0), count: n))
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: repeatElement(Float(0), count: n))
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: repeatElement(Float16(0), count: n))
        default:
            throw LoadOnlyError.input("\(name) is \(d.scalarType): no zero fill for it")
        }
        return array
    }

    /// NaN / inf values of a float32 / float16 array, read in place through its strides (nil for other types).
    static func nonFiniteCount(_ array: NDArray) -> Int? {
        let shape = array.shape, strides = array.strides
        switch array.scalarType {
        case .float32:
            return array.view(as: Float.self).withUnsafePointer { ptr, _, _ in count(ptr, shape, strides) { !$0.isFinite } }
        case .float16:
            return array.view(as: Float16.self).withUnsafePointer { ptr, _, _ in count(ptr, shape, strides) { !$0.isFinite } }
        default:
            return nil
        }
    }

    static func count<T>(_ ptr: UnsafePointer<T>, _ shape: [Int], _ strides: [Int], _ bad: (T) -> Bool) -> Int {
        let total = shape.reduce(1, *)
        var dense = [Int](repeating: 1, count: shape.count)
        for d in stride(from: shape.count - 2, through: 0, by: -1) { dense[d] = dense[d + 1] * shape[d + 1] }
        var n = 0
        if strides == dense || strides.isEmpty {
            for i in 0..<total where bad(ptr[i]) { n += 1 }
            return n
        }
        for flat in 0..<total {
            var rem = flat, offset = 0
            for d in 0..<shape.count {
                let i = rem / dense[d]
                rem -= i * dense[d]
                offset += i * strides[d]
            }
            if bad(ptr[offset]) { n += 1 }
        }
        return n
    }

    static func seconds(_ d: Duration) -> Double {
        Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }
}
