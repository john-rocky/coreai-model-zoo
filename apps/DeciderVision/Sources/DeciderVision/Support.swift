// Support — NDArray fill/read/zero on the system CoreAI framework (strides honored), the contract checks the loaders
// share, and the two NumPy reductions the reference numbers go through (float64 pairwise sum, softmax at T = 1), so
// a probability here is the same double the Python gate computes from the same logits.

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public enum DeciderVisionError: Error, CustomStringConvertible, Sendable {
    case contract(String)
    case image(String)
    case prompt(String)
    case bundle(String)

    public var description: String {
        switch self {
        case .contract(let s): return "contract: \(s)"
        case .image(let s): return "image: \(s)"
        case .prompt(let s): return "prompt: \(s)"
        case .bundle(let s): return "bundle: \(s)"
        }
    }
}

@available(macOS 27, iOS 27, *)
enum ND {
    static func descriptor(_ value: InferenceValue.Descriptor?) -> NDArrayDescriptor? {
        guard case .ndArray(let d) = value else { return nil }
        return d
    }

    /// A new array for `descriptor`, filled row-major from `values`.
    static func make<T: BitwiseCopyable>(_ values: [T], _ descriptor: NDArrayDescriptor) -> NDArray {
        var array = NDArray(descriptor: descriptor)
        var view = array.mutableView(as: T.self)
        view.copyElements(fromContentsOf: values)
        return array
    }

    /// Row-major copy of an array's scalars, honoring its strides (in elements).
    static func read<T: BitwiseCopyable>(_ array: NDArray, as type: T.Type) -> [T] {
        let shape = array.shape
        let count = shape.reduce(1, *)
        return array.view(as: T.self).withUnsafePointer { ptr, _, strides in
            var expected = 1
            var contiguous = true
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                if shape[d] > 1 && strides[d] != expected {
                    contiguous = false
                    break
                }
                expected *= shape[d]
            }
            if contiguous { return Array(UnsafeBufferPointer(start: ptr, count: count)) }
            var out = [T]()
            out.reserveCapacity(count)
            var index = [Int](repeating: 0, count: shape.count)
            for _ in 0..<count {
                var offset = 0
                for d in 0..<shape.count { offset += index[d] * strides[d] }
                out.append(ptr[offset])
                var d = shape.count - 1
                while d >= 0 {
                    index[d] += 1
                    if index[d] < shape[d] { break }
                    index[d] = 0
                    d -= 1
                }
            }
            return out
        }
    }

    /// Sets every scalar of the array (and any padding between its rows) to zero.
    static func zero(_ array: inout NDArray) {
        let shape = array.shape
        switch array.scalarType {
        case .float16: zeroScalars(&array, Float16(0), shape)
        case .float32: zeroScalars(&array, Float(0), shape)
        case .int32: zeroScalars(&array, Int32(0), shape)
        default:
            fatalError("ND.zero: unsupported scalar type \(array.scalarType)")
        }
    }

    private static func zeroScalars<T: BitwiseCopyable>(_ array: inout NDArray, _ zero: T, _ shape: [Int]) {
        array.mutableView(as: T.self).withUnsafeMutablePointer { ptr, _, strides in
            var extent = 1
            for d in 0..<shape.count where shape[d] > 0 { extent += (shape[d] - 1) * strides[d] }
            ptr.update(repeating: zero, count: extent)
        }
    }
}

/// The contract a loaded function must meet: input / output / state names with shapes (-1 = dynamic) and types.
@available(macOS 27, iOS 27, *)
struct TensorSpec: Equatable, CustomStringConvertible {
    let shape: [Int]
    let type: NDArray.ScalarType

    var description: String { "\(shape) \(type)" }

    static func of(_ value: InferenceValue.Descriptor?) -> TensorSpec? {
        ND.descriptor(value).map { TensorSpec(shape: $0.shape, type: $0.scalarType) }
    }
}

@available(macOS 27, iOS 27, *)
func describe(_ d: InferenceFunctionDescriptor) -> [String: Any] {
    func table(_ names: [String], _ get: (String) -> InferenceValue.Descriptor?) -> [String: Any] {
        var out: [String: Any] = [:]
        for n in names {
            if let s = TensorSpec.of(get(n)) { out[n] = ["shape": s.shape, "type": "\(s.type)"] } else { out[n] = "non-ndarray" }
        }
        return out
    }
    return ["inputs": table(d.inputNames, d.inputDescriptor(of:)),
            "outputs": table(d.outputNames, d.outputDescriptor(of:)),
            "states": table(d.stateNames, d.stateDescriptor(of:))]
}

/// NumPy's float64 `add.reduce` over a contiguous 1-D array (pairwise summation, 8 accumulators per block of at most
/// 128): the order `w.sum()` and `p.sum()` add in, so the weights and probabilities here equal the reference's bits.
func numpySum(_ a: [Double]) -> Double {
    a.withUnsafeBufferPointer { numpyPairwise($0.baseAddress!, a.count) }
}

private func numpyPairwise(_ a: UnsafePointer<Double>, _ n: Int) -> Double {
    if n < 8 {
        var res = 0.0
        for i in 0..<n { res += a[i] }
        return res
    } else if n <= 128 {
        var r = (0..<8).map { a[$0] }
        var i = 8
        while i < n - (n % 8) {
            for j in 0..<8 { r[j] += a[i + j] }
            i += 8
        }
        var res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n {
            res += a[i]
            i += 1
        }
        return res
    } else {
        var n2 = n / 2
        n2 -= n2 % 8
        return numpyPairwise(a, n2) + numpyPairwise(a + n2, n - n2)
    }
}

/// Softmax at T = 1 in float64, as the gates compute it: exp(x - max) / sum.
@available(macOS 27, iOS 27, *)
public func softmax(_ logits: [Double]) -> [Double] {
    guard let m = logits.max() else { return [] }
    let e = logits.map { Foundation.exp($0 - m) }
    let s = numpySum(e)
    return e.map { $0 / s }
}

/// The first index of the largest value (numpy.argmax).
func argmax<T: Comparable>(_ xs: [T]) -> Int {
    var best = 0
    for i in 1..<xs.count where xs[i] > xs[best] { best = i }
    return best
}

func secondsSince(_ t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}
