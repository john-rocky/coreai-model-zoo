// Decoder — the d1 decoder graph on the low-level runtime (AIModel + loadFunction + MutableViews states): one static-S
// function `main` that returns the final-norm hidden state at every position of a call (no vocabulary head in the
// graph). The loop is apps/Kev's KevDecoder (zoo d1-3b 8e82d36; Kev's is apps/ClefFlash's ClefDecoder) with d1's three
// states and its static image rows (ClefFlash's image_embeds input).
//
//   inputs   input_ids [1, S] i32 (static S = metadata language.prefill_chunk), position_ids [1, -1] i32 (0 ..< p + S
//            for a call after p ids), image_embeds [N, d] f16 (static; zero for a text request)
//   states   keyCache / valueCache [n_full, 1, n_kv, -1, head] f16 (the dynamic axis resolved to max_context_length),
//            convState [n_conv, 1, d, w] f16 — allocated once, zeroed at the start of every row
//   output   hidden [1, S, d] f16
//
// Every name, shape and type is the bundle's metadata.json `language.contract`; the loaded function's descriptor is
// checked against it at load (a different graph fails there, not in a probability).
//
// Read order (metadata `decision.readout`, conversion/d1/host.py §7): a row of T ids runs from zero states as
// ceil(T / S) calls of S ids, call c with position_ids 0 ..< p + cS + S; the last call is padded with the pad id and the
// padded positions' rows are dropped (causal: they cannot reach a real position). The image rows are one buffer of the
// decoder's, written once per request (`setImageRows`) and bound to every call; a text request leaves it zero.
//
// The shared prefix: the first k = floor(Ls / S) * S ids, the same in every row of a request, run once from zero
// states; the three states are then copied whole into a snapshot and every row runs the rest of its ids from a copy of
// it, positions continuing at k. On a static-S graph the calls and their inputs are a direct run's, so are the hidden
// rows (knowledge/kev-port.md "The shared prefix is exact on a static graph"). `prepare` / `runPrepared` are the same
// two halves with the states kept by the caller.

import CoreAI
import CryptoKit
import Foundation

public final class D1Decoder: @unchecked Sendable {
    /// One row's pass (or a row's part after a shared prefix): the hidden rows and each call's time.
    public struct Pass: Sendable {
        /// [tokens * d] fp16, row-major: the final-norm hidden state of every real position.
        public let hidden: [Float16]
        public let tokens: Int
        public let calls: Int
        public let callSeconds: [Double]
        /// zeroing (or restoring) the three states before the pass
        public let resetSeconds: Double
        public let seconds: Double
    }

    public let url: URL
    public let chunk: Int
    public let hidden: Int
    public let imageRows: Int
    public let maxContext: Int
    public let padID: Int32
    public let functionNames: [String]
    public let loadSeconds: (model: Double, function: Double)
    public let descriptor: JSONValue
    public let options: SpecializationOptions
    /// zeroing the image buffer and the three states at load
    public let allocationSeconds: Double

    private let main: InferenceFunction
    private let idsDescriptor: NDArrayDescriptor
    private let positions: NDArrayDescriptor
    /// The three states' descriptors, the dynamic length resolved to max_context_length.
    private let stateDescriptors: [String: NDArrayDescriptor]
    // The states, the image rows and the hidden buffer, allocated once. A pass moves them into locals for its whole run:
    // MutableViews borrows what it holds up to `run`, which a class property's access scope does not cover.
    private var buffers: Buffers?

    struct Buffers {
        var keyCache: NDArray
        var valueCache: NDArray
        var convState: NDArray
        var hidden: NDArray
        var image: NDArray
        /// how many leading image rows hold values (0 = the buffer is all zero)
        var imageRowsSet: Int
    }

    public static let stateNames = ["keyCache", "valueCache", "convState"]

    /// Loads the decoder asset (`.aimodelc` AOT or `.aimodel` JIT) and checks `main` against the bundle's contract.
    public init(contentsOf url: URL, contract: D1GraphContract, maxContext: Int, padID: Int,
                options: SpecializationOptions) async throws
    {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = d1Seconds(since: t0)
        functionNames = model.functionNames
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: contract.function),
              let mainFn = try model.loadFunction(named: contract.function)
        else {
            throw D1Error.contract("decoder \(url.lastPathComponent): no \"\(contract.function)\" (functions \(model.functionNames))")
        }
        let tMain = d1Seconds(since: t1)
        try D1Contract.check(md, what: "decoder \(url.lastPathComponent) \(contract.function)", inputs: contract.inputs,
                             outputs: contract.outputs, states: contract.states)
        guard Set(contract.states.keys) == Set(Self.stateNames), contract.inputs["input_ids"] != nil,
              contract.inputs["position_ids"] != nil, contract.inputs["image_embeds"] != nil, contract.outputs["hidden"] != nil
        else { throw D1Error.contract("metadata.json language.contract: not input_ids / position_ids / image_embeds -> hidden "
            + "with keyCache / valueCache / convState") }
        let t2 = ContinuousClock.now
        var resolved: [String: NDArrayDescriptor] = [:]
        for n in Self.stateNames {
            let d = ND.descriptor(md.stateDescriptor(of: n))!
            resolved[n] = d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? maxContext : $0 })
        }
        func zeroed(_ d: NDArrayDescriptor) -> NDArray {
            var a = NDArray(descriptor: d)
            ND.zero(&a)
            return a
        }
        let imageDescriptor = ND.descriptor(md.inputDescriptor(of: "image_embeds"))!
        let hiddenDescriptor = ND.descriptor(md.outputDescriptor(of: "hidden"))!
        buffers = Buffers(keyCache: zeroed(resolved["keyCache"]!), valueCache: zeroed(resolved["valueCache"]!),
                          convState: zeroed(resolved["convState"]!), hidden: NDArray(descriptor: hiddenDescriptor),
                          image: zeroed(imageDescriptor), imageRowsSet: 0)
        allocationSeconds = d1Seconds(since: t2)
        stateDescriptors = resolved
        self.url = url
        chunk = contract.chunk
        hidden = contract.hidden
        imageRows = contract.imageRows
        self.maxContext = maxContext
        self.padID = Int32(padID)
        self.options = options
        loadSeconds = (tModel, tMain)
        descriptor = D1Contract.describe(md)
        main = mainFn
        idsDescriptor = ND.descriptor(md.inputDescriptor(of: "input_ids"))!
        positions = ND.descriptor(md.inputDescriptor(of: "position_ids"))!
    }

    /// The image rows of the next request: `rows` = n * d fp16 (n <= N) written at the top of the buffer, the rest
    /// zero; nil = a text request (zero; nothing is written when the buffer is already zero). -> seconds.
    @discardableResult
    public func setImageRows(_ rows: [Float16]?) throws -> Double {
        guard var b = buffers else { throw D1Error.contract("decoder busy: one request at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        let hidden = self.hidden
        let n = (rows?.count ?? 0) / hidden
        guard (rows?.count ?? 0) == n * hidden, n <= imageRows else {
            throw D1Error.request("images: \(n) image tokens over the graph's \(imageRows) image rows")
        }
        if n == 0 && b.imageRowsSet == 0 { return 0 }
        ND.zero(&b.image)
        if let rows, n > 0 {
            let d = hidden
            b.image.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, strides in
                // row-major [N, d]: rows of d values, `strides[0]` apart (the descriptor's own row stride)
                let rowStride = strides[0]
                rows.withUnsafeBufferPointer { src in
                    for r in 0..<n {
                        (ptr + r * rowStride).update(from: src.baseAddress! + r * d, count: d)
                    }
                }
            }
        }
        b.imageRowsSet = n
        return d1Seconds(since: t0)
    }

    /// Zeroes the states and runs one row of graph ids in S-id calls.
    public func run(ids: [Int]) async throws -> Pass {
        guard var b = buffers else { throw D1Error.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        zeroStates(&b)
        let reset = d1Seconds(since: t0)
        let (h, secs) = try await pieces(ids, from: 0, &b)
        return Pass(hidden: h, tokens: ids.count, calls: secs.count, callSeconds: secs, resetSeconds: reset,
                    seconds: d1Seconds(since: t0))
    }

    /// The shared prefix: `prefix` (a multiple of S) once from zero states, its states kept; then each of `tails` (a
    /// row after the prefix) from a copy of them, positions continuing at prefix.count -> the prefix's pass and one pass
    /// per tail.
    public func runShared(prefix: [Int], tails: [[Int]]) async throws -> (prefix: Pass, tails: [Pass]) {
        guard prefix.count % chunk == 0 else {
            throw D1Error.contract("a shared prefix of \(prefix.count) ids is not a multiple of \(chunk)")
        }
        guard var b = buffers else { throw D1Error.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        zeroStates(&b)
        let reset = d1Seconds(since: t0)
        let (hp, sp) = try await pieces(prefix, from: 0, &b)
        let tSnap = ContinuousClock.now
        var snap = Dictionary(uniqueKeysWithValues: Self.stateNames.map { ($0, NDArray(descriptor: stateDescriptors[$0]!)) })
        ND.copy(b.keyCache, into: &snap["keyCache"]!)
        ND.copy(b.valueCache, into: &snap["valueCache"]!)
        ND.copy(b.convState, into: &snap["convState"]!)
        let pre = Pass(hidden: hp, tokens: prefix.count, calls: sp.count, callSeconds: sp,
                       resetSeconds: reset + d1Seconds(since: tSnap), seconds: d1Seconds(since: t0))
        var out: [Pass] = []
        for tail in tails {
            let t1 = ContinuousClock.now
            restore(snap, into: &b)
            let restoreS = d1Seconds(since: t1)
            let (h, secs) = try await pieces(tail, from: prefix.count, &b)
            out.append(Pass(hidden: h, tokens: tail.count, calls: secs.count, callSeconds: secs, resetSeconds: restoreS,
                            seconds: d1Seconds(since: t1)))
        }
        return (pre, out)
    }

    /// The prepared state: `prefix` (a multiple of S; empty = nothing to keep) once from zero states -> its pass and a
    /// copy of the three states after it, newly allocated (the caller keeps it past this call).
    public func prepare(prefix: [Int]) async throws -> (pass: Pass, states: [String: NDArray]) {
        guard prefix.count % chunk == 0 else {
            throw D1Error.contract("a prepared prefix of \(prefix.count) ids is not a multiple of \(chunk)")
        }
        guard var b = buffers else { throw D1Error.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        zeroStates(&b)
        let reset = d1Seconds(since: t0)
        let (hp, sp) = prefix.isEmpty ? ([Float16](), [Double]()) : try await pieces(prefix, from: 0, &b)
        var states = Dictionary(uniqueKeysWithValues: Self.stateNames.map { ($0, NDArray(descriptor: stateDescriptors[$0]!)) })
        ND.copy(b.keyCache, into: &states["keyCache"]!)
        ND.copy(b.valueCache, into: &states["valueCache"]!)
        ND.copy(b.convState, into: &states["convState"]!)
        return (Pass(hidden: hp, tokens: prefix.count, calls: sp.count, callSeconds: sp, resetSeconds: reset,
                     seconds: d1Seconds(since: t0)), states)
    }

    /// Each of `tails` (a row after a prepared prefix of k ids) from a copy of the prepared `states`, positions
    /// continuing at k: the shared prefix's second half on states kept from an earlier `prepare`. A nil tail runs its
    /// `whole` row from zero states instead (a row that does not start with the prepared ids).
    public func runPrepared(states: [String: NDArray], from k: Int, tails: [[Int]?], whole: [[Int]]) async throws -> [Pass] {
        guard Set(states.keys) == Set(Self.stateNames) else { throw D1Error.contract("prepared states \(states.keys.sorted())") }
        guard var b = buffers else { throw D1Error.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        var out: [Pass] = []
        for (i, tail) in tails.enumerated() {
            let t1 = ContinuousClock.now
            if tail != nil {
                restore(states, into: &b)
            } else {
                zeroStates(&b)
            }
            let restoreS = d1Seconds(since: t1)
            let ids = tail ?? whole[i]
            let (h, secs) = try await pieces(ids, from: tail == nil ? 0 : k, &b)
            out.append(Pass(hidden: h, tokens: ids.count, calls: secs.count, callSeconds: secs, resetSeconds: restoreS,
                            seconds: d1Seconds(since: t1)))
        }
        return out
    }

    /// One call of pad ids from zero states, the hidden read back and dropped: the runtime's first call in a process
    /// (its one-time cost) done before any request. -> seconds.
    public func warmUp() async throws -> Double {
        guard var b = buffers else { throw D1Error.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        zeroStates(&b)
        let t = ContinuousClock.now
        _ = try await call([Int32](repeating: padID, count: chunk), from: 0, &b)
        return d1Seconds(since: t)
    }

    private func zeroStates(_ b: inout Buffers) {
        ND.zero(&b.keyCache)
        ND.zero(&b.valueCache)
        ND.zero(&b.convState)
    }

    private func restore(_ snap: [String: NDArray], into b: inout Buffers) {
        ND.copy(snap["keyCache"]!, into: &b.keyCache)
        ND.copy(snap["valueCache"]!, into: &b.valueCache)
        ND.copy(snap["convState"]!, into: &b.convState)
    }

    /// ids from position p0 on, S at a time (the last call padded) -> hidden fp16 rows and each call's seconds.
    private func pieces(_ ids: [Int], from p0: Int, _ b: inout Buffers) async throws -> ([Float16], [Double]) {
        let n = (ids.count + chunk - 1) / chunk
        let end = p0 + n * chunk
        guard end <= maxContext - 1 else {
            throw D1Error.graphLimit("\(p0 + ids.count) ids (padded \(end)) > the graph's \(maxContext - 1) positions")
        }
        var out = [Float16]()
        out.reserveCapacity(ids.count * hidden)
        var secs: [Double] = []
        for c in 0..<n {
            let t = ContinuousClock.now
            var x = [Int32](repeating: padID, count: chunk)
            let r = min(chunk, ids.count - c * chunk)
            for i in 0..<r { x[i] = Int32(ids[c * chunk + i]) }
            let h = try await call(x, from: p0 + c * chunk, &b)
            out += h[0..<(r * hidden)]
            secs.append(d1Seconds(since: t))
        }
        return (out, secs)
    }

    /// One call: S ids after p earlier ids (position_ids 0 ..< p + S), the states and the image rows in place ->
    /// hidden [S * d] fp16.
    private func call(_ x: [Int32], from p: Int, _ b: inout Buffers) async throws -> [Float16] {
        let end = p + chunk
        let inputs: [String: NDArray] = [
            "input_ids": ND.make(x, idsDescriptor),
            "position_ids": ND.make((0..<end).map { Int32($0) }, positions.resolvingDynamicDimensions([1, end])),
            "image_embeds": b.image,
        ]
        var states = InferenceFunction.MutableViews()
        states.insert(&b.keyCache, for: "keyCache")
        states.insert(&b.valueCache, for: "valueCache")
        states.insert(&b.convState, for: "convState")
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&b.hidden, for: "hidden")
        _ = try await main.run(inputs: inputs, states: consume states, outputViews: consume outputs)
        let rows = ND.read(b.hidden, as: Float16.self)
        guard rows.count == chunk * hidden else {
            throw D1Error.contract("hidden of \(rows.count) values for a call of \(chunk) ids")
        }
        return rows
    }

    /// sha256 of fp16 values as their little-endian bytes (NumPy's `tobytes()` of a float16 array).
    public static func sha256(_ values: [Float16]) -> String {
        values.withUnsafeBytes { SHA256.hash(data: $0) }.map { String(format: "%02x", $0) }.joined()
    }
}

/// A graph function's contract as metadata.json writes it: {name: [shape, dtype]} for inputs, outputs and states
/// (-1 = a dynamic axis).
public struct D1GraphContract: Sendable {
    public let function: String
    public let inputs: [String: D1TensorSpec]
    public let outputs: [String: D1TensorSpec]
    public let states: [String: D1TensorSpec]

    /// The decoder's S (input_ids [1, S]), d (hidden [1, S, d]) and N (image_embeds [N, d]).
    public var chunk: Int { inputs["input_ids"]?.shape.last ?? 0 }
    public var hidden: Int { outputs["hidden"]?.shape.last ?? 0 }
    public var imageRows: Int { inputs["image_embeds"]?.shape.first ?? 0 }

    /// `j` = {"function"?, "inputs": {...}, "outputs": {...}, "states"?: {...}}.
    public init(_ j: JSONValue, what: String) throws {
        func table(_ v: JSONValue?) throws -> [String: D1TensorSpec] {
            var out: [String: D1TensorSpec] = [:]
            for m in v?.members ?? [] {
                guard let pair = m.value.array, pair.count == 2, let shape = pair[0].array?.compactMap(\.intValue),
                      let dt = pair[1].string, let spec = D1TensorSpec(shape: shape, dtype: dt)
                else { throw D1Error.bundle("\(what): \(m.key) is not [shape, dtype]") }
                out[m.key] = spec
            }
            return out
        }
        function = j["function"]?.string ?? "main"
        inputs = try table(j["inputs"])
        outputs = try table(j["outputs"])
        states = try table(j["states"])
        guard !inputs.isEmpty, !outputs.isEmpty else { throw D1Error.bundle("\(what): no inputs / outputs") }
    }
}

/// A tensor's shape (-1 = dynamic) and scalar type, as a contract states it and as a loaded function's descriptor has it.
public struct D1TensorSpec: Equatable, Sendable, CustomStringConvertible {
    public let shape: [Int]
    public let dtype: String

    public init?(shape: [Int], dtype: String) {
        guard ["int32", "float16", "float32"].contains(dtype) else { return nil }
        self.shape = shape
        self.dtype = dtype
    }

    init?(_ value: InferenceValue.Descriptor?) {
        guard let d = ND.descriptor(value) else { return nil }
        let dt: String
        switch d.scalarType {
        case .int32: dt = "int32"
        case .float16: dt = "float16"
        case .float32: dt = "float32"
        default: dt = "\(d.scalarType)"
        }
        shape = d.shape
        dtype = dt
    }

    public var description: String { "\(shape) \(dtype)" }
}

enum D1Contract {
    /// The loaded function against the contract: the same input / output / state names, each with its shape and type.
    static func check(_ d: InferenceFunctionDescriptor, what: String, inputs: [String: D1TensorSpec],
                      outputs: [String: D1TensorSpec], states: [String: D1TensorSpec]) throws
    {
        var bad: [String] = []
        for (part, names, want, get) in [
            ("inputs", d.inputNames, inputs, d.inputDescriptor(of:)),
            ("outputs", d.outputNames, outputs, d.outputDescriptor(of:)),
            ("states", d.stateNames, states, d.stateDescriptor(of:)),
        ] as [(String, [String], [String: D1TensorSpec], (String) -> InferenceValue.Descriptor?)] {
            if Set(names) != Set(want.keys) {
                bad.append("\(part) \(names.sorted()) != \(want.keys.sorted())")
                continue
            }
            for (n, w) in want.sorted(by: { $0.key < $1.key }) where D1TensorSpec(get(n)) != w {
                bad.append("\(part) \(n): \(D1TensorSpec(get(n))?.description ?? "not an ndarray") != \(w)")
            }
        }
        if !bad.isEmpty { throw D1Error.contract("\(what): \(bad.joined(separator: "; "))") }
    }

    static func describe(_ d: InferenceFunctionDescriptor) -> JSONValue {
        func table(_ names: [String], _ get: (String) -> InferenceValue.Descriptor?) -> JSONValue {
            .object(names.map { n in
                if let s = D1TensorSpec(get(n)) {
                    return JSONMember(n, .array([.array(s.shape.map { .int($0) }), .string(s.dtype)]))
                }
                return JSONMember(n, .string("non-ndarray"))
            })
        }
        return .object([JSONMember("inputs", table(d.inputNames, d.inputDescriptor(of:))),
                        JSONMember("outputs", table(d.outputNames, d.outputDescriptor(of:))),
                        JSONMember("states", table(d.stateNames, d.stateDescriptor(of:)))])
    }
}

/// `SpecializationOptions` in words (the trace's record of how an asset was loaded).
public func d1Describe(_ o: SpecializationOptions) -> String {
    if o == .default { return "SpecializationOptions.default" }
    let pref = o.preferredComputeUnitKind.map { "\($0)" } ?? "none"
    return "preferred \(pref), allowed \(o.allowedComputeUnitKinds.map { "\($0)" }.sorted()), "
        + "expectFrequentReshapes \(o.expectFrequentReshapes)"
}

func d1Seconds(since t: ContinuousClock.Instant) -> Double {
    let d = ContinuousClock.now - t
    return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
}

// MARK: - NDArray helpers (apps/Kev/Sources/Kev/Support.swift `ND`, itself apps/ClefFlash's; zoo d1-3b 8e82d36)

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
            ptr.update(repeating: zero, count: extent(shape, strides))
        }
    }

    /// The elements an array spans from its first scalar to its last, padding included.
    static func extent(_ shape: [Int], _ strides: Span<Int>) -> Int {
        var e = 1
        for d in 0..<shape.count where shape[d] > 0 { e += (shape[d] - 1) * strides[d] }
        return e
    }

    /// Copies `source` into `dest`, scalar for scalar and padding included: two fp16 arrays of one descriptor (the
    /// state snapshot of the shared prefix).
    static func copy(_ source: NDArray, into dest: inout NDArray) {
        precondition(source.shape == dest.shape && source.scalarType == dest.scalarType && source.strides == dest.strides,
                     "ND.copy: arrays of different descriptors")
        let shape = source.shape
        source.view(as: Float16.self).withUnsafePointer { src, _, strides in
            let n = extent(shape, strides)
            dest.mutableView(as: Float16.self).withUnsafeMutablePointer { dst, _, _ in
                dst.update(from: src, count: n)
            }
        }
    }
}
