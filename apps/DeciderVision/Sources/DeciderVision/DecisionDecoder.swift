// DecisionDecoder — the ids-input decider-2b-vision decoder on the low-level runtime (AIModel + loadFunction +
// MutableViews states; the pipelined engine exposes no logits). Two functions share four states:
//
//   main     input_ids [1, 1]   prefill  input_ids [1, C] (C = 16, the bundle's language.prefill_chunk)
//   both     position_ids [1, t + S] (0 ... t + S - 1), image_embeds [256, 2048] f16, image_rc [256, 2] i32,
//            rope_shift_start [1] i32, rope_shift_amount [1] i32 -> logits [1, 1, V] f16 (the last position)
//   states   keyCache / valueCache [6, 1, 2, -1, 256] f16 (the dynamic length resolved to max_context_length),
//            convState [18, 1, 6144, 3] f16, recState [18, 1, 16, 128, 128] f16 — zero at the start of every row.
//
// Read order (the bundle's `decision.readout`, the supervisor's check_chunk.py): cursor = 0; for each slot s
// ascending, while s - cursor + 1 >= C run "prefill" on ids[cursor ..< cursor + C] (its logits are the slot's when
// cursor + C - 1 == s) and advance C; then "main" one token at a time up to and including s (the logits at
// cursor == s are the slot's). The last slot is the row's last token.

import CoreAI
import Foundation

@available(macOS 27, iOS 27, *)
public final class DecisionDecoder: @unchecked Sendable {
    public static let imageRows = 256
    public static let hidden = 2048

    /// The logits of one slot and the call that produced them.
    public struct Slot: Sendable {
        public let position: Int
        /// Full-vocabulary fp16 logits at the slot.
        public let logits: [Float16]
        /// "prefill" or "main".
        public let readFrom: String
    }

    /// One row's decode: the slots, and each call's kind and wall time.
    public struct Pass: Sendable {
        public let slots: [Slot]
        /// true = prefill, false = main, per call in order
        public let callIsPrefill: [Bool]
        public let callSeconds: [Double]
        /// zeroing the four states before the row
        public let resetSeconds: Double
        public let seconds: Double
    }

    public let url: URL
    public let vocab: Int
    public let maxContext: Int
    /// nil when the asset has no "prefill" function (S = 1 order only).
    public let chunk: Int?
    public let functionNames: [String]
    public let loadSeconds: (model: Double, main: Double, prefill: Double?)
    public let descriptors: [String: Any]

    private let main: InferenceFunction
    private let prefill: InferenceFunction?
    private let idsMain: NDArrayDescriptor
    private let idsPrefill: NDArrayDescriptor?
    private let positions: NDArrayDescriptor
    private let embedsDescriptor: NDArrayDescriptor
    private let rcDescriptor: NDArrayDescriptor
    private let shiftStartDescriptor: NDArrayDescriptor
    private let shiftAmountDescriptor: NDArrayDescriptor
    // The four states and the logits buffer, allocated once. A row moves them into locals for its whole pass:
    // MutableViews borrows what it holds up to `run`, which a class property's access scope does not cover.
    private var buffers: Buffers?

    struct Buffers {
        var keyCache: NDArray
        var valueCache: NDArray
        var convState: NDArray
        var recState: NDArray
        var logits: NDArray
    }

    static let stateNames = ["keyCache", "valueCache", "convState", "recState"]
    static let inputNames: Set<String> = ["input_ids", "position_ids", "image_embeds", "image_rc", "rope_shift_start",
                                          "rope_shift_amount"]

    /// Loads the decoder asset (`.aimodel` or `.aimodelc`) and checks both functions against the contract.
    public init(contentsOf url: URL, vocab: Int, maxContext: Int, prefillChunk: Int?, options: SpecializationOptions)
        async throws
    {
        let t0 = ContinuousClock.now
        let model = try await AIModel(contentsOf: url, options: options)
        let tModel = secondsSince(t0)
        functionNames = model.functionNames
        let t1 = ContinuousClock.now
        guard let md = model.functionDescriptor(for: "main"), let mainFn = try model.loadFunction(named: "main") else {
            throw DeciderVisionError.contract("decoder \(url.lastPathComponent): no \"main\" (functions \(model.functionNames))")
        }
        let tMain = secondsSince(t1)
        var pd: InferenceFunctionDescriptor? = nil
        var prefillFn: InferenceFunction? = nil
        var tPrefill: Double? = nil
        if let c = prefillChunk, c > 1 {
            let t2 = ContinuousClock.now
            guard let d = model.functionDescriptor(for: "prefill"), let f = try model.loadFunction(named: "prefill") else {
                throw DeciderVisionError.contract(
                    "decoder \(url.lastPathComponent): prefill_chunk \(c) but no \"prefill\" (functions \(model.functionNames))")
            }
            tPrefill = secondsSince(t2)
            pd = d
            prefillFn = f
        }
        try Self.check(md, name: "main", queryLength: 1, vocab: vocab)
        if let pd, let c = prefillChunk {
            try Self.check(pd, name: "prefill", queryLength: c, vocab: vocab)
            for s in Self.stateNames where TensorSpec.of(pd.stateDescriptor(of: s)) != TensorSpec.of(md.stateDescriptor(of: s)) {
                throw DeciderVisionError.contract("prefill state \(s) differs from main's")
            }
        }
        func nd(_ d: InferenceFunctionDescriptor, input n: String) -> NDArrayDescriptor { ND.descriptor(d.inputDescriptor(of: n))! }
        func state(_ n: String) -> NDArray {
            let d = ND.descriptor(md.stateDescriptor(of: n))!
            var a = NDArray(descriptor: d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? maxContext : $0 }))
            ND.zero(&a)
            return a
        }
        self.url = url
        self.vocab = vocab
        self.maxContext = maxContext
        self.chunk = prefillFn == nil ? nil : prefillChunk
        self.loadSeconds = (tModel, tMain, tPrefill)
        var desc: [String: Any] = ["main": describe(md)]
        if let pd { desc["prefill"] = describe(pd) }
        self.descriptors = desc
        self.main = mainFn
        self.prefill = prefillFn
        self.idsMain = nd(md, input: "input_ids")
        self.idsPrefill = pd.map { nd($0, input: "input_ids") }
        self.positions = nd(md, input: "position_ids")
        self.embedsDescriptor = nd(md, input: "image_embeds")
        self.rcDescriptor = nd(md, input: "image_rc")
        self.shiftStartDescriptor = nd(md, input: "rope_shift_start")
        self.shiftAmountDescriptor = nd(md, input: "rope_shift_amount")
        let ld = ND.descriptor(md.outputDescriptor(of: "logits"))!
        buffers = Buffers(keyCache: state("keyCache"), valueCache: state("valueCache"), convState: state("convState"),
                          recState: state("recState"), logits: NDArray(descriptor: ld.resolvingDynamicDimensions(ld.shape)))
    }

    private static func check(_ d: InferenceFunctionDescriptor, name: String, queryLength: Int, vocab: Int) throws {
        let want: [String: TensorSpec] = [
            "input_ids": TensorSpec(shape: [1, queryLength], type: .int32),
            "position_ids": TensorSpec(shape: [1, -1], type: .int32),
            "image_embeds": TensorSpec(shape: [imageRows, hidden], type: .float16),
            "image_rc": TensorSpec(shape: [imageRows, 2], type: .int32),
            "rope_shift_start": TensorSpec(shape: [1], type: .int32),
            "rope_shift_amount": TensorSpec(shape: [1], type: .int32),
        ]
        var bad: [String] = []
        if Set(d.inputNames) != inputNames { bad.append("inputs \(d.inputNames)") }
        for (n, w) in want where TensorSpec.of(d.inputDescriptor(of: n)) != w {
            bad.append("\(n) \(TensorSpec.of(d.inputDescriptor(of: n))?.description ?? "missing") != \(w)")
        }
        if d.outputNames != ["logits"] || TensorSpec.of(d.outputDescriptor(of: "logits")) != TensorSpec(shape: [1, 1, vocab], type: .float16) {
            bad.append("outputs \(d.outputNames) \(TensorSpec.of(d.outputDescriptor(of: "logits"))?.description ?? "")")
        }
        if Set(d.stateNames) != Set(stateNames) { bad.append("states \(d.stateNames)") }
        for s in stateNames where TensorSpec.of(d.stateDescriptor(of: s))?.type != .float16 { bad.append("state \(s) not f16") }
        if !bad.isEmpty {
            throw DeciderVisionError.contract("decoder function \(name): \(bad.joined(separator: "; "))")
        }
    }

    /// The static inputs of one row: tower rows 0 ..< N as fp16 (the rest zero), image_rc[k] = (k / G, k % G).
    public struct StaticInputs: @unchecked Sendable {
        let embeds: NDArray
        let rc: NDArray
        let start: NDArray
        let amount: NDArray
    }

    /// `towerEmbeds` = the tower output [G^2 * 2048] float32 (nil for a text-only row).
    public func staticInputs(towerEmbeds: [Float]?, grid: Int?, start: Int32, amount: Int32) throws -> StaticInputs {
        var emb = [Float16](repeating: 0, count: Self.imageRows * Self.hidden)
        var rc = [Int32](repeating: 0, count: Self.imageRows * 2)
        if let towerEmbeds, let g = grid {
            let n = g * g
            guard n <= Self.imageRows, towerEmbeds.count == n * Self.hidden else {
                throw DeciderVisionError.contract("image_embeds: \(towerEmbeds.count) values for \(n) rows")
            }
            for i in 0..<(n * Self.hidden) { emb[i] = Float16(towerEmbeds[i]) }
            for k in 0..<n {
                rc[2 * k] = Int32(k / g)
                rc[2 * k + 1] = Int32(k % g)
            }
        }
        return StaticInputs(embeds: ND.make(emb, embedsDescriptor), rc: ND.make(rc, rcDescriptor),
                            start: ND.make([start], shiftStartDescriptor), amount: ND.make([amount], shiftAmountDescriptor))
    }

    /// One call of the row's schedule: prefill on ids[start ..< start + count] (count = C) or main on ids[start].
    struct Step {
        let prefill: Bool
        let start: Int
        let count: Int
        /// the slot this call's logits answer, if any
        let reads: Int?
    }

    /// The chunk order (the bundle's `decision.readout`); S = 1 throughout when `chunk` is nil.
    static func schedule(slots: [Int], chunk: Int?) -> [Step] {
        var steps: [Step] = []
        var cursor = 0
        for slot in slots {
            if let c = chunk {
                while slot - cursor + 1 >= c {
                    steps.append(Step(prefill: true, start: cursor, count: c, reads: cursor + c - 1 == slot ? slot : nil))
                    cursor += c
                }
            }
            while cursor <= slot {
                steps.append(Step(prefill: false, start: cursor, count: 1, reads: cursor == slot ? slot : nil))
                cursor += 1
            }
        }
        return steps
    }

    /// Zeroes the states and reads one row's slots in the chunk order (S = 1 only when `useChunks` is false or the
    /// asset has no prefill). One row at a time per decoder: the states are this object's.
    public func run(ids: [Int], slots: [Int], inputs s: StaticInputs, useChunks: Bool = true) async throws -> Pass {
        guard !slots.isEmpty, slots == slots.sorted(), let last = slots.last, last == ids.count - 1 else {
            throw DeciderVisionError.prompt("slots \(slots) do not end the row (\(ids.count) ids)")
        }
        guard ids.count <= maxContext else {
            throw DeciderVisionError.prompt("\(ids.count) ids > max_context_length \(maxContext)")
        }
        guard var b = buffers else { throw DeciderVisionError.contract("decoder busy: one row at a time") }
        buffers = nil
        defer { buffers = b }
        let t0 = ContinuousClock.now
        ND.zero(&b.keyCache)
        ND.zero(&b.valueCache)
        ND.zero(&b.convState)
        ND.zero(&b.recState)
        let reset = secondsSince(t0)
        let steps = Self.schedule(slots: slots, chunk: useChunks && prefill != nil ? chunk : nil)
        var got: [Slot] = []
        var kinds: [Bool] = [], secs: [Double] = []
        for step in steps {
            let t = ContinuousClock.now
            let fn = step.prefill ? prefill! : main
            let end = step.start + step.count
            let inputs: [String: NDArray] = [
                "input_ids": ND.make(ids[step.start..<end].map { Int32($0) }, step.prefill ? idsPrefill! : idsMain),
                "position_ids": ND.make((0..<end).map { Int32($0) }, positions.resolvingDynamicDimensions([1, end])),
                "image_embeds": s.embeds, "image_rc": s.rc, "rope_shift_start": s.start, "rope_shift_amount": s.amount,
            ]
            var states = InferenceFunction.MutableViews()
            states.insert(&b.keyCache, for: "keyCache")
            states.insert(&b.valueCache, for: "valueCache")
            states.insert(&b.convState, for: "convState")
            states.insert(&b.recState, for: "recState")
            var outputs = InferenceFunction.MutableViews()
            outputs.insert(&b.logits, for: "logits")
            _ = try await fn.run(inputs: inputs, states: consume states, outputViews: consume outputs)
            if let slot = step.reads {
                got.append(Slot(position: slot, logits: ND.read(b.logits, as: Float16.self),
                                readFrom: step.prefill ? "prefill" : "main"))
            }
            kinds.append(step.prefill)
            secs.append(secondsSince(t))
        }
        guard steps.last.map({ $0.start + $0.count }) == ids.count, got.count == slots.count else {
            throw DeciderVisionError.prompt("read \(got.count) slots; the schedule does not end at the row's last id")
        }
        return Pass(slots: got, callIsPrefill: kinds, callSeconds: secs, resetSeconds: reset, seconds: secondsSince(t0))
    }
}
