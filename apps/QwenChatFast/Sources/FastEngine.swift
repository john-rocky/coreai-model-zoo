// Qwen3.5-0.8B HOST-CACHE + CHUNKED (ANE fast path) on-device generation engine for iOS (Core AI).
//
// Mirrors the gemma4 ANE GO solution: the Core AI ANE compiler rejects the in-graph indexed KV write
// (`mutable_slice_update`) and OOMs on the 24-layer monolith. So the model is exported as N host-cache
// chunks (ondevice/export_qwen3_5_hostcache.py): each chunk holds ~6 layers, the KV caches + the
// GatedDeltaNet conv/rec states are plain I/O — the graph reads the host-written past, `cat`s the
// current token's K/V, runs masked SDPA, and RETURNS the current K/V columns + new conv/rec; THIS
// engine writes the K/V columns into the per-chunk caches at `position` between steps (a cheap numpy-
// style assignment, no in-graph op) and threads `hidden` chunk -> chunk. No Core AI state, no indexed
// write, each graph small -> compiles for + runs on the ANE.
//
//   chunk 0      : input_ids[1,1]      -> embed -> 6 layers -> hidden[1,1,H]
//   chunk 1..N-2 : hidden[1,1,H]       -> 6 layers          -> hidden[1,1,H]
//   chunk N-1    : hidden[1,1,H]       -> 6 layers -> norm + lm_head -> logits[1,1,V]
// Each chunk also: in  past_k/past_v[nf,1,HKV,B,D] + conv_state[nl,1,Cc,K-1] + rec_state[nl,1,nV,Dk,Dv]
//                  out k_cur/v_cur[nf,1,HKV,1,D]   + conv_cur[nl,...]        + rec_cur[nl,...]
//
// SPEED (issue B fix): the old path round-tripped EVERY state through `[Float]` per step —
// `flattenAsFloat` (fp16->fp32) of rec_cur (1.3M elems/chunk) + `.map { Float16($0) }` (fp32->fp16) on
// refill — which dominated at ~1 tok/s. This version stays in **Float16**:
//   * KV + hidden  : host round-trip kept (KV must be scattered column-wise; hidden is tiny 1024) but
//                    via `readF16` (stride-safe fp16, no fp32) — the PROVEN structure, just fp16.
//   * conv/rec     : the 1.3M-elem bottleneck. DEFAULT threads the chunk's `*_cur` output NDArray
//                    straight back as next step's `*_state` input — zero copy, zero conversion
//                    (the prescribed fix). Set env `QWEN_RT=1` to fall back to the proven
//                    fp16 host round-trip (still no fp32) if direct NDArray threading misbehaves on
//                    device — so one device window validates both without a rebuild.
// This is numerically the exact recurrence the GPU `--verify` runs (conv[ci]=conv_cur, rec[ci]=rec_cur,
// kcache[...,p,:]=k_cur), just conversion-free.

import CoreAI
import CoreAIShared
import Foundation
import Tokenizers
import os

private let log = Logger(subsystem: "com.coreai.qwenchat.fast", category: "engine")

@MainActor
final class FastEngine: ObservableObject {
    enum Status: Equatable {
        case idle, loading, ready, generating, error(String)
        var label: String {
            switch self {
            case .idle: return "idle"
            case .loading: return "loading model…"
            case .ready: return "ready"
            case .generating: return "generating…"
            case .error(let m): return "error: \(m)"
            }
        }
    }

    @Published var status: Status = .idle
    @Published var output: String = ""
    @Published var stats: String = ""

    // qwen3.5-0.8B static dims (== export_qwen3_5_hostcache.py).
    private let vocab = 248320
    // B: fixed cache capacity (host-managed). MUST equal the exported --max-ctx (mask length, KV
    // strides and the prompt+max guard all derive from it). QWEN_CTX selects between exports.
    // Default = 2048, matching the deployed release monolith (ctx-2048, 27.7 tok/s GPU).
    private let maxCtx = Int(ProcessInfo.processInfo.environment["QWEN_CTX"] ?? "2048") ?? 2048
    private let hidden = 1024
    private let hkv = 2, headDim = 256       // KV: [nf,1,hkv,B,headDim]
    private let convDim = 6144, convK1 = 3   // conv: [nl,1,convDim,convK1]
    private let numV = 16, dk = 128, dv = 128 // rec: [nl,1,numV,dk,dv]
    private let maskNeg: Float16 = -1.0e4

    private var fns: [InferenceFunction] = []
    private var descs: [InferenceFunctionDescriptor] = []
    private var nFull: [Int] = []            // full-attn layers per chunk
    private var nLin: [Int] = []             // SSM layers per chunk
    private var tokenizer: Tokenizer?
    private(set) var computeUnit = "ane"
    // QWEN_KIND selects the decode artifact family. Default = "int8v3" (the fused-kernel release
    // config: 42.5-45.4 tok/s, GPU argmax head). QWEN_KIND=fp16 selects the previous fp16
    // monolith (qwen3_5_0_8b_ios_hc0.aimodel, 27.7-28.3 tok/s).
    private let kindSuffix: String = {
        let k = ProcessInfo.processInfo.environment["QWEN_KIND"] ?? "int8v3"
        return (k.isEmpty || k == "fp16") ? "" : "_\(k)"
    }()
    // Kernel exports replace the last chunk's logits with two-level argmax partials
    // (head_pv fp32 / head_pi int32, vocab/8 each); detected from the descriptor at load.
    private var argmaxHead = false
    // QWEN_PREFILL=<q> selects the q>1 chunked-prefill companion graph (147 tok/s prefill):
    // qwen3_5_0_8b_ios_hc_prefill_q<q>_b<maxCtx>(_int8).aimodel. The prompt is consumed in FULL
    // q-token blocks (one weight pass per block instead of per token); the remainder (< q) and
    // all generation stay on the q=1 decode graph. Default 16 (the measured optimum: q=32 has
    // the same per-token block cost but a bigger q=1 remainder); QWEN_PREFILL=0 disables.
    // Monolith-only (requires QWEN_CHUNKS=1). Partial blocks are never fed: the SSM conv/rec
    // states cannot be partially written back, so padding would poison the recurrence.
    // If the artifact is absent the engine degrades gracefully to q=1 prefill.
    private let prefillQ = Int(ProcessInfo.processInfo.environment["QWEN_PREFILL"] ?? "16") ?? 16
    private var prefillFn: InferenceFunction?
    private var prefillDesc: InferenceFunctionDescriptor?
    // conv/rec: false = thread output NDArray directly (fast, default); true = fp16 host round-trip.
    private let convRecRoundTrip = ProcessInfo.processInfo.environment["QWEN_RT"] == "1"
    // Repetition penalty over the full fp32 logits (CoreML-LLM's fix for the ANE fp16 248K-vocab argmax
    // collapse -> greedy "the the the" loop). 1.0 disables. Applied to already-generated token ids.
    private let repPenalty = Float(ProcessInfo.processInfo.environment["QWEN_REP_PENALTY"] ?? "1.3") ?? 1.3

    // MARK: load

    func load(modelDir: URL, tokenizerFolder: URL, cu: String, numChunks: Int) async {
        status = .loading; output = ""; stats = ""; computeUnit = cu
        fns = []; descs = []; nFull = []; nLin = []; prefillFn = nil; prefillDesc = nil
        do {
            let kind: ComputeUnitKind = cu == "cpu" ? .cpu : cu == "gpu" ? .gpu : .neuralEngine
            var opts = SpecializationOptions(preferredComputeUnitKind: kind)
            opts.expectFrequentReshapes = false
            for ci in 0..<numChunks {
                let url = modelDir.appendingPathComponent("qwen3_5_0_8b_ios_hc\(ci)\(kindSuffix).aimodel")
                let model = try await AIModel(contentsOf: url, options: opts)
                guard let d = model.functionDescriptor(for: "main"),
                      let f = try model.loadFunction(named: "main") else {
                    fail("chunk \(ci): no 'main'"); return
                }
                descs.append(d); fns.append(f)
                nFull.append(stateLeadDim(d, "past_k"))
                nLin.append(stateLeadDim(d, "conv_state"))
                print("[QwenChatFast] chunk \(ci) loaded nFull=\(nFull[ci]) nLin=\(nLin[ci])")
            }
            if case .ndArray(_)? = descs[descs.count - 1].outputDescriptor(of: "head_pv") {
                argmaxHead = true  // greedy-only GPU argmax head; logits (and repPenalty) unavailable
            }
            if prefillQ > 0 && numChunks == 1 {
                // Prefer the int8-palettized prefill graph: fp16 (1.5 GB) + the decode monolith
                // do NOT fit in the app memory budget together (std::bad_alloc on device).
                let pbase = "qwen3_5_0_8b_ios_hc_prefill_q\(prefillQ)_b\(maxCtx)"
                let p8 = modelDir.appendingPathComponent("\(pbase)_int8.aimodel")
                let pfp = modelDir.appendingPathComponent("\(pbase).aimodel")
                let purl = FileManager.default.fileExists(atPath: p8.path) ? p8 : pfp
                if FileManager.default.fileExists(atPath: purl.path) {
                    let pmodel = try await AIModel(contentsOf: purl, options: opts)
                    guard let pd = pmodel.functionDescriptor(for: "main"),
                          let pf = try pmodel.loadFunction(named: "main") else {
                        fail("prefill graph: no 'main'"); return
                    }
                    prefillDesc = pd; prefillFn = pf
                    print("[QwenChatFast] prefill graph loaded q=\(prefillQ)")
                } else {
                    print("[QwenChatFast] prefill artifact missing (\(p8.lastPathComponent)) — q=1 prefill")
                }
            } else if prefillQ > 0 {
                print("[QwenChatFast] QWEN_PREFILL needs QWEN_CHUNKS=1 — q=1 prefill")
            }
            tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolder)
            status = .ready
            print("[QwenChatFast] READY \(numChunks) host-cache chunks cu=\(cu) maxCtx=\(maxCtx) kind=\(kindSuffix.isEmpty ? "fp16" : String(kindSuffix.dropFirst())) argmaxHead=\(argmaxHead) prefillQ=\(prefillQ) convRecRoundTrip=\(convRecRoundTrip)")
        } catch {
            fail("\(error)")
        }
    }

    // MARK: generation

    func generate(prompt: String, maxTokens: Int) async {
        guard status == .ready, let tok = tokenizer else { return }
        status = .generating; output = ""; stats = ""
        let promptIds = tok.encode(text: prompt).map { Int32($0) }
        let M = promptIds.count
        if M + maxTokens > maxCtx { fail("prompt+max exceeds \(maxCtx)"); return }
        print("[QwenChatFast] generate \"\(prompt)\" ids=\(M) maxTokens=\(maxTokens)")
        do {
            let N = fns.count

            // Per-chunk element counts (from the resolved input descriptor shapes).
            var kvCount = [Int](), convCount = [Int](), recCount = [Int]()
            for ci in 0..<N {
                let d = descs[ci]
                kvCount.append(shape(d, "past_k", "input").reduce(1, *))
                convCount.append(shape(d, "conv_state", "input").reduce(1, *))
                recCount.append(shape(d, "rec_state", "input").reduce(1, *))
            }

            // KV + hidden: Float16 host caches (proven structure, no fp32). KV scattered column-wise.
            var kc = [[Float16]](), vc = [[Float16]]()
            for ci in 0..<N {
                kc.append([Float16](repeating: 0, count: kvCount[ci]))
                vc.append([Float16](repeating: 0, count: kvCount[ci]))
            }
            var hiddenF = [Float16]()

            // conv/rec: either threaded NDArrays (default) or fp16 host arrays (QWEN_RT=1).
            var convND = [NDArray](), recND = [NDArray]()
            var convF = [[Float16]](), recF = [[Float16]]()
            for ci in 0..<N {
                if convRecRoundTrip {
                    convF.append([Float16](repeating: 0, count: convCount[ci]))
                    recF.append([Float16](repeating: 0, count: recCount[ci]))
                } else {
                    convND.append(zeroInput(descs[ci], "conv_state"))
                    recND.append(zeroInput(descs[ci], "rec_state"))
                }
            }

            var penalized = Set<Int>()   // already-generated token ids, penalized in the fp32 logits
            var runSecAccum = 0.0        // wall-clock spent inside fn.run (Core AI inference) across all chunks/steps

            func decodeStep(_ tokenId: Int32, _ pos: Int) async throws -> Int {
                let mask = causalMask(pos: pos)
                var logits: NDArray? = nil
                var headTok: Int? = nil
                for ci in 0..<N {
                    let d = descs[ci], fn = fns[ci]
                    let isLast = ci == N - 1
                    let inName = ci == 0 ? "input_ids" : "hidden_in"

                    var posA = alloc(d, "position_ids", [1, 1], kind: "input")
                    fillNDArray(&posA, as: Int32.self, with: [Int32(pos)])
                    var maskA = alloc(d, "causal_mask", [1, 1, 1, maxCtx + 1], kind: "input")
                    fillNDArray(&maskA, as: Float16.self, with: mask)
                    var pkA = alloc(d, "past_k", shape(d, "past_k", "input"), kind: "input")
                    fillNDArray(&pkA, as: Float16.self, with: kc[ci])
                    var pvA = alloc(d, "past_v", shape(d, "past_v", "input"), kind: "input")
                    fillNDArray(&pvA, as: Float16.self, with: vc[ci])
                    var inputs: [String: NDArray] = [
                        "position_ids": posA, "causal_mask": maskA, "past_k": pkA, "past_v": pvA,
                    ]
                    if convRecRoundTrip {
                        var cvA = alloc(d, "conv_state", shape(d, "conv_state", "input"), kind: "input")
                        fillNDArray(&cvA, as: Float16.self, with: convF[ci])
                        var rcA = alloc(d, "rec_state", shape(d, "rec_state", "input"), kind: "input")
                        fillNDArray(&rcA, as: Float16.self, with: recF[ci])
                        inputs["conv_state"] = cvA; inputs["rec_state"] = rcA
                    } else {
                        inputs["conv_state"] = convND[ci]; inputs["rec_state"] = recND[ci]
                    }
                    if ci == 0 {
                        var idA = alloc(d, inName, [1, 1], kind: "input")
                        fillNDArray(&idA, as: Int32.self, with: [tokenId])
                        inputs[inName] = idA
                    } else {
                        var hA = alloc(d, inName, [1, 1, hidden], kind: "input")
                        fillNDArray(&hA, as: Float16.self, with: hiddenF)
                        inputs[inName] = hA
                    }

                    let runT0 = Date()
                    var out = try await fn.run(inputs: inputs)
                    runSecAccum += Date().timeIntervalSince(runT0)
                    inputs.removeAll(keepingCapacity: false)

                    if isLast {
                        if argmaxHead {
                            if let pvA = out.remove("head_pv")?.ndArray,
                               let piA = out.remove("head_pi")?.ndArray {
                                headTok = reduceArgmaxPartials(pvA, piA)
                            }
                        } else {
                            logits = out.remove("logits")?.ndArray
                        }
                    } else if let h = out.remove("hidden_out")?.ndArray { hiddenF = readF16(h) }

                    // HOST write-back: current K/V columns -> this chunk's caches at `pos` (fp16).
                    if nFull[ci] > 0, let kCur = out.remove("k_cur")?.ndArray,
                       let vCur = out.remove("v_cur")?.ndArray {
                        writeColumn(&kc[ci], readF16(kCur), nf: nFull[ci], pos: pos)
                        writeColumn(&vc[ci], readF16(vCur), nf: nFull[ci], pos: pos)
                    }
                    // conv/rec recurrence: thread output NDArray (default) or fp16 round-trip.
                    if nLin[ci] > 0 {
                        if convRecRoundTrip {
                            if let cc = out.remove("conv_cur")?.ndArray { convF[ci] = readF16(cc) }
                            if let rc = out.remove("rec_cur")?.ndArray { recF[ci] = readF16(rc) }
                        } else {
                            if let cc = out.remove("conv_cur")?.ndArray { convND[ci] = cc }
                            if let rc = out.remove("rec_cur")?.ndArray { recND[ci] = rc }
                        }
                    }
                }
                if let t = headTok { return t }      // GPU argmax head (greedy; no logits returned)
                return selectToken(logits!, penalized)
            }

            let t0 = Date(); var last = 0
            var p = 0
            if let pfn = prefillFn, let pd = prefillDesc, M - 1 >= prefillQ {
                // q-block prefill: FULL blocks only over prompt[0..M-2] (the last prompt token and
                // the remainder go through decodeStep). conv/rec ride a host fp16 round-trip at
                // block rate (q x cheaper than per-token), then seed the decode chain once.
                var convH = [Float16](repeating: 0, count: convCount[0])
                var recH = [Float16](repeating: 0, count: recCount[0])
                while p + prefillQ <= M - 1 {
                    var idsA = alloc(pd, "input_ids", [1, prefillQ], kind: "input")
                    fillNDArray(&idsA, as: Int32.self, with: Array(promptIds[p..<(p + prefillQ)]))
                    var posA = alloc(pd, "position_ids", [1, prefillQ], kind: "input")
                    fillNDArray(&posA, as: Int32.self, with: (p..<(p + prefillQ)).map { Int32($0) })
                    var maskA = alloc(pd, "causal_mask", [1, 1, prefillQ, maxCtx + prefillQ], kind: "input")
                    fillNDArray(&maskA, as: Float16.self, with: prefillMask(pos: p))
                    var pkA = alloc(pd, "past_k", shape(pd, "past_k", "input"), kind: "input")
                    fillNDArray(&pkA, as: Float16.self, with: kc[0])
                    var pvA = alloc(pd, "past_v", shape(pd, "past_v", "input"), kind: "input")
                    fillNDArray(&pvA, as: Float16.self, with: vc[0])
                    var cvA = alloc(pd, "conv_state", shape(pd, "conv_state", "input"), kind: "input")
                    fillNDArray(&cvA, as: Float16.self, with: convH)
                    var rcA = alloc(pd, "rec_state", shape(pd, "rec_state", "input"), kind: "input")
                    fillNDArray(&rcA, as: Float16.self, with: recH)
                    let runT0 = Date()
                    var out = try await pfn.run(inputs: [
                        "input_ids": idsA, "position_ids": posA, "causal_mask": maskA,
                        "past_k": pkA, "past_v": pvA, "conv_state": cvA, "rec_state": rcA,
                    ])
                    runSecAccum += Date().timeIntervalSince(runT0)
                    if let kCur = out.remove("k_cur")?.ndArray,
                       let vCur = out.remove("v_cur")?.ndArray {
                        writeColumns(&kc[0], readF16(kCur), nf: nFull[0], pos: p, cols: prefillQ)
                        writeColumns(&vc[0], readF16(vCur), nf: nFull[0], pos: p, cols: prefillQ)
                    }
                    if let cc = out.remove("conv_cur")?.ndArray { convH = readF16(cc) }
                    if let rc = out.remove("rec_cur")?.ndArray { recH = readF16(rc) }
                    p += prefillQ
                }
                if p > 0 {  // hand the block states to the decode chain
                    if convRecRoundTrip {
                        convF[0] = convH; recF[0] = recH
                    } else {
                        var cvA = alloc(descs[0], "conv_state", shape(descs[0], "conv_state", "input"), kind: "input")
                        fillNDArray(&cvA, as: Float16.self, with: convH)
                        convND[0] = cvA
                        var rcA = alloc(descs[0], "rec_state", shape(descs[0], "rec_state", "input"), kind: "input")
                        fillNDArray(&rcA, as: Float16.self, with: recH)
                        recND[0] = rcA
                    }
                }
            }
            for i in p..<M { last = try await decodeStep(promptIds[i], i) }
            let prefillSec = Date().timeIntervalSince(t0)

            penalized.insert(last)
            var gen = [last]; output = tok.decode(tokens: gen)
            let t1 = Date(); let runBeforeDecode = runSecAccum
            for i in 0..<(maxTokens - 1) {
                last = try await decodeStep(Int32(last), M + i)
                penalized.insert(last)
                gen.append(last); output = tok.decode(tokens: gen)
            }
            let decodeSec = Date().timeIntervalSince(t1)
            let decodeRunSec = runSecAccum - runBeforeDecode          // time inside fn.run (Core AI inference)
            let decodeHostSec = max(0, decodeSec - decodeRunSec)      // alloc/fill/readF16/writeColumn/argmax
            let dTps = decodeSec > 0 ? Double(maxTokens - 1) / decodeSec : 0
            let runPct = decodeSec > 0 ? 100 * decodeRunSec / decodeSec : 0
            stats = String(format: "%@ host-cache%@ · prefill %d/%.2fs=%.1f t/s · decode %d/%.2fs=%.1f t/s · run %.2fs(%.0f%%) host %.2fs",
                           computeUnit.uppercased(), convRecRoundTrip ? " RT" : "",
                           M, prefillSec, prefillSec > 0 ? Double(M)/prefillSec : 0,
                           maxTokens - 1, decodeSec, dTps, decodeRunSec, runPct, decodeHostSec)
            status = .ready
            print("[QwenChatFast] DONE output=\"\(output)\"")
            print("[QwenChatFast] STATS \(stats)")
        } catch { fail("\(error)") }
    }

    // MARK: helpers

    private func fail(_ m: String) { status = .error(m); print("[QwenChatFast] ERROR \(m)"); log.error("\(m, privacy: .public)") }

    private func alloc(_ d: InferenceFunctionDescriptor, _ name: String, _ s: [Int], kind: String) -> NDArray {
        let io = kind == "input" ? d.inputDescriptor(of: name) : d.outputDescriptor(of: name)
        guard case .ndArray(let nd)? = io else { fatalError("\(name) not ndarray") }
        return NDArray(descriptor: nd.resolvingDynamicDimensions(s))
    }

    private func shape(_ d: InferenceFunctionDescriptor, _ name: String, _ kind: String) -> [Int] {
        let io = kind == "input" ? d.inputDescriptor(of: name) : d.outputDescriptor(of: name)
        guard case .ndArray(let nd)? = io else { fatalError("\(name)") }
        return nd.shape.map { $0 < 0 ? maxCtx : $0 }
    }

    private func stateLeadDim(_ d: InferenceFunctionDescriptor, _ name: String) -> Int {
        guard case .ndArray(let nd)? = d.inputDescriptor(of: name) else { return 0 }
        return nd.shape.first.map { $0 < 0 ? 1 : $0 } ?? 0
    }

    // Persistent zero-initialised Float16 input NDArray (conv/rec direct-threading start state).
    private func zeroInput(_ d: InferenceFunctionDescriptor, _ name: String) -> NDArray {
        let s = shape(d, name, "input")
        var a = alloc(d, name, s, kind: "input")
        let count = s.reduce(1, *)
        fillNDArray(&a, as: Float16.self, count: count) { _ in Float16(0) }
        return a
    }

    // Read an NDArray to [Float16] (no fp32, vs flattenAsFloat). Uses the public CoreAIShared wrapper
    // (same proven path as fillNDArray/flattenAsFloat) so the app needs no direct view(as:) call.
    // Assumes row-major contiguous output — "the common case for Core AI outputs" and what the old
    // code's flattenAsFloat fast path already relied on; these host-cache outputs (stack/standard ops)
    // are contiguous.
    private func readF16(_ array: NDArray) -> [Float16] {
        let total = array.shape.reduce(1, *)
        return readNDArray(array, as: Float16.self, count: total)
    }

    // Write k_cur [nf,1,hkv,q,D] into cache [nf,1,hkv,B,D] at columns pos..pos+cols-1 (prefill blocks).
    private func writeColumns(_ cache: inout [Float16], _ cur: [Float16], nf: Int, pos: Int, cols: Int) {
        guard nf > 0 else { return }
        let B = maxCtx, D = headDim, HKV = hkv
        for f in 0..<nf {
            for h in 0..<HKV {
                for c in 0..<cols {
                    let dst = ((f * HKV + h) * B + pos + c) * D
                    let src = ((f * HKV + h) * cols + c) * D
                    for e in 0..<D { cache[dst + e] = cur[src + e] }
                }
            }
        }
    }

    // Additive block mask [1,1,q,B+q] for the prefill graph: past cols < pos valid for ALL rows;
    // current-block col t (at index B+t) valid for rows i >= t; everything else masked.
    private func prefillMask(pos: Int) -> [Float16] {
        let q = prefillQ, W = maxCtx + q
        var m = [Float16](repeating: maskNeg, count: q * W)
        for i in 0..<q {
            let row = i * W
            for j in 0..<pos { m[row + j] = 0 }
            for t in 0...i { m[row + maxCtx + t] = 0 }
        }
        return m
    }

    // Write k_cur [nf,1,hkv,1,D] into cache [nf,1,hkv,B,D] at sequence column `pos` (contiguous fp16).
    private func writeColumn(_ cache: inout [Float16], _ cur: [Float16], nf: Int, pos: Int) {
        guard nf > 0 else { return }
        let B = maxCtx, D = headDim, HKV = hkv
        for f in 0..<nf {
            for h in 0..<HKV {
                let dst = ((f * HKV + h) * B + pos) * D
                let src = (f * HKV + h) * D
                for e in 0..<D { cache[dst + e] = cur[src + e] }
            }
        }
    }

    // additive mask [1,1,1,B+1]: 0 for past cols 0..pos-1 + current col (index B), else maskNeg.
    private func causalMask(pos: Int) -> [Float16] {
        var m = [Float16](repeating: maskNeg, count: maxCtx + 1)
        for j in 0..<pos { m[j] = 0 }
        m[maxCtx] = 0
        return m
    }

    // Reduce the head kernel's per-threadgroup (max value fp32, global index int32) partials to the
    // greedy token: token = pi[argmax(pv)]. Strict '>' keeps the lowest index on ties == torch.argmax.
    // pv is ~vocab/8 = 31040 floats — ~1/8 the readback of full logits, and already fp32.
    private func reduceArgmaxPartials(_ pvA: NDArray, _ piA: NDArray) -> Int {
        let pv = flattenAsFloat(pvA)
        let pi = readNDArray(piA, as: Int32.self, count: pv.count)
        var best = 0
        var bv = pv[0]
        for j in 1..<pv.count where pv[j] > bv { bv = pv[j]; best = j }
        return Int(pi[best])
    }

    // Greedy token over logits [1,1,vocab] with a full-vocab fp32 repetition penalty. flattenAsFloat is
    // stride/dtype-safe; the ANE emits fp16 logits whose 248K-vocab argmax collapses to a repeated token
    // ("the the the") — penalizing already-generated ids in fp32 lets a fresh token win (CoreML-LLM's fix).
    private func selectToken(_ nd: NDArray, _ penalized: Set<Int>) -> Int {
        var flat = flattenAsFloat(nd)
        if repPenalty != 1.0 {
            for id in penalized where id >= 0 && id < flat.count {
                flat[id] = flat[id] > 0 ? flat[id] / repPenalty : flat[id] * repPenalty
            }
        }
        var best = 0; var bv = flat[0]
        for j in 1..<flat.count where flat[j] > bv { bv = flat[j]; best = j }
        return best
    }
}
