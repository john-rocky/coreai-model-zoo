// MiniCPMVLBackend — MiniCPM-V-4.6 VLM on the Core AI pipelined engine.
//
// Two downloaded bundles (placed under <Documents>/models by ModelDownloader):
//   * minicpmv46_vlm_decode_int8lin — decoder LanguageBundle (input_ids -> logits +
//     a single static `image_embeds`[64,1024] f16 buffer; in-graph gather
//     embedding = ids < V ? embed_tokens[ids] : image_embeds[ids - V]).
//   * minicpmv46_vision — fixed-grid SigLIP .aimodel, run once per image:
//     pixel_values[1,3,448,448] f16 -> image_features[64,1024].
//
// Host contract (mirrors the gated python pipeline + the iPhone PipelinedBench gate):
//   * preprocess: resize 448x448, normalize x/127.5-1 (mean/std 0.5), CHW [1,3,448,448].
//   * prompt: ChatML; 64x<|image_pad|>(248056) inline in the user turn, rewritten to V+slot.
//   * one image serves a whole chat (each generate re-prefills).

import CoreAI
import CoreAILanguageModels
import CoreAIShared
import CoreGraphics
import Foundation
import Metal
import Tokenizers

@MainActor
final class MiniCPMVLBackend {
    // `nonisolated` so the (nonisolated) download spec in MiniCPMVIEngine can name them.
    nonisolated static let decoderBundle = "minicpmv46_vlm_decode_int8lin"
    nonisolated static let visionDir = "minicpmv46_vision"

    private let V: Int32 = 248_094
    private let IMG_TOK: Int32 = 248_056
    private let N = 64           // merged vision tokens (8x8 @ 448px)
    private let HID = 1024
    private let S = 448

    private var engine: (any InferenceEngine)?
    private var tokenizer: Tokenizer?
    private var visionFn: InferenceFunction?
    private var visionD: InferenceFunctionDescriptor?
    private var visionModel: AIModel?
    private var imgBuf: MTLBuffer?
    private(set) var ctx = 4096
    private(set) var imageAttached = false
    private var eosId = 248_044

    var loaded: Bool { engine != nil }

    /// Loads the decoder + vision bundles from `modelsDir` (where ModelDownloader placed them).
    func load(modelsDir: URL) async throws {
        if getenv("COREAI_CHUNK_THRESHOLD") == nil { setenv("COREAI_CHUNK_THRESHOLD", "1", 1) }
        let bundle = try LanguageBundle(at: modelsDir.appendingPathComponent(Self.decoderBundle))
        ctx = bundle.maxContextLength

        guard let device = MTLCreateSystemDefaultDevice() else { throw Self.err("no Metal device") }
        let img = device.makeBuffer(length: N * HID * 2, options: .storageModeShared)!
        memset(img.contents(), 0, img.length)
        imgBuf = img

        let config = ModelConfig(
            name: bundle.name, tokenizer: bundle.tokenizer, vocabSize: bundle.vocabSize,
            maxContextLength: bundle.maxContextLength, serializedModel: [bundle.modelAssetPath],
            function: bundle.language.functionMap?.name(for: "main") ?? "main")
        engine = try await EngineFactory.createEngine(
            config: try JSONEncoder().encode(config),
            modelURL: try bundle.requireModelURL(for: ModelBundle.ComponentKey.main),
            options: EngineOptions(staticInputBuffers: ["image_embeds": StaticInputBuffer(img)]))
        let tok = try await bundle.loadTokenizer()
        tokenizer = tok
        if let e = tok.eosTokenId { eosId = e }

        let visURL = modelsDir.appendingPathComponent(Self.visionDir)
            .appendingPathComponent("\(Self.visionDir).aimodel")
        guard FileManager.default.fileExists(atPath: visURL.path) else {
            throw Self.err("missing \(Self.visionDir).aimodel")
        }
        var vo = SpecializationOptions(preferredComputeUnitKind: .gpu)
        vo.expectFrequentReshapes = false
        let vm = try await AIModel(contentsOf: visURL, options: vo)
        guard let fn = try vm.loadFunction(named: "main") else { throw Self.err("vision main missing") }
        visionModel = vm; visionFn = fn; visionD = fn.descriptor

        _ = try await run(ids: [9707], maxTokens: 1, onText: { _ in })  // LLM warmup
        try? await warmupVision()  // compile the SigLIP graph NOW (at load) so the first photo is warm
    }

    // Run the vision encoder once on dummy pixels so its MPSGraph->Metal compile (the bulk of the
    // cold first-photo latency — encode itself is ~tens of ms warm) happens during load, not on the
    // user's first real photo. Output is discarded.
    private func warmupVision() async throws {
        guard let visionFn, let visionD else { return }
        guard case .ndArray(let pin)? = visionD.inputDescriptor(of: "pixel_values"),
              case .ndArray(let e0)? = visionD.outputDescriptor(of: "image_features") else { return }
        var pArr = NDArray(descriptor: pin.resolvingDynamicDimensions([1, 3, S, S]))
        fillNDArray(&pArr, as: Float16.self, with: [Float16](repeating: 0, count: 3 * S * S))
        var embOut = NDArray(descriptor: e0.resolvingDynamicDimensions([N, HID]))
        var out = InferenceFunction.MutableViews()
        out.insert(&embOut, for: "image_features")
        _ = try await visionFn.run(inputs: ["pixel_values": pArr],
                                   states: InferenceFunction.MutableViews(), outputViews: consume out)
    }

    // MARK: - Image attach (preprocess + on-device vision encode -> static buffer)

    /// Preprocess + vision-encode a CGImage into the static image buffer.
    func attach(cgImage: CGImage) async throws {
        // [1*3*448*448] f16, CHW, x/127.5-1
        try await attach(pixels: Self.preprocess(cgImage: cgImage, side: S))
    }

    /// Vision-encode already-preprocessed pixels (a Sendable `[Float16]`) into the static buffer.
    /// Splitting this out lets a nonisolated caller (e.g. a Visual Intelligence query) do the
    /// CGImage -> pixels work off the main actor and hand only Sendable values across the actor
    /// boundary (a CGImage is not Sendable).
    func attach(pixels: [Float16]) async throws {
        guard let visionFn, let visionD, let imgBuf else { throw Self.err("not loaded") }
        guard pixels.count == 3 * S * S else { throw Self.err("bad pixel count \(pixels.count)") }
        guard case .ndArray(let pin)? = visionD.inputDescriptor(of: "pixel_values") else {
            throw Self.err("pixel_values input missing")
        }
        var pArr = NDArray(descriptor: pin.resolvingDynamicDimensions([1, 3, S, S]))
        fillNDArray(&pArr, as: Float16.self, with: pixels)
        guard case .ndArray(let e0)? = visionD.outputDescriptor(of: "image_features") else {
            throw Self.err("image_features output missing")
        }
        var embOut = NDArray(descriptor: e0.resolvingDynamicDimensions([N, HID]))
        var out = InferenceFunction.MutableViews()
        out.insert(&embOut, for: "image_features")
        _ = try await visionFn.run(inputs: ["pixel_values": pArr],
                                   states: InferenceFunction.MutableViews(), outputViews: consume out)
        let emb = flattenAsFloat(embOut)
        let p = imgBuf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<min(emb.count, N * HID) { p[i] = Float16(emb[i]) }
        imageAttached = true
    }

    func detachImage() {
        imageAttached = false
        if let imgBuf { memset(imgBuf.contents(), 0, imgBuf.length) }
    }

    // MARK: - Generate

    /// Greedy answer; streams the decoded text, returns the engine's prompt ids + generated ids.
    @discardableResult
    func generate(_ prompt: String, maxNew: Int, onUpdate: @escaping (String) -> Void) async throws
        -> (promptIds: [Int32], tokens: [Int])
    {
        guard let tokenizer else { throw Self.err("not loaded") }
        var ids: [Int32]
        if imageAttached {
            let text = "<|im_start|>user\n"
                + String(repeating: "<|image_pad|>", count: N)
                + "\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer.encode(text: text).map { Int32($0) }
            var slot: Int32 = 0
            for i in 0..<ids.count where ids[i] == IMG_TOK {
                ids[i] = V + slot; slot += 1
            }
            guard slot == Int32(N) else { throw Self.err("expected \(N) image pads, got \(slot)") }
        } else {
            let text = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
            ids = tokenizer.encode(text: text).map { Int32($0) }
        }
        guard ids.count < ctx - 1 else { throw Self.err("prompt too long (\(ids.count))") }
        let budget = min(maxNew, ctx - ids.count - 1)
        let gen = try await run(ids: ids, maxTokens: budget) { gen in
            onUpdate(tokenizer.decode(tokens: gen, skipSpecialTokens: true))
        }
        return (ids, gen)
    }

    @discardableResult
    private func run(ids: [Int32], maxTokens: Int, onText: @escaping ([Int]) -> Void) async throws -> [Int] {
        guard let engine else { throw Self.err("not loaded") }
        try await engine.reset()
        let stream = try await engine.generate(
            with: ids, samplingConfiguration: SamplingConfiguration(temperature: 0),
            inferenceOptions: InferenceOptions(maxTokens: maxTokens))
        var gen: [Int] = []
        for try await step in stream {
            let tok = Int(step.tokenId)
            if tok == eosId { break }
            gen.append(tok)
            onText(gen)
        }
        return gen
    }

    // MARK: - Preprocess (resize 448 + x/127.5-1, CHW [1,3,448,448])

    // `nonisolated` (pure function, no actor state) so a nonisolated caller — e.g. a Visual
    // Intelligence query off the main actor — can preprocess to Sendable pixels itself.
    nonisolated static func preprocess(cgImage: CGImage, side S: Int) -> [Float16] {
        var rgba = [UInt8](repeating: 0, count: S * S * 4)
        let ctx = CGContext(data: &rgba, width: S, height: S, bitsPerComponent: 8,
                            bytesPerRow: S * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: S, height: S))
        var out = [Float16](repeating: 0, count: 3 * S * S)
        for c in 0..<3 {
            let cBase = c * S * S
            for y in 0..<S {
                for x in 0..<S {
                    let v = Float(rgba[(y * S + x) * 4 + c]) / 127.5 - 1.0
                    out[cBase + y * S + x] = Float16(v)
                }
            }
        }
        return out
    }

    private static func err(_ m: String) -> Error {
        NSError(domain: "MiniCPMVLBackend", code: 1, userInfo: [NSLocalizedDescriptionKey: m])
    }
}
