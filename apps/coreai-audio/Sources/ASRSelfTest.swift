// Headless self-test: ASR_SELFTEST=1 loads Qwen3-ASR (local conversion artifacts), transcribes the
// wav at ASR_SELFTEST_WAV, and writes the result to /tmp/asr_swift_result.txt + NSLog. Verifies the
// CoreAIKit ASR path (KitASRModel) end-to-end on Mac without driving the GUI.

import CoreAIKit
import Foundation

func runASRSelfTest() async {
    NSLog("ASR selftest: start")
    let env = ProcessInfo.processInfo.environment
    let wav = env["ASR_SELFTEST_WAV"] ?? "/tmp/asr_test/ja1.wav"
    let artifacts =
        (NSHomeDirectory() + "/code/coreai/coreai-models-community/conversion/qwen3_asr/artifacts")
    let out = "/tmp/asr_swift_result.txt"

    func write(_ s: String) {
        try? s.write(toFile: out, atomically: true, encoding: .utf8)
        NSLog("ASR selftest: %@", s)
    }

    do {
        let dir = URL(filePath: artifacts)
        let t0 = Date()
        let model = try await KitASRModel(
            decoderBundleAt: dir.appending(path: "qwen3_asr_1.7b_decode_int8hu_n390_s1"),
            encoderModelAt: dir.appending(path: "qwen3_asr_1.7b_audio_encoder_fp16_k30.aimodel"),
            arch: .qwen3ASR1_7B)
        NSLog("ASR selftest: model loaded (%.1fs)", Date().timeIntervalSince(t0))

        guard let pcm = AudioLoader.load16kMono(URL(filePath: wav)) else {
            write("FAIL: could not decode \(wav)")
            return
        }
        NSLog("ASR selftest: clip %@ = %.1fs", wav, Double(pcm.count) / 16000)

        let t1 = Date()
        let result = try await model.transcribe(samples: pcm)
        let ms = Date().timeIntervalSince(t1) * 1000
        write("OK  lang=[\(result.language)]  (\(Int(ms)) ms)\n\(result.text)")
    } catch {
        write("FAIL: \(error)")
    }
}
