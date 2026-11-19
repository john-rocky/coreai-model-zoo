// Headless self-test + speed bench: PARAKEET_SELFTEST=1 loads the Parakeet bundle (the sideloaded
// Documents/Models/Parakeet on device, or the local conversion artifacts on a Mac), transcribes the
// libri1 wav, and reports load time + per-run transcribe time + RTF (audio seconds ÷ transcribe
// seconds). Writes Documents/parakeet_selftest_result.txt (pullable via devicectl) + NSLog.

import CoreAIKit
import Foundation

private let kGold = "With her white paint and her scarlet smokestack, the Inverashiel"

func runParakeetSelfTest() async {
    NSLog("Parakeet selftest: start")
    let fm = FileManager.default
    let env = ProcessInfo.processInfo.environment
    let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let out = docs.appending(path: "parakeet_selftest_result.txt")

    func write(_ s: String) {
        try? s.write(to: out, atomically: true, encoding: .utf8)
        NSLog("Parakeet selftest: %@", s)
    }

    do {
        // bundle: prefer the sideloaded device bundle (AOT encoder); else assemble from Mac artifacts.
        let bundle: URL
        let sideload = docs.appending(path: "Models/Parakeet")
        if (try? fm.contentsOfDirectory(at: sideload, includingPropertiesForKeys: nil))?
            .contains(where: { $0.lastPathComponent.lowercased().contains("encoder") }) == true {
            bundle = sideload
        } else {
            let art = URL(filePath:
                (NSHomeDirectory() + "/code/coreai/coreai-models-community/conversion/parakeet/artifacts"))
            let stage = URL(filePath: NSTemporaryDirectory()).appending(path: "parakeet_bundle")
            try? fm.removeItem(at: stage)
            try fm.createDirectory(at: stage, withIntermediateDirectories: true)
            try fm.createSymbolicLink(at: stage.appending(path: "encoder.aimodel"),
                withDestinationURL: art.appending(path: "parakeet_encoder_float16_L2885.aimodel"))
            try fm.createSymbolicLink(at: stage.appending(path: "predict.aimodel"),
                withDestinationURL: art.appending(path: "parakeet_predict_float32.aimodel"))
            try fm.createSymbolicLink(at: stage.appending(path: "joint.aimodel"),
                withDestinationURL: art.appending(path: "parakeet_joint_float32.aimodel"))
            for f in ["tokenizer.json", "tokenizer_config.json"] {
                try? fm.copyItem(at: art.appending(path: "bundle_assets/\(f)"), to: stage.appending(path: f))
            }
            bundle = stage
        }

        // wav: env override, else Documents/libri1.wav (device), else /tmp/libri1.wav (Mac).
        let wav: URL = {
            if let p = env["PARAKEET_SELFTEST_WAV"] { return URL(filePath: p) }
            let d = docs.appending(path: "libri1.wav")
            return fm.fileExists(atPath: d.path) ? d : URL(filePath: "/tmp/libri1.wav")
        }()
        guard let pcm = AudioLoader.load16kMono(wav) else { write("FAIL: could not decode \(wav.path)"); return }
        let clipSec = Double(pcm.count) / 16000

        let t0 = Date()
        let model = try await KitParakeetModel(bundleAt: bundle)
        let loadMs = Date().timeIntervalSince(t0) * 1000
        NSLog("Parakeet selftest: loaded (%.0f ms), clip %.2fs", loadMs, clipSec)

        // 4 runs: run 1 is cold (any lazy specialization), runs 2-4 warm. Report each + RTF.
        var times: [Double] = []
        var text = ""
        for i in 0..<4 {
            let t = Date()
            let r = try await model.transcribe(samples: pcm)
            times.append(Date().timeIntervalSince(t) * 1000)
            text = r.text
            NSLog("Parakeet selftest: run %d = %.0f ms", i + 1, times[i])
        }
        let warm = times.dropFirst().min() ?? times[0]          // best warm run
        let rtf = clipSec / (warm / 1000)                        // audio sec ÷ transcribe sec = ×real-time
        let pass = text.contains(kGold)
        write(String(format:
            "%@  clip %.2fs\nload %.0f ms\nruns ms: cold %.0f | warm %.0f/%.0f/%.0f\nbest warm %.0f ms → %.1f× real-time\n%@",
            pass ? "PASS" : "MISMATCH", clipSec, loadMs,
            times[0], times[1], times[2], times[3], warm, rtf, text))
    } catch {
        write("FAIL: \(error)")
    }
}
