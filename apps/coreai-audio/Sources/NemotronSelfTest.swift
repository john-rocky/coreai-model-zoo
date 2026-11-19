// Headless self-test + speed bench: NEMOTRON_SELFTEST=1 loads the Nemotron streaming bundle (the
// sideloaded Documents/Models/Nemotron on device, or the local conversion artifacts on a Mac),
// STREAMS the libri1 wav through a NemotronStreamSession in 100 ms mic-sized packets, and reports
// load time + per-chunk latency (avg/max vs the 320 ms audio budget) + RTF. Writes
// Documents/nemotron_selftest_result.txt (pullable via devicectl) + NSLog.

import CoreAIKit
import CoreAIKitVision
import Foundation

private let kGold = "With her white paint and her scarlet smoke stack, the inver is shiel"

func runNemotronSelfTest() async {
    NSLog("Nemotron selftest: start")
    let fm = FileManager.default
    let env = ProcessInfo.processInfo.environment
    let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
    let out = docs.appending(path: "nemotron_selftest_result.txt")

    func write(_ s: String) {
        try? s.write(to: out, atomically: true, encoding: .utf8)
        NSLog("Nemotron selftest: %@", s)
    }

    // NEMOTRON_CLEAN=<relative path under Documents>: delete it and exit — devicectl has no
    // remove verb, so re-pushing a directory bundle needs the app to clear the old tree first.
    if let rel = env["NEMOTRON_CLEAN"] {
        let target = docs.appending(path: rel)
        do {
            try fm.removeItem(at: target)
            write("CLEANED \(rel)")
        } catch {
            write("CLEAN FAILED \(rel): \(error)")
        }
        return
    }

    do {
        // bundle: prefer the sideloaded device bundle (AOT graphs); else assemble from Mac artifacts.
        let bundle: URL
        let sideload = docs.appending(path: "Models/Nemotron")
        if (try? fm.contentsOfDirectory(at: sideload, includingPropertiesForKeys: nil))?
            .contains(where: { $0.pathExtension == "aimodelc" || $0.pathExtension == "aimodel" })
            == true {
            bundle = sideload
        } else {
            let art = URL(filePath:
                (NSHomeDirectory() + "/code/coreai/coreai-models-community/conversion/nemotron_asr/artifacts"))
            let stage = URL(filePath: NSTemporaryDirectory()).appending(path: "nemotron_bundle")
            try? fm.removeItem(at: stage)
            try fm.createDirectory(at: stage, withIntermediateDirectories: true)
            for name in [
                "nemotron_asr_stream_pre_first_float16.aimodel",
                "nemotron_asr_stream_pre_float16.aimodel",
                "nemotron_asr_stream_conformer_float16.aimodel",
                "nemotron_asr_predict_float32.aimodel",
                "nemotron_asr_joint_float32.aimodel",
            ] {
                try fm.createSymbolicLink(at: stage.appending(path: name),
                    withDestinationURL: art.appending(path: name))
            }
            for f in ["tokenizer.json", "tokenizer_config.json"] {
                try? fm.copyItem(at: art.appending(path: "bundle_assets/\(f)"),
                    to: stage.appending(path: f))
            }
            bundle = stage
        }

        // wav: env override, else Documents/libri1.wav (device), else /tmp/libri1.wav (Mac).
        let wav: URL = {
            if let p = env["NEMOTRON_SELFTEST_WAV"] { return URL(filePath: p) }
            let d = docs.appending(path: "libri1.wav")
            return fm.fileExists(atPath: d.path) ? d : URL(filePath: "/tmp/libri1.wav")
        }()
        guard let pcm = AudioLoader.load16kMono(wav) else {
            write("FAIL: could not decode \(wav.path)")
            return
        }
        let clipSec = Double(pcm.count) / 16000

        // Diagnostics: write the on-device bundle tree, then probe each graph load separately so
        // a failure names the exact graph (a bare POSIX-2 from AIModel says nothing).
        var listing = "bundle: \(bundle.path)\n"
        var graphURLs: [URL] = []
        if let it = fm.enumerator(at: bundle, includingPropertiesForKeys: [.fileSizeKey]) {
            while let u = it.nextObject() as? URL {
                if u.pathExtension == "aimodel" || u.pathExtension == "aimodelc" {
                    graphURLs.append(u)
                }
                if (try? u.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory != true {
                    let sz = (try? u.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? -1
                    listing += "  \(sz)\t\(u.path.replacingOccurrences(of: bundle.path + "/", with: ""))\n"
                }
            }
        }
        NSLog("Nemotron selftest listing:\n%@", listing)
        // NEMOTRON_PROBE=1: load each graph separately (diagnostics). NEMOTRON_PROBE_FILTER
        // narrows to name substrings (comma-separated), preserving the given order.
        var probeURLs = graphURLs.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
        if let filter = env["NEMOTRON_PROBE_FILTER"] {
            probeURLs = filter.split(separator: ",").compactMap { sub in
                probeURLs.first { $0.lastPathComponent.contains(sub) }
            }
        }
        for u in probeURLs where env["NEMOTRON_PROBE"] != nil {
            do {
                let t = Date()
                _ = try await GraphModel(contentsOf: u, computeUnits: .gpu)
                NSLog("Nemotron selftest: probe %@ OK (%.0f ms)",
                      u.lastPathComponent, Date().timeIntervalSince(t) * 1000)
            } catch {
                let ns = error as NSError
                write("FAIL probing \(u.lastPathComponent): \(error)\n"
                    + "domain=\(ns.domain) code=\(ns.code)\nuserInfo=\(ns.userInfo)\n"
                    + "underlying=\((ns.userInfo[NSUnderlyingErrorKey] as? NSError)?.userInfo ?? [:])\n\n\(listing)")
                return
            }
        }

        let t0 = Date()
        let model = try await KitNemotronModel(bundleAt: bundle)
        let loadMs = Date().timeIntervalSince(t0) * 1000
        NSLog("Nemotron selftest: loaded (%.0f ms), clip %.2fs", loadMs, clipSec)

        // 2 streamed runs: run 1 is cold (lazy graph specialization on the first chunks), run 2 warm.
        var results: [(text: String, chunks: Int, avgMs: Double, maxMs: Double, wallMs: Double)] = []
        for run in 0..<2 {
            let session = try model.makeSession(language: "en-US")
            let tRun = Date()
            var i = 0
            while i < pcm.count {
                let end = min(i + 1600, pcm.count)             // 100 ms mic packets
                _ = try await session.feed(samples: Array(pcm[i..<end]))
                i = end
            }
            let r = try await session.finish()
            let wallMs = Date().timeIntervalSince(tRun) * 1000
            let avgMs = session.totalChunkSeconds / Double(max(session.chunkCount, 1)) * 1000
            results.append((r.text, session.chunkCount, avgMs, session.maxChunkSeconds * 1000, wallMs))
            NSLog("Nemotron selftest: run %d — %d chunks, avg %.1f ms, max %.1f ms",
                  run + 1, session.chunkCount, avgMs, session.maxChunkSeconds * 1000)
        }

        let warm = results[1]
        let pass = warm.text.hasPrefix(kGold)
        // RTF = compute time per second of audio (chunk latency ÷ 320 ms chunk audio).
        write(String(format: """
            %@  clip %.2fs
            load %.0f ms
            cold: %d chunks, avg %.1f ms, max %.1f ms
            warm: %d chunks, avg %.1f ms/chunk (audio 320 ms) → RTF %.3f (%.1f× real-time), max %.1f ms
            %@
            """,
            pass ? "PASS" : "MISMATCH", clipSec, loadMs,
            results[0].chunks, results[0].avgMs, results[0].maxMs,
            warm.chunks, warm.avgMs, warm.avgMs / 320, 320 / warm.avgMs, warm.maxMs,
            warm.text))
    } catch {
        write("FAIL: \(error)")
    }
}
