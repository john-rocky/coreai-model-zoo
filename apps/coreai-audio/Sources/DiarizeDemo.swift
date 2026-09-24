// DiarizeDemo — the Transcribe tab's 8-speaker Diarize as a hands-off run, for a screen recording
// (record-demo.sh), plus the pieces the playback-synced run uses: the clip player, the device name and the
// run's numbers. Launch environment:
//   DIARIZE_DEMO=<clip>        open Transcribe, turn Diarize on, load the ASR and the diarizer, wait 2 s, then play
//                              <clips>/<clip>.wav with the speaker timeline, the transcript and the summary line
//   DIARIZE_DEMO_TRIGGER=1     after loading, wait for Documents/autoplay-<clip>-<epoch>.trigger files (put there by
//                              `devicectl device copy to`); each new one plays its clip, 2 s after it appears
//   DIARIZE_DEMO_LOG=1         mirror the numbers to Documents/diarize_demo/<launch epoch>.log (`devicectl device
//                              copy from`); "DONE <clip>" after each clip, "ERROR …" when something fails
//   DIARIZE_DEMO_ENGINE=…      parakeet | whisper | nemotron | qwen3asr (default: Parakeet; on iPhone only when it is
//                              sideloaded, else a sideloaded Nemotron, else Whisper from the Hub)
//   N3D_DEMO=<dir>             the clips (default <N3DAssets>/demo); DIARIZE_DEMO_VOLUME=0 plays silently (Mac checks)
//   DIARIZE_DEMO_SNAPSHOTS=1   the result view rendered off-screen every 2 s of a clip's run and at its end, as PNGs in
//                              Documents/diarize_demo/snapshots (no window needed: a locked Mac makes none)
// DIARIZE_DEMO_TRIGGER, DIARIZE_DEMO_LOG and DIARIZE_DEMO_SNAPSHOTS also take a directory path instead of 1. The run
// reads a file and plays it; it never touches the microphone.
import AVFoundation
import Darwin
import Foundation
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// One line of an 8-speaker Diarize transcript: the speaker's lane number and what the ASR heard in that turn.
struct DiarizedLine: Identifiable, Equatable {
    let id: Int
    let speaker: Int
    let text: String
}

/// One diarizer turn and its ASR (`seconds` nil: too short to transcribe), for the demo log.
struct DiarizedTurn {
    let speaker: Int
    let start: Double
    let end: Double
    let seconds: Double?
    let text: String
}

struct DiarizeDemoOptions: Sendable {
    let clip: String?
    let trigger: URL?
    let logDirectory: URL?
    let engine: TranscribeModel.Engine?
    let clips: URL
    let volume: Float
    let snapshots: URL?

    /// Nil unless the launch asked for the demo (DIARIZE_DEMO or DIARIZE_DEMO_TRIGGER).
    static let current: DiarizeDemoOptions? = {
        let env = ProcessInfo.processInfo.environment
        func directory(_ key: String, _ fallback: URL) -> URL? {
            guard let v = env[key], !v.isEmpty, v != "0" else { return nil }
            return v.hasPrefix("/") ? URL(fileURLWithPath: v) : fallback
        }
        let clip = env["DIARIZE_DEMO"].flatMap { $0.isEmpty ? nil : $0 }
        let trigger = directory("DIARIZE_DEMO_TRIGGER", URL.documentsDirectory)
        guard clip != nil || trigger != nil else { return nil }
        let engine = env["DIARIZE_DEMO_ENGINE"].flatMap { name in
            TranscribeModel.Engine.allCases.first { $0.rawValue.lowercased() == name.lowercased() }
        }
        let logs = URL.documentsDirectory.appendingPathComponent("diarize_demo")
        return DiarizeDemoOptions(
            clip: clip, trigger: trigger, logDirectory: directory("DIARIZE_DEMO_LOG", logs), engine: engine,
            clips: env["N3D_DEMO"].map { URL(fileURLWithPath: $0) }
                ?? NemotronDiarizerBridge.location.appendingPathComponent("demo"),
            volume: env["DIARIZE_DEMO_VOLUME"].flatMap(Float.init) ?? 1,
            snapshots: directory("DIARIZE_DEMO_SNAPSHOTS", logs.appendingPathComponent("snapshots")))
    }()
}

/// `<epoch> <HH:mm:ss.SSS> <text>` lines, to stdout and (with DIARIZE_DEMO_LOG) to <dir>/<launch epoch>.log.
@MainActor
final class DiarizeDemoLog {
    private let url: URL?
    private let clock: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    init(directory: URL?) {
        guard let directory else { url = nil; return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        url = directory.appendingPathComponent("\(Int(Date().timeIntervalSince1970)).log")
    }

    func write(_ text: String) {
        let now = Date()
        let line = String(format: "%.3f %@ %@\n", now.timeIntervalSince1970, clock.string(from: now), text)
        print("[DIARIZE_DEMO] \(line)", terminator: "")
        guard let url else { return }
        if let handle = try? FileHandle(forWritingTo: url) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: Data(line.utf8))
            try? handle.close()
        } else {
            try? line.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}

/// A playback-synced run's numbers, for the log (the screen shows only the summary line).
struct DiarizeRunStats {
    let clipSeconds: Double
    let engine: String
    var chunks = 0
    var frames = 0
    var graphMs: [Double] = []
    var hostMs: [Double] = []
    /// How long after its audio (look-ahead included) had played each chunk started.
    var lateMs: [Double] = []
    /// How far the lanes trail the audio when a chunk lands, at its first and at its last frame.
    var behindFirst: [Double] = []
    var behindLast: [Double] = []
    var speakers = 0
    var turns = 0
    var lines = 0
    var asrTurns: [DiarizedTurn] = []
    var playStart: Date?, playEnd: Date?, asrStart: Date?, asrEnd: Date?
    var summary = ""

    init(clipSeconds: Double, engine: String) {
        self.clipSeconds = clipSeconds
        self.engine = engine
    }

    mutating func addChunk(late: Double, graph: Double, host: Double, behindFirst first: Double, behindLast last: Double,
                           frames count: Int) {
        chunks += 1
        frames += count
        lateMs.append(late * 1e3)
        graphMs.append(graph * 1e3)
        hostMs.append(host * 1e3)
        guard count > 0 else { return }
        behindFirst.append(first)
        behindLast.append(last)
    }

    var logLines: [String] {
        func median(_ v: [Double]) -> Double {
            let s = v.sorted()
            return s.isEmpty ? 0 : s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
        }
        func p90(_ v: [Double]) -> Double { v.isEmpty ? 0 : v.sorted()[min(v.count - 1, Int(Double(v.count) * 0.9))] }
        func epoch(_ d: Date?) -> String { d.map { String(format: "%.3f", $0.timeIntervalSince1970) } ?? "-" }
        func span(_ a: Date?, _ b: Date?) -> Double {
            guard let a, let b else { return 0 }
            return b.timeIntervalSince(a)
        }
        return [
            String(format: "streaming: %d chunks, %d frames; graph %.2f ms median (p90 %.2f, max %.2f), host %.2f ms median; "
                   + "chunk start after its audio %.1f ms median (max %.1f)",
                   chunks, frames, median(graphMs), p90(graphMs), graphMs.max() ?? 0, median(hostMs),
                   median(lateMs), lateMs.max() ?? 0),
            String(format: "lanes behind the audio when a chunk lands: %.2f s at its first frame, %.2f s at its last "
                   + "(medians; max %.2f / %.2f); clip %.2f s played in %.2f s",
                   median(behindFirst), median(behindLast), behindFirst.max() ?? 0, behindLast.max() ?? 0,
                   clipSeconds, span(playStart, playEnd)),
            String(format: "turns %d, speakers %d, transcript lines %d; ASR %@ %.2f s for %d calls (%.2f s median)",
                   turns, speakers, lines, engine, span(asrStart, asrEnd), calls.count, median(calls)),
            "marks play_start=\(epoch(playStart)) play_end=\(epoch(playEnd)) asr_start=\(epoch(asrStart)) asr_end=\(epoch(asrEnd))",
            "summary \(summary)",
        ] + asrTurns.enumerated().map { i, t in
            String(format: "turn %d: Speaker %d %.2f-%.2f s: %@", i + 1, t.speaker, t.start, t.end,
                   t.seconds.map { String(format: "%.2f s ", $0) + (t.text.isEmpty ? "(empty)" : "\"\(t.text)\"") }
                       ?? "(too short)")
        }
    }

    private var calls: [Double] { asrTurns.compactMap(\.seconds) }
}

/// Plays a 16 kHz mono clip (the samples the diarizer reads) through AVAudioPlayer and says how far it has
/// played; the Diarize run paces its chunks and draws the playhead by it.
@MainActor
final class DiarizePlayback {
    let duration: Double
    private let player: AVAudioPlayer
    private var started = false

    struct Failed: LocalizedError {
        var errorDescription: String? { "Could not play the clip." }
    }

    init(samples: [Float], sampleRate: Int = 16000, volume: Float = 1) throws {
        player = try AVAudioPlayer(data: Self.wav(samples, sampleRate: sampleRate), fileTypeHint: AVFileType.wav.rawValue)
        player.volume = volume
        duration = Double(samples.count) / Double(sampleRate)
        player.prepareToPlay()
    }

    func play() throws {
        #if os(iOS)
        // playback only: the clip plays through the speaker, the microphone is not involved
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
        try? AVAudioSession.sharedInstance().setActive(true)
        #endif
        guard player.play() else { throw Failed() }
        started = true
    }

    func stop() { player.stop() }

    /// Played out (or stopped).
    var ended: Bool { started && !player.isPlaying }

    /// Seconds played: the player's position while it plays, the whole clip once it has ended.
    var position: Double { ended ? duration : player.currentTime }

    /// Returns once playback has passed `second` (or has ended).
    func wait(untilSecond second: Double) async {
        while !ended && !Task.isCancelled {
            let remaining = second - player.currentTime
            if remaining <= 0 { return }
            try? await Task.sleep(for: .milliseconds(max(2, min(50, Int(remaining * 1000)))))
        }
    }

    func waitUntilEnded() async {
        while !ended && !Task.isCancelled { try? await Task.sleep(for: .milliseconds(20)) }
    }

    /// 16-bit PCM WAV of the samples (the clips are 16-bit; AudioLoader's floats go back exactly).
    private static func wav(_ samples: [Float], sampleRate: Int) -> Data {
        let bytes = samples.count * 2
        var d = Data(capacity: 44 + bytes)
        func tag(_ s: String) { d.append(contentsOf: Array(s.utf8)) }
        func u32(_ v: Int) { withUnsafeBytes(of: UInt32(v).littleEndian) { d.append(contentsOf: $0) } }
        func u16(_ v: Int) { withUnsafeBytes(of: UInt16(v).littleEndian) { d.append(contentsOf: $0) } }
        tag("RIFF"); u32(36 + bytes); tag("WAVE")
        tag("fmt "); u32(16); u16(1); u16(1); u32(sampleRate); u32(sampleRate * 2); u16(2); u16(16)
        tag("data"); u32(bytes)
        let pcm = samples.map { Int16(clamping: $0.isFinite ? Int(($0 * 32768).rounded()) : 0).littleEndian }
        pcm.withUnsafeBytes { d.append(contentsOf: $0) }
        return d
    }
}

/// The device on the summary line: the iPhone model (utsname, named as `devicectl` lists these devices), the chip on
/// a Mac. An identifier not in the table is shown as is.
enum DeviceName {
    static let names = [
        "iPhone15,4": "iPhone 15", "iPhone18,1": "iPhone 17 Pro", "iPhone18,2": "iPhone 17 Pro Max",
        "iPhone18,3": "iPhone 17", "iPhone19,2": "iPhone 18 Pro", "iPad17,4": "iPad Pro 13-inch (M5)",
    ]

    /// utsname.machine, e.g. "iPhone18,1" ("arm64" on a Mac).
    static let machine: String = {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { raw in String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self) }
    }()

    static let current: String = {
        #if os(macOS)
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var chars = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("machdep.cpu.brand_string", &chars, &size, nil, 0)
        let brand = String(decoding: chars.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        return brand.isEmpty ? "Mac" : brand
        #else
        return names[machine] ?? machine
        #endif
    }()
}

extension TranscribeModel {
    /// The Transcribe tab's model on a DIARIZE_DEMO launch. The App's init starts the run on it before any window
    /// exists (a Mac app launched while the screen is locked makes none), and the tab shows this same instance.
    static let demo: TranscribeModel? = DiarizeDemoOptions.current == nil ? nil : TranscribeModel()
    private static var demoStarted = false

    /// Starts the DIARIZE_DEMO run once per process, in its own task (not tied to a view).
    func startDiarizeDemoIfRequested() {
        guard let options = DiarizeDemoOptions.current, !Self.demoStarted else { return }
        Self.demoStarted = true
        Task { await runDiarizeDemo(options) }
    }

    private func runDiarizeDemo(_ options: DiarizeDemoOptions) async {
        setvbuf(stdout, nil, _IOLBF, 0)
        let log = DiarizeDemoLog(directory: options.logDirectory)
        log.write("start: clip \(options.clip ?? "-"), trigger \(options.trigger?.path ?? "off"), clips \(options.clips.path), "
                  + "device \(DeviceName.current) (\(DeviceName.machine))")
        guard diarizesEightSpeakers else {
            status = "The 8-speaker diarizer is not staged at \(NemotronDiarizerBridge.location.path)."
            log.write("ERROR \(status)")
            return
        }
        let (choice, why) = options.engine.map { ($0, "DIARIZE_DEMO_ENGINE") } ?? Self.demoEngine()
        if engine != choice { engine = choice }
        diarize = true
        let t0 = Date()
        await load()
        guard loaded else { log.write("ERROR ASR: \(status)"); return }
        log.write(String(format: "asr %@ (%@) loaded in %.2f s", engine.title, why, Date().timeIntervalSince(t0)))
        do {
            let bridge = try await ensureNemotronDiarizer()
            log.write(String(format: "diarizer %@ loaded in %.2f s (graph %.2f s), first call %.2f s",
                             bridge.modelURL.lastPathComponent, diarizerLoadSeconds.load, bridge.loadSeconds,
                             diarizerLoadSeconds.firstCall))
            status = "Model ready."
        } catch {
            status = "Diarizer load failed: \(error.localizedDescription)"
            log.write("ERROR \(status)")
            return
        }
        if let directory = options.trigger {
            await watchTriggers(in: directory, log: log) { clip, take in
                await self.playDemoClip(clip, take: take, options: options, log: log)
            }
        } else if let clip = options.clip {
            await playDemoClip(clip, take: nil, options: options, log: log)
        }
    }

    /// The demo's ASR: Parakeet (on iPhone only a sideloaded one: the Hub graph's on-device JIT stalls), else a
    /// sideloaded Nemotron, else Whisper from the Hub.
    private static func demoEngine() -> (Engine, String) {
        #if os(iOS)
        if sideloadedBundle(named: "Parakeet") != nil { return (.parakeet, "sideloaded") }
        if sideloadedBundle(named: "Nemotron") != nil { return (.nemotron, "sideloaded; no Parakeet") }
        return (.whisper, "no sideloaded Parakeet or Nemotron")
        #else
        return (.parakeet, "default")
        #endif
    }

    /// Plays every new `autoplay-<clip>-<epoch>.trigger` in `directory`, one at a time. `devicectl device copy to` puts
    /// them there and the Mac cannot delete them, so each take has a new name; the ones there before the watch
    /// began are old takes and are skipped. A played trigger is deleted.
    private func watchTriggers(in directory: URL, log: DiarizeDemoLog, play: (String, String) async -> Void) async {
        let fm = FileManager.default
        func triggers() -> [String] {
            ((try? fm.contentsOfDirectory(atPath: directory.path)) ?? [])
                .filter { $0.hasPrefix("autoplay-") && $0.hasSuffix(".trigger") }.sorted()
        }
        var seen = Set(triggers())
        for old in seen { try? fm.removeItem(at: directory.appendingPathComponent(old)) }
        log.write("READY: waiting for \(directory.path)/autoplay-<clip>-<epoch>.trigger (\(seen.count) old skipped)")
        while !Task.isCancelled {
            for name in triggers() where !seen.contains(name) {
                seen.insert(name)
                let body = name.dropFirst("autoplay-".count).dropLast(".trigger".count)
                if let dash = body.lastIndex(of: "-") {
                    await play(String(body[..<dash]), String(body[body.index(after: dash)...]))
                } else {
                    log.write("skipped \(name): not autoplay-<clip>-<epoch>.trigger")
                }
                try? fm.removeItem(at: directory.appendingPathComponent(name))
            }
            try? await Task.sleep(for: .milliseconds(250))
        }
    }

    private func playDemoClip(_ clip: String, take: String?, options: DiarizeDemoOptions, log: DiarizeDemoLog) async {
        let file = clip.hasSuffix(".wav") ? clip : clip + ".wav"
        let url = options.clips.appendingPathComponent(file)
        let done = "\(clip)\(take.map { " " + $0 } ?? "")"
        guard let pcm = AudioLoader.load16kMono(url), !pcm.isEmpty else {
            status = "Could not decode \(file)."
            log.write("ERROR cannot read \(url.path) (\(done))")
            return
        }
        setClip(pcm, from: url)
        log.write(String(format: "clip %@: %.2f s (%d samples)%@", file, Double(pcm.count) / 16000, pcm.count,
                         take.map { ", take \($0)" } ?? ""))
        try? await Task.sleep(for: .seconds(2))
        let name = "\(clip)\(take.map { "_" + $0 } ?? "")"
        var snapper: Task<Void, Never>?
        if let dir = options.snapshots {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            snapper = Task {
                var n = 0
                while !Task.isCancelled {
                    n += 1
                    writeSnapshot(to: dir.appendingPathComponent(String(format: "%@_%03d.png", name, n)))
                    try? await Task.sleep(for: .seconds(2))
                }
            }
        }
        let stats = await transcribeDiarizedSynced(pcm)
        snapper?.cancel()
        if let dir = options.snapshots { writeSnapshot(to: dir.appendingPathComponent("\(name)_final.png")) }
        guard let stats else {
            log.write("ERROR \(status) (\(done))")
            return
        }
        for line in stats.logLines { log.write(line) }
        log.write("DONE \(done)")
    }

    /// The Diarize result view rendered off-screen at phone width (402 pt, light) to a PNG.
    private func writeSnapshot(to url: URL) {
        let content = DiarizeResultView(model: self, scrolls: false)
            .padding(20)
            .frame(width: 402)
            .background(Color.white)
            .environment(\.colorScheme, .light)
        let renderer = ImageRenderer(content: content)
        renderer.scale = 2
        guard let image = renderer.cgImage,
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }
}
