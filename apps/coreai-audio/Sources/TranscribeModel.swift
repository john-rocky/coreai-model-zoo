// TranscribeModel — view model for the Transcribe tab. Turns a recorded/chosen/demo clip into TEXT
// (speech-to-text) with a CHOICE of four Core AI ASR models, distinct from the Understand tab (which
// describes sounds):
//   • Whisper large-v3-turbo (Apple-recipe export on the stock runtime, via CoreAIKit's
//     KitWhisperModel) — published, downloads from the Hub on both platforms.
//   • Qwen3-ASR-1.7B (the zoo's first ASR, via CoreAIKit's KitASRModel).
//   • Parakeet-TDT-0.6B (the zoo's first transducer / TDT, via CoreAIKit's KitParakeetModel).
//   • Nemotron 3.5 Streaming 0.6B (the zoo's first STREAMING ASR, via CoreAIKit's
//     KitNemotronModel) — adds a LIVE mode: continuous mic transcription in 320 ms chunks,
//     any-length audio, 40 locales, punctuation built in.
//
// All models download from the Hugging Face Hub on first load (iOS + macOS) and cache on-device.

import CoreAIKit
import Foundation

@MainActor
final class TranscribeModel: ObservableObject {
    /// The selectable transcription engines.
    enum Engine: String, CaseIterable, Identifiable {
        case whisper, qwen3ASR, parakeet, nemotron
        var id: String { rawValue }
        var title: String {
            switch self {
            case .whisper: return "Whisper large-v3-turbo"
            case .qwen3ASR: return "Qwen3-ASR-1.7B"
            case .parakeet: return "Parakeet TDT 0.6B"
            case .nemotron: return "Nemotron Streaming 0.6B"
            }
        }
        var blurb: String {
            switch self {
            case .whisper: return "OpenAI Whisper on Core AI (stock runtime). 100 languages, ≤30 s."
            case .qwen3ASR: return "The zoo's first ASR. 52 languages, ≤30 s."
            case .parakeet: return "NVIDIA FastConformer transducer — the zoo's first TDT/RNN-T. 25 EU languages, ≤30 s."
            case .nemotron: return "The zoo's first STREAMING ASR — live mic, 320 ms chunks, 40 locales, any length."
            }
        }
    }

    @Published var engine: Engine = .whisper { didSet { if oldValue != engine { unload() } } }
    @Published var status = "Pick an engine, then tap “Load model”."
    @Published var loaded = false
    @Published var busy = false
    @Published var recording = false
    @Published var live = false
    @Published var clipName = "No audio loaded."
    @Published var transcript = ""
    @Published var language = ""
    /// Diarize mode: label each speaker turn ("who said what") by running the chosen ASR on every
    /// speaker turn — from the 8-speaker Nemotron-3 diarizer when its bundle is staged, else from the
    /// 4-speaker Sortformer. Available only when one of the two bundles is staged.
    @Published var diarize = false
    /// An 8-speaker Diarize result replaces the plain transcript: the speaker timeline, one colored
    /// "Speaker N: text" line per turn, and a summary line ("90.0 s · 4 speakers · 50 turns · iPhone 17 Pro").
    @Published var showsDiarization = false
    @Published var diarizedLines: [DiarizedLine] = []
    @Published var diarizeSummary = ""
    let timeline = DiarizeTimeline()
    /// The live screen (DiarizeLiveView) of a playback-synced run; `liveScreen`: a chosen file's run shows it full
    /// screen over the tab until Close.
    let diarizeLive = DiarizeLive()
    @Published var liveScreen = false

    private var whisper: KitWhisperModel?
    private var asr: KitASRModel?
    private var parakeet: KitParakeetModel?
    private var nemotron: KitNemotronModel?
    private var diarizer: SortformerDiarizer?
    private var nemotronDiarizer: NemotronDiarizerBridge?
    /// The 8-speaker diarizer's load and its first (warm-up) call, in seconds, for the demo log.
    private(set) var diarizerLoadSeconds = (load: 0.0, firstCall: 0.0)
    private var samples: [Float]?
    /// The file `samples` came from (nil for a mic recording): with 8-speaker Diarize a file plays with the timeline.
    private(set) var clipURL: URL?
    private var playback: DiarizePlayback?
    /// Plays the chosen clip aloud when transcription starts, so you can hear what you picked
    /// (the file importer gives no preview). 16 kHz mono — the same PCM fed to the model. Recreated
    /// per clip so a fresh Transcribe restarts audio instead of queueing behind the previous clip.
    private var player = AudioPlayer()

    /// Whether a diarize bundle is present (dev symlink on macOS / sideload on device).
    var diarizeAvailable: Bool { NemotronDiarizerBridge.staged != nil || DiarizeAssets.root != nil }
    /// The toggle runs the 8-speaker Nemotron-3 diarizer whenever it is staged (else the Sortformer).
    var diarizesEightSpeakers: Bool { NemotronDiarizerBridge.staged != nil }
    private let recorder = MicRecorder()
    private let micStreamer = MicStreamer()
    private var liveTask: Task<Void, Never>?

    /// The batch bundles transcribe ≤30 s clips; Nemotron streams any length.
    private let maxClipSeconds = 30

    // MARK: - Loading

    func load() async {
        guard !loaded, !busy else { return }
        busy = true
        status = "Loading \(engine.title)…"
        do {
            switch engine {
            case .whisper:
                whisper = try await KitWhisperModel(model: .largeV3Turbo) { progress in
                    Task { @MainActor [weak self] in
                        self?.status = String(
                            format: "Downloading %@ — %.0f%%",
                            (progress.currentFile as NSString).lastPathComponent,
                            progress.fraction * 100)
                    }
                }
            case .qwen3ASR:
                asr = try await KitASRModel(model: .qwen3ASR1_7B) { progress in
                    Task { @MainActor [weak self] in
                        self?.status = String(
                            format: "Downloading %@ — %.0f%%",
                            (progress.currentFile as NSString).lastPathComponent,
                            progress.fraction * 100)
                    }
                }
            case .parakeet:
                // Prefer a sideloaded bundle (AOT-compiled encoder) — the 1.2 GB encoder's on-device
                // JIT specialization stalls, so on iPhone we ship a precompiled `.aimodelc`.
                if let local = Self.sideloadedBundle(named: "Parakeet") {
                    status = "Loading Parakeet (sideloaded)…"
                    parakeet = try await KitParakeetModel(bundleAt: local)
                } else {
                    parakeet = try await KitParakeetModel(model: .parakeetTDT) { progress in
                        Task { @MainActor [weak self] in
                            self?.status = String(
                                format: "Downloading %@ — %.0f%%",
                                (progress.currentFile as NSString).lastPathComponent,
                                progress.fraction * 100)
                        }
                    }
                }
            case .nemotron:
                // Same AOT story as Parakeet: the 1.2 GB conformer graph ships precompiled for iPhone.
                if let local = Self.sideloadedBundle(named: "Nemotron") {
                    status = "Loading Nemotron (sideloaded)…"
                    nemotron = try await KitNemotronModel(bundleAt: local)
                } else {
                    nemotron = try await KitNemotronModel(model: .nemotronASRStreaming) { progress in
                        Task { @MainActor [weak self] in
                            self?.status = String(
                                format: "Downloading %@ — %.0f%%",
                                (progress.currentFile as NSString).lastPathComponent,
                                progress.fraction * 100)
                        }
                    }
                }
            }
            loaded = true
            status = "Model ready. Record / Choose / Demo, then Transcribe."
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
        busy = false
    }

    /// A sideloaded bundle in `Documents/Models/<name>` — the AOT-encoder path for iPhone, where
    /// the big encoder graphs' on-device JIT specialization stalls. Returns the directory if it
    /// holds any graph (`.aimodelc` AOT or `.aimodel`), else nil (→ Hub download).
    static func sideloadedBundle(named name: String) -> URL? {
        let fm = FileManager.default
        let dir = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appending(path: "Models/\(name)")
        guard let entries = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        else { return nil }
        let hasGraph = entries.contains {
            $0.pathExtension == "aimodelc" || $0.pathExtension == "aimodel"
        }
        return hasGraph ? dir : nil
    }

    /// Drop the loaded model(s) — called when the engine selection changes.
    private func unload() {
        stopLive()
        whisper = nil
        asr = nil
        parakeet = nil
        nemotron = nil
        loaded = false
        transcript = ""; language = ""
        clearDiarization()
        status = "Engine: \(engine.title). Tap “Load model”."
    }

    func loadFile(_ url: URL) {
        guard let pcm = AudioLoader.load16kMono(url) else {
            status = "Could not decode \(url.lastPathComponent)."
            return
        }
        setClip(pcm, from: url)
        status = "Audio loaded. Tap Transcribe."
        // 8-speaker Diarize: a chosen file starts right away, playing with the speaker timeline.
        if diarize && diarizesEightSpeakers && loaded && !busy && !live && !recording {
            Task { await transcribeClip() }
        }
    }

    func setClip(_ pcm: [Float], from url: URL) {
        samples = pcm
        clipURL = url
        transcript = ""; language = ""
        clearDiarization()
        clipName = "\(url.lastPathComponent)  (\(String(format: "%.1f", Double(pcm.count) / 16000))s)"
    }

    private func clearDiarization() {
        showsDiarization = false
        diarizedLines = []
        diarizeSummary = ""
    }

    func toggleRecord() {
        if recording {
            recording = false
            status = "Processing recording…"
            recorder.stop { [weak self] pcm in
                Task { @MainActor in
                    guard let self else { return }
                    self.samples = pcm
                    self.clipURL = nil
                    self.clipName = String(format: "Mic clip (%.1fs)", Double(pcm.count) / 16000)
                    self.status = pcm.isEmpty ? "No audio captured." : "Recorded. Tap Transcribe."
                }
            }
        } else {
            recording = true
            transcript = ""; language = ""
            clearDiarization()
            clipName = "Recording… tap Stop when done."
            status = "Listening…"
            recorder.start { [weak self] error in
                Task { @MainActor in
                    guard let self, let error else { return }
                    self.recording = false
                    self.clipName = "No audio loaded."
                    self.status = "Mic error: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Live streaming (Nemotron)

    /// Toggle continuous live-mic transcription — the Nemotron differentiator: 320 ms chunks
    /// stream through the cache-aware encoder while you speak; the transcript grows in place.
    func toggleLive() {
        if live { stopLive(); return }
        guard engine == .nemotron, let nemotron, !busy, !recording else { return }
        live = true
        transcript = ""; language = ""
        clearDiarization()
        clipName = "Live — speak; tap Stop when done."
        status = "Listening… (streaming on-device)"
        liveTask = Task { [weak self] in
            guard let self else { return }
            do {
                let session = try nemotron.makeSession(language: "en-US")
                let stream = try await self.micStreamer.start()
                var seconds = 0.0
                for await packet in stream {
                    let partial = try await session.feed(samples: packet)
                    seconds += Double(packet.count) / 16000
                    let s = seconds
                    await MainActor.run {
                        if !partial.isEmpty { self.transcript = partial }
                        self.clipName = String(format: "Live — %.0fs. Tap Stop when done.", s)
                    }
                }
                let result = try await session.finish()
                let avgMs = session.totalChunkSeconds / Double(max(session.chunkCount, 1)) * 1000
                await MainActor.run {
                    if !result.text.isEmpty { self.transcript = result.text }
                    self.language = result.language
                    self.status = String(
                        format: "Live done — %d chunks, avg %.0f ms/chunk (320 ms audio).",
                        session.chunkCount, avgMs)
                }
            } catch {
                await MainActor.run {
                    self.status = "Live failed: \(error.localizedDescription)"
                    self.live = false
                }
            }
        }
    }

    private func stopLive() {
        guard live else { return }
        live = false
        clipName = "No audio loaded."
        status = "Finishing live transcript…"
        micStreamer.stop()   // ends the packet stream; the live task then finish()es the session
    }

    // MARK: - Transcription

    func transcribeClip() async {
        guard loaded, let samples, !busy, !live else { return }
        // a chosen file plays on the live screen (a mic recording is diarized after the fact)
        if diarize && diarizesEightSpeakers && clipURL != nil {
            await transcribeDiarizedSynced(samples, fullScreen: true)
            return
        }
        playClip(samples)   // hear the clip you picked, in sync with transcription
        if diarize { await transcribeDiarized(samples); return }
        busy = true
        transcript = ""; language = ""
        clearDiarization()
        // Nemotron streams — no bucket, feed the whole clip; the batch engines cap at 30 s.
        let clip = engine == .nemotron ? samples : Array(samples.prefix(16000 * maxClipSeconds))
        status = "Transcribing… (on-device)"
        // Stream the running transcript into the UI as the decoder emits it. The callback fires off
        // the main actor, so hop back; the length guard keeps a late-arriving partial from
        // overwriting the final text with an earlier (shorter) snapshot.
        let onPartial: @Sendable (String) -> Void = { [weak self] partial in
            Task { @MainActor in
                guard let self else { return }
                if partial.count >= self.transcript.count { self.transcript = partial }
            }
        }
        do {
            let result: Transcription
            switch engine {
            case .whisper:
                guard let whisper else { busy = false; return }
                result = try await whisper.transcribe(samples: clip, onPartial: onPartial)
            case .qwen3ASR:
                guard let asr else { busy = false; return }
                status = "Encoding audio…"
                try await asr.attach(samples: clip)  // mel → AuT encoder → static buffer
                status = "Transcribing… (on-device)"
                result = try await asr.transcribe(onPartial: onPartial)
            case .parakeet:
                guard let parakeet else { busy = false; return }
                result = try await parakeet.transcribe(samples: clip, onPartial: onPartial)
            case .nemotron:
                guard let nemotron else { busy = false; return }
                result = try await nemotron.transcribe(samples: clip, onPartial: onPartial)
            }
            transcript = result.text
            language = result.language
            status = "Done."
        } catch {
            status = "Transcription failed: \(error.localizedDescription)"
        }
        busy = false
    }

    // MARK: - Diarized transcription ("who said what")

    /// Diarize the clip into speaker turns (Nemotron-3 Diarization or Streaming Sortformer on Core AI),
    /// then transcribe each turn's audio slice with the selected ASR engine and stitch a
    /// "Speaker N [t0–t1]: text" transcript. No ASR word timestamps are needed — the diarizer supplies
    /// the turn boundaries.
    private func transcribeDiarized(_ samples: [Float]) async {
        busy = true; transcript = ""; language = ""
        clearDiarization()
        defer { busy = false }
        status = "Diarizing — who spoke when…"
        do {
            if diarizesEightSpeakers {
                // 8 speakers: the whole timeline at once, then the colored lines
                let bridge = try await ensureNemotronDiarizer()
                status = "Diarizing — who spoke when…"
                let (turns, out) = try await bridge.diarize(samples)
                let clip = NemotronDiarizerBridge.clipProbs(out, samples: samples.count)
                showsDiarization = true
                timeline.show(probs: clip.probs, frames: clip.frames, duration: Double(samples.count) / 16000)
                _ = try await transcribeTurns(turns, samples)
                diarizeSummary = Self.summaryLine(seconds: Double(samples.count) / 16000,
                                                  speakers: timeline.speakerCount, turns: turns.count)
                status = "Done."
                return
            }
            let segs = try await diarizeTurns(samples)
            guard !segs.isEmpty else { transcript = "(no speech detected)"; status = "Done."; return }

            let sr = 16000.0
            let pad = 0.1                      // widen each turn slightly so onsets aren't clipped
            let minTurn = Int(0.3 * sr)        // skip sub-0.3 s turns (too short to transcribe)
            var speakerLabel: [Int: Int] = [:] // raw speaker -> display order (1-based, first-seen)
            var lines: [String] = []

            for (i, seg) in segs.enumerated() {
                status = "Transcribing turn \(i + 1)/\(segs.count)…"
                let a = max(0, Int((seg.startSec - pad) * sr))
                let b = min(samples.count, Int((seg.endSec + pad) * sr))
                guard b - a >= minTurn else { continue }
                let text = try await transcribeSlice(Array(samples[a..<b]))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                let label = speakerLabel[seg.speaker] ?? (speakerLabel.count + 1)
                speakerLabel[seg.speaker] = label
                lines.append(String(format: "Speaker %d [%.1f–%.1fs]: %@",
                                    label, seg.startSec, seg.endSec, text))
                transcript = lines.joined(separator: "\n\n")   // stream turns into the UI as they land
            }
            transcript = lines.isEmpty ? "(no speech transcribed)" : lines.joined(separator: "\n\n")
            status = "Done — \(speakerLabel.count) speaker(s), \(segs.count) turn(s)."
        } catch {
            status = "Diarized transcription failed: \(error.localizedDescription)"
        }
    }

    /// Speaker turns of the clip: the 8-speaker Nemotron-3 diarizer when it is staged, else the Sortformer.
    private func diarizeTurns(_ samples: [Float]) async throws -> [SpeakerSegment] {
        guard diarizesEightSpeakers else { return try await ensureDiarizer().diarize(samples).segments }
        return try await ensureNemotronDiarizer().diarize(samples).turns
    }

    /// Diarize while the clip plays, on the live screen (DiarizeLiveView): each 0.72 s chunk goes through the
    /// graph once its audio, look-ahead included, has played, and its frames extend the lanes about a second
    /// behind what you hear. When the clip ends, every turn goes through the selected ASR into a colored line
    /// (`withTranscript`), then the summary line. `fullScreen` (a chosen file): the live screen goes up over the
    /// tab until Close, READY for half a second before the clip plays, as LiveView.kt does for a picked clip.
    /// Returns the run's numbers (nil when it failed; the status line says why).
    @discardableResult
    func transcribeDiarizedSynced(_ samples: [Float], withTranscript: Bool = true, fullScreen: Bool = false) async
        -> DiarizeRunStats? {
        busy = true; transcript = ""; language = ""
        clearDiarization()
        defer { busy = false }
        var stats = DiarizeRunStats(clipSeconds: Double(samples.count) / 16000,
                                    engine: withTranscript ? engine.title : nil)
        diarizeLive.setClip(samples)
        if fullScreen { liveScreen = true }
        do {
            let bridge = try await ensureNemotronDiarizer()
            let stream = NemotronDiarizerStream(bridge: bridge, samples: samples)
            let playback = try DiarizePlayback(samples: samples, volume: DiarizeDemoOptions.current?.volume ?? 1)
            self.playback = playback
            if fullScreen { try? await Task.sleep(for: .milliseconds(500)) }
            showsDiarization = true
            timeline.start(duration: stats.clipSeconds, playhead: { [weak playback] in playback?.position ?? 0 })
            status = "Listening — speakers appear as they talk."
            try playback.play()
            diarizeLive.start { [weak playback] in playback?.position ?? 0 }
            stats.playStart = Date()
            var probs: [Float] = []
            probs.reserveCapacity(stream.clipFrames * NemotronDiarizerBridge.speakers)
            for k in 0..<stream.chunks.count {
                let due = Double(stream.readsUpTo(step: k)) / 16000
                await playback.wait(untilSecond: due)
                let late = playback.position - due
                let step = try await stream.step()
                timeline.append(probs: step.probs, frames: step.frames)
                diarizeLive.append(probs: step.probs, frames: step.frames)
                diarizeLive.chunkDone(k, seconds: step.graphSeconds + step.hostSeconds)
                probs.append(contentsOf: step.probs)
                let at = playback.position, f = NemotronDiarizerBridge.frameSec
                stats.addChunk(late: late, graph: step.graphSeconds, host: step.hostSeconds,
                               behindFirst: at - Double(step.firstFrame) * f,
                               behindLast: at - Double(step.firstFrame + step.frames) * f, frames: step.frames)
            }
            await playback.waitUntilEnded()
            stats.playEnd = Date()
            timeline.finish()
            diarizeLive.finish()
            stats.screen = [diarizeLive.latencyText, diarizeLive.stepText]

            let turns = NemotronDiarizerBridge.turns(from: probs, frames: stream.clipFrames)
            stats.turns = turns.count
            stats.speakers = timeline.speakerCount
            if withTranscript {
                stats.asrStart = Date()
                stats.asrTurns = try await transcribeTurns(turns, samples)
                stats.asrEnd = Date()
                stats.lines = diarizedLines.count
            }
            diarizeSummary = Self.summaryLine(seconds: stats.clipSeconds, speakers: stats.speakers, turns: stats.turns)
            stats.summary = diarizeSummary
            status = "Done."
            return stats
        } catch {
            playback?.stop()
            timeline.finish()
            diarizeLive.fail(error.localizedDescription)
            if fullScreen { liveScreen = false }
            status = "Diarized transcription failed: \(error.localizedDescription)"
            return nil
        }
    }

    /// Every turn through the selected ASR as a colored "Speaker N: text" line (N = the turn's lane on the
    /// timeline), streamed into `diarizedLines`. Sliced like the 4-speaker path: 0.1 s padding, turns too short
    /// to transcribe and empty results skipped. Returns what happened to each turn (the demo log lists them).
    private func transcribeTurns(_ turns: [SpeakerSegment], _ samples: [Float]) async throws -> [DiarizedTurn] {
        let sr = 16000.0
        let pad = 0.1
        let minTurn = Int(0.3 * sr)
        var done: [DiarizedTurn] = []
        for (i, seg) in turns.enumerated() {
            status = "Transcribing turn \(i + 1)/\(turns.count)…"
            let number = timeline.number(forSpeaker: seg.speaker) ?? timeline.speakerCount + 1
            let a = max(0, Int((seg.startSec - pad) * sr))
            let b = min(samples.count, Int((seg.endSec + pad) * sr))
            guard b - a >= minTurn else {
                done.append(DiarizedTurn(speaker: number, start: seg.startSec, end: seg.endSec, seconds: nil, text: ""))
                continue
            }
            let t0 = Date()
            let text = try await transcribeSlice(Array(samples[a..<b]))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            done.append(DiarizedTurn(speaker: number, start: seg.startSec, end: seg.endSec,
                                     seconds: Date().timeIntervalSince(t0), text: text))
            guard !text.isEmpty else { continue }
            diarizedLines.append(DiarizedLine(id: diarizedLines.count, speaker: number, text: text))
        }
        return done
    }

    /// "90.0 s · 4 speakers · 50 turns · iPhone 17 Pro": the clip, the speakers on the timeline, the diarizer's
    /// turns, and this device.
    static func summaryLine(seconds: Double, speakers: Int, turns: Int) -> String {
        String(format: "%.1f s · %d speaker%@ · %d turn%@ · %@", seconds, speakers, speakers == 1 ? "" : "s",
               turns, turns == 1 ? "" : "s", DeviceName.current)
    }

    /// Lazily load the 8-speaker diarizer from the staged bundle (GPU on Mac / device), then one call on a
    /// silent clip: the graph's first call after a load is slow (2.7 s on the iPhone gate), and a playing clip
    /// must not wait for it.
    func ensureNemotronDiarizer() async throws -> NemotronDiarizerBridge {
        if let d = nemotronDiarizer { return d }
        status = "Loading Nemotron-3 diarizer…"
        let t0 = Date()
        let d = try await NemotronDiarizerBridge.load()
        let t1 = Date()
        _ = try await d.diarize([])
        diarizerLoadSeconds = (t1.timeIntervalSince(t0), Date().timeIntervalSince(t1))
        nemotronDiarizer = d
        return d
    }

    /// Lazily build the diarizer from the staged bundle (GPU on Mac / device).
    private func ensureDiarizer() async throws -> SortformerDiarizer {
        if let d = diarizer { return d }
        guard let murl = DiarizeAssets.modelURL, let filters = DiarizeAssets.melFilters() else {
            throw NSError(domain: "diarize", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Diarize model not staged at \(DiarizeAssets.location.path)"])
        }
        let d = try await SortformerDiarizer(model: murl, melFilters: filters, computeUnits: .gpu)
        diarizer = d
        return d
    }

    /// Start playing the clip (16 kHz mono) from the top. A fresh AudioPlayer per call means the old
    /// one deallocates and stops, so re-transcribing restarts audio cleanly instead of queueing.
    private func playClip(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        player = AudioPlayer()
        player.play(samples, sampleRate: 16000)
    }

    /// Transcribe one speaker turn's audio with the currently-selected engine (text only).
    private func transcribeSlice(_ clip: [Float]) async throws -> String {
        switch engine {
        case .whisper:
            guard let whisper else { return "" }
            return try await whisper.transcribe(samples: clip).text
        case .qwen3ASR:
            guard let asr else { return "" }
            return try await asr.transcribe(samples: clip).text
        case .parakeet:
            guard let parakeet else { return "" }
            return try await parakeet.transcribe(samples: clip).text
        case .nemotron:
            guard let nemotron else { return "" }
            return try await nemotron.transcribe(samples: clip).text
        }
    }
}
