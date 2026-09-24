// coreai-audio — on-device audio understanding (Qwen2.5-Omni Thinker on Core AI), built on CoreAIKit.

import SwiftUI

@main
struct CoreAIAudioApp: App {
    @State private var tab = 0
    #if os(iOS)
    /// The Diarize live screen is portrait only.
    @UIApplicationDelegateAdaptor(DiarizeOrientation.self) private var orientation
    #endif

    init() {
        // Headless ASR self-test: kicked from init() (not .task) so it runs without the GUI window
        // appearing — reliable for a CLI-launched verification run.
        if ProcessInfo.processInfo.environment["ASR_SELFTEST"] != nil {
            Task.detached { await runASRSelfTest() }
        }
        // Same init()-launch (not .task) as ASR — reliable for a headless CLI verification run.
        if ProcessInfo.processInfo.environment["VOXCPM_SELFTEST"] != nil {
            Task.detached { await runVoxCPMSelfTest() }
        }
        if ProcessInfo.processInfo.environment["VOXCPM2_SELFTEST"] != nil {
            Task.detached { await runVoxCPM2SelfTest() }
        }
        if ProcessInfo.processInfo.environment["PARAKEET_SELFTEST"] != nil {
            Task.detached { await runParakeetSelfTest() }
        }
        if ProcessInfo.processInfo.environment["NEMOTRON_SELFTEST"] != nil {
            Task.detached { await runNemotronSelfTest() }
        }
        if ProcessInfo.processInfo.environment["MUSIC_SELFTEST"] != nil {
            Task.detached { await runMusicSelfTest() }
        }
        if ProcessInfo.processInfo.environment["SEPARATE_SELFTEST"] != nil {
            Task.detached { await runSeparateSelfTest() }
        }
        if ProcessInfo.processInfo.environment["DIARIZE_SELFTEST"] != nil {
            Task.detached { await runDiarizeSelfTest() }
        }
        if ProcessInfo.processInfo.environment["VIBEVOICE_SELFTEST"] != nil {
            Task.detached { await runVibeVoiceSelfTest() }
        }
        if ProcessInfo.processInfo.environment["DIALOGUE_SELFTEST"] != nil {
            Task.detached { await runDialogueSelfTest() }
        }
        if ProcessInfo.processInfo.environment["ENCBENCH_SELFTEST"] != nil {
            Task.detached { await runEncoderBenchSelfTest() }
        }
        // DIARIZE_DEMO: from init() too, on the model the Transcribe tab shows (it needs no window to run)
        TranscribeModel.demo?.startDiarizeDemoIfRequested()
    }

    var body: some Scene {
        WindowGroup("coreai-audio") {
            Group {
                if let demo = TranscribeModel.demo {
                    // DIARIZE_DEMO: the Diarize live screen and nothing else (DiarizeDemo.swift)
                    DiarizeLiveView(model: demo)
                    #if os(macOS)
                        .frame(width: 402, height: 874)
                    #endif
                } else {
                    tabs
                }
            }
            // Non-blocking self-test (KOKORO_SELFTEST=1): the iOS launch watchdog
            // kills any main-thread block, so run it as a normal async task.
            .task {
                if ProcessInfo.processInfo.environment["KOKORO_SELFTEST"] != nil {
                    await runKokoroSelfTest()
                }
                // env var (device: devicectl --environment-variables) or, on macOS where `open`
                // can't inherit env, a ~/.dots_selftest trigger file.
                var dotsTrigger = ProcessInfo.processInfo.environment["DOTS_SELFTEST"] != nil
                #if os(macOS)
                dotsTrigger = dotsTrigger || FileManager.default.fileExists(
                    atPath: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".dots_selftest").path)
                #endif
                if dotsTrigger { await runDotsSelfTest() }
            }
        }
        #if os(macOS)
            .defaultSize(width: 560, height: 520)
        #endif
    }

    private var tabs: some View {
        TabView(selection: $tab) {
            ContentView()
                .tabItem { Label("Understand", systemImage: "ear") }.tag(0)
            TranscribeView()
                .tabItem { Label("Transcribe", systemImage: "text.bubble") }.tag(1)
            KokoroView()
                .tabItem { Label("Speak", systemImage: "speaker.wave.2") }.tag(2)
            VoxCPMView()
                .tabItem { Label("Voice", systemImage: "waveform") }.tag(3)
            VoxCPM2View()
                .tabItem { Label("Voice 2B", systemImage: "waveform.badge.plus") }.tag(4)
            DotsView()
                .tabItem { Label("Voice ML", systemImage: "globe") }.tag(5)
            MusicGenView()
                .tabItem { Label("Music", systemImage: "music.note") }.tag(6)
            SeparateView()
                .tabItem { Label("Separate", systemImage: "music.mic") }.tag(7)
            DialogueView()
                .tabItem { Label("Dialogue", systemImage: "person.2.wave.2") }.tag(8)
        }
    }
}
