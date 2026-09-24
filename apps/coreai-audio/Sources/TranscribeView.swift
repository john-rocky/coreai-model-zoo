// TranscribeView — record from the mic, choose a file, then turn SPEECH into TEXT with one of two
// local Core AI ASR models (Whisper large-v3-turbo or Qwen3-ASR-1.7B, selectable). Distinct from
// the Understand tab (which describes sounds).

import SwiftUI
import UniformTypeIdentifiers

struct TranscribeView: View {
    @StateObject private var model = TranscribeModel.demo ?? TranscribeModel()
    @State private var showImporter = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("coreai-audio — on-device transcription")
                .font(.title2).bold()

            Picker("Model", selection: $model.engine) {
                ForEach(TranscribeModel.Engine.allCases) { engine in
                    Text(engine.title).tag(engine)
                }
            }
            .pickerStyle(.segmented)
            .disabled(model.busy || model.recording)

            if !model.showsDiarization {   // a phone needs the room for the timeline
                Text(model.engine.blurb)
                    .font(.callout).foregroundStyle(.secondary)
            }

            if model.diarizeAvailable {
                Toggle(isOn: $model.diarize) {
                    Label("Diarize — who said what", systemImage: "person.2.wave.2")
                }
                .disabled(model.busy || model.recording || model.live)
                Text(model.diarizesEightSpeakers
                     ? "Nemotron-3 Diarization labels each speaker turn (up to 8 speakers), then \(model.engine.title) transcribes it."
                     : "Streaming Sortformer labels each speaker turn, then \(model.engine.title) transcribes it.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if !model.loaded {
                Button { Task { await model.load() } } label: {
                    Label("Load model", systemImage: "arrow.down.circle")
                }.disabled(model.busy)
            }

            HStack(spacing: 12) {
                if model.engine == .nemotron {
                    // The streaming differentiator: transcribe WHILE speaking, no stop-then-wait.
                    Button {
                        model.toggleLive()
                    } label: {
                        Label(model.live ? "Stop" : "Live",
                              systemImage: model.live ? "stop.circle.fill" : "waveform.badge.mic")
                    }
                    .disabled(!model.loaded || model.busy || model.recording)
                    .tint(model.live ? .red : .accentColor)
                }

                Button {
                    model.toggleRecord()
                } label: {
                    Label(model.recording ? "Stop" : "Record",
                          systemImage: model.recording ? "stop.circle.fill" : "mic.circle")
                }
                .disabled(!model.loaded || model.busy || model.live)
                .tint(model.recording ? .red : .accentColor)

                Button { showImporter = true } label: {
                    Label("Choose…", systemImage: "waveform")
                }.disabled(!model.loaded || model.busy || model.recording || model.live)

                Button { Task { await model.transcribeClip() } } label: {
                    Label("Transcribe", systemImage: "text.bubble")
                }.disabled(!model.loaded || model.busy || model.recording || model.live)
            }

            Text(model.clipName).font(.footnote).foregroundStyle(.secondary)

            if model.busy { ProgressView().controlSize(.small) }

            if model.showsDiarization {
                DiarizeResultView(model: model)
            } else {
                ScrollView {
                    Text(model.transcript.isEmpty ? " " : model.transcript)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                        .padding(10)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }.frame(minHeight: 110)
            }

            if !model.language.isEmpty {
                Text("Detected language: \(model.language)")
                    .font(.footnote).foregroundStyle(.secondary)
            }

            Text(model.status).font(.footnote).foregroundStyle(.secondary)
            Spacer()
        }
        .padding(20)
        #if os(macOS)
            .frame(minWidth: 520, minHeight: 460)
        #endif
        .fileImporter(
            isPresented: $showImporter, allowedContentTypes: [.audio], allowsMultipleSelection: false
        ) { result in
            if case .success(let urls) = result, let url = urls.first { model.loadFile(url) }
        }
        // 8-speaker Diarize of a chosen file: the live screen over everything until Close
        #if os(iOS)
        .fullScreenCover(isPresented: $model.liveScreen) {
            DiarizeLiveView(model: model) { model.liveScreen = false }
        }
        #else
        .sheet(isPresented: $model.liveScreen) {
            DiarizeLiveView(model: model) { model.liveScreen = false }
                .frame(width: 402, height: 874)
        }
        #endif
    }
}
