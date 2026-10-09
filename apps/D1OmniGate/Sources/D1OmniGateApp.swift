// D1OmniGate — the device gate of the d1-omni-600M Core AI port on the D1Omni library. Starts the gate on launch (no
// UI input), keeps the iPhone's screen awake while it runs, shows progress and the result lines, and leaves
// result.json + result.log + memory.tsv for `devicectl device copy from` (../_run.sh). On the Mac it quits when the gate
// is done (D1_EXIT_WHEN_DONE=0 keeps the window), so ../_run_mac.sh can wait for the process.
// Copied from apps/KevGate/Sources/KevGateApp.swift with the names changed.

import SwiftUI
#if os(iOS)
import UIKit
#else
import AppKit
#endif

@main
struct D1OmniGateApp: App {
    @State private var model = GateModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .task { await model.start() }
        }
    }
}

@MainActor
@Observable
final class GateModel {
    var stage = "starting"
    var lines: [String] = []
    var verdict: Bool?
    private var started = false

    func start() async {
        guard !started else { return }
        started = true
        // the battery level and state for every log line (UIDevice is read on the main actor), every 5 s
        BatteryCache.shared.set(DeviceInfo.battery())
        let poller = Task { @MainActor in
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(5))
                BatteryCache.shared.set(DeviceInfo.battery())
            }
        }
        defer { poller.cancel() }
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = true
        #else
        // A Mac app whose window is covered is put in App Nap: its threads drop to background priority and every timed
        // number with it. The activity holds the app at user-initiated priority for the whole gate.
        let activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .latencyCritical], reason: "d1-omni gate: timed decisions")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        #endif
        let config = GateConfig.fromEnvironment()
        let runner = GateRunner(
            config: config,
            emit: { line in Task { @MainActor in self.lines.append(line) } },
            setStage: { s in Task { @MainActor in self.stage = s } })
        verdict = await runner.run()
        #if os(iOS)
        UIApplication.shared.isIdleTimerDisabled = false
        #else
        if config.exitWhenDone { NSApplication.shared.terminate(nil) }
        #endif
    }
}

struct ContentView: View {
    let model: GateModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("d1-omni-600M gate").font(.headline)
            HStack {
                Text(model.stage).font(.subheadline.monospaced())
                Spacer()
                if let v = model.verdict {
                    Text(v ? "PASS" : "FAIL").font(.headline).foregroundStyle(v ? .green : .red)
                } else {
                    ProgressView()
                }
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(model.lines.suffix(400).enumerated()), id: \.offset) { i, line in
                            Text(line).font(.system(size: 10, design: .monospaced)).id(i)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: model.lines.count) { _, n in
                    if n > 0 { proxy.scrollTo(min(n, 400) - 1, anchor: .bottom) }
                }
            }
        }
        .padding()
    }
}
