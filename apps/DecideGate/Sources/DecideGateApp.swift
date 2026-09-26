// DecideGate — the iPhone gate of the GLiNER2.5-Decide Core AI port. Starts the gate on launch (no UI
// input), keeps the screen awake while it runs, shows progress and the result lines, and leaves
// Documents/decide_gate/result.json + result.log for `devicectl device copy from` (../_run.sh).

import SwiftUI
import UIKit

@main
struct DecideGateApp: App {
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
        UIApplication.shared.isIdleTimerDisabled = true
        let runner = GateRunner(
            config: .fromEnvironment(),
            emit: { line in Task { @MainActor in self.lines.append(line) } },
            setStage: { s in Task { @MainActor in self.stage = s } })
        verdict = await runner.run()
        UIApplication.shared.isIdleTimerDisabled = false
    }
}

struct ContentView: View {
    let model: GateModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("GLiNER2.5-Decide gate").font(.headline)
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
                        ForEach(Array(model.lines.enumerated()), id: \.offset) { i, line in
                            Text(line).font(.system(size: 10, design: .monospaced)).id(i)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .onChange(of: model.lines.count) { _, n in
                    if n > 0 { proxy.scrollTo(n - 1, anchor: .bottom) }
                }
            }
        }
        .padding()
    }
}
