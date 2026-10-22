// App.swift — TimestralGate on the iPhone: one run at launch from the launch arguments
// (`devicectl device process launch --terminate-existing com.coreai.timestralgate -- --mode gate --bundle <path> …`),
// bundles from Documents/models/, input packs from Documents/inputs/, everything out to Documents/out/<tag>/, then it
// idles showing the result. The screen is kept on from launch (a locked phone suspends the app and its GPU work, and
// refuses the next launch); only `--mode thermal`, a slot's last launch, gives the idle timer back. The view shows
// the status and the last log line only (UI redraws are GPU work too).

import SwiftUI
import UIKit

@MainActor
final class Runner: ObservableObject {
    @Published var status = "idle"
    @Published var last = ""

    func run(_ arguments: [String]) async {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let args = arguments.filter { $0 != "--" }
        guard !args.isEmpty else {
            status = "no arguments: launch with --mode gate|timing|warm|probe|thermal …"
            return
        }
        UIApplication.shared.isIdleTimerDisabled = true
        var log: GateLog?
        do {
            let options = try GateOptions.parse(args, root: docs)
            let gateLog = try GateLog(directory: options.out) { line in
                Task { @MainActor in self.last = line }
            }
            log = gateLog
            status = "running \(options.tag)"
            let done = try await Task.detached { try await Gate(options: options, log: gateLog).run() }.value
            gateLog.write("DONE \(done)")
            status = "DONE \(options.tag)"
            if options.mode == .thermal { UIApplication.shared.isIdleTimerDisabled = false }
        } catch {
            let message = "ERROR: \(error)"
            if let log {
                log.write(message)
            } else {
                // The arguments did not parse: leave a log under the tag the poller waits for.
                let tag = args.firstIndex(of: "--tag").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } ?? "error"
                let dir = docs.appendingPathComponent("out").appendingPathComponent(tag)
                try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                try? Data("t=   0.000 | \(message)\n".utf8).write(to: dir.appendingPathComponent("log.txt"))
            }
            status = message
        }
    }
}

@main
struct TimestralGateApp: App {
    @StateObject private var runner = Runner()

    var body: some Scene {
        WindowGroup {
            VStack(alignment: .leading, spacing: 8) {
                Text("TimestralGate").font(.title)
                Text(runner.status).font(.headline)
                Text(runner.last).font(.system(size: 10, design: .monospaced))
                Spacer()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .task { await runner.run(Array(CommandLine.arguments.dropFirst())) }
        }
    }
}
