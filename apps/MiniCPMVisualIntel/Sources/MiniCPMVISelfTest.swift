// Headless device check: `-mvi-selftest <out> -mvi-cases a,b` -> Documents/<out>.json (ids, text, timings).
import Foundation

enum MiniCPMVISelfTest {
    @MainActor
    static func runIfRequested() async {
        let d = UserDefaults.standard
        guard let out = d.string(forKey: "mvi-selftest"), !out.isEmpty else { return }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let cases = (d.string(forKey: "mvi-cases") ?? "").split(separator: ",").map(String.init)
        let prompt = d.string(forKey: "mvi-prompt") ?? "Describe this image."
        let maxNew = d.object(forKey: "mvi-max") == nil ? 32 : d.integer(forKey: "mvi-max")
        let thermal = { ProcessInfo.processInfo.thermalState.rawValue }  // 0 nominal ... 3 critical
        let free = try? docs.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        var rec: [String: Any] = ["thermal_start": thermal(), "prompt": prompt, "max_new": maxNew,
                                  "free_bytes": free?.volumeAvailableCapacityForImportantUsage ?? -1]
        var rows: [[String: Any]] = []
        do {
            if !cases.isEmpty {  // no cases = a thermal-only probe (the model is not loaded)
                let t0 = Date()
                let b = try await MiniCPMVIEngine.shared.ready()
                rec["load_s"] = Date().timeIntervalSince(t0)
                for c in cases {
                    let data = try Data(contentsOf: docs.appendingPathComponent("selftest/\(c).f16"))
                    let px = data.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) }
                    let t1 = Date()
                    try await b.attach(pixels: px)
                    let t2 = Date()
                    var text = ""
                    let r = try await b.generate(prompt, maxNew: maxNew) { text = $0 }
                    let t3 = Date()
                    rows.append([
                        "case": c, "pixel_count": px.count,
                        "pixel_sum": px.reduce(0.0) { $0 + Double($1) },
                        "prompt_ids": r.promptIds.map { Int($0) }, "gen_ids": r.tokens, "text": text,
                        "attach_s": t2.timeIntervalSince(t1), "generate_s": t3.timeIntervalSince(t2),
                    ])
                }
            }
        } catch {
            rec["error"] = "\(error)"
        }
        rec["cases"] = rows
        rec["thermal_end"] = thermal()
        if let json = try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]) {
            try? json.write(to: docs.appendingPathComponent("\(out).json"), options: .atomic)
        }
    }
}
