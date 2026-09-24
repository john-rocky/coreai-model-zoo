// Helpers of the self-test that only the macOS CLI needs: arguments and the load average. The shared
// part (WAV input, float files, JSON, the comparison metrics of ../../gate_closed_loop.py, timing
// statistics) is the N3DGateSupport library, which the iPhone gate app (apps/N3DGate) uses too.

import Darwin
import Foundation
import N3DGateSupport

struct Args {
    private var values: [String: String] = [:]
    private var flags: Set<String> = []

    init(_ argv: [String]) throws {
        var i = 1
        while i < argv.count {
            let a = argv[i]
            guard a.hasPrefix("--") else { throw SelfTestError.usage("unexpected argument \(a)") }
            let key = String(a.dropFirst(2))
            if i + 1 < argv.count, !argv[i + 1].hasPrefix("--") {
                values[key] = argv[i + 1]
                i += 2
            } else {
                flags.insert(key)
                i += 1
            }
        }
    }

    subscript(_ key: String) -> String? { values[key] }
    func has(_ key: String) -> Bool { flags.contains(key) || values[key] != nil }
    func int(_ key: String) -> Int? { values[key].flatMap { Int($0) } }
    func url(_ key: String) -> URL? {
        values[key].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath).standardizedFileURL }
    }
}

func loadAverage() -> [Double] {
    var l = [Double](repeating: 0, count: 3)
    _ = getloadavg(&l, 3)
    return l.map { ($0 * 100).rounded() / 100 }
}
