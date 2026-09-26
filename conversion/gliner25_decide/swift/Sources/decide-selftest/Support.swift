// Helpers of the self-test that only the macOS CLI needs: arguments and the load average. The
// shared part (fixtures, the host rule, the comparison, timing statistics) is the DecideGateSupport library,
// which the iPhone gate app (apps/DecideGate) uses too.

import Darwin
import DecideGateSupport
import Foundation

/// `--key value [value ...]` and bare `--flag`s; every value up to the next `--` belongs to the key before it.
struct Args {
    private var values: [String: [String]] = [:]
    private var flags: Set<String> = []

    init(_ argv: [String]) throws {
        var key: String? = nil
        for a in argv.dropFirst() {
            if a.hasPrefix("--") {
                key = String(a.dropFirst(2))
                flags.insert(key!)
            } else {
                guard let k = key else { throw GateError.usage("unexpected argument \(a)") }
                values[k, default: []].append(a)
            }
        }
    }

    subscript(_ key: String) -> String? { values[key]?.first }
    func has(_ key: String) -> Bool { flags.contains(key) }
    func int(_ key: String) -> Int? { self[key].flatMap { Int($0) } }
    func url(_ key: String) -> URL? { urls(key).first }
    func urls(_ key: String) -> [URL] {
        (values[key] ?? []).map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath).standardizedFileURL }
    }
}

func loadAverage() -> [Double] {
    var l = [Double](repeating: 0, count: 3)
    _ = getloadavg(&l, 3)
    return l.map { ($0 * 100).rounded() / 100 }
}
