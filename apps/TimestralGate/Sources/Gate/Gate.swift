// Gate.swift — TimestralGate, shared by the iPhone app and the Mac command-line tool: one Timestral-209M graph bundle
// (an .aimodel specialized at load = JIT, or an .aimodelc = AOT) on the packed fixture windows
// (conversion/timestral/device_pack.py), its raw outputs written for the Mac to score
// (conversion/timestral/device_score.py), or several bundles timed in turns in one process, with the load time,
// memory, thermal state and power around every step. No verdict here: the Mac applies the port's own measures.
//
// The graph: patches [R, L, 64] + cos [L+h, 64] + sin [L+h, 64] + attn_mask [1, 1, L+h, L+h] -> quantiles
// [R, h*32, 21], one input dtype (float32, or float16 for the fp16-compute forms). A window has two rows [s, -s]
// (symmetric averaging): an R = 2 graph takes both in one call, an R = 1 graph in two calls (row 0, then row 1).
//
// Modes (--mode):
//   gate     one bundle (--bundle, --compute): every window of the bucket's pack (<inputs>/L<L>_h<h>/), --passes
//            times; pass<k>/quantiles.f32.bin = [N, 2, h*32, 21] float32 little-endian in the pack's window order.
//            Pass k is compared with pass 1 bit for bit here, and window 0 with sanity_ref.f32.bin (a first look).
//   timing   one or more --config <bundle>@<compute>: each loaded once (with --cold, its cache entry deleted first)
//            and its first forecast on --window timed (cold); then --rounds rounds A/B/C… (odd rounds reversed): per
//            round and config, --warm forecasts and then --calls timed forecasts. A forecast = the graph work of one
//            symmetric-averaging forecast (one R = 2 call, or two R = 1 calls), each output copied out. The loop
//            stops at --budget s of calls (the 18 Pro GPU slows down after about 20 s of back-to-back calls).
//   warm     every --config loaded once and run once on --window (a first Neural Engine load compiles).
//   probe    the device, thermal state, power, free disk, and what models/ and inputs/ hold; then DONE.
//   thermal  the thermal state every 10 s until nominal (DONE) or --cap s.
// Common: --tag <name> (the output folder), --root <dir> (models/, inputs/, out/; the phone: Documents), --inputs
//   <dir> (default <root>/inputs), --note, --cold. A bundle is a path under <root>/models/ or an absolute path.
//   A bare `--` (devicectl passes it through) is skipped. Every load writes `loading …` before it starts.
// Output: <root>/out/<tag>/log.txt, flushed line by line, whose last line is `DONE …` or `ERROR: …` (what the poller
//   waits for), and <root>/out/<tag>/run.json (written on errors too).

import CoreAI
import CryptoKit
import Darwin
import Foundation
#if os(iOS)
import UIKit
#endif

// MARK: - Options

struct GateError: Error, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

struct BundleConfig: Sendable {
    let url: URL
    let compute: String

    /// gpu-jit, gpu-aot, ane-jit, ane-aot, cpuonly-jit …: the compute preference and how the bundle is specialized.
    var route: String { "\(compute)-\(url.pathExtension == "aimodelc" ? "aot" : "jit")" }

    /// The bundle's name without the extension; an AOT asset is named by its folder (`<form>-h19p-ane/<form>.h19p.aimodelc`).
    var form: String {
        if url.pathExtension == "aimodelc" { return url.deletingLastPathComponent().lastPathComponent }
        return url.deletingPathExtension().lastPathComponent
    }

    var label: String { "\(form)@\(route)" }
}

struct GateOptions: Sendable {
    enum Mode: String, Sendable, CaseIterable { case gate, timing, warm, probe, thermal }

    var mode = Mode.gate
    var configs: [BundleConfig] = []
    var window: String?
    var warm = 1
    var calls = 4
    var rounds = 8
    var passes = 2
    var budget = 20.0
    var tag: String
    var note: String?
    var root: URL
    var inputs: URL
    var cold = false
    var coolCap = 300.0
    var cap = 900.0

    var out: URL { root.appendingPathComponent("out").appendingPathComponent(tag) }

    static func specialization(_ name: String) throws -> SpecializationOptions {
        switch name {
        case "gpu": return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case "ane": return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case "cpu": return SpecializationOptions(preferredComputeUnitKind: .cpu)
        case "cpuonly": return .cpuOnly
        case "default": return .default
        default: throw GateError("compute is gpu | ane | cpu | cpuonly | default, not '\(name)'")
        }
    }

    static func parse(_ arguments: [String], root defaultRoot: URL) throws -> GateOptions {
        var values: [String: String] = [:]
        var configs: [String] = []
        var cold = false
        var rest = arguments[...]
        while let arg = rest.popFirst() {
            switch arg {
            case "--": continue
            case "--cold": cold = true
            case "--config":
                guard let value = rest.popFirst() else { throw GateError("--config needs <bundle>@<compute>") }
                configs.append(value)
            case "--mode", "--bundle", "--compute", "--window", "--warm", "--calls", "--rounds", "--passes", "--budget",
                 "--tag", "--note", "--root", "--inputs", "--cool-cap", "--cap":
                guard let value = rest.popFirst() else { throw GateError("\(arg) needs a value") }
                values[arg] = value
            default:
                throw GateError("unknown argument '\(arg)'")
            }
        }
        let root = values["--root"].map { URL(fileURLWithPath: $0) } ?? defaultRoot
        guard let mode = Mode(rawValue: values["--mode"] ?? "gate") else {
            throw GateError("--mode is one of \(Mode.allCases.map(\.rawValue).joined(separator: " | "))")
        }
        func bundleURL(_ name: String) -> URL {
            name.hasPrefix("/") ? URL(fileURLWithPath: name) : root.appendingPathComponent("models").appendingPathComponent(name)
        }
        /// <bundle>@<compute>; the bundle may hold '/' (a path under models/), the last '@' splits.
        func config(_ spec: String) throws -> BundleConfig {
            guard let sep = spec.lastIndex(of: "@"), sep != spec.startIndex else {
                throw GateError("a config is <bundle>@<compute>, not '\(spec)'")
            }
            let compute = String(spec[spec.index(after: sep)...])
            _ = try specialization(compute)
            return BundleConfig(url: bundleURL(String(spec[..<sep])), compute: compute)
        }
        var parsed: [BundleConfig] = []
        if let bundle = values["--bundle"] {
            let compute = values["--compute"] ?? "gpu"
            _ = try specialization(compute)
            parsed.append(BundleConfig(url: bundleURL(bundle), compute: compute))
        }
        let many = try configs.map(config)
        switch mode {
        case .gate:
            guard parsed.count == 1, many.isEmpty else { throw GateError("--mode gate takes one --bundle (and --compute)") }
        case .timing, .warm:
            guard !many.isEmpty, parsed.isEmpty else {
                throw GateError("--mode \(mode.rawValue) takes --config <bundle>@<compute> (repeat for A/B/C)")
            }
        case .probe, .thermal:
            break
        }
        func count(_ key: String, _ fallback: Int, minimum: Int) throws -> Int {
            guard let text = values[key] else { return fallback }
            guard let n = Int(text), n >= minimum else { throw GateError("\(key) takes a whole number ≥ \(minimum)") }
            return n
        }
        func seconds(_ key: String, _ fallback: Double) throws -> Double {
            guard let text = values[key] else { return fallback }
            guard let s = Double(text), s >= 0 else { throw GateError("\(key) takes seconds ≥ 0") }
            return s
        }
        var options = GateOptions(
            mode: mode, configs: parsed + many,
            tag: values["--tag"] ?? "\(mode.rawValue)-\(Int(Date().timeIntervalSince1970))",
            note: values["--note"], root: root,
            inputs: values["--inputs"].map { URL(fileURLWithPath: $0) } ?? root.appendingPathComponent("inputs"))
        options.window = values["--window"]
        options.warm = try count("--warm", 1, minimum: 0)
        options.calls = try count("--calls", 4, minimum: 1)
        options.rounds = try count("--rounds", 8, minimum: 1)
        options.passes = try count("--passes", 2, minimum: 1)
        options.budget = try seconds("--budget", 20)
        options.cold = cold
        options.coolCap = try seconds("--cool-cap", 300)
        options.cap = try seconds("--cap", 900)
        guard !options.tag.contains("/") else { throw GateError("--tag is a folder name, without '/'") }
        return options
    }
}

// MARK: - Log

/// Progress lines, flushed to <out>/log.txt as they happen (the poller pulls the file while the run goes) and handed
/// to `sink` (the phone's status view, the Mac's stderr).
final class GateLog: @unchecked Sendable {
    private let lock = NSLock()
    private let handle: FileHandle
    private let start = Date()
    private let sink: @Sendable (String) -> Void
    let url: URL

    init(directory: URL, sink: @escaping @Sendable (String) -> Void) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        url = directory.appendingPathComponent("log.txt")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try FileHandle(forWritingTo: url)
        self.sink = sink
    }

    func write(_ message: String) {
        let line = String(format: "t=%8.3f | ", Date().timeIntervalSince(start)) + message + "\n"
        lock.lock()
        defer { lock.unlock() }
        handle.write(Data(line.utf8))
        try? handle.synchronize()
        sink(line)
    }
}

// MARK: - Device, memory, thermal, power

enum Probe {
    static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return "" }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    /// `iPhone19,2`, `Mac16,9`: the hardware model, not a marketing name.
    static var machine: String {
        #if os(macOS)
        return sysctl("hw.model")
        #else
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
        #endif
    }

    static var osBuild: String { sysctl("kern.osversion") }

    /// The process's physical footprint (`task_vm_info.phys_footprint`, what jetsam counts).
    static func footprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Int64(info.phys_footprint) : -1
    }

    /// What the process may still allocate before jetsam (`os_proc_available_memory`); nil on the Mac.
    static func available() -> Int64? {
        #if os(iOS)
        return Int64(os_proc_available_memory())
        #else
        return nil
        #endif
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static var hot: Bool { ["serious", "critical"].contains(thermal()) }

    /// Free space on the volume of `url` (devicectl does not report the phone's).
    static func freeBytes(_ url: URL) -> Int64? {
        (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?.volumeAvailableCapacityForImportantUsage
    }

    /// Process CPU time (all threads), nanoseconds.
    static func cpuNanoseconds() -> UInt64 { clock_gettime_nsec_np(CLOCK_PROCESS_CPUTIME_ID) }

    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

    /// The Core AI framework's own version strings, when the bundle answers.
    static func coreAIVersion() -> [String: Any] {
        for id in ["com.apple.CoreAI", "com.apple.coreai"] {
            if let info = Bundle(identifier: id)?.infoDictionary {
                return ["bundle_id": id, "short": orNull(info["CFBundleShortVersionString"]),
                        "version": orNull(info["CFBundleVersion"])]
            }
        }
        return ["bundle_id": NSNull()]
    }
}

struct PowerState: Sendable {
    let battery: String
    let level: Double?
    let lowPowerMode: Bool

    var json: [String: Any] { ["battery": battery, "level": orNull(level), "low_power_mode": lowPowerMode] }
    var line: String { "battery=\(battery) level=\(level.map { String(format: "%.2f", $0) } ?? "-") lpm=\(lowPowerMode)" }

    static func read() async -> PowerState {
        #if os(iOS)
        return await MainActor.run {
            UIDevice.current.isBatteryMonitoringEnabled = true
            let state: String
            switch UIDevice.current.batteryState {
            case .charging: state = "charging"
            case .full: state = "full"
            case .unplugged: state = "unplugged"
            default: state = "unknown"
            }
            let level = UIDevice.current.batteryLevel
            return PowerState(battery: state, level: level >= 0 ? Double(level) : nil,
                              lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
        }
        #else
        return PowerState(battery: "mac", level: nil, lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled)
        #endif
    }
}

/// The peak footprint and the least available memory, sampled every 50 ms while a run goes, and every thermal-state
/// change with its time.
final class Sampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: Int64 = 0
    private var least: Int64?
    private var changes: [(seconds: Double, state: String)] = []
    private var running = true
    private let start = Date()
    private var observer: NSObjectProtocol?

    func begin() {
        observer = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil
        ) { [weak self] _ in
            guard let self else { return }
            let state = Probe.thermal()
            self.lock.lock()
            self.changes.append((Date().timeIntervalSince(self.start), state))
            self.lock.unlock()
        }
        let thread = Thread { [weak self] in
            while let self, self.sample() {
                Thread.sleep(forTimeInterval: 0.05)
            }
        }
        thread.start()
    }

    /// One sample; false once `end()` was called.
    private func sample() -> Bool {
        let footprint = Probe.footprint()
        let available = Probe.available()
        lock.lock()
        defer { lock.unlock() }
        peak = max(peak, footprint)
        if let available { least = min(least ?? available, available) }
        return running
    }

    func end() -> [String: Any] {
        _ = sample()
        lock.lock()
        defer { lock.unlock() }
        running = false
        if let observer { NotificationCenter.default.removeObserver(observer) }
        return ["footprint_peak_mb": mb(peak), "available_least_mb": orNull(least.map(mb)),
                "thermal_changes": changes.map { ["t_s": round3($0.seconds), "state": $0.state] as [String: Any] }]
    }
}

func mb(_ bytes: Int64) -> Double { (Double(bytes) / 1_000_000 * 10).rounded() / 10 }
func round3(_ x: Double) -> Double { (x * 1000).rounded() / 1000 }
/// An optional as a JSON value: the value, or null.
func orNull<T>(_ value: T?) -> Any { value.map { $0 as Any } ?? NSNull() }

// MARK: - Statistics and JSON

enum Stats {
    static func percentile(_ values: [Double], _ q: Double) -> Double {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return .nan }
        let position = q * Double(sorted.count - 1)
        let low = Int(position.rounded(.down)), high = Int(position.rounded(.up))
        return sorted[low] + (sorted[high] - sorted[low]) * (position - Double(low))
    }

    static func median(_ values: [Double]) -> Double { percentile(values, 0.5) }
}

enum JSON {
    /// NaN / ±inf become null (JSONSerialization refuses them).
    static func clean(_ value: Any) -> Any {
        switch value {
        case let d as Double: return d.isFinite ? d : NSNull()
        case let f as Float: return f.isFinite ? Double(f) : NSNull()
        case let a as [Any]: return a.map(clean)
        case let o as [String: Any]: return o.mapValues(clean)
        default: return value
        }
    }

    static func write(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: clean(object), options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }

    static func read(_ url: URL) throws -> [String: Any] {
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw GateError("\(url.lastPathComponent) is not a JSON object")
        }
        return object
    }
}

func md5Hex(_ data: Data) -> String { Insecure.MD5.hash(data: data).map { String(format: "%02x", $0) }.joined() }

// MARK: - Arrays

enum DType: String, Sendable {
    case f16, f32

    init(_ scalarType: NDArray.ScalarType) throws {
        switch scalarType {
        case .float16: self = .f16
        case .float32: self = .f32
        default: throw GateError("unsupported I/O scalar type \(scalarType)")
        }
    }

    var scalarType: NDArray.ScalarType { self == .f16 ? .float16 : .float32 }
}

/// Every element of a strided array (rank ≤ 4) in row-major order; `body` gets the element's offset in the source.
@inline(__always)
func forEachOffset(shape: [Int], strides: [Int], _ body: (Int) -> Void) {
    let pad = 4 - shape.count
    let s = [Int](repeating: 1, count: pad) + shape, st = [Int](repeating: 0, count: pad) + strides
    for a in 0..<s[0] {
        for b in 0..<s[1] {
            for c in 0..<s[2] {
                let base = a * st[0] + b * st[1] + c * st[2]
                for d in 0..<s[3] { body(base + d * st[3]) }
            }
        }
    }
}

/// Row-major (dimensions of size 1 ignored).
@inline(__always)
func rowMajor(_ shape: [Int], _ strides: [Int]) -> Bool {
    var expect = 1
    for i in stride(from: shape.count - 1, through: 0, by: -1) where shape[i] > 1 {
        if strides[i] != expect { return false }
        expect *= shape[i]
    }
    return true
}

/// A new array of `scalarType` filled from `values` (row-major) through the array's own strides: a straight copy
/// (the framework's init(scalars:) walks a generic Sequence one element at a time).
func makeNDArray<T: BitwiseCopyable>(_ values: UnsafeBufferPointer<T>, shape: [Int], scalarType: NDArray.ScalarType) -> NDArray {
    let n = shape.reduce(1, *)
    precondition(values.count == n, "makeNDArray: \(values.count) values for shape \(shape)")
    var array = NDArray(shape: shape, scalarType: scalarType)
    array.mutableView(as: T.self).withUnsafeMutablePointer { p, _, st in
        var ds = [Int]()
        for i in 0..<st.count { ds.append(st[i]) }
        if rowMajor(shape, ds) {
            p.update(from: values.baseAddress!, count: n)
        } else {
            var i = 0
            forEachOffset(shape: shape, strides: ds) { p[$0] = values[i]; i += 1 }
        }
    }
    return array
}

/// A float16 / float32 array of rank ≤ 4 copied out as Float, row-major, into `dst` (exactly `count` elements).
func copyOut(_ array: NDArray, into dst: UnsafeMutablePointer<Float>, count: Int) throws {
    guard array.interleaveLayout == nil else { throw GateError("the output comes interleaved (\(String(describing: array.interleaveLayout)))") }
    let shape = array.shape, strides = array.strides
    guard shape.count <= 4, strides.count == shape.count else { throw GateError("output rank \(shape.count): only rank ≤ 4 is read") }
    guard shape.reduce(1, *) == count else { throw GateError("output shape \(shape), expected \(count) elements") }
    let plain = rowMajor(shape, strides)
    switch array.scalarType {
    case .float32:
        array.view(as: Float.self).withUnsafePointer { p, _, _ in
            if plain {
                dst.update(from: p, count: count)
            } else {
                var i = 0
                forEachOffset(shape: shape, strides: strides) { dst[i] = p[$0]; i += 1 }
            }
        }
    case .float16:
        array.view(as: Float16.self).withUnsafePointer { p, _, _ in
            if plain {
                for i in 0..<count { dst[i] = Float(p[i]) }
            } else {
                var i = 0
                forEachOffset(shape: shape, strides: strides) { dst[i] = Float(p[$0]); i += 1 }
            }
        }
    default:
        throw GateError("expected a float16 / float32 output, got \(array.scalarType)")
    }
}

/// Bit-identical (NaN payloads included) and the largest |a - b| over the finite pairs that differ.
func compareBits(_ a: UnsafeBufferPointer<Float>, _ b: UnsafeBufferPointer<Float>) -> (identical: Bool, different: Int, maxAbs: Double) {
    guard a.count == b.count else { return (false, max(a.count, b.count), .infinity) }
    var different = 0
    var worst = 0.0
    for i in 0..<a.count where a[i].bitPattern != b[i].bitPattern {
        different += 1
        if a[i].isFinite && b[i].isFinite { worst = max(worst, Double(abs(a[i] - b[i]))) } else { worst = .infinity }
    }
    return (different == 0, different, worst)
}

func compareBits(_ a: [Float], _ b: [Float]) -> (identical: Bool, different: Int, maxAbs: Double) {
    a.withUnsafeBufferPointer { pa in b.withUnsafeBufferPointer { pb in compareBits(pa, pb) } }
}

func nonFinite(_ values: UnsafeBufferPointer<Float>) -> Int { values.reduce(0) { $0 + ($1.isFinite ? 0 : 1) } }

/// Output floats at a fixed address: the calls copy into it between awaits (an Array's storage cannot be held
/// across a suspension).
final class FloatBuffer: @unchecked Sendable {
    let pointer: UnsafeMutablePointer<Float>
    let count: Int

    init(count: Int) {
        self.count = count
        pointer = .allocate(capacity: max(count, 1))
        pointer.initialize(repeating: .nan, count: max(count, 1))
    }

    deinit { pointer.deallocate() }

    func view(_ range: Range<Int>) -> UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: pointer + range.lowerBound, count: range.count)
    }

    var all: UnsafeBufferPointer<Float> { view(0..<count) }

    func write(to url: URL) throws {
        try Data(bytes: pointer, count: count * MemoryLayout<Float>.stride).write(to: url)
    }
}

// MARK: - The inputs pack (conversion/timestral/device_pack.py)

/// One bucket's packed windows in one dtype: patches [N, 2, L, 64], cos / sin [N, S, 64], mask [N, 1, 1, S, S],
/// S = L + h, kept as mapped bytes; window n's graph inputs are made on demand.
struct Pack: @unchecked Sendable {
    let dir: URL
    let name: String
    let nCtx: Int
    let nHor: Int
    let dtype: DType
    let ids: [String]
    let manifestMD5: String
    let patches: Data
    let cos: Data
    let sin: Data
    let mask: Data
    let sanity: [Float]?

    var seq: Int { nCtx + nHor }
    var count: Int { ids.count }
    var horizonSteps: Int { nHor * 32 }

    static func load(inputs: URL, nCtx: Int, nHor: Int, dtype: DType) throws -> Pack {
        let name = "L\(nCtx)_h\(nHor)"
        let dir = inputs.appendingPathComponent(name)
        let manifestURL = dir.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else { throw GateError("no inputs pack at \(dir.path)") }
        let raw = try Data(contentsOf: manifestURL)
        guard let m = try JSONSerialization.jsonObject(with: raw) as? [String: Any] else { throw GateError("\(name)/manifest.json is not a JSON object") }
        guard m["n_ctx"] as? Int == nCtx, m["n_hor"] as? Int == nHor else {
            throw GateError("\(name)/manifest.json is for n_ctx \(String(describing: m["n_ctx"])) n_hor \(String(describing: m["n_hor"]))")
        }
        let windows = m["windows"] as? [[String: Any]] ?? []
        let ids = windows.compactMap { $0["id"] as? String }
        guard !ids.isEmpty, ids.count == windows.count else { throw GateError("\(name)/manifest.json lists no windows") }
        let seq = nCtx + nHor
        let scalarBytes = dtype == .f16 ? 2 : 4
        func file(_ key: String, perWindow: Int) throws -> Data {
            let url = dir.appendingPathComponent("\(key).\(dtype.rawValue).bin")
            let data = try Data(contentsOf: url, options: .alwaysMapped)
            guard data.count == perWindow * ids.count * scalarBytes else {
                throw GateError("\(name)/\(url.lastPathComponent): \(data.count) bytes, expected \(perWindow * ids.count * scalarBytes)")
            }
            return data
        }
        var sanity: [Float]?
        let sanityURL = dir.appendingPathComponent("sanity_ref.f32.bin")
        if let data = try? Data(contentsOf: sanityURL), data.count == 2 * nHor * 32 * 21 * 4 {
            sanity = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        return Pack(dir: dir, name: name, nCtx: nCtx, nHor: nHor, dtype: dtype, ids: ids, manifestMD5: md5Hex(raw),
                    patches: try file("patches", perWindow: 2 * nCtx * 64), cos: try file("cos", perWindow: seq * 64),
                    sin: try file("sin", perWindow: seq * 64), mask: try file("mask", perWindow: seq * seq), sanity: sanity)
    }

    private func slice(_ data: Data, offset: Int, count: Int, shape: [Int]) -> NDArray {
        switch dtype {
        case .f32:
            return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> NDArray in
                makeNDArray(UnsafeBufferPointer(rebasing: raw.bindMemory(to: Float.self)[offset..<(offset + count)]),
                            shape: shape, scalarType: .float32)
            }
        case .f16:
            return data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> NDArray in
                makeNDArray(UnsafeBufferPointer(rebasing: raw.bindMemory(to: Float16.self)[offset..<(offset + count)]),
                            shape: shape, scalarType: .float16)
            }
        }
    }

    /// Window n's calls: one feed with both rows for an R = 2 graph, two feeds (row 0, row 1) for an R = 1 graph.
    func feeds(window n: Int, rows: Int) throws -> [[String: NDArray]] {
        guard n >= 0, n < count else { throw GateError("window \(n) is outside the pack (\(count))") }
        let s = seq, l = nCtx
        let cos = slice(self.cos, offset: n * s * 64, count: s * 64, shape: [s, 64])
        let sin = slice(self.sin, offset: n * s * 64, count: s * 64, shape: [s, 64])
        let mask = slice(self.mask, offset: n * s * s, count: s * s, shape: [1, 1, s, s])
        switch rows {
        case 2:
            let p = slice(patches, offset: n * 2 * l * 64, count: 2 * l * 64, shape: [2, l, 64])
            return [["patches": p, "cos": cos, "sin": sin, "attn_mask": mask]]
        case 1:
            return (0..<2).map { r in
                ["patches": slice(patches, offset: (n * 2 + r) * l * 64, count: l * 64, shape: [1, l, 64]),
                 "cos": cos, "sin": sin, "attn_mask": mask]
            }
        default:
            throw GateError("a graph with R = \(rows) rows: only 1 or 2")
        }
    }

    func index(of id: String?) throws -> Int {
        guard let id else { return 0 }
        guard let n = ids.firstIndex(of: id) else { throw GateError("window '\(id)' is not in \(name)") }
        return n
    }
}

// MARK: - A loaded graph

struct Graph: @unchecked Sendable {
    let config: BundleConfig
    let model: AIModel                // retained for as long as the function is used
    let function: InferenceFunction
    let rows: Int
    let nCtx: Int
    let nHor: Int
    let dtype: DType
    let outType: String
    let loadMilliseconds: Double
    let record: [String: Any]

    var horizonSteps: Int { nHor * 32 }
    /// Floats of one window's output (both rows).
    var perWindow: Int { 2 * horizonSteps * 21 }

    /// Specializes (JIT) or opens (AOT) the bundle and checks its contract: patches [R, L, 64], cos / sin [S, 64],
    /// attn_mask [1, 1, S, S] in, quantiles [R, (S - L) * 32, 21] out, one input dtype.
    static func load(_ config: BundleConfig, cold: Bool, log: GateLog) async throws -> Graph {
        guard FileManager.default.fileExists(atPath: config.url.path) else { throw GateError("no bundle at \(config.url.path)") }
        #if os(macOS)
        // An .aimodelc compiled for another architecture (the phone's h19p) must never run on the Mac.
        let files = (try? FileManager.default.contentsOfDirectory(atPath: config.url.path)) ?? []
        if let foreign = files.first(where: { $0.hasPrefix("main-") && $0.hasSuffix(".mlirb") && $0 != "main-\(AIModel.deviceArchitectureName).mlirb" }) {
            throw GateError("\(config.url.lastPathComponent) holds \(foreign), not this Mac's \(AIModel.deviceArchitectureName): refused")
        }
        #endif
        let options = try GateOptions.specialization(config.compute)
        var deletedCache = false
        if cold {
            do {
                try AIModelCache.default.deleteEntry(for: config.url, options: options)
                deletedCache = true
            } catch {
                log.write("cold: no cache entry deleted (\(error))")
            }
        }
        log.write("loading \(config.label) …")
        let before = Probe.footprint()
        let cpu0 = Probe.cpuNanoseconds()
        let t0 = Probe.now()
        let model = try await AIModel(contentsOf: config.url, options: options)
        guard let function = try model.loadFunction(named: "main") else { throw GateError("\(config.url.lastPathComponent) has no function 'main'") }
        let loadMilliseconds = Double(Probe.now() - t0) / 1e6
        let loadCPU = Double(Probe.cpuNanoseconds() - cpu0) / 1e6
        let d = function.descriptor
        guard Set(d.inputNames) == ["patches", "cos", "sin", "attn_mask"], d.outputNames == ["quantiles"], d.stateNames.isEmpty,
              case .ndArray(let p) = d.inputDescriptor(of: "patches"),
              case .ndArray(let c) = d.inputDescriptor(of: "cos"),
              case .ndArray(let s) = d.inputDescriptor(of: "sin"),
              case .ndArray(let m) = d.inputDescriptor(of: "attn_mask"),
              case .ndArray(let q) = d.outputDescriptor(of: "quantiles") else {
            throw GateError("contract mismatch: inputs \(d.inputNames) outputs \(d.outputNames) states \(d.stateNames)")
        }
        guard p.shape.count == 3, p.shape[2] == 64, c.shape.count == 2, c.shape[1] == 64, s.shape == c.shape,
              m.shape == [1, 1, c.shape[0], c.shape[0]], c.shape[0] > p.shape[1],
              q.shape == [p.shape[0], (c.shape[0] - p.shape[1]) * 32, 21],
              p.scalarType == c.scalarType, p.scalarType == s.scalarType, p.scalarType == m.scalarType else {
            throw GateError("contract mismatch: patches \(p.shape) cos \(c.shape) sin \(s.shape) attn_mask \(m.shape) quantiles \(q.shape)")
        }
        let dtype = try DType(p.scalarType)
        _ = try DType(q.scalarType)
        let rows = p.shape[0], nCtx = p.shape[1], nHor = c.shape[0] - p.shape[1]
        let record: [String: Any] = [
            "bundle": config.url.lastPathComponent, "form": config.form, "label": config.label, "path_under_root": config.url.path,
            "compute": config.compute, "route": config.route,
            "specialization": "\(options)", "allowed_compute_units": options.allowedComputeUnitKinds.map { "\($0)" }.sorted(),
            "preferred_compute_unit": orNull(options.preferredComputeUnitKind.map { "\($0)" }),
            "io_dtype": dtype.rawValue, "output_dtype": "\(q.scalarType)", "rows": rows, "n_ctx": nCtx, "n_hor": nHor,
            "load_ms": round3(loadMilliseconds), "load_cpu_ms": round3(loadCPU), "cold_requested": cold,
            "cache_entry_deleted": deletedCache, "footprint_before_load_mb": mb(before),
            "footprint_after_load_mb": mb(Probe.footprint()), "available_after_load_mb": orNull(Probe.available().map(mb)),
            "thermal_after_load": Probe.thermal(),
        ]
        log.write(String(format: "loaded %@ (R%ld L%ld h%ld %@) in %.1f ms (CPU %.1f ms); footprint %.0f MB (was %.0f)",
                         config.label, rows, nCtx, nHor, dtype.rawValue, loadMilliseconds, loadCPU, mb(Probe.footprint()), mb(before)))
        return Graph(config: config, model: model, function: function, rows: rows, nCtx: nCtx, nHor: nHor, dtype: dtype,
                     outType: "\(q.scalarType)", loadMilliseconds: loadMilliseconds, record: record)
    }

    /// One window: every call (one for R = 2, two for R = 1), the outputs copied into `buffer` from `offset`
    /// (perWindow floats, row 0 then row 1). Returns the wall ms of each call (run + copy) and of the whole window.
    func forecast(_ feeds: [[String: NDArray]], into buffer: FloatBuffer, at offset: Int) async throws -> (calls: [Double], total: Double) {
        let per = perWindow / feeds.count
        guard offset >= 0, offset + perWindow <= buffer.count else { throw GateError("output buffer too small") }
        var calls: [Double] = []
        let t0 = Probe.now()
        for (r, feed) in feeds.enumerated() {
            let tc = Probe.now()
            var outputs = try await function.run(inputs: feed)
            guard let array = outputs.remove("quantiles")?.ndArray else { throw GateError("no quantiles output") }
            try copyOut(array, into: buffer.pointer + offset + r * per, count: per)
            calls.append(Double(Probe.now() - tc) / 1e6)
        }
        return (calls, Double(Probe.now() - t0) / 1e6)
    }
}

// MARK: - The gate

struct Gate {
    let options: GateOptions
    let log: GateLog

    func header() async -> [String: Any] {
        let power = await PowerState.read()
        let free = Probe.freeBytes(options.root)
        log.write("timestralgate mode=\(options.mode.rawValue) tag=\(options.tag) configs=\(options.configs.map(\.label)) window=\(options.window ?? "-") cold=\(options.cold)")
        log.write("device=\(Probe.machine) arch=\(AIModel.deviceArchitectureName) os=\(ProcessInfo.processInfo.operatingSystemVersionString) build=\(Probe.osBuild) thermal=\(Probe.thermal()) \(power.line) footprint=\(mb(Probe.footprint()))MB available=\(Probe.available().map { "\(mb($0))MB" } ?? "-") free_disk=\(free.map { String(format: "%.1fGB", Double($0) / 1e9) } ?? "-")")
        return [
            "schema": "timestralgate-run/1", "mode": options.mode.rawValue, "tag": options.tag, "note": orNull(options.note),
            "started": ISO8601DateFormatter().string(from: Date()),
            "device": ["machine": Probe.machine, "arch": AIModel.deviceArchitectureName,
                       "os": ProcessInfo.processInfo.operatingSystemVersionString, "build": Probe.osBuild,
                       "available_compute_units": ComputeUnitKind.availableKinds.map { "\($0)" }.sorted(),
                       "physical_memory_gb": (Double(ProcessInfo.processInfo.physicalMemory) / 1e9 * 100).rounded() / 100]
                as [String: Any],
            "coreai": Probe.coreAIVersion(),
            "thermal_start": Probe.thermal(), "power_start": power.json,
            "footprint_start_mb": mb(Probe.footprint()), "available_start_mb": orNull(Probe.available().map(mb)),
            "free_disk_gb": orNull(free.map { (Double($0) / 1e9 * 10).rounded() / 10 }),
            "args": ["configs": options.configs.map(\.label), "window": orNull(options.window), "warm": options.warm,
                     "calls": options.calls, "rounds": options.rounds, "passes": options.passes, "budget_s": options.budget,
                     "cold": options.cold, "cool_cap_s": options.coolCap, "inputs": options.inputs.path] as [String: Any],
        ]
    }

    /// Runs the mode; writes run.json (with `error` on failure) and returns the DONE line's text.
    func run() async throws -> String {
        var record = await header()
        let sampler = Sampler()
        sampler.begin()
        do {
            let done: String
            switch options.mode {
            case .gate: done = try await gate(&record)
            case .timing: done = try await timing(&record)
            case .warm: done = try await warmUp(&record)
            case .probe: done = try probe(&record)
            case .thermal: done = try await thermalWait(&record)
            }
            record.merge(sampler.end()) { $1 }
            record["power_end"] = await PowerState.read().json
            record["thermal_end"] = Probe.thermal()
            record["finished"] = ISO8601DateFormatter().string(from: Date())
            try JSON.write(record, to: options.out.appendingPathComponent("run.json"))
            return done
        } catch {
            record.merge(sampler.end()) { $1 }
            record["error"] = "\(error)"
            record["error_reflecting"] = String(reflecting: error)
            record["thermal_end"] = Probe.thermal()
            record["finished"] = ISO8601DateFormatter().string(from: Date())
            try? JSON.write(record, to: options.out.appendingPathComponent("run.json"))
            throw error
        }
    }

    // MARK: gate

    private func gate(_ record: inout [String: Any]) async throws -> String {
        let g = try await Graph.load(options.configs[0], cold: options.cold, log: log)
        record["bundle"] = g.record
        let pack = try Pack.load(inputs: options.inputs, nCtx: g.nCtx, nHor: g.nHor, dtype: g.dtype)
        let per = g.perWindow
        log.write("gate: \(pack.name) \(pack.count) windows x \(options.passes) passes, R\(g.rows) = \(g.rows == 2 ? "one call" : "two calls") per window, \(g.dtype.rawValue) inputs")
        var passes: [[String: Any]] = []
        var first: FloatBuffer?
        var firstCall: Double?
        var sanity: [String: Any] = [:]
        for pass in 1...options.passes {
            let dir = options.out.appendingPathComponent("pass\(pass)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let out = FloatBuffer(count: pack.count * per)
            var callMs: [Double] = [], windowMs: [Double] = []
            let cpu0 = Probe.cpuNanoseconds()
            let wall0 = Probe.now()
            for n in 0..<pack.count {
                let feeds = try pack.feeds(window: n, rows: g.rows)
                let (calls, total) = try await g.forecast(feeds, into: out, at: n * per)
                if firstCall == nil { firstCall = calls.first }
                callMs += calls
                windowMs.append(total)
            }
            let wall = Double(Probe.now() - wall0) / 1e9
            let cpuPerCall = Double(Probe.cpuNanoseconds() - cpu0) / 1e6 / Double(max(callMs.count, 1))
            let file = "pass\(pass)/quantiles.f32.bin"
            try out.write(to: options.out.appendingPathComponent(file))
            let bad = nonFinite(out.all)
            var row: [String: Any] = [
                "pass": pass, "file": file, "calls": callMs.count, "call_ms": callMs.map(round3), "window_ms": windowMs.map(round3),
                "median_call_ms": round3(Stats.median(callMs)), "p90_call_ms": round3(Stats.percentile(callMs, 0.9)),
                "median_window_ms": round3(Stats.median(windowMs)), "cpu_ms_per_call": round3(cpuPerCall), "wall_s": round3(wall),
                "nonfinite": bad, "thermal_end": Probe.thermal(), "footprint_end_mb": mb(Probe.footprint()),
            ]
            if let first {
                let c = compareBits(first.all, out.all)
                row["identical_to_pass1"] = c.identical
                row["different_vs_pass1"] = c.different
                row["max_abs_vs_pass1"] = c.maxAbs
            } else {
                first = out
                if let ref = pack.sanity {
                    let w0 = out.view(0..<per)
                    var worst = 0.0
                    for i in 0..<per { worst = max(worst, Double(abs(w0[i] - ref[i]))) }
                    sanity = ["window": pack.ids[0], "max_abs_vs_released_fp32": worst, "nonfinite": nonFinite(w0)]
                    log.write(String(format: "sanity %@: max|d| vs the released fp32 %.3g, non-finite %ld", pack.ids[0], worst, nonFinite(w0)))
                }
            }
            log.write(String(format: "pass %ld/%ld: %ld windows, call median %.2f ms p90 %.2f ms, window median %.2f ms, CPU %.2f ms/call, %.1f s, non-finite %ld%@, thermal %@",
                             pass, options.passes, pack.count, Stats.median(callMs), Stats.percentile(callMs, 0.9), Stats.median(windowMs),
                             cpuPerCall, wall, bad,
                             pass == 1 ? "" : ((row["identical_to_pass1"] as? Bool ?? false) ? ", identical to pass 1" : ", DIFFERS from pass 1"),
                             Probe.thermal()))
            passes.append(row)
        }
        record["output"] = [
            "bucket": pack.name, "windows": pack.count, "window_ids": pack.ids, "horizon_steps": g.horizonSteps, "quantiles": 21,
            "rows_per_window": 2, "format": "float32 little-endian [windows, 2, horizon_steps, 21], rows = [series, mirror]",
            "passes": passes, "inputs_manifest_md5": pack.manifestMD5, "inputs_dtype": pack.dtype.rawValue,
        ] as [String: Any]
        record["sanity"] = sanity
        record["first_call_ms"] = orNull(firstCall.map(round3))
        let last = passes.last?["median_call_ms"] as? Double ?? .nan
        let same = passes.dropFirst().allSatisfy { $0["identical_to_pass1"] as? Bool ?? false }
        return String(format: "%@ %@ R%ld %@: %ld windows x %ld passes (identical %@); load %.0f ms, first call %.1f ms, median call (last pass) %.2f ms; sanity max|d| %.3g; peak %.0f MB",
                      g.config.form, g.config.route, g.rows, pack.name, pack.count, options.passes, same ? "yes" : "NO",
                      g.loadMilliseconds, firstCall ?? .nan, last, sanity["max_abs_vs_released_fp32"] as? Double ?? .nan,
                      mb(Probe.footprint()))
    }

    // MARK: timing

    private func timing(_ record: inout [String: Any]) async throws -> String {
        struct Arm {
            let graph: Graph
            let pack: Pack
            let window: Int
            let feeds: [[String: NDArray]]
            let coldOut: FloatBuffer
            var row: [String: Any]
            var rounds: [[String: Any]] = []
            var kept: [Double] = []        // the round medians of rounds 1...
            var keptCalls: [Double] = []   // every timed forecast of rounds 1...
            var drift = 0
        }
        var packs: [String: Pack] = [:]
        var arms: [Arm] = []
        var failed: [[String: Any]] = []
        for c in options.configs {
            // A config that cannot load or run (an AOT asset the device refuses) is recorded and left out; the rest are timed.
            let g: Graph
            let pack: Pack
            let n: Int
            let feeds: [[String: NDArray]]
            let out: FloatBuffer
            let calls: [Double], total: Double
            do {
                g = try await Graph.load(c, cold: options.cold, log: log)
                let key = "L\(g.nCtx)_h\(g.nHor).\(g.dtype.rawValue)"
                pack = try packs[key] ?? Pack.load(inputs: options.inputs, nCtx: g.nCtx, nHor: g.nHor, dtype: g.dtype)
                packs[key] = pack
                n = try pack.index(of: options.window)
                feeds = try pack.feeds(window: n, rows: g.rows)
                out = FloatBuffer(count: g.perWindow)
                let first = try await g.forecast(feeds, into: out, at: 0)
                calls = first.calls
                total = first.total
            } catch {
                failed.append(["label": c.label, "error": "\(error)", "error_reflecting": String(reflecting: error)])
                log.write("FAILED \(c.label): \(error) (left out of the rounds)")
                continue
            }
            var row = g.record
            row["window"] = pack.ids[n]
            row["bucket"] = pack.name
            row["cold_forecast_ms"] = round3(total)
            row["cold_call_ms"] = calls.map(round3)
            row["cold_output_nonfinite"] = nonFinite(out.all)
            log.write(String(format: "cold %@: load %.1f ms, first forecast %.2f ms (%@)", c.label, g.loadMilliseconds, total,
                             calls.map { String(format: "%.2f", $0) }.joined(separator: " + ")))
            arms.append(Arm(graph: g, pack: pack, window: n, feeds: feeds, coldOut: out, row: row))
        }
        guard !arms.isEmpty else { throw GateError("no config loaded: \(failed.map { "\($0["label"] ?? "?"): \($0["error"] ?? "?")" })") }
        log.write("timing: \(arms.count) configs, \(options.rounds) rounds x (\(options.warm) warm-up + \(options.calls) timed forecasts), budget \(options.budget) s; footprint \(mb(Probe.footprint())) MB")
        let scratch = FloatBuffer(count: arms.map(\.graph.perWindow).max() ?? 0)
        let start = Probe.now()
        var cooledTotal = 0.0
        var truncated: String?
        var completed = 0
        rounds: for round in 0..<options.rounds {
            let order = round % 2 == 0 ? Array(arms.indices) : Array(arms.indices.reversed())
            for i in order {
                let busy = Double(Probe.now() - start) / 1e9 - cooledTotal
                if busy >= options.budget {
                    truncated = String(format: "budget %.0f s reached before round %ld %@", options.budget, round, arms[i].graph.config.label)
                    break rounds
                }
                var waited = 0.0
                let before = Probe.thermal()
                if Probe.hot {
                    let t0 = Date()
                    log.write("thermal \(before) before round \(round) \(arms[i].graph.config.label): waiting for fair or nominal (cap \(Int(options.coolCap)) s)")
                    while Probe.hot && Date().timeIntervalSince(t0) < options.coolCap {
                        try await Task.sleep(for: .seconds(5))
                    }
                    waited = Date().timeIntervalSince(t0)
                    cooledTotal += waited
                }
                let arm = arms[i]
                var warmMs: [Double] = [], ms: [Double] = [], callMs: [[Double]] = []
                var drift = 0
                let roundStart = Double(Probe.now() - start) / 1e9
                for _ in 0..<options.warm {
                    warmMs.append(try await arm.graph.forecast(arm.feeds, into: scratch, at: 0).total)
                }
                for _ in 0..<options.calls {
                    let r = try await arm.graph.forecast(arm.feeds, into: scratch, at: 0)
                    ms.append(r.total)
                    callMs.append(r.calls)
                    if !compareBits(arm.coldOut.all, scratch.view(0..<arm.graph.perWindow)).identical { drift += 1 }
                }
                let median = Stats.median(ms)
                let power = await PowerState.read()
                arms[i].rounds.append([
                    "round": round, "t_start_s": round3(roundStart), "warm_ms": warmMs.map(round3), "ms": ms.map(round3),
                    "call_ms": callMs.map { $0.map(round3) }, "median_ms": round3(median), "drift": drift,
                    "thermal_before": before, "thermal_after": Probe.thermal(), "cooled_s": round3(waited), "power": power.json,
                    "footprint_mb": mb(Probe.footprint()), "available_mb": orNull(Probe.available().map(mb)),
                ])
                arms[i].drift += drift
                if round >= 1 {
                    arms[i].kept.append(median)
                    arms[i].keptCalls += ms
                }
                log.write(String(format: "round %ld %@: median %.2f ms [%.2f-%.2f] (warm-up %@), drift %ld, thermal %@ -> %@, %@%@",
                                 round, arm.graph.config.label, median, ms.min() ?? .nan, ms.max() ?? .nan,
                                 warmMs.map { String(format: "%.2f", $0) }.joined(separator: " "), drift, before, Probe.thermal(),
                                 power.line, waited > 0 ? String(format: ", cooled %.0f s", waited) : ""))
            }
            completed = round + 1
        }
        let loopS = Double(Probe.now() - start) / 1e9
        var summaries: [[String: Any]] = []
        var lines: [String] = []
        for arm in arms {
            var s = arm.row
            s["rounds"] = arm.rounds
            s["rounds_kept"] = arm.kept.count
            s["warm_median_ms"] = arm.kept.isEmpty ? NSNull() : round3(Stats.median(arm.kept)) as Any
            s["warm_round_range_ms"] = arm.kept.isEmpty ? NSNull() : [round3(arm.kept.min()!), round3(arm.kept.max()!)] as Any
            s["warm_forecast_range_ms"] = arm.keptCalls.isEmpty ? NSNull() : [round3(arm.keptCalls.min()!), round3(arm.keptCalls.max()!)] as Any
            s["drift_forecasts"] = arm.drift
            summaries.append(s)
            let cold = arm.row["cold_forecast_ms"] as? Double ?? .nan
            lines.append(String(format: "%@ cold load %.0f ms + %.2f ms, warm %.2f ms [%.2f-%.2f] (%ld rounds)", arm.graph.config.label,
                                arm.graph.loadMilliseconds, cold, arm.kept.isEmpty ? .nan : Stats.median(arm.kept),
                                arm.kept.min() ?? .nan, arm.kept.max() ?? .nan, arm.kept.count))
        }
        record["timing"] = [
            "order": arms.map(\.graph.config.label), "configs": summaries, "failed": failed, "rounds_completed": completed,
            "loop_s": round3(loopS), "cooled_s": round3(cooledTotal), "truncated": orNull(truncated),
            "note": "one process: every bundle loaded once (cold = that load + its first forecast), then rounds A/B/C... (odd rounds reversed); a forecast = one R = 2 call or two R = 1 calls, each with its output copied out; warm = rounds 1.., the round value = the median of its timed forecasts",
        ] as [String: Any]
        if let truncated { log.write("timing stopped early: \(truncated)") }
        let left = failed.isEmpty ? "" : "; FAILED " + failed.map { "\($0["label"] ?? "?")" }.joined(separator: ", ")
        return "timing: " + lines.joined(separator: "; ") + String(format: "; loop %.1f s", loopS) + left
    }

    // MARK: warm

    private func warmUp(_ record: inout [String: Any]) async throws -> String {
        var rows: [[String: Any]] = []
        var lines: [String] = []
        for c in options.configs {
            let g = try await Graph.load(c, cold: options.cold, log: log)
            let pack = try Pack.load(inputs: options.inputs, nCtx: g.nCtx, nHor: g.nHor, dtype: g.dtype)
            let n = try pack.index(of: options.window)
            let feeds = try pack.feeds(window: n, rows: g.rows)
            let out = FloatBuffer(count: g.perWindow)
            let (calls, total) = try await g.forecast(feeds, into: out, at: 0)
            var r = g.record
            r["window"] = pack.ids[n]
            r["first_forecast_ms"] = round3(total)
            r["first_call_ms"] = calls.map(round3)
            r["output_nonfinite"] = nonFinite(out.all)
            rows.append(r)
            lines.append(String(format: "%@ load %.0f ms first forecast %.1f ms", c.label, g.loadMilliseconds, total))
            log.write(lines.last!)
        }
        record["warm"] = rows
        return "warm: " + lines.joined(separator: "; ")
    }

    // MARK: probe

    private func probe(_ record: inout [String: Any]) throws -> String {
        let fm = FileManager.default
        func size(_ url: URL) -> Int64 {
            var total: Int64 = 0
            if let e = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
                for case let f as URL in e {
                    total += Int64((try? f.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
                }
            }
            return total
        }
        func list(_ dir: URL) -> [[String: Any]] {
            let names = ((try? fm.contentsOfDirectory(atPath: dir.path)) ?? []).sorted()
            return names.map { ["name": $0, "mb": mb(size(dir.appendingPathComponent($0)))] }
        }
        let models = list(options.root.appendingPathComponent("models"))
        let inputs = list(options.inputs)
        record["models"] = models
        record["inputs"] = inputs
        log.write("models: \(models.map { "\($0["name"] ?? "")" }.joined(separator: ", "))")
        log.write("inputs: \(inputs.map { "\($0["name"] ?? "")" }.joined(separator: ", "))")
        return "probe ok: \(models.count) model entries, \(inputs.count) input packs, thermal \(Probe.thermal())"
    }

    // MARK: thermal

    private func thermalWait(_ record: inout [String: Any]) async throws -> String {
        let t0 = Date()
        var states: [[String: Any]] = []
        while true {
            let state = Probe.thermal()
            let power = await PowerState.read()
            states.append(["t_s": round3(Date().timeIntervalSince(t0)), "state": state, "power": power.json])
            log.write("thermal \(state) \(power.line) after \(Int(Date().timeIntervalSince(t0))) s")
            if state == "nominal" || Date().timeIntervalSince(t0) >= options.cap { break }
            try await Task.sleep(for: .seconds(10))
        }
        record["thermal_wait"] = states
        return "thermal=\(Probe.thermal()) after \(Int(Date().timeIntervalSince(t0))) s"
    }
}
