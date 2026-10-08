// DeviceInfo — what the gate records about the machine: model identifier (utsname.machine), OS version and build,
// Core AI architecture, memory, thermal state, low-power mode, battery (level and power source: the 18 Pro sits on USB,
// so "charging" / "full" = USB power), the app's memory footprint and headroom, free space, a directory's size on disk,
// what the Core AI specialization cache holds for this app, and the Neural Engine regions a compiled asset holds.
// MemoryMonitor reads the footprint every 100 ms for the app's whole life and appends every reading to memory.tsv as
// it goes (a killed process leaves its series; a file that stops growing while the process lives is a frozen app).
// The DeviceInfo enum is apps/KevGate/Sources/DeviceInfo.swift's, unchanged; MemoryMonitor replaces its per-section
// sampler (one series, labelled with the stage or section that runs, statistics per label).

import CoreAI
import Darwin
import Foundation
#if os(iOS)
import os
import UIKit
#endif

enum DeviceInfo {
    static func snapshot() -> [String: Any] {
        let p = ProcessInfo.processInfo
        let v = p.operatingSystemVersion
        #if os(iOS)
        let os = "iOS"
        #else
        let os = "macOS"
        #endif
        return ["machine": machine(), "hw_model": sysctlString("hw.model"), "cpu": sysctlString("machdep.cpu.brand_string"),
                "os": "\(os) \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)", "os_build": sysctlString("kern.osversion"),
                "os_version_string": p.operatingSystemVersionString,
                "physical_memory_gb": Double(p.physicalMemory) / 1_073_741_824, "processor_count": p.processorCount,
                "coreai_architecture": AIModel.deviceArchitectureName, "thermal": thermal(),
                "low_power_mode": p.isLowPowerModeEnabled]
    }

    /// utsname.machine, e.g. "iPhone19,2" (on a Mac, "arm64").
    static func machine() -> String {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: &u.machine) { raw in
            String(decoding: raw.prefix { $0 != 0 }, as: UTF8.self)
        }
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

    /// The app's physical footprint (what jetsam counts), in MB.
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : -1
    }

    /// Battery level (0...1; -1 unknown) and state (unplugged / charging / full / unknown). On the iPhone, "charging" or
    /// "full" means the phone is on external (USB) power.
    @MainActor static func battery() -> (level: Double, state: String) {
        #if os(iOS)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let state: String
        switch UIDevice.current.batteryState {
        case .unplugged: state = "unplugged"
        case .charging: state = "charging"
        case .full: state = "full"
        default: state = "unknown"
        }
        return (Double(UIDevice.current.batteryLevel), state)
        #else
        return (-1, "unknown")
        #endif
    }

    /// "USB power" when the battery reports charging or full, "battery" when unplugged.
    static func powerSource(_ state: String) -> String {
        switch state {
        case "charging", "full": return "USB power"
        case "unplugged": return "battery"
        default: return "unknown"
        }
    }

    /// os_proc_available_memory(): what the app can still allocate before its memory limit (jetsam), in MB
    /// (-1 on macOS, where the call does not exist).
    static func availableMB() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory()) / 1e6
        #else
        return -1
        #endif
    }

    /// This app's half of Core AI's specialization cache. iPhone: the container's Library/Caches/coreai-cache
    /// (<OS build>/<bundle id>/<hash>/<hash>/; the system half is out of the app's reach). Mac: the cache is shared by
    /// every process of the user (~/Library/Caches/coreai-cache/<OS build>/<bundle id or tool name>/), so only this
    /// app's directory is walked.
    static func coreAICacheDir() -> URL {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("coreai-cache")
        #if os(iOS)
        return caches
        #else
        return caches.appendingPathComponent(sysctlString("kern.osversion"))
            .appendingPathComponent(Bundle.main.bundleIdentifier ?? ProcessInfo.processInfo.processName)
        #endif
    }

    static func storageSnapshot() -> [String: Any] {
        var s: [String: Any] = ["coreai_cache": dirSnapshot(coreAICacheDir(), depth: 4)]
        #if os(iOS)
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        s["caches"] = dirSnapshot(caches, depth: 2)
        s["tmp"] = dirSnapshot(URL(fileURLWithPath: NSTemporaryDirectory()), depth: 2)
        #endif
        s["free_gb"] = freeGB(coreAICacheDir().deletingLastPathComponent())
        return s
    }

    /// Bytes and regular files under a directory, in total and per subdirectory down to `depth` levels
    /// (paths relative to the directory, at most `maxEntries`, largest first).
    static func dirSnapshot(_ root: URL, depth: Int, maxEntries: Int = 40) -> [String: Any] {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue,
              let e = FileManager.default.enumerator(atPath: root.path) else {
            return ["path": root.path, "exists": false, "bytes": 0, "files": 0]
        }
        var bytes = 0, files = 0
        var perDir: [String: (bytes: Int, files: Int)] = [:]
        while let rel = e.nextObject() as? String {
            guard let attrs = e.fileAttributes, attrs[.type] as? FileAttributeType == .typeRegular else { continue }
            let size = (attrs[.size] as? NSNumber)?.intValue ?? 0
            bytes += size
            files += 1
            let parts = rel.split(separator: "/")
            for d in stride(from: 1, through: min(depth, parts.count - 1), by: 1) {
                let key = parts[0..<d].joined(separator: "/")
                perDir[key, default: (0, 0)].bytes += size
                perDir[key, default: (0, 0)].files += 1
            }
        }
        let entries = perDir.sorted { $0.value.bytes != $1.value.bytes ? $0.value.bytes > $1.value.bytes : $0.key < $1.key }
            .prefix(maxEntries).map { ["dir": $0.key, "bytes": $0.value.bytes, "files": $0.value.files] as [String: Any] }
        return ["path": root.path, "exists": true, "bytes": bytes, "files": files, "dirs": entries,
                "dirs_total": perDir.count]
    }

    /// Free space for important data on the volume holding `url`, in GB (-1 when unknown).
    static func freeGB(_ url: URL) -> Double {
        let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage.map { Double($0) / 1e9 } ?? -1
    }

    /// Bytes and regular files under a directory (a bundle's size as sideloaded).
    static func tree(_ url: URL) -> (bytes: Int, files: Int) {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) else {
            return (0, 0)
        }
        var bytes = 0, files = 0
        for case let f as URL in e {
            guard let v = try? f.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]), v.isRegularFile == true else { continue }
            bytes += v.fileSize ?? 0
            files += 1
        }
        return (bytes, files)
    }
}

extension DeviceInfo {
    /// The Neural Engine regions a compiled asset holds: the distinct names before the first "." of every file or
    /// directory whose name contains "_ANE_region_" (a plain find lists each region more than once: lane memory
    /// reference_coreai_ane_requires_palettization). A GPU compile has none.
    static func aneRegions(_ asset: URL) -> (regions: Int, entries: Int) {
        guard let e = FileManager.default.enumerator(atPath: asset.path) else { return (0, 0) }
        var names = Set<String>()
        var entries = 0
        while let rel = e.nextObject() as? String {
            let name = (rel as NSString).lastPathComponent
            guard name.contains("_ANE_region_") else { continue }
            entries += 1
            names.insert(String(name.split(separator: ".", maxSplits: 1).first ?? Substring(name)))
        }
        return (names.count, entries)
    }

    /// The bytes of a file or of every regular file under a directory.
    static func bytes(_ url: URL) -> Int {
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else { return 0 }
        if !isDir.boolValue { return (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0 }
        return tree(url).bytes
    }
}

/// memory.tsv in the output directory, one line every `interval` seconds from launch to exit, written as taken (a plain
/// write(2): a process that is killed keeps what it wrote, and ../_run.sh reads a file that stops growing while the
/// process lives as a frozen app). Columns: t_app_s, label (the stage or section running), footprint_mb, available_mb,
/// thermal, battery. Per label: the readings, the peak footprint (and when), the least available memory (and when),
/// the first and last footprint.
final class MemoryMonitor: @unchecked Sendable {
    private struct Stats {
        var samples = 0
        var t0 = 0.0, t1 = 0.0
        var peak = -1.0, peakAt = 0.0
        var minAvailable = -1.0, minAvailableAt = 0.0
        var first = -1.0, last = -1.0
    }

    private let lock = NSLock()
    private let done = DispatchSemaphore(value: 0)
    private let t0: ContinuousClock.Instant
    private let interval: Double
    private var handle: FileHandle?
    private var label = "launch"
    private var stats: [String: Stats] = [:]
    private var running = false
    private(set) var lines = 0

    init(url: URL, t0: ContinuousClock.Instant, interval: Double = 0.1) {
        self.t0 = t0
        self.interval = interval
        FileManager.default.createFile(
            atPath: url.path, contents: Data("t_app_s\tlabel\tfootprint_mb\tavailable_mb\tthermal\tbattery\n".utf8))
        handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
    }

    func start() {
        lock.withLock { running = true }
        let thread = Thread { [self] in
            while lock.withLock({ running }) {
                sample()
                Thread.sleep(forTimeInterval: interval)
            }
            done.signal()
        }
        thread.name = "d1o.memory-monitor"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func stop() {
        let was = lock.withLock { () -> Bool in
            let r = running
            running = false
            return r
        }
        if was { done.wait() }
        sample()
        lock.withLock {
            try? handle?.close()
            handle = nil
        }
    }

    /// The label of the readings from now on (a stage, or a section inside one: "parity_jit load L256").
    func set(_ newLabel: String) { lock.withLock { label = newLabel } }

    var current: String { lock.withLock { label } }

    /// The statistics of one label so far.
    func summary(_ of: String) -> [String: Any] {
        lock.withLock {
            guard let s = stats[of] else { return ["label": of, "samples": 0] }
            return ["label": of, "samples": s.samples, "from_s": s.t0, "to_s": s.t1, "peak_footprint_mb": s.peak,
                    "peak_at_s": s.peakAt, "min_available_mb": s.minAvailable, "min_available_at_s": s.minAvailableAt,
                    "footprint_mb_first": s.first, "footprint_mb_last": s.last, "interval_s": interval]
        }
    }

    private func sample() {
        let fp = DeviceInfo.footprintMB(), av = DeviceInfo.availableMB(), th = DeviceInfo.thermal()
        let battery = BatteryCache.shared.text
        let t = seconds(since: t0)
        let line: String = lock.withLock {
            var s = stats[label] ?? Stats()
            if s.samples == 0 {
                s.t0 = t
                s.first = fp
            }
            s.samples += 1
            s.t1 = t
            s.last = fp
            if fp > s.peak {
                s.peak = fp
                s.peakAt = t
            }
            if av >= 0 && (s.minAvailable < 0 || av < s.minAvailable) {
                s.minAvailable = av
                s.minAvailableAt = t
            }
            stats[label] = s
            lines += 1
            return String(format: "%.2f\t%@\t%.1f\t%.1f\t%@\t%@\n", t, label, fp, av, th, battery)
        }
        lock.withLock { try? handle?.write(contentsOf: Data(line.utf8)) }
    }
}
