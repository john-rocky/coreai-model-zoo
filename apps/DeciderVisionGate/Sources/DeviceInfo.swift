// DeviceInfo — what the gate records about the machine: model identifier (utsname.machine), OS version and build,
// Core AI architecture, memory, thermal state, low-power mode, battery (level and power source: the 18 Pro sits on USB,
// so "charging" / "full" = USB power), the app's memory footprint and headroom, free space, a directory's size on disk,
// and what the Core AI specialization cache holds for this app. MemorySampler reads the footprint every 100 ms while a
// load or a run of decisions goes on, keeps the readings in its record and appends every one of them to memory.tsv as it
// goes, so a process the system kills still leaves its series behind (FunASRGate's DeviceInfo, for this gate).

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

/// memory.tsv in the output directory: one line per sampler reading, from every sampler of the run, written as it is
/// taken (a plain write(2): a process that is killed keeps what it wrote). Columns: t_app_s, label, t_s (in the
/// sampler), footprint_mb, available_mb, thermal.
final class MemoryLog: @unchecked Sendable {
    private let lock = NSLock()
    private var handle: FileHandle?
    private let t0: ContinuousClock.Instant

    init(url: URL, t0: ContinuousClock.Instant) {
        self.t0 = t0
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: Data("t_app_s\tlabel\tt_s\tfootprint_mb\tavailable_mb\tthermal\n".utf8))
        }
        handle = try? FileHandle(forWritingTo: url)
        _ = try? handle?.seekToEnd()
    }

    func write(label: String, t: Double, footprint: Double, available: Double, thermal: String) {
        let s = String(format: "%.2f\t%@\t%.2f\t%.1f\t%.1f\t%@\n", seconds(since: t0), label, t, footprint, available, thermal)
        lock.withLock { try? handle?.write(contentsOf: Data(s.utf8)) }
    }

    func close() { lock.withLock { try? handle?.close(); handle = nil } }
}

/// The app's memory while a load or a run of decisions goes on: the footprint (phys_footprint, what jetsam counts) and
/// os_proc_available_memory every `interval` seconds, on a thread of its own so that a load holding its thread cannot
/// stall the readings. The record keeps every reading of the first `fullSeconds` ([t s, footprint MB, available MB])
/// and every 10th after that; every reading also goes to memory.tsv. Every 10 s it logs a progress line; with
/// `timelineEvery`, it also records [t s, thermal state, footprint MB, available MB] at that period. start() takes
/// the first reading, stop() the last one and returns the record.
final class MemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private let done = DispatchSemaphore(value: 0)
    private let label: String
    private let interval: Double
    private let fullSeconds: Double
    private let timelineEvery: Double?
    private let memoryLog: MemoryLog?
    private let progress: @Sendable (String) -> Void
    private var running = false
    private var t0 = ContinuousClock.now
    private var n = 0
    private var peak = -1.0, peakAt = 0.0
    private var minAvailable = -1.0, minAvailableAt = 0.0
    private var firstFootprint = -1.0, lastFootprint = -1.0
    private var firstAvailable = -1.0, lastAvailable = -1.0
    private var series: [[Double]] = []
    private var timeline: [[Any]] = []
    private var nextTimeline = 0.0

    init(_ label: String, interval: Double = 0.1, fullSeconds: Double = 600, timelineEvery: Double? = nil,
         memoryLog: MemoryLog?, progress: @escaping @Sendable (String) -> Void) {
        self.label = label
        self.interval = interval
        self.fullSeconds = fullSeconds
        self.timelineEvery = timelineEvery
        self.memoryLog = memoryLog
        self.progress = progress
    }

    func start() {
        lock.withLock {
            running = true
            t0 = .now
        }
        sample()
        let thread = Thread { [self] in
            while lock.withLock({ running }) {
                Thread.sleep(forTimeInterval: interval)
                sample()
            }
            done.signal()
        }
        thread.name = "dv.memory-sampler"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func stop() -> [String: Any] {
        lock.withLock { running = false }
        done.wait()
        sample(final: true)
        return lock.withLock {
            var r: [String: Any] = ["label": label, "samples": n, "interval_s": interval, "duration_s": seconds(since: t0),
                                    "peak_footprint_mb": peak, "peak_at_s": peakAt,
                                    "footprint_mb_start": firstFootprint, "footprint_mb_end": lastFootprint,
                                    "min_available_mb": minAvailable, "min_available_at_s": minAvailableAt,
                                    "available_mb_start": firstAvailable, "available_mb_end": lastAvailable,
                                    "series_full_s": fullSeconds, "series": series,
                                    "series_columns": ["t_s", "footprint_mb", "available_mb"]]
            if timelineEvery != nil { r["timeline"] = timeline }
            return r
        }
    }

    private func sample(final: Bool = false) {
        let fp = DeviceInfo.footprintMB(), av = DeviceInfo.availableMB(), th = DeviceInfo.thermal()
        let (note, t): (String?, Double) = lock.withLock {
            let t = seconds(since: t0)
            if n == 0 {
                firstFootprint = fp
                firstAvailable = av
            }
            n += 1
            lastFootprint = fp
            lastAvailable = av
            if fp > peak {
                peak = fp
                peakAt = t
            }
            if av >= 0 && (minAvailable < 0 || av < minAvailable) {
                minAvailable = av
                minAvailableAt = t
            }
            if t <= fullSeconds || n % 10 == 1 || final { series.append([(t * 100).rounded() / 100, fp, av]) }
            if let every = timelineEvery, t >= nextTimeline || final {
                timeline.append([(t * 10).rounded() / 10, th, fp, av])
                nextTimeline += every
                while nextTimeline <= t { nextTimeline += every }
            }
            guard n % 100 == 0 else { return (nil, t) }
            return ("\(label): \(String(format: "%.0f", t)) s, footprint \(String(format: "%.0f", fp)) MB (peak "
                + "\(String(format: "%.0f", peak))), available \(String(format: "%.0f", av)) MB, thermal \(th)", t)
        }
        memoryLog?.write(label: label, t: t, footprint: fp, available: av, thermal: th)
        if let s = note { progress(s) }
    }
}
