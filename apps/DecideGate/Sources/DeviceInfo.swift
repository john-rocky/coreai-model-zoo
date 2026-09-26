// DeviceInfo — what the gate records about the phone: model identifier (utsname.machine), OS version and
// build, memory, thermal state, low-power mode, the app's memory footprint and headroom, free space, a bundle's
// size on disk, and what the app's container holds in its caches (the Core AI specialization cache among them).
// MemorySampler reads the footprint every 100 ms while a load or a call runs.

import Darwin
import Foundation
import DecideGateSupport
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
        let os = "macOS"                                     // a Mac dry run of the gate code
        #endif
        return ["machine": machine(), "hw_model": sysctlString("hw.model"),
                "os": "\(os) \(v.majorVersion).\(v.minorVersion).\(v.patchVersion)", "os_build": sysctlString("kern.osversion"),
                "os_version_string": p.operatingSystemVersionString,
                "physical_memory_gb": Double(p.physicalMemory) / 1_073_741_824, "processor_count": p.processorCount,
                "thermal": thermal(), "low_power_mode": p.isLowPowerModeEnabled]
    }

    /// utsname.machine, e.g. "iPhone18,1".
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

    /// Battery level (0...1; -1 unknown) and state (unplugged / charging / full / unknown): the phone charges on USB,
    /// and charging heats it.
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

    /// os_proc_available_memory(): what the app can still allocate before its memory limit (jetsam), in MB
    /// (-1 where the call does not exist).
    static func availableMB() -> Double {
        #if os(iOS)
        return Double(os_proc_available_memory()) / 1e6
        #else
        return -1
        #endif
    }

    /// The app container's caches after a load: Library/Caches/coreai-cache (the container half of Core AI's
    /// specialization cache, <OS build>/<bundle id>/<hash>/<hash>/; the system half is out of the app's reach),
    /// Library/Caches as a whole, and tmp.
    static func storageSnapshot() -> [String: Any] {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return ["coreai_cache": dirSnapshot(caches.appendingPathComponent("coreai-cache"), depth: 4),
                "caches": dirSnapshot(caches, depth: 2),
                "tmp": dirSnapshot(URL(fileURLWithPath: NSTemporaryDirectory()), depth: 2)]
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

/// The app's memory while a load or a call runs: the footprint (phys_footprint, what jetsam counts) and
/// os_proc_available_memory every `interval` seconds, on a thread of its own so that a load holding its thread
/// cannot stall the readings. Every 10 s it logs a progress line: a load that is killed leaves its last reading in
/// result.log. start() takes the first reading, stop() the last one and returns the record.
final class MemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private let done = DispatchSemaphore(value: 0)
    private let label: String
    private let interval: Double
    private let progress: @Sendable (String) -> Void
    private var running = false
    private var t0 = ContinuousClock.now
    private var n = 0
    private var peak = -1.0, peakAt = 0.0
    private var minAvailable = -1.0
    private var firstFootprint = -1.0, lastFootprint = -1.0
    private var firstAvailable = -1.0, lastAvailable = -1.0
    private var series: [[Double]] = []                 // every 10th reading: [t s, footprint MB, available MB]

    init(_ label: String, interval: Double = 0.1, progress: @escaping @Sendable (String) -> Void) {
        self.label = label
        self.interval = interval
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
        thread.name = "decide.memory-sampler"
        thread.qualityOfService = .userInitiated
        thread.start()
    }

    func stop() -> [String: Any] {
        lock.withLock { running = false }
        done.wait()
        sample()
        return lock.withLock {
            ["samples": n, "interval_s": interval, "duration_s": seconds(since: t0),
             "peak_footprint_mb": peak, "peak_at_s": peakAt,
             "footprint_mb_start": firstFootprint, "footprint_mb_end": lastFootprint,
             "min_available_mb": minAvailable, "available_mb_start": firstAvailable, "available_mb_end": lastAvailable,
             "series_1s": series]
        }
    }

    private func sample() {
        let fp = DeviceInfo.footprintMB(), av = DeviceInfo.availableMB()
        let note: String? = lock.withLock {
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
            if av >= 0 && (minAvailable < 0 || av < minAvailable) { minAvailable = av }
            if n % 10 == 1 { series.append([(t * 10).rounded() / 10, fp, av]) }
            guard n % 100 == 0 else { return nil }
            return "\(label): \(String(format: "%.0f", t)) s, footprint \(String(format: "%.0f", fp)) MB (peak "
                + "\(String(format: "%.0f", peak))), available \(String(format: "%.0f", av)) MB"
        }
        if let s = note { progress(s) }
    }
}
