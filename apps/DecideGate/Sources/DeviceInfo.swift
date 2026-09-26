// DeviceInfo — what the gate records about the phone: model identifier (utsname.machine), OS version and
// build, memory, thermal state, low-power mode, the app's memory footprint, free space, and a bundle's size on disk.

import Darwin
import Foundation
import DecideGateSupport

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
