#if os(macOS)
import AppKit
import CoreAI
import CryptoKit
import Darwin
import Foundation

/// Launch arguments for hands-off runs on the Mac (checks, timings, screenshots). The run presses the
/// same controls a person would (ContentView.runAutoplay). Run the binary inside the app:
/// `CoreAIImageGenMac.app/Contents/MacOS/CoreAIImageGenMac -autoplay 1 -model int8 -prompt "a red bicycle" -out ~/r -quit 1`.
///
/// - `-autoplay 1`: press Download & Load (a bundle already on disk loads without a download), wait
///   until the model is ready, then press Generate.
/// - `-model <key>`: the catalog entry to pick (`int8`, `fp16`, `glm512`, `glm1024`, `zimage512`,
///   `zimage1024`; default: the first one).
/// - `-local <folder>`: press Local… with this folder instead of Download & Load.
/// - `-reference <image file>`: switch to Edit with this reference; the prompt is the instruction.
/// - `-prompt "<text>"`, `-steps <n>`, `-guidance <x>`, `-seed <text>`: set the fields before pressing.
/// - `-runs <n>`: press Generate / Apply Edit n times (default 1).
/// - `-stopAfterStep <n>`: press Stop once step n has finished.
/// - `-out <folder>`: where the images (`run-<n>.png`), `result-<launch epoch>.json` and, with `-log 1`,
///   `autoplay-<launch epoch>.log` go (default ~/Library/Application Support/CoreAIImageGen/autoplay).
/// - `-hold <s>`: wait this long before each press and after each image (default 2).
/// - `-quit 1`: quit when the run is done.
///
/// A launch from a script can come up with no window when the screen is locked; adding
/// `-ApplePersistenceIgnoreState YES` brings the window back.
enum LaunchOptions {
    private static var defaults: UserDefaults { .standard }
    static var autoplay: Bool { defaults.bool(forKey: "autoplay") }
    static var log: Bool { defaults.bool(forKey: "log") }
    static var model: String? { defaults.string(forKey: "model") }
    static var local: String? { defaults.string(forKey: "local") }
    static var reference: String? { defaults.string(forKey: "reference") }
    static var prompt: String? { defaults.string(forKey: "prompt") }
    static var steps: Int? { defaults.object(forKey: "steps") == nil ? nil : defaults.integer(forKey: "steps") }
    static var guidance: Float? { defaults.object(forKey: "guidance") == nil ? nil : defaults.float(forKey: "guidance") }
    static var seed: String? { defaults.string(forKey: "seed") }
    static var runs: Int { defaults.object(forKey: "runs") == nil ? 1 : defaults.integer(forKey: "runs") }
    static var stopAfterStep: Int? {
        defaults.object(forKey: "stopAfterStep") == nil ? nil : defaults.integer(forKey: "stopAfterStep")
    }
    static var out: URL {
        if let path = defaults.string(forKey: "out") {
            return URL(filePath: (path as NSString).expandingTildeInPath, directoryHint: .isDirectory)
        }
        return URL.applicationSupportDirectory.appending(path: "CoreAIImageGen/autoplay")
    }
    static var hold: Double { defaults.object(forKey: "hold") == nil ? 2 : defaults.double(forKey: "hold") }
    static var quit: Bool { defaults.bool(forKey: "quit") }
}

struct AutoplayError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

enum Telemetry {
    static let launchEpoch = Int(Date().timeIntervalSince1970)

    /// With `-log 1`, stdout and stderr (these lines and Core AI's own messages) go to the log file.
    static func begin() {
        guard LaunchOptions.log else { return }
        try? FileManager.default.createDirectory(at: LaunchOptions.out, withIntermediateDirectories: true)
        let path = LaunchOptions.out.appending(path: "autoplay-\(launchEpoch).log").path
        let descriptor = path.withCString { Darwin.open($0, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR) }
        if descriptor >= 0 {
            dup2(descriptor, STDOUT_FILENO)
            dup2(descriptor, STDERR_FILENO)
            Darwin.close(descriptor)
        }
    }

    static func line(_ text: String) {
        let time = Date().formatted(.iso8601.year().month().day().time(includingFractionalSeconds: true))
        print("IMAGEGEN \(time) \(text)")
        fflush(nil)
    }
}

/// Every measured number of one hands-off launch; written as JSON when the run ends.
struct RunReport: Codable {
    struct Load: Codable {
        var title: String
        var status: String
        /// Download & Load / Local… pressed → model ready (a download, if any, is not included).
        var seconds: Double?
        var image_size: String
        var edit: Bool
    }
    struct Generation: Codable {
        var mode: String
        var prompt: String
        var steps: Int
        var guidance: Float
        var seed: String
        var reference: String?
        /// Generate pressed → image returned (text encoding, every step, VAE decode).
        var seconds: Double?
        var notice: String?
        var error: String?
        var width: Int?
        var height: Int?
        var png_path: String?
        var png_sha256: String?
        /// SHA-256 of the image's own 8-bit bytes (independent of the PNG encoder).
        var pixels_sha256: String?
    }

    var status = "RUNNING"
    var launch_epoch = Telemetry.launchEpoch
    var model = RunReport.sysctl("hw.model")
    var os_build = RunReport.sysctl("kern.osversion")
    var architecture = AIModel.deviceArchitectureName
    var arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
    var thermal_at_launch = RunReport.thermal()
    var thermal_at_end: String?
    var footprint_mb_at_end: Double?
    var peak_footprint_mb: Double?
    var loads: [Load] = []
    var generations: [Generation] = []

    static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "unavailable" }
        var bytes = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return "unavailable" }
        return String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
    }

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// The process's physical footprint now and its peak, in MB (GPU buffers count: memory is unified).
    static func footprint() -> (now: Double, peak: Double)? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return (Double(info.phys_footprint) / 1_048_576, Double(info.ledger_phys_footprint_peak) / 1_048_576)
    }
}

@MainActor
enum Autoplay {
    private static var started = false

    /// True once per launch, when `-autoplay 1` is given (a second window does not start it again).
    static func claim() -> Bool {
        guard LaunchOptions.autoplay, !started else { return false }
        started = true
        Telemetry.begin()
        Telemetry.line("LAUNCH epoch=\(Telemetry.launchEpoch) pid=\(getpid()) model=\(RunReport.sysctl("hw.model")) os_build=\(RunReport.sysctl("kern.osversion")) arch=\(AIModel.deviceArchitectureName) thermal=\(RunReport.thermal()) args=\(ProcessInfo.processInfo.arguments.dropFirst().joined(separator: " "))")
        return true
    }

    static func pause() async throws {
        try await Task.sleep(for: .seconds(LaunchOptions.hold))
    }

    static func until(_ what: String, timeout: Double, _ condition: () -> Bool) async throws {
        let start = ContinuousClock.now
        while !condition() {
            let waited = start.duration(to: .now)
            if Double(waited.components.seconds) > timeout {
                throw AutoplayError("Timed out after \(Int(timeout)) s waiting for \(what)")
            }
            try await Task.sleep(for: .milliseconds(100))
        }
    }

    /// Copies the PNG the app wrote for Save / Share into `-out` and adds its hashes to the record.
    static func record(_ image: CGImage, png: URL?, run: Int, into generation: inout RunReport.Generation) {
        generation.width = image.width
        generation.height = image.height
        if let data = image.dataProvider?.data as Data? {
            generation.pixels_sha256 = sha256(data)
        }
        guard let png else { return }
        let target = LaunchOptions.out.appending(path: "run-\(run).png")
        do {
            try FileManager.default.createDirectory(at: LaunchOptions.out, withIntermediateDirectories: true)
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.copyItem(at: png, to: target)
            generation.png_path = target.path
            generation.png_sha256 = sha256(try Data(contentsOf: target))
        } catch {
            Telemetry.line("ERROR writing \(target.path): \(error)")
        }
    }

    static func finish(_ report: RunReport, outcome: String) {
        var report = report
        report.status = outcome
        report.thermal_at_end = RunReport.thermal()
        if let footprint = RunReport.footprint() {
            report.footprint_mb_at_end = footprint.now
            report.peak_footprint_mb = footprint.peak
        }
        let url = LaunchOptions.out.appending(path: "result-\(Telemetry.launchEpoch).json")
        do {
            try FileManager.default.createDirectory(at: LaunchOptions.out, withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: url, options: .atomic)
            Telemetry.line("DONE status=\(outcome) result=\(url.path) peak_footprint_mb=\(report.peak_footprint_mb ?? -1)")
        } catch {
            Telemetry.line("DONE status=\(outcome) result=not-written error=\(error)")
        }
        if LaunchOptions.quit { NSApplication.shared.terminate(nil) }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
#endif
