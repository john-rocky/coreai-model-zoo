// GateRunner — the d1-3B device gate on the D1 library (D1Decider: host.py's request checks, text and rows, the option
// table, the float64 readout and the answers; D1Graph: the decoder `main` in S = 16 calls from zero states — or the
// shared prefix — and the vision tower once per crop), over the fixture the port was gated on (361 text records / 393
// questions, the 12 picture records / 24 questions, round 4's 5 red arms), on the iPhone or on the Mac. What the phone
// adds to the Mac runs of the same library (rounds 5c / 6b): the on-device JIT specialization of the decoder's and the
// tower's `.aimodel` (the iPhone AOT `.aimodelc` h19p is the second arm), their loads, footprint and headroom at the
// default memory limit, the probabilities on the phone's own GPU, thermal and battery, and the time of a decision.
// Copied from apps/KevGate/Sources/GateRunner.swift (zoo d1-3b 4955a23) and made d1's: the stage machinery, the memory
// sampler, the thermal / battery records and result.json are KevGate's; new here: the tower and the pictures, the red
// arms, the refused request's questions alone, the deadline (a 30-minute device window), D1_SKIP_DONE (a later launch
// finishes the fixture), the whole-run 1 s memory sampler (memory.tsv grows while the app lives: ../_run.sh reads a
// still file as a frozen app).
// Results: <out>/result.json (rewritten when a stage starts, after every stage and every 5 records; "status" running ->
// done), result.log (one line per event, each with the thermal state and the battery), memory.tsv (every memory reading,
// written as taken); <out> = Documents/d1_gate on the iPhone. <out>/p_ref_<jit|aot>.json keeps every e2e row's p bits,
// hidden sha256 and checks across launches (D1_SKIP_DONE, e2e_aot's comparison, the union of the launches' rows).
//
// Assets (<assets> = Library/Application Support/D1Assets on the iPhone; D1_ASSETS on the Mac), laid out by
// ../_stage.sh and pushed by ../_install.sh:
//   decoder/    the bundle d1_3b_decode_int8mlp_pf16: metadata.json, d1_3b_decode_int8mlp_pf16.aimodel (main S=16, MLP
//               int8 per block of 32, the rest fp16, attention projections fp32), tokenizer/, head/option_rows.*
//   tower/      the bundle d1_3b_vision_fp16w32: metadata.json, d1_3b_vision_fp16w32.aimodel (static, fp16 weights,
//               fp32 compute), host/position_embedding.safetensors
//   aot/        d1_3b_decode_int8mlp_pf16.h19p.aimodelc (the iPhone 18 Pro's AOT of the decoder, compiled with
//               --expect-frequent-reshapes; pushed apart with MD5SUMS_AOT; never opened on a Mac)
//   fixtures/   requests.json, oracle_slim.json, mac_ref.json, red_arms.json, bench.json, images/ — Fixtures.swift
//   MD5SUMS
//
// Stages, in the order D1_STAGES gives (default assets,load_jit,warm,red,e2e_images,e2e_fixture,reset):
//   assets       MD5SUMS: every file present, md5 of every file up to 16 MB (the model files wait for "md5": reading
//                4 GB right before the first load would warm the file cache under the cold-load number); stops when the
//                volume has less than D1_MIN_FREE_GB (default 12) free
//   md5          md5 of the files "assets" left (or of every file of D1_MD5SUMS, another list beside MD5SUMS)
//   load_jit     the decider dropped; then, each under the 100 ms memory sampler with the Core AI cache sized before and
//                after: the text side (D1Decider: tokenizer, option table), the tower alone (D1Tower on the tower's
//                `.aimodel`, GPU preferred, no frequent reshapes: its specialization when cold), and the decoder
//                (D1Decider.loadGraph(.jit): the bundle's `.aimodel`, GPU preferred + expectFrequentReshapes, with the
//                tower again from its cache) — the library's split (AIModel, main, state allocation, tower)
//   load_aot     the same with the decoder's AOT asset D1_AOT (default aot/<name>.h19p.aimodelc, SpecializationOptions
//                .default); the tower as D1_TOWER_ASSET says (jit = its `.aimodel`; aot = the h16c asset beside the
//                tower bundle: the Mac)
//   warm         one decision (D1_WARM, default card_refund): the first calls after the load, scored, not in the bar
//   red          round 4's five red arms: every perturbed request against its base record's questions on this graph
//                (FACTS §7 inverted: an arm is red when an argmax moves on a non-near-tie question, max |dp| > 0.02 or
//                the mean of its rows' mean |dp| > 0.002), and the graph's |dp| against the provider's on the same rows
//   e2e_images   the picture records (D1_IMAGE_IDS / D1_IMAGE_LIMIT): each file decoded and cut here (D1Pixels), every
//                crop through the tower, the decoder on the image rows; per question the ids / groups / keys against the
//                oracle, p against the oracle (both forms) and the Mac's run (p bits, |dp|, hidden sha256), per crop the
//                tower's output sha256 and ms; a record of 2+ questions also shared (= direct bit for bit)
//   e2e_fixture  the text records (D1_IDS / D1_LIMIT; D1_SKIP_DONE leaves out the records an earlier launch finished),
//                the same; a request the host refuses: its text, then each valid question alone
//   reset        the first e2e record of this process again (its pictures too): hidden rows and p bit-equal
//   bench        bench.json `bench` (D1_BENCH_ITEMS filters): per item rest D1_BENCH_REST s (default 60), wait up to
//                D1_WAIT_NOMINAL s (300) for the thermal state nominal, 1 warm-up and `reps` decisions (D1_BENCH_RUNS
//                overrides) with the item's rest between them; every decision with its start offset, every call's ms,
//                the tower's ms, thermal, battery and footprint (the 18 Pro's GPU slows after ~20 s of back-to-back work)
//   e2e_aot      on the decider in use: the fixture's first D1_AOT_LIMIT (60) text records, scored as e2e and against
//                p_ref_jit.json (p bits, hidden sha256)
//   bench_aot    bench.json `bench_aot` on the decider in use
//   load_tower_aot  the tower's AOT asset alone (D1_TOWER_AOT, default aot/d1_3b_vision_fp16w32.h19p.aimodelc,
//                SpecializationOptions.default) under the sampler with the cache sized around it; D1_TOWER_RECORD's
//                crops through it twice (bit-equal re-run) and through the tower's `.aimodel` (its JIT), per crop the
//                output's sha256 against the Mac's (mac_ref.json) and max |d| / cosine against the JIT's and the Mac's
//                values (D1_TOWER_REF, default aot/tower_ref/<record>.f32, when staged); the outputs go to
//                <out>/tower_out_<record>_<aot|jit>.f32
//   probe_noefr  on the decider in use (the decoder AOT without --expect-frequent-reshapes: one specialization per new
//                position length): D1_PROBE_RECORD (card_refund) D1_PROBE_PASSES (2) times, direct; every call's ms
//                with its position length and whether it was new, storage and memory around each pass, p bits and
//                hidden sha256 against the first pass, the oracle and the Mac
//   delete       D1_DELETE (comma-separated): paths under the assets directory, cache:<hex> = the Core AI cache
//                entries of this app whose directory name starts with that hash (the phone has no delete verb), or
//                tmp:<name> = a directory under the app's tmp (iPhone only: MPSGraph's scratch)
//
// The disk guard (iPhone): the volume's free-space reading holds still within a launch (Kev round 8), so the gate keeps
// the launch's reading and subtracts what the container has written since (Library/Caches with the Core AI cache, tmp
// with MPSGraph's scratch); statfs's reading beside it. When the smaller falls below D1_MIN_FREE_GB + D1_DISK_MARGIN_GB
// (8) the loops stop as at the deadline (checked at most every 10 s) and result.json carries `disk_stop`.
//
// A stage writes its start into result.json before it runs. A launch that finds result.json still "running" records the
// stage the previous launch died in (a crash or a jetsam kill) and skips that stage if it is planned again
// (D1_RETRY_DIED=1 runs it anyway): a configuration that died is not retried by accident. D1_DEADLINE_S (seconds after
// the launch) stops the e2e and bench loops before a record or an item that would start within D1_RESERVE_S (40) of it,
// and skips every later stage but reset, md5 and delete: a device window has an end.
//
// Environment (devicectl device process launch --environment-variables on the iPhone; the shell on the Mac), all
// optional except D1_ASSETS on the Mac: D1_RUN_ID, D1_STAGES, D1_ASSETS, D1_OUT, D1_DECODER, D1_TOWER, D1_TOWER_ASSET,
// D1_AOT, D1_LIMIT, D1_IDS, D1_IMAGE_LIMIT, D1_IMAGE_IDS, D1_SHARED (multi | none), D1_SKIP_DONE, D1_DEADLINE_S,
// D1_RESERVE_S, D1_WARM, D1_WAIT_NOMINAL, D1_BENCH_REST, D1_BENCH_RUNS, D1_BENCH_ITEMS, D1_AOT_LIMIT, D1_MIN_FREE_GB,
// D1_DISK_MARGIN_GB, D1_DELETE, D1_MD5SUMS (comma-separated lists: MD5SUMS = the files "assets" left), D1_ORACLE
// (another oracle_slim: the Mac's red control), D1_RECORD_PAUSE, D1_RETRY_DIED, D1_TOWER_AOT, D1_TOWER_RECORD,
// D1_TOWER_REF, D1_PROBE_RECORD, D1_PROBE_PASSES, D1_EXIT_WHEN_DONE (Mac, default 1). Every D1_* variable is echoed into
// result.json (config.env). The launch line records os_proc_available_memory and whether the build carries
// com.apple.developer.kernel.increased-memory-limit (in its code signature and in its embedded profile).

import CoreAI
import D1
import Foundation

struct GateConfig: Sendable {
    static let defaultStages = ["assets", "load_jit", "warm", "red", "e2e_images", "e2e_fixture", "reset"]
    static let knownStages = ["assets", "md5", "load_jit", "load_aot", "warm", "red", "e2e_images", "e2e_fixture", "reset",
                              "bench", "e2e_aot", "bench_aot", "delete", "load_tower_aot", "probe_noefr"]
    /// stages that still run after the deadline (short, or the cleanup)
    static let afterDeadline: Set<String> = ["reset", "md5", "delete"]

    let runID: String
    /// nil on a Mac without D1_ASSETS (the run stops with a fatal line)
    let assets: URL?
    let out: URL
    let stages: [String]
    let decoderPath: String
    let towerPath: String
    /// "jit" (the tower bundle's .aimodel, specialized here) or "aot" (the h16c asset beside the tower bundle: the Mac)
    let towerAsset: String
    let aotPath: String
    /// load_tower_aot: the tower's AOT asset (the iPhone's h19p; the Mac's h16c on a Mac run), the record whose crops go
    /// through it, the Mac's output values of those crops (nil = aot/tower_ref/<record>.f32 when it exists)
    let towerAOTPath: String
    let towerRecord: String
    let towerRefPath: String?
    /// probe_noefr's record and passes
    let probeRecord: String
    let probePasses: Int
    /// the loops stop when the free space left falls below D1_MIN_FREE_GB + this (the disk guard, iPhone)
    let diskMarginGB: Double
    let limit: Int
    let ids: [String]
    let imageLimit: Int
    let imageIDs: [String]
    /// "multi": a request of 2+ questions also runs shared; "none": direct only
    let shared: String
    let skipDone: Bool
    let deadline: Double?
    let reserve: Double
    let warmRecord: String
    let waitNominalSeconds: Double
    let benchRest: Double
    let benchRuns: Int?
    let benchItems: [String]
    let aotLimit: Int
    /// load_jit / load_aot do not start below this much free space (a cold specialization that runs out of disk leaves
    /// partial caches behind)
    let minFreeGB: Double
    let deletePaths: [String]
    let md5Sums: String
    let oraclePath: String?
    let recordPause: Double
    let retryDied: Bool
    let exitWhenDone: Bool
    /// Files up to this many bytes are md5-checked in "assets"; the larger ones in "md5".
    let md5SmallLimit: Int
    let env: [String: String]

    static func fromEnvironment() -> GateConfig {
        let env = ProcessInfo.processInfo.environment
        let home = URL(fileURLWithPath: NSHomeDirectory())
        func path(_ s: String) -> URL { s.hasPrefix("/") ? URL(fileURLWithPath: s) : home.appendingPathComponent(s) }
        func list(_ k: String) -> [String] {
            (env[k] ?? "").split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        }
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        #if os(iOS)
        let assets: URL? = env["D1_ASSETS"].map(path) ?? support.appendingPathComponent("D1Assets")
        let out = env["D1_OUT"].map(path)
            ?? fm.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("d1_gate")
        let exitWhenDone = false
        #else
        let assets: URL? = env["D1_ASSETS"].map(path)
        let out = env["D1_OUT"].map(path) ?? support.appendingPathComponent("D1Gate/out")
        let exitWhenDone = env["D1_EXIT_WHEN_DONE"] != "0"
        #endif
        let stages = list("D1_STAGES")
        return GateConfig(
            runID: env["D1_RUN_ID"] ?? ISO8601DateFormatter().string(from: Date()),
            assets: assets, out: out, stages: stages.isEmpty ? defaultStages : stages,
            decoderPath: env["D1_DECODER"] ?? "decoder",
            towerPath: env["D1_TOWER"] ?? "tower",
            towerAsset: env["D1_TOWER_ASSET"] == "aot" ? "aot" : "jit",
            aotPath: env["D1_AOT"] ?? "aot/d1_3b_decode_int8mlp_pf16.h19p.aimodelc",
            towerAOTPath: env["D1_TOWER_AOT"] ?? "aot/d1_3b_vision_fp16w32.h19p.aimodelc",
            towerRecord: env["D1_TOWER_RECORD"] ?? "img01_shapes_384x384",
            towerRefPath: env["D1_TOWER_REF"],
            probeRecord: env["D1_PROBE_RECORD"] ?? "card_refund",
            probePasses: max(1, Int(env["D1_PROBE_PASSES"] ?? "") ?? 2),
            diskMarginGB: max(0, Double(env["D1_DISK_MARGIN_GB"] ?? "") ?? 8),
            limit: max(0, Int(env["D1_LIMIT"] ?? "") ?? Int.max),
            ids: list("D1_IDS"),
            imageLimit: max(0, Int(env["D1_IMAGE_LIMIT"] ?? "") ?? Int.max),
            imageIDs: list("D1_IMAGE_IDS"),
            shared: env["D1_SHARED"] == "none" ? "none" : "multi",
            skipDone: env["D1_SKIP_DONE"] == "1",
            deadline: Double(env["D1_DEADLINE_S"] ?? "").flatMap { $0 > 0 ? $0 : nil },
            reserve: max(0, Double(env["D1_RESERVE_S"] ?? "") ?? 40),
            warmRecord: env["D1_WARM"] ?? "card_refund",
            waitNominalSeconds: max(0, Double(env["D1_WAIT_NOMINAL"] ?? "") ?? 300),
            benchRest: max(0, Double(env["D1_BENCH_REST"] ?? "") ?? 60),
            benchRuns: Int(env["D1_BENCH_RUNS"] ?? "").map { max(1, $0) },
            benchItems: list("D1_BENCH_ITEMS"),
            aotLimit: max(0, Int(env["D1_AOT_LIMIT"] ?? "") ?? 60),
            minFreeGB: Double(env["D1_MIN_FREE_GB"] ?? "") ?? 12,
            deletePaths: list("D1_DELETE"),
            md5Sums: env["D1_MD5SUMS"] ?? "MD5SUMS",
            oraclePath: env["D1_ORACLE"],
            recordPause: max(0, Double(env["D1_RECORD_PAUSE"] ?? "") ?? 0),
            retryDied: env["D1_RETRY_DIED"] == "1",
            exitWhenDone: exitWhenDone,
            md5SmallLimit: 16 << 20,
            env: env.filter { $0.key.hasPrefix("D1_") })
    }

    var json: [String: Any] {
        ["assets": assets?.path ?? "(unset)", "out": out.path, "stages": stages, "decoder": decoderPath, "tower": towerPath,
         "tower_asset": towerAsset, "aot": aotPath, "tower_aot": towerAOTPath, "tower_record": towerRecord,
         "tower_ref": towerRefPath ?? "(aot/tower_ref/<record>.f32 when present)", "probe_record": probeRecord,
         "probe_passes": probePasses, "disk_margin_gb": diskMarginGB, "limit": limit == Int.max ? -1 : limit, "ids": ids,
         "image_limit": imageLimit == Int.max ? -1 : imageLimit, "image_ids": imageIDs, "shared": shared,
         "skip_done": skipDone, "deadline_s": deadline ?? -1, "reserve_s": reserve, "warm_record": warmRecord,
         "wait_nominal_s": waitNominalSeconds, "bench_rest_s": benchRest, "bench_runs": benchRuns ?? -1,
         "bench_items": benchItems, "aot_limit": aotLimit, "min_free_gb": minFreeGB, "delete": deletePaths,
         "md5sums": md5Sums, "oracle": oraclePath ?? "fixtures/oracle_slim.json", "record_pause_s": recordPause,
         "retry_died": retryDied, "md5_small_limit_bytes": md5SmallLimit, "env": env]
    }
}

actor GateRunner {
    let config: GateConfig
    let emit: @Sendable (String) -> Void
    let setStage: @Sendable (String) -> Void
    private let t0: ContinuousClock.Instant
    private let sink: LogSink
    private var memoryLog: MemoryLog?
    private var appSampler: MemorySampler?
    private var report: [String: Any] = [:]
    private var stageResults: [String: Any] = [:]
    private var stageOrder: [String] = []
    private var fixtures: Fixtures?
    /// The decider with its graph, its kind ("jit" / "aot") and the decoder asset it loaded.
    private var d1: D1Decider?
    private var graphKind: String?
    private var graphAsset: String?
    /// Per row key ("<record>:<question>"): the e2e score of this process (the decider kind of the stage that ran it).
    private var scores: [String: Fixtures.RowScore] = [:]
    private var scoreKind: [String: String] = [:]
    /// The first e2e record of this process, kept for the reset proof.
    private var first: (id: String, hidden: [[Float16]], bits: [[String]])?
    /// (path, md5) of the files "assets" left for "md5"
    private var deferredMD5: [(rel: String, sum: String)] = []
    private var assetsChecked = false
    /// The stage the previous launch died in (result.json left "running"), if any.
    private var diedStage: String?
    /// Stages the deadline cut short or skipped.
    private var deadlineCut: [String] = []
    /// The disk guard: the launch's free space, what the container held at the launch, the reason once it tripped.
    private var launchFreeGB = -1.0
    private var launchWrittenBytes = 0
    private var diskStopped: String?
    private var lastDiskCheck: ContinuousClock.Instant?
    /// Position lengths the decider in use has run in probe_noefr (an efr-less AOT specializes each new one once).
    private var lengthsSeen = Set<Int>()
    static let increasedMemoryLimit = "com.apple.developer.kernel.increased-memory-limit"

    init(config: GateConfig, emit: @escaping @Sendable (String) -> Void, setStage: @escaping @Sendable (String) -> Void) {
        self.config = config
        self.emit = emit
        self.setStage = setStage
        let t0 = ContinuousClock.now
        self.t0 = t0
        sink = LogSink(t0: t0, emit: emit)
    }

    private var assets: URL { config.assets ?? URL(fileURLWithPath: "/nonexistent") }
    private func assetURL(_ p: String) -> URL { p.hasPrefix("/") ? URL(fileURLWithPath: p) : assets.appendingPathComponent(p) }
    private var bundleURL: URL { assetURL(config.decoderPath) }
    private var towerURL: URL { assetURL(config.towerPath) }
    private var aotURL: URL { assetURL(config.aotPath) }
    private var fixturesDir: URL { assets.appendingPathComponent("fixtures") }

    static var buildConfiguration: String {
        #if DEBUG
        return "Debug"
        #else
        return "Release"
        #endif
    }

    /// True once the launch is within D1_RESERVE_S of D1_DEADLINE_S, or once the disk guard tripped.
    private func pastDeadline() -> Bool {
        if diskStopped != nil { return true }
        if lastDiskCheck == nil || seconds(since: lastDiskCheck!) >= 10 {
            lastDiskCheck = .now
            if let why = diskCheck() {
                diskStopped = why
                report["disk_stop"] = why
                line("DISK STOP: \(why)")
                return true
            }
        }
        guard let d = config.deadline else { return false }
        return elapsed() >= d - config.reserve
    }

    /// The free space left: the launch's reading minus what the app's container has written since (Caches with the Core
    /// AI cache, tmp with MPSGraph's scratch), and statfs's reading beside it; a reason when the smaller one is below
    /// D1_MIN_FREE_GB + D1_DISK_MARGIN_GB. nil on a Mac.
    private func diskCheck() -> String? {
        #if os(iOS)
        guard launchFreeGB >= 0 else { return nil }
        let written = DeviceInfo.containerWrittenBytes() - launchWrittenBytes
        let est = launchFreeGB - Double(written) / 1e9
        let st = DeviceInfo.statfsFreeGB(config.out)
        let free = st >= 0 ? min(est, st) : est
        let floor = config.minFreeGB + config.diskMarginGB
        guard free < floor else { return nil }
        return "free space \(f1(free)) GB (launch \(f1(launchFreeGB)) GB - written since \(mb(written)) MB = \(f1(est)) GB; "
            + "statfs \(f1(st)) GB) < D1_MIN_FREE_GB \(f1(config.minFreeGB)) + D1_DISK_MARGIN_GB \(f1(config.diskMarginGB))"
        #else
        return nil
        #endif
    }

    // MARK: - the run

    /// Every stage in order; true when every stage that ran passed.
    func run() async -> Bool {
        let fm = FileManager.default
        try? fm.createDirectory(at: config.out, withIntermediateDirectories: true)
        let resultURL = config.out.appendingPathComponent("result.json")
        let logURL = config.out.appendingPathComponent("result.log")
        let memURL = config.out.appendingPathComponent("memory.tsv")
        let previous = previousLaunch(resultURL)
        try? fm.removeItem(at: resultURL)
        try? fm.removeItem(at: memURL)
        fm.createFile(atPath: logURL.path, contents: nil)
        sink.open(logURL)
        memoryLog = MemoryLog(url: memURL, t0: t0)
        // the whole run, every second: the footprint over the launch, and a memory.tsv that grows while the app lives
        let app = MemorySampler("app", interval: 1.0, fullSeconds: 0, memoryLog: memoryLog, progress: { [sink] in sink.line($0) })
        app.start()
        appSampler = app

        let launchIndex = bumpLaunchCount()
        var device = DeviceInfo.snapshot()
        let battery = await DeviceInfo.battery()
        BatteryCache.shared.set(battery)
        device["battery_level"] = battery.level
        device["battery_state"] = battery.state
        device["power_source"] = DeviceInfo.powerSource(battery.state)
        device["footprint_mb"] = DeviceInfo.footprintMB()
        device["available_mb"] = DeviceInfo.availableMB()
        device["free_gb"] = DeviceInfo.freeGB(config.out)
        device["free_gb_statfs"] = DeviceInfo.statfsFreeGB(config.out)
        let iml = DeviceInfo.entitlement(Self.increasedMemoryLimit)
        device["entitlement_increased_memory_limit"] = iml
        launchFreeGB = device["free_gb"] as? Double ?? -1
        launchWrittenBytes = DeviceInfo.containerWrittenBytes()
        device["container_bytes_at_launch"] = launchWrittenBytes
        report = ["app": "D1Gate", "run_id": config.runID, "status": "running", "started": Self.now(),
                  "launch_index": launchIndex, "device": device,
                  "model": "d1-3B: decoder d1_3b_decode_int8mlp_pf16 (main S=16, MLP int8 per block of 32, hidden output) "
                      + "+ tower d1_3b_vision_fp16w32 + the option readout on the host (float64), D1 library (D1Decider)",
                  "options": ["decoder_jit": d1Describe(D1Paths.decoderJITOptions), "tower_jit": d1Describe(D1Paths.towerJITOptions),
                              "aot": d1Describe(SpecializationOptions.default),
                              "tower_aot": d1Describe(SpecializationOptions.default)],
                  "build": ["configuration": Self.buildConfiguration],
                  "bar": ["argmax": "every question whose oracle top-2 margin is above 0.02 (near-ties counted apart)",
                          "max_abs_dp": 0.02, "mean_of_run_mean_abs_dp": 0.002,
                          "mean_definition": "mean over runs (one run = one question) of the run's mean |dp| over its "
                              + "options (readout_gate.py BAR)",
                          "ids": "row ids (picture slots mapped back to <image>), readout groups and keys = the oracle's",
                          "finite": "every hidden value finite, no all-zero row", "reset": "bit-equal re-run",
                          "red": "every red arm moves its rows past the bar (an argmax on a non-near-tie question, max |dp| "
                              + "> 0.02 or the mean of the rows' mean |dp| > 0.002)"],
                  "config": config.json]
        if let p = previous { report["previous_launch"] = p }
        line("D1Gate run \(config.runID) (launch \(launchIndex) here), \(Self.buildConfiguration) build")
        line("device \(device["machine"] ?? "?") \(device["hw_model"] ?? "?"), \(device["os"] ?? "?") (build "
             + "\(device["os_build"] ?? "?")), Core AI arch \(device["coreai_architecture"] ?? "?"), low power "
             + "\(device["low_power_mode"] ?? "?"), power \(DeviceInfo.powerSource(battery.state)), footprint "
             + "\(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB, free "
             + "\(f1(DeviceInfo.freeGB(config.out))) GB, physical memory \(f2(device["physical_memory_gb"] as? Double ?? -1)) GB"
             + (config.deadline.map { ", deadline \(Int($0)) s after the launch" } ?? ""))
        line("increased-memory-limit: code signature \(iml["signature"] ?? "?"), profile \(iml["profile"] ?? "?") | "
             + "os_proc_available_memory \(f1(DeviceInfo.availableMB())) MB at launch | free \(f1(launchFreeGB)) GB (statfs "
             + "\(f1(device["free_gb_statfs"] as? Double ?? -1)) GB), container \(mb(launchWrittenBytes)) MB in Caches + tmp")
        if let p = previous {
            line("previous launch: run \(p["run_id"] ?? "?") ended \(p["status"] ?? "?")"
                 + (diedStage.map { " — it died in stage \($0) (last result.json update \(p["updated"] ?? "?"))" } ?? ""))
        }
        writeReport()

        guard config.assets != nil else {
            line("FATAL D1_ASSETS is not set (the Mac reads the stage directory from it)")
            return finish(ok: false, fatal: "D1_ASSETS not set")
        }
        #if os(macOS)
        // an iPhone AOT bundle (.h18p. / .h19p. ...) must never be loaded on a Mac (D1_AOT names the Mac's h16c asset)
        if config.stages.contains("load_aot"), config.aotPath.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil {
            line("FATAL refusing an iPhone AOT bundle on macOS: \(config.aotPath)")
            return finish(ok: false, fatal: "iPhone AOT bundle on macOS: \(config.aotPath)")
        }
        if config.stages.contains("load_tower_aot"),
           config.towerAOTPath.range(of: #"\.h[0-9]+p\."#, options: .regularExpression) != nil {
            line("FATAL refusing an iPhone AOT bundle on macOS: \(config.towerAOTPath)")
            return finish(ok: false, fatal: "iPhone AOT bundle on macOS: \(config.towerAOTPath)")
        }
        #endif
        do {
            let fx = try Fixtures(root: fixturesDir, oracleOverride: config.oraclePath.map(assetURL))
            fixtures = fx
            report["fixtures"] = fx.files
            line("fixtures: \(fx.records.count) records (text \(fx.recordsOf(set: "fixture").count), pictures "
                 + "\(fx.recordsOf(set: "image").count), skipped \(fx.skipped)); oracle \(fx.oracle.count), Mac reference "
                 + "\(fx.mac.count); red arms \(fx.red.count); bench \(fx.bench.count) items, bench_aot \(fx.benchAOT.count)")
        } catch {
            line("FATAL fixtures: \(error)")
            return finish(ok: false, fatal: "fixtures: \(error)")
        }

        var allOK = true
        for stage in config.stages {
            setStage(stage)
            let s0 = elapsed()
            var result: [String: Any]
            if stage == diedStage && !config.retryDied {
                line("\(stage): SKIPPED — the previous launch died in this stage (D1_RETRY_DIED=1 runs it again)")
                result = ["skipped": true, "reason": "the previous launch died in this stage", "pass": false]
            } else if pastDeadline() && !GateConfig.afterDeadline.contains(stage) {
                line("\(stage): SKIPPED — " + (diskStopped.map { "the disk guard (\($0))" }
                     ?? "past the deadline (\(f1(elapsed())) s of \(f1(config.deadline ?? -1)) s)"))
                result = ["skipped": true, "reason": diskStopped == nil ? "deadline" : "disk", "deadline_skipped": true]
                deadlineCut.append(stage)
            } else {
                // the stage's start goes into result.json before it runs: a launch that dies here leaves it behind
                writePartial(stage, ["step": "start", "started_at_s": s0])
                switch stage {
                case "assets": result = stageAssets()
                case "md5": result = stageMD5()
                case "load_jit": result = await stageLoad(kind: "jit", key: stage)
                case "load_aot": result = await stageLoad(kind: "aot", key: stage)
                case "load_tower_aot": result = await stageLoadTowerAOT()
                case "probe_noefr": result = await stageProbeNoEFR()
                case "warm": result = await stageWarm()
                case "red": result = await stageRed()
                case "e2e_images": result = await stageE2E(key: stage, records: selected(set: "image"))
                case "e2e_fixture": result = await stageE2E(key: stage, records: selected(set: "fixture"))
                case "reset": result = await stageReset()
                case "bench": result = await stageBench(items: fixtures?.bench ?? [], key: stage)
                case "e2e_aot": result = await stageE2EAOT()
                case "bench_aot": result = await stageBench(items: fixtures?.benchAOT ?? [], key: stage)
                case "delete": result = stageDelete()
                default:
                    line("unknown stage \(stage) (D1_STAGES takes \(GateConfig.knownStages.joined(separator: ", ")))")
                    result = ["pass": false, "error": "unknown stage \(stage)"]
                }
            }
            result["t_start_s"] = s0
            result["t_end_s"] = elapsed()
            stageResults[stage] = result
            stageOrder.append(stage)
            if result["deadline_skipped"] as? Bool != true { allOK = allOK && (result["pass"] as? Bool ?? false) }
            if result["deadline_stop"] as? Bool == true { deadlineCut.append(stage) }
            report["e2e_summary"] = overallSummary()
            writeReport()
            if result["stop"] as? Bool == true {
                line("stopping after \(stage): the later stages need what it failed to provide")
                break
            }
        }
        if stageOrder.isEmpty {
            line("no stage ran")
            allOK = false
        }
        return finish(ok: allOK, fatal: nil)
    }

    /// The text or picture records the e2e stage runs: D1_IDS / D1_IMAGE_IDS filter, D1_LIMIT / D1_IMAGE_LIMIT cap.
    private func selected(set: String) -> [Fixtures.Record] {
        guard let fx = fixtures else { return [] }
        var recs = fx.recordsOf(set: set)
        let ids = set == "image" ? config.imageIDs : config.ids
        if !ids.isEmpty { recs = recs.filter { ids.contains($0.id) } }
        return Array(recs.prefix(set == "image" ? config.imageLimit : config.limit))
    }

    /// The previous launch's result.json, when it did not finish: its run id, status, the stage it was in, its last
    /// update. A copy is kept as runs/<run id>.died.json.
    private func previousLaunch(_ url: URL) -> [String: Any]? {
        guard let d = try? Data(contentsOf: url), let r = try? JSONSerialization.jsonObject(with: d) as? [String: Any] else {
            return nil
        }
        let status = r["status"] as? String ?? "?"
        var p: [String: Any] = ["run_id": r["run_id"] ?? "?", "status": status, "updated": r["updated"] ?? "?",
                                "stage_order": r["stage_order"] ?? []]
        if status == "running" {
            let order = r["stage_order"] as? [String] ?? []
            let stages = r["stages"] as? [String: Any] ?? [:]
            if let last = order.last, (stages[last] as? [String: Any])?["partial"] as? Bool == true {
                diedStage = last
                p["died_in_stage"] = last
                p["died_stage_record"] = stages[last]
            }
            let runs = config.out.appendingPathComponent("runs")
            try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
            let safe = "\(r["run_id"] ?? "unknown")".replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
            try? FileManager.default.copyItem(at: url, to: runs.appendingPathComponent("\(safe).died.json"))
            p["copy"] = "runs/\(safe).died.json"
        }
        return p
    }

    // MARK: - assets

    func stageAssets() -> [String: Any] {
        let fm = FileManager.default
        let c0 = ContinuousClock.now
        var j: [String: Any] = ["dir": assets.path]
        guard let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) else {
            line("assets: no MD5SUMS in \(assets.path)")
            return ["dir": assets.path, "error": "no MD5SUMS", "pass": false, "stop": true]
        }
        var listed = 0, checked = 0, checkedBytes = 0, deferredBytes = 0
        var missing: [String] = [], mismatched: [String] = []
        var tops = Set<String>()
        deferredMD5 = []
        for row in text.split(separator: "\n") {
            let parts = row.split(separator: " ", maxSplits: 1)
            guard parts.count == 2 else { continue }
            listed += 1
            let sum = String(parts[0]), rel = parts[1].trimmingCharacters(in: .whitespaces)
            tops.insert(String(rel.split(separator: "/").first ?? ""))
            let url = assets.appendingPathComponent(rel)
            guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize else {
                missing.append(rel)
                continue
            }
            if size > config.md5SmallLimit {
                deferredMD5.append((rel, sum))
                deferredBytes += size
                continue
            }
            do {
                if try md5Hex(of: url) != sum { mismatched.append(rel) }
                checked += 1
                checkedBytes += size
            } catch {
                mismatched.append("\(rel) (\(error))")
            }
        }
        let names = ((try? fm.contentsOfDirectory(atPath: assets.path)) ?? []).sorted()
        let unlisted = names.filter { !$0.hasPrefix("MD5SUMS") && !tops.contains($0) }
        let aotDir = assets.appendingPathComponent("aot")
        let aot = ((try? fm.contentsOfDirectory(atPath: aotDir.path)) ?? []).sorted().map { n -> [String: Any] in
            let t = DeviceInfo.tree(aotDir.appendingPathComponent(n))
            return ["name": n, "bytes": t.bytes, "files": t.files]
        }
        let dec = DeviceInfo.tree(bundleURL), tw = DeviceInfo.tree(towerURL)
        let required = ["metadata.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "head/option_rows.json",
                        "head/option_rows.safetensors"].map { bundleURL.appendingPathComponent($0) }
            + ["metadata.json", "host/position_embedding.safetensors"].map { towerURL.appendingPathComponent($0) }
            + ["requests.json", "oracle_slim.json", "mac_ref.json", "red_arms.json", "bench.json"].map { fixturesDir.appendingPathComponent($0) }
        let absent = required.filter { !fm.fileExists(atPath: $0.path) }.map(\.path)
        let free = DeviceInfo.freeGB(assets)
        j["md5sums_listed"] = listed
        j["missing"] = missing
        j["md5_mismatch"] = mismatched
        j["md5_checked"] = checked
        j["md5_checked_bytes"] = checkedBytes
        j["md5_deferred"] = deferredMD5.map(\.rel)
        j["md5_deferred_bytes"] = deferredBytes
        j["unlisted_entries"] = unlisted
        j["required_absent"] = absent
        j["decoder_bytes"] = dec.bytes
        j["tower_bytes"] = tw.bytes
        j["aot"] = aot
        j["free_gb"] = free
        j["min_free_gb"] = config.minFreeGB
        j["storage"] = DeviceInfo.storageSnapshot()
        j["seconds"] = seconds(since: c0)
        var ok = listed > 0 && missing.isEmpty && mismatched.isEmpty && absent.isEmpty
        assetsChecked = true
        line("assets \(assets.path): \(listed) files in MD5SUMS, \(missing.count) missing, md5 \(checked) checked "
             + "(\(mb(checkedBytes)) MB) with \(mismatched.count) different, \(deferredMD5.count) model files "
             + "(\(mb(deferredBytes)) MB) left for the md5 stage; decoder \(mb(dec.bytes)) MB, tower \(mb(tw.bytes)) MB; aot "
             + "\(aot.map { "\($0["name"] ?? "?") \(mb($0["bytes"] as? Int ?? 0)) MB" }); unlisted \(unlisted); free \(f1(free)) GB")
        if !missing.isEmpty { line("assets: missing \(missing.prefix(8).joined(separator: ", "))") }
        if !mismatched.isEmpty { line("assets: md5 differs \(mismatched.prefix(8).joined(separator: ", "))") }
        if !absent.isEmpty { line("assets: required file absent \(absent.joined(separator: ", "))") }
        if free >= 0 && free < config.minFreeGB {
            line("assets: STOP free space \(f1(free)) GB < D1_MIN_FREE_GB \(f1(config.minFreeGB)) GB")
            j["error"] = "free space \(free) GB < \(config.minFreeGB) GB"
            ok = false
        }
        j["pass"] = ok
        if !ok { j["stop"] = true }
        return j
    }

    func stageMD5() -> [String: Any] {
        let c0 = ContinuousClock.now
        var files: [(rel: String, sum: String)] = []
        let lists = config.md5Sums.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        for list in lists where list != "MD5SUMS" {
            // another list (e.g. the AOT assets'): every file in it
            guard let text = try? String(contentsOf: assets.appendingPathComponent(list), encoding: .utf8) else {
                line("md5: no \(list) in \(assets.path)")
                return ["error": "no \(list)", "pass": false]
            }
            files += text.split(separator: "\n").compactMap { row -> (rel: String, sum: String)? in
                let p = row.split(separator: " ", maxSplits: 1)
                return p.count == 2 ? (p[1].trimmingCharacters(in: .whitespaces), String(p[0])) : nil
            }
        }
        if lists.contains("MD5SUMS") {
            if !assetsChecked {
                if let text = try? String(contentsOf: assets.appendingPathComponent("MD5SUMS"), encoding: .utf8) {
                    deferredMD5 = text.split(separator: "\n").compactMap { row in
                        let p = row.split(separator: " ", maxSplits: 1)
                        guard p.count == 2 else { return nil }
                        let rel = p[1].trimmingCharacters(in: .whitespaces)
                        let size = (try? assets.appendingPathComponent(rel).resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                        return size > config.md5SmallLimit ? (rel, String(p[0])) : nil
                    }
                }
            }
            files += deferredMD5
        }
        var mismatched: [String] = []
        var bytes = 0
        for (rel, sum) in files {
            let url = assets.appendingPathComponent(rel)
            do {
                if try md5Hex(of: url) != sum { mismatched.append(rel) }
                bytes += (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            } catch {
                mismatched.append("\(rel) (\(error))")
            }
        }
        let s = seconds(since: c0)
        line("md5 (\(config.md5Sums)): \(files.count) files, \(mb(bytes)) MB in \(f1(s)) s, \(mismatched.count) different"
             + (mismatched.isEmpty ? "" : ": \(mismatched.joined(separator: ", "))"))
        return ["list": config.md5Sums, "files": files.map(\.rel), "bytes": bytes, "md5_mismatch": mismatched, "seconds": s,
                "pass": !files.isEmpty && mismatched.isEmpty]
    }

    // MARK: - loads

    /// The Core AI cache of this app at `at` into `storage`; its bytes and files.
    private func cacheSnapshot(_ at: String, _ storage: inout [String: Any]) -> (bytes: Int, files: Int) {
        let s = DeviceInfo.storageSnapshot()
        storage[at] = s
        let c = s["coreai_cache"] as? [String: Any] ?? [:]
        return (c["bytes"] as? Int ?? 0, c["files"] as? Int ?? 0)
    }

    /// Drops the decider in use and waits a moment for the memory to go.
    private func dropDecider(_ j: inout [String: Any]) async {
        let had = graphKind
        j["footprint_mb_with_previous_decider"] = DeviceInfo.footprintMB()
        d1 = nil
        graphKind = nil
        graphAsset = nil
        lengthsSeen = []
        try? await Task.sleep(for: .seconds(2))
        j["dropped_decider"] = had ?? "none"
        j["footprint_mb_after_drop"] = DeviceInfo.footprintMB()
        j["available_mb_after_drop"] = DeviceInfo.availableMB()
    }

    private func hashHex(_ asset: URL) -> String? {
        (try? Data(contentsOf: asset.appendingPathComponent("main.hash"))).map { $0.map { String(format: "%02x", $0) }.joined() }
    }

    /// The sampler's record without its series (the series is in memory.tsv and in the stage's own `memory`).
    private func memoryBrief(_ m: [String: Any]) -> [String: Any] {
        ["peak_footprint_mb": m["peak_footprint_mb"] ?? -1, "min_available_mb": m["min_available_mb"] ?? -1,
         "footprint_mb_start": m["footprint_mb_start"] ?? -1, "footprint_mb_end": m["footprint_mb_end"] ?? -1,
         "available_mb_start": m["available_mb_start"] ?? -1, "available_mb_end": m["available_mb_end"] ?? -1,
         "samples": m["samples"] ?? 0, "duration_s": m["duration_s"] ?? -1]
    }

    /// load_jit (the decoder's .aimodel, D1Paths.decoderJITOptions) or load_aot (D1_AOT, SpecializationOptions.default),
    /// each with the tower (D1_TOWER_ASSET). Three steps under the sampler: the text side, the tower alone, the decoder
    /// with the tower (D1Decider.loadGraph).
    func stageLoad(kind: String, key: String) async -> [String: Any] {
        var j: [String: Any] = ["kind": kind]
        await dropDecider(&j)
        let bundle = bundleURL, tower = towerURL
        let towerKind: D1Graph.Asset = config.towerAsset == "aot" ? .aot : .jit
        let meta: D1Decider.Metadata
        let towerModel: URL
        do {
            meta = try D1Decider.Metadata(bundle: bundle)
            let tm = try JSONParser.parse(Data(contentsOf: tower.appendingPathComponent("metadata.json")))
            guard let towerName = tm["name"]?.string else { throw GateError.assets("\(tower.path)/metadata.json: no name") }
            // the paths D1Decider.loadGraph opens: the bundle's .aimodel (metadata assets.main), or the h16c AOT beside it
            towerModel = towerKind == .jit ? try D1Tower.modelAsset(bundle: tower) : D1Paths.aot(bundle: tower, name: towerName)
        } catch {
            line("\(key): ERROR bundle metadata: \(error)")
            return ["kind": kind, "error": "\(error)", "pass": false]
        }
        let decAsset = kind == "jit" ? bundle.appendingPathComponent(meta.asset) : aotURL
        let decOptions = kind == "jit" ? D1Paths.decoderJITOptions : SpecializationOptions.default
        let towerOptions = towerKind == .jit ? D1Paths.towerJITOptions : SpecializationOptions.default
        let dt = DeviceInfo.tree(decAsset), tt = DeviceInfo.tree(towerModel)
        j["bundle"] = bundle.path
        j["asset"] = decAsset.path
        j["asset_bytes"] = dt.bytes
        j["asset_files"] = dt.files
        j["asset_main_hash"] = hashHex(decAsset) ?? "?"
        let pkg = Self.aotPackage(decAsset)
        if let pkg {
            j["asset_package"] = pkg
            j["asset_efr"] = pkg["efr"] ?? NSNull()
        }
        j["options"] = d1Describe(decOptions)
        j["tower_bundle"] = tower.path
        j["tower_asset"] = towerModel.path
        j["tower_asset_kind"] = towerKind.rawValue
        j["tower_asset_bytes"] = tt.bytes
        j["tower_asset_main_hash"] = hashHex(towerModel) ?? "?"
        j["tower_options"] = d1Describe(towerOptions)
        var storage: [String: Any] = [:]
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["footprint_mb_before"] = DeviceInfo.footprintMB()
        j["available_mb_before"] = DeviceInfo.availableMB()
        let free = DeviceInfo.freeGB(assets)
        j["free_gb_before"] = free
        let b0 = await DeviceInfo.battery()
        j["battery_start"] = ["level": b0.level, "state": b0.state, "power": DeviceInfo.powerSource(b0.state)]
        j["thermal_start"] = DeviceInfo.thermal()
        line("\(key): decoder \(decAsset.lastPathComponent) (\(mb(dt.bytes)) MB, \(dt.files) files, main.hash "
             + "\((hashHex(decAsset) ?? "?").prefix(12))…\(pkg.map { ", efr \($0["efr"] ?? "?")" } ?? "")), "
             + "\(d1Describe(decOptions)); tower \(towerModel.lastPathComponent) "
             + "(\(mb(tt.bytes)) MB), \(d1Describe(towerOptions)); Core AI cache \(mb(cache0.bytes)) MB in \(cache0.files) files; "
             + "footprint \(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB; free \(f1(free)) GB")
        for (what, u) in [("decoder", decAsset), ("tower", towerModel)] where !FileManager.default.fileExists(atPath: u.path) {
            line("\(key): ERROR no \(what) asset at \(u.path)")
            j["error"] = "no \(what) asset at \(u.path)"
            j["pass"] = false
            return j
        }
        if free >= 0 && free < config.minFreeGB {
            line("\(key): STOP free space \(f1(free)) GB < D1_MIN_FREE_GB \(f1(config.minFreeGB)) GB")
            j["error"] = "free space \(free) GB < \(config.minFreeGB) GB"
            j["pass"] = false
            j["stop"] = true
            return j
        }

        // 1. the text side: metadata, tokenizer, option table
        j["step"] = "text"
        j["storage"] = storage
        writePartial(key, j)
        let (dres, dmem, dwall) = await sampled("\(key) text") { try await D1Decider(bundle: bundle) }
        let decider: D1Decider
        switch dres {
        case .success(let d):
            decider = d
            j["text"] = ["wall_s": dwall, "memory": memoryBrief(dmem), "table_ids": d.table.ids.count,
                         "hidden": d.table.hidden, "tokenizer_json_sha256": d.tokenizer.tokenizerJSONSHA256 ?? "?",
                         "bundle_name": d.metadata.name, "chunk": d.metadata.chunk, "max_context": d.metadata.maxContext]
            line("\(key) text: \(f2(dwall)) s (tokenizer + option table \(d.table.ids.count) x \(d.table.hidden)), peak footprint "
                 + "\(f1(dmem["peak_footprint_mb"] as? Double ?? -1)) MB")
        case .failure(let error):
            line("\(key): ERROR text side: \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
            return j
        }

        // 2. the tower alone: its specialization when its cache entry is cold
        j["step"] = "tower"
        writePartial(key, j)
        let ct0 = cacheSnapshot("before_tower", &storage)
        let (tres, tmem, twall) = await sampled("\(key) tower") {
            try await D1Tower(bundle: tower, asset: towerModel, options: towerOptions).loadSeconds
        }
        try? await Task.sleep(for: .seconds(1))
        let ct1 = cacheSnapshot("after_tower", &storage)
        switch tres {
        case .success(let loadS):
            j["tower"] = ["wall_s": twall, "library_load_s": loadS, "memory": tmem, "cache_bytes_added": ct1.bytes - ct0.bytes,
                          "cache_files_added": ct1.files - ct0.files]
            line("\(key) tower alone: wall \(f2(twall)) s (AIModel + main \(f2(loadS)) s) | peak footprint "
                 + "\(f1(tmem["peak_footprint_mb"] as? Double ?? -1)) MB, least available \(f1(tmem["min_available_mb"] as? Double ?? -1)) MB "
                 + "| Core AI cache +\(mb(ct1.bytes - ct0.bytes)) MB")
        case .failure(let error):
            line("\(key): ERROR tower: \(error)")
            j["tower"] = ["wall_s": twall, "memory": tmem, "error": "\(error)", "error_detail": Self.errorRecord(error)]
            j["error"] = "tower: \(error)"
            j["pass"] = false
            j["storage"] = storage
            return j
        }

        // 3. the decoder (cold or warm), the tower again from its cache: D1Decider.loadGraph
        j["step"] = "decoder"
        j["storage"] = storage
        writePartial(key, j)
        let cd0 = cacheSnapshot("before_decoder", &storage)
        let (gres, gmem, gwall) = await sampled("\(key) decoder") {
            try await decider.loadGraph(asset: kind == "jit" ? .jit : .aot, assetURL: decAsset, options: decOptions,
                                        tower: tower, towerAsset: towerKind)
        }
        try? await Task.sleep(for: .seconds(1))
        let cd1 = cacheSnapshot("after_decoder", &storage)
        j["memory"] = gmem
        j["wall_s"] = gwall
        switch gres {
        case .success(let g):
            d1 = decider
            graphKind = kind
            graphAsset = decAsset.path
            j["decoder"] = ["wall_s": gwall, "graph_load_s": g.loadSeconds, "decoder_model_s": g.decoder.loadSeconds.model,
                            "decoder_function_s": g.decoder.loadSeconds.function,
                            "state_allocation_s": g.decoder.allocationSeconds, "tower_load_s": g.tower?.loadSeconds ?? -1,
                            "functions": g.decoder.functionNames, "descriptor": JSONWriter.compact(g.decoder.descriptor),
                            "options": d1Describe(g.decoder.options), "tower_options": g.tower.map { d1Describe($0.options) } ?? "none",
                            "tower_descriptor": g.tower.map { JSONWriter.compact($0.descriptor) } ?? "none",
                            "chunk": g.metadata.chunk, "max_context": g.metadata.maxContext, "hidden": g.metadata.hidden,
                            "image_rows": g.metadata.imageRows, "memory": memoryBrief(gmem),
                            "cache_bytes_added": cd1.bytes - cd0.bytes, "cache_files_added": cd1.files - cd0.files]
            line("\(key) decoder: wall \(f2(gwall)) s = AIModel \(f2(g.decoder.loadSeconds.model)) s + main "
                 + "\(f2(g.decoder.loadSeconds.function)) s + states \(f2(g.decoder.allocationSeconds)) s + tower "
                 + "\(f2(g.tower?.loadSeconds ?? -1)) s | peak footprint \(f1(gmem["peak_footprint_mb"] as? Double ?? -1)) MB, "
                 + "least available \(f1(gmem["min_available_mb"] as? Double ?? -1)) MB | Core AI cache +\(mb(cd1.bytes - cd0.bytes)) MB")
            j["step"] = "done"
            j["pass"] = true
        case .failure(let error):
            line("\(key): ERROR decoder: \(error)")
            j["error"] = "decoder: \(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
        }
        let cache1 = cacheSnapshot("after", &storage)
        j["cache_bytes_after"] = cache1.bytes
        j["cache_bytes_added"] = cache1.bytes - cache0.bytes
        j["footprint_mb_after"] = DeviceInfo.footprintMB()
        j["available_mb_after"] = DeviceInfo.availableMB()
        j["free_gb_after"] = DeviceInfo.freeGB(assets)
        line("\(key): Core AI cache \(mb(cache0.bytes)) -> \(mb(cache1.bytes)) MB (+\(mb(cache1.bytes - cache0.bytes)) MB); "
             + "footprint \(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB")
        let b1 = await DeviceInfo.battery()
        j["battery_end"] = ["level": b1.level, "state": b1.state, "power": DeviceInfo.powerSource(b1.state)]
        j["thermal_end"] = DeviceInfo.thermal()
        j["storage"] = storage
        return j
    }

    // MARK: - decisions

    private func noDecider(_ key: String) -> [String: Any] {
        line("\(key): ERROR no decider loaded (run load_jit or load_aot first)")
        return ["error": "no decider loaded", "pass": false]
    }

    private static func isRefusal(_ e: D1Error) -> Bool {
        switch e {
        case .request, .graphLimit, .json: return true
        default: return false
        }
    }

    /// The pictures of a record: decoded and cut here (D1Pixels), the crops' four inputs with the tower bundle's position
    /// table; nil for a text record.
    private func pictures(_ rec: Fixtures.Record, _ d1: D1Decider) throws -> (D1TowerInputs?, Double) {
        guard !rec.images.isEmpty else { return (nil, 0) }
        let c = ContinuousClock.now
        let p = try d1.pictures(files: rec.images.map { fixturesDir.appendingPathComponent($0) })
        return (p, seconds(since: c) * 1e3)
    }

    /// A decision's times and calls (ms): latency = the tower calls and the image rows + the graph calls + the readout
    /// and the response (decide.py's `latency_ms`; the rows' build is `plan`).
    private func traceTimes(_ tr: D1Trace) -> [String: Any] {
        let s = tr.seconds
        let latency = ((s["images"] ?? 0) + (s["graph"] ?? 0) + (s["readout"] ?? 0)) * 1e3
        return ["mode": tr.mode, "shared_k": tr.sharedK, "calls": tr.callSeconds.count,
                "call_ms": tr.callSeconds.map { ($0 * 1e5).rounded() / 100 }, "reset_ms": tr.resetSeconds * 1e3,
                "latency_ms": latency, "plan_ms": (s["plan"] ?? 0) * 1e3, "images_ms": (s["images"] ?? 0) * 1e3,
                "graph_ms": (s["graph"] ?? 0) * 1e3, "readout_ms": (s["readout"] ?? 0) * 1e3, "wall_ms": (s["wall"] ?? 0) * 1e3,
                "tower_ms": tr.towerSeconds.map { ($0 * 1e5).rounded() / 100 }, "image_rows": tr.imageRows,
                "input_tokens": tr.plan.inputTokens, "state_tokens": tr.plan.stateTokens,
                "rows_tokens": tr.plan.rows.map { $0.ids.count }]
    }

    private func responseSHA(_ v: JSONValue) -> String {
        sha256Hex(Data(PythonFormat.dumps(v, indent: 2, asciiOnly: false).utf8))
    }

    /// The image rows a trace bound (each crop's first n_tokens rows of its tower output, cast to float16): sha256.
    private func imageRowsSHA(_ tr: D1Trace, _ p: D1TowerInputs, width: Int) -> String {
        var rows: [Float16] = []
        for (o, c) in zip(tr.towerOutputs, p.pictures.flatMap(\.crops)) { rows += o[0..<(c.tokens * width)].map { Float16($0) } }
        return sha256Hex(of: rows)
    }

    private func pText(_ ps: [[Double]]) -> String {
        ps.map { p in "[" + p.map { String(format: "%.4f", $0) }.joined(separator: " ") + "]" }.joined(separator: " ")
    }

    private func rowJSON(_ s: Fixtures.RowScore, _ r: D1GraphRow, _ p: [Double], _ o: Fixtures.OracleQuestion) -> [String: Any] {
        var j: [String: Any] = [
            "key": s.key, "name": r.row.name, "type": r.row.kind.rawValue, "row_len": r.ids.count, "slot": r.slot,
            "ids_sha256": sha256Hex(Data(PythonFormat.dumps(.ints(r.ids)).utf8)), "ids_equal_oracle": s.idsEqualOracle,
            "hidden_sha256": s.hiddenSHA, "hidden_finite": s.finite, "hidden_all_zero": s.allZero, "p": p, "p_bits": s.pBits,
            "argmax": s.argmax, "argmax_oracle": s.argmaxOracle, "argmax_equal": s.argmaxEqual, "near_tie": s.nearTie,
            "top2_margin_oracle": o.top2Margin, "max_abs_dp": s.maxAbsDp, "mean_abs_dp": s.meanAbsDp,
        ]
        if let a = s.apiMaxAbsDp { j["api_max_abs_dp"] = a }
        if !s.idsEqualOracle { j["row_ids"] = r.ids; j["read_groups"] = r.readGroups }
        if let x = s.macHiddenEqual { j["mac_hidden_sha256_equal"] = x }
        if let x = s.macPBitEqual { j["mac_p_bit_equal"] = x }
        if let x = s.macMaxAbsDp { j["mac_max_abs_dp"] = x }
        if let x = s.macArgmaxEqual { j["mac_argmax_equal"] = x }
        return j
    }

    /// One record, direct (and shared when it asks 2+ questions and D1_SHARED is multi): its JSON, its scored rows, the
    /// hidden rows and p bits of its direct run (the reset proof). A request the host refuses: its text, then (a text
    /// record) each of its valid questions alone, as readout_gate.py runs the provider's rows.
    private func decideRecord(_ rec: Fixtures.Record, _ d1: D1Decider, _ fx: Fixtures, shared wantShared: Bool) async throws
        -> (json: [String: Any], scores: [Fixtures.RowScore], hidden: [[Float16]], bits: [[String]])
    {
        guard let orc = fx.oracle[rec.id] else { throw GateError.fixture("no oracle record \(rec.id)") }
        let mac = fx.mac[rec.id]
        var rj: [String: Any] = ["id": rec.id, "set": rec.set, "source": rec.source]
        let width = d1.graph?.metadata.hidden ?? 0
        let (pics, pixelsMs) = try pictures(rec, d1)
        if let pics {
            rj["pixels_ms"] = pixelsMs
            rj["crops"] = pics.pictures.flatMap(\.crops).count
            rj["pictures"] = pics.pictures.map { ["id": $0.id, "size": [$0.width, $0.height], "crops": $0.crops.count] }
        }
        var scores: [Fixtures.RowScore] = []
        var rows: [[String: Any]] = []
        do {
            let request = try D1Request(json: rec.json)
            let tr = try await d1.trace(request: request, mode: .direct, pictures: pics)
            for (k, r) in tr.plan.rows.enumerated() {
                guard let o = orc.question(r.row.name) else {
                    rows.append(["name": r.row.name, "error": "no oracle question"])
                    continue
                }
                let s = Fixtures.score(key: "\(rec.id):\(r.row.name)", set: rec.set, row: r, hidden: tr.hidden[k],
                                       p: tr.probabilities[k], oracle: o, mac: mac?[r.row.name])
                scores.append(s)
                rows.append(rowJSON(s, r, tr.probabilities[k], o))
            }
            rj["accepted"] = true
            rj["rows"] = rows
            rj["direct"] = traceTimes(tr)
            rj["response_sha256"] = responseSHA(tr.response)
            rj["answers"] = JSONWriter.compact(tr.response["answers"] ?? .null)
            if let pics, !tr.towerOutputs.isEmpty {
                rj["tower_outputs_sha256"] = tr.towerOutputs.map { sha256Hex(of: $0) }
                rj["image_rows_sha256"] = imageRowsSHA(tr, pics, width: width)
            }
            if wantShared && request.questions.count > 1 {
                let ts = try await d1.trace(request: request, mode: .shared, pictures: pics)
                let hEq = ts.hidden.count == tr.hidden.count && zip(ts.hidden, tr.hidden).allSatisfy { a, b in
                    a.count == b.count && a.withUnsafeBytes { x in b.withUnsafeBytes { y in x.elementsEqual(y) } }
                }
                let pEq = ts.probabilities.map { $0.map(\.bitPattern) } == tr.probabilities.map { $0.map(\.bitPattern) }
                let respEq = responseSHA(ts.response) == responseSHA(tr.response)
                var sh: [String: Any] = ["hidden_bit_equal_direct": hEq, "p_bit_equal_direct": pEq,
                                         "response_equal_direct": respEq, "times": traceTimes(ts)]
                if !ts.towerOutputs.isEmpty {
                    sh["tower_outputs_bit_equal_direct"] = ts.towerOutputs.count == tr.towerOutputs.count
                        && zip(ts.towerOutputs, tr.towerOutputs).allSatisfy { $0.0.map(\.bitPattern) == $0.1.map(\.bitPattern) }
                }
                rj["shared"] = sh
            }
            return (rj, scores, tr.hidden, tr.probabilities.map { $0.map(Fixtures.hexBits) })
        } catch let e as D1Error where Self.isRefusal(e) {
            rj["accepted"] = false
            rj["refusal"] = e.message
            var hidden: [[Float16]] = [], bits: [[String]] = []
            // a refused text request's questions alone (the readout gate's rows); a refused request with pictures stops here
            for m in pics == nil ? (rec.json["questions"]?.members ?? []) : [] {
                guard (try? D1Request.validateQuestion(name: m.key, m.value)) != nil, let o = orc.question(m.key) else { continue }
                let one = try D1Request(json: .obj([("state", rec.json["state"] ?? .null), ("questions", .object([m]))]))
                let t = try await d1.trace(request: one, mode: .direct)
                let s = Fixtures.score(key: "\(rec.id):\(m.key)", set: rec.set, row: t.plan.rows[0], hidden: t.hidden[0],
                                       p: t.probabilities[0], oracle: o, mac: mac?[m.key])
                scores.append(s)
                var r = rowJSON(s, t.plan.rows[0], t.probabilities[0], o)
                r["direct"] = traceTimes(t)
                rows.append(r)
                hidden.append(t.hidden[0])
                bits.append(t.probabilities[0].map(Fixtures.hexBits))
            }
            rj["sub"] = rows
            return (rj, scores, hidden, bits)
        }
    }

    func stageWarm() async -> [String: Any] {
        guard let d1, let fx = fixtures else { return noDecider("warm") }
        guard let rec = fx.byID[config.warmRecord] else { return ["error": "no record \(config.warmRecord)", "pass": false] }
        var j: [String: Any] = ["record": rec.id, "kind": graphKind ?? ""]
        j["thermal_start"] = DeviceInfo.thermal()
        let (result, memory, wall) = await sampled("warm") { try await self.decideRecord(rec, d1, fx, shared: false) }
        do {
            let r = try result.get()
            j["run"] = r.json
            j["memory"] = memoryBrief(memory)
            j["wall_s"] = wall
            j["thermal_end"] = DeviceInfo.thermal()
            let ok = !r.scores.isEmpty && r.scores.allSatisfy { $0.idsEqualOracle && $0.finite && ($0.argmaxEqual || $0.nearTie) }
            j["pass"] = ok
            let t = r.json["direct"] as? [String: Any] ?? [:]
            let calls = t["call_ms"] as? [Double] ?? []
            line("warm \(rec.id) (\(graphKind ?? "?")): latency \(f1(t["latency_ms"] as? Double ?? -1)) ms, \(calls.count) calls "
                 + "(first \(f1(calls.first ?? -1)) ms, median \(f1(median(calls))) ms), wall \(f1(wall * 1e3)) ms | max|dp| "
                 + "\(f6(r.scores.map(\.maxAbsDp).max() ?? .nan)), argmax \(r.scores.filter(\.argmaxEqual).count)/\(r.scores.count), "
                 + "Mac p bits \(r.scores.filter { $0.macPBitEqual == true }.count)/\(r.scores.count)")
        } catch {
            line("warm: ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
        }
        return j
    }

    // MARK: - red arms

    /// Round 4's red arms on this graph: each perturbed row against its base row (the same question of the base record,
    /// direct on this graph), FACTS §7 inverted; and the graph's |dp| against the provider's on the same rows.
    func stageRed() async -> [String: Any] {
        guard let d1, let fx = fixtures else { return noDecider("red") }
        var base: [String: [String: [Double]]] = [:]
        var baseRuns: [[String: Any]] = []
        var arms: [[String: Any]] = []
        var allRed = true
        var cut = false
        let c0 = ContinuousClock.now
        do {
            for arm in fx.red {
                var items: [[String: Any]] = []
                for rq in arm.requests {
                    // a deadline or the disk guard stops the arms between requests (an efr-less AOT specializes every new
                    // row length: one arm can take minutes and gigabytes)
                    if pastDeadline() {
                        cut = true
                        line("red: \(diskStopped == nil ? "deadline" : "disk guard") — stopping before \(arm.id) \(rq.name) at "
                             + "\(f1(elapsed())) s")
                        break
                    }
                    if base[rq.base] == nil {
                        guard let b = fx.byID[rq.base] else { throw GateError.fixture("red \(arm.id): no base record \(rq.base)") }
                        let tb = try await d1.trace(request: D1Request(json: b.json), mode: .direct)
                        var m: [String: [Double]] = [:]
                        for (k, r) in tb.plan.rows.enumerated() { m[r.row.name] = tb.probabilities[k] }
                        base[rq.base] = m
                        baseRuns.append(["record": rq.base, "calls": tb.callSeconds.count,
                                         "p_bits": tb.probabilities.map { $0.map(Fixtures.hexBits) },
                                         "hidden_sha256": tb.hidden.map { sha256Hex(of: $0) }])
                    }
                    let tr = try await d1.trace(request: D1Request(json: rq.json), mode: .direct)
                    for (k, r) in tr.plan.rows.enumerated() where rq.only == nil || rq.only!.contains(r.row.name) {
                        guard let o = fx.oracle[rq.base]?.question(r.row.name), let pb = base[rq.base]?[r.row.name] else {
                            throw GateError.fixture("red \(arm.id) \(rq.name): no base row \(rq.base):\(r.row.name)")
                        }
                        let p = tr.probabilities[k]
                        let dp = zip(p, pb).map { abs($0 - $1) }
                        var item: [String: Any] = [
                            "request": rq.name, "question": r.row.name, "base": "\(rq.base):\(r.row.name)", "tokens": r.ids.count,
                            "keys_equal_base": r.row.keys == o.keys, "oracle_top2_margin": o.top2Margin, "near_tie": o.nearTie,
                            "argmax_moved": Fixtures.firstArgmax(p) != Fixtures.firstArgmax(pb),
                            "max_abs_dp_vs_base": dp.max() ?? .nan, "mean_abs_dp_vs_base": dp.reduce(0, +) / Double(max(1, dp.count)),
                            "base_p_bits": pb.map(Fixtures.hexBits), "perturbed_p_bits": p.map(Fixtures.hexBits),
                            "hidden_sha256": sha256Hex(of: tr.hidden[k]),
                        ]
                        if let po = rq.oracle[r.row.name], po.count == p.count, o.probs.count == p.count {
                            // (graph perturbed - graph base) - (oracle perturbed - oracle base)
                            let dd = (0..<p.count).map { abs((p[$0] - pb[$0]) - (po[$0] - o.probs[$0])) }
                            item["oracle_max_abs_dp_vs_base"] = zip(po, o.probs).map { abs($0 - $1) }.max() ?? .nan
                            item["graph_dp_minus_oracle_dp_max_abs"] = dd.max() ?? .nan
                        }
                        items.append(item)
                    }
                }
                if cut && items.isEmpty { break }
                let movedFar = items.filter { ($0["argmax_moved"] as? Bool ?? false) && !($0["near_tie"] as? Bool ?? true) }.count
                let mdp = items.compactMap { $0["max_abs_dp_vs_base"] as? Double }.max() ?? 0
                let means = items.compactMap { $0["mean_abs_dp_vs_base"] as? Double }
                let meanRun = means.isEmpty ? 0 : means.reduce(0, +) / Double(means.count)
                let facts7 = ["a_argmax_non_near_tie": movedFar >= 1, "b_max_abs_dp": mdp > 0.02, "c_mean_of_run_means": meanRun > 0.002]
                let red = !items.isEmpty && facts7.values.contains(true)
                let dd = items.compactMap { $0["graph_dp_minus_oracle_dp_max_abs"] as? Double }.max()
                allRed = allRed && red
                arms.append(["id": arm.id, "kind": arm.kind, "rows": items.count, "argmax_moved_non_near_tie": movedFar,
                             "max_abs_dp_vs_base": mdp, "mean_of_run_mean_abs_dp_vs_base": meanRun, "red_facts7": facts7,
                             "red": red, "graph_dp_minus_oracle_dp_max_abs": dd ?? Double.nan, "items": items])
                line("red \(arm.id) (\(arm.kind)): \(items.count) rows, max|dp| vs base \(f6(mdp)), mean \(f6(meanRun)), argmax "
                     + "moved (non-near-tie) \(movedFar) -> \(red ? "RED" : "not red")"
                     + (dd.map { " | graph dp - oracle dp max \(f6($0))" } ?? ""))
                if cut { break }
            }
        } catch {
            line("red: ERROR \(error)")
            return ["arms": arms, "base_runs": baseRuns, "error": "\(error)", "error_detail": Self.errorRecord(error), "pass": false]
        }
        let red = arms.filter { $0["red"] as? Bool == true }.count
        line("red: \(red)/\(arms.count) arms red in \(f1(seconds(since: c0))) s")
        return ["arms": arms, "base_runs": baseRuns, "red_arms": red, "arm_count": arms.count, "arms_planned": fx.red.count,
                "deadline_stop": cut, "seconds": seconds(since: c0),
                "rule": "red = the perturbed rows against their base rows fail FACTS §7: (a) an argmax moves on a question "
                    + "whose oracle top-2 margin is above 0.02, or (b) max |dp| > 0.02, or (c) the mean over the arm's rows "
                    + "of the row's mean |dp| > 0.002", "pass": allRed && !arms.isEmpty]
    }

    // MARK: - e2e

    /// The records in order, direct (and shared), scored; a deadline stops the loop before a record.
    func stageE2E(key: String, records recs: [Fixtures.Record], compareRef: [String: [String: Any]]? = nil,
                  keepFirst: Bool = true) async -> [String: Any] {
        guard let d1, let fx = fixtures else { return noDecider(key) }
        let kind = graphKind ?? "?"
        let prior = config.skipDone ? loadPRef(kind: kind) : [:]
        var j: [String: Any] = ["kind": kind, "asset": graphAsset ?? "", "records_planned": recs.count,
                                "questions_planned": recs.reduce(0) { $0 + (fx.oracle[$1.id]?.questions.count ?? 0) }]
        j["thermal_start"] = DeviceInfo.thermal()
        line("\(key): \(recs.count) records (\(j["questions_planned"] ?? 0) questions) on the \(kind) decider"
             + (config.skipDone ? ", leaving out the records p_ref_\(kind).json already holds (\(prior.count) rows)" : ""))
        var out: [[String: Any]] = []
        var done: [Fixtures.RowScore] = []
        var errors = 0
        var skippedDone: [String] = []
        var sharedRecs = 0, sharedHEq = 0, sharedPEq = 0, sharedRespEq = 0, sharedTowerEq = 0
        var refEqP = 0, refEqHidden = 0, refRows = 0
        var refMaxDp = 0.0
        var deadlineStop = false
        var timeline: [[Any]] = [await timelinePoint(0)]
        var nextTimeline = 20.0
        let sampler = MemorySampler(key, fullSeconds: 60, timelineEvery: 20, memoryLog: memoryLog,
                                    progress: { [sink] in sink.line($0) })
        sampler.start()
        let e0 = ContinuousClock.now
        for (i, rec) in recs.enumerated() {
            if pastDeadline() {
                deadlineStop = true
                line("\(key): \(diskStopped == nil ? "deadline" : "disk guard") — stopping before record \(i + 1)/\(recs.count) "
                     + "(\(rec.id)) at \(f1(elapsed())) s")
                break
            }
            if config.skipDone, let o = fx.oracle[rec.id], !o.questions.isEmpty,
               o.questions.allSatisfy({ prior["\(rec.id):\($0.name)"] != nil }) {
                skippedDone.append(rec.id)
                continue
            }
            let tStart = seconds(since: e0)
            if tStart >= nextTimeline {
                timeline.append(await timelinePoint(tStart))
                while nextTimeline <= tStart { nextTimeline += 20 }
            }
            do {
                let r = try await decideRecord(rec, d1, fx, shared: config.shared == "multi")
                var rj = r.json
                rj["t_start_s"] = tStart
                rj["thermal"] = DeviceInfo.thermal()
                rj["footprint_mb"] = DeviceInfo.footprintMB()
                rj["available_mb"] = DeviceInfo.availableMB()
                rj["battery_level"] = BatteryCache.shared.get().level
                if keepFirst && first == nil && !r.hidden.isEmpty { first = (rec.id, r.hidden, r.bits) }
                for s in r.scores {
                    scores[s.key] = s
                    scoreKind[s.key] = kind
                }
                if let ref = compareRef {
                    var eqP = 0, eqH = 0
                    for s in r.scores {
                        guard let x = ref[s.key] else { continue }
                        refRows += 1
                        let bits = x["p_bits"] as? [String] ?? []
                        if bits == s.pBits { eqP += 1 }
                        if (x["hidden_sha256"] as? String) == s.hiddenSHA { eqH += 1 }
                        if bits.count == s.pBits.count {
                            let d = zip(bits, s.pBits).map { abs(Fixtures.fromBits($0) - Fixtures.fromBits($1)) }.max() ?? 0
                            refMaxDp = max(refMaxDp, d)
                        }
                    }
                    refEqP += eqP
                    refEqHidden += eqH
                    rj["ref_p_bit_equal_rows"] = eqP
                    rj["ref_hidden_sha256_equal_rows"] = eqH
                }
                var sharedText = ""
                if let sh = rj["shared"] as? [String: Any] {
                    sharedRecs += 1
                    let h = sh["hidden_bit_equal_direct"] as? Bool ?? false, p = sh["p_bit_equal_direct"] as? Bool ?? false
                    let resp = sh["response_equal_direct"] as? Bool ?? false
                    sharedHEq += h ? 1 : 0
                    sharedPEq += p ? 1 : 0
                    sharedRespEq += resp ? 1 : 0
                    if let t = sh["tower_outputs_bit_equal_direct"] as? Bool { sharedTowerEq += t ? 1 : 0 } else { sharedTowerEq += 1 }
                    let st = sh["times"] as? [String: Any] ?? [:]
                    sharedText = " | shared \(st["calls"] ?? 0) calls \(f1(st["latency_ms"] as? Double ?? -1)) ms, = direct \(h && p && resp)"
                }
                done += r.scores
                out.append(rj)
                let t = rj["direct"] as? [String: Any] ?? [:]
                let pt = r.scores.map { s in "[" + s.pBits.map { String(format: "%.4f", Fixtures.fromBits($0)) }.joined(separator: " ") + "]" }
                line("\(key) \(i + 1)/\(recs.count) \(rec.id): \(r.scores.count) rows"
                     + ((rj["accepted"] as? Bool) == false ? " (refused whole: each question alone)" : "")
                     + ", \(t["calls"] ?? "-") calls, latency \(f1(t["latency_ms"] as? Double ?? -1)) ms"
                     + ((rj["crops"] as? Int).map { ", \($0) crops (pixels \(f1(rj["pixels_ms"] as? Double ?? -1)) ms)" } ?? "")
                     + " | \(pt.joined(separator: " ")) | max|dp| \(f6(r.scores.map(\.maxAbsDp).max() ?? .nan)), argmax "
                     + "\(r.scores.filter(\.argmaxEqual).count)/\(r.scores.count), ids \(r.scores.filter(\.idsEqualOracle).count)/"
                     + "\(r.scores.count), Mac p bits \(r.scores.filter { $0.macPBitEqual == true }.count)/\(r.scores.count), Mac "
                     + "max|dp| \(f6(r.scores.compactMap(\.macMaxAbsDp).max() ?? .nan))" + sharedText)
            } catch {
                errors += 1
                out.append(["id": rec.id, "set": rec.set, "error": "\(error)", "error_detail": Self.errorRecord(error)])
                line("\(key) \(i + 1)/\(recs.count) \(rec.id): ERROR \(error)")
            }
            if (out.count) % 5 == 0 || i + 1 == recs.count {
                j["records"] = out
                j["summary"] = Fixtures.summarize(done)
                j["timeline"] = timeline
                j["skipped_done"] = skippedDone
                writePartial(key, j)
            }
            if errors >= 3 && done.isEmpty {
                line("\(key): 3 errors and no record done: stopping the stage")
                break
            }
            if config.recordPause > 0 { try? await Task.sleep(for: .seconds(config.recordPause)) }
        }
        timeline.append(await timelinePoint(seconds(since: e0)))
        let memory = sampler.stop()
        let summary = Fixtures.summarize(done)
        j["records"] = out
        j["records_done"] = out.filter { $0["error"] == nil }.count
        j["skipped_done"] = skippedDone
        j["summary"] = summary
        j["errors"] = errors
        j["memory"] = memory
        j["timeline"] = timeline
        j["timeline_columns"] = ["t_s", "thermal", "battery_level", "battery_state", "footprint_mb", "available_mb"]
        j["thermal_end"] = DeviceInfo.thermal()
        j["seconds"] = seconds(since: e0)
        j["deadline_stop"] = deadlineStop
        let callMs = out.flatMap { ($0["direct"] as? [String: Any])?["call_ms"] as? [Double] ?? [] }
        j["calls_total"] = callMs.count
        j["call_ms_median"] = median(callMs)
        j["call_ms_p90"] = percentile(callMs, 0.9)
        let sharedOK = sharedHEq == sharedRecs && sharedPEq == sharedRecs && sharedRespEq == sharedRecs && sharedTowerEq == sharedRecs
        j["shared_summary"] = ["records": sharedRecs, "hidden_bit_equal_direct": sharedHEq, "p_bit_equal_direct": sharedPEq,
                               "response_equal_direct": sharedRespEq, "tower_outputs_equal_direct": sharedTowerEq,
                               "rule": "static S: the shared run is bit-equal to direct", "pass": sharedOK]
        if compareRef != nil {
            j["ref_summary"] = ["rows_compared": refRows, "p_bit_equal": refEqP, "hidden_sha256_equal": refEqHidden,
                                "max_abs_dp": refMaxDp]
        }
        line("\(key) every 20 s (t, thermal, battery, footprint): " + timeline.map {
            "\($0[0])s \($0[1]) \(String(format: "%.0f", ($0[2] as? Double ?? -1) * 100))% \($0[3]) \(String(format: "%.0f", $0[4] as? Double ?? -1)) MB"
        }.joined(separator: ", "))
        line(Fixtures.summaryLine(key, summary))
        line("\(key): \(callMs.count) calls, call ms median \(f2(median(callMs))) (p90 \(f2(percentile(callMs, 0.9)))), "
             + "\(f1(seconds(since: e0))) s; shared \(sharedRecs) records: hidden = direct \(sharedHEq), p = direct \(sharedPEq), "
             + "response = direct \(sharedRespEq); skipped (done earlier) \(skippedDone.count); errors \(errors)"
             + (deadlineStop ? "; STOPPED by the deadline" : ""))
        if compareRef != nil {
            line("\(key) vs p_ref_jit: p bits \(refEqP)/\(refRows), hidden sha256 \(refEqHidden)/\(refRows), max|dp| \(f6(refMaxDp))")
        }
        savePRef(kind: kind, rows: done)
        // the union of every launch's rows of this set (this launch's and the earlier ones' in p_ref)
        if let set = recs.first?.set {
            let u = unionSummary(kind: kind, set: set)
            j["union_summary"] = u
            line(Fixtures.summaryLine("\(key) union of the launches (\(kind), \(set))", u))
        }
        let plannedRows = done.count
        j["pass"] = errors == 0 && plannedRows > 0 && (summary["bar_pass"] as? Bool ?? false) && sharedOK
        return j
    }

    /// The first e2e record of this process again: hidden rows and p bit-equal.
    func stageReset() async -> [String: Any] {
        guard let d1, let fx = fixtures else { return noDecider("reset") }
        guard let f = first, let rec = fx.byID[f.id] else {
            line("reset: ERROR no e2e record ran in this process")
            return ["error": "no e2e record in this process", "pass": false]
        }
        do {
            let r = try await decideRecord(rec, d1, fx, shared: false)
            let hEq = r.hidden.count == f.hidden.count && zip(r.hidden, f.hidden).allSatisfy { a, b in
                a.count == b.count && a.withUnsafeBytes { x in b.withUnsafeBytes { y in x.elementsEqual(y) } }
            }
            let pEq = r.bits == f.bits
            let t = r.json["direct"] as? [String: Any] ?? [:]
            line("reset \(rec.id) again (\(graphKind ?? "?")): hidden bit-equal \(hEq), p bit-equal \(pEq), latency "
                 + "\(f1(t["latency_ms"] as? Double ?? -1)) ms")
            return ["record": rec.id, "kind": graphKind ?? "", "hidden_bit_equal": hEq, "p_bit_equal": pEq,
                    "rows": r.hidden.count, "latency_ms": t["latency_ms"] ?? -1, "thermal": DeviceInfo.thermal(), "pass": hEq && pEq]
        } catch {
            line("reset: ERROR \(error)")
            return ["error": "\(error)", "error_detail": Self.errorRecord(error), "pass": false]
        }
    }

    /// On the decider in use (the AOT asset): the fixture's first D1_AOT_LIMIT text records, scored, and against
    /// p_ref_jit.json (the JIT run of an earlier launch).
    func stageE2EAOT() async -> [String: Any] {
        guard d1 != nil, let fx = fixtures else { return noDecider("e2e_aot") }
        let subset = Array(fx.recordsOf(set: "fixture").prefix(config.aotLimit))
        let ref = loadPRef(kind: "jit")
        line("e2e_aot: \(subset.count) records on the \(graphKind ?? "?") decider; p_ref_jit.json \(ref.count) rows")
        var j = await stageE2E(key: "e2e_aot", records: subset, compareRef: ref.isEmpty ? nil : ref, keepFirst: false)
        j["p_ref_jit_rows"] = ref.count
        if graphKind != "aot" { j["note"] = "the decider in use is \(graphKind ?? "none"), not the AOT asset" }
        return j
    }

    // MARK: - bench

    /// Per item: rest, wait for nominal, 1 warm-up, then `reps` decisions (the item's rest between them).
    func stageBench(items all: [Fixtures.BenchItem], key: String) async -> [String: Any] {
        guard let d1, let fx = fixtures else { return noDecider(key) }
        let items = config.benchItems.isEmpty ? all : all.filter { config.benchItems.contains($0.name) }
        var j: [String: Any] = ["kind": graphKind ?? "", "asset": graphAsset ?? "", "rest_s": config.benchRest,
                                "wait_nominal_cap_s": config.waitNominalSeconds, "items_planned": items.map(\.name)]
        var out: [[String: Any]] = []
        var ok = true
        var deadlineStop = false
        let b0 = ContinuousClock.now
        for item in items {
            if pastDeadline() {
                deadlineStop = true
                line("\(key): deadline — stopping before \(item.name) at \(f1(elapsed())) s")
                break
            }
            var r: [String: Any] = ["item": item.name, "record": item.record, "questions": item.questions ?? [],
                                    "mode": item.shared ? "shared" : "direct", "rep_rest_s": item.repRest]
            do {
                guard let rec = fx.byID[item.record] else { throw GateError.fixture("no record \(item.record)") }
                let sub = Fixtures.subRequest(rec.json, names: item.questions)
                let req = try D1Request(json: sub)
                let (pics, pixelsMs) = try pictures(rec, d1)
                if pics != nil { r["pixels_ms"] = pixelsMs }
                if config.benchRest > 0 {
                    line("\(key) \(item.name): resting \(Int(config.benchRest)) s before the item")
                    try? await Task.sleep(for: .seconds(config.benchRest))
                }
                if config.waitNominalSeconds > 0 { r["wait_nominal"] = await waitForNominal("\(key) \(item.name)") }
                let bs = await DeviceInfo.battery()
                BatteryCache.shared.set(bs)
                r["thermal_start"] = DeviceInfo.thermal()
                r["battery_start"] = ["level": bs.level, "state": bs.state, "power": DeviceInfo.powerSource(bs.state)]
                let reps = config.benchRuns ?? item.reps
                var decisions: [[String: Any]] = []
                var lat: [Double] = []
                var firstBits: [[UInt64]]? = nil
                var sameReps = true, sameE2E = 0, comparedE2E = 0
                let c0 = ContinuousClock.now
                var lastEnd = 0.0, firstTimedStart = -1.0
                var stoppedByDeadline = false
                for i in 0...reps {                    // i = 0: one warm-up decision, not in the statistics
                    if i >= 1 && item.repRest > 0 { try? await Task.sleep(for: .seconds(item.repRest)) }
                    if i > 0 && pastDeadline() { stoppedByDeadline = true; break }
                    let tStart = seconds(since: c0)
                    let tr = try await d1.trace(request: req, mode: item.shared ? .shared : .direct, pictures: pics)
                    let tEnd = seconds(since: c0)
                    let b = BatteryCache.shared.get()
                    let pb = tr.probabilities.map { $0.map(\.bitPattern) }
                    if i == 1 { firstBits = pb; firstTimedStart = tStart }
                    if i > 1 { sameReps = sameReps && pb == firstBits! }
                    var eq: Bool? = nil
                    let keys = tr.plan.rows.map { "\(item.record):\($0.row.name)" }
                    if keys.allSatisfy({ scores[$0] != nil && scoreKind[$0] == graphKind }) {
                        eq = zip(keys, tr.probabilities).allSatisfy { scores[$0.0]!.pBits == $0.1.map(Fixtures.hexBits) }
                        comparedE2E += 1
                        if eq == true { sameE2E += 1 }
                    }
                    var d = traceTimes(tr)
                    d["rep"] = i
                    d["warmup"] = i == 0
                    d["t_start_s"] = tStart
                    d["t_end_s"] = tEnd
                    d["t_bench_s"] = seconds(since: b0)
                    d["thermal"] = DeviceInfo.thermal()
                    d["battery_level"] = b.level
                    d["battery_state"] = b.state
                    d["footprint_mb"] = DeviceInfo.footprintMB()
                    d["p_bits"] = tr.probabilities.map { $0.map(Fixtures.hexBits) }
                    d["p_bits_equal_e2e"] = eq.map { $0 as Any } ?? NSNull()
                    decisions.append(d)
                    if i > 0 {
                        lat.append(d["latency_ms"] as? Double ?? .nan)
                        lastEnd = tEnd
                    }
                }
                let b1 = await DeviceInfo.battery()
                BatteryCache.shared.set(b1)
                r["decisions"] = decisions
                r["summary"] = ["n": lat.count, "latency_ms": lat, "median": median(lat), "min": lat.min() ?? Double.nan,
                                "max": lat.max() ?? Double.nan]
                r["row_tokens"] = (decisions.first?["rows_tokens"] as? [Int]) ?? []
                r["reps"] = reps
                r["timed_runs_start_s"] = firstTimedStart
                r["timed_runs_end_s"] = lastEnd
                r["timed_runs_in_first_20s"] = lastEnd <= 20
                r["reps_p_bit_equal"] = sameReps
                r["p_bits_equal_e2e"] = ["equal": sameE2E, "compared": comparedE2E]
                r["thermal_end"] = DeviceInfo.thermal()
                r["battery_end"] = ["level": b1.level, "state": b1.state, "power": DeviceInfo.powerSource(b1.state)]
                r["deadline_stop"] = stoppedByDeadline
                let itemOK = sameReps && sameE2E == comparedE2E && !lat.isEmpty
                r["pass"] = itemOK
                ok = ok && itemOK
                if stoppedByDeadline { deadlineStop = true }
                let tower = (decisions.last?["tower_ms"] as? [Double]) ?? []
                line("\(key) \(item.name) (\(item.shared ? "shared" : "direct"), rows \(r["row_tokens"] ?? []) tokens, "
                     + "\(lat.count) of \(reps) reps, thermal \(r["thermal_start"] ?? "?") -> \(DeviceInfo.thermal()), battery "
                     + "\(String(format: "%.0f", bs.level * 100)) -> \(String(format: "%.0f", b1.level * 100)) % "
                     + "\(DeviceInfo.powerSource(b1.state))): median \(f1(median(lat))) ms (min \(f1(lat.min() ?? .nan)), max "
                     + "\(f1(lat.max() ?? .nan)); " + lat.map { f1($0) }.joined(separator: " ") + ")"
                     + (tower.isEmpty ? "" : " | tower \(tower.map { f1($0) }.joined(separator: " ")) ms")
                     + " | timed runs end at \(f1(lastEnd)) s" + (lastEnd <= 20 ? " (inside the first 20 s)" : " (past 20 s)")
                     + " | reps p bit-equal \(sameReps), p = e2e \(sameE2E)/\(comparedE2E)"
                     + (stoppedByDeadline ? " | STOPPED by the deadline" : ""))
            } catch {
                line("\(key) \(item.name): ERROR \(error)")
                r["error"] = "\(error)"
                r["error_detail"] = Self.errorRecord(error)
                r["pass"] = false
                ok = false
            }
            out.append(r)
            j["items"] = out
            writePartial(key, j)
            if deadlineStop { break }
        }
        j["items"] = out
        j["deadline_stop"] = deadlineStop
        j["pass"] = ok && !out.isEmpty
        return j
    }

    // MARK: - delete

    /// D1_DELETE: paths under the assets directory, or cache:<hex> = this app's Core AI cache entries whose directory
    /// name starts with the hash (at least 8 hex digits).
    func stageDelete() -> [String: Any] {
        let fm = FileManager.default
        var done: [[String: Any]] = []
        var ok = true
        let free0 = DeviceInfo.freeGB(assets)
        for p in config.deletePaths {
            if p.hasPrefix("tmp:") {
                // a directory under the app's tmp (MPSGraph's scratch); on a Mac tmp is every process's: refused
                let rel = String(p.dropFirst(4))
                #if os(iOS)
                guard !rel.isEmpty, !rel.hasPrefix("/"), !rel.contains("..") else {
                    done.append(["path": p, "error": "want tmp:<a name under the app's tmp>"])
                    ok = false
                    continue
                }
                let u = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(rel)
                let t = DeviceInfo.tree(u)
                var rec: [String: Any] = ["path": u.path, "bytes": t.bytes, "files": t.files]
                if fm.fileExists(atPath: u.path) {
                    do {
                        try fm.removeItem(at: u)
                        rec["deleted"] = true
                    } catch {
                        rec["error"] = "\(error)"
                        ok = false
                    }
                } else {
                    rec["deleted"] = false
                    rec["absent"] = true
                }
                done.append(rec)
                line("delete \(p): \(mb(t.bytes)) MB in \(t.files) files, deleted \(rec["deleted"] as? Bool ?? false)")
                #else
                done.append(["path": p, "error": "tmp: is refused on a Mac (its tmp is every process's)"])
                ok = false
                #endif
                continue
            }
            if p.hasPrefix("cache:") {
                let hex = String(p.dropFirst(6)).lowercased()
                guard hex.count >= 8, hex.allSatisfy(\.isHexDigit) else {
                    done.append(["path": p, "error": "want cache:<at least 8 hex digits>"])
                    ok = false
                    continue
                }
                let root = DeviceInfo.coreAICacheDir()
                var hits: [[String: Any]] = []
                if let e = fm.enumerator(at: root, includingPropertiesForKeys: [.isDirectoryKey]) {
                    for case let u as URL in e where u.lastPathComponent.lowercased().hasPrefix(hex) {
                        if (try? u.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                            hits.append(["dir": u.path, "bytes": DeviceInfo.tree(u).bytes])
                            e.skipDescendants()
                        }
                    }
                }
                for h in hits {
                    do { try fm.removeItem(atPath: h["dir"] as! String) } catch {
                        ok = false
                        done.append(["path": h["dir"] ?? "", "error": "\(error)"])
                    }
                }
                done.append(["path": p, "cache_entries": hits])
                line("delete \(p): \(hits.count) cache entries, \(mb(hits.reduce(0) { $0 + ($1["bytes"] as? Int ?? 0) })) MB")
                continue
            }
            guard !p.hasPrefix("/"), !p.contains(".."), !p.isEmpty else {
                done.append(["path": p, "error": "only a relative path under the assets directory"])
                ok = false
                continue
            }
            let u = assets.appendingPathComponent(p)
            let t = DeviceInfo.tree(u)
            var rec: [String: Any] = ["path": u.path, "bytes": t.bytes, "files": t.files]
            if let h = hashHex(u) { rec["main_hash"] = h }
            do {
                try fm.removeItem(at: u)
                rec["deleted"] = true
            } catch {
                rec["error"] = "\(error)"
                ok = false
            }
            done.append(rec)
            line("delete \(p): \(mb(t.bytes)) MB in \(t.files) files, deleted \(rec["deleted"] as? Bool ?? false)"
                 + (rec["main_hash"].map { " (main.hash \($0))" } ?? ""))
        }
        let free1 = DeviceInfo.freeGB(assets)
        line("delete: free \(f1(free0)) -> \(f1(free1)) GB")
        return ["deleted": done, "free_gb_before": free0, "free_gb_after": free1, "storage": DeviceInfo.storageSnapshot(),
                "pass": ok && !config.deletePaths.isEmpty]
    }

    // MARK: - the tower's AOT

    /// The files of an AOT asset's MPSGraph package (<main>-<arch>-delegates/MPSGraph/mpsExecutable.mpsgraphpackage), name
    /// -> bytes, and what coreai-build put in it: a dynamic graph with --expect-frequent-reshapes keeps the original
    /// beside a specialized model (efr), without it only the original (specialized at the load, once per new input shape
    /// for a dynamic graph); a static graph gets a specialized model only. nil for a `.aimodel`.
    static func aotPackage(_ asset: URL) -> [String: Any]? {
        guard asset.pathExtension == "aimodelc" else { return nil }
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(atPath: asset.path) else { return nil }
        var files: [String: Int] = [:]
        for d in dirs where d.hasSuffix("-delegates") {
            let pkg = asset.appendingPathComponent(d).appendingPathComponent("MPSGraph/mpsExecutable.mpsgraphpackage")
            for n in (try? fm.contentsOfDirectory(atPath: pkg.path)) ?? [] {
                files["\(d)/\(n)"] = (try? pkg.appendingPathComponent(n).resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? -1
            }
        }
        let names = files.keys.map { ($0 as NSString).lastPathComponent }
        let original = names.contains { $0.hasPrefix("original_model") }
        let specialized = names.contains { $0.hasPrefix("specialized_model") }
        let kind = original && specialized ? "original + specialized (efr)"
            : original ? "original only (no efr: specialized on the device)" : specialized ? "specialized only (static)" : "?"
        return ["files": files, "efr": original && specialized, "original": original, "specialized": specialized, "kind": kind]
    }

    /// float32 little-endian, the arrays one after the other.
    static func writeFloats(_ xs: [[Float]], to url: URL) throws {
        var d = Data()
        for x in xs { x.withUnsafeBytes { d.append(contentsOf: $0) } }
        try d.write(to: url)
    }

    static func readFloats(_ url: URL) -> [Float]? {
        guard let d = try? Data(contentsOf: url), d.count % 4 == 0, !d.isEmpty else { return nil }
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// max |a - b|, the cosine (float64 sums) and the number of values whose bits differ, over two equal-length arrays.
    static func compareFloats(_ a: [Float], _ b: [Float]) -> [String: Any] {
        guard a.count == b.count, !a.isEmpty else { return ["error": "lengths \(a.count) and \(b.count)"] }
        var maxd = 0.0, dot = 0.0, na = 0.0, nb = 0.0, diff = 0
        for i in 0..<a.count {
            let x = Double(a[i]), y = Double(b[i])
            maxd = max(maxd, abs(x - y))
            dot += x * y
            na += x * x
            nb += y * y
            if a[i].bitPattern != b[i].bitPattern { diff += 1 }
        }
        return ["max_abs": maxd, "cos": dot / (na.squareRoot() * nb.squareRoot()), "values": a.count, "bit_different": diff]
    }

    static func brief(_ v: Any?) -> String {
        guard let m = v as? [String: Any] else { return v.map { "\($0)" } ?? "-" }
        if let e = m["error"] { return "error \(e)" }
        return "max|d| \(String(format: "%.3g", m["max_abs"] as? Double ?? .nan)), cos "
            + "\(String(format: "%.10f", m["cos"] as? Double ?? .nan)), bit-different \(m["bit_different"] ?? "?")"
    }

    /// mac_ref.json's tower_outputs_sha256 of a picture record (the Mac's run of the same crops).
    static func macTowerSHA(_ url: URL, record: String) -> [String]? {
        guard let d = try? Data(contentsOf: url), let doc = try? JSONParser.parse(d),
              let a = doc["records"]?[record]?["tower_outputs_sha256"]?.array else { return nil }
        let s = a.compactMap(\.string)
        return s.count == a.count ? s : nil
    }

    /// load_tower_aot: the tower's AOT asset (D1_TOWER_AOT, SpecializationOptions.default) loaded alone under the 100 ms
    /// sampler with the Core AI cache sized around it; D1_TOWER_RECORD's crops through it twice, then through the tower's
    /// `.aimodel` (D1Paths.towerJITOptions: its cache entry when warm). Per crop: the output's sha256 against the Mac's,
    /// max |d| and cosine against the JIT's and the Mac's values (D1_TOWER_REF; default aot/tower_ref/<record>.f32).
    func stageLoadTowerAOT() async -> [String: Any] {
        let key = "load_tower_aot"
        var j: [String: Any] = ["kind": "tower_aot"]
        await dropDecider(&j)
        let tower = towerURL
        let asset = assetURL(config.towerAOTPath)
        let at = DeviceInfo.tree(asset)
        j["tower_bundle"] = tower.path
        j["asset"] = asset.path
        j["asset_bytes"] = at.bytes
        j["asset_files"] = at.files
        j["asset_main_hash"] = hashHex(asset) ?? "?"
        j["asset_package"] = Self.aotPackage(asset) ?? [:]
        j["options"] = d1Describe(SpecializationOptions.default)
        j["record"] = config.towerRecord
        guard FileManager.default.fileExists(atPath: asset.path) else {
            line("\(key): ERROR no tower AOT asset at \(asset.path)")
            j["error"] = "no tower AOT asset at \(asset.path)"
            j["pass"] = false
            return j
        }
        guard let fx = fixtures, let rec = fx.byID[config.towerRecord], !rec.images.isEmpty else {
            line("\(key): ERROR no picture record \(config.towerRecord)")
            j["error"] = "no picture record \(config.towerRecord)"
            j["pass"] = false
            return j
        }
        var storage: [String: Any] = [:]
        let cache0 = cacheSnapshot("before", &storage)
        j["cache_bytes_before"] = cache0.bytes
        j["footprint_mb_before"] = DeviceInfo.footprintMB()
        j["available_mb_before"] = DeviceInfo.availableMB()
        j["thermal_start"] = DeviceInfo.thermal()
        line("\(key): tower \(asset.lastPathComponent) (\(mb(at.bytes)) MB, \(at.files) files, main.hash "
             + "\((hashHex(asset) ?? "?").prefix(12))…), \(d1Describe(SpecializationOptions.default)); Core AI cache "
             + "\(mb(cache0.bytes)) MB; footprint \(f1(DeviceInfo.footprintMB())) MB, available \(f1(DeviceInfo.availableMB())) MB")
        j["step"] = "load"
        j["storage"] = storage
        writePartial(key, j)
        let (res, mem, wall) = await sampled("\(key) load") {
            try await D1Tower(bundle: tower, asset: asset, options: SpecializationOptions.default)
        }
        try? await Task.sleep(for: .seconds(1))
        let cache1 = cacheSnapshot("after_load", &storage)
        j["storage"] = storage
        let aotTower: D1Tower
        switch res {
        case .success(let t):
            aotTower = t
            j["load"] = ["wall_s": wall, "library_load_s": t.loadSeconds, "memory": memoryBrief(mem),
                         "cache_bytes_added": cache1.bytes - cache0.bytes, "cache_files_added": cache1.files - cache0.files,
                         "descriptor": JSONWriter.compact(t.descriptor), "options": d1Describe(t.options)]
            line("\(key) load: wall \(f2(wall)) s (AIModel + main \(f2(t.loadSeconds)) s) | peak footprint "
                 + "\(f1(mem["peak_footprint_mb"] as? Double ?? -1)) MB, least available \(f1(mem["min_available_mb"] as? Double ?? -1)) MB "
                 + "| Core AI cache +\(mb(cache1.bytes - cache0.bytes)) MB")
        case .failure(let error):
            line("\(key): ERROR load: \(error)")
            j["load"] = ["wall_s": wall, "memory": memoryBrief(mem)]
            j["error"] = "load: \(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
            return j
        }
        do {
            j["step"] = "crops"
            writePartial(key, j)
            let crops = try D1TowerInputs.pictures(files: rec.images.map { fixturesDir.appendingPathComponent($0) },
                                                   table: aotTower.positionTable).pictures.flatMap(\.crops)
            func encodeAll(_ t: D1Tower) async throws -> (outs: [[Float]], ms: [Double]) {
                var outs: [[Float]] = [], ms: [Double] = []
                for c in crops {
                    let c0 = ContinuousClock.now
                    outs.append(try await t.encode(c))
                    ms.append(seconds(since: c0) * 1e3)
                }
                return (outs, ms)
            }
            let a1 = try await encodeAll(aotTower)
            let a2 = try await encodeAll(aotTower)
            let rerunEqual = a1.outs.count == a2.outs.count
                && zip(a1.outs, a2.outs).allSatisfy { $0.0.map(\.bitPattern) == $0.1.map(\.bitPattern) }
            let finite = a1.outs.allSatisfy { $0.allSatisfy(\.isFinite) }
            let allZero = a1.outs.contains { $0.allSatisfy { $0 == 0 } }
            let shas = a1.outs.map { sha256Hex(of: $0) }
            try? Self.writeFloats(a1.outs, to: config.out.appendingPathComponent("tower_out_\(rec.id)_aot.f32"))
            var cmp: [String: Any] = [:]
            if let m = Self.macTowerSHA(fixturesDir.appendingPathComponent("mac_ref.json"), record: rec.id) {
                cmp["mac_sha256_equal"] = zip(shas, m).filter { $0.0 == $0.1 }.count
                cmp["mac_sha256_crops"] = m.count
            }
            let refURL = assetURL(config.towerRefPath ?? "aot/tower_ref/\(rec.id).f32")
            if let ref = Self.readFloats(refURL) {
                cmp["mac_values"] = Self.compareFloats(a1.outs.flatMap { $0 }, ref)
                cmp["mac_values_file"] = refURL.path
            } else {
                cmp["mac_values"] = "absent (\(refURL.lastPathComponent))"
            }
            j["step"] = "jit"
            writePartial(key, j)
            let cj0 = cacheSnapshot("before_jit", &storage)
            let (jres, jmem, jwall) = await sampled("\(key) jit") {
                try await D1Tower(bundle: tower, asset: try D1Tower.modelAsset(bundle: tower), options: D1Paths.towerJITOptions)
            }
            let cj1 = cacheSnapshot("after_jit", &storage)
            j["storage"] = storage
            switch jres {
            case .success(let tj):
                let jo = try await encodeAll(tj)
                try? Self.writeFloats(jo.outs, to: config.out.appendingPathComponent("tower_out_\(rec.id)_jit.f32"))
                let bitEq = zip(a1.outs, jo.outs).filter { $0.0.map(\.bitPattern) == $0.1.map(\.bitPattern) }.count
                cmp["jit"] = ["load_wall_s": jwall, "load_s": tj.loadSeconds, "memory": memoryBrief(jmem),
                              "cache_bytes_added": cj1.bytes - cj0.bytes, "crop_ms": jo.ms, "bit_equal_crops": bitEq,
                              "vs_aot": Self.compareFloats(jo.outs.flatMap { $0 }, a1.outs.flatMap { $0 }),
                              "sha256": jo.outs.map { sha256Hex(of: $0) }]
            case .failure(let error):
                cmp["jit"] = ["error": "\(error)", "load_wall_s": jwall]
            }
            j["crops"] = crops.count
            j["crop_ms"] = ["first_run": a1.ms, "second_run": a2.ms]
            j["outputs_sha256"] = shas
            j["rerun_bit_equal"] = rerunEqual
            j["finite"] = finite
            j["all_zero_crop"] = allZero
            j["compare"] = cmp
            j["step"] = "done"
            j["thermal_end"] = DeviceInfo.thermal()
            j["pass"] = rerunEqual && finite && !allZero
            let jit = cmp["jit"] as? [String: Any]
            line("\(key): \(rec.id) \(crops.count) crops, ms \(a1.ms.map { f1($0) }.joined(separator: " ")) then "
                 + "\(a2.ms.map { f1($0) }.joined(separator: " ")) | re-run bit-equal \(rerunEqual), finite \(finite) | Mac sha256 "
                 + "\(cmp["mac_sha256_equal"] ?? "-")/\(cmp["mac_sha256_crops"] ?? "-") | Mac values \(Self.brief(cmp["mac_values"])) "
                 + "| JIT bit-equal \(jit?["bit_equal_crops"] ?? "-")/\(crops.count), JIT vs AOT \(Self.brief(jit?["vs_aot"]))")
        } catch {
            line("\(key): ERROR \(error)")
            j["error"] = "\(error)"
            j["error_detail"] = Self.errorRecord(error)
            j["pass"] = false
        }
        return j
    }

    // MARK: - the efr-less AOT

    /// probe_noefr: on the decider in use (meant for the decoder AOT without --expect-frequent-reshapes, which
    /// specializes once per new position length), one record (D1_PROBE_RECORD) D1_PROBE_PASSES times, direct, each pass
    /// under the memory sampler with the container's storage sized around it: every call's ms with its position length (a
    /// row's call c binds (c + 1) S positions) and whether the decider had run that length in this stage before (a new
    /// length's call = the phone's specialization), the rows' p bits and hidden sha256 against the first pass, the oracle
    /// and the Mac.
    func stageProbeNoEFR() async -> [String: Any] {
        let key = "probe_noefr"
        guard let d1, let fx = fixtures else { return noDecider(key) }
        guard let rec = fx.byID[config.probeRecord] else {
            line("\(key): ERROR no record \(config.probeRecord)")
            return ["error": "no record \(config.probeRecord)", "pass": false]
        }
        let S = d1.graph?.metadata.chunk ?? 16
        var j: [String: Any] = ["kind": graphKind ?? "", "asset": graphAsset ?? "", "record": rec.id, "chunk": S,
                                "passes_planned": config.probePasses, "stages_before": stageOrder,
                                "asset_package": graphAsset.flatMap { Self.aotPackage(URL(fileURLWithPath: $0)) } ?? [:]]
        var passes: [[String: Any]] = []
        var first: (bits: [String], hidden: [String])? = nil
        var ok = true, stopped = false
        line("\(key): \(rec.id) \(config.probePasses) times, direct, on the \(graphKind ?? "?") decider "
             + "\((graphAsset ?? "").split(separator: "/").last ?? "")")
        for p in 1...config.probePasses {
            if p > 1 && pastDeadline() {
                stopped = true
                line("\(key): stopping before pass \(p) at \(f1(elapsed())) s (\(diskStopped ?? "deadline"))")
                break
            }
            var storage: [String: Any] = [:]
            let s0 = DeviceInfo.containerWrittenBytes()
            let c0 = cacheSnapshot("before", &storage)
            let (res, mem, wall) = await sampled("\(key) pass \(p)") { try await self.decideRecord(rec, d1, fx, shared: false) }
            let c1 = cacheSnapshot("after", &storage)
            let s1 = DeviceInfo.containerWrittenBytes()
            var pj: [String: Any] = ["pass": p, "wall_s": wall, "memory": memoryBrief(mem),
                                     "coreai_cache_bytes_added": c1.bytes - c0.bytes, "container_bytes_written": s1 - s0,
                                     "storage": storage, "thermal": DeviceInfo.thermal(), "available_mb_after": DeviceInfo.availableMB()]
            switch res {
            case .success(let r):
                let t = r.json["direct"] as? [String: Any] ?? [:]
                let callMs = t["call_ms"] as? [Double] ?? []
                let rowsTokens = t["rows_tokens"] as? [Int] ?? []
                var calls: [[String: Any]] = []
                var idx = 0
                for (k, n) in rowsTokens.enumerated() {
                    for c in 0..<((n + S - 1) / S) {
                        let len = (c + 1) * S
                        let isNew = !lengthsSeen.contains(len)
                        lengthsSeen.insert(len)
                        calls.append(["row": k, "call": c, "position_length": len, "ms": idx < callMs.count ? callMs[idx] : -1,
                                      "new_length": isNew])
                        idx += 1
                    }
                }
                let bits = r.scores.map { $0.pBits.joined(separator: ",") }
                let hidden = r.scores.map(\.hiddenSHA)
                if first == nil { first = (bits, hidden) }
                let newCalls = calls.filter { $0["new_length"] as? Bool == true }
                let newMs = newCalls.compactMap { $0["ms"] as? Double }
                let oldMs = calls.filter { $0["new_length"] as? Bool != true }.compactMap { $0["ms"] as? Double }
                let pEq = bits == first!.bits, hEq = hidden == first!.hidden
                pj["calls"] = calls
                pj["calls_match_trace"] = idx == callMs.count
                pj["new_lengths"] = newCalls.compactMap { $0["position_length"] as? Int }
                pj["new_length_ms"] = newMs
                pj["seen_length_ms_median"] = median(oldMs)
                pj["latency_ms"] = t["latency_ms"] ?? -1
                pj["rows"] = r.json["rows"] ?? NSNull()
                pj["summary"] = Fixtures.summarize(r.scores)
                pj["p_bit_equal_pass1"] = pEq
                pj["hidden_sha256_equal_pass1"] = hEq
                ok = ok && pEq && hEq && idx == callMs.count && r.scores.allSatisfy { $0.finite && $0.idsEqualOracle }
                line("\(key) pass \(p): \(calls.count) calls in \(f1(wall)) s | new lengths \(newCalls.compactMap { $0["position_length"] as? Int }) "
                     + "at \(newMs.map { f1($0) }.joined(separator: " ")) ms, lengths run before: median \(f1(median(oldMs))) ms | "
                     + "p = pass 1 \(pEq), hidden = pass 1 \(hEq) | max|dp| \(f6(r.scores.map(\.maxAbsDp).max() ?? .nan)), argmax "
                     + "\(r.scores.filter(\.argmaxEqual).count)/\(r.scores.count), Mac p bits \(r.scores.filter { $0.macPBitEqual == true }.count) "
                     + "| peak footprint \(f1(mem["peak_footprint_mb"] as? Double ?? -1)) MB, least available "
                     + "\(f1(mem["min_available_mb"] as? Double ?? -1)) MB | container +\(mb(s1 - s0)) MB (Core AI cache "
                     + "+\(mb(c1.bytes - c0.bytes)) MB)")
            case .failure(let error):
                pj["error"] = "\(error)"
                pj["error_detail"] = Self.errorRecord(error)
                ok = false
                line("\(key) pass \(p): ERROR \(error)")
            }
            passes.append(pj)
            j["passes"] = passes
            writePartial(key, j)
            if pj["error"] != nil { break }
        }
        j["passes"] = passes
        j["deadline_stop"] = stopped
        j["pass"] = ok && !passes.isEmpty && (passes.count == config.probePasses || stopped)
        return j
    }

    // MARK: - p references across launches

    private func prefURL(_ kind: String) -> URL { config.out.appendingPathComponent("p_ref_\(kind).json") }

    private func loadPRef(kind: String) -> [String: [String: Any]] {
        guard let d = try? Data(contentsOf: prefURL(kind)),
              let r = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let rows = r["rows"] as? [String: [String: Any]] else { return [:] }
        return rows
    }

    /// Merges this stage's rows (p bits, hidden sha256, the row checks) into p_ref_<kind>.json.
    private func savePRef(kind: String, rows: [Fixtures.RowScore]) {
        guard kind == "jit" || kind == "aot", !rows.isEmpty else { return }
        var all = loadPRef(kind: kind)
        for s in rows {
            all[s.key] = ["p_bits": s.pBits, "hidden_sha256": s.hiddenSHA, "ids_equal_oracle": s.idsEqualOracle,
                          "finite": s.finite, "all_zero": s.allZero, "set": s.set, "run_id": config.runID]
        }
        do {
            try writeJSON(["kind": kind, "asset": graphAsset ?? "", "updated": Self.now(), "rows": all], to: prefURL(kind))
        } catch {
            line("ERROR writing p_ref_\(kind).json: \(error)")
        }
    }

    /// The bar over every row of `set` that p_ref_<kind>.json holds (this launch's and the earlier launches').
    private func unionSummary(kind: String, set: String) -> [String: Any] {
        guard let fx = fixtures else { return [:] }
        var rows: [Fixtures.RowScore] = []
        var runs = Set<String>()
        for (key, v) in loadPRef(kind: kind) where v["set"] as? String == set {
            let parts = key.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2, let o = fx.oracle[parts[0]]?.question(parts[1]) else { continue }
            rows.append(Fixtures.stored(key: key, set: set, pBits: v["p_bits"] as? [String] ?? [],
                                        hiddenSHA: v["hidden_sha256"] as? String ?? "", idsEqual: v["ids_equal_oracle"] as? Bool ?? false,
                                        finite: v["finite"] as? Bool ?? false, allZero: v["all_zero"] as? Bool ?? true, oracle: o,
                                        mac: fx.mac[parts[0]]?[parts[1]]))
            if let r = v["run_id"] as? String { runs.insert(r) }
        }
        var s = Fixtures.summarize(rows.sorted { $0.key < $1.key })
        s["runs"] = runs.sorted()
        s["rows_expected"] = fx.recordsOf(set: set).reduce(0) { $0 + (fx.oracle[$1.id]?.questions.count ?? 0) }
        return s
    }

    // MARK: - summaries

    /// The e2e sets of this process by decider kind, and the reset proof.
    private func overallSummary() -> [String: Any] {
        var s: [String: Any] = [:]
        for kind in Set(scoreKind.values).sorted() {
            let mine = scores.filter { scoreKind[$0.key] == kind }.map(\.value).sorted { $0.key < $1.key }
            var k: [String: Any] = ["all": Fixtures.summarize(mine)]
            for set in ["fixture", "image"] {
                let sel = mine.filter { $0.set == set }
                if !sel.isEmpty { k[set] = Fixtures.summarize(sel) }
            }
            s[kind] = k
        }
        if let r = stageResults["reset"] as? [String: Any] { s["reset"] = ["hidden_bit_equal": r["hidden_bit_equal"] ?? false,
                                                                           "p_bit_equal": r["p_bit_equal"] ?? false] }
        return s
    }

    // MARK: - thermal, sampling, records

    /// [t s, thermal, battery level, battery state, footprint MB, available MB]
    private func timelinePoint(_ t: Double) async -> [Any] {
        let b = await DeviceInfo.battery()
        BatteryCache.shared.set(b)
        return [(t * 10).rounded() / 10, DeviceInfo.thermal(), b.level, b.state, DeviceInfo.footprintMB(), DeviceInfo.availableMB()]
    }

    /// Polls ProcessInfo.thermalState every 5 s until it is nominal, at most D1_WAIT_NOMINAL seconds.
    func waitForNominal(_ key: String) async -> [String: Any] {
        let cap = config.waitNominalSeconds
        let before = DeviceInfo.thermal()
        let c0 = ContinuousClock.now
        var polls = 0
        if before != "nominal" { line("\(key): thermal \(before); waiting for nominal (5 s steps, cap \(Int(cap)) s)") }
        while DeviceInfo.thermal() != "nominal" && seconds(since: c0) < cap {
            try? await Task.sleep(for: .seconds(5))
            polls += 1
        }
        let waited = seconds(since: c0)
        let after = DeviceInfo.thermal()
        if before != "nominal" {
            line("\(key): thermal \(after) after \(f1(waited)) s" + (after == "nominal" ? "" : " (cap reached: running anyway)"))
        }
        return ["cap_s": cap, "state_before": before, "state_after": after, "waited_s": waited, "polls": polls,
                "reached_nominal": after == "nominal"]
    }

    /// `body` under a MemorySampler (100 ms): its result or error, the sampler's record, the wall seconds.
    func sampled<T>(_ what: String, _ body: () async throws -> T) async -> (Result<T, Error>, [String: Any], Double) {
        let sampler = MemorySampler(what, memoryLog: memoryLog, progress: { [sink] in sink.line($0) })
        sampler.start()
        let c0 = ContinuousClock.now
        let result: Result<T, Error>
        do {
            result = .success(try await body())
        } catch {
            result = .failure(error)
        }
        let wall = seconds(since: c0)
        return (result, sampler.stop(), wall)
    }

    /// An error as result.json keeps it: the text, the Swift type and case, and the NSError bridge.
    static func errorRecord(_ error: Error) -> [String: Any] {
        let ns = error as NSError
        return ["description": "\(error)", "reflecting": String(reflecting: error), "type": String(reflecting: type(of: error)),
                "ns_domain": ns.domain, "ns_code": ns.code,
                "ns_user_info": Dictionary(uniqueKeysWithValues: ns.userInfo.map { ($0.key, "\($0.value)") })]
    }

    /// The stage's record so far into result.json (a stage that is killed still leaves this much).
    func writePartial(_ key: String, _ j: [String: Any]) {
        var partial = j
        partial["partial"] = true
        let order = stageOrder
        let previous = stageResults[key]
        stageResults[key] = partial
        stageOrder.append(key)
        writeReport()
        stageOrder = order
        stageResults[key] = previous
    }

    func finish(ok: Bool, fatal: String?) -> Bool {
        report["status"] = fatal == nil ? "done" : "failed"
        if let f = fatal { report["fatal"] = f }
        report["pass"] = ok
        report["complete"] = deadlineCut.isEmpty
        report["deadline_cut"] = deadlineCut
        report["finished"] = Self.now()
        report["elapsed_s"] = elapsed()
        report["device_end"] = ["thermal": DeviceInfo.thermal(), "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
                                "footprint_mb": DeviceInfo.footprintMB(), "available_mb": DeviceInfo.availableMB(),
                                "free_gb": DeviceInfo.freeGB(config.out), "battery_level": BatteryCache.shared.get().level,
                                "battery_state": BatteryCache.shared.get().state]
        if let app = appSampler {
            let m = app.stop()
            report["memory_whole_run"] = memoryBrief(m)
            appSampler = nil
        }
        report["e2e_summary"] = overallSummary()
        let verdicts = stageOrder.map { k -> String in
            let s = stageResults[k] as? [String: Any] ?? [:]
            if (s["deadline_skipped"] as? Bool) == true { return "\(k)=SKIPPED(deadline)" }
            return "\(k)=\((s["skipped"] as? Bool) == true ? "SKIPPED" : ((s["pass"] as? Bool) == true ? "PASS" : "FAIL"))"
                + ((s["deadline_stop"] as? Bool) == true ? "(cut)" : "")
        }
        report["summary"] = verdicts
        if let e = report["e2e_summary"] as? [String: Any] {
            for kind in ["jit", "aot"] {
                if let k = e[kind] as? [String: Any], let all = k["all"] as? [String: Any] {
                    line(Fixtures.summaryLine("all e2e rows of this launch (\(kind))", all))
                }
            }
        }
        line("GATE_SUMMARY \(verdicts.joined(separator: " ")) VERDICT=\(ok ? "PASS" : "FAIL")"
             + (deadlineCut.isEmpty ? "" : " (cut by the deadline: \(deadlineCut.joined(separator: ", ")))")
             + (fatal.map { " (\($0))" } ?? ""))
        writeReport()
        let runs = config.out.appendingPathComponent("runs")
        try? FileManager.default.createDirectory(at: runs, withIntermediateDirectories: true)
        let safe = config.runID.replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: ":", with: "-")
        try? FileManager.default.copyItem(at: config.out.appendingPathComponent("result.json"),
                                          to: runs.appendingPathComponent("\(safe).json"))
        line("DONE \(config.runID)")
        sink.close()
        memoryLog?.close()
        setStage(ok ? "done: PASS" : "done: FAIL")
        return ok
    }

    func writeReport() {
        var r = report
        r["stages"] = stageResults
        r["stage_order"] = stageOrder
        r["updated"] = Self.now()
        r["elapsed_s"] = elapsed()
        do {
            try writeJSON(r, to: config.out.appendingPathComponent("result.json"))
        } catch {
            line("ERROR writing result.json: \(error)")
        }
    }

    nonisolated func line(_ s: String) { sink.line(s) }

    func bumpLaunchCount() -> Int {
        let url = config.out.appendingPathComponent("launch_count")
        let n = (try? String(contentsOf: url, encoding: .utf8)).flatMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) } ?? 0
        try? "\(n + 1)\n".write(to: url, atomically: true, encoding: .utf8)
        return n + 1
    }

    func elapsed() -> Double { seconds(since: t0) }

    static func now() -> String { ISO8601DateFormatter().string(from: Date()) }
}
