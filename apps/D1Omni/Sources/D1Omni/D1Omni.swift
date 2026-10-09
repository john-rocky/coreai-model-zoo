// D1Omni — a SystemOne request -> the SystemOne response on d1-omni-600M's decision graphs, in host.py's order:
//
//   let d1 = try await D1Omni(folders: D1Omni.folders(macos: dir))        // the .aimodel of every bucket (JIT)
//   let response = try await d1.systemOne(state: state, questions: questions)
//
//   request ──Prompt.rows (the publisher's encode, per mode)──> one row per question
//           ──GraphInputs at bucket_for(P + n)──> the bucket's graph (one call per row) ──> scores[P + markers]
//           ──Readout.probabilities (÷ T on text rows, float32 softmax, noul flipped)──> Readout.response
//
// Every bucket folder holds metadata.json (the decision block: the graph's contract, the nine token ids, the
// temperatures, max_length / image_text_length / audio_text_length / min_text_positions) and the bucket's `.aimodel`;
// the tokenizer folder holds tokenizer.json + tokenizer_config.json. What differs from the contract fails at load. A
// bucket's graph loads on its first row (or every one at init with `preload`). `assets` replaces a bucket's `.aimodel`
// with another file to load (the Mac's AOT `.aimodelc`, compiled from the same `.aimodel`).
//
// Text is round 8. Images and audio are round 9: `systemOne(state:questions:images:)` decodes each image, cuts and
// resizes its crops, runs the vision graph once per crop and concatenates the prefixes (ImagePreprocess, VisionGraph);
// `systemOne(state:questions:audio:)` reads the clip, makes the mel and the masks at its bucket and runs that bucket's
// audio graph (AudioPreprocess, AudioGraph); the media folders are `vision-<precision>/` (with position_table.f32) and
// `audio-<precision>-<sec>s/` (with mel_filters_128x257_f32.bin) beside the decision buckets (`MediaFolders`). The
// rows, the graph inputs and the readout are the same for every mode. One decision at a time per instance.

import CoreAI
import Foundation

public final class D1Omni: @unchecked Sendable {
    public struct Bucket: Sendable {
        public let length: Int
        public let folder: URL
        /// what loads: `assets[L]`, or the folder's `.aimodel` (metadata.json assets.main)
        public let asset: URL
    }

    /// One row's decision: the marker logits (float32, the graph's), the reported probabilities, the bucket, seconds.
    public struct Decision: Sendable {
        public let row: Row
        public let bucket: Int
        public let logits: [Float]
        public let probabilities: [Float]
        /// graph inputs, the call, the readout
        public let seconds: (inputs: Double, graph: Double, readout: Double)
    }

    public let buckets: [Int: Bucket]
    public let lengths: [Int]
    public let config: D1Config
    public let tokenizer: D1Tokenizer
    public let options: SpecializationOptions
    public let media: MediaFolders?
    private var graphs: [Int: DecisionGraph] = [:]
    private var visionGraph: VisionGraph? = nil
    private var audioGraphs: [Int: AudioGraph] = [:]
    private var positionTable: [Float]? = nil
    private var melFilterbank: [Float]? = nil

    /// The media bundles beside the decision buckets: the vision folder and one audio folder per clip bucket, each
    /// with metadata.json naming its `.aimodel` (`assets.main`); `assets` replaces a graph's file ("vision",
    /// "audio-<sec>") with another one to load (the Mac's AOT `.aimodelc`).
    public struct MediaFolders: Sendable {
        public let vision: URL?
        public let audio: [Int: URL]
        public let assets: [String: URL]

        public init(vision: URL?, audio: [Int: URL], assets: [String: URL] = [:]) {
            self.vision = vision
            self.audio = audio
            self.assets = assets
        }

        /// `vision-<precision>/` and `audio-<precision>-<sec>s/` under a `macos/` directory (the Hugging Face layout and
        /// the conversion's work folder name them alike).
        public static func find(macos dir: URL, precision: String = "fp16", assets: [String: URL] = [:]) -> MediaFolders {
            let fm = FileManager.default
            let v = dir.appendingPathComponent("vision-\(precision)")
            var a: [Int: URL] = [:]
            for sec in AudioPreprocess.bucketSeconds {
                let f = dir.appendingPathComponent("audio-\(precision)-\(sec)s")
                if fm.fileExists(atPath: f.appendingPathComponent("metadata.json").path) { a[sec] = f }
            }
            return MediaFolders(vision: fm.fileExists(atPath: v.appendingPathComponent("metadata.json").path) ? v : nil,
                                audio: a, assets: assets)
        }

        /// The file a media graph loads: `assets[key]`, or the folder's metadata.json `assets.main`.
        public func asset(_ key: String, folder: URL) throws -> URL {
            if let u = assets[key] { return u }
            let j = try JSONParser.parse(Data(contentsOf: folder.appendingPathComponent("metadata.json")))
            guard let main = j["assets"]?["main"]?.string else { throw D1OmniError.bundle("\(folder.path): no assets.main") }
            return folder.appendingPathComponent(main)
        }
    }

    /// The bucket folders under a `macos/` directory: `decide-<precision>-L<L>` (the Hugging Face layout) or
    /// `<precision>-L<L>` (the conversion's work folder).
    public static func folders(macos dir: URL, precision: String = "fp16") throws -> [Int: URL] {
        var out: [Int: URL] = [:]
        for name in try FileManager.default.contentsOfDirectory(atPath: dir.path) {
            for prefix in ["decide-\(precision)-L", "\(precision)-L"] where name.hasPrefix(prefix) {
                if let L = Int(name.dropFirst(prefix.count)), out[L] == nil {
                    out[L] = dir.appendingPathComponent(name)
                }
            }
        }
        if out.isEmpty { throw D1OmniError.bundle("\(dir.path): no decide-\(precision)-L<L> or \(precision)-L<L> folder") }
        return out
    }

    /// `folders`: L -> the bucket's folder; `assets`: L -> the file to load instead of the folder's `.aimodel`;
    /// `tokenizerFolder`: nil = the smallest bucket's `tokenizer/`.
    public init(folders: [Int: URL], assets: [Int: URL] = [:], tokenizerFolder: URL? = nil,
                options: SpecializationOptions = DecisionGraph.gpuOptions, preload: Bool = false,
                media: MediaFolders? = nil) async throws {
        self.media = media
        var buckets: [Int: Bucket] = [:]
        var config: D1Config? = nil
        var tokenIDs: [String: Int]? = nil
        for (L, folder) in folders.sorted(by: { $0.key < $1.key }) {
            let url = folder.appendingPathComponent("metadata.json")
            let j = try JSONParser.parse(Data(contentsOf: url))
            guard let d = j["decision"], let main = j["assets"]?["main"]?.string, d["seq_len"]?.intValue == L,
                  let ids = d["token_ids"]?.members, let temps = d["temperatures"]?.members,
                  let maxLength = d["max_length"]?.intValue, let image = d["image_text_length"]?.intValue,
                  let audio = d["audio_text_length"]?.intValue, let minText = d["min_text_positions"]?.intValue,
                  d["prefix_hidden"]?.intValue == DecisionGraph.hidden
            else { throw D1OmniError.bundle("\(url.path): no decision block for L = \(L) (seq_len, token_ids, temperatures, lengths)") }
            let c = D1Config(maxLength: maxLength, imageTextLength: image, audioTextLength: audio, minTextPositions: minText,
                             temperatures: Dictionary(uniqueKeysWithValues: try temps.map { m in
                                 guard let t = m.value.double else { throw D1OmniError.bundle("\(url.path): temperature \(m.key)") }
                                 return (m.key, t)
                             }))
            let t = Dictionary(uniqueKeysWithValues: try ids.map { m in
                guard let i = m.value.intValue else { throw D1OmniError.bundle("\(url.path): token id \(m.key)") }
                return (m.key, i)
            })
            if let config, config.temperatures != c.temperatures || config.maxLength != c.maxLength
                || config.imageTextLength != c.imageTextLength || config.audioTextLength != c.audioTextLength
                || config.minTextPositions != c.minTextPositions {
                throw D1OmniError.bundle("\(url.path): the decision block differs from the other buckets'")
            }
            if let tokenIDs, tokenIDs != t { throw D1OmniError.bundle("\(url.path): token ids differ from the other buckets'") }
            config = c
            tokenIDs = t
            buckets[L] = Bucket(length: L, folder: folder, asset: assets[L] ?? folder.appendingPathComponent(main))
        }
        guard let config, let tokenIDs, let first = buckets.keys.min() else { throw D1OmniError.bundle("no bucket folders") }
        self.buckets = buckets
        self.lengths = buckets.keys.sorted()
        self.config = config
        self.options = options
        tokenizer = try await D1Tokenizer.load(folder: tokenizerFolder ?? buckets[first]!.folder.appendingPathComponent("tokenizer"),
                                               expected: tokenIDs)
        if preload { for L in lengths { _ = try await graph(L) } }
    }

    /// The bucket's graph, loaded on first use.
    public func graph(_ L: Int) async throws -> DecisionGraph {
        if let g = graphs[L] { return g }
        guard let b = buckets[L] else { throw D1OmniError.graphLimit("no bucket of length \(L) (buckets \(lengths))") }
        let g = try await DecisionGraph(contentsOf: b.asset, length: L, options: options)
        graphs[L] = g
        return g
    }

    /// The loaded graphs, by length.
    public var loaded: [Int: DecisionGraph] { graphs }

    /// host.py `request_rows` with this bundle's config.
    public func rows(state: JSONValue?, questions: JSONValue, mode: Mode = .text, prefixLength: Int = 0) throws -> [Row] {
        try Prompt.rows(tokenizer, state: state, questions: questions, mode: mode, prefixLength: prefixLength, config: config)
    }

    /// host.py `bucket_for` over this bundle's buckets.
    public func bucket(for row: Row) throws -> Int {
        guard let L = GraphInputs.bucket(positions: row.positions, buckets: lengths) else {
            throw D1OmniError.graphLimit("question '\(row.qid)': \(row.positions) positions, over the largest bucket \(lengths.last ?? 0)")
        }
        return L
    }

    /// One row through its bucket's graph (`prefix`: the media prefix [P * 1024] when P > 0) and the readout.
    public func decide(_ row: Row, prefix: [Float]? = nil, bucket: Int? = nil) async throws -> Decision {
        let L = try bucket ?? self.bucket(for: row)
        let g = try await graph(L)
        let t0 = ContinuousClock.now
        let x = try GraphInputs(row: row, length: L)
        let t1 = ContinuousClock.now
        let s = try await g.scores(x, prefix: prefix)
        let t2 = ContinuousClock.now
        let logits = x.markers.map { s[$0] }
        let p = Readout.probabilities(logits: logits, question: row.question, calibrate: row.calibrate, config: config)
        let t3 = ContinuousClock.now
        return Decision(row: row, bucket: L, logits: logits, probabilities: p,
                        seconds: (Self.seconds(t0, t1), Self.seconds(t1, t2), Self.seconds(t2, t3)))
    }

    static func seconds(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
        let d = b - a
        return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
    }

    /// A text request's response (`system_one(state, questions)`): `questions` a {name: question} object.
    public func systemOne(state: JSONValue?, questions: JSONValue) async throws -> JSONValue {
        let rows = try self.rows(state: state, questions: questions)
        var probs: [[Float]] = []
        for r in rows { probs.append(try await decide(r).probabilities) }
        return Readout.response(rows: rows, probabilities: probs)
    }

    /// An image or audio request's response, the media prefix [P * 1024] given (the vision / audio graphs' rows, in
    /// request order; round 9 builds them in Swift).
    public func systemOne(state: JSONValue?, questions: JSONValue, mode: Mode, prefix: [Float]) async throws -> JSONValue {
        guard mode != .text, prefix.count % DecisionGraph.hidden == 0, !prefix.isEmpty else {
            throw D1OmniError.request("a media request needs its prefix (P x \(DecisionGraph.hidden) values)")
        }
        let rows = try self.rows(state: state, questions: questions, mode: mode, prefixLength: prefix.count / DecisionGraph.hidden)
        var probs: [[Float]] = []
        for r in rows { probs.append(try await decide(r, prefix: prefix).probabilities) }
        return Readout.response(rows: rows, probabilities: probs)
    }

    // MARK: - media (round 9)

    /// The vision graph, loaded on first use (with the bundle's position table).
    public func vision() async throws -> VisionGraph {
        if let g = visionGraph { return g }
        guard let folder = media?.vision else { throw D1OmniError.bundle("no vision folder (MediaFolders.vision)") }
        let g = try await VisionGraph(contentsOf: try media!.asset("vision", folder: folder), options: options)
        visionGraph = g
        return g
    }

    /// The vision bundle's position table [16][16][768].
    public func visionPositionTable() throws -> [Float] {
        if let t = positionTable { return t }
        guard let folder = media?.vision else { throw D1OmniError.bundle("no vision folder (MediaFolders.vision)") }
        let t = try ImagePreprocess.positionTable(contentsOf: folder.appendingPathComponent("position_table.f32"))
        positionTable = t
        return t
    }

    /// The audio graph of one clip bucket, loaded on first use.
    public func audio(_ bucket: AudioBucket) async throws -> AudioGraph {
        if let g = audioGraphs[bucket.seconds] { return g }
        guard let folder = media?.audio[bucket.seconds] else { throw D1OmniError.bundle("no audio folder for the \(bucket.seconds) s bucket") }
        let g = try await AudioGraph(contentsOf: try media!.asset("audio-\(bucket.seconds)", folder: folder), bucket: bucket,
                                     options: options)
        audioGraphs[bucket.seconds] = g
        return g
    }

    /// The audio bundles' filterbank [128][257] (every bucket's file is the same; the first bucket's is read).
    public func audioFilterbank() throws -> [Float] {
        if let f = melFilterbank { return f }
        guard let folder = media?.audio.sorted(by: { $0.key < $1.key }).first?.value else { throw D1OmniError.bundle("no audio folder") }
        let f = try AudioPreprocess.filterbank(contentsOf: folder.appendingPathComponent(AudioPreprocess.filterbankFile))
        melFilterbank = f
        return f
    }

    /// Every crop's inputs of every image, in request order (host.image_crops_inputs with the NumPy forms).
    public func imageInputs(_ images: [RGBImage]) throws -> [CropInputs] {
        let table = try visionPositionTable()
        return try images.flatMap { try ImagePreprocess.crops($0).map { try ImagePreprocess.inputs($0, table: table) } }
    }

    /// The images' prefix [P * 1024]: each crop's first (ph / 2)(pw / 2) rows, crops in order, images in order.
    public func imagePrefix(_ crops: [CropInputs]) async throws -> [Float] {
        let g = try await vision()
        var prefix: [Float] = []
        for c in crops { prefix += try await g.prefix(c) }
        return prefix
    }

    /// The clip's inputs at its bucket (host.audio_inputs with mel_numpy).
    public func audioInputs(samples: [Int16]) throws -> AudioInputs {
        try AudioPreprocess.inputs(samples: samples, filterbank: try audioFilterbank())
    }

    /// The clip's prefix [P * 1024] from its bucket's graph.
    public func audioPrefix(_ x: AudioInputs) async throws -> [Float] {
        try await audio(x.bucket).prefix(x)
    }

    /// An image request's response: the images (files, in request order) -> their prefix -> the image rows.
    public func systemOne(state: JSONValue?, questions: JSONValue, images: [URL]) async throws -> JSONValue {
        let decoded = try images.map { try ImagePreprocess.decode(contentsOf: $0) }
        let prefix = try await imagePrefix(try imageInputs(decoded))
        let p = try MediaLength.imagePrefixLength(decoded.map { ($0.width, $0.height) })
        guard prefix.count == p * DecisionGraph.hidden else { throw D1OmniError.contract("an image prefix of \(prefix.count / DecisionGraph.hidden) rows, P = \(p)") }
        return try await systemOne(state: state, questions: questions, mode: .image, prefix: prefix)
    }

    /// An audio request's response: the clip (a 16 kHz mono 16-bit WAV) -> its prefix -> the audio rows.
    public func systemOne(state: JSONValue?, questions: JSONValue, audio: URL) async throws -> JSONValue {
        try await systemOne(state: state, questions: questions, samples: try AudioPreprocess.samples(contentsOf: audio))
    }

    /// An audio request's response from the clip's int16 samples.
    public func systemOne(state: JSONValue?, questions: JSONValue, samples: [Int16]) async throws -> JSONValue {
        let x = try audioInputs(samples: samples)
        let prefix = try await audioPrefix(x)
        return try await systemOne(state: state, questions: questions, mode: .audio, prefix: prefix)
    }
}
