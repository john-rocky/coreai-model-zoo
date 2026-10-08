// AudioPreprocess — a 16 kHz mono clip -> the audio graph's five inputs at its clip bucket (conversion/d1_omni/host.py
// §6 and mel_host.py: the publisher's waveform() and MelFrontend, in mel_host.mel_numpy's form, the Swift spec):
//
//   samples   a RIFF/WAVE file of 16-bit PCM, mono, 16 kHz, read as int16 (soundfile's `read(dtype="int16")`); no
//             resampling, no channel mix (the publisher's waveform() takes 16 kHz mono as is)
//   waveform  cut to the first 480,000 samples (30 s); x / 32768 in float32 (exact); zero-padded at the end to 8,000
//             (0.5 s). n = the length after this step; columns = 1 + n // 160, frames = n // 160 (the valid columns)
//   mel       in float64, cast to float32 at the end (mel_numpy, float64 = the spec, round 6):
//             preemphasis y[0] = x[0], y[i] = x[i] - 0.97 x[i-1]; 256 zeros on both sides; column t = the 512 samples
//             from t * 160 times the window (Hann(400, periodic=False) = 0.5 - 0.5 cos(2 pi k / 399) centred, 56 zeros
//             each side); the 512-point real DFT, bins 0..256 (vDSP's double FFT); power = sqrt(re^2 + im^2)^2;
//             mel = filterbank [128, 257] (the bundle's mel_filters_128x257_f32.bin, as float64) @ power^T, one
//             cblas_dgemm as NumPy's matmul calls it; log(mel + 2^-24); per mel row over the valid columns: mean =
//             NumPy's pairwise sum / frames, std = sqrt(pairwise sum of (x - mean)^2 / (frames - 1)) (nan -> 0),
//             x = (x - mean) / (std + 1e-5); columns t >= frames 0.0
//   bucket    the smallest of 5 / 10 / 20 / 30 s whose F = 1 + 100 * sec holds the clip's columns; the mel is
//             zero-padded to F
//   masks     mask_f / mask_f2 / mask_f4 / mask_t: 1.0 on t < l0, l1, l2, l3 with l0 = frames and l(k+1) =
//             (lk - 1) // 2 + 1 (ConvSubsampling's lengths); P = l3 prefix rows (MediaLength.audioPrefixLength)
//
// What differs from NumPy: the FFT (vDSP's, not pocketfft's); every other step is the same arithmetic in the same
// order (NumPy's float64 log and cos are libm's, its matmul is Accelerate's dgemm), so the mel matches mel_numpy to
// the float32 rounding of a float64 difference near 1e-16 (round 9 measures it).

import Accelerate
import Foundation

/// One clip bucket's static lengths (host.audio_bucket_shapes).
public struct AudioBucket: Sendable, Equatable, Hashable {
    public let seconds: Int
    public let F: Int
    public let F2: Int
    public let F4: Int
    public let T: Int

    public init(seconds: Int) {
        self.seconds = seconds
        F = 1 + seconds * AudioPreprocess.sampleRate / AudioPreprocess.hop
        F2 = MediaLength.subsample(F)
        F4 = MediaLength.subsample(F2)
        T = MediaLength.subsample(F4)
    }

    public static let all = AudioPreprocess.bucketSeconds.map(AudioBucket.init(seconds:))
}

/// One clip's audio-graph inputs at a bucket, flat row-major.
public struct AudioInputs: Sendable {
    public let bucket: AudioBucket
    /// [1, 128, F]
    public let mel: [Float]
    public let maskF: [Float]
    public let maskF2: [Float]
    public let maskF4: [Float]
    public let maskT: [Float]
    /// n after waveform(), the valid columns (n // 160), the mel's own columns (1 + n // 160), P
    public let samples: Int
    public let frames: Int
    public let columns: Int
    public let prefixRows: Int

    public init(bucket: AudioBucket, mel: [Float], maskF: [Float], maskF2: [Float], maskF4: [Float], maskT: [Float],
                samples: Int, frames: Int, columns: Int, prefixRows: Int) {
        self.bucket = bucket
        self.mel = mel
        self.maskF = maskF
        self.maskF2 = maskF2
        self.maskF4 = maskF4
        self.maskT = maskT
        self.samples = samples
        self.frames = frames
        self.columns = columns
        self.prefixRows = prefixRows
    }
}

public enum AudioPreprocess {
    public static let sampleRate = 16000, minSamples = 8000, maxSeconds = 30, hop = 160
    public static let nFFT = 512, window = 400, features = 128, bins = 257
    public static let preemphasis = 0.97
    public static let logGuard = 0x1p-24
    public static let normEps = 1e-5
    public static let bucketSeconds = [5, 10, 20, 30]
    public static let filterbankFile = "mel_filters_128x257_f32.bin"

    // MARK: samples

    /// The int16 samples of a RIFF/WAVE file: PCM (or WAVE_FORMAT_EXTENSIBLE with the PCM subformat), 16 bits, mono,
    /// 16 kHz; anything else throws (the publisher does not resample or mix).
    public static func samples(wav data: Data, name: String = "audio") throws -> [Int16] {
        let b = [UInt8](data)
        func u16(_ i: Int) -> Int { Int(b[i]) | Int(b[i + 1]) << 8 }
        func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }
        guard b.count >= 12, b[0..<4].elementsEqual("RIFF".utf8), b[8..<12].elementsEqual("WAVE".utf8) else {
            throw D1OmniError.request("\(name): not a RIFF/WAVE file")
        }
        var i = 12
        var format: (tag: Int, channels: Int, rate: Int, bits: Int)? = nil
        while i + 8 <= b.count {
            let id = String(decoding: b[i..<(i + 4)], as: UTF8.self), size = u32(i + 4), body = i + 8
            guard body + size <= b.count || id == "data" else { throw D1OmniError.request("\(name): chunk \(id) past the end") }
            if id == "fmt " {
                guard size >= 16 else { throw D1OmniError.request("\(name): short fmt chunk") }
                var tag = u16(body)
                if tag == 0xFFFE, size >= 40 { tag = u16(body + 24) }   // WAVE_FORMAT_EXTENSIBLE: the subformat GUID's first two bytes
                format = (tag, u16(body + 2), u32(body + 4), u16(body + 14))
            } else if id == "data" {
                guard let f = format else { throw D1OmniError.request("\(name): data before fmt") }
                guard f.tag == 1, f.bits == 16, f.channels == 1, f.rate == sampleRate else {
                    throw D1OmniError.request("\(name): format \(f.tag), \(f.bits) bits, \(f.channels) channels, \(f.rate) Hz; "
                                              + "the model takes 16-bit PCM mono at 16 kHz")
                }
                let end = min(body + size, b.count)
                let n = (end - body) / 2
                return (0..<n).map { Int16(bitPattern: UInt16(b[body + 2 * $0]) | UInt16(b[body + 2 * $0 + 1]) << 8) }
            }
            i = body + size + (size & 1)
        }
        throw D1OmniError.request("\(name): no data chunk")
    }

    public static func samples(contentsOf url: URL) throws -> [Int16] {
        try samples(wav: Data(contentsOf: url), name: url.lastPathComponent)
    }

    /// waveform(): cut to 30 s, / 32768 in float32, zero-padded to 0.5 s.
    public static func waveform(_ s: [Int16]) -> [Float] {
        var x = s.prefix(maxSeconds * sampleRate).map { Float($0) / 32768 }
        if x.count < minSamples { x += [Float](repeating: 0, count: minSamples - x.count) }
        return x
    }

    // MARK: bucket and masks

    /// host.audio_bucket_for: the smallest bucket whose F holds 1 + n // 160 (n after waveform()).
    public static func bucket(forSamples raw: Int) throws -> AudioBucket {
        let n = max(min(raw, maxSeconds * sampleRate), minSamples)
        guard let b = AudioBucket.all.first(where: { 1 + n / hop <= $0.F }) else { throw D1OmniError.request("\(raw) samples fit no bucket") }
        return b
    }

    static func mask(_ count: Int, ones: Int) -> [Float] { (0..<count).map { $0 < ones ? 1 : 0 } }

    // MARK: mel

    /// NumPy's float64 add.reduce of a contiguous run (the identity 0 plus pairwise_sum, loops_utils.h.src): below 8
    /// values a left-to-right sum from -0.0; up to 128 eight partial sums combined ((r0 + r1) + (r2 + r3)) +
    /// ((r4 + r5) + (r6 + r7)), then the rest left to right; above, the two halves (n / 2 rounded down to a multiple of
    /// 8). Round 9 checked the order against numpy 2.3.5 / 2.5.3 on 5,120 rows of 50..3,000 values.
    static func pairwiseSum(_ a: UnsafeBufferPointer<Double>, _ start: Int, _ n: Int) -> Double {
        if n < 8 {
            var res = -0.0
            for i in 0..<n { res += a[start + i] }
            return res
        } else if n <= 128 {
            var r = (a[start], a[start + 1], a[start + 2], a[start + 3], a[start + 4], a[start + 5], a[start + 6], a[start + 7])
            var i = 8
            while i < n - (n % 8) {
                r.0 += a[start + i]; r.1 += a[start + i + 1]; r.2 += a[start + i + 2]; r.3 += a[start + i + 3]
                r.4 += a[start + i + 4]; r.5 += a[start + i + 5]; r.6 += a[start + i + 6]; r.7 += a[start + i + 7]
                i += 8
            }
            var res = ((r.0 + r.1) + (r.2 + r.3)) + ((r.4 + r.5) + (r.6 + r.7))
            while i < n {
                res += a[start + i]
                i += 1
            }
            return res
        }
        var n2 = n / 2
        n2 -= n2 % 8
        return pairwiseSum(a, start, n2) + pairwiseSum(a, start + n2, n - n2)
    }

    /// The 512-point frame window: Hann(400, periodic=False) centred, 56 zeros each side (mel_host.hann_window_numpy).
    public static let hann: [Double] = {
        var w = [Double](repeating: 0, count: nFFT)
        let offset = (nFFT - window) / 2
        for k in 0..<window { w[offset + k] = 0.5 - 0.5 * cos(2.0 * Double.pi * Double(k) / Double(window - 1)) }
        return w
    }()

    /// mel_host.mel_numpy on waveform() samples -> (mel [128][columns] float32, frames).
    public static func mel(_ x: [Float], filterbank: [Float]) throws -> (mel: [Float], frames: Int, columns: Int) {
        guard filterbank.count == features * bins else { throw D1OmniError.bundle("filterbank of \(filterbank.count) values, not 128 x 257") }
        let n = x.count
        let frames = n / hop, columns = 1 + n / hop
        // preemphasis in float64, padded by 256 zeros on both sides
        var padded = [Double](repeating: 0, count: n + nFFT)
        let half = nFFT / 2
        padded[half] = Double(x[0])
        for i in 1..<n { padded[half + i] = Double(x[i]) - preemphasis * Double(x[i - 1]) }
        // power spectrum [columns][257]: vDSP's double real FFT of each windowed column
        let log2n = vDSP_Length(9)
        guard let setup = vDSP_create_fftsetupD(log2n, FFTRadix(kFFTRadix2)) else { throw D1OmniError.contract("vDSP FFT setup") }
        defer { vDSP_destroy_fftsetupD(setup) }
        var power = [Double](repeating: 0, count: columns * bins)
        var frame = [Double](repeating: 0, count: nFFT)
        var re = [Double](repeating: 0, count: nFFT / 2), im = [Double](repeating: 0, count: nFFT / 2)
        for t in 0..<columns {
            for k in 0..<nFFT { frame[k] = padded[t * hop + k] * hann[k] }
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPDoubleSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    frame.withUnsafeBytes { raw in
                        vDSP_ctozD(raw.bindMemory(to: DSPDoubleComplex.self).baseAddress!, 2, &split, 1, vDSP_Length(nFFT / 2))
                    }
                    vDSP_fft_zripD(setup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                }
            }
            // vDSP returns 2 X[k] for k in 1..255, 2 X[0] in re[0] and 2 X[256] in im[0]
            let row = t * bins
            func put(_ k: Int, _ r: Double, _ i: Double) {
                let m = (r * r + i * i).squareRoot()
                power[row + k] = m * m
            }
            put(0, re[0] / 2, 0)
            put(nFFT / 2, im[0] / 2, 0)
            for k in 1..<(nFFT / 2) { put(k, re[k] / 2, im[k] / 2) }
        }
        // mel = filterbank @ power^T: NumPy's matmul = cblas_dgemm(RowMajor, NoTrans, Trans, 128, columns, 257)
        let fb = filterbank.map(Double.init)
        var melD = [Double](repeating: 0, count: features * columns)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(features), Int32(columns), Int32(bins), 1.0,
                    fb, Int32(bins), power, Int32(bins), 0.0, &melD, Int32(columns))
        for i in 0..<melD.count { melD[i] = log(melD[i] + logGuard) }
        // per-row normalisation over the valid columns
        var out = [Float](repeating: 0, count: features * columns)
        var dev = [Double](repeating: 0, count: frames)
        melD.withUnsafeBufferPointer { m in
            for r in 0..<features {
                let row = r * columns
                let mean = (0.0 + pairwiseSum(m, row, frames)) / Double(frames)
                for t in 0..<frames {
                    let d = m[row + t] - mean
                    dev[t] = d * d
                }
                var std = dev.withUnsafeBufferPointer { (0.0 + pairwiseSum($0, 0, frames)) / Double(frames - 1) }.squareRoot()
                if std.isNaN { std = 0 }
                for t in 0..<frames { out[row + t] = Float((m[row + t] - mean) / (std + normEps)) }
            }
        }
        return (out, frames, columns)
    }

    // MARK: inputs

    /// One clip -> the audio graph's inputs at `bucket` (default: the smallest that holds it), from the bundle's
    /// filterbank.
    public static func inputs(samples s: [Int16], filterbank: [Float], bucket chosen: AudioBucket? = nil) throws -> AudioInputs {
        let x = waveform(s)
        let b = try chosen ?? bucket(forSamples: s.count)
        let (m, frames, columns) = try mel(x, filterbank: filterbank)
        guard columns <= b.F else { throw D1OmniError.request("a clip of \(columns) mel columns does not fit the \(b.seconds) s bucket (\(b.F))") }
        var mel = [Float](repeating: 0, count: features * b.F)
        for r in 0..<features { mel.replaceSubrange((r * b.F)..<(r * b.F + columns), with: m[(r * columns)..<((r + 1) * columns)]) }
        let l1 = MediaLength.subsample(frames), l2 = MediaLength.subsample(l1), l3 = MediaLength.subsample(l2)
        guard l3 == MediaLength.audioPrefixLength(samples: s.count) else {
            throw D1OmniError.contract("P \(l3) != MediaLength.audioPrefixLength \(MediaLength.audioPrefixLength(samples: s.count))")
        }
        return AudioInputs(bucket: b, mel: mel, maskF: mask(b.F, ones: frames), maskF2: mask(b.F2, ones: l1),
                           maskF4: mask(b.F4, ones: l2), maskT: mask(b.T, ones: l3), samples: x.count, frames: frames,
                           columns: columns, prefixRows: l3)
    }

    /// The bundle's filterbank [128][257] float32 (mel_filters_128x257_f32.bin, little-endian).
    public static func filterbank(contentsOf url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count == features * bins * 4 else { throw D1OmniError.bundle("\(url.path): \(data.count) bytes, not 128x257 float32") }
        return data.withUnsafeBytes { raw in
            (0..<(features * bins)).map { Float(bitPattern: UInt32(littleEndian: raw.load(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }
}
