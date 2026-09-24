// N3DMel / N3DEmbedder — ../mel_frontend.py in Swift: the log-mel features of transformers'
// NemotronAsrStreamingFeatureExtractor, the model card's streaming chunker, and the embedder
// (8-frame stacking + Linear 1024 -> 512) that runs on the host.

import Accelerate
import Foundation

/// One streaming chunk of the audio (mel_frontend.stream_chunks): samples [start, end), center-padded
/// when it is the first chunk. `end` of the last chunk is the audio length.
public struct N3DChunk: Sendable, Equatable {
    public let start: Int
    public let end: Int
    public let isFirst: Bool
    public let isLast: Bool
}

/// log_mel(audio, center) -> [F, 128] float32, frame-major:
///   1. preemphasis 0.97 on the chunk, its first sample kept: y[0] = x[0], y[n] = x[n] - 0.97 x[n-1]
///   2. frames of 512 samples every 160: center pads 256 zeros on both sides, otherwise no padding
///   3. window: the shipped Hann(400) (hann_window_400.f32le) at [56, 456) of the 512 frame, zeros elsewhere
///   4. power = sqrt(re^2 + im^2)^2 of the 512-point DFT (257 bins), float32
///   5. mel = power @ filters.T (librosa slaney [128, 257]); log(mel + 2^-24); no normalization
///   6. valid frames: center: floor(n / 160); not center: floor((n - 512) / 160) + 1
/// The DFT is a matmul of the float32 windowed samples against cos/sin tables (angle reduced exactly,
/// (k * j) mod 512), summed in Double and rounded to float32 re / im -- what NumPy's complex64 rfft
/// returns up to its own rounding (a float32 sum put the log-mel up to 1.2e-4 away from NumPy's in
/// near-silent cells). Only the 400 window taps enter the sum (the other 112 frame samples are zero).
/// Frames go through in blocks of 256 so a long recording needs no frame-sized Double buffers.
public struct N3DMel: Sendable {
    public static let sampleRate = 16_000
    public static let nFFT = 512, winLength = 400, hop = 160, nMels = 128, stack = 8
    public static let nFreq = nFFT / 2 + 1                      // 257
    public static let preemphasis: Float = 0.97
    public static let logGuard: Float = 0x1p-24
    static let winOffset = (nFFT - winLength) / 2              // 56
    static let block = 256

    let window: [Float]          // [400]
    let cosBasis: [Double]       // [400, 257]  cos(2π k (i + 56) / 512)
    let sinBasis: [Double]       // [400, 257]  sin(2π k (i + 56) / 512)
    let filtersT: [Float]        // [257, 128]

    public init(melFilters: [Float], hannWindow: [Float]) {
        precondition(melFilters.count == Self.nMels * Self.nFreq, "mel filters must be [128, 257]")
        precondition(hannWindow.count == Self.winLength, "window must be [400]")
        window = hannWindow
        let K = Self.winLength, F = Self.nFreq
        var c = [Double](repeating: 0, count: K * F)
        var s = [Double](repeating: 0, count: K * F)
        for i in 0..<K {
            let j = i + Self.winOffset
            for k in 0..<F {
                let a = 2.0 * Double.pi * Double((k * j) % Self.nFFT) / Double(Self.nFFT)
                c[i * F + k] = cos(a)
                s[i * F + k] = sin(a)
            }
        }
        cosBasis = c
        sinBasis = s
        var t = [Float](repeating: 0, count: F * Self.nMels)
        for m in 0..<Self.nMels {
            for k in 0..<F { t[k * Self.nMels + m] = melFilters[m * F + k] }
        }
        filtersT = t
    }

    /// Python floor division (the frame-count formulas go negative for audio shorter than a frame).
    static func floorDiv(_ a: Int, _ b: Int) -> Int {
        let q = a / b
        return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q
    }

    public static func validFrames(samples n: Int, center: Bool) -> Int {
        center ? floorDiv(n, hop) : floorDiv(n - nFFT, hop) + 1
    }

    /// Log-mel of one audio buffer: (mel [F, 128] frame-major, F).
    public func logMel(_ audio: UnsafeBufferPointer<Float>, center: Bool) -> (mel: [Float], frames: Int) {
        let n = audio.count
        let nFrames = max(0, Self.validFrames(samples: n, center: center))
        guard nFrames > 0 else { return ([], 0) }
        let pad = center ? Self.nFFT / 2 : 0
        var y = [Float](repeating: 0, count: n + 2 * pad)
        y[pad] = audio[0]
        for t in 1..<n { y[pad + t] = audio[t] - Self.preemphasis * audio[t - 1] }

        let K = Self.winLength, nF = Self.nFreq, nM = Self.nMels, B = Self.block
        var windowed = [Double](repeating: 0, count: B * K)
        var re = [Double](repeating: 0, count: B * nF)
        var im = [Double](repeating: 0, count: B * nF)
        var power = [Float](repeating: 0, count: B * nF)
        var melBlock = [Float](repeating: 0, count: B * nM)
        var mel = [Float](repeating: 0, count: nFrames * nM)
        var f0 = 0
        while f0 < nFrames {
            let nb = min(B, nFrames - f0)
            for f in 0..<nb {
                let base = (f0 + f) * Self.hop + Self.winOffset
                for i in 0..<K { windowed[f * K + i] = Double(y[base + i] * window[i]) }     // float32 product
            }
            vDSP_mmulD(windowed, 1, cosBasis, 1, &re, 1, vDSP_Length(nb), vDSP_Length(nF), vDSP_Length(K))
            vDSP_mmulD(windowed, 1, sinBasis, 1, &im, 1, vDSP_Length(nb), vDSP_Length(nF), vDSP_Length(K))
            for i in 0..<(nb * nF) {
                let a = Float(re[i]), b = Float(im[i])
                let m = (a * a + b * b).squareRoot()
                power[i] = m * m
            }
            vDSP_mmul(power, 1, filtersT, 1, &melBlock, 1, vDSP_Length(nb), vDSP_Length(nM), vDSP_Length(nF))
            for i in 0..<(nb * nM) { mel[f0 * nM + i] = logf(melBlock[i] + Self.logGuard) }
            f0 += nb
        }
        return (mel, nFrames)
    }

    public func logMel(_ audio: ArraySlice<Float>, center: Bool) -> (mel: [Float], frames: Int) {
        audio.withUnsafeBufferPointer { logMel($0, center: center) }
    }

    /// The model card's streaming chunks for (chunk, look-ahead) encoder frames: chunk 0 =
    /// audio[0, ((c+la)*8 - 1)*160 + 200), center-padded; chunk k >= 1 starts at 160 * k*c*8 - 256 and
    /// holds (c+la)*8*160 + 400 samples, not padded; the first chunk that would run past the audio instead
    /// takes the rest and is the last one (no look-ahead).
    public static func streamChunks(samples n: Int, chunkFrames c: Int, lookaheadFrames la: Int) -> [N3DChunk] {
        let melPerChunk = (c + la) * stack, melPerStep = c * stack
        let samplesFirst = (melPerChunk - 1) * hop + winLength / 2
        let samplesLater = melPerChunk * hop + winLength
        var chunks = [N3DChunk(start: 0, end: samplesFirst, isFirst: true, isLast: false)]
        var melIdx = melPerStep
        var start = melIdx * hop - nFFT / 2
        while start + samplesLater <= n {
            chunks.append(N3DChunk(start: start, end: start + samplesLater, isFirst: false, isLast: false))
            melIdx += melPerStep
            start = melIdx * hop - nFFT / 2
        }
        chunks.append(N3DChunk(start: start, end: n, isFirst: false, isLast: true))
        return chunks
    }
}

/// embed(mel, projection) -> [ceil(F/8), 512]: zero-pad the frame count to a multiple of 8 (zeros in the
/// log-mel domain, as transformers does), stack 8 frames -> 1024, @ projection.T (float32).
public struct N3DEmbedder: Sendable {
    public static let hidden = 512
    static let width = N3DMel.nMels * N3DMel.stack             // 1024

    let projectionT: [Float]     // [1024, 512]

    public init(projection: [Float]) {
        precondition(projection.count == Self.hidden * Self.width, "projection must be [512, 1024]")
        var t = [Float](repeating: 0, count: Self.width * Self.hidden)
        for o in 0..<Self.hidden {
            for i in 0..<Self.width { t[i * Self.hidden + o] = projection[o * Self.width + i] }
        }
        projectionT = t
    }

    /// Frame-major mel [F, 128] -> (embeds [rows, 512], rows = ceil(F / 8)).
    public func embed(mel: [Float], frames: Int) -> (embeds: [Float], rows: Int) {
        let rows = (frames + N3DMel.stack - 1) / N3DMel.stack
        guard rows > 0 else { return ([], 0) }
        var stacked = [Float](repeating: 0, count: rows * Self.width)
        stacked.replaceSubrange(0..<(frames * N3DMel.nMels), with: mel[0..<(frames * N3DMel.nMels)])
        var out = [Float](repeating: 0, count: rows * Self.hidden)
        vDSP_mmul(stacked, 1, projectionT, 1, &out, 1, vDSP_Length(rows), vDSP_Length(Self.hidden), vDSP_Length(Self.width))
        return (out, rows)
    }
}
