// ImagePreprocess — image -> tower patches, the host half of the vision contract (conversion/decider_vision/host.py):
//
//   RGB8 -> bicubic resize to (32 * grid)^2 in Pillow's pass order -> /255 -> (x - 0.5) / 0.5
//        -> merge-block-major patches [4 * grid^2, 1536] float32, vector (C, T = 2, 16, 16), the frame twice.
//
// The resize is `host.resize_bicubic` written out: Pillow's antialiased bicubic (a = -0.5, support 2 x the reduction
// factor), the HORIZONTAL pass first, the intermediate rounded (floor(x + 0.5)) and clipped to 0...255 before the
// vertical pass, every sum in float64 in NumPy's order (weights normalized by NumPy's pairwise sum, taps accumulated in
// window order with a fused multiply-add from zero, as `np.einsum` does on arm64). The aspect ratio is not kept: a
// 256x240 frame is stretched to 256x256, as the oracle's grid arms were captured. Host resampling is part of the
// numerics (knowledge/qwen3.8-27b-port.md): a CGContext resize is a different image.
//
// Decoding keeps the file's 8-bit samples as they are, the way Pillow's `convert("RGB")` does: no color matching, alpha
// dropped. Formats the direct path does not read (16-bit, CMYK, premultiplied alpha) are drawn into an sRGB context
// instead and say so in `RGB8Image.decodePath`.

import CoreGraphics
import Foundation
import ImageIO

/// 8-bit RGB pixels, row-major [height][width][3].
@available(macOS 27, iOS 27, *)
public struct RGB8Image: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]
    /// How the pixels were obtained: "direct <layout>" (the decoded samples) or "drawn sRGB" (color-matched).
    public var decodePath: String

    public init(width: Int, height: Int, pixels: [UInt8], decodePath: String = "raw") {
        precondition(pixels.count == width * height * 3, "RGB8Image: \(pixels.count) bytes for \(width)x\(height)")
        self.width = width
        self.height = height
        self.pixels = pixels
        self.decodePath = decodePath
    }
}

@available(macOS 27, iOS 27, *)
public enum ImagePreprocess {
    public static let patch = 16
    public static let merge = 2
    public static let temporal = 2
    public static let channels = 3
    public static let patchVector = channels * temporal * patch * patch   // 1536

    /// Pixels per side of the square tile a merged `grid` x `grid` covers (g256: 256, g448: 448).
    public static func tileSide(grid: Int) -> Int { patch * merge * grid }

    // MARK: Decode

    public static func loadCGImage(url: URL) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw DeciderVisionError.image("cannot decode \(url.path)") }
        return image
    }

    /// The image's 8-bit RGB samples, alpha dropped (Pillow `convert("RGB")`).
    public static func rgb8(_ image: CGImage) throws -> RGB8Image {
        if let direct = directRGB8(image) { return direct }
        return try drawnRGB8(image)
    }

    private static func directRGB8(_ image: CGImage) -> RGB8Image? {
        guard image.bitsPerComponent == 8, let space = image.colorSpace,
              let data = image.dataProvider?.data, let base = CFDataGetBytePtr(data)
        else { return nil }
        let w = image.width, h = image.height, rowBytes = image.bytesPerRow
        let bpp = image.bitsPerPixel / 8
        let alpha = image.alphaInfo
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let little = order == .byteOrder32Little || order == .byteOrder16Little
        var out = [UInt8](repeating: 0, count: w * h * 3)
        switch space.model {
        case .rgb:
            // offsets of R, G, B inside one pixel as stored
            var offsets: [Int]
            let layout: String
            switch (bpp, alpha) {
            case (3, .none):
                offsets = [0, 1, 2]; layout = "RGB"
            case (4, .noneSkipLast), (4, .last):
                offsets = [0, 1, 2]; layout = alpha == .last ? "RGBA" : "RGBX"
            case (4, .noneSkipFirst), (4, .first):
                offsets = [1, 2, 3]; layout = alpha == .first ? "ARGB" : "XRGB"
            default:
                return nil
            }
            if little && bpp == 4 { offsets = offsets.map { 3 - $0 } }
            for y in 0..<h {
                let row = base + y * rowBytes
                for x in 0..<w {
                    let p = row + x * bpp, o = (y * w + x) * 3
                    out[o] = p[offsets[0]]
                    out[o + 1] = p[offsets[1]]
                    out[o + 2] = p[offsets[2]]
                }
            }
            return RGB8Image(width: w, height: h, pixels: out, decodePath: "direct \(layout)\(little && bpp == 4 ? " little-endian" : "")")
        case .monochrome:
            let lOffset: Int
            switch (bpp, alpha) {
            case (1, .none): lOffset = 0
            case (2, .last), (2, .noneSkipLast): lOffset = little ? 1 : 0
            case (2, .first), (2, .noneSkipFirst): lOffset = little ? 0 : 1
            default: return nil
            }
            for y in 0..<h {
                let row = base + y * rowBytes
                for x in 0..<w {
                    let v = row[x * bpp + lOffset], o = (y * w + x) * 3
                    out[o] = v; out[o + 1] = v; out[o + 2] = v
                }
            }
            return RGB8Image(width: w, height: h, pixels: out, decodePath: "direct gray")
        default:
            return nil
        }
    }

    private static func drawnRGB8(_ image: CGImage) throws -> RGB8Image {
        let w = image.width, h = image.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw DeciderVisionError.image("cannot make an RGB context \(w)x\(h)") }
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw DeciderVisionError.image("empty context") }
        let src = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3] = src[i * 4]; out[i * 3 + 1] = src[i * 4 + 1]; out[i * 3 + 2] = src[i * 4 + 2]
        }
        return RGB8Image(width: w, height: h, pixels: out, decodePath: "drawn sRGB")
    }

    // MARK: Resize (host.resize_bicubic)

    /// Pillow's bicubic kernel at distance `x` (already divided by the filter scale), a = -0.5, in NumPy's
    /// operation order (`_smoke/lfm25vl_preprocess._kernel`).
    static func bicubic(_ t: Double) -> Double {
        let x = abs(t)
        let a = -0.5
        if x < 1.0 {
            var w = (a + 2.0) * x
            w = w - (a + 3.0)
            w = w * x
            w = w * x
            return w + 1.0
        }
        if x < 2.0 {
            var w = x - 5.0
            w = w * x
            w = w + 8.0
            w = w * x
            w = w - 4.0
            return w * a
        }
        return 0.0
    }

    /// Per-output-sample window start and normalized weights (`_filter_weights`).
    static func filterWeights(inSize: Int, outSize: Int) -> (starts: [Int], weights: [[Double]]) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(1.0, scale)
        let support = 2.0 * filterscale
        var starts = [Int](repeating: 0, count: outSize)
        var weights = [[Double]](repeating: [], count: outSize)
        for i in 0..<outSize {
            let center = (Double(i) + 0.5) * scale
            let xmin = Int(max(0.0, (center - support + 0.5).rounded(.down)))
            let xmax = Int(min(Double(inSize), (center + support + 0.5).rounded(.up)))
            var w = [Double]()
            w.reserveCapacity(max(0, xmax - xmin))
            for x in xmin..<max(xmin, xmax) {
                w.append(bicubic(((Double(x) + 0.5) - center) / filterscale))
            }
            let total = numpySum(w)
            if total > 0 { w = w.map { $0 / total } }
            starts[i] = xmin
            weights[i] = w
        }
        return (starts, weights)
    }

    /// One pass along `axis` (1 = horizontal, 0 = vertical) of a float [H][W][3] plane, then floor(x + 0.5) and clip.
    private static func pass(_ src: [Double], h: Int, w: Int, axis: Int, out: Int) -> [Double] {
        let inSize = axis == 1 ? w : h
        let (starts, weights) = filterWeights(inSize: inSize, outSize: out)
        let oh = axis == 1 ? h : out, ow = axis == 1 ? out : w
        var dst = [Double](repeating: 0, count: oh * ow * 3)
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                for y in 0..<oh {
                    for x in 0..<ow {
                        let i = axis == 1 ? x : y
                        let ws = weights[i], s0 = starts[i]
                        for c in 0..<3 {
                            var acc = 0.0
                            for k in 0..<ws.count {
                                let sy = axis == 1 ? y : s0 + k, sx = axis == 1 ? s0 + k : x
                                acc = acc.addingProduct(ws[k], s[(sy * w + sx) * 3 + c])
                            }
                            d[(y * ow + x) * 3 + c] = min(255.0, max(0.0, (acc + 0.5).rounded(.down)))
                        }
                    }
                }
            }
        }
        return dst
    }

    /// `host.resize_bicubic`: horizontal pass first (when the width changes), the uint8 intermediate, then vertical.
    public static func resizeBicubic(_ image: RGB8Image, width: Int, height: Int) -> RGB8Image {
        var x = image.pixels.map(Double.init)
        var h = image.height, w = image.width
        if w != width {
            x = pass(x, h: h, w: w, axis: 1, out: width)
            w = width
        }
        if h != height {
            x = pass(x, h: h, w: w, axis: 0, out: height)
            h = height
        }
        return RGB8Image(width: w, height: h, pixels: x.map { UInt8($0) }, decodePath: image.decodePath)
    }

    // MARK: Patches (host.patchify)

    /// Resized tile -> [4 * grid^2, 1536] float32: (x / 255 - 0.5) / 0.5 in float64, then float32; patches in
    /// (block row, block col, row in block, col in block) order, each vector (C, T, 16, 16) with the frame at both T.
    public static func patches(_ tile: RGB8Image) throws -> [Float] {
        let side = tile.width
        guard tile.height == side, side % (patch * merge) == 0 else {
            throw DeciderVisionError.image("tile \(tile.width)x\(tile.height) is not a square multiple of \(patch * merge)")
        }
        let g = side / patch             // patches per side (2 * grid)
        let blocks = g / merge
        var lut = [Float](repeating: 0, count: 256)
        for v in 0..<256 { lut[v] = Float((Double(v) / 255.0 - 0.5) / 0.5) }
        var out = [Float](repeating: 0, count: g * g * patchVector)
        tile.pixels.withUnsafeBufferPointer { px in
            out.withUnsafeMutableBufferPointer { o in
                for bh in 0..<blocks {
                    for bw in 0..<blocks {
                        for mh in 0..<merge {
                            for mw in 0..<merge {
                                let n = ((bh * blocks + bw) * merge + mh) * merge + mw
                                let row0 = (bh * merge + mh) * patch, col0 = (bw * merge + mw) * patch
                                for c in 0..<channels {
                                    for t in 0..<temporal {
                                        let base = n * patchVector + (c * temporal + t) * patch * patch
                                        for py in 0..<patch {
                                            for pxx in 0..<patch {
                                                let v = px[((row0 + py) * side + col0 + pxx) * 3 + c]
                                                o[base + py * patch + pxx] = lut[Int(v)]
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// Everything the tower needs from one image at one grid, with each step's time.
    public struct Prepared: Sendable {
        public let decoded: RGB8Image
        public let resized: RGB8Image
        public let patches: [Float]
        public let decodeSeconds: Double
        public let resizeSeconds: Double
        public let patchSeconds: Double
    }

    public static func prepare(_ image: CGImage, grid: Int) throws -> Prepared {
        let t0 = ContinuousClock.now
        let rgb = try rgb8(image)
        let t1 = ContinuousClock.now
        let side = tileSide(grid: grid)
        let resized = resizeBicubic(rgb, width: side, height: side)
        let t2 = ContinuousClock.now
        let p = try patches(resized)
        let t3 = ContinuousClock.now
        func s(_ a: ContinuousClock.Instant, _ b: ContinuousClock.Instant) -> Double {
            let d = b - a
            return Double(d.components.seconds) + Double(d.components.attoseconds) * 1e-18
        }
        return Prepared(decoded: rgb, resized: resized, patches: p, decodeSeconds: s(t0, t1),
                        resizeSeconds: s(t1, t2), patchSeconds: s(t2, t3))
    }
}
