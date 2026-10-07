// ImagePixels — a picture file -> per crop the four fixed-shape inputs of the vision tower graph, the pixel half of
// conversion/d1/vision_host.py (§1 the picture and cap_pixels, §3 both resamplers, §4 pixels, §7 the position table,
// §8 unshuffle and `tower_inputs`; the crops come from `D1Vision.plan`, §2). The rules and their sources in
// transformers 5.19 / torchvision 0.24.1 / torch 2.9.1 / Pillow 12.3 are K/results/vision_rules.md (d)(f)(g);
// conversion/d1/gate_swift_pixels.py holds every crop of the fixture to vision_host.tower_inputs bit for bit.
//
//   1 decode     ImageIO, the file's 8-bit samples as stored (no color matching), alpha dropped without compositing
//                (Pillow `convert("RGB")`), then the EXIF orientation applied as `ImageOps.exif_transpose` does (the
//                card's `load_image`): 2 mirror, 3 rotate 180, 4 flip, 5 transpose, 6 rotate 90 clockwise, 7 transverse,
//                8 rotate 90 counter-clockwise. A PNG gives Pillow's bytes; a JPEG is ImageIO's decoder, not libjpeg's
//                (levels can differ). Layouts the direct path does not read (16-bit, CMYK, premultiplied alpha) are
//                drawn into an sRGB context and come back with `exact` false.
//   2 cap        w * h > 1024 * 1024 -> Pillow's BICUBIC to `D1Vision.capSize`: Resample.c's 8-bit resampler (22-bit
//                fixed-point weights, the horizontal pass first into uint8, then the vertical one), the ClefFlash copy
//                (apps/ClefFlash/Sources/ClefFlash/ImagePreprocess.swift)
//   3 crops      a single crop or the thumbnail = the capped picture resized to the plan's smart size; the tiles = the
//                capped picture resized to (512 rows, 512 cols) and cut row-major
//   4 resize     torch's antialiased uint8 bicubic (torchvision resizes uint8 natively on CPU; UpSampleKernel.cpp
//                `_compute_index_ranges_int16_weights`, `basic_loop_separable_1d_*<uint8_t>`), per axis in double:
//                  scale = in / out, support = 2 * scale (or 2 when enlarging), kmax = ceil(support) * 2 + 1,
//                  center = scale * (i + 0.5), xmin = max(trunc(center - support + 0.5), 0),
//                  n = clamp(min(trunc(center + support + 0.5), in) - xmin, 0, kmax),
//                  w_j = cubic((j + xmin - center + 0.5) * invscale) (A = -0.5), divided by their sum,
//                  p counts up from 0 while int(0.5 + max_w * 2^(p+1)) < 2^15, at most 22 (max_w over the whole axis),
//                  w16 = w * 2^p rounded half away from zero, out = clamp((2^(p-1) + sum src * w16) >> p, 0, 255);
//                width first into uint8, then height; an unchanged axis is skipped. Not Pillow's weights: the two
//                land 1 level apart on some pixels, and the processor's (torch's) are the ones the tower saw.
//   5 pixels     (x - 127.5) / 127.5 in Float (subtract, then divide); patches [H/16 * W/16, 768] row-major, inside a
//                patch [y][x][c] (channel fastest); 0.0 rows to 1024; key_bias 0 for a real patch, -inf for padding
//   6 positions  the [256, d] table as [16, 16, d] resized to the crop's (h, w) patches by torch's float32
//                antialiased bilinear (`HelperInterpLinear`): scale = Float(16) / out, support = scale (or 1),
//                center = Float(scale * (i + 0.5)), w = 1 - |x| at x = Float((Float(j + xmin) - center + 0.5) *
//                invscale), divided by the Float sum; width first, then height; taps accumulated in order as
//                acc = t0 * w0, acc = fma(t_j, w_j, acc) (torch's aarch64 build fuses `output += t * wts`; two roundings
//                land up to 4.8e-7 away); rows past h * w are 0 (HF writes row 0 there; the tower masks those keys
//                and drops those outputs, so either is exact)
//   7 unshuffle  merged token k = i * (w / 2) + j reads patches [2i w + 2j, 2i w + 2j + 1, (2i+1) w + 2j,
//                (2i+1) w + 2j + 1] in that order; rows k >= h * w / 4 are 0 and their outputs are dropped

import CoreGraphics
import Foundation
import ImageIO

public enum D1Pixels {
    public static let maxPatches = 1024
    public static let patchDim = D1Vision.patch * D1Vision.patch * 3      // 768
    public static let maxTokens = 256
    public static let positionSide = 16

    /// 8-bit RGB pixels, row-major [height][width][3].
    public struct RGB: Sendable, Equatable {
        public let width: Int
        public let height: Int
        public var bytes: [UInt8]

        public init(width: Int, height: Int, bytes: [UInt8]) {
            precondition(bytes.count == width * height * 3, "RGB: \(bytes.count) bytes for \(width)x\(height)")
            self.width = width
            self.height = height
            self.bytes = bytes
        }
    }

    /// A decoded picture and how it was read.
    public struct Decoded: Sendable {
        /// the pixels after the EXIF orientation (what the processor receives before the cap)
        public let rgb: RGB
        /// the size as stored in the file (before the orientation)
        public let storedWidth: Int
        public let storedHeight: Int
        /// the EXIF orientation 1...8 (1 when the file has none; values outside 1...8 are ignored, as Pillow does)
        public let orientation: Int
        /// the file type ImageIO reports (UTI)
        public let type: String
        /// how the samples were read: "RGBX", "RGBA", "XRGB little-endian", "gray", "gray+alpha", "indexed", "RGB",
        /// or "drawn sRGB" (a layout the direct path does not read)
        public let layout: String
        /// true when the bytes are the file's own 8-bit samples (Pillow's for a PNG); false when drawn
        public let exact: Bool
    }

    // MARK: 1 decode

    /// A picture file -> its RGB pixels with the EXIF orientation applied (Pillow: `ImageOps.exif_transpose`,
    /// `convert("RGB")`).
    public static func decode(url: URL) throws -> Decoded {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), CGImageSourceGetCount(source) > 0 else {
            throw D1Error.request("picture \(url.lastPathComponent): not an image ImageIO can open")
        }
        return try decode(source: source, name: url.lastPathComponent)
    }

    /// The same for the bytes of a picture file.
    public static func decode(data: Data) throws -> Decoded {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) > 0 else {
            throw D1Error.request("picture (\(data.count) bytes): not an image ImageIO can open")
        }
        return try decode(source: source, name: "\(data.count) bytes")
    }

    static func decode(source: CGImageSource, name: String) throws -> Decoded {
        let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let tag = (props?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        let orientation = (1...8).contains(tag) ? tag : 1
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw D1Error.request("picture \(name): ImageIO cannot decode it")
        }
        let read: (rgb: RGB, layout: String, exact: Bool)
        if let direct = directSamples(image) {
            read = (direct.rgb, direct.layout, true)
        } else {
            read = (try drawnSRGB(image), "drawn sRGB", false)
        }
        return Decoded(rgb: oriented(read.rgb, orientation), storedWidth: image.width, storedHeight: image.height,
                       orientation: orientation, type: CGImageSourceGetType(source) as String? ?? "?",
                       layout: read.layout, exact: read.exact)
    }

    /// The image's 8-bit samples as decoded, alpha dropped; nil for a layout this does not read.
    static func directSamples(_ image: CGImage) -> (rgb: RGB, layout: String)? {
        guard image.bitsPerComponent == 8, !image.bitmapInfo.contains(.floatComponents), let space = image.colorSpace,
              let data = image.dataProvider?.data, let base = CFDataGetBytePtr(data)
        else { return nil }
        let w = image.width, h = image.height, rowBytes = image.bytesPerRow
        let bpp = image.bitsPerPixel / 8
        guard w > 0, h > 0, CFDataGetLength(data) >= (h - 1) * rowBytes + w * bpp else { return nil }
        let alpha = image.alphaInfo
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let little = order == .byteOrder32Little || order == .byteOrder16Little
        var out = [UInt8](repeating: 0, count: w * h * 3)
        switch space.model {
        case .rgb:
            var offsets: [Int]
            var layout: String
            switch (bpp, alpha) {
            case (3, .none): offsets = [0, 1, 2]; layout = "RGB"
            case (4, .noneSkipLast): offsets = [0, 1, 2]; layout = "RGBX"
            case (4, .last): offsets = [0, 1, 2]; layout = "RGBA"
            case (4, .noneSkipFirst): offsets = [1, 2, 3]; layout = "XRGB"
            case (4, .first): offsets = [1, 2, 3]; layout = "ARGB"
            default: return nil      // premultiplied alpha: the stored samples are not recoverable
            }
            if little && bpp == 4 {
                offsets = offsets.map { 3 - $0 }
                layout += " little-endian"
            } else if little {
                return nil
            }
            let (o0, o1, o2) = (offsets[0], offsets[1], offsets[2])
            out.withUnsafeMutableBufferPointer { d in
                for y in 0..<h {
                    let row = base + y * rowBytes
                    for x in 0..<w {
                        let p = row + x * bpp, o = (y * w + x) * 3
                        d[o] = p[o0]
                        d[o + 1] = p[o1]
                        d[o + 2] = p[o2]
                    }
                }
            }
            return (RGB(width: w, height: h, bytes: out), layout)
        case .monochrome:
            let lOffset: Int
            let layout: String
            switch (bpp, alpha) {
            case (1, .none): lOffset = 0; layout = "gray"
            case (2, .last), (2, .noneSkipLast): lOffset = little ? 1 : 0; layout = "gray+alpha"
            case (2, .first), (2, .noneSkipFirst): lOffset = little ? 0 : 1; layout = "alpha+gray"
            default: return nil
            }
            out.withUnsafeMutableBufferPointer { d in
                for y in 0..<h {
                    let row = base + y * rowBytes
                    for x in 0..<w {
                        let v = row[x * bpp + lOffset], o = (y * w + x) * 3
                        d[o] = v
                        d[o + 1] = v
                        d[o + 2] = v
                    }
                }
            }
            return (RGB(width: w, height: h, bytes: out), layout)
        case .indexed:
            // Pillow's "P" -> "RGB": each index through the palette
            guard bpp == 1, alpha == .none, let baseSpace = space.baseColorSpace, baseSpace.model == .rgb,
                  baseSpace.numberOfComponents == 3, let table = space.colorTable, table.count >= 3
            else { return nil }
            let entries = table.count / 3
            out.withUnsafeMutableBufferPointer { d in
                for y in 0..<h {
                    let row = base + y * rowBytes
                    for x in 0..<w {
                        let i = Int(row[x]), o = (y * w + x) * 3
                        guard i < entries else { continue }      // past the palette: 0, 0, 0
                        d[o] = table[i * 3]
                        d[o + 1] = table[i * 3 + 1]
                        d[o + 2] = table[i * 3 + 2]
                    }
                }
            }
            return (RGB(width: w, height: h, bytes: out), "indexed")
        default:
            return nil
        }
    }

    /// A layout the direct path does not read, color-matched into sRGB by Core Graphics (not Pillow's numbers).
    static func drawnSRGB(_ image: CGImage) throws -> RGB {
        let w = image.width, h = image.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { throw D1Error.request("picture \(w)x\(h): cannot make an RGB context") }
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw D1Error.request("picture \(w)x\(h): empty context") }
        let src = data.bindMemory(to: UInt8.self, capacity: w * h * 4)
        var out = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            out[i * 3] = src[i * 4]
            out[i * 3 + 1] = src[i * 4 + 1]
            out[i * 3 + 2] = src[i * 4 + 2]
        }
        return RGB(width: w, height: h, bytes: out)
    }

    /// `ImageOps.exif_transpose`: the stored pixels turned the way the EXIF orientation says (Pillow's transpose
    /// for 2...8; anything else unchanged).
    public static func oriented(_ s: RGB, _ orientation: Int) -> RGB {
        guard (2...8).contains(orientation) else { return s }
        let w = s.width, h = s.height
        let swap = orientation >= 5
        let ow = swap ? h : w, oh = swap ? w : h
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        s.bytes.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { d in
                for y in 0..<oh {
                    for x in 0..<ow {
                        let sx: Int, sy: Int
                        switch orientation {
                        case 2: (sx, sy) = (w - 1 - x, y)              // FLIP_LEFT_RIGHT
                        case 3: (sx, sy) = (w - 1 - x, h - 1 - y)      // ROTATE_180
                        case 4: (sx, sy) = (x, h - 1 - y)              // FLIP_TOP_BOTTOM
                        case 5: (sx, sy) = (y, x)                      // TRANSPOSE
                        case 6: (sx, sy) = (y, h - 1 - x)              // ROTATE_270 (90 degrees clockwise)
                        case 7: (sx, sy) = (w - 1 - y, h - 1 - x)      // TRANSVERSE
                        default: (sx, sy) = (w - 1 - y, x)             // 8: ROTATE_90 (90 degrees counter-clockwise)
                        }
                        let i = (sy * w + sx) * 3, o = (y * ow + x) * 3
                        d[o] = src[i]
                        d[o + 1] = src[i + 1]
                        d[o + 2] = src[i + 2]
                    }
                }
            }
        }
        return RGB(width: ow, height: oh, bytes: out)
    }

    // MARK: 2 / 4 the two integer resamplers

    /// One axis of a separable integer resampler: per output index the first input index, the tap count and
    /// `ksize` integer weights (zeros past the count), applied as clamp((2^(p-1) + sum src * w) >> p, 0, 255).
    struct Kernel {
        let outSize: Int
        let ksize: Int
        let starts: [Int]
        let counts: [Int]
        let weights: [Int]
        let precision: Int
    }

    /// `HelperInterpCubic::aa_filter<double, true>` (A = -0.5; cubic_convolution1 / 2 of UpSample.h).
    static func torchCubic(_ t: Double) -> Double {
        let a = -0.5
        let x = abs(t)
        if x < 1.0 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2.0 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return 0.0
    }

    /// torch's uint8 antialiased bicubic weights for one axis (`_compute_indices_min_size_weights_aa` in double,
    /// then `_compute_index_ranges_int16_weights`; align_corners false, no scale factor) = vision_host.torch_u8_weights.
    static func torchKernel(inSize: Int, outSize: Int) -> Kernel {
        let scale = Double(inSize) / Double(outSize)
        let support = scale >= 1.0 ? (4 * 0.5) * scale : 4 * 0.5
        let kmax = Int(support.rounded(.up)) * 2 + 1
        let invscale = scale >= 1.0 ? 1.0 / scale : 1.0
        var starts = [Int](repeating: 0, count: outSize)
        var counts = [Int](repeating: 0, count: outSize)
        var wf = [Double](repeating: 0, count: outSize * kmax)
        var wtMax = 0.0
        for i in 0..<outSize {
            let center = scale * (Double(i) + 0.5)
            let xmin = max(Int(center - support + 0.5), 0)
            let xsize = min(max(min(Int(center + support + 0.5), inSize) - xmin, 0), kmax)
            var total = 0.0
            for j in 0..<xsize {
                let v = torchCubic((Double(j + xmin) - center + 0.5) * invscale)
                wf[i * kmax + j] = v
                total += v
            }
            if total != 0.0 {
                for j in 0..<xsize {
                    wf[i * kmax + j] /= total
                    wtMax = max(wtMax, wf[i * kmax + j])
                }
            }
            starts[i] = xmin
            counts[i] = xsize
        }
        var precision = 0
        while precision < 22 {
            if Int(0.5 + wtMax * Double(1 << (precision + 1))) >= (1 << 15) { break }
            precision += 1
        }
        let one = Double(1 << precision)
        let weights = wf.map { w -> Int in
            let v = w * one
            return v < 0 ? Int(-0.5 + v) : Int(0.5 + v)
        }
        return Kernel(outSize: outSize, ksize: kmax, starts: starts, counts: counts, weights: weights, precision: precision)
    }

    static let pillowPrecision = 32 - 8 - 2      // Resample.c PRECISION_BITS

    /// Resample.c `bicubic_filter` (a = -0.5) in its operation order.
    static func pillowCubic(_ t: Double) -> Double {
        let x = t < 0 ? -t : t
        let a = -0.5
        if x < 1.0 { return ((a + 2.0) * x - (a + 3.0)) * x * x + 1 }
        if x < 2.0 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0.0
    }

    /// Resample.c `precompute_coeffs` (in0 = 0, in1 = inSize) + `normalize_coeffs_8bpc` = vision_host.pil_weights.
    static func pillowKernel(inSize: Int, outSize: Int) -> Kernel {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1.0)
        let support = 2.0 * filterscale
        let ksize = Int(support.rounded(.up)) * 2 + 1
        var starts = [Int](repeating: 0, count: outSize)
        var counts = [Int](repeating: 0, count: outSize)
        var weights = [Int](repeating: 0, count: outSize * ksize)
        let one = Double(1 << pillowPrecision)
        for xx in 0..<outSize {
            let center = 0.0 + (Double(xx) + 0.5) * scale
            var ww = 0.0
            let ss = 1.0 / filterscale
            var xmin = Int(center - support + 0.5)          // C (int): truncation toward zero
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax = min(xmax - xmin, ksize)
            var k = [Double](repeating: 0, count: ksize)
            for x in 0..<max(xmax, 0) {
                let w = pillowCubic((Double(x + xmin) - center + 0.5) * ss)
                k[x] = w
                ww += w
            }
            for x in 0..<max(xmax, 0) where ww != 0.0 { k[x] /= ww }
            for x in 0..<ksize {
                weights[xx * ksize + x] = k[x] < 0 ? Int(-0.5 + k[x] * one) : Int(0.5 + k[x] * one)
            }
            starts[xx] = xmin
            counts[xx] = max(xmax, 0)
        }
        return Kernel(outSize: outSize, ksize: ksize, starts: starts, counts: counts, weights: weights,
                      precision: pillowPrecision)
    }

    /// One integer pass over an RGB image: along the width (`horizontal`) or the height.
    static func pass(_ src: [UInt8], width w: Int, height h: Int, horizontal: Bool, _ k: Kernel) -> [UInt8] {
        let ow = horizontal ? k.outSize : w, oh = horizontal ? h : k.outSize
        var dst = [UInt8](repeating: 0, count: ow * oh * 3)
        let half = 1 << (k.precision - 1), p = k.precision
        @inline(__always) func clamp8(_ v: Int) -> UInt8 { UInt8(min(max(v >> p, 0), 255)) }
        src.withUnsafeBufferPointer { s in
            dst.withUnsafeMutableBufferPointer { d in
                k.weights.withUnsafeBufferPointer { wt in
                    if horizontal {
                        for y in 0..<oh {
                            let row = y * w * 3
                            for x in 0..<ow {
                                let lo = k.starts[x], n = k.counts[x], kb = x * k.ksize
                                var s0 = half, s1 = half, s2 = half
                                for t in 0..<n {
                                    let q = row + (lo + t) * 3, c = wt[kb + t]
                                    s0 += Int(s[q]) * c
                                    s1 += Int(s[q + 1]) * c
                                    s2 += Int(s[q + 2]) * c
                                }
                                let o = (y * ow + x) * 3
                                d[o] = clamp8(s0)
                                d[o + 1] = clamp8(s1)
                                d[o + 2] = clamp8(s2)
                            }
                        }
                    } else {
                        for y in 0..<oh {
                            let lo = k.starts[y], n = k.counts[y], kb = y * k.ksize
                            for x in 0..<ow {
                                var s0 = half, s1 = half, s2 = half
                                for t in 0..<n {
                                    let q = ((lo + t) * w + x) * 3, c = wt[kb + t]
                                    s0 += Int(s[q]) * c
                                    s1 += Int(s[q + 1]) * c
                                    s2 += Int(s[q + 2]) * c
                                }
                                let o = (y * ow + x) * 3
                                d[o] = clamp8(s0)
                                d[o + 1] = clamp8(s1)
                                d[o + 2] = clamp8(s2)
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    /// The processor's resize: torch's uint8 antialiased bicubic to (height, width), width first (§4).
    public static func resizeTorchU8Bicubic(_ image: RGB, width: Int, height: Int) -> RGB {
        if image.width == width && image.height == height { return image }
        var x = image.bytes
        var w = image.width
        let h = image.height
        if w != width {
            x = pass(x, width: w, height: h, horizontal: true, torchKernel(inSize: w, outSize: width))
            w = width
        }
        if h != height { x = pass(x, width: w, height: h, horizontal: false, torchKernel(inSize: h, outSize: height)) }
        return RGB(width: width, height: height, bytes: x)
    }

    /// Pillow's `Image.resize((width, height), BICUBIC)` of an RGB image: the horizontal pass when the width changes,
    /// then the vertical pass when the height changes, each on uint8 (§2).
    public static func resizePillowBicubic(_ image: RGB, width: Int, height: Int) -> RGB {
        var x = image.bytes
        var w = image.width
        let h = image.height
        if w != width {
            x = pass(x, width: w, height: h, horizontal: true, pillowKernel(inSize: w, outSize: width))
            w = width
        }
        if h != height { x = pass(x, width: w, height: h, horizontal: false, pillowKernel(inSize: h, outSize: height)) }
        return RGB(width: width, height: height, bytes: x)
    }

    /// `runner.cap_pixels`: over 1024 * 1024 pixels, Pillow's BICUBIC down to `D1Vision.capSize`; otherwise as is.
    public static func capPixels(_ image: RGB) -> RGB {
        let c = D1Vision.capSize(width: image.width, height: image.height)
        if c.width == image.width && c.height == image.height { return image }
        return resizePillowBicubic(image, width: c.width, height: c.height)
    }

    // MARK: 3 crops

    public struct Picture: Sendable {
        public let decoded: Decoded
        /// the picture the processor receives (after the cap)
        public let capped: RGB
        public let plan: D1Vision.Plan
        /// each crop's pixels after the resize, in plan order
        public let crops: [RGB]
    }

    /// `vision_host.crop_images`: every crop's uint8 pixels in plan order (one resize per distinct target).
    public static func cropImages(
        _ image: RGB, plan: D1Vision.Plan,
        resize: (RGB, Int, Int) -> RGB = { D1Pixels.resizeTorchU8Bicubic($0, width: $1, height: $2) }
    ) -> [RGB] {
        var tiles: RGB?
        var out: [RGB] = []
        for c in plan.crops {
            switch c.kind {
            case .single, .thumbnail:
                out.append(resize(image, c.width, c.height))
            case .tile:
                if tiles == nil { tiles = resize(image, D1Vision.tile * plan.cols, D1Vision.tile * plan.rows) }
                out.append(cut(tiles!, x0: (c.col - 1) * D1Vision.tile, y0: (c.row - 1) * D1Vision.tile,
                               width: D1Vision.tile, height: D1Vision.tile))
            }
        }
        return out
    }

    static func cut(_ s: RGB, x0: Int, y0: Int, width: Int, height: Int) -> RGB {
        var out = [UInt8](repeating: 0, count: width * height * 3)
        s.bytes.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { d in
                for y in 0..<height {
                    let from = ((y0 + y) * s.width + x0) * 3
                    for i in 0..<(width * 3) { d[y * width * 3 + i] = src[from + i] }
                }
            }
        }
        return RGB(width: width, height: height, bytes: out)
    }

    /// A decoded picture through the cap, the plan and the crops' resize.
    public static func picture(
        _ decoded: Decoded,
        resize: (RGB, Int, Int) -> RGB = { D1Pixels.resizeTorchU8Bicubic($0, width: $1, height: $2) }
    ) -> Picture {
        let capped = capPixels(decoded.rgb)
        let plan = D1Vision.plan(height: capped.height, width: capped.width)
        return Picture(decoded: decoded, capped: capped, plan: plan, crops: cropImages(capped, plan: plan, resize: resize))
    }

    public static func picture(url: URL) throws -> Picture { picture(try decode(url: url)) }

    // MARK: 5 pixels

    /// (v - 127.5) / 127.5 in Float for every byte value (the processor's fused rescale + normalize).
    public static let normalizeTable: [Float] = (0..<256).map { (Float($0) - 127.5) / 127.5 }

    /// A crop -> [H/16 * W/16, 768] patch rows (no padding): patches row-major, [y][x][c] inside.
    public static func patches(_ crop: RGB) throws -> [Float] {
        let p = D1Vision.patch
        guard crop.width % p == 0, crop.height % p == 0 else {
            throw D1Error.request("crop \(crop.width)x\(crop.height) is not a multiple of \(p)")
        }
        let gh = crop.height / p, gw = crop.width / p, w = crop.width
        var out = [Float](repeating: 0, count: gh * gw * patchDim)
        let lut = normalizeTable
        crop.bytes.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { o in
                for py in 0..<gh {
                    for px in 0..<gw {
                        let base = (py * gw + px) * patchDim
                        for y in 0..<p {
                            let from = ((py * p + y) * w + px * p) * 3, to = base + y * p * 3
                            for i in 0..<(p * 3) { o[to + i] = lut[Int(s[from + i])] }
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: 6 positions

    /// The vision tower's 16 x 16 position table [256, dim] (float32), resized per crop on the host.
    public struct PositionTable: Sendable {
        public let dim: Int
        public let values: [Float]

        public init(dim: Int, values: [Float]) throws {
            guard dim > 0, values.count == 256 * dim else {
                throw D1Error.contract("position table: \(values.count) values for [256, \(dim)]")
            }
            self.dim = dim
            self.values = values
        }

        /// `host/position_embedding.safetensors` (one F32 tensor "position_embedding" [256, d], as export_vision.py
        /// writes it) or a raw little-endian float32 file of 256 * d values.
        public static func load(url: URL) throws -> PositionTable {
            let data = try Data(contentsOf: url)
            let name = url.lastPathComponent
            guard url.pathExtension == "safetensors" else {
                guard data.count > 0, data.count % (256 * 4) == 0 else {
                    throw D1Error.contract("position table \(name): \(data.count) bytes is not 256 * d float32 values")
                }
                return try PositionTable(dim: data.count / (256 * 4), values: floats(data, 0, data.count / 4))
            }
            guard data.count >= 8 else { throw D1Error.contract("position table \(name): no safetensors header") }
            let n = data.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self))) }
            guard n > 0, 8 + n <= data.count,
                  let header = try? JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + n))) as? [String: Any],
                  let entry = header["position_embedding"] as? [String: Any]
            else { throw D1Error.contract("position table \(name): no \"position_embedding\" in the safetensors header") }
            guard let dtype = entry["dtype"] as? String, dtype == "F32",
                  let shape = entry["shape"] as? [Int], shape.count == 2, shape[0] == 256, shape[1] > 0,
                  let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                  offsets[1] - offsets[0] == 256 * shape[1] * 4, 8 + n + offsets[1] <= data.count
            else {
                throw D1Error.contract("position table \(name): want F32 [256, d], got \(entry["dtype"] ?? "?") "
                                       + "\(entry["shape"] ?? "?")")
            }
            return try PositionTable(dim: shape[1], values: floats(data, 8 + n + offsets[0], 256 * shape[1]))
        }

        static func floats(_ data: Data, _ offset: Int, _ count: Int) -> [Float] {
            [Float](unsafeUninitializedCapacity: count) { buf, initialized in
                data.withUnsafeBytes { raw in
                    for i in 0..<count {
                        buf[i] = Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset + 4 * i,
                                                                                          as: UInt32.self)))
                    }
                }
                initialized = count
            }
        }

        /// torch's float32 antialiased bilinear weights for one axis (`HelperInterpLinear`, scalar_t = float)
        /// = vision_host.torch_linear_aa_weights_f32: (first input index, weights) per output index.
        static func weights(inSize: Int, outSize: Int) -> [(start: Int, w: [Float])] {
            let scale = Float(inSize) / Float(outSize)
            let support: Float = scale >= 1.0 ? Float((2 * 0.5) * Double(scale)) : Float(2 * 0.5)
            let kmax = Int(Double(support).rounded(.up)) * 2 + 1
            let invscale: Float = scale >= 1.0 ? Float(1.0 / Double(scale)) : 1.0
            var out: [(start: Int, w: [Float])] = []
            for i in 0..<outSize {
                let center = Float(Double(scale) * (Double(i) + 0.5))
                let xmin = max(Int(Double(center - support) + 0.5), 0)
                let xsize = min(max(min(Int(Double(center + support) + 0.5), inSize) - xmin, 0), kmax)
                var ws: [Float] = []
                var total: Float = 0
                for j in 0..<xsize {
                    let arg = Float((Double(Float(j + xmin) - center) + 0.5) * Double(invscale))
                    let x = abs(arg)
                    let w: Float = x < 1.0 ? Float(1.0 - Double(x)) : 0
                    ws.append(w)
                    total = total + w
                }
                if total != 0 { ws = ws.map { $0 / total } }
                out.append((xmin, ws))
            }
            return out
        }

        /// One float32 pass over [rows, cols, dim] along the rows (`alongRows`, the height) or the columns.
        static func linearPass(_ t: [Float], rows: Int, cols: Int, dim: Int, alongRows: Bool, out n: Int,
                               fused: Bool) -> [Float]
        {
            let wts = weights(inSize: alongRows ? rows : cols, outSize: n)
            let orows = alongRows ? n : rows, ocols = alongRows ? cols : n
            var out = [Float](repeating: 0, count: orows * ocols * dim)
            t.withUnsafeBufferPointer { s in
                out.withUnsafeMutableBufferPointer { o in
                    for r in 0..<orows {
                        for c in 0..<ocols {
                            let (start, ws) = alongRows ? wts[r] : wts[c]
                            let dst = (r * ocols + c) * dim
                            guard !ws.isEmpty else { continue }
                            for k in 0..<dim {
                                @inline(__always) func tap(_ j: Int) -> Float {
                                    alongRows ? s[((start + j) * cols + c) * dim + k] : s[(r * cols + start + j) * dim + k]
                                }
                                var acc = tap(0) * ws[0]
                                for j in 1..<ws.count {
                                    acc = fused ? acc.addingProduct(tap(j), ws[j]) : acc + tap(j) * ws[j]
                                }
                                o[dst + k] = acc
                            }
                        }
                    }
                }
            }
            return out
        }

        /// The table resized to (h, w) patches -> [h * w, dim] row-major (§6): the width pass, then the height pass,
        /// each skipped when that side stays 16. `fused: false` rounds every product before its add (the gate's
        /// negative control; torch fuses).
        public func resized(height h: Int, width w: Int, fused: Bool = true) -> [Float] {
            var t = values
            let side = D1Pixels.positionSide
            if w != side { t = Self.linearPass(t, rows: side, cols: side, dim: dim, alongRows: false, out: w, fused: fused) }
            if h != side { t = Self.linearPass(t, rows: side, cols: w, dim: dim, alongRows: true, out: h, fused: fused) }
            return t
        }
    }

    // MARK: 7 unshuffle and the tower's inputs

    /// [h/2 * w/2, 4] int32: the four patch rows merged token k concatenates, in channel order.
    public static func unshuffleIndex(height h: Int, width w: Int) throws -> [Int32] {
        let f = D1Vision.factor
        guard h % f == 0, w % f == 0 else { throw D1Error.request("grid \(h)x\(w) not divisible by \(f)") }
        var out: [Int32] = []
        out.reserveCapacity(h * w)
        for i in 0..<(h / f) {
            for j in 0..<(w / f) {
                out += [Int32(2 * i * w + 2 * j), Int32(2 * i * w + 2 * j + 1), Int32((2 * i + 1) * w + 2 * j),
                        Int32((2 * i + 1) * w + 2 * j + 1)]
            }
        }
        return out
    }

    /// The fixed-shape inputs of the vision tower graph for one crop (`vision_host.tower_inputs`).
    public struct TowerInputs: Sendable {
        /// [1024, 768] float32, rows past the crop's patches 0.0
        public let patches: [Float]
        /// [1024, d] float32, rows past the crop's patches 0.0 (empty without a table)
        public let posTable: [Float]
        /// [1024] float32: 0 for a real patch, -inf for padding
        public let keyBias: [Float]
        /// [256, 4] int32, rows past the crop's tokens 0
        public let unshuffleIndex: [Int32]
        public let gridHeight: Int
        public let gridWidth: Int
        public var patchCount: Int { gridHeight * gridWidth }
        public var tokenCount: Int { (gridHeight / D1Vision.factor) * (gridWidth / D1Vision.factor) }
    }

    /// One crop's uint8 pixels -> its tower inputs. `positions` is the table already resized to this crop's grid
    /// ([h * w, d], `PositionTable.resized`); nil leaves `posTable` empty.
    public static func towerInputs(crop: RGB, positions: [Float]?) throws -> TowerInputs {
        let rows = try patches(crop)
        let gh = crop.height / D1Vision.patch, gw = crop.width / D1Vision.patch, n = gh * gw
        guard n <= maxPatches else {
            throw D1Error.request("crop of \(n) patches over \(maxPatches) (the processor would not pad it)")
        }
        var padded = [Float](repeating: 0, count: maxPatches * patchDim)
        padded.replaceSubrange(0..<rows.count, with: rows)
        var keyBias = [Float](repeating: -Float.infinity, count: maxPatches)
        for i in 0..<n { keyBias[i] = 0 }
        var index = [Int32](repeating: 0, count: maxTokens * 4)
        let idx = try unshuffleIndex(height: gh, width: gw)
        index.replaceSubrange(0..<idx.count, with: idx)
        var pos: [Float] = []
        if let positions {
            guard positions.count % n == 0, positions.count > 0 else {
                throw D1Error.contract("position rows: \(positions.count) values for a \(gh)x\(gw) grid")
            }
            pos = [Float](repeating: 0, count: maxPatches * (positions.count / n))
            pos.replaceSubrange(0..<positions.count, with: positions)
        }
        return TowerInputs(patches: padded, posTable: pos, keyBias: keyBias, unshuffleIndex: index, gridHeight: gh,
                           gridWidth: gw)
    }

    /// Every crop of a picture -> its tower inputs, the position table resized once per distinct grid.
    public static func towerInputs(_ picture: Picture, table: PositionTable?, fused: Bool = true) throws -> [TowerInputs] {
        var resized: [Int: [Float]] = [:]
        return try picture.crops.map { crop in
            let gh = crop.height / D1Vision.patch, gw = crop.width / D1Vision.patch
            var positions: [Float]?
            if let table {
                let key = gh * 100_000 + gw
                if resized[key] == nil { resized[key] = table.resized(height: gh, width: gw, fused: fused) }
                positions = resized[key]
            }
            return try towerInputs(crop: crop, positions: positions)
        }
    }
}
