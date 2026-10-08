// ImagePreprocess — an image file -> the vision graph's four inputs, one set per crop (conversion/d1_omni/host.py §5,
// the publisher's vision.preprocess() written out; round 9 gated each step against the Python host bit for bit):
//
//   decode        PNG / JPEG -> RGB uint8 as PIL's `Image.open(f).convert("RGB")` reads it: ImageIO's decoded bytes,
//                 no EXIF rotation, no colour management (an embedded ICC profile is not applied), alpha dropped
//   crops         MediaLength.layout: a tiled image is resized to its grid (columns x 512, rows x 512) and cut into
//                 512 px tiles, row-major; then the thumbnail, resized from the image (the same size: no resize)
//   resize        torchvision's resize(uint8, BILINEAR, antialias=True) as it runs on a CPU without AVX2 (Apple
//                 silicon; host.resize_uint8_antialias_numpy): float32, torch's separable antialias kernel — the width
//                 pass, then the height pass, a pass skipped when its side does not change; per output index the
//                 triangle weights of `aaWeights`; out = t0 * w0, then out = fma(tj, wj, out) — rounded half to even
//                 and cast back to uint8
//   pixel_values  (x - 127.5) / 127.5 in float32; 16 px patches row-major over (ph, pw), [py][px][c] inside one
//                 (channel fastest); 0.0 past ph * pw, 1,024 rows of 768
//   pos_embed     the bundle's position table [16, 16, 768] (position_table.f32) resized to (ph, pw) by the same
//                 float32 kernel (host.position_embeddings_numpy), row-major; 0.0 past ph * pw
//   patch_mask    1.0 on the crop's patches; unshuffle_index [256, 4] int32: token (i, j) = row i * (pw / 2) + j takes
//                 the patches 2i*pw + 2j, 2i*pw + 2j+1, (2i+1)*pw + 2j, (2i+1)*pw + 2j+1 (host.unshuffle_index)
//
// The crop's prefix is the vision graph's first (ph / 2)(pw / 2) rows (VisionGraph); an image's prefix is its crops'
// in order, a request's is its images' in request order.

import CoreGraphics
import Foundation
import ImageIO

/// An RGB image as PIL's `convert("RGB")` holds it: interleaved [height][width][3] uint8.
public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    public let pixels: [UInt8]
    /// which decoder read the file: "imageio" or "jpeg-baseline" (JPEGBaseline, libjpeg's arithmetic)
    public let decoder: String

    public init(width: Int, height: Int, pixels: [UInt8], decoder: String = "imageio") {
        precondition(pixels.count == width * height * 3)
        self.width = width
        self.height = height
        self.pixels = pixels
        self.decoder = decoder
    }

    /// The planar copy [3][height][width] (host.rgb_uint8's layout).
    public var planar: Planes { Planes(rgb: self) }
}

/// A uint8 image in planes: [3][height][width] (the publisher's CHW tensor).
public struct Planes: Sendable {
    public let height: Int
    public let width: Int
    public var values: [UInt8]

    public init(height: Int, width: Int, values: [UInt8]) {
        precondition(values.count == 3 * height * width)
        self.height = height
        self.width = width
        self.values = values
    }

    init(rgb: RGBImage) {
        let n = rgb.width * rgb.height
        var v = [UInt8](repeating: 0, count: 3 * n)
        rgb.pixels.withUnsafeBufferPointer { src in
            v.withUnsafeMutableBufferPointer { dst in
                for i in 0..<n {
                    dst[i] = src[3 * i]
                    dst[n + i] = src[3 * i + 1]
                    dst[2 * n + i] = src[3 * i + 2]
                }
            }
        }
        self.init(height: rgb.height, width: rgb.width, values: v)
    }

    /// Rows [top, top + h) and columns [left, left + w) of every plane.
    func cut(top: Int, left: Int, height h: Int, width w: Int) -> Planes {
        var v = [UInt8](repeating: 0, count: 3 * h * w)
        for c in 0..<3 {
            for y in 0..<h {
                let src = c * height * width + (top + y) * width + left
                let dst = c * h * w + y * w
                v.replaceSubrange(dst..<(dst + w), with: values[src..<(src + w)])
            }
        }
        return Planes(height: h, width: w, values: v)
    }
}

/// One crop's four vision-graph inputs (host.crop_inputs), flat row-major, and what the host reads back.
public struct CropInputs: Sendable {
    /// (ph, pw): the crop's patch grid
    public let grid: (rows: Int, columns: Int)
    /// (ph / 2)(pw / 2): the prefix rows this crop gives
    public let tokens: Int
    public let pixelValues: [Float]      // [1, 1024, 768]
    public let posEmbed: [Float]         // [1, 1024, 768]
    public let patchMask: [Float]        // [1, 1024]
    public let unshuffleIndex: [Int32]   // [256, 4]

    public init(grid: (rows: Int, columns: Int), tokens: Int, pixelValues: [Float], posEmbed: [Float], patchMask: [Float],
                unshuffleIndex: [Int32]) {
        self.grid = grid
        self.tokens = tokens
        self.pixelValues = pixelValues
        self.posEmbed = posEmbed
        self.patchMask = patchMask
        self.unshuffleIndex = unshuffleIndex
    }
}

public enum ImagePreprocess {
    public static let patch = 16, tile = 512, maxPatches = 1024, maxTokens = 256, patchDim = 768, hidden = 768

    // MARK: decode

    /// The file's first image as PIL reads it (`convert("RGB")`): ImageIO's decoded bytes, unrotated, no colour
    /// matching. 8-bit RGB / RGBA / RGBX (any byte order), 8-bit gray (+ alpha), 16-bit components (the high byte, as
    /// PIL's ";16B" unpackers), an indexed colour table; anything else throws.
    public static func decode(contentsOf url: URL) throws -> RGBImage {
        try decode(data: Data(contentsOf: url), name: url.lastPathComponent)
    }

    public static func decode(data: Data, name: String = "image") throws -> RGBImage {
        // a JPEG of the form JPEGBaseline reads is decoded with libjpeg's arithmetic (ImageIO's JPEG decoder differs
        // from PIL's libjpeg-turbo by 1 to 3 levels); any other JPEG, and every other format, goes to ImageIO
        if data.count > 2, data[data.startIndex] == 0xFF, data[data.startIndex + 1] == 0xD8, let rgb = try JPEGBaseline.decode(data) {
            return rgb
        }
        let options = [kCGImageSourceShouldAllowFloat: false, kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, options),
              let image = CGImageSourceCreateImageAtIndex(source, 0, options)
        else { throw D1OmniError.request("\(name): not an image ImageIO can decode") }
        return try rgb(of: image, name: name)
    }

    /// The decoded bytes of a CGImage (its data provider: before any colour matching) as RGB.
    public static func rgb(of image: CGImage, name: String = "image") throws -> RGBImage {
        let w = image.width, h = image.height
        guard w > 0, h > 0, let provider = image.dataProvider, let cf = provider.data else {
            throw D1OmniError.request("\(name): no pixel data")
        }
        let bpc = image.bitsPerComponent, bpp = image.bitsPerPixel, bpr = image.bytesPerRow
        let alpha = image.alphaInfo
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let model = image.colorSpace?.model ?? .unknown
        let length = CFDataGetLength(cf)
        guard let base = CFDataGetBytePtr(cf), length >= bpr * (h - 1) + (bpp * w + 7) / 8 else {
            throw D1OmniError.request("\(name): \(length) bytes for \(w)x\(h), \(bpr) per row")
        }
        guard bpc == 8 || bpc == 16 else { throw D1OmniError.request("\(name): \(bpc) bits per component") }
        let premultiplied = alpha == .premultipliedLast || alpha == .premultipliedFirst
        let hasAlphaFirst = alpha == .first || alpha == .premultipliedFirst || alpha == .noneSkipFirst
        let little = order == .byteOrder32Little || order == .byteOrder16Little
        var out = [UInt8](repeating: 0, count: w * h * 3)
        var alphaBelowOpaque = false

        // one component (8 or 16 bits) of pixel x, component index k in memory order
        func component(_ row: UnsafePointer<UInt8>, _ x: Int, _ k: Int, _ perPixel: Int) -> UInt8 {
            if bpc == 8 { return row[x * perPixel + k] }
            let p = row + (x * perPixel + k) * 2
            return little ? p[1] : p[0]   // the high byte of the 16-bit value
        }

        switch model {
        case .rgb:
            let perPixel = bpp / bpc
            guard perPixel == 3 || perPixel == 4 else { throw D1OmniError.request("\(name): \(bpp) bits per pixel") }
            // memory order of R, G, B (and A) within a pixel
            var idx: (r: Int, g: Int, b: Int, a: Int?)
            if perPixel == 3 {
                idx = (0, 1, 2, nil)
            } else if hasAlphaFirst {
                idx = little && bpc == 8 ? (2, 1, 0, 3) : (1, 2, 3, 0)          // BGRA (32Little) or ARGB
            } else {
                idx = little && bpc == 8 ? (3, 2, 1, 0) : (0, 1, 2, 3)          // ABGR (32Little) or RGBA
            }
            if bpc == 16 && little && perPixel == 4 {
                // 16Little reverses the bytes of each component, not the components
                idx = hasAlphaFirst ? (1, 2, 3, 0) : (0, 1, 2, 3)
            }
            for y in 0..<h {
                let row = base + y * bpr
                for x in 0..<w {
                    let o = (y * w + x) * 3
                    out[o] = component(row, x, idx.r, perPixel)
                    out[o + 1] = component(row, x, idx.g, perPixel)
                    out[o + 2] = component(row, x, idx.b, perPixel)
                    if premultiplied, let a = idx.a, component(row, x, a, perPixel) != 255 { alphaBelowOpaque = true }
                }
            }
        case .monochrome:
            let perPixel = bpp / bpc
            guard perPixel == 1 || perPixel == 2 else { throw D1OmniError.request("\(name): gray with \(bpp) bits per pixel") }
            let g = perPixel == 2 && hasAlphaFirst ? 1 : 0
            for y in 0..<h {
                let row = base + y * bpr
                for x in 0..<w {
                    let v = component(row, x, g, perPixel)
                    let o = (y * w + x) * 3
                    out[o] = v
                    out[o + 1] = v
                    out[o + 2] = v
                    if premultiplied, perPixel == 2, component(row, x, 1 - g, perPixel) != 255 { alphaBelowOpaque = true }
                }
            }
        case .indexed:
            guard bpc == 8, bpp == 8, let space = image.colorSpace, let baseSpace = space.baseColorSpace,
                  baseSpace.model == .rgb, let table = space.colorTable
            else { throw D1OmniError.request("\(name): indexed colours other than 8-bit RGB") }
            let count = table.count / 3
            for y in 0..<h {
                let row = base + y * bpr
                for x in 0..<w {
                    let i = Int(row[x])
                    guard i < count else { throw D1OmniError.request("\(name): colour index \(i) past the table (\(count))") }
                    let o = (y * w + x) * 3
                    out[o] = table[3 * i]
                    out[o + 1] = table[3 * i + 1]
                    out[o + 2] = table[3 * i + 2]
                }
            }
        default:
            throw D1OmniError.request("\(name): colour model \(model.rawValue) (RGB, gray and indexed are read)")
        }
        if alphaBelowOpaque {
            throw D1OmniError.request("\(name): ImageIO premultiplied a translucent pixel; PIL's stored RGB is not recoverable")
        }
        return RGBImage(width: w, height: h, pixels: out)
    }

    // MARK: resize (torch's float32 antialias kernel)

    /// One output index of one axis: its first tap and the taps' weights.
    public struct Taps: Sendable {
        public let start: Int
        public let weights: [Float]
    }

    /// The antialiased bilinear (triangle) weights of one axis as aten's _compute_index_ranges_weights makes them for
    /// a float32 input (host._aa_weights_f32): scale = n_in / n_out and support = max(scale, 1) in float32; the centre
    /// scale * (i + 0.5) through double; the first tap trunc(centre - support + 0.5), the count up to
    /// trunc(centre + support + 0.5) clamped to n_in and to ceil(support) * 2 + 1; each tap's filter argument
    /// (j + first - centre + 0.5) * invscale through double, the weight 1 - |x| (0 past 1) through double; the weights
    /// summed and divided in float32.
    public static func aaWeights(input nIn: Int, output nOut: Int) -> [Taps] {
        let scale = Float(nIn) / Float(nOut)
        let support: Float = scale >= 1 ? scale : 1
        let invscale: Float = scale >= 1 ? Float(1.0 / Double(scale)) : 1
        let maxTaps = Int(support.rounded(.up)) * 2 + 1
        var rows: [Taps] = []
        rows.reserveCapacity(nOut)
        for i in 0..<nOut {
            let centre = Float(Double(scale) * (Double(i) + 0.5))
            let lo = max(Int(Double(centre - support) + 0.5), 0)
            let size = min(max(min(Int(Double(centre + support) + 0.5), nIn) - lo, 0), maxTaps)
            var w = [Float](repeating: 0, count: size)
            var total: Float = 0
            for j in 0..<size {
                let x = abs(Float((Double(Float(j + lo) - centre) + 0.5) * Double(invscale)))
                w[j] = x < 1 ? Float(1.0 - Double(x)) : 0
                total += w[j]
            }
            if total != 0 { for j in 0..<size { w[j] /= total } }
            rows.append(Taps(start: lo, weights: w))
        }
        return rows
    }

    /// One separable pass of torch's kernel over a float32 array viewed as [outer][n][inner] along the middle axis:
    /// out[o][i][k] = src[o][s][k] * w0, then fma(src[o][s + j][k], wj, out) (host._aa_pass_f32).
    static func pass(_ src: [Float], outer: Int, n: Int, inner: Int, taps: [Taps]) -> [Float] {
        let m = taps.count
        var out = [Float](repeating: 0, count: outer * m * inner)
        src.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                for o in 0..<outer {
                    for (i, t) in taps.enumerated() {
                        let dst = (o * m + i) * inner
                        let first = (o * n + t.start) * inner
                        let w0 = t.weights.first ?? 0
                        for k in 0..<inner { d[dst + k] = s[first + k] * w0 }
                        if t.weights.count > 1 {
                            for j in 1..<t.weights.count {
                                let wj = t.weights[j]
                                let row = first + j * inner
                                for k in 0..<inner { d[dst + k] = d[dst + k].addingProduct(s[row + k], wj) }
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    /// torchvision's resize(uint8 [3, h, w], [height, width], BILINEAR, antialias=True) on a CPU without AVX2
    /// (host.resize_uint8_antialias_numpy).
    public static func resize(_ image: Planes, height: Int, width: Int) -> Planes {
        if height == image.height && width == image.width { return image }
        var x = image.values.map { Float($0) }
        var h = image.height, w = image.width
        if width != w {
            x = pass(x, outer: 3 * h, n: w, inner: 1, taps: aaWeights(input: w, output: width))
            w = width
        }
        if height != h {
            x = pass(x, outer: 3, n: h, inner: w, taps: aaWeights(input: h, output: height))
            h = height
        }
        let v = x.map { UInt8(min(max($0.rounded(.toNearestOrEven), 0), 255)) }
        return Planes(height: h, width: w, values: v)
    }

    /// vision.preprocess()'s crops of one image, in order (host.crop_pixels_numpy).
    public static func crops(_ rgb: RGBImage) throws -> [Planes] {
        let plan = try MediaLength.layout(width: rgb.width, height: rgb.height)
        let planes = rgb.planar
        var out: [Planes] = []
        if plan.tiled {
            let (gw, gh) = plan.grid
            let big = resize(planes, height: gh * tile, width: gw * tile)
            for r in 0..<gh {
                for c in 0..<gw { out.append(big.cut(top: r * tile, left: c * tile, height: tile, width: tile)) }
            }
        }
        out.append(resize(planes, height: plan.thumbnail.0, width: plan.thumbnail.1))
        return out
    }

    // MARK: graph inputs

    /// The checkpoint's 16x16 position table [16][16][768] float32 (the vision bundle's position_table.f32).
    public static func positionTable(contentsOf url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count == 16 * 16 * hidden * 4 else { throw D1OmniError.bundle("\(url.path): \(data.count) bytes, not 16x16x768 float32") }
        return data.withUnsafeBytes { raw in (0..<(16 * 16 * hidden)).map { Float(bitPattern: UInt32(littleEndian: raw.load(fromByteOffset: $0 * 4, as: UInt32.self))) } }
    }

    /// The table resized to (ph, pw) (host.position_embeddings_numpy = F.interpolate(bilinear, antialias) in float32):
    /// the width pass, then the height pass -> [ph * pw][768].
    public static func positions(_ table: [Float], rows ph: Int, columns pw: Int) -> [Float] {
        var x = table
        if pw != 16 { x = pass(x, outer: 16, n: 16, inner: hidden, taps: aaWeights(input: 16, output: pw)) }
        if ph != 16 { x = pass(x, outer: 1, n: 16, inner: pw * hidden, taps: aaWeights(input: 16, output: ph)) }
        return x
    }

    /// host.unshuffle_index: [256][4] int32, 0 past the crop's tokens.
    public static func unshuffleIndex(rows ph: Int, columns pw: Int) throws -> [Int32] {
        guard ph % 2 == 0, pw % 2 == 0, ph * pw <= maxPatches else {
            throw D1OmniError.request("grid \(ph)x\(pw): needs even sides and at most \(maxPatches) patches")
        }
        var index = [Int32](repeating: 0, count: maxTokens * 4)
        let half = pw / 2
        for t in 0..<((ph / 2) * half) {
            let i = t / half, j = t % half
            index[4 * t] = Int32(2 * i * pw + 2 * j)
            index[4 * t + 1] = Int32(2 * i * pw + 2 * j + 1)
            index[4 * t + 2] = Int32((2 * i + 1) * pw + 2 * j)
            index[4 * t + 3] = Int32((2 * i + 1) * pw + 2 * j + 1)
        }
        return index
    }

    /// One crop -> its four inputs (host.crop_inputs with the NumPy forms).
    public static func inputs(_ crop: Planes, table: [Float]) throws -> CropInputs {
        let ph = crop.height / patch, pw = crop.width / patch
        let n = ph * pw
        guard crop.height % patch == 0, crop.width % patch == 0, n <= maxPatches else {
            throw D1OmniError.request("crop \(crop.height)x\(crop.width): not a multiple of \(patch) within \(maxPatches) patches")
        }
        var pixels = [Float](repeating: 0, count: maxPatches * patchDim)
        let hw = crop.height * crop.width
        crop.values.withUnsafeBufferPointer { v in
            pixels.withUnsafeMutableBufferPointer { p in
                for py in 0..<ph {
                    for px in 0..<pw {
                        let row = (py * pw + px) * patchDim
                        for y in 0..<patch {
                            for x in 0..<patch {
                                let src = (py * patch + y) * crop.width + px * patch + x
                                let dst = row + (y * patch + x) * 3
                                for c in 0..<3 { p[dst + c] = (Float(v[c * hw + src]) - 127.5) / 127.5 }
                            }
                        }
                    }
                }
            }
        }
        var pos = [Float](repeating: 0, count: maxPatches * hidden)
        pos.replaceSubrange(0..<(n * hidden), with: positions(table, rows: ph, columns: pw))
        var mask = [Float](repeating: 0, count: maxPatches)
        for i in 0..<n { mask[i] = 1 }
        return CropInputs(grid: (ph, pw), tokens: n / 4, pixelValues: pixels, posEmbed: pos, patchMask: mask,
                          unshuffleIndex: try unshuffleIndex(rows: ph, columns: pw))
    }
}
