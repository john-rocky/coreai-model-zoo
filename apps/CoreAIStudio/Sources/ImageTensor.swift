// ImageTensor.swift — CGImage <-> NDArray conversion, tiling, and pad/crop helpers shared by
// UpscaleEngine (AdcSR) and FrameInterpolationEngine (RIFE).
//
// API usage here is verified against the REAL macOS 27.0 SDK (Xcode-beta 27.0), read directly
// from CoreAIRuntime's .swiftinterface this session — not guessed from documentation drafts.
// `import CoreAI` re-exports CoreAIDelegates -> CoreAIAsset/CoreAICommon/CoreAICompiler/
// CoreAIRuntime, so AIModel/InferenceFunction/NDArray/SpecializationOptions/ComputeUnitKind are
// all available without any external package (see project.yml).
//
// CGImage traps (knowledge/adcsr-super-resolution.md, apps/CoreAISegment's
// SegmentationEngine.renderOverlay): read the CGContext's REAL bytesPerRow (pass 0, never
// assume width*4 — CG pads non-16-aligned widths); a standard top-down CGBitmapContext needs
// NO y-flip (drawing already maps the image's top to row 0).

import CoreAI
import CoreGraphics
import Foundation

enum ImageTensorError: Error {
    case contextCreationFailed
    case imageCreationFailed
    case unexpectedShape([Int])
}

// MARK: - CGImage -> NDArray

/// image -> NDArray [1,3,H,W] float32 RGB in [0,1]. Ignores alpha (both AdcSR and RIFE are
/// opaque-RGB models).
func ndArray(from image: CGImage) throws -> NDArray {
    let w = image.width, h = image.height
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { throw ImageTensorError.contextCreationFailed }
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    guard let data = ctx.data else { throw ImageTensorError.contextCreationFailed }
    let bytesPerRow = ctx.bytesPerRow
    let buf = data.bindMemory(to: UInt8.self, capacity: bytesPerRow * h)

    var arr = NDArray(shape: [1, 3, h, w], scalarType: .float32)
    let view = arr.mutableView(as: Float.self)
    view.withUnsafeMutablePointer { ptr, shape, strides in
        let cStride = strides[1], hStride = strides[2], wStride = strides[3]
        for y in 0..<h {
            let row = buf + y * bytesPerRow
            for x in 0..<w {
                let p = row + x * 4  // RGBA8, premultipliedLast
                let base = y * hStride + x * wStride
                ptr[base + 0 * cStride] = Float(p[0]) / 255.0
                ptr[base + 1 * cStride] = Float(p[1]) / 255.0
                ptr[base + 2 * cStride] = Float(p[2]) / 255.0
            }
        }
    }
    return arr
}

// MARK: - NDArray -> CGImage

/// NDArray [1,3,H,W] or [3,H,W] float32 (any range) -> CGImage, clamping to [0,1] and mapping
/// to [0,255]. `scale`/`offset` let a caller feed a [-1,1]-range tensor directly
/// (AdcSR's raw `sr` output) without a separate normalization pass: displayed = clamp(x*scale+offset).
func cgImage(from arr: NDArray, scale: Float = 1.0, offset: Float = 0.0) throws -> CGImage {
    let shape = arr.shape
    let (h, w): (Int, Int)
    let chanAxis: Int
    switch shape.count {
    case 4 where shape[0] == 1: h = shape[2]; w = shape[3]; chanAxis = 1
    case 3: h = shape[1]; w = shape[2]; chanAxis = 0
    default: throw ImageTensorError.unexpectedShape(shape)
    }
    guard shape[chanAxis] == 3 else { throw ImageTensorError.unexpectedShape(shape) }

    var pixels = [UInt8](repeating: 255, count: w * h * 4)
    let view = arr.view(as: Float.self)
    view.withUnsafePointer { ptr, _, strides in
        let base0 = shape.count == 4 ? 0 : 0
        let cStride = strides[chanAxis]
        let hStride = strides[chanAxis + 1]
        let wStride = strides[chanAxis + 2]
        for y in 0..<h {
            for x in 0..<w {
                let off = base0 + y * hStride + x * wStride
                let r = ptr[off + 0 * cStride] * scale + offset
                let g = ptr[off + 1 * cStride] * scale + offset
                let b = ptr[off + 2 * cStride] * scale + offset
                let p = (y * w + x) * 4
                pixels[p + 0] = UInt8(max(0, min(255, r * 255.0 + 0.5)))
                pixels[p + 1] = UInt8(max(0, min(255, g * 255.0 + 0.5)))
                pixels[p + 2] = UInt8(max(0, min(255, b * 255.0 + 0.5)))
            }
        }
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: &pixels, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ), let out = ctx.makeImage() else { throw ImageTensorError.imageCreationFailed }
    return out
}

// MARK: - Resize / pad / crop (CGImage level — kept separate from the tensor boundary)

func resizeImage(_ image: CGImage, toWidth w: Int, height h: Int) throws -> CGImage {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { throw ImageTensorError.contextCreationFailed }
    ctx.interpolationQuality = .high
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
    guard let out = ctx.makeImage() else { throw ImageTensorError.imageCreationFailed }
    return out
}

/// Pads (bottom/right, replicated edge) to the next multiple of `multiple`. Returns the padded
/// image plus the original (unpadded) size so the caller can crop back after inference — the
/// RIFE static-shape contract (conversion/export_rife.py: padded_H/W % 64 == 0).
func padImage(_ image: CGImage, toMultipleOf multiple: Int) throws -> (padded: CGImage, originalWidth: Int, originalHeight: Int) {
    let w = image.width, h = image.height
    let paddedW = ((w + multiple - 1) / multiple) * multiple
    let paddedH = ((h + multiple - 1) / multiple) * multiple
    if paddedW == w && paddedH == h { return (image, w, h) }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(
        data: nil, width: paddedW, height: paddedH, bitsPerComponent: 8, bytesPerRow: 0,
        space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { throw ImageTensorError.contextCreationFailed }
    // CGContext is bottom-left origin; draw the source flush with the TOP-left of the padded
    // canvas (padding added bottom/right) by offsetting Y by the extra height.
    ctx.draw(image, in: CGRect(x: 0, y: CGFloat(paddedH - h), width: CGFloat(w), height: CGFloat(h)))
    guard let out = ctx.makeImage() else { throw ImageTensorError.imageCreationFailed }
    return (out, w, h)
}

/// Crop `image` (assumed padded bottom/right, see `padImage`) back to `width`x`height` at the
/// top-left.
func cropImage(_ image: CGImage, toWidth width: Int, height: Int) throws -> CGImage {
    // CGImage.cropping(to:) works in the image's own top-left-origin pixel space directly.
    guard let out = image.cropping(to: CGRect(x: 0, y: 0, width: width, height: height)) else {
        throw ImageTensorError.imageCreationFailed
    }
    return out
}

func cropTile(_ image: CGImage, x: Int, y: Int, width: Int, height: Int) throws -> CGImage {
    guard let out = image.cropping(to: CGRect(x: x, y: y, width: width, height: height)) else {
        throw ImageTensorError.imageCreationFailed
    }
    return out
}

// MARK: - RGBBuffer (interleaved HWC float32) — the tiling/blend working format

/// A plain interleaved-HWC float32 RGB buffer, values nominally in [0,1] unless noted. Used
/// wherever per-tile CGImage<->NDArray round trips would be needlessly expensive (tiling,
/// accumulation, color-match) — mirrors the NumPy-array-as-working-format design of
/// apps/CoreAIStudio/reference/adcsr_host_reference.py.
struct RGBBuffer {
    var width: Int
    var height: Int
    var pixels: [Float]  // length width*height*3, row-major HWC

    init(width: Int, height: Int, fill: Float = 0) {
        self.width = width
        self.height = height
        self.pixels = [Float](repeating: fill, count: width * height * 3)
    }

    init(cgImage: CGImage) throws {
        let w = cgImage.width, h = cgImage.height
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw ImageTensorError.contextCreationFailed }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let data = ctx.data else { throw ImageTensorError.contextCreationFailed }
        let bytesPerRow = ctx.bytesPerRow
        let buf = data.bindMemory(to: UInt8.self, capacity: bytesPerRow * h)

        self.width = w
        self.height = h
        var px = [Float](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            let row = buf + y * bytesPerRow
            for x in 0..<w {
                let p = row + x * 4
                let o = (y * w + x) * 3
                px[o + 0] = Float(p[0]) / 255.0
                px[o + 1] = Float(p[1]) / 255.0
                px[o + 2] = Float(p[2]) / 255.0
            }
        }
        self.pixels = px
    }

    /// NDArray [1,3,H,W] or [3,H,W] float32 -> RGBBuffer, applying `value * scale + offset` on
    /// read (e.g. scale=0.5, offset=0.5 to map a model's raw [-1,1] output into [0,1]).
    init(ndArray arr: NDArray, scale: Float = 1, offset: Float = 0) throws {
        let shape = arr.shape
        let (h, w): (Int, Int)
        let chanAxis: Int
        switch shape.count {
        case 4 where shape[0] == 1: h = shape[2]; w = shape[3]; chanAxis = 1
        case 3: h = shape[1]; w = shape[2]; chanAxis = 0
        default: throw ImageTensorError.unexpectedShape(shape)
        }
        guard shape[chanAxis] == 3 else { throw ImageTensorError.unexpectedShape(shape) }

        self.width = w
        self.height = h
        var px = [Float](repeating: 0, count: w * h * 3)
        let view = arr.view(as: Float.self)
        view.withUnsafePointer { ptr, _, strides in
            let cStride = strides[chanAxis], hStride = strides[chanAxis + 1], wStride = strides[chanAxis + 2]
            for y in 0..<h {
                for x in 0..<w {
                    let off = y * hStride + x * wStride
                    let o = (y * w + x) * 3
                    px[o + 0] = ptr[off + 0 * cStride] * scale + offset
                    px[o + 1] = ptr[off + 1 * cStride] * scale + offset
                    px[o + 2] = ptr[off + 2 * cStride] * scale + offset
                }
            }
        }
        self.pixels = px
    }

    /// -> NDArray [1,3,H,W] float32, applying `value * scale + offset` on write (e.g.
    /// scale=2, offset=-1 to map [0,1] pixels into a model's expected [-1,1] input range).
    func toNDArray(scale: Float = 1, offset: Float = 0) -> NDArray {
        var arr = NDArray(shape: [1, 3, height, width], scalarType: .float32)
        let view = arr.mutableView(as: Float.self)
        view.withUnsafeMutablePointer { ptr, _, strides in
            let cStride = strides[1], hStride = strides[2], wStride = strides[3]
            for y in 0..<height {
                for x in 0..<width {
                    let o = (y * width + x) * 3
                    let base = y * hStride + x * wStride
                    ptr[base + 0 * cStride] = pixels[o + 0] * scale + offset
                    ptr[base + 1 * cStride] = pixels[o + 1] * scale + offset
                    ptr[base + 2 * cStride] = pixels[o + 2] * scale + offset
                }
            }
        }
        return arr
    }

    func toCGImage() throws -> CGImage {
        var out = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            let o = i * 3
            out[i * 4 + 0] = UInt8(max(0, min(255, pixels[o + 0] * 255.0 + 0.5)))
            out[i * 4 + 1] = UInt8(max(0, min(255, pixels[o + 1] * 255.0 + 0.5)))
            out[i * 4 + 2] = UInt8(max(0, min(255, pixels[o + 2] * 255.0 + 0.5)))
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &out, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let img = ctx.makeImage() else { throw ImageTensorError.imageCreationFailed }
        return img
    }

    /// Extract a `w`x`h` crop at (x,y) — used to pull one LR tile out of the (possibly capped)
    /// source buffer.
    func cropped(x: Int, y: Int, width w: Int, height h: Int) -> RGBBuffer {
        var out = RGBBuffer(width: w, height: h)
        for row in 0..<h {
            let srcOff = ((y + row) * width + x) * 3
            let dstOff = row * w * 3
            out.pixels.withUnsafeMutableBufferPointer { dst in
                pixels.withUnsafeBufferPointer { src in
                    dst.baseAddress!.advanced(by: dstOff).update(
                        from: src.baseAddress!.advanced(by: srcOff), count: w * 3)
                }
            }
        }
        return out
    }

    /// Accumulate `tile` (an SR tile, already resolution-matched to this buffer's local scale)
    /// into `self` at (x,y), weighted by `weight` (row-major, length tile.width*tile.height) —
    /// the feather-blend accumulation step. `weightSum` is co-accumulated in the same loop.
    mutating func accumulate(_ tile: RGBBuffer, weight: [Float], at x: Int, y: Int, into weightSum: inout [Float]) {
        for row in 0..<tile.height {
            let dstRowBase = (y + row) * width
            let srcRowBase = row * tile.width
            for col in 0..<tile.width {
                let w = weight[srcRowBase + col]
                let dst = (dstRowBase + x + col) * 3
                let src = (srcRowBase + col) * 3
                pixels[dst + 0] += tile.pixels[src + 0] * w
                pixels[dst + 1] += tile.pixels[src + 1] * w
                pixels[dst + 2] += tile.pixels[src + 2] * w
                weightSum[dstRowBase + x + col] += w
            }
        }
    }

    /// In place: divide every pixel by the corresponding weight (post feather-blend
    /// normalization); weights <= `minWeight` are clamped to avoid a divide-by-zero.
    mutating func normalize(by weightSum: [Float], minWeight: Float = 1e-6) {
        for i in 0..<(width * height) {
            let w = max(weightSum[i], minWeight)
            pixels[i * 3 + 0] /= w
            pixels[i * 3 + 1] /= w
            pixels[i * 3 + 2] /= w
        }
    }

    /// Per-channel mean/std.
    func channelStats() -> (mean: (Float, Float, Float), std: (Float, Float, Float)) {
        let n = Float(width * height)
        var sum: (Float, Float, Float) = (0, 0, 0)
        for i in 0..<(width * height) {
            sum.0 += pixels[i * 3 + 0]; sum.1 += pixels[i * 3 + 1]; sum.2 += pixels[i * 3 + 2]
        }
        let mean = (sum.0 / n, sum.1 / n, sum.2 / n)
        var sq: (Float, Float, Float) = (0, 0, 0)
        for i in 0..<(width * height) {
            let d0 = pixels[i * 3 + 0] - mean.0, d1 = pixels[i * 3 + 1] - mean.1, d2 = pixels[i * 3 + 2] - mean.2
            sq.0 += d0 * d0; sq.1 += d1 * d1; sq.2 += d2 * d2
        }
        let std = (sqrtf(sq.0 / n), sqrtf(sq.1 / n), sqrtf(sq.2 / n))
        return (mean, std)
    }

    /// GLOBAL per-channel color-match: rescale so self's mean/std match `target`'s, then clamp
    /// to [0,1]. Must be applied ONCE, after stitching — never per-tile (a uniform tile's std
    /// is near-zero, and dividing by it blows up to a pure-white square; see
    /// adcsr_host_reference.py's module docstring).
    mutating func colorMatch(to target: RGBBuffer) {
        let (tMean, tStd) = target.channelStats()
        let (sMean, sStd) = channelStats()
        let means = [sMean.0, sMean.1, sMean.2], stds = [sStd.0, sStd.1, sStd.2]
        let tMeans = [tMean.0, tMean.1, tMean.2], tStds = [tStd.0, tStd.1, tStd.2]
        for i in 0..<(width * height) {
            for c in 0..<3 {
                let v = (pixels[i * 3 + c] - means[c]) / max(stds[c], 1e-6) * tStds[c] + tMeans[c]
                pixels[i * 3 + c] = max(0, min(1, v))
            }
        }
    }
}

// MARK: - Overlap-tile coverage + feather blend (ported from
// apps/CoreAIStudio/reference/adcsr_host_reference.py — see that file's docstring for why the
// exact stride/feather-curve choices here are an authored contract, not a reverse-engineered
// clone of the closed-source CoreAIKitVision `SuperResolver`).

/// Tile top-left offsets covering `size`, every tile fully inside bounds, last tile clamped
/// flush with the far edge (no gaps, no padding needed as long as size >= tile).
func tileOrigins(size: Int, tile: Int, stride: Int) -> [Int] {
    if size <= tile { return [0] }
    var origins = Swift.stride(from: 0, through: size - tile, by: stride).map { $0 }
    if origins.last != size - tile { origins.append(size - tile) }
    return origins
}

/// 1D weight for one axis: linear 0->1 ramp only on edges that have a NEIGHBORING tile — an
/// edge on the true image boundary (no neighbor) keeps full weight. Ramping every tile's outer
/// border down regardless (the bug the Python reference's self-test gate caught first) leaves
/// the whole canvas's outer rim at weight-sum == 0, a real hole, not a rounding artifact.
private func featherAxis(tilePx: Int, ramp: Int, hasPrev: Bool, hasNext: Bool) -> [Float] {
    var w = [Float](repeating: 1, count: tilePx)
    if ramp > 0 && hasPrev {
        for i in 0..<ramp { w[i] = Float(i) / Float(ramp) }
    }
    if ramp > 0 && hasNext {
        for i in 0..<ramp { w[tilePx - 1 - i] = Float(i) / Float(ramp) }
    }
    return w
}

/// 2D separable linear feather weight for ONE tile at a specific grid position.
func featherWeight(tilePx: Int, overlapPx: Int, scale: Int, hasLeft: Bool, hasRight: Bool, hasTop: Bool, hasBottom: Bool) -> [Float] {
    let ramp = overlapPx * scale
    let wx = featherAxis(tilePx: tilePx, ramp: ramp, hasPrev: hasLeft, hasNext: hasRight)
    let wy = featherAxis(tilePx: tilePx, ramp: ramp, hasPrev: hasTop, hasNext: hasBottom)
    var out = [Float](repeating: 0, count: tilePx * tilePx)
    for y in 0..<tilePx {
        for x in 0..<tilePx {
            out[y * tilePx + x] = wy[y] * wx[x]
        }
    }
    return out
}
