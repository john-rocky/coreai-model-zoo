// JPEGBaseline — a baseline JPEG decoded as PIL decodes it (Pillow's bundled libjpeg-turbo with the library defaults:
// JDCT_ISLOW, no DCT scaling, YCbCr -> RGB), for the files ImageIO decodes differently: ImageIO's JPEG decoder is not
// libjpeg's, and round 9 measured it 1 to 3 levels off PIL on 19 % of the bytes of a 640 x 480 photo.
//
//   entropy     baseline / extended sequential Huffman (SOF0 / SOF1), 8-bit samples, restart intervals
//   IDCT        jidctint.c jpeg_idct_islow: the integer slow-but-accurate IDCT (CONST_BITS 13, PASS1_BITS 2), the
//               coefficients dequantized in natural order, the post-IDCT range limit (10-bit wrap, then clamp)
//   colour      jdcolor.c ycc_rgb_convert: the 16-bit fixed-point tables (FIX(1.40200), FIX(1.77200), -FIX(0.71414),
//               -FIX(0.34414) + ONE_HALF), the sample range limit; one component = gray (R = G = B)
//
// Read here: one or three components with equal sampling factors (no chroma upsampling), which is what the fixture's
// JPEG is. Everything else — chroma subsampling (libjpeg's fancy upsampling is not written here), progressive or
// arithmetic coding, 12-bit, CMYK or an Adobe RGB transform — returns nil, and the caller decodes with ImageIO (not
// bit-equal to PIL for a JPEG).

import Foundation

public enum JPEGBaseline {
    struct Huffman {
        // (code length, code) -> symbol, as libjpeg's canonical code assignment
        var lookup: [UInt32: UInt8] = [:]   // key = length << 16 | code
        var maxLength = 0
    }

    struct Component {
        let id: Int
        let h: Int
        let v: Int
        let tq: Int
        var td = 0
        var ta = 0
    }

    struct BitReader {
        let data: [UInt8]
        var pos: Int
        var buffer: UInt32 = 0
        var bits = 0
        var marker: UInt8? = nil

        mutating func fill() {
            while bits <= 24 {
                var byte: UInt8 = 0
                if marker == nil, pos < data.count {
                    byte = data[pos]
                    if byte == 0xFF {
                        let next = pos + 1 < data.count ? data[pos + 1] : 0
                        if next == 0x00 {
                            pos += 2
                        } else {
                            marker = next   // a marker ends the segment: feed zeros (libjpeg does the same)
                            byte = 0
                        }
                    } else {
                        pos += 1
                    }
                }
                buffer |= UInt32(byte) << UInt32(24 - bits)
                bits += 8
            }
        }

        mutating func bit() -> Int {
            if bits == 0 { fill() }
            let b = Int(buffer >> 31)
            buffer <<= 1
            bits -= 1
            return b
        }

        mutating func receive(_ n: Int) -> Int {
            var v = 0
            for _ in 0..<n { v = (v << 1) | bit() }
            return v
        }

        mutating func decode(_ t: Huffman) throws -> UInt8 {
            var code: UInt32 = 0
            for length in 1...16 {
                code = (code << 1) | UInt32(bit())
                if let s = t.lookup[UInt32(length) << 16 | code] { return s }
            }
            throw D1OmniError.request("jpeg: bad Huffman code")
        }

        /// Restart: drop the buffered bits, skip the RSTn marker.
        mutating func restart() throws {
            buffer = 0
            bits = 0
            if marker == nil {
                // the marker may not have been reached by the fill yet
                while pos + 1 < data.count, !(data[pos] == 0xFF && data[pos + 1] != 0 && data[pos + 1] != 0xFF) { pos += 1 }
                guard pos + 1 < data.count else { throw D1OmniError.request("jpeg: missing restart marker") }
                marker = data[pos + 1]
            }
            guard let m = marker, (0xD0...0xD7).contains(m) else { throw D1OmniError.request("jpeg: expected RSTn") }
            // pos is at the 0xFF of the marker when the fill stopped there
            if pos + 1 < data.count, data[pos] == 0xFF, data[pos + 1] == m { pos += 2 }
            marker = nil
        }
    }

    static let zigzag: [Int] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
        28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
        54, 47, 55, 62, 63,
    ]

    /// The file as PIL reads it, or nil when its form is not one read here (see the header).
    public static func decode(_ data: Data) throws -> RGBImage? {
        let b = [UInt8](data)
        guard b.count > 4, b[0] == 0xFF, b[1] == 0xD8 else { return nil }
        var quant = [[Int]](repeating: [], count: 4)
        var dc = [Huffman](repeating: Huffman(), count: 4), ac = [Huffman](repeating: Huffman(), count: 4)
        var comps: [Component] = []
        var width = 0, height = 0, restartInterval = 0
        var adobeTransform: Int? = nil
        var sawJFIF = false
        var i = 2
        func u16(_ k: Int) -> Int { Int(b[k]) << 8 | Int(b[k + 1]) }
        while i + 4 <= b.count {
            guard b[i] == 0xFF else { throw D1OmniError.request("jpeg: marker expected at \(i)") }
            let m = b[i + 1]
            if m == 0xFF { i += 1; continue }
            if m == 0xD8 || (0xD0...0xD7).contains(m) { i += 2; continue }
            if m == 0xD9 { break }
            let length = u16(i + 2), body = i + 4, end = i + 2 + length
            guard end <= b.count else { throw D1OmniError.request("jpeg: segment past the end") }
            switch m {
            case 0xC0, 0xC1:
                guard b[body] == 8 else { return nil }
                height = u16(body + 1)
                width = u16(body + 3)
                let n = Int(b[body + 5])
                for k in 0..<n {
                    let o = body + 6 + 3 * k
                    comps.append(Component(id: Int(b[o]), h: Int(b[o + 1] >> 4), v: Int(b[o + 1] & 15), tq: Int(b[o + 2])))
                }
            case 0xC2, 0xC3, 0xC5...0xC7, 0xC9...0xCB, 0xCD...0xCF:
                return nil   // progressive, lossless, hierarchical or arithmetic
            case 0xDB:
                var k = body
                while k < end {
                    let pq = Int(b[k] >> 4), tq = Int(b[k] & 15)
                    var table = [Int](repeating: 0, count: 64)
                    for z in 0..<64 { table[zigzag[z]] = pq == 0 ? Int(b[k + 1 + z]) : u16(k + 1 + 2 * z) }
                    quant[tq] = table
                    k += 1 + (pq == 0 ? 64 : 128)
                }
            case 0xC4:
                var k = body
                while k < end {
                    let tc = Int(b[k] >> 4), th = Int(b[k] & 15)
                    let counts = (0..<16).map { Int(b[k + 1 + $0]) }
                    var t = Huffman()
                    var code: UInt32 = 0, s = k + 17
                    for length in 1...16 {
                        for _ in 0..<counts[length - 1] {
                            t.lookup[UInt32(length) << 16 | code] = b[s]
                            code += 1
                            s += 1
                            t.maxLength = length
                        }
                        code <<= 1
                    }
                    if tc == 0 { dc[th] = t } else { ac[th] = t }
                    k = s
                }
            case 0xDD:
                restartInterval = u16(body)
            case 0xEE:
                if length >= 12, b[body..<(body + 5)].elementsEqual("Adobe".utf8) { adobeTransform = Int(b[body + 11]) }
            case 0xE0:
                if length >= 7, b[body..<(body + 5)].elementsEqual([0x4A, 0x46, 0x49, 0x46, 0x00]) { sawJFIF = true }
            case 0xDA:
                let ns = Int(b[body])
                // libjpeg's colour space guess (jdapimin.c): three components are YCbCr unless an Adobe marker says
                // otherwise or, without JFIF / Adobe markers, the component ids are 'R', 'G', 'B'
                let rgbIDs = comps.map(\.id) == [82, 71, 66] && adobeTransform == nil && !sawJFIF
                guard comps.count == 1 || comps.count == 3, ns == comps.count, !rgbIDs,
                      Set(comps.map(\.h)).count == 1, Set(comps.map(\.v)).count == 1,
                      comps.count == 3 || (comps[0].h == 1 && comps[0].v == 1),
                      adobeTransform == nil || adobeTransform == (comps.count == 3 ? 1 : 0) else { return nil }
                for k in 0..<ns {
                    let cid = Int(b[body + 1 + 2 * k]), t = b[body + 2 + 2 * k]
                    guard let ci = comps.firstIndex(where: { $0.id == cid }) else { throw D1OmniError.request("jpeg: scan component \(cid)") }
                    comps[ci].td = Int(t >> 4)
                    comps[ci].ta = Int(t & 15)
                }
                let planes = try scan(b, start: end, width: width, height: height, comps: comps, quant: quant, dc: dc, ac: ac,
                                      restartInterval: restartInterval)
                return rgb(planes, width: width, height: height)
            default:
                break   // APPn, COM and the rest
            }
            i = end
        }
        return nil
    }

    /// One interleaved baseline scan (every component sampled alike: one block per component per MCU) -> each
    /// component's samples [height][width].
    static func scan(_ b: [UInt8], start: Int, width: Int, height: Int, comps: [Component], quant: [[Int]], dc: [Huffman],
                     ac: [Huffman], restartInterval: Int) throws -> [[UInt8]] {
        // an interleaved scan's MCU holds h x v blocks of each component (row-major); a one-component scan's MCU is one
        // block
        let (h, v) = comps.count == 1 ? (1, 1) : (comps[0].h, comps[0].v)
        let mcuW = 8 * h, mcuH = 8 * v
        let blocksPerMCU = h * v
        let mcusX = (width + mcuW - 1) / mcuW, mcusY = (height + mcuH - 1) / mcuH
        let stride = mcusX * mcuW
        var planes = [[UInt8]](repeating: [UInt8](repeating: 0, count: stride * mcusY * mcuH), count: comps.count)
        var reader = BitReader(data: b, pos: start)
        var pred = [Int](repeating: 0, count: comps.count)
        var coef = [Int](repeating: 0, count: 64)
        var left = restartInterval
        for my in 0..<mcusY {
            for mx in 0..<mcusX {
                if restartInterval > 0 {
                    if left == 0 {
                        try reader.restart()
                        pred = [Int](repeating: 0, count: comps.count)
                        left = restartInterval
                    }
                    left -= 1
                }
                for (ci, c) in comps.enumerated() {
                    for blk in 0..<blocksPerMCU {
                        for k in 0..<64 { coef[k] = 0 }
                        // DC
                        let s = Int(try reader.decode(dc[c.td]))
                        var diff = s == 0 ? 0 : reader.receive(s)
                        if s > 0, diff < 1 << (s - 1) { diff += (-1 << s) + 1 }
                        pred[ci] += diff
                        coef[0] = pred[ci]
                        // AC
                        var k = 1
                        while k < 64 {
                            let rs = Int(try reader.decode(ac[c.ta]))
                            let r = rs >> 4, sz = rs & 15
                            if sz == 0 {
                                if r != 15 { break }
                                k += 16
                                continue
                            }
                            k += r
                            guard k < 64 else { throw D1OmniError.request("jpeg: AC index past 63") }
                            var v = reader.receive(sz)
                            if v < 1 << (sz - 1) { v += (-1 << sz) + 1 }
                            coef[zigzag[k]] = v
                            k += 1
                        }
                        let bx = mx * h + blk % h, by = my * v + blk / h
                        idctIslow(coef, quant[c.tq], into: &planes[ci], stride: stride, x: bx * 8, y: by * 8)
                    }
                }
            }
        }
        // crop to the image
        return planes.map { p in
            var out = [UInt8](repeating: 0, count: width * height)
            for y in 0..<height { out.replaceSubrange((y * width)..<((y + 1) * width), with: p[(y * stride)..<(y * stride + width)]) }
            return out
        }
    }

    /// jidctint.c jpeg_idct_islow on one block (natural-order coefficients), written at (x, y) of a plane.
    static func idctIslow(_ c: [Int], _ q: [Int], into out: inout [UInt8], stride: Int, x: Int, y: Int) {
        let constBits = 13, pass1Bits = 2
        func descale(_ v: Int, _ n: Int) -> Int { (v + (1 << (n - 1))) >> n }
        func limit(_ v: Int) -> UInt8 {
            let m = v & 1023
            let s = m < 512 ? m : m - 1024
            return UInt8(min(max(s + 128, 0), 255))
        }
        var ws = [Int](repeating: 0, count: 64)
        for col in 0..<8 {
            func dq(_ r: Int) -> Int { c[r * 8 + col] * q[r * 8 + col] }
            if c[8 + col] == 0 && c[16 + col] == 0 && c[24 + col] == 0 && c[32 + col] == 0 && c[40 + col] == 0
                && c[48 + col] == 0 && c[56 + col] == 0 {
                let dcval = dq(0) << pass1Bits
                for r in 0..<8 { ws[r * 8 + col] = dcval }
                continue
            }
            var z2 = dq(2), z3 = dq(6)
            var z1 = (z2 + z3) * 4433
            let tmp2e = z1 + z3 * -15137
            let tmp3e = z1 + z2 * 6270
            z2 = dq(0)
            z3 = dq(4)
            let tmp0e = (z2 + z3) << constBits
            let tmp1e = (z2 - z3) << constBits
            let tmp10 = tmp0e + tmp3e, tmp13 = tmp0e - tmp3e, tmp11 = tmp1e + tmp2e, tmp12 = tmp1e - tmp2e
            var tmp0 = dq(7), tmp1 = dq(5), tmp2 = dq(3), tmp3 = dq(1)
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * 9633
            tmp0 *= 2446
            tmp1 *= 16819
            tmp2 *= 25172
            tmp3 *= 12299
            z1 *= -7373
            z2 *= -20995
            z3 *= -16069
            z4 *= -3196
            z3 += z5
            z4 += z5
            tmp0 += z1 + z3
            tmp1 += z2 + z4
            tmp2 += z2 + z3
            tmp3 += z1 + z4
            ws[0 * 8 + col] = descale(tmp10 + tmp3, constBits - pass1Bits)
            ws[7 * 8 + col] = descale(tmp10 - tmp3, constBits - pass1Bits)
            ws[1 * 8 + col] = descale(tmp11 + tmp2, constBits - pass1Bits)
            ws[6 * 8 + col] = descale(tmp11 - tmp2, constBits - pass1Bits)
            ws[2 * 8 + col] = descale(tmp12 + tmp1, constBits - pass1Bits)
            ws[5 * 8 + col] = descale(tmp12 - tmp1, constBits - pass1Bits)
            ws[3 * 8 + col] = descale(tmp13 + tmp0, constBits - pass1Bits)
            ws[4 * 8 + col] = descale(tmp13 - tmp0, constBits - pass1Bits)
        }
        for row in 0..<8 {
            let w = row * 8, o = (y + row) * stride + x
            if ws[w + 1] == 0 && ws[w + 2] == 0 && ws[w + 3] == 0 && ws[w + 4] == 0 && ws[w + 5] == 0 && ws[w + 6] == 0
                && ws[w + 7] == 0 {
                let v = limit(descale(ws[w], pass1Bits + 3))
                for k in 0..<8 { out[o + k] = v }
                continue
            }
            var z2 = ws[w + 2], z3 = ws[w + 6]
            var z1 = (z2 + z3) * 4433
            let tmp2e = z1 + z3 * -15137
            let tmp3e = z1 + z2 * 6270
            let tmp0e = (ws[w] + ws[w + 4]) << constBits
            let tmp1e = (ws[w] - ws[w + 4]) << constBits
            let tmp10 = tmp0e + tmp3e, tmp13 = tmp0e - tmp3e, tmp11 = tmp1e + tmp2e, tmp12 = tmp1e - tmp2e
            var tmp0 = ws[w + 7], tmp1 = ws[w + 5], tmp2 = ws[w + 3], tmp3 = ws[w + 1]
            z1 = tmp0 + tmp3
            z2 = tmp1 + tmp2
            z3 = tmp0 + tmp2
            var z4 = tmp1 + tmp3
            let z5 = (z3 + z4) * 9633
            tmp0 *= 2446
            tmp1 *= 16819
            tmp2 *= 25172
            tmp3 *= 12299
            z1 *= -7373
            z2 *= -20995
            z3 *= -16069
            z4 *= -3196
            z3 += z5
            z4 += z5
            tmp0 += z1 + z3
            tmp1 += z2 + z4
            tmp2 += z2 + z3
            tmp3 += z1 + z4
            let s = constBits + pass1Bits + 3
            out[o + 0] = limit(descale(tmp10 + tmp3, s))
            out[o + 7] = limit(descale(tmp10 - tmp3, s))
            out[o + 1] = limit(descale(tmp11 + tmp2, s))
            out[o + 6] = limit(descale(tmp11 - tmp2, s))
            out[o + 2] = limit(descale(tmp12 + tmp1, s))
            out[o + 5] = limit(descale(tmp12 - tmp1, s))
            out[o + 3] = limit(descale(tmp13 + tmp0, s))
            out[o + 4] = limit(descale(tmp13 - tmp0, s))
        }
    }

    /// The colour conversion's tables (jdcolor.c build_ycc_rgb_table).
    static let tables: (crR: [Int], cbB: [Int], crG: [Int], cbG: [Int]) = {
        let scale = 16, half = 1 << 15
        func fix(_ x: Double) -> Int { Int(x * Double(1 << scale) + 0.5) }
        var crR = [Int](repeating: 0, count: 256), cbB = crR, crG = crR, cbG = crR
        for i in 0..<256 {
            let x = i - 128
            crR[i] = (fix(1.40200) * x + half) >> scale
            cbB[i] = (fix(1.77200) * x + half) >> scale
            crG[i] = -fix(0.71414) * x
            cbG[i] = -fix(0.34414) * x + half
        }
        return (crR, cbB, crG, cbG)
    }()

    /// ycc_rgb_convert (three components) or gray -> interleaved RGB.
    static func rgb(_ planes: [[UInt8]], width: Int, height: Int) -> RGBImage {
        let n = width * height
        var out = [UInt8](repeating: 0, count: 3 * n)
        func clamp(_ v: Int) -> UInt8 { UInt8(min(max(v, 0), 255)) }
        if planes.count == 1 {
            for i in 0..<n { out[3 * i] = planes[0][i]; out[3 * i + 1] = planes[0][i]; out[3 * i + 2] = planes[0][i] }
        } else {
            let t = tables
            for i in 0..<n {
                let yy = Int(planes[0][i]), cb = Int(planes[1][i]), cr = Int(planes[2][i])
                out[3 * i] = clamp(yy + t.crR[cr])
                out[3 * i + 1] = clamp(yy + ((t.cbG[cb] + t.crG[cr]) >> 16))
                out[3 * i + 2] = clamp(yy + t.cbB[cb])
            }
        }
        return RGBImage(width: width, height: height, pixels: out, decoder: "jpeg-baseline")
    }
}
