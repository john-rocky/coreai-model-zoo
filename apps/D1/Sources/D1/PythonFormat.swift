// PythonFormat — the texts CPython prints, which is what d1's prompt puts into the model's input and the response:
//
//   json.dumps(v, ensure_ascii=False, indent=2)   d1's state block (`prompt.state_block`): "," + newline + 2 spaces per
//                     level between items, ": " after keys, "[]" / "{}" for empty containers, members in their order
//                     (no sort), floats as repr, ints as their digits, true / false / null, NaN / Infinity
//   json.dumps(v)     with its defaults (", " and ": ", ensure_ascii: non-ASCII as \uXXXX, a surrogate pair above
//                     U+FFFF) — the answers' text
//   repr(float), str(float), round(x, n)
//   str.strip(), str.isalpha(), repr(str)      `prompt.option_codes` and the option-table refusal's text
//
// Copied from apps/Kev/Sources/Kev/PythonFormat.swift (zoo d1-3b b1a082a; `floatRepr`, `isFloatLiteral`,
// `numberValue` and `pyRound` are apps/ClefFlash's, zoo main 5ef2247, gated against CPython on 200,000 doubles);
// `dumps(_:indent:asciiOnly:)`, `strip`, `isAlpha`, `isPrintable` and `reprString` are new. The Unicode properties
// (str.isalpha, str.isprintable) come from Swift's `Unicode.Scalar.Properties.generalCategory`: the same definition
// as CPython's (categories Lu Ll Lt Lm Lo; Cc Cf Cs Co Cn Zl Zp Zs), so the two differ only on code points whose
// category changed between their Unicode versions (CPython 3.12 ships Unicode 15.0).

import Foundation

public enum PythonFormat {
    /// A literal json.loads turns into a float (it has a fraction or an exponent, or is NaN / ±Infinity).
    public static func isFloatLiteral(_ s: String) -> Bool {
        s.contains(where: { $0 == "." || $0 == "e" || $0 == "E" }) || s == "NaN" || s.hasSuffix("Infinity")
    }

    /// The value json.loads makes of a number literal, as a Double (ints converted).
    public static func numberValue(_ s: String) -> Double {
        switch s {
        case "NaN": return .nan
        case "Infinity": return .infinity
        case "-Infinity": return -.infinity
        default: return Double(s) ?? .nan
        }
    }

    /// The decimal digits of the int json.loads makes of an int literal ("-0" is 0; the digits are kept in full).
    static func intDigits(_ s: String) -> String {
        let digits = s.hasPrefix("-") ? String(s.dropFirst()) : s
        if digits.allSatisfy({ $0 == "0" }) { return "0" }
        return s
    }

    /// What `json.dumps` prints for the value `json.loads` made of the literal `s`.
    public static func numberJSONText(_ s: String) -> String {
        isFloatLiteral(s) ? floatRepr(numberValue(s)) : intDigits(s)
    }

    /// Python's `str()` of the value `json.loads` made of the literal `s` (the author's `render` of a number).
    public static func numberStr(_ s: String) -> String {
        isFloatLiteral(s) ? floatStr(numberValue(s)) : intDigits(s)
    }

    /// Python's `str(float)`: repr for a finite value, "nan" / "inf" / "-inf" otherwise.
    public static func floatStr(_ x: Double) -> String {
        if x.isNaN { return "nan" }
        if x.isInfinite { return x < 0 ? "-inf" : "inf" }
        return floatRepr(x)
    }

    /// Python's `repr(float)` (and json.dumps's text for a float; non-finite values as json.dumps writes them).
    public static func floatRepr(_ x: Double) -> String {
        if x.isNaN { return "NaN" }
        if x.isInfinite { return x < 0 ? "-Infinity" : "Infinity" }
        if x == 0 { return x.sign == .minus ? "-0.0" : "0.0" }
        // Swift's description is the shortest digit string that round-trips (the closest one when several do), the
        // same digits as Python's dtoa mode 0; only the layout differs, so take the digits and the exponent.
        var s = x.magnitude.description
        var exp10 = 0
        if let e = s.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            exp10 = Int(s[s.index(after: e)...].replacingOccurrences(of: "+", with: ""))!
            s = String(s[..<e])
        }
        let parts = s.split(separator: ".", omittingEmptySubsequences: false)
        let intPart = String(parts[0])
        let frac = parts.count > 1 ? String(parts[1]) : ""
        var digits = Array(intPart + frac)
        var decpt = intPart.count + exp10            // value = 0.d1d2... x 10^decpt
        while let f = digits.first, f == "0" {
            digits.removeFirst()
            decpt -= 1
        }
        while let l = digits.last, l == "0" { digits.removeLast() }
        let sign = x < 0 ? "-" : ""
        if decpt <= -4 || decpt > 16 {
            let exp = decpt - 1
            var m = String(digits[0])
            if digits.count > 1 { m += "." + String(digits[1...]) }
            let e = abs(exp) < 10 ? "0\(abs(exp))" : "\(abs(exp))"
            return sign + m + "e" + (exp < 0 ? "-" : "+") + e
        }
        if decpt <= 0 {
            return sign + "0." + String(repeating: "0", count: -decpt) + String(digits)
        }
        if decpt < digits.count {
            return sign + String(digits[..<decpt]) + "." + String(digits[decpt...])
        }
        return sign + String(digits) + String(repeating: "0", count: decpt - digits.count) + ".0"
    }

    /// Python's `round(x, ndigits)` for a float: the correctly rounded decimal with `ndigits` places (ties to even on
    /// the exact binary value, as dtoa mode 3 and the C library's `%.*f` both do), read back as the nearest double.
    public static func pyRound(_ x: Double, _ ndigits: Int = 4) -> Double {
        guard x.isFinite else { return x }
        return Double(String(format: "%.\(ndigits)f", x))!
    }

    private static let hex: [Character] = Array("0123456789abcdef")

    private static func appendHex4(_ v: UInt32, to out: inout String.UnicodeScalarView) {
        out.append(contentsOf: "\\u".unicodeScalars)
        for shift in stride(from: 12, through: 0, by: -4) {
            out.append(contentsOf: String(hex[Int((v >> UInt32(shift)) & 0xF)]).unicodeScalars)
        }
    }

    /// A JSON string literal as json.dumps writes it: `"` and `\` escaped, \b \f \n \r \t as such, other controls as
    /// \u00xx; with `asciiOnly` (ensure_ascii, json.dumps's default) everything outside ' '...'~' as \uxxxx (a UTF-16
    /// surrogate pair above U+FFFF), lowercase hex.
    public static func quote(_ s: String, asciiOnly: Bool) -> String {
        var out = String.UnicodeScalarView()
        out.append("\"")
        for c in s.unicodeScalars {
            switch c {
            case "\"": out.append(contentsOf: "\\\"".unicodeScalars)
            case "\\": out.append(contentsOf: "\\\\".unicodeScalars)
            case "\u{08}": out.append(contentsOf: "\\b".unicodeScalars)
            case "\u{0C}": out.append(contentsOf: "\\f".unicodeScalars)
            case "\n": out.append(contentsOf: "\\n".unicodeScalars)
            case "\r": out.append(contentsOf: "\\r".unicodeScalars)
            case "\t": out.append(contentsOf: "\\t".unicodeScalars)
            default:
                if c.value < 0x20 {
                    appendHex4(c.value, to: &out)
                } else if asciiOnly && c.value > 0x7E {
                    if c.value >= 0x10000 {
                        let v = c.value - 0x10000
                        appendHex4(0xD800 | (v >> 10), to: &out)
                        appendHex4(0xDC00 | (v & 0x3FF), to: &out)
                    } else {
                        appendHex4(c.value, to: &out)
                    }
                } else {
                    out.append(c)
                }
            }
        }
        out.append("\"")
        return String(out)
    }

    /// `json.dumps(v)`: members in their order; `asciiOnly` = ensure_ascii (the default, true); separators (", ",
    /// ": ") as json.dumps uses without an indent.
    public static func dumps(_ v: JSONValue, asciiOnly: Bool = true, separators: (item: String, key: String) = (", ", ": "))
        -> String
    {
        var out = ""
        write(v, asciiOnly: asciiOnly, separators: separators, into: &out)
        return out
    }

    private static func write(_ v: JSONValue, asciiOnly: Bool, separators: (item: String, key: String), into out: inout String) {
        switch v {
        case .null: out += "null"
        case .bool(let b): out += b ? "true" : "false"
        case .number(let s): out += numberJSONText(s)
        case .string(let s): out += quote(s, asciiOnly: asciiOnly)
        case .array(let a):
            out += "["
            for (k, x) in a.enumerated() {
                if k > 0 { out += separators.item }
                write(x, asciiOnly: asciiOnly, separators: separators, into: &out)
            }
            out += "]"
        case .object(let m):
            out += "{"
            for (k, x) in m.enumerated() {
                if k > 0 { out += separators.item }
                out += quote(x.key, asciiOnly: asciiOnly)
                out += separators.key
                write(x.value, asciiOnly: asciiOnly, separators: separators, into: &out)
            }
            out += "}"
        }
    }

    /// The characters Python's `str.isspace()` names (`_PyUnicode_IsWhitespace`: bidirectional WS / B / S or
    /// category Zs), which `str.strip` / `lstrip` remove.
    public static func isPythonSpace(_ c: Unicode.Scalar) -> Bool {
        switch c.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
            return true
        default:
            return false
        }
    }

    /// Python's `str.lstrip()`.
    public static func lstrip(_ s: String) -> String {
        let u = s.unicodeScalars
        guard let first = u.firstIndex(where: { !isPythonSpace($0) }) else { return "" }
        return String(u[first...])
    }

    /// `json.dumps(v, ensure_ascii=asciiOnly, indent=indent)`: CPython's `_make_iterencode` — an item separator ","
    /// then a newline and `indent` spaces per level, ": " after a key, "[]" / "{}" for an empty container, members in
    /// their order (no sort_keys).
    public static func dumps(_ v: JSONValue, indent: Int, asciiOnly: Bool) -> String {
        var out = ""
        writeIndented(v, indent: indent, level: 0, asciiOnly: asciiOnly, into: &out)
        return out
    }

    private static func writeIndented(_ v: JSONValue, indent: Int, level: Int, asciiOnly: Bool, into out: inout String) {
        switch v {
        case .array(let a) where !a.isEmpty:
            let pad = "\n" + String(repeating: " ", count: indent * (level + 1))
            out += "["
            for (k, x) in a.enumerated() {
                out += k == 0 ? pad : "," + pad
                writeIndented(x, indent: indent, level: level + 1, asciiOnly: asciiOnly, into: &out)
            }
            out += "\n" + String(repeating: " ", count: indent * level) + "]"
        case .object(let m) where !m.isEmpty:
            let pad = "\n" + String(repeating: " ", count: indent * (level + 1))
            out += "{"
            for (k, x) in m.enumerated() {
                out += k == 0 ? pad : "," + pad
                out += quote(x.key, asciiOnly: asciiOnly) + ": "
                writeIndented(x.value, indent: indent, level: level + 1, asciiOnly: asciiOnly, into: &out)
            }
            out += "\n" + String(repeating: " ", count: indent * level) + "}"
        default:
            out += dumps(v, asciiOnly: asciiOnly)
        }
    }

    /// Python's `str.strip()` (no argument): the `isPythonSpace` characters off both ends.
    public static func strip(_ s: String) -> String {
        let u = s.unicodeScalars
        guard let first = u.firstIndex(where: { !isPythonSpace($0) }),
              let last = u.lastIndex(where: { !isPythonSpace($0) }) else { return "" }
        return String(u[first...last])
    }

    /// Python's `str.isalpha()`: non-empty and every code point a letter (general category Lu, Ll, Lt, Lm or Lo).
    public static func isAlpha(_ s: String) -> Bool {
        guard !s.unicodeScalars.isEmpty else { return false }
        return s.unicodeScalars.allSatisfy { c in
            switch c.properties.generalCategory {
            case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
            default: return false
            }
        }
    }

    /// CPython's `Py_UNICODE_ISPRINTABLE`: every code point but the categories Cc, Cf, Cs, Co, Cn, Zl, Zp and Zs, with
    /// the space (U+0020) printable.
    public static func isPrintable(_ c: Unicode.Scalar) -> Bool {
        if c == " " { return true }
        switch c.properties.generalCategory {
        case .control, .format, .surrogate, .privateUse, .unassigned, .lineSeparator, .paragraphSeparator, .spaceSeparator:
            return false
        default:
            return true
        }
    }

    /// Python's `repr(str)` (`unicode_repr`): single quotes unless the text holds a single quote and no double quote;
    /// the quote and the backslash escaped; \t \n \r; other ASCII controls and DEL as \xhh; a non-printable code point
    /// as \xhh / \uhhhh / \Uhhhhhhhh (lowercase hex); everything else as is.
    public static func reprString(_ s: String) -> String {
        let u = s.unicodeScalars
        let quote: Unicode.Scalar = u.contains("'") && !u.contains("\"") ? "\"" : "'"
        var out = String.UnicodeScalarView()
        out.append(quote)
        func hexEscape(_ v: UInt32, prefix: String, digits: Int) {
            let h = String(v, radix: 16)
            out.append(contentsOf: (prefix + String(repeating: "0", count: max(0, digits - h.count)) + h).unicodeScalars)
        }
        for c in u {
            if c == quote || c == "\\" {
                out.append("\\")
                out.append(c)
            } else if c == "\t" {
                out.append(contentsOf: "\\t".unicodeScalars)
            } else if c == "\n" {
                out.append(contentsOf: "\\n".unicodeScalars)
            } else if c == "\r" {
                out.append(contentsOf: "\\r".unicodeScalars)
            } else if c.value < 0x20 || c.value == 0x7F {
                hexEscape(c.value, prefix: "\\x", digits: 2)
            } else if c.value < 0x7F {
                out.append(c)
            } else if isPrintable(c) {
                out.append(c)
            } else if c.value <= 0xFF {
                hexEscape(c.value, prefix: "\\x", digits: 2)
            } else if c.value <= 0xFFFF {
                hexEscape(c.value, prefix: "\\u", digits: 4)
            } else {
                hexEscape(c.value, prefix: "\\U", digits: 8)
            }
        }
        out.append(quote)
        return String(out)
    }

    /// Python's `str(x)` of a value d1 formats into its prompt with an f-string: a string as is, `None` for null.
    public static func strOrNone(_ v: JSONValue?) -> String {
        guard let v, !v.isNull else { return "None" }
        if case .string(let s) = v { return s }
        return dumps(v)
    }
}
