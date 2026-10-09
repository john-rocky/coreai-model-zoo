// Support — the error type and the file helpers the host shares.

import CryptoKit
import Foundation

public enum D1Error: Error, CustomStringConvertible, Sendable {
    /// the tokenizer, the asset or the bundle does not match the contract the host was written for
    case contract(String)
    /// the request is not one the host accepts (host.py raises ValueError with the same text)
    case request(String)
    /// a row does not fit the exported graph (ceil(T / S) * S > max_context_length - 1)
    case graphLimit(String)
    /// round 3a: the decoder graph is not called yet (the rows, the readout and the answers are)
    case graphNotWired
    case bundle(String)
    case json(String)

    /// The text alone (host.py's ValueError text for a request error).
    public var message: String {
        switch self {
        case .contract(let s), .request(let s), .graphLimit(let s), .bundle(let s), .json(let s): return s
        case .graphNotWired: return "the decoder graph is not wired in this build (round 3a: text, ids, readout only)"
        }
    }

    public var description: String {
        switch self {
        case .contract: return "contract: \(message)"
        case .request: return "request: \(message)"
        case .graphLimit: return "graph limit: \(message)"
        case .graphNotWired: return "graph: \(message)"
        case .bundle: return "bundle: \(message)"
        case .json: return "json: \(message)"
        }
    }
}

enum D1Files {
    static func sha256(_ url: URL) throws -> String {
        let d = try Data(contentsOf: url)
        return SHA256.hash(data: d).map { String(format: "%02x", $0) }.joined()
    }
}
