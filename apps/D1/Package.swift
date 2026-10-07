// swift-tools-version: 6.1
// D1 — the Swift host of LiquidAI d1-3B on Core AI: a System One request -> one decoder row per question -> the option
// readout -> the System One response, the provider's contract end to end (conversion/d1/host.py is the specification
// it copies: the request checks, `prompt.render`'s text, the option codes / aliases / readout groups / keys, the rows
// and the Tree split, the option table, the float64 readout, `api.answer`, the usage; conversion/d1/vision_host.py
// for a picture's crop plan and token run). Only the system frameworks (CoreAI, Accelerate) and swift-transformers'
// tokenizer. The CLI's directory is `Sources/d1-cli`: on a case-insensitive volume `Sources/d1` would be the
// library's `Sources/D1`. `D1BLAS` is a C target so that NumPy's Accelerate calls can be made with the new-interface
// defines (no unsafe flags).
import PackageDescription

let package = Package(
    name: "D1",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "D1", targets: ["D1"]),
        .executable(name: "d1", targets: ["D1CLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "D1BLAS",
            path: "Sources/D1BLAS",
            cSettings: [.define("ACCELERATE_NEW_LAPACK"), .define("ACCELERATE_LAPACK_ILP64")],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .target(
            name: "D1",
            dependencies: [
                "D1BLAS", .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "D1CLI",
            dependencies: ["D1"],
            path: "Sources/d1-cli"
        ),
        // the pixel path's test CLI (ImagePixels.swift -> raw files for conversion/d1/gate_swift_pixels.py)
        .executableTarget(
            name: "d1-pixels-test",
            dependencies: ["D1"],
            path: "Sources/d1-pixels-test"
        ),
    ]
)
