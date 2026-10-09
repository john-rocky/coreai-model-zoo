// swift-tools-version: 6.1
// D1Omni — the Swift host of LiquidAI/d1-omni-600M on Core AI: a SystemOne request (a state and typed questions) ->
// one token row per question -> the decision graph's six inputs at the row's bucket -> the scores at the markers ->
// the publisher's probabilities and answers. conversion/d1_omni/host.py is the specification every file copies (the
// publisher's prompt.py, the per-mode dispatch of modeling_d1.py, graph_inputs, bucket_for,
// probabilities_from_logits; round 9: the image and audio paths of host.py §5–6). Only the system CoreAI framework,
// Accelerate, ImageIO / CoreGraphics (the image decode) and swift-transformers' tokenizer, pinned to the version the
// gate ran (a different tokenizer version could cut a row differently). The CLI's directory is
// `Sources/d1omni-cli`: on a case-insensitive volume `Sources/d1omni` would be the library's `Sources/D1Omni`.
import PackageDescription

let package = Package(
    name: "D1Omni",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "D1Omni", targets: ["D1Omni"]),
        .executable(name: "d1omni", targets: ["D1OmniCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(
            name: "D1Omni",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            linkerSettings: [.linkedFramework("CoreAI"), .linkedFramework("Accelerate")]
        ),
        .executableTarget(
            name: "D1OmniCLI",
            dependencies: ["D1Omni"],
            path: "Sources/d1omni-cli"
        ),
    ]
)
