// swift-tools-version: 6.1
// DeciderVision — the Swift read-out of decider-2b-vision on Core AI: image + questions -> the probability of
// every option, the author's contract end to end (Pillow-order bicubic, the author's prompt, the slot rule, the
// fixed-grid vision tower, the 2-function decoder in its chunk order, the letter softmax). Only the system
// CoreAI framework and swift-transformers' tokenizer.
import PackageDescription

let package = Package(
    name: "DeciderVision",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "DeciderVision", targets: ["DeciderVision"]),
        .executable(name: "decider-vision", targets: ["decider-vision"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
    ],
    targets: [
        .target(
            name: "DeciderVision",
            dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .executableTarget(
            name: "decider-vision",
            dependencies: ["DeciderVision"]
        ),
    ]
)
