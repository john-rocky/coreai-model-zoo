// swift-tools-version: 6.0
// GLiNERDecide — the Swift host of the GLiNER2.5-Decide Core AI port (zero-shot classification, one fixed-shape
// graph per S). DecideGraph loads the exported graph (../../export_gliner25_decide.py) and runs one call:
// input_ids / attention_mask [1, S] + label_idx [1, MMAX] -> logits [1, MMAX]. DecideGateSupport is a 1:1 port
// of the export script's host side (graph_inputs / probs_of / decide / compare_case / summarize) plus the
// fixture reader, shared by the Mac self-test (decide-selftest) and the iPhone gate app (apps/DecideGate), so a
// number from the phone means what the same number from the Mac means. Tokenization is not here: the fixtures
// carry the oracle's input_ids.
//
// No package dependencies; links the system CoreAI framework (Xcode 27).
//
//   export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
//   swift build -c release
//   D=~/code/coreai/_gliner25_decide
//   .build/release/decide-selftest --bundle $D/exports/gliner25-decide_float16_s256_m32.aimodel \
//       --fixtures $D/fixtures/readme21.json $D/fixtures/fast_decisions_s256.json \
//       --reference $D/exports/reference_s256.json --pygpu $D/results/gate_s256_gpu.json --json ../_work/selftest_s256.json
import PackageDescription

let package = Package(
    name: "GLiNERDecide",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "DecideGraph", targets: ["DecideGraph"]),
        .library(name: "DecideGateSupport", targets: ["DecideGateSupport"]),
        .executable(name: "decide-selftest", targets: ["decide-selftest"]),
    ],
    targets: [
        .target(
            name: "DecideGraph",
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .target(
            name: "DecideGateSupport",
            dependencies: ["DecideGraph"]
        ),
        .executableTarget(
            name: "decide-selftest",
            dependencies: ["DecideGraph", "DecideGateSupport"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
    ]
)
