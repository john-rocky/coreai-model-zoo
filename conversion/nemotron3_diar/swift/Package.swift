// swift-tools-version: 6.0
// NemotronDiarizer — the Swift host of the Nemotron-3-Diarization Core AI port (8-speaker streaming
// Sortformer). A 1:1 port of ../host_loop.py and ../mel_frontend.py: log-mel -> 8-frame stacking +
// projection -> left-packed [cache | FIFO | chunk] graph input -> sigmoid / 8x average pool -> speaker
// cache (AOSC + FIFO), around the exported fixed-T graph (T = 541 streaming, 684 offline). The Python
// files are the reference; where a choice was needed, this follows them.
//
// No package dependencies; links the system CoreAI framework (Xcode 27: the stable Xcode SDK has none).
// N3DGateSupport holds the gate metrics the Mac self-test and the iPhone gate app (apps/N3DGate) share.
//
//   export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
//   swift build -c release
//   .build/release/n3d-selftest --assets ../_work/artifacts \
//       --bundle ../_work/artifacts/n3d_streaming_float16.aimodel \
//       --wav ~/code/coreai/_n3d/fixtures/diarization_example_16k.wav --mode ll \
//       --golden ../_work/golden/diarization_example_ll_probs.f32le
import PackageDescription

let package = Package(
    name: "NemotronDiarizer",
    platforms: [.macOS("27.0"), .iOS("27.0")],
    products: [
        .library(name: "NemotronDiarizer", targets: ["NemotronDiarizer"]),
        .library(name: "N3DGateSupport", targets: ["N3DGateSupport"]),
        .executable(name: "n3d-selftest", targets: ["n3d-selftest"]),
    ],
    targets: [
        .target(
            name: "NemotronDiarizer",
            linkerSettings: [.linkedFramework("CoreAI"), .linkedFramework("Accelerate")]
        ),
        .target(
            name: "N3DGateSupport",
            dependencies: ["NemotronDiarizer"]
        ),
        .executableTarget(
            name: "n3d-selftest",
            dependencies: ["NemotronDiarizer", "N3DGateSupport"],
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
    ]
)
