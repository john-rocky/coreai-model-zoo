// CoreAIStudio — a macOS-native Core AI media enhancer: ×4 super-resolution (AdcSR) and
// frame interpolation (RIFE v4.26), both self-contained on the low-level Core AI runtime
// (apple/coreai-models), no coreai-kit. Two modes exercise two different compute units on
// purpose: Upscale runs on the GPU (a large diffusion-derived graph), Interpolate splits
// across ANE (flow-estimation) and GPU (warp/merge) per the measured routing in
// conversion/rife_compute_router.py — the whole chip, not just one part of it.

import SwiftUI

@main
struct CoreAIStudioApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 980, minHeight: 680)
        }
        .windowResizability(.contentMinSize)
    }
}
