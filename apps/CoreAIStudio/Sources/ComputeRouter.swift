// ComputeRouter.swift — Swift port of conversion/rife_compute_router.py. Keep the
// (bundleKind, mode) -> unit mapping IDENTICAL to the Python source; that identity is what
// "router parity" means and is checked directly (see the app's own self-test entry point).
//
// Full rationale + the Tier-1/Tier-2 measured numbers backing these defaults live in
// conversion/rife_compute_router.py and zoo/rife-v4.26.md "Compute routing" — not duplicated
// here beyond the short form needed to justify each branch at the point of use.

import CoreAI

enum ComputeBundleKind: String {
    case monolith
    case split
}

enum ComputeMode: String {
    case interactive  // image-pair tween: one call, latency-visible to a human
    case video        // FPS-multiplication: many back-to-back calls, throughput-bound
}

struct RoutePlan {
    /// Function name (within the bundle) -> preferred compute unit. {"main": ...} for
    /// monolith, {"flow": ..., "warpmerge": ...} for split.
    let functions: [String: ComputeUnitKind]
    let rationale: String
}

enum ComputeRouter {
    // The one knob Tier-2 (real Core AI device numbers) was expected to turn — Tier-2 has now
    // run (M2 Pro, macOS 27.0, 384x640 fp32): GPU-preferred 24.72 ms vs ANE-preferred 24.94 ms,
    // statistically tied. ANE stays the default per the project's Neural-Engine-maximization
    // directive; flip only if a future power measurement shows it does NOT win there.
    static let defaultInteractiveUnit: ComputeUnitKind = .neuralEngine
    static let defaultVideoUnit: ComputeUnitKind = .gpu

    static func route(bundleKind: ComputeBundleKind, mode: ComputeMode) -> RoutePlan {
        switch bundleKind {
        case .split:
            // Matches the measured CPU_AND_NE ceiling directly (Tier 1): the flow-estimation
            // function is 100% ANE-eligible (conv/activation-only) at every resolution
            // profiled; warp+merge contains the never-ANE-eligible gather ops -> GPU (not CPU
            // — GPU wins gather-heavy work by a wide margin; CPU_AND_NE only isolated the
            // ANE-eligibility question).
            return RoutePlan(
                functions: ["flow": .neuralEngine, "warpmerge": .gpu],
                rationale: "split bundle: flow-estimation -> ANE (100% eligible), warp+merge -> GPU (never ANE-eligible)."
            )
        case .monolith:
            switch mode {
            case .video:
                // Tier-2 measured GPU/ANE statistically tied at this size, but GPU is kept as
                // the video default: it's the generalizable choice as resolution/batch scale
                // (Tier-1's GPU-preferred share only grew with resolution), and throughput-
                // bound many-frame workloads are where any such headroom matters most.
                return RoutePlan(
                    functions: ["main": defaultVideoUnit],
                    rationale: "monolith bundle, video/bulk mode: GPU (generalizable default; measured tied with ANE at this size)."
                )
            case .interactive:
                // Tier-2 MEASURED: GPU 24.72 ms vs ANE 24.94 ms warm-median — tied. ANE stays
                // default per the project's directive + its standard (here unmeasured) power
                // argument.
                return RoutePlan(
                    functions: ["main": defaultInteractiveUnit],
                    rationale: "monolith bundle, interactive mode: ANE (measured tied with GPU; kept for the Neural-Engine directive + unmeasured power argument)."
                )
            }
        }
    }

    /// Convenience: SpecializationOptions for one function's routed unit.
    static func specializationOptions(for unit: ComputeUnitKind) -> SpecializationOptions {
        SpecializationOptions(preferredComputeUnitKind: unit)
    }
}
