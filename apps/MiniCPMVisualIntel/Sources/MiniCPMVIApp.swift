// MiniCPMVIApp.swift — app entry + navigation router.
//
// The router bridges the system-facing intents back into the UI: an `OpenIntent` (tap a Visual
// Intelligence result) or the "Continue in app" schema intent runs `perform()`, which drives this
// `@Observable` router, which the view presents. The engine and the router are registered as App
// Intents dependencies in `init()` so `@Dependency` resolves them — crucially also during the
// background launch the system uses to run the Visual Intelligence query.

import AppIntents
import SwiftUI

@MainActor
@Observable
final class AppRouter {
    static let shared = AppRouter()

    /// A capture handed off from Visual Intelligence, to (re)answer in the foreground where the
    /// app has its full memory budget. Carried as JPEG `Data` (Sendable) — the view decodes it.
    var pendingImageData: Data?
    /// An answer already produced inside the VI background launch (flag A), to show directly.
    var pendingAnswer: String?

    /// A tapped VI result. An already-produced answer is shown; a tapped teaser re-answers in the
    /// foreground from the thumbnail it carried.
    func open(answer: String, alreadyAnswered: Bool, thumbnail: Data?) {
        if alreadyAnswered {
            pendingAnswer = answer
        } else if let thumbnail {
            pendingImageData = thumbnail
        }
    }

    /// "Continue in app" / "More results": run MiniCPM-V in the foreground on the captured frame.
    func continueInApp(imageData: Data?) {
        pendingAnswer = nil
        pendingImageData = imageData
    }
}

@main
struct MiniCPMVisualIntelApp: App {
    init() {
        // Make the engine and router resolvable via `@Dependency` in the intents. App Intents may
        // launch the app in the background to answer a Visual Intelligence query; `init()` still
        // runs first, so the registrations are in place before any `perform()`/`values(for:)`.
        // Resolve the singletons to locals first (App.init is main-actor) so the registration does
        // not capture a main-actor static accessor in a Sendable closure.
        let engine = MiniCPMVIEngine.shared
        let router = AppRouter.shared
        AppDependencyManager.shared.add(dependency: engine)
        AppDependencyManager.shared.add(dependency: router)
        // Headless on-device check (`-mvi-selftest <out>`); does nothing on a normal launch.
        Task { @MainActor in await MiniCPMVISelfTest.runIfRequested() }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(AppRouter.shared)
        }
    }
}
