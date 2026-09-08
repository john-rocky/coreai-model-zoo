import SwiftUI

@main
struct CoreAIAgentApp: App {
    init() {
        // Prefill as S=1 steps. The engine's default prefill runs the whole prompt as one
        // dynamic-length graph call, and on iPhone every new length re-specializes (measured
        // 95–160 s per tool turn); S=1 steps reuse one shape at the decode rate (~27 tok/s).
        if ProcessInfo.processInfo.environment["COREAI_CHUNK_THRESHOLD"] == nil {
            setenv("COREAI_CHUNK_THRESHOLD", "1", 1)
        }
    }

    var body: some Scene {
        WindowGroup {
            AgentView()
        }
    }
}
