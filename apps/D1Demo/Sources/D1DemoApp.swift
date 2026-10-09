// D1DemoApp — d1-3B's demo: a support inbox triaged in one tap (8 messages and 4 with a photo of the delivery, three
// questions each, a probability for every option), then an order log's audit questions, on the phone. The decision path
// is the Swift host apps/D1 (D1Decider, linked by path), unchanged. The model and the samples are sideloaded into the
// app's container (Library/Application Support/D1Assets: _stage.sh, _install.sh); nothing is bundled and nothing is
// fetched. The screens are shown alone, full height, no status bar and no navigation chrome; `-autoplay` drives them
// for a recording (Autoplay.swift); `-autoplay story` shows the video's pages instead (StoryView.swift).

import SwiftUI

@main
struct D1DemoApp: App {
    @State private var model: DemoModel
    @State private var autoplay: Autoplay

    init() {
        let model = DemoModel(config: .fromDefaults())
        let autoplay = Autoplay()
        _model = State(initialValue: model)
        _autoplay = State(initialValue: autoplay)
        #if os(macOS)
        // a Mac whose screen is locked opens no window, and the window's task never starts: an autoplayed run starts
        // here (it runs once; the window's task finds it started)
        if autoplay.mode != nil { Task { await autoplay.run(model) } }
        #endif
    }

    var body: some Scene {
        WindowGroup {
            RootView(model: model, autoplay: autoplay)
                .environment(\.colorScheme, .dark)
                #if os(iOS)
                .statusBarHidden(true)
                .persistentSystemOverlays(.hidden)
                #else
                // the video's pages on the Mac: the phone's screen in points, so a window capture is the phone's frame
                .frame(width: autoplay.mode == .story ? 402 : nil, height: autoplay.mode == .story ? 874 : nil)
                #endif
        }
        #if os(macOS)
        .defaultSize(width: 402, height: 874)
        .windowResizability(.contentSize)
        #endif
    }
}

struct RootView: View {
    let model: DemoModel
    let autoplay: Autoplay

    var body: some View {
        let manual = autoplay.mode == nil
        Group {
            switch model.screen {
            case .inbox: InboxView(model: model, manual: manual)
            case .log: LogView(model: model, manual: manual)
            case .end: EndCardView()
            case .story: StoryView(model: model)
            }
        }
        .task {
            if autoplay.mode != nil { await autoplay.run(model) } else { await model.load() }
        }
    }
}
