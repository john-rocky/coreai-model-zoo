import SwiftUI

struct AgentView: View {
    @State private var agent = AgentModel()
    @State private var prompt = ""
    @FocusState private var focused: Bool

    private let presets = [
        "What's on my calendar tomorrow?",
        "Remind me 15 minutes before the first one.",
        "How much battery and storage do I have left?",
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                header
                Divider()
                switch agent.phase {
                case .needsDownload, .downloading, .failed:
                    setup
                default:
                    transcript
                }
                Divider()
                composer
            }
            .navigationTitle("On-device agent")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Toggle("Thinking", isOn: Binding(
                            get: { agent.thinking },
                            set: { on in Task { await agent.setThinking(on) } }))
                        Button("New conversation") { agent.newConversation() }
                    } label: { Image(systemName: "ellipsis.circle") }
                }
            }
            .task {
                if agent.phase == .loading { await agent.load() }
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Label(agent.online ? "Online" : "Offline",
                  systemImage: agent.online ? "wifi" : "airplane")
                .font(.caption.weight(.semibold))
                .foregroundStyle(agent.online ? Color.secondary : Color.orange)
            Spacer()
            Text("MiniCPM5-2B · int8 · Core AI")
                .font(.caption)
                .foregroundStyle(.secondary)
            phaseBadge
        }
        .padding(.horizontal)
        .padding(.vertical, 6)
    }

    @ViewBuilder
    private var phaseBadge: some View {
        switch agent.phase {
        case .loading:
            ProgressView().controlSize(.mini)
        case .ready:
            Text(String(format: "loaded %.1f s", agent.loadSeconds))
                .font(.caption2).foregroundStyle(.secondary)
        case .generating:
            HStack(spacing: 4) {
                ProgressView().controlSize(.mini)
                Text("running").font(.caption2).foregroundStyle(.secondary)
            }
        default:
            EmptyView()
        }
    }

    private var setup: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "iphone.gen3").font(.system(size: 40)).foregroundStyle(.secondary)
            Text("MiniCPM5-2B (2.7 GB) is not on this phone yet.")
                .multilineTextAlignment(.center)
            switch agent.phase {
            case .downloading:
                ProgressView(value: agent.downloader.fraction)
                    .padding(.horizontal, 40)
                Text(agent.downloader.detail).font(.caption).foregroundStyle(.secondary)
            case .failed(let why):
                Text(why).font(.caption).foregroundStyle(.red).padding(.horizontal)
                Button("Retry") { Task { await agent.download() } }
            default:
                Button("Download from Hugging Face") { Task { await agent.download() } }
                    .buttonStyle(.borderedProminent)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(agent.entries) { entry in
                        EntryView(entry: entry).id(entry.id)
                    }
                    if let last = agent.stats.last, agent.phase == .ready {
                        StatsView(stats: last)
                    }
                }
                .padding()
            }
            .onChange(of: agent.entries.count) { _, _ in
                if let last = agent.entries.last { withAnimation { proxy.scrollTo(last.id, anchor: .bottom) } }
            }
        }
    }

    private var composer: some View {
        VStack(spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack {
                    ForEach(presets, id: \.self) { p in
                        Button(p) { submit(p) }
                            .buttonStyle(.bordered)
                            .font(.caption)
                            .disabled(agent.phase != .ready)
                    }
                }
                .padding(.horizontal)
            }
            HStack {
                TextField("Ask the phone…", text: $prompt)
                    .textFieldStyle(.roundedBorder)
                    .focused($focused)
                    .onSubmit { submit(prompt) }
                Button {
                    submit(prompt)
                } label: { Image(systemName: "arrow.up.circle.fill").font(.title2) }
                .disabled(agent.phase != .ready || prompt.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding([.horizontal, .bottom])
        }
        .padding(.top, 6)
    }

    private func submit(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, agent.phase == .ready else { return }
        prompt = ""
        focused = false
        Task { await agent.send(t) }
    }
}

struct EntryView: View {
    let entry: TurnEntry
    @State private var showReasoning = false

    var body: some View {
        switch entry.kind {
        case .prompt:
            HStack {
                Spacer(minLength: 40)
                Text(entry.text)
                    .padding(10)
                    .background(Color.accentColor.opacity(0.15), in: RoundedRectangle(cornerRadius: 12))
            }
        case .reasoning:
            DisclosureGroup(isExpanded: $showReasoning) {
                Text(entry.text).font(.caption).foregroundStyle(.secondary).padding(.top, 4)
            } label: {
                Label("thinking", systemImage: "brain").font(.caption).foregroundStyle(.secondary)
            }
        case .toolCall(let name, let arguments):
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "wrench.and.screwdriver").foregroundStyle(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).font(.system(.caption, design: .monospaced).weight(.semibold))
                    Text(arguments).font(.system(.caption2, design: .monospaced)).foregroundStyle(.secondary)
                }
            }
            .padding(8)
            .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        case .toolResult(let name):
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "arrow.turn.down.right").foregroundStyle(.green)
                VStack(alignment: .leading, spacing: 2) {
                    Text(name).font(.system(.caption, design: .monospaced))
                    Text(entry.text).font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(8)
            .background(Color.green.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
        case .response:
            Text(entry.text)
                .padding(10)
                .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
        case .error:
            Text(entry.text).font(.caption).foregroundStyle(.red)
        }
    }
}

struct StatsView: View {
    let stats: TurnStats
    var body: some View {
        Text(String(
            format: "%.1f s · %d prompt tokens (%d cached) · %d generated · %d tool call%@",
            stats.seconds, stats.promptTokens, stats.cachedTokens, stats.outputTokens,
            stats.toolCalls, stats.toolCalls == 1 ? "" : "s"))
            .font(.caption2)
            .foregroundStyle(.secondary)
    }
}
