# CoreAIAgent — an on-device agent on iPhone (MiniCPM5-2B, Core AI, FoundationModels tools)

A small SwiftUI app that puts [MiniCPM5-2B](../../models/minicpm5-2b/README.md) (int8 block-32,
2.7 GB, Apple Core AI) behind FoundationModels' `LanguageModelSession` with three Swift `Tool`s
that touch the phone:

| tool | what it does | permission |
|---|---|---|
| `get_calendar_events(day)` | lists the day's events from EventKit | Calendar (full access) |
| `create_reminder(title, day, time)` | creates an EventKit reminder with an alarm | Reminders (full access) |
| `get_device_status()` | battery level / charging state / free storage | none |

The model, the tool loop and the answer all run on the phone — the header shows the network
state, so the recording can be made in airplane mode. The provider is the zoo's
[`ZooFMProvider`](../../swift/README.md) with its `MiniCPMDialect`: MiniCPM5 emits calls as
`<function name=…><param name=…>…</param></function>` XML, which the dialect parses into the
typed arguments the framework hands to the Swift `Tool`.

First launch downloads `int8/` from `mlboydaisuke/MiniCPM5-2B-CoreAI` into
`Documents/models/minicpm5_2b_int8` (the shared `ModelDownloader`); the app needs
`com.apple.developer.kernel.increased-memory-limit` for the 2.7 GB cold specialization.
Generation is capped at 220 tokens per turn and the presets are three turns long on purpose: the
shipped pipelined engine caps the iOS growing-KV at 1024 tokens (see the model card), and the
engine keeps decoding to the cap after EOS.

## Verify on the Mac (no phone, no TCC dialog)

The same sources build as `CoreAIAgentMac`; `AGENT_MOCK_TOOLS=1` makes the three tools answer with
fixed data (the gate's calendar), so the whole session — dialect, tool round trips, caps — runs
under the app's own settings:

```bash
xcodebuild -project CoreAIAgent.xcodeproj -scheme CoreAIAgentMac -configuration Release \
  -destination generic/platform=macOS -derivedDataPath build_mac CODE_SIGN_IDENTITY="-" CODE_SIGNING_REQUIRED=NO build
cp -cR "<the published int8 bundle>" ~/Documents/models/minicpm5_2b_int8      # the Mac build does not download
AGENT_SELFTEST=1 AGENT_MOCK_TOOLS=1 build_mac/Build/Products/Release/CoreAIAgentMac.app/Contents/MacOS/CoreAIAgentMac
```

2026-09-08, final settings (S=1 prefill, trace off, cap 120), two runs: calendar read (80-token
answer) → `create_reminder` at 09:45 for the 10:00 first event → device status, 7.7 / 9.9 / 11.3 s
per turn on an M4 Max. With the trace ON and a 220 cap the second turn failed — the trace spent the
budget and the call was cut mid-XML (`ToolCallError: GeneratedContent does not contain …`); the
trace needs ~400 tokens, which three turns cannot afford under the iOS 1024-token KV cap, so the
Thinking toggle is macOS-only and the phone runs without it.

## Status

- **Mac (gate):** `swift run -c release zoo-fm-gate <bundle> agent` runs this exact flow with fixed
  calendar data — PASS with the model's thinking on (cap 220) and off (cap 120): calendar read →
  reminder at 09:45 for a 10:00 first meeting → device status.
- **Build:** compiles for iOS (unsigned `xcodebuild … CODE_SIGNING_ALLOWED=NO`, 2026-09-08).
- **iPhone 17 Pro (2026-09-08, `AGENT_SELFTEST=1`):** the three presets ran end to end on the
  phone against the real calendar — read tomorrow's event, created the reminder 15 minutes before
  it, reported battery and storage; download 2.68 GB in ~1.5 min, cold load 22 s, warm 4 s. Turn
  times were 95–160 s with the engine's default prefill (every new prompt length re-specializes
  the dynamic graph on iPhone; `COREAI_CHUNK_THRESHOLD=128` did not help), so the app sets
  `COREAI_CHUNK_THRESHOLD=1` itself (S=1 prefill at the decode rate) and defaults to thinking off
  with a 100-token cap. One caveat seen on device: the "15 minutes before" arithmetic came out
  17:45 in one run and 17:55 (correct) in another for an 18:10 event — a 2B model doing time math.
- **Signing:** the target borrows `com.daisukemajima.llmbench014`, an app id whose provisioning
  profile already carries both kernel entitlements (no app with that id on the phone), because
  automatic provisioning of a new id needs an Apple ID signed in to Xcode. Switch the id back
  once `com.daisukemajima.CoreAIAgent` is registered with the same capabilities.

```bash
cd apps/CoreAIAgent && xcodegen generate
xcodebuild -project CoreAIAgent.xcodeproj -scheme CoreAIAgent -configuration Release \
  -destination generic/platform=iOS -derivedDataPath build -allowProvisioningUpdates build
```
