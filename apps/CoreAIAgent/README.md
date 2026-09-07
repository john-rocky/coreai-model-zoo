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

## Status

- **Mac (gate):** `swift run -c release zoo-fm-gate <bundle> agent` runs this exact flow with fixed
  calendar data — PASS with the model's thinking on (cap 220) and off (cap 120): calendar read →
  reminder at 09:45 for a 10:00 first meeting → device status.
- **Build:** compiles for iOS (unsigned `xcodebuild … CODE_SIGNING_ALLOWED=NO`, 2026-09-08).
- **iPhone:** not yet run on a device (the phone was disconnected when the app was written). The
  signed build needs the app id registered with the two kernel entitlements, which automatic
  provisioning can only do with an Apple ID signed in to Xcode — open the project in Xcode once
  with the team signed in and Run, then `xcodebuild … -allowProvisioningUpdates` works from the CLI.

```bash
cd apps/CoreAIAgent && xcodegen generate
xcodebuild -project CoreAIAgent.xcodeproj -scheme CoreAIAgent -configuration Release \
  -destination generic/platform=iOS -derivedDataPath build -allowProvisioningUpdates build
```
