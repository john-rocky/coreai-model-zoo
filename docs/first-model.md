# Put one model in your Swift app

This page was written for CoreAIKit 0.4.1. The current version and its first-run steps are in [CoreAIKit's README](https://github.com/john-rocky/coreai-kit#quickstart).

[**CoreAIKit 0.4.1: requirements, install and first answer →**](https://github.com/john-rocky/coreai-kit#readme)

Start with **`qwen3-0.6b`** in the release's built-in catalog. CoreAIKit downloads the matching
platform bundle once and caches it; your prompts and inference stay on the device.
The first download is about **352 MB on Mac** (the separate iPhone variant is about
**456 MB**). Follow the Kit README for the exact tested OS/SDK and current device
verification. A Mac run is not evidence for the iPhone variant.

```swift
import CoreAIKit

guard let modelID = ModelCatalog.builtin.entry(id: "qwen3-0.6b")?.modelID else {
    throw CoreAIKitError.modelNotAvailableOnPlatform(id: "qwen3-0.6b")
}
let chat = try await ChatSession(model: modelID)
let reply = try await chat.respond(to: "What is the capital of Japan?")
print(reply)
```

The same published release includes the
[ChatDemo app and CLI](https://github.com/john-rocky/coreai-kit/tree/0.4.1/Examples/ChatDemo):

```bash
git clone --branch 0.4.1 --depth 1 https://github.com/john-rocky/coreai-kit
cd coreai-kit/Examples/ChatDemo
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-Beta.5.app/Contents/Developer
swift run -c release chat-cli --model qwen3-0.6b --prompt "What is the capital of Japan?"
```

Adjust `DEVELOPER_DIR` to your installed beta 5 app if its name differs.
Expect a text answer mentioning Tokyo after the initial download and model load.
For a conversation, keep one `ChatSession` and call `respond(to:)` again. The GUI is
in `ChatDemo.xcodeproj`; use My Mac or a physical iPhone with the required OS and SDK.

After that first success, try text-to-speech with
[Speak and VoxCPM 0.5B](https://github.com/john-rocky/coreai-kit/tree/0.4.1/Examples/Speak)
from the same release (a separate download, about 1.37 GB on Mac). Each model card
below links its own runner, minimal Swift function and integration requirements.

For FoundationModels apps, `KitLanguageModel(model: modelID)` supplies a chat
model to `LanguageModelSession`. Tool calling depends on the model's dialect; guided
generation also needs a compatible sequential engine. Check the
[capability matrix](https://github.com/john-rocky/coreai-kit#works-with-apples-foundationmodels-api).
The catalog includes non-chat tasks, which use their own Kit APIs.

If a run fails, start with the Kit README's troubleshooting links and the
[observed error index](../knowledge/coreai-error-index.md). For details about using
Apple's adapter directly, see [FoundationModels integration notes](../knowledge/fm-provider.md).
