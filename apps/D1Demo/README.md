# D1Demo: a support inbox triaged on the phone with d1-3B

An iPhone app that shows [d1-3B](../../models/d1-3b/) doing the job its provider recommends it for, routing and triage:
one tap answers three questions about each of twelve support messages, eight short texts and four with a picture of
the delivery, and every answer is one of the question's options with its probability. A second screen asks three
audit questions about a day's order log (55 entries, about 3,400 tokens) in one request. Every decision runs on the
phone through the Swift host [`apps/D1`](../D1/) (`D1Decider.decide(requestJSON:images:shared:)`, linked by path and
unchanged): the iPhone's int8mlp decoder and the vision tower, each `.aimodel` specialized on the device.

The screens show what the model returned and the seconds the app's own `ContinuousClock` measured around each
`decide` call (the picture's decoding and the tower included, the load not). A pill reads "in-app samples · offline"
when the phone has no Wi-Fi and no cellular path (`NWPathMonitor`); the samples ship with the assets, nothing is
fetched.

## What it reads

The model and the samples are sideloaded into the app's container, `Library/Application Support/D1Assets/`:

| path | what |
|---|---|
| `decoder/` | `d1_3b_decode_int8mlp_pf64_s` from [mlboydaisuke/d1-3B-CoreAI](https://huggingface.co/mlboydaisuke/d1-3B-CoreAI) (`gpu-pipelined/`) |
| `tower/` | `d1_3b_vision_fp16w32_s` from the same repository |
| `demo/` | `samples.json`, `requests/*.json` (one System One request per item, sent as written), `pictures/*.png` |

`make_samples.py` writes `demo/`: the twelve messages (written for the demo; they name no person, company or product),
the four pictures (drawn with PIL, CC0-1.0, the same bytes on every run), the order log (the d1-3B port's fixture
`long_34k`, the record's first three questions) and three warm-up requests the app runs once after loading and never
shows.

```bash
python make_samples.py --fixtures <the port's fixtures directory> --out <dir>
```

## Build, stage, install

```bash
export DEVELOPER_DIR=/Applications/Xcode-27.0.0-RC.app/Contents/Developer
D1_PROVISIONING_UPDATES=1 ./_build.sh   # iOS, Release, with D1Demo.entitlements (increased-memory-limit)
./_build.sh --mac                       # macOS, for checking a run against the Python reference
./_stage.sh <download>/gpu-pipelined/d1_3b_decode_int8mlp_pf64_s <download>/gpu-pipelined/d1_3b_vision_fp16w32_s <dir>
D1_ALLOWED_DEVICES=<udid>,<coredevice id> D1_HOLD_SCRIPT=<hold name> ./_install.sh <udid>
```

The iPhone build needs `com.apple.developer.kernel.increased-memory-limit`: without it the decoder's on-device
specialization does not fit (apps/D1Gate). A new bundle id gets the capability with `-allowProvisioningUpdates`
(`D1_PROVISIONING_UPDATES=1`): Xcode's account registers the App ID with it and fetches its profile. `_stage.sh` checks
the bundles against the repository's `SHA256SUMS`; `_install.sh` pushes `D1Assets/` and pulls it back to compare every
file's md5 (a `devicectl device copy to` can exit 0 with a file cut short), never passes `--remove-existing-content`,
and runs only on a listed device while the shared device hold is this run's.

## Run it hands-off

```bash
./_run.sh launch <udid> <run id> -autoplay all -trigger t-<epoch>.trigger -delay 1.5
./_run.sh front <udid>                       # a terminating launch can leave the home screen showing
./_run.sh trigger <udid> t-<epoch>.trigger   # the app presses Triage inbox 1.5 s later
./_run.sh wait <udid> <run id> "DONE run-" 120 <dir>
./_run.sh pull <udid> <run id> <dir>
```

`-autoplay all` loads, waits at READY for the trigger file, presses Triage inbox, opens each picture message in turn,
shows the summary, moves to the order log, scrolls it, presses Ask and ends on a card with the model's name and links;
`-autoplay inbox` / `log` run one screen. With `-log 1` (added by `_run.sh launch`) every step is a line of
`Documents/d1demo/autoplay.log` and `Documents/d1demo/run-<id>.json` keeps what the screen showed: each answer's words
and probability, the response bodies as `json.dumps(indent=2)` writes them, every second and the line it went into.
After the run the app writes its thermal state every 10 s for a while, so a script can hand a shared phone back cool.
The holds between steps are arguments (`-detail`, `-hold`, `-scroll`, `-logHold`, `-end`).

On the Mac, `-assets <dir>` (with `demo/` inside), `-decoder` / `-tower` (bundle directories) and `-asset aot` run the
same screens on a bundle's AOT asset; pass `-NSAppSleepDisabled YES`, or App Nap slows the decisions of a window in
the background several times over. A Mac whose screen is locked opens no window: an autoplayed run starts from the app
itself and does not need one.

### The video's layout: `-autoplay story`

`-autoplay story` shows the same run one message to a screen, for a recording read at a phone feed's width: a blank
screen at READY; after the trigger, a title card while the run starts behind it (Triage inbox's run, all twelve
messages back to back, then Ask on the log); then a few messages one at a time (`-items`, default
`m01,m09,m04,m12,m05`), each alone for `-msg` s, its three answers `-step` s apart, then the seconds the app timed for
that message, held `-itemHold` s; the whole inbox's count and seconds (`-scaleHold` s); the log's first question, its
answer and the log's seconds (`-logMsg`, `-logHold`); the end card. The message text is 44 pt, an answer 48 pt, the
seconds 46 pt, and every page sits in a 9:16 band at the screen's center (402 x 715 of 402 x 874 points), so the
recording can be cut to 9:16 without losing anything. The pages only set the pace of showing the run: every second on
them is the run's (a message's seconds are its own `decide`, measured in that run), and the run JSON keeps every page
with the time it came up and its words (`story`). On the Mac, `-render <dir>` also draws each page at the phone's size
(1206 x 2622) once it is complete.

## Checked

The Mac build on the fp16 decoder's and the tower's AOT h16c assets, autoplay `all`: all 39 questions (12 messages and
the log) give the same p bits as `conversion/d1/decide.py` on the same assets, the 13 response bodies are byte-equal,
the words on screen equal the responses, and every second on screen equals the run JSON's.

The iPhone 18 Pro (iOS 27.2 24B5099f), the released int8mlp decoder and tower specialized on the phone, autoplay `all`,
2026-10-09, four runs (two of them recorded): every answer's option equals the Mac's read-out of the same int8mlp bundle
on all 39 questions (max |dp| 0.0009; the two GPUs give different bits). The screen showed the twelve messages decided
in 4.72–4.78 s (a text 0.19–0.29 s, the run's first one slowest; a message with a picture 0.73–0.80 s) and the log's
three questions in 3.02–3.05 s, with or without the screen being recorded. The first launch after the install
specialized the two `.aimodel` in 49.2 s; later launches loaded in 4.9–5.2 s.

`-autoplay story` (2026-10-09, the same phone and bundles): the Mac build on the fp16 AOT assets gives the same p bits
as `decide.py` on all 39 questions and every page's words equal the run's values; on the phone, three runs (two of them
recorded) gave every answer's option equal to the Mac's int8mlp read-out (max |dp| 0.0009) and showed the inbox in
4.69–4.73 s and the log in 2.99–3.01 s, the pages up for 2.0 s (the title card), 3.6 s (a message), 3.0 s and 4.1 s.
