#!/bin/bash
# Reproduce the speculative-decoding table on the muse-glimmer-30b card.
#
# Lossless greedy n-gram (prompt-lookup) spec-decode on the SHIPPED decode bundle —
# no re-export, no drafter model, no extra bytes. The tool is
# `coreai-models/swift/Sources/Tools/spec-decode`:
#
#   DEVELOPER_DIR=<Xcode-beta>/Contents/Developer \
#     swift build -c release --product spec-decode
#
# RUN THIS ALONE. Two things bite otherwise, both learned the hard way:
#   * a second GPU job makes a run report a lossless FAIL that does not reproduce
#     solo, on top of wrecking the timings;
#   * sustained load throttles the machine, and a throttled half corrupts the A/B
#     ratio in EITHER direction (a collapsed baseline once read a flattering 2.10x).
# Each row's `spec off` half is its own baseline, taken seconds before the `on`
# half; a row whose baseline leaves ~26-28 tok/s on an M4 Max is not usable.
set -u

BIN=${BIN:-coreai-models/.build/release/spec-decode}
BUNDLE=${BUNDLE:-coreai-models/exports/muse_glimmer_30b_decode_int4hu_block32_sym}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/chat.txt" <<'EOF'
Why do people find it so hard to change their minds, even when they are shown good evidence? Answer in a few short paragraphs, in your own words, without lists.
EOF

cat > "$WORK/code.txt" <<'EOF'
Here is a Python module:

```python
class RingBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def push(self, item):
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[1:]

    def pop(self):
        return self.items.pop(0)

    def peek(self):
        return self.items[0]

    def __len__(self):
        return len(self.items)
```

Rewrite this class so that push, pop and peek are all O(1) by using a fixed-size list with head and tail indices instead of list slicing. Keep the same public method names and behaviour, raise IndexError with a clear message when popping or peeking an empty buffer, and add a short docstring to every method.
EOF

cat > "$WORK/tools.txt" <<'EOF'
Set up the quarterly planning review for the on-device inference team. Find a 90 minute slot where taro.yamada@example.com, lena.ortiz@example.com and priya.raman@example.com are all free between 2026-08-18T09:00:00-07:00 and 2026-08-20T17:00:00-07:00, book it in Room Kirin with the title "Q3 on-device inference planning review", and put this agenda in the description: decode throughput targets, speculative decoding rollout, quantization backlog, hiring. Then email the same three people to tell them it is booked.
EOF

cat > "$WORK/tools.json" <<'EOF'
[
  {"name": "calendar.create_event",
   "description": "Create a calendar event and invite attendees.",
   "parameters": {"type": "object", "properties": {
     "title": {"type": "string", "description": "Event title shown in the calendar"},
     "start_time": {"type": "string", "description": "ISO 8601 start time"},
     "duration_minutes": {"type": "integer", "description": "Length of the event in minutes"},
     "attendees": {"type": "array", "items": {"type": "string"}, "description": "Email addresses to invite"},
     "location": {"type": "string", "description": "Room name or video link"},
     "description": {"type": "string", "description": "Agenda text for the invite body"}},
     "required": ["title", "start_time", "duration_minutes", "attendees"]}},
  {"name": "calendar.find_free_slot",
   "description": "Find a time window where every attendee is free.",
   "parameters": {"type": "object", "properties": {
     "attendees": {"type": "array", "items": {"type": "string"}, "description": "Email addresses to check"},
     "earliest": {"type": "string", "description": "ISO 8601 earliest acceptable start"},
     "latest": {"type": "string", "description": "ISO 8601 latest acceptable end"},
     "duration_minutes": {"type": "integer", "description": "Required window length in minutes"}},
     "required": ["attendees", "earliest", "latest", "duration_minutes"]}},
  {"name": "email.send",
   "description": "Send an email message.",
   "parameters": {"type": "object", "properties": {
     "to": {"type": "array", "items": {"type": "string"}, "description": "Recipient email addresses"},
     "subject": {"type": "string", "description": "Subject line"},
     "body": {"type": "string", "description": "Plain text body"}},
     "required": ["to", "subject", "body"]}}
]
EOF

echo "=== 1. logits-path gate: does one S=8 forward return all 8 rows? ==="
# The kill-switch. Row i of a [1,q,vocab] verify forward must equal the argmax of a
# plain S=1 decode at that position, or none of the rest is meaningful.
"$BIN" --model "$BUNDLE" --mode rows --rows 8 --prompt-file "$WORK/code.txt"

echo
echo "=== 2. verify cost per query length (the staircase that sets K) ==="
"$BIN" --model "$BUNDLE" --mode verify-cost --sweep-s "1,2,3,4,5,6,7,8,9,12,16" \
       --prompt-file "$WORK/tools.txt" --tools-file "$WORK/tools.json"

echo
echo "=== 3. the card's table: best draft length per workload, off vs on, 2 pairs ==="
for W in chat code tools; do
  EXTRA=""; CFG="-k 2"
  [ "$W" = tools ] && EXTRA="--tools-file $WORK/tools.json"
  [ "$W" = code ] && CFG="-k 7 --adaptive"      # code earns the longer plateau; see the note
  echo "--- $W ($CFG)"
  "$BIN" --model "$BUNDLE" --mode ab --max-tokens 256 --repeat 2 $CFG \
         --label "$W" --prompt-file "$WORK/$W.txt" $EXTRA \
    | grep -E "^RUN|LOSSLESS|SPEC OFF"
done
