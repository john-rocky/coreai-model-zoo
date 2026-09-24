#!/bin/zsh
# Record the Diarize demo from the Mac: QuickTime mirrors the USB-connected iPhone (screen and sound), the app
# plays N3DAssets/demo/<clip>.wav by itself (DIARIZE_DEMO, Sources/DiarizeDemo.swift): the speaker lanes grow
# with the audio, then the transcript and the summary line. The recording is exported to ~/Desktop and cut for X.
#   ./record-demo.sh [clip]                 default demo_clip
#   TRIGGER=1 ./record-demo.sh [clip]       load before recording: the app waits for a trigger file, then a launch
#                                           without --terminate-existing brings it to the front (when a terminate
#                                           launch leaves the home screen showing)
#   ./record-demo.sh --cut <raw.mov> [t1 t2]   cut a recording for X again (t1..t2 = the part played at 3x)
# Knobs: UDID, ENGINE (parakeet | whisper | nemotron | qwen3asr), SLOW (the transcript part plays at 3x when it
# takes longer than this, default 10 s), HOLD (seconds kept after DONE, default 4), CAP (5 s polls, default 120).
# Before: the phone unlocked, on USB, in airplane mode (no banners), with the app, Library/Application Support/
# N3DAssets (demo/<clip>.wav inside) and an ASR on it (Documents/Models/Parakeet, else Nemotron; Whisper would have
# to download). QuickTime's last movie-recording source must be the iPhone, as camera and as microphone.
# X cut: scale=-2:1900, sound kept; 1x, except the transcript part at 3x with a badge when it is long.
set -u
cd "$(dirname "$0")"
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode-27.0.0-RC.app/Contents/Developer}
zmodload zsh/datetime
UDID=${UDID:-A6F3E849-1947-5202-9AD1-9C881CA58EEF}
BID=com.daisukemajima.coreaiaudio
SLOW=${SLOW:-10}
TS=$(date +%Y%m%d_%H%M%S)
LOG=_record_$TS.log
WORK=$(mktemp -d /tmp/diarize-demo.XXXXXX)
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

# PNG badge for the 3x part (brew ffmpeg has no drawtext)
badge() {
  python3 - "$1" <<'PY' 2>/dev/null
import sys
from PIL import Image, ImageDraw, ImageFont
W, H = 240, 88
img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([0, 0, W - 1, H - 1], radius=24, fill=(0, 0, 0, 175))
font = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 50)
text = "3× speed"
l, t, r, b = d.textbbox((0, 0), text, font=font)
d.text(((W - (r - l)) / 2 - l, (H - (b - t)) / 2 - t), text, font=font, fill=(255, 255, 255, 255))
img.save(sys.argv[1])
PY
}

# cut_x <raw.mov> <out.mp4> [t1 t2]: 1x with the sound; with t1 t2, that part at 3x (sound too) under the badge
cut_x() {
  local IN=$1 OUT=$2 T1=${3:-} T2=${4:-}
  local ENC=(-c:v libx264 -preset medium -crf 20 -movflags +faststart)
  local AUD=$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$IN" | head -1)
  [[ -z "$AUD" ]] && say "WARNING: $IN has no sound track (QuickTime's microphone was not the iPhone); cutting without sound"
  local ACODEC=(-c:a aac -b:a 160k); [[ -z "$AUD" ]] && ACODEC=(-an)
  if [[ -z "$T1" || -z "$T2" ]]; then
    ffmpeg -v error -y -i "$IN" -filter:v "fps=30,scale=-2:1900:flags=lanczos,format=yuv420p" $ENC $ACODEC "$OUT"
    return
  fi
  local B=$WORK/badge.png
  badge $B || { say "WARNING: no badge (python3 + PIL); the 3x part goes without it"; B=""; }
  local V="[0:v]fps=30,split=3[v0][v1][v2];[v0]trim=0:$T1,setpts=PTS-STARTPTS[va];[v1]trim=$T1:$T2,setpts=(PTS-STARTPTS)/3[vb0];[v2]trim=$T2,setpts=PTS-STARTPTS[vc]"
  local IMG=()
  if [[ -n "$B" ]]; then V="$V;[vb0][1:v]overlay=W-w-40:48[vb]"; IMG=(-i $B); else V="$V;[vb0]null[vb]"; fi
  if [[ -n "$AUD" ]]; then
    local A="[0:a]asplit=3[a0][a1][a2];[a0]atrim=0:$T1,asetpts=PTS-STARTPTS[aa];[a1]atrim=$T1:$T2,asetpts=PTS-STARTPTS,atempo=3[ab];[a2]atrim=$T2,asetpts=PTS-STARTPTS[ac]"
    ffmpeg -v error -y -i "$IN" $IMG -filter_complex \
      "$V;$A;[va][aa][vb][ab][vc][ac]concat=n=3:v=1:a=1[vv][ao];[vv]fps=30,scale=-2:1900:flags=lanczos,format=yuv420p[vo]" \
      -map "[vo]" -map "[ao]" $ENC $ACODEC "$OUT"
  else
    ffmpeg -v error -y -i "$IN" $IMG -filter_complex \
      "$V;[va][vb][vc]concat=n=3:v=1:a=0[vv];[vv]fps=30,scale=-2:1900:flags=lanczos,format=yuv420p[vo]" \
      -map "[vo]" $ENC -an "$OUT"
  fi
}

report_x() {
  local d=$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$1" | cut -c1-6)
  say "X: $1 ($(du -h "$1" | cut -f1), ${d} s$( (( ${d%.*} > 140 )) && echo '; over the 2:20 X limit' ))"
}

if [[ "${1:-}" == "--cut" ]]; then
  RAW=${2:?usage: record-demo.sh --cut <raw.mov> [t1 t2]}
  XOUT=${RAW:r}-x.mp4
  cut_x "$RAW" "$XOUT" "${3:-}" "${4:-}" && report_x "$XOUT"
  exit
fi

CLIP=${1:-demo_clip}
RAW=$HOME/Desktop/coreai-audio-diarize-$TS.mov
XOUT=$HOME/Desktop/coreai-audio-diarize-$TS-x.mp4
ENVJ="\"DIARIZE_DEMO_LOG\":\"1\"${ENGINE:+,\"DIARIZE_DEMO_ENGINE\":\"$ENGINE\"}"
dc() { xcrun devicectl device $1 $2 --device $UDID "${@:3}"; }   # --device before a launch's bundle id

# preflight: nobody else on the phone, no devicectl running, the clip is in the app's container
HOLD_FILE=$HOME/code/coreai/ondevice/.device_hold
if [[ -e $HOLD_FILE && "${IGNORE_HOLD:-0}" != 1 ]]; then
  say "device hold present ($HOLD_FILE: $(head -c 200 $HOLD_FILE)); IGNORE_HOLD=1 to go anyway"; exit 2
fi
busy() { ps -axo pid,command | grep -qE "^ *[0-9]+ +(/[^ ]*/)?(xcrun )?devicectl device (process launch|install app|copy (to|from))"; }
for w in 1 2 3 4 5 6; do busy || break; sleep 10; done
busy && { say "device busy: another devicectl launch / install / copy is running"; exit 2; }
L=$(dc info files --domain-type appDataContainer --domain-identifier $BID --subdirectory "Library/Application Support/N3DAssets/demo" 2>&1)
if echo "$L" | grep -q ERROR || ! echo "$L" | grep -q "$CLIP.wav"; then
  say "N3DAssets/demo/$CLIP.wav not in the app's container (app installed? phone unlocked, on USB?)"
  echo "$L" | tail -5; exit 2
fi

launch() {  # launch [--terminate-existing] [env JSON]: retried while the phone answers Busy
  local args=() out
  [[ "${1:-}" == --terminate-existing ]] && { args+=(--terminate-existing); shift; }
  [[ -n "${1:-}" ]] && args+=(--environment-variables "$1")
  for t in {1..8}; do
    out=$(dc process launch $args $BID 2>&1); echo "$out" >> $LOG
    echo "$out" | grep -q "Launched application" && return 0
    say "launch retry $t: $(echo "$out" | grep -m1 -iE 'error|busy' | cut -c1-140)"; sleep 15
  done
  return 1
}

# poll_log <since epoch> <done line>: pull Documents/diarize_demo every 5 s, print new lines of the newest log from
# this launch, stop at the done line or an ERROR (the directory lands at the destination or one level under it)
LOGFILE=""
poll_log() {
  local since=$1 want=$2 seen=0 gone=0 f n
  for i in {1..${CAP:-120}}; do
    sleep 5
    rm -rf $WORK/pull
    dc copy from --domain-type appDataContainer --domain-identifier $BID --source Documents/diarize_demo \
      --destination $WORK/pull >/dev/null 2>&1
    f=$(find $WORK/pull -name '[0-9]*.log' 2>/dev/null | while read -r p; do
          n=${p:t:r}; (( n >= since - 5 )) && echo "$n $p"; done | sort -n | tail -1 | cut -d' ' -f2-)
    if [[ -n "$f" ]]; then
      LOGFILE=$f
      n=$(wc -l < $f)
      (( n > seen )) && { tail -n $((n - seen)) $f | cut -c1-200 | tee -a $LOG; seen=$n; }
      grep -q " ERROR " $f && return 1
      grep -qF "$want" $f && return 0
    fi
    if (( i % 12 == 0 )); then   # every minute: is the app still running? (two misses in a row = gone)
      if dc info processes 2>/dev/null | grep -q "coreai-audio"; then gone=0; else gone=$((gone + 1)); fi
      (( gone >= 2 )) && { say "the app is gone (crash?)"; return 1; }
    fi
  done
  say "no '$want' after $(( ${CAP:-120} * 5 )) s"; return 1
}

RECORDING=0
stop_recording() {
  (( RECORDING )) || return 0
  RECORDING=0
  osascript -e 'tell application "QuickTime Player" to stop document "Movie Recording"'
}
trap stop_recording EXIT

if [[ "${TRIGGER:-0}" == 1 ]]; then
  LAUNCH0=${EPOCHREALTIME%.*}
  launch --terminate-existing "{\"DIARIZE_DEMO_TRIGGER\":\"1\",$ENVJ}" || { say "ERROR launch never accepted"; exit 1; }
  say "launched (trigger mode); waiting for READY (the models load first)"
  poll_log $LAUNCH0 "READY" || { say "ERROR the app did not get ready"; exit 1; }
fi

osascript <<'AS' || { say "ERROR QuickTime: the movie-recording source is not the iPhone"; exit 1; }
tell application "QuickTime Player"
    activate
    new movie recording
end tell
delay 3
tell application "System Events" to tell process "QuickTime Player"
    set s to size of window "Movie Recording"
    if (item 2 of s) < (item 1 of s) then error "QuickTime source is not the iPhone (landscape preview)"
end tell
AS
osascript -e 'tell application "QuickTime Player" to start document "Movie Recording"'
RECORDING=1
REC0=$EPOCHREALTIME
say "recording since $(date +%H:%M:%S)"

if [[ "${TRIGGER:-0}" == 1 ]]; then
  E=${EPOCHREALTIME%.*}
  T=$WORK/autoplay-$CLIP-$E.trigger
  : > $T
  sleep 2
  dc copy to --domain-type appDataContainer --domain-identifier $BID --source $T --destination Documents/${T:t} >> $LOG 2>&1
  launch || say "WARNING: the foreground launch was not accepted"
  say "trigger ${T:t} placed"
  poll_log $LAUNCH0 "DONE $CLIP $E"; ok=$?
else
  LAUNCH0=${EPOCHREALTIME%.*}
  launch --terminate-existing "{\"DIARIZE_DEMO\":\"$CLIP\",$ENVJ}" || { say "ERROR launch never accepted"; exit 1; }
  sleep 3
  launch || say "WARNING: the foreground launch was not accepted"   # a terminate launch can leave it behind
  say "launched; the app loads, waits 2 s, then plays $CLIP"
  poll_log $LAUNCH0 "DONE $CLIP"; ok=$?
fi

sleep ${HOLD:-4}
stop_recording
sleep 2
osascript <<AS
tell application "QuickTime Player"
    set d to document 1
    export d in POSIX file "$RAW" using settings preset "1080p"
    delay 2
    close d saving no
end tell
AS
for i in {1..60}; do [[ -s "$RAW" ]] && break; sleep 2; done
[[ -s "$RAW" ]] || { say "ERROR no export at $RAW"; exit 1; }
say "raw: $RAW ($(du -h "$RAW" | cut -f1)); app log: ${LOGFILE:-none} (copied to ${LOG:r}.app.log)"
[[ -n "$LOGFILE" ]] && cp $LOGFILE ${LOG:r}.app.log
(( ok == 0 )) || say "WARNING: the run did not end with DONE; cutting what was recorded"

# the transcript part in the recording's time: the app's asr marks (phone clock) minus the recording start
T1="" T2=""
if [[ -n "$LOGFILE" ]] && grep -q "asr_start=" $LOGFILE; then
  A0=$(grep -o 'asr_start=[0-9.]*' $LOGFILE | tail -1 | cut -d= -f2)
  A1=$(grep -o 'asr_end=[0-9.]*' $LOGFILE | tail -1 | cut -d= -f2)
  if (( A1 - A0 > SLOW )); then
    T1=$(printf %.2f $(( A0 - REC0 + 0.3 ))); T2=$(printf %.2f $(( A1 - REC0 )))
    say "transcript part $(printf %.1f $(( A1 - A0 ))) s > ${SLOW} s: 3x from ${T1} s to ${T2} s of the recording (redo: ./record-demo.sh --cut $RAW <t1> <t2>)"
  else
    say "transcript part $(printf %.1f $(( A1 - A0 ))) s: all 1x"
  fi
fi
cut_x "$RAW" "$XOUT" $T1 $T2 && report_x "$XOUT"
rm -rf $WORK
