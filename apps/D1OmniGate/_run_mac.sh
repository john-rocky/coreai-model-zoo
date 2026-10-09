#!/bin/zsh
# Run the macOS build of D1OmniGate on this Mac on the stage directory, and check it against the Mac's earlier Swift run
# (round 10's apps/D1Omni parity on the same stripped graphs) before the phone: the check that the gate app scores what
# the library scored, row for row and bit for bit. The JIT form reads the stage's jit/ (the bytes the phone gets); the AOT
# form reads the Mac's own h16c AOT of the same bundles ($D1_WORK/mac/aot, ./_stage.sh --mac): the app refuses an iPhone
# AOT (.h19p.) on a Mac, so the stage's aot/ and ane/ are never opened here. --ane adds the Neural Engine stage on the
# Mac's own ANE AOT ($D1_WORK/mac/ane: round 7's compile of the same graphs) to try the stage's code; its numbers are
# not the phone's. The bench is short (D1_BENCH_TIMED 5, 1 s rests): a dry run of the code, not a measurement.
# Adapted from apps/KevGate/_run_mac.sh. This script takes no lock and waits for nothing: run it through
#   ~/code/standup/tools/quiet/quiet_wait.py -- <K>/scripts/r11_run.sh <tag> -- ./_run_mac.sh
# (quiet_wait at the start, r7_yield stops the app while another lane's Mac timing window is open).
#   ./_run_mac.sh            D1_STAGES=assets,parity_jit,parity_aot,bench
#   ./_run_mac.sh --ane      + ane
#   any D1_* knob of GateRunner.swift passes through the environment
# The Mac check (exit 0 only when it holds): status done; parity_jit and parity_aot each 78/78 rows with the bar PASS,
# logits and probabilities bit-equal to mac_ref.json on every row, every media array (6 items) equal to the Mac's, the
# control FAIL; the AOT rows bit-equal to the JIT rows of the same process; every bench item's outputs PASS.
# Output: $D1_WORK/mac_runs/<run id>/{result.json,result.log,memory.tsv,app.stdout,run.out,mac_check.json}
set -u
DIR=${0:A:h}
W=${D1_WORK:-$HOME/code/coreai/_d1_omni/device}
APP=$(cat $W/app_path_mac.txt 2>/dev/null)
[ -d "$APP" ] || { echo "no macOS build: run ./_build.sh --mac"; exit 1; }
BIN=$APP/Contents/MacOS/D1OmniGate
S=${D1_ASSETS:-$W/assets/D1OmniAssets}
[ -f $S/MD5SUMS ] || { echo "nothing staged at $S: run ./_stage.sh"; exit 1; }
ANE=0
for a in "$@"; do
  case $a in
    --ane) ANE=1 ;;
    *) echo "unknown option $a (--ane)"; exit 1 ;;
  esac
done
RUN_ID=${D1_RUN_ID:-mac-$(date +%Y%m%d-%H%M%S)}
OUT=$W/mac_runs/$RUN_ID
mkdir -p $OUT
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/run.out; }
export D1_ASSETS=$S D1_OUT=$OUT D1_RUN_ID=$RUN_ID
export D1_AOT_DIR=${D1_AOT_DIR:-$W/mac/aot} D1_ANE_DIR=${D1_ANE_DIR:-$W/mac/ane}
export D1_STAGES=${D1_STAGES:-assets,parity_jit,parity_aot,bench,long$( (( ANE )) && echo ,ane)}
export D1_BENCH_TIMED=${D1_BENCH_TIMED:-5} D1_BENCH_REST=${D1_BENCH_REST:-1} D1_WAIT_NOMINAL=${D1_WAIT_NOMINAL:-0}
[ -d $D1_AOT_DIR ] || { say "no Mac AOT at $D1_AOT_DIR: run ./_stage.sh --mac"; exit 1; }
CAP=${D1_MAC_CAP:-900}
say "run $RUN_ID: $BIN on $S, stages $D1_STAGES, AOT $D1_AOT_DIR, ANE $D1_ANE_DIR, bench timed $D1_BENCH_TIMED, cap $CAP s"
t0=$SECONDS
$BIN > $OUT/app.stdout 2>&1 &
pid=$!
echo $pid > $OUT/app.pid
while kill -0 $pid 2>/dev/null; do
  if (( SECONDS - t0 > CAP )); then
    say "the app ran past $CAP s: terminating pid $pid"
    kill $pid; sleep 10; kill -9 $pid 2>/dev/null
    break
  fi
  sleep 2
done
wait $pid 2>/dev/null
rc=$?
say "app exit $rc after $((SECONDS - t0)) s"
if [ ! -f $OUT/result.json ]; then say "no result.json (see $OUT/app.stdout)"; exit 1; fi
echo "--- last lines of result.log"; tail -4 $OUT/result.log | cut -c1-230
echo "--- summary"; $DIR/_run.sh --summary $OUT/result.json $RUN_ID | tee -a $OUT/run.out
/usr/bin/python3 -I - $OUT/result.json $OUT/mac_check.json <<'PY' | tee -a $OUT/run.out
import json, sys
r = json.load(open(sys.argv[1]))
S = r.get("stages", {})
checks = {"status done": r.get("status") == "done"}
long = S.get("long", {})
parts = [(k, S.get(k), 74) for k in ("parity_jit", "parity_aot")]
parts += [(f"long {k}", long.get(k), 4) for k in ("parity_jit", "parity_aot") if "long" in S]
for k, s, want in parts:
    if s is None:
        continue
    sm, m = s.get("summary", {}), s.get("summary", {}).get("mac", {})
    n = s.get("rows_planned")
    checks[f"{k}: {s.get('rows_done')}/{n} rows"] = s.get("rows_done") == n == want
    checks[f"{k}: bar PASS"] = bool(sm.get("bar_pass"))
    checks[f"{k}: logits bit-equal to mac_ref {m.get('logits_bit_equal')}/{want}"] = m.get("logits_bit_equal") == want
    checks[f"{k}: probabilities bit-equal to mac_ref {m.get('probs_bit_equal')}/{want}"] = m.get("probs_bit_equal") == want
    if want == 74:
        ma = s.get("media_arrays_equal_mac", {})
        checks[f"{k}: media arrays = Mac {ma.get('equal')}/{ma.get('items')}"] = ma.get("equal") == ma.get("items") == 6
        checks[f"{k}: control FAIL"] = bool(s.get("control", {}).get("caught"))
    if k.endswith("parity_aot"):
        v = s.get("vs_jit", {})
        checks[f"{k}: logits = JIT {v.get('logits_bit_equal')}/{v.get('rows')}"] = v.get("logits_bit_equal") == v.get("rows") == want
if "long" in S:
    for it in long.get("bench", {}).get("items", []):
        checks[f"long bench {it.get('workload')}: outputs {it.get('output_check', {}).get('status')}"] = it.get("output_check", {}).get("status") == "PASS"
for k in ("parity_jit", "parity_aot"):
    s = S.get(k)
    if s is None:
        continue
    if s.get("rows_l64_planned"):
        l = s.get("summary_l64", {})
        checks[f"{k}: L64 {s.get('rows_l64_done')}/{s.get('rows_l64_planned')} rows, bar PASS"] = (
            s.get("rows_l64_done") == s.get("rows_l64_planned") == 18 and bool(l.get("bar_pass")))
        checks[f"{k}: L64 logits / p bit-equal to mac_ref {l.get('mac', {}).get('logits_bit_equal')} / "
               f"{l.get('mac', {}).get('probs_bit_equal')} of 18"] = (
            l.get("mac", {}).get("logits_bit_equal") == l.get("mac", {}).get("probs_bit_equal") == 18)
    if k == "parity_aot":
        if s.get("rows_l64_planned"):
            v = s.get("vs_jit_l64", {})
            checks[f"parity_aot: L64 logits = parity_jit {v.get('logits_bit_equal')}/{v.get('rows')}"] = v.get("logits_bit_equal") == v.get("rows") == 18
for it in S.get("bench", {}).get("items", []):
    checks[f"bench {it.get('workload')}: outputs {it.get('output_check', {}).get('status')}"] = it.get("output_check", {}).get("status") == "PASS"
ok = all(checks.values())
json.dump({"run_id": r.get("run_id"), "checks": checks, "mac_check": "PASS" if ok else "FAIL"}, open(sys.argv[2], "w"), indent=1)
for c, v in checks.items():
    print(f"  {'ok ' if v else 'NO '} {c}")
print(f"MAC CHECK {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 3)
PY
mrc=${pipestatus[1]}
echo "files: $OUT"
exit $mrc
