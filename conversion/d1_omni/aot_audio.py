#!/usr/bin/env python3
"""AOT-compile an exported audio bundle (one clip bucket) for this Mac (h16c), or probe it for the Neural Engine.

    python3 conversion/d1_omni/aot_audio.py --precision wfp16 --sec 10
    python3 conversion/d1_omni/aot_audio.py --precision fp16 --sec 10 --preferred-compute neural-engine --tag r7

Compiles <work>/bundles/d1-omni-600m/macos/audio-<precision>-<sec>s/ (export_audio.py; its files are checked against the
export manifest first) with aot_decide.py's command

    $(xcrun -f coreai-build) compile <bundle>.aimodel --output <dir> --platform macOS
        --preferred-compute gpu|neural-engine --architecture h16c

(static shapes: no --expect-frequent-reshapes) into <work>/compiled/audio-<precision>-<sec>s-h16c/ (gpu) or -ane/
(neural-engine; --tag <t> appends -<t>), and records the same manifest as aot_vision.py (seconds, exit code, files,
resources.bin bytes, main.hash = the compile-cache entry a load creates, the ANE regions counted by unique name,
coreai-build inspect of both assets, disk free) in <dir>/provenance/aot-manifest.json and
results/aot_audio_<precision>_<sec>s_<h16c|ane>[_<t>].json. A neural-engine probe with no region keeps only the
manifest (the .aimodelc is removed). An existing folder is never replaced.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import work_path  # noqa: E402

from aot_decide import ARCHITECTURE, ane_regions, coreai_build, disk_free, inspect, main_hash, tool_env  # noqa: E402
from export_audio import bundle_dir  # noqa: E402
from export_decide import environment, file_inventory, sha256_file, write_json  # noqa: E402

import d1_omni_audio as da  # noqa: E402

WORK = work_path("_d1_omni")
SUFFIX = {"gpu": "h16c", "neural-engine": "ane"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=da.PRECISIONS, required=True)
    parser.add_argument("--sec", type=int, choices=da.CLIP_SECONDS, required=True)
    parser.add_argument("--preferred-compute", choices=sorted(SUFFIX), default="gpu")
    parser.add_argument("--tag", default=None, help="compile into <dir>-<tag>/ (a new folder and results file)")
    args = parser.parse_args()
    source_dir = bundle_dir(args.precision, args.sec)
    manifest = json.loads((source_dir / "provenance" / "export-manifest.json").read_text())
    bundle = source_dir / manifest["bundle"]
    for item in manifest["files"]:
        if sha256_file(bundle / item["path"]) != item["sha256"]:
            raise SystemExit(f"{bundle / item['path']} changed since its export")
    suffix = SUFFIX[args.preferred_compute] + (f"-{args.tag}" if args.tag else "")
    out_dir = WORK / "compiled" / f"audio-{args.precision}-{args.sec}s-{suffix}"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists: an existing folder is never replaced")
    out_dir.mkdir(parents=True)
    cb = coreai_build()
    argv = [cb, "compile", str(bundle), "--output", str(out_dir), "--platform", "macOS", "--preferred-compute",
            args.preferred_compute, "--architecture", ARCHITECTURE]
    before = disk_free()
    print("$", " ".join(argv), flush=True)
    t0 = time.perf_counter()
    done = subprocess.run(argv, capture_output=True, text=True, env=tool_env())
    seconds = time.perf_counter() - t0
    after = disk_free()
    (out_dir / "provenance").mkdir()
    (out_dir / "provenance" / "coreai-build.log").write_text(f"$ {' '.join(argv)}\n{done.stdout}\n--- stderr\n{done.stderr}")
    compiled = sorted(out_dir.glob("*.aimodelc"))
    record = {"precision": args.precision, "graph": "audio", "bucket_s": args.sec,
              "preferred_compute": args.preferred_compute, "architecture": ARCHITECTURE, "tag": args.tag,
              "platform": "macOS", "command": argv,
              "developer_dir": tool_env()["DEVELOPER_DIR"],
              "coreai_build_version": subprocess.run([cb, "--version"], capture_output=True, text=True).stdout.strip(),
              "seconds": seconds, "exit_code": done.returncode, "stderr_tail": done.stderr[-4000:],
              "disk_free_before": before, "disk_free_after": after,
              "source": {"folder": str(source_dir), "bundle": manifest["bundle"], "bytes": manifest["bytes"],
                         "export_manifest_sha256": sha256_file(source_dir / "provenance" / "export-manifest.json"),
                         "export_status": manifest["status"], "precision": manifest["precision"]},
              "source_inspect": inspect(cb, bundle)}
    result = WORK / "results" / f"aot_audio_{args.precision}_{args.sec}s_{suffix.replace('-', '_')}.json"
    if done.returncode != 0 or len(compiled) != 1:
        record["status"] = "FAILED"
        write_json(out_dir / "provenance" / "aot-manifest.json", record)
        write_json(result, record)
        raise SystemExit(f"coreai-build exit {done.returncode}, {len(compiled)} .aimodelc:\n{done.stderr[-2000:]}")
    aimodelc = compiled[0]
    files = file_inventory(aimodelc)
    resources = [{"path": f["path"], "bytes": f["bytes"]} for f in files if Path(f["path"]).name == "resources.bin"]
    regions = ane_regions(aimodelc)
    record.update(status="COMPILED", aimodelc=aimodelc.name, bytes=sum(f["bytes"] for f in files),
                  files=files, resources_bin=resources, resources_bin_bytes=sum(r["bytes"] for r in resources),
                  resources_bin_over_source_bytes=sum(r["bytes"] for r in resources) / manifest["bytes"],
                  hashes=main_hash(aimodelc), ane=regions, compiled_inspect=inspect(cb, aimodelc),
                  code_sha256={"aot_audio.py": sha256_file(HERE / "aot_audio.py")},
                  environment=environment(("coreai-core", "coreai-torch")))
    kept = True
    if args.preferred_compute == "neural-engine" and regions["regions"] == 0:
        shutil.rmtree(aimodelc)  # no region: the evidence is this manifest, not the bundle
        kept = False
    record["aimodelc_kept"] = kept
    write_json(out_dir / "provenance" / "aot-manifest.json", record)
    write_json(result, record)
    print(json.dumps({"folder": str(out_dir), "aimodelc": aimodelc.name, "kept": kept, "seconds": round(seconds, 1),
                      "bytes": record["bytes"], "resources_bin_bytes": record["resources_bin_bytes"],
                      "source_bytes": manifest["bytes"], "ane_regions": regions["regions"],
                      "ane_ir_bytes": regions["ir_bytes"], "hashes": record["hashes"]}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
