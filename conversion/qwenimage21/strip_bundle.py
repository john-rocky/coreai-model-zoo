"""Re-save a Core AI ``.aimodel`` with its export-time debug locations stripped: the copy that ships.

coreai-torch records every op's Python source location in the bytecode, so an exported ``main.mlirb`` holds the
export machine's absolute paths (the source tree, the venv's torch files). This loads ``<bundle>.aimodel``, runs
coreai-torch's ``strip_debug_info`` on its program (every location becomes an unknown one; the ops and the weights
are not touched) and saves ``<out>.aimodel``: a new ``main.mlirb`` and ``main.hash``, and a ``metadata.json`` that is
the source's, or ``--metadata-from``'s (a copy whose description was already cleaned), with a
``creatorDefinedMetadata.strip`` record appended: the source and the stripped ``main.mlirb`` bytes and sha256, the
tool and its versions. It stops, and deletes the output, when an absolute path is left in ``main.mlirb`` or
``metadata.json``, or when the function signatures, the storage / compute types or the op counts changed.

A stripped asset is a new asset (new ``main.hash``, a new compile, a new runtime cache entry), so it is gated again:
its outputs should equal the unstripped bundle's bit for bit.

``--aot`` compiles the stripped asset the way every bundle of this port is compiled (``xcrun coreai-build compile
... --platform macOS --architecture h16c --preferred-compute gpu --expect-frequent-reshapes``) into
``<out dir>_aot_efr/<name>.h16c.aimodelc``. Both steps refuse to start below 60 GiB of free disk.

Keep one program object: ``AIModelAsset.program`` builds a new object on every access, so
``strip_debug_info(asset.program)`` followed by ``asset.program.save_asset(...)`` saves the unstripped program.

Run (coreai-models base venv, from conversion/qwenimage21/; E = the exports dir):
  python strip_bundle.py $E/qi21_vae_256_fp32/qi21_vae_256_fp32.aimodel \\
      $E/qi21_vae_256_fp32_s/qi21_vae_256_fp32.aimodel --aot --json strip_qi21_vae_256_fp32.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

AOT_FLAGS = ["--platform", "macOS", "--architecture", "h16c", "--preferred-compute", "gpu",
             "--expect-frequent-reshapes"]
MIN_FREE_GIB = 60
CHUNK = 1 << 24


def needles() -> dict[str, bytes]:
    """What a shipped file must not contain: absolute machine paths, this machine's home and user name."""
    home = Path.home()
    return {n: n.encode() for n in ("/Users/", "/home/", "/private/", "/tmp/claude", str(home), home.name)}


def scan(p: Path, keys: dict[str, bytes]) -> dict:
    """One read: bytes, sha256 and the occurrences of every needle (matches across chunk seams included)."""
    h, counts, tail = hashlib.sha256(), dict.fromkeys(keys, 0), b""
    keep = max(len(v) for v in keys.values()) - 1
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
            w = tail + b
            for k, v in keys.items():
                counts[k] += w.count(v) - tail.count(v)
            tail = w[-keep:]
    return {"bytes": p.stat().st_size, "sha256": h.hexdigest(), "occurrences": counts}


def grep_count(p: Path, pattern: str = "/Users/") -> int:
    """``grep -c -a`` (lines that match), the check the ship rule names."""
    r = subprocess.run(["grep", "-c", "-a", "-F", pattern, str(p)], capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"grep failed on {p}: {r.stderr}")
    return int(r.stdout.strip() or 0)


def summary(asset) -> dict:
    """Signatures, storage / compute types and op counts: what a strip must leave unchanged."""
    s = asset.summary(include_statistics=True)
    fns = {f: {"inputs": s.function_inputs(f), "outputs": s.function_outputs(f), "states": s.function_states(f)}
           for f in s.function_names}
    return json.loads(json.dumps({"functions": fns, "storage_types": sorted(s.storage_types),
                                  "compute_types": sorted(s.compute_types),
                                  "operation_distribution": sorted(s.operation_distribution),
                                  "compatibility_version": s._compatibility_version}))


def runtime_json(d: dict) -> str:
    """The runtime's own metadata.json layout (2-space indent, ``"key" : value``, ``/`` escaped, no final newline),
    so a diff against the source file shows only the appended record."""
    return json.dumps(d, indent=2, separators=(",", " : "), ensure_ascii=False).replace("/", "\\/")


def free_gib() -> float:
    return shutil.disk_usage(Path.home()).free / 2**30


def need_disk(what: str) -> None:
    if free_gib() < MIN_FREE_GIB:
        sys.exit(f"{what}: {free_gib():.0f} GiB free < {MIN_FREE_GIB} GiB, stopped")


def aot_compile(asset: Path, out_dir: Path) -> dict:
    target = out_dir / f"{asset.stem}.h16c.aimodelc"
    need_disk("AOT compile")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["xcrun", "coreai-build", "compile", str(asset), "--output", str(out_dir), *AOT_FLAGS]
    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"cmd": cmd, "seconds": round(time.monotonic() - t0, 1), "returncode": proc.returncode,
           "coreai_build": subprocess.run(["xcrun", "coreai-build", "--version"], capture_output=True,
                                          text=True).stdout.strip(),
           "stderr_tail": proc.stderr.splitlines()[-20:], "aimodelc": str(target)}
    if proc.returncode != 0 or not target.is_dir():
        print(json.dumps(rec, indent=1), flush=True)
        sys.exit("coreai-build compile failed")
    rec["main_hash_hex"] = (target / "main.hash").read_bytes().hex()
    rec["files"] = {str(p.relative_to(target)): p.stat().st_size for p in sorted(target.rglob("*")) if p.is_file()}
    rec["bytes"] = sum(rec["files"].values())
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", help="the exported <name>.aimodel directory")
    ap.add_argument("out", help="the stripped <name>.aimodel to write (its parent directory is created)")
    ap.add_argument("--metadata-from", default=None, help="the metadata.json to copy (default: the bundle's own)")
    ap.add_argument("--aot", action="store_true", help="also compile the stripped asset into <out dir>_aot_efr/")
    ap.add_argument("--json", default=None, help="write the record here as well")
    a = ap.parse_args()
    src, out = Path(a.bundle).expanduser().resolve(), Path(a.out).expanduser().absolute()
    if src.suffix != ".aimodel" or not (src / "main.mlirb").is_file():
        sys.exit(f"{src}: not an .aimodel directory with a main.mlirb")
    if out.suffix != ".aimodel":
        sys.exit(f"{out}: the output must be a <name>.aimodel directory")
    if out.exists():
        sys.exit(f"{out} exists: a stripped asset is written once (remove it on purpose, then run again)")
    aot_dir = out.parent.parent / f"{out.parent.name}_aot_efr"
    if a.aot and (aot_dir / f"{out.stem}.h16c.aimodelc").exists():
        sys.exit(f"{aot_dir / (out.stem + '.h16c.aimodelc')} exists: an AOT compile never overwrites an asset")
    keys = needles()
    meta_src = Path(a.metadata_from).expanduser() if a.metadata_from else src / "metadata.json"
    base_text = meta_src.read_text()
    base = json.loads(base_text)
    held = [k for k, v in keys.items() if v.decode() in json.dumps(base, ensure_ascii=False)]
    if held:
        sys.exit(f"{meta_src} holds {held}: pass --metadata-from a copy whose description names the source instead")
    if "strip" in base.get("creatorDefinedMetadata", {}):
        sys.exit(f"{meta_src} already carries a strip record: strip the exported asset, not a stripped one")
    need_disk("strip")

    from coreai.authoring import AIModelAsset
    from coreai_torch.debugging.debug_info import strip_debug_info

    before = scan(src / "main.mlirb", keys)
    before["grep_c_users"] = grep_count(src / "main.mlirb")
    src_asset = AIModelAsset.load(src)
    shape_before = summary(src_asset)
    t0 = time.monotonic()
    program = src_asset.program                      # one object: strip and save the same program
    strip_debug_info(program)
    out.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(out)
    del program
    secs = round(time.monotonic() - t0, 1)
    after = scan(out / "main.mlirb", keys)
    after["grep_c_users"] = grep_count(out / "main.mlirb")
    shape_after = summary(AIModelAsset.load(out))

    record = {
        "what": "the .aimodel re-saved after coreai-torch's strip_debug_info: every op's source location (file, "
                "line, call stack) replaced by an unknown location; the ops and the weights are not touched",
        "from_main_mlirb": {"bytes": before["bytes"], "sha256": before["sha256"]},
        "main_mlirb": {"bytes": after["bytes"], "sha256": after["sha256"]},
        "tool": "conversion/qwenimage21/strip_bundle.py",
        "versions": {"coreai-torch": version("coreai-torch"), "coreai-core": version("coreai-core")},
    }
    meta = dict(base)
    meta["creatorDefinedMetadata"] = {**base.get("creatorDefinedMetadata", {}), "strip": record}
    (out / "metadata.json").write_text(runtime_json(meta))
    m = AIModelAsset.load(out).metadata              # the runtime must read back what was written
    stamp = (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(m.creation_date)) if m.creation_date is not None else None)
    meta_scan = scan(out / "metadata.json", keys)

    problems = []
    if after["sha256"] == before["sha256"]:
        problems.append("main.mlirb did not change")
    if any(after["occurrences"].values()) or after["grep_c_users"]:
        problems.append(f"main.mlirb still holds paths: {after['occurrences']}, grep -c {after['grep_c_users']}")
    if any(meta_scan["occurrences"].values()):
        problems.append(f"metadata.json holds paths: {meta_scan['occurrences']}")
    if (out / "main.hash").read_bytes().hex() != after["sha256"]:
        problems.append("main.hash != sha256(main.mlirb)")
    if shape_after != shape_before:
        problems.append("signatures / types / op counts changed")
    if (m.model_description, m.license, m._producer, stamp) != (
            base.get("description", ""), base.get("license", ""), base.get("producer", ""), base.get("creationDate")):
        problems.append("the runtime reads the metadata back differently")
    if m.creator_defined_metadata.get("strip") != record:
        problems.append("the runtime reads the strip record back differently")

    rec = {"bundle": str(src), "out": str(out), "metadata_from": str(meta_src), "strip_seconds": secs,
           "before": before, "after": after, "summary_unchanged": shape_after == shape_before,
           "summary": shape_before, "metadata": {"sha256": meta_scan["sha256"], "bytes": meta_scan["bytes"],
                                                 "occurrences": meta_scan["occurrences"]},
           "record": record, "problems": problems}
    if problems:
        shutil.rmtree(out)
        print(json.dumps(rec, indent=1), flush=True)
        sys.exit("strip refused: " + "; ".join(problems) + f" ({out} removed)")
    if a.aot:
        rec["aot"] = aot_compile(out, aot_dir)
    print(json.dumps(rec, indent=1), flush=True)
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(rec, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
