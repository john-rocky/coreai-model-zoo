#!/usr/bin/env python3
"""The media prefix lengths the decision graph's buckets must hold, from host.py's copies, each checked against the
publisher's own code on synthetic inputs (no weights, no real image or clip needed).

    python3 conversion/d1_omni/media_lengths.py    # -> $ZOO_WORK_ROOT/_d1_omni/results/media_lengths.json

Images: host.layout / host.image_crops / host.image_prefix_length vs the publisher's vision.layout and
vision.preprocess -> prefix_length on a flat PIL image of each size (the crop shapes do not depend on the pixels).
Audio: host.audio_frames / audio_prefix_length / audio_encoder_steps vs the publisher's waveform() -> MelFrontend ->
ConvSubsampling (built with one channel: the time arithmetic does not depend on the width) on seeded noise.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import host  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402
from encode_rows import load_upstream  # noqa: E402

WORK = work_path("_d1_omni")
IMAGE_SIZES = [(384, 384), (640, 480), (1024, 768), (1536, 1024), (2048, 1536)]  # (width, height)
AUDIO_SECONDS = [5, 10, 20, 30]
AUDIO_EDGES = {"0.3 s (padded to 0.5 s)": 4800, "35 s (cut to 30 s)": 560000}


def image_row(up, width: int, height: int, origin: str) -> dict:
    from PIL import Image

    plan = host.layout(width, height)
    assert plan == up.vision.layout(width, height), (width, height)
    crops = host.image_crops(width, height)
    inputs = up.vision.preprocess(Image.new("RGB", (width, height), (90, 120, 150)))
    theirs = [tuple(s) for s in inputs["spatial_shapes"].tolist()]
    mine = [(h // host.PATCH, w // host.PATCH) for h, w in crops]
    assert mine == theirs, (width, height, mine, theirs)
    p = host.image_prefix_length([(width, height)])
    assert p == up.vision.prefix_length(inputs), (width, height)
    text = min(896, 16384 - p)
    return {"px": [width, height], "origin": origin, "tiled": plan["tiled"], "grid_wxh": list(plan["grid"]),
            "thumbnail_hxw": list(plan["thumbnail"]), "crops": len(crops),
            "patches_per_crop": [h * w for h, w in mine], "prefix": p, "text_max": text,
            "positions_max": p + text, "bucket_at_text_max": host.bucket_for(p + text)}


def audio_row(up, n: int, origin: str, rng) -> dict:
    x = up.audio.waveform((rng.standard_normal(n) * 3000).astype(np.int16))
    frontend = up.audio.MelFrontend(128)
    mel, frames = frontend(x)
    sub = up.audio.ConvSubsampling(128, 1, 8).eval()
    with torch.no_grad():
        y, lengths = sub(mel.transpose(1, 2), frames)
    mine = host.audio_frames(n)
    theirs = {"samples": x.shape[1], "stft_frames": mel.shape[2], "valid_frames": int(frames[0])}
    assert mine == theirs, (n, mine, theirs)
    assert host.audio_prefix_length(n) == int(lengths[0]) and host.audio_encoder_steps(n) == y.shape[1], n
    p = host.audio_prefix_length(n)
    return {"samples_in": n, "seconds_in": round(n / 16000, 4), "origin": origin, **mine,
            "encoder_steps": y.shape[1], "prefix": p, "text_max": min(15360, 16384 - p)}


def main() -> int:
    t0 = time.time()
    snapshot = Path(hf_snapshot(host.MODEL_ID, revision=host.MODEL_SHA))
    up = load_upstream(snapshot)
    fixtures = json.loads((WORK / "fixtures" / "records.json").read_text())["records"]
    images, sizes = [], {}
    for w, h in IMAGE_SIZES:
        sizes[(w, h)] = "candidate"
    for r in fixtures:
        if r["media"] and "images" in r["media"]:
            px = tuple((r["provenance"].get("image_info") or {}).get("px") or r["provenance"]["px"])
            sizes[px] = (sizes.get(px, "") + " " if px in sizes else "") + r["id"]
    for (w, h), origin in sizes.items():
        images.append(image_row(up, w, h, origin.strip()))
    rng = np.random.default_rng(0)
    audio = [audio_row(up, 16000 * s, f"{s} s", rng) for s in AUDIO_SECONDS]
    audio += [audio_row(up, n, label, rng) for label, n in AUDIO_EDGES.items()]
    for r in fixtures:
        if r["media"] and "audio" in r["media"]:
            prov = r["provenance"]
            n = prov["clip"]["frames"] if "clip" in prov else prov["audio_info"]["frames"]
            audio.append(audio_row(up, n, r["id"], rng))
    worst_untiled = max(host.image_prefix_length([(w, h)]) for w in range(32, 2049, 32) for h in range(32, 2049, 32)
                        if not host.layout(w, h)["tiled"])
    doc = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": {"hf_id": host.MODEL_ID, "revision": host.MODEL_SHA},
        "checks": "host layout / crops / prefix == vision.layout / preprocess / prefix_length; host audio lengths == "
                  "waveform -> MelFrontend -> ConvSubsampling (asserted per row)",
        "image": images,
        "image_notes": {
            "per_crop": "a crop of h x w px gives (h/16) x (w/16) patches and (h/32) x (w/32) prefix positions (<= 256)",
            "worst_untiled_prefix_scanned": worst_untiled,
            "worst_rows": {"untiled": f"{worst_untiled} + 896 = {worst_untiled + 896} positions",
                           "tiled": "10 tiles x 256 + thumbnail <= 256 + 896 = 3712 positions"},
        },
        "audio": audio,
        "audio_notes": {
            "lengths": "valid frames = samples // 160 (after the 30 s cut / 0.5 s pad); prefix = three times "
                       "(l - 1) // 2 + 1; the conformer runs over 1 + samples // 160 STFT columns subsampled the same way "
                       "and returns the first `prefix` steps",
            "per_second": "12.5 prefix positions per second (one per 80 ms)",
        },
        "bucket_proposal": {
            "image": "decision L 1024 holds an untiled image row only while text <= 1024 - P (P <= 256); the worst untiled "
                     f"row is {worst_untiled + 896} positions (L 2048) and a tiled one up to 3712 (L 4096): image rows "
                     "need L 512 / 2048 / 4096 as well as 256 (fixture: see encode_rows.json native_buckets_by_mode)",
            "audio": "mel buckets F = 1 + 100 s for s = 5 / 10 / 20 / 30 s: F 501 / 1001 / 2001 / 3001 -> encoder steps "
                     "63 / 126 / 251 / 376, full-clip prefix 63 / 125 / 250 / 375; decision rows = prefix + text",
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = WORK / "results" / "media_lengths.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print(f"{out} ({doc['elapsed_s']} s)")
    for r in images:
        print(f"  image {r['px']} ({r['origin']}): tiled {r['tiled']} grid {r['grid_wxh']} thumb {r['thumbnail_hxw']} "
              f"crops {r['crops']} prefix {r['prefix']} -> max {r['positions_max']} (L {r['bucket_at_text_max']})")
    for r in audio:
        print(f"  audio {r['origin']}: {r['seconds_in']} s in, stft {r['stft_frames']} valid {r['valid_frames']} "
              f"steps {r['encoder_steps']} prefix {r['prefix']}")
    print("  worst untiled prefix:", worst_untiled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
