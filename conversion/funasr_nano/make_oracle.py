#!/usr/bin/env python3
# Community port — NOT an Apple model.
"""Fun-ASR-Nano fp32 oracle: the official funasr pipeline on every fixture clip, tensors saved.

For each clip in ``fixtures/meta.json`` this runs, on CPU in fp32 with greedy decoding:

1. the official path — ``AutoModel.generate(input=[wav], language=None, itn=True, hotwords=[],
   llm_kwargs={"do_sample": False, "num_beams": 1})`` -> ``text_official``;
2. the same pipeline by hand — ``FunASRNano.inference_prepare`` -> ``llm.generate(inputs_embeds,
   attention_mask, ...)`` -> ``batch_decode`` -> the same whitespace clean-up ``inference_llm``
   applies -> ``text_manual`` (the 5 model-repo examples must give ``text_official == text_manual``);
3. ``oracle/<name>.npz``: ``speech [L,560]`` (fbank+LFR, captured before the encoder scales it in
   place), ``speech_len``, ``encoder_out [L,512]``, ``adaptor_out [L,1024]``, ``fake_token_len``,
   ``fbank_beg``, ``source_ids [Sp]`` (audio placeholders are id 0), ``gen_ids`` (EOS included),
   ``inputs_embeds [Sp,1024]``.

Pinned behaviour (all checked at run time and written to ``oracle.json``):
- dither 0.0 (funasr's default 1.0 adds random noise), passed as ``frontend_conf`` at construction —
  setting ``frontend.dither`` afterwards does not reach inference — and asserted on every call;
- the LLM is built in bf16 (``llm_conf.llm_dtype``) and ``inference_llm`` casts it to fp32 on the
  first call — after ``inference_prepare`` has already built that call's ``inputs_embeds`` in bf16.
  A cold call on the first example is recorded separately (``cold_call``); every clip below runs
  with the fp32 LLM, which is what every later official call uses.

Run with the private oracle venv (funasr 1.4.16, CPU torch):
    ~/code/coreai/_funasr_nano/venv-oracle/bin/python conversion/funasr_nano/make_oracle.py

A full run also writes ``<--shared-dir>/oracle_transcripts.json`` (one transcript per clip); it needs
``--shared-dir`` or ``--no-shared-copy``.

Prompt variants (``get_prompt`` options: hotwords, language, itn) run one clip under a case name and
write ``oracle_hotwords/<case>.{npz,json}`` instead — ``oracle/`` and the shared copy are not touched:
    ... make_oracle.py --case zh_hotwords --only zh --hotwords 开放时间
    ... make_oracle.py --case zh_language --only zh --language 中文
    ... make_oracle.py --case yue_no_itn --only yue --no-itn
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import record_path, work_path  # noqa: E402

WORK = work_path("_funasr_nano")
OFFICIAL = WORK / "official"
FIXTURES = WORK / "fixtures"
ORACLE = WORK / "oracle"
VARIANTS = WORK / "oracle_hotwords"
EXAMPLES = ["zh", "en", "ja", "ko", "yue"]
VLLM_EXPECTED_ZH = "开饭时间早上九点至下午五点。"   # Fun-ASR-Nano-2512-vllm MODEL_PROVENANCE.json
RUNTIME = dict(cache={}, batch_size=1, language=None, itn=True, hotwords=[],
               llm_kwargs={"do_sample": False, "num_beams": 1})
NOTE = ("oracle = fp32 steady state (2nd and later calls). The first call after the model is built "
        "embeds the prompt with the bf16 LLM, so that call's inputs_embeds (audio rows included) are "
        "rounded to bf16; inference_llm then casts the LLM to fp32 for good. cold_call keeps one clip's "
        "first-call text.")


def clean(response: str) -> str:
    """``FunASRNano.inference_llm``'s ``text`` field."""
    return re.sub(r"\s+", " ", response.replace("/sil", " "))


class DitherCheck:
    """Fail closed if any feature extraction runs with dither != 0 (it would add random noise)."""

    def __init__(self) -> None:
        from funasr.frontends.wav_frontend import WavFrontend

        self.calls = 0
        forward = WavFrontend.forward
        check = self

        def checked(frontend, *args, **kwargs):
            if frontend.dither != 0.0:
                raise RuntimeError(f"WavFrontend.forward ran with dither={frontend.dither}")
            check.calls += 1
            return forward(frontend, *args, **kwargs)

        WavFrontend.forward = checked


class SpeechTap:
    """Clone the encoder input: ``SenseVoiceEncoderSmall.forward`` scales ``xs_pad`` in place."""

    def __init__(self, encoder: torch.nn.Module) -> None:
        self.speech = None
        self.lengths = None
        encoder.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        self.speech = args[0].detach().clone()
        self.lengths = args[1].detach().clone()


def _copy(runtime: dict) -> dict:
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in runtime.items()}


def official(am, wav: str, runtime: dict = RUNTIME) -> tuple[str, float]:
    t0 = time.perf_counter()
    res = am.generate(input=[wav], **_copy(runtime))
    return res[0]["text"], time.perf_counter() - t0


@torch.no_grad()
def manual(am, wav: str, name: str, runtime: dict = RUNTIME) -> dict:
    model = am.model
    am._reset_runtime_configs()
    kwargs = am.kwargs
    kwargs.pop("cache", None)
    am._merge_runtime_config(kwargs, _copy(runtime))
    prompt = model.get_prompt(kwargs.get("hotwords", []), kwargs.get("language", None), kwargs.get("itn", True))
    data_in = [model.generate_chatml(prompt, wav)]
    inputs_embeds, contents, batch, source_ids, meta = model.inference_prepare(
        data_in, data_lengths=None, key=[name], **kwargs)
    model.llm = model.llm.to(torch.float32)
    inputs_embeds = inputs_embeds.to(torch.float32)
    attention_mask = batch.get("attention_mask", None)
    gen = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=kwargs.get("max_length", 512),
        pad_token_id=model.llm.config.pad_token_id or model.llm.config.eos_token_id,
        **kwargs.get("llm_kwargs", {}),
    )
    response = kwargs["tokenizer"].batch_decode(gen, skip_special_tokens=kwargs.get("skip_special_tokens", True))[0]
    return {
        "prompt": prompt,
        "text": clean(kwargs.get("prev_text", "") + response),
        "gen_ids": gen[0].cpu().numpy().astype(np.int64),
        "inputs_embeds": inputs_embeds[0].cpu().numpy().astype(np.float32),
        "source_ids": source_ids[0].cpu().numpy().astype(np.int64),
        "fake_token_len": int(batch["fake_token_len"][0, 0]),
        "fbank_beg": int(batch["fbank_beg"][0, 0]),
        "attn_len": int(attention_mask.shape[-1]),
        "encoder_out": meta["encoder_out"][0].cpu().numpy().astype(np.float32),
        "adaptor_out": meta["audio_adaptor_out"][0].cpu().numpy().astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="clip names to run (default: all in meta.json)")
    ap.add_argument("--ncpu", type=int, default=8)
    ap.add_argument("--shared-dir", type=Path,
                    help="folder that gets oracle_transcripts.json (a full run needs it or --no-shared-copy)")
    ap.add_argument("--no-shared-copy", action="store_true")
    ap.add_argument("--case", help="prompt-variant run: output name under oracle_hotwords/ (needs one --only clip)")
    ap.add_argument("--hotwords", nargs="+", default=[], help="get_prompt hotwords (prompt-variant run)")
    ap.add_argument("--language", help="get_prompt language, e.g. 中文 (prompt-variant run)")
    ap.add_argument("--no-itn", action="store_true", help="get_prompt itn=False (prompt-variant run)")
    args = ap.parse_args()
    variant = bool(args.case or args.hotwords or args.language or args.no_itn)
    runtime = RUNTIME | {"hotwords": list(args.hotwords), "language": args.language, "itn": not args.no_itn}
    if variant and not (args.case and args.only and len(args.only) == 1):
        raise SystemExit("a prompt-variant run needs --case and exactly one --only clip")
    if not args.no_shared_copy and not args.only and args.shared_dir is None:
        ap.error("the transcripts file (oracle_transcripts.json) needs --shared-dir DIR; --no-shared-copy skips it")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    import funasr
    import torchaudio
    import transformers
    from funasr import AutoModel

    versions = {"funasr": funasr.__version__, "torch": torch.__version__,
                "torchaudio": torchaudio.__version__, "transformers": transformers.__version__}
    meta = json.loads((FIXTURES / "meta.json").read_text())
    clips = [c for c in meta["clips"] if not args.only or c["name"] in args.only]
    (VARIANTS if variant else ORACLE).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    dither_seen = DitherCheck()
    # dither has to go in at construction: AutoModel deep-copies kwargs (frontend included) when it
    # is built and restores that copy before every call, so setting frontend.dither afterwards is
    # silently undone (funasr 1.4.16 _store_base_configs / _reset_runtime_configs).
    am = AutoModel(model=str(OFFICIAL), device="cpu", disable_update=True, hub="hf",
                   disable_pbar=True, disable_log=True, ncpu=args.ncpu, frontend_conf={"dither": 0.0})
    model = am.model
    # The frontend inference really uses is the construction-time copy that _reset_runtime_configs
    # restores; both must equal config.yaml's frontend_conf plus dither 0.0 (the partial
    # frontend_conf override is deep-merged, not a replacement).
    from omegaconf import OmegaConf

    keys = ("fs", "window", "n_mels", "frame_length", "frame_shift", "lfr_m", "lfr_n", "cmvn_file", "dither")
    expected = dict(OmegaConf.to_container(OmegaConf.load(OFFICIAL / "config.yaml").frontend_conf)) | {"dither": 0.0}
    am._reset_runtime_configs()
    frontend = am.kwargs["frontend"]
    effective = {k: getattr(frontend, k) for k in keys}
    assert frontend is am._base_kwargs_map["kwargs"]["frontend"]
    assert effective == {k: expected.get(k) for k in keys}, (effective, expected)
    info = {
        "dither": "AutoModel(frontend_conf={'dither': 0.0}); every WavFrontend.forward call asserted to run with 0.0",
        "model_dir": record_path(OFFICIAL.resolve()),
        "official_revision": "FunAudioLLM/Fun-ASR-Nano-2512@272c57b82523ada6fd87095e955f8e29100979ab",
        "frontend_effective": {k: record_path(v) if isinstance(v, str) else v for k, v in effective.items()}
                              | {"snip_edges": frontend.snip_edges,
                                           "upsacle_samples": frontend.upsacle_samples,
                                           "cmvn_loaded": frontend.cmvn is not None},
        "frontend_expected_config_yaml_plus_dither0": {k: expected.get(k) for k in keys},
        "frontend_matches_config": True,
        "runtime": {k: v for k, v in RUNTIME.items() if k != "cache"},
        "ncpu": args.ncpu,
        "llm_dtype_at_build": str(next(model.llm.parameters()).dtype),
        "ctc_decoder_active": model.ctc_decoder is not None,
        "generation_config": model.llm.generation_config.to_dict(),
    }
    print(json.dumps(info, ensure_ascii=False, default=str), flush=True)
    tap = SpeechTap(model.audio_encoder)

    # Cold call: the first official call still embeds the prompt with the bf16 LLM.
    first = next((c for c in clips if c["name"] == "zh"), clips[0])
    cold_text, cold_s = official(am, str(FIXTURES / first["path"]))
    info["cold_call"] = {"name": first["name"], "text": cold_text, "wall_s": round(cold_s, 3),
                         "llm_dtype_after": str(next(model.llm.parameters()).dtype)}
    print(f"[cold] {first['name']}: {cold_text!r} ({cold_s:.1f} s), llm now {info['cold_call']['llm_dtype_after']}",
          flush=True)

    rows = []
    for i, c in enumerate(clips):
        wav = str(FIXTURES / c["path"])
        text_official, wall = official(am, wav, runtime)
        m = manual(am, wav, c["name"], runtime)
        speech = tap.speech[0].cpu().numpy().astype(np.float32)
        L = int(tap.lengths[0])
        assert speech.shape == (L, 560) and m["encoder_out"].shape == (L, 512) and m["adaptor_out"].shape == (L, 1024)
        if variant:
            row = {"case": args.case, "name": c["name"], "hotwords": runtime["hotwords"],
                   "language": runtime["language"], "itn": runtime["itn"], "prompt": m["prompt"],
                   "text": text_official, "text_manual": m["text"], "manual_matches": text_official == m["text"],
                   "gen_ids": m["gen_ids"].tolist(), "source_ids": m["source_ids"].tolist(),
                   "N": m["fake_token_len"], "L": L, "Sp": int(m["source_ids"].shape[0]),
                   "fbank_beg": m["fbank_beg"], "wall_s": round(wall, 3), "versions": versions,
                   "note": NOTE + " Prompt variant: get_prompt(hotwords, language, itn) as listed."}
            np.savez(VARIANTS / f"{args.case}.npz", source_ids=m["source_ids"], gen_ids=m["gen_ids"],
                     fake_token_len=np.int64(m["fake_token_len"]), fbank_beg=np.int64(m["fbank_beg"]),
                     adaptor_out=m["adaptor_out"])
            (VARIANTS / f"{args.case}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1, default=str))
            print(f"[variant {args.case}] {c['name']}: prompt {m['prompt']!r} N={row['N']} Sp={row['Sp']} "
                  f"gen={len(row['gen_ids'])} match={row['manual_matches']}  {text_official!r}", flush=True)
            if not row["manual_matches"]:
                raise SystemExit(f"text_official != text_manual: {text_official!r} vs {m['text']!r}")
            return
        np.savez(ORACLE / f"{c['name']}.npz", speech=speech, speech_len=np.int64(L),
                 encoder_out=m["encoder_out"], adaptor_out=m["adaptor_out"],
                 fake_token_len=np.int64(m["fake_token_len"]), fbank_beg=np.int64(m["fbank_beg"]),
                 source_ids=m["source_ids"], gen_ids=m["gen_ids"], inputs_embeds=m["inputs_embeds"])
        row = {"name": c["name"], "text": text_official, "text_manual": m["text"],
               "manual_matches": text_official == m["text"], "gen_ids": m["gen_ids"].tolist(),
               "N": m["fake_token_len"], "L": L, "Sp": int(m["source_ids"].shape[0]),
               "fbank_beg": m["fbank_beg"], "attn_len": m["attn_len"], "wall_s": round(wall, 3)}
        rows.append(row)
        print(f"[{i + 1}/{len(clips)}] {c['name']}: L={L} N={row['N']} Sp={row['Sp']} "
              f"gen={len(row['gen_ids'])} match={row['manual_matches']} {wall:.1f}s  {text_official!r}", flush=True)
        if c["name"] in EXAMPLES and not row["manual_matches"]:
            raise SystemExit(f"text_official != text_manual on {c['name']}: {text_official!r} vs {m['text']!r}")

    info["frontend_calls_checked"] = dither_seen.calls
    zh = next((r for r in rows if r["name"] == "zh"), None)
    out = {"versions": versions,
           "note": NOTE,
           "info": info,
           "zh_vs_vllm": None if zh is None else {"oracle": zh["text"], "vllm_expected": VLLM_EXPECTED_ZH,
                                                  "equal": zh["text"] == VLLM_EXPECTED_ZH},
           "clips": rows}
    (ORACLE / "oracle.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print(f"[oracle] {len(rows)} clips -> {ORACLE}", flush=True)
    if not args.no_shared_copy and not args.only:
        path = write_shared(args.shared_dir.expanduser(), versions, info, rows)
        print(f"[oracle] transcripts -> {path}", flush=True)


def write_shared(shared_dir: Path, versions: dict, info: dict, rows: list) -> Path:
    """oracle_transcripts.json in shared_dir: the versions, the note, the setup and each clip's transcript."""
    shared_dir.mkdir(parents=True, exist_ok=True)
    path = shared_dir / "oracle_transcripts.json"
    path.write_text(json.dumps(
        {"versions": versions, "note": NOTE,
         "setup": {"official_revision": info["official_revision"], "runtime": info["runtime"],
                   "dither": 0.0, "device": "cpu", "dtype": "fp32 (steady state)"},
         "transcripts": {r["name"]: r["text"] for r in rows}}, ensure_ascii=False, indent=1))
    return path


if __name__ == "__main__":
    main()
