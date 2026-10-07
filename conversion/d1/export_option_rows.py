#!/usr/bin/env python3
"""The option-row table: the tied embedding rows of every token a d1 readout can read, for the host's gather.

d1's readout (`host.py` section 5) needs z[id] = h_slot . E[id] for the ids of a question's readout groups, E = the tied
embedding `model.language_model.embed_tokens.weight` ([128000, 2048], bf16 in the checkpoint). The graph has no
vocabulary head, so a host keeps the rows a readout can ever read: the ids of every string the provider's prompt can
turn into a readout id (option codes, their " " + code forms, the noul words, the score digits) that the tokenizer
encodes as one token.

    candidate_strings()      every code family a choice can print — A..Z, a..z, 0..9, 00..99, 100..999, #0..#199,
                             AA..ZZ (option_codes and the provider's prompt._FALLBACK_POOL) — the " " + code form of
                             each, and the noul words yes Yes YES no No NO
    candidate_ids(tok)       the ids of the single-token ones, deduplicated, ascending
    write_table(E_rows_fn, ids, out_dir)
                             -> <out_dir>/option_rows.safetensors (`rows` [n, d] fp32, `ids` [n] int32, ascending)
                                + <out_dir>/option_rows.json (id -> strings, n, sha256, the source); the file is read
                                back and must equal the rows written bit for bit
    read_table(dir)          -> {id: row} (a bundle directory or its head/)

The checkpoint's rows (`safetensors_rows`): `safetensors.safe_open(framework="pt")` reads only the requested rows
(`get_slice`, one row at a time, the table never loaded whole) and widens bf16 -> fp32, which is exact. A second path
(`memmap_rows`: the raw bf16 bytes through a NumPy memmap, `u16 << 16` as fp32) must give the same rows bit for bit.
In a bundle the table sits at `head/`; `export_decoder.py` writes it (the toy bundle from the toy embedding). A host
reads no id outside it: `host.build_request(..., table_ids=)` refuses such a request whole (host.py section 3).

    cd conversion/d1
    $PY export_option_rows.py --check        # render_ids.json: every readout group id is in the candidate set
    $PY export_option_rows.py --self-test    # a small bf16 safetensors through both readers, write -> read
    $PY export_option_rows.py --write <dir>  # round 3: the checkpoint's rows (needs model.safetensors)

-> `--check` writes $ZOO_WORK_ROOT/_d1_3b/results/option_rows_check.json (`--out` to change). Only `tokenizers`,
NumPy and safetensors are needed for `--check`; torch for the checkpoint reader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import host  # noqa: E402
from _paths import hf_snapshot, work_path  # noqa: E402

LANE = work_path("_d1_3b")
MODEL = {"hf_id": "LiquidAI/d1-3B", "revision": "da1fe36a861f24690f27f622dca1d8688503d113"}
EMBED_KEY = "model.language_model.embed_tokens.weight"
SHAPE = (128000, 2048)
TABLE_FILE, TABLE_JSON = "option_rows.safetensors", "option_rows.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- the candidate set
CODE_FAMILIES = ("A..Z", "a..z", "0..9", "00..99", "100..999", "#0..#199", "AA..ZZ")


def code_family(name: str) -> list[str]:
    up = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    return {"A..Z": up, "a..z": [chr(c) for c in range(ord("a"), ord("z") + 1)], "0..9": [str(i) for i in range(10)],
            "00..99": [f"{i:02d}" for i in range(100)], "100..999": [str(i) for i in range(100, 1000)],
            "#0..#199": [f"#{i}" for i in range(200)], "AA..ZZ": [a + b for a in up for b in up]}[name]


def candidate_strings() -> list[tuple[str, str]]:
    """[(family, string)] in a fixed order: every code family, its " " + code form, the noul words."""
    out = [(fam, s) for fam in CODE_FAMILIES for s in code_family(fam)]
    out += [(f"' {fam}'", " " + s) for fam in CODE_FAMILIES for s in code_family(fam)]
    return out + [("noul", s) for s in list(host.YES_FORMS) + list(host.NO_FORMS)]


def candidate_table(tok) -> dict:
    """Every candidate string's encoding: {"strings": [{family, text, ids, single}], "ids": ascending single ids,
    "strings_of": {id: [texts]}, "by_family": {family: "single/total"}}."""
    rows, by_id, fam_count = [], {}, {}
    for fam, s in candidate_strings():
        ids = host.token_ids(tok, s)
        single = ids[0] if len(ids) == 1 else None
        rows.append({"family": fam, "text": s, "ids": ids, "single": single})
        n_single, n = fam_count.get(fam, (0, 0))
        fam_count[fam] = (n_single + (single is not None), n + 1)
        if single is not None:
            by_id.setdefault(single, []).append(s)
    return {"strings": rows, "ids": sorted(by_id), "strings_of": {str(i): by_id[i] for i in sorted(by_id)},
            "by_family": {f: f"{a}/{b}" for f, (a, b) in fam_count.items()}}


def candidate_ids(tok) -> list[int]:
    """The ids of the candidate strings that are one token, deduplicated, ascending."""
    return candidate_table(tok)["ids"]


# --------------------------------------------------------------------------- the checkpoint's rows
def header_entry(path: Path, key: str) -> tuple[dict, int]:
    """A safetensors file's header entry for `key` and the byte offset where the data section starts."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header[key], 8 + n


def safetensors_rows(path: Path, key: str = EMBED_KEY, shape: tuple[int, int] = SHAPE) -> Callable[[list[int]], np.ndarray]:
    """E_rows_fn over a checkpoint file: rows `ids` of `key` (bf16) as fp32, read one row at a time (get_slice)."""
    def rows(ids: list[int]) -> np.ndarray:
        import torch
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            sl = f.get_slice(key)
            if sl.get_dtype() != "BF16" or tuple(sl.get_shape()) != tuple(shape):
                raise SystemExit(f"{path}:{key} is {sl.get_dtype()} {sl.get_shape()}, not BF16 {list(shape)}")
            out = np.empty((len(ids), shape[1]), np.float32)
            for k, i in enumerate(ids):
                if not 0 <= int(i) < shape[0]:
                    raise SystemExit(f"id {i} outside the table's {shape[0]} rows")
                out[k] = sl[int(i):int(i) + 1].to(torch.float32).numpy()[0]
        return out
    return rows


def memmap_rows(path: Path, ids: list[int], key: str = EMBED_KEY, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """The second path: the raw bf16 bytes of `key` through a memmap, widened by `u16 << 16`."""
    info, start = header_entry(path, key)
    if info["dtype"] != "BF16" or tuple(info["shape"]) != tuple(shape):
        raise SystemExit(f"{path}:{key} header {info}")
    raw = np.memmap(path, dtype="<u2", mode="r", offset=start + info["data_offsets"][0], shape=tuple(shape))
    return (np.asarray(raw[np.asarray(ids, np.int64)]).astype(np.uint32) << 16).view(np.float32)


def checkpoint_file(snapshot: Path) -> Path:
    """The safetensors file holding EMBED_KEY (a single file, or the shard the index names)."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        return snapshot / json.loads(index.read_text())["weight_map"][EMBED_KEY]
    return snapshot / "model.safetensors"


# --------------------------------------------------------------------------- write / read
def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def write_table(E_rows_fn: Callable[[list[int]], np.ndarray], ids, out_dir: Path, *, strings: dict | None = None,
                source: dict | None = None) -> dict:
    """The rows E_rows_fn(ids) -> out_dir/option_rows.safetensors + option_rows.json; read back bit for bit.
    Never overwrites. -> the json record."""
    from safetensors.numpy import save_file

    ids = [int(i) for i in ids]
    if ids != sorted(set(ids)):
        raise SystemExit("option-row ids must be distinct and ascending")
    out = Path(out_dir)
    st, js = out / TABLE_FILE, out / TABLE_JSON
    if st.exists() or js.exists():
        raise SystemExit(f"{st} exists: the table is never overwritten (remove it first, on purpose)")
    rows = np.ascontiguousarray(E_rows_fn(ids), np.float32)
    if rows.ndim != 2 or rows.shape[0] != len(ids):
        raise SystemExit(f"E_rows_fn gave {rows.shape} for {len(ids)} ids")
    if not np.isfinite(rows).all():
        raise SystemExit("the option rows are not all finite")
    out.mkdir(parents=True, exist_ok=True)
    id_arr = np.asarray(ids, np.int32)
    save_file({"rows": rows, "ids": id_arr}, str(st),
              metadata={"what": "d1 option rows: the tied embedding rows of the readout candidate ids, fp32",
                        "source": json.dumps(source or {}, sort_keys=True)[:4000]})
    got_ids, got_rows = table_arrays(out)
    readback = {"ids_equal": bool(np.array_equal(got_ids, id_arr)),
                "rows_bit_equal": bool(got_rows.shape == rows.shape and np.array_equal(_bits(got_rows), _bits(rows)))}
    if not (readback["ids_equal"] and readback["rows_bit_equal"]):
        raise SystemExit(f"{st}: read-back differs from the rows written: {readback}")
    rec = {"schema": "d1-option-rows/1", "file": TABLE_FILE, "bytes": st.stat().st_size, "sha256": sha256_file(st),
           "n": len(ids), "hidden": int(rows.shape[1]), "dtype": "float32", "ids_dtype": "int32",
           "layout": "tensors `rows` [n, hidden] fp32 and `ids` [n] int32 (ascending); row k = E[ids[k]]",
           "use": "z[id] = h_slot . rows[k] in float64 for the ids of the question's groups (host.option_logits)",
           "ids": ids, "strings": {str(i): (strings or {}).get(str(i), (strings or {}).get(i)) for i in ids},
           "source": source, "readback": readback,
           "rows_absmax": float(np.abs(rows).max()), "rows_l2_min_max": [float(np.linalg.norm(rows, axis=1).min()),
                                                                         float(np.linalg.norm(rows, axis=1).max())],
           "written": datetime.now().astimezone().isoformat(timespec="seconds")}
    js.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return rec


def _table_dir(path: Path) -> Path:
    p = Path(path)
    return p / "head" if (p / "head" / TABLE_FILE).exists() else p


def table_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ids [n] int32, rows [n, d] fp32) of a table directory (or a bundle directory holding head/)."""
    from safetensors.numpy import load_file

    z = load_file(str(_table_dir(path) / TABLE_FILE))
    return z["ids"], z["rows"]


def read_table(path: Path) -> dict[int, np.ndarray]:
    """{id: row (fp32)} of a table directory (or a bundle directory holding head/)."""
    ids, rows = table_arrays(path)
    return {int(i): rows[k] for k, i in enumerate(ids)}


def rows_for(table: dict[int, np.ndarray], ids: list[int]) -> np.ndarray:
    """The table's rows for `ids` in that order ([n, d] fp32); an id outside the table stops the caller."""
    missing = [i for i in ids if int(i) not in table]
    if missing:
        raise SystemExit(f"ids {missing[:8]} are not in the option-row table (a readout outside the table)")
    return np.stack([table[int(i)] for i in ids])


# --------------------------------------------------------------------------- the checkpoint's table (round 3)
def write_checkpoint_table(out_dir: Path, snapshot: Path | None = None) -> dict:
    """The real table: candidate ids from the snapshot's tokenizer, rows from its checkpoint (bf16 -> fp32), the
    second reader's rows bit-equal."""
    snap = Path(snapshot) if snapshot else Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))
    tok = host.load_tokenizer(snap / "tokenizer.json")
    cand = candidate_table(tok)
    ck = checkpoint_file(snap)
    if not ck.exists():
        raise SystemExit(f"no checkpoint file {ck} (round 3: model.safetensors is not downloaded yet)")
    info, start = header_entry(ck, EMBED_KEY)
    rec = write_table(safetensors_rows(ck), cand["ids"], out_dir, strings=cand["strings_of"],
                      source={"hf_id": MODEL["hf_id"], "revision": MODEL["revision"], "file": ck.name,
                              "file_sha256": sha256_file(ck), "key": EMBED_KEY, "header": info, "data_start": start,
                              "conversion": "bf16 -> fp32 (exact)",
                              "tokenizer_json_sha256": sha256_file(snap / "tokenizer.json"),
                              "candidates": cand["by_family"]})
    _, rows = table_arrays(out_dir)
    second = memmap_rows(ck, cand["ids"])
    same = bool(np.array_equal(_bits(second), _bits(rows)))
    if not same:
        raise SystemExit("memmap_rows differs from the safetensors reader on the checkpoint")
    rec["second_reader_bit_equal"] = same
    (Path(out_dir) / TABLE_JSON).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
    return rec


# --------------------------------------------------------------------------- --check / --self-test
def check(render_ids: Path, snapshot: Path) -> dict:
    """Every readout group id of every fixture question (results/render_ids.json) is in the candidate set; the ids
    outside it are the contract's holes."""
    tok = host.load_tokenizer(snapshot / "tokenizer.json")
    cand = candidate_table(tok)
    have = set(cand["ids"])
    doc = json.loads(render_ids.read_text())
    holes, n_q, used = [], 0, set()
    for r in doc["records"]:
        for q in r["questions"]:
            n_q += 1
            for g in q["groups"]:
                for i in g:
                    used.add(i)
                    if i not in have:
                        holes.append({"record": r["id"], "question": q["name"], "id": i,
                                      "token": tok.id_to_token(i) if hasattr(tok, "id_to_token") else None})
    multi = [{"text": s["text"], "ids": s["ids"]} for s in cand["strings"] if s["single"] is None]
    ctok = CachedTok(tok)
    cap = alias_capacity(ctok)
    probe = alias_rule_probe(ctok, have, cap)
    controls = refusal_controls(ctok, cand["ids"], cap)
    return {"schema": "d1-option-rows-check/1",
            "what": "every readout group id of every fixture question is in the option-row candidate set",
            "render_ids": {"path": str(render_ids), "sha256": sha256_file(render_ids),
                           "fixtures_sha256": doc.get("fixtures_sha256")},
            "tokenizer": {"path": str(snapshot / "tokenizer.json"), "sha256": sha256_file(snapshot / "tokenizer.json")},
            "candidate_strings": len(cand["strings"]), "single_token_strings": sum(s["single"] is not None for s in cand["strings"]),
            "candidate_ids": len(cand["ids"]), "by_family": cand["by_family"],
            "strings_sharing_an_id": {i: v for i, v in cand["strings_of"].items() if len(v) > 1},
            "multi_token_strings": multi,
            "questions": n_q, "group_ids_used": len(used), "group_ids_used_list": sorted(used),
            "holes": holes, "refusal_controls": controls,
            "pass": not holes and all(c["as_expected"] for c in controls.values()),
            "alias_rule_probe": probe,
            "table_rows_fp32_bytes": len(cand["ids"]) * SHAPE[1] * 4,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds")}


PROBE_COUNTS = (*range(2, 31), 99, 100, 101, 110, 152, 153, 200, 300, 500, 999, 1000, 1001, 1026, 1027, 1053, 1100, 1500)
PROBE_NATIVE = (("ascii lower", list("abcd")), ("ascii upper", list("WXYZ")), ("latin-1", list("éßñø")),
                ("greek", list("αβγδ")), ("cyrillic", list("абвг")), ("cjk", list("日月火水")),
                ("hiragana", list("あいうえ")), ("fullwidth latin", list("ＡＢＣＤ")))


class CachedTok:
    """A tokenizers.Tokenizer whose encodings are memoized (the alias rule re-encodes its pool for every fallback)."""

    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def encode(self, text, add_special_tokens=False):
        key = (text, add_special_tokens)
        if key not in self.cache:
            self.cache[key] = self.tok.encode(text, add_special_tokens=add_special_tokens)
        return self.cache[key]

    def encode_batch(self, *a, **k):
        return self.tok.encode_batch(*a, **k)

    def token_to_id(self, t):
        return self.tok.token_to_id(t)

    def id_to_token(self, i):
        return self.tok.id_to_token(i)


def alias_rule_probe(tok, have: set, capacity: dict) -> dict:
    """The host's alias rule (host.aliases + host.readout_groups) on requests the fixture does not hold: choice
    questions of n positional labels and of native one-letter labels (Python's isalpha takes any script). -> per case
    the readout ids outside the candidate set (a host reading them has no row)."""
    def outside(labels: list[str]) -> dict:
        q = {"type": "choice", "instructions": "x", "criteria": {lab: None for lab in labels}}
        try:
            groups = host.readout_groups(tok, q)
        except ValueError as e:
            return {"error": str(e)}
        ids = host.group_ids(groups)
        bad = [i for i in ids if i not in have]
        return {"ids": len(ids), "outside": len(bad),
                "outside_examples": [{"id": i, "token": tok.id_to_token(i)} for i in bad[:6]]}
    counts = {str(n): outside([f"opt{i}" for i in range(n)]) for n in PROBE_COUNTS}
    native = {name: {"labels": labs, **outside(labs)} for name, labs in PROBE_NATIVE}
    first_bad = next((int(n) for n, v in counts.items() if v.get("outside")), None)
    return {"what": "host.readout_groups on choice questions the fixture does not hold, against the candidate set",
            "positional_counts": counts, "first_positional_count_outside": first_bad,
            "alias_capacity": capacity,
            "native_one_letter_labels": native,
            "native_scripts_outside": [k for k, v in native.items() if v.get("outside")]}


def alias_capacity(tok, lo: int = 2, hi: int = 4000) -> dict:
    """The largest positional choice the alias rule can code (host.aliases raises one option later)."""
    def ok(n: int) -> bool:
        try:
            host.aliases(tok, [f"opt{i}" for i in range(n)])
            return True
        except ValueError:
            return False
    if ok(hi):
        return {"max_options": None, "note": f"no refusal up to {hi}"}
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return {"max_options": lo, "first_refused": hi}


REFUSAL_CONTROLS = (("native_hiragana", "refuse_table", {"あ": "first", "い": "second"}),
                    ("choice_101", "accept", {f"opt{i}": f"option {i}" for i in range(101)}))


def refusal_controls(tok, table_ids: list[int], cap: dict) -> dict:
    """host.build_request(..., table_ids=) on the controls: a native-label choice outside the table is refused with the
    table's message, a 101-option choice (codes 00..100) is accepted, and the first count past the alias pool is
    refused by the alias rule."""
    def run(questions: dict) -> str:
        try:
            host.build_request({"state": "s", "questions": {"q": questions}}, tok, table_ids=table_ids)
            return "accept"
        except ValueError as e:
            return f"refuse: {e}"
    out = {}
    for name, want, crit in REFUSAL_CONTROLS:
        got = run({"type": "choice", "instructions": "Which?", "criteria": crit})
        ok = (got == "accept") if want == "accept" else ("is not in the option table" in got)
        out[name] = {"expected": want, "host": got, "as_expected": ok}
    if cap.get("first_refused"):
        n = cap["first_refused"]
        got = run({"type": "choice", "instructions": "Which?", "criteria": {f"opt{i}": None for i in range(n)}})
        out[f"choice_{n}"] = {"expected": "refuse_alias_rule", "host": got[:160],
                              "as_expected": got.startswith("refuse") and "alias" in got}
    return out


def self_test() -> dict:
    """A small bf16 safetensors with the real key: safetensors_rows == memmap_rows == torch's bf16 -> fp32, bit for
    bit; write_table -> read_table bit for bit; a second write into the same directory refused."""
    import torch
    from safetensors.torch import save_file

    shape = (300, 24)
    g = torch.Generator().manual_seed(7)
    w = (torch.randn(shape, generator=g) * 0.05).to(torch.bfloat16)
    w[5, 3] = 1e-40   # a bf16 subnormal: the widening must keep it
    ids = [0, 5, 41, 73, 299]
    with tempfile.TemporaryDirectory(prefix="d1_option_rows_") as td:
        ck = Path(td) / "model.safetensors"
        save_file({EMBED_KEY: w, "other.weight": torch.zeros(2, 2, dtype=torch.bfloat16)}, str(ck))
        a = safetensors_rows(ck, shape=shape)(ids)
        b = memmap_rows(ck, ids, shape=shape)
        c = w[torch.tensor(ids)].to(torch.float32).numpy()
        rec = write_table(safetensors_rows(ck, shape=shape), ids, Path(td) / "head",
                          source={"self_test": True, "shape": list(shape)})
        back = read_table(Path(td))
        try:
            write_table(safetensors_rows(ck, shape=shape), ids, Path(td) / "head")
            refused = False
        except SystemExit:
            refused = True
        out = {"safetensors_vs_torch_bit_equal": bool(np.array_equal(_bits(a), _bits(c))),
               "memmap_vs_torch_bit_equal": bool(np.array_equal(_bits(b), _bits(c))),
               "subnormal_kept": bool(a[1, 3] != 0.0),
               "write_read_bit_equal": bool(all(np.array_equal(_bits(back[i]), _bits(c[k])) for k, i in enumerate(ids))),
               "table_sha256": rec["sha256"], "second_write_refused": refused}
    out["pass"] = all(v for k, v in out.items() if k != "table_sha256")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="render_ids.json's readout ids against the candidate set")
    ap.add_argument("--self-test", action="store_true", help="both checkpoint readers and write -> read on a small file")
    ap.add_argument("--write", metavar="DIR", help="round 3: write the checkpoint's table into DIR (a bundle's head/)")
    ap.add_argument("--render-ids", default=str(LANE / "results" / "render_ids.json"))
    ap.add_argument("--snapshot", default=None, help="the d1-3B snapshot directory (default: the pinned revision)")
    ap.add_argument("--out", default=None, help="--check / --self-test record (default results/option_rows_check.json)")
    args = ap.parse_args()
    if not (args.check or args.self_test or args.write):
        ap.error("one of --check, --self-test, --write")
    snap = Path(args.snapshot) if args.snapshot else Path(hf_snapshot(MODEL["hf_id"], revision=MODEL["revision"]))
    rec: dict = {}
    if args.check:
        rec["check"] = check(Path(args.render_ids), snap)
        c = rec["check"]
        print(f"candidates: {c['single_token_strings']}/{c['candidate_strings']} strings single -> {c['candidate_ids']} ids "
              f"{json.dumps(c['by_family'])}; fixture group ids used {c['group_ids_used']} over {c['questions']} "
              f"questions; holes {len(c['holes'])} -> {'PASS' if c['pass'] else 'FAIL'}")
    if args.self_test:
        rec["self_test"] = self_test()
        print("self-test:", json.dumps(rec["self_test"]))
    if args.write:
        rec["write"] = write_checkpoint_table(Path(args.write), snap)
        print(f"wrote {args.write}: {rec['write']['n']} rows, sha256 {rec['write']['sha256']}")
    if args.check or args.self_test:
        out = Path(args.out) if args.out else LANE / "results" / "option_rows_check.json"
        if out.exists():
            raise SystemExit(f"{out} exists: records are never overwritten")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {out}")
    ok = all(v.get("pass", True) for v in rec.values() if isinstance(v, dict))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
