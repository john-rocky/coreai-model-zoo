#!/usr/bin/env python3
"""Self-test for the decision rules: doctor's source lint and verify's verdict.

The negative half matters more than the positive half. A lint that flags the documented
WORKAROUND as if it were the bug is worse than no lint — it teaches people to ignore it.
The same goes for the gate: a verdict that calls an fp16 knife-edge tie a failure trains
people to override it, and then it catches nothing.

    python3 selftest.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_doctor as doctor  # noqa: E402
import coreai_eval as ev  # noqa: E402

# (rule id, line that must fire, line that must NOT fire)
CASES: list[tuple[str, str, str]] = [
    ("SRC-CAST-ROUNDTRIP",
     "y = (x + 64.0).long().float() - 64.0",
     "y = torch.div(x * 2.0, 2.0, rounding_mode='floor')"),
    ("SRC-FLOOR-ON-GPU",
     "y = torch.floor(x)",
     "y = torch.div(x * 2.0, 2.0, rounding_mode='floor')"),
    ("SRC-FLOORDIV-ONE",
     'y = torch.div(x, 1, rounding_mode="floor")',
     'y = torch.div(x * 2.0, 2.0, rounding_mode="floor")'),
    ("SRC-INT64-BOOL-MASK",
     "mask = ((ix0 >= 0) & (ix0 < W)).to(dtype)",
     "mask = 1 - (x - x.clamp(0, W)).abs().clamp(max=1)"),
    ("SRC-ARANGE-FLOAT",
     "t = torch.arange(8.0, dtype=x.dtype)",
     "t = torch.arange(8, dtype=x.dtype)"),
    ("SRC-FP16-DECOMP-OVERFLOW",
     "y = F.softplus(x)",
     "y = torch.clamp(x, min=0) + torch.log1p(torch.exp(-x.abs()))"),
    ("SRC-OPTIMIZE-AXIS-MOVE",
     "s2 = torch.sum(y ** 2, dim=-1).unsqueeze(-2)",
     "s2 = torch.sum(y ** 2, dim=-1).reshape(1, 1, -1)"),
    ("SRC-SQUEEZE-DIM",
     "x = x.squeeze(1)",
     "x = x.squeeze()"),
    ("SRC-COMPLEX-OPS",
     "f = torch.polar(mag, ang)",
     "f = torch.stack([cos, sin], -1)"),
    ("SRC-REMAINDER",
     "i = torch.remainder(pos, W)",
     "i = torch.where(pos >= W, pos - W, pos)"),
    ("SRC-F-NORMALIZE",
     "q = F.normalize(q, dim=-1)",
     "q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)"),
    ("SRC-TORCH-ASSERT",
     "torch._assert(n > 0, 'positive')",
     "assert isinstance(n, int)"),
    ("SRC-WHILE-LOOP",
     "out = torch.ops.higher_order.while_loop(cond, body, carry)",
     "out = step(carry)  # loop-free single step at S=1"),
    ("SRC-DATA-INDEXED-KV-WRITE",
     "cache = slice_update(cache, col, begin=in_step)",
     "cache = cache * (1 - write_mask) + col * write_mask"),
]


def findings_for(text: str) -> set[str]:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "m.py"
        f.write_text(text + "\n")
        rep = doctor.Report(target=str(f), kind="source")
        doctor.check_source_files([f], rep)
        return {x.rule.id for x in rep.findings}



def check_eval() -> int:
    """eval's refusals. A comparison tool that prints a delta it cannot justify is worse
    than no tool — it launders a configuration difference into a claim about the models,
    which is exactly how this project published a 12-point gap that was a token budget."""
    failures: list[str] = []

    def arm(**over):
        base = {"task": "gsm8k", "n": 10, "data_digest": "d0", "instruction_digest": "i0",
                "template_digest": "t0", "max_new_tokens": 2048, "temperature": 0.0,
                "stop": "eos", "arm": "x"}
        base.update(over)
        return base

    def side(correct, truncated=0, offformat=0, missing=0, **over):
        rows = [{"i": i, "ok": i < correct, "unmarked": i >= 10 - truncated,
                 "truncated": i >= 10 - truncated, "missing": False,
                 "pred": None, "gold": None} for i in range(10)]
        return {"arm": arm(**over),
                "score": {"n": 10, "correct": correct, "accuracy": correct / 10,
                          "unmarked": truncated + offformat,
                          "unmarked_rate": (truncated + offformat) / 10,
                          "truncated": truncated, "truncated_rate": truncated / 10,
                          "offformat": offformat, "missing": missing, "rows": rows}}

    cases = [
        ("matched protocol compares", side(8), side(9), 0),
        ("budget mismatch refuses", side(8), side(9, max_new_tokens=600), 3),
        ("template mismatch refuses", side(8), side(9, template_digest="t1"), 3),
        ("different questions refuse", side(8), side(9, data_digest="d1"), 3),
        ("both unrecorded still refuses",
         side(8, template_digest="unrecorded", stop="unrecorded"),
         side(9, template_digest="unrecorded", stop="unrecorded"), 3),
        ("a free field never blocks", side(8, arm="mac"), side(9, arm="iphone"), 0),
        # A run that crashed part way through scores its missing items wrong, so its
        # accuracy is a floor. Comparing against it reads a broken run as a worse model —
        # the misattribution this whole file exists to stop.
        ("an incomplete arm refuses", side(8), side(7, missing=3), 3),
    ]
    for label, a, b, expected in cases:
        code, _lines = ev.compare(a, b)
        if code != expected:
            failures.append(f"eval.compare — {label}: got {code}, expected {expected}")

    # Truncation is reported even when the protocol matches: equal budgets do not mean
    # equal room to answer, and the delta is partly measuring the difference.
    _code, lines = ev.compare(side(8), side(7, truncated=3))
    if not any("truncation differs" in x for x in lines):
        failures.append("eval.compare — a 30% truncation gap went unreported")
    _code, lines = ev.compare(side(8), side(9))
    if any("truncation differs" in x for x in lines):
        failures.append("eval.compare — warned about truncation when there was none")

    # An arm that finished and ignored the answer format must NOT be told to raise its
    # budget: it did not run out of one. Found by a real run answering in \boxed{} at a
    # third of its budget, and reported by this tool as a budget problem.
    _code, lines = ev.compare(side(8), side(5, offformat=5))
    if any("truncation differs" in x for x in lines):
        failures.append("eval.compare — called an off-format arm truncated")
    if not any("ignoring the requested format" in x for x in lines):
        failures.append("eval.compare — an off-format arm went unreported")

    # Scoring: formatting is not disagreement, and a missing marker is not a wrong answer.
    task = ev.BUILTIN_TASKS["gsm8k"]
    if not ev.score_one(task, "so\n#### 1,234", "x\n#### 1234")["ok"]:
        failures.append("eval.score_one — a thousands separator counted as wrong")
    if not ev.score_one(task, "so\n#### 18.0", "x\n#### 18")["ok"]:
        failures.append("eval.score_one — 18.0 counted as different from 18")
    if not ev.score_one(task, "first #### 9 then #### 18", "x\n#### 18")["ok"]:
        failures.append("eval.score_one — took the first restated number, not the last")
    truncated = ev.score_one(task, "let me work out the number of", "x\n#### 18")
    if truncated["ok"] or not truncated["unmarked"]:
        failures.append("eval.score_one — a truncated generation was not flagged unmarked")
    wrong = ev.score_one(task, "so\n#### 19", "x\n#### 18")
    if wrong["ok"] or wrong["unmarked"]:
        failures.append("eval.score_one — a plain wrong answer was called truncated")

    # Absence of a run is not truncation of one. Scoring them the same way makes a driver
    # that died look like a budget that was too small.
    absent = ev.score_one(task, None, "x\n#### 18")
    if not absent["missing"] or absent["unmarked"]:
        failures.append("eval.score_one — a missing generation was counted as truncated")

    # The real case: finished, wrong format, well under budget.
    boxed = ev.score_one(task, "the answer is \\boxed{0}", "x\n#### 18", capped=False)
    if boxed["truncated"] is not False or not boxed["unmarked"]:
        failures.append("eval.score_one — an off-format finish was called truncated")
    cut = ev.score_one(task, "we start by computing the", "x\n#### 18", capped=True)
    if cut["truncated"] is not True:
        failures.append("eval.score_one — a budget-capped generation was not marked")
    unknown = ev.score_one(task, "the answer is \\boxed{0}", "x\n#### 18")
    if unknown["truncated"] is not None:
        failures.append("eval.score_one — guessed truncation with no token count")

    for f in failures:
        print(f"FAIL {f}")
    if failures:
        raise SystemExit(1)
    return len(cases) + 13


def check_host_build() -> int:
    """The two asset rules Apple fixed in OS 27 beta 5 must be fatal only on a build that
    refuses the artifact. 'fatal' on a release build is false, and a lint that is false gets
    muted — which is worse than no lint."""
    failures: list[str] = []

    parse = [("26A5416b", (26, "A", 5416, True)), ("26A353", (26, "A", 353, False)),
             ("24A5408d", (24, "A", 5408, True)), ("26B31", (26, "B", 31, False)),
             ("garbage", None), ("", None), (None, None)]
    for build, want in parse:
        if doctor.parse_build(build) != want:
            failures.append(f"parse_build({build!r}) = {doctor.parse_build(build)}, want {want}")

    # 0.4.0-era IR: loads on beta 1; refused on every build measured since (26A5416b,
    # 2026-09-04) — Apple's beta 5 note did not change that. A release build inherits
    # nothing: it flips only once IR040_MEASURED_OK_FROM names a build a sweep measured.
    ir = [("26A5353q", "info"),   # beta 1 loads it
          ("26A5368g", "fatal"),  # beta 2: the break
          ("26A5378j", "fatal"),  # beta 3
          ("26A5406e", "fatal"),  # beta 5: the note said fixed; the load says no
          ("26A5416b", "fatal"),  # measured
          ("26A353", "fatal"),    # a release build: not measured, so not assumed
          ("26B31", "fatal"),     # 27.1: same
          ("27A5100a", "fatal"),  # a later major: same
          ("24A5355q", "info"),   # iOS beta 1
          ("24A5380h", "fatal"),  # iOS beta 3
          ("25A353", "fatal"),    # ambiguous major: unreadable premise, worst case
          (None, "fatal")]        # sw_vers unavailable: worst case
    for build, want in ir:
        got = doctor.ir040_severity(build)
        if got != want:
            failures.append(f"ir040_severity({build!r}) = {got}, want {want}")
    # ... and the flip, when a sweep measures it: everything at or past that build is info.
    saved = doctor.IR040_MEASURED_OK_FROM
    doctor.IR040_MEASURED_OK_FROM = {"26A": "26A353", "24A": "24A353"}
    try:
        for build, want in [("26A353", "info"), ("26A5416b", "fatal"), ("26B31", "info"),
                            ("26A5353q", "info"), ("24A353", "info"), ("24A5408d", "fatal")]:
            got = doctor.ir040_severity(build)
            if got != want:
                failures.append(f"ir040_severity({build!r}) after the flip = {got}, want {want}")
    finally:
        doctor.IR040_MEASURED_OK_FROM = saved

    # Pre-beta-3 AOT (181264112) is the break; merely older than the installed toolchain is
    # info; same or newer is nothing.
    aotc = [("3600.70.1", "3600.82.1", "fatal"), ("3600.70.1", None, "fatal"),
            ("3600.75.3", "3600.82.1", "info"), ("3600.75.3", None, None),
            ("3600.82.1", "3600.82.1", None), ("3600.90.0", "3600.82.1", None)]
    for producer, installed, want in aotc:
        got = doctor.aotc_severity(producer, installed)
        if got != want:
            failures.append(f"aotc_severity({producer}, {installed!r}) = {got}, want {want}")

    # The effective severity is what the report sorts and exits on.
    rep = doctor.Report(target="x", kind="asset", host_build="26A5353q")
    rep.add(doctor.IR_040, "x", "e", "f", severity=doctor.ir040_severity(rep.host_build))
    if rep.defects() or not rep.requirements():
        failures.append("a host-conditional info finding was counted as a defect")
    rep = doctor.Report(target="x", kind="asset", host_build="26A5378j")
    rep.add(doctor.IR_040, "x", "e", "f", severity=doctor.ir040_severity(rep.host_build))
    if not rep.defects():
        failures.append("a host-conditional fatal finding was not counted as a defect")

    for f in failures:
        print(f"FAIL {f}")
    if failures:
        raise SystemExit(1)
    return len(parse) + len(ir) + 6 + len(aotc) + 2


def check_rule_urls(failures: list[str]) -> int:
    """Every rule's `see` URL must resolve against this checkout: the page exists and, when
    the URL carries an anchor, a heading on that page slugs to it the way the site does.
    A finding that points at a dead anchor is a finding nobody can follow up."""
    repo = Path(__file__).resolve().parents[1]
    checked = 0
    for r in doctor.RULES.values():
        checked += 1
        if r.url.startswith(doctor.SITE + "/knowledge/"):
            page, _, anchor = r.url[len(doctor.SITE) + len("/knowledge/"):].partition("#")
            md = repo / "knowledge" / (page.removesuffix(".html") + ".md")
            if not md.exists():
                failures.append(f"{r.id}: url points at a page that is not in the checkout: {md.name}")
                continue
            if anchor:
                headings = [line.lstrip("#").strip() for line in md.read_text().splitlines()
                            if line.startswith("#")]
                if anchor not in {doctor.slug(h) for h in headings}:
                    failures.append(f"{r.id}: no heading in {md.name} slugs to #{anchor}")
        elif r.url.startswith(doctor.REPO_URL + "/blob/main/"):
            rel = r.url[len(doctor.REPO_URL) + len("/blob/main/"):]
            if not (repo / rel).exists():
                failures.append(f"{r.id}: url points at a file that is not in the checkout: {rel}")
        elif not r.url.startswith("https://github.com/apple/"):
            failures.append(f"{r.id}: url is neither a knowledge page, a repo file, nor an Apple issue: {r.url}")
    # The slug must be what the site produces, or the anchors above are validated against
    # the wrong thing. These three were read off the live site on 2026-09-07.
    live = [("Critical ordering — register kernels BEFORE add_exported_program",
             "critical-ordering--register-kernels-before-add_exported_program"),
            ("Re-verifying a recovered bundle: `conversion/coreai_gate.py`",
             "re-verifying-a-recovered-bundle-conversioncoreai_gatepy"),
            ("The chunk-threshold dial (`COREAI_CHUNK_THRESHOLD` / `llm-runner --chunk-size`)",
             "the-chunk-threshold-dial-coreai_chunk_threshold--llm-runner---chunk-size")]
    for heading, want in live:
        checked += 1
        if doctor.slug(heading) != want:
            failures.append(f"slug({heading!r}) = {doctor.slug(heading)!r}, the site has {want!r}")
    return checked


def main() -> int:
    failures: list[str] = []
    for rule_id, trigger, workaround in CASES:
        if rule_id not in findings_for(trigger):
            failures.append(f"{rule_id}: MISSED its trigger  -> {trigger}")
        if rule_id in findings_for(workaround):
            failures.append(f"{rule_id}: FIRED on the fix    -> {workaround}")

    # Counting rules: one write site is normal, several on one handle is the bug.
    if "SRC-CHAINED-STATE-WRITES" in findings_for("s = update_states(s, new)"):
        failures.append("SRC-CHAINED-STATE-WRITES: fired on a single write site")
    many = "\n".join(f"s{i} = update_states(s{i}, new{i})" for i in range(3))
    if "SRC-CHAINED-STATE-WRITES" not in findings_for(many):
        failures.append("SRC-CHAINED-STATE-WRITES: missed three write sites")

    # Absence rule: slice_update without remove_functionalization.
    if "SRC-MISSING-DEFUNCTIONALIZE" not in findings_for("ep = slice_update(ep, v)"):
        failures.append("SRC-MISSING-DEFUNCTIONALIZE: missed a bare slice_update")
    paired = "ep = slice_update(ep, v)\nep = remove_functionalization(ep)"
    if "SRC-MISSING-DEFUNCTIONALIZE" in findings_for(paired):
        failures.append("SRC-MISSING-DEFUNCTIONALIZE: fired despite remove_functionalization")

    # Comments are documentation, not code.
    if findings_for("# y = torch.floor(x)  -- do not do this"):
        failures.append("a commented-out line was reported")

    covered = {r.id for r, _p, _f in doctor.SRC_RULES}
    tested = {c[0] for c in CASES} | {"SRC-CHAINED-STATE-WRITES", "SRC-MISSING-DEFUNCTIONALIZE"}
    if untested := covered - tested:
        failures.append(f"source rules with no self-test: {', '.join(sorted(untested))}")

    n_verify = check_verify(failures)
    n_eval = check_eval()
    n_host = check_host_build()
    n_urls = check_rule_urls(failures)

    for line in failures:
        print("FAIL  " + line)
    print(f"\n{len(CASES) + 4} checks over {len(covered)} source rules, "
          f"{n_verify} over verify's verdict, {n_eval} over eval's refusals, "
          f"{n_host} over the host-build rules, {n_urls} over the rules' URLs: "
          f"{'FAILED' if failures else 'all pass'}")
    return 1 if failures else 0


def oracle(texts: list[str], margins: list[float]) -> dict:
    """A synthetic oracle result: cumulative decodes plus the per-step top-2 margins."""
    prefixes, acc = [], ""
    for t in texts:
        acc += t
        prefixes.append(acc)
    return {"gen_text": acc, "step_prefixes": prefixes, "margins": margins,
            "gen_ids": list(range(len(texts)))}


def check_verify(failures: list[str]) -> int:
    """The verdict rule, on both sides of the margin floor.

    A gate that cannot distinguish 'the conversion is wrong' from 'fp16 broke a tie' is
    not a gate — it either blocks good bundles or waves through broken ones.
    """
    import coreai_verify as verify

    confident = [0.9] * 4
    knife_edge = [0.9, 0.9, 0.004, 0.9]
    cases = [
        # (label, oracle, bundle text, floor, expected verdict)
        ("identical", oracle([" A", " B", " C", " D"], confident), " A B C D", 0.1, "PASS"),
        ("real divergence, high margin", oracle([" A", " B", " C", " D"], confident),
         " A B X D", 0.1, "FAIL"),
        ("knife-edge tie below the floor", oracle([" A", " B", " C", " D"], knife_edge),
         " A B X D", 0.1, "PASS"),
        ("same tie, floor lowered under it", oracle([" A", " B", " C", " D"], knife_edge),
         " A B X D", 0.001, "FAIL"),
        ("diverges at the very first token", oracle([" A", " B"], confident), " X B", 0.1, "FAIL"),
        ("bundle stopped early", oracle([" A", " B", " C"], confident), " A B", 0.1, "PASS"),
    ]
    for label, orc, got, floor, expected in cases:
        result, _line = verify.judge(orc, got, floor)
        if result != expected:
            failures.append(f"verify.judge — {label}: got {result}, expected {expected}")

    # Prompt validation: a position under the floor makes the prompt unusable, in either
    # direction. This is computable from the oracle alone, before any bundle exists.
    if verify.validate_prompt(oracle([" A", " B"], [0.9, 0.9]), 0.1):
        failures.append("verify.validate_prompt rejected a clean prompt")
    weak = verify.validate_prompt(oracle([" A", " B"], [0.9, 0.012]), 0.1)
    if len(weak) != 1:
        failures.append(f"verify.validate_prompt missed a 0.012-margin tie (got {weak})")
    return len(cases) + 2


if __name__ == "__main__":
    raise SystemExit(main())
