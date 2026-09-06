#!/usr/bin/env python3
"""coreai doctor — pre-flight lint for Core AI conversions.

    coreai_doctor.py <bundle-dir | *.aimodel | *.aimodelc | checkpoint-dir | hf-id | *.py>

The failure this exists for is not "the conversion errored". It is "the conversion
succeeded, the bundle loaded, and the output is quietly wrong" — the class that only a
device run or an oracle gate catches, hours later. Every rule encodes one incident that
already cost this project real time, and carries a citation back to the note or upstream
issue that recorded it.

Four things doctor can read, and what each is good for:

  ASSET       .aimodel / .aimodelc directory   IR provenance, AOT staleness
  GRAPH       via `xcrun coreai-build inspect` states, io shapes, op distribution
  BUNDLE      LanguageBundle directory         runtime contract, tokenizer, chat surface
  CHECKPOINT  HF checkpoint dir or repo id     quant recipe, eos, activation scales
  SOURCE      PyTorch modelling code           converter and delegate op traps

Findings are split into DEFECTS (something is wrong with the artifact) and REQUIREMENTS
(the artifact is fine and its host must do something specific, or it breaks). Only defects
affect the exit status: 2 for fatal/silent, 1 for runaway/perf, 0 otherwise.

  fatal    it will not convert, will not load, or will not execute
  silent   it converts, loads and runs — and the numbers are wrong
  runaway  it runs and produces output, but the app-level behaviour is broken
  perf     it works and is slower or larger than it needs to be
  requires the artifact is fine; whatever drives it must do this
  info     worth knowing before you ship

The full table, including rules this prototype does not yet implement, is DOCTOR_RULES.md
next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {
    "fatal": 0, "silent": 1, "runaway": 2, "perf": 3, "requires": 4, "info": 5,
}
DEFECT_SEVERITIES = ("fatal", "silent", "runaway", "perf")


@dataclass(frozen=True)
class Rule:
    id: str
    scope: str  # asset | graph | bundle | checkpoint | source
    severity: str
    title: str
    source: str  # where this rule is written down, as a citation
    url: str  # the same record as one fetchable URL, anchored to the section


RULES: dict[str, Rule] = {}


def rule(**kw) -> Rule:
    r = Rule(**kw)
    assert r.severity in SEVERITY_ORDER, r.severity
    assert r.url.startswith("https://"), r.id
    RULES[r.id] = r
    return r


# Every rule links to the page and section that recorded the incident, so a finding is one
# fetch away from its evidence. The published knowledge base is a Jekyll site; a section's
# anchor is what kramdown makes of the heading (measured against the live site 2026-09-07:
# everything but letters, digits, spaces, underscores and hyphens is dropped, leading
# non-letters are dropped, spaces become hyphens, lower-cased). `slug` reproduces that, and
# cli/selftest.py resolves every URL below against the checkout so a renamed heading fails
# the self-test instead of leaving a dead link in the report.

SITE = "https://john-rocky.github.io/coreai-model-zoo"
REPO_URL = "https://github.com/john-rocky/coreai-model-zoo"
ERROR_INDEX = "knowledge/coreai-error-index.md"


def slug(heading: str) -> str:
    """kramdown's auto_id for a heading, as the published site renders it."""
    s = re.sub(r"^[^a-zA-Z]+", "", heading)
    s = re.sub(r"[^a-zA-Z0-9 _-]", "", s)
    return s.replace(" ", "-").lower() or "section"


def note(page: str, heading: str | None = None) -> str:
    """URL of a knowledge note (`page` is the file name), anchored to `heading` if given."""
    url = f"{SITE}/knowledge/{page.removesuffix('.md')}.html"
    return f"{url}#{slug(heading)}" if heading else url


def err(exact_string: str) -> str:
    """URL of one entry in the error index: the H2 there is the exact error string."""
    return note(ERROR_INDEX.removeprefix("knowledge/"), exact_string)


# --- ASSET -----------------------------------------------------------------

IR_040 = rule(
    id="IR-040-DEBUG-LOC",
    url=err("LLVM ERROR: cannot unwrap empty odiec_module_t"),
    scope="asset", severity="fatal",
    title="asset carries coreai-torch 0.4.0-era IR (no producer stamp) — every OS 27 build from "
          "beta 2 on refuses its debug locations at load (measured through 26A5416b, 2026-09-04); "
          "only beta 1 loads it. Severity follows the host build; see --host-build",
    source="knowledge/coreai-torch-041-ir-incident.md; apple/coreai-torch#37, #44; "
           "RECOVERY_STATUS.md; models/_SMOKE.json (the measured builds)",
)

AOTC_STALE = rule(
    id="AOTC-STALE-TOOLCHAIN",
    url=note("coreai-torch-041-ir-incident.md", "Environment the fix needs"),
    scope="asset", severity="fatal",
    title="compiled .aimodelc comes from an Xcode 27 beta-2-or-earlier coreai-build (below "
          "3600.75.3) — OS 27 beta 3 and later fail to specialize it (Apple 181264112). An "
          "artifact merely older than the installed toolchain is info, not a defect",
    source="knowledge/coreai-torch-041-ir-incident.md 'Environment the fix needs' (Apple 181264112)",
)

AOTC_NO_SOURCE = rule(
    id="AOTC-NO-SOURCE-ASSET",
    url=note("coreai-torch-041-ir-incident.md", "UPDATE 2026-07-21 — the in-place fix: `strip_debug_info` (no re-conversion needed)"),
    scope="asset", severity="info",
    title="compiled-only artifact: its 0.4.0 provenance cannot be read from it, and it cannot be "
          "repaired in place — strip_debug_info works on .aimodel only",
    source="RECOVERY_STATUS.md 'NOT recoverable by strip'",
)

SYMLINKED_ASSET = rule(
    id="ASSET-SYMLINK",
    url=err("failedToSpecialize"),
    scope="asset", severity="fatal",
    title="asset path is a symlink — Swift AIModel(contentsOf:) does not follow symlinks and "
          "cannot resolve the per-arch delegates directory (failedToSpecialize)",
    source="GLM_IMAGE_KICKOFF.md 'Swift Phase 3' gotcha 2",
)

AOTC_LOAD_OPTIONS = rule(
    id="AOTC-LOAD-OPTIONS",
    url=err("failedToSpecialize"),
    scope="asset", severity="requires",
    title="an AOT .aimodelc must be loaded with SpecializationOptions.default() or .cpuOnly() — "
          "asking for preferredComputeUnitKind re-specializes the baked graph on device",
    source="GLM_IMAGE_KICKOFF.md 'Swift Phase 3' gotcha 1; "
           "knowledge/aot-and-specialization.md 'expectFrequentReshapes on a FIXED-shape graph'",
)

# --- GRAPH (from coreai-build inspect) --------------------------------------

GRAPH_STATE_COUNT = rule(
    id="GRAPH-STATE-COUNT",
    url=err('invalidOutputType("Expected 2 states (KV cache), got 4: [keyCache, valueCache, convState, recState]")'),
    scope="graph", severity="requires",
    title="the graph declares a number of states the stock sequential engine cannot drive — it "
          "hard-requires exactly 2 and fails engine-create with invalidOutputType",
    source="knowledge/pipelined-engine.md 'Driving hybrid bundles from an APP'; "
           "knowledge/stateful-kv-cache.md 'Hybrid / multi-state caches'",
)

GRAPH_S1_CONTRACT = rule(
    id="GRAPH-S1-RUN-CONTRACT",
    url=err("Shape at dimension 1 of 256 is not a valid substitution for source shape 1"),
    scope="graph", severity="requires",
    title="static S=1 decode graph: the engine's default warmup prefills 256 tokens and this "
          "graph rejects the substitution, and any multi-token prefill chunk is fatal",
    source="knowledge/pipelined-engine.md 'Run contract'; "
           "knowledge/coreai-torch-041-ir-incident.md 'Non-obvious things the gate encodes'",
)

GRAPH_VOCAB_MISMATCH = rule(
    id="GRAPH-VOCAB-MISMATCH",
    url=note("pipelined-engine.md", "What a model needs to ride the engine"),
    scope="graph", severity="silent",
    title="metadata language.vocab_size disagrees with the graph's logits width — the sampler is "
          "sized from metadata and indexes the real logits, so this drifts without an error",
    source="knowledge/pipelined-engine.md 'A full LanguageBundle directory'",
)

GRAPH_TOKENIZER_OVERFLOW = rule(
    id="GRAPH-TOKENIZER-OVERFLOW",
    url=note("gliner2-pii.md", "3. swift-transformers tokenizer routing — the make-or-break"),
    scope="graph", severity="info",
    title="the tokenizer can produce ids the head cannot emit (max token id >= logits width). "
          "Harmless when the overflow is a multimodal placeholder a text-only port never sees; "
          "a real bug when those markers carry meaning, because the host silently resolves "
          "them to UNK",
    source="knowledge/gliner2-pii.md 'GLiNER special markers live above the Unigram vocab'",
)

GRAPH_CUSTOM_METAL = rule(
    id="GRAPH-CUSTOM-METAL-KERNEL",
    url=note("custom-metal-kernels.md", "Constraints & gotchas"),
    scope="graph", severity="requires",
    title="the graph embeds custom Metal kernels — GPU-only by construction (the ANE runs fixed "
          "hardware ops and can never execute MSL)",
    source="knowledge/custom-metal-kernels.md; knowledge/compute-units-and-authoring.md",
)

GRAPH_FLOOR_IDENTITY = rule(
    id="GRAPH-FLOOR-IDENTITY",
    url=note("conversion-guide.md", "Detection transformers"),
    scope="graph", severity="silent",
    title="the graph contains floor / trunc / ceil, which execute as the IDENTITY function on "
          "the GPU delegate — the same asset is correct on CPU, so a cpu_only parity gate "
          "cannot see this",
    source="apple/coreai-torch#10; knowledge/conversion-guide.md 'Detection transformers' 3",
)

GRAPH_ROUND_TIES = rule(
    id="GRAPH-ROUND-TIES",
    url="https://github.com/apple/coreai-torch/issues/10",
    scope="graph", severity="info",
    title="the graph contains round, which uses ties-away-from-zero on the GPU delegate instead "
          "of ties-to-even — a 1-LSB divergence that matters most inside a quantize path",
    source="apple/coreai-torch#10",
)

GRAPH_DYNAMIC_KV_IOS = rule(
    id="GRAPH-DYNAMIC-KV-IOS-2048",
    url=note("undocumented-answers.md", "Does iOS behave differently from macOS for a dynamically-sized KV cache?"),
    scope="graph", severity="requires",
    title="growing (dynamic-seq) KV state plus a declared context of 2048 or more — the iOS "
          "on-device compiler miscompiles this graph class once the bound KV seq dim reaches "
          "2048, and output is corrupt from token 1 at full pipeline speed. The host engine "
          "must carry the capacity guard; the artifact itself is fine",
    source="apple/coreai-models#124; coreai-kit#5; coreai-models bd8dcf7 (the shipped engine "
           "guard); coreai-kit Sources/CoreAIKit/ModelRuntime.swift; "
           "knowledge/undocumented-answers.md; NANBEIGE42_PR6_STATE.md (the bisect)",
)

# --- BUNDLE ----------------------------------------------------------------

BUNDLE_INCOMPLETE = rule(
    id="BUNDLE-INCOMPLETE",
    url=note("pipelined-engine.md", "What a model needs to ride the engine"),
    scope="bundle", severity="fatal",
    title="LanguageBundle is missing a required key or file — a bare .aimodel directory is not "
          "loadable by LanguageBundle/EngineFactory",
    source="knowledge/pipelined-engine.md 'What a model needs to ride the engine'",
)

NOT_A_BUNDLE_MANIFEST = rule(
    id="BUNDLE-NOT-A-MANIFEST",
    url=note("pipelined-engine.md", "What a model needs to ride the engine"),
    scope="bundle", severity="info",
    title="metadata.json here is not a Core AI bundle manifest (no metadata_version, no assets) "
          "— it is some other sidecar that happens to share the name",
    source="knowledge/pipelined-engine.md 'What a model needs to ride the engine'",
)

BUNDLE_ASSET_MISSING = rule(
    id="BUNDLE-ASSET-MISSING",
    url=note("pipelined-engine.md", "What a model needs to ride the engine"),
    scope="bundle", severity="fatal",
    title="metadata.json assets.main does not resolve to anything on disk",
    source="knowledge/pipelined-engine.md 'What a model needs to ride the engine'",
)

NO_CHAT_TEMPLATE = rule(
    id="CHAT-TEMPLATE-MISSING",
    url=note("cross-runtime-quality-benchmarking.md", "Ops notes"),
    scope="bundle", severity="runaway",
    title="an instruction-tuned model with no chat template in the bundle — the runtime silently "
          "falls back to raw completion and the model never sees turn markers",
    source="knowledge/cross-runtime-quality-benchmarking.md 'Ops notes'",
)

NO_CHAT_TEMPLATE_INFO = rule(
    id="CHAT-TEMPLATE-ABSENT",
    url=note("cross-runtime-quality-benchmarking.md", "Ops notes"),
    scope="bundle", severity="info",
    title="no chat template in the bundle. Expected for a non-chat decoder (ASR, OCR, drafter); "
          "a defect if this model is meant to hold a conversation",
    source="knowledge/cross-runtime-quality-benchmarking.md 'Ops notes'",
)

EOS_NOT_IN_TEMPLATE = rule(
    id="EOS-NOT-EMITTED-BY-TEMPLATE",
    url=note("minicpm5-1b.md", "App integration (CoreAIChat — applies to any Think-mode model)"),
    scope="bundle", severity="runaway",
    title="the declared eos_token is never emitted by the chat template — generation runs to the "
          "token cap instead of stopping at the turn terminator",
    source="GEMMA4_12B_STATE.md 'THE ONE REAL FIX — chat-EOS'",
)

TOKENIZER_CLASS_UNKNOWN = rule(
    id="TOKENIZER-CLASS-UNREGISTERED",
    url=err("unsupportedTokenizer"),
    scope="bundle", severity="fatal",
    title="tokenizer_class is not in swift-transformers' registry — a strict load throws "
          "unsupportedTokenizer, and the non-strict fallback is BPE for everything, which is "
          "silently wrong for a SentencePiece/Unigram or WordPiece model",
    source="knowledge/ship-playbook.md 'Cross-cutting traps'; knowledge/gliner2-pii.md; "
           "swift-transformers Sources/Tokenizers/Tokenizer.swift knownTokenizers",
)

BIG_IOS_JIT = rule(
    id="IOS-LARGE-GRAPH-JIT",
    url=note("aot-and-specialization.md", "The 4B wall — large decoders MUST ship AOT, not as a portable IR"),
    scope="bundle", severity="requires",
    title="a GB-class graph's on-device specialization is the coin-flip step — some bundles of "
          "this size JIT fine on iPhone and some abort; measure it before you ship",
    source="knowledge/aot-and-specialization.md 'The 4B wall'; knowledge/ship-playbook.md; "
           "knowledge/pipelined-engine.md 'Run contract'; NANBEIGE42_PR6_STATE.md (the counter-case)",
)

IPHONE_MEMORY_ENTITLEMENT = rule(
    id="IPHONE-MEMORY-ENTITLEMENT",
    url=err("libc++abi: terminating due to uncaught exception of type std::bad_alloc: std::bad_alloc"),
    scope="bundle", severity="requires",
    title="at this size cold specialization hits the default jetsam limit and dies with "
          "std::bad_alloc — the app needs the increased-memory-limit entitlement",
    source="knowledge/pipelined-engine.md 'Run contract'",
)

# --- CHECKPOINT ------------------------------------------------------------

ACT_SCALE_QAT = rule(
    id="QAT-STATIC-ACTIVATION-SCALES",
    url=note("gemma4-wna8o8-requires-int8-activations.md"),
    scope="checkpoint", severity="silent",
    title="the checkpoint carries static activation / KV-cache scales — these weights were "
          "trained with a learned activation clamp in the loop, and exporting them into an "
          "fp16-activation graph loses about half the reasoning accuracy while every "
          "equivalence gate still passes",
    source="knowledge/gemma4-wna8o8-requires-int8-activations.md",
)

PREQUANTIZED_CHECKPOINT = rule(
    id="CHECKPOINT-PREQUANTIZED",
    url=note("cross-runtime-quality-benchmarking.md", "Bits are not a spec"),
    scope="checkpoint", severity="info",
    title="the checkpoint carries a quantization_config — its weights are already fitted to one "
          "runtime's grid, so re-quantizing it to 'int4' produces a different product, not a "
          "comparable one",
    source="knowledge/cross-runtime-quality-benchmarking.md 'Bits are not a spec'",
)

EOS_SOURCE_MISMATCH = rule(
    id="EOS-SOURCE-MISMATCH",
    url=note("minicpm5-1b.md", "App integration (CoreAIChat — applies to any Think-mode model)"),
    scope="checkpoint", severity="runaway",
    title="generation_config lists more eos ids than tokenizer_config's eos_token resolves to — "
          "pick the turn terminator and retag the exported bundle, or anything that stops on a "
          "single eosTokenId runs to the cap",
    source="GEMMA4_12B_STATE.md 'THE ONE REAL FIX — chat-EOS'; knowledge/minicpm5-1b.md 'Chat EOS'",
)

TIED_EMBEDDINGS = rule(
    id="TIED-EMBEDDINGS-SKIP-QUANT",
    url=note("pipelined-engine.md", "Quantization on the GPU delegate (measured, qwen3.5-0.8B, M4 Max p128/g256)"),
    scope="checkpoint", severity="perf",
    title="tie_word_embeddings is set — the eager quantizer silently skips tied weights, so the "
          "lm_head ships full precision; on a bandwidth-bound phone that head is a large share "
          "of the per-token read",
    source="knowledge/pipelined-engine.md 'Quantization on the GPU delegate'",
)

BLOCK_DIVISIBILITY = rule(
    id="QUANT-BLOCK-DIVISIBILITY",
    url=note("compression-reference.md", "Pitfalls"),
    scope="checkpoint", severity="silent",
    title="a weight dimension is not divisible by the intended quant block size — per-block "
          "quantization and per-grouped-channel palettization silently skip those layers and "
          "they ship uncompressed",
    source="knowledge/compression-reference.md 'Pitfalls'",
)

# --- ENV -------------------------------------------------------------------

ENV_CLONE_SHADOW = rule(
    id="ENV-CONVERTER-SHADOWED",
    url=note("coreai-torch-041-ir-incident.md", "Environment the fix needs"),
    scope="env", severity="silent",
    title="a coreai_torch source checkout in the working directory shadows the installed wheel "
          "through sys.path[0] — exports run on the checkout's version, whatever is pip-installed",
    source="knowledge/coreai-torch-041-ir-incident.md 'Environment the fix needs'",
)

ENV_OVERLAY_MISSING = rule(
    id="ENV-OVERLAY-MISSING",
    url=f"{REPO_URL}/blob/main/conversion/overlay/README.md",
    scope="env", severity="fatal",
    title="the interpreter has coreai_models without the zoo overlay applied — every recipe that "
          "re-authors a model imports overlay-only classes and dies on import",
    source="conversion/overlay/README.md; conversion/zoo_convert.py doctor",
)

# --- SOURCE (PyTorch modelling code) ---------------------------------------

SRC_RULES: list[tuple[Rule, re.Pattern, str]] = []
SRC_FIX: dict[str, str] = {}


def src_rule(pattern: str, fix: str, **kw) -> Rule:
    r = rule(scope="source", **kw)
    SRC_RULES.append((r, re.compile(pattern), fix))
    SRC_FIX[r.id] = fix
    return r


src_rule(
    id="SRC-CAST-ROUNDTRIP", severity="silent",
    url=note("conversion-guide.md", "Detection transformers"),
    title="float->int->float cast round-trip: the converter cancels the pair and drops the "
          "truncation, so the bounded-floor idiom compiles to the identity function — on every "
          "compute unit, CPU included",
    source="apple/coreai-torch#9; knowledge/conversion-guide.md 'Detection transformers' 3",
    pattern=r"\.(long|int)\(\s*\)\s*\.(float|half)\(|\.to\(\s*torch\.(int64|int32|long)\s*\)\s*\.to\(\s*torch\.(float|half)",
    fix="Use torch.div(x * 2.0, 2.0, rounding_mode='floor') — floor-div with a divisor != 1 "
        "lowers correctly on every unit and the x2/2 scale is exact in floating point.",
)

src_rule(
    id="SRC-FLOOR-ON-GPU", severity="silent",
    url=note("conversion-guide.md", "Detection transformers"),
    title="aten.floor / trunc / ceil execute as the identity function on the GPU delegate, and "
          "the same asset is correct on CPU",
    source="apple/coreai-torch#10; knowledge/conversion-guide.md 'Detection transformers' 3",
    pattern=r"torch\.(floor|trunc|ceil)\s*\(|\.(floor|trunc|ceil)\s*\(\s*\)",
    fix="The floor that survives every compute unit is torch.div(x * 2.0, 2.0, "
        "rounding_mode='floor') — floor-div with a divisor != 1 lowers correctly. Verify on "
        "GPU, not CPU. Check first whether the call is on the traced path at all: drop-path and "
        "other training-only branches match this pattern and never reach the graph.",
)

src_rule(
    id="SRC-FLOORDIV-ONE", severity="silent",
    url="https://github.com/apple/coreai-torch/issues/10",
    title="div(x, 1, rounding_mode='floor') is simplified to the identity at conversion time — "
          "the divisor-1 fold drops the rounding semantics. A divisor other than 1 is fine",
    source="apple/coreai-torch#10",
    pattern=r"div\s*\([^,()]+,\s*1(\.0*)?\s*,\s*rounding_mode\s*=\s*[\"']floor[\"']",
    fix="Only the divisor-1 form is affected. Rewrite it as "
        "torch.div(x * 2.0, 2.0, rounding_mode='floor').",
)

src_rule(
    id="SRC-INT64-BOOL-MASK", severity="silent",
    url=note("conversion-guide.md", "Detection transformers"),
    title="an int64-comparison -> bool -> float mask chain corrupts an unrelated, still-live "
          "tensor elsewhere in the graph (even a declared graph output); clone()/contiguous() "
          "barriers do not protect the victim and skipping optimize() does not help",
    source="apple/coreai-torch#11; knowledge/conversion-guide.md 'Detection transformers' 2",
    pattern=r"\(\s*[\w\.\[\]]+\s*[<>]=?\s*[\w\.\[\]\-]+\s*\)\s*[&|]\s*\(\s*[\w\.\[\]]+\s*[<>]=?",
    fix="Compute 0/1 masks in float arithmetic: 1 - (x - x.clamp(lo, hi)).abs().clamp(max=1) is "
        "exact on integer-valued floats. Diagnosis pattern: a tensor is provably computed right "
        "(another consumer sees exact values) but reads wrong later.",
)

src_rule(
    id="SRC-ARANGE-FLOAT", severity="fatal",
    url=err("bad_optional_access"),
    title="torch.arange with float start/end/step aborts the converter with a C++ "
          "bad_optional_access — no Python traceback, the process just dies",
    source="apple/coreai-torch#8; knowledge/conversion-guide.md 'Detection transformers' 1",
    pattern=r"\barange\s*\(\s*[^)\n]*?(\d\.\d|\bfloat\(|\w+\s*/\s*\d)",
    fix="Precompute the vector in Python and bake it as a constant — that also removes the "
        "runtime arange/floordiv/pow chain. DETR-family models hit this via "
        "gen_sineembed_for_position(..., d_model / 2).",
)

src_rule(
    id="SRC-FP16-DECOMP-OVERFLOW", severity="silent",
    url="https://github.com/apple/coreai-torch/issues/21",
    title="softplus / mish / logsumexp / logcumsumexp get PyTorch's naive decomposition, which "
          "overflows fp16 on the ANE — output collapses to 0, or to inf/NaN",
    source="apple/coreai-torch#21 (and #5); the same fixes landed in apple/coremltools#2725-2727",
    pattern=r"\b(softplus|mish|logsumexp|logcumsumexp)\b",
    fix="Use the stable forms: softplus(x) = max(x,0) + log1p(exp(-|x|)); logsumexp with a "
        "max-shift. Or preserve the op in the decomposition table instead of letting it decompose.",
)

src_rule(
    id="SRC-OPTIMIZE-AXIS-MOVE", severity="silent",
    url="https://github.com/apple/coreai-torch/issues/49",
    title="AIProgram.optimize() removes a broadcasting-significant axis move in the expanded "
          "squared-distance form, so ||y_i||^2 broadcasts where ||y_j||^2 belongs; for "
          "equal-length inputs the output shape is still right and nothing is emitted",
    source="apple/coreai-torch#49",
    pattern=r"sum\([^)\n]*\*\*\s*2[^)\n]*\)\s*\.unsqueeze\(\s*-2\s*\)|keepdim\s*=\s*True\s*\)\s*\.transpose\(\s*-1\s*,\s*-2\s*\)",
    fix="Compare the distance expression against eager torch with optimize() ON — an unoptimized "
        "arm passing proves nothing here.",
)

src_rule(
    id="SRC-SQUEEZE-DIM", severity="fatal",
    url=err("dimension to be shrunk must have size 1, got N"),
    title="squeeze(dim) is a no-op in torch when that dim != 1, but coreai-torch lowers it to a "
          "hard shrink and aborts",
    source="knowledge/conversion-guide.md 'Gotchas that cost real time'",
    pattern=r"\.squeeze\(\s*-?\d+\s*\)",
    fix="Guard it so the trace resolves it away: if x.shape[1] == 1: x = x.squeeze(1).",
)

src_rule(
    id="SRC-COMPLEX-OPS", severity="fatal",
    url=note("conversion-guide.md", "Gotchas that cost real time"),
    title="complex tensor ops (torch.polar, view_as_complex/view_as_real, complex multiply) do "
          "not lower — the usual source is a complex-valued RoPE",
    source="knowledge/conversion-guide.md 'Gotchas that cost real time'",
    pattern=r"torch\.(polar|view_as_complex|view_as_real)\b",
    fix="Rewrite RoPE as real cos/sin: rope returns stack([cos, sin], -1); apply is "
        "(x_re*cos - x_im*sin, x_re*sin + x_im*cos).",
)

src_rule(
    id="SRC-REMAINDER", severity="fatal",
    url=err("Unsupported ATen op: sym_max"),
    title="aten.remainder (tensor modulo) is unsupported and surfaces at add_exported_program "
          "validate time, not at runtime",
    source="knowledge/conversion-guide.md; knowledge/stateful-kv-cache.md 'Sliding-window ring buffer'",
    pattern=r"torch\.remainder\s*\(|\.remainder\s*\(",
    fix="Compute the modulo with a scalar symint plus where(), or with floor/where arithmetic "
        "(and on GPU use the div(x*2, 2, floor) form of floor).",
)

src_rule(
    id="SRC-F-NORMALIZE", severity="silent",
    url=note("conversion-guide.md", "Gotchas that cost real time"),
    title="F.normalize loses its eps denominator clamp, so near-zero-norm vectors blow up "
          "(~1e13); it is input-dependent, hides at small sequence lengths and surfaces at large",
    source="knowledge/conversion-guide.md 'Gotchas that cost real time'",
    pattern=r"\b(F|functional)\.normalize\s*\(",
    fix="Write the norm explicitly: x * rsqrt(mean(x**2) + eps) for RMS, "
        "x * rsqrt(sum(x**2) + eps) for L2.",
)

src_rule(
    id="SRC-TORCH-ASSERT", severity="fatal",
    url=note("conversion-guide.md", "Detection transformers"),
    title="torch._assert on a data-dependent comparison breaks torch.export non-strict "
          "(GuardOnDataDependentSymNode) — often added upstream FOR export compatibility",
    source="knowledge/conversion-guide.md 'Detection transformers' 4",
    pattern=r"torch\._assert\s*\(",
    fix="For static-shape exports the check is vacuous: no-op torch._assert around "
        "torch.export.export and restore it after.",
)

src_rule(
    id="SRC-WHILE-LOOP", severity="fatal",
    url=err("'scf.while' region type mismatch"),
    title="a recurrent scan (torch.ops.higher_order.while_loop) does not lower on the MPSGraph "
          "GPU delegate ('scf.while' region type mismatch), and on the macOS-27 beta the same "
          "bundle fails even cpu_only — so 'it verified on CPU' proves nothing",
    source="knowledge/pipelined-engine.md 'The export trick: decode-only, loop-free'",
    pattern=r"while_loop\s*\(",
    fix="Export at S=1 with a loop-free single-step recurrence (set the loop-free flag BEFORE "
        "quantization and tracing) — at S=1 a scan is one step, so it is numerically identical.",
)

src_rule(
    id="SRC-CHAINED-STATE-WRITES", severity="silent",
    url=note("pipelined-engine.md", "State & precision traps on the GPU delegate (found by the LFM2.5 port)"),
    title="more than one per-layer write to the same fixed-shape state handle: the GPU delegate "
          "drops all but one, so position 0 is fine (a fresh state IS zero) and everything after "
          "decodes garbage",
    source="knowledge/pipelined-engine.md 'State & precision traps'; "
           "knowledge/rwkv7-recurrent-linear-attention-coreai.md",
    pattern=r"update_states\s*\(",
    fix="Collect each layer's new state slice and issue ONE fused full-state slice_update per "
        "step. Reads stay per-layer narrows and the slots are disjoint, so semantics are "
        "identical. The growing KV pair is exempt — its written values are re-read in-graph.",
)

src_rule(
    id="SRC-DATA-INDEXED-KV-WRITE", severity="fatal",
    url=err("EXC_BREAKPOINT (SIGTRAP, code 5)"),
    title="a KV write position derived in-graph from runtime data (the in_step index) does not "
          "lower on the WWDC26 betas: SIGTRAP on Mac GPU, SIGSEGV on iPhone GPU, and on the ANE "
          "it corrupts the compile cache so the next load ENOENTs. Conversion succeeds",
    source="knowledge/coreai-beta-mpsgraph-kvwrite-bug.md; apple/coreai-models#5; FB23024751; "
           "apple/coreai-torch#6 for the ANECompiler variant",
    pattern=r"\bin_step\b",
    fix="Either derive the write position from a shape symint, or hand the graph a host-built "
        "one-hot write mask input and blend: sl*(1-m) + col*m — no data-derived index anywhere.",
)

src_rule(
    id="SRC-MISSING-DEFUNCTIONALIZE", severity="silent",
    url=note("conversion-guide.md", "Gotchas that cost real time"),
    title="in-place state writes without remove_functionalization(ep): the mutation is dropped at "
          "conversion and the state never updates",
    source="knowledge/conversion-guide.md 'Gotchas that cost real time'; knowledge/stateful-kv-cache.md",
    pattern=r"slice_update\s*\(",  # judged as an absence, after the scan
    fix="Call remove_functionalization(ep) after run_decompositions and before converting.",
)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    rule: Rule
    where: str
    evidence: str
    fix: str
    # The severity this finding carries HERE. Normally the rule's own; the two loader-side
    # incidents Apple fixed in OS 27 beta 5 report by the host build instead — fatal on a
    # build that refuses the artifact, info on one that loads it — because "fatal" on a
    # release build would be false, and a lint that is false gets muted.
    severity: str = ""

    def __post_init__(self) -> None:
        self.severity = self.severity or self.rule.severity
        assert self.severity in SEVERITY_ORDER, self.severity


@dataclass
class Report:
    target: str
    kind: str
    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    host_build: str | None = None  # OS build the asset rules judge against (default: this Mac)

    def add(self, r: Rule, where: str, evidence: str, fix: str, severity: str = "") -> None:
        self.findings.append(Finding(r, where, evidence, fix, severity))

    def ran(self, *rule_ids: str) -> None:
        self.checked.extend(rule_ids)

    def defects(self) -> list[Finding]:
        return [f for f in self.findings if f.severity in DEFECT_SEVERITIES]

    def requirements(self) -> list[Finding]:
        return [f for f in self.findings if f.severity not in DEFECT_SEVERITIES]


# ---------------------------------------------------------------------------
# Host OS build — the two asset rules above are true only on some OS 27 builds
# ---------------------------------------------------------------------------

BUILD = re.compile(r"^(\d+)([A-Z])(\d+)([a-z]?)$")

# OS 27 seed builds, recorded from device-attributed measurements in the zoo's cards and
# notes (macOS 26A… / iOS 24A…). Apple seeds carry a 4-digit build number starting with 5;
# the RC/GA build drops it (macOS 26 shipped as 25A353 after 25A5xxx seeds), so a release
# build sorts AFTER every seed.
#   beta 1  26A5353q / 24A5355q   loads 0.4.0-era IR
#   beta 2  26A5368g               first build to refuse 0.4.0-era IR
#   beta 3  26A5378j / 24A5380h   first build to refuse beta-2-or-earlier AOT (181264112)
# iOS beta 2's build is not recorded here; anything past beta 1 is treated as beta 2+.
#
# Apple's beta 5 release notes list both incidents as fixed (177008303, 181264112). The
# first is NOT what a load shows: on 26A5416b (2026-09-04, coreai-build 3600.82.1) a
# 0.4.0-era asset still aborts at AIModel.load AND at coreai-build compile with the July
# signature (conversion/zoo_smoke.py, models/_SMOKE.json). A release note is not a
# measurement. IR040_MEASURED_OK_FROM names the first build MEASURED to load such an asset;
# it stays None until a sweep on that build says so, and a release build inherits nothing.
OS27_BETA2 = {"26A": "26A5368g", "24A": "24A5356"}
OS27_BETA3 = {"26A": "26A5378j", "24A": "24A5380h"}
IR040_MEASURED_OK_FROM: dict[str, str] | None = None   # e.g. {"26A": "26A353", "24A": "24A353"}
# coreai-build shipped with Xcode 27 beta 3 — an AOT artifact whose producer is older than
# this was compiled by beta 2 or earlier, the class 181264112 describes.
COREAI_BUILD_BETA3 = (3600, 75, 3)


def parse_build(build: str | None) -> tuple[int, str, int, bool] | None:
    """'26A5416b' -> (26, 'A', 5416, seed=True); '26A353' -> (26, 'A', 353, seed=False)."""
    m = BUILD.match((build or "").strip())
    if not m:
        return None
    major, train, num = int(m.group(1)), m.group(2), m.group(3)
    return major, train, int(num), len(num) >= 4 and num.startswith("5")


def host_os_build() -> str | None:
    try:
        out = subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


# Build majors of the OS 27 generation: macOS 27 = 26x, iOS 27 = 24x. Majors above these are
# later releases (macOS 28 = 27A). 25A is ambiguous — macOS 26 or iOS 28 — and 23A and below
# predate Core AI, so those read as unknown and the rules take the worst case.
OS27_MAJORS = {26, 24}


def build_at_least(build: str | None, thresholds: dict[str, str] | None) -> bool | None:
    """Is this OS build at or past the threshold build of its train? None = cannot be known.

    Within a train, every seed (26A5xxx) precedes the release build (26A353) regardless of
    the number, so a threshold that is a release build is NOT reached by any seed. A later
    major or a later train (26B = 27.1) is past every 27.0 threshold. A None table means
    "no such build has been measured", which is False for every host.
    """
    if thresholds is None:
        return False
    p = parse_build(build)
    if p is None:
        return None
    major, train, num, seed = p
    if major > max(OS27_MAJORS):
        return True
    if major not in OS27_MAJORS:
        return None
    threshold = parse_build(thresholds.get(f"{major}{train}"))
    if threshold is None:
        return True
    return (0 if seed else 1, num) >= (0 if threshold[3] else 1, threshold[2])


def ir040_severity(build: str | None) -> str:
    """0.4.0-era IR: loads on OS 27 beta 1, refused on every build measured since. Unknown
    host = worst case. Flips to info only past a build MEASURED to load it again."""
    past_beta2 = build_at_least(build, OS27_BETA2)
    if past_beta2 is None:
        return "fatal"
    if not past_beta2:
        return "info"
    return "info" if build_at_least(build, IR040_MEASURED_OK_FROM) else "fatal"


def aotc_severity(producer_version: str, installed: str | None) -> str | None:
    """fatal: compiled by a pre-beta-3 coreai-build (181264112). info: older than the
    installed toolchain, which is not a known break. None: nothing to report."""
    if version_tuple(producer_version) < COREAI_BUILD_BETA3:
        return "fatal"
    if installed and version_tuple(producer_version) < version_tuple(installed):
        return "info"
    return None


def build_label(build: str | None) -> str:
    p = parse_build(build)
    if p is None:
        return "an unreadable host build"
    return f"{build} ({'seed' if p[3] else 'release build'})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_json(path: Path) -> dict | None:
    try:
        with open(path, "rb") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def token_text(value) -> str | None:
    """tokenizer_config token fields are either a string or an AddedToken dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("content")
    return None


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v))


def run_xcrun(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["xcrun", *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def installed_coreai_build() -> str | None:
    out = run_xcrun(["coreai-build", "--version"]) or ""
    m = re.search(r"coreai-build\s+([\d.]+)", out)
    return m.group(1) if m else None


SHAPE = re.compile(r"NDArray \(([^,]+), ([^)]+)\)")


def parse_type(t: str) -> tuple[str | None, list[int | None]]:
    """'NDArray (Float16, 44 x 1 x 8 x ? x 128)' -> ('Float16', [44,1,8,None,128])."""
    m = SHAPE.match(t.replace("×", "x"))
    if not m:
        return None, []
    dtype, dims = m.group(1), m.group(2)
    shape = [None if d.strip() == "?" else int(d.strip())
             for d in dims.split("x") if d.strip()]
    return dtype, shape


def inspect_asset(path: Path) -> dict | None:
    out = run_xcrun(["coreai-build", "inspect", str(path), "--ops", "--json"])
    if not out:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# ASSET + GRAPH checks
# ---------------------------------------------------------------------------


def check_asset(path: Path, rep: Report, *, inside_bundle: bool = False,
                bundle_meta: dict | None = None, tok_dir: Path | None = None,
                profile: str = "any") -> None:
    meta = read_json(path / "metadata.json") or {}
    producer = meta.get("producer")
    compiled = path.suffix == ".aimodelc"

    rep.ran(SYMLINKED_ASSET.id)
    if path.is_symlink() or (not inside_bundle and path.parent.is_symlink()):
        rep.add(
            SYMLINKED_ASSET, str(path),
            "the asset (or its parent directory) is a symlink",
            "Resolve the real path before loading: url.resolvingSymlinksInPath(). The Python "
            "runtime follows symlinks and Swift does not, so this passes every Python gate and "
            "fails only in the app.",
        )

    if not compiled:
        rep.ran(IR_040.id)
        if producer is None:
            severity = ir040_severity(rep.host_build)
            if severity == "fatal":
                verdict = (f"this host, {build_label(rep.host_build)}, refuses it at load — "
                           f"measured on 26A5416b (2026-09-04); no later build has been measured "
                           f"to accept it, and Apple's beta 5 note (177008303) did not change this"
                           if rep.host_build else
                           "the host build could not be read, so this assumes a build that "
                           "refuses it (every OS 27 build from beta 2 on)")
            else:
                verdict = (f"this host, {build_label(rep.host_build)}, loads it; every OS 27 build "
                           f"from beta 2 on refuses it, so no one else can")
            rep.add(
                IR_040, str(path / "metadata.json"),
                "no 'producer' key — a 0.4.1+ asset carries "
                '{"producer": "coreai-core 1.0.0b2", ...}; a 0.4.0 one carries only assetVersion. '
                + verdict,
                "strip_debug_info is the cheap fix: weights stay byte-identical, minutes per "
                "model, no re-export. It needs an isolated coreai-torch 0.4.0 + coreai-core "
                "1.0.0b1 venv to parse the old locations, then a b2 re-save to restamp the "
                "producer. Note coreai-build inspect reads a broken asset happily — inspect "
                "succeeding is not evidence the asset loads.",
                severity=severity,
            )
        else:
            rep.notes.append(f"asset producer: {producer}")
    else:
        rep.ran(AOTC_NO_SOURCE.id, AOTC_LOAD_OPTIONS.id)
        if not list(path.parent.glob("*.aimodel")):
            rep.add(
                AOTC_NO_SOURCE, str(path),
                "no source .aimodel next to the compiled artifact",
                "Keep the source .aimodel with anything you publish. A compiled-only artifact "
                "cannot be strip-repaired and needs a full re-export plus AOT recompile.",
            )
        rep.add(
            AOTC_LOAD_OPTIONS, str(path),
            "compiled AOT artifact",
            "Load with SpecializationOptions.default() (or .cpuOnly()). "
            "init(preferredComputeUnitKind: .gpu) makes the runtime re-specialize the baked "
            "graph, which wedges large graphs on OS 27 (failedToSpecialize). "
            "expectFrequentReshapes is already baked in and must NOT be requested at load time — "
            "on a fixed-shape graph asking for it segfaults the on-device MPSGraph compiler.",
        )

        rep.ran(AOTC_STALE.id)
        installed = installed_coreai_build()
        m = re.search(r"coreai-build-([\d.]+)", producer or "")
        if m:
            beta3 = ".".join(map(str, COREAI_BUILD_BETA3))
            severity = aotc_severity(m.group(1), installed)
            if severity == "fatal":
                rep.add(
                    AOTC_STALE, str(path / "metadata.json"),
                    f"compiled by coreai-build {m.group(1)}, older than Xcode 27 beta 3's {beta3}"
                    + (f"; installed toolchain is {installed}" if installed else "")
                    + " — OS 27 beta 3 and later fail to specialize it (181264112)",
                    "Recompile: xcrun coreai-build compile <source>.aimodel --output <dir> "
                    "--platform <iOS|macOS> --architecture <arch> --preferred-compute gpu",
                )
            elif severity == "info":
                rep.add(
                    AOTC_STALE, str(path / "metadata.json"),
                    f"compiled by coreai-build {m.group(1)}; installed toolchain is {installed}. "
                    f"Not a known break (181264112 needs a producer below {beta3}) — whether a "
                    f"release toolchain still specializes it is a device check, not a lint",
                    "Recompile with the current toolchain when the device sweep says so, or for "
                    "its own improvements: xcrun coreai-build compile <source>.aimodel --output "
                    "<dir> --platform <iOS|macOS> --architecture <arch> --preferred-compute gpu",
                    severity="info",
                )
            else:
                rep.notes.append(
                    f"AOT producer coreai-build {m.group(1)}"
                    + (f" matches the installed {installed}" if installed else ""))
        arch = next((re.match(r"main-([a-z0-9]+)\.mlirb$", f.name).group(1)
                     for f in path.iterdir()
                     if re.match(r"main-([a-z0-9]+)\.mlirb$", f.name)), None)
        if arch:
            rep.notes.append(
                f"compiled for architecture '{arch}' — arch names track the DEVICE IDENTIFIER "
                f"major version (iPhone18,1 -> h18p, Mac16,x -> h16c), not the marketing name, "
                f"and coreai-build exits 0 for any arch you ask for"
            )

    check_graph(path, rep, bundle_meta=bundle_meta, tok_dir=tok_dir, profile=profile)


IDENTITY_OPS = {"floor", "trunc", "ceil"}


def check_graph(path: Path, rep: Report, *, bundle_meta: dict | None,
                tok_dir: Path | None, profile: str) -> None:
    info = inspect_asset(path)
    if info is None:
        rep.notes.append(
            "coreai-build inspect unavailable (needs the Xcode 27 toolchain) — graph-level "
            "rules were skipped"
        )
        return

    summary = info.get("summary") or {}
    functions = summary.get("functions") or []
    main = next((f for f in functions if f.get("name") == "main"), functions[0] if functions else {})
    states = main.get("states") or []
    inputs = {i["name"]: i["type"] for i in main.get("inputs") or []}
    outputs = {o["name"]: o["type"] for o in main.get("outputs") or []}
    ops = {o["name"]: o["count"] for o in summary.get("operationDistribution") or []}

    rep.notes.append(
        f"graph: {len(functions)} function(s), {len(states)} state(s), "
        f"{sum(ops.values())} ops in {len(ops)} kinds"
    )

    # --- states vs the engine that has to drive it
    rep.ran(GRAPH_STATE_COUNT.id)
    if states and len(states) != 2:
        rep.add(
            GRAPH_STATE_COUNT, str(path),
            f"{len(states)} states: " + ", ".join(s["name"] for s in states),
            "The coreai-sequential engine variant hard-requires exactly 2 states and fails "
            "engine-create with invalidOutputType(\"Expected 2 states ... got N\"). Route this "
            "to coreai-pipelined with the extra-states patch, and remember "
            "COREAI_CHUNK_THRESHOLD is read live at prefill time.",
        )

    # --- S=1 decode graphs come with a run contract
    rep.ran(GRAPH_S1_CONTRACT.id)
    ids_type = inputs.get("input_ids")
    if ids_type:
        _, ids_shape = parse_type(ids_type)
        if ids_shape and all(d is not None for d in ids_shape) and ids_shape[-1] == 1:
            rep.add(
                GRAPH_S1_CONTRACT, str(path),
                f"input_ids is static {ids_type}",
                "Set COREAI_CHUNK_THRESHOLD=1 BEFORE engine creation, never call "
                "engine.warmup() (it warms query length 256 and the static [1,1] graph "
                "rejects it), and drive llm-runner with --warmup exact --warmup-length 1. "
                "In an app, a 1-token generate after load IS the warmup.",
            )

    # --- vocab agreement, graph-first
    logits_type = next((t for n, t in outputs.items() if "logit" in n), None)
    logits_width = None
    if logits_type:
        _, lshape = parse_type(logits_type)
        logits_width = lshape[-1] if lshape else None
    if logits_width:
        rep.ran(GRAPH_VOCAB_MISMATCH.id)
        declared = ((bundle_meta or {}).get("language") or {}).get("vocab_size")
        if declared and declared != logits_width:
            rep.add(
                GRAPH_VOCAB_MISMATCH, str(path),
                f"metadata language.vocab_size={declared}, graph logits width={logits_width}",
                "Make metadata match the graph. The engine sizes its sampler from metadata and "
                "indexes the real logits buffer.",
            )
        if tok_dir is not None:
            rep.ran(GRAPH_TOKENIZER_OVERFLOW.id)
            top = max_token_id(tok_dir)
            if top is not None and top >= logits_width:
                rep.add(
                    GRAPH_TOKENIZER_OVERFLOW, str(tok_dir),
                    f"highest tokenizer id {top} >= logits width {logits_width}",
                    "Either widen the head/embedding to cover the added tokens, or map those "
                    "ids host-side. Tokens above the head are unreachable, and fed back as "
                    "input ids they index outside the embedding table.",
                )

    # --- custom Metal kernels pin the compute unit
    rep.ran(GRAPH_CUSTOM_METAL.id)
    if ops.get("metal4_kernel"):
        rep.add(
            GRAPH_CUSTOM_METAL, str(path),
            f"{ops['metal4_kernel']} metal4_kernel ops",
            "This bundle is GPU-only: the ANE runs fixed hardware ops and can never execute "
            "hand-written MSL. Custom kernels do survive AOT (the .aimodelc embeds the MSL and "
            "the compiled outputs are bit-identical). If the kernel is gather_qmm-class, note "
            "its logits shape asserts in the pipelined GrowingLogitsBuffer.",
        )

    # --- rounding ops that lower wrongly on the GPU delegate
    rep.ran(GRAPH_FLOOR_IDENTITY.id, GRAPH_ROUND_TIES.id)
    if present := {k: v for k, v in ops.items() if k in IDENTITY_OPS}:
        rep.add(
            GRAPH_FLOOR_IDENTITY, str(path),
            ", ".join(f"{k} x{v}" for k, v in sorted(present.items())),
            "Verify these on the GPU, not on CPU — CPU executes them correctly from the same "
            "asset. The floor that survives every unit is "
            "torch.div(x * 2.0, 2.0, rounding_mode='floor'); div(x, 1, rounding_mode='floor') "
            "and a float->int->float round-trip are both folded away.",
        )
    if n := ops.get("round"):
        rep.add(
            GRAPH_ROUND_TIES, str(path), f"round x{n}",
            "Check whether any of these sit in a quantize/dequantize path, where a "
            "ties-away-from-zero result differs from torch by one LSB.",
        )

    # --- growing KV + iOS
    rep.ran(GRAPH_DYNAMIC_KV_IOS.id)
    growing = [s["name"] for s in states if None in parse_type(s["type"])[1]]
    ctx = ((bundle_meta or {}).get("language") or {}).get("max_context_length")
    if growing:
        rep.notes.append("growing (dynamic-seq) states: " + ", ".join(growing))
    if growing and profile in ("iphone", "ios") and isinstance(ctx, int) and ctx >= 2048:
        rep.add(
            GRAPH_DYNAMIC_KV_IOS, str(path),
            f"dynamic-seq state(s) {', '.join(growing)} with declared "
            f"max_context_length={ctx}",
            "Drive it with an engine that caps the resolved iOS KV capacity at 1024 (the guard "
            "shipped in coreai-models bd8dcf7), or clamp prompt + maxTokens below 1024 at the "
            "app layer. Capacity <=1024 is clean every time; .fixedSize(4096) is WORSE, because "
            "every generation then rides the broken shape, and a full cache evict plus fresh "
            "compile still soups. Mac is clean at every shape, so this cannot reproduce on your "
            "development machine — and a device gate capped at 256 generated tokens cannot see "
            "it either, which is exactly how it reached a chat app.",
        )


def max_token_id(tok_dir: Path) -> int | None:
    ids: list[int] = []
    if added := read_json(tok_dir / "added_tokens.json"):
        ids += [v for v in added.values() if isinstance(v, int)]
    cfg = read_json(tok_dir / "tokenizer_config.json") or {}
    ids += [int(k) for k in (cfg.get("added_tokens_decoder") or {}) if str(k).isdigit()]
    data = read_json(tok_dir / "tokenizer.json")
    if data:
        ids += [t["id"] for t in data.get("added_tokens") or [] if isinstance(t.get("id"), int)]
        vocab = (data.get("model") or {}).get("vocab")
        if isinstance(vocab, dict):
            ids.append(len(vocab) - 1)
        elif isinstance(vocab, list):
            ids.append(len(vocab) - 1)
    return max(ids) if ids else None


# ---------------------------------------------------------------------------
# BUNDLE checks
# ---------------------------------------------------------------------------

# swift-transformers Sources/Tokenizers/Tokenizer.swift, TokenizerModel.knownTokenizers.
# A "Fast" suffix is stripped before the lookup.
SWIFT_KNOWN_TOKENIZERS = {
    "BertTokenizer", "CodeGenTokenizer", "CodeLlamaTokenizer", "CohereTokenizer",
    "DistilbertTokenizer", "DistilBertTokenizer", "FalconTokenizer", "GemmaTokenizer",
    "GPT2Tokenizer", "LlamaTokenizer", "RobertaTokenizer", "T5Tokenizer",
    "TokenizersBackend", "PreTrainedTokenizer", "Qwen2Tokenizer", "WhisperTokenizer",
    "XLMRobertaTokenizer", "Xlm-RobertaTokenizer",
}

IOS_JIT_WATCH_BYTES = 1 << 30          # knowledge/aot-and-specialization.md "the 4B wall"
ENTITLEMENT_BYTES = 2 * (1 << 30)      # knowledge/pipelined-engine.md "Run contract"


def chat_template_text(tok_dir: Path, cfg: dict) -> tuple[str | None, str]:
    if (jinja := tok_dir / "chat_template.jinja").exists():
        return jinja.read_text(errors="ignore"), str(jinja)
    tmpl = cfg.get("chat_template")
    if isinstance(tmpl, str):
        return tmpl, str(tok_dir / "tokenizer_config.json") + " (chat_template)"
    if isinstance(tmpl, list) and tmpl:  # the named-template form
        return "\n".join(t.get("template", "") for t in tmpl), str(tok_dir / "tokenizer_config.json")
    return None, ""


JINJA_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def renders_eos_variable(template: str) -> bool:
    """True when the template emits the eos_token VARIABLE rather than a literal.

    Templates reference it in the middle of an expression as often as at the start
    ("{{ '<|assistant|>' + message['content'] + eos_token }}"), so look for the bare
    identifier with string literals removed rather than matching right after '{{'.
    """
    return re.search(r"\beos_token\b", JINJA_STRING.sub("", template)) is not None


def special_tokens(tok_dir: Path, cfg: dict) -> set[str]:
    """Special/added token strings, from the small files first.

    tokenizer.json is routinely tens of MB (Gemma's is 32), so it is the last resort.
    """
    out: set[str] = set()
    for entry in (cfg.get("added_tokens_decoder") or {}).values():
        if c := token_text(entry):
            out.add(c)
    for key, value in cfg.items():
        if key.endswith("_token") and (c := token_text(value)):
            out.add(c)
    if added := read_json(tok_dir / "added_tokens.json"):
        out.update(added.keys())
    if stm := read_json(tok_dir / "special_tokens_map.json"):
        for value in stm.values():
            if c := token_text(value):
                out.add(c)
            elif isinstance(value, list):
                out.update(filter(None, (token_text(v) for v in value)))
    if not out and (data := read_json(tok_dir / "tokenizer.json")):
        out.update(t["content"] for t in data.get("added_tokens") or [] if t.get("content"))
    return out


def check_bundle(path: Path, rep: Report, profile: str) -> None:
    meta = read_json(path / "metadata.json")

    # A directory can hold a metadata.json that is not a Core AI bundle manifest at all
    # (ship sidecars, app configs). Lint the assets it contains and say so, rather than
    # holding it to a contract it never claimed.
    if meta is not None and "metadata_version" not in meta and "assets" not in meta:
        rep.ran(NOT_A_BUNDLE_MANIFEST.id)
        assets = [c for c in sorted(path.iterdir())
                  if c.is_dir() and c.suffix in (".aimodel", ".aimodelc")]
        rep.add(
            NOT_A_BUNDLE_MANIFEST, str(path / "metadata.json"),
            f"keys: {', '.join(sorted(meta)[:8])}",
            f"Linting the {len(assets)} asset(s) in the directory directly. If this is meant to "
            f"load through LanguageBundle, it needs metadata_version, kind, assets.main, the "
            f"language block and a tokenizer/ directory.",
        )
        for asset in assets:
            check_asset(asset, rep, inside_bundle=True, bundle_meta=None, tok_dir=None,
                        profile=profile)
        return

    rep.ran(BUNDLE_INCOMPLETE.id)
    missing: list[str] = []
    if meta is None:
        missing.append("metadata.json")
        meta = {}
    lang = meta.get("language") or {}
    tok_dir = path / "tokenizer"
    required: list[tuple[str, dict, str]] = [
        ("kind", meta, "metadata.kind"),
        ("assets", meta, "metadata.assets"),
    ]
    # The language block and the tokenizer directory are what LanguageBundle reads. A
    # vision-encoder or drafter asset ships neither by design, so only hold `kind: llm`
    # to that contract.
    is_language_bundle = meta.get("kind") == "llm"
    if is_language_bundle:
        required += [
            ("tokenizer", lang, "metadata.language.tokenizer"),
            ("vocab_size", lang, "metadata.language.vocab_size"),
            ("max_context_length", lang, "metadata.language.max_context_length"),
            ("function_map", lang, "metadata.language.function_map"),
        ]
        if not tok_dir.is_dir():
            missing.append("tokenizer/")
    for key, holder, label in required:
        if key not in holder:
            missing.append(label)
    if missing:
        rep.add(
            BUNDLE_INCOMPLETE, str(path),
            "missing: " + ", ".join(missing),
            "LanguageBundle/EngineFactory need the whole directory: metadata.json (kind llm, "
            "assets.main, language.{tokenizer,vocab_size,max_context_length,function_map}) + "
            "tokenizer/ + the .aimodel.",
        )

    rep.ran(BUNDLE_ASSET_MISSING.id)
    asset_name = (meta.get("assets") or {}).get("main")
    asset_path = path / asset_name if asset_name else None
    if asset_name and (asset_path is None or not asset_path.exists()):
        rep.add(
            BUNDLE_ASSET_MISSING, str(path / "metadata.json"),
            f"assets.main = {asset_name!r} does not exist on disk",
            "Point assets.main at the .aimodel (or .aimodelc) directory shipped in the bundle.",
        )
        asset_path = None

    if asset_path is not None:
        check_asset(asset_path, rep, inside_bundle=True, bundle_meta=meta,
                    tok_dir=tok_dir if tok_dir.is_dir() else None, profile=profile)
        size = dir_size(asset_path)
        rep.notes.append(f"asset {asset_path.name}: {human(size)}")

        if profile in ("iphone", "ios"):
            rep.ran(BIG_IOS_JIT.id)
            if asset_path.suffix == ".aimodel" and size >= IOS_JIT_WATCH_BYTES:
                rep.add(
                    BIG_IOS_JIT, str(asset_path),
                    f"{human(size)} portable IR, no .aimodelc in the bundle",
                    "Time a cold on-device specialization before shipping this as portable IR. "
                    "It is graph-shaped, not purely size-driven: a 4.6 GB static-S=1 int8 "
                    "pipelined bundle JITs on iPhone in ~54 s, while a 4B dense GPU bundle "
                    "exhausts the device's scratch disk and a 2 GB-constants graph fails "
                    "mmap allocation. If it aborts: xcrun coreai-build compile <m>.aimodel "
                    "--platform iOS --architecture h18p --preferred-compute gpu "
                    "--min-deployment-version 27.0.",
                )
            rep.ran(IPHONE_MEMORY_ENTITLEMENT.id)
            if size >= ENTITLEMENT_BYTES:
                rep.add(
                    IPHONE_MEMORY_ENTITLEMENT, str(asset_path),
                    f"{human(size)} is past the ~2 GB point where cold specialization hits the "
                    f"default jetsam limit",
                    "Ship com.apple.developer.kernel.increased-memory-limit. Also budget for "
                    "the failure mode: an aborted cold specialization leaves partial e-caches "
                    "that make every later attempt fail as NSPOSIXErrorDomain code=2, so carry "
                    "a Library/Caches/coreai-cache wipe hook.",
                )

    if not tok_dir.is_dir():
        return
    cfg = read_json(tok_dir / "tokenizer_config.json") or {}

    rep.ran(TOKENIZER_CLASS_UNKNOWN.id)
    tclass = cfg.get("tokenizer_class")
    if not tclass:
        rep.add(
            TOKENIZER_CLASS_UNKNOWN, str(tok_dir / "tokenizer_config.json"),
            "no tokenizer_class key",
            "swift-transformers throws missingTokenizerClassInConfig without it.",
        )
    elif tclass.replace("Fast", "") not in SWIFT_KNOWN_TOKENIZERS:
        rep.add(
            TOKENIZER_CLASS_UNKNOWN, str(tok_dir / "tokenizer_config.json"),
            f"tokenizer_class={tclass!r} (looked up as {tclass.replace('Fast', '')!r}) is not in "
            f"swift-transformers' knownTokenizers",
            "Retag tokenizer_config.json at upload time to a registered class with the RIGHT "
            "algorithm: BPETokenizer-backed names for byte-level BPE, XLMRobertaTokenizer for "
            "SentencePiece/Unigram, BertTokenizer for WordPiece. The non-strict fallback is BPE "
            "for everything, which tokenizes a Unigram model wrongly and never errors.",
        )

    template, template_where = chat_template_text(tok_dir, cfg)
    rep.ran(NO_CHAT_TEMPLATE.id, NO_CHAT_TEMPLATE_INFO.id)
    if template is None:
        source_id = (meta.get("source") or {}).get("hf_model_id", "")
        instruct = bool(re.search(r"(-it\b|-it-|instruct|-chat\b|chat-)", source_id, re.I))
        rep.add(
            NO_CHAT_TEMPLATE if instruct else NO_CHAT_TEMPLATE_INFO, str(tok_dir),
            "neither chat_template.jinja nor a chat_template key in tokenizer_config.json"
            + (f"; source model {source_id!r} looks instruction-tuned" if instruct
               else f"; source model {source_id!r} does not look instruction-tuned"),
            "Copy the source model's chat template into the bundle. --apply-chat-template "
            "defaults to true and does NOT warn when there is nothing to apply — the model "
            "simply never sees turn markers, and any quality number measured this way is "
            "measuring raw completion.",
        )
    else:
        rep.ran(EOS_NOT_IN_TEMPLATE.id)
        eos = token_text(cfg.get("eos_token"))
        if eos and eos not in template and not renders_eos_variable(template):
            emitted = sorted(
                t for t in special_tokens(tok_dir, cfg)
                if t != eos and len(t) > 2 and (f"'{t}'" in template or f'"{t}"' in template)
            )
            rep.add(
                EOS_NOT_IN_TEMPLATE, template_where,
                f"eos_token={eos!r} never appears in the template; the template does emit "
                + (", ".join(repr(t) for t in emitted[:6]) if emitted else "no known special token"),
                "Set the bundle's eos_token to the turn terminator the template actually emits. "
                "swift-transformers' StopSequences stops only on tokenizer.eosTokenId, so a "
                "document-end eos means a generic app never sees the turn end and runs to "
                "maxTokens. This is render-safe as long as the template emits the terminator "
                "literally rather than through {{ eos_token }}.",
            )


# ---------------------------------------------------------------------------
# CHECKPOINT checks
# ---------------------------------------------------------------------------

ACTIVATION_SCALE_KEYS = (
    "input_activation_scale", "output_activation_scale", "k_cache_scale", "v_cache_scale",
)


def safetensors_keys(path: Path) -> list[str]:
    """Tensor names from safetensors headers, without loading any weights."""
    names: list[str] = []
    for shard in sorted(path.glob("*.safetensors")):
        try:
            with open(shard, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            names.extend(k for k in header if k != "__metadata__")
        except (OSError, ValueError, struct.error):
            continue
    return names


def remote_tensor_names(repo_id: str, rep: Report) -> list[str]:
    """Tensor names for a remote checkpoint, headers only — no weights are downloaded."""
    try:
        from huggingface_hub import HfApi

        return list(HfApi().get_safetensors_metadata(repo_id).weight_map)
    except Exception as exc:  # offline, gated repo, non-safetensors layout
        rep.notes.append(f"could not read remote tensor names ({type(exc).__name__}) — "
                         f"the activation-scale rule needs a local snapshot")
        return []


def check_checkpoint(path: Path, rep: Report, block_size: int,
                     repo_id: str | None = None) -> None:
    config = read_json(path / "config.json") or {}
    gen = read_json(path / "generation_config.json") or {}
    tcfg = read_json(path / "tokenizer_config.json") or {}
    text_cfg = config.get("text_config") or config

    rep.notes.append(
        f"architectures={config.get('architectures') or text_cfg.get('architectures')} "
        f"model_type={config.get('model_type')}"
    )

    rep.ran(PREQUANTIZED_CHECKPOINT.id)
    if qc := config.get("quantization_config"):
        rep.add(
            PREQUANTIZED_CHECKPOINT, str(path / "config.json"),
            f"quant_method={qc.get('quant_method')!r} num_bits={qc.get('num_bits')} "
            f"quantize_embeddings={qc.get('quantize_embeddings')}"
            + (f", {len(qc['module_quant_configs'])} per-module overrides"
               if qc.get("module_quant_configs") else ""),
            "Read the per-module bit-widths before choosing a recipe, and never compare this "
            "against a generic same-bit build and call the delta 'runtime quality'. To build a "
            "matched pair, compile every arm from the same UNQUANTIZED checkpoint at the same "
            "block size.",
        )

    rep.ran(ACT_SCALE_QAT.id)
    index = read_json(path / "model.safetensors.index.json") or {}
    names = list((index.get("weight_map") or {}).keys()) or safetensors_keys(path)
    if not names and repo_id:
        names = remote_tensor_names(repo_id, rep)
    hits = sorted({k for k in names if any(s in k for s in ACTIVATION_SCALE_KEYS)})
    if hits:
        rep.add(
            ACT_SCALE_QAT, str(path),
            f"{len(hits)} activation/KV scale tensors, e.g. " + ", ".join(hits[:3]),
            "Implement per-linear static activation quantization in the export (quantize with "
            "input_activation_scale -> int8 matmul -> requantize with output_activation_scale "
            "-> dequantize), then int8 KV storage for the remainder. Do NOT gate this against "
            "an fp16 oracle built from the same weights: an equivalence gate cannot see a "
            "defect its reference shares, and single-hop recall prompts cannot see it at all.",
        )
    elif names:
        rep.notes.append(f"{len(names)} weight tensors, no activation-scale tensors")

    rep.ran(EOS_SOURCE_MISMATCH.id)
    gen_eos = gen.get("eos_token_id")
    gen_eos = gen_eos if isinstance(gen_eos, list) else ([gen_eos] if gen_eos is not None else [])
    tok_eos = token_text(tcfg.get("eos_token"))
    if len(gen_eos) > 1:
        named = {int(k): token_text(v) for k, v in (tcfg.get("added_tokens_decoder") or {}).items()
                 if token_text(v)}
        rep.add(
            EOS_SOURCE_MISMATCH, str(path / "generation_config.json"),
            f"generation_config eos_token_id = "
            f"[{', '.join(f'{i}' + (f' ({named[i]})' if i in named else '') for i in gen_eos)}], "
            f"tokenizer_config eos_token = {tok_eos!r}",
            "Decide which id ends a CHAT TURN and set the exported bundle's eos_token to it at "
            "export time, in the save_tokenizer step after the verbatim HF copy. Anything that "
            "stops on a single eosTokenId — swift-transformers does — otherwise misses the turn "
            "terminator and runs to the cap.",
        )

    rep.ran(TIED_EMBEDDINGS.id)
    if text_cfg.get("tie_word_embeddings"):
        rep.add(
            TIED_EMBEDDINGS, str(path / "config.json"),
            "tie_word_embeddings = true",
            "Clone the embedding table before quantizing if you want the head quantized, and "
            "measure the result on the PHONE: the untied head looked like a no-win on Mac for "
            "three models in a row and was +17-40% on device, where it is a large share of the "
            "per-token read. Big-vocab heads want plain 'symmetric' absmax, never "
            "symmetric_with_clipping (that flips outlier head rows).",
        )

    rep.ran(BLOCK_DIVISIBILITY.id)
    dims = {k: text_cfg[k] for k in
            ("hidden_size", "intermediate_size", "head_dim", "vocab_size", "moe_intermediate_size")
            if isinstance(text_cfg.get(k), int)}
    if bad := {k: v for k, v in dims.items() if v % block_size}:
        rep.add(
            BLOCK_DIVISIBILITY, str(path / "config.json"),
            f"not divisible by block size {block_size}: "
            + ", ".join(f"{k}={v}" for k, v in bad.items()),
            "Check the resulting bundle SIZE against the theoretical one before believing the "
            "recipe applied. Per-block quant and per-grouped-channel palettization skip "
            "non-divisible layers without a word and ship them uncompressed.",
        )
    elif dims:
        rep.notes.append(f"dims divisible by {block_size}: "
                         + ", ".join(f"{k}={v}" for k, v in dims.items()))

    if sources := sorted(path.glob("*.py")):
        check_source_files(sources, rep)
    elif config.get("auto_map"):
        rep.notes.append(
            "config declares auto_map but no .py is present locally — fetch the modelling files "
            "and re-run with --torch-src to lint the graph-level traps"
        )


# ---------------------------------------------------------------------------
# SOURCE checks
# ---------------------------------------------------------------------------

COMMENT = re.compile(r"^\s*#")


def check_source_files(files: list[Path], rep: Report) -> None:
    rep.ran(*(r.id for r, _p, _f in SRC_RULES))
    if not files:
        return

    saw_slice_update = False
    saw_defunctionalize = False
    state_writes: dict[Path, int] = {}

    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        if "remove_functionalization" in text:
            saw_defunctionalize = True
        for n, line in enumerate(text.splitlines(), 1):
            if COMMENT.match(line):
                continue
            for r, pat, fix in SRC_RULES:
                if not pat.search(line):
                    continue
                if r.id == "SRC-MISSING-DEFUNCTIONALIZE":
                    saw_slice_update = True
                elif r.id == "SRC-CHAINED-STATE-WRITES":
                    state_writes[f] = state_writes.get(f, 0) + 1
                else:
                    rep.add(r, f"{f}:{n}", line.strip()[:150], fix)

    # These two are about a COUNT and an ABSENCE, so they are judged after the scan.
    for f, count in state_writes.items():
        if count > 1:
            r = RULES["SRC-CHAINED-STATE-WRITES"]
            rep.add(r, str(f), f"{count} state-write call sites in one module", SRC_FIX[r.id])
    if saw_slice_update and not saw_defunctionalize:
        r = RULES["SRC-MISSING-DEFUNCTIONALIZE"]
        rep.add(r, str(files[0].parent),
                "slice_update present, remove_functionalization absent from the scanned files",
                SRC_FIX[r.id])


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------

OVERLAY_CANARY = "coreai_models.models.macos.qwen3_5"  # mirrors conversion/zoo_convert.py


def check_env(rep: Report, python: str | None = None) -> None:
    """Environment defects whose output is a bad asset rather than an error message.

    `conversion/zoo_convert.py doctor` checks the same interpreter wiring and stays the
    entry point people already type. This is the superset: the overlay probe is the same
    one, plus the shadowing defect that produces a silently-0.4.0 export.
    """
    if python:
        rep.ran(ENV_OVERLAY_MISSING.id)
        probe = (f"import coreai_models, importlib; "
                 f"importlib.import_module('{OVERLAY_CANARY}'); print(coreai_models.__file__)")
        res = subprocess.run([python, "-c", probe], capture_output=True, text=True)
        if res.returncode != 0:
            tail = (res.stderr.strip().splitlines() or ["unknown error"])[-1]
            rep.add(
                ENV_OVERLAY_MISSING, python, tail,
                "The interpreter needs coreai_models with conversion/overlay/ applied — clone "
                "the pinned base, run apply.py, pip install -e python/. See "
                "conversion/overlay/README.md, or `python3 conversion/zoo_convert.py doctor`.",
            )
        else:
            rep.notes.append(f"overlay OK ({OVERLAY_CANARY} imports from {res.stdout.strip()})")

    rep.ran(ENV_CLONE_SHADOW.id)
    cwd = Path.cwd()
    shadows = sorted(cwd.glob("coreai_torch*.egg-info")) + sorted(cwd.glob("coreai_torch/__init__.py"))
    if shadows:
        rep.add(
            ENV_CLONE_SHADOW, str(cwd),
            "working directory contains " + ", ".join(p.name for p in shadows[:3]),
            "Never run an export with a coreai-torch clone as the working directory. Its "
            "egg-info takes sys.path[0] priority over the installed wheel, so a 0.4.1 "
            "environment silently exports 0.4.0-era IR — an asset with no producer stamp that "
            "OS 27 beta 2-4 refuse at load and every producer audit flags. cd somewhere else "
            "and re-export.",
        )


HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")

# Small files worth fetching for a checkpoint lint. Weights are never downloaded.
HF_SMALL_FILES = (
    "config.json", "generation_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
    "model.safetensors.index.json",
)


def fetch_hf(repo_id: str, cache: Path) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not importable — install it, or point doctor at a local "
            "checkpoint directory."
        )
    out = cache / repo_id.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    available = set(list_repo_files(repo_id))
    wanted = [f for f in HF_SMALL_FILES if f in available]
    wanted += [f for f in available if f.endswith(".py") and "/" not in f]
    for name in wanted:
        (out / name).write_bytes(Path(hf_hub_download(repo_id, name)).read_bytes())
    print(f"fetched {len(wanted)} metadata files for {repo_id} -> {out}", file=sys.stderr)
    return out


def classify(path: Path) -> str:
    if path.suffix in (".aimodel", ".aimodelc"):
        return "asset"
    if (path / "metadata.json").exists() and (
        list(path.glob("*.aimodel")) or list(path.glob("*.aimodelc"))
    ):
        return "bundle"
    if (path / "config.json").exists():
        return "checkpoint"
    if path.is_file() and path.suffix == ".py":
        return "source"
    if list(path.glob("*.py")):
        return "source"
    raise SystemExit(
        f"cannot tell what {path} is — expected a LanguageBundle directory, a "
        f".aimodel/.aimodelc, an HF checkpoint directory (config.json), or PyTorch source"
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

BADGE = {"fatal": "FATAL  ", "silent": "SILENT ", "runaway": "RUNAWAY",
         "perf": "PERF   ", "requires": "REQUIRE", "info": "INFO   "}

CLOSING = """
What doctor does NOT tell you: it is a static lint and it cannot check numerics.
Gate the bundle against an HF oracle before trusting it — teacher-forced top-1 over a
prompt whose fp32 top-2 margin clears 0.1 at every position, 16/16. And gate the ENGINE
path, not only the python runtime.
"""


def render(rep: Report, verbose: bool) -> None:
    print(f"coreai doctor  {rep.target}")
    print(f"kind           {rep.kind}")
    if rep.host_build:
        print(f"host build     {build_label(rep.host_build)}")
    for note in rep.notes:
        print(f"note           {note}")
    print(f"rules run      {len(rep.checked)}")
    print()

    defects, requirements = rep.defects(), rep.requirements()
    if not defects:
        print("no known failure pattern detected.")
        print()

    for group, heading in ((defects, "DEFECTS"), (requirements, "NOTES AND SHIP REQUIREMENTS")):
        if not group:
            continue
        group.sort(key=lambda f: SEVERITY_ORDER[f.severity])
        print(f"--- {heading} " + "-" * (66 - len(heading)))
        print()
        for f in group:
            print(f"{BADGE[f.severity]} {f.rule.id}"
                  + (f"  (host-conditional: {f.rule.severity} on a build that refuses it)"
                     if f.severity != f.rule.severity else ""))
            print(f"  {f.rule.title}")
            print(f"  where : {f.where}")
            print(f"  found : {f.evidence}")
            print(f"  fix   : {f.fix}")
            print(f"  see   : {f.rule.url}")
            if verbose:
                print(f"  source: {f.rule.source}")
            print()

    counts: dict[str, int] = {}
    for f in rep.findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    if counts:
        print("summary        " + ", ".join(
            f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: SEVERITY_ORDER[kv[0]])))
    print(CLOSING)


def to_json(rep: Report) -> str:
    return json.dumps({
        "target": rep.target,
        "kind": rep.kind,
        "rules_run": rep.checked,
        "notes": rep.notes,
        "host_build": rep.host_build,
        "findings": [{
            "id": f.rule.id, "severity": f.severity, "rule_severity": f.rule.severity,
            "scope": f.rule.scope, "title": f.rule.title, "where": f.where,
            "evidence": f.evidence, "fix": f.fix, "source": f.rule.source,
            "url": f.rule.url,
        } for f in sorted(rep.findings, key=lambda f: SEVERITY_ORDER[f.severity])],
    }, indent=2)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?",
                    help="LanguageBundle dir, .aimodel/.aimodelc, HF checkpoint dir, HF repo id, "
                         "or PyTorch source")
    ap.add_argument("--profile", default="iphone", choices=["iphone", "ios", "mac", "any"],
                    help="device class the bundle targets; gates the device-only rules "
                         "(default: iphone)")
    ap.add_argument("--torch-src", action="append", default=[],
                    help="extra PyTorch modelling file or directory to lint (repeatable)")
    ap.add_argument("--block-size", type=int, default=32,
                    help="quantization block size the recipe intends (default: 32)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--verbose", "-v", action="store_true", help="print each finding's source")
    ap.add_argument("--rules", action="store_true", help="list every rule and exit")
    ap.add_argument("--host-build", metavar="BUILD",
                    help="judge the OS-build-conditional asset rules against this OS build "
                         "(e.g. 26A5378j, 24A5408d, 26A353) instead of this Mac's sw_vers")
    ap.add_argument("--cache", default=None, help="directory for fetched HF metadata")
    ap.add_argument("--env", nargs="?", const="", default=None, metavar="PYTHON",
                    help="also probe an interpreter for the zoo overlay (superset of "
                         "`zoo_convert.py doctor`); defaults to the sibling coreai-models venv")
    args = ap.parse_args()

    if args.rules:
        for r in sorted(RULES.values(),
                        key=lambda r: (r.scope, SEVERITY_ORDER[r.severity], r.id)):
            print(f"{r.scope:<11}{r.severity:<9}{r.id}")
            print(f"           {r.title}")
            print(f"           see:    {r.url}")
            print(f"           source: {r.source}\n")
        print(f"{len(RULES)} rules")
        return
    if not args.target:
        ap.error("a target is required (or --rules)")

    if HF_ID.match(args.target) and not Path(args.target).exists():
        path = fetch_hf(args.target, Path(args.cache or Path.home() / ".cache" / "coreai-doctor"))
        kind, label, repo_id = "checkpoint", args.target, args.target
    else:
        path = Path(args.target).resolve()
        if not path.exists():
            raise SystemExit(f"no such path: {args.target}")
        kind, label, repo_id = classify(path), str(path), None

    env_python = args.env
    if env_python == "":
        sibling = Path.home() / "code/coreai/coreai-models/.venv/bin/python"
        env_python = str(sibling) if sibling.exists() else None
    rep = Report(target=label, kind=kind, host_build=args.host_build or host_os_build())
    check_env(rep, env_python)
    if kind == "asset":
        check_asset(path, rep, profile=args.profile)
    elif kind == "bundle":
        check_bundle(path, rep, args.profile)
    elif kind == "checkpoint":
        check_checkpoint(path, rep, args.block_size, repo_id)
    elif kind == "source":
        check_source_files([path] if path.is_file() else sorted(path.glob("*.py")), rep)

    extra: list[Path] = []
    for spec in args.torch_src:
        p = Path(spec)
        extra.extend([p] if p.is_file() else sorted(p.rglob("*.py")))
    if extra:
        check_source_files(extra, rep)

    print(to_json(rep) if args.json else "", end="")
    if not args.json:
        render(rep, args.verbose)

    worst = min((SEVERITY_ORDER[f.rule.severity] for f in rep.defects()), default=99)
    raise SystemExit(2 if worst <= 1 else (1 if worst <= 3 else 0))


if __name__ == "__main__":
    main()
