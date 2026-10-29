#!/usr/bin/env python3
"""Write the chat template an LFM3 Core AI bundle carries: the vendor's chat_template.jinja with eight rewrites.

    python3 lfm3_chat_template.py OUT [--vendor PATH] [--repo LiquidAI/LFM3-1.7B] [--revision REV]

OUT is the file to write; hand it to the exporter as `--chat-template OUT` (export_lfm3_decode_pipelined.py copies
it into the bundle's tokenizer/ as chat_template.jinja). The vendor template comes from the Hugging Face snapshot
of --repo at --revision (hf_hub_download: the local cache first, the Hub when the file is not there) unless
--vendor names a file.

No rewrite changes a string transformers' jinja2 renders. They make the template parse, and print JSON the same
way, in swift-jinja, which the Swift runtimes render it with (knowledge/lfm3-1.7b-port.md, "The chat template"):
  R1     `for tool in tools or []` -> `for tool in (tools or [])`. swift-jinja 2.3.2 (the pin of
         apple/coreai-models) parses a for-loop iterable as a primary expression only ("Expected '%}' after for
         loop.. Got or instead"; fixed in 2.3.6, swift-jinja #59). In parentheses it is one in every engine, and
         jinja2 evaluates the same value.
  R2     a `_json(v)` macro after the enable_thinking line. swift-jinja 2.3.x's tojson is a JSONEncoder (sorted
         keys, no spaces, non-ASCII and '/' escaped); transformers' is json.dumps(ensure_ascii=False). `_json`
         calls tojson where tojson already prints json.dumps' format (jinja2 under transformers: the vendor's own
         filter call, so the same string for every input) and writes that format out otherwise.
  R3-R8  the six lines holding the seven `x | tojson` calls -> `_json(x)`.
Each rewrite is one exact-substring replacement that must match once: a template edit that silently does nothing
is the failure this guards against. Prints the sha256 of the vendor file and of OUT. The vendor file of
LiquidAI/LFM3-1.7B at ac2586a3 (sha256 f45005c5...) must give the template mlboydaisuke/LFM3-1.7B-CoreAI ships
(48f23f85...); the script exits 1 when it does not.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = "LiquidAI/LFM3-1.7B"
REVISION = "ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7"
# The vendor template at REVISION, and what the rewrites make of it (the bundle's tokenizer/chat_template.jinja).
VENDOR_SHA256 = "f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818"
BUNDLE_SHA256 = "48f23f8579dda4c39f48ff0a9c5f22e68e6b3267898dc9316e9b4161e56643be"

JSON_MACRO = r'''{%- macro _json(v) -%}
    {%- if ([1, 2] | tojson) == "[1, 2]" -%}
        {{- v | tojson -}}
    {%- elif v is string -%}
        {{- '"' + (v | replace('\\', '\\\\') | replace('"', '\\"') | replace('\n', '\\n') | replace('\r', '\\r') | replace('\t', '\\t') | replace('\b', '\\b') | replace('\f', '\\f')) + '"' -}}
    {%- elif v is boolean -%}
        {{- "true" if v else "false" -}}
    {%- elif v is none -%}
        {{- "null" -}}
    {%- elif v is mapping -%}
        {%- set _jn = namespace(out="{", first=true) -%}
        {%- for _jk, _jv in v.items() -%}
            {%- set _jn.out = _jn.out + ("" if _jn.first else ", ") + _json(_jk | string) + ": " + _json(_jv) -%}
            {%- set _jn.first = false -%}
        {%- endfor -%}
        {{- _jn.out + "}" -}}
    {%- elif v is sequence -%}
        {%- set _jn = namespace(out="[", first=true) -%}
        {%- for _jx in v -%}
            {%- set _jn.out = _jn.out + ("" if _jn.first else ", ") + _json(_jx) -%}
            {%- set _jn.first = false -%}
        {%- endfor -%}
        {{- _jn.out + "]" -}}
    {%- else -%}
        {{- v | string -}}
    {%- endif -%}
{%- endmacro -%}

'''

REWRITES = [
    ("R1", "{%- for tool in tools or [] -%}", "{%- for tool in (tools or []) -%}"),
    ("R2", "{%- set enable_thinking = enable_thinking | default(true) -%}\n\n",
     "{%- set enable_thinking = enable_thinking | default(true) -%}\n\n" + JSON_MACRO),
    ("R3", "        {{- v | tojson -}}\n    {%- else -%}\n        {{- v -}}",
     "        {{- _json(v) -}}\n    {%- else -%}\n        {{- v -}}"),
    ("R4", "    {%- elif content is mapping -%}\n        {{- content | tojson -}}",
     "    {%- elif content is mapping -%}\n        {{- _json(content) -}}"),
    ("R5", "{%- set _ns.result = _ns.result + (item | tojson) -%}", "{%- set _ns.result = _ns.result + _json(item) -%}"),
    ("R6", "    {%- elif content is not none -%}\n        {{- content | tojson -}}\n    {%- endif -%}\n{%- endmacro -%}\n\n{%- macro render_tool_calls",
     "    {%- elif content is not none -%}\n        {{- _json(content) -}}\n    {%- endif -%}\n{%- endmacro -%}\n\n{%- macro render_tool_calls"),
    ("R7", "{{- k | tojson -}}{{- \": \" -}}{{- v | tojson -}}", "{{- _json(k) -}}{{- \": \" -}}{{- _json(v) -}}"),
    ("R8", "    {%- elif content is not none -%}\n        {{- content | tojson -}}\n    {%- endif -%}\n    {{- \"\\n</tool_response>\" -}}",
     "    {%- elif content is not none -%}\n        {{- _json(content) -}}\n    {%- endif -%}\n    {{- \"\\n</tool_response>\" -}}"),
]


def rewrite(text: str) -> str:
    out = text
    for tag, old, new in REWRITES:
        n = out.count(old)
        if n != 1:
            raise SystemExit(f"{tag}: expected 1 match in the vendor template, found {n}")
        out = out.replace(old, new)
    # The only tojson calls left are the two inside _json itself.
    rest = out.replace("{{- v | tojson -}}", "", 1).replace("([1, 2] | tojson)", "", 1)
    if "| tojson" in rest:
        raise SystemExit("a tojson call site was left unrewritten")
    return out


def vendor_template(path: str | None, repo: str, revision: str) -> Path:
    if path:
        return Path(path)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, "chat_template.jinja", revision=revision))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="the template file to write (the exporter's --chat-template)")
    ap.add_argument("--vendor", default=None,
                    help="the vendor chat_template.jinja (default: from the Hugging Face snapshot of --repo)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--revision", default=REVISION)
    args = ap.parse_args()

    src = vendor_template(args.vendor, args.repo, args.revision)
    raw = src.read_bytes()
    out = rewrite(raw.decode("utf-8")).encode("utf-8")
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(out)
    print(f"vendor {sha256(raw)} {len(raw)} B  {src}")
    print(f"bundle {sha256(out)} {len(out)} B -> {dest}")
    if sha256(raw) != VENDOR_SHA256:
        print(f"NOTE: not the vendor template the shipped bundle was built from ({VENDOR_SHA256[:8]}...); every "
              "rewrite matched once, but read the result before shipping it")
    elif sha256(out) != BUNDLE_SHA256:
        print(f"FAIL: the pinned vendor template must give {BUNDLE_SHA256}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
