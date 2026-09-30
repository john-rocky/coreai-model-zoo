#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["coreai-core==1.0.0b2", "numpy>=2.2", "tokenizers>=0.22"]
# ///
"""Julia-1 on Core AI from Python: the reference host (text in, the publisher's typed answers out).

    python3 julia_coreai.py <folder> --state "I was charged twice for the same order." \\
        --questions '{"team": {"type": "choice", "instructions": "Which team should handle this request?",
                      "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery",
                                   "access": "Account access and login"}}}'

`<folder>` is one exported variant folder (e.g. macos/fp32-s1024/ of mlboydaisuke/Julia-1-CoreAI): the
.aimodel, tokenizer/ and metadata.json. The same call shape as the publisher's
`engine.predict(state=..., questions=...)`: every named question becomes one row (the criteria text as the
options, julia/typed.py), one `main` call per question, and the answer is julia/typed.py's — full softmax
probabilities at T = 1, choice = the winning id, score = the expected rubric index, noul = p[true].
Strict encoding: a question that does not fit the folder's window is refused, never cut.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _julia_host import Tokenizer, answer, build_row, gather_markers, graph_inputs, render_named  # noqa: E402


class JuliaCoreAI:
    """One loaded variant folder. `compute` is "gpu" (what ships) or "cpu_only" (the parity option)."""

    def __init__(self, folder: str | Path, compute: str = "gpu"):
        self.folder = Path(folder)
        metadata = json.loads((self.folder / "metadata.json").read_text())
        decision = metadata["decision"]
        if decision.get("layout") != "julia":
            raise ValueError(f"{self.folder} is not a Julia decision bundle (layout {decision.get('layout')!r})")
        self.window, self.head_length = decision["window"], decision["head_max_len"]
        self.tokenizer = Tokenizer(self.folder / "tokenizer" / "tokenizer.json")
        options = (rt.SpecializationOptions.cpu_only() if compute == "cpu_only" else
                   rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        self._loop = asyncio.new_event_loop()
        self._model = self._loop.run_until_complete(rt.AIModel.load(self.folder / metadata["assets"]["main"], options))
        self._main = self._model.load_function(decision["functions"]["main"])

    def row(self, request: dict) -> dict:
        """The publisher's row for one request {state, question, options, type} (strict)."""
        return build_row(self.tokenizer, request, self.window, self.head_length, strict=True)

    def logits(self, request: dict) -> list[float]:
        """The raw option logits for one request, in option order (the publisher's engine.logits)."""
        row = self.row(request)
        inputs = {k: rt.NDArray(v) for k, v in graph_inputs(row, self.window).items()}
        out = self._loop.run_until_complete(self._main(inputs))
        return gather_markers(out["token_logits"].numpy(), row["markers"])

    def predict(self, state, questions: dict) -> dict:
        """The publisher's predict(state=..., questions=...): {"answers": {id: answer}}."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty mapping")
        answers = {}
        for question_id, question in questions.items():
            if not isinstance(question_id, str) or not question_id or not isinstance(question, dict):
                raise ValueError("Questions require nonempty string IDs and question objects")
            kind, keys, labels = render_named(question)
            request = {"state": state, "question": question.get("instructions"), "type": kind, "options": labels}
            answers[question_id] = answer(kind, keys, self.logits(request))
        return {"answers": answers}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--state", required=True, help="text, or JSON when it parses as an object or a list")
    parser.add_argument("--questions", required=True, help="the named questions, JSON")
    parser.add_argument("--compute", choices=["gpu", "cpu_only"], default="gpu")
    args = parser.parse_args()
    try:
        state = json.loads(args.state)
        state = state if isinstance(state, (dict, list)) else args.state
    except json.JSONDecodeError:
        state = args.state
    julia = JuliaCoreAI(args.folder, args.compute)
    print(json.dumps(julia.predict(state, json.loads(args.questions)), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
