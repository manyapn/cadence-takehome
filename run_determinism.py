#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Run one case many times, keep every output, and report how often the runs agree.

`make determinism` reports 100% decision stability when every run fails the same way
and keeps no outputs. This counts only valid outputs and saves them so a difference can be traced.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from core import triage_submission

ROOT = Path(__file__).resolve().parent


def agreement(values: list[str]) -> int:
    return max(Counter(values).values(), default=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(ROOT / "data" / "patients_sample_50.jsonl"))
    parser.add_argument("--record-index", type=int, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "determinism"))
    args = parser.parse_args()

    cases = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    case = cases[args.record_index]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs, errors = [], []
    for _ in range(args.runs):
        try:
            outputs.append(triage_submission(case["submission"], model=args.model).model_dump(mode="json"))
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")

    canonical = lambda value: json.dumps(value, sort_keys=True)
    summary = {
        "case_id": case["case_id"],
        "label": case["label"]["decision"],
        "model": args.model,
        "runs": args.runs,
        "valid_outputs": len(outputs),
        "decision_agreement": agreement([output["decision"] for output in outputs]),
        "issue_agreement": agreement([canonical(sorted(issue["category"] for issue in output["issues"])) for output in outputs]),
        "exact_output_agreement": agreement([canonical(output) for output in outputs]),
        "decisions": dict(Counter(output["decision"] for output in outputs)),
        "runs_with_unverified_facts": sum(bool(output["unverified"]) for output in outputs),
        "errors": errors,
    }
    (output_dir / f"{case['case_id']}_outputs.jsonl").write_text("".join(canonical(output) + "\n" for output in outputs))
    (output_dir / f"{case['case_id']}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
