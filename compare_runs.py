#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic>=2.8.0",
# ]
# ///

"""Score saved output files against the labels and print one row per file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BLOOD_PRESSURE = re.compile(r"(?<![\d/])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/]|\.\d)")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_note_only_blood_pressure(submission: dict) -> tuple[int, int] | None:
    """The last BP written in the newest note that has one, when no structured BP is as recent.

    A regex over the sample's note formats. It is the answer key for this check, not part of the system.
    """

    by_date: dict[str, list[tuple[int, int]]] = {}
    for document in submission["documents"]:
        pairs = [(int(s), int(d)) for s, d in BLOOD_PRESSURE.findall(document["text"])]
        pairs = [(s, d) for s, d in pairs if 50 <= s <= 300 and 20 <= d < s]
        if pairs:
            by_date.setdefault(document["date"], []).extend(pairs)
    if not by_date:
        return None
    newest = max(by_date)
    structured = [vital["date"][:10] for vital in submission["vitals"] if vital["type"] == "blood_pressure"]
    return by_date[newest][-1] if newest > max(structured, default="") else None


def score(cases: list[dict], rows: list[dict]) -> dict[str, object]:
    outputs = {row["record_index"]: row.get("output") for row in rows}
    summary = dict.fromkeys(
        ("valid", "decision", "categories", "both", "exclusions_caught", "non_ready_returned_ready", "ready_returned_ready"), 0
    )
    note_only = {"cases": 0, "correct": 0, "wrong": []}
    quotes = {"checked": 0, "not_verbatim": []}
    misses = []

    for index, case in enumerate(cases):
        label, output = case["label"], outputs.get(index)
        valid = isinstance(output, dict) and "decision" in output and isinstance(output.get("issues"), list)
        decision = output["decision"] if valid else "INVALID"
        categories = sorted({issue["category"] for issue in output["issues"]}) if valid else []
        decision_ok = decision == label["decision"]
        categories_ok = valid and categories == sorted(set(label["categories"]))
        summary["valid"] += valid
        summary["decision"] += decision_ok
        summary["categories"] += categories_ok
        summary["both"] += decision_ok and categories_ok
        summary["exclusions_caught"] += label["decision"] == "NOT_CLEARED" and decision == "NOT_CLEARED"
        summary["non_ready_returned_ready"] += label["decision"] != "READY" and decision == "READY"
        summary["ready_returned_ready"] += label["decision"] == "READY" and decision == "READY"
        if not (decision_ok and categories_ok):
            misses.append({"case": index, "label": [label["decision"], *sorted(label["categories"])], "got": [decision, *categories]})

        checks = {check["name"]: check for check in output.get("checks", [])} if valid else {}
        if not checks:
            continue
        expected = latest_note_only_blood_pressure(case["submission"])
        if expected:
            note_only["cases"] += 1
            governing = [item.get("value") for item in checks["blood_pressure"]["evidence"][:1]]
            if governing == [f"{expected[0]}/{expected[1]} mmHg"]:
                note_only["correct"] += 1
            else:
                note_only["wrong"].append({"case": index, "expected": expected, "got": governing})
        texts = {document["doc_id"]: document["text"] for document in case["submission"]["documents"]}
        for check in checks.values():
            for item in check["evidence"]:
                if item["source"] == "note" and item.get("quote"):
                    quotes["checked"] += 1
                    if item["quote"] not in texts.get(item["reference"], ""):
                        quotes["not_verbatim"].append({"case": index, "quote": item["quote"]})

    result = {**summary, "misses": misses}
    if note_only["cases"] or quotes["checked"]:
        result |= {"latest_bp_note_only": note_only, "evidence_quotes": quotes}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs", nargs="+", help="Output JSONL files written by run_baseline.py")
    parser.add_argument("--input", default=str(ROOT / "data" / "patients_sample_50.jsonl"))
    parser.add_argument("--summary", help="Write the full summary here as JSON")
    args = parser.parse_args()

    cases = load_jsonl(Path(args.input))
    summaries = {path: score(cases, load_jsonl(Path(path))) for path in args.outputs}

    columns = ("valid", "decision", "categories", "both", "exclusions_caught", "non_ready_returned_ready", "ready_returned_ready")
    print("| file | " + " | ".join(columns) + " | latest BP (note-only) | quotes not verbatim |")
    print("|---|" + "---|" * (len(columns) + 2))
    for path, summary in summaries.items():
        note_only, quotes = summary.get("latest_bp_note_only"), summary.get("evidence_quotes")
        print(
            f"| {Path(path).name} | "
            + " | ".join(str(summary[column]) for column in columns)
            + f" | {f'''{note_only['correct']}/{note_only['cases']}''' if note_only else 'n/a'}"
            + f" | {f'''{len(quotes['not_verbatim'])}/{quotes['checked']}''' if quotes else 'n/a'} |"
        )
    if args.summary:
        Path(args.summary).write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
