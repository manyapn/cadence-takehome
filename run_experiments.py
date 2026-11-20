#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "openai>=2.0.0",
#   "pydantic>=2.8.0",
# ]
# ///

"""Step 10 experiments: one call vs three focused calls, verification on vs off, and perturbed charts.

  capture   one extraction and three focused extractions per case, saved (200 model calls)
  perturb   8 cases, each reworded and each with its deciding fact changed (16 model calls)
  report    scores whatever has been captured (no model calls)
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from pydantic import create_model

from core import PatientSubmission
from extract import EXTRACTION_PROMPT, extract
from observations import Observations, Reading, VerifiedObservations, VitalObservation
from pipeline import evaluate
from resolve import resolve_anticoagulants, resolve_vitals
from rules import TriageResult, respond, run_checks
from verify import find_quote, to_fahrenheit

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments"
FOCUS = (
    ("vitals",),
    ("hp_documents", "consents", "scheduling_statements"),
    ("anticoagulant_mentions", "anticoagulation_plans"),
)


def load_cases() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data" / "patients_sample_50.jsonl").read_text().splitlines() if line]


# -------------------------
# One call vs three focused calls
# -------------------------


def focused_prompt(fields: tuple[str, ...]) -> str:
    blocks = EXTRACTION_PROMPT.split("\n\n")
    kept = [block for block in blocks if block.split("\n")[0] not in Observations.model_fields or block.split("\n")[0] in fields]
    return "\n\n".join(kept)


def extract_focused(submission: PatientSubmission, model: str) -> Observations:
    merged = {}
    for fields in FOCUS:
        schema = create_model(
            "_".join(fields), **{name: (Observations.model_fields[name].annotation, ...) for name in fields}
        )
        merged |= extract(submission, model=model, instructions=focused_prompt(fields), schema=schema).model_dump()
    return Observations.model_validate(merged)


def capture(model: str) -> None:
    def one(case: dict) -> dict:
        submission = PatientSubmission.model_validate(case["submission"])
        row = {"case_id": case["case_id"]}
        for arm, run in (("single", lambda: extract(submission, model=model)), ("focused", lambda: extract_focused(submission, model))):
            try:
                row[arm] = run().model_dump()
            except Exception as error:
                row[arm], row[f"{arm}_error"] = None, f"{type(error).__name__}: {error}"
        return row

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, load_cases()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "extractions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"captured {len(rows)} cases -> {OUT / 'extractions.jsonl'}")


# -------------------------
# Verification off
# -------------------------


def evaluate_without_verification(submission: dict, observations: Observations) -> TriageResult:
    """The same resolver and rules, with every extracted observation taken at its word."""

    parsed = PatientSubmission.model_validate(submission)
    documents = {document.doc_id: document for document in parsed.documents}
    readings = []
    for vital in observations.vitals:
        document = documents.get(vital.doc_id)
        values = (vital.systolic, vital.diastolic) if vital.kind == "blood_pressure" else (vital.temperature, vital.unit)
        if document is None or None in values:
            continue
        found = find_quote(vital.quote, document.text or "")
        readings.append(
            Reading(
                kind=vital.kind,
                systolic=vital.systolic if vital.kind == "blood_pressure" else None,
                diastolic=vital.diastolic if vital.kind == "blood_pressure" else None,
                temperature_f=to_fahrenheit(Decimal(str(vital.temperature)), vital.unit) if vital.kind == "temperature" else None,
                source="note",
                reference=vital.doc_id,
                date=document.date,
                time=vital.time.zfill(5) if vital.time and re.fullmatch(r"\d{1,2}:\d{2}", vital.time) else None,
                quote=vital.quote,
                offset=found[0] if found else len(document.text or ""),
            )
        )
    trusted = VerifiedObservations(
        readings=readings,
        **{
            name: [item for item in getattr(observations, name) if item.doc_id in documents]
            for name in Observations.model_fields
            if name != "vitals"
        },
    )
    checks = run_checks(parsed, trusted, resolve_vitals(parsed, readings), resolve_anticoagulants(parsed, trusted))
    return respond(checks)


def with_fabricated_normal_reading(case: dict, observations: Observations, honest: TriageResult) -> Observations:
    """Add a made-up normal reading, later than everything else, for each vital that excludes the patient."""

    newest = max(case["submission"]["documents"], key=lambda document: document["date"])
    empty = {"systolic": None, "diastolic": None, "temperature": None, "unit": None, "time": None}
    made_up = {
        "blood_pressure": {"systolic": 118, "diastolic": 74, "quote": "Repeat blood pressure 118/74 after rest."},
        "temperature": {"temperature": 98.4, "unit": "F", "quote": "Repeat temperature 98.4 F."},
    }
    failing = [check.name for check in honest.checks if check.rule == "Rule 4" and check.status == "fail"]
    extra = [VitalObservation(doc_id=newest["doc_id"], kind=kind, **{**empty, **made_up[kind]}) for kind in failing]
    return observations.model_copy(update={"vitals": observations.vitals + extra})


# -------------------------
# Perturbed charts
# -------------------------

PERTURBATIONS = [
    # (case, text to find, reworded with the same meaning, deciding fact changed, decision expected after the change)
    (
        16,
        "ADDENDUM 08:30: Re-measured 137/78 mmHg following 24 minutes in a quiet room.",
        "ADDENDUM 08:30: Pressure taken again once the patient had sat quietly for 24 minutes, now 137/78.",
        "ADDENDUM 08:30: Re-measured 184/101 mmHg following 24 minutes in a quiet room.",
        "NOT_CLEARED",
    ),
    (
        35,
        "Obtained at 08:30: BP 121/77, T 38.7 C,",
        "Taken at 08:30: BP 121/77, temp 38.7 degrees Celsius,",
        "Obtained at 08:30: BP 121/77, T 37.2 C,",
        "READY",
    ),
    (
        47,
        "09:05  BP: 161/117, 37.2 °C tympanic,",
        "09:05  B/P 161/117 (left arm, seated), 37.2 °C tympanic,",
        "09:05  BP: 161/97, 37.2 °C tympanic,",
        "READY",
    ),
    (
        20,
        "Post-operative anticoagulation will be addressed after surgery.",
        "When to restart the blood thinner will be decided once the operation is done.",
        "Resume warfarin at the usual home dose the evening of surgery if hemostasis is adequate.",
        "READY",
    ),
    (
        10,
        "Patient signature: ______________________   Date/Time: __________\n[Form printed for patient review; not yet signed]",
        "Patient signature: ______________________   Date/Time: __________\n[Paper copy handed to the patient to read; she has not signed it]",
        "Patient signature: /s/ Ava Nguyen (electronically signed 03/08/2026 14:02)",
        "READY",
    ),
    (
        27,
        "Patient also reports taking Eliquis 5 mg (BID), prescribed by an outside cardiologist; not on the reconciled medication list - flagged for provider review.",
        "He mentions that an outside heart doctor has him on Eliquis 5 mg twice a day. It is missing from our medication list; provider asked to review.",
        "He mentions that an outside heart doctor had him on Eliquis until last year. He no longer takes it.",
        "READY",
    ),
    (
        19,
        "completed 6 months of warfarin, discontinued by hematology in early 2020 with no recurrence. Not currently anticoagulated.",
        "took warfarin for half a year; hematology ended it in early 2020 and there have been no clots since. Not on any blood thinner now.",
        "completed 6 months of warfarin in 2020; warfarin was restarted last month after a new clot and he takes it daily.",
        "NEEDS_FOLLOW_UP",
    ),
    (
        7,
        "Signed by patient: Danielle Lopez, 03/03/2026 16:44. Witness: Lauren Hernandez.",
        "Danielle Lopez signed this form on 03/03/2026 at 16:44 in front of witness Lauren Hernandez.",
        "Patient signature: pending",
        "NEEDS_FOLLOW_UP",
    ),
]


def perturbed(case: dict, find: str, replace: str) -> dict:
    submission = copy.deepcopy(case["submission"])
    hits = [document for document in submission["documents"] if find in document["text"]]
    assert len(hits) == 1 and hits[0]["text"].count(find) == 1, (case["case_id"], find)
    hits[0]["text"] = hits[0]["text"].replace(find, replace)
    return submission


def perturb(model: str) -> None:
    cases = load_cases()
    jobs = [
        (number, kind, perturbed(cases[number], find, text), expected)
        for number, find, reworded, changed, after in PERTURBATIONS
        for kind, text, expected in (("reworded", reworded, cases[number]["label"]["decision"]), ("fact_changed", changed, after))
    ]

    def one(job: tuple) -> dict:
        number, kind, submission, expected = job
        parsed = PatientSubmission.model_validate(submission)
        result = evaluate(parsed, extract(parsed, model=model))
        return {
            "case": number,
            "variant": kind,
            "original": cases[number]["label"]["decision"],
            "expected": expected,
            "got": result.decision,
            "categories": sorted({issue.category for issue in result.issues}),
            "open_checks": {check.name: check.summary for check in result.checks if check.status in ("fail", "unknown")},
            "unverified": [fact.model_dump() for fact in result.unverified],
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, jobs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "perturbations.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"ran {len(rows)} perturbed charts -> {OUT / 'perturbations.json'}")


# -------------------------
# Report
# -------------------------


def correct(result: TriageResult, label: dict) -> bool:
    return result.decision == label["decision"] and sorted({issue.category for issue in result.issues}) == sorted(label["categories"])


def report() -> None:
    cases = load_cases()
    summary: dict[str, object] = {}

    if (OUT / "extractions.jsonl").exists():
        rows = [json.loads(line) for line in (OUT / "extractions.jsonl").read_text().splitlines()]
        arms = {arm: {"correct": 0, "observations": 0, "dropped": 0, "failed_calls": 0, "misses": []} for arm in ("single", "focused")}
        off = {"correct": 0, "decisions_changed_by_verification": []}
        injected = {"exclusion_cases": 0, "still_not_cleared_with_verification": 0, "still_not_cleared_without": 0}
        for index, (case, row) in enumerate(zip(cases, rows)):
            for arm, stats in arms.items():
                if row[arm] is None:
                    stats["failed_calls"] += 1
                    continue
                observations = Observations.model_validate(row[arm])
                result = evaluate(case["submission"], observations)
                stats["observations"] += sum(len(getattr(observations, name)) for name in Observations.model_fields)
                stats["dropped"] += len(result.unverified)
                stats["correct"] += correct(result, case["label"])
                if not correct(result, case["label"]):
                    stats["misses"].append({"case": index, "got": [result.decision, *sorted({i.category for i in result.issues})]})
            if row["single"] is None:
                continue
            observations = Observations.model_validate(row["single"])
            verified, trusted = evaluate(case["submission"], observations), evaluate_without_verification(case["submission"], observations)
            off["correct"] += correct(trusted, case["label"])
            if trusted.decision != verified.decision:
                off["decisions_changed_by_verification"].append({"case": index, "with": verified.decision, "without": trusted.decision})
            if case["label"]["decision"] == "NOT_CLEARED" and verified.decision == "NOT_CLEARED":
                tampered = with_fabricated_normal_reading(case, observations, verified)
                injected["exclusion_cases"] += 1
                injected["still_not_cleared_with_verification"] += evaluate(case["submission"], tampered).decision == "NOT_CLEARED"
                injected["still_not_cleared_without"] += evaluate_without_verification(case["submission"], tampered).decision == "NOT_CLEARED"
        summary["one_call_vs_three"] = arms
        summary["verification_off_on_real_extractions"] = off
        summary["fabricated_normal_reading_added_to_exclusions"] = injected

    if (OUT / "perturbations.json").exists():
        rows = json.loads((OUT / "perturbations.json").read_text())
        summary["perturbations"] = {
            kind: {
                "as_expected": sum(row["got"] == row["expected"] for row in rows if row["variant"] == kind),
                "total": sum(row["variant"] == kind for row in rows),
                "not_as_expected": [
                    {key: row[key] for key in ("case", "expected", "got", "categories")}
                    for row in rows
                    if row["variant"] == kind and row["got"] != row["expected"]
                ],
            }
            for kind in ("reworded", "fact_changed")
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["capture", "perturb", "report"])
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()
    {"capture": lambda: capture(args.model), "perturb": lambda: perturb(args.model), "report": report}[args.experiment]()
