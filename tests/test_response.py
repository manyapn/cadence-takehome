from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cases import ROOT, load_case
from core import PatientSubmission
from observations import ConsentObservation, VitalObservation
from pipeline import evaluate
from rules import Check, TriageResult
from verify import verify

FIXTURES = sorted((ROOT / "tests" / "fixtures").glob("case_*_observations.json"))
CHECK_NAMES = {
    "history_and_physical",
    "surgical_consent",
    "cbc",
    "cmp",
    "anticoagulation_plan",
    "blood_pressure",
    "temperature",
}
RECHECK = "2026-03-27T09:00:00Z"
SIGNED_CONSENT = {
    "doc_id": "signed-consent",
    "type": "Surgery Consent",
    "date": "2026-03-27",
    "text": "Procedure: thyroidectomy\nPatient signature: Ethan Lopez, signed in clinic 03/27/2026.",
}


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)


def check_named(result: TriageResult, name: str) -> Check:
    return next(check for check in result.checks if check.name == name)


def categories(result: TriageResult) -> list[str]:
    return sorted(issue.category for issue in result.issues)


def recheck(systolic: int = 120, diastolic: int = 80) -> dict[str, object]:
    return {"type": "blood_pressure", "systolic": systolic, "diastolic": diastolic, "date": RECHECK, "source": "Pre-op clinic"}


def test_case_28_fixture_verifies() -> None:
    submission, _, observations = load_case(28)

    assert verify(PatientSubmission.model_validate(submission), observations).dropped == []


def test_case_28_as_written_is_not_cleared_with_both_issues() -> None:
    submission, label, observations = load_case(28)

    result = evaluate(submission, observations)

    assert result.decision == label["decision"] == "NOT_CLEARED"
    assert categories(result) == sorted(label["categories"]) == ["ACUTE_SAFETY_EXCLUSION", "REQUIRED_DOCUMENTATION"]
    assert "148/110 mmHg" in check_named(result, "blood_pressure").summary
    assert check_named(result, "surgical_consent").status == "fail"


def test_case_28_with_a_later_normal_blood_pressure_needs_only_the_consent() -> None:
    submission, _, observations = load_case(28)
    submission["vitals"].append(recheck())

    result = evaluate(submission, observations)

    assert result.decision == "NEEDS_FOLLOW_UP"
    assert categories(result) == ["REQUIRED_DOCUMENTATION"]
    assert "supersedes 148/110 mmHg" in check_named(result, "blood_pressure").summary


def test_case_28_with_a_later_normal_blood_pressure_and_a_signed_consent_is_ready() -> None:
    submission, _, observations = load_case(28)
    submission["vitals"].append(recheck())
    submission["documents"].append(SIGNED_CONSENT)
    observations.consents.append(
        ConsentObservation(
            doc_id="signed-consent",
            patient_signed="yes",
            signature_quote="Patient signature: Ethan Lopez, signed in clinic 03/27/2026.",
            procedure_named="thyroidectomy",
            procedure_quote="Procedure: thyroidectomy",
        )
    )

    result = evaluate(submission, observations)

    assert result.unverified == []
    assert result.decision == "READY"
    assert result.issues == []


@pytest.mark.parametrize(
    ("systolic", "diastolic", "status", "decision"),
    [
        (179, 80, "pass", "NEEDS_FOLLOW_UP"),
        (180, 80, "fail", "NOT_CLEARED"),
        (120, 109, "pass", "NEEDS_FOLLOW_UP"),
        (120, 110, "fail", "NOT_CLEARED"),
    ],
)
def test_blood_pressure_excludes_at_180_systolic_or_110_diastolic(
    systolic: int, diastolic: int, status: str, decision: str
) -> None:
    submission, _, observations = load_case(28)
    submission["vitals"].append(recheck(systolic, diastolic))

    result = evaluate(submission, observations)

    assert check_named(result, "blood_pressure").status == status
    assert result.decision == decision


@pytest.mark.parametrize(
    ("value_f", "status", "decision"),
    [(100.4, "pass", "NEEDS_FOLLOW_UP"), (100.5, "fail", "NOT_CLEARED")],
)
def test_temperature_excludes_only_above_100_4_fahrenheit(value_f: float, status: str, decision: str) -> None:
    submission, _, observations = load_case(28)
    submission["vitals"] += [recheck(), {"type": "temperature", "value_f": value_f, "date": RECHECK, "source": "Pre-op clinic"}]

    result = evaluate(submission, observations)

    assert check_named(result, "temperature").status == status
    assert result.decision == decision


@pytest.mark.parametrize(
    ("celsius", "status", "decision"),
    [(38.0, "pass", "NEEDS_FOLLOW_UP"), (38.1, "fail", "NOT_CLEARED")],
)
def test_celsius_note_reading_converts_exactly_at_the_limit(celsius: float, status: str, decision: str) -> None:
    submission, _, observations = load_case(28)
    submission["vitals"].append(recheck())
    submission["documents"].append(
        {"doc_id": "recheck-note", "type": "Nursing Note", "date": RECHECK[:10], "text": f"Recheck 09:00  Temp {celsius} C (oral)"}
    )
    observations.vitals.append(
        VitalObservation(
            doc_id="recheck-note",
            kind="temperature",
            systolic=None,
            diastolic=None,
            temperature=celsius,
            unit="C",
            time="09:00",
            quote=f"Recheck 09:00  Temp {celsius} C",
        )
    )

    result = evaluate(submission, observations)

    temperature = check_named(result, "temperature")
    assert result.unverified == []
    assert temperature.evidence[0].reference == "recheck-note"
    assert temperature.status == status
    assert result.decision == decision


def test_chart_with_no_temperature_is_a_chart_gap() -> None:
    submission, _, observations = load_case(16)
    submission["vitals"] = [vital for vital in submission["vitals"] if vital["type"] != "temperature"]
    observations.vitals = [vital for vital in observations.vitals if vital.kind != "temperature"]

    result = evaluate(submission, observations)

    temperature = check_named(result, "temperature")
    assert (temperature.status, temperature.unknown_cause) == ("unknown", "chart_gap")
    assert temperature.sources_checked
    (issue,) = result.issues
    assert (issue.category, issue.unknown_cause) == ("MISSING_REQUIRED_DATA", "chart_gap")
    assert issue.sources_checked == temperature.sources_checked
    assert result.decision == "NEEDS_FOLLOW_UP"


def test_fabricated_quote_on_the_latest_temperature_is_a_processing_failure() -> None:
    submission, _, observations = load_case(16)
    latest = next(vital for vital in observations.vitals if vital.temperature == 37.1)
    latest.quote = "08:06  T 37.1 C"

    result = evaluate(submission, observations)

    temperature = check_named(result, "temperature")
    assert (temperature.status, temperature.unknown_cause) == ("unknown", "processing_failure")
    (issue,) = result.issues
    assert (issue.category, issue.unknown_cause) == ("MISSING_REQUIRED_DATA", "processing_failure")
    (dropped,) = result.unverified
    assert (dropped.kind, dropped.doc_id, dropped.reason) == ("vital", latest.doc_id, "quote not found in document")
    assert dropped.claim["temperature"] == 37.1
    assert result.decision == "NEEDS_FOLLOW_UP"


def test_verified_exclusion_survives_a_dropped_consent() -> None:
    submission, _, observations = load_case(28)
    observations.consents[0].signature_quote = "Patient signature on file."

    result = evaluate(submission, observations)

    consent = check_named(result, "surgical_consent")
    assert (consent.status, consent.unknown_cause) == ("unknown", "processing_failure")
    assert [fact.kind for fact in result.unverified] == ["consent"]
    assert check_named(result, "blood_pressure").status == "fail"
    assert result.decision == "NOT_CLEARED"
    assert categories(result) == ["ACUTE_SAFETY_EXCLUSION", "MISSING_REQUIRED_DATA"]


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.stem[:10])
def test_response_invariants_hold_for_every_fixture(fixture: Path) -> None:
    submission, _, observations = load_case(int(fixture.stem.split("_")[1]))
    documents = {document["doc_id"]: document["text"] for document in submission["documents"]}

    result = evaluate(submission, observations)

    issue_categories = {issue.category for issue in result.issues}
    assert (result.decision == "NOT_CLEARED") == ("ACUTE_SAFETY_EXCLUSION" in issue_categories)
    assert (result.decision == "READY") == (result.issues == [])
    assert sorted(check.name for check in result.checks) == sorted(CHECK_NAMES)
    for check in result.checks:
        assert (check.status == "unknown") == (check.unknown_cause is not None), check.name
    assert len(result.issues) == sum(check.status in ("fail", "unknown") for check in result.checks)
    for issue in result.issues:
        assert issue.evidence or issue.sources_checked, issue.finding
    for item in (item for check in result.checks for item in check.evidence):
        if item.source == "note" and item.quote:
            assert item.quote in documents[item.reference], item.quote
    assert result.explanation.splitlines()[0] == result.decision
