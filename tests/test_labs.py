from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from cases import ROOT, load_case
from core import LabResult, PatientSubmission
from pipeline import evaluate
from resolve import lab_test
from rules import Check, TriageResult
from verify import verify

LAB_TESTS = {
    ("CBC", "Complete Blood Count"): "CBC",
    ("CBC", "Hemogram/CBC"): "CBC",
    ("LAB-CBC", "Complete Blood Count w/ Differential"): "CBC",
    ("CMP", "Comprehensive Metabolic Panel"): "CMP",
    ("CMP", "CMP (Comprehensive Metabolic Panel)"): "CMP",
    ("LAB-CMP", "Comprehensive Metabolic Panel w/ Magnesium"): "CMP",
    ("BMP", "Basic Metabolic Panel"): None,
    ("LAB-BMP", "Basic Metabolic Panel w/ eGFR"): None,
    ("HBA1C", "Hemoglobin A1c"): None,
    ("PT-INR", "Prothrombin Time with INR"): None,
    ("T&S", "Type and Screen"): None,
    ("TSH", "Thyroid Stimulating Hormone"): None,
    ("UA", "Urinalysis with Reflex Microscopy"): None,
}
PROCEDURE_DATE = date(2026, 3, 17)


def check_named(result: TriageResult, name: str) -> Check:
    return next(check for check in result.checks if check.name == name)


def failed(result: TriageResult) -> list[str]:
    return [check.name for check in result.checks if check.status == "fail"]


def lab(test: str, days_before: int, status: str = "final") -> dict[str, object]:
    return {
        "id": f"{test}-{days_before}",
        "code": test,
        "display": test,
        "effective_at": f"{PROCEDURE_DATE - timedelta(days=days_before)}T08:00:00Z",
        "status": status,
    }


def case_16_with(risk: str, *labs: dict[str, object]) -> TriageResult:
    submission, _, observations = load_case(16)
    assert submission["procedure"]["procedure_date"] == str(PROCEDURE_DATE)
    submission["procedure"]["procedure_risk"] = risk
    submission["labs"] = list(labs)
    return evaluate(submission, observations)


@pytest.mark.parametrize("number", [4, 6, 17, 41, 11, 29])
def test_case_matches_its_label(number: int) -> None:
    submission, label, observations = load_case(number)

    assert verify(PatientSubmission.model_validate(submission), observations).dropped == []
    result = evaluate(submission, observations)
    assert result.decision == label["decision"]
    assert {issue.category for issue in result.issues} == set(label["categories"])


def test_every_lab_name_in_the_sample_maps_to_cbc_cmp_or_neither() -> None:
    with (ROOT / "data" / "patients_sample_50.jsonl").open() as handle:
        pairs = {(entry["code"], entry["display"]) for case in map(json.loads, handle) for entry in case["submission"]["labs"]}

    assert pairs == set(LAB_TESTS)
    for (code, display), test in LAB_TESTS.items():
        assert lab_test(LabResult(code=code, display=display)) == test, (code, display)


@pytest.mark.parametrize(("number", "days"), [(4, 24), (41, 11)])
def test_entered_in_error_cbc_falls_back_to_the_prior_final_one(number: int, days: int) -> None:
    submission, _, observations = load_case(number)

    cbc = check_named(evaluate(submission, observations), "cbc")

    assert cbc.status == "pass"
    assert f"is {days} days before the procedure" in cbc.summary
    assert "is marked entered-in-error and was not counted" in cbc.summary


def test_case_6_fails_because_the_cbc_left_after_the_entered_in_error_one_is_too_old() -> None:
    submission, _, observations = load_case(6)

    result = evaluate(submission, observations)

    (issue,) = result.issues
    assert issue.category == "REQUIRED_TESTING"
    assert failed(result) == ["cbc"]
    assert "Latest CBC (2026-01-22) is 44 days before the procedure (limit 30 for MODERATE risk)" in issue.finding
    assert "A newer CBC dated 2026-03-04 is marked entered-in-error and was not counted." in issue.finding


def test_case_17_bmp_does_not_stand_in_for_the_cmp_a_high_risk_procedure_needs() -> None:
    submission, _, observations = load_case(17)

    result = evaluate(submission, observations)

    assert failed(result) == ["cmp"]
    assert check_named(result, "cmp").summary == "No final CMP result among 3 lab results."
    assert check_named(result, "cbc").status == "pass"


@pytest.mark.parametrize(
    ("risk", "labs", "decision", "failing"),
    [
        ("LOW", [lab("CBC", 30)], "READY", []),
        ("LOW", [lab("CBC", 31)], "NEEDS_FOLLOW_UP", ["cbc"]),
        ("MODERATE", [lab("CBC", 30)], "READY", []),
        ("MODERATE", [lab("CBC", 31)], "NEEDS_FOLLOW_UP", ["cbc"]),
        ("HIGH", [lab("CBC", 14), lab("CMP", 14)], "READY", []),
        ("HIGH", [lab("CBC", 15), lab("CMP", 14)], "NEEDS_FOLLOW_UP", ["cbc"]),
        ("HIGH", [lab("CBC", 14), lab("CMP", 15)], "NEEDS_FOLLOW_UP", ["cmp"]),
        ("HIGH", [lab("CBC", 14)], "NEEDS_FOLLOW_UP", ["cmp"]),
    ],
)
def test_lab_window_on_both_sides_of_the_limit(risk: str, labs: list[dict[str, object]], decision: str, failing: list[str]) -> None:
    result = case_16_with(risk, *labs)

    assert result.decision == decision
    assert failed(result) == failing
    assert {issue.category for issue in result.issues} == ({"REQUIRED_TESTING"} if failing else set())


@pytest.mark.parametrize(
    ("labs", "decision"),
    [
        ([lab("CBC", 5), lab("CBC", 60)], "READY"),
        ([lab("CBC", 60), lab("CBC", 5)], "READY"),
        ([lab("CBC", 60)], "NEEDS_FOLLOW_UP"),
        ([lab("CBC", 60), lab("CBC", 5, "entered-in-error")], "NEEDS_FOLLOW_UP"),
        ([lab("CBC", 60), lab("CBC", 5, "preliminary")], "NEEDS_FOLLOW_UP"),
        ([lab("CBC", 60), lab("BMP", 5), lab("CMP", 5)], "NEEDS_FOLLOW_UP"),
        ([lab("CBC", 5, "entered-in-error")], "NEEDS_FOLLOW_UP"),
    ],
)
def test_only_the_most_recent_final_cbc_counts(labs: list[dict[str, object]], decision: str) -> None:
    result = case_16_with("LOW", *labs)

    assert result.decision == decision
    assert failed(result) == ([] if decision == "READY" else ["cbc"])


@pytest.mark.parametrize(("number", "missing", "unknown"), [(11, "procedure date", "cbc"), (29, "procedure risk", "cbc"), (29, "procedure risk", "cmp")])
def test_missing_date_or_risk_leaves_the_lab_window_unknown(number: int, missing: str, unknown: str) -> None:
    submission, _, observations = load_case(number)

    result = evaluate(submission, observations)

    check = check_named(result, unknown)
    assert (check.status, check.unknown_cause) == ("unknown", "chart_gap")
    assert f"{missing} is missing" in check.summary.lower()
    assert failed(result) == []
    assert {issue.category for issue in result.issues} == {"MISSING_REQUIRED_DATA"}
    assert {issue.unknown_cause for issue in result.issues} == {"chart_gap"}


def test_case_11_missing_date_also_leaves_the_h_and_p_window_unknown() -> None:
    submission, _, observations = load_case(11)

    history_and_physical = check_named(evaluate(submission, observations), "history_and_physical")

    assert (history_and_physical.status, history_and_physical.unknown_cause) == ("unknown", "chart_gap")


@pytest.mark.parametrize("number", [11, 29])
def test_absent_cbc_fails_even_when_the_date_or_risk_is_missing(number: int) -> None:
    submission, _, observations = load_case(number)
    submission["labs"] = [entry for entry in submission["labs"] if lab_test(LabResult(**entry)) != "CBC"]

    result = evaluate(submission, observations)

    cbc = check_named(result, "cbc")
    assert cbc.status == "fail"
    assert cbc.summary.startswith("No final CBC result")
    assert {issue.category for issue in result.issues} == {"REQUIRED_TESTING", "MISSING_REQUIRED_DATA"}
    assert result.decision == "NEEDS_FOLLOW_UP"
