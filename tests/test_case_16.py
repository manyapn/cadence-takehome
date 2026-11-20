from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from core import PatientSubmission
from observations import Observations
from pipeline import evaluate
from resolve import resolve_vitals
from verify import verify

ROOT = Path(__file__).resolve().parent.parent
RECHECK = "ADDENDUM 08:30"


@pytest.fixture
def submission() -> dict[str, object]:
    with (ROOT / "data" / "patients_sample_50.jsonl").open() as handle:
        cases = [json.loads(line) for line in handle]
    return next(case["submission"] for case in cases if case["case_id"] == "case_00016")


@pytest.fixture
def observations() -> Observations:
    return Observations.model_validate_json((ROOT / "tests" / "fixtures" / "case_00016_observations.json").read_text())


@pytest.fixture(autouse=True)
def no_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)


def test_every_fixture_observation_verifies(submission: dict[str, object], observations: Observations) -> None:
    assert verify(PatientSubmission.model_validate(submission), observations).dropped == []


def test_recheck_supersedes_the_arrival_reading(submission: dict[str, object], observations: Observations) -> None:
    parsed = PatientSubmission.model_validate(submission)
    resolution = resolve_vitals(parsed, verify(parsed, observations).readings)["blood_pressure"]

    (governing,) = resolution.governing
    assert (governing.systolic, governing.diastolic, governing.time) == (137, 78, "08:30")
    (arrival,) = [entry for entry in resolution.superseded if entry.reading.systolic == 189]
    assert arrival.reading.time == "08:06"
    assert (arrival.by.systolic, arrival.by.time) == (137, "08:30")
    assert arrival.reason == "later in the same note"


def test_case_is_ready_with_no_issues(submission: dict[str, object], observations: Observations) -> None:
    result = evaluate(submission, observations)

    blood_pressure = next(check for check in result.checks if check.name == "blood_pressure")
    assert blood_pressure.status == "pass"
    assert "supersedes 189/112 mmHg" in blood_pressure.summary
    assert result.decision == "READY"
    assert result.issues == []


def test_output_does_not_depend_on_input_order(submission: dict[str, object], observations: Observations) -> None:
    reordered_submission = {**submission, "vitals": submission["vitals"][::-1]}
    reordered_observations = Observations(
        **{name: getattr(observations, name)[::-1] for name in Observations.model_fields}
    )

    assert evaluate(reordered_submission, reordered_observations) == evaluate(submission, observations)


def test_without_the_recheck_the_arrival_reading_excludes(submission: dict[str, object], observations: Observations) -> None:
    observations.vitals = [vital for vital in observations.vitals if not vital.quote.startswith(RECHECK)]

    result = evaluate(submission, observations)

    assert result.decision == "NOT_CLEARED"
    assert [issue.category for issue in result.issues] == ["ACUTE_SAFETY_EXCLUSION"]
    assert "189/112 mmHg" in result.issues[0].finding
