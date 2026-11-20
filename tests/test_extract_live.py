from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core import PatientSubmission
from extract import extract
from observations import Observations
from pipeline import evaluate
from verify import verify

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("MODEL", "gpt-4.1-mini")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("RUN_LIVE_TESTS")),
    reason="makes a real model call; set RUN_LIVE_TESTS=1 with OPENAI_API_KEY to run",
)


@pytest.fixture(scope="module")
def submission() -> PatientSubmission:
    with (ROOT / "data" / "patients_sample_50.jsonl").open() as handle:
        cases = [json.loads(line) for line in handle]
    return PatientSubmission.model_validate(
        next(case["submission"] for case in cases if case["case_id"] == "case_00016")
    )


@pytest.fixture(scope="module")
def observations(submission: PatientSubmission) -> Observations:
    return extract(submission, model=MODEL)


def test_extraction_finds_the_arrival_reading_and_the_recheck(
    submission: PatientSubmission, observations: Observations
) -> None:
    verified = verify(submission, observations)
    blood_pressures = {
        (reading.systolic, reading.diastolic, reading.time)
        for reading in verified.readings
        if reading.kind == "blood_pressure"
    }

    assert {(189, 112, "08:06"), (137, 78, "08:30")} <= blood_pressures


def test_case_16_is_ready_end_to_end(submission: PatientSubmission, observations: Observations) -> None:
    result = evaluate(submission, observations)

    assert result.decision == "READY"
    assert result.issues == []
