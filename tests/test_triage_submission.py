from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import baseline
import extract
from core import PatientSubmission, triage_submission
from observations import Observations
from rules import TriageResult

HP_TEXT = "PRE-OPERATIVE EVALUATION\nSeen ahead of the planned hernia repair.\nBP 120/80, temperature 98.6 F."
CONSENT_TEXT = "CONSENT\nOperation: hernia repair\nSigned by the patient on 01/22/2026."


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1", "name": {"given": "Ines", "family": "Okoro"}},
        "procedure": {
            "case_id": "case-1",
            "procedure_type": "Elective hernia repair",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [],
        "labs": [
            {
                "code": "CBC",
                "display": "Complete blood count",
                "effective_at": "2026-01-20",
                "status": "final",
            }
        ],
        "medications": [],
        "conditions": [],
        "documents": [
            {"doc_id": "hp-1", "type": "history_and_physical", "date": "2026-01-20", "text": HP_TEXT},
            {"doc_id": "consent-1", "type": "surgical_consent", "date": "2026-01-22", "text": CONSENT_TEXT},
        ],
    }


def vital(kind: str, quote: str, **values) -> dict[str, object]:
    empty = {"systolic": None, "diastolic": None, "temperature": None, "unit": None, "time": None}
    return {"doc_id": "D1", "kind": kind, "quote": quote, **empty, **values}


@pytest.fixture
def extracted() -> Observations:
    return Observations.model_validate(
        {
            "vitals": [
                vital("blood_pressure", "BP 120/80", systolic=120, diastolic=80),
                vital("temperature", "temperature 98.6 F", temperature=98.6, unit="F"),
            ],
            "hp_documents": [{"doc_id": "D1", "kind": "preop_evaluation", "quote": "PRE-OPERATIVE EVALUATION"}],
            "consents": [
                {
                    "doc_id": "D2",
                    "patient_signed": "yes",
                    "signature_quote": "Signed by the patient on 01/22/2026.",
                    "procedure_named": "hernia repair",
                    "procedure_quote": "Operation: hernia repair",
                }
            ],
            "anticoagulant_mentions": [],
            "anticoagulation_plans": [],
            "scheduling_statements": [],
        }
    )


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch, extracted: Observations) -> Mock:
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(status="completed", output=[], output_parsed=extracted)
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps({"decision": "READY", "issues": [], "explanation": "All required criteria are satisfied."}),
    )
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **settings: client))
    extract.shared_client.cache_clear()
    yield client
    extract.shared_client.cache_clear()


def test_triage_submission_makes_one_strict_extraction_call(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    output = triage_submission(submission_payload, model="test-model")

    assert isinstance(output, TriageResult)
    assert output.decision == "READY"
    assert output.issues == []
    assert output.unverified == []

    assert openai_client.responses.parse.call_count == 1
    assert openai_client.responses.create.call_count == 0
    call = openai_client.responses.parse.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["instructions"] == extract.EXTRACTION_PROMPT
    assert call["text_format"] is Observations


def test_triage_submission_sends_only_the_documents(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    triage_submission(PatientSubmission.model_validate(submission_payload), model="test-model")

    chart = openai_client.responses.parse.call_args.kwargs["input"]
    assert HP_TEXT in chart and CONSENT_TEXT in chart
    assert "=== document D1 | type: history_and_physical | date: 2026-01-20 ===" in chart
    assert "Okoro" not in chart and "patient-1" not in chart


def test_one_client_is_shared_across_calls(
    openai_client: Mock,
    submission_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = Mock(return_value=openai_client)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=built))

    triage_submission(submission_payload, model="test-model")
    triage_submission(submission_payload, model="test-model")

    assert built.call_count == 1
    assert built.call_args.kwargs == {"timeout": 60.0, "max_retries": 2}


def unresolved(output: TriageResult) -> dict[str, str]:
    return {check.name: check.status for check in output.checks}


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status="incomplete", incomplete_details="max_output_tokens", output=[], output_parsed=None),
        SimpleNamespace(
            status="completed",
            output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="cannot help")])],
            output_parsed=None,
        ),
        SimpleNamespace(status="completed", output=[], output_parsed=None),
    ],
    ids=["incomplete", "refusal", "no parsed output"],
)
def test_unusable_model_response_is_follow_up_not_ready(
    openai_client: Mock,
    submission_payload: dict[str, object],
    response: SimpleNamespace,
) -> None:
    openai_client.responses.parse.return_value = response

    output = triage_submission(submission_payload, model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert unresolved(output) == {
        "history_and_physical": "unknown",
        "surgical_consent": "unknown",
        "cbc": "pass",
        "cmp": "not_applicable",
        "anticoagulation_plan": "unknown",
        "blood_pressure": "unknown",
        "temperature": "unknown",
    }
    assert {issue.category for issue in output.issues} == {"MISSING_REQUIRED_DATA"}
    assert {issue.unknown_cause for issue in output.issues} == {"processing_failure"}
    assert "ExtractionError" in output.issues[0].finding


def test_failed_model_call_is_follow_up_not_a_crash(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    openai_client.responses.parse.side_effect = TimeoutError("request timed out")
    submission_payload["vitals"] = [
        {"type": "blood_pressure", "systolic": 190, "diastolic": 115, "date": "2026-01-25T09:00:00Z"}
    ]

    output = triage_submission(submission_payload, model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert "TimeoutError: request timed out" in output.explanation
    blood_pressure = next(check for check in output.checks if check.name == "blood_pressure")
    assert blood_pressure.status == "unknown"
    assert "190/115 mmHg" in blood_pressure.summary


def test_fabricated_extraction_is_not_ready(
    openai_client: Mock,
    submission_payload: dict[str, object],
    extracted: Observations,
) -> None:
    for vital_observation in extracted.vitals:
        vital_observation.quote = "not in the chart"

    output = triage_submission(submission_payload, model="test-model")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert len(output.unverified) == 2


def test_baseline_is_still_callable_with_its_one_loose_call(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    output = baseline.triage_submission(submission_payload, model="test-model")

    assert isinstance(output, baseline.TriageOutput)
    assert output.decision == "READY"
    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    assert call["instructions"] == baseline.BASELINE_SYSTEM_PROMPT
    assert call["text"]["format"]["schema"] == baseline.triage_output_json_schema()
    assert call["text"]["format"]["strict"] is False


def test_baseline_rejects_invalid_model_json(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    openai_client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps({"decision": "MAYBE", "issues": [], "explanation": "Not a valid triage decision."})
    )

    with pytest.raises(ValidationError):
        baseline.triage_submission(submission_payload, model="test-model")
