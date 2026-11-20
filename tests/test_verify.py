from __future__ import annotations

from decimal import Decimal

import pytest

from core import PatientSubmission
from observations import (
    AnticoagulantMention,
    AnticoagulationPlan,
    ConsentObservation,
    HpDocument,
    Observations,
    PlanSide,
    SchedulingStatement,
    VitalObservation,
)
from verify import verify

NURSING_TEXT = """NURSING CHECK-IN
Readings:
    07:50   BP 172/96,  HR 69,  Temp 38.0 C
    08:20   BP 150/88 on repeat
Oral temperature 99.5 F at discharge desk.
Patient states she takes Eliquis every morning."""

PLAN_TEXT = """MEDICATION PLAN
Eliquis: last dose three days before the operation.
Restart the evening after the operation if no bleeding."""

CONSENT_TEXT = """CONSENT FORM
Operation: left knee arthroscopy on 04/09/2026
Signed by the patient, 04/02/2026."""


@pytest.fixture
def submission() -> PatientSubmission:
    return PatientSubmission.model_validate(
        {
            "documents": [
                {"doc_id": "nursing", "date": "2026-04-07", "text": NURSING_TEXT},
                {"doc_id": "plan", "date": "2026-04-01", "text": PLAN_TEXT},
                {"doc_id": "consent", "date": "2026-04-02", "text": CONSENT_TEXT},
            ]
        }
    )


def observations(**lists: list) -> Observations:
    empty = {name: [] for name in Observations.model_fields}
    return Observations(**{**empty, **lists})


def blood_pressure(quote: str, systolic: int = 172, diastolic: int = 96, **fields) -> VitalObservation:
    return VitalObservation(
        **{
            "doc_id": "nursing",
            "kind": "blood_pressure",
            "systolic": systolic,
            "diastolic": diastolic,
            "temperature": None,
            "unit": None,
            "time": None,
            "quote": quote,
            **fields,
        }
    )


def temperature(quote: str, value: float, unit: str, **fields) -> VitalObservation:
    return VitalObservation(
        **{
            "doc_id": "nursing",
            "kind": "temperature",
            "systolic": None,
            "diastolic": None,
            "temperature": value,
            "unit": unit,
            "time": None,
            "quote": quote,
            **fields,
        }
    )


def test_verbatim_quote_becomes_a_reading_dated_from_the_document(submission: PatientSubmission) -> None:
    quote = "08:20   BP 150/88 on repeat"
    result = verify(submission, observations(vitals=[blood_pressure(quote, 150, 88, time="08:20")]))

    assert result.dropped == []
    (reading,) = result.readings
    assert (reading.systolic, reading.diastolic, reading.time) == (150, 88, "08:20")
    assert (reading.source, reading.reference, reading.date) == ("note", "nursing", "2026-04-07")
    assert reading.offset == NURSING_TEXT.index(quote)


def test_quote_with_collapsed_indentation_passes(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[blood_pressure("07:50 BP 172/96, HR 69", time="07:50")]))

    assert result.dropped == []
    assert len(result.readings) == 1


def test_fabricated_quote_is_dropped_and_recorded(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[blood_pressure("08:40 BP 128/76 after rest", 128, 76)]))

    assert result.readings == []
    (dropped,) = result.dropped
    assert (dropped.kind, dropped.doc_id, dropped.reason) == ("vital", "nursing", "quote not found in document")
    assert dropped.claim["systolic"] == 128


def test_real_quote_with_wrong_numbers_is_dropped(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[blood_pressure("08:20 BP 150/88 on repeat", 130, 88)]))

    assert result.readings == []
    assert result.dropped[0].reason == "quote does not contain 130/88"


def test_systolic_and_diastolic_must_be_written_as_a_pair(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[blood_pressure("07:50 BP 172/96, HR 69", 172, 69)]))

    assert result.readings == []
    assert result.dropped[0].reason == "quote does not contain 172/69"


def test_blood_pressure_quote_must_hold_one_blood_pressure(submission: PatientSubmission) -> None:
    quote = "07:50 BP 172/96, HR 69, Temp 38.0 C 08:20 BP 150/88"
    result = verify(submission, observations(vitals=[blood_pressure(quote, 150, 88)]))

    assert result.readings == []
    assert result.dropped[0].reason == "quote contains 2 blood pressures"


def test_a_date_in_the_quote_is_not_a_second_blood_pressure() -> None:
    text = "Seen 04/07, 12/10 visit reviewed. 08:20 BP 150/88"
    submission = PatientSubmission.model_validate({"documents": [{"doc_id": "nursing", "date": "2026-04-07", "text": text}]})
    result = verify(submission, observations(vitals=[blood_pressure(text, 150, 88, time="08:20")]))

    assert result.dropped == []
    assert len(result.readings) == 1


def test_time_order_must_agree_with_text_order_within_a_note() -> None:
    def note(doc_id: str, first: str, second: str) -> dict[str, str]:
        return {"doc_id": doc_id, "date": "2026-04-07", "text": f"{first} BP 172/96\n{second} BP 150/88"}

    submission = PatientSubmission.model_validate(
        {"documents": [note("backwards", "9:40", "09:10"), note("forwards", "9:10", "09:40")]}
    )

    def readings(doc_id: str, first: str, second: str) -> list[VitalObservation]:
        return [
            blood_pressure(f"{first} BP 172/96", 172, 96, time=first, doc_id=doc_id),
            blood_pressure(f"{second} BP 150/88", 150, 88, time=second, doc_id=doc_id),
        ]

    result = verify(
        submission,
        observations(vitals=readings("backwards", "9:40", "09:10") + readings("forwards", "9:10", "09:40")),
    )

    assert [(reading.reference, reading.time) for reading in result.readings] == [("forwards", "09:10"), ("forwards", "09:40")]
    assert [(dropped.doc_id, dropped.reason) for dropped in result.dropped] == [
        ("backwards", "time order and text order disagree in this note")
    ] * 2


def test_time_followed_by_a_colon_passes(submission: PatientSubmission) -> None:
    document = submission.documents[0]
    document.text += "\nLATE ENTRY 08:40: BP 128/76 after rest"
    result = verify(submission, observations(vitals=[blood_pressure("LATE ENTRY 08:40: BP 128/76", 128, 76, time="08:40")]))

    assert result.dropped == []


def test_twelve_hour_time_is_dropped() -> None:
    submission = PatientSubmission.model_validate(
        {"documents": [{"doc_id": "nursing", "date": "2026-04-07", "text": "2:15 PM BP 150/88"}]}
    )
    result = verify(submission, observations(vitals=[blood_pressure("2:15 PM BP 150/88", 150, 88, time="2:15")]))

    assert result.readings == []
    assert result.dropped[0].reason == "2:15 is a 12-hour time"


def test_quote_from_a_different_document_is_dropped(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[blood_pressure("BP 150/88 on repeat", 150, 88, doc_id="plan")]))

    assert result.readings == []
    assert result.dropped[0].reason == "quote not found in document"


def test_unknown_document_is_dropped(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[blood_pressure("BP 150/88 on repeat", 150, 88, doc_id="missing")]))

    assert result.readings == []
    assert result.dropped[0].reason == "cited document is not in the submission"


def test_time_must_be_in_the_quote(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[blood_pressure("BP 150/88 on repeat", 150, 88, time="09:15")]))

    assert result.readings == []
    assert result.dropped[0].reason == "quote does not contain the time 09:15"


def test_celsius_converts_exactly(submission: PatientSubmission) -> None:
    result = verify(
        submission,
        observations(
            vitals=[
                temperature("Temp 38.0 C", 38.0, "C"),
                temperature("Oral temperature 99.5 F", 99.5, "F"),
            ]
        ),
    )

    assert result.dropped == []
    assert [reading.temperature_f for reading in result.readings] == [Decimal("100.4"), Decimal("99.5")]
    assert not result.readings[0].temperature_f > Decimal("100.4")


def test_temperature_with_the_wrong_unit_is_dropped(submission: PatientSubmission) -> None:
    result = verify(submission, observations(vitals=[temperature("Temp 38.0 C", 38.0, "F")]))

    assert result.readings == []
    assert result.dropped[0].reason == "quote does not contain 38.0 F"


def test_consent_needs_its_quotes(submission: PatientSubmission) -> None:
    def consent(**fields) -> ConsentObservation:
        return ConsentObservation(
            **{
                "doc_id": "consent",
                "patient_signed": "yes",
                "signature_quote": "Signed by the patient, 04/02/2026.",
                "procedure_named": "left knee arthroscopy",
                "procedure_quote": "Operation: left knee arthroscopy on 04/09/2026",
                **fields,
            }
        )

    good = consent()
    result = verify(
        submission,
        observations(
            consents=[
                good,
                consent(signature_quote=None),
                consent(signature_quote="Signed by the patient and a witness."),
                consent(procedure_named="hip replacement"),
            ]
        ),
    )

    assert result.consents == [good]
    assert [dropped.reason for dropped in result.dropped] == [
        "signed consent without a signature quote",
        "quote not found in document",
        "quote does not name hip replacement",
    ]


def test_anticoagulant_mention_must_name_the_drug_in_its_quote(submission: PatientSubmission) -> None:
    def mention(drug: str) -> AnticoagulantMention:
        return AnticoagulantMention(
            doc_id="nursing",
            drug_as_written=drug,
            generic_name="apixaban",
            status="taking",
            quote="Patient states she takes Eliquis every morning.",
        )

    result = verify(submission, observations(anticoagulant_mentions=[mention("Eliquis"), mention("Xarelto")]))

    assert result.anticoagulant_mentions == [mention("Eliquis")]
    assert result.dropped[0].reason == "quote does not name Xarelto"


def test_plan_sides_need_quotes_from_a_document_that_names_the_drug(submission: PatientSubmission) -> None:
    hold = PlanSide(label="instruction", quote="last dose three days before the operation.")
    restart = PlanSide(label="instruction", quote="Restart the evening after the operation if no bleeding.")

    def plan(**fields) -> AnticoagulationPlan:
        return AnticoagulationPlan(**{"doc_id": "plan", "drug_as_written": "Eliquis", "preop": hold, "postop": restart, **fields})

    good = plan()
    half = plan(postop=PlanSide(label="absent", quote=None))
    result = verify(
        submission,
        observations(
            anticoagulation_plans=[
                good,
                half,
                plan(drug_as_written="warfarin"),
                plan(postop=PlanSide(label="deferred", quote=None)),
                plan(postop=PlanSide(label="instruction", quote="Restart on the second day after the operation.")),
            ]
        ),
    )

    assert result.anticoagulation_plans == [good, half]
    assert [dropped.reason for dropped in result.dropped] == [
        "document does not name warfarin",
        "deferred without a quote",
        "quote not found in document",
    ]


def test_scheduling_date_must_be_in_the_quote(submission: PatientSubmission) -> None:
    def statement(procedure_date: str | None, pending: bool = False) -> SchedulingStatement:
        return SchedulingStatement(
            doc_id="consent",
            procedure_date=procedure_date,
            pending=pending,
            quote="Operation: left knee arthroscopy on 04/09/2026",
        )

    result = verify(
        submission,
        observations(scheduling_statements=[statement("2026-04-09"), statement("2026-04-19"), statement(None)]),
    )

    assert result.scheduling_statements == [statement("2026-04-09")]
    assert [dropped.reason for dropped in result.dropped] == [
        "quote does not contain the date 2026-04-19",
        "states neither a date nor that it is pending",
    ]


def test_hp_document_quote_must_be_in_its_document(submission: PatientSubmission) -> None:
    real = HpDocument(doc_id="plan", kind="wellness_or_other", quote="MEDICATION PLAN")
    fake = HpDocument(doc_id="plan", kind="preop_evaluation", quote="PRE-OPERATIVE EVALUATION")

    result = verify(submission, observations(hp_documents=[real, fake]))

    assert result.hp_documents == [real]
    assert result.dropped[0].kind == "hp_document"
