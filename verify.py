"""Check each extracted observation against the document it cites."""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal

from core import Document, PatientSubmission
from observations import (
    AnticoagulantMention,
    AnticoagulationPlan,
    ConsentObservation,
    DroppedFact,
    HpDocument,
    Observations,
    Reading,
    SchedulingStatement,
    VerifiedObservations,
    VitalObservation,
)

MATCHED = re.compile(r"[A-Za-z0-9./:]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
BLOOD_PRESSURE = re.compile(r"(?<![\d/])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/]|\.\d)")
UNIT_AFTER_VALUE = re.compile(r"[^A-Za-z0-9]*(?:degrees?\s*)?([FC])(?:ahrenheit|elsius)?\b")
NOT_FOUND = "quote not found in document"


def find_quote(quote: str, text: str) -> tuple[int, str] | None:
    """Where the quote sits in the text, and the text's own wording of it.

    Only letters, digits and . / : have to match. Whitespace and other symbols are ignored.
    """

    wanted = "".join(MATCHED.findall(quote))
    matched = [(index, char) for index, char in enumerate(text) if MATCHED.match(char)]
    at = "".join(char for _, char in matched).find(wanted)
    if not wanted or at < 0:
        return None
    start, end = matched[at][0], matched[at + len(wanted) - 1][0] + 1
    return start, text[start:end]


def blood_pressures(text: str) -> list[tuple[int, int]]:
    pairs = [(int(systolic), int(diastolic)) for systolic, diastolic in BLOOD_PRESSURE.findall(text)]
    # A date like 03/15 or 12/10 is not a blood pressure.
    return [(systolic, diastolic) for systolic, diastolic in pairs if 50 <= systolic <= 300 and 20 <= diastolic < systolic]


def numbers_in(text: str) -> set[Decimal]:
    return {Decimal(number) for number in NUMBER.findall(text)}


def names(quote: str, value: str) -> bool:
    matched = lambda s: "".join(MATCHED.findall(s)).casefold()
    return matched(value) in matched(quote)


def to_fahrenheit(value: Decimal, unit: str) -> Decimal:
    return value if unit == "F" else value * Decimal("1.8") + 32


def vital_problem(vital: VitalObservation) -> str | None:
    if vital.kind == "blood_pressure":
        if vital.systolic is None or vital.diastolic is None:
            return "blood pressure without both systolic and diastolic"
        pairs = blood_pressures(vital.quote)
        if len(pairs) > 1:
            return f"quote contains {len(pairs)} blood pressures"
        if pairs != [(vital.systolic, vital.diastolic)]:
            return f"quote does not contain {vital.systolic}/{vital.diastolic}"
    else:
        if vital.temperature is None or vital.unit is None:
            return "temperature without a value and unit"
        value = Decimal(str(vital.temperature))
        stated = any(
            Decimal(number.group()) == value
            and (unit := UNIT_AFTER_VALUE.match(vital.quote, number.end()))
            and unit.group(1) == vital.unit
            for number in NUMBER.finditer(vital.quote)
        )
        if not stated:
            return f"quote does not contain {value} {vital.unit}"
    if vital.time is not None:
        if not re.fullmatch(r"\d{1,2}:\d{2}", vital.time):
            return f"{vital.time} is not a clock time"
        written = re.search(rf"(?<![\d:]){vital.time}(?!\d|:\d)(\s*[ap]\.?m\b)?", vital.quote, re.IGNORECASE)
        if written is None:
            return f"quote does not contain the time {vital.time}"
        if written.group(1):
            return f"{vital.time} is a 12-hour time"
    return None


def checked_hp(hp: HpDocument, text: str) -> HpDocument | str:
    found = find_quote(hp.quote, text)
    return hp.model_copy(update={"quote": found[1]}) if found else NOT_FOUND


def checked_consent(consent: ConsentObservation, text: str, patient: tuple[str, ...] = ()) -> ConsentObservation | str:
    if consent.patient_signed == "yes" and consent.signature_quote is None:
        return "signed consent without a signature quote"
    if consent.procedure_named is not None and consent.procedure_quote is None:
        return "procedure named without a procedure quote"
    as_written = {}
    for field in ("signature_quote", "procedure_quote"):
        if (quote := getattr(consent, field)) is not None:
            if (found := find_quote(quote, text)) is None:
                return NOT_FOUND
            as_written[field] = found[1]
    consent = consent.model_copy(update=as_written)
    # A clinician's sign-off quoted as the signature must not pass as the patient's.
    if consent.patient_signed == "yes" and not any(names(consent.signature_quote, who) for who in ("patient", *patient)):
        return "signature quote does not refer to the patient"
    if consent.procedure_named is not None and not names(consent.procedure_quote, consent.procedure_named):
        return f"quote does not name {consent.procedure_named}"
    return consent


def checked_mention(mention: AnticoagulantMention, text: str) -> AnticoagulantMention | str:
    if (found := find_quote(mention.quote, text)) is None:
        return NOT_FOUND
    if not names(found[1], mention.drug_as_written):
        return f"quote does not name {mention.drug_as_written}"
    return mention.model_copy(update={"quote": found[1]})


def checked_plan(plan: AnticoagulationPlan, text: str) -> AnticoagulationPlan | str:
    # A side's quote may not repeat the drug, so the document as a whole has to name it.
    if not names(text, plan.drug_as_written):
        return f"document does not name {plan.drug_as_written}"
    as_written = {}
    for field in ("preop", "postop"):
        side = getattr(plan, field)
        if side.label != "absent" and side.quote is None:
            return f"{side.label} without a quote"
        if side.quote is not None:
            if (found := find_quote(side.quote, text)) is None:
                return NOT_FOUND
            as_written[field] = side.model_copy(update={"quote": found[1]})
    return plan.model_copy(update=as_written)


def checked_scheduling(statement: SchedulingStatement, text: str) -> SchedulingStatement | str:
    if (found := find_quote(statement.quote, text)) is None:
        return NOT_FOUND
    statement = statement.model_copy(update={"quote": found[1]})
    if statement.procedure_date is None:
        return statement if statement.pending else "states neither a date nor that it is pending"
    try:
        stated = date.fromisoformat(statement.procedure_date)
    except ValueError:
        return f"{statement.procedure_date} is not an ISO date"
    if not {stated.year, stated.month, stated.day} <= numbers_in(statement.quote):
        return f"quote does not contain the date {statement.procedure_date}"
    return statement


def verify(submission: PatientSubmission, observations: Observations) -> VerifiedObservations:
    documents: dict[str, Document] = {d.doc_id: d for d in submission.documents if d.doc_id}
    name = submission.patient.name if submission.patient else None
    patient = tuple(part for part in (name.given, name.family) if part) if name else ()
    verified = VerifiedObservations()

    def cited(kind: str, observation) -> Document | None:
        document = documents.get(observation.doc_id)
        if document is None:
            drop(kind, observation, "cited document is not in the submission")
        return document

    def drop(kind: str, observation, reason: str) -> None:
        verified.dropped.append(
            DroppedFact(kind=kind, doc_id=observation.doc_id, reason=reason, claim=observation.model_dump())
        )

    readings: list[tuple[VitalObservation, Reading]] = []
    for vital in observations.vitals:
        if (document := cited("vital", vital)) is None:
            continue
        found = find_quote(vital.quote, document.text or "")
        if found is None:
            drop("vital", vital, NOT_FOUND)
            continue
        offset, as_written = found
        # The claimed values are checked against the document's words, not the model's copy of them.
        if problem := vital_problem(vital.model_copy(update={"quote": as_written})):
            drop("vital", vital, problem)
            continue
        readings.append(
            (
                vital,
                Reading(
                    kind=vital.kind,
                    systolic=vital.systolic if vital.kind == "blood_pressure" else None,
                    diastolic=vital.diastolic if vital.kind == "blood_pressure" else None,
                    temperature_f=(
                        to_fahrenheit(Decimal(str(vital.temperature)), vital.unit)
                        if vital.kind == "temperature"
                        else None
                    ),
                    source="note",
                    reference=vital.doc_id,
                    date=document.date,
                    time=vital.time.zfill(5) if vital.time else None,
                    quote=as_written,
                    offset=offset,
                ),
            )
        )

    for vital, reading in readings:
        timed = sorted(
            (other for _, other in readings if (other.reference, other.kind) == (reading.reference, reading.kind) and other.time),
            key=lambda other: other.offset,
        )
        if [other.time for other in timed] != sorted(other.time for other in timed):
            drop("vital", vital, "time order and text order disagree in this note")
        else:
            verified.readings.append(reading)

    checks = (
        ("hp_document", observations.hp_documents, checked_hp, verified.hp_documents),
        ("consent", observations.consents, checked_consent, verified.consents),
        ("anticoagulant_mention", observations.anticoagulant_mentions, checked_mention, verified.anticoagulant_mentions),
        ("anticoagulation_plan", observations.anticoagulation_plans, checked_plan, verified.anticoagulation_plans),
        ("scheduling_statement", observations.scheduling_statements, checked_scheduling, verified.scheduling_statements),
    )
    for kind, items, checked, kept in checks:
        for item in items:
            if (document := cited(kind, item)) is None:
                continue
            outcome = checked(item, document.text or "", *((patient,) if kind == "consent" else ()))
            if isinstance(outcome, str):
                drop(kind, item, outcome)
            else:
                kept.append(outcome)

    return verified
