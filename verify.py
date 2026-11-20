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
    Observations,
    Reading,
    SchedulingStatement,
    VerifiedObservations,
    VitalObservation,
)

NUMBER = re.compile(r"\d+(?:\.\d+)?")
BLOOD_PRESSURE = re.compile(r"(?<![\d/])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/]|\.\d)")
UNIT_AFTER_VALUE = re.compile(r"\s*(?:°|degrees?)?\s*([FC])(?:ahrenheit|elsius)?\b")


def find_quote(quote: str, text: str) -> int | None:
    words = quote.split()
    if not words:
        return None
    match = re.search(r"\s+".join(re.escape(word) for word in words), text)
    return match.start() if match else None


def blood_pressures(text: str) -> list[tuple[int, int]]:
    pairs = [(int(systolic), int(diastolic)) for systolic, diastolic in BLOOD_PRESSURE.findall(text)]
    # A date like 03/15 or 12/10 is not a blood pressure.
    return [(systolic, diastolic) for systolic, diastolic in pairs if 50 <= systolic <= 300 and 20 <= diastolic < systolic]


def numbers_in(text: str) -> set[Decimal]:
    return {Decimal(number) for number in NUMBER.findall(text)}


def names(quote: str, value: str) -> bool:
    squash = lambda s: " ".join(s.split()).casefold()
    return squash(value) in squash(quote)


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


def consent_problem(consent: ConsentObservation, text: str) -> str | None:
    if consent.patient_signed == "yes" and consent.signature_quote is None:
        return "signed consent without a signature quote"
    if consent.procedure_named is not None and consent.procedure_quote is None:
        return "procedure named without a procedure quote"
    for quote in (consent.signature_quote, consent.procedure_quote):
        if quote is not None and find_quote(quote, text) is None:
            return "quote not found in document"
    if consent.procedure_named is not None and not names(consent.procedure_quote, consent.procedure_named):
        return f"quote does not name {consent.procedure_named}"
    return None


def mention_problem(mention: AnticoagulantMention, text: str) -> str | None:
    if find_quote(mention.quote, text) is None:
        return "quote not found in document"
    if not names(mention.quote, mention.drug_as_written):
        return f"quote does not name {mention.drug_as_written}"
    return None


def plan_problem(plan: AnticoagulationPlan, text: str) -> str | None:
    # A side's quote may not repeat the drug, so the document as a whole has to name it.
    if not names(text, plan.drug_as_written):
        return f"document does not name {plan.drug_as_written}"
    for side in (plan.preop, plan.postop):
        if side.label != "absent" and side.quote is None:
            return f"{side.label} without a quote"
        if side.quote is not None and find_quote(side.quote, text) is None:
            return "quote not found in document"
    return None


def scheduling_problem(statement: SchedulingStatement, text: str) -> str | None:
    if find_quote(statement.quote, text) is None:
        return "quote not found in document"
    if statement.procedure_date is None:
        return None if statement.pending else "states neither a date nor that it is pending"
    try:
        stated = date.fromisoformat(statement.procedure_date)
    except ValueError:
        return f"{statement.procedure_date} is not an ISO date"
    if not {stated.year, stated.month, stated.day} <= numbers_in(statement.quote):
        return f"quote does not contain the date {statement.procedure_date}"
    return None


def verify(submission: PatientSubmission, observations: Observations) -> VerifiedObservations:
    documents: dict[str, Document] = {d.doc_id: d for d in submission.documents if d.doc_id}
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
        offset = find_quote(vital.quote, document.text or "")
        problem = "quote not found in document" if offset is None else vital_problem(vital)
        if problem:
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
                    quote=vital.quote,
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

    for hp in observations.hp_documents:
        if (document := cited("hp_document", hp)) is None:
            continue
        if find_quote(hp.quote, document.text or "") is None:
            drop("hp_document", hp, "quote not found in document")
        else:
            verified.hp_documents.append(hp)

    checks = (
        ("consent", observations.consents, consent_problem, verified.consents),
        ("anticoagulant_mention", observations.anticoagulant_mentions, mention_problem, verified.anticoagulant_mentions),
        ("anticoagulation_plan", observations.anticoagulation_plans, plan_problem, verified.anticoagulation_plans),
        ("scheduling_statement", observations.scheduling_statements, scheduling_problem, verified.scheduling_statements),
    )
    for kind, items, find_problem, kept in checks:
        for item in items:
            if (document := cited(kind, item)) is None:
                continue
            if problem := find_problem(item, document.text or ""):
                drop(kind, item, problem)
            else:
                kept.append(item)

    return verified
