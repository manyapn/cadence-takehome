"""Decide which reading governs, and record what it superseded."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from core import Medication, PatientSubmission
from observations import AnticoagulantMention, AnticoagulationPlan, Reading, VerifiedObservations

ANTICOAGULANTS = {
    "warfarin": "warfarin",
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "apixaban": "apixaban",
    "eliquis": "apixaban",
    "rivaroxaban": "rivaroxaban",
    "xarelto": "rivaroxaban",
    "dabigatran": "dabigatran",
    "pradaxa": "dabigatran",
    "edoxaban": "edoxaban",
    "savaysa": "edoxaban",
    "enoxaparin": "enoxaparin",
    "lovenox": "enoxaparin",
    "dalteparin": "dalteparin",
    "fragmin": "dalteparin",
    "fondaparinux": "fondaparinux",
    "arixtra": "fondaparinux",
    "heparin": "heparin",
}
ANTIPLATELETS = {
    "aspirin",
    "asa",
    "clopidogrel",
    "plavix",
    "prasugrel",
    "effient",
    "ticagrelor",
    "brilinta",
    "dipyridamole",
    "cilostazol",
}


class Superseded(BaseModel):

    reading: Reading
    by: Reading
    reason: str


class VitalResolution(BaseModel):

    governing: list[Reading]
    superseded: list[Superseded]


class DrugStatus(BaseModel):

    drug: str
    recognised: bool
    status: Literal["taking", "discontinued", "uncertain"]
    medications: list[Medication] = []
    mentions: list[AnticoagulantMention] = []
    plans: list[AnticoagulationPlan] = []


def anticoagulant(name: str) -> str | None:
    for word in re.findall(r"[a-z]+", name.lower()):
        if word in ANTICOAGULANTS:
            return ANTICOAGULANTS[word]
    return None


def resolve_anticoagulants(submission: PatientSubmission, verified: VerifiedObservations) -> list[DrugStatus]:
    dates = {document.doc_id: document.date or "" for document in submission.documents}
    drugs: dict[str, DrugStatus] = {}

    def entry(drug: str, recognised: bool) -> DrugStatus:
        return drugs.setdefault(drug, DrugStatus(drug=drug, recognised=recognised, status="uncertain"))

    for medication in submission.medications:
        if drug := anticoagulant(medication.name or ""):
            entry(drug, True).medications.append(medication)
    for mention in verified.anticoagulant_mentions:
        if drug := anticoagulant(mention.drug_as_written) or anticoagulant(mention.generic_name):
            entry(drug, True).mentions.append(mention)
        elif not ANTIPLATELETS & set(re.findall(r"[a-z]+", f"{mention.drug_as_written} {mention.generic_name}".lower())):
            # The extractor called it an anticoagulant and the table cannot confirm or deny it.
            entry(mention.generic_name.strip().lower() or mention.drug_as_written.lower(), False).mentions.append(mention)
    for plan in verified.anticoagulation_plans:
        drug = anticoagulant(plan.drug_as_written) or plan.drug_as_written.strip().lower()
        if drug in drugs:
            drugs[drug].plans.append(plan)

    for drug in drugs.values():
        says = {{True: "taking", False: "discontinued", None: "uncertain"}[medication.active] for medication in drug.medications}
        if drug.mentions:
            newest = max(dates[mention.doc_id] for mention in drug.mentions)
            says |= {mention.status for mention in drug.mentions if dates[mention.doc_id] == newest}
        drug.status = says.pop() if len(says) == 1 else "uncertain"
    return sorted(drugs.values(), key=lambda drug: drug.drug)


def structured_readings(submission: PatientSubmission) -> list[Reading]:
    readings = []
    for vital in submission.vitals:
        timestamp = vital.date or ""
        fields = {
            "source": "structured",
            # Not the list index: the output must not depend on input order.
            "reference": f"{vital.source or 'structured vitals'}, {vital.date}",
            "date": timestamp[:10] or None,
            "time": timestamp[11:16] or None,
        }
        systolic, diastolic = getattr(vital, "systolic", None), getattr(vital, "diastolic", None)
        value_f = getattr(vital, "value_f", None)
        if vital.type == "blood_pressure" and systolic is not None and diastolic is not None:
            readings.append(Reading(kind="blood_pressure", systolic=systolic, diastolic=diastolic, **fields))
        elif vital.type == "temperature" and value_f is not None:
            readings.append(Reading(kind="temperature", temperature_f=Decimal(str(value_f)), **fields))
    return readings


def later(a: Reading, b: Reading) -> tuple[Reading, str] | None:
    if a.date is None or b.date is None:
        return None
    if a.date != b.date:
        return max(a, b, key=lambda reading: reading.date), "later date"
    if a.source == b.source == "note" and a.reference == b.reference:
        if a.offset == b.offset:
            return None
        return max(a, b, key=lambda reading: reading.offset), "later in the same note"
    if a.time and b.time and a.time != b.time:
        return max(a, b, key=lambda reading: reading.time), "later time on the same day"
    return None


def resolve_vital(readings: list[Reading]) -> VitalResolution:
    readings = sorted(readings, key=lambda reading: reading.model_dump_json())
    governing, superseded = [], []
    for reading in readings:
        beaten_by = [
            (other, outcome[1])
            for other in readings
            if (outcome := later(reading, other)) and outcome[0] is other
        ]
        if beaten_by:
            by, reason = max(beaten_by, key=lambda pair: (pair[0].date, pair[0].time or "", pair[0].offset or 0))
            superseded.append(Superseded(reading=reading, by=by, reason=reason))
        else:
            governing.append(reading)
    return VitalResolution(governing=governing, superseded=superseded)


def resolve_vitals(submission: PatientSubmission, note_readings: list[Reading]) -> dict[str, VitalResolution]:
    readings = structured_readings(submission) + note_readings
    return {
        kind: resolve_vital([reading for reading in readings if reading.kind == kind])
        for kind in ("blood_pressure", "temperature")
    }
