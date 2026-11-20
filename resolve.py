"""Decide which reading governs, and record what it superseded."""

from __future__ import annotations

import re
from decimal import Decimal

from pydantic import BaseModel

from core import PatientSubmission
from observations import Reading

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
    "heparin": "heparin",
}


class Superseded(BaseModel):

    reading: Reading
    by: Reading
    reason: str


class VitalResolution(BaseModel):

    governing: list[Reading]
    superseded: list[Superseded]


def anticoagulant(name: str) -> str | None:
    for word in re.findall(r"[a-z]+", name.lower()):
        if word in ANTICOAGULANTS:
            return ANTICOAGULANTS[word]
    return None


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
