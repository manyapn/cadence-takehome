from __future__ import annotations

from observations import Reading
from resolve import resolve_vital
from rules import check_vital


def blood_pressure(systolic: int, source: str, reference: str, time: str | None, offset: int | None = None) -> Reading:
    return Reading(
        kind="blood_pressure",
        systolic=systolic,
        diastolic=80,
        source=source,
        reference=reference,
        date="2026-04-07",
        time=time,
        offset=offset,
    )


def test_later_clock_time_governs_across_sources() -> None:
    structured = blood_pressure(184, "structured", "Pre-op clinic", "09:05")
    note = blood_pressure(128, "note", "nursing", "10:05", offset=40)

    resolution = resolve_vital([structured, note])

    assert resolution.governing == [note]
    assert resolution.superseded[0].reason == "later time on the same day"
    assert check_vital("blood_pressure", resolution).status == "pass"


def test_untimed_reading_from_another_source_cannot_be_ordered() -> None:
    structured = blood_pressure(128, "structured", "Pre-op clinic", "09:05")
    note = blood_pressure(184, "note", "nursing", None, offset=40)

    for readings in ([structured, note], [note, structured]):
        resolution = resolve_vital(readings)

        assert sorted(reading.source for reading in resolution.governing) == ["note", "structured"]
        assert resolution.superseded == []
        assert check_vital("blood_pressure", resolution).status == "fail"


def test_text_order_decides_within_one_note() -> None:
    first = blood_pressure(184, "note", "nursing", "08:06", offset=10)
    second = blood_pressure(128, "note", "nursing", None, offset=90)

    resolution = resolve_vital([second, first])

    assert resolution.governing == [second]
    assert resolution.superseded[0].reason == "later in the same note"


def test_later_date_governs_whatever_the_time() -> None:
    older = blood_pressure(184, "note", "nursing", "23:00", offset=10)
    newer = blood_pressure(128, "structured", "Pre-op clinic", None).model_copy(update={"date": "2026-04-08"})

    assert resolve_vital([older, newer]).governing == [newer]
