"""PLAN §5.5 number guard helpers: reading written numbers, nearness, writing in the same style.
(Number-word normalization, `digits`/`numbers_in`, is tested in test_evals.py.)"""

import pytest

from cohr.numbers import (
    WRITTEN,
    Reading,
    native_readings,
    near,
    one_edit,
    ordinals_in,
    readings,
    write_like,
)


def _values(body: str) -> set[float]:
    return {r.value for r in readings(body)}


def test_readings_cover_both_separator_conventions() -> None:
    assert _values("42500") == {42500}
    assert _values("42,500") == {42500, 42.5}  # thousands comma, or decimal comma
    assert _values("42.500") == {42.5, 42500}  # decimal point, or thousands dot (de)
    assert _values("1,234.5") == {1234.5}
    assert _values("1.234,5") == {1234.5}
    assert _values("42 500") == {42500}
    assert _values("42 500") == {42500}
    assert _values("1.2.3") == set()  # a version, not a number
    assert _values("4,20,500") == set()  # not a grouping either convention allows


def test_readings_record_the_written_style() -> None:
    assert readings("420,500")[0] == Reading(420500, ",", None)
    assert readings("336")[0] == Reading(336, None, None)
    assert readings("42.500")[0] == Reading(42.5, None, ".", 3)


@pytest.mark.parametrize(
    ("value", "source"),
    [(420500, 42500), (570000, 57000), (3360, 336), (2, 22), (1180, 11800), (59600, 59500)],
)
def test_near_one_digit_edit_or_factor(value: float, source: float) -> None:
    assert near(value, source)


@pytest.mark.parametrize(("value", "source"), [(43600, 42500), (42500, 42500), (2024, 2002)])
def test_not_near(value: float, source: float) -> None:
    assert not near(value, source)


def test_one_edit() -> None:
    assert one_edit("4250", "42500") and one_edit("42500", "4250")
    assert one_edit("336", "386")
    assert not one_edit("336", "336") and not one_edit("336", "363")


def test_write_like_keeps_separators() -> None:
    assert write_like(42500, Reading(420500, ",", None)) == "42,500"
    assert write_like(42500, Reading(420500, ".", None)) == "42.500"
    assert write_like(42500, Reading(420500, None, None)) == "42500"
    assert write_like(1234.5, Reading(12345, ".", None)) == "1.234,5"
    assert write_like(336, Reading(3360, ",", None)) == "336"
    assert write_like(42.5, Reading(420.5, None, ".", 3)) == "42.500"
    assert write_like(11800, Reading(1180.5, ",", ".", 2)) == "11,800"
    assert write_like(4650, Reading(4.65, None, ",", 4)) == "4650"  # 4b-1b: not "4650,0000"


def test_written_skips_names_and_reads_suffixes() -> None:
    found = [
        (m["body"], m["ordinal"], m["scale"])
        for m in WRITTEN.finditer(
            "Q3 and R7B in 3D: $420,500 by the 2nd, 42.5k users, up 34%, v1.2."
        )
    ]
    assert found == [
        ("420,500", None, None),
        ("2", "nd", None),
        ("42.5", None, "k"),
        ("34", None, None),
    ]


def test_ordinals_in_reads_digit_and_word_days() -> None:
    text = "by the 22nd of November, the twenty-second, the third of October, 12 seats, 1st"
    assert ordinals_in(text) == {22.0, 3.0, 1.0}
    assert ordinals_in("the thirteenth of October and twenty first") == {13.0, 21.0}
    assert ordinals_in("12 seats and forty-two items") == set()


def test_native_readings_follow_the_decimal_mark() -> None:
    assert [r.value for r in native_readings(readings("770,000"), "en")] == [770000.0]
    assert [r.value for r in native_readings(readings("770.000"), "de")] == [770000.0]
    assert len(native_readings(readings("770,000"), "es")) == 2  # not in the table
    assert [r.value for r in native_readings(readings("3,36"), "en")] == [3.36]  # fallback
