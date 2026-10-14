"""PLAN §5.5 number guard helpers: reading written numbers, nearness, writing in the same style.
(Number-word normalization, `digits`/`numbers_in`, is tested in test_evals.py.)"""

import pytest

from cohr.numbers import (
    WRITTEN,
    Reading,
    native_readings,
    near,
    notes_numbers,
    one_edit,
    ordinals_in,
    readings,
    source_numbers_in,
    write_like,
    written_values,
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


def test_notes_numbers_read_the_notes_language() -> None:
    # 6d-1: German notes write "3.550" and "80.000 $"; read in English they were 3.55 and 80.
    assert notes_numbers("Neue Trial-Signups: 3.550, Budget 80.000 $", "de") == {3550, 80000}
    assert notes_numbers("Ladezeit 0,74 s, Quote 37 %", "de") == {0.74, 37}
    # English notes are read as before (number words too): the German reading must not fire.
    assert notes_numbers("3.55 points, up from 3,550, forty two", "en") == {3.55, 3550, 42}
    assert notes_numbers("3.55 points", None) == {3.55}
    # A language not in DECIMAL_MARK keeps both readings (ar writes Western digits, "42,500").
    assert notes_numbers("الميزانية 42,500 دولار", "ar") == {42500, 42.5}
    assert written_values("42,500 دولار و 34%", "ar") == [{42500, 42.5}, {34}]
    assert written_values("Q3: 42.5k", "de") == [{42500}]  # "Q3" is a name; scale applied


@pytest.mark.parametrize(
    ("text", "values"),
    [  # the forms Transcribe wrote in 6d-2's step 0 (say → Transcribe)
        ("es de 42.500 dólares", {42500}),  # es_ES voice
        ("es de 42,500 dólares", {42500}),  # es_MX voice
        ("354 mil dólares, 1,5 millones", {354000, 1500000}),
        ("frente a 7.700 dólares y 34%, 3,5 puntos, 1.250.000", {7700, 34, 3.5, 1250000}),
    ],
)
def test_source_numbers_in_spanish(text: str, values: set[float]) -> None:
    assert source_numbers_in(text, "es") == values


@pytest.mark.parametrize(
    ("text", "values"),
    [
        ("予算は42,500ドルです", {42500}),
        ("予算は4万2500ドルです", {42500}),
        ("予算は四万二千五百ドルです", {42500}),
        ("月間経常収益は35万4000ドル、先月は13万4000ドル", {354000, 134000}),
        ("1万1800ドル、三百二十件、10月20日", {11800, 320, 10, 20}),
    ],
)
def test_source_numbers_in_japanese(text: str, values: set[float]) -> None:
    assert source_numbers_in(text, "ja") == values


def test_source_numbers_in_english_and_other_languages() -> None:
    from cohr.numbers import numbers_in

    text = "$42,500 and fourteen thousand five hundred on the 22nd"
    assert source_numbers_in(text, "en") == source_numbers_in(text, None) == numbers_in(text)
    # German digits get their own reading as well as the English one
    assert {42500.0, 42.5} <= source_numbers_in("Budget 42.500 €", "de")
