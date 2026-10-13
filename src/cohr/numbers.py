"""Number normalization (PLAN §5.5, §18): number words, separators, `k`/`thousand` and ordinal
suffixes → plain digits. English only. Shared by the Map number guard and the evals, so both
agree on what a number is."""

import math
import re
from collections.abc import Sequence
from typing import NamedTuple

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_ORDINALS) | set(_SCALES)
_WORD = re.compile(r"[a-z]+|\d[\d,]*(?:\.\d+)?|[^\sa-z\d]", re.I)
_ORDINAL_SUFFIX = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I)
_SUFFIXED = re.compile(r"\b(\d+(?:\.\d+)?)\s*(k|m|thousand|million)\b", re.I)


def _words_value(words: Sequence[str]) -> int:
    """'fourteen thousand five hundred' → 14500; 'twenty first' → 21."""
    total = current = 0
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w in _ORDINALS:
            current += _ORDINALS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        else:  # thousand, million
            total += max(current, 1) * _SCALES[w]
            current = 0
    return total + current


def digits(text: str) -> str:
    """Number words and digit forms → plain digits: "$14,500" → "$14500", "fourteen thousand
    five hundred" → "14500", "the twentieth" → "the 20", "14.5k" → "14500"."""
    text = _ORDINAL_SUFFIX.sub(r"\1", text)
    text = _SUFFIXED.sub(
        lambda m: _fmt(float(m.group(1)) * (1e6 if m.group(2).lower() in "mmillion" else 1e3)),
        text,
    )
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            out.append(str(_words_value(run)))
            run.clear()

    for tok in _WORD.findall(text):
        low = tok.lower()
        if low in _NUMBER_WORDS and not (low in _SCALES and not run):
            run.append(low)
        elif low == "and" and run:
            continue  # "one hundred and five"
        else:
            flush()
            out.append(tok.replace(",", "") if tok[0].isdigit() else tok)
    flush()
    return " ".join(out)


def _fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else str(v)


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def ordinals_in(text: str) -> set[float]:
    """Ordinals, the way days are said: "the 22nd" and "the twenty-second" → 22."""
    out = {float(m.group(1)) for m in _ORDINAL_SUFFIX.finditer(text)}
    run: list[str] = []
    for tok in [*_WORD.findall(text), ""]:
        low = tok.lower()
        if low in _NUMBER_WORDS:
            run.append(low)
        elif not (low == "-" and run):  # "twenty-second" goes on
            if run and run[-1] in _ORDINALS:
                out.add(float(_words_value(run)))
            run = []
    return out


def numbers_in(text: str) -> set[float]:
    return {float(m) for m in _NUMBER.findall(digits(text))}


# --- numbers as written (Map number guard, §5.5) -------------------------------------------

NBSP, NNBSP = "\u00a0", "\u202f"
# (thousands, decimal) separator pairs, English style first: when two readings lead to the same
# source number, a repair is written in the first one's style.
_SEPARATORS = (
    (None, None),
    (",", "."),
    (None, "."),
    (".", ","),
    (NBSP, ","),
    (NNBSP, ","),
    (NBSP, "."),
    (NNBSP, "."),
    (" ", ","),
    (" ", "."),
    (None, ","),
)
# A number written with digits: not glued to a letter or digit on either side ("Q3", "R7B" and
# "3D" are names, not numbers), optionally with an ordinal or a scale suffix. A plain space
# groups thousands only as "42 500" (1–3 digits, then groups of exactly 3).
WRITTEN = re.compile(
    r"(?<![\w.,])(?P<body>\d{1,3}(?: \d{3})+(?:[.,]\d+)?|\d(?:[\d,.\u00a0\u202f]*\d)?)"
    r"(?:(?P<ordinal>st|nd|rd|th)|\s?(?P<scale>k|K|thousand|million))?(?!\w)"
)
SCALE = {"k": 1e3, "K": 1e3, "thousand": 1e3, "million": 1e6}
FACTORS = (10, 100, 1000)


class Reading(NamedTuple):
    value: float
    group: str | None  # thousands separator as written
    decimal: str | None
    places: int = 0  # digits after the decimal separator as written ("42.500" → 3)


def _read(body: str, group: str | None, decimal: str | None) -> float | None:
    if any(not c.isdigit() and c not in (group, decimal) for c in body):
        return None
    whole, _, frac = body.partition(decimal) if decimal else (body, "", "")
    if decimal and decimal in body and (not frac or decimal in frac or (group and group in frac)):
        return None
    if group and group in whole:
        parts = whole.split(group)
        if not 1 <= len(parts[0]) <= 3 or any(len(p) != 3 for p in parts[1:]):
            return None
        whole = "".join(parts)
    return float(f"{whole}.{frac}" if frac else whole)


def readings(body: str) -> list[Reading]:
    """Every value a written number can mean: "42,500" is 42500 or (decimal comma) 42.5;
    "42.500" is 42.5 or 42500. A separator that can't be grouping or decimal gives nothing."""
    out: list[Reading] = []
    for group, decimal in _SEPARATORS:
        v = _read(body, group, decimal)
        if v is not None and all(r.value != v for r in out):
            dec = decimal if decimal and decimal in body else None
            places = len(body.rpartition(dec)[2]) if dec else 0
            out.append(Reading(v, group if group and group in body else None, dec, places))
    return out


# Decimal mark by target language, for the guard's repair candidates (6b-2): "." languages group
# with ",", "," languages with "." or a space. Not listed, so both readings stay: languages
# written both ways (es: Spain vs. Mexico) or with their own separators (ar, fa).
DECIMAL_MARK = {
    **dict.fromkeys(("en", "he", "hi", "ja", "ko", "zh"), "."),
    **dict.fromkeys(
        ("cs", "de", "el", "fr", "id", "it", "nl", "pl", "pt", "ro", "ru", "tr", "uk", "vi"), ","
    ),
}


def native_readings(reads: list[Reading], language: str | None) -> list[Reading]:
    """The readings written the way `language` writes numbers: in English "770,000" is 770000,
    never 770.0. All of them when the language isn't in DECIMAL_MARK or no reading fits it
    (a malformed "3,36" can only be the decimal-comma 3.36)."""
    mark = DECIMAL_MARK.get(language or "")
    if mark is None:
        return reads
    return [r for r in reads if r.decimal in (None, mark) and r.group != mark] or reads


def one_edit(a: str, b: str) -> bool:
    """One character inserted, dropped or changed."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))


def near(value: float, source: float) -> bool:
    """`source` is one digit edit or a factor of 10, 100 or 1000 away from `value`."""
    if value == source:
        return False
    if one_edit(_fmt(value), _fmt(source)):
        return True
    return any(math.isclose(value * f, source) or math.isclose(source * f, value) for f in FACTORS)


def write_like(value: float, reading: Reading) -> str:
    """`value` in the written style of `reading`: same thousands and decimal separators, and
    a fraction padded to as many places ("420.500" → "42.500", which reads right either way).
    A whole value gets no fraction: "4,6500" (only readable as 4.65) → 4650 is "4650"."""
    whole, _, frac = _fmt(value).partition(".")
    frac = frac.ljust(reading.places, "0") if frac else ""
    if reading.group:
        whole = f"{int(whole):,}".replace(",", reading.group)
    decimal = reading.decimal or ("," if reading.group == "." else ".")
    return f"{whole}{decimal}{frac}" if frac else whole
