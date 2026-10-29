"""Due dates as words (PLAN §5.6, D40): "the 17th of November" and "17 November" share a key,
so the Reduce due guard and the evals agree on what a due says. English words only, except
`due_said`'s digits-only check for other languages. Map's dates are checked here too (8c):
"November 1" where the segment says "11 November"."""

import re

from cohr import numbers
from cohr.fts import fold_digits
from cohr.numbers import digits, one_edit

FILLER = {"by", "the", "of", "on", "at", "before", "until", "due", "end"}
MONTH_ABBR = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}
# Extra words allowed between a due's own words when it is said: "by the end of next week"
# says "end of next week", but an AMI segment with "end" and "meeting" 200 words apart doesn't
# say "End of meeting" (8b).
WINDOW = 4


def words(text: str) -> list[str]:
    """Lowercase words, numbers as digits (ordinal suffixes dropped), month abbreviations
    spelled out: "the 17th of Nov" → ["the", "17", "of", "november"]."""
    return [MONTH_ABBR.get(w, w) for w in re.findall(r"\w+", digits(text).lower())]


def due_key(due: str) -> frozenset[str]:
    """ "the twentieth of October" and "Oct 20" → {"20", "october"}; "end of next week" →
    {"next", "week", "end"} ("end" kept when it's what distinguishes the date)."""
    ws = words(due)
    key = {w for w in ws if w not in FILLER}
    if "end" in ws:
        key.add("end")
    return frozenset(key)


def said(due: str, text: str) -> bool:
    """`text` says `due`: all its key words within a few words of each other."""
    key = due_key(due)
    if not key:
        return True
    ws = words(text)
    n = len(key) + WINDOW
    return any(key <= set(ws[i : i + n]) for i in range(max(1, len(ws) - n + 1)))


def due_said(due: str, text: str, notes_language: str, source_language: str) -> bool:
    """`text` (in the meeting's language) says `due` (in the notes'). English notes of an
    English meeting: `said`. Otherwise only digits can be read, each in its own language
    (D40): Hindi "17 नवंबर" on an English meeting is said when the text says 17; a due with
    no digits ("next week" in another language) is kept."""
    if notes_language == source_language == "en":
        return said(due, text)
    want = numbers.notes_numbers(fold_digits(due), notes_language)
    return want <= numbers.source_numbers_in(fold_digits(text), source_language)


MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# "11 November", "the 11th of Nov", "November 11", "Nov. the 11th": the day's digits and the
# month, next to each other.
DATE = re.compile(
    rf"\b(?:(?P<day>\d{{1,2}})(?P<suffix>st|nd|rd|th)?(?:\s+of)?\s+(?P<month>{_MONTH})"
    rf"|(?P<month2>{_MONTH})\.?\s+(?:the\s+)?(?P<day2>\d{{1,2}})(?P<suffix2>st|nd|rd|th)?)\b",
    re.I,
)

Date = tuple[str, int]  # month, day


def _date(m: re.Match[str]) -> Date:
    month = (m.group("month") or m.group("month2")).lower()
    return next(full for full in MONTHS if full.startswith(month[:3])), int(
        m.group("day") or m.group("day2")
    )


def dates_in(text: str) -> set[Date]:
    """The dates `text` says, number words read ("the fourteenth of November")."""
    return {_date(m) for m in DATE.finditer(digits(text))}


def _ordinal(day: int) -> str:
    return "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def check_dates(text: str, said_dates: set[Date]) -> tuple[str, list[tuple[str, str]], list[str]]:
    """Map's dates against the ones its segment says (8c): Tiny Aya wrote "November 1" for "11
    November", and the number guard kept it because the segment says 1 somewhere else. A date
    said is kept; one whose day is one digit edit from exactly one said date of the same month
    is repaired in place (the day's digits, and its suffix if it has one); otherwise it is kept
    and flagged, never guessed. Returns the text, the repairs (original, repaired), and the
    flagged dates."""
    repairs: list[tuple[str, str]] = []
    unverified: list[str] = []

    def check(m: re.Match[str]) -> str:
        month, day = _date(m)
        if (month, day) in said_dates:
            return m.group(0)
        near = {d for mo, d in said_dates if mo == month and one_edit(str(day), str(d))}
        if len(near) != 1:
            unverified.append(m.group(0))
            return m.group(0)
        [fixed_day] = near
        key = "day" if m.group("day") else "day2"
        suffix = m.group("suffix" if key == "day" else "suffix2")
        start, end = m.span(key)
        if suffix:
            end += len(suffix)
        new = str(fixed_day) + (_ordinal(fixed_day) if suffix else "")
        written = m.group(0)
        offset = m.start()
        fixed = written[: start - offset] + new + written[end - offset :]
        repairs.append((written, fixed))
        return fixed

    return DATE.sub(check, text), repairs, unverified
