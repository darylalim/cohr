"""Due dates as words (PLAN §5.6, D40): "the 17th of November" and "17 November" share a key,
so the Reduce due guard and the evals agree on what a due says. English words only."""

import re

from cohr.numbers import digits

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
