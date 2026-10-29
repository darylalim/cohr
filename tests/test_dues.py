import pytest

from cohr import dues


@pytest.mark.parametrize(
    ("due", "text", "said"),
    [
        ("17 November", "Tom, finish the post mortem by the 17th of November.", True),
        ("Friday", "Laura will configure the alerts by Friday.", True),
        ("end of next week", "Maria, the quotes by the end of next week, please.", True),
        ("Oct 20", "Priya has until October twentieth.", True),
        ("1 November", "Grace summarise the competitor research by 11 November.", False),
        # Said elsewhere in the meeting is not said here.
        ("end of next week", "The dashboard for revenue was updated on Monday.", False),
        # The words apart: "end" and "meeting" 10 words away don't say "End of meeting".
        (
            "End of meeting",
            "At the end we should talk about the colours, before this whole meeting",
            False,
        ),
    ],
)
def test_said(due: str, text: str, said: bool) -> None:
    assert dues.said(due, text) is said


def test_due_key_keeps_end() -> None:
    assert dues.due_key("by the end of next week") == {"end", "next", "week"}
    assert dues.due_key("the twentieth of Oct") == {"20", "october"}


@pytest.mark.parametrize(
    ("due", "text", "notes", "source", "said"),
    [
        ("the 17th of November", "Tom, the post mortem by 17 November.", "en", "en", True),
        ("End of next week", "The revenue dashboard was updated on Monday.", "en", "en", False),
        # Other languages: digits only, each read in its own language (D40).
        ("17 नवंबर", "Tom, the post mortem by the 17th of November.", "hi", "en", True),
        ("Do. 4:44", "Tom will draft the hiring plan by the end of next week.", "de", "en", False),
        ("अगले हफ्ते के अंत तक", "Tom will draft it by the end of next week.", "hi", "en", True),
    ],
)
def test_due_said(due: str, text: str, notes: str, source: str, said: bool) -> None:
    assert dues.due_said(due, text, notes, source) is said


def test_dates_in_reads_words_and_either_order() -> None:
    text = "Grace, by 11 November. Tom by the fourteenth of Nov, Priya by October 3rd."
    assert dues.dates_in(text) == {("november", 11), ("november", 14), ("october", 3)}


@pytest.mark.parametrize(
    ("text", "fixed", "repairs"),
    [
        # 8c (long-en a-f82): said "11 November", Map wrote "November 1".
        (
            "Summarize competitor research by November 1.",
            "Summarize competitor research by November 11.",
            [("November 1", "November 11")],
        ),
        (
            "by the 1st of November",
            "by the 11th of November",
            [("1st of November", "11th of November")],
        ),
    ],
)
def test_check_dates_repairs_a_day_one_edit_from_one_said_date(
    text: str, fixed: str, repairs: list[tuple[str, str]]
) -> None:
    assert dues.check_dates(text, {("november", 11), ("october", 1)}) == (fixed, repairs, [])


def test_check_dates_must_not_fire() -> None:
    said = {("november", 11), ("november", 14)}
    # Said, in other words.
    assert dues.check_dates("by Nov. the 14th", said) == ("by Nov. the 14th", [], [])
    # Two said days one edit away: flagged, never guessed.
    assert dues.check_dates("by November 1", said) == ("by November 1", [], ["November 1"])
    # Another month's day isn't a candidate.
    assert dues.check_dates("by 11 October", said) == ("by 11 October", [], ["11 October"])
    # Not a date: a year after a month, a number with no month.
    assert dues.check_dates("in May 2025, 11 tickets", said) == ("in May 2025, 11 tickets", [], [])
