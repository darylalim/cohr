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
