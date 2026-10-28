"""Owners back in the meeting's spelling (6e-2, carry-over 3)."""

import pytest

from cohr import names

# Shaped like the eval transcripts: names mostly start sentences ("Victor, can you…").
SAID = [
    "Okay team, quick sync. Action item for Laura send the contract to legal by Friday. "
    "Omar will draft the hiring plan. Victor, can you write the release notes? Maria, can you "
    "prepare the customer survey? Action item for Tom record the onboarding video. "
    "The team agreed."
]


@pytest.mark.parametrize(
    ("owner", "said"),
    [
        # Devanagari (clean-hi, 6e-1), Arabic (clean-ar, 6d-2), Bengali, Japanese
        *[
            (o, n)
            for o, n in zip(
                ["लॉरा", "ओमर", "विक्टर", "मारिया", "टॉम"],
                ["Laura", "Omar", "Victor", "Maria", "Tom"],
                strict=True,
            )
        ],
        *[
            (o, n)
            for o, n in zip(
                ["لورا", "عمر", "فيكتور", "ماريا", "توم"],
                ["Laura", "Omar", "Victor", "Maria", "Tom"],
                strict=True,
            )
        ],
        *[
            (o, n)
            for o, n in zip(
                ["লরা", "ওমর", "ভিক্টর", "মারিয়া", "টম"],
                ["Laura", "Omar", "Victor", "Maria", "Tom"],
                strict=True,
            )
        ],
        *[
            (o, n)
            for o, n in zip(
                ["ローラ", "オマール", "マリア", "トム"],
                ["Laura", "Omar", "Maria", "Tom"],
                strict=True,
            )
        ],
    ],
)
def test_restore_finds_the_name_said(owner: str, said: str) -> None:
    assert names.restore(owner, SAID) == said


@pytest.mark.parametrize(
    "owner",
    [
        "Laura",  # Latin: as written
        "María",  # Latin with an accent
        "मार्केटिंग मैनेजर",  # more than one word ("marketing manager"): a role, not a name
        "प्रिया",  # Priya: not said
        "राम",  # Ram: one consonant pair nobody said
    ],
)
def test_restore_keeps_the_owner_as_written(owner: str) -> None:
    assert names.restore(owner, SAID) == owner


def test_restore_keeps_an_owner_said_in_its_own_script() -> None:
    """A Hindi meeting: the name is in the transcript as written."""
    assert names.restore("लॉरा", ["लॉरा अनुबंध भेजेगी। Lara"]) == "लॉरा"


def test_words_that_start_sentences_are_not_names() -> None:
    """Must not fire: "Team" is also said as "the team", so it's no candidate for टॉम."""
    assert names.restore("टॉम", ["Team, welcome. The team agreed."]) == "टॉम"
    said = names.said_names(SAID)
    assert "The" not in said and {"Laura", "Omar", "Victor", "Maria", "Tom"} <= set(said)


def test_two_names_equally_close_keep_the_owner() -> None:
    assert names.restore("مر", ["Mira and Mara spoke."]) == "مر"


@pytest.mark.parametrize(
    ("name", "text", "said"),
    [
        ("Tom", "Action item for Tom record the video.", True),
        ("Tom", "When we talked to customers, it came up.", False),  # not the "tom" of customers
        ("Maria", "よかったですMARIAさん来週までに", True),  # xling-ja: Latin inside Japanese
        ("Tom", "トムさんへのアクションアイテム", True),  # katakana, same consonants (D39)
        ("Tom", "トライアル登録に移りましょう", False),
        ("Laura", "لورا سترسل العقد", False),  # only Latin names are read from katakana
    ],
)
def test_said_in(name: str, text: str, said: bool) -> None:
    assert names.said_in(name, text) is said
