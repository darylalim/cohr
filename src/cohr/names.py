"""Owner names back in the meeting's spelling (6e-2): Map writes some owners in the notes'
script ("लॉरा" for Laura in Hindi notes, 6d-2/6e-1) although its prompt says to keep names
as they appear. A non-Latin owner that the cited transcript doesn't contain is matched to a
Latin name said there by its consonants after transliteration (anyascii)."""

import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

from anyascii import anyascii

# Letters transliteration blurs: Arabic writes long vowels as w/y (و ي), Japanese has no l/r
# contrast, Arabic no v (ف = f), Bengali and Japanese write v as b(h), Latin spells k as c
# ("Victor"). An h after a consonant is aspiration (Bengali "ভ" = bh).
_SAME = str.maketrans("bclvwy", "fkrfui")
_VOWELS = set("aeiou")
_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*")


def latin(text: str) -> bool:
    """Every letter is Latin script (accents allowed: "María")."""
    return all(unicodedata.name(ch, "").startswith("LATIN") for ch in text if ch.isalpha())


def plain(name: str) -> str:
    """Transliterated, lowercase letters only, doubles collapsed, blurred letters merged."""
    out = ""
    for ch in anyascii(name).lower():
        if not ch.isalpha() or out.endswith(ch):
            continue
        if ch == "h" and out and out[-1] not in "aeiou":
            continue
        out += ch
    return out.translate(_SAME)


def skeleton(name: str) -> str:
    """The consonants of `plain`: "विक्टर" → "fktr" = "Victor"; "لورا" → "rr" = "Laura"."""
    return "".join(ch for ch in plain(name) if ch not in _VOWELS)


def said_names(texts: Iterable[str]) -> list[str]:
    """Capitalised Latin words never written in lowercase, in first-seen order: names, not
    the words that start sentences ("Team" when "the team" is said too)."""
    seen: dict[str, None] = {}
    lower: set[str] = set()
    for text in texts:
        for word in _WORD.findall(text):
            if word[0].isupper() and latin(word):
                seen.setdefault(word, None)
            elif word[0].islower():
                lower.add(word)
    return [w for w in seen if w.lower() not in lower]


def restore(owner: str, texts: Iterable[str]) -> str:
    """The name said in `texts` that a non-Latin, one-word `owner` transliterates to: same
    consonants, closest spelling (Arabic "ماريا" → Maria, not Omar). The owner as written when
    it's Latin, said as written, or matches no name, or two names equally."""
    texts = list(texts)
    if latin(owner) or " " in owner.strip() or any(owner in t for t in texts):
        return owner
    target = skeleton(owner)
    if len(target) < 2:
        return owner
    scored = sorted(
        (
            (SequenceMatcher(None, plain(owner), plain(name)).ratio(), name)
            for name in said_names(texts)
            if skeleton(name) == target
        ),
        reverse=True,
    )
    if not scored or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return owner
    return scored[0][1]
