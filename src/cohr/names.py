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
_LATIN_LETTER = r"[A-Za-zÀ-ɏ]"
_KATAKANA = re.compile(r"[ァ-ヺー]{2,}")  # ー (long vowel) included, ・ not


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


def said_in(name: str, text: str) -> bool:
    """`name` is said in `text`: as a whole word ("Tom", not the "tom" of "customers"; also
    "MARIAさん"), or, for a Latin name, as a katakana word with the same consonants (トム =
    Tom, D39)."""
    letter = _LATIN_LETTER if latin(name) else r"[^\W\d_]"  # kana next to "MARIA" is no letter
    if re.search(rf"(?<!{letter}){re.escape(name)}(?!{letter})", text, re.I):
        return True
    target = skeleton(name)
    return (
        latin(name)
        and len(target) >= 2
        and any(skeleton(k) == target for k in _KATAKANA.findall(text))
    )


def fold(text: str) -> str:
    """Accents dropped: the Spanish transcript says "Víctor", Map's English notes "Victor"."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# --- who a task is for (English, 8d) --------------------------------------------------------

# Capitalised words `said_names` keeps that are never a task's owner.
_NOT_PEOPLE = {
    *["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    *["mondays", "tuesdays", "wednesdays", "thursdays", "fridays"],
    *["january", "february", "march", "april", "may", "june", "july", "august"],
    *["september", "october", "november", "december"],
    *["i", "i'm", "i'll", "i'd", "i've"],
    *["we", "you", "he", "she", "they", "it", "there", "this", "that"],
    *["nobody", "everyone", "everybody", "someone", "somebody"],
    # replies and fillers that start a sentence: "Okay, will do." is no one's task (8e)
    *["okay", "ok", "yes", "yeah", "yep", "no", "sure", "thanks", "thank", "sorry", "right"],
    *["great", "good", "well", "alright", "hi", "hello", "hey", "please", "action"],
}
_PERSON = r"[A-Z][\w’'-]*(?: [A-Z][\w’'-]*)?"
# "Tom will update …", "Laura to send …": Map's way of writing a task as a decision or a task
# with its owner inside (8d: long-en f22, f31).
_SUBJECT = re.compile(
    rf"^(?P<name>{_PERSON})\s+(?:will|shall|should|is going to|to)\s+(?P<rest>\S.*)$", re.S
)
# Where a meeting says who does a task: "Maria, can you …", "Grace will …", "Action item for
# Tom: …" (8d).
_CUES = (
    re.compile(
        rf"^(?:(?i:okay|ok|so|right|and|then),?\s+)*(?P<name>{_PERSON}),\s+"
        r"(?i:can|could|would|will|please)\b"
    ),
    re.compile(rf"^(?P<name>{_PERSON})\s+(?:will|is going to|should|has to|needs to)\b"),
    re.compile(rf"\b(?i:action item for)\s+(?P<name>{_PERSON})"),
)


def person(name: str, said: set[str]) -> bool:
    """`name` is someone in the meeting: each word a said name (`said_names`, accents folded),
    not a weekday, month or pronoun ("Design will …" on AMI is no one: "design" is said too)."""
    words = name.split()
    return bool(words) and all(
        len(w) > 1 and fold(w) in said and w.casefold() not in _NOT_PEOPLE for w in words
    )  # one letter is no one: ASR's "the F site" for "the offsite" (8e)


def subject_task(text: str, said: set[str]) -> tuple[str, str] | None:
    """`text` says a person will do something: (the person, the rest), "Tom will update the
    pricing page" → ("Tom", "update the pricing page")."""
    m = _SUBJECT.match(text.strip())
    if m and person(m["name"], said):
        return m["name"], m["rest"]
    return None


def addressee(sentence: str, said: set[str]) -> str | None:
    """The one person a sentence gives a task to, in a cue position ("Maria, can you prepare
    …?"); None when it names nobody there, or names anyone else ("Tom, can you ask Maria …"):
    the owner would be a guess."""
    sentence = sentence.strip()
    cued: dict[str, str] = {}  # the person → the words the cue took
    for c in _CUES:
        if (m := c.search(sentence)) and person(name := _first_name(m["name"], said), said):
            cued[name] = m["name"]
    if len(cued) != 1:
        return None
    ((name, taken),) = cued.items()
    rest = _WORD.findall(sentence)[1:]  # the first word is capitalised anyway ("Okay", "Action")
    others = {w for w in rest if w[0].isupper() and person(w, said) and w not in taken.split()}
    return None if others else name


def _first_name(name: str, said: set[str]) -> str:
    """The first word of a two-word name when it is a person alone: ASR drops the colon of
    "Action item for Laura: schedule …" and capitalises the verb ("Laura Schedule", 8e)."""
    first = name.split()[0]
    return first if person(first, said) else name


def said_folded(name: str, text: str) -> bool:
    """`said_in`, else with accents dropped on both sides ("Victor" in "Víctor, ¿puedes …").
    Folding comes second: it would also drop katakana voicing marks (ビ → ヒ)."""
    return said_in(name, text) or said_in(fold(name), fold(text))


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
