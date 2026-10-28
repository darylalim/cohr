"""SQLite FTS5 query terms, shared by Ask's ranking (PLAN §5.7, D27) and the cross-meeting
search index (§13). Pure: no models.

Space-separated languages use the `unicode61` tokenizer and words (English stopwords left
out). Chinese, Japanese and Korean have no spaces between words, so text with any of them is
matched with the `trigram` tokenizer, as every 3-character window of each run (D17: trigram
can't match shorter terms).
"""

import re
import unicodedata
from collections.abc import Sequence

from cohr.fuse import STOPWORDS

CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_TERM = re.compile(r"[^\W_]+")
TRIGRAM = 3  # FTS5's trigram tokenizer matches terms of at least 3 characters
WORDS = "unicode61 remove_diacritics 2"
CJK_LANGUAGES = frozenset({"ja", "ko", "zh"})  # rows in these are searched by trigram


class _Digits(dict[int, int]):
    """str.translate table: every Unicode decimal digit (Bengali ৯, Arabic-Indic ٩, Devanagari
    ९, fullwidth ９ …) to its ASCII digit; other characters map to themselves."""

    def __missing__(self, cp: int) -> int:
        ch = chr(cp)
        out = ord(str(unicodedata.decimal(ch))) if unicodedata.category(ch) == "Nd" else cp
        self[cp] = out
        return out


_DIGITS = _Digits()


def fold_digits(text: str) -> str:
    """ASCII digits for every script's: Tiny Aya writes Bengali notes with Bengali digits
    ("৯৫০"), and a search for "950" must find them (Phase 7 search check)."""
    return text.translate(_DIGITS)


def query_terms(text: str) -> tuple[list[str], bool]:
    """FTS5 terms for a query and whether to use the trigram tokenizer: words (English
    stopwords left out) for space-separated languages; for text with Chinese, Japanese or
    Korean, every 3-character window of each run (trigram can't match shorter terms)."""
    cjk = bool(CJK.search(text))
    out: list[str] = []
    for run in _TERM.findall(text.casefold()):
        if not cjk:
            if run not in STOPWORDS and (len(run) > 1 or run.isdigit()):
                out.append(run)
            continue
        out += [run[i : i + TRIGRAM] for i in range(len(run) - TRIGRAM + 1)]
    return list(dict.fromkeys(out)), cjk


def tokenizer(cjk: bool) -> str:
    return "trigram" if cjk else WORDS


def match(terms: Sequence[str], op: str = "OR") -> str:
    """An FTS5 MATCH expression: each term quoted (so no term is read as an operator), joined
    by `op` ("OR" or "AND")."""
    return f" {op} ".join('"' + t.replace('"', '""') + '"' for t in terms)
