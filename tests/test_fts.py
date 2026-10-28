"""FTS5 query terms shared by Ask (§5.7) and the search index (§13)."""

from cohr import fts


def test_query_terms() -> None:
    assert fts.query_terms("What is the monthly cloud bill?") == (
        ["monthly", "cloud", "bill"],
        False,
    )
    assert fts.query_terms("Who owns item 7?") == (["owns", "item", "7"], False)
    assert fts.query_terms("予算は いくら？") == (["予算は", "いくら"], True)
    assert fts.query_terms("予算") == ([], True)  # under 3 characters: trigram can't match


def test_match_quotes_every_term() -> None:
    assert fts.match(["cloud", "bill"]) == '"cloud" OR "bill"'
    assert fts.match(["cloud", "bill"], "AND") == '"cloud" AND "bill"'
    # A term is never read as an FTS5 operator or column filter.
    assert fts.match(["or", 'a"b', "text:x"]) == '"or" OR "a""b" OR "text:x"'


def test_fold_digits() -> None:
    assert (
        fts.fold_digits("$৪২,৫০০ and ٩٥٠ ms, ९५० ms, ９５０") == "$42,500 and 950 ms, 950 ms, 950"
    )
    assert fts.fold_digits("Budget ²½ Ⅳ") == "Budget ²½ Ⅳ"  # not decimal digits: kept
