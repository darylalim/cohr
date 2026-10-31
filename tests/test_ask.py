"""PLAN §5.7 Ask: history trimming, language fallback, citations in answers (no models)."""

from pathlib import Path

import pytest

from cohr import ask, pipeline, reduce, schemas
from cohr.ask import Answer, Sources, Turn
from cohr.config import Config, ReduceConfig, load_prompt
from cohr.schemas import (
    Citation,
    Claim,
    MapFile,
    Segment,
    SegmentsFile,
    SlidesFile,
    Transcript,
    TranscriptItem,
)
from tests.conftest import fake_manager, make_run
from tests.test_reduce import FakeR7B, _items, _map, _segments, _slide

# Reduce's config hash: Ask must not move it (no re-runs of existing notes). Moved on purpose
# only by Reduce changes: 6c reduce.md (one fact per item); 6c two-pass (reduce_pass2 in
# the prompt list, NotesDoc v4; reduce_part for long-meeting pass 1); 6e-2 reduce.max_tokens
# 2048 → 8192 (NotesDoc v5, truncated).
REDUCE_HASH = "5d92d62a572d76ee906387a0a72476cd041b34db1f74f59c0a684933d2e595e6"


def sources() -> Sources:
    return Sources(
        segments=_segments(), notes=_map(), slides=[_slide(0, "Q4 Budget")], items=_items()
    )


def sure(lang: str) -> ask.Detect:
    return lambda _text: [(lang, 0.9), ("en", 0.05)]


def turns(n: int, size: int = 100) -> list[Turn]:
    return [Turn(f"question {k} " + "?" * size, f"answer {k} " + "." * size) for k in range(n)]


def lm(reply: str = "", tokens_per_char: float = 1.0) -> FakeR7B:
    return FakeR7B(reply, tokens_per_char=tokens_per_char)


# --- prompt and history --------------------------------------------------------------------


def test_instruction_puts_the_question_and_language_last() -> None:
    text = ask.instruction("Wann startet die Beta?", turns(1, 0), "de")
    assert text.rstrip().endswith("Question: Wann startet die Beta?\n\nAnswer in German.")
    assert "Q: question 0 \nA: answer 0" in text
    assert text.index("Q: question 0") < text.index("Question: Wann")
    alone = ask.instruction("Wann?", [], "de")
    assert "Earlier questions" not in alone and "{" not in alone


def test_history_is_dropped_oldest_first() -> None:
    base = ask.fit(lm(), "Q", [], "en", sources(), 10**9).tokens
    history = turns(3)
    one_turn = ask.instruction("Q", history[2:], "en")
    cap = base + len(one_turn) - len(ask.instruction("Q", [], "en"))
    model = lm()
    fitted = ask.fit(model, "Q", history, "en", sources(), cap)
    assert fitted.turns_dropped == 2 and not fitted.ranked and fitted.tokens <= cap
    sent = model.instructions[-1]
    assert history[2].question in sent  # the newest turn stays
    assert history[0].question not in sent and history[1].question not in sent


def test_only_the_last_three_turns_are_sent() -> None:
    history = turns(5)
    model = lm()
    assert ask.fit(model, "Q", history, "en", sources(), 10**9).turns_dropped == 0
    sent = model.instructions[-1]
    assert [t.question in sent for t in history] == [False, False, True, True, True]


# --- ranking over the cap (§5.7, D27) ------------------------------------------------------

TOPICS = [
    "The support queue has 390 open tickets this week.",
    "Cloud costs: the monthly bill is $11,800.",
    "Hiring: 75 candidates are in the pipeline.",
]


def topic_sources() -> Sources:
    segs = [
        Segment(
            seg_id=k,
            t0=100.0 * k,
            t1=100.0 * k + 90,
            chunk_ids=[k],
            transcript_text=f"{text} " + "filler words " * 20,
            slide_ids=[],
        )
        for k, text in enumerate(TOPICS)
    ]
    items = [
        TranscriptItem(chunk_id=k, t0=100.0 * k, t1=100.0 * k + 90, text=text)
        for k, text in enumerate(TOPICS)
    ]
    return Sources(segments=segs, notes={}, slides=[_slide(0, "Hiring plan")], items=items)


def test_rank_by_bm25_then_meeting_order() -> None:
    docs, _ = reduce.documents(topic_sources().segments, {}, topic_sources().slides)
    assert ask.rank("How many candidates are in the hiring pipeline?", docs)[:2] == [2, 3]
    assert ask.rank("cloud bill", docs) == [1, 0, 2, 3]
    assert ask.rank("the", docs) == [0, 1, 2, 3]  # no terms: meeting order


def test_rank_cjk_with_trigrams() -> None:
    docs = [
        {"title": "0:00–1:00", "snippet": "サポートの未処理チケットは390件です。"},
        {"title": "1:00–2:00", "snippet": "来四半期の予算は14,500ドルに決まりました。"},
    ]
    assert ask.rank("来四半期の予算はいくらですか？", docs) == [1, 0]


def test_documents_over_the_cap_are_ranked_and_sent_in_meeting_order() -> None:
    src = topic_sources()
    question = "What is the cloud bill and the hiring pipeline?"
    cap = ask.fit(lm(), question, [], "en", src, 10**9).tokens - 200  # one segment too many
    fitted = ask.fit(lm(), question, [], "en", src, cap)
    assert fitted.ranked and fitted.tokens <= cap
    assert fitted.targets == [("seg", 1), ("seg", 2), ("slide", 0)]  # meeting order
    assert TOPICS[0] not in fitted.prompt
    assert fitted.prompt.index(TOPICS[1]) < fitted.prompt.index(TOPICS[2])


def test_ranked_answers_cite_through_the_documents_sent() -> None:
    src = topic_sources()
    cap = ask.fit(lm(), "Q", [], "en", src, 10**9).tokens - 200
    model = lm("<co>75 candidates</co: 0:[1]> are in the pipeline.")
    config = Config(reduce=ReduceConfig(max_input_tokens=cap))
    answer = ask.ask(model, "How many hiring candidates?", [], src, config, sure("en"))
    assert answer.ranked is not None and ("seg", 2) in answer.ranked
    assert [(c.kind, c.id) for c in answer.claims[0].citations] == [answer.ranked[1]]
    assert answer.scope() == "Answer based on the 2 most relevant segments and 1 slide"


def test_follow_ups_rank_with_the_last_turn() -> None:
    src = topic_sources()
    history = [Turn("What is the cloud bill?", "The monthly bill is $11,800.")]
    full = ask.fit(lm(), "And who owns that?", history, "en", src, 10**9).tokens
    fitted = ask.fit(lm(), "And who owns that?", history, "en", src, full - 500)
    assert fitted.ranked and fitted.turns_dropped == 0
    assert fitted.targets[0] == ("seg", 1)  # only the last turn's terms match anything


def test_ranking_sends_at_most_max_ranked_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ask, "MAX_RANKED", 2)
    src = topic_sources()
    fitted = ask.fit(lm(), "hiring cloud tickets", [], "en", src, 10**6 - 1)
    assert not fitted.ranked  # under the cap: everything, MAX_RANKED doesn't apply
    cap = ask.fit(lm(), "hiring cloud tickets", [], "en", src, 10**9).tokens - 1
    fitted = ask.fit(lm(), "hiring cloud tickets", [], "en", src, cap)
    assert fitted.ranked and len(fitted.targets) == 2


def test_scope_text() -> None:
    def scope(ranked: list[reduce.Target] | None) -> str | None:
        return Answer("q", [], "en", 1, 0, ranked, 0, "").scope()

    assert scope(None) is None
    assert scope([("seg", 3)]) == "Answer based on the 1 most relevant segment"
    assert scope([("seg", 1), ("seg", 3), ("slide", 2), ("slide", 4)]) == (
        "Answer based on the 2 most relevant segments and 2 slides"
    )


def test_too_long_for_even_one_document_is_a_clear_error() -> None:
    with pytest.raises(ask.AskError, match="too long for Ask"):
        ask.fit(lm(), "Q", [], "en", sources(), 10)


def test_history_answers_carry_no_citation_markup() -> None:
    model = lm("<|START_RESPONSE|><co>We approved the 14,500 budget.</co: 0:[0]><|END_RESPONSE|>")
    first = ask.ask(model, "What was approved?", [], sources(), Config(), sure("en"))
    assert first.turn() == Turn("What was approved?", "We approved the 14,500 budget.")
    model.replies = ["<co>Priya</co: 0:[1]> owns it."]
    ask.ask(model, "Who owns that?", [first.turn()], sources(), Config(), sure("en"))
    sent = model.instructions[-1]
    assert "A: We approved the 14,500 budget." in sent
    assert "<co" not in sent and "</co" not in sent


# --- answer language -----------------------------------------------------------------------


def test_language_is_the_questions_when_lingua_is_sure() -> None:
    assert ask.answer_language("Wer ist dafür verantwortlich?", "en", sure("de")) == "de"


@pytest.mark.parametrize(
    "ranked",
    [
        [("nl", 0.25), ("en", 0.25)],  # a tie: no lead
        [("fr", 0.15), ("it", 0.05)],  # under the minimum confidence
        [],
    ],
)
def test_language_falls_back_when_lingua_is_unsure(ranked: list[tuple[str, float]]) -> None:
    assert ask.answer_language("Wat is het budget?", "ja", lambda _t: ranked) == "ja"


def test_one_word_questions_use_the_target_language() -> None:
    def never(_text: str) -> list[tuple[str, float]]:
        raise AssertionError("not detected")

    assert ask.answer_language("Priya?", "de", never) == "de"
    assert ask.answer_language("谁负责？", "en", sure("zh")) == "zh"  # CJK has no spaces


def test_real_lingua_on_sample_questions() -> None:
    assert ask.answer_language("Wer ist dafür verantwortlich?", "en") == "de"
    assert ask.answer_language("When does the beta launch?", "de") == "en"
    assert ask.answer_language("Priya?", "en") == "en"


def test_prompt_asks_for_the_answer_language() -> None:
    model = lm("<co>Ja.</co: 0:[0]>")
    answer = ask.ask(model, "Wann startet die Beta?", [], sources(), Config(), sure("de"))
    assert answer.language == "de"
    assert model.instructions[-1].rstrip().endswith("Answer in German.")


# --- citations -----------------------------------------------------------------------------


def test_answer_citations_are_parsed_and_refined() -> None:
    reply = (
        "<|START_RESPONSE|><co>We approved the 14,500 budget</co: 0:[0,2]> today. "
        "The beta is <co>on 3 November</co: 0:[1]>.<|END_RESPONSE|>"
    )
    answer = ask.ask(lm(reply), "What happened?", [], sources(), Config(), sure("en"))
    first, second = answer.claims
    assert [(c.kind, c.id) for c in first.citations] == [("seg", 0), ("slide", 0)]
    assert first.citations[0].t == 20.0  # chunk 1 ("We approved the 14,500 budget today.")
    assert [(c.kind, c.id, c.t) for c in second.citations] == [("seg", 1, 300.0)]
    assert answer.dropped_citations == 0 and answer.uncited == 0


def test_bad_indices_are_dropped_and_uncited_sentences_kept() -> None:
    reply = "The meeting doesn't say. <co>Maybe</co: 0:[99]> later."
    answer = ask.ask(lm(reply), "When?", [], sources(), Config(), sure("en"))
    assert [c.text for c in answer.claims] == ["The meeting doesn't say.", "Maybe later."]
    assert all(c.citations == [] for c in answer.claims)
    assert answer.dropped_citations == 1 and answer.uncited == 2


def test_generation_uses_the_runs_reduce_settings() -> None:
    model = lm("Ok.")
    config = Config(seed=5, reduce=ReduceConfig(temperature=0.2))
    ask.ask(model, "Anything?", [], sources(), config, sure("en"))
    assert model.calls == [{"max_tokens": ask.MAX_ANSWER_TOKENS, "temperature": 0.2, "seed": 5}]


def test_empty_run_and_empty_question_are_refused() -> None:
    empty = Sources(segments=[], notes={}, slides=[], items=[])
    with pytest.raises(ask.AskError, match="no transcript"):
        ask.ask(lm(), "Anything?", [], empty, Config(), sure("en"))
    with pytest.raises(ask.AskError):
        ask.ask(lm(), "  ", [], sources(), Config(), sure("en"))


# --- near-miss: periods and groups the meeting never gives (8j, D52) -----------------------

# Fixture documents say: "Welcome everyone to the Atlas sync. We approved the 14,500 budget
# today. Beta launch is on 3 November. Tom writes release notes.", Map's due "end of next
# week" and slide "Q4 Budget".
SAID = ask.said_text(sources())


@pytest.mark.parametrize(
    ("question", "unsaid"),
    [
        ("What was the approved budget two years ago?", ["two years ago"]),
        ("What is the budget for the marketing team?", ["for the marketing team"]),
        ("What is the budget for next year?", ["next year"]),
        ("How much is the budget in 2023?", ["in 2023"]),
        ("What's the budget in March?", ["in March"]),
        (
            "What is the budget for next year for the sales team?",
            ["next year", "for the sales team"],
        ),
        # Must not fire: no qualifier, or one the documents say.
        ("What is the approved budget?", []),
        ("What is the budget for Q4?", []),  # the slide's own label
        ("What is due next week?", []),  # Map's due: "end of next week"
        ("What is the budget for the team?", []),  # names no group
        ("What is the budget for the Atlas team?", []),  # "Atlas" is said
        # Not a value question, or not English: left to R7B and ask.md.
        ("What does Tom need to do by next year?", []),
        ("Who owns the budget for the marketing team?", []),
        ("Wie hoch ist das Budget für nächstes Jahr?", []),
    ],
)
def test_unsaid_qualifiers(question: str, unsaid: list[str]) -> None:
    assert ask.unsaid_qualifiers(question, SAID) == unsaid


def test_qualifiers_are_read_in_any_wording_the_meeting_uses() -> None:
    said = ask.said_text(
        Sources(
            segments=[],
            notes={},
            slides=[_slide(0, "Costs", "Cloud bill 2 years ago: 8,200. Third quarter: 9,000.")],
            items=[],
        )
    )
    assert ask.unsaid_qualifiers("What was the cloud bill two years ago?", said) == []
    assert ask.unsaid_qualifiers("What is the cloud bill in Q3?", said) == []
    long_en = ask.fold("Read the marketing emails. The cloud bill for the Central region.")
    assert ask.unsaid_qualifiers("What is the cloud bill for the Central region?", long_en) == []
    # A group the meeting names anywhere counts as said: R7B answers it alone (no false "not
    # given" on a number the meeting does give for that group).
    q = "What is the cloud bill for the Central region for the marketing team?"
    assert ask.unsaid_qualifiers(q, long_en) == []
    assert ask.fold("the previous quarter, a year ago") == "the last quarter, 1 year ago"


def _claims(*texts: str) -> list[Claim]:
    return [Claim(text=t, citations=[Citation(kind="seg", id=0, t=20.0)]) for t in texts]


def test_qualifier_copied_onto_a_meeting_number() -> None:  # shape (a)
    q = "What is the monthly recurring revenue for the marketing team?"
    out = ask.deny_unsaid(
        q,
        _claims("The monthly recurring revenue for the marketing team is $273,000."),
        ["for the marketing team"],
    )
    assert [c.text for c in out] == [
        "The meeting doesn't give the monthly recurring revenue for the marketing team.",
        "The monthly recurring revenue is $273,000.",
    ]
    assert out[0].citations == []  # nothing to cite: uncited, flagged like any (§5.6)
    assert out[1].citations == [Citation(kind="seg", id=0, t=20.0)]


def test_qualifier_dropped_and_the_number_given() -> None:  # shape (b)
    claims = _claims("The monthly cloud bill is $8,900.", "Last month it was $8,200.")
    out = ask.deny_unsaid("What is the monthly cloud bill for next year?", claims, ["next year"])
    assert [c.text for c in out] == [
        "The meeting doesn't give the monthly cloud bill for next year.",
        "The monthly cloud bill is $8,900.",
        "Last month it was $8,200.",
    ]
    assert out[1:] == claims


def test_related_numbers_before_the_denial() -> None:  # shape (c)
    claims = _claims(
        "The approved budget for next quarter is $57,000.",
        "The approved budget two years ago is not mentioned in the meeting.",
    )
    out = ask.deny_unsaid("What was the approved budget two years ago?", claims, ["two years ago"])
    assert [c.text for c in out] == [
        "The meeting doesn't give the approved budget two years ago.",
        "The approved budget for next quarter is $57,000.",
    ]


def test_an_answer_that_opens_with_the_denial_is_kept() -> None:
    claims = _claims(
        "The meeting doesn't give the page load time for next year.",
        "It says the average page load time is 370 milliseconds.",
    )
    q = "What is the page load time for next year?"
    assert ask.deny_unsaid(q, claims, ["next year"]) == claims
    assert ask.deny_unsaid(q, claims, []) == claims


def test_ask_opens_with_not_given_on_an_unsaid_qualifier() -> None:
    reply = "<co>The approved budget for the marketing team is 14,500.</co: 0:[0,2]>"
    q = "What is the approved budget for the marketing team?"
    answer = ask.ask(lm(reply), q, [], sources(), Config(), sure("en"))
    assert answer.unsaid == ["for the marketing team"]
    first, second = answer.claims
    assert first.text == "The meeting doesn't give the approved budget for the marketing team."
    assert second.text == "The approved budget is 14,500."
    assert [(c.kind, c.id) for c in second.citations] == [("seg", 0), ("slide", 0)]
    assert second.citations[0].t == 20.0  # refined on the reply's own text, as before
    assert answer.raw == reply  # what R7B wrote is kept as it was


@pytest.mark.parametrize(
    ("language", "config"),
    [("de", Config()), ("en", Config(source_language="ja"))],
)
def test_the_check_is_for_english_questions_on_english_meetings(
    language: str, config: Config
) -> None:
    reply = "<co>The approved budget for the marketing team is 14,500.</co: 0:[0]>"
    q = "What is the approved budget for the marketing team?"
    answer = ask.ask(lm(reply), q, [], sources(), config, sure(language))
    assert answer.unsaid == []
    assert [c.text for c in answer.claims] == [
        "The approved budget for the marketing team is 14,500."
    ]


def test_not_given_sentence_is_a_prompt_file() -> None:
    assert "{subject}" in load_prompt("ask_not_given")


# --- run(): the ModelManager and the run's config --------------------------------------------


def _ask_run(tmp_path: Path, config: Config) -> Path:
    run_dir = make_run(tmp_path, config=config)
    schemas.save(run_dir, SlidesFile(slides=[_slide(0, "Q4 Budget")]))
    schemas.save(run_dir, SegmentsFile(segments=_segments(), unaligned_slide_ids=[]))
    schemas.save(
        run_dir, MapFile(target_language=config.target_language, notes=list(_map().values()))
    )
    schemas.save(run_dir, Transcript(language="en", runtime="mlx-speech", items=_items()))
    return run_dir


def test_run_uses_the_runs_config_and_restores_the_managers(tmp_path: Path) -> None:
    run_dir = _ask_run(tmp_path, Config(seed=7, target_language="de"))
    model = lm("<co>Ja.</co: 0:[0]>")
    manager_config = Config(seed=1)
    models = fake_manager(manager_config, tmp_path, reduce=model)
    answer = ask.run(models, run_dir, "Priya?", [], detect=sure("en"))
    assert isinstance(answer, Answer) and answer.language == "de"  # one word: run's target
    assert model.calls[0]["seed"] == 7
    assert models.config is manager_config


# --- Reduce is unchanged -------------------------------------------------------------------


def test_reduce_config_hash_is_unchanged() -> None:
    assert pipeline.config_hash(reduce.STAGE, Config()) == REDUCE_HASH
