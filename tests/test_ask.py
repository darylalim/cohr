"""PLAN §5.7 Ask: history trimming, language fallback, citations in answers (no models)."""

from pathlib import Path

import pytest

from cohr import ask, pipeline, reduce, schemas
from cohr.ask import Answer, Sources, Turn
from cohr.config import Config, ReduceConfig
from cohr.schemas import MapFile, SegmentsFile, SlidesFile, Transcript
from tests.conftest import fake_manager, make_run
from tests.test_reduce import FakeR7B, _items, _map, _segments, _slide

# Reduce's config hash before Phase 5b: Ask must not move it (no re-runs of existing notes).
REDUCE_HASH = "bc243656d5a2b39c8087857bcb67ee36ad82435698811232789290da2fe4123e"


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
    (_, _, base, _), _ = ask.fit(lm(), "Q", [], "en", sources(), 10**9)
    history = turns(3)
    one_turn = ask.instruction("Q", history[2:], "en")
    cap = base + len(one_turn) - len(ask.instruction("Q", [], "en"))
    model = lm()
    (_, _, size, limit), dropped = ask.fit(model, "Q", history, "en", sources(), cap)
    assert dropped == 2 and limit is None and size <= cap
    sent = model.instructions[-1]
    assert history[2].question in sent  # the newest turn stays
    assert history[0].question not in sent and history[1].question not in sent


def test_only_the_last_three_turns_are_sent() -> None:
    history = turns(5)
    model = lm()
    _, dropped = ask.fit(model, "Q", history, "en", sources(), 10**9)
    assert dropped == 0
    sent = model.instructions[-1]
    assert [t.question in sent for t in history] == [False, False, True, True, True]


def test_documents_alone_over_the_cap_trim_transcripts() -> None:
    segs = [s.model_copy(update={"transcript_text": "word " * 400}) for s in _segments()]
    src = Sources(segments=segs, notes=_map(), slides=[], items=_items())
    model = lm()
    (_, _, size, limit), dropped = ask.fit(model, "Q", turns(2), "en", src, 2500)
    assert dropped == 2 and limit is not None and size <= 2500
    assert "Earlier questions" not in model.instructions[-1]


def test_too_long_even_trimmed_is_a_clear_error() -> None:
    with pytest.raises(ask.AskError, match="too long for Ask"):
        ask.fit(lm(), "Q", [], "en", sources(), 10)


def test_history_answers_carry_no_citation_markup() -> None:
    model = lm("<|START_RESPONSE|><co>We approved the 14,500 budget.</co: 0:[0]><|END_RESPONSE|>")
    first = ask.ask(model, "What was approved?", [], sources(), Config(), sure("en"))
    assert first.turn() == Turn("What was approved?", "We approved the 14,500 budget.")
    model.reply = "<co>Priya</co: 0:[1]> owns it."
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
