"""Ask eval (PLAN §18 Ask row, Phase 6a): template questions, scoring, the answer cache, and
the end-to-end path through ask.run with a rule-based fake R7B. No models."""

import re
from pathlib import Path

import pytest

from cohr import evals, schemas
from cohr.ask import Answer
from cohr.config import Config
from cohr.evals import ItemReport, ItemSpec, Truth, TruthSlide
from cohr.schemas import Citation, Claim, MapFile, SegmentsFile, SlidesFile, Span, Transcript
from evals.gen.script import TOPICS
from tests.conftest import fake_manager, make_run
from tests.test_evals import fact
from tests.test_reduce import FakeR7B, _items, _map, _segments, _slide


def truth(item_id: str = "t") -> Truth:
    return Truth(
        item=ItemSpec(id=item_id, seed=1),
        gen_hash="x",
        language="en",
        duration_s=340.0,
        media="meeting.mp4",
        script_text="We approved the fourteen thousand five hundred budget. Last quarter it was "
        "nine thousand. There are 409 open tickets. Tom writes release notes by Friday.",
        lines=[],
        facts=[
            fact(
                "number", id="n1", text="Approved budget: $14,500", value=14500.0, t0=20.0, slide=0
            ),
            fact(
                "action",
                id="a1",
                text="write the release notes",
                owner="Tom",
                due="Friday",
                t0=320.0,
            ),
            fact("decision", id="d1", text="keep the onboarding checklist", t0=30.0),
            fact("number", id="n2", text="Open tickets: 409", value=409.0, t0=300.0),
        ],
        speech_spans=[Span(t0=0.0, t1=340.0)],
        gap_spans=[],
        slides=[TruthSlide(slide_id=0, title="Q4 Budget", cells=["14,500"], t0=0.0, t1=40.0)],
    )


def item_report_for(item_id: str) -> ItemReport:
    return ItemReport(
        id=item_id,
        run_id="r",
        seed=1,
        duration_s=340.0,
        metrics={"claims": 0},
        facts=[],
        made_up=[],
        wrong_fields=[],
        uncited=[],
        gap_texts=[],
        stages={},
    )


def answer(*claims: Claim, language: str = "en") -> Answer:
    return Answer(
        question="q",
        claims=list(claims),
        language=language,
        prompt_tokens=100,
        turns_dropped=0,
        ranked=None,
        dropped_citations=0,
        raw="",
    )


def seg(t: float, seg_id: int = 0) -> list[Citation]:
    return [Citation(kind="seg", id=seg_id, t=t)]


SLIDE0 = [Citation(kind="slide", id=0)]


# --- questions -----------------------------------------------------------------------------


def test_questions_are_deterministic_and_cover_every_kind() -> None:
    qs = evals.ask_questions(truth())
    assert qs == evals.ask_questions(truth())
    assert {q.kind for q in qs} == set(evals.ASK_KINDS)
    by_id = {q.id: q for q in qs}
    assert [q.question for q in qs if q.kind == "number"] == [
        "What is the approved budget?",
        "What is the number of open tickets?",
    ]
    assert by_id["a-a1"].question == "What does Tom need to do?"
    assert by_id["followup"].after == ["a-a1"] and by_id["followup"].fact == "a1"
    assert by_id["de-alone"].question == "Wie hoch ist das genehmigte Budget?"
    assert by_id["de-history"].fact == "n2" and by_id["de-history"].after == ["a-a1", "followup"]
    assert not any(q.kind == "number" and q.fact == "d1" for q in qs)  # decisions: no question
    # Every near-miss template, each on another number from the last, wrapping around.
    misses = [q for q in qs if q.kind == "near_miss"]
    assert [q.question for q in misses] == [
        "What was the number of open tickets two years ago?",
        "What is the approved budget for the marketing team?",
        "What is the number of open tickets for next year?",
    ]
    assert [q.fact for q in misses] == ["n2", "n1", "n2"]
    assert [q.id for q in misses] == ["near-miss-1", "near-miss-2", "near-miss-3"]


def test_every_generator_label_has_question_words() -> None:
    assert {t.label for t in TOPICS} <= set(evals.NUMBER_LABELS)
    periods = {t.period for t in TOPICS}
    for template in evals.NEAR_MISS:  # never a period the meeting states ("last quarter")
        assert not any(f" {p}" in template for p in periods)


# --- scoring -------------------------------------------------------------------------------


def score(q_id: str, a: Answer, t: Truth | None = None) -> evals.AskResult:
    t = t or truth()
    q = next(q for q in evals.ask_questions(t) if q.id == q_id)
    return evals.score_answer(q, a, {f.id: f for f in t.facts}, {0: 0}, evals.meeting_numbers(t))


def test_number_right_fact_and_citation() -> None:
    r = score("n-n1", answer(Claim(text="The approved budget is $14,500.", citations=SLIDE0)))
    assert r.stated and r.citation_ok and r.correct
    r = score("n-n1", answer(Claim(text="It is 14,500.", citations=seg(25.0))))
    assert r.correct and r.t_error == 5.0  # within ±15 s of t0 = 20
    r = score("n-n1", answer(Claim(text="It is 14,500.", citations=seg(200.0))))
    assert r.stated and r.citation_ok is False and not r.correct
    r = score("n-n1", answer(Claim(text="It is 14,000.", citations=SLIDE0)))
    assert not r.stated and r.citation_ok is None and not r.correct


def test_action_wrong_owner_is_not_stated() -> None:
    right = Claim(text="Tom needs to write the release notes.", citations=seg(321.0, 1))
    assert score("a-a1", answer(right)).correct
    wrong = Claim(text="Omar needs to write the release notes.", citations=seg(321.0, 1))
    assert not score("a-a1", answer(wrong)).stated
    other_task = Claim(text="Tom needs to book the venue.", citations=seg(321.0, 1))
    assert not score("a-a1", answer(other_task)).stated


def test_followup_needs_owner_and_due() -> None:
    both = answer(Claim(text="Tom owns it, by Friday.", citations=seg(320.0, 1)))
    assert score("followup", both).correct
    split = answer(
        Claim(text="Tom owns it.", citations=[]),
        Claim(text="It is due on Friday.", citations=seg(322.0, 1)),
    )
    assert score("followup", split).correct  # one claim stating it cites right
    no_due = answer(Claim(text="Tom owns it.", citations=seg(320.0, 1)))
    assert not score("followup", no_due).stated


def test_german_numbers_with_dot_separators() -> None:
    de = answer(
        Claim(text="Das genehmigte Budget beträgt 14.500 $.", citations=SLIDE0), language="de"
    )
    r = score("de-alone", de)
    assert r.correct and r.language == "de"
    assert evals.answer_numbers("42.500 $ und 1.5 Stunden") >= {42500.0, 1.5}
    nothing = answer(Claim(text="Ich konnte keine Informationen finden.", citations=[]))
    assert not score("de-history", nothing).correct


def test_near_miss_rule() -> None:
    planted = evals.meeting_numbers(truth())
    assert {14500.0, 9000.0, 409.0} <= planted and 4.0 not in planted  # 1–31 excluded
    related_first = answer(
        Claim(text="There are 409 open tickets.", citations=seg(300.0, 1)),
        Claim(text="The meeting doesn't give the marketing team's number.", citations=[]),
    )
    assert not score("near-miss-1", related_first).correct
    not_given_first = answer(
        Claim(text="The meeting doesn't give the marketing team's number.", citations=[]),
        Claim(text="It gives 409 open tickets overall.", citations=seg(300.0, 1)),
    )
    assert score("near-miss-1", not_given_first).correct
    same_sentence = answer(
        Claim(text="That isn't stated; overall there are 409 open tickets.", citations=[])
    )
    assert score("near-miss-1", same_sentence).correct
    no_number = answer(Claim(text="Only the overall queue is discussed.", citations=[]))
    assert score("near-miss-1", no_number).correct
    r = score("near-miss-1", related_first)
    assert r.citation_ok is None and r.stated is False


def test_ask_metrics() -> None:
    def res(kind: str, stated: bool, ok: bool | None, correct: bool) -> evals.AskResult:
        return evals.AskResult(
            id=kind,
            kind=kind,  # ty: ignore[invalid-argument-type]
            question="q",
            fact="f",
            answer="a",
            citations=[],
            language="en",
            stated=stated,
            citation_ok=ok,
            correct=correct,
        )

    m = evals.ask_metrics(
        [
            res("number", True, True, True),
            res("number", True, False, False),
            res("action", False, None, False),
            res("action", True, True, True),
            res("followup", True, True, True),
            res("de_alone", True, True, True),
            res("de_history", False, None, False),
            res("near_miss", False, None, False),
        ]
    )
    assert m["ask_questions"] == 8
    assert m["ask_accuracy"] == 0.5 and m["ask_stated"] == 0.75
    assert m["ask_citation_accuracy"] == round(2 / 3, 4)
    assert m["ask_followup"] == 1.0 and m["ask_xlang_alone"] == 1.0
    assert m["ask_xlang_history"] == 0.0 and m["ask_near_miss"] == 0.0


# --- cache ---------------------------------------------------------------------------------


def test_cache_key() -> None:
    config = Config()
    turn = Answer.turn(answer(Claim(text="Tom writes the notes.", citations=[])))
    base = evals.ask_cache_key("Q?", [turn], config, "src")
    assert base == evals.ask_cache_key("Q?", [turn], Config(), "src")
    assert base != evals.ask_cache_key("Q2?", [turn], config, "src")
    assert base != evals.ask_cache_key("Q?", [], config, "src")
    assert base != evals.ask_cache_key("Q?", [turn], config.model_copy(update={"seed": 1}), "src")
    assert base != evals.ask_cache_key("Q?", [turn], config, "other documents")
    models = dict(config.models)
    models["reduce"] = models["reduce"].model_copy(update={"revision": "f" * 40})
    assert base != evals.ask_cache_key(
        "Q?", [turn], config.model_copy(update={"models": models}), "src"
    )


def test_cache_key_follows_the_ask_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    base = evals.ask_cache_key("Q?", [], Config(), "src")
    real = evals.prompt_hash
    monkeypatch.setattr(evals, "prompt_hash", lambda p: "changed" if p == "ask_turn" else real(p))
    assert evals.ask_cache_key("Q?", [], Config(), "src") != base


# --- end to end through ask.run ---------------------------------------------------------------


class RuleR7B(FakeR7B):
    """Replies by the question in the last instruction; documents: 0 = seg 0 (0–40 s),
    1 = seg 1 (300–340 s), 2 = slide 0."""

    REPLIES = {
        "What is the approved budget?": "<co>The approved budget is 14,500.</co: 0:[2]>",
        "What is the number of open tickets?": "<co>There are 409 open tickets.</co: 0:[0]>",
        "What does Tom need to do?": "<co>Tom writes release notes.</co: 0:[1]>",
        "Who owns that, and by when?": "<co>Tom owns it, by Friday.</co: 0:[1]>",
        "Wie hoch ist das genehmigte Budget?": "<co>Das Budget beträgt 14.500.</co: 0:[2]>",
        "Wie hoch ist die Anzahl offener Tickets?": "Ich konnte keine Informationen finden.",
    }
    NEAR_MISS = "The meeting doesn't give that. <co>Overall: 409 open tickets.</co: 0:[1]>"

    def __init__(self) -> None:
        super().__init__("")

    def generate(self, prompt: str, *, max_tokens: int, temperature: float, seed: int) -> str:
        super().generate(prompt, max_tokens=max_tokens, temperature=temperature, seed=seed)
        question = re.findall(r"^Question: (.*)$", self.instructions[-1], re.M)[-1]
        return self.REPLIES.get(question, self.NEAR_MISS)


def ask_run(tmp_path: Path) -> Path:
    run_dir = make_run(tmp_path)
    schemas.save(run_dir, SlidesFile(slides=[_slide(0, "Q4 Budget")]))
    schemas.save(run_dir, SegmentsFile(segments=_segments(), unaligned_slide_ids=[]))
    schemas.save(run_dir, MapFile(target_language="en", notes=list(_map().values())))
    schemas.save(run_dir, Transcript(language="en", runtime="mlx-speech", items=_items()))
    return run_dir


def test_ask_item_end_to_end_and_cached(tmp_path: Path) -> None:
    run_dir = ask_run(tmp_path)
    lm = RuleR7B()
    models = fake_manager(Config(), tmp_path, reduce=lm)
    asker = evals.ask_session(models)
    asker.open(evals.ask_cache(tmp_path, "t"))
    rep = evals.ask_item(item_report_for("t"), truth(), run_dir, asker)
    asker.save()

    by_id = {r.id: r for r in rep.ask}
    assert by_id["n-n1"].correct and by_id["n-n1"].citations == ["slide 0"]
    assert by_id["n-n2"].stated and not by_id["n-n2"].citation_ok  # seg 0, fact at 300 s
    assert by_id["a-a1"].correct and by_id["followup"].correct
    assert by_id["de-alone"].correct and by_id["de-alone"].language == "de"
    assert not by_id["de-history"].correct and by_id["near-miss-1"].correct
    # The follow-up saw the first answer as history; the German one saw both turns.
    follow = next(i for i in lm.instructions if "Question: Who owns that" in i)
    assert "Tom writes release notes." in follow
    de = next(i for i in lm.instructions if "Question: Wie hoch ist die Anzahl" in i)
    assert "Tom owns it, by Friday." in de and "Tom writes release notes." in de
    m = rep.metrics
    assert m["ask_questions"] == 9 and m["ask_accuracy"] == round(2 / 3, 4)
    assert m["ask_followup"] == 1.0 and m["ask_near_miss"] == 1.0
    assert m["ask_xlang_alone"] == 1.0 and m["ask_xlang_history"] == 0.0
    assert asker.calls == 9 and asker.hits == 0

    # Re-scoring with nothing stale asks no model.
    def no_model(*_a: object) -> Answer:
        raise AssertionError("the model was asked")

    cached = evals.CachedAsk(no_model)
    cached.open(evals.ask_cache(tmp_path, "t"))
    again = evals.ask_item(item_report_for("t"), truth(), run_dir, cached)
    assert again == rep and cached.hits == 9 and cached.calls == 0

    # A run with another pipeline seed is a cache miss (D16 runs don't share answers).
    config = schemas.load(run_dir, schemas.Manifest)
    config.config["seed"] = 2
    schemas.save(run_dir, config)
    other = evals.CachedAsk(lambda *_a: answer())
    other.open(evals.ask_cache(tmp_path, "t"))
    evals.ask_item(item_report_for("t"), truth(), run_dir, other)
    assert other.calls == 9


def test_ask_without_models_and_no_cache_is_a_clear_error(tmp_path: Path) -> None:
    run_dir = ask_run(tmp_path)
    asker = evals.ask_session(None)
    asker.open(evals.ask_cache(tmp_path, "t"))
    with pytest.raises(RuntimeError, match="needs a ModelManager"):
        evals.ask_item(item_report_for("t"), truth(), run_dir, asker)


def test_report_shows_ask(tmp_path: Path) -> None:
    run_dir = ask_run(tmp_path)
    asker = evals.ask_session(fake_manager(Config(), tmp_path, reduce=RuleR7B()))
    asker.open(evals.ask_cache(tmp_path, "t"))
    rep = evals.ask_item(item_report_for("t"), truth(), run_dir, asker)
    summary = evals.summarize([rep])
    assert summary["ask_accuracy"] == rep.metrics["ask_accuracy"]
    report = evals.Report(
        created_at=evals.EPOCH,
        set="quick",
        git_sha=None,
        models={},
        prompts={},
        items=[rep],
        summary=summary,
        stages={},
    )
    md = evals.render_markdown(report)
    assert "| ask_accuracy | 66.7% |" in md
    assert "- ask number cited off" in md and "- ask de_history missed" in md
    assert evals.load_report(evals.write_report(report, tmp_path)) == report
