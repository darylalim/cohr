"""Judge metrics in the evals (PLAN §18, D15): support rate, judge recall, agreement, the
cache across scorings, and the gate staying judge-free. A fake judge, no models."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cohr import evals, schemas
from cohr.config import Config
from cohr.evals import (
    Calibration,
    CalibrationPair,
    CalibrationResult,
    FactResult,
    ItemReport,
    Report,
)
from cohr.judge import CachedJudge, Label, Supported
from cohr.models import ModelManager
from cohr.schemas import Citation, Claim, Segment, SegmentsFile, SlidesFile
from tests.conftest import make_run
from tests.test_evals import item, notes, report, slide

ROOT = Path(__file__).resolve().parents[1]
SEG0 = "Kenji will fix the login bug by Monday. Revenue is $248,000."
SEG1 = "We agreed to hire a contract designer."


class FakeJudge:
    """Answers by rule: a support prompt is "yes" when the claim's last word is in the cited
    text, "no" otherwise, and an unparseable reply for claims containing "garble"; coverage
    is true when the fact's first word is in a claim."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "Fact:\n<<<" in prompt:
            fact = _between(prompt, "Fact:")
            claims = _between(prompt, "Claims:")
            covered = fact.split()[0].casefold() in claims.casefold()
            return json.dumps({"reason": "rule", "covered": covered})
        claim = _between(prompt, "Claim:")
        if "garble" in claim:
            return "I think it's fine"
        cited = _between(prompt, "Cited text:")
        word = re.findall(r"\w+", claim)[-1].casefold()
        label = "yes" if word in cited.casefold() else "no"
        return json.dumps({"reason": "rule", "supported": label})


def seg(t: float, seg_id: int = 0) -> list[Citation]:
    return [Citation(kind="seg", id=seg_id, t=t)]


def _between(prompt: str, header: str) -> str:
    return prompt.split(f"{header}\n<<<\n", 1)[1].split("\n>>>", 1)[0]


def judge_run(tmp_path: Path) -> Path:
    run_dir = make_run(tmp_path)
    schemas.save(
        run_dir,
        SegmentsFile(
            segments=[
                Segment(seg_id=0, t0=0, t1=60, chunk_ids=[0], transcript_text=SEG0, slide_ids=[0]),
                Segment(seg_id=1, t0=60, t1=90, chunk_ids=[1], transcript_text=SEG1, slide_ids=[]),
            ],
            unaligned_slide_ids=[],
        ),
    )
    schemas.save(run_dir, SlidesFile(slides=[slide(0, "Revenue", ["| MRR | $248,000 |"], 0.0)]))
    schemas.save(
        run_dir,
        notes(
            key_points=[
                Claim(text="MRR is 248,000", citations=[*seg(1.0), *seg(2.0)]),  # deduped
                Claim(text="Signups are 4,450", citations=seg(1.0)),  # "450" not cited: no
                Claim(text="Something garble", citations=seg(1.0)),  # unjudged
                Claim(text="An uncited claim", citations=[]),  # not judged
            ],
            decisions=[Claim(text="Hire a contract designer", citations=seg(70.0, 1))],
            action_items=[item("Fix the login bug", "Kenji", "Monday", citations=seg(1.0))],
        ),
    )
    return run_dir


def item_report(facts: list[FactResult] | None = None) -> ItemReport:
    return ItemReport(
        id="t",
        run_id="r",
        seed=1,
        duration_s=90.0,
        metrics={"claims": 6},
        facts=facts or [],
        made_up=[],
        wrong_fields=[],
        uncited=[],
        gap_texts=[],
        stages={},
    )


def manager(fake: FakeJudge, loads: list[str] | None = None) -> ModelManager:
    def load(path: Path) -> FakeJudge:
        if loads is not None:
            loads.append(str(path))
        return fake

    return ModelManager(Config(), loaders={"judge": load}, resolve=lambda spec: Path("/x"))


def test_claim_text_and_cited_text() -> None:
    action = item("Fix the login bug", "Kenji", "Monday", citations=[])
    assert evals.claim_text(action) == "Fix the login bug (owner: Kenji; due: Monday)"
    assert evals.claim_text(action, "action_items").startswith("Action item: Fix")
    assert evals.claim_text(Claim(text="Why?", citations=[]), "open_questions") == (
        "Open question: Why?"
    )
    segment = Segment(seg_id=0, t0=0, t1=65, chunk_ids=[], transcript_text="Hello.", slide_ids=[0])
    s = slide(0, "Revenue", ["| MRR | 1 |"])
    claim = Claim(
        text="x",
        citations=[*seg(1.0), *seg(2.0), *seg(1.0, 9)] + [Citation(kind="slide", id=0)],
    )
    text = evals.cited_text(claim, {0: segment}, {0: s})
    assert text == "[transcript 0:00–1:05]\nHello.\n\n[slide 1]\nRevenue\n| MRR | 1 |"


def test_judge_item_support_rate_and_judge_recall(tmp_path: Path) -> None:
    run_dir = judge_run(tmp_path)
    fake = FakeJudge()
    cj = CachedJudge("rev", lambda: fake.generate)
    cj.open(tmp_path / "cache.json")
    facts = [
        FactResult(id="d1", kind="decision", text="hire a designer", found=False, judge_only=True),
        FactResult(id="d2", kind="decision", text="freeze features", found=False, judge_only=True),
        FactResult(id="n1", kind="number", text="MRR 248,000", found=True),
    ]
    rep = evals.judge_item(item_report(facts), run_dir, cj)

    by_claim = {s.claim: s for s in rep.support}
    assert set(by_claim) == {
        "Key point: MRR is 248,000",
        "Key point: Signups are 4,450",
        "Key point: Something garble",
        "Decision: Hire a contract designer",
        "Action item: Fix the login bug (owner: Kenji; due: Monday)",
    }
    assert by_claim["Key point: MRR is 248,000"].supported == "yes"
    assert by_claim["Key point: MRR is 248,000"].citations == ["seg 0", "seg 0"]
    assert by_claim["Key point: Signups are 4,450"].supported == "no"
    assert by_claim["Key point: Something garble"].supported == "unjudged"
    assert by_claim["Decision: Hire a contract designer"].supported == "yes"  # seg 1's text
    assert by_claim["Action item: Fix the login bug (owner: Kenji; due: Monday)"].supported == (
        "yes"
    )
    m = rep.metrics
    assert m["claims"] == 6  # scoring metrics kept
    assert (m["cited_claims"], m["support_rate"]) == (5, 0.6)  # unjudged stays in the denominator
    assert (m["support_partial"], m["support_no"], m["support_unjudged"]) == (0, 1, 1)
    assert [f.judge_covered for f in rep.facts] == [True, False, None]
    assert (m["judge_covered"], m["judge_recall"]) == (1, 0.5)
    assert not any(f.found for f in rep.facts if f.judge_only)  # exact recall untouched

    # The seg 0 cited text is the transcript only, never the segment's attached slides.
    support_prompts = [p for p in fake.prompts if "Claim:\n<<<\nKey point: MRR" in p]
    assert support_prompts and "[slide 1]" not in support_prompts[0]


def test_judge_items_rescoring_loads_no_model(tmp_path: Path) -> None:
    run_dir = judge_run(tmp_path)
    loads: list[str] = []
    fake = FakeJudge()
    logs: list[str] = []
    first, cost = evals.judge_items(
        [(item_report(), run_dir)], root=tmp_path, models=manager(fake, loads), log=logs.append
    )
    assert loads == ["/x"] and cost.loaded and cost.calls == 5 and cost.cached == 0
    assert cost.s_per_audio_hour is not None
    assert (tmp_path / evals.JUDGE_CACHE_DIR / "t.json").is_file()
    assert any("releasing the pipeline models" in m for m in logs)

    calls = len(fake.prompts)
    again, cost2 = evals.judge_items(
        [(item_report(), run_dir)], root=tmp_path, models=manager(fake, loads), log=logs.append
    )
    assert loads == ["/x"] and len(fake.prompts) == calls  # nothing stale: no load, no call
    assert (cost2.loaded, cost2.calls, cost2.cached, cost2.peak_mem_gb) == (False, 0, 5, None)
    assert again == first


def test_judge_releases_pipeline_models_before_loading(tmp_path: Path) -> None:
    run_dir = judge_run(tmp_path)
    fake = FakeJudge()
    models = ModelManager(
        Config(),
        loaders={"judge": lambda p: fake, "reduce": lambda p: object()},
        resolve=lambda spec: Path("/x"),
    )
    with models.use("reduce"):
        pass
    assert models.loaded() == ["reduce"]  # resident policy keeps it
    seen: list[list[str]] = []
    original = fake.generate

    def spy(prompt: str) -> str:
        seen.append(models.loaded())
        return original(prompt)

    setattr(fake, "generate", spy)  # noqa: B010
    evals.judge_items([(item_report(), run_dir)], root=tmp_path, models=models, log=lambda m: None)
    assert seen and all(loaded == ["judge"] for loaded in seen)
    assert models.loaded() == []  # the judge is released afterwards too


def test_summary_support_metrics_and_gate_ignores_them(tmp_path: Path) -> None:
    fake = FakeJudge()
    cj = CachedJudge("rev", lambda: fake.generate)
    cj.open(tmp_path / "c.json")
    judged = evals.judge_item(item_report(), judge_run(tmp_path), cj)
    s = evals.summarize([judged, item_report()])
    assert s["support_rate"] == 0.6 and s["support_no"] == 1 and s["support_unjudged"] == 1
    assert s["judge_recall"] is None  # no judge_only facts
    unjudged = evals.summarize([item_report()])
    assert unjudged["support_rate"] is None and unjudged["support_no"] is None

    base = report(fact_recall=0.8, support_rate=1.0, judge_recall=1.0)
    worse = report(fact_recall=0.8, support_rate=0.1, judge_recall=0.0)
    assert evals.compare(worse, base) == []


def test_report_renders_judge_rows(tmp_path: Path) -> None:
    fake = FakeJudge()
    cj = CachedJudge("rev", lambda: fake.generate)
    cj.open(tmp_path / "c.json")
    rep = evals.judge_item(item_report(), judge_run(tmp_path), cj)
    full = Report(
        created_at=datetime(2026, 10, 11, tzinfo=UTC),
        set="quick",
        git_sha=None,
        models={},
        prompts={},
        items=[rep],
        summary=evals.summarize([rep]),
        stages={},
        judge=evals.JudgeCost(
            loaded=True, calls=5, cached=0, seconds=3.0, s_per_audio_hour=120.0, peak_mem_gb=2.5
        ),
    )
    md = evals.render_markdown(full)
    assert "| support_rate | 60.0% |" in md
    assert "| judge (eval only) | 120 | 2.5 |" in md and "Judge: 5 calls, 0 cached" in md
    assert "- support no (seg 0): Key point: Signups are 4,450 · rule" in md
    assert "- support unjudged (seg 0): Key point: Something garble" in md
    assert evals.load_report(evals.write_report(full, tmp_path)) == full


def result(label: Label, judged: Supported) -> CalibrationResult:
    return CalibrationResult(id="c", label=label, judge=judged, reason="")


def test_agreement_three_way_and_binary() -> None:
    results = [
        result("yes", "yes"),
        result("partial", "no"),  # 3-way miss, binary agree (both not-yes)
        result("no", "yes"),  # both miss
        result("no", "unjudged"),  # unjudged disagrees with everything
    ]
    assert evals.agreement(results) == (0.25, 0.5)
    assert evals.agreement([]) == (0.0, 0.0)
    matrix = evals.confusion(results)
    assert matrix["no"] == {"yes": 1, "partial": 0, "no": 0, "unjudged": 1}
    assert matrix["partial"]["no"] == 1 and sum(matrix["yes"].values()) == 1


def write_calibration(root: Path, pairs: list[dict[str, Any]]) -> None:
    cal = Calibration(
        about="test",
        sources={"m/seg0": "[transcript 0:00–1:00]\nKenji fixes the bug. Revenue is $248,000."},
        pairs=[CalibrationPair.model_validate(p) for p in pairs],
    )
    path = root / evals.CALIBRATION
    path.parent.mkdir(parents=True)
    path.write_text(cal.model_dump_json())


def test_calibrate_reports_agreement_and_caches(tmp_path: Path) -> None:
    pair: Callable[[str, str, str], dict[str, Any]] = lambda i, claim, label: {  # noqa: E731
        "id": i,
        "source": "test",
        "claim": claim,
        "cited": ["m/seg0"],
        "label": label,
        "reason": "",
    }
    write_calibration(
        tmp_path,
        [
            pair("c1", "Key point: Revenue is $248,000", "yes"),
            pair("c2", "Action item: Victor fixes the bug", "no"),  # fake says yes ("bug")
            pair("c3", "Key point: Signups 4,450", "no"),
            pair("c4", "Key point: garble", "partial"),
            pair("c5", "Key point: the bug", "yes"),
        ],
    )
    fake = FakeJudge()
    loads: list[str] = []
    rep = evals.calibrate(root=tmp_path, models=manager(fake, loads), log=lambda m: None)
    assert [r.judge for r in rep.results] == ["yes", "yes", "no", "unjudged", "yes"]
    assert (rep.agreement, rep.agreement_binary, rep.passed) == (0.6, 0.6, False)
    assert rep.cost.calls == 5 and loads == ["/x"]
    md = evals.render_calibration(rep)
    assert "3-way agreement 60.0% (**FAIL** at 80%)" in md
    assert "- c2: label no, judge yes · rule" in md
    assert "| no | 1 | 0 | 1 | 0 |" in md
    path = evals.write_calibration_report(rep, tmp_path)
    assert path.name.endswith("-judge-calibration.json") and path.with_suffix(".md").is_file()

    again = evals.calibrate(root=tmp_path, models=manager(fake, loads), log=lambda m: None)
    assert loads == ["/x"] and again.cost.cached == 5 and again.results == rep.results


def test_committed_calibration_set_is_consistent() -> None:
    cal = evals.load_calibration(ROOT)
    labels = [p.label for p in cal.pairs]
    assert len(cal.pairs) == 30
    assert (labels.count("yes"), labels.count("partial"), labels.count("no")) == (12, 8, 10)
    assert len({p.id for p in cal.pairs}) == 30
    for p in cal.pairs:
        assert p.cited and all(k in cal.sources for k in p.cited), p.id
        assert p.reason and p.claim.split(":", 1)[0] in evals.SECTION_LABEL.values(), p.id
    assert set(cal.sources) == {k for p in cal.pairs for k in p.cited}  # no unused sources


@pytest.mark.parametrize("flag", ["--no-judge", "--calibrate-judge"])
def test_cli_judge_flags_parse(flag: str) -> None:
    from cohr.cli import build_parser

    args = build_parser().parse_args(["eval", flag])
    assert args.no_judge == (flag == "--no-judge")
    assert args.calibrate_judge == (flag == "--calibrate-judge")
