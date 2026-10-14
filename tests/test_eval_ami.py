"""AMI in the full set (PLAN §18, 6e-1): summaries, downloads, coverage and support scoring.
A fake judge, no network, no models."""

import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cohr import ami, evals
from cohr.evals import AMI_HOME, SentenceResult
from cohr.judge import CachedJudge
from tests.test_eval_judge import FakeJudge, judge_run, manager

ROOT = Path(__file__).resolve().parents[1]

SUMMARY = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
<abstract nite:id="X.abstract.1">
<sentence nite:id="X.s.1">Kenji presented the login   bug.</sentence>
<sentence nite:id="X.s.2">The caf\xe9 budget was discussed.</sentence>
</abstract>
<actions nite:id="X.actions.1">
<sentence nite:id="X.s.3">Kenji will fix the login bug.</sentence>
</actions>
<decisions nite:id="X.decisions.1">
<sentence nite:id="X.s.4">Hire a contract designer.</sentence>
<sentence nite:id="X.s.5">Revenue target stays.</sentence>
</decisions>
<problems nite:id="X.problems.1">
</problems>
</nite:root>
""".encode("iso-8859-1")

REFS = {
    "abstract": ["MRR was reviewed.", "Weather was mentioned."],
    "decisions": ["Hire a contract designer."],
    "actions": ["Fix the login bug."],
    "problems": ["Nobody knows the launch date."],
}


def test_full_set_has_the_five_test_meetings() -> None:
    full = evals.load_set("full", ROOT)
    assert full.ami is not None
    assert full.ami.meetings == ["ES2004b", "ES2004d", "IS1009b", "IS1009c", "TS3003c"]


def test_every_part_has_a_home_section() -> None:
    assert set(AMI_HOME) == set(ami.PARTS)
    assert AMI_HOME["abstract"] == ("summary", "key_points")
    assert AMI_HOME["decisions"] == ("decisions",)


def test_parse_summary_reads_latin1_and_every_part() -> None:
    out = ami.parse_summary(SUMMARY)
    assert out == {
        "abstract": ["Kenji presented the login bug.", "The café budget was discussed."],
        "decisions": ["Hire a contract designer.", "Revenue target stays."],
        "actions": ["Kenji will fix the login bug."],
        "problems": [],
    }


def test_reference_reads_the_meeting_from_the_zip(tmp_path: Path) -> None:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("abstractive/X.abssumm.xml", SUMMARY)
    assert ami.reference("X", path)["actions"] == ["Kenji will fix the login bug."]
    with pytest.raises(KeyError):
        ami.reference("Y", path)


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, length: int):
        super().__init__(data)
        self.headers = {"Content-Length": str(length)}


def test_download_is_atomic_and_checks_the_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urls: list[str] = []

    def urlopen(url: str, timeout: float) -> Any:
        urls.append(url)
        return FakeResponse(b"abc", 3 if url.endswith("ok") else 9)

    monkeypatch.setattr(ami.urllib.request, "urlopen", urlopen)
    dest = tmp_path / "d" / "f.wav"
    with pytest.raises(OSError, match="got 3 of 9 bytes"):
        ami.download("u/short", dest, log=lambda _: None)
    assert not dest.exists() and list(dest.parent.iterdir()) == []  # no partial file

    assert ami.download("u/ok", dest, log=lambda _: None) == dest
    assert dest.read_bytes() == b"abc"
    ami.download("u/ok", dest, log=lambda _: None)  # kept: not fetched again
    assert urls == ["u/short", "u/ok"]


def test_ensure_reads_the_summary_before_fetching_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_path = tmp_path / ami.ANNOTATIONS
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("abstractive/X.abssumm.xml", SUMMARY)
    fetched: list[str] = []

    def download(url: str, dest: Path, log: Any = None) -> Path:
        fetched.append(url)
        return dest

    monkeypatch.setattr(ami, "download", download)
    with pytest.raises(KeyError):  # no summary for Y: its audio is never fetched
        ami.ensure("Y", tmp_path, log=lambda _: None)
    assert fetched == [ami.ANNOTATIONS_URL]
    wav, refs = ami.ensure("X", tmp_path, log=lambda _: None)
    assert wav.name == "X.Mix-Headset.wav" and refs["decisions"][0] == "Hire a contract designer."
    assert fetched[-1] == ami.AUDIO_URL.format(id="X")


def test_ami_meetings_follow_the_item_filter() -> None:
    spec = ami.AmiSpec(meetings=["A1", "B2"])
    assert evals.ami_meetings(spec, []) == ["A1", "B2"]
    assert evals.ami_meetings(spec, ["ami"]) == ["A1", "B2"]
    assert evals.ami_meetings(spec, ["ami-B2", "fleurs"]) == ["B2"]
    assert evals.ami_meetings(spec, ["clean-en"]) == []
    assert evals.ami_meetings(None, []) == []


def test_coverage_counts_unjudged_as_missed() -> None:
    s = [
        SentenceResult(part="decisions", text="a", covered=True),
        SentenceResult(part="decisions", text="b", covered=None),
        SentenceResult(part="actions", text="c"),
    ]
    assert evals.ami_coverage(s, "decisions") == 0.5
    assert evals.ami_coverage(s, "actions") is None  # not judged at all
    assert evals.ami_coverage(s, "problems") is None  # no sentences
    assert evals.ami_coverage(s) == 0.3333


def scored(run_dir: Path) -> evals.AmiReport:
    return evals.score_ami("X", REFS, run_dir, 90.0, datetime(1970, 1, 1, tzinfo=UTC))


def test_score_and_judge_a_meeting_per_part(tmp_path: Path) -> None:
    run_dir = judge_run(tmp_path)
    rep = scored(run_dir)
    assert rep.id == "ami-X"
    assert (rep.metrics["claims"], rep.metrics["uncited_claims"]) == (6, 1)
    assert rep.uncited == ["An uncited claim"]
    assert [s.covered for s in rep.sentences] == [None] * 5

    fake = FakeJudge()
    cj = CachedJudge("rev", lambda: fake.generate)
    cj.open(tmp_path / "c.json")
    judged = evals.judge_ami(rep, run_dir, cj)
    covered = {s.text: s.covered for s in judged.sentences}
    # FakeJudge: covered when the sentence's first word is in a claim of its home sections
    assert covered["MRR was reviewed."] is True  # abstract found in Key points
    assert covered["Weather was mentioned."] is False
    assert covered["Hire a contract designer."] is True
    assert covered["Fix the login bug."] is True
    assert covered["Nobody knows the launch date."] is False  # no open questions: no claims
    m = judged.metrics
    assert (m["coverage_abstract"], m["coverage_decisions"], m["coverage_problems"]) == (0.5, 1, 0)
    assert m["coverage"] == 0.6
    assert (m["cited_claims"], m["support_rate"]) == (5, 0.6)

    # a decision in Key points doesn't count: sentences are looked for in their home only
    decision_prompts = [p for p in fake.prompts if "Fact:\n<<<\nHire" in p]
    assert decision_prompts and "MRR" not in decision_prompts[0]


def test_judge_items_judges_meetings_in_the_same_session(tmp_path: Path) -> None:
    run_dir = judge_run(tmp_path)
    loads: list[str] = []
    items, cost, meetings = evals.judge_items(
        [],
        root=tmp_path,
        models=manager(FakeJudge(), loads),
        log=lambda _: None,
        meetings=[(scored(run_dir), run_dir)],
    )
    assert items == [] and loads == ["/x"] and len(meetings) == 1
    assert meetings[0].metrics["coverage"] == 0.6
    assert (tmp_path / evals.JUDGE_CACHE_DIR / "ami-X.json").is_file()
    assert cost.calls == 5 + 5  # five supports, five coverage questions


def test_report_shows_meetings_and_keeps_them_out_of_the_gate(tmp_path: Path) -> None:
    run_dir = judge_run(tmp_path)
    cj = CachedJudge("rev", lambda: FakeJudge().generate)
    cj.open(tmp_path / "c.json")
    meeting = evals.judge_ami(scored(run_dir), run_dir, cj)
    base = evals.load_report(ROOT / "evals" / "baseline.json")
    report = base.model_copy(
        update={
            "summary": base.summary | evals.ami_summary([meeting]),
            "ami": [meeting],
            "ami_stages": evals.stage_summary([meeting]),
        }
    )
    assert report.summary["ami_coverage"] == 0.6
    assert report.summary["ami_coverage_problems"] == 0
    assert report.summary["ami_uncited_claims"] == 1
    assert report.summary["uncited_claims"] == base.summary["uncited_claims"]  # not added
    assert evals.compare(report, base) == []
    md = evals.render_markdown(report)
    assert "## AMI (real meetings, report-only)" in md
    assert "| ami-X | 1.5 | 60.0% |" in md
    assert "- missed problems: Nobody knows the launch date." in md
    assert "- uncited: An uncited claim" in md
    assert evals.Report.model_validate_json(report.model_dump_json()) == report
