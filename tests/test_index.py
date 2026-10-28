"""Cross-meeting search index (PLAN §13): built from run files, synced on read, disposable."""

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from cohr import index, schemas, storage
from cohr.config import Config
from cohr.schemas import (
    ActionItem,
    Citation,
    Claim,
    Manifest,
    NotesDoc,
    NotesSections,
    Segment,
    SegmentsFile,
    Slide,
    SlidesFile,
    StageState,
    Transcript,
    TranscriptItem,
    TranslatedNotes,
)
from cohr.storage import Source

CHUNK_S = 30.0


def sections(
    summary: list[Claim] | None = None, actions: list[ActionItem] | None = None
) -> NotesSections:
    return NotesSections(
        summary=summary or [],
        key_points=[],
        decisions=[],
        action_items=actions or [],
        open_questions=[],
    )


def make_run(
    runs: Path,
    name: str,
    lines: list[str],
    *,
    lang: str = "en",
    notes: NotesSections | None = None,
    slides: Sequence[str] = (),
    created: datetime = datetime(2026, 10, 20, 9, 0, tzinfo=UTC),
) -> Path:
    """A run with one transcript chunk per line, two chunks per segment, and notes if given."""
    src = runs.parent / "media" / f"{name}.wav"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"x")
    config = Config(source_language=lang, target_language=lang)
    run_dir = storage.create_run(runs, [Source(src, "audio")], config, name=name, now=created)
    items = [
        TranscriptItem(chunk_id=k, t0=CHUNK_S * k, t1=CHUNK_S * (k + 1), text=line)
        for k, line in enumerate(lines)
    ]
    schemas.save(run_dir, Transcript(language=lang, runtime="mlx-speech", items=items))
    segments = [
        Segment(
            seg_id=s,
            t0=2 * CHUNK_S * s,
            t1=2 * CHUNK_S * (s + 1),
            chunk_ids=[k for k in (2 * s, 2 * s + 1) if k < len(lines)],
            transcript_text=" ".join(lines[2 * s : 2 * s + 2]),
            slide_ids=[0] if s == 0 and slides else [],
        )
        for s in range((len(lines) + 1) // 2)
    ]
    schemas.save(run_dir, SegmentsFile(segments=segments, unaligned_slide_ids=[]))
    schemas.save(
        run_dir,
        SlidesFile(
            slides=[
                Slide(
                    slide_id=k,
                    frame_id=k,
                    t=5.0 if k == 0 else None,
                    t_source="aligned" if k == 0 else "none",
                    title=title,
                    text_md="",
                    tables_md=[],
                    chart_notes="",
                    visual_notes="a blue bar chart",
                )
                for k, title in enumerate(slides)
            ]
        ),
    )
    if notes is not None:
        schemas.save(run_dir, NotesDoc(sections=notes, passes=1, language=lang, raw=""))
        manifest = schemas.load(run_dir, Manifest)
        manifest.stages["render"] = StageState(status="done")
        schemas.save(run_dir, manifest)
    return run_dir


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def dump(runs: Path) -> dict[str, list[tuple]]:
    """The index's rows without the autoincrement ids, and what FTS finds for a probe term."""
    with closing(sqlite3.connect(runs / index.FILENAME)) as con:
        out = {
            "runs": sorted(con.execute("SELECT * FROM runs")),
            "items": sorted(
                con.execute(
                    "SELECT run_id, kind, lang, section, seg_id, slide_id, t, t1, citations, text"
                    " FROM items"
                ),
                key=repr,
            ),
            "actions": sorted(con.execute("SELECT * FROM action_items"), key=repr),
        }
        for table in ("fts_words", "fts_tri"):
            out[table] = sorted(
                con.execute(
                    f"SELECT items.run_id, items.text FROM {table} JOIN items"
                    f" ON items.id = {table}.rowid WHERE {table} MATCH 'budget'"
                )
            )
    return out


def find(
    runs: Path,
    query: str,
    *,
    kinds: Sequence[str] | None = None,
    langs: Sequence[str] | None = None,
    since: date | None = None,
    until: date | None = None,
) -> index.Results:
    with index.open_index(runs) as con:
        return index.search(con, query, kinds=kinds, langs=langs, since=since, until=until)


BUDGET = Claim(text="The budget is $14,500.", citations=[Citation(kind="seg", id=1, t=75.0)])
LOGO = Claim(text="Use the new logo.", citations=[Citation(kind="slide", id=0)])
TASK = ActionItem(
    text="Send the budget sheet",
    citations=[Citation(kind="seg", id=0, t=12.0)],
    owner="Priya",
    due="Friday",
)


def test_rebuild_equals_incremental(runs: Path) -> None:
    a = make_run(runs, "alpha", ["hello everyone", "the budget is set"], notes=sections([BUDGET]))
    index.sync(runs)
    b = make_run(runs, "beta", ["we talk about the logo", "and the budget"], slides=["Budget"])
    index.sync(runs)
    schemas.save(a, NotesDoc(sections=sections([BUDGET, LOGO]), passes=1, language="en", raw=""))
    storage.prune_media(b, runs)
    c = make_run(runs, "gamma", ["budget again"], notes=sections(actions=[TASK]))
    index.sync(runs)
    storage.delete_run(c, runs)
    result = index.sync(runs)
    assert result.removed == [c.name] and not result.rebuilt
    incremental = dump(runs)
    assert index.rebuild(runs).rebuilt
    assert dump(runs) == incremental
    assert {r[0] for r in incremental["runs"]} == {a.name, b.name}
    assert len(incremental["fts_words"]) == len(incremental["fts_tri"]) == 4


def test_unchanged_run_is_not_reindexed(runs: Path) -> None:
    run = make_run(runs, "alpha", ["the budget is set"], notes=sections([BUDGET]))
    assert index.sync(runs).indexed == [run.name]
    assert index.sync(runs).indexed == []
    # Files the index doesn't read don't count: the log, audio, frames.
    (run / "log.txt").write_text("stage reduce: done\n")
    (run / "audio.wav").write_bytes(b"RIFF")
    assert index.sync(runs).indexed == []
    schemas.save(run, NotesDoc(sections=sections([LOGO]), passes=1, language="en", raw=""))
    assert index.sync(runs).indexed == [run.name]
    assert find(runs, "logo").hits and not find(runs, "budget", kinds=["note"]).hits


def test_delete_and_prune_update_rows(runs: Path) -> None:
    a = make_run(runs, "alpha", ["the budget is set"], notes=sections([BUDGET]))
    b = make_run(runs, "beta", ["budget talk"], notes=sections([BUDGET]))
    assert {h.run_id for h in find(runs, "budget").hits} == {a.name, b.name}
    storage.prune_media(a, runs)
    storage.delete_run(b, runs)
    assert {h.run_id for h in find(runs, "budget").hits} == {a.name}
    with closing(sqlite3.connect(runs / index.FILENAME)) as con:
        assert con.execute("SELECT run_id, pruned FROM runs").fetchall() == [(a.name, 1)]
        assert (
            con.execute("SELECT count(*) FROM items WHERE run_id = ?", (b.name,)).fetchone()[0] == 0
        )


@pytest.mark.parametrize("damage", ["missing", "garbage", "old_schema", "no_fts"])
def test_unusable_file_is_rebuilt(runs: Path, damage: str) -> None:
    run = make_run(runs, "alpha", ["the budget is set"])
    index.sync(runs)
    db = runs / index.FILENAME
    if damage == "missing":
        db.unlink()
    elif damage == "garbage":
        db.write_bytes(b"not a database at all" * 100)
    else:
        with closing(sqlite3.connect(db)) as con:
            if damage == "old_schema":
                con.execute("UPDATE meta SET value = '0'")
            else:
                con.execute("DROP TABLE fts_tri")
            con.commit()
    result = index.sync(runs)
    assert result.rebuilt and result.indexed == [run.name]
    assert [h.run_id for h in find(runs, "budget").hits] == [run.name]
    assert not index.sync(runs).rebuilt  # and the rebuilt file is kept


def test_partial_run_is_indexed_by_what_it_has(runs: Path) -> None:
    run = make_run(runs, "alpha", ["hello", "the budget is set"])  # no notes yet
    hits = find(runs, "budget").hits
    assert [(h.kind, h.seg_id, h.t) for h in hits] == [("transcript", 0, CHUNK_S)]
    with closing(sqlite3.connect(runs / index.FILENAME)) as con:
        assert con.execute("SELECT status, notes_lang FROM runs").fetchone() == ("partial", None)
    # A broken file is skipped; the rest of the run is still indexed.
    (run / SlidesFile.FILENAME).write_text("{")
    assert index.sync(runs).indexed == [run.name]
    assert find(runs, "budget").hits


def test_unreadable_manifest_leaves_the_run_out(runs: Path) -> None:
    good = make_run(runs, "alpha", ["the budget is set"])
    bad = make_run(runs, "beta", ["budget too"])
    (bad / Manifest.FILENAME).write_text("{")
    result = index.sync(runs)
    assert result.failed == [bad.name] and good.name in result.indexed
    assert [h.run_id for h in find(runs, "budget").hits] == [good.name]


def test_note_hits_keep_citations_and_jump_time(runs: Path) -> None:
    make_run(
        runs,
        "alpha",
        ["intro", "numbers", "the logo"],
        notes=sections([BUDGET, LOGO], [TASK]),
        slides=["Brand refresh"],
    )
    [budget] = find(runs, "14,500", kinds=["note"]).hits
    assert (budget.section, budget.seg_id, budget.t) == ("summary", 1, 75.0)
    assert budget.citations == BUDGET.citations
    [logo] = find(runs, "logo", kinds=["note"]).hits
    assert (logo.seg_id, logo.slide_id, logo.t) == (None, 0, None)  # slide-only citation
    # An action item is found by its owner and listed in the cross-meeting table.
    [task] = find(runs, "Priya").hits
    assert task.section == "action_items" and task.t == 12.0
    with index.open_index(runs) as con:
        [row] = index.action_items(con)
        assert (row.owner, row.due, row.text, row.t) == ("Priya", "Friday", TASK.text, 12.0)
        assert index.action_items(con, owners=["Tom"]) == []
        assert index.owners(con) == ["Priya"]


def test_slides_index_text_not_visual_notes(runs: Path) -> None:
    make_run(runs, "alpha", ["hello"], slides=["Quarterly budget"])
    [hit] = find(runs, "quarterly").hits
    assert (hit.kind, hit.slide_id, hit.seg_id, hit.t) == ("slide", 0, 0, 5.0)
    assert not find(runs, "blue").hits  # Vision's visual description isn't searched


def test_every_term_first_then_any(runs: Path) -> None:
    make_run(runs, "alpha", ["the budget is set", "the logo is new", "budget and logo"])
    # One chunk has both words: only it, and the fallback must not fire.
    both = find(runs, "budget logo")
    assert not both.any_term and [h.t for h in both.hits] == [2 * CHUNK_S]
    # No chunk has both: any word, flagged.
    either = find(runs, "budget hiring")
    assert either.any_term and len(either.hits) == 2
    assert find(runs, "hiring").hits == [] and not find(runs, "hiring").any_term
    assert find(runs, "the is").hits == []  # stopwords only: no terms


def test_cjk_uses_trigram_windows(runs: Path) -> None:
    make_run(
        runs,
        "jp",
        ["サポートの未処理チケットは390件です。", "来四半期の予算は14,500ドルに決まりました。"],
        lang="ja",
    )
    [hit] = find(runs, "予算は").hits
    assert hit.t == CHUNK_S and index.MARK_START in hit.snippet
    assert find(runs, "来四半期の予算").hits
    assert find(runs, "予算").hits == []  # 2 characters: trigram can't match (D17)
    # Latin digits inside a Japanese run: unicode61 reads the run as one token, so a Latin
    # query searches Japanese rows by trigram.
    [hit] = find(runs, "390").hits
    assert hit.t == 0.0


def test_substrings_match_only_in_cjk_rows(runs: Path) -> None:
    make_run(runs, "en", ["the budget is set"])
    make_run(runs, "jp", ["Budget会議"], lang="ja")
    # "udget" is a substring of both rows, but only the Japanese one is searched by trigram.
    assert [h.lang for h in find(runs, "udget").hits] == ["ja"]
    assert {h.lang for h in find(runs, "budget").hits} == {"en", "ja"}


def test_translated_notes_are_what_is_searched(runs: Path) -> None:
    run = make_run(runs, "sw", ["the budget is set"], notes=sections([BUDGET]))
    translated = TranslatedNotes(
        sections=sections(
            [Claim(text="Bajeti ni $14,500.", citations=BUDGET.citations)],
        ),
        language="sw",
        source_language="en",
        untranslated=0,
        raw=[],
    )
    schemas.save(run, translated)
    assert [h.lang for h in find(runs, "bajeti").hits] == ["sw"]
    assert find(runs, "budget", kinds=["note"]).hits == []


def test_digits_match_across_scripts(runs: Path) -> None:
    notes = sections([Claim(text="সাইটের গতি: ৯৫০ মিলিসেকেন্ড", citations=BUDGET.citations)])
    make_run(runs, "bn", ["page load is 950 ms"], lang="en", notes=notes)
    assert {h.kind for h in find(runs, "950").hits} == {"transcript", "note"}
    assert {h.kind for h in find(runs, "৯৫০").hits} == {"transcript", "note"}
    [note] = find(runs, "950", kinds=["note"]).hits
    assert f"{index.MARK_START}950{index.MARK_END}" in note.snippet


def test_filters(runs: Path) -> None:
    make_run(
        runs,
        "old",
        ["budget in october"],
        notes=sections([BUDGET]),
        created=datetime(2026, 10, 1, tzinfo=UTC),
    )
    make_run(
        runs, "de", ["das Budget steht"], lang="de", created=datetime(2026, 10, 25, tzinfo=UTC)
    )
    assert len(find(runs, "budget").hits) == 3
    assert [h.kind for h in find(runs, "budget", kinds=["note"]).hits] == ["note"]
    assert [h.lang for h in find(runs, "budget", langs=["de"]).hits] == ["de"]
    assert [h.lang for h in find(runs, "budget", since=date(2026, 10, 2)).hits] == ["de"]
    assert len(find(runs, "budget", until=date(2026, 10, 1)).hits) == 2
    with index.open_index(runs) as con:
        assert index.languages(con) == ["de", "en"]
        assert index.run_count(con) == 2


def test_transcript_hit_jumps_to_the_matched_sentence(runs: Path) -> None:
    line = "Hello everyone and welcome back. We agreed to use the new logo on the website."
    make_run(runs, "alpha", ["intro", line])
    [hit] = find(runs, "new logo").hits
    offset = line.index("We agreed") / len(line)
    assert hit.t == round(CHUNK_S + offset * CHUNK_S, 2)
    # A match in the first sentence keeps the chunk start.
    assert [h.t for h in find(runs, "welcome").hits] == [CHUNK_S]


def test_snippet_marks_the_match(runs: Path) -> None:
    make_run(runs, "alpha", ["so the [draft] budget is $14,500 for now"])
    [hit] = find(runs, "budget").hits
    assert f"{index.MARK_START}budget{index.MARK_END}" in hit.snippet
    assert "[draft]" in hit.snippet  # raw text: the page escapes it


def test_index_lives_beside_the_runs_and_is_not_a_run(runs: Path) -> None:
    make_run(runs, "alpha", ["hello"])
    index.sync(runs)
    assert (runs / index.FILENAME).is_file()
    assert [p.name for p in storage.list_runs(runs)] == [
        p.name for p in runs.iterdir() if storage.is_run(p)
    ]
    assert not list(runs.glob(f".{index.FILENAME}.*"))  # no temp file left behind
