"""Cross-meeting search index (PLAN §13, Phase 7): SQLite + FTS5 at `<runs_dir>/index.db`.

Disposable and rebuilt from the run files at any time: nothing is written here that isn't on
disk first, and run files are read only through schemas.py. Deleting index.db is always safe.

What's indexed, per run: transcript items (one row per chunk, ≤30 s, so a hit jumps near its
words), slides (title, text, tables, chart notes) and the notes people read
(`render.shown_notes`: claims with their citations); action items also go in their own table
for the cross-meeting list. A run without notes yet is indexed by the files it has.

Sync on read (§13, Phase 7 step 0): `open_index` lists the runs in `runs_dir` and re-indexes
each run whose key changed (mtime and size of manifest.json and every indexed file), one
transaction per run, and drops runs no longer on disk. So stage finishes, Delete, Prune media,
CLI runs and hand edits all reach the index with no hooks in the pipeline or storage code. A
missing, corrupt or older-schema file is rebuilt into a temp file and swapped in.

Two FTS5 tables over the same rows (external content, the text is stored once): `unicode61`
for space-separated languages and `trigram` for Chinese, Japanese and Korean. The query picks
one with `fts.query_terms`, as Ask's ranking does (D17, D27).
"""

import hashlib
import json
import logging
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Literal, get_args

from pydantic import ValidationError

from cohr import atomic, fts, render, schemas, storage
from cohr.fuse import tokens
from cohr.mapper import clock
from cohr.reduce import SECTIONS, sentence_offset, slide_label
from cohr.schemas import (
    ActionItem,
    Citation,
    Manifest,
    NotesDoc,
    RunFile,
    SegmentsFile,
    SlidesFile,
    Transcript,
    TranslatedNotes,
)

log = logging.getLogger("cohr")

SCHEMA_VERSION = 1  # bump on any DDL or row-content change: older files are rebuilt
FILENAME = "index.db"
Kind = Literal["transcript", "slide", "note"]
KINDS: tuple[Kind, ...] = get_args(Kind)
# The files a run's rows come from; their stats (with the manifest's) are the run's key.
_FILES: tuple[type[RunFile], ...] = (
    Manifest,
    Transcript,
    SegmentsFile,
    SlidesFile,
    NotesDoc,
    TranslatedNotes,
)
# snippet() marks matches with these; the Search page turns them into bold after escaping
# the text, so a bracket or `$` in a meeting can't break the markdown.
MARK_START = "\x02"
MARK_END = "\x03"
SNIPPET_TOKENS = 16
TIMEOUT_S = 10.0  # another session re-indexing holds the write lock briefly

_DDL = f"""
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, key TEXT NOT NULL, created_at TEXT NOT NULL,
  source_lang TEXT, notes_lang TEXT, status TEXT NOT NULL, pruned INTEGER NOT NULL
);
CREATE TABLE items (
  id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL, lang TEXT,
  section TEXT, seg_id INTEGER, slide_id INTEGER, t REAL, t1 REAL, citations TEXT,
  text TEXT NOT NULL
);
CREATE INDEX items_run ON items(run_id);
CREATE TABLE action_items (
  run_id TEXT NOT NULL, owner TEXT, due TEXT, text TEXT NOT NULL, lang TEXT,
  seg_id INTEGER, slide_id INTEGER, t REAL, citations TEXT NOT NULL
);
CREATE INDEX action_items_run ON action_items(run_id);
CREATE VIRTUAL TABLE fts_words USING fts5(
  text, content='items', content_rowid='id', tokenize='{fts.WORDS}'
);
CREATE VIRTUAL TABLE fts_tri USING fts5(
  text, content='items', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER items_ai AFTER INSERT ON items BEGIN
  INSERT INTO fts_words(rowid, text) VALUES (new.id, new.text);
  INSERT INTO fts_tri(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER items_ad AFTER DELETE ON items BEGIN
  INSERT INTO fts_words(fts_words, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO fts_tri(fts_tri, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


# --- rows from a run (pure, given the files) ------------------------------------------------


@dataclass(frozen=True)
class Item:
    kind: Kind
    lang: str | None
    text: str
    section: str | None = None
    seg_id: int | None = None
    slide_id: int | None = None
    t: float | None = None  # where a hit jumps: chunk start, slide time, or the claim's [mm:ss]
    t1: float | None = None  # a transcript chunk's end: search places t at the matched sentence
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class Action:
    item: Item  # the task alone (the searchable note row also holds owner and due)
    owner: str | None
    due: str | None


@dataclass(frozen=True)
class RunRows:
    run_id: str
    key: str
    created_at: str
    source_lang: str | None
    notes_lang: str | None
    status: Literal["done", "partial"]
    pruned: bool
    items: list[Item]
    actions: list[Action]


def run_key(run_dir: Path) -> str:
    """Changes whenever a file the index reads changes (written, replaced or removed)."""
    parts = []
    for model in _FILES:
        path = run_dir / model.FILENAME
        try:
            s = path.stat()
            parts.append(f"{model.FILENAME}:{s.st_mtime_ns}:{s.st_size}")
        except FileNotFoundError:
            parts.append(f"{model.FILENAME}:-")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:32]


def _load[M: RunFile](run_dir: Path, model: type[M]) -> M | None:
    """A run file, or None when it's missing or unreadable (the rest of the run is indexed)."""
    if not (run_dir / model.FILENAME).is_file():
        return None
    try:
        return schemas.load(run_dir, model)
    except (ValidationError, OSError, ValueError) as e:
        log.warning("index: skipping %s/%s: %s", run_dir.name, model.FILENAME, e)
        return None


def _jump(
    citations: Sequence[Citation], seg_t0: dict[int, float]
) -> tuple[int | None, float | None, int | None]:
    """(seg_id, t, slide_id) a claim opens at: its first seg citation (refined t, else the
    segment start), else its first slide citation."""
    for c in citations:
        if c.kind == "seg":
            return c.id, c.t if c.t is not None else seg_t0.get(c.id), None
    for c in citations:
        if c.kind == "slide":
            return None, None, c.id
    return None, None, None


def action_text(item: ActionItem) -> str:
    """An action item's searchable text: the task, then owner and due, so a search for a name
    finds that person's tasks."""
    return " · ".join(p for p in (item.text, item.owner, item.due) if p)


def run_rows(run_dir: Path) -> RunRows:
    """Everything the index holds for one run, from its files on disk."""
    key = run_key(run_dir)
    manifest = schemas.load(run_dir, Manifest)
    source_lang = manifest.config.get("source_language")
    transcript = _load(run_dir, Transcript)
    segs = _load(run_dir, SegmentsFile)
    slides = _load(run_dir, SlidesFile)
    notes = None
    if (run_dir / NotesDoc.FILENAME).is_file():
        try:
            notes = render.shown_notes(run_dir)
        except (ValidationError, OSError, ValueError) as e:
            log.warning("index: skipping the notes of %s: %s", run_dir.name, e)

    seg_of_chunk: dict[int, int] = {}
    seg_of_slide: dict[int, int] = {}
    seg_t0: dict[int, float] = {}
    if segs is not None:
        for seg in segs.segments:
            seg_t0[seg.seg_id] = seg.t0
            for chunk_id in seg.chunk_ids:
                seg_of_chunk.setdefault(chunk_id, seg.seg_id)
            for slide_id in seg.slide_ids:
                seg_of_slide.setdefault(slide_id, seg.seg_id)

    items: list[Item] = []
    if transcript is not None:
        lang = transcript.language
        for it in transcript.items:
            if it.text.strip():
                items.append(
                    Item(
                        "transcript",
                        lang,
                        it.text,
                        seg_id=seg_of_chunk.get(it.chunk_id),
                        t=it.t0,
                        t1=it.t1,
                    )
                )
    if slides is not None:
        lang = source_lang if isinstance(source_lang, str) else None
        for s in slides.slides:
            text = "\n".join(p for p in (s.title, s.text_md, *s.tables_md, s.chart_notes) if p)
            if text.strip():
                seg_id = seg_of_slide.get(s.slide_id)
                items.append(Item("slide", lang, text, seg_id=seg_id, slide_id=s.slide_id, t=s.t))
    actions: list[Action] = []
    if notes is not None:
        for section in SECTIONS:
            for claim in getattr(notes.sections, section):
                seg_id, t, slide_id = _jump(claim.citations, seg_t0)
                item = Item(
                    "note",
                    notes.language,
                    claim.text,
                    section=section,
                    seg_id=seg_id,
                    slide_id=slide_id,
                    t=t,
                    citations=tuple(claim.citations),
                )
                if isinstance(claim, ActionItem):
                    actions.append(Action(item, claim.owner, claim.due))
                    item = replace(item, text=action_text(claim))
                items.append(item)

    render_stage = manifest.stages.get("render")
    done = render_stage is not None and render_stage.status == "done" and notes is not None
    return RunRows(
        run_id=run_dir.name,
        key=key,
        created_at=manifest.created_at.isoformat(),
        source_lang=source_lang if isinstance(source_lang, str) else None,
        notes_lang=notes.language if notes is not None else None,
        status="done" if done else "partial",
        pruned=manifest.pruned,
        items=items,
        actions=actions,
    )


# --- writing ---------------------------------------------------------------------------------


def _citations_json(citations: Sequence[Citation]) -> str:
    return json.dumps([c.model_dump(mode="json") for c in citations], separators=(",", ":"))


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path, timeout=TIMEOUT_S, isolation_level=None)


def _create(con: sqlite3.Connection) -> None:
    con.executescript(_DDL)
    con.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))


def _delete_run(con: sqlite3.Connection, run_id: str) -> None:
    con.execute("DELETE FROM items WHERE run_id = ?", (run_id,))  # the trigger updates FTS
    con.execute("DELETE FROM action_items WHERE run_id = ?", (run_id,))
    con.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def _insert_run(con: sqlite3.Connection, rows: RunRows) -> None:
    con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            rows.run_id,
            rows.key,
            rows.created_at,
            rows.source_lang,
            rows.notes_lang,
            rows.status,
            int(rows.pruned),
        ),
    )
    # Digits are folded to ASCII (fts.fold_digits) in the stored text, so FTS matches "950"
    # against Bengali "৯৫০" and snippets show the digits the query used.
    con.executemany(
        "INSERT INTO items(run_id, kind, lang, section, seg_id, slide_id, t, t1, citations, text)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                rows.run_id,
                i.kind,
                i.lang,
                i.section,
                i.seg_id,
                i.slide_id,
                i.t,
                i.t1,
                _citations_json(i.citations) if i.kind == "note" else None,
                fts.fold_digits(i.text),
            )
            for i in rows.items
        ],
    )
    con.executemany(
        "INSERT INTO action_items VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                rows.run_id,
                a.owner,
                a.due,
                a.item.text,
                a.item.lang,
                a.item.seg_id,
                a.item.slide_id,
                a.item.t,
                _citations_json(a.item.citations),
            )
            for a in rows.actions
        ],
    )


@contextmanager
def _transaction(con: sqlite3.Connection) -> Generator[None]:
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK")
        raise
    con.execute("COMMIT")


@dataclass(frozen=True)
class SyncResult:
    indexed: list[str]  # runs (re-)indexed
    removed: list[str]  # runs no longer on disk
    failed: list[str]  # runs whose manifest couldn't be read (left out, retried next time)
    rebuilt: bool


def _sync(con: sqlite3.Connection, runs_dir: Path) -> tuple[list[str], list[str], list[str]]:
    stored = dict(con.execute("SELECT run_id, key FROM runs").fetchall())
    on_disk = {p.name: p for p in storage.list_runs(runs_dir)}
    indexed: list[str] = []
    failed: list[str] = []
    for run_id, run_dir in on_disk.items():
        if stored.get(run_id) == run_key(run_dir):
            continue
        try:
            rows = run_rows(run_dir)
        except (ValidationError, OSError, ValueError) as e:
            log.warning("index: can't read %s: %s", run_id, e)
            failed.append(run_id)
            rows = None
        with _transaction(con):
            _delete_run(con, run_id)
            if rows is not None:
                _insert_run(con, rows)
        if rows is not None:
            indexed.append(run_id)
    removed = sorted(set(stored) - set(on_disk))
    if removed:
        with _transaction(con):
            for run_id in removed:
                _delete_run(con, run_id)
    return indexed, removed, failed


def _usable(path: Path) -> bool:
    """The file exists, is a SQLite database and has this SCHEMA_VERSION."""
    if not path.is_file():
        return False
    try:
        with closing(_connect(path)) as con:
            row = con.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
            con.execute("SELECT count(*) FROM fts_tri").fetchone()  # FTS tables are there too
    except sqlite3.DatabaseError as e:
        log.info("index: %s unusable (%s), rebuilding", path, e)
        return False
    return row is not None and row[0] == str(SCHEMA_VERSION)


def rebuild(runs_dir: Path) -> SyncResult:
    """Build a fresh index from every run on disk into a temp file, then swap it in."""
    dest = runs_dir / FILENAME
    with atomic.atomic_path(dest) as tmp, closing(_connect(tmp)) as con:
        _create(con)
        indexed, _, failed = _sync(con, runs_dir)
    for suffix in ("-journal", "-wal", "-shm"):  # a stale journal must not replay into the new file
        Path(f"{dest}{suffix}").unlink(missing_ok=True)
    return SyncResult(indexed=indexed, removed=[], failed=failed, rebuilt=True)


def sync(runs_dir: Path) -> SyncResult:
    """Bring the index up to date with the runs on disk; rebuild it when it's missing,
    corrupt or from another SCHEMA_VERSION."""
    path = runs_dir / FILENAME
    if not _usable(path):
        return rebuild(runs_dir)
    with closing(_connect(path)) as con:
        indexed, removed, failed = _sync(con, runs_dir)
    return SyncResult(indexed=indexed, removed=removed, failed=failed, rebuilt=False)


@contextmanager
def open_index(runs_dir: Path) -> Generator[sqlite3.Connection]:
    """A synced index, read-only use. Creates runs_dir if needed (the index lives in it)."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    sync(runs_dir)
    with closing(_connect(runs_dir / FILENAME)) as con:
        con.row_factory = sqlite3.Row
        yield con


# --- reading ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    run_id: str
    created_at: str
    kind: Kind
    lang: str | None
    section: str | None
    seg_id: int | None
    slide_id: int | None
    t: float | None
    citations: list[Citation]
    snippet: str  # match marked with MARK_START … MARK_END


def where(t: float | None, slide_id: int | None) -> str:
    """Where a hit or action item opens: its [mm:ss], else its slide, else "-"."""
    if t is not None:
        return clock(t)
    return slide_label(slide_id) if slide_id is not None else "-"


@dataclass(frozen=True)
class Results:
    hits: list[Hit]
    any_term: bool  # no result had every term, so these match any of them


def _filters(
    kinds: Sequence[str] | None,
    langs: Sequence[str] | None,
    since: date | None,
    until: date | None,
) -> tuple[str, list[object]]:
    where: list[str] = []
    args: list[object] = []
    if kinds:
        where.append(f"items.kind IN ({','.join('?' * len(kinds))})")
        args += kinds
    if langs:
        where.append(f"items.lang IN ({','.join('?' * len(langs))})")
        args += langs
    if since is not None:
        where.append("substr(runs.created_at, 1, 10) >= ?")
        args.append(since.isoformat())
    if until is not None:
        where.append("substr(runs.created_at, 1, 10) <= ?")
        args.append(until.isoformat())
    return "".join(f" AND {w}" for w in where), args


def _citations(text: str | None) -> list[Citation]:
    return [Citation.model_validate(c) for c in json.loads(text)] if text else []


def search(
    con: sqlite3.Connection,
    query: str,
    *,
    kinds: Sequence[str] | None = None,
    langs: Sequence[str] | None = None,
    since: date | None = None,
    until: date | None = None,
    limit: int = 50,
) -> Results:
    """BM25-ranked hits for every term of the query; when none has them all (and there's
    more than one term), hits for any term, flagged with `any_term`.

    A query with Chinese, Japanese or Korean searches every row through the trigram table.
    Any other query searches rows in those languages through the trigram table too (with its
    terms of 3+ characters): unicode61 reads a CJK run with digits or Latin letters in it
    ("チケットは390件") as one token, so "390" would never match it."""
    query = fts.fold_digits(query)
    terms, cjk = fts.query_terms(query)
    if not terms:
        return Results([], any_term=False)
    where, args = _filters(kinds, langs, since, until)
    cjk_rows = f"items.lang IN ({','.join(repr(lang) for lang in sorted(fts.CJK_LANGUAGES))})"
    if cjk:
        parts = [("fts_tri", terms, "1")]
    else:
        parts = [("fts_words", terms, f"(items.lang IS NULL OR NOT {cjk_rows})")]
        if long := [t for t in terms if len(t) >= fts.TRIGRAM]:
            parts.append(("fts_tri", long, cjk_rows))
    selects = [
        "SELECT items.*, runs.created_at, "
        f"snippet({table}, 0, ?, ?, '…', {SNIPPET_TOKENS}) AS snip, bm25({table}) AS score "
        f"FROM {table} JOIN items ON items.id = {table}.rowid "
        "JOIN runs ON runs.run_id = items.run_id "
        f"WHERE {table} MATCH ? AND {rows}{where}"
        for table, _, rows in parts
    ]
    sql = " UNION ALL ".join(selects) + " ORDER BY score, id LIMIT ?"
    query_tokens = set(tokens(query))

    def jump(r: sqlite3.Row) -> float | None:
        """A transcript chunk's hit plays from the sentence that best matches the query, placed
        by character position (D19's even speaking rate): chunks are up to 30 s long."""
        t, t1 = r["t"], r["t1"]
        if r["kind"] != "transcript" or t is None or t1 is None:
            return t
        return round(t + sentence_offset(query_tokens, r["text"]) * (t1 - t), 2)

    def run(op: str) -> list[Hit]:
        params: list[object] = []
        for _, part_terms, _ in parts:
            params += [MARK_START, MARK_END, fts.match(part_terms, op), *args]
        rows = con.execute(sql, [*params, limit])
        return [
            Hit(
                run_id=r["run_id"],
                created_at=r["created_at"],
                kind=r["kind"],
                lang=r["lang"],
                section=r["section"],
                seg_id=r["seg_id"],
                slide_id=r["slide_id"],
                t=jump(r),
                citations=_citations(r["citations"]),
                snippet=r["snip"],
            )
            for r in rows
        ]

    hits = run("AND")
    if hits or len(terms) == 1:
        return Results(hits, any_term=False)
    return Results(run("OR"), any_term=True)


@dataclass(frozen=True)
class ActionRow:
    run_id: str
    created_at: str
    owner: str | None
    due: str | None
    text: str
    lang: str | None
    seg_id: int | None
    slide_id: int | None
    t: float | None
    citations: list[Citation]


def action_items(con: sqlite3.Connection, owners: Sequence[str] | None = None) -> list[ActionRow]:
    """Every run's action items, newest run first, in notes order; only `owners`' if given."""
    sql = (
        "SELECT action_items.*, runs.created_at FROM action_items "
        "JOIN runs ON runs.run_id = action_items.run_id"
    )
    args: list[object] = []
    if owners:
        sql += f" WHERE action_items.owner IN ({','.join('?' * len(owners))})"
        args += owners
    sql += " ORDER BY runs.created_at DESC, runs.run_id DESC, action_items.rowid"
    return [
        ActionRow(
            run_id=r["run_id"],
            created_at=r["created_at"],
            owner=r["owner"],
            due=r["due"],
            text=r["text"],
            lang=r["lang"],
            seg_id=r["seg_id"],
            slide_id=r["slide_id"],
            t=r["t"],
            citations=_citations(r["citations"]),
        )
        for r in con.execute(sql, args)
    ]


def owners(con: sqlite3.Connection) -> list[str]:
    rows = con.execute(
        "SELECT DISTINCT owner FROM action_items WHERE owner IS NOT NULL ORDER BY owner"
    )
    return [r[0] for r in rows]


def languages(con: sqlite3.Connection) -> list[str]:
    rows = con.execute("SELECT DISTINCT lang FROM items WHERE lang IS NOT NULL ORDER BY lang")
    return [r[0] for r in rows]


def run_count(con: sqlite3.Connection) -> int:
    return con.execute("SELECT count(*) FROM runs").fetchone()[0]
