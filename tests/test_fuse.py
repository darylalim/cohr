"""PLAN §5.4: slide text alignment (monotonic DP, threshold, skips), windows, token budget."""

from pathlib import Path

from cohr import fuse, schemas
from cohr.config import Config, FuseConfig
from cohr.schemas import SegmentsFile, Slide, SlidesFile, Transcript, TranscriptItem
from tests.conftest import ctx_for, make_run


def _item(cid: int, t0: float, t1: float, text: str = "") -> TranscriptItem:
    return TranscriptItem(chunk_id=cid, t0=t0, t1=t1, text=text)


def _slide(sid: int, title: str = "", t: float | None = None, text: str = "") -> Slide:
    return Slide(
        slide_id=sid,
        frame_id=sid,
        t=t,
        t_source="pts" if t is not None else "none",
        title=title,
        text_md=text,
        tables_md=[],
        chart_notes="",
        visual_notes="",
    )


# --- tokens and scores ---------------------------------------------------------------------


def test_tokens_casefold_drop_stopwords_and_keep_numbers() -> None:
    assert fuse.tokens("The Q4 Budget is $14,500 and 38%!") == ["q4", "budget", "14500", "38"]
    assert fuse.tokens("a I x") == []


def test_tokens_split_cjk_into_bigrams() -> None:
    assert fuse.tokens("予算は承認") == ["予算", "算は", "は承", "承認"]
    assert fuse.tokens("新 launch") == ["新", "launch"]


def test_overlap_uses_smaller_set_with_a_floor() -> None:
    small = {"budget", "launch"}
    assert fuse.overlap(small, {"budget", "launch", "x", "y"}) == 2 / fuse.MIN_DENOMINATOR
    big = {f"w{i}" for i in range(20)}
    assert fuse.overlap(big, set(list(big)[:10]) | {"other"}) == 10 / 11
    assert fuse.overlap(set(), big) == 0.0


def test_common_tokens_are_dropped_from_scores() -> None:
    chunks = ["atlas budget approved", "atlas launch date", "atlas retention", "atlas support"]
    scores = fuse.score_matrix(["Atlas"], chunks)
    assert scores == [[0.0, 0.0, 0.0, 0.0]]
    assert fuse.score_matrix(["Atlas"], chunks[:3])[0][0] > 0  # too few chunks to tell


# --- monotonic alignment -------------------------------------------------------------------


def test_align_is_monotonic_and_maximizes_the_total() -> None:
    # Slide 0 fits chunk 2 best, but placing it there would push slides 1 and 2 out.
    scores = [
        [0.1, 0.0, 0.9],
        [0.9, 0.0, 0.0],
        [0.0, 0.9, 0.0],
    ]
    assert fuse.align(scores, 0.15) == [None, 0, 1]


def test_align_respects_threshold_and_skips() -> None:
    scores = [[0.5, 0.1], [0.1, 0.14], [0.0, 0.6]]
    assert fuse.align(scores, 0.15) == [0, None, 1]


def test_align_allows_two_slides_on_one_chunk_and_breaks_ties_early() -> None:
    assert fuse.align([[0.0, 0.5], [0.0, 0.5]], 0.15) == [1, 1]
    assert fuse.align([[0.5, 0.5]], 0.15) == [0]


def test_align_with_no_chunks_or_slides() -> None:
    assert fuse.align([[], []], 0.15) == [None, None]
    assert fuse.align([], 0.15) == []


def test_align_slides_keeps_pts_and_is_idempotent() -> None:
    items = [
        _item(0, 0, 30, "welcome to the weekly planning meeting"),
        _item(1, 30, 60, "the approved budget for next quarter is 14,500 dollars"),
        _item(2, 60, 90, "we decided to launch the beta on 3 November"),
    ]
    slides = [
        _slide(0, "Q4 Budget", text="Approved: $14,500 for the quarter"),
        _slide(1, "Video frame", t=5.0),
        _slide(2, "Beta Launch Plan", text="Launch: 3 November"),
        _slide(3, "Hiring", text="Two engineers in January"),
    ]
    out = fuse.align_slides(slides, items, 0.15)
    assert [(s.t, s.t_source) for s in out] == [
        (30, "aligned"),
        (5.0, "pts"),
        (60, "aligned"),
        (None, "none"),
    ]
    assert out[0].align_score is not None and out[0].align_score >= 0.15
    strict = fuse.align_slides(out, items, 0.99)  # re-run on aligned output
    assert [s.t_source for s in strict] == ["none", "pts", "none", "none"]
    assert strict[0].align_score is None
    assert fuse.align_slides(strict, items, 0.15) == out


# --- windows, slides per segment, token budget ---------------------------------------------


def test_windows_never_span_more_than_window_s() -> None:
    items = [_item(i, i * 30, i * 30 + 28) for i in range(12)]  # 0 … 358 s
    groups = fuse.windows(items, 300)
    assert [len(g) for g in groups] == [10, 2]
    assert all(g[-1].t1 - g[0].t0 <= 300 for g in groups)
    assert fuse.windows([], 300) == []


def test_attach_slides_by_time_and_on_screen_at_start() -> None:
    groups = [[_item(0, 10, 40)], [_item(1, 100, 130)], [_item(2, 200, 230)]]
    slides = [
        _slide(0, t=0.0),  # before the first group → first group
        _slide(1, t=20.0),  # during group 0, still on screen when group 1 starts
        _slide(2, t=70.0),  # shown in the silence before group 1 → group 1
        _slide(3, t=210.0),
        _slide(4, t=500.0),  # after the end → last group
        _slide(5),  # unaligned: in no group
    ]
    assert fuse.attach(groups, slides) == [[0, 1], [2], [2, 3, 4]]


def test_attach_adds_the_slide_on_screen_when_the_group_starts() -> None:
    groups = [[_item(0, 0, 30)], [_item(1, 30, 60)], [_item(2, 60, 90)]]
    assert fuse.attach(groups, [_slide(0, t=5.0)]) == [[0], [0], [0]]


def test_build_segments_splits_over_budget() -> None:
    items = [_item(i, i * 10, i * 10 + 9, "word " * 100) for i in range(8)]  # ~170 tokens each
    cfg = FuseConfig(window_s=300, max_segment_tokens=400)
    segments, warnings = fuse.build_segments(items, [], cfg)
    assert [s.chunk_ids for s in segments] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert [s.seg_id for s in segments] == [0, 1, 2, 3]
    assert (segments[1].t0, segments[1].t1) == (20, 39)
    assert warnings == []


def test_build_segments_counts_slide_text_and_warns_on_a_single_chunk() -> None:
    items = [_item(0, 0, 9, "hello"), _item(1, 10, 19, "there")]
    big = _slide(0, "Big", t=12.0, text="table " * 2000)
    segments, warnings = fuse.build_segments(items, [big], FuseConfig(max_segment_tokens=1000))
    assert [s.chunk_ids for s in segments] == [[0], [1]]
    assert segments[1].slide_ids == [0]
    assert len(warnings) == 1 and warnings[0].startswith("segment 1:")


def test_estimate_tokens_counts_cjk_per_character() -> None:
    assert fuse.estimate_tokens("abcdef") == 2
    assert fuse.estimate_tokens("予算ab") == 3


# --- stage ---------------------------------------------------------------------------------


def test_stage_writes_segments_and_aligned_slides(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path, config=Config(fuse=FuseConfig(window_s=60)))
    items = [
        _item(0, 0, 25, "welcome everyone to the planning meeting"),
        _item(1, 30, 55, "the approved budget is 14,500 dollars for user testing"),
        _item(2, 70, 95, "retention went up to 38 percent thanks to the checklist"),
    ]
    schemas.save(run_dir, Transcript(language="en", runtime="mlx-speech", items=items))
    slides = [
        _slide(0, "Budget", text="Approved 14,500 user testing"),
        _slide(1, "Retention", text="38% week-2 return; onboarding checklist"),
        _slide(2, "Appendix", text="Glossary of internal codenames"),
    ]
    schemas.save(run_dir, SlidesFile(slides=slides))

    fuse.run_stage(ctx_for(run_dir, "fuse"))

    out = schemas.load(run_dir, SlidesFile).slides
    assert [(s.t, s.t_source) for s in out] == [(30, "aligned"), (70, "aligned"), (None, "none")]
    seg = schemas.load(run_dir, SegmentsFile)
    assert [(s.t0, s.t1, s.chunk_ids, s.slide_ids) for s in seg.segments] == [
        (0, 55, [0, 1], [0]),
        (70, 95, [2], [1]),  # slide 1 replaced slide 0 on screen at 70 s
    ]
    assert seg.segments[0].transcript_text.splitlines()[0] == items[0].text
    assert seg.unaligned_slide_ids == [2]


def test_stage_with_no_speech_leaves_every_slide_unaligned(tmp_path: Path) -> None:
    run_dir = make_run(tmp_path)
    schemas.save(run_dir, Transcript(language="en", runtime="mlx-speech", items=[]))
    schemas.save(run_dir, SlidesFile(slides=[_slide(0, "A"), _slide(1, "B")]))
    fuse.run_stage(ctx_for(run_dir, "fuse"))
    seg = schemas.load(run_dir, SegmentsFile)
    assert seg.segments == [] and seg.unaligned_slide_ids == [0, 1]
