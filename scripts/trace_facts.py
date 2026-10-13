"""Phase 6b-2 step 0: trace each planted fact transcript → Map → notes (no models). Throwaway.

  uv run python scripts/trace_facts.py evals/reports/<ts>-quick.json

Reads the report's runs under evals/data/runs and the items' truth.json under evals/data/gen
(so run it right after the eval, before another run replaces those directories). Prints:
- per segment: duration, planted facts inside it, Map list sizes and how many planted facts
  each Map key point holds (a bundle shows up as 2 or more);
- per fact: ASR right in a ±2 s window, the Map lists that hold it (×N = facts in that string),
  the notes section the eval found it in and its citation; the ASR window when ASR missed it.
Map matching is the eval's rule, simplified: numbers by value, text by coverage ≥ 0.6, owner
by name.
"""

import json
import sys
from pathlib import Path

from cohr import schemas
from cohr.evals import TEXT_MATCH_MIN, Truth, TruthFact, coverage, due_key
from cohr.numbers import numbers_in
from cohr.schemas import MapFile, SegmentNotes, SegmentsFile, Transcript


def map_strings(n: SegmentNotes) -> list[tuple[str, str]]:
    out = [("summary", n.summary)]
    out += [("key_points", s) for s in n.key_points]
    out += [("decisions", s) for s in n.decisions]
    out += [
        ("action_items", f"{a.owner or '?'}: {a.task} (due {a.due or '?'})") for a in n.action_items
    ]
    out += [("open_questions", s) for s in n.open_questions]
    return out


def fits(fact: TruthFact, text: str) -> bool:
    if fact.kind == "number":
        return fact.value in numbers_in(text)
    if coverage(fact.text, text) < TEXT_MATCH_MIN:
        return False
    return fact.owner is None or fact.owner.casefold() in text.casefold()


def asr_ok(fact: TruthFact, text: str) -> bool:
    if fact.kind == "number":
        return fact.value in numbers_in(text)
    ok = coverage(fact.text, text) >= TEXT_MATCH_MIN
    if fact.owner:
        ok = ok and fact.owner.casefold() in text.casefold()
    if fact.due:
        ok = ok and due_key(fact.due) <= due_key(text) | set(text.casefold().split())
    return ok


def main(report_path: Path, root: Path = Path()) -> None:
    report = json.loads(report_path.read_text())
    seg_rows, fact_rows = [], []
    for item in report["items"]:
        run = root / "evals/data/runs" / item["id"] / item["run_id"]
        truth = Truth.model_validate_json(
            (root / "evals/data/gen" / item["id"] / "truth.json").read_text()
        )
        segs = schemas.load(run, SegmentsFile).segments
        notes = {n.seg_id: n for n in schemas.load(run, MapFile).notes}
        transcript = schemas.load(run, Transcript).items
        found = {f["id"]: f for f in item["facts"]}
        facts = truth.facts
        for s in segs:
            n = notes[s.seg_id]
            per_kp = [
                sum(fits(f, t) for f in facts) for k, t in map_strings(n) if k == "key_points"
            ]
            in_seg = sum(s.t0 <= f.t0 < s.t1 for f in facts)
            seg_rows.append(
                (item["id"], s.seg_id, round(s.t1 - s.t0), in_seg, len(n.key_points), per_kp,
                 len(n.decisions), len(n.action_items), len(n.open_questions), n.raw)
            )  # fmt: skip
        for f in facts:
            seg = next((s for s in segs if s.t0 <= f.t0 < s.t1), None)
            window = " ".join(i.text for i in transcript if i.t1 >= f.t0 - 2 and i.t0 <= f.t1 + 2)
            hits = []
            if seg is not None:
                for k, t in map_strings(notes[seg.seg_id]):
                    if fits(f, t):
                        hits.append(f"{k}×{sum(fits(g, t) for g in facts)}")
            r = found.get(f.id, {})
            asr = asr_ok(f, window)
            fact_rows.append(
                (item["id"], f.id, f.kind, f.text[:48], f.owner or "", f.due or "",
                 "ok" if asr else "MISS", seg.seg_id if seg else "-", ",".join(hits) or "—",
                 r.get("section") if r.get("found") else "—",
                 {True: "ok", False: "off"}.get(r.get("citation_ok"), ""), "" if asr else window)
            )  # fmt: skip
    print("## Segments: item | seg | s | facts in seg | kp | facts per kp | dec | act | oq | raw")
    for row in seg_rows:
        print(*row, sep=" | ")
    print("\n## Facts: item | id | kind | text | owner | due | ASR | seg | Map | notes | cite")
    for row in fact_rows:
        print(*row, sep=" | ")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path())
