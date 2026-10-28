"""Search check without models (Phase 7, report-only): query each planted fact against an index
built from the eval runs, the way a person searching across meetings would.

    uv run python scripts/search_check.py [--runs evals/data/runs] [--gen evals/data/gen]

The index is built in a temp folder of symlinks to evals/data/runs/<item>/<run>/, so nothing
is written under evals/ (the app never indexes eval runs). Every run is indexed, AMI included,
so a fact competes with the other meetings' rows; only items with a truth.json are queried.

Per fact, three queries, each searched across every run and scored on the fact's own run:
- number: the value as a person types it ("14,500", "2.5"); misspoken values left out
- owner: the action owner's name
- words: the fact's text (AND first, then any word: the Search page's rule)
Each cell is "transcript · notes": how many of the facts' queries have, among the top TOP
hits in the fact's run, a transcript hit (at its matched sentence) or a note (at its [mm:ss])
within ±TOLERANCE_S of the fact's span (D19). "Any word" counts the queries where no row had
every word, so Search fell back to any of them.
"""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from cohr import index, schemas, storage
from cohr.schemas import Manifest

TOP = 5
TOLERANCE_S = 15.0
QUERIES = ("number", "owner", "words")


def number_text(value: float) -> str:
    return f"{int(value):,}" if value == int(value) else f"{value:g}"


def queries(fact: dict) -> dict[str, str]:
    out = {"words": fact["text"]}
    if fact["value"] is not None and not fact["misspoken"]:
        out["number"] = number_text(fact["value"])
    if fact["owner"]:
        out["owner"] = fact["owner"]
    return out


def at_moment(hits: list[index.Hit], fact: dict, kind: str) -> bool:
    lo, hi = fact["t0"] - TOLERANCE_S, fact["t1"] + TOLERANCE_S
    return any(h.kind == kind and h.t is not None and lo <= h.t <= hi for h in hits[:TOP])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default="evals/data/runs", type=Path)
    parser.add_argument("--gen", default="evals/data/gen", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="cohr-search-check-") as tmp:
        links = Path(tmp) / "runs"
        links.mkdir()
        item_of: dict[str, str] = {}
        for item_dir in sorted(p for p in args.runs.iterdir() if p.is_dir()):
            for run_dir in storage.list_runs(item_dir):
                (links / run_dir.name).symlink_to(run_dir.resolve(), target_is_directory=True)
                item_of[run_dir.name] = item_dir.name
        result = index.rebuild(links)
        print(f"indexed {len(result.indexed)} runs ({len(result.failed)} unreadable)\n")

        # item → query kind → [transcript at moment, note at moment, asked, any-word fallbacks]
        table: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0] * 4))
        langs: dict[str, str] = {}
        with index.open_index(links) as con:
            for run_id, item in sorted(item_of.items(), key=lambda kv: kv[1]):
                truth_file = args.gen / item / "truth.json"
                if not truth_file.is_file():
                    continue
                truth = json.loads(truth_file.read_text())
                config = schemas.load(links / run_id, Manifest).config
                langs[item] = f"{config.get('source_language')} → {config.get('target_language')}"
                for fact in truth["facts"]:
                    for kind, q in queries(fact).items():
                        results = index.search(con, q, limit=1000)
                        hits = [h for h in results.hits if h.run_id == run_id]
                        row = table[item][kind]
                        row[0] += at_moment(hits, fact, "transcript")
                        row[1] += at_moment(hits, fact, "note")
                        row[2] += 1
                        row[3] += results.any_term

    def cell(row: list[int]) -> str:
        tr, note, n, _ = row
        return f"{tr}/{n} · {note}/{n}" if n else "-"

    print(f"top {TOP} hits in the fact's run within ±{TOLERANCE_S:.0f} s: transcript · notes\n")
    print("| item | languages | " + " | ".join(QUERIES) + " | any word |")
    print("|---|---|" + "---|" * (len(QUERIES) + 1))
    totals: dict[str, list[int]] = defaultdict(lambda: [0] * 4)
    for item in sorted(table):
        for kind in QUERIES:
            totals[kind] = [a + b for a, b in zip(totals[kind], table[item][kind], strict=True)]
        cells = " | ".join(cell(table[item][kind]) for kind in QUERIES)
        fallbacks = sum(table[item][k][3] for k in QUERIES)
        print(f"| {item} | {langs[item]} | {cells} | {fallbacks} |")
    fallbacks = sum(totals[k][3] for k in QUERIES)
    print(f"| **all** | | {' | '.join(cell(totals[k]) for k in QUERIES)} | {fallbacks} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
