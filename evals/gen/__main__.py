"""Generate one eval item.

Usage (from the repo root):
  uv run python -m evals.gen ITEM OUT_DIR [--set quick] [--seed N]

ITEM is an item id from evals/sets/<set>.yaml, or an ItemSpec as JSON (what `cohr eval` passes).
OUT_DIR is replaced as a whole; keep it under evals/data/ (gitignored).
"""

import argparse
from pathlib import Path

from cohr.evals import ItemSpec, load_set
from evals.gen.build import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("item", help="item id, or an ItemSpec as JSON")
    parser.add_argument("out", type=Path)
    parser.add_argument("--set", default="quick", help="eval set to look the item up in")
    parser.add_argument("--seed", type=int, help="override the item's seed")
    args = parser.parse_args()
    if args.item.lstrip().startswith("{"):
        spec = ItemSpec.model_validate_json(args.item)
    else:
        found = [s for s in load_set(args.set).items if s.id == args.item]
        if not found:
            raise SystemExit(f"no item {args.item!r} in set {args.set!r}")
        spec = found[0]
    if args.seed is not None:
        spec = spec.model_copy(update={"seed": args.seed})
    truth = build(spec, args.out)
    print(
        f"{spec.id}: {truth.duration_s:.0f} s, {len(truth.facts)} facts, "
        f"{len(truth.slides)} slides, gaps {[(g.t0, g.t1) for g in truth.gap_spans]} -> {args.out}"
    )


if __name__ == "__main__":
    main()
