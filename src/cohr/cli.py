"""Command line (PLAN §6, §12, §18): `cohr run`, `resume`, `runs`, `prune`, `rm`, `models`,
`eval`."""

import argparse
import sys
from pathlib import Path

from cohr import models, pipeline, schemas, storage
from cohr.asr import LANGUAGES
from cohr.config import Config, runs_dir
from cohr.ingest import IngestError, probe
from cohr.models import ModelManager, ModelsMissing
from cohr.pipeline import Cancelled, ProgressEvent, StageFailed
from cohr.schemas import Manifest
from cohr.storage import DiskSpaceError, Source


def _print_progress(event: ProgressEvent) -> None:
    if event.status == "progress":
        print(f"  {event.stage}: {event.fraction:4.0%} {event.message}", file=sys.stderr)
    else:
        print(f"{event.stage}: {event.status} {event.message}".rstrip(), file=sys.stderr)


def _execute(run_dir: Path) -> int:
    config = Config.model_validate(schemas.load(run_dir, Manifest).config)
    try:
        pipeline.run(run_dir, progress=_print_progress, models=ModelManager(config))
    except ModelsMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted; resume with `cohr resume`", file=sys.stderr)
        return 130
    except Cancelled:
        return 130
    except StageFailed as e:
        print(f"error: {e} (see {run_dir / 'log.txt'})", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


def _resolve_run(arg: str) -> Path:
    path = Path(arg)
    if not storage.is_run(path):
        path = runs_dir() / arg
    if not storage.is_run(path):
        raise SystemExit(f"error: no run {arg!r} (looked in {runs_dir()})")
    return path


def cmd_run(args: argparse.Namespace) -> int:
    if (args.audio is None) == (args.video is None):
        raise SystemExit("error: give exactly one of AUDIO or --video")
    media = Source(Path(args.audio or args.video), "audio" if args.audio else "video")
    sources = [media, *(Source(Path(p), "slides") for p in args.slides)]
    config = Config(
        source_language=args.lang,
        target_language=args.target,
        source_mode="copy" if args.copy else "link",
    )
    try:
        duration = probe(media.path).duration_s
        run_dir = storage.create_run(
            runs_dir(), sources, config, name=args.name, duration_s=duration
        )
    except (FileNotFoundError, DiskSpaceError, IngestError) as e:
        raise SystemExit(f"error: {e}") from e
    print(f"run: {run_dir.name}", file=sys.stderr)
    return _execute(run_dir)


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run)
    pipeline.recover_interrupted(run_dir)
    return _execute(run_dir)


def cmd_runs(_args: argparse.Namespace) -> int:
    for run_dir in storage.list_runs(runs_dir()):
        manifest = schemas.load(run_dir, Manifest)
        stages = " ".join(f"{k}:{v.status}" for k, v in manifest.stages.items()) or "new"
        pruned = " pruned" if manifest.pruned else ""
        size = storage.run_size(run_dir) / storage.MB
        print(f"{run_dir.name}  {size:8.1f} MB  {stages}{pruned}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    freed = storage.prune_media(_resolve_run(args.run))
    print(f"freed {freed / storage.MB:.1f} MB")
    return 0


def cmd_rm(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run)
    if not args.yes:
        answer = input(f"Delete {run_dir}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return 1
    storage.delete_run(run_dir)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    config = Config()
    names = args.names or list(models.PIPELINE_MODELS)
    unknown = [n for n in names if n not in config.models]
    if unknown:
        raise SystemExit(f"error: unknown model(s) {unknown}; choose from {list(config.models)}")
    if args.action == "download":
        specs = {n: config.models[n] for n in names}
        try:
            fetched = models.download(specs)
        except DiskSpaceError as e:
            raise SystemExit(f"error: {e}") from e
        print(f"downloaded: {', '.join(fetched)}" if fetched else "all cached", file=sys.stderr)
    for name, spec, size in models.cached_report(config, names):
        state = f"{size / storage.GB:6.2f} GB" if size is not None else "   missing"
        print(f"{name:<11} {state}  {spec.repo}@{spec.revision[:7]}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from cohr import evals  # pulls in the eval dependency group lazily

    root = Path(args.root)
    if args.calibrate_judge:
        return _calibrate_judge(root)
    try:
        overrides = dict(evals.parse_override(o) for o in args.config)
        report = evals.run_set(
            args.set,
            root=root,
            items=args.item,
            seed=args.seed,
            overrides=overrides,
            fresh=args.fresh,
            models=ModelManager(Config()),
            judge_=not args.no_judge,
            log=lambda msg: print(msg, file=sys.stderr),
            progress=_print_progress,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        raise SystemExit(f"error: {e}") from e
    except ModelsMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except StageFailed as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    baseline = Path(args.baseline) if args.baseline else root / evals.BASELINE
    if baseline.is_file() and not args.update_baseline:
        report.baseline = str(baseline)
        report.regressions = evals.compare(report, evals.load_report(baseline))
    path = evals.write_report(report, root)
    print(evals.render_markdown(report))
    print(f"report: {path} (+ .md)", file=sys.stderr)
    if args.update_baseline:
        print(f"baseline updated: {evals.write_baseline(report, root)}", file=sys.stderr)
    return 1 if report.regressions else 0


def _calibrate_judge(root: Path) -> int:
    from cohr import evals

    try:
        report = evals.calibrate(
            root=root, models=ModelManager(Config()), log=lambda m: print(m, file=sys.stderr)
        )
    except FileNotFoundError as e:
        raise SystemExit(f"error: {e}") from e
    except ModelsMissing as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    path = evals.write_calibration_report(report, root)
    print(evals.render_calibration(report))
    print(f"report: {path} (+ .md)", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cohr", description="Cited meeting notes on MLX.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="create a run and process it")
    p.add_argument("audio", nargs="?", help="audio file")
    p.add_argument("--video", help="video file (audio + slide frames)")
    p.add_argument("--slides", nargs="*", default=[], help="PDF or image slide files")
    p.add_argument("--lang", default="en", choices=LANGUAGES, help="source language (default en)")
    p.add_argument("--target", default="en", help="target language for the notes (default en)")
    p.add_argument("--copy", action="store_true", help="copy sources instead of linking")
    p.add_argument("--name", help="run name (default: the media file name)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="re-run a run from its first incomplete stage")
    p.add_argument("run")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("runs", help="list runs with size and stage status")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("prune", help="remove media (audio, frames, sources), keep notes")
    p.add_argument("run")
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("rm", help="delete a run")
    p.add_argument("run")
    p.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("models", help="list pipeline models and whether they're downloaded")
    p.add_argument("action", nargs="?", choices=["list", "download"], default="list")
    p.add_argument("names", nargs="*", help="model names (default: the pipeline models)")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("eval", help="run an eval set and score it (PLAN §18)")
    p.add_argument("--set", default="quick", help="evals/sets/<name>.yaml (default quick)")
    p.add_argument("--item", nargs="*", default=[], help="only these item ids")
    p.add_argument("--seed", type=int, help="override every item's seed")
    p.add_argument(
        "-c",
        "--config",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. fuse.window_s=120 (repeatable)",
    )
    p.add_argument("--fresh", action="store_true", help="new runs instead of reusing stages")
    p.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the judge metrics (support rate, judge recall)",
    )
    p.add_argument(
        "--calibrate-judge",
        action="store_true",
        help="only run the judge on evals/gen/judge_calibration.json (D15)",
    )
    p.add_argument("--baseline", help="report to compare with (default evals/baseline.json)")
    p.add_argument(
        "--update-baseline", action="store_true", help="write this report as the baseline"
    )
    p.add_argument("--root", default=".", help="repo root (default: current directory)")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
