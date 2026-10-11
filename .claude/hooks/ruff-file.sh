#!/usr/bin/env bash
# H3 (PLAN §14): format + lint each edited Python file.
f=$(jq -r '.tool_input.file_path // ""')
case "$f" in *.py) ;; *) exit 0 ;; esac

cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 0
# Before `uv init` / `uv add --dev ruff` (start of Phase 0) there's nothing to run.
[ -f pyproject.toml ] && [ -x .venv/bin/ruff ] || exit 0
[ -f "$f" ] || exit 0

uv run --frozen ruff format -q "$f" >/dev/null 2>&1
if ! out=$(uv run --frozen ruff check --fix -q "$f" 2>&1); then
  echo "ruff found lint errors in $f:" >&2
  echo "$out" >&2
  exit 2
fi
exit 0
