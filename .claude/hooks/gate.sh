#!/usr/bin/env bash
# H4 (PLAN §14): run the §6 gate before Claude finishes.
input=$(cat)
[ "$(jq -r '.stop_hook_active // false' <<<"$input")" = "true" ] && exit 0

cd "${CLAUDE_PROJECT_DIR:-$PWD}" || exit 0
[ -f pyproject.toml ] || exit 0
# Nothing Python changed (e.g. a planning-only turn).
[ -z "$(git status --porcelain -- '*.py' pyproject.toml)" ] && exit 0

run() {
  local out code
  out=$("$@" 2>&1)
  code=$?
  # pytest exit 5 = no tests collected (early Phase 0, before tests/ exists).
  [ "$1 $2 $3 $4" = "uv run --frozen pytest" ] && [ $code -eq 5 ] && return 0
  if [ $code -ne 0 ]; then
    echo "Gate failed: $*" >&2
    tail -n 40 <<<"$out" >&2
    exit 2
  fi
}
run uv run --frozen ruff format --check .
run uv run --frozen ruff check .
run uv run --frozen ty check
run uv run --frozen pytest -q -x -m "not models"
exit 0
