#!/usr/bin/env bash
# H2 (PLAN §14): block edits to generated and private files.
path=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // ""')
[ -z "$path" ] && exit 0

root="${CLAUDE_PROJECT_DIR:-$PWD}"
rel="${path#"$root"/}"
name=$(basename "$path")

block() { echo "Blocked: $1" >&2; exit 2; }
[ "$rel" = "uv.lock" ] && block "uv.lock is managed by uv; change pyproject.toml or run uv add"
case "$rel" in runs/*) block "run outputs are produced by the pipeline, not edited by hand" ;; esac
case "$name" in .env | .env.*) block "$name may hold secrets (HF token)" ;; esac

for cache in "${COHR_HF_HOME:-}" "${HF_HOME:-}" "$HOME/.cache/huggingface"; do
  [ -n "$cache" ] && case "$path" in "$cache"/*) block "files in the Hugging Face cache are never edited" ;; esac
done
exit 0
