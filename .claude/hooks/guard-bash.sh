#!/usr/bin/env bash
# H1 (PLAN §14): enforce uv, block risky commands, ask before outward actions.
# A guardrail against mistakes, not a security boundary; regexes are best effort.
cmd=$(jq -r '.tool_input.command // ""')

block() { echo "Blocked: $1" >&2; exit 2; }
grep -Eq '(^|[;&|[:space:]])(pip3?|python3?[[:space:]]+-m[[:space:]]+pip)[[:space:]]+install' <<<"$cmd" && block "use uv add / uv add --dev"
grep -Eq '(^|[;&|[:space:]])sudo[[:space:]]' <<<"$cmd" && block "no sudo; ask the user (see PLAN §4)"
grep -Eq 'rm[[:space:]]+-[a-zA-Z]*r[a-zA-Z]*f?.*(runs|\.cache/huggingface|HF_HOME)' <<<"$cmd" && block "use cohr rm/prune or hf cache delete; ask first"
grep -Eq "pytest.*-m[[:space:]]*[\"']?models" <<<"$cmd" && block "model smoke test loads ~15 GB; ask the user first"
grep -Eq 'cohr[[:space:]]+eval.*--set[[:space:]=]+full' <<<"$cmd" && block "full eval takes ~1 h; ask the user first"
grep -Eq 'git[[:space:]]+push.*(--force|[[:space:]]-f([[:space:]]|$))' <<<"$cmd" && block "no force push"

ask() {
  jq -n --arg r "$1" '{hookSpecificOutput: {hookEventName: "PreToolUse",
    permissionDecision: "ask", permissionDecisionReason: $r}}'
  exit 0
}
grep -Eq '(^|[;&|[:space:]])git[[:space:]]+push' <<<"$cmd" && ask "Pushes to GitHub."
grep -Eq 'gh[[:space:]]+repo[[:space:]]+(create|edit|delete)' <<<"$cmd" && ask "Changes the GitHub repository."
grep -Eq 'gh[[:space:]]+api.*(-X|--method)[[:space:]=]*(PUT|POST|PATCH|DELETE)' <<<"$cmd" && ask "Writes via the GitHub API."
grep -Eq 'gh[[:space:]]+release[[:space:]]+(create|edit|delete|upload)' <<<"$cmd" && ask "Changes GitHub releases."
grep -Eq 'gh[[:space:]]+pr[[:space:]]+(merge|close)' <<<"$cmd" && ask "Merges or closes a pull request."
exit 0
