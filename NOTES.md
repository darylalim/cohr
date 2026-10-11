# cohr: implementation notes

Findings, measurements and decisions per phase. Results for the PLAN §20 decisions table are
recorded here in detail and copied into the table's **Result** column.

Entry template:

```markdown
## YYYY-MM-DD · Phase N (<name>)

### Findings
- …

### Decisions (PLAN §20)
| # | Result | Evidence |
|---|---|---|

### Plan changes
- …

### Next
- …
```

---

## 2026-10-10 · Phase 0 (checks before building)

### Findings
- `.gitignore` created from PLAN §6; `git check-ignore` check passes (runs/, *.mp4,
  *.safetensors, .env ignored; uv.lock, .claude/settings.json, NOTES.md not ignored).

### Decisions (PLAN §20)
| # | Result | Evidence |
|---|---|---|

### Plan changes
- None yet.

### Next
- Hooks (§14), then the uv project and model checks.
