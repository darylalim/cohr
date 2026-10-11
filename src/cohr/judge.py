"""Judge (PLAN §18): Qwen3 on mlx-lm, eval only, for the secondary metrics.

Not a model the pipeline uses (no self-preference). Two questions, JSON only, temperature 0:
- support: claim + the text it cites → {"supported": "yes|partial|no", "reason"};
- coverage: a fact + one section's claims → {"covered": bool, "reason"} (cross-language
  facts with no exact field to match, §18).
Invalid JSON is retried once with `judge_retry.md`; after that the answer is "unjudged" (or
`covered: None`), never counted as supported or covered.

`CachedJudge` keeps answers per item on disk, keyed by the question's text, the prompt hashes
and the judge revision, and asks for the model only on a cache miss. Re-scoring with nothing
stale therefore loads no model.
"""

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from cohr import atomic
from cohr.config import load_prompt, prompt_hash
from cohr.mapper import fill

MAX_TOKENS = 256
PROMPTS = ("judge", "judge_coverage", "judge_retry")

Label = Literal["yes", "partial", "no"]
Supported = Literal["yes", "partial", "no", "unjudged"]
Generate = Callable[[str], str]

_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SupportAnswer(_Model):
    supported: Supported
    reason: str


class CoverageAnswer(_Model):
    covered: bool | None  # None = unjudged
    reason: str


# --- parsing (pure) ------------------------------------------------------------------------


def _object(text: str) -> dict[str, Any] | None:
    m = _JSON_OBJECT.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _reason(obj: dict[str, Any]) -> str:
    reason = obj.get("reason")
    return reason.strip() if isinstance(reason, str) else ""


def parse_support(text: str) -> SupportAnswer | None:
    obj = _object(text)
    if obj is None:
        return None
    label = obj.get("supported")
    label = label.strip().casefold() if isinstance(label, str) else None
    if label not in ("yes", "partial", "no"):
        return None
    return SupportAnswer(supported=label, reason=_reason(obj))


def parse_coverage(text: str) -> CoverageAnswer | None:
    obj = _object(text)
    if obj is None:
        return None
    covered = obj.get("covered")
    if isinstance(covered, str) and covered.strip().casefold() in ("true", "false"):
        covered = covered.strip().casefold() == "true"
    if not isinstance(covered, bool):
        return None
    return CoverageAnswer(covered=covered, reason=_reason(obj))


def _ask[T](generate: Generate, prompt: str, parse: Callable[[str], T | None]) -> T | None:
    """The parsed reply, retrying once with the retry instruction appended."""
    for template in (prompt, f"{prompt}\n\n{load_prompt('judge_retry')}"):
        if (answer := parse(generate(template))) is not None:
            return answer
    return None


def support_prompt(claim: str, cited: str) -> str:
    return fill(load_prompt("judge"), claim=claim, cited=cited)


def coverage_prompt(fact: str, claims: Sequence[str]) -> str:
    listed = "\n".join(f"- {c}" for c in claims) or "(none)"
    return fill(load_prompt("judge_coverage"), fact=fact, claims=listed)


def judge_support(generate: Generate, claim: str, cited: str) -> SupportAnswer:
    answer = _ask(generate, support_prompt(claim, cited), parse_support)
    return answer or SupportAnswer(supported="unjudged", reason="no valid JSON after a retry")


def judge_coverage(generate: Generate, fact: str, claims: Sequence[str]) -> CoverageAnswer:
    if not claims:
        return CoverageAnswer(covered=False, reason="no claims in the section")
    answer = _ask(generate, coverage_prompt(fact, claims), parse_coverage)
    return answer or CoverageAnswer(covered=None, reason="no valid JSON after a retry")


# --- cache ---------------------------------------------------------------------------------


def cache_key(kind: str, revision: str, *texts: str) -> str:
    """Question kind + its texts + every judge prompt's hash + the judge revision."""
    h = hashlib.sha256()
    for part in (kind, revision, *(prompt_hash(p) for p in PROMPTS), *texts):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class CachedJudge:
    """Support and coverage answers through an on-disk cache (one JSON file per item).

    `model` returns the generate function and is called only on the first cache miss, so the
    caller can release the pipeline models and load the judge lazily (§4)."""

    def __init__(self, revision: str, model: Callable[[], Generate]):
        self.revision = revision
        self._model = model
        self._generate: Generate | None = None
        self.calls = 0  # questions sent to the model
        self.hits = 0  # questions answered from the cache
        self._path: Path | None = None
        self._entries: dict[str, dict[str, Any]] = {}

    @property
    def loaded(self) -> bool:
        return self._generate is not None

    def open(self, path: Path) -> None:
        """Switch to another item's cache file (saving the current one first)."""
        self.save()
        self._path = path
        self._entries = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def save(self) -> None:
        if self._path is not None:
            atomic.write_text(self._path, json.dumps(self._entries, indent=1, sort_keys=True))

    def _gen(self, prompt: str) -> str:
        if self._generate is None:
            self._generate = self._model()
        return self._generate(prompt)

    def _cached[T: _Model](self, key: str, model: type[T], ask: Callable[[], T]) -> T:
        if (entry := self._entries.get(key)) is not None:
            try:
                answer = model.model_validate(entry)
            except ValidationError:
                pass
            else:
                self.hits += 1
                return answer
        self.calls += 1
        answer = ask()
        self._entries[key] = answer.model_dump()
        return answer

    def support(self, claim: str, cited: str) -> SupportAnswer:
        key = cache_key("support", self.revision, claim, cited)
        return self._cached(key, SupportAnswer, lambda: judge_support(self._gen, claim, cited))

    def coverage(self, fact: str, claims: Sequence[str]) -> CoverageAnswer:
        key = cache_key("coverage", self.revision, fact, *claims)
        return self._cached(key, CoverageAnswer, lambda: judge_coverage(self._gen, fact, claims))


# --- model (MLX) ---------------------------------------------------------------------------


class Qwen3Judge:
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )
        return generate(
            self.model, self.tokenizer, text, max_tokens=MAX_TOKENS, sampler=make_sampler(temp=0.0)
        )


def load(path: Path) -> Qwen3Judge:
    from mlx_lm import load as load_lm

    loaded = load_lm(str(path))  # (model, tokenizer); typed as maybe + config
    return Qwen3Judge(loaded[0], loaded[1])
