"""Judge wrapper (PLAN §18): parsing, the retry, prompts and the answer cache. No models."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from cohr import judge
from cohr.config import load_prompt
from cohr.judge import (
    CachedJudge,
    CoverageAnswer,
    SupportAnswer,
    cache_key,
    judge_coverage,
    judge_support,
    parse_coverage,
    parse_support,
)


def replies(*texts: str) -> tuple[Callable[[str], str], list[str]]:
    """A fake generate that returns `texts` in turn and records every prompt."""
    prompts: list[str] = []
    it = iter(texts)

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        return next(it)

    return generate, prompts


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ('{"reason": "stated", "supported": "yes"}', "yes"),
        ('```json\n{"supported": "Partial", "reason": "x"}\n```', "partial"),
        ('Sure! {"supported": " no ", "reason": "x"} Hope that helps.', "no"),
        ('{"supported": "maybe", "reason": "x"}', None),
        ('{"supported": true}', None),
        ('{"reason": "x"}', None),
        ("yes", None),
        ('{"supported": "yes", "reason": ', None),
        ('["yes"]', None),
    ],
)
def test_parse_support(text: str, label: str | None) -> None:
    answer = parse_support(text)
    assert (answer.supported if answer else None) == label


def test_parse_support_keeps_reason_and_tolerates_missing_one() -> None:
    assert parse_support('{"reason": " ok ", "supported": "yes"}') == SupportAnswer(
        supported="yes", reason="ok"
    )
    assert parse_support('{"supported": "no"}') == SupportAnswer(supported="no", reason="")


@pytest.mark.parametrize(
    ("text", "covered"),
    [
        ('{"reason": "x", "covered": true}', True),
        ('{"covered": false}', False),
        ('{"covered": "True"}', True),
        ('{"covered": "yes"}', None),
        ('{"covered": 1}', None),
        ("no JSON", None),
    ],
)
def test_parse_coverage(text: str, covered: bool | None) -> None:
    answer = parse_coverage(text)
    assert (answer.covered if answer else None) == covered


def test_prompts_fill_values_once() -> None:
    prompt = judge.support_prompt("Revenue is {cited}.", "The text says {claim}.")
    assert "Revenue is {cited}." in prompt and "The text says {claim}." in prompt
    assert '{"reason": "one short sentence"' in prompt  # literal JSON braces survive
    listed = judge.coverage_prompt("keep the checklist", ["Die Checkliste bleibt.", "Andere."])
    assert "- Die Checkliste bleibt.\n- Andere." in listed


def test_invalid_json_is_retried_once_with_the_retry_instruction() -> None:
    generate, prompts = replies("not json", '{"supported": "partial", "reason": "r"}')
    assert judge_support(generate, "claim", "cited") == SupportAnswer(
        supported="partial", reason="r"
    )
    assert len(prompts) == 2
    assert prompts[1] == prompts[0] + "\n\n" + load_prompt("judge_retry")


def test_two_invalid_replies_are_unjudged_never_supported() -> None:
    generate, prompts = replies("nope", '{"supported": "probably"}')
    assert judge_support(generate, "c", "t").supported == "unjudged"
    assert len(prompts) == 2
    generate, _ = replies("nope", "still nope")
    assert judge_coverage(generate, "fact", ["a claim"]).covered is None


def test_coverage_with_no_claims_asks_nothing() -> None:
    generate, prompts = replies()
    assert judge_coverage(generate, "fact", []) == CoverageAnswer(
        covered=False, reason="no claims in the section"
    )
    assert prompts == []


def test_cache_key_depends_on_text_kind_and_revision() -> None:
    base = cache_key("support", "rev1", "claim", "cited")
    assert base == cache_key("support", "rev1", "claim", "cited")
    assert base != cache_key("support", "rev2", "claim", "cited")
    assert base != cache_key("coverage", "rev1", "claim", "cited")
    assert base != cache_key("support", "rev1", "claim", "cited ")
    assert cache_key("support", "r", "ab", "c") != cache_key("support", "r", "a", "bc")


def test_cache_key_changes_with_a_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = cache_key("support", "r", "c", "t")
    monkeypatch.setattr(judge, "prompt_hash", lambda name: f"edited-{name}")
    assert cache_key("support", "r", "c", "t") != before


def test_cached_judge_loads_the_model_only_on_a_miss(tmp_path: Path) -> None:
    loads: list[int] = []
    generate, prompts = replies(
        '{"supported": "yes", "reason": "a"}', '{"covered": true, "reason": "b"}'
    )

    def model() -> Callable[[str], str]:
        loads.append(1)
        return generate

    cj = CachedJudge("rev", model)
    cj.open(tmp_path / "item.json")
    assert not cj.loaded
    assert cj.support("c", "t").supported == "yes"
    assert cj.coverage("fact", ["claim"]).covered is True
    assert cj.support("c", "t").supported == "yes"  # same question: from the cache
    assert (cj.calls, cj.hits, len(loads), len(prompts)) == (2, 1, 1, 2)
    cj.save()

    # A second scoring with nothing stale never asks for the model.
    def no_model() -> Callable[[str], str]:
        raise AssertionError("the judge model was loaded")

    again = CachedJudge("rev", no_model)
    again.open(tmp_path / "item.json")
    assert again.support("c", "t") == SupportAnswer(supported="yes", reason="a")
    assert again.coverage("fact", ["claim"]) == CoverageAnswer(covered=True, reason="b")
    assert (again.calls, again.hits, again.loaded) == (0, 2, False)

    # Another revision is a miss.
    other = CachedJudge("rev2", no_model)
    other.open(tmp_path / "item.json")
    with pytest.raises(AssertionError, match="loaded"):
        other.support("c", "t")


def test_cache_files_are_per_item_and_unjudged_answers_are_kept(tmp_path: Path) -> None:
    generate, _ = replies("bad", "bad", '{"supported": "no", "reason": "x"}')
    cj = CachedJudge("rev", lambda: generate)
    cj.open(tmp_path / "a.json")
    assert cj.support("c", "t").supported == "unjudged"
    cj.open(tmp_path / "b.json")  # saves a.json
    assert cj.support("c", "t").supported == "no"  # b's cache doesn't have it
    cj.save()
    a = json.loads((tmp_path / "a.json").read_text())
    b = json.loads((tmp_path / "b.json").read_text())
    assert [v["supported"] for v in a.values()] == ["unjudged"]
    assert [v["supported"] for v in b.values()] == ["no"]


def test_a_corrupt_cache_entry_is_asked_again(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    path.write_text(json.dumps({cache_key("support", "rev", "c", "t"): {"supported": "?"}}))
    generate, prompts = replies('{"supported": "yes", "reason": ""}')
    cj = CachedJudge("rev", lambda: generate)
    cj.open(path)
    assert cj.support("c", "t").supported == "yes" and len(prompts) == 1


def test_mlx_is_imported_only_inside_functions() -> None:
    source = Path(judge.__file__).read_text(encoding="utf-8")
    top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert top and not any("mlx" in line for line in top)
