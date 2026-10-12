"""Meeting scripts with planted facts, from a seed (PLAN §18). Pure: no audio, no files.

A script is a list of spoken lines grouped into topics (one slide per topic), with N decisions,
M action items (owner + due date), one number per topic and one or two open questions planted on
known lines, plus filler talk that carries no facts. Facts come from small templates and word
lists, so the truth is exact. The same seed and variant always give the same script.

Numbers are kept apart so a value in the notes can only mean one fact: every number spoken or
shown (facts, previous values, due-date days, misspoken values) is distinct, and fact values
are never in 1–31, so they can't be confused with a date.

Long meetings (`long-en`, 6c) repeat the topic loop once per region, with more discussion per
topic. The region is in the spoken metric, the fact label and the slide title, so a topic that
comes back stays unambiguous; extra decisions, tasks and questions come from MORE_* lists used
only when a variant needs more than the quick lists hold, so quick scripts are unchanged.
"""

import random
from dataclasses import dataclass, field
from typing import Literal

FactKind = Literal["decision", "action", "number", "open_question"]
GapKind = Literal["silence", "noise", "music"]

VOICES = ("Samantha", "Daniel", "Karen")  # en_US, en_GB, en_AU (`say -v '?'`)
# One common spelling each: "Sofia" came back as "Sophia" and "Priya" as "Kreia" (4b-1 run 1),
# "Elena" as "Eleanor" (6b, 4 of 9 say/ASR round trips), and an owner is matched exactly.
OWNERS = ("Tom", "Maria", "Omar", "Kenji", "Nadia", "Victor", "Grace", "Laura")
PROJECTS = ("Atlas", "Borealis", "Cedar", "Harbor", "Juniper", "Meridian")


@dataclass(frozen=True)
class Topic:
    title: str
    metric: str  # as spoken: "the {metric} is …"
    label: str  # table row label on the slide
    unit: Literal["money", "percent", "count", "ms"]
    lo: int
    hi: int
    step: int
    bullets: tuple[str, str]
    period: str  # "That's up from … last {period}"


TOPICS = (
    Topic(
        "Quarterly Budget",
        "approved budget for next quarter",
        "Approved budget",
        "money",
        40_000,
        95_000,
        500,
        ("Spend plan by team", "User testing and onboarding"),
        "quarter",
    ),
    Topic(
        "Retention",
        "week two return rate for trial users",
        "Week-2 return rate",
        "percent",
        34,
        89,
        1,
        ("Trial users coming back", "Onboarding checklist impact"),
        "month",
    ),
    Topic(
        "Support Queue",
        "number of open support tickets",
        "Open tickets",
        "count",
        120,
        480,
        1,
        ("Ticket backlog", "Duplicates and stale tickets"),
        "week",
    ),
    Topic(
        "Trial Signups",
        "number of new trial signups this month",
        "New trial signups",
        "count",
        1_200,
        4_800,
        50,
        ("Signup funnel", "Campaign results"),
        "month",
    ),
    Topic(
        "Site Speed",
        "average page load time in milliseconds",
        "Page load (ms)",
        "ms",
        320,
        950,
        10,
        ("Performance work", "Image and script sizes"),
        "release",
    ),
    Topic(
        "Revenue",
        "monthly recurring revenue",
        "Monthly recurring revenue",
        "money",
        120_000,
        390_000,
        1_000,
        ("Revenue trend", "Expansion and churn"),
        "month",
    ),
    Topic(
        "Hiring",
        "number of candidates in the pipeline",
        "Candidates in pipeline",
        "count",
        32,
        99,
        1,
        ("Open roles", "Interview capacity"),
        "month",
    ),
    Topic(
        "Cloud Costs",
        "monthly cloud bill",
        "Monthly cloud bill",
        "money",
        6_000,
        12_000,
        100,
        ("Compute and storage", "Reserved capacity"),
        "month",
    ),
)

DECISIONS = (
    "move the weekly sync to Thursdays",
    "switch the payment provider to Northwind",
    "drop the legacy export feature",
    "keep the onboarding checklist",
    "freeze new features until the beta ships",
    "hire a contract designer for the redesign",
    "use the new logo on the website",
    "publish the roadmap to customers",
    "retire the old mobile app",
)
# Long meetings only (more than the lists above hold). Every planted text in all lists differs
# from every other by more than the matcher's 0.6 token overlap (test).
MORE_DECISIONS = (
    "adopt quarterly planning cycles",
    "close the Denver office",
    "standardize on Postgres for analytics",
    "cancel the trade show booth",
    "require two reviewers for every merge",
    "sunset the free tier",
    "rebuild the help center search",
    "raise the referral bonus",
    "pause the podcast series",
    "build an engineering hub in Lisbon",
    "switch the newsletter to monthly issues",
    "replace the intranet wiki with Notion",
    "introduce annual billing discounts",
    "consolidate the design tools",
)
DECISION_LINES = (
    "We decided to {d}.",
    "OK, so the decision is to {d}.",
    "We agreed to {d}.",
)

TASKS = (
    "send the contract to legal",
    "update the pricing page",
    "book the venue for the offsite",
    "write the release notes",
    "prepare the customer survey",
    "fix the login timeout bug",
    "draft the hiring plan",
    "review the security audit",
    "record the onboarding video",
)
MORE_TASKS = (
    "migrate the staging database",
    "reconcile the vendor invoices",
    "schedule the board presentation",
    "clean up the shared drive",
    "interview the shortlisted engineers",
    "renew the domain certificates",
    "finish the incident postmortem",
    "order laptops for the new starters",
    "configure the error alerts",
    "proofread the marketing emails",
    "collect quotes from moving companies",
    "test the backup restore",
    "organize the team lunch",
    "summarize the competitor research",
)
# Asked and left unresolved: what Reduce's Open questions section should hold (6b). No numbers,
# and no words shared with the decisions or tasks, so a match can only mean this question.
OPEN_QUESTIONS = (
    "partner with a reseller in Japan",
    "charge extra for the premium reports",
    "offer a tablet version",
    "outsource the translation work",
    "rename the company blog",
    "run the user conference online",
    "sponsor the local hackathon",
)
MORE_OPEN_QUESTIONS = (
    "allow pets at work",
    "join the industry alliance",
    "accept bitcoin",
    "add a weekend support shift",
    "disclose salary bands internally",
    "try a shorter work week",
)
OPEN_QUESTION_LINES = (
    "One thing we haven't settled: should we {q}?",
    "We still don't know whether we should {q}. Let's leave that open.",
    "Open question for next time: do we {q}? Nobody has an answer yet.",
)
ACTION_LINES = (
    "{owner} will {task} by {due}.",
    "Action item for {owner}: {task} by {due}.",
    "{owner}, can you {task} by {due}?",  # followed by a short yes from another voice
)
WEEKDAYS = ("Monday", "Wednesday", "Friday")
RELATIVE_DUES = ("the end of next week", "the end of the month")
MONTHS = ("October", "November")
ORDINALS = {
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
    21: "twenty first",
    22: "twenty second",
    23: "twenty third",
    24: "twenty fourth",
    25: "twenty fifth",
    26: "twenty sixth",
    27: "twenty seventh",
    28: "twenty eighth",
}

OPENING = (
    "Good morning everyone, this is the weekly sync for the {project} project.",
    "Morning. Can everyone see my screen?",
    "Yes, we can see it.",
)
FILLER = (
    "Sorry, I was on mute for a second.",
    "Thanks, that's helpful.",
    "Good point.",
    "Let me pull up the next slide.",
    "Can you say that again? The audio cut out.",
    "I think that covers it for this topic.",
    "Makes sense to me.",
    "Let me check my notes quickly.",
    "OK, sounds good.",
    "Right, moving on.",
)
# Talk about a topic that carries no decision, action item or number.
DISCUSSION = (
    "I looked at the {topic} figures in more detail yesterday, and the trend is roughly what "
    "we expected.",
    "A few people asked me about {topic} this week, so it's good that we're covering it.",
    "Honestly, {topic} has been more stable than I thought it would be a month ago.",
    "The dashboard for {topic} was updated on Monday, so these are fresh numbers.",
    "When we talked to customers, {topic} came up a couple of times, mostly in passing.",
    "There's some seasonality in {topic}, so I wouldn't read too much into one period.",
    "For context, the way we measure {topic} hasn't changed since the spring.",
)
# Long meetings only: more talk per topic than DISCUSSION holds. Statements about the past or
# the data only: no intention, request, offer or question, which Map would list as a decision,
# action item or open question (6c: "I'd like us to keep an eye on …" became 36 decisions).
MORE_DISCUSSION = (
    "I spent some time with the {topic} numbers, and they look about the way they did before.",
    "Nothing about {topic} surprised me this time, which is a nice change.",
    "We had a long thread about {topic} in the chat, but it mostly went in circles.",
    "The way {topic} is reported is a bit confusing, and I keep having to explain it.",
    "Some of the context on {topic} lives in old documents that nobody reads anymore.",
    "Overall, {topic} has been steady.",
    "I checked {topic} against what we heard from the field, and it lines up.",
    "I don't have much to add on {topic} beyond what's on the slide.",
    "My impression is that {topic} moves slowly, so small changes don't mean much.",
    "The {topic} chart on this slide looks like the chart we used last time.",
    "People in the field mention {topic} now and then, but not often.",
    "The {topic} figures come from the same source as before.",
    "I remember when {topic} was much harder to read than it is now.",
)
YES = ("Sure, I can do that.", "Yes, no problem.", "OK, will do.")
CLOSING = ("That's everything for today. Thanks, everyone.", "Thanks, bye.")
TOPIC_INTROS = ("Let's start with {topic}.", "Next topic is {topic}.", "Let's move on to {topic}.")

# Slide-conflict variant: a teen value on the slide, spoken as its "-ty" twin (14 ↔ 40).
TEEN_TO_TY = {13: 30, 14: 40, 15: 50, 16: 60, 17: 70, 18: 80, 19: 90}


@dataclass(frozen=True)
class Variant:
    """What a quick/full item changes about the meeting (evals/sets/*.yaml)."""

    topics: int = 8
    decisions: int = 4
    actions: int = 5
    gaps: tuple[GapKind, ...] = ("silence",)  # one gap per entry, after topics 1, 2, …
    gap_s: float = 12.0
    conflict: bool = False  # one money fact misspoken in speech, correct on the slide
    open_questions: tuple[int, int] = (1, 2)  # how many are planted: random in this range
    regions: tuple[str, ...] = ()  # long meetings: the topic loop once per region (6c)
    discussion: int = 4  # discussion lines per topic


@dataclass(frozen=True)
class PlannedFact:
    id: str
    kind: FactKind
    text: str  # decisions: the decision; actions: the task; numbers: "label: value";
    # open questions: what's asked ("open the beta to customers in Europe")
    line: int  # index into Script.lines where it's spoken
    topic: int
    owner: str | None = None
    due: str | None = None  # canonical: "Friday", "end of next week", "October 20"
    value: float | None = None
    misspoken: float | None = None  # conflict variant: the value said aloud
    on_slide: bool = False


@dataclass(frozen=True)
class Line:
    voice: str
    text: str
    topic: int | None  # None for opening/closing talk


@dataclass(frozen=True)
class SlideSpec:
    title: str
    bullets: tuple[str, ...]
    table: tuple[tuple[str, ...], ...]


@dataclass
class Script:
    project: str
    lines: list[Line] = field(default_factory=list)
    facts: list[PlannedFact] = field(default_factory=list)
    slides: list[SlideSpec] = field(default_factory=list)
    gaps_after: dict[int, GapKind] = field(default_factory=dict)  # line index → gap after it


def money(v: int) -> str:
    return f"${v:,}"


def display(topic: Topic, v: int) -> str:
    """How a value is written in the script (say reads it aloud) and on the slide."""
    if topic.unit == "money":
        return money(v)
    if topic.unit == "percent":
        return f"{v}%"
    return f"{v:,}"


def spoken(topic: Topic, v: int) -> str:
    """Written so `say` reads it as intended ("38 percent", not "38 per cent sign")."""
    if topic.unit == "percent":
        return f"{v} percent"
    if topic.unit == "ms":
        return f"{v:,} milliseconds"
    return display(topic, v)


class _Picker:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.used: set[int] = set(range(1, 32))  # fact numbers never look like a day

    def value(self, lo: int, hi: int, step: int) -> int:
        while True:
            v = self.rng.randrange(lo, hi + 1, step)
            if v not in self.used:
                self.used.add(v)
                return v

    def due(self) -> tuple[str, str]:
        """(canonical, spoken)."""
        kind = self.rng.randrange(3)
        if kind == 0:
            day = self.rng.choice(WEEKDAYS)
            return day, day
        if kind == 1:
            rel = self.rng.choice(RELATIVE_DUES)
            return rel.removeprefix("the "), rel
        month = self.rng.choice(MONTHS)
        day = self.rng.choice(sorted(ORDINALS))
        return f"{month} {day}", f"the {ORDINALS[day]} of {month}"


def _pool(base: tuple[str, ...], more: tuple[str, ...], n: int) -> tuple[str, ...]:
    """The quick list, or with the long-meeting extras when it holds fewer than n."""
    return base if n <= len(base) else base + more


def _discussion(rng: random.Random, n: int) -> list[str]:
    """n discussion lines: a sample of DISCUSSION (quick), or shuffled rounds of every line."""
    if n <= len(DISCUSSION):
        return rng.sample(DISCUSSION, n)
    pool = DISCUSSION + MORE_DISCUSSION
    out: list[str] = []
    while len(out) < n:
        out += rng.sample(pool, len(pool))
    return out[:n]


def generate(seed: int, variant: Variant) -> Script:
    """The meeting for this seed and variant. Same inputs → same script."""
    p = _Picker(seed)
    rng = p.rng
    regions: tuple[str | None, ...] = variant.regions or (None,)
    topics: list[Topic] = []
    for _ in regions:
        topics += rng.sample(TOPICS, variant.topics)
    region = [r for r in regions for _ in range(variant.topics)]
    if variant.conflict and not any(t.unit == "money" for t in topics):
        topics[0] = next(t for t in TOPICS if t.unit == "money")
    conflict_topic = (
        next(k for k, t in enumerate(topics) if t.unit == "money") if variant.conflict else None
    )
    decisions = rng.sample(_pool(DECISIONS, MORE_DECISIONS, variant.decisions), variant.decisions)
    tasks = rng.sample(_pool(TASKS, MORE_TASKS, variant.actions), variant.actions)
    owners = (
        rng.sample(OWNERS, variant.actions)
        if variant.actions <= len(OWNERS)
        else [rng.choice(OWNERS) for _ in range(variant.actions)]
    )
    # Spread decisions and actions over the topics, at most two of each per topic.
    slots = [k for k in range(len(topics)) for _ in range(2)]
    decision_topic = sorted(rng.sample(slots, variant.decisions))
    action_topic = sorted(rng.sample(slots, variant.actions))

    # Open questions draw from their own generator, so planting them left the other facts as
    # they were before 6b.
    q_rng = random.Random(f"{seed}-open-questions")
    n_questions = q_rng.randint(*variant.open_questions)
    questions = q_rng.sample(_pool(OPEN_QUESTIONS, MORE_OPEN_QUESTIONS, n_questions), n_questions)
    question_topic = dict(
        zip(q_rng.sample(range(len(topics)), n_questions), questions, strict=True)
    )

    script = Script(project=rng.choice(PROJECTS))
    last_voice = ""

    def say(text: str, topic: int | None, voice: str | None = None) -> int:
        nonlocal last_voice
        voice = voice or rng.choice([v for v in VOICES if v != last_voice])
        last_voice = voice
        script.lines.append(Line(voice, text, topic))
        return len(script.lines) - 1

    def next_id() -> str:
        return f"f{len(script.facts) + 1}"

    for i, text in enumerate(OPENING):
        say(text.format(project=script.project), None, VOICES[i % len(VOICES)])

    gaps = list(variant.gaps)
    for k, topic in enumerate(topics):
        where = f" for the {region[k]} region" if region[k] else ""
        intro = TOPIC_INTROS[0] if k == 0 else rng.choice(TOPIC_INTROS[1:])
        say(intro.format(topic=topic.title.lower() + where), k)

        if k == conflict_topic:
            teen = rng.choice(sorted(TEEN_TO_TY))
            value, said = teen * 1000, TEEN_TO_TY[teen] * 1000
            p.used |= {value, said}
        else:
            value = p.value(topic.lo, topic.hi, topic.step)
            said = value
        prev = p.value(topic.lo, topic.hi, topic.step)
        line = say(f"The {topic.metric}{where} is {spoken(topic, said)}.", k)
        label = f"{topic.label} ({region[k]})" if region[k] else topic.label
        script.facts.append(
            PlannedFact(
                id=next_id(),
                kind="number",
                text=f"{label}: {display(topic, value)}",
                line=line,
                topic=k,
                value=float(value),
                misspoken=float(said) if said != value else None,
                on_slide=True,
            )
        )
        say(f"That's compared with {spoken(topic, prev)} last {topic.period}.", k)
        for text in _discussion(rng, variant.discussion):
            say(text.format(topic=topic.title.lower()), k)
        if k in question_topic:
            q = question_topic[k]
            voice = q_rng.choice([v for v in VOICES if v != last_voice])
            line = say(q_rng.choice(OPEN_QUESTION_LINES).format(q=q), k, voice)
            script.facts.append(PlannedFact(next_id(), "open_question", q, line, k))

        # ("decision", index) and ("action", index) planted in this topic, in random order.
        items = [("decision", i) for i, t in enumerate(decision_topic) if t == k]
        items += [("action", j) for j, t in enumerate(action_topic) if t == k]
        rng.shuffle(items)
        for what, i in items:
            if what == "decision":
                line = say(rng.choice(DECISION_LINES).format(d=decisions[i]), k)
                script.facts.append(PlannedFact(next_id(), "decision", decisions[i], line, k))
            else:
                due, due_said = p.due()
                template = rng.choice(ACTION_LINES)
                line = say(template.format(owner=owners[i], task=tasks[i], due=due_said), k)
                script.facts.append(
                    PlannedFact(next_id(), "action", tasks[i], line, k, owner=owners[i], due=due)
                )
                if template.endswith("?"):
                    say(rng.choice(YES), k)
            say(rng.choice(FILLER), k)

        script.slides.append(
            SlideSpec(
                title=f"{topic.title} ({region[k]})" if region[k] else topic.title,
                bullets=topic.bullets,
                table=(
                    ("Metric", "Value"),
                    (topic.label, display(topic, value)),
                    (f"Last {topic.period}", display(topic, prev)),
                ),
            )
        )
        if gaps and k < len(topics) - 1:
            script.gaps_after[len(script.lines) - 1] = gaps.pop(0)

    for text in CLOSING:
        say(text, None)
    return script
