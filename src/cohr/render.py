"""Render (PLAN §5.6): notes.json → notes.md, and the in-app markdown for the Notes page.

Section headings come from a static table in the target language (English when missing); the
model's own headings are never used. Two modes from the same notes:
- portable (notes.md, Download): plain `[12:47]` / `[slide 4]` citations and a Sources appendix;
- app (Notes page, §7): `[12:47](#seg-3)` / `[slide 4](#slide-3)` links to the Sources panel, in
  Streamlit markdown: `$` escaped (it would start LaTeX), gray text + gray badge for uncited
  claims, and a count badge in the section's semantic color (§7) on Decisions, Action items
  and Open questions;
- chat (Ask answers, §5.7): Streamlit markdown like app, but citations are plain `[12:47]`
  labels, because the Ask page has no Sources panel to link to (its pills open the drawer).
Uncited claims are kept and flagged in every mode.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from cohr import atomic, schemas
from cohr.mapper import clock
from cohr.pipeline import Stage, StageContext
from cohr.reduce import SECTIONS, slide_label
from cohr.schemas import (
    ActionItem,
    Citation,
    Claim,
    NotesDoc,
    Segment,
    SegmentsFile,
    Slide,
    SlidesFile,
    TranslatedNotes,
)

FORMAT_VERSION = 2  # of notes.md; bump when the rendering changes (2: RTL isolates, 6d-1)

Notes = NotesDoc | TranslatedNotes  # what render shows: the same sections and citations
FILENAME = "notes.md"
EXCERPT_CHARS = 200

Mode = Literal["portable", "app", "chat"]

_KEYS = (*SECTIONS, "sources", "owner", "due", "uncited")
_LABELS = {
    "ar": (
        "الملخص",
        "النقاط الرئيسية",
        "القرارات",
        "المهام المطلوبة",
        "أسئلة مفتوحة",
        "المصادر",
        "المسؤول",
        "الموعد",
        "بلا مصدر",
    ),
    "cs": (
        "Shrnutí",
        "Hlavní body",
        "Rozhodnutí",
        "Úkoly",
        "Otevřené otázky",
        "Zdroje",
        "odpovídá",
        "termín",
        "bez zdroje",
    ),
    "de": (
        "Zusammenfassung",
        "Kernpunkte",
        "Entscheidungen",
        "Aufgaben",
        "Offene Fragen",
        "Quellen",
        "verantwortlich",
        "fällig",
        "ohne Quelle",
    ),
    "el": (
        "Περίληψη",
        "Βασικά σημεία",
        "Αποφάσεις",
        "Ενέργειες",
        "Ανοιχτά ερωτήματα",
        "Πηγές",
        "υπεύθυνος",
        "προθεσμία",
        "χωρίς πηγή",
    ),
    "en": (
        "Summary",
        "Key points",
        "Decisions",
        "Action items",
        "Open questions",
        "Sources",
        "owner",
        "due",
        "uncited",
    ),
    "es": (
        "Resumen",
        "Puntos clave",
        "Decisiones",
        "Tareas",
        "Preguntas abiertas",
        "Fuentes",
        "responsable",
        "fecha",
        "sin fuente",
    ),
    "fa": (
        "خلاصه",
        "نکات کلیدی",
        "تصمیم‌ها",
        "اقدامات",
        "پرسش‌های باز",
        "منابع",
        "مسئول",
        "مهلت",
        "بدون منبع",
    ),
    "fr": (
        "Résumé",
        "Points clés",
        "Décisions",
        "Actions",
        "Questions ouvertes",
        "Sources",
        "responsable",
        "échéance",
        "sans source",
    ),
    "he": (
        "סיכום",
        "נקודות עיקריות",
        "החלטות",
        "משימות",
        "שאלות פתוחות",
        "מקורות",
        "אחראי",
        "מועד",
        "ללא מקור",
    ),
    "hi": (
        "सारांश",
        "मुख्य बिंदु",
        "निर्णय",
        "कार्य",
        "खुले प्रश्न",
        "स्रोत",
        "ज़िम्मेदार",
        "समय-सीमा",
        "बिना स्रोत",
    ),
    "id": (
        "Ringkasan",
        "Poin utama",
        "Keputusan",
        "Tindak lanjut",
        "Pertanyaan terbuka",
        "Sumber",
        "penanggung jawab",
        "tenggat",
        "tanpa sumber",
    ),
    "it": (
        "Riepilogo",
        "Punti chiave",
        "Decisioni",
        "Azioni",
        "Domande aperte",
        "Fonti",
        "responsabile",
        "scadenza",
        "senza fonte",
    ),
    "ja": (
        "概要",
        "要点",
        "決定事項",
        "アクションアイテム",
        "未解決の質問",
        "出典",
        "担当",
        "期限",
        "出典なし",
    ),
    "ko": (
        "요약",
        "핵심 사항",
        "결정 사항",
        "실행 항목",
        "미해결 질문",
        "출처",
        "담당",
        "기한",
        "출처 없음",
    ),
    "nl": (
        "Samenvatting",
        "Belangrijkste punten",
        "Besluiten",
        "Actiepunten",
        "Open vragen",
        "Bronnen",
        "eigenaar",
        "deadline",
        "zonder bron",
    ),
    "pl": (
        "Podsumowanie",
        "Najważniejsze punkty",
        "Decyzje",
        "Zadania",
        "Otwarte pytania",
        "Źródła",
        "odpowiedzialny",
        "termin",
        "bez źródła",
    ),
    "pt": (
        "Resumo",
        "Pontos principais",
        "Decisões",
        "Ações",
        "Perguntas em aberto",
        "Fontes",
        "responsável",
        "prazo",
        "sem fonte",
    ),
    "ro": (
        "Rezumat",
        "Puncte cheie",
        "Decizii",
        "Acțiuni",
        "Întrebări deschise",
        "Surse",
        "responsabil",
        "termen",
        "fără sursă",
    ),
    "ru": (
        "Резюме",
        "Ключевые моменты",
        "Решения",
        "Задачи",
        "Открытые вопросы",
        "Источники",
        "ответственный",
        "срок",
        "без источника",
    ),
    "tr": (
        "Özet",
        "Önemli noktalar",
        "Kararlar",
        "Eylem maddeleri",
        "Açık sorular",
        "Kaynaklar",
        "sorumlu",
        "tarih",
        "kaynaksız",
    ),
    "uk": (
        "Підсумок",
        "Ключові моменти",
        "Рішення",
        "Завдання",
        "Відкриті питання",
        "Джерела",
        "відповідальний",
        "термін",
        "без джерела",
    ),
    "vi": (
        "Tóm tắt",
        "Điểm chính",
        "Quyết định",
        "Việc cần làm",
        "Câu hỏi mở",
        "Nguồn",
        "phụ trách",
        "hạn",
        "không có nguồn",
    ),
    "zh": ("摘要", "要点", "决定", "待办事项", "未决问题", "来源", "负责人", "截止", "无来源"),
}
# LABELS[lang][key]: the five section headings plus the words render adds itself.
LABELS: dict[str, dict[str, str]] = {
    lang: dict(zip(_KEYS, row, strict=True)) for lang, row in _LABELS.items()
}


def labels(language: str) -> dict[str, str]:
    return LABELS.get(language, LABELS["en"])


# Right-to-left notes languages (6d-1). Streamlit's markdown and most Markdown viewers lay a
# paragraph out left to right, so a line mixing Arabic with Latin names, dates or citations
# came out reordered: "(owner: Tom; due: Monday)" split the sentence and the citation ended up
# on the far side. Each line is wrapped in a right-to-left isolate (RLI … PDI, Unicode bidi):
# the reading order is right; the bullet and the alignment stay on the left (no custom CSS).
RTL_LANGUAGES = {"ar", "fa", "he", "ur"}
RLI, PDI = "\u2067", "\u2069"


def isolate(text: str, language: str) -> str:
    return f"{RLI}{text}{PDI}" if language in RTL_LANGUAGES else text


# Semantic colors on the Notes page (§7); the same names in light and dark.
SECTION_COLORS = {"decisions": "green", "action_items": "orange", "open_questions": "violet"}


def app_escape(text: str) -> str:
    """Model text inside Streamlit markdown: `$` starts LaTeX, brackets close a `:gray[…]`."""
    return text.replace("\\", "\\\\").replace("$", "\\$").replace("[", "\\[").replace("]", "\\]")


# --- markdown (pure) -----------------------------------------------------------------------


def citation_md(c: Citation, segments: Mapping[int, Segment], mode: Mode) -> str:
    if c.kind == "slide":
        text, anchor = slide_label(c.id), f"slide-{c.id}"
    else:
        t = c.t if c.t is not None else segments[c.id].t0
        text, anchor = clock(t), f"seg-{c.id}"
    if mode == "app":
        return f"[{text}](#{anchor})"
    return f"\\[{text}\\]" if mode == "chat" else f"[{text}]"


def claim_md(
    claim: Claim, segments: Mapping[int, Segment], words: dict[str, str], mode: Mode
) -> str:
    text = claim.text
    if isinstance(claim, ActionItem):
        extra = [f"{words[k]}: {v}" for k, v in (("owner", claim.owner), ("due", claim.due)) if v]
        if extra:
            text += f" ({'; '.join(extra)})"
    if mode != "portable":
        text = app_escape(text)
        if not claim.citations:
            return f":gray[{text}] :gray-badge[{words['uncited']}]"
    elif not claim.citations:
        return f"{text} [{words['uncited']}]"
    return f"{text} " + " ".join(citation_md(c, segments, mode) for c in claim.citations)


def answer_md(claims: Sequence[Claim], segments: Sequence[Segment], language: str) -> str:
    """An Ask answer as Streamlit markdown: one paragraph, uncited sentences flagged."""
    by_seg = {s.seg_id: s for s in segments}
    return isolate(
        " ".join(claim_md(c, by_seg, labels(language), "chat") for c in claims), language
    )


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return (cut.rsplit(" ", 1)[0] if " " in cut else cut) + " …"


def sources_md(
    notes: Notes, segments: Mapping[int, Segment], slides: Mapping[int, Slide]
) -> list[str]:
    """One entry per cited segment (time range, slides, transcript excerpt) and per cited slide."""
    cited = [
        c for name in SECTIONS for claim in getattr(notes.sections, name) for c in claim.citations
    ]
    seg_ids = sorted({c.id for c in cited if c.kind == "seg" and c.id in segments})
    slide_ids = sorted({c.id for c in cited if c.kind == "slide" and c.id in slides})
    lines = []
    for i in seg_ids:
        seg = segments[i]
        where = f"[{clock(seg.t0)}–{clock(seg.t1)}]"
        if seg.slide_ids:
            where += " · " + ", ".join(slide_label(s) for s in seg.slide_ids)
        lines.append(f"- {where}: {excerpt(seg.transcript_text)}")
    for i in slide_ids:
        title = slides[i].title or excerpt(slides[i].text_md, 80)
        lines.append(f"- [{slide_label(i)}]: {title}".rstrip(": "))
    return lines


def render(
    notes: Notes, segments: Sequence[Segment], slides: Sequence[Slide], mode: Mode = "portable"
) -> str:
    words = labels(notes.language)
    by_seg = {s.seg_id: s for s in segments}
    out: list[str] = []
    for name in SECTIONS:
        items = getattr(notes.sections, name)
        heading = f"## {words[name]}"
        if mode == "app" and items and name in SECTION_COLORS:
            heading += f" :{SECTION_COLORS[name]}-badge[{len(items)}]"
        out += [heading, ""]
        rendered = [claim_md(c, by_seg, words, mode) for c in items]
        if not rendered:
            out.append("—")
        elif name == "summary":
            out.append(isolate(" ".join(rendered), notes.language))
        else:
            out += [f"- {isolate(r, notes.language)}" for r in rendered]
        out.append("")
    if mode == "portable" and (
        sources := sources_md(notes, by_seg, {s.slide_id: s for s in slides})
    ):
        out += [f"## {words['sources']}", "", *sources, ""]
    return "\n".join(out).rstrip() + "\n"


def shown_notes(run_dir: Path) -> Notes:
    """The notes people read: notes_translated.json for an "Other (beta)" target (the translate
    stage deletes it otherwise), else notes.json."""
    if (run_dir / TranslatedNotes.FILENAME).is_file():
        return schemas.load(run_dir, TranslatedNotes)
    return schemas.load(run_dir, NotesDoc)


# --- stage ---------------------------------------------------------------------------------


def run_stage(ctx: StageContext) -> None:
    notes = shown_notes(ctx.run_dir)
    segments = schemas.load(ctx.run_dir, SegmentsFile).segments
    slides = schemas.load(ctx.run_dir, SlidesFile).slides
    atomic.write_text(ctx.run_dir / FILENAME, render(notes, segments, slides, "portable"))


STAGE = Stage(
    name="render",
    run=run_stage,
    outputs=(FILENAME,),
    format_version=FORMAT_VERSION,
)
