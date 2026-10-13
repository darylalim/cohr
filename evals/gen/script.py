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

Meetings in Spanish and Japanese (6d-2) use the same plan with that language's tables (`LANGS`):
the random draws are the same as in English, only the words change. Owners keep their Latin
names, as people's names do in real meetings. Numbers are written as plain digits ("42500
dólares"), which `say` reads the same way in every locale, so how they come back (digits or
words, which separators) is up to Transcribe. Due dates stay English in the truth ("Friday",
"October 20"): cross-language scoring compares only their digits. Long meetings are English
only (the es/ja tables have no MORE_* lists).
"""

import random
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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


# --- languages (6d-2) ----------------------------------------------------------------------


@dataclass(frozen=True)
class Lang:
    """Everything a meeting says or shows, in one language. Lists line up with the English
    ones (same length, same order: TOPICS[k] is topics[k], WEEKDAYS[i] is weekdays[i]), so the
    same seed plants the same facts in every language."""

    code: str
    voices: tuple[str, str, str]
    topics: tuple[Topic, ...]
    decisions: tuple[str, ...]
    decision_lines: tuple[str, ...]
    tasks: tuple[str, ...]
    action_lines: tuple[str, ...]  # a line ending in a question gets a short yes after it
    open_questions: tuple[str, ...]
    open_question_lines: tuple[str, ...]
    weekdays: tuple[str, ...]  # as said after "by"
    relative_dues: tuple[str, ...]
    date: Callable[[int, int], str]  # (MONTHS index, day) → as said
    opening: tuple[str, ...]
    filler: tuple[str, ...]
    discussion: tuple[str, ...]
    yes: tuple[str, ...]
    closing: tuple[str, ...]
    topic_intros: tuple[str, ...]
    number_line: str  # {metric} {where} {value}
    prev_line: str  # {value} {period}
    where: str  # long meetings: {region}
    table_header: tuple[str, str]
    last_row: Callable[[str], str]  # period → the slide's previous-value row label
    display: Callable[[Topic, int], str]  # as written on the slide
    spoken: Callable[[Topic, int], str]  # as written in the script, for `say`
    more_decisions: tuple[str, ...] = ()
    more_tasks: tuple[str, ...] = ()
    more_open_questions: tuple[str, ...] = ()
    more_discussion: tuple[str, ...] = ()


def _en_date(month: int, day: int) -> str:
    return f"the {ORDINALS[day]} of {MONTHS[month]}"


EN = Lang(
    code="en",
    voices=VOICES,
    topics=TOPICS,
    decisions=DECISIONS,
    decision_lines=DECISION_LINES,
    tasks=TASKS,
    action_lines=ACTION_LINES,
    open_questions=OPEN_QUESTIONS,
    open_question_lines=OPEN_QUESTION_LINES,
    weekdays=WEEKDAYS,
    relative_dues=RELATIVE_DUES,
    date=_en_date,
    opening=OPENING,
    filler=FILLER,
    discussion=DISCUSSION,
    yes=YES,
    closing=CLOSING,
    topic_intros=TOPIC_INTROS,
    number_line="The {metric}{where} is {value}.",
    prev_line="That's compared with {value} last {period}.",
    where=" for the {region} region",
    table_header=("Metric", "Value"),
    last_row=lambda period: f"Last {period}",
    display=display,
    spoken=spoken,
    more_decisions=MORE_DECISIONS,
    more_tasks=MORE_TASKS,
    more_open_questions=MORE_OPEN_QUESTIONS,
    more_discussion=MORE_DISCUSSION,
)


def _localize(*texts: tuple[str, str, str, tuple[str, str], str]) -> tuple[Topic, ...]:
    """TOPICS with (title, metric, label, bullets, period) in another language."""
    return tuple(
        replace(t, title=a, metric=b, label=c, bullets=d, period=e)
        for t, (a, b, c, d, e) in zip(TOPICS, texts, strict=True)
    )


# Spanish (Spain style on the slides: "42.500 $", "34 %"). Day numbers said as words, like
# the English ordinals: "el veinte de octubre".
_ES_DAYS = {
    2: "dos",
    3: "tres",
    4: "cuatro",
    5: "cinco",
    6: "seis",
    7: "siete",
    8: "ocho",
    9: "nueve",
    10: "diez",
    11: "once",
    12: "doce",
    13: "trece",
    14: "catorce",
    15: "quince",
    16: "dieciséis",
    17: "diecisiete",
    18: "dieciocho",
    19: "diecinueve",
    20: "veinte",
    21: "veintiuno",
    22: "veintidós",
    23: "veintitrés",
    24: "veinticuatro",
    25: "veinticinco",
    26: "veintiséis",
    27: "veintisiete",
    28: "veintiocho",
}
_ES_MONTHS = ("octubre", "noviembre")


def _es_group(v: int) -> str:
    return f"{v:,}".replace(",", ".")


def _es_display(topic: Topic, v: int) -> str:
    if topic.unit == "money":
        return f"{_es_group(v)} $"
    if topic.unit == "percent":
        return f"{v} %"
    return _es_group(v)


def _es_spoken(topic: Topic, v: int) -> str:
    unit = {"money": " dólares", "percent": " por ciento", "ms": " milisegundos"}
    return f"{v}{unit.get(topic.unit, '')}"


ES = Lang(
    code="es",
    voices=("Mónica", "Paulina", "Eddy (Spanish (Spain))"),  # es_ES, es_MX, es_ES
    topics=_localize(
        (
            "Presupuesto trimestral",
            "El presupuesto aprobado para el próximo trimestre",
            "Presupuesto aprobado",
            ("Gasto por equipo", "Pruebas con usuarios"),
            "el trimestre pasado",
        ),
        (
            "Retención",
            "La tasa de retorno en la segunda semana de los usuarios de prueba",
            "Retorno semana 2",
            ("Usuarios de prueba que vuelven", "Efecto de la lista de bienvenida"),
            "el mes pasado",
        ),
        (
            "Cola de soporte",
            "El número de tickets de soporte abiertos",
            "Tickets abiertos",
            ("Tickets pendientes", "Duplicados y tickets antiguos"),
            "la semana pasada",
        ),
        (
            "Registros de prueba",
            "El número de registros de prueba nuevos este mes",
            "Registros de prueba nuevos",
            ("Embudo de registro", "Resultados de la campaña"),
            "el mes pasado",
        ),
        (
            "Velocidad del sitio",
            "El tiempo medio de carga de la página en milisegundos",
            "Carga de página (ms)",
            ("Trabajo de rendimiento", "Tamaño de imágenes y scripts"),
            "en la versión anterior",
        ),
        (
            "Ingresos",
            "El ingreso recurrente mensual",
            "Ingreso recurrente mensual",
            ("Tendencia de ingresos", "Expansión y bajas"),
            "el mes pasado",
        ),
        (
            "Contratación",
            "El número de candidatos en el proceso",
            "Candidatos en proceso",
            ("Puestos abiertos", "Capacidad de entrevistas"),
            "el mes pasado",
        ),
        (
            "Costes de nube",
            "La factura mensual de la nube",
            "Factura mensual de nube",
            ("Cómputo y almacenamiento", "Capacidad reservada"),
            "el mes pasado",
        ),
    ),
    decisions=(
        "pasar la reunión semanal a los jueves",
        "cambiar el proveedor de pagos a Northwind",
        "eliminar la función de exportación antigua",
        "mantener la lista de bienvenida",
        "congelar las funciones nuevas hasta que salga la beta",
        "contratar a un diseñador externo para el rediseño",
        "usar el logotipo nuevo en la web",
        "publicar la hoja de ruta para los clientes",
        "retirar la aplicación móvil vieja",
    ),
    decision_lines=("Decidimos {d}.", "Bien, la decisión es {d}.", "Acordamos {d}."),
    tasks=(
        "enviar el contrato al departamento legal",
        "actualizar la página de precios",
        "reservar el local para el retiro",
        "escribir las notas de la versión",
        "preparar la encuesta a clientes",
        "corregir el fallo de tiempo de espera del inicio de sesión",
        "redactar el plan de contratación",
        "revisar la auditoría de seguridad",
        "grabar el vídeo de bienvenida",
    ),
    action_lines=(
        "{owner} va a {task} para {due}.",
        "Tarea para {owner}: {task} para {due}.",
        "{owner}, ¿puedes {task} para {due}?",
    ),
    open_questions=(
        "asociarnos con un revendedor en Japón",
        "cobrar aparte los informes premium",
        "ofrecer una versión para tabletas",
        "externalizar el trabajo de traducción",
        "cambiar el nombre del blog de la empresa",
        "hacer la conferencia de usuarios en línea",
        "patrocinar el hackatón local",
    ),
    open_question_lines=(
        "Algo que no hemos resuelto: ¿deberíamos {q}?",
        "Todavía no sabemos si deberíamos {q}. Dejémoslo abierto.",
        "Pregunta abierta para la próxima vez: ¿vamos a {q}? Nadie tiene la respuesta aún.",
    ),
    weekdays=("el lunes", "el miércoles", "el viernes"),
    relative_dues=("el final de la próxima semana", "el final del mes"),
    date=lambda month, day: f"el {_ES_DAYS[day]} de {_ES_MONTHS[month]}",
    opening=(
        "Buenos días a todos, esta es la reunión semanal del proyecto {project}.",
        "Buenos días. ¿Todos ven mi pantalla?",
        "Sí, la vemos.",
    ),
    filler=(
        "Perdón, tenía el micrófono silenciado.",
        "Gracias, eso ayuda.",
        "Buena observación.",
        "Voy a poner la siguiente diapositiva.",
        "¿Puedes repetirlo? Se cortó el audio.",
        "Creo que con esto cerramos este tema.",
        "Me parece bien.",
        "Déjame revisar mis notas un momento.",
        "Vale, de acuerdo.",
        "Bien, sigamos.",
    ),
    discussion=(
        "Ayer miré las cifras de {topic} con más detalle, y la tendencia es más o menos la "
        "que esperábamos.",
        "Varias personas me preguntaron por {topic} esta semana, así que está bien que lo veamos.",
        "La verdad es que {topic} ha estado más estable de lo que pensaba hace un mes.",
        "El panel de {topic} se actualizó el lunes, así que son datos recientes.",
        "Cuando hablamos con clientes, {topic} salió un par de veces, casi siempre de pasada.",
        "Hay algo de estacionalidad en {topic}, así que no le daría mucha importancia a un "
        "solo periodo.",
        "Como contexto, la forma de medir {topic} no ha cambiado desde la primavera.",
    ),
    yes=("Claro, me encargo.", "Sí, sin problema.", "Vale, lo hago."),
    closing=("Eso es todo por hoy. Gracias a todos.", "Gracias, adiós."),
    topic_intros=(
        "Empecemos con el tema {topic}.",
        "El siguiente tema es {topic}.",
        "Pasemos al tema {topic}.",
    ),
    number_line="{metric}{where} es de {value}.",
    prev_line="Eso se compara con {value} {period}.",
    where=" en la región {region}",
    table_header=("Métrica", "Valor"),
    last_row=lambda period: period[0].upper() + period[1:],
    display=_es_display,
    spoken=_es_spoken,
)


def _ja_display(topic: Topic, v: int) -> str:
    if topic.unit == "money":
        return f"{v:,}ドル"
    if topic.unit == "percent":
        return f"{v}%"
    return f"{v:,}"


def _ja_spoken(topic: Topic, v: int) -> str:
    unit = {"money": "ドル", "percent": "パーセント", "ms": "ミリ秒"}
    return f"{v}{unit.get(topic.unit, '')}"


# Japanese: task, decision and question texts end in the dictionary form, so every line
# template can attach its ending ("…ことに決めました", "…までに…予定です").
JA = Lang(
    code="ja",
    voices=("Kyoko", "Eddy (Japanese (Japan))", "Reed (Japanese (Japan))"),
    topics=_localize(
        (
            "四半期予算",
            "来四半期の承認済み予算",
            "承認済み予算",
            ("チーム別の支出計画", "ユーザーテストとオンボーディング"),
            "前四半期",
        ),
        (
            "リテンション",
            "トライアルユーザーの2週目の再訪率",
            "2週目の再訪率",
            ("戻ってくるトライアルユーザー", "オンボーディングチェックリストの効果"),
            "先月",
        ),
        (
            "サポート窓口",
            "未対応のサポートチケットの数",
            "未対応チケット",
            ("チケットの滞留", "重複と古いチケット"),
            "先週",
        ),
        (
            "トライアル登録",
            "今月の新規トライアル登録数",
            "新規トライアル登録",
            ("登録ファネル", "キャンペーンの結果"),
            "先月",
        ),
        (
            "サイトの速度",
            "ページの平均読み込み時間",
            "ページ読み込み (ms)",
            ("パフォーマンス改善", "画像とスクリプトのサイズ"),
            "前回のリリース",
        ),
        (
            "売上",
            "月間経常収益",
            "月間経常収益",
            ("売上の推移", "拡大と解約"),
            "先月",
        ),
        (
            "採用",
            "選考中の候補者の数",
            "選考中の候補者",
            ("募集中のポジション", "面接の対応枠"),
            "先月",
        ),
        (
            "クラウド費用",
            "クラウドの月額請求",
            "クラウド月額請求",
            ("計算資源とストレージ", "予約済みの容量"),
            "先月",
        ),
    ),
    decisions=(
        "週次定例を木曜日に移す",
        "決済代行をNorthwindに切り替える",
        "旧エクスポート機能を廃止する",
        "オンボーディングのチェックリストを残す",
        "ベータ版が出るまで新機能の開発を止める",
        "リデザインのために外部デザイナーと契約する",
        "ウェブサイトに新しいロゴを使う",
        "ロードマップを顧客に公開する",
        "古いモバイルアプリを終了する",
    ),
    decision_lines=(
        "{d}ことに決めました。",
        "では、{d}ということで決定です。",
        "{d}ことで合意しました。",
    ),
    tasks=(
        "契約書を法務部に送る",
        "料金ページを更新する",
        "合宿の会場を予約する",
        "リリース時の変更点をまとめる",
        "顧客アンケートを準備する",
        "ログインのタイムアウトの不具合を直す",
        "採用計画の案を作る",
        "セキュリティ監査の結果を確認する",
        "新人向けの紹介動画を撮影する",
    ),
    action_lines=(
        "{owner}さんが{due}までに{task}予定です。",
        "{owner}さんへのアクションアイテム：{due}までに{task}こと。",
        "{owner}さん、{due}までに{task}ことはできますか？",
    ),
    open_questions=(
        "日本の販売代理店と提携する",
        "プレミアムレポートを有料にする",
        "タブレット版を出す",
        "翻訳作業を外注する",
        "会社ブログの名前を変える",
        "ユーザーカンファレンスをオンラインで開く",
        "地元のハッカソンに協賛する",
    ),
    open_question_lines=(
        "まだ決まっていないことがあります。{q}べきでしょうか？",
        "{q}べきかどうかは、まだ分かりません。保留にしましょう。",
        "次回への宿題です。{q}かどうか、まだ誰も答えを持っていません。",
    ),
    weekdays=("月曜日", "水曜日", "金曜日"),
    relative_dues=("来週の終わり", "月末"),
    date=lambda month, day: f"{10 + month}月{day}日",
    opening=(
        "皆さん、おはようございます。{project}プロジェクトの週次定例を始めます。",
        "おはようございます。画面は見えていますか？",
        "はい、見えています。",
    ),
    filler=(
        "すみません、ミュートになっていました。",
        "ありがとうございます、助かります。",
        "いい指摘ですね。",
        "次のスライドを出します。",
        "もう一度言ってもらえますか？音声が途切れました。",
        "この話題はこれで大丈夫だと思います。",
        "それでいいと思います。",
        "ちょっとメモを確認します。",
        "はい、了解です。",
        "では、次に進みましょう。",
    ),
    discussion=(
        "昨日{topic}の数字を詳しく見ましたが、傾向はだいたい予想どおりでした。",
        "今週何人かから{topic}について聞かれたので、ここで取り上げられてよかったです。",
        "正直なところ、{topic}は一か月前に思っていたより安定しています。",
        "{topic}のダッシュボードは月曜日に更新されたので、これは新しい数字です。",
        "お客さんと話したとき、{topic}の話が何度か出ましたが、ほとんどは軽く触れただけでした。",
        "{topic}には季節性があるので、一つの期間だけで判断しないほうがいいと思います。",
        "補足すると、{topic}の測り方は春から変わっていません。",
    ),
    yes=("はい、やっておきます。", "はい、大丈夫です。", "了解です、対応します。"),
    closing=("今日は以上です。皆さん、ありがとうございました。", "ありがとうございました。"),
    topic_intros=(
        "まず{topic}から始めましょう。",
        "次の議題は{topic}です。",
        "では{topic}に移りましょう。",
    ),
    number_line="{where}{metric}は{value}です。",
    prev_line="{period}は{value}でした。",
    where="{region}地域の",
    table_header=("指標", "値"),
    last_row=lambda period: period,
    display=_ja_display,
    spoken=_ja_spoken,
)

LANGS = {lang.code: lang for lang in (EN, ES, JA)}


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
    language: str = "en"  # script language: a LANGS key (6d-2)


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

    def due(self, lang: Lang = EN) -> tuple[str, str]:
        """(canonical, spoken). Canonical is English in every language; the draws are the
        same, so a seed gives the same due dates in every language."""
        kind = self.rng.randrange(3)
        if kind == 0:
            i = self.rng.randrange(len(WEEKDAYS))
            return WEEKDAYS[i], lang.weekdays[i]
        if kind == 1:
            i = self.rng.randrange(len(RELATIVE_DUES))
            return RELATIVE_DUES[i].removeprefix("the "), lang.relative_dues[i]
        month = self.rng.randrange(len(MONTHS))
        day = self.rng.choice(sorted(ORDINALS))
        return f"{MONTHS[month]} {day}", lang.date(month, day)


def _pool(base: tuple[str, ...], more: tuple[str, ...], n: int) -> tuple[str, ...]:
    """The quick list, or with the long-meeting extras when it holds fewer than n."""
    return base if n <= len(base) else base + more


def _discussion(rng: random.Random, n: int, lang: Lang = EN) -> list[str]:
    """n discussion lines: a sample of DISCUSSION (quick), or shuffled rounds of every line."""
    if n <= len(lang.discussion):
        return rng.sample(lang.discussion, n)
    pool = lang.discussion + lang.more_discussion
    out: list[str] = []
    while len(out) < n:
        out += rng.sample(pool, len(pool))
    return out[:n]


def generate(seed: int, variant: Variant) -> Script:
    """The meeting for this seed and variant. Same inputs → same script."""
    lang = LANGS[variant.language]
    p = _Picker(seed)
    rng = p.rng
    regions: tuple[str | None, ...] = variant.regions or (None,)
    topics: list[Topic] = []
    for _ in regions:
        topics += rng.sample(lang.topics, variant.topics)
    region = [r for r in regions for _ in range(variant.topics)]
    if variant.conflict and not any(t.unit == "money" for t in topics):
        topics[0] = next(t for t in lang.topics if t.unit == "money")
    conflict_topic = (
        next(k for k, t in enumerate(topics) if t.unit == "money") if variant.conflict else None
    )
    decisions = rng.sample(
        _pool(lang.decisions, lang.more_decisions, variant.decisions), variant.decisions
    )
    tasks = rng.sample(_pool(lang.tasks, lang.more_tasks, variant.actions), variant.actions)
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
    questions = q_rng.sample(
        _pool(lang.open_questions, lang.more_open_questions, n_questions), n_questions
    )
    question_topic = dict(
        zip(q_rng.sample(range(len(topics)), n_questions), questions, strict=True)
    )

    script = Script(project=rng.choice(PROJECTS))
    last_voice = ""

    def say(text: str, topic: int | None, voice: str | None = None) -> int:
        nonlocal last_voice
        voice = voice or rng.choice([v for v in lang.voices if v != last_voice])
        last_voice = voice
        script.lines.append(Line(voice, text, topic))
        return len(script.lines) - 1

    def next_id() -> str:
        return f"f{len(script.facts) + 1}"

    for i, text in enumerate(lang.opening):
        say(text.format(project=script.project), None, lang.voices[i % len(lang.voices)])

    gaps = list(variant.gaps)
    for k, topic in enumerate(topics):
        where = lang.where.format(region=region[k]) if region[k] else ""
        intro = lang.topic_intros[0] if k == 0 else rng.choice(lang.topic_intros[1:])
        say(intro.format(topic=topic.title.lower() + where), k)

        if k == conflict_topic:
            teen = rng.choice(sorted(TEEN_TO_TY))
            value, said = teen * 1000, TEEN_TO_TY[teen] * 1000
            p.used |= {value, said}
        else:
            value = p.value(topic.lo, topic.hi, topic.step)
            said = value
        prev = p.value(topic.lo, topic.hi, topic.step)
        number = lang.number_line.format(
            metric=topic.metric, where=where, value=lang.spoken(topic, said)
        )
        line = say(number, k)
        label = f"{topic.label} ({region[k]})" if region[k] else topic.label
        script.facts.append(
            PlannedFact(
                id=next_id(),
                kind="number",
                text=f"{label}: {lang.display(topic, value)}",
                line=line,
                topic=k,
                value=float(value),
                misspoken=float(said) if said != value else None,
                on_slide=True,
            )
        )
        say(lang.prev_line.format(value=lang.spoken(topic, prev), period=topic.period), k)
        for text in _discussion(rng, variant.discussion, lang):
            say(text.format(topic=topic.title.lower()), k)
        if k in question_topic:
            q = question_topic[k]
            voice = q_rng.choice([v for v in lang.voices if v != last_voice])
            line = say(q_rng.choice(lang.open_question_lines).format(q=q), k, voice)
            script.facts.append(PlannedFact(next_id(), "open_question", q, line, k))

        # ("decision", index) and ("action", index) planted in this topic, in random order.
        items = [("decision", i) for i, t in enumerate(decision_topic) if t == k]
        items += [("action", j) for j, t in enumerate(action_topic) if t == k]
        rng.shuffle(items)
        for what, i in items:
            if what == "decision":
                line = say(rng.choice(lang.decision_lines).format(d=decisions[i]), k)
                script.facts.append(PlannedFact(next_id(), "decision", decisions[i], line, k))
            else:
                due, due_said = p.due(lang)
                template = rng.choice(lang.action_lines)
                line = say(template.format(owner=owners[i], task=tasks[i], due=due_said), k)
                script.facts.append(
                    PlannedFact(next_id(), "action", tasks[i], line, k, owner=owners[i], due=due)
                )
                if template.endswith(("?", "？")):
                    say(rng.choice(lang.yes), k)
            say(rng.choice(lang.filler), k)

        script.slides.append(
            SlideSpec(
                title=f"{topic.title} ({region[k]})" if region[k] else topic.title,
                bullets=topic.bullets,
                table=(
                    lang.table_header,
                    (topic.label, lang.display(topic, value)),
                    (lang.last_row(topic.period), lang.display(topic, prev)),
                ),
            )
        )
        if gaps and k < len(topics) - 1:
            script.gaps_after[len(script.lines) - 1] = gaps.pop(0)

    for text in lang.closing:
        say(text, None)
    return script
