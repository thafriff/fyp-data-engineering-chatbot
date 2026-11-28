"""
Study plan (feature A; docs/feature_design_2026-11-28.md, approved by the author on 2026-11-28; suggested by the author's lecturer).
APP LAYER ONLY: the frozen SYSTEM_PROMPT, the notebook 04 runners and single-turn chat are untouched.

"I have 3 hours to study data warehousing" / "plan 3 days for SQL" -> a time-boxed plan grounded in the knowledge base. CODE BUILDS
EVERYTHING STRUCTURAL:
- detection and parsing (detect): a planning phrase and a duration, or an explicit "study plan" request; minutes, hours, days or weeks
  (Malay jam / minit / hari / minggu too); days mode at PLAN_HOURS_PER_DAY unless the student says "N hours a day";
- coverage (build): the topic is searched (plus up to PLAN_MAX_EXPANSION_QUERIES phrases from TOPIC_DESCRIPTIONS when it names one of
  the 8 course topics), chunks at or above SIMILARITY_THRESHOLD are kept, and the plan needs PLAN_MIN_CHUNKS of them from
  PLAN_MIN_SECTIONS sections, otherwise it is declined;
- sub-topics are the retrieved chunks grouped by section and NAMED AFTER THE REAL SECTION TITLE, so no sub-topic can be invented;
- order: the source's difficulty label (beginner, intermediate, advanced), then the most relevant source first, then course order
  (chapter / section / module number), then score;
- time: 45-minute sessions with a 10-minute break (session_lengths), minutes per sub-topic from its amount of material; sub-topics that
  do not fit go under "If you have more time", and spare time becomes review;
- quiz checkpoints after every PLAN_QUIZ_EVERY_SESSIONS sessions (hours mode) or at the end of each day (days mode).
THE MODEL ONLY WRITES each session's one-sentence goal and its suggested questions (write_session_text). A suggested question is kept
only if, searched like a student message, its top 5 holds one of its session's chunks at or above the threshold (the chatbot would
answer it from the same material); otherwise "What is <sub-topic>?" is tried, else the question is dropped.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from config import settings
from src.core import llm
from src.core.retriever import RetrievedChunk

_DIFFICULTY = {"beginner": 0, "intermediate": 1, "advanced": 2}
_UNITS = sorted(settings.PLAN_UNITS, key=len, reverse=True)
_NUMBER = r"\d+(?:\.\d+)?|" + "|".join(settings.PLAN_NUMBER_WORDS)
_DURATION = re.compile(rf"\bhalf an hour\b|\b(?P<n>{_NUMBER})\s*(?P<unit>{'|'.join(map(re.escape, _UNITS))})\b", re.IGNORECASE)
_PER_DAY = re.compile(settings.PLAN_PER_DAY_PATTERN, re.IGNORECASE)
_EXPLICIT = [re.compile(p, re.IGNORECASE) for p in settings.PLAN_EXPLICIT_PATTERNS]
_REQUEST_WORDS = re.compile(settings.PLAN_REQUEST_WORDS, re.IGNORECASE)
_QUESTION_START = re.compile(settings.PLAN_QUESTION_START, re.IGNORECASE)
_FILLERS = [re.compile(rf"\b{re.escape(w)}\b", re.IGNORECASE) for w in sorted(settings.PLAN_FILLER_WORDS, key=len, reverse=True)]
_BROAD = [re.compile(p, re.IGNORECASE) for p in settings.PLAN_BROAD_PATTERNS]
_ID_IN_BRACKETS = re.compile(r"\s*\[[^\]]*\]")
_FIRST_NUMBER = re.compile(r"(\d+(?:\.\d+)?)")


@dataclass
class PlanRequest:
    """A parsed study-plan request. minutes is the whole budget (None: no time given)."""
    topic: str
    minutes: int | None
    mode: str = "hours"                     # "hours" or "days"
    days: int | None = None
    minutes_per_day: int | None = None
    per_day_given: bool = False


# ---------------------------------------------------------------------------
# Detection and parsing
# ---------------------------------------------------------------------------

def _number(text: str) -> float:
    text = text.lower()
    return float(settings.PLAN_NUMBER_WORDS[text]) if text in settings.PLAN_NUMBER_WORDS else float(text)


def detect(text: str) -> PlanRequest | None:
    """The parsed request when the (cleaned) message asks for a study plan, otherwise None."""
    t = re.sub(r"[\s.?!]+$", "", text.strip())
    explicit = any(p.search(t) for p in _EXPLICIT)
    if _QUESTION_START.match(t) and not re.search(r"\bplan\b", t, re.IGNORECASE):
        return None
    per_day = _PER_DAY.search(t)
    rest = (t[:per_day.start()] + " " + t[per_day.end():]) if per_day else t
    durations = list(_DURATION.finditer(rest))
    if not (explicit or (_REQUEST_WORDS.search(t) and durations)):
        return None
    minutes_per_day = round(_number(per_day.group("n")) * 60) if per_day else None
    total, mode, days = None, "hours", None
    if durations:
        total = 0.0
        for m in durations:
            if m.group(0).lower() == "half an hour":
                total += 30
                continue
            unit = settings.PLAN_UNITS[m.group("unit").lower()]
            if unit >= 1440:
                mode = "days"
            total += _number(m.group("n")) * unit
        if mode == "days":
            days = max(1, round(total / 1440))
            total = days * (minutes_per_day or settings.PLAN_HOURS_PER_DAY * 60)
    topic = _DURATION.sub(" ", rest)
    for filler in _FILLERS:
        topic = filler.sub(" ", topic)
    topic = re.sub(r"\s+", " ", re.sub(r"[,;:]", " ", topic)).strip(" -")
    return PlanRequest(topic=topic, minutes=None if total is None else round(total), mode=mode, days=days,
                       minutes_per_day=(minutes_per_day or settings.PLAN_HOURS_PER_DAY * 60) if mode == "days" else None,
                       per_day_given=per_day is not None)


def problem(request: PlanRequest) -> tuple[str, str] | None:
    """A fixed reply before any search: (kind, message) for a missing topic or time, a budget outside the limits, or the whole
    field. None when a plan can be built."""
    if not request.topic:
        return "ask_topic", settings.PLAN_ASK_TOPIC
    if request.minutes is None:
        return "ask_time", settings.PLAN_ASK_TIME
    if request.minutes < settings.PLAN_MIN_MINUTES:
        return "too_short", settings.PLAN_TOO_SHORT.format(minutes=request.minutes, topic=request.topic)
    too_many_days = request.mode == "days" and (request.days or 0) > settings.PLAN_MAX_DAYS
    if too_many_days or request.minutes > settings.PLAN_MAX_HOURS * 60:
        return "too_long", settings.PLAN_TOO_LONG.format(max_days=settings.PLAN_MAX_DAYS, max_hours=settings.PLAN_MAX_HOURS,
                                                         topic=request.topic)
    if any(p.search(request.topic) for p in _BROAD):
        overview = "\n".join(f"- **{name}**: {desc.split(':')[0]}" for name, desc in settings.TOPIC_DESCRIPTIONS.items())
        return "broad", settings.PLAN_BROAD.format(minutes=request.minutes, overview=overview)
    return None


# ---------------------------------------------------------------------------
# Retrieval, sub-topics and order
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words} - {"and", "or", "the"}


def course_topic(topic: str) -> str | None:
    """The course topic (a TOPIC_DESCRIPTIONS key) the request names exactly, e.g. "etl and elt" -> "ETL/ELT"; None otherwise."""
    wanted = _tokens(topic)
    return next((name for name in settings.TOPIC_DESCRIPTIONS if _tokens(name) == wanted), None)


def queries(topic: str) -> list[str]:
    """The topic itself, plus the first phrases of its course-topic description when it names one of the 8 course topics."""
    name = course_topic(topic)
    if name is None:
        return [topic]
    phrases = [p.strip().removeprefix("and ").strip() for p in settings.TOPIC_DESCRIPTIONS[name].split(":", 1)[1].split(",")]
    return [topic] + [p for p in phrases if p][:settings.PLAN_MAX_EXPANSION_QUERIES]


def gather(topic: str, search: Callable[[str], list[RetrievedChunk]]) -> list[RetrievedChunk]:
    """Every chunk at or above the threshold over the plan's searches, each with its best score, best first."""
    best: dict[str, RetrievedChunk] = {}
    for query in queries(topic):
        for chunk in search(query):
            if chunk.score >= settings.SIMILARITY_THRESHOLD and (chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score):
                best[chunk.chunk_id] = chunk
    return sorted(best.values(), key=lambda c: -c.score)


def section_title(section: str | None) -> tuple[str, str]:
    """(title, parent) of a section path: "A > B > C" -> ("C", "A > B"). The title is real section text, never generated."""
    parts = [p.strip() for p in (section or "").split(">") if p.strip()]
    return (parts[-1], " > ".join(parts[:-1])) if parts else ("(untitled section)", "")


def in_scope(chunks: list[RetrievedChunk]) -> bool:
    """The in-scope rule: enough material at or above the threshold, from enough sections."""
    return (len(chunks) >= settings.PLAN_MIN_CHUNKS
            and len({(c.source_id, c.section) for c in chunks}) >= settings.PLAN_MIN_SECTIONS)


def subtopics(chunks: list[RetrievedChunk], difficulty: Callable[[str], str]) -> list[dict]:
    """The chunks grouped by section (titles that normalise to the same text are merged), in plan order: difficulty, the most relevant
    source first, course order, score."""
    groups: dict[str, dict] = {}
    for c in chunks:
        title, parent = section_title(c.section)
        key = " ".join(re.findall(r"[a-z0-9]+", title.lower()))
        group = groups.setdefault(key, {"title": title, "parent": parent, "source_id": c.source_id, "source_name": c.source_name,
                                        "difficulty": difficulty(c.chunk_id), "chunk_ids": [], "best": c.score,
                                        "order": float(m.group(1)) if (m := _FIRST_NUMBER.search(c.section or "")) else 999.0})
        group["chunk_ids"].append(c.chunk_id)
        group["best"] = max(group["best"], c.score)
    source_rank = {}
    for c in chunks:                                   # chunks are best first, so the first time a source appears is its rank
        source_rank.setdefault(c.source_id, len(source_rank))
    return sorted(groups.values(), key=lambda g: (_DIFFICULTY.get(g["difficulty"], 1), source_rank[g["source_id"]], g["order"],
                                                  -g["best"]))


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def session_lengths(budget: int) -> list[int]:
    """Session minutes for one block of time: as many 45-minute sessions (10-minute breaks between) as fit; spare time of at least a
    break plus a 20-minute session becomes a short final session, smaller spare time is spread over the sessions. A budget shorter
    than one session is one session."""
    full, gap = settings.PLAN_SESSION_MINUTES, settings.PLAN_BREAK_MINUTES
    if budget <= full:
        return [budget]
    n = max(1, (budget + gap) // (full + gap))
    spare = budget - n * full - (n - 1) * gap
    if spare >= gap + settings.PLAN_MIN_SESSION_MINUTES:
        return [full] * n + [spare - gap]
    lengths = [full + spare // n] * n
    for i in range(spare % n):
        lengths[i] += 1
    return lengths


def need_minutes(group: dict) -> int:
    """Time a sub-topic needs: PLAN_MINUTES_PER_CHUNK per chunk, at least PLAN_MIN_SUBTOPIC_MINUTES and at most one session."""
    return min(settings.PLAN_MAX_SUBTOPIC_MINUTES,
               max(settings.PLAN_MIN_SUBTOPIC_MINUTES, settings.PLAN_MINUTES_PER_CHUNK * len(group["chunk_ids"])))


def allocate(groups: list[dict], study_minutes: int) -> tuple[list[dict], list[dict], int]:
    """Minutes per sub-topic. When everything fits, each gets its need and the spare time is returned as review minutes. Otherwise
    sub-topics are taken in order of relevance (best score) while their need fits, the last one shortened to what is left (if that is
    at least the minimum); the rest go under "If you have more time". The kept ones stay in plan order."""
    need = {id(g): need_minutes(g) for g in groups}
    if sum(need.values()) <= study_minutes:
        return [{**g, "minutes": need[id(g)]} for g in groups], [], study_minutes - sum(need.values())
    chosen: dict[int, int] = {}
    left = study_minutes
    for g in sorted(groups, key=lambda g: -g["best"]):
        if left < settings.PLAN_MIN_SUBTOPIC_MINUTES:
            break
        chosen[id(g)] = min(need[id(g)], left)
        left -= chosen[id(g)]
    placed = [{**g, "minutes": chosen[id(g)]} for g in groups if id(g) in chosen]
    for i in range(left):                              # a few spare minutes (under the minimum) go to the sub-topics in order
        placed[i % len(placed)]["minutes"] += 1
    more = sorted((g for g in groups if id(g) not in chosen), key=lambda g: -g["best"])
    return placed, more, 0


def pack(items: list[dict], lengths: list[int]) -> list[list[dict]]:
    """Fill the sessions in order; a sub-topic longer than what is left of a session continues in the next one. A gap under
    PLAN_MIN_SUBTOPIC_MINUTES at the end of a session goes to the sub-topic before it, so no sub-topic is cut into a tiny piece."""
    sessions: list[list[dict]] = [[] for _ in lengths]
    s, left = 0, lengths[0]
    for item in items:
        remaining, part = item["minutes"], 0
        while remaining > 0 and s < len(lengths):
            if 0 < left < settings.PLAN_MIN_SUBTOPIC_MINUTES and remaining > left and sessions[s] and s + 1 < len(lengths):
                sessions[s][-1]["minutes"] += left       # the small gap extends the previous sub-topic
                left = 0
            if left == 0:
                s += 1
                if s == len(lengths):
                    break
                left = lengths[s]
            take = min(remaining, left)
            sessions[s].append({**item, "minutes": take, "continued": part > 0})
            remaining, left, part = remaining - take, left - take, part + 1
    return sessions


def build(request: PlanRequest, chunks: list[RetrievedChunk], difficulty: Callable[[str], str]) -> dict:
    """The plan's structure, from code only (no model): sessions, times, sub-topics, citations, checkpoints."""
    groups = subtopics(chunks, difficulty)
    if request.mode == "days":
        day_lengths = session_lengths(request.minutes_per_day)
        lengths = day_lengths * request.days
        day_of = [d + 1 for d in range(request.days) for _ in day_lengths]
    else:
        lengths = session_lengths(request.minutes)
        day_of = [None] * len(lengths)
    placed, more, review = allocate(groups, sum(lengths))
    if review:
        placed.append({"title": "Review and quiz", "parent": "", "review": True, "chunk_ids": [], "minutes": review,
                       "difficulty": None, "source_name": "", "source_id": ""})
    packed = pack(placed, lengths)
    sessions = []
    for n, (items, minutes) in enumerate(zip(packed, lengths), 1):
        last_of_day = request.mode == "days" and (n == len(lengths) or day_of[n] != day_of[n - 1])
        checkpoint = last_of_day if request.mode == "days" else (n % settings.PLAN_QUIZ_EVERY_SESSIONS == 0 or n == len(lengths))
        gap = 0 if n == len(lengths) or last_of_day else settings.PLAN_BREAK_MINUTES
        sessions.append({"n": n, "day": day_of[n - 1], "minutes": minutes, "break_after": gap, "items": items,
                         "checkpoint": checkpoint,
                         "goal": "", "questions": []})
    covered = sorted({g["title"] for g in groups})
    return {"topic": request.topic, "mode": request.mode, "days": request.days, "minutes_per_day": request.minutes_per_day,
            "per_day_given": request.per_day_given, "budget_minutes": request.minutes, "sessions": sessions,
            "more_time": [{"title": g["title"], "parent": g["parent"], "chunk_ids": g["chunk_ids"]} for g in more],
            "review_minutes": review, "chunk_ids": [c.chunk_id for c in chunks], "section_count": len(groups),
            "note": (f"The course materials cover {request.topic} in {len(groups)} section(s); the spare time is for review and a quiz."
                     if review else ""), "covered_titles": covered}


def total_minutes(plan: dict) -> int:
    """Study minutes plus breaks: what the plan uses of the budget."""
    return sum(s["minutes"] + s["break_after"] for s in plan["sessions"])


def session_chunk_ids(session: dict) -> list[str]:
    return [i for item in session["items"] for i in item["chunk_ids"]]


# ---------------------------------------------------------------------------
# Goals and suggested questions (the only part the model writes)
# ---------------------------------------------------------------------------

def _prompt(sessions: list[dict], chunk_text: Callable[[str], str]) -> list[dict]:
    blocks = []
    for s in sessions:
        lines = [f"Session {s['n']}: " + "; ".join(i["title"] for i in s["items"] if not i.get("review"))]
        for item in s["items"]:
            if item.get("review") or not item["chunk_ids"]:
                continue
            first = item["chunk_ids"][0]
            lines.append(f"[{first}] {chunk_text(first)[:settings.PLAN_CHUNK_CHARS_IN_PROMPT]}")
        blocks.append("\n".join(lines))
    user = "\n\n".join(blocks) + "\n\n" + settings.PLAN_INSTRUCTION.format(n=settings.PLAN_QUESTIONS_PER_SESSION)
    return [{"role": "system", "content": settings.PLAN_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _repair_brackets(text: str) -> str:
    """Drop a closing bracket that does not match the open one, and add any closers still missing at the end. Fixes the one fault
    DeepSeek's JSON mode showed in evaluation rounds 2-3: a stray "}" after a question list (..."q3"}]}]} for ..."q3"]}]}). Text
    inside strings is left alone."""
    pairs, stack, out, in_string, escaped = {"{": "}", "[": "]"}, [], [], False, False
    for ch in text:
        if in_string:
            escaped = ch == "\\" and not escaped
            in_string = not (ch == '"' and not escaped)
            out.append(ch)
            continue
        if ch == '"':
            in_string = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                continue                               # a stray closer: dropped
            stack.pop()
        out.append(ch)
    return "".join(out + stack[::-1])


def _parse(text: str, numbers: list[int]) -> dict[int, dict]:
    """The goals and questions by session number. Accepts {"sessions": [...]}, a bare list, or one session object on its own (a
    one-session plan, round 1 of the evaluation), and numbers written as "Session 2"; a row without a number takes its place in the
    order of the sessions asked for."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = {}
    start, end = text.find("{"), text.rfind("}")
    candidates = [text, _repair_brackets(text)] + ([text[start:end + 1]] if 0 <= start < end else [])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except ValueError:
            continue
    if isinstance(data, dict) and isinstance(data.get("sessions"), list):
        rows = data["sessions"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and ("questions" in data or "goal" in data):
        rows = [data]
    else:
        rows = []
    out = {}
    for position, row in enumerate(r for r in rows if isinstance(r, dict)):
        found = re.search(r"\d+", str(row.get("session", "")))
        number = int(found.group()) if found else (numbers[position] if position < len(numbers) else None)
        if number in numbers:
            out[number] = row
    return out


def _plain_title(title: str) -> str:
    """A section title without its numbering, for a template question: "8.5 Data Warehousing" -> "Data Warehousing"."""
    return re.sub(r"^(?:chapter\s+)?\d+(?:\.\d+)*[\s.:\-]*", "", title, flags=re.IGNORECASE).strip() or title


def _grounded(question: str, own: set[str], search: Callable[[str], list[RetrievedChunk]]) -> bool:
    return any(h.chunk_id in own and h.score >= settings.SIMILARITY_THRESHOLD for h in search(question))


def write_session_text(plan: dict, model: str, search: Callable[[str], list[RetrievedChunk]], chunk_text: Callable[[str], str],
                       on_status: Callable[[str], None] | None = None) -> dict:
    """Fill each study session's goal and suggested questions (one cloud call, or local calls of PLAN_LOCAL_SESSIONS_PER_CALL
    sessions), then keep only grounded questions. A failed call leaves template questions, so a plan is always shown."""
    start = time.perf_counter()
    study = [s for s in plan["sessions"] if any(not i.get("review") for i in s["items"])]
    cloud = llm.is_cloud_model(model)
    size = len(study) if cloud else settings.PLAN_LOCAL_SESSIONS_PER_CALL
    written: dict[int, dict] = {}
    usage, errors, retried = None, [], 0
    for i in range(0, len(study), max(1, size)):
        messages = _prompt(study[i:i + size], chunk_text)
        numbers = [s["n"] for s in study[i:i + size]]
        # One retry when the reply cannot be used: DeepSeek's JSON mode sometimes closes a list with "}" (evaluation round 2).
        for attempt in range(1 + settings.PLAN_UNUSABLE_REPLY_RETRIES):
            try:
                if cloud:
                    reply = llm.cloud_complete(model, messages, settings.PLAN_MAX_OUTPUT_TOKENS, json_output=True,
                                               on_status=on_status)
                    text = reply.text
                    part = {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                            "cost_usd_list": reply.cost_usd_list, "latency_seconds": reply.latency_seconds}
                    usage = part if usage is None else {k: usage[k] + part[k] for k in usage}
                else:
                    text = llm.local_complete_json(messages, model, settings.PLAN_MAX_OUTPUT_TOKENS, settings.PLAN_JSON_SCHEMA)
            except Exception as error:     # cloud or Ollama failure: template questions instead, the plan is still shown
                errors.append(f"{type(error).__name__}: {str(error)[:200]}")
                break
            rows = _parse(text, numbers)
            if rows:
                written.update(rows)
                break
            errors.append("unusable reply: " + " ".join(text[:400].split()))   # kept (cut) for the evaluation
            retried += attempt == 0
    stats = {"generated": 0, "grounded_before_replacement": 0, "replaced": 0, "dropped": 0, "template_only_sessions": 0}
    for s in plan["sessions"]:
        if not any(not i.get("review") for i in s["items"]):
            s["goal"] = "Review the plan's sub-topics, then take the quiz."
            continue
        own = set(session_chunk_ids(s))
        titles = [i["title"] for i in s["items"] if not i.get("review")]
        row = written.get(s["n"], {})
        goal = _ID_IN_BRACKETS.sub("", str(row.get("goal") or "")).strip()
        s["goal"] = goal or "Learn " + ", ".join(dict.fromkeys(titles)) + "."
        proposed = [str(q).strip() for q in row.get("questions") or [] if str(q).strip()][:settings.PLAN_QUESTIONS_PER_SESSION]
        if not proposed:
            stats["template_only_sessions"] += 1
        kept, templates = [], [f"What is {_plain_title(t)}?" for t in dict.fromkeys(titles)]
        for q in proposed:
            stats["generated"] += 1
            if _grounded(q, own, search):
                stats["grounded_before_replacement"] += 1
                kept.append(q)
                continue
            while templates:
                candidate = templates.pop(0)
                if candidate not in kept and _grounded(candidate, own, search):
                    stats["replaced"] += 1
                    kept.append(candidate)
                    break
            else:
                stats["dropped"] += 1
        if not proposed:
            kept = [t for t in templates[:2] if _grounded(t, own, search)]
        s["questions"] = kept
    plan.update(model=model, usage=usage, question_stats=stats, write_errors=errors, unusable_reply_retries=retried,
                write_seconds=round(time.perf_counter() - start, 1))
    return plan


def to_markdown(plan: dict) -> str:
    """The plan as Markdown, for the download button (chunk IDs kept, so the student can find the sources)."""
    budget = (f"{plan['days']} day(s) x {plan['minutes_per_day'] // 60} h" if plan["mode"] == "days"
              else f"{plan['budget_minutes']} minutes")
    lines = [f"# Study plan: {plan['topic']}", "", f"Time: {budget}. Sessions of up to {settings.PLAN_SESSION_MINUTES} minutes.", ""]
    for s in plan["sessions"]:
        day = f"Day {s['day']} · " if s["day"] else ""
        lines += [f"## {day}Session {s['n']} ({s['minutes']} min)", "", s["goal"], ""]
        for item in s["items"]:
            ids = ", ".join(item["chunk_ids"])
            cont = " (continued)" if item.get("continued") else ""
            lines.append(f"- {item['title']}{cont}: {item['minutes']} min" + (f" [{ids}]" if ids else ""))
        if s["questions"]:
            lines += ["", "Ask the chatbot:"] + [f"- {q}" for q in s["questions"]]
        if s["checkpoint"]:
            lines += ["", "Quiz checkpoint."]
        if s["break_after"]:
            lines += ["", f"Break: {s['break_after']} minutes."]
        lines.append("")
    if plan["more_time"]:
        lines += ["## If you have more time", ""] + [f"- {m['title']}" for m in plan["more_time"]]
    return "\n".join(lines)
