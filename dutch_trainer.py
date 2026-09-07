from __future__ import annotations

import hmac
import html
import json
import random
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo
from urllib.parse import quote

import requests
import streamlit as st
import streamlit.components.v1 as components

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


st.set_page_config(page_title="Dutch Trainer", page_icon="🇳🇱", layout="centered")


# -----------------------------------------------------------------------------
# Secrets / Supabase REST
# -----------------------------------------------------------------------------

def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return default


SUPA_URL = secret("SUPABASE_URL").rstrip("/")
SUPA_KEY = secret("SUPABASE_SERVICE_KEY") or secret("SUPABASE_KEY")
OPENAI_KEY = secret("OPENAI_API_KEY")
OPENAI_MODEL = secret("OPENAI_MODEL", "gpt-5.6-luna")
TRAINER_TIMEZONE = secret("TRAINER_TIMEZONE", "Europe/Amsterdam")
REST_DAYS_ALLOWED = 2  # same forgiving rule as Napraten: up to 2 consecutive rest days


def configured() -> bool:
    return bool(SUPA_URL and SUPA_KEY)


def headers(prefer: str | None = None) -> dict[str, str]:
    # Supabase's newer sb_secret_/sb_publishable_ keys are API keys, not JWTs.
    # They belong in the apikey header only. Legacy service_role/anon JWT keys
    # (typically starting with eyJ...) may also be sent as Bearer tokens.
    h = {
        "apikey": SUPA_KEY,
        "Content-Type": "application/json",
    }
    if SUPA_KEY and not SUPA_KEY.startswith(("sb_secret_", "sb_publishable_")):
        h["Authorization"] = f"Bearer {SUPA_KEY}"
    if prefer:
        h["Prefer"] = prefer
    return h


def supa_get(table: str, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
    r = requests.get(
        f"{SUPA_URL}/rest/v1/{table}",
        params=params or {},
        headers=headers(),
        timeout=12,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def supa_post(table: str, rows: list[dict[str, Any]], upsert: bool = False) -> list[dict[str, Any]]:
    prefer = "return=representation"
    if upsert:
        prefer = "resolution=merge-duplicates,return=representation"
    r = requests.post(
        f"{SUPA_URL}/rest/v1/{table}",
        headers=headers(prefer),
        json=rows,
        timeout=15,
    )
    r.raise_for_status()
    if not r.text.strip():
        return []
    data = r.json()
    return data if isinstance(data, list) else []


def supa_patch(table: str, filters: dict[str, str], values: dict[str, Any]) -> None:
    r = requests.patch(
        f"{SUPA_URL}/rest/v1/{table}",
        params=filters,
        headers=headers("return=minimal"),
        json=values,
        timeout=12,
    )
    r.raise_for_status()


def supa_get_paged(
    table: str,
    params: dict[str, str] | None = None,
    page_size: int = 1000,
    max_pages: int = 100,
) -> list[dict[str, Any]]:
    """Fetch rows in pages so streak history is not truncated at Supabase's row cap."""
    out: list[dict[str, Any]] = []
    base = dict(params or {})
    for page in range(max_pages):
        start = page * page_size
        stop = start + page_size - 1
        h = headers()
        h["Range"] = f"{start}-{stop}"
        r = requests.get(
            f"{SUPA_URL}/rest/v1/{table}",
            params=base,
            headers=h,
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or not data:
            break
        out.extend(data)
        if len(data) < page_size:
            break
    return out


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def fetch_items(include_inactive: bool = False) -> list[dict[str, Any]]:
    params = {
        "select": "id,item_type,dutch,english,chinese,example_nl,example_en,accepted_answers,level,theme,notes,source,active,created_at",
        "order": "created_at.desc",
    }
    if not include_inactive:
        params["active"] = "eq.true"
    return supa_get("trainer_items", params)


def fetch_progress() -> dict[str, dict[str, Any]]:
    rows = supa_get("trainer_progress", {"select": "*"})
    return {str(r["item_id"]): r for r in rows}


def get_setting(key: str, default: Any) -> Any:
    rows = supa_get("trainer_settings", {"key": f"eq.{key}", "select": "value", "limit": "1"})
    if not rows:
        return default
    value = rows[0].get("value", default)
    return default if value is None else value


def set_setting(key: str, value: Any) -> None:
    supa_post("trainer_settings", [{"key": key, "value": value}], upsert=True)


def mastery_target() -> int:
    try:
        value = int(get_setting("mastery_target", 5))
    except Exception:
        value = 5
    return 3 if value == 3 else 5


def ensure_progress(item_id: str) -> dict[str, Any]:
    rows = supa_get("trainer_progress", {"item_id": f"eq.{item_id}", "select": "*"})
    if rows:
        return rows[0]
    made = supa_post(
        "trainer_progress",
        [{"item_id": item_id, "status": "new", "due_on": date.today().isoformat()}],
        upsert=True,
    )
    return made[0] if made else {
        "item_id": item_id,
        "status": "new",
        "due_on": date.today().isoformat(),
        "interval_days": 0,
        "difficulty": 0,
        "attempts": 0,
        "correct": 0,
        "typo": 0,
        "wrong": 0,
        "dont_know": 0,
        "consecutive_correct": 0,
    }


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r"[.,!?;:\"“”()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answer_variants(item: dict[str, Any]) -> list[str]:
    vals = [item.get("dutch", "")]
    extra = item.get("accepted_answers") or []
    if isinstance(extra, str):
        try:
            extra = json.loads(extra)
        except Exception:
            extra = []
    if isinstance(extra, list):
        vals.extend(str(x) for x in extra if x)
    return [normalize(v) for v in vals if normalize(v)]


def classify_answer(item: dict[str, Any], answer: str) -> str:
    mine = normalize(answer)
    variants = answer_variants(item)
    if mine in variants:
        return "correct"
    if not mine:
        return "wrong"

    # Conservative typo detection: close spelling only. Short Dutch words are
    # deliberately not auto-forgiven because one character can change meaning.
    best = max((SequenceMatcher(None, mine, v).ratio() for v in variants), default=0.0)
    target_len = max((len(v) for v in variants), default=0)
    if target_len >= 5 and best >= 0.90:
        return "typo"
    if target_len >= 12 and best >= 0.94:
        return "typo"
    return "wrong"


# Learning is deliberately spaced across days. A same-day rescue after a miss
# is useful practice, but it does not count as another mastery success.
LEARNING_INTERVALS = {
    3: [2, 7],          # correct on day 0 -> +2d -> +7d -> retire on 3rd success
    5: [1, 3, 7, 14],  # stronger track: roughly day 0, 1, 4, 11, 25
}
RETIRED_INTERVALS = [45, 90, 180, 365]


def _next_retired_interval(before: int) -> int:
    return next((x for x in RETIRED_INTERVALS if x > before), RETIRED_INTERVALS[-1])


def _learning_interval(successes: int, target: int) -> int:
    steps = LEARNING_INTERVALS[target]
    if successes <= 0:
        return 1
    idx = min(successes - 1, len(steps) - 1)
    return steps[idx]


def schedule_after(
    progress: dict[str, Any],
    grade: str,
    target: int,
    was_retired: bool = False,
) -> tuple[dict[str, Any], int, str]:
    before = int(progress.get("interval_days") or 0)
    attempts = int(progress.get("attempts") or 0) + 1
    correct = int(progress.get("correct") or 0)
    typo = int(progress.get("typo") or 0)
    wrong = int(progress.get("wrong") or 0)
    dont = int(progress.get("dont_know") or 0)
    successes = int(progress.get("consecutive_correct") or 0)
    difficulty = int(progress.get("difficulty") or 0)

    last_day = _review_local_date(str(progress.get("last_reviewed_at") or ""))
    today = local_today()
    same_day = last_day == today
    transition = ""

    if was_retired:
        if grade == "correct":
            correct += 1
            difficulty = max(0, difficulty - 1)
            after = _next_retired_interval(before)
            status = "mastered"
            transition = "retired_pass"
        elif grade == "typo":
            typo += 1
            # A typo is not evidence of forgetting. Keep it retired, but check
            # it again sooner than a clean recall.
            after = 30
            status = "mastered"
            transition = "retired_typo"
        else:
            if grade == "dont_know":
                dont += 1
                difficulty += 2
            else:
                wrong += 1
                difficulty += 1
            successes = 0
            after = 0
            status = "new"
            transition = "relearn"
    else:
        if grade == "correct":
            correct += 1
            difficulty = max(0, difficulty - 1)
            # Only one clean recall per calendar day advances mastery. This
            # stops same-session repetition from producing false mastery.
            if not same_day:
                successes += 1
            if successes >= target:
                after = RETIRED_INTERVALS[0]
                status = "mastered"
                transition = "retire"
            else:
                after = _learning_interval(successes, target)
                status = "familiar" if successes >= max(2, target - 2) else "learning"
        elif grade == "typo":
            typo += 1
            # Keep mastery credit; a spelling slip neither advances nor resets it.
            after = max(1, min(before or 2, 3))
            status = "familiar" if successes >= max(2, target - 2) else "learning"
        elif grade == "dont_know":
            dont += 1
            successes = 0
            difficulty += 2
            after = 1
            status = "learning"
        else:
            wrong += 1
            successes = 0
            difficulty += 1
            after = 1
            status = "learning"

    values = {
        "status": status,
        "due_on": (today + timedelta(days=after)).isoformat(),
        "interval_days": after,
        "difficulty": difficulty,
        "attempts": attempts,
        "correct": correct,
        "typo": typo,
        "wrong": wrong,
        "dont_know": dont,
        "consecutive_correct": successes,
        "last_grade": grade,
        "last_reviewed_at": datetime.now(_trainer_tz()).isoformat(),
    }
    return values, before, transition


def save_review(item: dict[str, Any], answer: str, grade: str, prompt: str) -> dict[str, Any]:
    """Save one review and return the learning transition."""
    progress = ensure_progress(str(item["id"]))
    target = mastery_target()
    was_retired = not bool(item.get("active", True)) or progress.get("status") == "mastered"
    values, before, transition = schedule_after(progress, grade, target, was_retired=was_retired)
    after = int(values["interval_days"])

    supa_patch("trainer_progress", {"item_id": f"eq.{item['id']}"}, values)
    supa_post(
        "trainer_reviews",
        [{
            "item_id": str(item["id"]),
            "direction": "meaning_to_dutch",
            "prompt": prompt,
            "answer": answer,
            "grade": grade,
            "interval_before": before,
            "interval_after": after,
        }],
    )

    if transition == "retire":
        supa_patch("trainer_items", {"id": f"eq.{item['id']}"}, {"active": False})
    elif transition == "relearn":
        # A failed long-term review proves the item is no longer secure. Put it
        # back in the ordinary New/Learning pool immediately.
        supa_patch("trainer_items", {"id": f"eq.{item['id']}"}, {"active": True})

    fetch_activity_dates.clear()
    return {
        "transition": transition,
        "interval_after": after,
        "successes": int(values.get("consecutive_correct") or 0),
        "target": target,
    }


def practice_candidates(limit: int = 40) -> list[dict[str, Any]]:
    # Include retired items because their sparse long-term checks must return
    # automatically when due. Permanently hidden/manual inactive items without
    # mastered progress remain excluded.
    items = fetch_items(include_inactive=True)
    prog = fetch_progress()
    today = local_today().isoformat()

    scored: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    for item in items:
        p = prog.get(str(item["id"]))
        active = bool(item.get("active", True))

        if not p:
            if not active:
                continue
            pri = (2, 0, item.get("created_at", ""))  # unseen after scheduled reviews
        else:
            due = str(p.get("due_on") or today)
            if due > today:
                continue
            status = str(p.get("status") or "new")
            difficulty = int(p.get("difficulty") or 0)
            if not active and status == "mastered":
                pri = (0, -difficulty, due)  # sparse retired review: don't miss it
            elif active:
                pri = (1, -difficulty, due)
            else:
                continue
        scored.append((pri, item))

    scored.sort(key=lambda x: x[0])
    return [x[1] for x in scored[:limit]]


def migrate_scheduler_v2() -> None:
    """Undo the brief v1.5 one-correct retirement rule once."""
    try:
        version = int(get_setting("scheduler_version", 1))
    except Exception:
        version = 1
    if version >= 2:
        return

    target = mastery_target()
    items = fetch_items(include_inactive=True)
    prog = fetch_progress()
    for item in items:
        if item.get("active", True):
            continue
        p = prog.get(str(item["id"]))
        if not p or p.get("status") != "mastered":
            continue
        if int(p.get("consecutive_correct") or 0) < target:
            supa_patch("trainer_items", {"id": f"eq.{item['id']}"}, {"active": True})
            supa_patch(
                "trainer_progress",
                {"item_id": f"eq.{item['id']}"},
                {
                    "status": "learning",
                    "due_on": local_today().isoformat(),
                    "interval_days": 0,
                },
            )
    set_setting("scheduler_version", 2)


# -----------------------------------------------------------------------------
# Streak helpers
# -----------------------------------------------------------------------------


def _trainer_tz() -> ZoneInfo:
    try:
        return ZoneInfo(TRAINER_TIMEZONE)
    except Exception:
        return ZoneInfo("Europe/Amsterdam")


def local_today() -> date:
    return datetime.now(_trainer_tz()).date()


def _review_local_date(value: str) -> date | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(_trainer_tz()).date()
    except Exception:
        return None


@st.cache_data(ttl=30, show_spinner=False)
def fetch_activity_dates() -> list[date]:
    rows = supa_get_paged(
        "trainer_reviews",
        {"select": "reviewed_at", "order": "reviewed_at.asc"},
    )
    dates = {_review_local_date(str(r.get("reviewed_at") or "")) for r in rows}
    return sorted(d for d in dates if d is not None)


def streak_summary(activity_dates: list[date]) -> dict[str, Any]:
    today = local_today()
    dates = sorted(set(d for d in activity_dates if d <= today))
    if not dates:
        return {
            "current": 0,
            "best": 0,
            "today_done": False,
            "gap": None,
            "rest_days_current": 0,
            "last7": [],
            "message": "🌱 Practice one item today and your streak begins!",
        }

    allowed_gap = REST_DAYS_ALLOWED + 1
    clusters: list[list[date]] = []
    cluster = [dates[0]]
    for d in dates[1:]:
        if (d - cluster[-1]).days <= allowed_gap:
            cluster.append(d)
        else:
            clusters.append(cluster)
            cluster = [d]
    clusters.append(cluster)

    historical_best = max((c[-1] - c[0]).days + 1 for c in clusters)
    last = dates[-1]
    gap = (today - last).days
    today_done = today in set(dates)

    if gap <= allowed_gap:
        current_cluster = clusters[-1]
        start = current_cluster[0]
        current = (today - start).days + 1
        active_set = set(current_cluster)
        rest_days_current = sum(
            1 for i in range(current)
            if start + timedelta(days=i) not in active_set
        )
    else:
        current = 0
        rest_days_current = 0

    best = max(historical_best, current)

    if current == 0:
        message = "🌱 Your previous streak ended. Practice today to start a new one."
    elif today_done:
        message = "✅ Today done — lekker bezig!"
    elif gap <= REST_DAYS_ALLOWED:
        remaining = REST_DAYS_ALLOWED + 1 - gap
        message = f"🧊 Rest day — your streak is safe. {remaining} day{'s' if remaining != 1 else ''} of leeway left."
    else:
        message = "⚠️ Last chance — practice today to keep your streak."

    last7 = []
    active_all = set(dates)
    for offset in range(6, -1, -1):
        d = today - timedelta(days=offset)
        last7.append({"date": d, "active": d in active_all, "today": offset == 0})

    return {
        "current": current,
        "best": best,
        "today_done": today_done,
        "gap": gap,
        "rest_days_current": rest_days_current,
        "last7": last7,
        "message": message,
    }


def render_streak_banner() -> None:
    try:
        summary = streak_summary(fetch_activity_dates())
    except Exception as e:
        st.caption(f"Streak temporarily unavailable: {e}")
        return

    dots = []
    for d in summary["last7"]:
        if d["active"]:
            cls = "on"
        else:
            cls = "off"
        if d["today"]:
            cls += " today"
        dots.append(f'<span class="trainer-dot {cls}" title="{d["date"].isoformat()}"></span>')

    current = int(summary["current"])
    best = int(summary["best"])
    rests = int(summary["rest_days_current"])
    noun = "day" if current == 1 else "days"
    rest_noun = "rest day" if rests == 1 else "rest days"
    flame = "🔥" if summary["today_done"] else ("🧊" if current else "🌱")

    st.markdown(
        f"""
        <style>
          .trainer-streak {{
            border:1px solid rgba(128,128,128,.30); border-radius:18px;
            padding:18px 20px; margin:4px 0 18px 0;
          }}
          .trainer-streak-top {{display:flex;gap:14px;align-items:center;flex-wrap:wrap;}}
          .trainer-streak-num {{font-size:2.2rem;font-weight:750;line-height:1;}}
          .trainer-streak-label {{font-size:1rem;letter-spacing:.08em;text-transform:uppercase;opacity:.72;}}
          .trainer-dots {{margin-left:auto;display:flex;gap:8px;align-items:center;}}
          .trainer-dot {{width:15px;height:15px;border-radius:50%;display:inline-block;border:1px solid rgba(128,128,128,.45);}}
          .trainer-dot.on {{background:#d68a00;border-color:#d68a00;}}
          .trainer-dot.off {{background:rgba(214,138,0,.12);}}
          .trainer-dot.today {{outline:2px solid currentColor;outline-offset:3px;}}
          .trainer-streak-msg {{margin-top:12px;font-size:1rem;}}
          .trainer-streak-sub {{margin-top:6px;opacity:.72;font-size:.92rem;}}
          @media (max-width: 520px) {{.trainer-dots{{margin-left:0;width:100%;}}}}
        </style>
        <div class="trainer-streak">
          <div class="trainer-streak-top">
            <span style="font-size:2rem">{flame}</span>
            <span class="trainer-streak-num">{current}</span>
            <span class="trainer-streak-label">{noun} streak · best: {best}</span>
            <span class="trainer-dots">{''.join(dots)}</span>
          </div>
          <div class="trainer-streak-msg">{summary['message']}</div>
          <div class="trainer-streak-sub">🧊 {rests} {rest_noun} in this streak · up to {REST_DAYS_ALLOWED} consecutive rest days allowed</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# AI generation
# -----------------------------------------------------------------------------

def generate_items(level: str, topic: str, count: int, kinds: list[str]) -> list[dict[str, Any]]:
    if not OPENAI_KEY or OpenAI is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=OPENAI_KEY)
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_type": {"type": "string", "enum": ["word", "phrase", "sentence"]},
                        "dutch": {"type": "string"},
                        "english": {"type": "string"},
                        "example_nl": {"type": "string"},
                        "example_en": {"type": "string"},
                        "accepted_answers": {"type": "array", "items": {"type": "string"}},
                        "theme": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "item_type", "dutch", "english", "example_nl",
                        "example_en", "accepted_answers", "theme", "notes"
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }

    prompt = f"""
Create exactly {count} useful Dutch learning items for an adult learner at CEFR {level}.
Allowed item types: {', '.join(kinds)}.
Topic: {topic or 'balanced everyday Dutch'}.

Requirements:
- Natural contemporary Dutch used in the Netherlands.
- Mix practical everyday language with useful work/social language when the topic allows.
- For nouns, include the correct article in the Dutch field (de/het).
- Phrases should be useful chunks, not arbitrary fragments.
- Sentences should be natural and worth memorising, generally 4-14 words.
- English must be a concise cue suitable for a typing exercise.
- Do not generate Chinese translations.
- example_nl/example_en are optional in spirit but must be strings; for a sentence item, they may repeat the sentence/meaning.
- accepted_answers should contain only genuinely equivalent Dutch variants, not looser paraphrases.
- Avoid duplicates or trivial variants of the same item.
- Do not include pronunciation respellings.
""".strip()

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "dutch_trainer_items",
                "strict": True,
                "schema": schema,
            }
        },
    )
    data = json.loads(response.output_text)
    out = []
    for raw in data.get("items", []):
        if raw.get("item_type") not in kinds:
            continue
        dutch = str(raw.get("dutch", "")).strip()
        english = str(raw.get("english", "")).strip()
        if not dutch or not english:
            continue
        out.append({
            "item_type": raw["item_type"],
            "dutch": dutch,
            "english": english,
            "example_nl": str(raw.get("example_nl", "")).strip(),
            "example_en": str(raw.get("example_en", "")).strip(),
            "accepted_answers": raw.get("accepted_answers") or [],
            "level": level,
            "theme": str(raw.get("theme", topic or "general")).strip() or "general",
            "notes": str(raw.get("notes", "")).strip(),
            "source": "ai",
            "active": True,
        })
    return out


def insert_items(rows: list[dict[str, Any]]) -> tuple[int, list[str]]:
    added = 0
    skipped: list[str] = []
    for row in rows:
        try:
            result = supa_post("trainer_items", [row])
            if result:
                item_id = str(result[0]["id"])
                supa_post(
                    "trainer_progress",
                    [{"item_id": item_id, "status": "new", "due_on": date.today().isoformat()}],
                    upsert=True,
                )
            added += 1
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 409:
                skipped.append(row.get("dutch", ""))
            else:
                raise
    return added, skipped


# -----------------------------------------------------------------------------
# Pronunciation helper
# -----------------------------------------------------------------------------

def pronunciation_box(text: str, key: str) -> None:
    safe = json.dumps(text, ensure_ascii=False).replace("</", "<\\/")
    forvo = f"https://forvo.com/search/{quote(text)}/nl/"
    block = f"""
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 4px;">
      <button id="speak-{key}" style="font:600 14px system-ui;padding:8px 12px;border-radius:9px;border:1px solid #aaa;background:transparent;cursor:pointer;">🔊 Pronounce</button>
      <a href="{forvo}" target="_blank" rel="noopener" style="font:600 14px system-ui;">Forvo ↗</a>
    </div>
    <script>
    (() => {{
      const b = document.getElementById('speak-{key}');
      if (!b) return;
      b.addEventListener('click', () => {{
        if (!('speechSynthesis' in window)) {{ b.textContent = 'Audio unavailable'; return; }}
        const u = new SpeechSynthesisUtterance({safe});
        u.lang = 'nl-NL';
        const voices = speechSynthesis.getVoices();
        const nl = voices.find(v => /^nl(-|_)/i.test(v.lang)) || voices.find(v => /^nl/i.test(v.lang));
        if (nl) u.voice = nl;
        u.rate = 0.92;
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
      }});
    }})();
    </script>
    """
    components.html(block, height=55)


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

st.title("🇳🇱 Dutch Trainer")
st.caption("Words · phrases · sentences — synced through Supabase")


def require_password() -> None:
    expected = secret("TRAINER_PASSWORD")
    if not expected:
        st.error("Trainer access is not configured yet. Add TRAINER_PASSWORD to Streamlit secrets.")
        st.stop()

    if st.session_state.get("trainer_authenticated"):
        return

    with st.form("trainer_login"):
        entered = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Open trainer", type="primary")

    if submitted:
        if hmac.compare_digest(entered, expected):
            st.session_state["trainer_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_password()

if not configured():
    st.error("Supabase is not configured yet. Add SUPABASE_URL and SUPABASE_SERVICE_KEY to Streamlit secrets.")
    st.stop()

# Fail early with a useful, non-secret diagnostic instead of a redacted traceback.
try:
    supa_get("trainer_items", {"select": "id", "limit": "1"})
except requests.HTTPError as exc:
    status = exc.response.status_code if exc.response is not None else "unknown"
    if status == 401:
        st.error("Supabase rejected the API key (HTTP 401). Check SUPABASE_SERVICE_KEY in Streamlit Secrets, then save and reboot the app.")
    elif status == 404:
        st.error("Supabase connected, but trainer_items was not found (HTTP 404). The trainer schema may not have been created in this project.")
    else:
        st.error(f"Supabase connection failed (HTTP {status}). Open Manage app → Logs for details.")
    st.stop()

try:
    migrate_scheduler_v2()
except Exception as e:
    st.caption(f"Learning-schedule migration skipped: {e}")

render_streak_banner()

pages = st.tabs(["Practice", "Generate", "Add", "Library", "Progress"])


# Practice --------------------------------------------------------------------
with pages[0]:
    st.subheader("Practice")

    current_target = mastery_target()
    with st.expander("⚙️ Learning settings"):
        chosen_target = st.radio(
            "Retire an item after",
            [3, 5],
            index=0 if current_target == 3 else 1,
            horizontal=True,
            format_func=lambda x: "3 correct recalls · faster" if x == 3 else "5 correct recalls · stronger (recommended)",
            key="mastery_target_choice",
        )
        st.caption(
            "Only one correct recall per calendar day counts toward retirement. "
            "Retired items return after 45, 90, 180 and then 365 days. "
            "If you forget one on a retired review, it returns to the New pool."
        )
        if int(chosen_target) != current_target:
            set_setting("mastery_target", int(chosen_target))
            current_target = int(chosen_target)
            st.success(f"Mastery target changed to {current_target} correct recalls.")

    if "practice_queue" not in st.session_state:
        st.session_state.practice_queue = []
        st.session_state.practice_index = 0
        st.session_state.practice_feedback = None

    c1, c2 = st.columns([1, 1])
    if c1.button("Start / refresh session", type="primary", use_container_width=True):
        st.session_state.practice_queue = practice_candidates(40)
        st.session_state.practice_index = 0
        st.session_state.practice_feedback = None
        st.rerun()
    if c2.button("Clear session", use_container_width=True):
        st.session_state.practice_queue = []
        st.session_state.practice_index = 0
        st.session_state.practice_feedback = None
        st.rerun()

    queue = st.session_state.practice_queue
    idx = int(st.session_state.practice_index)

    if not queue:
        st.info("Press Start / refresh session. If nothing appears, add or generate some items first.")
    elif idx >= len(queue):
        st.success("Session finished.")
        if st.button("Start another session"):
            st.session_state.practice_queue = practice_candidates(40)
            st.session_state.practice_index = 0
            st.session_state.practice_feedback = None
            st.rerun()
    else:
        item = queue[idx]
        cue = item.get("english", "")

        item_progress = ensure_progress(str(item["id"]))
        item_retired = not bool(item.get("active", True)) or item_progress.get("status") == "mastered"
        if item_retired:
            stage = f"Retired review · last interval {int(item_progress.get('interval_days') or 0)} days"
        else:
            stage = f"Recall {int(item_progress.get('consecutive_correct') or 0)}/{current_target} toward retirement"
        st.caption(
            f"{item.get('item_type','item').title()} · {item.get('level','')} · {stage} · {idx + 1}/{len(queue)}"
        )
        st.markdown(f"### {html.escape(cue)}")

        answer_key = f"ans_{item['id']}_{idx}"
        typed = st.text_input("Type the Dutch", key=answer_key, autocomplete="off")

        b1, b2 = st.columns(2)
        submit = b1.button("Check", type="primary", use_container_width=True, disabled=bool(st.session_state.practice_feedback))
        dont = b2.button("I don't know", use_container_width=True, disabled=bool(st.session_state.practice_feedback))

        if submit or dont:
            grade = "dont_know" if dont else classify_answer(item, typed)
            result = save_review(item, typed, grade, cue)
            transition = result.get("transition", "")
            st.session_state.practice_feedback = {
                "grade": grade,
                "typed": typed,
                "correct": item.get("dutch", ""),
                "transition": transition,
                "interval_after": result.get("interval_after", 0),
                "successes": result.get("successes", 0),
                "target": result.get("target", current_target),
            }
            # Real mistakes return later in the same session after a few cards.
            if grade in {"wrong", "dont_know"}:
                insert_at = min(len(queue), idx + 4)
                queue.insert(insert_at, item)
            elif transition in {"retire", "retired_pass", "retired_typo"}:
                # Remove stale future copies. Long-term review timing now lives in Supabase.
                queue = queue[:idx + 1] + [
                    q for q in queue[idx + 1:] if str(q.get("id")) != str(item.get("id"))
                ]
            st.session_state.practice_queue = queue
            st.rerun()

        fb = st.session_state.practice_feedback
        if fb:
            labels = {
                "correct": "✅ Correct",
                "typo": "⌨️ Small typing mistake — treated more gently",
                "wrong": "❌ Not quite",
                "dont_know": "🌱 Marked as not known yet",
            }
            st.write(labels.get(fb["grade"], fb["grade"]))
            transition = fb.get("transition", "")
            if transition == "retire":
                st.caption(
                    f"🏁 Learned: {fb.get('target')} spaced correct recalls. "
                    f"Retired for now; next memory check in {fb.get('interval_after')} days."
                )
            elif transition == "retired_pass":
                st.caption(f"🧠 Long-term memory confirmed. Next review in {fb.get('interval_after')} days.")
            elif transition == "retired_typo":
                st.caption(f"⌨️ Still retired, but scheduled a closer check in {fb.get('interval_after')} days.")
            elif transition == "relearn":
                st.caption("🔄 This retired item was forgotten, so it has returned to the New pool.")
            elif fb.get("grade") == "correct":
                st.caption(
                    f"Memory strength: {fb.get('successes')}/{fb.get('target')} spaced correct recalls."
                )
            st.markdown(f"**Dutch:** {html.escape(str(item.get('dutch','')))}")
            if item.get("example_nl"):
                st.caption(str(item.get("example_nl")))
            pronunciation_box(str(item.get("dutch", "")), f"practice-{idx}")

            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.practice_index = idx + 1
                st.session_state.practice_feedback = None
                st.rerun()


# Generate --------------------------------------------------------------------
with pages[1]:
    st.subheader("Generate new material")
    st.caption("AI is used only to create new items. Once saved, ordinary practice does not call AI.")

    if not OPENAI_KEY:
        st.warning("Add OPENAI_API_KEY to Streamlit secrets to enable generation.")
    else:
        level = st.selectbox("Level", ["A2", "B1", "B2"], index=1)
        topic = st.text_input("Topic", value="everyday Dutch")
        kinds = st.multiselect("Include", ["word", "phrase", "sentence"], default=["word", "phrase", "sentence"])
        count = st.slider("How many", 3, 20, 8)

        if st.button("Generate", type="primary", disabled=not kinds):
            with st.spinner("Generating Dutch material…"):
                try:
                    st.session_state.generated_preview = generate_items(level, topic, count, kinds)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

        preview = st.session_state.get("generated_preview") or []
        if preview:
            st.write(f"Generated {len(preview)} items. Review them before saving.")
            chosen_ids = []
            for i, row in enumerate(preview):
                label = f"{row['item_type'].title()}: {row['dutch']} — {row['english']}"
                if st.checkbox(label, value=True, key=f"gen_keep_{i}"):
                    chosen_ids.append(i)
            if st.button("Save selected to trainer", type="primary"):
                rows = [preview[i] for i in chosen_ids]
                try:
                    added, skipped = insert_items(rows)
                    st.success(f"Saved {added} item(s)." + (f" Skipped duplicates: {', '.join(skipped)}" if skipped else ""))
                    st.session_state.generated_preview = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Saving failed: {e}")


# Add -------------------------------------------------------------------------
with pages[2]:
    st.subheader("Add your own")
    with st.form("add_item"):
        item_type = st.selectbox("Type", ["word", "phrase", "sentence"])
        dutch = st.text_input("Dutch")
        english = st.text_input("English cue")
        level = st.selectbox("Level", ["A1", "A2", "B1", "B2", "C1"], index=2, key="manual_level")
        theme = st.text_input("Theme", value="general")
        example_nl = st.text_input("Dutch example (optional)")
        example_en = st.text_input("English example (optional)")
        alternatives = st.text_input("Accepted Dutch alternatives (separate with |)")
        notes = st.text_area("Notes (optional)")
        add = st.form_submit_button("Add to trainer", type="primary")

    if add:
        if not dutch.strip() or not english.strip():
            st.error("Dutch and English cue are required.")
        else:
            row = {
                "item_type": item_type,
                "dutch": dutch.strip(),
                "english": english.strip(),
                "example_nl": example_nl.strip(),
                "example_en": example_en.strip(),
                "accepted_answers": [x.strip() for x in alternatives.split("|") if x.strip()],
                "level": level,
                "theme": theme.strip() or "general",
                "notes": notes.strip(),
                "source": "manual",
                "active": True,
            }
            try:
                added, skipped = insert_items([row])
                if added:
                    st.success("Added.")
                elif skipped:
                    st.info("That item is already active in the trainer.")
            except Exception as e:
                st.error(f"Could not add item: {e}")


# Library ---------------------------------------------------------------------
with pages[3]:
    st.subheader("Library")
    st.caption("Items retire only after your chosen mastery target, then return automatically for sparse long-term reviews.")
    try:
        items = fetch_items(include_inactive=True)
        library_progress = fetch_progress()
        library_target = mastery_target()
    except Exception as e:
        st.error(f"Could not load library: {e}")
        items = []
        library_progress = {}
        library_target = 5

    f1, f2 = st.columns([2, 1])
    search = f1.text_input("Search Dutch / English", key="library_search")
    state_view = f2.selectbox("Status", ["Active", "Retired", "All"], key="library_status")

    f3, f4 = st.columns(2)
    typ = f3.selectbox("Type", ["all", "word", "phrase", "sentence"], key="library_type")
    level_view = f4.selectbox("Level", ["all", "A2", "B1", "B2"], key="library_level")

    filtered = []
    q = search.strip().lower()
    for item in items:
        active = bool(item.get("active", True))
        if state_view == "Active" and not active:
            continue
        if state_view == "Retired" and active:
            continue
        if typ != "all" and item.get("item_type") != typ:
            continue
        if level_view != "all" and item.get("level") != level_view:
            continue
        hay = f"{item.get('dutch','')} {item.get('english','')} {item.get('theme','')}".lower()
        if q and q not in hay:
            continue
        filtered.append(item)

    active_n = sum(1 for x in items if x.get("active", True))
    retired_n = len(items) - active_n
    st.caption(
        f"{len(filtered)} shown · {active_n} active · {retired_n} retired · {len(items)} total"
    )

    # Keep very large libraries usable instead of rendering hundreds of expanders.
    PAGE_SIZE = 25
    pages_n = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = st.selectbox(
        "Page",
        list(range(1, pages_n + 1)),
        index=0,
        key=f"library_page_{state_view}_{typ}_{level_view}_{q}",
    )
    lo = (int(page) - 1) * PAGE_SIZE
    hi = lo + PAGE_SIZE

    if not filtered:
        st.info("No items match these filters.")

    for item in filtered[lo:hi]:
        active = bool(item.get("active", True))
        p = library_progress.get(str(item["id"]), {})
        if active:
            successes = int(p.get("consecutive_correct") or 0)
            status_label = f"learning {successes}/{library_target}"
        else:
            due = str(p.get("due_on") or "")
            status_label = f"retired · next review {due}" if due else "retired"
        with st.expander(f"{item.get('dutch','')} — {item.get('english','')} · {status_label}"):
            st.write(f"Type: {item.get('item_type')} · Level: {item.get('level')} · Theme: {item.get('theme')}")
            if item.get("example_nl"):
                st.write(item["example_nl"])
            pronunciation_box(str(item.get("dutch", "")), f"lib-{item['id']}")
            if not active:
                if st.button("Return to learning now", key=f"reactivate_{item['id']}"):
                    try:
                        supa_patch("trainer_items", {"id": f"eq.{item['id']}"}, {"active": True})
                        supa_patch(
                            "trainer_progress",
                            {"item_id": f"eq.{item['id']}"},
                            {
                                "status": "new",
                                "due_on": local_today().isoformat(),
                                "interval_days": 0,
                                "consecutive_correct": 0,
                            },
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not return item to learning: {e}")


# Progress --------------------------------------------------------------------
with pages[4]:
    st.subheader("Progress")
    try:
        items = fetch_items(include_inactive=True)
        prog = fetch_progress()
        reviews = supa_get("trainer_reviews", {"select": "id,grade,reviewed_at", "order": "reviewed_at.desc", "limit": "5000"})
    except Exception as e:
        st.error(f"Could not load progress: {e}")
        items, prog, reviews = [], {}, []

    counts = {"new": 0, "learning": 0, "familiar": 0}
    due = 0
    retired_due = 0
    retired = 0
    today = local_today().isoformat()
    for item in items:
        p = prog.get(str(item["id"]))
        if not item.get("active", True):
            retired += 1
            if p and str(p.get("due_on") or today) <= today:
                due += 1
                retired_due += 1
            continue
        if not p:
            counts["new"] += 1
            due += 1
            continue
        status = p.get("status", "new")
        if status == "mastered":
            status = "familiar"
        counts[status] = counts.get(status, 0) + 1
        if str(p.get("due_on") or today) <= today:
            due += 1

    cols = st.columns(4)
    cols[0].metric("New", counts["new"])
    cols[1].metric("Learning", counts["learning"])
    cols[2].metric("Familiar", counts["familiar"])
    cols[3].metric("Retired", retired)
    st.metric("Due now", due)
    if retired_due:
        st.caption(f"{retired_due} of the due items are long-term reviews of retired material.")
    st.caption(
        f"Current mastery rule: retire after {mastery_target()} spaced correct recalls; "
        "retired reviews expand to 45 → 90 → 180 → 365 days when remembered."
    )

    if reviews:
        grades = {"correct": 0, "typo": 0, "wrong": 0, "dont_know": 0}
        for r in reviews:
            grades[r.get("grade", "wrong")] = grades.get(r.get("grade", "wrong"), 0) + 1
        total = sum(grades.values())
        if total:
            st.write(
                f"Recent review history: {grades['correct']} correct · {grades['typo']} typo · "
                f"{grades['wrong']} wrong · {grades['dont_know']} don't know"
            )
