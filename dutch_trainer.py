from __future__ import annotations

import hmac
import html
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
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


@st.cache_resource
def _http_session() -> requests.Session:
    """Reuse HTTPS connections instead of doing a fresh TLS setup for every call."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


@st.cache_resource
def _sync_writer() -> ThreadPoolExecutor:
    # A single background writer preserves review order while keeping the UI fast.
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="dutch-trainer-sync")


HTTP = _http_session()
SYNC_WRITER = _sync_writer()


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
    r = HTTP.get(
        f"{SUPA_URL}/rest/v1/{table}",
        params=params or {},
        headers=headers(),
        timeout=12,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def supa_post(
    table: str,
    rows: list[dict[str, Any]],
    upsert: bool = False,
    returning: bool = True,
) -> list[dict[str, Any]]:
    if returning:
        prefer = "return=representation"
        if upsert:
            prefer = "resolution=merge-duplicates,return=representation"
    else:
        prefer = "return=minimal"
        if upsert:
            prefer = "resolution=merge-duplicates,return=minimal"
    r = HTTP.post(
        f"{SUPA_URL}/rest/v1/{table}",
        headers=headers(prefer),
        json=rows,
        timeout=15,
    )
    r.raise_for_status()
    if not returning or not r.text.strip():
        return []
    data = r.json()
    return data if isinstance(data, list) else []


def supa_patch(table: str, filters: dict[str, str], values: dict[str, Any]) -> None:
    r = HTTP.patch(
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
    """Fetch rows in pages so long-term history is not truncated by row limits."""
    out: list[dict[str, Any]] = []
    base = dict(params or {})
    for page in range(max_pages):
        start = page * page_size
        stop = start + page_size - 1
        h = headers()
        h["Range"] = f"{start}-{stop}"
        r = HTTP.get(
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
# Data helpers / scheduler / activity
# -----------------------------------------------------------------------------

def fetch_items(include_inactive: bool = False) -> list[dict[str, Any]]:
    params = {
        "select": "id,item_type,dutch,english,example_nl,example_en,accepted_answers,level,theme,notes,source,active,created_at",
        "order": "created_at.desc",
    }
    if not include_inactive:
        params["active"] = "eq.true"
    return supa_get("trainer_items", params)


def fetch_progress() -> dict[str, dict[str, Any]]:
    rows = supa_get("trainer_progress", {"select": "*"})
    return {str(r["item_id"]): r for r in rows}


def _settings_cache(force: bool = False) -> dict[str, Any]:
    if force or "trainer_settings_cache" not in st.session_state:
        rows = supa_get("trainer_settings", {"select": "key,value"})
        st.session_state.trainer_settings_cache = {
            str(r.get("key")): r.get("value") for r in rows if r.get("key") is not None
        }
    return st.session_state.trainer_settings_cache


def get_setting(key: str, default: Any) -> Any:
    value = _settings_cache().get(key, default)
    return default if value is None else value


def set_setting(key: str, value: Any) -> None:
    supa_post("trainer_settings", [{"key": key, "value": value}], upsert=True, returning=False)
    _settings_cache()[key] = value


def set_setting_async(key: str, value: Any) -> None:
    """Update the local settings cache now and save to Supabase in the background."""
    _settings_cache()[key] = value
    future = SYNC_WRITER.submit(
        supa_post,
        "trainer_settings",
        [{"key": key, "value": value}],
        True,
        False,
    )
    _pending_syncs().append(future)


def save_resume_state_async(mode: str, item_id: str | None) -> None:
    """Remember the next unfinished practice item across browser/app sessions."""
    rows = [
        {"key": "practice_resume_mode", "value": mode},
        {"key": "practice_resume_item_id", "value": item_id or ""},
    ]
    _settings_cache()["practice_resume_mode"] = mode
    _settings_cache()["practice_resume_item_id"] = item_id or ""
    future = SYNC_WRITER.submit(supa_post, "trainer_settings", rows, True, False)
    _pending_syncs().append(future)


def mastery_target() -> int:
    try:
        value = int(get_setting("mastery_target", 5))
    except Exception:
        value = 5
    return 3 if value == 3 else 5


def _default_progress(item_id: str) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "status": "new",
        "due_on": local_today().isoformat(),
        "interval_days": 0,
        "difficulty": 0,
        "attempts": 0,
        "correct": 0,
        "typo": 0,
        "wrong": 0,
        "dont_know": 0,
        "consecutive_correct": 0,
        "last_grade": None,
        "last_reviewed_at": None,
    }


def ensure_progress(item_id: str) -> dict[str, Any]:
    rows = supa_get("trainer_progress", {"item_id": f"eq.{item_id}", "select": "*"})
    if rows:
        return rows[0]
    p = _default_progress(item_id)
    supa_post("trainer_progress", [p], upsert=True, returning=False)
    return p


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

    # Conservative typo detection. Short Dutch words are not auto-forgiven,
    # because one character can change meaning.
    best = max((SequenceMatcher(None, mine, v).ratio() for v in variants), default=0.0)
    target_len = max((len(v) for v in variants), default=0)
    if target_len >= 5 and best >= 0.90:
        return "typo"
    if target_len >= 12 and best >= 0.94:
        return "typo"
    return "wrong"


# One clean recall per calendar day can advance mastery. Retired items receive
# deliberately sparse memory checks.
LEARNING_INTERVALS = {
    3: [2, 7],
    5: [1, 3, 7, 14],
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


def _persist_review_bundle(
    item_id: str,
    values: dict[str, Any],
    review_row: dict[str, Any],
    transition: str,
    usage_rows: list[dict[str, Any]],
) -> None:
    """Cloud persistence runs in one ordered background worker."""
    supa_patch("trainer_progress", {"item_id": f"eq.{item_id}"}, values)
    supa_post("trainer_reviews", [review_row], returning=False)
    if transition == "retire":
        supa_patch("trainer_items", {"id": f"eq.{item_id}"}, {"active": False})
    elif transition == "relearn":
        supa_patch("trainer_items", {"id": f"eq.{item_id}"}, {"active": True})
    if usage_rows:
        supa_post("trainer_settings", usage_rows, upsert=True, returning=False)


def _pending_syncs() -> list[Any]:
    return st.session_state.setdefault("pending_syncs", [])


def drain_pending_syncs() -> tuple[int, list[str]]:
    pending = []
    errors: list[str] = []
    for future in _pending_syncs():
        if future.done():
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
        else:
            pending.append(future)
    st.session_state.pending_syncs = pending
    if errors:
        st.session_state.setdefault("sync_errors", []).extend(errors)
    return len(pending), errors


def save_review_async(
    item: dict[str, Any],
    answer: str,
    grade: str,
    prompt: str,
    target: int,
) -> dict[str, Any]:
    """Update local state immediately; persist to Supabase in the background."""
    progress = dict(item.get("_progress") or _default_progress(str(item["id"])))
    was_retired = not bool(item.get("active", True)) or progress.get("status") == "mastered"
    values, before, transition = schedule_after(progress, grade, target, was_retired=was_retired)
    after = int(values["interval_days"])

    # Local state becomes authoritative for the current session immediately.
    progress.update(values)
    item["_progress"] = progress
    if transition == "retire":
        item["active"] = False
    elif transition == "relearn":
        item["active"] = True

    st.session_state.did_review_today = True
    st.session_state.activity_review_count = int(st.session_state.get("activity_review_count", 0)) + 1
    dates = set(st.session_state.get("activity_dates", []))
    dates.add(local_today())
    st.session_state.activity_dates = sorted(dates)

    usage_rows = prepare_usage_sync()
    review_row = {
        "item_id": str(item["id"]),
        "direction": "meaning_to_dutch",
        "prompt": prompt,
        "answer": answer,
        "grade": grade,
        "interval_before": before,
        "interval_after": after,
    }
    future = SYNC_WRITER.submit(
        _persist_review_bundle,
        str(item["id"]),
        values,
        review_row,
        transition,
        usage_rows,
    )
    _pending_syncs().append(future)

    return {
        "transition": transition,
        "interval_after": after,
        "successes": int(values.get("consecutive_correct") or 0),
        "target": target,
    }


def practice_candidates(
    limit: int = 40,
    item_types: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fetch once per session/filter; attach progress to each queued item."""
    items = fetch_items(include_inactive=True)
    prog = fetch_progress()
    today = local_today().isoformat()
    missing_progress: list[dict[str, Any]] = []

    stats = {"in_progress": 0, "retired": 0, "due": 0, "total": 0}
    scored: list[tuple[tuple[int, int, str, float], dict[str, Any]]] = []

    for raw in items:
        if item_types and raw.get("item_type") not in item_types:
            continue
        item = dict(raw)
        stats["total"] += 1
        p = prog.get(str(item["id"]))
        active = bool(item.get("active", True))

        if active:
            stats["in_progress"] += 1
        elif p and p.get("status") == "mastered":
            stats["retired"] += 1

        if not p and active:
            p = _default_progress(str(item["id"]))
            missing_progress.append(p)
            prog[str(item["id"])] = p
        item["_progress"] = dict(p) if p else None

        if not p:
            if not active:
                continue
            due = today
            pri = (2, 0, due, random.random())
        else:
            due = str(p.get("due_on") or today)
            if due > today:
                continue
            status = str(p.get("status") or "new")
            difficulty = int(p.get("difficulty") or 0)
            if not active and status == "mastered":
                pri = (0, -difficulty, due, random.random())
            elif active:
                pri = (1, -difficulty, due, random.random())
            else:
                continue

        stats["due"] += 1
        scored.append((pri, item))

    if missing_progress:
        supa_post("trainer_progress", missing_progress, upsert=True, returning=False)

    scored.sort(key=lambda x: x[0])
    return [x[1] for x in scored[:limit]], stats


def migrate_scheduler_v2() -> None:
    """Undo the brief one-correct retirement rule once."""
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
                {"status": "learning", "due_on": local_today().isoformat(), "interval_days": 0},
            )
    set_setting("scheduler_version", 2)


# -----------------------------------------------------------------------------
# Streak + time helpers
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


@st.cache_data(ttl=60, show_spinner=False)
def fetch_activity_snapshot() -> dict[str, Any]:
    rows = supa_get_paged(
        "trainer_reviews",
        {"select": "reviewed_at", "order": "reviewed_at.asc"},
    )
    dates = {_review_local_date(str(r.get("reviewed_at") or "")) for r in rows}
    return {
        "dates": sorted(d for d in dates if d is not None),
        "review_count": len(rows),
    }


def session_activity_snapshot() -> dict[str, Any]:
    if "activity_dates" not in st.session_state:
        snap = fetch_activity_snapshot()
        st.session_state.activity_dates = list(snap["dates"])
        st.session_state.activity_review_count = int(snap["review_count"])
    if st.session_state.get("did_review_today"):
        dates = set(st.session_state.activity_dates)
        dates.add(local_today())
        st.session_state.activity_dates = sorted(dates)
    return {
        "dates": list(st.session_state.activity_dates),
        "review_count": int(st.session_state.get("activity_review_count", 0)),
    }


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
            "message": "Practice one item today and your streak begins!",
            "current_start": None,
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
            1 for i in range(current) if start + timedelta(days=i) not in active_set
        )
    else:
        start = None
        current = 0
        rest_days_current = 0

    best = max(historical_best, current)
    if current == 0:
        message = "Your previous streak ended. Practice today to start a new one."
    elif today_done:
        message = "Today done — lekker bezig!"
    elif gap <= REST_DAYS_ALLOWED:
        remaining = REST_DAYS_ALLOWED + 1 - gap
        message = f"Rest day — streak safe. {remaining} day{'s' if remaining != 1 else ''} of leeway left."
    else:
        message = "Last chance — practice today to keep your streak."

    last7 = []
    active_all = set(dates)
    for offset in range(6, -1, -1):
        d = today - timedelta(days=offset)
        if d in active_all:
            state = "active"
        elif start and d >= start and d <= today:
            state = "rest"
        else:
            state = "off"
        last7.append({"date": d, "state": state, "today": offset == 0})

    return {
        "current": current,
        "best": best,
        "today_done": today_done,
        "gap": gap,
        "rest_days_current": rest_days_current,
        "last7": last7,
        "message": message,
        "current_start": start,
    }


def init_usage_clock() -> None:
    if "usage_session_seconds" not in st.session_state:
        st.session_state.usage_session_seconds = 0.0
        st.session_state.usage_synced_session_seconds = 0.0
        st.session_state.usage_last_tick = time.monotonic()
        st.session_state.usage_prev_page = None


def tick_usage(current_page: str) -> str | None:
    init_usage_clock()
    now = time.monotonic()
    last = float(st.session_state.get("usage_last_tick", now))
    previous_page = st.session_state.get("usage_prev_page")
    delta = max(0.0, now - last)
    # Count time spent on the Practice screen between interactions, but cap long
    # idle gaps so an abandoned browser tab does not inflate the total.
    if previous_page == "Practice" and delta > 0:
        st.session_state.usage_session_seconds += min(delta, 300.0)
    st.session_state.usage_last_tick = now
    st.session_state.usage_prev_page = current_page
    return previous_page


def _int_setting(key: str) -> int:
    try:
        return int(float(get_setting(key, 0)))
    except Exception:
        return 0


def usage_display() -> dict[str, int]:
    cache = _settings_cache()
    today = local_today()
    day_key = f"usage_day:{today.isoformat()}"
    synced = float(st.session_state.get("usage_synced_session_seconds", 0.0))
    session_secs = float(st.session_state.get("usage_session_seconds", 0.0))
    pending = max(0, int(session_secs - synced))

    cloud_total = _int_setting("usage_total_seconds")
    cloud_today = _int_setting(day_key)
    week_total = 0
    for i in range(7):
        d = today - timedelta(days=i)
        week_total += _int_setting(f"usage_day:{d.isoformat()}")
    week_total += pending

    return {
        "today": cloud_today + pending,
        "week": week_total,
        "total": cloud_total + pending,
    }


def prepare_usage_sync() -> list[dict[str, Any]]:
    """Fold unsynced practice seconds into two tiny settings rows."""
    init_usage_clock()
    session_secs = int(st.session_state.usage_session_seconds)
    synced = int(st.session_state.usage_synced_session_seconds)
    delta = max(0, session_secs - synced)
    if delta <= 0:
        return []

    today = local_today()
    day_key = f"usage_day:{today.isoformat()}"
    new_total = _int_setting("usage_total_seconds") + delta
    new_today = _int_setting(day_key) + delta
    cache = _settings_cache()
    cache["usage_total_seconds"] = new_total
    cache[day_key] = new_today
    st.session_state.usage_synced_session_seconds = session_secs
    return [
        {"key": "usage_total_seconds", "value": new_total},
        {"key": day_key, "value": new_today},
    ]


def _format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins:02d}m" if mins else f"{hours}h"


def render_activity_banner() -> None:
    try:
        snap = session_activity_snapshot()
        summary = streak_summary(snap["dates"])
        usage = usage_display()
    except Exception as exc:
        st.caption(f"Activity summary temporarily unavailable: {exc}")
        return

    dots = []
    for d in summary["last7"]:
        cls = d["state"]
        if d["today"]:
            cls += " today"
        dots.append(f'<span class="trainer-dot {cls}" title="{d["date"].isoformat()}"></span>')

    current = int(summary["current"])
    best = int(summary["best"])
    rests = int(summary["rest_days_current"])
    flame = "🔥" if summary["today_done"] else ("🧊" if current else "🌱")

    st.markdown(
        f"""
        <div class="activity-card">
          <div class="activity-top">
            <div class="streak-pill">{flame} <strong>{current}</strong> DAYS</div>
            <div class="trainer-dots">{''.join(dots)}</div>
          </div>
          <div class="activity-message">{html.escape(summary['message'])}</div>
          <div class="activity-stats">
            <div><strong>{_format_duration(usage['today'])}</strong><span>today</span></div>
            <div><strong>{_format_duration(usage['week'])}</strong><span>this week</span></div>
            <div><strong>{_format_duration(usage['total'])}</strong><span>total</span></div>
            <div><strong>{snap['review_count']}</strong><span>answers</span></div>
          </div>
          <div class="activity-foot">Best streak: {best} days · {rests} rest day{'s' if rests != 1 else ''} in current streak · up to {REST_DAYS_ALLOWED} consecutive rest days allowed</div>
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
                    [{"item_id": item_id, "status": "new", "due_on": local_today().isoformat()}],
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


def font_scale() -> float:
    try:
        value = float(get_setting("font_scale", 1.0))
    except Exception:
        value = 1.0
    choices = [0.90, 1.00, 1.15, 1.30]
    return min(choices, key=lambda x: abs(x - value))


def apply_app_css(scale: float) -> None:
    st.markdown(
        f"""
        <style>
          :root {{ --trainer-scale: {scale}; color-scheme:light !important; }}
          html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            color-scheme:light !important;
          }}
          .stApp {{ background:#f5f8fb; color:#173a68 !important; }}
          .block-container {{ max-width:760px; padding-top:1.05rem; padding-bottom:2.5rem; }}
          #MainMenu, footer {{ visibility:hidden; }}
          header[data-testid="stHeader"] {{ background:transparent; }}

          /* Android may inherit Streamlit's dark widget palette even though this app
             deliberately uses a light background. Force readable widget colours. */
          [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
          [data-testid="stRadio"] label, [data-testid="stRadio"] label p,
          [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
            color:#426b98 !important; -webkit-text-fill-color:#426b98 !important;
          }}
          [data-testid="stRadio"] svg {{ color:#4773a6 !important; }}
          [data-baseweb="input"], [data-baseweb="textarea"],
          .stTextInput input, .stTextArea textarea {{
            background:#ffffff !important; color:#173a68 !important;
            -webkit-text-fill-color:#173a68 !important;
          }}
          .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
            color:#9aaabd !important; -webkit-text-fill-color:#9aaabd !important; opacity:1 !important;
          }}
          button[kind="secondary"] {{
            background:#ffffff !important; color:#426b98 !important;
            border-color:#cbd9e8 !important;
          }}
          button[kind="primary"] {{
            background:#173a68 !important; color:#ffffff !important;
            border-color:#173a68 !important;
          }}
          button[kind="secondary"] p, button[kind="primary"] p {{ color:inherit !important; -webkit-text-fill-color:inherit !important; }}
          button:disabled {{ opacity:.48 !important; }}

          .trainer-kicker {{ color:#4773a6; letter-spacing:.18em; font-size:.78rem; font-weight:800; margin-bottom:.15rem; }}
          .trainer-title {{ color:#173a68; font-family:Georgia, 'Times New Roman', serif; font-size:{2.55 * scale:.3f}rem; line-height:.95; margin-bottom:.65rem; }}
          .trainer-sub {{ color:#6c829e; font-size:{0.95 * scale:.3f}rem; margin-bottom:.2rem; }}

          .activity-card {{ background:rgba(255,255,255,.88); border:1px solid #d9e3ef; border-radius:18px; padding:14px 16px; margin:8px 0 14px; box-shadow:0 5px 18px rgba(44,77,116,.06); }}
          .activity-top {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }}
          .streak-pill {{ border:1px solid #f0cda9; background:#fff7ef; color:#c96f18; border-radius:999px; padding:7px 13px; font-size:{0.90 * scale:.3f}rem; letter-spacing:.05em; }}
          .trainer-dots {{ display:flex; gap:7px; align-items:center; }}
          .trainer-dot {{ width:12px; height:12px; border-radius:50%; display:inline-block; border:1px solid #c7d6e6; background:#edf3f8; }}
          .trainer-dot.active {{ background:#4773a6; border-color:#4773a6; }}
          .trainer-dot.rest {{ background:#fff2dc; border-color:#e6c58d; }}
          .trainer-dot.today {{ outline:2px solid #173a68; outline-offset:2px; }}
          .activity-message {{ color:#526d8d; margin-top:8px; font-size:{0.92 * scale:.3f}rem; }}
          .activity-stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:12px; text-align:center; }}
          .activity-stats div {{ border-top:1px solid #e3ebf3; padding-top:9px; }}
          .activity-stats strong {{ display:block; color:#173a68; font-size:{1.08 * scale:.3f}rem; }}
          .activity-stats span {{ display:block; color:#7590ad; font-size:{0.74 * scale:.3f}rem; }}
          .activity-foot {{ color:#7890a8; font-size:{0.76 * scale:.3f}rem; margin-top:9px; text-align:center; }}

          .cue-card {{ background:#fff; border:1px solid #d6e1ed; border-radius:12px; padding:30px 18px; margin:12px 0 14px; box-shadow:0 8px 22px rgba(44,77,116,.07); text-align:center; }}
          .cue-kicker {{ color:#7893b0; letter-spacing:.16em; font-size:{0.76 * scale:.3f}rem; font-weight:700; }}
          .cue-text {{ color:#173a68; font-family:Georgia, 'Times New Roman', serif; font-size:{2.05 * scale:.3f}rem; line-height:1.2; margin-top:12px; overflow-wrap:anywhere; }}
          .stage-line {{ color:#66819f; font-size:{0.80 * scale:.3f}rem; letter-spacing:.04em; margin:.3rem 0 .25rem; }}
          .compact-stats {{ color:#65809e; text-align:center; font-size:{0.82 * scale:.3f}rem; margin-top:12px; }}
          .sync-line {{ color:#7890a8; text-align:center; font-size:{0.76 * scale:.3f}rem; margin-top:3px; }}
          .keyboard-hint {{ color:#8297ae; text-align:center; font-size:.72rem; margin:-.15rem 0 .25rem; }}

          .stTextInput input, .stTextArea textarea {{ font-size:{1.05 * scale:.3f}rem !important; }}
          .stButton button, .stFormSubmitButton button {{ font-size:{0.95 * scale:.3f}rem !important; border-radius:10px !important; min-height:2.7rem; }}
          [data-testid="stRadio"] label p {{ font-size:{0.88 * scale:.3f}rem !important; }}
          div[data-testid="stRadio"] > div {{ gap:.45rem; flex-wrap:wrap; }}
          .stCaptionContainer, [data-testid="stCaptionContainer"] {{ color:#7088a3 !important; font-size:{0.80 * scale:.3f}rem !important; }}

          /* Keep the small A− / A+ controls in one row on narrow Android screens. */
          .st-key-font_controls [data-testid="stHorizontalBlock"] {{ flex-wrap:nowrap !important; align-items:center !important; }}
          .st-key-font_controls [data-testid="stColumn"] {{ min-width:0 !important; width:auto !important; }}

          @media (max-width:560px) {{
            .block-container {{ padding-left:.7rem; padding-right:.7rem; padding-top:.8rem; }}
            .trainer-title {{ font-size:{2.15 * scale:.3f}rem; }}
            .trainer-sub {{ font-size:{0.88 * scale:.3f}rem; }}
            .activity-stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
            .cue-card {{ padding:22px 12px; }}
            .cue-text {{ font-size:{1.75 * scale:.3f}rem; }}
            [data-testid="stRadio"] label p {{ font-size:{0.82 * scale:.3f}rem !important; }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def practice_type_set(mode: str) -> set[str] | None:
    return {
        "Words": {"word"},
        "Phrases": {"phrase"},
        "Sentences": {"sentence"},
    }.get(mode)


def load_practice(mode: str) -> None:
    queue, stats = practice_candidates(40, practice_type_set(mode))

    # Resume at the last unfinished card if it is still due and still belongs to
    # this practice mode. The remainder of the queue can be rebuilt safely:
    # already answered cards have their next due date in Supabase and therefore
    # do not suddenly reappear when the app is reopened.
    saved_mode = str(get_setting("practice_resume_mode", mode) or mode)
    saved_item_id = str(get_setting("practice_resume_item_id", "") or "")
    if saved_mode == mode and saved_item_id:
        for pos, queued in enumerate(queue):
            if str(queued.get("id")) == saved_item_id:
                queue = queue[pos:] + queue[:pos]
                break

    st.session_state.practice_queue = queue
    st.session_state.practice_stats = stats
    st.session_state.practice_index = 0
    st.session_state.practice_feedback = None
    st.session_state.practice_loaded_mode = mode
    next_id = str(queue[0].get("id")) if queue else None
    save_resume_state_async(mode, next_id)


def advance_practice(mode: str) -> None:
    """Advance before the rerun so clicking Next needs only one Streamlit pass."""
    queue = st.session_state.get("practice_queue", [])
    idx = int(st.session_state.get("practice_index", 0)) + 1
    st.session_state.practice_index = idx
    st.session_state.practice_feedback = None
    next_id = str(queue[idx].get("id")) if idx < len(queue) else None
    save_resume_state_async(mode, next_id)


def install_keyboard_shortcuts() -> None:
    """Laptop shortcuts without changing the phone UI.

    Enter inside the answer form is handled natively by Streamlit and submits
    the first form button (Check). This tiny browser-side listener adds Escape
    for “I don't know” and Enter for “Next” only when those buttons are visible.
    """
    components.html(
        """
        <script>
        (() => {
          try {
            const w = window.parent;
            const d = w.document;
            if (w.__dutchTrainerShortcutHandler) return;
            const visibleButton = (label) => Array.from(d.querySelectorAll('button')).find((b) => {
              const text = (b.innerText || '').trim();
              const visible = !!(b.offsetWidth || b.offsetHeight || b.getClientRects().length);
              return visible && text === label && !b.disabled;
            });
            w.__dutchTrainerShortcutHandler = (e) => {
              if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
              const active = d.activeElement;
              const tag = active && active.tagName ? active.tagName.toLowerCase() : '';
              if (e.key === 'Escape') {
                const btn = visibleButton("I don't know");
                if (btn) { e.preventDefault(); e.stopPropagation(); btn.click(); }
                return;
              }
              if (e.key === 'Enter') {
                // While typing, let the Streamlit form submit Check normally.
                if (tag === 'input' || tag === 'textarea' || tag === 'button') return;
                const btn = visibleButton('Next →') || visibleButton('Next');
                if (btn) { e.preventDefault(); e.stopPropagation(); btn.click(); }
              }
            };
            d.addEventListener('keydown', w.__dutchTrainerShortcutHandler, true);
          } catch (err) {
            // Keyboard shortcuts are optional; the visible buttons always remain.
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


require_password()

if not configured():
    st.error("Supabase is not configured yet. Add SUPABASE_URL and SUPABASE_SERVICE_KEY to Streamlit secrets.")
    st.stop()

# Do the network diagnostics and one-time migration only once per browser session,
# not after every answer.
if not st.session_state.get("trainer_connection_checked"):
    try:
        supa_get("trainer_items", {"select": "id", "limit": "1"})
        st.session_state.trainer_connection_checked = True
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 401:
            st.error("Supabase rejected the API key (HTTP 401). Check SUPABASE_SERVICE_KEY in Streamlit Secrets, then save and reboot the app.")
        elif status == 404:
            st.error("Supabase connected, but trainer_items was not found (HTTP 404). The trainer schema may not have been created in this project.")
        else:
            st.error(f"Supabase connection failed (HTTP {status}). Open Manage app → Logs for details.")
        st.stop()

_settings_cache()
if not st.session_state.get("scheduler_migration_checked"):
    try:
        migrate_scheduler_v2()
    except Exception as exc:
        st.session_state.scheduler_migration_note = str(exc)
    st.session_state.scheduler_migration_checked = True

current_scale = font_scale()
apply_app_css(current_scale)

# Header + font-size controls. Keep the title simple, then place the controls
# in a dedicated row so Android does not stack A− and A+ into giant buttons.
st.markdown(
    """
    <div class="trainer-kicker">OEFENEN</div>
    <div class="trainer-title">Dutch word trainer</div>
    <div class="trainer-sub">Words · phrases · sentences · long-term review</div>
    """,
    unsafe_allow_html=True,
)
scales = [0.90, 1.00, 1.15, 1.30]
scale_idx = min(range(len(scales)), key=lambda i: abs(scales[i] - current_scale))
with st.container(key="font_controls"):
    spacer, a1, a2 = st.columns([7, 1, 1], gap="small")
    with a1:
        if st.button("A−", key="font_minus", use_container_width=True, disabled=scale_idx == 0):
            set_setting("font_scale", scales[max(0, scale_idx - 1)])
            st.rerun()
    with a2:
        if st.button("A+", key="font_plus", use_container_width=True, disabled=scale_idx == len(scales) - 1):
            set_setting("font_scale", scales[min(len(scales) - 1, scale_idx + 1)])
            st.rerun()

# Radio navigation is intentionally conditional instead of st.tabs. Streamlit
# executes every tab body on each rerun; that was causing unnecessary Supabase
# reads while practising.
page = st.radio(
    "Section",
    ["Practice", "Generate", "Add", "Library", "Progress"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav",
)

previous_page = tick_usage(page)
if previous_page == "Practice" and page != "Practice":
    # Save the final bit of practice time when leaving the practice screen.
    usage_rows = prepare_usage_sync()
    if usage_rows:
        _pending_syncs().append(
            SYNC_WRITER.submit(supa_post, "trainer_settings", usage_rows, True, False)
        )
drain_pending_syncs()
render_activity_banner()

if st.session_state.get("scheduler_migration_note"):
    st.caption(f"Learning-schedule migration skipped: {st.session_state.scheduler_migration_note}")


# Practice --------------------------------------------------------------------
if page == "Practice":
    current_target = mastery_target()
    install_keyboard_shortcuts()

    mode_options = ["Everything", "Words", "Phrases", "Sentences"]
    if "practice_mode" not in st.session_state:
        saved_mode = str(get_setting("practice_resume_mode", "Everything") or "Everything")
        st.session_state.practice_mode = saved_mode if saved_mode in mode_options else "Everything"

    mode = st.radio(
        "Practise",
        mode_options,
        horizontal=True,
        label_visibility="collapsed",
        key="practice_mode",
    )

    target_col, refresh_col = st.columns([3.4, 1])
    with target_col:
        chosen_target = st.radio(
            "Retire after",
            [3, 5],
            index=0 if current_target == 3 else 1,
            horizontal=True,
            format_func=lambda x: f"{x} correct",
            key="mastery_target_choice",
        )
        if int(chosen_target) != current_target:
            set_setting("mastery_target", int(chosen_target))
            current_target = int(chosen_target)
            # Existing local queue can keep going; the next answer uses the new target.
    with refresh_col:
        refresh = st.button("Refresh", use_container_width=True)

    st.caption(
        "Only one clean recall per calendar day advances mastery. Retired items return after "
        "45 → 90 → 180 → 365 days; if you forget one, it returns to learning."
    )
    st.markdown(
        "<div class='keyboard-hint'>Laptop: Enter = Check / Next · Esc = I don't know</div>",
        unsafe_allow_html=True,
    )

    if "practice_queue" not in st.session_state:
        st.session_state.practice_queue = []
        st.session_state.practice_stats = {"in_progress": 0, "retired": 0, "due": 0, "total": 0}
        st.session_state.practice_index = 0
        st.session_state.practice_feedback = None
        st.session_state.practice_loaded_mode = None

    if refresh or st.session_state.get("practice_loaded_mode") != mode:
        with st.spinner("Loading practice…"):
            load_practice(mode)

    queue = st.session_state.practice_queue
    idx = int(st.session_state.practice_index)
    stats = st.session_state.practice_stats

    if not queue:
        st.info(f"Nothing is due in {mode.lower()} right now. You can generate/add material or choose another practice type.")
        if st.button("Check again", type="primary"):
            load_practice(mode)
            st.rerun()
    elif idx >= len(queue):
        st.success("Session finished.")
        if st.button("Start another session", type="primary", use_container_width=True):
            load_practice(mode)
            st.rerun()
    else:
        item = queue[idx]
        cue = str(item.get("english", ""))
        item_progress = dict(item.get("_progress") or _default_progress(str(item["id"])))
        item_retired = not bool(item.get("active", True)) or item_progress.get("status") == "mastered"
        if item_retired:
            stage = f"Retired review · {int(item_progress.get('interval_days') or 0)}-day interval"
        else:
            stage = f"{int(item_progress.get('consecutive_correct') or 0)}/{current_target} recalls toward retirement"

        st.markdown(
            f'<div class="stage-line">{html.escape(str(item.get("item_type", "item")).upper())} · '
            f'{html.escape(str(item.get("level", "")))} · {html.escape(stage)} · {idx + 1}/{len(queue)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="cue-card"><div class="cue-kicker">SAY IT IN DUTCH</div>'
            f'<div class="cue-text">{html.escape(cue)}</div></div>',
            unsafe_allow_html=True,
        )

        fb = st.session_state.practice_feedback
        if not fb:
            placeholder = {
                "word": "type the Dutch word",
                "phrase": "type the Dutch phrase",
                "sentence": "type the Dutch sentence",
            }.get(str(item.get("item_type")), "type the Dutch")

            with st.form(f"answer_form_{item['id']}_{idx}", clear_on_submit=False):
                typed = st.text_input(
                    "Type the Dutch",
                    placeholder=placeholder,
                    label_visibility="collapsed",
                    autocomplete="off",
                )
                b1, b2 = st.columns(2)
                submit = b1.form_submit_button("Check", type="primary", use_container_width=True)
                dont = b2.form_submit_button("I don't know", use_container_width=True)

            if submit or dont:
                grade = "dont_know" if dont else classify_answer(item, typed)
                result = save_review_async(item, typed, grade, cue, current_target)
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

                # Real mistakes return after a few other cards. No second server rerun
                # is forced here; feedback renders immediately in this same run.
                if grade in {"wrong", "dont_know"}:
                    insert_at = min(len(queue), idx + 4)
                    queue.insert(insert_at, item)
                elif transition in {"retire", "retired_pass", "retired_typo"}:
                    queue = queue[:idx + 1] + [
                        q for q in queue[idx + 1:] if str(q.get("id")) != str(item.get("id"))
                    ]
                st.session_state.practice_queue = queue
                # The current card has now been answered. Persist the following
                # unfinished card so reopening the app continues from there.
                next_id = str(queue[idx + 1].get("id")) if idx + 1 < len(queue) else None
                save_resume_state_async(mode, next_id)
                fb = st.session_state.practice_feedback

        if fb:
            labels = {
                "correct": "✅ Correct",
                "typo": "⌨️ Small typing mistake — treated gently",
                "wrong": "❌ Not quite",
                "dont_know": "🌱 Not known yet",
            }
            st.markdown(f"### {labels.get(fb['grade'], fb['grade'])}")
            transition = fb.get("transition", "")
            if transition == "retire":
                st.caption(
                    f"Learned after {fb.get('target')} spaced correct recalls. "
                    f"Next long-term memory check in {fb.get('interval_after')} days."
                )
            elif transition == "retired_pass":
                st.caption(f"Long-term memory confirmed. Next review in {fb.get('interval_after')} days.")
            elif transition == "retired_typo":
                st.caption(f"Still retired; a closer check is scheduled in {fb.get('interval_after')} days.")
            elif transition == "relearn":
                st.caption("This retired item was forgotten, so it has returned to the learning pool.")
            elif fb.get("grade") == "correct":
                st.caption(f"Memory strength: {fb.get('successes')}/{fb.get('target')} spaced correct recalls.")

            st.markdown(f"**Dutch:** {html.escape(str(item.get('dutch', '')))}")
            if item.get("example_nl"):
                st.caption(str(item.get("example_nl")))
            pronunciation_box(str(item.get("dutch", "")), f"practice-{idx}")

            st.button(
                "Next →",
                type="primary",
                use_container_width=True,
                on_click=advance_practice,
                args=(mode,),
            )

    usage = usage_display()
    pending_n, _ = drain_pending_syncs()
    sync_text = "☁️ Saving…" if pending_n else "☁️ Synced"
    st.markdown(
        f'<div class="compact-stats"><strong>{stats.get("in_progress", 0)}</strong> in progress · '
        f'<strong>{stats.get("retired", 0)}</strong> retired · '
        f'<strong>{_format_duration(usage["today"])}</strong> today · '
        f'<strong>{_format_duration(usage["total"])}</strong> total</div>'
        f'<div class="sync-line">{sync_text}</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.get("sync_errors"):
        st.warning("A background cloud save failed. Your current screen still has the answer; refresh before continuing if this repeats.")
        st.session_state.sync_errors = []


# Generate --------------------------------------------------------------------
elif page == "Generate":
    st.subheader("Generate new material")
    st.caption("AI is used only when you press Generate. Ordinary practice, scoring, streaks and reviews do not call OpenAI.")

    if not OPENAI_KEY:
        st.warning("Add OPENAI_API_KEY to Streamlit secrets to enable generation.")
    else:
        level = st.selectbox("Level", ["A2", "B1", "B2"], index=1)
        topic = st.text_input("Topic", value="everyday Dutch")
        kinds = st.multiselect(
            "Include",
            ["word", "phrase", "sentence"],
            default=["word", "phrase", "sentence"],
        )
        count = st.slider("How many", 3, 20, 8)
        if st.button("Generate", type="primary"):
            if not kinds:
                st.error("Choose at least one item type.")
            else:
                try:
                    with st.spinner("Generating…"):
                        st.session_state.generated_preview = generate_items(level, topic, count, kinds)
                    st.success(f"Generated {len(st.session_state.generated_preview)} items. Review them before saving.")
                except Exception as exc:
                    st.error(f"Generation failed: {exc}")

        preview = st.session_state.get("generated_preview", [])
        if preview:
            chosen_ids = []
            for i, item in enumerate(preview):
                label = f"{item['item_type'].title()}: {item['dutch']} — {item['english']}"
                if st.checkbox(label, value=True, key=f"gen_keep_{i}"):
                    chosen_ids.append(i)
            if st.button("Save selected to trainer", type="primary"):
                rows = [preview[i] for i in chosen_ids]
                try:
                    added, skipped = insert_items(rows)
                    st.success(f"Saved {added} item(s)." + (f" Skipped duplicates: {', '.join(skipped)}" if skipped else ""))
                    st.session_state.generated_preview = []
                    st.rerun()
                except Exception as exc:
                    st.error(f"Saving failed: {exc}")


# Add -------------------------------------------------------------------------
elif page == "Add":
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
            except Exception as exc:
                st.error(f"Could not add item: {exc}")


# Library ---------------------------------------------------------------------
elif page == "Library":
    st.subheader("Library")
    st.caption("Retired items stay out of ordinary practice until their sparse long-term review is due.")
    try:
        items = fetch_items(include_inactive=True)
        library_progress = fetch_progress()
        library_target = mastery_target()
    except Exception as exc:
        st.error(f"Could not load library: {exc}")
        items = []
        library_progress = {}
        library_target = 5

    f1, f2 = st.columns([2, 1])
    search = f1.text_input("Search Dutch / English", key="library_search")
    state_view = f2.selectbox("Status", ["Active", "Retired", "All"], key="library_status")

    f3, f4 = st.columns(2)
    typ = f3.selectbox("Type", ["all", "word", "phrase", "sentence"], key="library_type")
    level_view = f4.selectbox("Level", ["all", "A1", "A2", "B1", "B2", "C1"], key="library_level")

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
    st.caption(f"{len(filtered)} shown · {active_n} active · {retired_n} retired · {len(items)} total")

    PAGE_SIZE = 25
    pages_n = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    page_no = st.selectbox(
        "Page",
        list(range(1, pages_n + 1)),
        index=0,
        key=f"library_page_{state_view}_{typ}_{level_view}_{q}",
    )
    lo = (int(page_no) - 1) * PAGE_SIZE
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
                            {"status": "new", "due_on": local_today().isoformat(), "interval_days": 0, "consecutive_correct": 0},
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not return item to learning: {exc}")


# Progress --------------------------------------------------------------------
elif page == "Progress":
    st.subheader("Progress")
    try:
        items = fetch_items(include_inactive=True)
        prog = fetch_progress()
    except Exception as exc:
        st.error(f"Could not load progress: {exc}")
        items, prog = [], {}

    counts = {"new": 0, "learning": 0, "familiar": 0}
    due = 0
    retired_due = 0
    retired = 0
    today_s = local_today().isoformat()
    for item in items:
        p = prog.get(str(item["id"]))
        if not item.get("active", True):
            retired += 1
            if p and str(p.get("due_on") or today_s) <= today_s:
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
        if str(p.get("due_on") or today_s) <= today_s:
            due += 1

    cols = st.columns(4)
    cols[0].metric("New", counts["new"])
    cols[1].metric("Learning", counts["learning"])
    cols[2].metric("Familiar", counts["familiar"])
    cols[3].metric("Retired", retired)
    st.metric("Due now", due)
    if retired_due:
        st.caption(f"{retired_due} due item(s) are long-term reviews of retired material.")

    snap = session_activity_snapshot()
    usage = usage_display()
    st.write(
        f"All-time answers: {snap['review_count']} · Today: {_format_duration(usage['today'])} · "
        f"This week: {_format_duration(usage['week'])} · Total practice: {_format_duration(usage['total'])}"
    )
    st.caption(
        f"Current mastery rule: retire after {mastery_target()} spaced correct recalls; "
        "retired reviews expand to 45 → 90 → 180 → 365 days when remembered."
    )
