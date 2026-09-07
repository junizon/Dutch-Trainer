from __future__ import annotations

import html
import json
import random
import re
from datetime import date, timedelta
from difflib import SequenceMatcher
from typing import Any
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


def configured() -> bool:
    return bool(SUPA_URL and SUPA_KEY)


def headers(prefer: str | None = None) -> dict[str, str]:
    h = {
        "apikey": SUPA_KEY,
        "Authorization": f"Bearer {SUPA_KEY}",
        "Content-Type": "application/json",
    }
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


INTERVALS = [1, 3, 7, 14, 30, 60, 120]


def schedule_after(progress: dict[str, Any], grade: str) -> tuple[dict[str, Any], int]:
    before = int(progress.get("interval_days") or 0)
    attempts = int(progress.get("attempts") or 0) + 1
    correct = int(progress.get("correct") or 0)
    typo = int(progress.get("typo") or 0)
    wrong = int(progress.get("wrong") or 0)
    dont = int(progress.get("dont_know") or 0)
    streak = int(progress.get("consecutive_correct") or 0)
    difficulty = int(progress.get("difficulty") or 0)

    if grade == "correct":
        correct += 1
        streak += 1
        difficulty = max(0, difficulty - 1)
        if before <= 0:
            after = 1
        else:
            after = next((x for x in INTERVALS if x > before), INTERVALS[-1])
    elif grade == "typo":
        typo += 1
        streak = max(0, streak - 1)
        after = max(1, min(before or 1, 3))
    elif grade == "dont_know":
        dont += 1
        streak = 0
        difficulty += 2
        after = 1
    else:
        wrong += 1
        streak = 0
        difficulty += 1
        after = 1

    if after >= 30 and streak >= 3:
        status = "mastered"
    elif after >= 7 and streak >= 2:
        status = "familiar"
    elif attempts > 0:
        status = "learning"
    else:
        status = "new"

    values = {
        "status": status,
        "due_on": (date.today() + timedelta(days=after)).isoformat(),
        "interval_days": after,
        "difficulty": difficulty,
        "attempts": attempts,
        "correct": correct,
        "typo": typo,
        "wrong": wrong,
        "dont_know": dont,
        "consecutive_correct": streak,
        "last_grade": grade,
        "last_reviewed_at": "now()",
    }
    return values, before


def save_review(item: dict[str, Any], answer: str, grade: str, prompt: str) -> None:
    progress = ensure_progress(str(item["id"]))
    values, before = schedule_after(progress, grade)
    after = int(values["interval_days"])

    # PostgREST cannot interpret now() as SQL inside JSON, so omit the field and
    # let updated_at remain server-side; reviewed_at is defaulted in reviews.
    values.pop("last_reviewed_at", None)
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


def practice_candidates(limit: int = 40) -> list[dict[str, Any]]:
    items = fetch_items()
    prog = fetch_progress()
    today = date.today().isoformat()

    scored: list[tuple[tuple[int, int, str], dict[str, Any]]] = []
    for item in items:
        p = prog.get(str(item["id"]))
        if not p:
            pri = (1, 0, item.get("created_at", ""))  # new comes after due reviews
        else:
            due = str(p.get("due_on") or today)
            if due > today:
                continue
            difficulty = int(p.get("difficulty") or 0)
            pri = (0, -difficulty, due)
        scored.append((pri, item))

    scored.sort(key=lambda x: x[0])
    due_items = [x[1] for x in scored]
    if len(due_items) > limit:
        due_items = due_items[:limit]
    return due_items


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
                        "chinese": {"type": "string"},
                        "example_nl": {"type": "string"},
                        "example_en": {"type": "string"},
                        "accepted_answers": {"type": "array", "items": {"type": "string"}},
                        "theme": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "item_type", "dutch", "english", "chinese", "example_nl",
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
- Chinese should be Traditional Chinese; leave it an empty string only if a concise natural equivalent is genuinely awkward.
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
            "chinese": str(raw.get("chinese", "")).strip(),
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

if not configured():
    st.error("Supabase is not configured yet. Add SUPABASE_URL and SUPABASE_SERVICE_KEY to Streamlit secrets.")
    st.stop()

pages = st.tabs(["Practice", "Generate", "Add", "Library", "Progress"])


# Practice --------------------------------------------------------------------
with pages[0]:
    st.subheader("Practice")

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
        english = item.get("english", "")
        chinese = item.get("chinese", "")
        cue = english
        if chinese:
            cue += f"\n\n{chinese}"

        st.caption(f"{item.get('item_type','item').title()} · {item.get('level','')} · {idx + 1}/{len(queue)}")
        st.markdown(f"### {html.escape(cue)}")

        answer_key = f"ans_{item['id']}_{idx}"
        typed = st.text_input("Type the Dutch", key=answer_key, autocomplete="off")

        b1, b2 = st.columns(2)
        submit = b1.button("Check", type="primary", use_container_width=True, disabled=bool(st.session_state.practice_feedback))
        dont = b2.button("I don't know", use_container_width=True, disabled=bool(st.session_state.practice_feedback))

        if submit or dont:
            grade = "dont_know" if dont else classify_answer(item, typed)
            save_review(item, typed, grade, cue)
            st.session_state.practice_feedback = {
                "grade": grade,
                "typed": typed,
                "correct": item.get("dutch", ""),
            }
            # Real mistakes return later in the same session after a few cards.
            if grade in {"wrong", "dont_know"}:
                insert_at = min(len(queue), idx + 4)
                queue.insert(insert_at, item)
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
                if row.get("chinese"):
                    st.caption(row["chinese"])
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
        chinese = st.text_input("Traditional Chinese cue (optional)")
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
                "chinese": chinese.strip(),
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
    try:
        items = fetch_items()
    except Exception as e:
        st.error(f"Could not load library: {e}")
        items = []

    search = st.text_input("Search Dutch / English", key="library_search")
    typ = st.selectbox("Show", ["all", "word", "phrase", "sentence"], key="library_type")
    filtered = []
    q = search.strip().lower()
    for item in items:
        if typ != "all" and item.get("item_type") != typ:
            continue
        hay = f"{item.get('dutch','')} {item.get('english','')} {item.get('theme','')}".lower()
        if q and q not in hay:
            continue
        filtered.append(item)

    st.caption(f"{len(filtered)} of {len(items)} active items")
    for item in filtered[:200]:
        with st.expander(f"{item.get('dutch','')} — {item.get('english','')}"):
            st.write(f"Type: {item.get('item_type')} · Level: {item.get('level')} · Theme: {item.get('theme')}")
            if item.get("chinese"):
                st.write(item["chinese"])
            if item.get("example_nl"):
                st.write(item["example_nl"])
            pronunciation_box(str(item.get("dutch", "")), f"lib-{item['id']}")
            if st.button("Archive", key=f"archive_{item['id']}"):
                try:
                    supa_patch("trainer_items", {"id": f"eq.{item['id']}"}, {"active": False})
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not archive: {e}")


# Progress --------------------------------------------------------------------
with pages[4]:
    st.subheader("Progress")
    try:
        items = fetch_items()
        prog = fetch_progress()
        reviews = supa_get("trainer_reviews", {"select": "id,grade,reviewed_at", "order": "reviewed_at.desc", "limit": "5000"})
    except Exception as e:
        st.error(f"Could not load progress: {e}")
        items, prog, reviews = [], {}, []

    counts = {"new": 0, "learning": 0, "familiar": 0, "mastered": 0}
    due = 0
    today = date.today().isoformat()
    for item in items:
        p = prog.get(str(item["id"]))
        if not p:
            counts["new"] += 1
            due += 1
            continue
        status = p.get("status", "new")
        counts[status] = counts.get(status, 0) + 1
        if str(p.get("due_on") or today) <= today:
            due += 1

    cols = st.columns(4)
    cols[0].metric("New", counts["new"])
    cols[1].metric("Learning", counts["learning"])
    cols[2].metric("Familiar", counts["familiar"])
    cols[3].metric("Mastered", counts["mastered"])
    st.metric("Due now", due)

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
