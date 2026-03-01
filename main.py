import os
import json
import uuid
import re
import io
import random
import unicodedata
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

from analysis_service import (
    AnalyzeResponse,
    AnalysisSummary,
    ANALYZE_FALLBACK,
    run_analysis,
)

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"


VALID_EMOTIONS = {"calm", "nervous", "angry", "sad", "defensive", "confident", "fearful"}
VALID_TONES = {"warm", "cold", "static"}

SUS_SCAN_FALLBACK: dict = {
    "narration": "The silence between words tells its own story.",
    "anomaly_delta": 0,
    "tone": "static",
    "reason_tags": [],
}

# Set ENABLE_ANALYSIS=true in .env to append a suspicion analysis to every
# interrogation response. Costs one extra Mistral call per question.
ENABLE_ANALYSIS: bool = os.getenv("ENABLE_ANALYSIS", "false").lower() == "true"

# ── Profanity filter (EN + ES) ────────────────────────────────────────────────

_PROFANITY: frozenset[str] = frozenset({
    # English
    "fuck", "fucking", "fucker", "fucked", "fucks", "motherfucker", "motherfucking",
    "shit", "shitty", "shithead", "bullshit", "dipshit",
    "ass", "asshole", "asses", "jackass", "dumbass", "smartass",
    "bitch", "bitches", "bitchy",
    "bastard", "bastards",
    "cunt", "cunts",
    "dick", "dicks", "dickhead",
    "cock", "cocks", "cocksucker",
    "pussy", "pussies",
    "whore", "whores",
    "slut", "slutty",
    "piss", "pissed", "pissoff",
    "faggot", "fag",
    "nigger", "nigga",
    "retard", "retarded",
    # Spanish
    "puta", "puto", "putas", "putos", "putada",
    "mierda", "mierdas",
    "cono",
    "joder", "jodes", "jodido", "jodida", "jodidos", "jodidas",
    "cabron", "cabrona", "cabrones", "cabronas",
    "pendejo", "pendeja", "pendejos", "pendejas",
    "chingada", "chingado", "chingados", "chinga", "chinguen", "chingon",
    "verga", "vergas",
    "culero", "culera", "culeros", "culeras", "culo",
    "maricon", "marica", "maricones",
    "perra", "perras",
    "pinche", "pinches",
    "hdp", "hijoputa", "hijosdeputa",
    "gilipollas",
    "hostia", "hostias",
    "cojones", "cojon",
    "mamada", "mamadas", "mamar",
    "putear", "puteado",
    "zorra", "zorras",
    "idiota",
})


def _normalise(text: str) -> str:
    """Lowercase, strip combining diacritics, replace non-letters with spaces."""
    nfd = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z]", " ", stripped)


def _has_profanity(text: str) -> bool:
    """Return True if any word in *text* is in the profanity list."""
    words = _normalise(text).split()
    return any(w in _PROFANITY for w in words)


def _build_pressure_guidance(is_real: bool, questions_used: int) -> str:
    """
    Returns a PRESSURE CONTEXT block that steers the model's emotion
    selection based on the suspect's role and how many questions have
    already been asked for this suspect in the current game.

    Impostors crack under mounting pressure → nervous / defensive / fearful.
    The real person stays grounded in truth → calm / sad / confident.
    """
    # questions_used is the count BEFORE this new question is processed,
    # so it ranges from 0 (first question) to MAX_QUESTIONS_PER_SUSPECT-1 (last).
    if is_real:
        if questions_used == 0:
            return (
                "PRESSURE CONTEXT: This is the first question of the interrogation. "
                "You are shaken by the unusual situation but still composed and cooperative. "
                "Preferred emotions for this moment: calm, sad, confident. "
                "Do NOT use angry — there is no reason to be hostile yet."
            )
        elif questions_used >= MAX_QUESTIONS_PER_SUSPECT - 1:
            return (
                "PRESSURE CONTEXT: This is the final question. "
                "You feel emotional exhaustion and desperation that no one believes you. "
                "Preferred emotions for this moment: sad, confident, defensive. "
                "angry is only acceptable if the question is directly accusatory."
            )
        else:
            return (
                "PRESSURE CONTEXT: The interrogation is deepening. "
                "You are determined to prove your identity; you feel misunderstood, not hostile. "
                "Preferred emotions for this moment: sad, confident, defensive. "
                "Do NOT use angry — frustration has not yet reached that level."
            )
    else:
        # impostor
        if questions_used == 0:
            return (
                "PRESSURE CONTEXT: This is the first question of the interrogation. "
                "Your cover story is solid and you feel fully in control. "
                "Preferred emotions for this moment: calm, confident. "
                "Do NOT use angry or nervous — you have nothing to fear yet."
            )
        elif questions_used >= MAX_QUESTIONS_PER_SUSPECT - 1:
            return (
                "PRESSURE CONTEXT: This is the final question — sustained pressure is wearing you down. "
                "Your composure is cracking as the deception becomes harder to maintain. "
                "Preferred emotions for this moment: nervous, defensive, fearful. "
                "Do NOT use angry — panic, not rage, is the natural response."
            )
        else:
            return (
                "PRESSURE CONTEXT: The pressure is gradually building. "
                "Maintaining the deception takes concentration; small cracks are forming. "
                "Preferred emotions for this moment: nervous, defensive. "
                "Do NOT use angry — irritation would draw unwanted attention."
            )

BASE_QUESTIONS = 5
EXTRA_QUESTIONS = 2
MAX_QUESTIONS_PER_SUSPECT = BASE_QUESTIONS + EXTRA_QUESTIONS
# Voz distinta para cada sospechoso
SUSPECT_VOICES = {
    "1": "pNInz6obpgDQGcFmaJgB",  # Adam - voz grave masculina
    "2": "EXAVITQu4vr4xnSDxMaL",  # Sarah - voz femenina
    "3": "VR6AewLTigWG4xSOukaG",  # Arnold - voz seria
}

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="DoppelMind API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory game store
games: dict[str, Any] = {}

# ── Game data validation & fallback ──────────────────────────────────────────

# Required case fields with generic fallbacks (used only if generation fails)
_CASE_REQUIRED = ["title", "description", "setting", "victim", "time", "crime"]

def _nonempty(value: Any) -> bool:
    """Return True if value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def _validate_game_data(data: dict) -> list[str]:
    """Return a list of missing-or-empty required field paths."""
    errors: list[str] = []

    case = data.get("case") or {}
    for field in _CASE_REQUIRED:
        if not _nonempty(case.get(field)):
            errors.append(f"case.{field}")

    suspects = data.get("suspects") or []
    if len(suspects) != 3:
        errors.append(f"suspects.count={len(suspects)} (expected 3)")
    for s in suspects:
        for field in (
    "id",
    "name",
    "age",
    "occupation",
    "relationship_to_victim",
    "appearance",
    "initial_statement",
    "personality",
    "hidden_knowledge",
    "is_real",
):
            if not isinstance(s.get("age"), int):
                errors.append(f"suspects[{s.get('id','?')}].age")
            alibi = s.get("alibi")
            if not isinstance(alibi, dict) or not isinstance(alibi.get("cooperative"), bool) or not _nonempty(alibi.get("statement")):
                errors.append(f"suspects[{s.get('id','?')}].alibi")
            
            if field == "is_real":
                if not isinstance(s.get(field), bool):
                    errors.append(f"suspects[{s.get('id','?')}].{field}")
            elif not _nonempty(str(s.get(field, ""))):
                errors.append(f"suspects[{s.get('id','?')}].{field}")

    requirements = data.get("requirements") or []
    if len(requirements) < 5:
        errors.append(f"requirements.count={len(requirements)} (expected 5)")

    if not _nonempty(data.get("solution", "")):
        errors.append("solution")

    return errors


def _normalize_suspects(data: dict) -> None:
    """
    Enforce a consistent suspects array:
    - exactly 3 suspects
    - unique ids
    - defined names
    - exactly one is_real=True
    """
    raw_suspects = data.get("suspects")
    suspects: list[dict] = [dict(s) for s in raw_suspects if isinstance(s, dict)] if isinstance(raw_suspects, list) else []

    inferred_name = next(
        (
            str(s.get("name", "")).strip()
            for s in suspects
            if _nonempty(str(s.get("name", "")))
        ),
        "Identidad desconocida",
    )
    template = suspects[0] if suspects else {}

    while len(suspects) < 3:
        suspects.append({
            "id": str(len(suspects) + 1),
            "name": inferred_name,
            "appearance": template.get("appearance", "Sin descripción."),
            "personality": template.get("personality", "Comportamiento reservado."),
            "hidden_knowledge": template.get("hidden_knowledge", "Recuerdos incompletos."),
            "initial_statement": template.get("initial_statement", "Soy la persona real."),
            "doppelganger_strategy": template.get(
                "doppelganger_strategy",
                "Responder con seguridad y evitar detalles verificables.",
            ),
            "is_real": False,
        })

    suspects = suspects[:3]

    for idx, suspect in enumerate(suspects, start=1):
        suspect["id"] = str(idx)
        if not _nonempty(str(suspect.get("name", ""))):
            suspect["name"] = inferred_name or f"Sospechoso {idx}"

    real_indices = [idx for idx, s in enumerate(suspects) if s.get("is_real") is True]
    if len(real_indices) != 1:
        chosen_real_idx = random.randrange(3)
        for idx, suspect in enumerate(suspects):
            suspect["is_real"] = idx == chosen_real_idx
    else:
        chosen_real_idx = real_indices[0]
        for idx, suspect in enumerate(suspects):
            suspect["is_real"] = idx == chosen_real_idx

    data["suspects"] = suspects


def _apply_case_fallbacks(data: dict) -> None:
    """
    Fill missing or empty case fields in-place using context from other
    parts of the generated data so fallbacks are always coherent.
    """
    case: dict = data.setdefault("case", {})
    suspects: list[dict] = data.get("suspects") or []

    # Derive a victim name from suspect names (all three claim the same identity)
    identity_name = suspects[0].get("name", "").strip() if suspects else ""
    background = case.get("background", "").strip()
    title = case.get("title", "").strip()

    fallbacks: dict[str, str] = {
        "title": "Caso de Suplantación de Identidad",
        "description": (
            background
            or "Tres personas reclaman ser la misma identidad. El detective debe descubrir quién es real."
        ),
        "setting": "Sala de interrogatorios de máxima seguridad",
        "victim": identity_name or "La persona cuya identidad fue usurpada",
        "time": "Hora no determinada",
        "crime": (
            f"Suplantación de identidad — {title}" if title else "Suplantación de identidad"
        ),
    }

    for field, default in fallbacks.items():
        if not _nonempty(case.get(field)):
            case[field] = default


# ── Mistral helper ───────────────────────────────────────────────────────────

async def call_mistral(
    messages: list[dict],
    json_mode: bool = False,
    temperature: float = 0.85,
) -> str:
    if not MISTRAL_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY is not configured. Add it to your .env file.",
        )

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": MISTRAL_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(MISTRAL_API_URL, headers=headers, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Mistral API error: {e.response.text}",
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Mistral API request timed out.")

    return resp.json()["choices"][0]["message"]["content"]


def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


# ── ElevenLabs voice settings ────────────────────────────────────────────────
#
# Two-layer system:
#   1. _VOICE_SETTINGS        – emotion layer  (drives expressiveness)
#   2. _BASE_SETTINGS_BY_SUSPECT – identity layer (preserves each voice's character)
#
# Parameters (all 0.0–1.0):
#   stability        – delivery consistency; lower = more variable / expressive
#   similarity_boost – adherence to the voice clone; lower = allows more drift
#   style            – stylistic exaggeration (eleven_multilingual_v2)
#   use_speaker_boost – boolean; enhances presence and clarity
#
# Blending strategy (see get_voice_settings):
#   style            → 100 % emotion   (primary carrier of expressiveness)
#   use_speaker_boost → 100 % emotion  (contextual clarity)
#   stability        → 15 % base + 85 % emotion  (emotion strongly dominates delivery)
#   similarity_boost → 50 % base + 50 % emotion  (balanced identity vs. emotion drift)
#
# Design principle: extreme values produce clearly audible differences.
#   Low stability  → erratic, expressive, broken delivery
#   High stability → flat, monotone, controlled delivery
#   High style     → exaggerated, energetic vocal performance
#   Low style      → restrained, neutral performance

# ── Emotion layer ─────────────────────────────────────────────────────────────
_VOICE_SETTINGS: dict[str, dict] = {
    # calm: neutral baseline — steady, clear, no exaggeration
    "calm":      {"stability": 0.75, "similarity_boost": 0.75, "style": 0.05, "use_speaker_boost": True},
    # nervous: unstable delivery, high style for jittery expressiveness
    "nervous":   {"stability": 0.08, "similarity_boost": 0.60, "style": 0.72, "use_speaker_boost": False},
    # angry: very unstable but forceful; maximum style for intensity
    "angry":     {"stability": 0.10, "similarity_boost": 0.82, "style": 0.90, "use_speaker_boost": True},
    # sad: moderate-low stability for natural variation, low style for subdued coloring,
    #      no speaker boost so it sounds heavy/low-energy rather than bright
    "sad":       {"stability": 0.52, "similarity_boost": 0.62, "style": 0.18, "use_speaker_boost": False},
    # defensive: measured tension — moderate stability, low-mid style
    "defensive": {"stability": 0.45, "similarity_boost": 0.80, "style": 0.28, "use_speaker_boost": True},
    # confident: steady but expressive — moderate-high style
    "confident": {"stability": 0.55, "similarity_boost": 0.82, "style": 0.55, "use_speaker_boost": True},
    # fearful: most unstable of all; high style for shaky, broken-sounding delivery
    "fearful":   {"stability": 0.04, "similarity_boost": 0.55, "style": 0.82, "use_speaker_boost": False},
}

_DEFAULT_VOICE_SETTINGS = _VOICE_SETTINGS["calm"]

# ── Identity (base) layer ─────────────────────────────────────────────────────
# Only stability and similarity_boost — the parameters that define vocal character.
# These small per-suspect offsets ensure each voice stays recognisable under any emotion.
_BASE_SETTINGS_BY_SUSPECT: dict[str, dict] = {
    "1": {"stability": 0.60, "similarity_boost": 0.82},  # Adam  – grave, grounded, controlled
    "2": {"stability": 0.45, "similarity_boost": 0.73},  # Sarah – expressive, wider dynamic range
    "3": {"stability": 0.68, "similarity_boost": 0.85},  # Arnold – deep, dry, composed
}

_DEFAULT_BASE_SETTINGS = {"stability": 0.55, "similarity_boost": 0.75}


def _blend(base: float, emotion: float, base_weight: float) -> float:
    """Weighted average of two values, clamped to [0.0, 1.0] and rounded."""
    return round(min(1.0, max(0.0, base * base_weight + emotion * (1.0 - base_weight))), 3)


def get_voice_settings(emotion: str, suspect_id: str = "1") -> dict:
    """
    Combines the suspect's base vocal signature with the emotion-driven settings.

    Blend weights (emotion-dominant by design):
      - stability        : 85 % emotion / 15 % base  → emotion strongly shifts delivery
      - similarity_boost : 50 % emotion / 50 % base  → balanced: identity vs. expressiveness
      - style            : 100 % emotion             → pure expressiveness, no identity concern
      - use_speaker_boost: 100 % emotion             → contextual clarity, boolean
    """
    emo  = _VOICE_SETTINGS.get(emotion, _DEFAULT_VOICE_SETTINGS)
    base = _BASE_SETTINGS_BY_SUSPECT.get(suspect_id, _DEFAULT_BASE_SETTINGS)

    return {
        "stability":        _blend(base["stability"],        emo["stability"],        0.15),
        "similarity_boost": _blend(base["similarity_boost"], emo["similarity_boost"], 0.50),
        "style":            emo["style"],
        "use_speaker_boost": emo["use_speaker_boost"],
    }


# ── ElevenLabs helper ────────────────────────────────────────────────────────

async def call_elevenlabs(text: str, voice_id: str, voice_settings: dict | None = None) -> bytes:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="ELEVENLABS_API_KEY is not configured. Add it to your .env file.",
        )

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": voice_settings or _DEFAULT_VOICE_SETTINGS,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"ElevenLabs API error: {e.response.text}",
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="ElevenLabs request timed out.")

    return resp.content


# ── Request / Response models ────────────────────────────────────────────────

class InterrogateRequest(BaseModel):
    game_id: str
    suspect_id: str
    question: str
    language: str = "es"


class StartGameRequest(BaseModel):
    language: str = "es"


class SusOScanResult(BaseModel):
    narration: str
    anomaly_delta: int
    tone: str
    reason_tags: list[str]
    sus_level: int


class InterrogateResponse(BaseModel):
    answer: str
    suspect_name: str
    questions_used: int
    questions_remaining: int
    emotion: str
    analysis: AnalysisSummary | None = None
    sus_scan: SusOScanResult | None = None


class AccuseRequest(BaseModel):
    game_id: str
    suspect_id: str


class AccuseResponse(BaseModel):
    correct: bool
    real_suspect_id: str
    real_suspect_name: str
    solution: str
    requirements: list[str]


class NarrateRequest(BaseModel):
    text: str
    suspect_id: str = "1"
    emotion: str = "calm"


class AnalyzeRequest(BaseModel):
    game_id: str
    suspect_id: str


class SuggestRequest(BaseModel):
    game_id: str
    suspect_id: str


def normalize_language(language: str | None) -> str:
    if language and language.lower() in {"es", "en"}:
        return language.lower()
    return "es"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/game/start")
async def start_game(req: StartGameRequest):
    """
    Generate a new Doppelganger mystery:
    - 1 real person (the target)
    - 2 doppelgangers pretending to be them
    - 5 requirements that only the real person can fulfill
    - Each suspect has an initial statement designed to confuse
    """

    language = normalize_language(req.language)
    language_instruction = (
        "IMPORTANT: Return ALL generated text strictly in Spanish."
        if language == "es"
        else "IMPORTANT: Return ALL generated text strictly in English."
    )

    prompt = language_instruction + """

Create a Doppelganger mystery game. The concept: ONE real person and TWO doppelgangers (perfect imposters) have been captured. The player must identify the real person by interrogating them.

Return ONLY a valid JSON object with this exact schema:

{
  "case": {
    "title": "string — dramatic case title",
    "description": "string — 3-4 sentences explaining the doppelganger situation and what is at stake",
    "setting": "string — vivid location where the interrogation takes place (e.g. 'Abandoned lighthouse, Norwegian coast')",
    "victim": "string — the real person whose identity is being disputed (their full name and one-line description, e.g. 'Dr. Amara Singh, lead quantum physicist at CERN')",
    "time": "string — when the imposters were captured or when the incident occurred (e.g. 'March 15, 2024 — 11:47 PM')",
    "crime": "string — the specific crime or threat the doppelgangers represent (e.g. 'Identity theft and corporate espionage')",
    "background": "string — who is the real person and why are the doppelgangers imitating them"
  },
  "requirements": [
    "string — specific, verifiable detail only the real person would know or have (e.g. a scar, a memory, a habit)",
    "string",
    "string",
    "string",
    "string"
  ],
  "suspects": [
    {
  "id": "1",
  "name": "string — all three suspects share the same full identity name",
  "age": 0,
  "occupation": "string — specific profession consistent with the case setting",
  "relationship_to_victim": "string — how this person knew or was connected to the victim",
  "appearance": "string — detailed but subtle physical description (all three look nearly identical)",
  "is_real": true,
  "personality": "string — how the REAL person speaks: genuine, specific memories, slight nervousness from the bizarre situation",
  "alibi": {
    "cooperative": true,
    "statement": "string — their public alibi. Should sound realistic and consistent."
  },
  "hidden_knowledge": "string — specific private memories and facts only the real person knows, tied to the investigation requirements",
  "initial_statement": "string — 3-4 sentence opening statement. The REAL person sounds authentic but slightly emotional or desperate. Subtly references 1-2 real facts without making it obvious.",
  "doppelganger_strategy": null
},
    {
  "id": "2",
  "name": "string — same full identity name as the real person",
  "age": 0,
  "occupation": "string — same profession as the real person",
  "relationship_to_victim": "string — same claimed relationship as the real person",
  "appearance": "string — nearly identical physical description with extremely subtle variation",
  "is_real": false,
  "personality": "string — overly confident, rehearsed, speaks in vague generalities and polished phrases",
  "alibi": {
    "cooperative": true,
    "statement": "string — confident and structured alibi, but contains subtle logical inconsistencies"
  },
  "hidden_knowledge": "string — information the doppelganger studied about the real person but includes subtle factual mistakes tied to the investigation requirements",
  "initial_statement": "string — 3-4 sentence opening statement. Sounds convincing and controlled, but slightly generic or rehearsed. References requirements incorrectly or vaguely.",
  "doppelganger_strategy": "string — describes how this doppelganger plans to deceive the interrogator (e.g., deflect emotional questions, remain calm, rely on memorized facts)"
},
    {
  "id": "3",
  "name": "string — same full identity name as the real person",
  "age": 0,
  "occupation": "string — same profession as the real person",
  "relationship_to_victim": "string — same claimed relationship as the real person",
  "appearance": "string — nearly identical physical description with extremely minor variation",
  "is_real": false,
  "personality": "string — overly emotional, dramatic, frequently shifts tone, tries to gain sympathy and portray themselves as misunderstood or persecuted",
  "alibi": {
    "cooperative": false,
    "statement": "string — emotionally charged alibi that sounds chaotic or exaggerated, may contain contradictions or over-explanations"
  },
  "hidden_knowledge": "string — information this doppelganger studied about the real person but includes different subtle mistakes than suspect 2, especially in emotional or personal memories",
  "initial_statement": "string — 3-4 sentence opening statement. Very emotional, possibly tearful. Tries to win sympathy. References investigation requirements with plausible but slightly incorrect or dramatized details.",
  "doppelganger_strategy": "string — explains how this doppelganger plans to manipulate the interrogator (e.g., appeal to empathy, act vulnerable, exaggerate trauma, create emotional pressure)"
}
  ],
  "solution": "string — 2-3 sentences explaining which is real and exactly how the requirements prove it"
}

RULES:
- All 3 suspects claim to be the SAME person (same name, same identity)
- Only 1 has is_real: true
- The 5 requirements must be specific enough to catch imposters (e.g. 'Has a crescent-shaped scar on the left wrist from a childhood accident', NOT vague things like 'loves music')
- Initial statements must sound convincing — the player should NOT be able to tell who is real just from the statement alone
- Make the setting and identity creative and surprising (avoid clichés)
- Return ONLY the raw JSON object"""

    # ── Generation with retry + fallback ─────────────────────────────────────
    MAX_ATTEMPTS = 2
    game_data: dict | None = None
    validation_errors: list[str] = []

    for _ in range(MAX_ATTEMPTS):
        try:
            raw = await call_mistral([{"role": "user", "content": prompt}], json_mode=True)
            candidate = json.loads(extract_json(raw))
        except json.JSONDecodeError as exc:
            validation_errors = [f"JSON parse error: {exc}"]
            continue  # retry

        _normalize_suspects(candidate)
        validation_errors = _validate_game_data(candidate)

        # Accept if only case fields are missing (fixable with fallbacks) or fully valid
        critical_errors = [e for e in validation_errors if not e.startswith("case.")]
        if not critical_errors:
            game_data = candidate
            break  # good enough — case gaps will be patched below

    if game_data is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate a valid game after {MAX_ATTEMPTS} attempts. "
                   f"Last errors: {validation_errors}",
        )

    # Patch any missing/empty case fields with coherent fallbacks
    _apply_case_fallbacks(game_data)

    game_id = str(uuid.uuid4())
    games[game_id] = {
        "data": game_data,
        "interrogation_history": {},
        "questions_used": {"1": 0, "2": 0, "3": 0},
        "extra_unlocked": {"1": False, "2": False, "3": False},
        "sus_level": {},
        "game_over": False,
        "language": language,
    }

    # Send public info only (no hidden_knowledge, no doppelganger_strategy, no is_real)
    public_suspects = [
    {
        "id": s["id"],
        "name": s["name"],
        "age": s.get("age"),
        "occupation": s.get("occupation"),
        "relationship_to_victim": s.get("relationship_to_victim"),
        "appearance": s.get("appearance"),
        "alibi_cooperative": (
            s["alibi"]["cooperative"] if isinstance(s.get("alibi"), dict)
            else False
        ),
        "personality": s.get("personality"),
        "initial_statement": s.get("initial_statement"),
    }
    for s in game_data["suspects"]
]

    return {
        "game_id": game_id,
        "case": game_data["case"],
        "requirements": game_data["requirements"],
        "suspects": public_suspects,
        "max_questions_per_suspect": MAX_QUESTIONS_PER_SUSPECT,
        "extra_questions_available": 2,
        "model": MISTRAL_MODEL,
        "language": language,
    }


async def _run_sus_scan(
    game: dict,
    suspect: dict,
    game_data: dict,
    question: str,
    answer: str,
    emotion: str,
    game_language: str,
) -> dict:
    """Calls Mistral to produce Sus-O-Scan signals. Updates sus_level in game store."""
    history = game["interrogation_history"].get(suspect["id"], [])

    # Build readable transcript (exclude the latest Q&A, already passed as params)
    transcript_lines = []
    for entry in history[:-2]:
        label = "Detective" if entry["role"] == "user" else suspect["name"]
        content = entry["content"]
        if entry["role"] == "assistant":
            try:
                content = json.loads(content).get("answer", content)
            except Exception:
                pass
        transcript_lines.append(f"{label}: {content}")
    transcript = "\n".join(transcript_lines) if transcript_lines else "(primer intercambio)"

    reqs = "\n".join(f"{i+1}. {r}" for i, r in enumerate(game_data["requirements"]))
    lang_rule = (
        "Write the narration in Spanish. Tags may be in English."
        if game_language == "es"
        else "Write the narration in English."
    )

    system_prompt = f"""You are Sus-O-Scan, a silent behavioral AI observing an interrogation.
{lang_rule}

CONTEXT:
Case: {game_data['case']['description']}
Suspect identity claimed: {suspect['name']}
Public alibi: {suspect.get('public_alibi', 'none')}
Requirements the real person must fulfill:
{reqs}

PRIOR CONVERSATION:
{transcript}

LATEST EXCHANGE:
Detective: {question}
{suspect['name']} [{emotion}]: {answer}

Output ONLY valid JSON — no markdown, no explanation:
{{
  "narration": "<one atmospheric line, max 12 words, no spoilers, never say liar/fake/real/doppelganger>",
  "anomaly_delta": <integer -2 to +2>,
  "tone": "<warm|cold|static>",
  "reason_tags": [<0 to 2 short behavioral micro-cues, 1-3 words each>]
}}

Rules:
- anomaly_delta:
    +2 = contradicts alibi, highly evasive, refuses to answer, very aggressive
    +1 = suspicious hesitation, vague details, slight inconsistency
     0 = neutral, answer neither helps nor hurts
    -1 = coherent and calm, details fit the alibi well
    -2 = very grounded, cooperative, consistent with all known facts
- tone (CRITICAL — must match the suspect's emotional state and anomaly_delta):
    warm  = suspect is nervous, defensive, angry, evasive, or emotionally agitated → RAISES suspicion
    cold  = suspect is calm, cooperative, coherent, composed → LOWERS suspicion
    static = signals are genuinely mixed or impossible to read
- The detected emotion is "{emotion}". Use it as strong evidence:
    angry/nervous/defensive/fearful → strongly lean toward warm
    calm/confident/cooperative     → strongly lean toward cold
    sad/ambiguous                  → lean toward static
- CONSISTENCY RULE: tone="warm" requires anomaly_delta >= 0; tone="cold" requires anomaly_delta <= 0
- narration must be cinematic and subtle — never a verdict"""

    messages = [{"role": "system", "content": system_prompt}]

    result: dict | None = None
    for attempt in range(3):
        try:
            raw = await call_mistral(messages, json_mode=True, temperature=0.7)
            parsed = json.loads(extract_json(raw))

            narration = str(parsed.get("narration") or "").strip()
            if not narration:
                raise ValueError("empty narration")

            delta = max(-2, min(2, int(parsed.get("anomaly_delta", 0))))
            tone = str(parsed.get("tone") or "static").strip().lower()
            if tone not in VALID_TONES:
                tone = "static"

            # Hard-correct tone if emotion strongly implies suspicion/warmth
            HOT_EMOTIONS = {"angry", "nervous", "defensive", "fearful"}
            COOL_EMOTIONS = {"calm", "confident"}
            if emotion in HOT_EMOTIONS and tone == "cold":
                tone = "warm"
            if emotion in COOL_EMOTIONS and tone == "warm" and delta <= 0:
                tone = "cold"
            # Enforce consistency: warm tone must have non-negative delta
            if tone == "warm" and delta < 0:
                delta = 0
            # cold tone must have non-positive delta
            if tone == "cold" and delta > 0:
                delta = 0

            tags = [str(t).strip() for t in (parsed.get("reason_tags") or []) if str(t).strip()][:2]

            result = {"narration": narration, "anomaly_delta": delta, "tone": tone, "reason_tags": tags}
            break
        except Exception:
            if attempt == 2:
                result = SUS_SCAN_FALLBACK.copy()

    if result is None:
        result = SUS_SCAN_FALLBACK.copy()

    sus_levels = game.setdefault("sus_level", {})
    current = sus_levels.get(suspect["id"], 5)
    new_level = max(0, min(10, current + result["anomaly_delta"]))
    sus_levels[suspect["id"]] = new_level
    result["sus_level"] = new_level

    return result


@app.post("/api/game/interrogate", response_model=InterrogateResponse)
async def interrogate(req: InterrogateRequest):
    """Ask a suspect a question. 5 base + 2 optional extra."""

    if _has_profanity(req.question):
        raise HTTPException(
            status_code=400,
            detail="Inappropriate language. Keep the interrogation professional.",
        )

    game = games.get(req.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")

    if game["game_over"]:
        raise HTTPException(status_code=400, detail="This game has already ended.")

    game_data = game["data"]
    game_language = normalize_language(game.get("language"))
    suspect = next((s for s in game_data["suspects"] if s["id"] == req.suspect_id), None)
    if not suspect:
        raise HTTPException(status_code=404, detail="Suspect not found.")

    questions_used = game["questions_used"].get(req.suspect_id, 0)
    extra_unlocked_map = game.setdefault("extra_unlocked", {})
    extra_unlocked = extra_unlocked_map.get(req.suspect_id, False)
    if questions_used >= BASE_QUESTIONS and not extra_unlocked:
        raise HTTPException(
            status_code=403,
            detail="BASE_LIMIT_REACHED"
    )
    if questions_used >= MAX_QUESTIONS_PER_SUSPECT:
        raise HTTPException(
            status_code=403,
            detail="MAX_LIMIT_REACHED"
    )
    

    if req.suspect_id not in game["interrogation_history"]:
        game["interrogation_history"][req.suspect_id] = []

    history: list[dict] = game["interrogation_history"][req.suspect_id]

    if suspect["is_real"]:
        role_instructions = f"""You ARE the real person. You are genuinely confused and scared by this situation.
Answer truthfully from your real memories. You may be emotional or frustrated that no one believes you.
Your private knowledge: {suspect['hidden_knowledge']}
Occasionally reference a specific real memory to prove your identity, but don't list all proofs at once — it would seem rehearsed."""
    else:
        role_instructions = f"""You are a DOPPELGANGER — a perfect physical copy pretending to be this person.
Your strategy: {suspect['doppelganger_strategy']}
What you studied about the real person (but with subtle errors): {suspect['hidden_knowledge']}
CRITICAL: If asked about the 5 requirements, give plausible but slightly wrong answers. Be confident but vague on specifics.
Never admit you are a doppelganger."""

    language_rule = (
        "ALWAYS reply in Spanish only."
        if game_language == "es"
        else "ALWAYS reply in English only."
    )

    pressure_guidance = _build_pressure_guidance(suspect["is_real"], questions_used)

    system_prompt = f"""You are one of three people claiming to be the same identity in a high-stakes interrogation.
{language_rule}

CASE: {game_data['case']['description']}
IDENTITY YOU CLAIM: {suspect['name']}
APPEARANCE: {suspect['appearance']}
PERSONALITY: {suspect['personality']}

THE 5 REQUIREMENTS the real person must fulfill:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(game_data['requirements']))}

YOUR ROLE:
{role_instructions}

{pressure_guidance}

RULES:
- Keep answers to 2-4 sentences. Stay fully in character.
- Never break character or acknowledge you are an AI.
- The interrogator only has {MAX_QUESTIONS_PER_SUSPECT} questions total for you — make each answer count toward your goal.
- Respond ONLY with a valid JSON object in this exact format: {{"answer": "your 2-4 sentence in-character response", "emotion": "one_word"}}
- The "emotion" field must be exactly one of: calm, nervous, angry, sad, defensive, confident, fearful
- Choose "emotion" based on your character's psychological state and the pressure context above
- The emotion word must NOT appear inside the "answer" text itself"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": req.question})

    raw_response = await call_mistral(messages, json_mode=True)

    try:
        parsed = json.loads(extract_json(raw_response))
        # Guard against None / non-string values for answer
        answer = str(parsed.get("answer") or "").strip() or raw_response
        # Normalize emotion: lowercase + strip before validation
        raw_emotion = str(parsed.get("emotion") or "").strip().lower()
        emotion = raw_emotion if raw_emotion in VALID_EMOTIONS else "calm"
    except (json.JSONDecodeError, AttributeError, TypeError):
        answer = raw_response
        emotion = "calm"

    # Store full raw JSON in history so the model maintains the expected format across turns
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": raw_response})
    game["questions_used"][req.suspect_id] = questions_used + 1

    new_count = game["questions_used"][req.suspect_id]

    # ── Optional inline analysis (ENABLE_ANALYSIS=true) ──────────────────────
    # Runs after history is updated so it sees the latest Q&A.
    # Any failure here is swallowed — the main game response is never affected.
    analysis: AnalysisSummary | None = None
    if ENABLE_ANALYSIS:
        try:
            ar = await run_analysis(game, suspect, game_data)
            if ar is not None:
                analysis = AnalysisSummary(
                    suspicion_score=ar.suspicion_score,
                    contradictions=ar.contradictions,
                    recommendation=ar.recommendation,
                )
        except Exception:
            pass

    # ── Sus-O-Scan ────────────────────────────────────────────────────────────
    sus_scan: SusOScanResult | None = None
    try:
        scan_dict = await _run_sus_scan(
            game=game,
            suspect=suspect,
            game_data=game_data,
            question=req.question,
            answer=answer,
            emotion=emotion,
            game_language=game_language,
        )
        sus_scan = SusOScanResult(**scan_dict)
    except Exception:
        pass

    return InterrogateResponse(
        answer=answer,
        suspect_name=suspect["name"],
        questions_used=new_count,
        questions_remaining=MAX_QUESTIONS_PER_SUSPECT - new_count,
        emotion=emotion,
        analysis=analysis,
        sus_scan=sus_scan,
    )
class UnlockRequest(BaseModel):
    game_id: str
    suspect_id: str


@app.post("/api/game/unlock-extra")
async def unlock_extra(req: UnlockRequest):

    game = games.get(req.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")

    if req.suspect_id not in game["questions_used"]:
        raise HTTPException(status_code=404, detail="Suspect not found.")

    # Solo permitir desbloquear si ya usó las 5 base
    if game["questions_used"][req.suspect_id] < BASE_QUESTIONS:
        raise HTTPException(
            status_code=400,
            detail="You must use the 5 base questions first."
        )

    extra_unlocked_map = game.setdefault("extra_unlocked", {})
    extra_unlocked_map[req.suspect_id] = True

    return {
        "extra_unlocked": True,
        "extra_questions_available": EXTRA_QUESTIONS
    }

class SusOScanRequest(BaseModel):
    game_id: str
    suspect_id: str


@app.post("/api/game/susoscan/scan", response_model=SusOScanResult)
async def susoscan_on_demand(req: SusOScanRequest):
    """On-demand Sus-O-Scan reading based on existing conversation history (no level change)."""
    game = games.get(req.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")

    game_data = game["data"]
    game_language = normalize_language(game.get("language"))
    suspect = next((s for s in game_data["suspects"] if s["id"] == req.suspect_id), None)
    if not suspect:
        raise HTTPException(status_code=404, detail="Suspect not found.")

    current_sus = game.get("sus_level", {}).get(req.suspect_id, 5)
    history = game["interrogation_history"].get(req.suspect_id, [])

    if not history:
        narration_no_data = (
            "La sala guarda silencio — aún no hay señales que leer."
            if game_language == "es"
            else "The room holds its breath — no signals to read yet."
        )
        return SusOScanResult(
            narration=narration_no_data,
            anomaly_delta=0,
            tone="static",
            reason_tags=[],
            sus_level=current_sus,
        )

    transcript_lines = []
    for entry in history:
        label = "Detective" if entry["role"] == "user" else suspect["name"]
        content = entry["content"]
        if entry["role"] == "assistant":
            try:
                content = json.loads(content).get("answer", content)
            except Exception:
                pass
        transcript_lines.append(f"{label}: {content}")
    transcript = "\n".join(transcript_lines)

    reqs = "\n".join(f"{i+1}. {r}" for i, r in enumerate(game_data["requirements"]))
    lang_rule = (
        "Write the narration in Spanish. Tags in English."
        if game_language == "es"
        else "Write the narration in English."
    )

    system_prompt = f"""You are Sus-O-Scan performing an on-demand behavioral assessment.
{lang_rule}

Case: {game_data['case']['description']}
Suspect: {suspect['name']} | Alibi: {suspect.get('public_alibi', 'none')}
Requirements:
{reqs}
Current sus_level: {current_sus}/10

Full conversation so far:
{transcript}

Output ONLY valid JSON (no markdown):
{{
  "narration": "<atmospheric assessment, max 12 words, no spoilers, no verdict>",
  "anomaly_delta": 0,
  "tone": "<warm|cold|static>",
  "reason_tags": [<0 to 2 behavioral micro-cues>]
}}
anomaly_delta MUST be 0 (read-only scan, no level change).
tone rules:
  warm  = suspect shows emotional heat across the conversation (nervous, angry, defensive, evasive) → raises suspicion
  cold  = suspect appears calm, cooperative, coherent, consistent → lowers suspicion
  static = signals are genuinely mixed or hard to read
Assess the overall pattern of the full conversation above, not just the last message."""

    result: dict = SUS_SCAN_FALLBACK.copy()
    for attempt in range(3):
        try:
            raw = await call_mistral(
                [{"role": "system", "content": system_prompt}],
                json_mode=True,
                temperature=0.65,
            )
            parsed = json.loads(extract_json(raw))
            narration = str(parsed.get("narration") or "").strip()
            if not narration:
                raise ValueError("empty narration")
            tone = str(parsed.get("tone") or "static").lower()
            if tone not in VALID_TONES:
                tone = "static"
            tags = [str(t).strip() for t in (parsed.get("reason_tags") or []) if str(t).strip()][:2]
            result = {"narration": narration, "anomaly_delta": 0, "tone": tone, "reason_tags": tags}
            break
        except Exception:
            if attempt == 2:
                result = SUS_SCAN_FALLBACK.copy()

    result["sus_level"] = current_sus
    return SusOScanResult(**result)


@app.post("/api/game/accuse", response_model=AccuseResponse)
async def accuse(req: AccuseRequest):
    """Player makes their final accusation — who is the REAL person?"""

    game = games.get(req.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")

    if game["game_over"]:
        raise HTTPException(status_code=400, detail="This game has already ended.")

    game["game_over"] = True

    game_data = game["data"]
    accused = next((s for s in game_data["suspects"] if s["id"] == req.suspect_id), None)
    if not accused:
        raise HTTPException(status_code=404, detail="Suspect not found.")

    suspects = game_data.get("suspects") or []
    real = next((s for s in suspects if s.get("is_real") is True), None)
    if not real and suspects:
        real = suspects[0]

    real_suspect_id = (
        str(real.get("id", "")).strip()
        if isinstance(real, dict)
        else ""
    ) or str(accused.get("id", "")).strip() or req.suspect_id
    real_suspect_name = (
        str(real.get("name", "")).strip()
        if isinstance(real, dict)
        else ""
    ) or str(accused.get("name", "")).strip() or "Unknown"
    accused_id = str(accused.get("id", "")).strip()

    return AccuseResponse(
        correct=accused_id == real_suspect_id,
        real_suspect_id=real_suspect_id,
        real_suspect_name=real_suspect_name,
        solution=game_data.get("solution", "Solution not recorded."),
        requirements=game_data["requirements"],
    )


@app.post("/api/game/narrate")
async def narrate(req: NarrateRequest):
    """Convert suspect answer to speech using ElevenLabs."""
    voice_id = SUSPECT_VOICES.get(req.suspect_id, SUSPECT_VOICES["1"])
    settings = get_voice_settings(req.emotion, req.suspect_id)
    audio_bytes = await call_elevenlabs(req.text, voice_id, settings)
    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/game/{game_id}/status")
async def game_status(game_id: str):
    """Returns current question usage per suspect without revealing answers."""
    game = games.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")

    return {
        "game_id": game_id,
        "game_over": game["game_over"],
        "questions_used": game["questions_used"],
        "questions_remaining": {
            sid: max(0, MAX_QUESTIONS_PER_SUSPECT - used)
            for sid, used in game["questions_used"].items()
        },
        "max_questions_per_suspect": MAX_QUESTIONS_PER_SUSPECT,
    }


@app.post("/api/game/analyze", response_model=AnalyzeResponse)
async def analyze_suspect(req: AnalyzeRequest):
    """
    Analyze one suspect's interrogation history using Mistral.
    Builds context exclusively from public game data (no hidden fields).
    """
    game = games.get(req.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")

    game_data = game["data"]
    suspect = next((s for s in game_data["suspects"] if s["id"] == req.suspect_id), None)
    if not suspect:
        raise HTTPException(status_code=404, detail="Suspect not found.")

    return await run_analysis(game, suspect, game_data) or ANALYZE_FALLBACK


@app.post("/api/game/suggest")
async def suggest_question(req: SuggestRequest):
    """Generate a suggested next question for the detective, based on public case data only."""
    game = games.get(req.game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")

    game_data = game["data"]
    suspect = next((s for s in game_data["suspects"] if s["id"] == req.suspect_id), None)
    if not suspect:
        raise HTTPException(status_code=404, detail="Suspect not found.")

    language  = normalize_language(game.get("language"))
    lang_rule = (
        "Respond entirely in Spanish."
        if language == "es"
        else "Respond entirely in English."
    )

    # Build transcript from interrogation history
    history: list[dict] = game["interrogation_history"].get(req.suspect_id, [])
    qa_pairs: list[str] = []
    for i in range(0, len(history) - 1, 2):
        q_text   = history[i]["content"]
        raw_ans  = history[i + 1]["content"]
        try:
            parsed_a = json.loads(extract_json(raw_ans))
            ans = str(parsed_a.get("answer") or "").strip() or raw_ans
        except (json.JSONDecodeError, AttributeError, TypeError):
            ans = raw_ans
        qa_pairs.append(f"Detective: {q_text}\nSuspect: {ans}")

    case = game_data["case"]
    requirements_text = "\n".join(
        f"{i + 1}. {r}" for i, r in enumerate(game_data["requirements"])
    )
    transcript = "\n\n".join(qa_pairs) if qa_pairs else "(No questions asked yet.)"

    fallback = (
        "¿Puede describir exactamente dónde estaba y con quién en ese momento?"
        if language == "es"
        else "Can you describe exactly where you were and who you were with at that time?"
    )

    prompt = f"""Detective advisor. {lang_rule}

CASE: {case['crime']} | Victim: {case['victim']} | When: {case['time']} | Where: {case['setting']}

REQUIREMENTS (only the real person satisfies all 5):
{requirements_text}

OPENING STATEMENT: {suspect['initial_statement']}

INTERROGATION SO FAR ({len(qa_pairs)} Q&A):
{transcript}

Generate ONE follow-up question the detective should ask next.
Rules:
- Target a gap, vagueness, or potential inconsistency in the alibi or requirements
- Do NOT reveal the answer or make it obvious who is real or fake
- One sentence, natural interrogation language, ends with ?
- Output ONLY: {{"suggested_question": "..."}}"""

    try:
        raw     = await call_mistral(
            [{"role": "user", "content": prompt}], json_mode=True, temperature=0.6
        )
        parsed  = json.loads(extract_json(raw))
        suggestion = str(parsed.get("suggested_question") or "").strip()
        if not suggestion or not suggestion.endswith("?"):
            return {"suggested_question": fallback}
        return {"suggested_question": suggestion}
    except Exception:
        return {"suggested_question": fallback}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MISTRAL_MODEL,
        "api_key_configured": bool(MISTRAL_API_KEY),
        "elevenlabs_configured": bool(ELEVENLABS_API_KEY),
        "active_games": len(games),
    }
