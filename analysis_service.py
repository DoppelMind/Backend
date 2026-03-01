"""
analysis_service.py — Mistral-powered forensic analysis of interrogation transcripts.

Public API
----------
    run_analysis(game, suspect, game_data) -> AnalyzeResponse | None
    AnalyzeResponse      — full Pydantic response (score + contradictions + evidence + rec)
    AnalysisSummary      — compact subset embedded in InterrogateResponse
    ANALYZE_FALLBACK     — stable default returned when all Mistral attempts fail
"""

import os
import json
import re
from typing import Any

import httpx
from fastapi import HTTPException
from pydantic import BaseModel

# ── Mistral config ────────────────────────────────────────────────────────────
# Reads the same env vars as main.py; no cross-module import needed.

_API_KEY  = os.getenv("MISTRAL_API_KEY")
_MODEL    = "mistral-small-latest"
_API_URL  = "https://api.mistral.ai/v1/chat/completions"

# ── Pydantic models ───────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    suspicion_score: int       # 0–100
    contradictions: list[str]  # 0–3 items
    supporting_evidence: list[str]  # 0–3 items
    recommendation: str        # "press" | "switch_suspect" | "ask_for_details"


class AnalysisSummary(BaseModel):
    """Compact subset embedded in InterrogateResponse when ENABLE_ANALYSIS=true."""
    suspicion_score: int
    contradictions: list[str]
    recommendation: str

# ── Constants ─────────────────────────────────────────────────────────────────

_VALID_RECOMMENDATIONS: frozenset[str] = frozenset(
    {"press", "switch_suspect", "ask_for_details"}
)

ANALYZE_FALLBACK = AnalyzeResponse(
    suspicion_score=50,
    contradictions=[],
    supporting_evidence=[],
    recommendation="ask_for_details",
)

_RETRY_MSG = (
    "Your previous response was not valid JSON matching the required schema. "
    "Respond ONLY with a raw JSON object — no markdown fences, no prose, no extra keys:\n"
    '{"suspicion_score": <int 0-100>, '
    '"contradictions": [<up to 3 strings>], '
    '"supporting_evidence": [<up to 3 strings>], '
    '"recommendation": "<press|switch_suspect|ask_for_details>"}'
)

# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown code fences if present, otherwise return text as-is."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return match.group(1).strip() if match else text


def _normalise_language(language: str | None) -> str:
    if language and language.lower() in {"es", "en"}:
        return language.lower()
    return "es"


async def _call_mistral(
    messages: list[dict],
    json_mode: bool = False,
    temperature: float = 0.3,
) -> str:
    """Minimal Mistral client for the analysis service (analysis always uses low temp)."""
    if not _API_KEY:
        raise HTTPException(
            status_code=500,
            detail="MISTRAL_API_KEY is not configured.",
        )

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": _MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(_API_URL, headers=headers, json=body)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Mistral API error: {e.response.text}",
            )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Mistral API request timed out.")

    return resp.json()["choices"][0]["message"]["content"]


def _parse_response(raw: str) -> AnalyzeResponse | None:
    """
    Strictly validate all required fields and types.
    Returns None on any failure — caller decides the fallback.
    """
    try:
        parsed = json.loads(_extract_json(raw))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None

    # suspicion_score — required, must be numeric
    raw_score = parsed.get("suspicion_score")
    if raw_score is None:
        return None
    try:
        score = max(0, min(100, int(raw_score)))
    except (TypeError, ValueError):
        return None

    # contradictions — required, must be a list
    raw_contra = parsed.get("contradictions")
    if raw_contra is None or not isinstance(raw_contra, list):
        return None
    contradictions = [str(c) for c in raw_contra if c][:3]

    # supporting_evidence — required, must be a list
    raw_support = parsed.get("supporting_evidence")
    if raw_support is None or not isinstance(raw_support, list):
        return None
    supporting = [str(s) for s in raw_support if s][:3]

    # recommendation — required, must be one of the allowed literals
    raw_rec = parsed.get("recommendation")
    if raw_rec is None:
        return None
    rec = str(raw_rec).strip().lower()
    if rec not in _VALID_RECOMMENDATIONS:
        return None

    return AnalyzeResponse(
        suspicion_score=score,
        contradictions=contradictions,
        supporting_evidence=supporting,
        recommendation=rec,
    )

# ── Public API ────────────────────────────────────────────────────────────────

async def run_analysis(
    game: dict,
    suspect: dict,
    game_data: dict,
) -> AnalyzeResponse | None:
    """
    Analyze a suspect's interrogation history using Mistral.

    Uses only public game data (no hidden_knowledge, no is_real).
    Retries up to 2 times with a reinforcement prompt on invalid responses.
    Returns None on total failure — callers should apply ANALYZE_FALLBACK.
    """
    try:
        # ── Transcript ───────────────────────────────────────────────────────
        history: list[dict] = game["interrogation_history"].get(suspect["id"], [])
        qa_pairs: list[str] = []
        for i in range(0, len(history) - 1, 2):
            question   = history[i]["content"]
            raw_answer = history[i + 1]["content"]
            try:
                parsed_a = json.loads(_extract_json(raw_answer))
                ans = str(parsed_a.get("answer") or "").strip() or raw_answer
            except (json.JSONDecodeError, AttributeError, TypeError):
                ans = raw_answer
            qa_pairs.append(f"Detective: {question}\n{suspect['name']}: {ans}")

        questions_asked    = len(qa_pairs)
        conversation_block = "\n\n".join(qa_pairs) if qa_pairs else "(No questions asked yet.)"

        # ── Context ──────────────────────────────────────────────────────────
        case = game_data["case"]
        requirements_text = "\n".join(
            f"{i + 1}. {r}" for i, r in enumerate(game_data["requirements"])
        )
        language  = _normalise_language(game.get("language"))
        lang_rule = (
            "Respond entirely in Spanish."
            if language == "es"
            else "Respond entirely in English."
        )

        # ── Prompt ───────────────────────────────────────────────────────────
        prompt = f"""Forensic interview analyst. {lang_rule}

CASE: {case['crime']} | Victim: {case['victim']} | When: {case['time']} | Where: {case['setting']}

REQUIREMENTS (only the real person satisfies all 5):
{requirements_text}

OPENING STATEMENT: {suspect['initial_statement']}

TRANSCRIPT ({questions_asked} Q&A):
{conversation_block}

OUTPUT: return ONLY this JSON object, nothing else — no markdown, no prose:
{{"suspicion_score": <int 0-100>, "contradictions": [<max 3>], "supporting_evidence": [<max 3>], "recommendation": "<press|switch_suspect|ask_for_details>"}}

RULES:
- NO_INVENT: every item in "contradictions" and "supporting_evidence" MUST be traceable to the OPENING STATEMENT, TRANSCRIPT, or CASE above. Never invent facts.
- Each contradiction string MUST start with its source tag: "contradicción con la hora:", "contradicción con el alibi:", "contradicción con requisito N:", or "internal contradiction:" — then quote the conflicting claim.
- Each supporting_evidence string MUST start with its source tag: "consistent with requisito N:", "alibi confirmed by:", or "transcript statement:" — then quote the supporting claim.
- If fewer than 2 Q&A turns exist, set both arrays to [] and use "ask_for_details".
- "recommendation": "press" = evasive/contradictory; "switch_suspect" = credible; "ask_for_details" = insufficient data.
- suspicion_score: 50 when uncertain, higher when evasive/contradictory, lower when consistent with requirements."""

        # ── Call Mistral with retry (1 initial + 2 retries) ──────────────────
        MAX_ATTEMPTS = 3
        messages: list[dict] = [{"role": "user", "content": prompt}]
        result: AnalyzeResponse | None = None

        for attempt in range(MAX_ATTEMPTS):
            raw    = await _call_mistral(messages, json_mode=True)
            result = _parse_response(raw)
            if result is not None:
                break
            if attempt < MAX_ATTEMPTS - 1:
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user",      "content": _RETRY_MSG})

        return result

    except Exception:
        return None
