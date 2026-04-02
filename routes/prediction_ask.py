"""
Prediction Ask — /api/predict/ask
===================================
LLM-powered Q&A about any consensus match prediction.

Strategy:
  1. Fetch full prediction context for the requested match
  2. Package it as structured text context
  3. Try Groq first (fastest, 14,400 req/day free)
  4. Fall back to Gemini if Groq fails
  5. Return the answer + which data panels to highlight

Environment variables required:
  GROQ_API_KEY   — from console.groq.com
  GEMINI_API_KEY — from aistudio.google.com
"""

import os
import json
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from database import get_connection

log = logging.getLogger(__name__)
router = APIRouter()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

SYSTEM_PROMPT = """You are PlusOne AI, an expert football match prediction analyst.
You answer user questions about a specific match prediction made by the PlusOne Consensus Engine.

RULES:
1. Answer ONLY using the data provided in the context below. Do NOT guess or use external knowledge.
2. Be concise and direct — 2-5 sentences maximum unless the question requires more detail.
3. Always reference specific numbers from the data (probabilities, xG values, etc.).
4. If the data doesn't contain enough information to answer, say so clearly.
5. Use plain language — no jargon unless the user specifically asks for technical detail.
6. When mentioning probabilities, always include the percentage (e.g. 63.2% not 0.632).
7. Highlight the most important insight first.

You are answering about the following match prediction:

{context}

Now answer the user's question concisely and clearly."""


class AskRequest(BaseModel):
    question: str
    match_id:     Optional[int] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    league_id:    Optional[int] = None
    season_id:    Optional[int] = None


def _build_context(match_data: dict) -> str:
    """Convert the full prediction dict into a readable text context for the LLM."""
    m   = match_data.get("match", {})
    con = match_data.get("consensus", {})
    eng = match_data.get("engines", {})
    mkt = match_data.get("markets", {})
    bets = match_data.get("best_bets", [])
    weights = match_data.get("weights_used", {})
    agreement = match_data.get("agreement", "unknown")

    def pct(v):
        try: return f"{round(float(v) * 100, 1)}%"
        except: return "N/A"

    def fmt_engine(name, data):
        if not data: return f"  {name}: no data"
        return (f"  {name}: {pct(data.get('home_win'))} Home Win | "
                f"{pct(data.get('draw'))} Draw | "
                f"{pct(data.get('away_win'))} Away Win → "
                f"Predicts: {data.get('predicted_outcome', 'N/A')}")

    lines = [
        f"MATCH: {m.get('home_team', 'Home')} vs {m.get('away_team', 'Away')}",
        f"League: {m.get('league', 'Unknown')} | Season: {m.get('season', 'Unknown')}",
        "",
        "=== CONSENSUS PREDICTION ===",
        f"Outcome: {con.get('predicted_outcome', 'N/A')}",
        f"Confidence: {con.get('confidence', 'N/A')} ({con.get('confidence_score', 0):.1f}/100)",
        f"Home Win: {pct(con.get('home_win'))} | Draw: {pct(con.get('draw'))} | Away Win: {pct(con.get('away_win'))}",
        f"Engine Agreement: {agreement.upper()} ({agreement} consensus among all engines)",
        "",
        "=== ENGINE BREAKDOWN ===",
        fmt_engine("Dixon-Coles (DC)", eng.get("dc", {})),
        fmt_engine("ML Ensemble", eng.get("ml", {})),
        fmt_engine("Legacy Heuristic", eng.get("legacy", {})),
        "",
        "=== ENGINE WEIGHTS ===",
        f"  DC: {pct(weights.get('dc'))} | ML: {pct(weights.get('ml'))} | Legacy: {pct(weights.get('legacy'))}",
        f"  Weight source: {weights.get('source', 'unknown')} (dynamic = based on recent accuracy, default = fixed starting weights)",
        "",
    ]

    if mkt:
        home_xg = mkt.get("home_xg", mkt.get("home_xg"))
        away_xg = mkt.get("away_xg", mkt.get("away_xg"))
        lines += [
            "=== MARKET PROBABILITIES ===",
            f"  Expected Goals (xG): Home {home_xg} — Away {away_xg}",
            f"  BTTS Yes: {pct(mkt.get('btts_yes'))} | BTTS No: {pct(mkt.get('btts_no'))}",
            f"  Over 1.5: {pct(mkt.get('over_1_5'))} | Over 2.5: {pct(mkt.get('over_2_5'))} | Over 3.5: {pct(mkt.get('over_3_5'))}",
            f"  Under 2.5: {pct(mkt.get('under_2_5'))} | Under 3.5: {pct(mkt.get('under_3_5'))}",
            f"  Double Chance 1X: {pct(mkt.get('dc_1x'))} | X2: {pct(mkt.get('dc_x2'))} | 12: {pct(mkt.get('dc_12'))}",
            f"  BTTS+Over2.5: {pct(mkt.get('btts_yes_over_2_5'))}",
            f"  Home Over 1.5 goals: {pct(mkt.get('home_over_1_5'))} | Away Over 1.5: {pct(mkt.get('away_over_1_5'))}",
            "",
        ]

    if bets:
        lines.append("=== BEST BETS (Ranked by confidence) ===")
        for b in bets[:6]:
            lines.append(
                f"  [{b.get('tier','').upper()}] {b.get('pick','?')} — "
                f"{b.get('probability_pct', 0):.1f}% probability | "
                f"Fair odds: {b.get('fair_odds', '?')}"
            )
        lines.append("")

    return "\n".join(lines)


async def _ask_groq(context: str, question: str) -> str:
    try:
        import httpx
    except ImportError:
        raise ValueError("httpx not installed — add 'httpx>=0.27.0' to requirements.txt")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    prompt = SYSTEM_PROMPT.format(context=context)
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": question},
                ],
                "max_tokens": 400,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


async def _ask_gemini(context: str, question: str) -> str:
    try:
        import httpx
    except ImportError:
        raise ValueError("httpx not installed — add 'httpx>=0.27.0' to requirements.txt")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    prompt = SYSTEM_PROMPT.format(context=context)
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": f"{prompt}\n\nUser question: {question}"}]}],
                "generationConfig": {"maxOutputTokens": 400, "temperature": 0.3},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _fetch_prediction_data(
    conn, match_id=None, home_team_id=None, away_team_id=None,
    league_id=None, season_id=None
) -> dict:
    """
    Fetch the latest consensus prediction for the match from prediction_log,
    then re-run consensus to get full market data.
    Falls back to a live consensus run if no log entry exists.
    """
    cur = conn.cursor()

    # Try to get stored prediction from prediction_log
    stored = None
    try:
        if match_id:
            cur.execute("""
                SELECT pl.*, m.home_score, m.away_score,
                       ht.name as home_team, at.name as away_team,
                       l.name as league_name
                FROM prediction_log pl
                LEFT JOIN matches m ON m.id = pl.match_id
                LEFT JOIN teams ht ON ht.id = pl.home_team_id
                LEFT JOIN teams at ON at.id = pl.away_team_id
                LEFT JOIN leagues l ON l.id = pl.league_id
                WHERE pl.match_id = %s
                ORDER BY pl.predicted_at DESC LIMIT 1
            """, (match_id,))
        elif home_team_id and away_team_id:
            cur.execute("""
                SELECT pl.*, m.home_score, m.away_score,
                       ht.name as home_team, at.name as away_team,
                       l.name as league_name
                FROM prediction_log pl
                LEFT JOIN matches m ON m.id = pl.match_id
                LEFT JOIN teams ht ON ht.id = pl.home_team_id
                LEFT JOIN teams at ON at.id = pl.away_team_id
                LEFT JOIN leagues l ON l.id = pl.league_id
                WHERE pl.home_team_id = %s AND pl.away_team_id = %s
                ORDER BY pl.predicted_at DESC LIMIT 1
            """, (home_team_id, away_team_id))
        stored = cur.fetchone()
    except Exception as e:
        log.debug("Could not fetch stored prediction: %s", e)

    # Run consensus live
    try:
        from ml.consensus_engine import run_consensus
        htid = home_team_id or (stored["home_team_id"] if stored else None)
        atid = away_team_id or (stored["away_team_id"] if stored else None)
        lid  = league_id  or (stored["league_id"]  if stored else None)
        sid  = season_id  or (stored["season_id"]  if stored else None)
        if htid and atid and lid and sid:
            result = run_consensus(htid, atid, lid, sid)
            # Enrich with actual score if available
            if stored and stored.get("home_score") is not None:
                result["actual_score"] = f"{stored['home_score']}-{stored['away_score']}"
                result["actual_outcome"] = stored.get("actual")
                result["prediction_correct"] = stored.get("correct")
            return result
    except Exception as e:
        log.warning("live consensus failed in ask endpoint: %s", e)

    # Last resort: build a minimal dict from what we know
    if stored:
        return {
            "match": {
                "home_team": stored.get("home_team") or "Home",
                "away_team": stored.get("away_team") or "Away",
                "league": stored.get("league_name") or "",
                "season": "",
            },
            "consensus": {
                "home_win": stored.get("home_win_prob"),
                "draw":     stored.get("draw_prob"),
                "away_win": stored.get("away_win_prob"),
                "predicted_outcome": stored.get("predicted_outcome"),
                "confidence": stored.get("confidence"),
                "confidence_score": stored.get("confidence_score"),
            },
            "engines": {},
            "markets": {},
            "best_bets": [],
            "weights_used": {},
            "agreement": "unknown",
        }

    return {}


@router.post("/ask")
async def ask_prediction(req: AskRequest):
    """
    Answer any question about a match prediction using LLM + full context.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Safety: reject if no valid API keys
    if not GROQ_API_KEY and not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="No LLM API keys configured. Set GROQ_API_KEY or GEMINI_API_KEY environment variables."
        )

    conn = get_connection()
    try:
        match_data = _fetch_prediction_data(
            conn,
            match_id=req.match_id,
            home_team_id=req.home_team_id,
            away_team_id=req.away_team_id,
            league_id=req.league_id,
            season_id=req.season_id,
        )
    finally:
        conn.close()

    if not match_data:
        raise HTTPException(status_code=404, detail="No prediction data found for this match.")

    context = _build_context(match_data)
    answer = None
    provider = None

    # Try Groq first (fastest)
    if GROQ_API_KEY:
        try:
            answer = await _ask_groq(context, req.question.strip())
            provider = "groq"
        except Exception as e:
            log.warning("Groq failed (%s), falling back to Gemini", e)

    # Fall back to Gemini
    if answer is None and GEMINI_API_KEY:
        try:
            answer = await _ask_gemini(context, req.question.strip())
            provider = "gemini"
        except Exception as e:
            log.error("Gemini also failed: %s", e)

    if not answer:
        raise HTTPException(status_code=502, detail="Both LLM providers failed. Please try again.")

    return {
        "answer":   answer,
        "provider": provider,
        "match":    match_data.get("match", {}),
        "context_snapshot": {
            "consensus_outcome":   match_data.get("consensus", {}).get("predicted_outcome"),
            "confidence":          match_data.get("consensus", {}).get("confidence"),
            "agreement":           match_data.get("agreement"),
            "best_bets_count":     len(match_data.get("best_bets", [])),
            "has_market_data":     bool(match_data.get("markets")),
        },
    }
