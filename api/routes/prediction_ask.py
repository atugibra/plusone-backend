"""
Prediction Ask — /api/predict/ask
===================================
LLM-powered Q&A about any consensus match prediction.

Strategy:
  1. Fetch full consensus prediction (all 3 engines + markets + best bets)
  2. Fetch rich DB context: last-5 form, season stats, H2H last-8, standings
  3. Ask the LLM to:
       a) Analyse the raw DB data independently
       b) Compare its own reading against the 3 engine predictions
       c) Agree or challenge the consensus, with reasoning
       d) Then answer the user's specific question
  4. Try the chosen provider first, fall back to the other
  5. Return answer + which provider/model was used

Environment variables (set in Railway):
  GROQ_API_KEY       — from console.groq.com
  GEMINI_API_KEY     — from aistudio.google.com

Optional overrides (no redeploy needed):
  GROQ_MODEL         — default: llama-3.3-70b-versatile
  GEMINI_MODEL       — default: gemini-2.0-flash

Available Groq models (all free):
  llama-3.3-70b-versatile   <- smartest, recommended
  llama-3.1-8b-instant      <- fastest
  mixtral-8x7b-32768        <- good for long context

Available Gemini models (free tier):
  gemini-2.0-flash          <- fast, good quality
  gemini-2.0-flash-lite     <- fastest, lower quality
"""

import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal

from database import get_connection

log = logging.getLogger(__name__)
router = APIRouter()

# API keys
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default models (overridable via env vars without redeploying)
DEFAULT_GROQ_MODEL   = os.getenv("GROQ_MODEL",   "llama-3.3-70b-versatile")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Available models catalogue (shown to frontend for selection UI)
AVAILABLE_MODELS = {
    "groq": [
        {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B  - Smartest",  "free": True},
        {"id": "llama-3.1-8b-instant",    "label": "Llama 3.1 8B   - Fastest",   "free": True},
        {"id": "mixtral-8x7b-32768",      "label": "Mixtral 8x7B   - Long ctx",  "free": True},
    ],
    "gemini": [
        {"id": "gemini-2.0-flash",      "label": "Gemini 2.0 Flash      - Fast & smart", "free": True},
        {"id": "gemini-2.0-flash-lite", "label": "Gemini 2.0 Flash Lite - Fastest",      "free": True},
    ],
}

GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

SYSTEM_PROMPT = """You are PlusOne AI, an expert football analyst and data scientist.

You have been given two sources of information about an upcoming match:

1. RAW DATABASE DATA - real historical facts: recent form, head-to-head history,
   season statistics, and league standings for both teams.

2. ENGINE PREDICTIONS - outputs from three independent prediction models
   (Dixon-Coles statistical model, ML ensemble, and a legacy heuristic model),
   which have already processed the data and produced probability estimates.

YOUR TASK - follow these steps in order:

STEP 1 - YOUR OWN ASSESSMENT:
Read the raw database data carefully. Based purely on the form, H2H record,
season stats and standings, decide what you think the most likely outcome is
and why. State this clearly and reference specific numbers.

STEP 2 - COMPARE WITH ENGINES:
Look at what each of the three engines predicted. Note where they agree and
where they disagree. Highlight any engine that contradicts your own reading
and explain why that might be.

STEP 3 - FINAL VERDICT:
State whether you agree or disagree with the consensus prediction, give a
confidence level (Low / Medium / High), and explain your reasoning concisely.

STEP 4 - ANSWER THE USER'S QUESTION:
Now directly answer the user's specific question using everything above.

RULES:
- Use plain text only. No markdown, no asterisks, no bullet symbols.
- Be specific - always reference actual numbers from the data.
- If data is missing or insufficient, say so clearly rather than guessing.
- Keep the full response under 400 words unless the question genuinely needs more.
- Probabilities should always be shown as percentages (e.g. 63.2% not 0.632).

Match context and data follow below.

{context}

User question: {question}"""


class AskRequest(BaseModel):
    question:     str
    match_id:     Optional[int] = None
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    league_id:    Optional[int] = None
    season_id:    Optional[int] = None
    provider:     Optional[Literal["groq", "gemini", "auto"]] = "auto"
    model:        Optional[str] = None


def _fetch_db_context(cur, home_team_id: int, away_team_id: int) -> dict:
    """Fetch rich real-world DB context: form, season stats, H2H, standings."""

    def recent_results(tid: int, n: int = 5):
        try:
            cur.execute("""
                SELECT
                    m.match_date,
                    CASE WHEN m.home_team_id = %(t)s THEN at2.name ELSE ht2.name END AS opponent,
                    CASE WHEN m.home_team_id = %(t)s THEN 'H' ELSE 'A'           END AS venue,
                    CASE WHEN m.home_team_id = %(t)s THEN m.home_score ELSE m.away_score END AS ts,
                    CASE WHEN m.home_team_id = %(t)s THEN m.away_score ELSE m.home_score END AS os
                FROM matches m
                JOIN teams ht2 ON ht2.id = m.home_team_id
                JOIN teams at2 ON at2.id = m.away_team_id
                WHERE (m.home_team_id = %(t)s OR m.away_team_id = %(t)s)
                  AND m.home_score IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT %(n)s
            """, {"t": tid, "n": n})
            out = []
            for r in cur.fetchall():
                ts = r["ts"] or 0; os_ = r["os"] or 0
                out.append({
                    "date":     str(r["match_date"]) if r["match_date"] else None,
                    "opponent": r["opponent"],
                    "venue":    r["venue"],
                    "score":    f"{ts}-{os_}",
                    "result":   "W" if ts > os_ else ("L" if ts < os_ else "D"),
                })
            return out
        except Exception as e:
            log.debug("recent_results failed: %s", e)
            return []

    def season_stats(tid: int):
        try:
            cur.execute("""
                SELECT
                    COUNT(*) AS played,
                    SUM(CASE WHEN (home_team_id=%(t)s AND home_score>away_score)
                              OR  (away_team_id=%(t)s AND away_score>home_score) THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN home_score=away_score THEN 1 ELSE 0 END) AS draws,
                    SUM(CASE WHEN (home_team_id=%(t)s AND home_score<away_score)
                              OR  (away_team_id=%(t)s AND away_score<home_score) THEN 1 ELSE 0 END) AS losses,
                    SUM(CASE WHEN home_team_id=%(t)s THEN home_score ELSE away_score END) AS gf,
                    SUM(CASE WHEN home_team_id=%(t)s THEN away_score ELSE home_score END) AS ga,
                    SUM(CASE WHEN (home_team_id=%(t)s AND away_score=0)
                              OR  (away_team_id=%(t)s AND home_score=0) THEN 1 ELSE 0 END) AS cs
                FROM matches
                WHERE (home_team_id=%(t)s OR away_team_id=%(t)s)
                  AND home_score IS NOT NULL
            """, {"t": tid})
            r = cur.fetchone()
            if not r or not r["played"]: return {}
            p = r["played"]
            return {
                "played":            p,
                "wins":              r["wins"]   or 0,
                "draws":             r["draws"]  or 0,
                "losses":            r["losses"] or 0,
                "goals_for":         r["gf"]     or 0,
                "goals_against":     r["ga"]     or 0,
                "avg_goals_for":     round((r["gf"] or 0) / p, 2),
                "avg_goals_against": round((r["ga"] or 0) / p, 2),
                "clean_sheets":      r["cs"]     or 0,
            }
        except Exception as e:
            log.debug("season_stats failed: %s", e)
            return {}

    def h2h(htid: int, atid: int, n: int = 8):
        try:
            cur.execute("""
                SELECT
                    m.match_date,
                    ht4.name AS home_team, at4.name AS away_team,
                    m.home_team_id, m.home_score, m.away_score
                FROM matches m
                JOIN teams ht4 ON ht4.id = m.home_team_id
                JOIN teams at4 ON at4.id = m.away_team_id
                WHERE ((m.home_team_id=%(h)s AND m.away_team_id=%(a)s)
                    OR (m.home_team_id=%(a)s AND m.away_team_id=%(h)s))
                  AND m.home_score IS NOT NULL
                ORDER BY m.match_date DESC
                LIMIT %(n)s
            """, {"h": htid, "a": atid, "n": n})
            rows = cur.fetchall()
            hw = dr = aw = 0
            matches = []
            for r in rows:
                hs = r["home_score"] or 0; as_ = r["away_score"] or 0
                if hs > as_:
                    winner = r["home_team"]
                    if r["home_team_id"] == htid: hw += 1
                    else: aw += 1
                elif hs < as_:
                    winner = r["away_team"]
                    if r["home_team_id"] == atid: hw += 1
                    else: aw += 1
                else:
                    winner = "Draw"; dr += 1
                matches.append({
                    "date":   str(r["match_date"]) if r["match_date"] else None,
                    "home":   r["home_team"],
                    "away":   r["away_team"],
                    "score":  f"{hs}-{as_}",
                    "winner": winner,
                })
            return {"summary": {"home_wins": hw, "draws": dr, "away_wins": aw}, "matches": matches}
        except Exception as e:
            log.debug("h2h failed: %s", e)
            return {}

    def standings(tid: int):
        try:
            cur.execute("""
                SELECT ls.wins, ls.ties, ls.losses, ls.games,
                       ls.goals_for, ls.goals_against, ls.points, ls.rank
                FROM league_standings ls
                WHERE ls.team_id = %s
                ORDER BY ls.season_id DESC LIMIT 1
            """, (tid,))
            r = cur.fetchone()
            if not r: return {}
            return {
                "rank":          r["rank"],
                "points":        r["points"],
                "played":        r["games"],
                "wins":          r["wins"],
                "draws":         r["ties"],
                "losses":        r["losses"],
                "goals_for":     r["goals_for"],
                "goals_against": r["goals_against"],
                "goal_diff":     (r["goals_for"] or 0) - (r["goals_against"] or 0),
            }
        except Exception as e:
            log.debug("standings failed: %s", e)
            return {}

    return {
        "home_form":      recent_results(home_team_id),
        "away_form":      recent_results(away_team_id),
        "home_season":    season_stats(home_team_id),
        "away_season":    season_stats(away_team_id),
        "h2h":            h2h(home_team_id, away_team_id),
        "home_standings": standings(home_team_id),
        "away_standings": standings(away_team_id),
    }


def _build_context(match_data: dict, db_ctx: dict) -> str:
    m         = match_data.get("match", {})
    con       = match_data.get("consensus", {})
    eng       = match_data.get("engines", {})
    mkt       = match_data.get("markets", {})
    bets      = match_data.get("best_bets", [])
    weights   = match_data.get("weights_used", {})
    agreement = match_data.get("agreement", "unknown")

    home = m.get("home_team", "Home")
    away = m.get("away_team", "Away")

    def pct(v):
        try: return f"{round(float(v) * 100, 1)}%"
        except: return "N/A"

    def fmt_engine(name, data):
        if not data: return f"  {name}: no data"
        return (f"  {name}: {pct(data.get('home_win'))} Home | "
                f"{pct(data.get('draw'))} Draw | "
                f"{pct(data.get('away_win'))} Away -> Predicts: {data.get('predicted_outcome', 'N/A')}")

    def fmt_form(form_list, team_name):
        if not form_list:
            return f"  {team_name}: no recent results available"
        lines = [f"  {team_name} last {len(form_list)} results:"]
        for f in form_list:
            lines.append(f"    {f.get('date','?')} [{f.get('venue','?')}] vs {f.get('opponent','?')} "
                         f"{f.get('score','?')} ({f.get('result','?')})")
        return "\n".join(lines)

    def fmt_stats(stats, team_name):
        if not stats:
            return f"  {team_name}: no season stats available"
        return (f"  {team_name}: P{stats.get('played',0)} "
                f"W{stats.get('wins',0)} D{stats.get('draws',0)} L{stats.get('losses',0)} | "
                f"GF {stats.get('goals_for',0)} GA {stats.get('goals_against',0)} | "
                f"Avg scored {stats.get('avg_goals_for','?')} conceded {stats.get('avg_goals_against','?')} | "
                f"Clean sheets: {stats.get('clean_sheets',0)}")

    def fmt_standings(st, team_name):
        if not st:
            return f"  {team_name}: no standings data"
        return (f"  {team_name}: Rank {st.get('rank','?')} | "
                f"{st.get('points','?')} pts | GD {st.get('goal_diff','?')}")

    def fmt_h2h(h2h_data, home_name, away_name):
        if not h2h_data or not h2h_data.get("matches"):
            return "  No H2H data available"
        s = h2h_data.get("summary", {})
        lines = [f"  Last {len(h2h_data.get('matches',[]))} meetings: "
                 f"{home_name} wins {s.get('home_wins',0)} | "
                 f"Draws {s.get('draws',0)} | "
                 f"{away_name} wins {s.get('away_wins',0)}"]
        for match in h2h_data.get("matches", [])[:5]:
            lines.append(f"    {match.get('date','?')}: {match.get('home','?')} vs {match.get('away','?')} "
                         f"{match.get('score','?')} -> {match.get('winner','?')}")
        return "\n".join(lines)

    lines = [
        "=" * 60,
        f"MATCH: {home} vs {away}",
        f"League: {m.get('league', 'Unknown')} | Season: {m.get('season', 'Unknown')}",
        "=" * 60,
        "",
        "--- SECTION 1: RAW DATABASE DATA ---",
        "",
        "RECENT FORM (last 5 matches):",
        fmt_form(db_ctx.get("home_form", []), home),
        fmt_form(db_ctx.get("away_form", []), away),
        "",
        "SEASON STATISTICS:",
        fmt_stats(db_ctx.get("home_season", {}), home),
        fmt_stats(db_ctx.get("away_season", {}), away),
        "",
        "LEAGUE STANDINGS:",
        fmt_standings(db_ctx.get("home_standings", {}), home),
        fmt_standings(db_ctx.get("away_standings", {}), away),
        "",
        "HEAD-TO-HEAD HISTORY:",
        fmt_h2h(db_ctx.get("h2h", {}), home, away),
        "",
        "--- SECTION 2: ENGINE PREDICTIONS ---",
        "",
        "CONSENSUS (blended output):",
        f"  Predicted outcome: {con.get('predicted_outcome', 'N/A')}",
        f"  Confidence: {con.get('confidence', 'N/A')} ({con.get('confidence_score', 0):.1f}/100)",
        f"  Home Win: {pct(con.get('home_win'))} | Draw: {pct(con.get('draw'))} | Away Win: {pct(con.get('away_win'))}",
        f"  Engine agreement: {agreement.upper()}",
        "",
        "INDIVIDUAL ENGINES:",
        fmt_engine("Dixon-Coles (Poisson statistical)", eng.get("dc", {})),
        fmt_engine("ML Ensemble (XGBoost + RandomForest)", eng.get("ml", {})),
        fmt_engine("Legacy Heuristic (rule-based)", eng.get("legacy", {})),
        "",
        "ENGINE WEIGHTS:",
        f"  DC: {pct(weights.get('dc'))} | ML: {pct(weights.get('ml'))} | Legacy: {pct(weights.get('legacy'))}",
        f"  Source: {weights.get('source', 'unknown')}",
        "",
    ]

    if mkt:
        lines += [
            "MARKET PROBABILITIES:",
            f"  xG: {home} {mkt.get('home_xg','?')} - {away} {mkt.get('away_xg','?')}",
            f"  BTTS Yes: {pct(mkt.get('btts_yes'))} | Over 2.5: {pct(mkt.get('over_2_5'))} | Under 2.5: {pct(mkt.get('under_2_5'))}",
            f"  Double Chance 1X: {pct(mkt.get('dc_1x'))} | X2: {pct(mkt.get('dc_x2'))}",
            "",
        ]

    if bets:
        lines.append("BEST BETS (by confidence):")
        for b in bets[:5]:
            lines.append(f"  [{b.get('tier','').upper()}] {b.get('pick','?')} - "
                         f"{b.get('probability_pct', 0):.1f}% | Fair odds: {b.get('fair_odds','?')}")
        lines.append("")

    return "\n".join(lines)


def _fetch_prediction_data(
    conn, match_id=None, home_team_id=None, away_team_id=None,
    league_id=None, season_id=None
) -> dict:
    cur = conn.cursor()
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

    try:
        from ml.consensus_engine import run_consensus
        htid = home_team_id or (stored["home_team_id"] if stored else None)
        atid = away_team_id or (stored["away_team_id"] if stored else None)
        lid  = league_id   or (stored["league_id"]    if stored else None)
        sid  = season_id   or (stored["season_id"]    if stored else None)
        if htid and atid and lid and sid:
            result = run_consensus(htid, atid, lid, sid)
            if stored and stored.get("home_score") is not None:
                result["actual_score"]       = f"{stored['home_score']}-{stored['away_score']}"
                result["actual_outcome"]     = stored.get("actual")
                result["prediction_correct"] = stored.get("correct")
            return result
    except Exception as e:
        log.warning("live consensus failed in ask endpoint: %s", e)

    if stored:
        return {
            "match": {
                "home_team":    stored.get("home_team") or "Home",
                "away_team":    stored.get("away_team") or "Away",
                "league":       stored.get("league_name") or "",
                "season":       "",
                "home_team_id": stored.get("home_team_id"),
                "away_team_id": stored.get("away_team_id"),
            },
            "consensus": {
                "home_win":          stored.get("home_win_prob"),
                "draw":              stored.get("draw_prob"),
                "away_win":          stored.get("away_win_prob"),
                "predicted_outcome": stored.get("predicted_outcome"),
                "confidence":        stored.get("confidence"),
                "confidence_score":  stored.get("confidence_score"),
            },
            "engines": {}, "markets": {}, "best_bets": [],
            "weights_used": {}, "agreement": "unknown",
        }

    return {}


async def _ask_groq(context: str, question: str, model: str) -> str:
    import httpx
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
                "temperature": 0.3,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _ask_gemini(context: str, question: str, model: str) -> str:
    import httpx
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=35.0) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 700, "temperature": 0.3},
            },
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


@router.get("/models")
async def list_models():
    """Return available providers and models so the frontend can build a selector."""
    return {
        "available": AVAILABLE_MODELS,
        "defaults": {
            "provider":     "groq",
            "groq_model":   DEFAULT_GROQ_MODEL,
            "gemini_model": DEFAULT_GEMINI_MODEL,
        },
        "configured": {
            "groq":   bool(GROQ_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
        },
    }


@router.post("/ask")
async def ask_prediction(req: AskRequest):
    """
    Answer any question about a match prediction.
    The LLM reads raw DB data (form, H2H, stats, standings), makes its own
    independent assessment, compares it against the three engine predictions,
    then directly answers the user's question.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not GROQ_API_KEY and not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="No LLM API keys configured. Set GROQ_API_KEY or GEMINI_API_KEY.",
        )

    provider     = req.provider or "auto"
    groq_model   = req.model if (provider == "groq"   and req.model) else DEFAULT_GROQ_MODEL
    gemini_model = req.model if (provider == "gemini" and req.model) else DEFAULT_GEMINI_MODEL

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
        if not match_data:
            raise HTTPException(status_code=404, detail="No prediction data found for this match.")

        htid = req.home_team_id or match_data.get("match", {}).get("home_team_id")
        atid = req.away_team_id or match_data.get("match", {}).get("away_team_id")

        cur = conn.cursor()
        db_ctx = _fetch_db_context(cur, htid, atid) if htid and atid else {}

    finally:
        conn.close()

    context       = _build_context(match_data, db_ctx)
    answer        = None
    used_provider = None
    used_model    = None

    if provider == "groq":
        if not GROQ_API_KEY:
            raise HTTPException(status_code=503, detail="GROQ_API_KEY not configured.")
        try:
            answer = await _ask_groq(context, req.question.strip(), groq_model)
            used_provider, used_model = "groq", groq_model
        except Exception as e:
            log.warning("Groq (%s) failed: %s - trying Gemini fallback", groq_model, e)
            if GEMINI_API_KEY:
                try:
                    answer = await _ask_gemini(context, req.question.strip(), gemini_model)
                    used_provider, used_model = "gemini", gemini_model
                except Exception as e2:
                    log.error("Gemini fallback also failed: %s", e2)

    elif provider == "gemini":
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured.")
        try:
            answer = await _ask_gemini(context, req.question.strip(), gemini_model)
            used_provider, used_model = "gemini", gemini_model
        except Exception as e:
            log.warning("Gemini (%s) failed: %s - trying Groq fallback", gemini_model, e)
            if GROQ_API_KEY:
                try:
                    answer = await _ask_groq(context, req.question.strip(), groq_model)
                    used_provider, used_model = "groq", groq_model
                except Exception as e2:
                    log.error("Groq fallback also failed: %s", e2)

    else:  # auto
        if GROQ_API_KEY:
            try:
                answer = await _ask_groq(context, req.question.strip(), groq_model)
                used_provider, used_model = "groq", groq_model
            except Exception as e:
                log.warning("Groq (%s) failed: %s - falling back to Gemini", groq_model, e)
        if answer is None and GEMINI_API_KEY:
            try:
                answer = await _ask_gemini(context, req.question.strip(), gemini_model)
                used_provider, used_model = "gemini", gemini_model
            except Exception as e:
                log.error("Gemini (%s) also failed: %s", gemini_model, e)

    if not answer:
        raise HTTPException(status_code=502, detail="All LLM providers failed. Please try again.")

    return {
        "answer":   answer,
        "provider": used_provider,
        "model":    used_model,
        "match":    match_data.get("match", {}),
        "context_snapshot": {
            "consensus_outcome": match_data.get("consensus", {}).get("predicted_outcome"),
            "confidence":        match_data.get("consensus", {}).get("confidence"),
            "agreement":         match_data.get("agreement"),
            "best_bets_count":   len(match_data.get("best_bets", [])),
            "has_market_data":   bool(match_data.get("markets")),
            "has_db_context":    bool(db_ctx),
            "home_form_games":   len(db_ctx.get("home_form", [])),
            "away_form_games":   len(db_ctx.get("away_form", [])),
            "h2h_games":         len(db_ctx.get("h2h", {}).get("matches", [])),
        },
    }
