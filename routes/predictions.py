"""
Predictions API Route
======================
Endpoints:
  GET  /api/predictions/status         → model status + accuracy
  POST /api/predictions/train          → train/retrain from all DB history
  POST /api/predictions/predict        → rich prediction for one matchup (by team IDs)
  GET  /api/predictions/upcoming       → predict all upcoming fixtures
  GET  /api/predictions/public         → public-facing predictions with bet recommendations
  GET  /api/predictions/fixtures       → list upcoming fixtures to pick from
  POST /api/predictions/generate       → legacy rule-based prediction (by team names)
  GET  /api/predictions/results        → recent completed match results
  GET  /api/predictions/accuracy       → accuracy trend by gameweek
  GET  /api/predictions/training-status → poll background training status
"""

import time
import logging
import threading
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from database import get_connection
import ml.prediction_engine as engine
from ml.prediction_engine import predict_upcoming_fast

log = logging.getLogger(__name__)
router = APIRouter()

# Tracks background training state so /status can report errors
_training_state: dict = {}

# Simple in-memory cache for public predictions (15-minute TTL)
_public_cache: dict = {"data": None, "expires_at": 0.0}


# ─── Request models ───────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    league_id:  Optional[int] = None
    season_id:  Optional[int] = None


class PredictRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    league_id:    int
    season_id:    int


class GenerateRequest(BaseModel):
    """Legacy rule-based prediction by team name."""
    home_team: str
    away_team: str
    league: Optional[str] = None


# ─── ML Routes ────────────────────────────────────────────────────────────────

@router.get("/status")
def prediction_status():
    """Return current model status: trained, accuracy, n_samples, n_features."""
    return engine.get_status()


def _run_training_in_background():
    """
    Runs in a daemon Thread — completely independent of the HTTP request lifecycle.
    This means training continues even if the user navigates away from the page.
    FastAPI BackgroundTasks can be cancelled when the request connection drops;
    a threading.Thread is not affected by that.
    """
    global _training_state
    _training_state = {"status": "running", "started_at": time.time()}
    try:
        result = engine.train_model()
        _training_state = {"status": "done", "result": result}
    except Exception as e:
        _training_state = {"status": "error", "error": str(e)}


@router.post("/train")
def train(req: TrainRequest = None):
    """
    Kick off model training in a background thread.
    Returns immediately — clients should poll /training-status for progress.
    Training continues even if the user navigates away (daemon thread).
    """
    if _training_state.get("status") == "running":
        elapsed = int(time.time() - _training_state.get("started_at", time.time()))
        return {"started": False, "message": f"Training already running ({elapsed}s elapsed). Poll /training-status."}
    t = threading.Thread(target=_run_training_in_background, daemon=True, name="model-training")
    t.start()
    return {"started": True, "message": "Training started. Poll /training-status for progress."}


@router.get("/training-status")
def training_status():
    """Poll this endpoint to check if background training has finished."""
    state = _training_state
    if not state:
        return {"status": "idle", "message": "No training has been triggered yet."}
    return state


# ─── Auto-log helper ──────────────────────────────────────────────────────────

def _log_prediction_to_db(result: dict, match_id: Optional[int] = None):
    try:
        match_info = result.get("match", {})
        probs      = result.get("probabilities", {})

        home_team  = match_info.get("home_team") or result.get("home_team")
        away_team  = match_info.get("away_team") or result.get("away_team")
        league     = match_info.get("league")    or result.get("league")
        predicted  = result.get("predicted_outcome")

        # match_date lives at root level in upcoming results, inside match for single predict
        match_date = (
            result.get("match_date")          # predict_upcoming_fast puts it here
            or match_info.get("date")         # DC markets puts it here
            or match_info.get("match_date")
        )

        if not predicted or not home_team or not away_team:
            return

        conn = get_connection()
        cur  = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO prediction_log
                    (match_id, home_team, away_team, league, match_date,
                     predicted, confidence, confidence_score,
                     home_win_prob, draw_prob, away_win_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO NOTHING
            """, (
                match_id,
                home_team, away_team, league, match_date,
                predicted,
                result.get("confidence"),
                float(result.get("confidence_score") or 0),   
                float(probs.get("home_win") or 0),             
                float(probs.get("draw") or 0),                 
                float(probs.get("away_win") or 0),            
            ))
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        log.warning("Could not log prediction: %s", exc)

@router.post("/predict")
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict the outcome of a specific match by team IDs.
    Returns rich output: probabilities, xG, key factors, H2H, team comparison.
    Automatically logs the prediction to prediction_log (non-blocking).
    Returns 422 if the model has not been trained yet.
    """
    if req.home_team_id == req.away_team_id:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")
    try:
        result = engine.predict_match(
            req.home_team_id,
            req.away_team_id,
            req.league_id,
            req.season_id,
        )
        if "error" in result:
            error_msg = result["error"]
            if "not trained" in error_msg.lower() or "no model" in error_msg.lower():
                raise HTTPException(
                    status_code=422,
                    detail="Model not trained yet. Click 'Train Model' first, then try again."
                )
            raise HTTPException(status_code=422, detail=error_msg)

        # Auto-log the prediction (best-effort, non-blocking)
        match_id = result.get("match", {}).get("match_id") or result.get("match_id")
        background_tasks.add_task(_log_prediction_to_db, result, match_id)

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upcoming")
def upcoming_predictions(
    background_tasks: BackgroundTasks,
    league_id: Optional[int] = Query(None, description="Filter by league ID"),
    limit:     int           = Query(50,   description="Max fixtures to predict"),
):
    """Predict all upcoming unplayed fixtures using the fast bulk-load path.
    Automatically logs every prediction to prediction_log (non-blocking)."""
    try:
        results = predict_upcoming_fast(league_id=league_id, limit=limit)

        # Auto-log all predictions in background
        def _bulk_log():
            for r in results:
                mid = r.get("fixture_id") or (r.get("match") or {}).get("match_id")
                _log_prediction_to_db(r, match_id=mid)

        background_tasks.add_task(_bulk_log)

        return {
            "count":       len(results),
            "predictions": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Public predictions endpoint (bet recommendations) ───────────────────────

def _derive_best_bets(predictions: list) -> list:
    """
    Add bet_recommendation field to each prediction.
    Logic:
    - High confidence + lead probability > 55% → "Strong pick: {outcome}"
    - Medium confidence + lead prob > 48% → "Value bet: {outcome}"
    - Both teams strong attack (predicted score >=2 goals each side) → "Both Teams to Score"
    - Under-valued draw (draw_prob > 35% but predicted outcome isn't draw) → flag it
    """
    enriched = []
    for p in predictions:
        probs    = p.get("probabilities", {})
        hw       = probs.get("home_win", 0.33)
        dr       = probs.get("draw", 0.33)
        aw       = probs.get("away_win", 0.34)
        conf     = p.get("confidence", "Low")
        outcome  = p.get("predicted_outcome", "")
        xg       = p.get("expected_goals", {}) or {}
        home_xg  = float(xg.get("home_xg") or 0)
        away_xg  = float(xg.get("away_xg") or 0)
        match    = p.get("match", {}) or {}

        bets = []

        # Main outcome bet
        lead_prob = max(hw, dr, aw)
        if conf == "High" and lead_prob >= 0.55:
            bets.append({
                "bet":  f"✅ Strong Pick — {outcome}",
                "prob": round(lead_prob * 100),
                "tier": "high",
            })
        elif conf in ("High", "Medium") and lead_prob >= 0.48:
            bets.append({
                "bet":  f"💡 Value Bet — {outcome}",
                "prob": round(lead_prob * 100),
                "tier": "medium",
            })

        # BTTS (Both Teams to Score)
        if home_xg >= 1.1 and away_xg >= 1.0:
            bets.append({
                "bet":  "⚽ Both Teams to Score (recommended)",
                "prob": None,
                "tier": "btts",
            })

        # Undervalued draw warning
        if dr >= 0.34 and outcome != "Draw":
            bets.append({
                "bet":  f"⚠️ Draw value — {round(dr * 100)}% probability, consider 1X or X2",
                "prob": round(dr * 100),
                "tier": "draw_value",
            })

        enriched.append({**p, "bet_recommendations": bets})
    return enriched


@router.get("/public")
def public_predictions(
    league_id: Optional[int] = Query(None, description="Filter by league ID"),
    limit:     int           = Query(30,   description="Max predictions"),
):
    """
    Public-facing upcoming match predictions with bet recommendations.
    Results are cached for 15 minutes to reduce DB load.
    No authentication required.
    """
    global _public_cache
    now = time.time()

    # Return cached data if still fresh
    if _public_cache["data"] is not None and now < _public_cache["expires_at"]:
        data = _public_cache["data"]
        if league_id:
            data = [p for p in data if (p.get("match") or {}).get("league_id") == league_id]
        return {"count": len(data), "predictions": data[:limit], "cached": True}

    try:
        raw = predict_upcoming_fast(league_id=None, limit=100)
        enriched = _derive_best_bets(raw)

        # Cache result for 15 minutes
        _public_cache = {"data": enriched, "expires_at": now + 900}

        if league_id:
            enriched = [p for p in enriched if (p.get("match") or {}).get("league_id") == league_id]

        return {
            "count":       len(enriched[:limit]),
            "predictions": enriched[:limit],
            "cached":      False,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Fixtures list ────────────────────────────────────────────────────────────

@router.get("/fixtures")
def list_upcoming_fixtures(
    league_id:  Optional[int] = Query(None),
    season_id:  Optional[int] = Query(None),
    limit:      int           = Query(100),
):
    """
    List upcoming unplayed fixtures with team names and IDs.
    Use this to discover team IDs before calling /predict.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        query = """
            SELECT m.id, m.match_date, m.gameweek, m.start_time,
                   ht.id AS home_team_id, ht.name AS home_team, ht.logo_url AS home_logo,
                   at.id AS away_team_id, at.name AS away_team, at.logo_url AS away_logo,
                   l.id  AS league_id,   l.name  AS league,
                   s.id  AS season_id,   s.name  AS season
            FROM matches m
            JOIN teams   ht ON ht.id = m.home_team_id
            JOIN teams   at ON at.id = m.away_team_id
            JOIN leagues l  ON l.id  = m.league_id
            JOIN seasons s  ON s.id  = m.season_id
            WHERE m.home_score IS NULL
              AND m.match_date >= CURRENT_DATE
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"; params.append(league_id)
        if season_id:
            query += " AND m.season_id = %s"; params.append(season_id)
        query += " ORDER BY m.match_date ASC LIMIT %s"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()
        return {
            "count":    len(rows),
            "fixtures": [dict(r) for r in rows],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── Historical Results Endpoint ─────────────────────────────────────────────

@router.get("/results")
def list_prediction_results(
    league_id:  Optional[int] = Query(None, description="Filter by league ID"),
    season_id:  Optional[int] = Query(None, description="Filter by season ID"),
    limit:      int           = Query(30,   description="Number of recent results"),
):
    """
    Return recent completed matches from the DB with real scores.
    Used by the frontend 'Predictions' page to show past match results.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        query = """
            SELECT m.id, m.match_date, m.gameweek, m.score_raw,
                   m.home_score, m.away_score,
                   ht.name AS home_team, at.name AS away_team,
                   l.name  AS league,    l.id    AS league_id,
                   s.name  AS season,    s.id    AS season_id
            FROM matches m
            JOIN teams   ht ON ht.id = m.home_team_id
            JOIN teams   at ON at.id = m.away_team_id
            JOIN leagues l  ON l.id  = m.league_id
            JOIN seasons s  ON s.id  = m.season_id
            WHERE m.home_score IS NOT NULL
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"; params.append(league_id)
        if season_id:
            query += " AND m.season_id = %s"; params.append(season_id)
        query += " ORDER BY m.match_date DESC LIMIT %s"
        params.append(limit)
        cur.execute(query, params)
        rows = cur.fetchall()
        return {"count": len(rows), "results": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


@router.get("/accuracy")
def prediction_accuracy_trend(
    league_id:  Optional[int] = Query(None),
    season_id:  Optional[int] = Query(None),
    weeks:      int           = Query(9, description="Number of recent gameweeks"),
):
    """
    Compute accuracy trend grouped by gameweek from real match data.
    Uses standings win-rate heuristic to score past predictions vs actual results.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        query = """
            SELECT m.gameweek,
                   COUNT(*)                              AS total,
                   COUNT(CASE
                     WHEN (m.home_score > m.away_score
                           AND ls_h.wins::float / NULLIF(ls_h.games,0) >
                               ls_a.wins::float / NULLIF(ls_a.games,0))
                       OR (m.away_score > m.home_score
                           AND ls_a.wins::float / NULLIF(ls_a.games,0) >
                               ls_h.wins::float / NULLIF(ls_h.games,0))
                       OR (m.home_score = m.away_score
                           AND ABS(ls_h.wins::float/NULLIF(ls_h.games,0)
                                 - ls_a.wins::float/NULLIF(ls_a.games,0)) < 0.05)
                     THEN 1 END)                        AS correct
            FROM matches m
            LEFT JOIN league_standings ls_h
                   ON ls_h.team_id = m.home_team_id
                  AND ls_h.league_id = m.league_id
                  AND ls_h.season_id = m.season_id
            LEFT JOIN league_standings ls_a
                   ON ls_a.team_id = m.away_team_id
                  AND ls_a.league_id = m.league_id
                  AND ls_a.season_id = m.season_id
            WHERE m.home_score IS NOT NULL
              AND m.gameweek IS NOT NULL
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"; params.append(league_id)
        if season_id:
            query += " AND m.season_id = %s"; params.append(season_id)
        query += """
            GROUP BY m.gameweek
            ORDER BY m.gameweek DESC
            LIMIT %s
        """
        params.append(weeks)
        cur.execute(query, params)
        rows = cur.fetchall()
        trend = []
        for r in reversed(rows):
            total   = int(r["total"] or 0)
            correct = int(r["correct"] or 0)
            trend.append({
                "week":        f"GW{r['gameweek']}",
                "predictions": total,
                "correct":     correct,
                "accuracy":    round(correct / total * 100) if total else 0,
            })
        return trend
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── Legacy Rule-based Route (kept for backward compatibility) ─────────────────

def _safe_div(a, b, default=0.0):
    try: return float(a) / float(b) if b else default
    except: return default


def _get_team_stats(cur, team_name: str):
    # Fixed: removed non-existent scraped_at column; use season_id DESC for ordering
    cur.execute("""
        SELECT ts.goals, ts.assists, ts.games, ts.possession, ts.avg_age
        FROM   team_squad_stats ts
        JOIN   teams t ON t.id = ts.team_id
        WHERE  LOWER(t.name) LIKE LOWER(%s) AND ts.split = 'for'
        ORDER  BY ts.season_id DESC LIMIT 1
    """, (f"%{team_name}%",))
    return cur.fetchone()


def _get_team_standings(cur, team_name: str):
    # Fixed: removed non-existent scraped_at column; use season_id DESC for ordering
    cur.execute("""
        SELECT ls.wins, ls.ties, ls.losses, ls.games,
               ls.goals_for, ls.goals_against, ls.points, ls.rank
        FROM   league_standings ls
        JOIN   teams t ON t.id = ls.team_id
        WHERE  LOWER(t.name) LIKE LOWER(%s)
        ORDER  BY ls.season_id DESC LIMIT 1
    """, (f"%{team_name}%",))
    return cur.fetchone()


def _get_actual_result(cur, home_team: str, away_team: str):
    cur.execute("""
        SELECT m.home_score, m.away_score, m.match_date, m.score_raw,
               h.name AS home_name, a.name AS away_name
        FROM   matches m
        JOIN   teams h ON h.id = m.home_team_id
        JOIN   teams a ON a.id = m.away_team_id
        WHERE  LOWER(h.name) LIKE LOWER(%s)
          AND  LOWER(a.name) LIKE LOWER(%s)
          AND  m.home_score IS NOT NULL
        ORDER  BY m.match_date DESC
        LIMIT  1
    """, (f"%{home_team}%", f"%{away_team}%"))
    return cur.fetchone()


def _compute_probabilities(h_sq, a_sq, h_st, a_st):
    h_gpg = _safe_div(h_sq["goals"] if h_sq else 0, h_sq["games"] if h_sq else 1, 1.2)
    a_gpg = _safe_div(a_sq["goals"] if a_sq else 0, a_sq["games"] if a_sq else 1, 1.0)
    h_wr  = _safe_div(h_st["wins"]  if h_st else 0, h_st["games"] if h_st else 1, 0.4)
    a_wr  = _safe_div(a_st["wins"]  if a_st else 0, a_st["games"] if a_st else 1, 0.35)
    h_str = 0.6 * h_gpg + 0.4 * h_wr
    a_str = 0.6 * a_gpg + 0.4 * a_wr
    total = h_str + a_str + 0.001
    r_h = max(0.1, h_str / total + 0.06)
    r_a = max(0.1, a_str / total - 0.03)
    r_d = max(0.1, 1 - r_h - r_a)
    s   = r_h + r_a + r_d
    home_p = round(r_h / s, 3)
    away_p = round(r_a / s, 3)
    draw_p = round(1 - home_p - away_p, 3)
    pred_h = max(0, round(h_gpg * 0.85))
    pred_a = max(0, round(a_gpg * 0.75))
    has_all = bool(h_sq and a_sq and h_st and a_st)
    confidence = "high" if has_all else ("medium" if (h_sq or a_sq) else "low")
    return home_p, draw_p, away_p, pred_h, pred_a, confidence


def _fmt_stats(sq, st):
    if not sq and not st:
        return None
    return {
        "goals_per_game":  round(_safe_div(sq["goals"] if sq else 0, sq["games"] if sq else 1, 0), 2) if sq else None,
        "win_rate":        round(_safe_div(st["wins"]  if st else 0, st["games"] if st else 1, 0), 2) if st else None,
        "possession":      float(sq["possession"]) if sq and sq.get("possession") else None,
        "avg_age":         float(sq["avg_age"])    if sq and sq.get("avg_age")    else None,
        "goals_for":       int(st["goals_for"])    if st and st.get("goals_for")  else None,
        "goals_against":   int(st["goals_against"]) if st and st.get("goals_against") else None,
        "rank":            int(st["rank"])         if st and st.get("rank")       else None,
        "points":          int(st["points"])       if st and st.get("points")     else None,
    }


@router.post("/generate")
async def generate_prediction(req: GenerateRequest):
    """Legacy rule-based endpoint: accepts team names, returns prediction."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        h_sq = _get_team_stats(cur,     req.home_team)
        a_sq = _get_team_stats(cur,     req.away_team)
        h_st = _get_team_standings(cur, req.home_team)
        a_st = _get_team_standings(cur, req.away_team)
        home_p, draw_p, away_p, pred_h, pred_a, conf = _compute_probabilities(h_sq, a_sq, h_st, a_st)
        actual = _get_actual_result(cur, req.home_team, req.away_team)
        actual_result = None
        if actual and actual["home_score"] is not None:
            ah = int(actual["home_score"])
            aa = int(actual["away_score"])
            pred_winner   = "home" if home_p > away_p and home_p > draw_p else ("away" if away_p > home_p and away_p > draw_p else "draw")
            actual_winner = "home" if ah > aa else ("away" if aa > ah else "draw")
            actual_result = {
                "home_score": ah,
                "away_score": aa,
                "score_raw":  actual["score_raw"],
                "match_date": str(actual["match_date"]) if actual.get("match_date") else None,
                "prediction_correct": pred_winner == actual_winner,
            }
        return {
            "success":      True,
            "home_team":    req.home_team,
            "away_team":    req.away_team,
            "home_win_prob": home_p,
            "draw_prob":    draw_p,
            "away_win_prob": away_p,
            "predicted_score": {"home": pred_h, "away": pred_a},
            "confidence":   conf,
            "home_stats":   _fmt_stats(h_sq, h_st),
            "away_stats":   _fmt_stats(a_sq, a_st),
            "actual_result": actual_result,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        cur.close()
        conn.close()
