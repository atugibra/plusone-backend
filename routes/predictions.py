"""
Predictions API Route
======================
Endpoints:
  GET  /api/predictions/status          → model status + accuracy
  POST /api/predictions/train           → train/retrain from all DB history
  GET  /api/predictions/training-status → poll background training status
  POST /api/predictions/recalibrate     → fit feedback calibrator from prediction_log
  POST /api/predictions/predict         → rich prediction for one matchup (by team IDs)
  GET  /api/predictions/upcoming        → predict all upcoming fixtures
  GET  /api/predictions/public          → public-facing predictions with bet recommendations
  GET  /api/predictions/fixtures        → list upcoming fixtures to pick from
  POST /api/predictions/generate        → legacy rule-based prediction (by team names)
  GET  /api/predictions/results         → recent completed match results
  GET  /api/predictions/accuracy        → accuracy trend by gameweek
  POST /api/predictions/consensus       → dynamic multi-engine consensus prediction
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

_training_state: dict = {}
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
    home_team: str
    away_team: str
    league: Optional[str] = None


class ConsensusRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    league_id:    int
    season_id:    int
    match_id:     Optional[int] = None   # allows prediction_log grading after match completes


# ─── ML Routes ────────────────────────────────────────────────────────────────

@router.get("/status")
def prediction_status():
    return engine.get_status()


def _run_training_in_background():
    global _training_state
    _training_state = {"status": "running", "started_at": time.time()}
    try:
        result = engine.train_model()
        _training_state = {"status": "done", "result": result}
    except Exception as e:
        _training_state = {"status": "error", "error": str(e)}


@router.post("/train")
def train(req: TrainRequest = None):
    if _training_state.get("status") == "running":
        elapsed = int(time.time() - _training_state.get("started_at", time.time()))
        return {"started": False, "message": f"Training already running ({elapsed}s elapsed). Poll /training-status."}
    t = threading.Thread(target=_run_training_in_background, daemon=True, name="model-training")
    t.start()
    return {"started": True, "message": "Training started. Poll /training-status for progress."}


@router.get("/training-status")
def training_status():
    state = _training_state
    if not state:
        return {"status": "idle", "message": "No training has been triggered yet."}
    return state


# ─── Feedback Calibration ─────────────────────────────────────────────────────

@router.post("/recalibrate")
def recalibrate():
    """
    Fit the feedback calibrator from evaluated prediction_log rows.

    Workflow:
      1. POST /api/predictions/upcoming      → generates predictions
      2. Wait for matches to complete
      3. POST /api/prediction-log/evaluate   → grades predictions vs real results
      4. POST /api/predictions/recalibrate   → THIS endpoint
         Calibrator learns from mistakes and adjusts future probabilities.

    Requires at least 15 evaluated predictions.
    Safe to call repeatedly — each call refits on all available data.
    The calibrator is automatically applied to all future predictions.
    """
    try:
        from ml.prediction_engine import recalibrate_from_log
        result = recalibrate_from_log()
        if not result.get("success"):
            raise HTTPException(status_code=422, detail=result.get("reason", "Calibration failed."))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Auto-log helper ──────────────────────────────────────────────────────────

def _log_prediction_to_db(
    result: dict,
    match_id: Optional[int] = None,
    dc_outcome: Optional[str] = None,
    ml_outcome: Optional[str] = None,
    legacy_outcome: Optional[str] = None,
    predicted_score: Optional[str] = None,
):
    """
    Persist one prediction to prediction_log with full deduplication/upsert logic.
    Extra per-engine outcomes are written so the feedback calibration loop has
    per-engine accuracy data to work with.
    """
    conn = None
    try:
        match_info = result.get("match", {})
        probs      = result.get("probabilities", {})

        home_team    = match_info.get("home_team")    or result.get("home_team")
        away_team    = match_info.get("away_team")    or result.get("away_team")
        home_team_id = match_info.get("home_team_id") or result.get("home_team_id")
        away_team_id = match_info.get("away_team_id") or result.get("away_team_id")
        league       = match_info.get("league")       or result.get("league")

        # If names are missing but we have IDs, look them up from DB
        if (not home_team or not away_team or not league) and (home_team_id or away_team_id):
            try:
                conn_inner = get_connection()
                cur_inner  = conn_inner.cursor()
                if not home_team or not away_team:
                    ids = [x for x in [home_team_id, away_team_id] if x]
                    if len(ids) == 2:
                        cur_inner.execute("SELECT id, name FROM teams WHERE id IN (%s, %s)", tuple(ids))
                    else:
                        cur_inner.execute("SELECT id, name FROM teams WHERE id = %s", (ids[0],))
                    name_map  = {r["id"]: r["name"] for r in cur_inner.fetchall()}
                    home_team = home_team or name_map.get(home_team_id)
                    away_team = away_team or name_map.get(away_team_id)
                if not league:
                    league_id = match_info.get("league_id") or result.get("league_id")
                    if league_id:
                        cur_inner.execute("SELECT name FROM leagues WHERE id = %s", (league_id,))
                        lg_row = cur_inner.fetchone()
                        league = lg_row["name"] if lg_row else None
                conn_inner.close()
            except Exception:
                pass

        predicted = result.get("predicted_outcome")

        # match_date: try all known keys
        match_date = (
            result.get("match_date")
            or match_info.get("match_date")
            or match_info.get("date")
        )

        if not predicted or not home_team or not away_team:
            log.warning("_log_prediction_to_db: missing fields — predicted=%s home=%s away=%s",
                        predicted, home_team, away_team)
            return

        # Markets — present in consensus (result.markets) or ML (result.expected_goals)
        markets  = result.get("markets") or {}
        xg_src   = result.get("expected_goals") or markets
        btts_yes = markets.get("btts_yes")
        over_2_5 = markets.get("over_2_5")
        home_xg  = xg_src.get("home_xg")
        away_xg  = xg_src.get("away_xg")

        conn = get_connection()
        cur  = conn.cursor()

        # ── Deduplication / Upsert ───────────────────────────────────────────
        existing = None

        # 1. Prefer matching exactly by match_id if provided
        if match_id is not None:
            cur.execute(
                "SELECT id, match_id, actual FROM prediction_log WHERE match_id = %s LIMIT 1",
                (match_id,)
            )
            existing = cur.fetchone()

        # 2. Fallback to team names + time window if no exact match_id found
        if not existing:
            cur.execute("""
                SELECT id, match_id, actual FROM prediction_log
                WHERE home_team = %s AND away_team = %s
                  AND (match_date >= CURRENT_DATE - INTERVAL '1 day'
                       OR created_at >= NOW() - INTERVAL '7 days')
                ORDER BY created_at DESC LIMIT 1
            """, (home_team, away_team))
            row = cur.fetchone()
            # Only use this fallback row if we aren't stealing it from a different known match
            if row and (row["match_id"] is None or row["match_id"] == match_id):
                existing = row

        if existing:
            if existing["actual"] is not None:
                log.debug("Skipping update for %s vs %s (already graded)", home_team, away_team)
                return

            new_id_to_set = match_id if match_id is not None else existing["match_id"]
            cur.execute("""
                UPDATE prediction_log SET
                    match_id                  = %s,
                    predicted                 = %s,
                    confidence                = %s,
                    confidence_score          = %s,
                    home_win_prob             = %s,
                    draw_prob                 = %s,
                    away_win_prob             = %s,
                    btts_yes                  = %s,
                    over_2_5                  = %s,
                    home_xg                   = %s,
                    away_xg                   = %s,
                    match_date                = COALESCE(%s, match_date),
                    league                    = COALESCE(%s, league),
                    dc_predicted_outcome      = COALESCE(%s, dc_predicted_outcome),
                    ml_predicted_outcome      = COALESCE(%s, ml_predicted_outcome),
                    legacy_predicted_outcome  = COALESCE(%s, legacy_predicted_outcome),
                    predicted_score           = COALESCE(%s, predicted_score)
                WHERE id = %s
            """, (
                new_id_to_set, predicted,
                result.get("confidence"),
                float(result.get("confidence_score") or 0),
                float(probs.get("home_win") or 0),
                float(probs.get("draw")     or 0),
                float(probs.get("away_win") or 0),
                float(btts_yes) if btts_yes is not None else None,
                float(over_2_5) if over_2_5 is not None else None,
                float(home_xg)  if home_xg  is not None else None,
                float(away_xg)  if away_xg  is not None else None,
                match_date, league,
                dc_outcome, ml_outcome, legacy_outcome, predicted_score,
                existing["id"],
            ))
            conn.commit()
            log.info("Updated prediction for %s vs %s", home_team, away_team)
            return

        # Guard against exact match_id collision for newly inserted rows
        if match_id is not None:
            cur.execute("SELECT id FROM prediction_log WHERE match_id = %s LIMIT 1", (match_id,))
            if cur.fetchone():
                log.debug("Skipping duplicate log for match_id=%s", match_id)
                return

        cur.execute("""
            INSERT INTO prediction_log
                (match_id, home_team, away_team, league, match_date,
                 predicted, confidence, confidence_score,
                 home_win_prob, draw_prob, away_win_prob,
                 btts_yes, over_2_5, home_xg, away_xg,
                 dc_predicted_outcome, ml_predicted_outcome,
                 legacy_predicted_outcome, predicted_score)
            VALUES
                (%s, %s, %s, %s, %s,
                 %s, %s, %s,
                 %s, %s, %s,
                 %s, %s, %s, %s,
                 %s, %s,
                 %s, %s)
        """, (
            match_id, home_team, away_team, league, match_date,
            predicted, result.get("confidence"),
            float(result.get("confidence_score") or 0),
            float(probs.get("home_win") or 0),
            float(probs.get("draw")     or 0),
            float(probs.get("away_win") or 0),
            float(btts_yes) if btts_yes is not None else None,
            float(over_2_5) if over_2_5 is not None else None,
            float(home_xg)  if home_xg  is not None else None,
            float(away_xg)  if away_xg  is not None else None,
            dc_outcome, ml_outcome, legacy_outcome, predicted_score,
        ))
        conn.commit()
        log.info("Logged prediction: %s vs %s → %s", home_team, away_team, predicted)
    except Exception as exc:
        log.error("Could not log prediction: %s", exc, exc_info=True)
        if conn:
            try: conn.rollback()
            except: pass
    finally:
        if conn:
            try: conn.close()
            except: pass


@router.post("/predict")
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    if req.home_team_id == req.away_team_id:
        raise HTTPException(status_code=400, detail="Home and away teams must be different.")
    try:
        result = engine.predict_match(
            req.home_team_id, req.away_team_id, req.league_id, req.season_id,
        )
        if "error" in result:
            error_msg = result["error"]
            if "not trained" in error_msg.lower() or "no model" in error_msg.lower():
                raise HTTPException(status_code=422, detail="Model not trained yet.")
            raise HTTPException(status_code=422, detail=error_msg)
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
    league_id: Optional[int] = Query(None),
    limit:     int           = Query(50),
):
    try:
        results = predict_upcoming_fast(league_id=league_id, limit=limit)
        def _bulk_log():
            for r in results:
                mid = r.get("fixture_id") or (r.get("match") or {}).get("match_id")
                _log_prediction_to_db(r, match_id=mid)
        background_tasks.add_task(_bulk_log)
        return {"count": len(results), "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Public predictions endpoint ─────────────────────────────────────────────

def _derive_best_bets(predictions: list) -> list:
    enriched = []
    for p in predictions:
        probs   = p.get("probabilities", {})
        hw      = probs.get("home_win", 0.33)
        dr      = probs.get("draw", 0.33)
        aw      = probs.get("away_win", 0.34)
        conf    = p.get("confidence", "Low")
        outcome = p.get("predicted_outcome", "")
        xg      = p.get("expected_goals", {}) or {}
        home_xg = float(xg.get("home_xg") or 0)
        away_xg = float(xg.get("away_xg") or 0)
        bets = []
        lead_prob = max(hw, dr, aw)
        if conf == "High" and lead_prob >= 0.55:
            bets.append({"bet": f"✅ Strong Pick — {outcome}", "prob": round(lead_prob * 100), "tier": "high"})
        elif conf in ("High", "Medium") and lead_prob >= 0.48:
            bets.append({"bet": f"💡 Value Bet — {outcome}", "prob": round(lead_prob * 100), "tier": "medium"})
        if home_xg >= 1.1 and away_xg >= 1.0:
            bets.append({"bet": "⚽ Both Teams to Score (recommended)", "prob": None, "tier": "btts"})
        if dr >= 0.34 and outcome != "Draw":
            bets.append({"bet": f"⚠️ Draw value — {round(dr * 100)}% probability, consider 1X or X2", "prob": round(dr * 100), "tier": "draw_value"})
        enriched.append({**p, "bet_recommendations": bets})
    return enriched


@router.get("/public")
def public_predictions(
    league_id: Optional[int] = Query(None),
    limit:     int           = Query(30),
):
    global _public_cache
    now = time.time()
    if _public_cache["data"] is not None and now < _public_cache["expires_at"]:
        data = _public_cache["data"]
        if league_id:
            data = [p for p in data if (p.get("match") or {}).get("league_id") == league_id]
        return {"count": len(data), "predictions": data[:limit], "cached": True}
    try:
        raw = predict_upcoming_fast(league_id=None, limit=100)
        enriched = _derive_best_bets(raw)
        _public_cache = {"data": enriched, "expires_at": now + 900}
        if league_id:
            enriched = [p for p in enriched if (p.get("match") or {}).get("league_id") == league_id]
        return {
            "count":        len(enriched[:limit]),
            "predictions":  enriched[:limit],
            "cached":       False,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@router.get("/fixtures")
def list_upcoming_fixtures(
    league_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),
    limit:     int           = Query(100),
):
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
            WHERE m.home_score IS NULL AND m.match_date >= CURRENT_DATE
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
        return {"count": len(rows), "fixtures": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── Results ──────────────────────────────────────────────────────────────────

@router.get("/results")
def list_prediction_results(
    league_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),
    limit:     int           = Query(30),
):
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
    league_id: Optional[int] = Query(None),
    season_id: Optional[int] = Query(None),
    weeks:     int           = Query(9),
):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        query = """
            SELECT m.gameweek,
                   COUNT(*) AS total,
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
                     THEN 1 END) AS correct
            FROM matches m
            LEFT JOIN league_standings ls_h
                   ON ls_h.team_id = m.home_team_id
                  AND ls_h.league_id = m.league_id
                  AND ls_h.season_id = m.season_id
            LEFT JOIN league_standings ls_a
                   ON ls_a.team_id = m.away_team_id
                  AND ls_a.league_id = m.league_id
                  AND ls_a.season_id = m.season_id
            WHERE m.home_score IS NOT NULL AND m.gameweek IS NOT NULL
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"; params.append(league_id)
        if season_id:
            query += " AND m.season_id = %s"; params.append(season_id)
        query += " GROUP BY m.gameweek ORDER BY m.gameweek DESC LIMIT %s"
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


# ─── Dynamic Consensus Engine (self-contained) ───────────────────────────────

_OUTCOME_LABELS  = ["Home Win", "Draw", "Away Win"]
_DEFAULT_WEIGHTS = {"dc": 0.45, "ml": 0.35, "legacy": 0.20}
_MIN_GRADED_ROWS = 10


def _consensus_safe_div(a, b, default=0.0):
    try:
        return float(a) / float(b) if b else default
    except Exception:
        return default


def _fetch_engine_weights(conn) -> dict:
    """Uses its own cursor. Rolls back on failure so conn stays usable."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN dc_correct     THEN 1 ELSE 0 END)::float AS dc_correct,
                SUM(CASE WHEN ml_correct     THEN 1 ELSE 0 END)::float AS ml_correct,
                SUM(CASE WHEN legacy_correct THEN 1 ELSE 0 END)::float AS legacy_correct
            FROM prediction_log
            WHERE correct IS NOT NULL
              AND evaluated_at >= NOW() - INTERVAL '30 days'
              AND dc_predicted_outcome     IS NOT NULL
              AND ml_predicted_outcome     IS NOT NULL
              AND legacy_predicted_outcome IS NOT NULL
        """)
        row   = cur.fetchone()
        total = int(row["total"] or 0) if row else 0
        if total < _MIN_GRADED_ROWS:
            return dict(_DEFAULT_WEIGHTS)
        dc_acc     = float(row["dc_correct"]     or 0) / total
        ml_acc     = float(row["ml_correct"]     or 0) / total
        legacy_acc = float(row["legacy_correct"] or 0) / total
        w_sum = dc_acc + ml_acc + legacy_acc
        if w_sum < 0.01:
            return dict(_DEFAULT_WEIGHTS)
        return {
            "dc":     round(dc_acc     / w_sum, 4),
            "ml":     round(ml_acc     / w_sum, 4),
            "legacy": round(legacy_acc / w_sum, 4),
        }
    except Exception as exc:
        log.warning("_fetch_engine_weights failed (%s) — using defaults", exc)
        try: conn.rollback()
        except: pass
        return dict(_DEFAULT_WEIGHTS)
    finally:
        try: cur.close()
        except: pass


def _run_legacy_inline(conn, home_id: int, away_id: int) -> dict:
    fallback = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35,
                "predicted_outcome": "Home Win"}
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT ls.wins, ls.games FROM league_standings ls
            WHERE ls.team_id = %s ORDER BY ls.season_id DESC LIMIT 1
        """, (home_id,))
        h_st = cur.fetchone()
        cur.execute("""
            SELECT ls.wins, ls.games FROM league_standings ls
            WHERE ls.team_id = %s ORDER BY ls.season_id DESC LIMIT 1
        """, (away_id,))
        a_st = cur.fetchone()
        cur.execute("""
            SELECT ts.goals, ts.games FROM team_squad_stats ts
            WHERE ts.team_id = %s AND ts.split = 'for'
            ORDER BY ts.season_id DESC LIMIT 1
        """, (home_id,))
        h_sq = cur.fetchone()
        cur.execute("""
            SELECT ts.goals, ts.games FROM team_squad_stats ts
            WHERE ts.team_id = %s AND ts.split = 'for'
            ORDER BY ts.season_id DESC LIMIT 1
        """, (away_id,))
        a_sq = cur.fetchone()

        h_gpg = _consensus_safe_div(h_sq["goals"] if h_sq else 0, h_sq["games"] if h_sq else 1, 1.2)
        a_gpg = _consensus_safe_div(a_sq["goals"] if a_sq else 0, a_sq["games"] if a_sq else 1, 1.0)
        h_wr  = _consensus_safe_div(h_st["wins"]  if h_st else 0, h_st["games"] if h_st else 1, 0.40)
        a_wr  = _consensus_safe_div(a_st["wins"]  if a_st else 0, a_st["games"] if a_st else 1, 0.35)

        h_str = 0.6 * h_gpg + 0.4 * h_wr
        a_str = 0.6 * a_gpg + 0.4 * a_wr
        total = h_str + a_str + 0.001
        r_h = max(0.1, h_str / total + 0.06)
        r_a = max(0.1, a_str / total - 0.03)
        r_d = max(0.1, 1 - r_h - r_a)
        s   = r_h + r_a + r_d
        home_p = round(r_h / s, 4)
        away_p = round(r_a / s, 4)
        draw_p = round(1 - home_p - away_p, 4)
        probs  = [home_p, draw_p, away_p]
        return {"home_win": home_p, "draw": draw_p, "away_win": away_p,
                "predicted_outcome": _OUTCOME_LABELS[probs.index(max(probs))]}
    except Exception as exc:
        log.warning("Legacy engine error: %s", exc)
        return fallback
    finally:
        try: cur.close()
        except: pass


def _blend_probs(dc: dict, ml: dict, legacy: dict, w: dict) -> dict:
    hw = w["dc"] * dc["home_win"] + w["ml"] * ml["home_win"] + w["legacy"] * legacy["home_win"]
    dr = w["dc"] * dc["draw"]     + w["ml"] * ml["draw"]     + w["legacy"] * legacy["draw"]
    aw = w["dc"] * dc["away_win"] + w["ml"] * ml["away_win"] + w["legacy"] * legacy["away_win"]
    t  = hw + dr + aw or 1.0
    return {"home_win": round(hw / t, 4), "draw": round(dr / t, 4), "away_win": round(aw / t, 4)}


def _entropy_confidence(probs: dict):
    import math
    vals        = [probs["home_win"], probs["draw"], probs["away_win"]]
    entropy     = -sum(p * math.log(max(p, 1e-10)) for p in vals)
    max_entropy = -math.log(1.0 / 3.0)
    score = round((1.0 - entropy / max_entropy) * 100, 1)
    label = "High" if score >= 30 else ("Medium" if score >= 15 else "Low")
    return score, label


@router.post("/consensus")
def consensus_predict(req: ConsensusRequest, background_tasks: BackgroundTasks):
    import math
    conn   = get_connection()
    errors = []
    try:
        # ── 1. Dynamic weights ─────────────────────────────────────────────────
        weights = _fetch_engine_weights(conn)
        weight_source = "dynamic_historical" if weights != _DEFAULT_WEIGHTS else "default_fallback"

        # ── 1b. Resolve team / league / season names BEFORE engine calls ───────
        # This ensures real names are logged even when the ML engine fails.
        _cur = conn.cursor()
        _cur.execute("SELECT id, name FROM teams WHERE id IN (%s, %s)",
                     (req.home_team_id, req.away_team_id))
        _name_map = {r["id"]: r["name"] for r in _cur.fetchall()}
        home_name   = _name_map.get(req.home_team_id, f"Team {req.home_team_id}")
        away_name   = _name_map.get(req.away_team_id, f"Team {req.away_team_id}")
        _cur.execute("SELECT name FROM leagues WHERE id = %s", (req.league_id,))
        _lg = _cur.fetchone()
        league_name = _lg["name"] if _lg else ""
        _cur.execute("SELECT name FROM seasons WHERE id = %s", (req.season_id,))
        _ss = _cur.fetchone()
        season_name = _ss["name"] if _ss else ""
        _cur.close()

        # ── 2. DC Engine ───────────────────────────────────────────────────────
        try:
            from ml.dc_engine import predict_dc_match
            dc_raw = predict_dc_match(req.home_team_id, req.away_team_id)
            if "error" in dc_raw:
                raise RuntimeError(dc_raw["error"])
            blended_key = dc_raw.get("calibrated") or dc_raw.get("blended") or {}
            dc_probs = {
                "home_win": float(blended_key.get("home_win", 0.35)),
                "draw":     float(blended_key.get("draw",     0.30)),
                "away_win": float(blended_key.get("away_win", 0.35)),
            }
            s = sum(dc_probs.values()) or 1
            dc_probs = {k: round(v / s, 4) for k, v in dc_probs.items()}
            dc_outcome = _OUTCOME_LABELS[[dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"]].index(max(dc_probs.values()))]
        except Exception as exc:
            log.warning("Consensus: DC error: %s", exc)
            errors.append(f"dc: {exc}")
            dc_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            dc_outcome = "Home Win"
            weights["dc"] = 0.0

        # ── 3. ML Engine ───────────────────────────────────────────────────────
        # home_name / away_name / league_name / season_name already resolved above;
        # only refine from ML result if it provides richer detail.
        expected_goals = {}
        try:
            import ml.prediction_engine as ml_engine
            ml_raw = ml_engine.predict_match(
                req.home_team_id, req.away_team_id, req.league_id, req.season_id)
            if "error" in ml_raw:
                raise RuntimeError(ml_raw["error"])
            ml_probs_inner = ml_raw.get("probabilities", {})
            ml_probs = {
                "home_win": float(ml_probs_inner.get("home_win", 0.35)),
                "draw":     float(ml_probs_inner.get("draw",     0.30)),
                "away_win": float(ml_probs_inner.get("away_win", 0.35)),
            }
            s = sum(ml_probs.values()) or 1
            ml_probs = {k: round(v / s, 4) for k, v in ml_probs.items()}
            ml_outcome = ml_raw.get("predicted_outcome", "Home Win")
            mi = ml_raw.get("match", {})
            # Refine names from ML result only if it returns valid strings
            home_name   = mi.get("home_team")  or home_name
            away_name   = mi.get("away_team")  or away_name
            league_name = mi.get("league")     or league_name
            season_name = mi.get("season")     or season_name
            expected_goals = ml_raw.get("expected_goals", {})
        except Exception as exc:
            log.warning("Consensus: ML error: %s", exc)
            errors.append(f"ml: {exc}")
            ml_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            ml_outcome = "Home Win"
            weights["ml"] = 0.0

        # ── 4. Legacy Engine ───────────────────────────────────────────────────
        try:
            legacy_raw     = _run_legacy_inline(conn, req.home_team_id, req.away_team_id)
            legacy_probs   = {k: v for k, v in legacy_raw.items()
                              if k in ("home_win", "draw", "away_win")}
            legacy_outcome = legacy_raw.get("predicted_outcome", "Home Win")
        except Exception as exc:
            log.warning("Consensus: Legacy error: %s", exc)
            errors.append(f"legacy: {exc}")
            legacy_probs   = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            legacy_outcome = "Home Win"
            weights["legacy"] = 0.0

        # ── 5. Re-normalise weights ────────────────────────────────────────────
        w_sum = sum(weights.values())
        weights = dict(_DEFAULT_WEIGHTS) if w_sum < 0.01 else {k: round(v / w_sum, 4) for k, v in weights.items()}

        # ── 6. Blend ───────────────────────────────────────────────────────────
        blended = _blend_probs(dc_probs, ml_probs, legacy_probs, weights)

        # ── 7. Outcome + confidence ────────────────────────────────────────────
        prob_list = [blended["home_win"], blended["draw"], blended["away_win"]]
        consensus_outcome = _OUTCOME_LABELS[prob_list.index(max(prob_list))]
        confidence_score, confidence_label = _entropy_confidence(blended)
        unique_outcomes = len(set([dc_outcome, ml_outcome, legacy_outcome]))
        agreement = "full" if unique_outcomes == 1 else ("majority" if unique_outcomes == 2 else "split")

        # ── 8. Markets ─────────────────────────────────────────────────────────
        home_xg  = float(expected_goals.get("home_xg", 1.35))
        away_xg  = float(expected_goals.get("away_xg", 1.10))
        btts_yes = max(0.0, min(1.0, round(
            1 - math.exp(-home_xg) - math.exp(-away_xg) + math.exp(-(home_xg + away_xg)), 4)))
        over_2_5 = max(0.0, min(1.0, round(
            1 - sum(math.exp(-(home_xg + away_xg)) * ((home_xg + away_xg) ** k) / math.factorial(k)
                    for k in range(3)), 4)))

        result = {
            "match": {
                "home_team":    home_name,
                "away_team":    away_name,
                "home_team_id": req.home_team_id,
                "away_team_id": req.away_team_id,
                "league":       league_name,
                "season":       season_name,
            },
            "consensus": {
                "home_win":          blended["home_win"],
                "draw":              blended["draw"],
                "away_win":          blended["away_win"],
                "predicted_outcome": consensus_outcome,
                "confidence":        confidence_label,
                "confidence_score":  confidence_score,
            },
            "engines": {
                "dc":     {**dc_probs,     "predicted_outcome": dc_outcome},
                "ml":     {**ml_probs,     "predicted_outcome": ml_outcome},
                "legacy": {**legacy_probs, "predicted_outcome": legacy_outcome},
            },
            "weights_used": {**weights, "source": weight_source},
            "agreement":    agreement,
            "markets": {
                "btts_yes":  btts_yes,
                "btts_no":   round(1 - btts_yes, 4),
                "over_2_5":  over_2_5,
                "under_2_5": round(1 - over_2_5, 4),
                "home_xg":   round(home_xg, 2),
                "away_xg":   round(away_xg, 2),
            },
            **({"_errors": errors} if errors else {}),
        }

        # ── 9. Log directly (synchronous) ──────────────────────────────────────
        # Fetch match_date from DB so it's stored even when caller didn't supply it.
        _match_date = None
        if req.match_id:
            try:
                _cur_date = conn.cursor()
                _cur_date.execute("SELECT match_date FROM matches WHERE id = %s", (req.match_id,))
                _dr = _cur_date.fetchone()
                if _dr and _dr["match_date"]:
                    _match_date = str(_dr["match_date"])
            except Exception:
                pass

        consensus = result["consensus"]
        _log_prediction_to_db(
            {
                "probabilities": {
                    "home_win": consensus["home_win"],
                    "draw":     consensus["draw"],
                    "away_win": consensus["away_win"],
                },
                "predicted_outcome": consensus["predicted_outcome"],
                "confidence":        consensus["confidence"],
                "confidence_score":  consensus["confidence_score"],
                "match":             {**result["match"], "match_date": _match_date},
                "markets":           result["markets"],
            },
            match_id=req.match_id,
            dc_outcome=dc_outcome,
            ml_outcome=ml_outcome,
            legacy_outcome=legacy_outcome,
            predicted_score=f"{round(home_xg)}-{round(away_xg)}",
        )

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try: conn.close()
        except: pass


# ─── Auto-consensus background job ────────────────────────────────────────────

_AUTO_CONSENSUS_INTERVAL_HOURS = 6
_auto_consensus_state: dict = {
    "last_run":       None,
    "last_count":     0,
    "last_error":     None,
    "interval_hours": _AUTO_CONSENSUS_INTERVAL_HOURS,
}


def _get_consensus_interval_hours() -> int:
    """Read consensus_interval_hours from app_settings, fallback to default."""
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("SELECT value FROM app_settings WHERE key = 'consensus_interval_hours' LIMIT 1")
        row = cur.fetchone()
        conn.close()
        if row and row["value"]:
            return max(1, min(int(row["value"]), 168))
    except Exception:
        pass
    return _AUTO_CONSENSUS_INTERVAL_HOURS


def _run_consensus_for_fixture(fx: dict, conn) -> bool:
    import math
    home_team_id = fx["home_team_id"]
    away_team_id = fx["away_team_id"]
    league_id    = fx["league_id"]
    season_id    = fx["season_id"]
    fixture_id   = fx["id"]
    home_name    = fx.get("home_name", f"Team {home_team_id}")
    away_name    = fx.get("away_name", f"Team {away_team_id}")
    league_name  = fx.get("league_name", "")
    season_name  = fx.get("season_name", "")

    weights = _fetch_engine_weights(conn)

    # DC
    try:
        from ml.dc_engine import predict_dc_match
        dc_raw      = predict_dc_match(home_team_id, away_team_id)
        if "error" in dc_raw: raise RuntimeError(dc_raw["error"])
        bk          = dc_raw.get("calibrated") or dc_raw.get("blended") or {}
        dc_probs    = {"home_win": float(bk.get("home_win", 0.35)), "draw": float(bk.get("draw", 0.30)), "away_win": float(bk.get("away_win", 0.35))}
        s           = sum(dc_probs.values()) or 1
        dc_probs    = {k: round(v / s, 4) for k, v in dc_probs.items()}
        dc_outcome  = _OUTCOME_LABELS[[dc_probs["home_win"], dc_probs["draw"], dc_probs["away_win"]].index(max(dc_probs.values()))]
    except Exception as exc:
        dc_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}; dc_outcome = "Home Win"; weights["dc"] = 0.0
        log.debug("Auto-consensus: DC error for %s vs %s: %s", home_name, away_name, exc)

    # ML
    expected_goals = {}
    try:
        import ml.prediction_engine as ml_engine
        ml_raw     = ml_engine.predict_match(home_team_id, away_team_id, league_id, season_id)
        if "error" in ml_raw: raise RuntimeError(ml_raw["error"])
        mpi        = ml_raw.get("probabilities", {})
        ml_probs   = {"home_win": float(mpi.get("home_win", 0.35)), "draw": float(mpi.get("draw", 0.30)), "away_win": float(mpi.get("away_win", 0.35))}
        s          = sum(ml_probs.values()) or 1
        ml_probs   = {k: round(v / s, 4) for k, v in ml_probs.items()}
        ml_outcome = ml_raw.get("predicted_outcome", "Home Win")
        mi         = ml_raw.get("match", {})
        home_name  = mi.get("home_team", home_name); away_name = mi.get("away_team", away_name)
        league_name = mi.get("league", league_name); season_name = mi.get("season", season_name)
        expected_goals = ml_raw.get("expected_goals", {})
    except Exception as exc:
        ml_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}; ml_outcome = "Home Win"; weights["ml"] = 0.0
        log.debug("Auto-consensus: ML error for %s vs %s: %s", home_name, away_name, exc)

    # Legacy
    try:
        legacy_raw    = _run_legacy_inline(conn, home_team_id, away_team_id)
        legacy_probs  = {k: v for k, v in legacy_raw.items() if k in ("home_win", "draw", "away_win")}
        legacy_outcome = legacy_raw.get("predicted_outcome", "Home Win")
    except Exception as exc:
        legacy_probs = {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}; legacy_outcome = "Home Win"; weights["legacy"] = 0.0
        log.debug("Auto-consensus: Legacy error for %s vs %s: %s", home_name, away_name, exc)

    w_sum   = sum(weights.values())
    weights = dict(_DEFAULT_WEIGHTS) if w_sum < 0.01 else {k: round(v / w_sum, 4) for k, v in weights.items()}
    blended = _blend_probs(dc_probs, ml_probs, legacy_probs, weights)
    prob_list = [blended["home_win"], blended["draw"], blended["away_win"]]
    consensus_outcome = _OUTCOME_LABELS[prob_list.index(max(prob_list))]
    confidence_score, confidence_label = _entropy_confidence(blended)

    home_xg  = float(expected_goals.get("home_xg", 1.35))
    away_xg  = float(expected_goals.get("away_xg", 1.10))
    btts_yes = max(0.0, min(1.0, round(
        1 - math.exp(-home_xg) - math.exp(-away_xg) + math.exp(-(home_xg + away_xg)), 4)))
    over_2_5 = max(0.0, min(1.0, round(
        1 - sum(math.exp(-(home_xg + away_xg)) * ((home_xg + away_xg) ** k) / math.factorial(k)
                for k in range(3)), 4)))

    _log_prediction_to_db(
        {
            "probabilities":     {"home_win": blended["home_win"], "draw": blended["draw"], "away_win": blended["away_win"]},
            "predicted_outcome": consensus_outcome,
            "confidence":        confidence_label,
            "confidence_score":  confidence_score,
            "match": {
                "home_team":    home_name,
                "away_team":    away_name,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "league":       league_name,
                "season":       season_name,
                "match_date":   str(fx.get("match_date")) if fx.get("match_date") else None,
            },
            "markets": {
                "btts_yes": btts_yes,
                "over_2_5": over_2_5,
                "home_xg":  round(home_xg, 2),
                "away_xg":  round(away_xg, 2),
            },
        },
        match_id=fixture_id,
        dc_outcome=dc_outcome,
        ml_outcome=ml_outcome,
        legacy_outcome=legacy_outcome,
        predicted_score=f"{round(home_xg)}-{round(away_xg)}",
    )
    return True


def auto_consensus_job():
    """
    Fetch all upcoming fixtures (next 7 days, unplayed) and run the
    Dynamic Consensus Engine on each. Results are automatically written
    to prediction_log (with dedup — existing rows are updated, not duplicated).
    Safe to call from a background thread.
    """
    global _auto_consensus_state
    _auto_consensus_state["last_error"] = None
    try:
        conn = get_connection()
        cur  = conn.cursor()
        cur.execute("""
            SELECT m.id, m.home_team_id, m.away_team_id,
                   m.league_id, m.season_id, m.match_date,
                   ht.name AS home_name, at.name AS away_name,
                   l.name  AS league_name, s.name AS season_name
            FROM   matches m
            JOIN   teams   ht ON ht.id = m.home_team_id
            JOIN   teams   at ON at.id = m.away_team_id
            JOIN   leagues l  ON l.id  = m.league_id
            JOIN   seasons s  ON s.id  = m.season_id
            WHERE  m.home_score IS NULL
              AND  m.match_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
            ORDER  BY m.match_date ASC
        """)
        fixtures = [dict(r) for r in cur.fetchall()]
        log.info("Auto-consensus job: found %d upcoming fixtures", len(fixtures))

        count = 0
        for fx in fixtures:
            try:
                _run_consensus_for_fixture(fx, conn)
                count += 1
            except Exception as exc:
                log.warning("Auto-consensus: skipped fixture id=%s (%s vs %s): %s",
                            fx.get("id"), fx.get("home_name"), fx.get("away_name"), exc)

        conn.close()
        import datetime
        _auto_consensus_state["last_run"]   = datetime.datetime.utcnow().isoformat() + "Z"
        _auto_consensus_state["last_count"] = count
        log.info("Auto-consensus job: logged/updated %d predictions", count)
    except Exception as exc:
        log.error("Auto-consensus job failed: %s", exc, exc_info=True)
        _auto_consensus_state["last_error"] = str(exc)


def _schedule_auto_consensus(interval_hours: int = None):
    def _run_and_reschedule():
        try: auto_consensus_job()
        finally: _schedule_auto_consensus()
    effective_hours = interval_hours or _get_consensus_interval_hours()
    _auto_consensus_state["interval_hours"] = effective_hours
    t = threading.Timer(effective_hours * 3600, _run_and_reschedule)
    t.daemon = True; t.name = "auto-consensus-scheduler"
    t.start()
    log.info("Auto-consensus next run in %dh", effective_hours)


@router.post("/auto-consensus")
def trigger_auto_consensus():
    """Manually trigger the automatic consensus prediction job for upcoming fixtures."""
    t = threading.Thread(target=auto_consensus_job, daemon=True, name="auto-consensus-manual")
    t.start()
    return {"started": True, "message": "Auto-consensus job started. Upcoming fixtures will be predicted and logged."}


@router.get("/auto-consensus/status")
def auto_consensus_status():
    """Return status of the last auto-consensus run."""
    return {
        "interval_hours":      _get_consensus_interval_hours(),
        "interval_hours_live": _auto_consensus_state.get("interval_hours", _AUTO_CONSENSUS_INTERVAL_HOURS),
        **{k: v for k, v in _auto_consensus_state.items() if k != "interval_hours"},
    }


# Start the scheduler 2 mins after import so the server finishes initialising first.
_initial_timer = threading.Timer(120, lambda: (_schedule_auto_consensus(), auto_consensus_job()))
_initial_timer.daemon = True
_initial_timer.name   = "auto-consensus-init"
_initial_timer.start()
log.info("Auto-consensus scheduler registered: first run in 2 min, then every %dh",
         _AUTO_CONSENSUS_INTERVAL_HOURS)

def _safe_div(a, b, default=0.0):
    try: return float(a) / float(b) if b else default
    except: return default


def _get_team_stats(cur, team_name: str):
    cur.execute("""
        SELECT ts.goals, ts.assists, ts.games, ts.possession, ts.avg_age
        FROM   team_squad_stats ts
        JOIN   teams t ON t.id = ts.team_id
        WHERE  LOWER(t.name) LIKE LOWER(%s) AND ts.split = 'for'
        ORDER  BY ts.season_id DESC LIMIT 1
    """, (f"%{team_name}%",))
    return cur.fetchone()


def _get_team_standings(cur, team_name: str):
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
        ORDER  BY m.match_date DESC LIMIT 1
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
        "goals_per_game": round(_safe_div(sq["goals"] if sq else 0, sq["games"] if sq else 1, 0), 2) if sq else None,
        "win_rate":       round(_safe_div(st["wins"]  if st else 0, st["games"] if st else 1, 0), 2) if st else None,
        "possession":     float(sq["possession"]) if sq and sq.get("possession") else None,
        "avg_age":        float(sq["avg_age"])    if sq and sq.get("avg_age")    else None,
        "goals_for":      int(st["goals_for"])    if st and st.get("goals_for")  else None,
        "goals_against":  int(st["goals_against"]) if st and st.get("goals_against") else None,
        "rank":           int(st["rank"])         if st and st.get("rank")       else None,
        "points":         int(st["points"])       if st and st.get("points")     else None,
    }


@router.post("/generate")
async def generate_prediction(req: GenerateRequest):
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
                "home_score":         ah,
                "away_score":         aa,
                "score_raw":          actual["score_raw"],
                "match_date":         str(actual["match_date"]) if actual.get("match_date") else None,
                "prediction_correct": pred_winner == actual_winner,
            }
        return {
            "success":         True,
            "home_team":       req.home_team,
            "away_team":       req.away_team,
            "home_win_prob":   home_p,
            "draw_prob":       draw_p,
            "away_win_prob":   away_p,
            "predicted_score": {"home": pred_h, "away": pred_a},
            "confidence":      conf,
            "home_stats":      _fmt_stats(h_sq, h_st),
            "away_stats":      _fmt_stats(a_sq, a_st),
            "actual_result":   actual_result,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        cur.close()
        conn.close()
