"""
Prediction Log — Standalone Performance Tracker
================================================
Tracks every prediction made by the ML engine against the actual result
once the match is completed. Gives real-world accuracy, not just training
cross-validation.

Endpoints:
  POST /api/prediction-log/record     → save a prediction before a match
  POST /api/prediction-log/evaluate   → scan completed matches, mark correct/wrong
  GET  /api/prediction-log/accuracy   → real-world accuracy stats and trend
  GET  /api/prediction-log            → full log with pagination

SQL to run ONCE in Supabase SQL Editor:
----------------------------------------
  CREATE TABLE IF NOT EXISTS prediction_log (
      id              SERIAL PRIMARY KEY,
      match_id        INT,            -- references matches(id), nullable (future fixtures)
      home_team       TEXT NOT NULL,
      away_team       TEXT NOT NULL,
      league          TEXT,
      match_date      DATE,
      predicted       TEXT NOT NULL,  -- 'Home Win' / 'Draw' / 'Away Win'
      confidence      TEXT,           -- 'High' / 'Medium' / 'Low'
      confidence_score FLOAT,
      home_win_prob   FLOAT,
      draw_prob       FLOAT,
      away_win_prob   FLOAT,
      actual          TEXT,           -- filled in by /evaluate after match
      correct         BOOLEAN,        -- filled in by /evaluate
      evaluated_at    TIMESTAMPTZ,
      created_at      TIMESTAMPTZ DEFAULT NOW()
  );
  CREATE INDEX IF NOT EXISTS prediction_log_match_id_idx ON prediction_log(match_id);
  CREATE INDEX IF NOT EXISTS prediction_log_correct_idx  ON prediction_log(correct);
  CREATE INDEX IF NOT EXISTS prediction_log_date_idx     ON prediction_log(match_date);
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from database import get_connection

router = APIRouter()


# ─── Request models ───────────────────────────────────────────────────────────

class RecordRequest(BaseModel):
    """Payload sent by the frontend/engine when a prediction is made."""
    match_id:        Optional[int]   = None
    home_team:       str
    away_team:       str
    league:          Optional[str]   = None
    match_date:      Optional[str]   = None   # ISO date string "YYYY-MM-DD"
    predicted:       str                      # "Home Win" / "Draw" / "Away Win"
    confidence:      Optional[str]   = None
    confidence_score: Optional[float] = None
    home_win_prob:   Optional[float] = None
    draw_prob:       Optional[float] = None
    away_win_prob:   Optional[float] = None


# ─── Helper ───────────────────────────────────────────────────────────────────

def _outcome_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:  return "Home Win"
    if away_score > home_score:  return "Away Win"
    return "Draw"


# ─── POST /record ─────────────────────────────────────────────────────────────

@router.post("/record")
def record_prediction(req: RecordRequest):
    """
    Save one prediction to the log before the match is played.
    Call this immediately after /api/predictions/predict returns.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO prediction_log
                (match_id, home_team, away_team, league, match_date,
                 predicted, confidence, confidence_score,
                 home_win_prob, draw_prob, away_win_prob)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            req.match_id, req.home_team, req.away_team,
            req.league,
            req.match_date,
            req.predicted, req.confidence, req.confidence_score,
            req.home_win_prob, req.draw_prob, req.away_win_prob,
        ))
        row = cur.fetchone()
        conn.commit()
        return {"saved": True, "log_id": row["id"]}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── POST /evaluate ───────────────────────────────────────────────────────────

@router.post("/evaluate")
def evaluate_predictions():
    """
    Scan all un-evaluated prediction_log rows whose match has now completed
    (i.e. matches table has home_score populated) and mark them correct/wrong.

    Safe to call repeatedly — only updates rows where correct IS NULL.
    Run this after each sync cycle or manually from the admin panel.
    Returns how many rows were evaluated.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        # Find unevaluated log rows that have a match_id and the match is now done
        cur.execute("""
            SELECT pl.id, pl.predicted, m.home_score, m.away_score
            FROM prediction_log pl
            JOIN matches m ON m.id = pl.match_id
            WHERE pl.correct IS NULL
              AND m.home_score IS NOT NULL
              AND m.away_score IS NOT NULL
        """)
        rows = cur.fetchall()

        updated = 0
        for r in rows:
            actual  = _outcome_label(int(r["home_score"]), int(r["away_score"]))
            correct = (actual == r["predicted"])
            cur.execute("""
                UPDATE prediction_log
                SET actual = %s, correct = %s, evaluated_at = NOW()
                WHERE id = %s
            """, (actual, correct, r["id"]))
            updated += 1

        conn.commit()
        return {
            "evaluated": updated,
            "message": f"{updated} prediction(s) graded against real results.",
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── GET /accuracy ────────────────────────────────────────────────────────────

@router.get("/accuracy")
def real_world_accuracy(
    league:    Optional[str] = Query(None, description="Filter by league name"),
    last_n:    int           = Query(50,   description="Consider last N evaluated predictions"),
):
    """
    Real-world accuracy computed from evaluated prediction_log rows.
    Unlike cv_accuracy (training data), this reflects actual match outcomes.
    Returns overall accuracy + breakdown by predicted outcome + recent trend.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        base = """
            FROM prediction_log
            WHERE correct IS NOT NULL
        """
        params: list = []
        if league:
            base += " AND league = %s"; params.append(league)

        # Overall accuracy
        cur.execute("SELECT COUNT(*) AS total, SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_count" + base, params)
        overall = cur.fetchone()
        total         = int(overall["total"] or 0)
        correct_count = int(overall["correct_count"] or 0)
        overall_pct   = round(correct_count / total * 100, 1) if total else None

        # Accuracy by predicted outcome
        cur.execute("""
            SELECT predicted,
                   COUNT(*) AS total,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_count
        """ + base + " GROUP BY predicted ORDER BY predicted", params)
        by_outcome = [
            {
                "predicted":   r["predicted"],
                "total":       int(r["total"]),
                "correct":     int(r["correct_count"] or 0),
                "accuracy_pct": round(int(r["correct_count"] or 0) / int(r["total"]) * 100, 1)
                               if int(r["total"]) > 0 else None,
            }
            for r in cur.fetchall()
        ]

        # Accuracy by confidence tier
        cur.execute("""
            SELECT confidence,
                   COUNT(*) AS total,
                   SUM(CASE WHEN correct THEN 1 ELSE 0 END) AS correct_count
        """ + base + " AND confidence IS NOT NULL GROUP BY confidence ORDER BY confidence", params)
        by_confidence = [
            {
                "confidence": r["confidence"],
                "total":      int(r["total"]),
                "correct":    int(r["correct_count"] or 0),
                "accuracy_pct": round(int(r["correct_count"] or 0) / int(r["total"]) * 100, 1)
                                if int(r["total"]) > 0 else None,
            }
            for r in cur.fetchall()
        ]

        # Rolling accuracy on last N evaluated predictions (trend)
        cur.execute("""
            SELECT correct, evaluated_at
        """ + base + f" ORDER BY evaluated_at DESC LIMIT {last_n}", params)
        recent = cur.fetchall()
        if recent:
            recent_correct = sum(1 for r in recent if r["correct"])
            recent_pct     = round(recent_correct / len(recent) * 100, 1)
        else:
            recent_pct = None

        return {
            "total_evaluated":  total,
            "correct":          correct_count,
            "overall_accuracy": overall_pct,
            f"last_{last_n}_accuracy": recent_pct,
            "by_outcome":       by_outcome,
            "by_confidence":    by_confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()


# ─── GET / (full log) ─────────────────────────────────────────────────────────

@router.get("")
def list_prediction_log(
    league:    Optional[str] = Query(None),
    correct:   Optional[bool] = Query(None, description="true=right, false=wrong, omit=all"),
    evaluated: Optional[bool] = Query(None, description="true=evaluated only, false=pending"),
    limit:     int            = Query(50),
    offset:    int            = Query(0),
):
    """
    Return prediction log rows with optional filters.
    Useful for an admin panel or dashboard table.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        query  = "SELECT * FROM prediction_log WHERE 1=1"
        params: list = []
        if league:
            query += " AND league = %s"; params.append(league)
        if correct is not None:
            query += " AND correct = %s"; params.append(correct)
        if evaluated is True:
            query += " AND correct IS NOT NULL"
        elif evaluated is False:
            query += " AND correct IS NULL"
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params += [limit, offset]
        cur.execute(query, params)
        rows = cur.fetchall()
        return {"count": len(rows), "rows": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
