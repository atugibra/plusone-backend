"""
Prediction Performance API Routes
===================================
GET /api/performance             — Overall metrics (Brier, RPS, accuracy, ROI)
GET /api/performance/drift       — Rolling 20-match Brier score over time
GET /api/performance/calibration — Calibration bin table
GET /api/performance/per-league  — Per-league accuracy and Brier breakdown
GET /api/performance/confusion   — Confusion matrix (where is the model wrong?)

All metrics are computed from the `prediction_log` table in Supabase.
Only rows where result_recorded = TRUE are included.
"""

import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from database import get_connection
from ml.metrics import MetricsEngine

router = APIRouter()
log    = logging.getLogger(__name__)


# ─── Helper: load completed predictions from DB ───────────────────────────────

def _load_completed(cur, league: str = None) -> pd.DataFrame:
    """Load completed predictions from prediction_log table."""
    query = """
        SELECT
            id, home_team, away_team, league, match_date,
            prob_home_win, prob_draw, prob_away_win,
            predicted_outcome, actual_outcome, correct,
            market_home_odds, market_draw_odds, market_away_odds,
            confidence_score
        FROM prediction_log
        WHERE actual IS NOT NULL
          AND home_win_prob   IS NOT NULL
          AND draw_prob       IS NOT NULL
          AND away_win_prob   IS NOT NULL
    """
    params = []
    if league:
        query  += " AND league = %s"
        params.append(league)
    query += " ORDER BY match_date ASC"

    try:
        cur.execute(query, params)
        rows = cur.fetchall()
    except Exception as e:
        log.warning("prediction_log query failed: %s", e)
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])

    # Map column names — prediction_log uses home_win_prob etc
    col_map = {
        "home_win_prob": "prob_home_win",
        "draw_prob":     "prob_draw",
        "away_win_prob": "prob_away_win",
    }
    df = df.rename(columns=col_map)

    # Map actual text outcome → int (0=Away Win, 1=Draw, 2=Home Win)
    outcome_map = {
        "Home Win": 2, "home_win": 2, "2": 2,
        "Draw":     1, "draw":     1, "1": 1,
        "Away Win": 0, "away_win": 0, "0": 0,
    }
    if "actual_outcome" in df.columns:
        df["actual_int"] = df["actual_outcome"].astype(str).map(outcome_map)
    elif "actual" in df.columns:
        df["actual_int"] = df["actual"].astype(str).map(outcome_map)
    else:
        return pd.DataFrame()

    # Map predicted outcome similarly
    if "predicted_outcome" in df.columns:
        df["predicted_int"] = df["predicted_outcome"].astype(str).map(outcome_map)
    elif "predicted" in df.columns:
        df["predicted_int"] = df["predicted"].astype(str).map(outcome_map)

    df = df.dropna(subset=["actual_int", "prob_home_win", "prob_draw", "prob_away_win"])
    return df


def _build_matrices(df: pd.DataFrame):
    """Build (N,3) probs array [p_aw, p_d, p_hw] and (N,) outcomes array."""
    p_aw = df["prob_away_win"].astype(float).values
    p_d  = df["prob_draw"].astype(float).values
    p_hw = df["prob_home_win"].astype(float).values
    probs    = np.stack([p_aw, p_d, p_hw], axis=1)
    outcomes = df["actual_int"].astype(int).values
    return probs, outcomes


# ─── GET /api/performance ─────────────────────────────────────────────────────

@router.get("")
def get_overall_performance(league: str = None):
    """
    Overall model performance metrics from the prediction_log table.
    Includes: accuracy, Brier score, RPS, log loss, statistical significance.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur, league)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions in prediction_log yet.",
                "n": 0}

    probs, outcomes = _build_matrices(df)
    summary = MetricsEngine.full_summary(probs, outcomes)

    # ROI
    roi_records = []
    for _, r in df.iterrows():
        pred = r.get("predicted_int")
        if pd.isna(pred):
            continue
        pred = int(pred)
        odds_cols = {2: "market_home_odds", 1: "market_draw_odds", 0: "market_away_odds"}
        col = odds_cols.get(pred)
        if col and col in r and pd.notna(r[col]):
            roi_records.append({
                "predicted_outcome": pred,
                "actual_outcome":    int(r["actual_int"]),
                "odds_taken":        float(r[col]),
            })

    roi = MetricsEngine.roi(roi_records) if roi_records else {}

    return {
        **summary,
        "roi":    roi,
        "league": league or "All Leagues",
    }


# ─── GET /api/performance/drift ───────────────────────────────────────────────

@router.get("/drift")
def get_rolling_drift(window: int = 20):
    """
    Rolling Brier score and accuracy over time (window = N most recent matches).
    Helps detect if model is degrading mid-season.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty or len(df) < window:
        return {"message": f"Need at least {window} completed predictions. Have {len(df)}.",
                "n": len(df)}

    df = df.reset_index(drop=True)
    rows = []
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i - window: i]
        probs, outcomes = _build_matrices(chunk)
        rows.append({
            "match_number":  i,
            "rolling_brier": round(MetricsEngine.brier_score(probs, outcomes), 4),
            "rolling_rps":   round(MetricsEngine.rps(probs, outcomes), 4),
            "rolling_acc":   round(MetricsEngine.accuracy(probs, outcomes) * 100, 2),
        })

    df_roll = pd.DataFrame(rows)
    latest  = rows[-1] if rows else {}
    return {
        "window":        window,
        "n_predictions": len(df),
        "latest":        latest,
        "drift":         rows,
    }


# ─── GET /api/performance/calibration ─────────────────────────────────────────

@router.get("/calibration")
def get_calibration():
    """
    Calibration table: when the model says X%, does it win X% of the time?
    A well-calibrated model's curve lies close to the diagonal.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions yet.", "bins": []}

    probs, outcomes = _build_matrices(df)
    cal_df = MetricsEngine.calibration(probs, outcomes)

    well_calibrated = int((cal_df["well_calibrated"]).sum())
    return {
        "n_predictions":  len(df),
        "n_bins":         len(cal_df),
        "well_calibrated_bins": well_calibrated,
        "bins":           cal_df.to_dict(orient="records"),
    }


# ─── GET /api/performance/per-league ─────────────────────────────────────────

@router.get("/per-league")
def get_per_league():
    """Per-league accuracy and Brier score breakdown."""
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty or "league" not in df.columns:
        return {"message": "No completed predictions yet.", "leagues": []}

    rows = []
    for league, grp in df.groupby("league"):
        if len(grp) < 5:
            continue
        probs, outcomes = _build_matrices(grp)
        rows.append({
            "league":    league,
            "matches":   len(grp),
            "accuracy":  round(MetricsEngine.accuracy(probs, outcomes) * 100, 2),
            "brier":     round(MetricsEngine.brier_score(probs, outcomes), 4),
            "rps":       round(MetricsEngine.rps(probs, outcomes), 4),
        })

    rows.sort(key=lambda x: x["brier"])
    return {"leagues": rows}


# ─── GET /api/performance/confusion ──────────────────────────────────────────

@router.get("/confusion")
def get_confusion_matrix():
    """
    Confusion matrix: where is the model going wrong?
    Shows how often Home Wins are predicted as Draws, Draws as Away Wins, etc.
    """
    conn = get_connection()
    cur  = conn.cursor()
    try:
        df = _load_completed(cur)
    finally:
        conn.close()

    if df.empty:
        return {"message": "No completed predictions yet."}

    probs, outcomes = _build_matrices(df)
    cm = MetricsEngine.confusion_matrix(probs, outcomes)

    return {
        "n_predictions": len(df),
        "matrix": cm.to_dict(),
        "labels": ["Away Win", "Draw", "Home Win"],
    }
