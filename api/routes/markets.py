"""
Betting Markets API Routes
===========================
GET  /api/markets              — Full market sheet for a fixture (needs DC model trained)
POST /api/markets/value        — Find value bets given bookmaker odds
GET  /api/markets/arb          — Scan for arb across multiple bookmakers
GET  /api/markets/dc/status    — DC model status
POST /api/markets/dc/train     — Train (or retrain) DC model
GET  /api/markets/dc/leaderboard — Elo leaderboard
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import Optional
import logging

from ml.dc_engine  import (predict_dc_match, train_dc_model,
                            dc_status, get_dc_predictor)
from ml.markets    import MarketPricer, ValueDetector, ArbitrageScanner

router = APIRouter()
log = logging.getLogger(__name__)


# ─── GET /api/markets ─────────────────────────────────────────────────────────

@router.get("")
def get_markets(
    home_team_id: int = Query(..., description="Home team ID"),
    away_team_id: int = Query(..., description="Away team ID"),
    n_sim: int        = Query(100_000, ge=10_000, le=500_000,
                              description="Monte Carlo simulations"),
):
    """
    Full betting market sheet for a fixture.
    Expected goals come from the DC ensemble model.
    Includes: 1X2, Asian Handicap, O/U, BTTS, Correct Score,
              Double Chance, Draw No Bet, Win to Nil, Team Goals.
    """
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(
            status_code=503,
            detail="DC model not trained. POST /api/markets/dc/train first.",
        )

    result = predict_dc_match(home_team_id, away_team_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    lam_h = result.get("exp_home_goals", 1.35)
    lam_a = result.get("exp_away_goals", 1.10)

    pricer = MarketPricer(lam_h, lam_a, n_sim=n_sim)
    sheet  = pricer.full_sheet()

    return {
        "fixture": {
            "home_team":    result["home_team"],
            "away_team":    result["away_team"],
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
        },
        "ensemble_prediction": {
            "calibrated":   result["calibrated"],
            "prediction":   result["prediction"],
            "confidence":   result["confidence"],
            "models":       result["models"],
        },
        "markets": sheet,
        "model_info": result.get("model_info", {}),
    }


# ─── POST /api/markets/value ──────────────────────────────────────────────────

class ValueRequest(BaseModel):
    home_team_id: int
    away_team_id: int
    market_odds: dict        # {"home_win": 2.45, "draw": 3.40, "away_win": 2.95}
    min_edge_pct: float = 3.0


@router.post("/value")
def get_value_bets(req: ValueRequest):
    """
    Find value bets by comparing DC model probabilities vs supplied bookmaker odds.
    Returns all outcomes where model edge > min_edge_pct%.
    """
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(status_code=503,
                            detail="DC model not trained. POST /api/markets/dc/train first.")

    result = predict_dc_match(req.home_team_id, req.away_team_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    cal = result["calibrated"]
    model_probs = {
        "home_win": cal["home_win"],
        "draw":     cal["draw"],
        "away_win": cal["away_win"],
    }

    detector = ValueDetector(min_edge=req.min_edge_pct / 100)
    bets = detector.scan(model_probs, req.market_odds, market_name="1x2")

    return {
        "fixture": {
            "home_team": result["home_team"],
            "away_team": result["away_team"],
        },
        "model_probs":  model_probs,
        "value_bets":   bets,
        "n_value_bets": len(bets),
    }


# ─── GET /api/markets/arb ─────────────────────────────────────────────────────

class ArbRequest(BaseModel):
    books_odds: dict    # {"home_win": {"Pinnacle": 2.45, "Bet365": 2.40}, ...}


@router.post("/arb")
def scan_arbitrage(req: ArbRequest):
    """
    Scan for arbitrage opportunities across multiple bookmakers.
    Pass {outcome: {bookmaker: decimal_odds}} and we'll find risk-free arbs.
    """
    result = ArbitrageScanner.find_arb(req.books_odds)
    return result


# ─── GET /api/markets/dc/status ──────────────────────────────────────────────

@router.get("/dc/status")
def get_dc_status():
    """Current DC model status."""
    return dc_status()


# ─── POST /api/markets/dc/train ──────────────────────────────────────────────

@router.post("/dc/train")
def train_dc(background_tasks: BackgroundTasks):
    """
    Train (or retrain) the DC + Elo + xG ensemble from DB data.
    Runs asynchronously — check /dc/status for completion.
    """
    def _train():
        result = train_dc_model()
        if result.get("trained"):
            log.info("DC retrain complete: %d matches", result.get("n_matches", 0))
        else:
            log.error("DC retrain failed: %s", result.get("error"))

    background_tasks.add_task(_train)
    return {"message": "DC model training started in background. Check /api/markets/dc/status."}


# ─── GET /api/markets/dc/predict ─────────────────────────────────────────────

@router.get("/dc/predict")
def dc_predict(
    home_team_id: int = Query(...),
    away_team_id: int = Query(...),
):
    """
    Full DC ensemble prediction for a fixture.
    Returns per-model breakdown (Dixon-Coles, Elo, xG, Ensemble, Calibrated),
    Monte Carlo markets, and confidence score.
    """
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(status_code=503,
                            detail="DC model not trained. POST /api/markets/dc/train first.")
    result = predict_dc_match(home_team_id, away_team_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ─── GET /api/markets/dc/leaderboard ─────────────────────────────────────────

@router.get("/dc/leaderboard")
def elo_leaderboard():
    """Elo rating leaderboard across all teams in the DC model."""
    dc = get_dc_predictor()
    if dc is None or not dc.fitted:
        raise HTTPException(status_code=503,
                            detail="DC model not trained. POST /api/markets/dc/train first.")
    return {"leaderboard": dc.elo_leaderboard()}
