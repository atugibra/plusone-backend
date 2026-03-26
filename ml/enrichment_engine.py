"""
Enrichment Machine Learning Engine
===================================
A standalone predictive engine strictly trained on Enrichment Data:
- ClubElo Goal Expectancy & Ratings
- Transfermarkt Injury Squad Decimation (Market Values)
- Historical Betting Odds Movement

This isolated engine guarantees that if a scraper drops missing data, the core DC/ML engines survive untouched, allowing Consensus to dynamically discard this engine's output.
"""

import os
import time
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from database import get_connection

from ml.enrichment_features import build_enrichment_features

log = logging.getLogger(__name__)

# ─── Singleton state ──────────────────────────────────────────────────────────

_engine = None

class EnrichmentPredictor:
    def __init__(self):
        # We use a blend of RF (low variance) and XGB (high accuracy)
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, 
                                 eval_metric='mlogloss', random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names_ = []
        self.n_samples = 0
        
    def train(self, X_df: pd.DataFrame, y_series: pd.Series):
        self.feature_names_ = list(X_df.columns)
        self.n_samples = len(X_df)
        
        X_scaled = self.scaler.fit_transform(X_df)
        self.rf.fit(X_scaled, y_series)
        self.xgb.fit(X_scaled, y_series)
        
        self.is_trained = True
        
    def predict_proba(self, feature_dict: dict) -> dict:
        if not self.is_trained:
            return {"home_win": 0.35, "draw": 0.30, "away_win": 0.35}
            
        row = []
        for col in self.feature_names_:
            row.append(feature_dict.get(col, 0.0))
            
        X_arr = np.array([row])
        X_scaled = self.scaler.transform(X_arr)
        
        rf_p = self.rf.predict_proba(X_scaled)[0]
        xgb_p = self.xgb.predict_proba(X_scaled)[0]
        
        # Blend 50/50
        hw = (rf_p[0] + xgb_p[0]) / 2.0
        dr = (rf_p[1] + xgb_p[1]) / 2.0
        aw = (rf_p[2] + xgb_p[2]) / 2.0
        
        return {"home_win": round(hw, 4), "draw": round(dr, 4), "away_win": round(aw, 4)}

    def save(self, filepath="enrichment_model.pkl"):
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            log.error(f"Failed to save enrichment model: {e}")

    @staticmethod
    def load(filepath="enrichment_model.pkl"):
        if os.path.exists(filepath):
            try:
                with open(filepath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                log.error(f"Failed to load enrichment model: {e}")
        return None

def _get_enrichment_engine() -> EnrichmentPredictor:
    global _engine
    if _engine is None:
        loaded = EnrichmentPredictor.load()
        if loaded and loaded.is_trained:
            _engine = loaded
    return _engine

# ─── Training Orchestrator ───────────────────────────────────────────────────

def train_enrichment_model() -> dict:
    """
    Extracts all historical matches, pulls their enrichment states at that point in time,
    trains the models, and persists the singleton.
    """
    global _engine
    conn = get_connection()
    cur = conn.cursor()
    try:
        t0 = time.time()
        
        # Target only matches with valid scores
        cur.execute("""
            SELECT id, home_team_id, away_team_id, match_date, home_score, away_score
            FROM matches
            WHERE home_score IS NOT NULL AND away_score IS NOT NULL
        """)
        matches = cur.fetchall()
        
        X_list = []
        y_list = []
        
        skipped = 0
        for m in matches:
            # We map outcomes to: 0 = Home, 1 = Draw, 2 = Away
            if m["home_score"] > m["away_score"]:
                outcome = 0
            elif m["home_score"] == m["away_score"]:
                outcome = 1
            else:
                outcome = 2
                
            m_date = str(m["match_date"]) if m["match_date"] else None
            try:
                feats = build_enrichment_features(cur, m["home_team_id"], m["away_team_id"], m_date)
                
                # We skip matches where ClubElo hasn't populated (value left as default 1500)
                # to prevent polluting the tree model with dead averages.
                if feats.get("clubelo_home_rating", 1500) == 1500 and feats.get("odds_home_prob", 0.35) == 0.35:
                    skipped += 1
                    continue
                    
                X_list.append(feats)
                y_list.append(outcome)
            except Exception as e:
                skipped += 1
                continue
                
        if len(X_list) < 20:
            return {"success": False, "error": f"Insufficient enriched training data ({len(X_list)} rows).", "trained": 0}
            
        X_df = pd.DataFrame(X_list)
        y_series = pd.Series(y_list)
        
        # Replace completely missing NaNs with 0
        X_df = X_df.fillna(0.0)
        
        model = EnrichmentPredictor()
        model.train(X_df, y_series)
        model.save()
        _engine = model
        
        elapsed = round(time.time() - t0, 1)
        
        return {
            "success": True,
            "trained_rows": len(X_list),
            "skipped_rows": skipped,
            "elapsed_seconds": elapsed,
            "n_features": len(model.feature_names_)
        }
    finally:
        conn.close()

# ─── Inference API ────────────────────────────────────────────────────────────

def predict_enrichment(home_team_id: int, away_team_id: int, match_date: str = None) -> dict:
    """
    Returns the enrichment model probabilities for a single upcoming match.
    """
    eng = _get_enrichment_engine()
    if eng is None or not eng.is_trained:
        return {"error": "Enrichment model not trained.", "home_win": 0.35, "draw": 0.30, "away_win": 0.35}
        
    conn = get_connection()
    cur = conn.cursor()
    try:
        feats = build_enrichment_features(cur, home_team_id, away_team_id, match_date)
        probs = eng.predict_proba(feats)
        
        idx = np.argmax([probs["home_win"], probs["draw"], probs["away_win"]])
        outcome_str = {0: "Home Win", 1: "Draw", 2: "Away Win"}[int(idx)]
        
        probs["predicted_outcome"] = outcome_str
        # Attach the raw features so we can display them in the console if needed
        probs["_features"] = feats
        
        return probs
    finally:
        conn.close()
