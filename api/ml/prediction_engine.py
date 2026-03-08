"""
Prediction Engine for PlusOne
==================================
Singleton-style engine that:
  1. Trains on ALL historical matches from the database
  2. Predicts match outcomes with rich, detailed output
  3. Persists trained model to disk (survives API restarts)
  4. Auto-loads saved model on startup if present
"""

import os
import time
from database import get_connection
from ml.feature_engineering import (
    build_match_features,
    build_training_dataset,
    compute_h2h,
)
from ml.ml_models import EnsemblePredictor
from ml.model_store import save_to_db, load_from_db
from ml.batch_features import build_training_dataset_fast, DataCache, _build_team_features

# ─── Singleton state ──────────────────────────────────────────────────────────

_engine: EnsemblePredictor = None
_meta = {
    "trained_at": None,
    "n_samples": 0,
    "train_accuracy": None,
    "cv_accuracy": None,
    "errors": 0,
    "feature_names": [],
}

OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}


def _get_engine() -> EnsemblePredictor:
    global _engine
    if _engine is None:
        # Try Supabase first (survives redeploys), fall back to local disk
        loaded = load_from_db()
        if loaded is None:
            loaded = EnsemblePredictor.load()
        if loaded and loaded.is_trained:
            _engine = loaded
            _meta["n_samples"]      = loaded.n_samples
            _meta["train_accuracy"] = loaded.train_accuracy
            _meta["cv_accuracy"]    = loaded.cv_accuracy
            _meta["feature_names"]  = loaded.feature_names_ or []
    return _engine


# ─── Training ─────────────────────────────────────────────────────────────────

def train_model():
    """
    Pull all completed matches from the database, build 70+ feature vectors,
    train the XGBoost+RandomForest ensemble, and save to disk.

    This may take 30-120 seconds depending on data size.
    Returns a status dict with: success, matches_trained, train_accuracy,
    cv_accuracy, errors_skipped, elapsed_seconds, n_features.
    """
    global _engine, _meta
    conn = get_connection()
    cur  = conn.cursor()
    try:
        t0 = time.time()
        # Fast path: bulk-load all tables in ~6 queries, build features in-memory
        X, y, match_ids, errors = build_training_dataset_fast(cur, skip_errors=True)
        conn.close()

        if len(X) < 20:
            return {
                "success": False,
                "error": f"Not enough training data. Found {len(X)} completed matches — need at least 20.",
                "matches_found": len(X),
            }

        model = EnsemblePredictor()

        conn3 = get_connection()
        cur3  = conn3.cursor()
        feat_names = []
        try:
            cur3.execute("""
                SELECT m.id, m.home_team_id, m.away_team_id, m.league_id, m.season_id
                FROM matches m
                WHERE m.home_score IS NOT NULL
                  AND m.league_id IS NOT NULL AND m.season_id IS NOT NULL
                LIMIT 1
            """)
            sample = cur3.fetchone()
            if sample:
                _, feat_names, _, _, _ = build_match_features(
                    cur3,
                    sample["home_team_id"],
                    sample["away_team_id"],
                    sample["league_id"],
                    sample["season_id"],
                )
        except Exception:
            pass
        finally:
            conn3.close()

        model.train(X, y, feature_names=feat_names or None)
        # Save to disk (fast, local backup) AND Supabase (survives redeploys)
        model.save()
        save_to_db(model)

        _engine = model
        _meta.update({
            "trained_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_samples":     model.n_samples,
            "train_accuracy":model.train_accuracy,
            "cv_accuracy":   model.cv_accuracy,
            "errors":        errors,
            "feature_names": model.feature_names_ or [],
        })

        elapsed = round(time.time() - t0, 1)
        return {
            "success": True,
            "matches_trained":  model.n_samples,
            "train_accuracy":   round(model.train_accuracy or 0, 4),
            "cv_accuracy":      round(model.cv_accuracy    or 0, 4),
            "errors_skipped":   errors,
            "elapsed_seconds":  elapsed,
            "n_features":       len(model.feature_names_ or []),
        }
    except Exception:
        conn.close()
        raise


# ─── Prediction ───────────────────────────────────────────────────────────────

def _compute_venue_xg(attacker_feats: dict, defender_feats: dict, venue: str) -> float:
    """
    Dixon-Coles style Expected Goals using real venue-split data from team_venue_stats.

    Attack rate: attacker's home_gf_pg (venue='home') or away_gf_pg (venue='away').
                 Falls back to season avg goals_for_pg if venue data is unavailable.
    Defence rate: opponent's concession rate at their venue (home defenders at home,
                 away defenders on the road).
    form_gf_avg: recent-form goals blended in as a short-term signal (35% weight).
    """
    # ── Attack: venue-specific goal rate from team_venue_stats ─────────────
    if venue == "home":
        attack_rate  = attacker_feats.get("home_gf_pg", 0.0)
        defence_rate = defender_feats.get("away_ga_pg", 0.0)   # away team concedes on road
    else:
        attack_rate  = attacker_feats.get("away_gf_pg", 0.0)
        defence_rate = defender_feats.get("home_ga_pg", 0.0)   # home team concedes at home

    # ── Fallbacks if venue data missing ────────────────────────────────
    season_gf = attacker_feats.get("goals_for_pg", 0.0)
    form_gf   = attacker_feats.get("form_gf_avg",  0.0)

    if attack_rate < 0.1:
        attack_rate = season_gf if season_gf > 0.1 else (form_gf if form_gf > 0.1 else 1.35)
    if defence_rate < 0.1:
        defence_rate = defender_feats.get("goals_against_pg", 1.35)

    # ── Blend in recent form (last-5 actual goals) as 35% short-term signal ──
    if form_gf > 0.1:
        attack_rate = 0.65 * attack_rate + 0.35 * form_gf

    # ── Dixon-Coles defensive adjustment ──────────────────────────────
    # defence_rate > 1.35 means leaky defence → attacker scores more
    # defence_rate < 1.35 means solid defence → attacker scores less
    LEAGUE_AVG_GA = 1.35
    defence_factor = defence_rate / LEAGUE_AVG_GA if defence_rate > 0 else 1.0
    xg = attack_rate * defence_factor

    # ── Clamp to realistic range ─────────────────────────────────────
    return round(max(0.5, min(xg, 5.0)), 2)


def _key_factors(home_feats: dict, away_feats: dict,
                 home_name: str, away_name: str) -> list:
    factors = []

    def pct_diff(h, a, label, higher_better=True):
        if h == 0 and a == 0:
            return
        diff = h - a
        if abs(diff) < 0.01:
            return
        better = home_name if (diff > 0) == higher_better else away_name
        worse  = away_name if better == home_name else home_name
        pct    = abs(round(diff / max(abs(a), abs(h), 0.001) * 100, 1))
        factors.append(f"{better} {pct}% better {label} than {worse}")

    pct_diff(home_feats.get("goals_per90",           0),
             away_feats.get("goals_per90",           0), "goals/90")
    pct_diff(home_feats.get("shots_on_target_per90", 0),
             away_feats.get("shots_on_target_per90", 0), "shots on target/90")
    pct_diff(home_feats.get("goals_per_shot",        0),
             away_feats.get("goals_per_shot",        0), "shot conversion rate")

    pct_diff(home_feats.get("gk_save_pct",        0),
             away_feats.get("gk_save_pct",        0), "GK save%")
    pct_diff(home_feats.get("gk_clean_sheets_pct",0),
             away_feats.get("gk_clean_sheets_pct",0), "clean sheet rate")

    hf = home_feats.get("form_score", 0.5)
    af = away_feats.get("form_score", 0.5)
    if abs(hf - af) > 0.1:
        better = home_name if hf > af else away_name
        hpts = round(hf * 15)
        apts = round(af * 15)
        factors.append(f"{better} better recent form: {hpts}/15 vs {apts}/15 pts (last 5)")

    hp = home_feats.get("prev_rank_norm", 0.5)
    ap = away_feats.get("prev_rank_norm", 0.5)
    if abs(hp - ap) > 0.15:
        better = home_name if hp > ap else away_name
        factors.append(f"{better} finished significantly higher last season")

    hga = home_feats.get("goals_against_pg", 0)
    aga = away_feats.get("goals_against_pg", 0)
    if aga > hga + 0.3:
        factors.append(f"{away_name} concede {round(aga, 1)} goals/game — vulnerable defence")
    elif hga > aga + 0.3:
        factors.append(f"{home_name} concede {round(hga, 1)} goals/game — defensive concern")

    if home_feats.get("attack_dependency") == 1.0:
        factors.append(f"{home_name} attack heavily reliant on one player (>40% of goals)")
    if away_feats.get("attack_dependency") == 1.0:
        factors.append(f"{away_name} attack heavily reliant on one player (>40% of goals)")

    h_blank = home_feats.get("blank_rate", 0)
    a_blank = away_feats.get("blank_rate", 0)
    if h_blank > 0.35: factors.append(f"{home_name} struggles to score (blanked in {int(h_blank*100)}% of games)")
    if a_blank > 0.35: factors.append(f"{away_name} struggles to score (blanked in {int(a_blank*100)}% of games)")

    h_blowout = home_feats.get("blowout_rate", 0)
    a_blowout = away_feats.get("blowout_rate", 0)
    if h_blowout > 0.25: factors.append(f"{home_name} has high explosive potential (3+ goals in {int(h_blowout*100)}% of games)")
    if a_blowout > 0.25: factors.append(f"{away_name} has high explosive potential (3+ goals in {int(a_blowout*100)}% of games)")

    h_collapse = home_feats.get("defensive_collapse_rate", 0)
    a_collapse = away_feats.get("defensive_collapse_rate", 0)
    if h_collapse > 0.25: factors.append(f"{home_name} prone to defensive collapse (conceded 3+ in {int(h_collapse*100)}% of games)")
    if a_collapse > 0.25: factors.append(f"{away_name} prone to defensive collapse (conceded 3+ in {int(a_collapse*100)}% of games)")

    pct_diff(home_feats.get("squad_depth_scorers", 0),
             away_feats.get("squad_depth_scorers", 0), "squad scoring depth")

    hp_pos = home_feats.get("possession", 50)
    ap_pos = away_feats.get("possession", 50)
    if abs(hp_pos - ap_pos) > 5:
        dom = home_name if hp_pos > ap_pos else away_name
        factors.append(f"{dom} dominates possession ({round(max(hp_pos,ap_pos),1)}% avg)")

    return factors[:6]


def predict_match(home_team_id: int, away_team_id: int,
                  league_id: int, season_id: int) -> dict:
    """
    Full rich prediction for one match. Returns comprehensive output dict with:
    - probabilities (home_win, draw, away_win)
    - predicted_outcome + confidence + confidence_score
    - expected_goals (home_xg, away_xg, predicted_score)
    - key_factors (human-readable list of deciding factors)
    - team_comparison (attack, defence, form, possession, etc.)
    - h2h head-to-head history
    - top_features from XGBoost feature importances
    - model_info (n_trained_on, cv_accuracy, trained_at)
    """
    engine = _get_engine()
    if engine is None or not engine.is_trained:
        return {"error": "Model not trained. POST /api/predictions/train first."}

    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute("SELECT id, name FROM teams WHERE id IN (%s, %s)",
                    (home_team_id, away_team_id))
        name_map = {r["id"]: r["name"] for r in cur.fetchall()}
        home_name = name_map.get(home_team_id, f"Team {home_team_id}")
        away_name = name_map.get(away_team_id, f"Team {away_team_id}")

        cur.execute("SELECT name FROM leagues WHERE id = %s", (league_id,))
        lg_row = cur.fetchone()
        league_name = lg_row["name"] if lg_row else ""

        cur.execute("SELECT name FROM seasons WHERE id = %s", (season_id,))
        ss_row = cur.fetchone()
        season_name = ss_row["name"] if ss_row else ""

        fv, feat_names, home_feats, away_feats, h2h = build_match_features(
            cur, home_team_id, away_team_id, league_id, season_id
        )

        proba = engine.predict_proba(fv)
        top_feats = engine.get_top_features(fv, n=6)

        home_xg = _compute_venue_xg(home_feats, away_feats, "home")
        away_xg = _compute_venue_xg(away_feats, home_feats, "away")

        factors = _key_factors(home_feats, away_feats, home_name, away_name)

        def cmp(feats, venue_key):
            return {
                "attack":    round(feats.get("attack_strength",   1.0), 3),
                "defence":   round(feats.get("defence_strength",  1.0), 3),
                "form":      round(feats.get(f"{venue_key}_form_score",
                                             feats.get("form_score", 0.5)), 3),
                "possession":round(feats.get("possession", 50.0), 1),
                "goals_pg":  round(feats.get("goals_for_pg", 0.0), 2),
                "concede_pg":round(feats.get("goals_against_pg", 0.0), 2),
                "gk_save_pct":round(feats.get("gk_save_pct", 0.0), 1),
                "shots_ot_pg":round(feats.get("shots_on_target_per90", 0.0), 2),
                "top_scorer_goals": int(feats.get("top_scorer_goals", 0)),
                "prev_rank_norm":   round(feats.get("prev_rank_norm", 0.5), 3),
                "prev_form":        round(feats.get("prev_form_score", 0.5), 3),
                "blank_rate":       round(feats.get("blank_rate", 0.0), 3),
                "clean_sheet_rate": round(feats.get("clean_sheet_rate", 0.0), 3),
            }

        return {
            "match": {
                "home_team":  home_name,
                "away_team":  away_name,
                "league":     league_name,
                "season":     season_name,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
            },
            "probabilities": {
                "home_win": proba["home_win"],
                "draw":     proba["draw"],
                "away_win": proba["away_win"],
            },
            "predicted_outcome":  proba["predicted_outcome"],
            "confidence":         proba["confidence"],
            "confidence_score":   proba["confidence_score"],
            "expected_goals": {
                "home_xg": home_xg,
                "away_xg": away_xg,
                "predicted_score": f"{round(home_xg)}-{round(away_xg)}",
            },
            "key_factors":    factors,
            "team_comparison": {
                "home": cmp(home_feats, "home"),
                "away": cmp(away_feats, "away"),
            },
            "h2h": {
                "home_wins":  h2h["h2h_home_wins"],
                "draws":      h2h["h2h_draws"],
                "away_wins":  h2h["h2h_away_wins"],
                "home_win_pct": round(h2h["h2h_home_win_pct"], 3),
                "away_win_pct": round(h2h["h2h_away_win_pct"], 3),
                "last_5":     h2h["h2h_last_5"],
            },
            "top_features": top_feats,
            "model_info": {
                "n_trained_on":   _meta.get("n_samples", 0),
                "cv_accuracy":    _meta.get("cv_accuracy"),
                "trained_at":     _meta.get("trained_at"),
            },
        }
    finally:
        conn.close()


# ─── Upcoming fixtures ────────────────────────────────────────────────────────

def predict_upcoming(league_id: int = None, limit: int = 50) -> list:
    """
    Predict all upcoming (unplayed) fixtures, optionally filtered by league.
    Returns list of rich prediction dicts, each containing all fields from
    predict_match() plus fixture_id, match_date, and gameweek.
    Returns empty list if model is not trained.
    """
    engine = _get_engine()
    if engine is None or not engine.is_trained:
        return []

    conn = get_connection()
    cur  = conn.cursor()
    try:
        query = """
            SELECT m.id, m.home_team_id, m.away_team_id,
                   m.league_id, m.season_id, m.match_date, m.gameweek
            FROM matches m
            WHERE m.home_score IS NULL
              AND m.match_date >= CURRENT_DATE
        """
        params = []
        if league_id:
            query += " AND m.league_id = %s"
            params.append(league_id)
        query += " ORDER BY m.match_date ASC LIMIT %s"
        params.append(limit)

        cur.execute(query, params)
        fixtures = cur.fetchall()
        conn.close()

        results = []
        for fx in fixtures:
            try:
                pred = predict_match(
                    fx["home_team_id"],
                    fx["away_team_id"],
                    fx["league_id"],
                    fx["season_id"],
                )
                pred["fixture_id"]   = fx["id"]
                pred["match_date"]   = str(fx["match_date"]) if fx["match_date"] else None
                pred["gameweek"]     = fx["gameweek"]
                results.append(pred)
            except Exception:
                continue
        return results
    except Exception:
        conn.close()
        return []


# ─── Fast upcoming predictions (single DB round-trip via DataCache) ───────────

def predict_upcoming_fast(league_id: int = None, limit: int = 50) -> list:
    """
    Bulk predict all upcoming fixtures using DataCache — one DB connection,
    all feature computations in memory.  ~10-30x faster than predict_upcoming()
    on large fixture lists.  Falls back to predict_upcoming() on any error.

    Returns the same schema as predict_upcoming().
    """
    eng = _get_engine()
    if eng is None or not eng.is_trained:
        return []

    conn = get_connection()
    cur  = conn.cursor()
    try:
        # 1. Fetch upcoming fixtures
        query = """
            SELECT m.id, m.home_team_id, m.away_team_id,
                   m.league_id, m.season_id, m.match_date, m.gameweek,
                   ht.name AS home_name, at.name AS away_name,
                   l.name  AS league_name, s.name AS season_name
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
            query += " AND m.league_id = %s"
            params.append(league_id)
        query += " ORDER BY m.match_date ASC LIMIT %s"
        params.append(limit)
        cur.execute(query, params)
        fixtures = [dict(r) for r in cur.fetchall()]

        if not fixtures:
            conn.close()
            return []

        # 2. Bulk-load all features into DataCache (one DB round-trip)
        cache = DataCache(cur)
        conn.close()

        # 3. Build feature vectors and predict for each fixture
        results = []
        for fx in fixtures:
            try:
                htid = fx["home_team_id"]
                atid = fx["away_team_id"]
                lid  = fx["league_id"]
                sid  = fx["season_id"]

                home_feats = _build_team_features(cache, htid, lid, sid)
                away_feats = _build_team_features(cache, atid, lid, sid)

                # Build diff feature vector (same as training)
                fv = eng.feature_names_ or []
                row = {}
                for feat in fv:
                    if feat.startswith("home_"):
                        k = feat[5:]
                        row[feat] = home_feats.get(k, home_feats.get(feat, 0.0))
                    elif feat.startswith("away_"):
                        k = feat[5:]
                        row[feat] = away_feats.get(k, away_feats.get(feat, 0.0))
                    elif feat.endswith("_diff"):
                        k = feat[:-5]
                        row[feat] = home_feats.get(k, 0.0) - away_feats.get(k, 0.0)
                    else:
                        row[feat] = home_feats.get(feat, 0.0)

                import numpy as np
                X = np.array([[row.get(f, 0.0) for f in fv]])
                proba = eng.predict_proba(X[0].tolist() if hasattr(X[0], 'tolist') else list(X[0]))

                home_xg = _compute_venue_xg(home_feats, away_feats, "home")
                away_xg = _compute_venue_xg(away_feats, home_feats, "away")
                factors = _key_factors(home_feats, away_feats,
                                       fx["home_name"], fx["away_name"])

                results.append({
                    "fixture_id":   fx["id"],
                    "match_date":   str(fx["match_date"]) if fx["match_date"] else None,
                    "gameweek":     fx["gameweek"],
                    "match": {
                        "home_team":    fx["home_name"],
                        "away_team":    fx["away_name"],
                        "league":       fx["league_name"],
                        "season":       fx["season_name"],
                        "home_team_id": htid,
                        "away_team_id": atid,
                    },
                    "probabilities": {
                        "home_win": proba["home_win"],
                        "draw":     proba["draw"],
                        "away_win": proba["away_win"],
                    },
                    "predicted_outcome":  proba["predicted_outcome"],
                    "confidence":         proba["confidence"],
                    "confidence_score":   proba["confidence_score"],
                    "expected_goals": {
                        "home_xg":        home_xg,
                        "away_xg":        away_xg,
                        "predicted_score": f"{round(home_xg)}-{round(away_xg)}",
                    },
                    "key_factors": factors,
                    "model_info": {
                        "n_trained_on": _meta.get("n_samples", 0),
                        "cv_accuracy":  _meta.get("cv_accuracy"),
                        "trained_at":   _meta.get("trained_at"),
                    },
                })
            except Exception:
                continue

        return results

    except Exception:
        try: conn.close()
        except Exception: pass
        # Graceful fallback to the original (slower) method
        return predict_upcoming(league_id=league_id, limit=limit)


# ─── Status ───────────────────────────────────────────────────────────────────

def get_status() -> dict:
    """
    Return current model status dict:
    - model_trained (bool)
    - trained_at (ISO timestamp or None)
    - n_samples (int: number of matches trained on)
    - train_accuracy (float or None)
    - cv_accuracy (float or None)
    - n_features (int)
    - feature_names (first 20, for inspection)
    """
    engine = _get_engine()
    return {
        "model_trained":   engine is not None and engine.is_trained,
        "trained_at":      _meta.get("trained_at"),
        "n_samples":       _meta.get("n_samples", 0),
        "train_accuracy":  _meta.get("train_accuracy"),
        "cv_accuracy":     _meta.get("cv_accuracy"),
        "n_features":      len(_meta.get("feature_names", [])),
        "feature_names":   _meta.get("feature_names", [])[:20],
    }


# Auto-load on import
_get_engine()
