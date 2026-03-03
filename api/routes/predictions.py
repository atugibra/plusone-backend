"""
Predictions API - Rule-based match outcome predictor.
Returns pre-match squad stats (BEFORE panel) + actual match result from DB (AFTER panel).
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from database import get_connection

router = APIRouter()


class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    league: Optional[str] = None


def _get_team_stats(cur, team_name: str):
    cur.execute("""
        SELECT ts.goals, ts.assists, ts.games, ts.possession, ts.avg_age
        FROM   team_squad_stats ts
        JOIN   teams t ON t.id = ts.team_id
        WHERE  LOWER(t.name) LIKE LOWER(%s) AND ts.split = 'for'
        ORDER  BY ts.scraped_at DESC LIMIT 1
    """, (f"%{team_name}%",))
    return cur.fetchone()


def _get_team_standings(cur, team_name: str):
    cur.execute("""
        SELECT ls.wins, ls.ties, ls.losses, ls.games,
               ls.goals_for, ls.goals_against, ls.points, ls.rank
        FROM   league_standings ls
        JOIN   teams t ON t.id = ls.team_id
        WHERE  LOWER(t.name) LIKE LOWER(%s)
        ORDER  BY ls.scraped_at DESC LIMIT 1
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


def _safe_div(a, b, default=0.0):
    try: return float(a) / float(b) if b else default
    except: return default


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
async def generate_prediction(req: PredictionRequest):
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
