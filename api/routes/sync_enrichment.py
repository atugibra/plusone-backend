from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from database import get_connection
from routes.deps import require_admin
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

class SyncEnrichmentPayload(BaseModel):
    league: str
    odds: Optional[List[dict]] = []
    injuries: Optional[List[dict]] = []
    clubelo: Optional[List[dict]] = []

def safe_num(val):
    if val is None or val == "": return None
    try: return float(val)
    except: return None


# ClubElo uses 2-4 char country/league codes; map each to the canonical full name
# used by FBref so we don't create duplicate league rows.
CLUBELO_LEAGUE_MAP = {
    # Top divisions
    "Aut": "Austrian Bundesliga",
    "Bel": "Belgian Pro League",
    "Bul": "Bulgarian First League",
    "Cro": "Croatian Football League",
    "Cze": "Czech First League",
    "Den": "Danish Superliga",
    "Eng": "Premier League",
    "Fra": "Ligue 1",
    "Ger": "Bundesliga",
    "Esp": "La Liga",
    "Gre": "Greek Super League",   # aligned with FBref canonical name
    "Hun": "Hungarian OTP Bank Liga",
    "Ita": "Serie A",
    "Ned": "Eredivisie",
    "Por": "Primeira Liga",
    "Rus": "Russian Premier League",
    "Sco": "Scottish Premiership",
    "Srb": "Serbian SuperLiga",
    "Sui": "Swiss Super League",
    "Tur": "Super Lig",             # aligned with FBref canonical name (was "Über Lig")
    "Ukr": "Ukrainian Premier League",
    # Second divisions
    "Eng2": "Championship",
    "Ger2": "2. Bundesliga",
    "Esp2": "La Liga 2",
    "Ita2": "Serie B",
    "Fra2": "Ligue 2",
    "Ned2": "Eerste Divisie",
    "Por2": "Liga Portugal 2",
    "Sco2": "Scottish Championship",
    # European / continental
    "UCL":  "UEFA Champions League",
    "UEL":  "UEFA Europa League",
    "UECL": "UEFA Europa Conference League",
}

# Football-Data.co.uk uses Div codes in their CSVs (E0, D1, SP1, etc.)
# Map these to canonical league names to prevent ghost league rows.
FOOTBALL_DATA_DIV_MAP = {
    # England
    "E0":  "Premier League",
    "E1":  "Championship",
    "E2":  "League One",
    "E3":  "League Two",
    "EC":  "National League",          # Non-League / Conference
    # Germany
    "D1":  "Bundesliga",
    "D2":  "2. Bundesliga",
    # Spain
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    # Italy
    "I1":  "Serie A",
    "I2":  "Serie B",
    # France
    "F1":  "Ligue 1",
    "F2":  "Ligue 2",
    # Netherlands
    "N1":  "Eredivisie",
    # Belgium
    "B1":  "Belgian Pro League",
    # Turkey
    "T1":  "Super Lig",
    # Scotland
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    # Greece
    "G1":  "Greek Super League",   # aligned with FBref canonical name
    # Portugal
    "P1":  "Primeira Liga",
    "P2":  "Liga Portugal 2",
    # Austria
    "A1":  "Austrian Bundesliga",
    # USA
    "USA": "MLS",
    # Brazil
    "BSA": "Brasileirao Serie A",
    # Argentina
    "ARG": "Argentine Primera Division",
    # Japan
    "J1":  "J1 League",
    # Catch-all codes seen in fixtures.csv
    "CONF": "UEFA Europa Conference League",
    "UCL":  "UEFA Champions League",
    "UEL":  "UEFA Europa League",
}

# ─── Team name aliases ────────────────────────────────────────────────────────
# Maps Football-Data.co.uk, ClubElo, and Transfermarkt names → FBref canonical
# names.  Add entries whenever a new source uses a different spelling.
TEAM_NAME_ALIASES: dict[str, str] = {
    # ── England ───────────────────────────────────────────────────────────────
    "man city":                   "Manchester City",
    "manchester city":            "Manchester City",
    "man utd":                    "Manchester United",
    "man united":                 "Manchester United",
    "manchester united":          "Manchester United",
    "nott'm forest":              "Nottingham Forest",
    "nottm forest":               "Nottingham Forest",
    "nottingham forest":          "Nottingham Forest",
    "wolverhampton wanderers":    "Wolverhampton Wanderers",
    "wolverhampton":              "Wolverhampton Wanderers",
    "sheffield utd":              "Sheffield United",
    "sheffield united":           "Sheffield United",
    "luton town":                 "Luton",
    "west bromwich albion":       "West Brom",
    "west brom":                  "West Brom",
    "queens park rangers":        "QPR",
    "brighton & hove albion":     "Brighton",
    "brighton and hove albion":   "Brighton",
    "huddersfield town":          "Huddersfield",
    "stoke city":                 "Stoke",
    "swansea city":               "Swansea",
    "cardiff city":               "Cardiff",
    "wigan athletic":             "Wigan",
    "blackburn rovers":           "Blackburn",
    "bolton wanderers":           "Bolton",
    "ipswich town":               "Ipswich",
    "oxford united":              "Oxford Utd",
    "bristol city":               "Bristol City",
    "birmingham city":            "Birmingham City",
    "leicester city":             "Leicester City",
    "newcastle united":           "Newcastle United",
    "norwich city":               "Norwich",
    "watford fc":                 "Watford",
    "crystal palace":             "Crystal Palace",
    "west ham united":            "West Ham United",
    "west ham":                   "West Ham United",
    "tottenham hotspur":          "Tottenham Hotspur",
    "tottenham":                  "Tottenham Hotspur",
    "aston villa":                "Aston Villa",
    # ── Germany ───────────────────────────────────────────────────────────────
    "eintr frankfurt":            "Eintracht Frankfurt",
    "eintracht frankfurt":        "Eintracht Frankfurt",
    "hertha berlin":              "Hertha BSC",
    "hertha bsc":                 "Hertha BSC",
    "m'gladbach":                 "Mönchengladbach",
    "b. monchengladbach":         "Mönchengladbach",
    "borussia m.gladbach":        "Mönchengladbach",
    "bayer leverkusen":           "Leverkusen",
    "rb leipzig":                 "RB Leipzig",
    "vfb stuttgart":               "Stuttgart",
    "sc freiburg":                 "Freiburg",
    "fc augsburg":                 "Augsburg",
    "vfl bochum":                  "Bochum",
    "fc koln":                     "Köln",
    "1. fc koln":                  "Köln",
    "fc heidenheim":               "Heidenheim",
    "sv darmstadt 98":             "Darmstadt 98",
    "vfl wolfsburg":               "Wolfsburg",
    "borussia dortmund":           "Dortmund",
    "fc bayern munich":            "Bayern Munich",
    "bayern munich":               "Bayern Munich",
    "tsg hoffenheim":              "Hoffenheim",
    "1899 hoffenheim":             "Hoffenheim",
    "mainz 05":                    "Mainz 05",
    "1. fsv mainz 05":             "Mainz 05",
    "vfb Stuttgart":               "Stuttgart",
    # ── Spain ─────────────────────────────────────────────────────────────────
    "athletic bilbao":            "Athletic Club",
    "atletico madrid":            "Atlético Madrid",
    "real madrid":                "Real Madrid",
    "barcelona":                  "Barcelona",
    "espanol":                    "Espanyol",
    "deportivo alaves":           "Alavés",
    "alaves":                     "Alavés",
    "real sociedad":              "Real Sociedad",
    "real betis":                 "Betis",
    "betis":                      "Betis",
    "rayo vallecano":             "Rayo Vallecano",
    "villarreal":                 "Villarreal",
    "valencia cf":                "Valencia",
    "girona fc":                  "Girona",
    "celta vigo":                 "Celta Vigo",
    "cadiz cf":                   "Cádiz",
    "cadiz":                      "Cádiz",
    "getafe cf":                  "Getafe",
    "las palmas":                 "Las Palmas",
    # ── France ────────────────────────────────────────────────────────────────
    "psg":                        "Paris Saint-Germain",
    "paris sg":                   "Paris Saint-Germain",
    "paris saint-germain":        "Paris Saint-Germain",
    "paris saint germain":        "Paris Saint-Germain",
    "st etienne":                 "Saint-Étienne",
    "saint-etienne":              "Saint-Étienne",
    "olympique marseille":        "Marseille",
    "olympique lyonnais":         "Lyon",
    "as monaco":                  "Monaco",
    "stade rennes":               "Rennes",
    "stade brest":                "Brest",
    "rc lens":                    "Lens",
    "losc lille":                 "Lille",
    "ogc nice":                   "Nice",
    "rc strasbourg":              "Strasbourg",
    "toulouse fc":                "Toulouse",
    "montpellier hsc":            "Montpellier",
    "nantes":                     "Nantes",
    # ── Italy ─────────────────────────────────────────────────────────────────
    "milan":                      "Milan",
    "ac milan":                   "Milan",
    "inter milan":                "Inter",
    "internazionale":             "Inter",
    "fc internazionale":          "Inter",
    "hellas verona":              "Verona",
    "as roma":                    "Roma",
    "ss lazio":                   "Lazio",
    "ssc napoli":                 "Napoli",
    "atalanta bc":                "Atalanta",
    "juventus fc":                "Juventus",
    "acf fiorentina":             "Fiorentina",
    "torino fc":                  "Torino",
    "genoa cfc":                  "Genoa",
    "udinese calcio":             "Udinese",
    "cagliari calcio":            "Cagliari",
    "bologna fc":                 "Bologna",
    "monza":                      "Monza",
    "ac monza":                   "Monza",
    "empoli fc":                  "Empoli",
    "us lecce":                   "Lecce",
    "frosinone calcio":           "Frosinone",
    "salernitana":                "Salernitana",
    # ── Netherlands ───────────────────────────────────────────────────────────
    "psv eindhoven":              "PSV",
    "ajax amsterdam":             "Ajax",
    "afc ajax":                   "Ajax",
    "feyenoord rotterdam":        "Feyenoord",
    "az alkmaar":                 "AZ",
    "fc utrecht":                 "Utrecht",
    "fc twente":                  "Twente",
    "sc heerenveen":              "Heerenveen",
    # ── Turkey ────────────────────────────────────────────────────────────────
    "besiktas jk":                "Beşiktaş",
    "besiktas":                   "Beşiktaş",
    "galatasaray sk":             "Galatasaray",
    "fenerbahce sk":              "Fenerbahçe",
    "fenerbahce":                 "Fenerbahçe",
    "trabzonspor":                "Trabzonspor",
    # ── Belgium ───────────────────────────────────────────────────────────────
    "club brugge":                "Club Brugge",
    "kv mechelen":                "Mechelen",
    "rsc anderlecht":             "Anderlecht",
    # ── Portugal ──────────────────────────────────────────────────────────────
    "sl benfica":                 "Benfica",
    "fc porto":                   "Porto",
    "sporting cp":                "Sporting CP",
    "sc braga":                   "Braga",
    # ── Scotland ──────────────────────────────────────────────────────────────
    "celtic fc":                  "Celtic",
    "rangers fc":                 "Rangers",
    "heart of midlothian":        "Hearts",
    "hibernian fc":               "Hibernian",
    # ── FBref abbreviation reverse aliases ───────────────────────────────────
    # These handle FBref's own abbreviations arriving from fixtures/stats pages
    "nott'ham forest":            "Nottingham Forest",
    "newcastle utd":              "Newcastle United",
    "sheffield utd":              "Sheffield United",
    "sheffield wed":              "Sheffield Wednesday",
    "manchester utd":             "Manchester United",
    "leeds utd":                  "Leeds United",
    "wolves":                     "Wolverhampton Wanderers",
    "oxford utd":                 "Oxford United",
    "sunderland afc":             "Sunderland",
    "norwich city":               "Norwich City",
    "ipswich town":               "Ipswich Town",
    "luton town":                 "Luton Town",
    "swansea city":               "Swansea City",
    "cardiff city":               "Cardiff City",
    "stoke city":                 "Stoke City",
    "huddersfield town":          "Huddersfield Town",
}


def _normalize_team_name(raw: str) -> str:
    """Map external data source names → FBref canonical names.
    Falls through unchanged if no alias is found."""
    return TEAM_NAME_ALIASES.get(raw.strip().lower(), raw.strip())


def _get_league(cur, name: str):
    """Resolve a league name (including ClubElo short codes and Football-Data Div
    codes) to a leagues.id.  Short codes are mapped to canonical full names to
    avoid duplicate league rows.
    """
    raw = name.strip()

    # Resolve ClubElo short code to its full canonical name (case-insensitive)
    canonical = (
        CLUBELO_LEAGUE_MAP.get(raw)
        or CLUBELO_LEAGUE_MAP.get(raw.capitalize())
        or CLUBELO_LEAGUE_MAP.get(raw.upper())
        # Resolve Football-Data.co.uk Div codes (E0, D1, SP1 …)
        or FOOTBALL_DATA_DIV_MAP.get(raw)
        or FOOTBALL_DATA_DIV_MAP.get(raw.upper())
        or raw
    )

    # 1. Exact match (case-insensitive)
    cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (canonical,))
    res = cur.fetchone()
    if res:
        return res["id"]

    # 2. Partial keyword match: helps when FBref used a slightly different spelling
    #    e.g. canonical="Austrian Bundesliga", DB has "Austrian Football Bundesliga"
    keyword = canonical.split()[0]  # use the most distinctive first word
    if len(keyword) > 3:  # skip very short words like 'La', 'De'
        cur.execute("SELECT id FROM leagues WHERE name ILIKE %s LIMIT 1", (f"%{keyword}%",))
        res = cur.fetchone()
        if res:
            return res["id"]

    # 3. Insert with canonical full name (never insert a bare short code)
    cur.execute("INSERT INTO leagues (name) VALUES (%s) RETURNING id", (canonical,))
    return cur.fetchone()["id"]


def _get_team(cur, name: str, league_id: int, team_cache: dict):
    raw   = name.strip()
    clean = _normalize_team_name(raw)   # map aliases → FBref canonical names

    # In-memory deduplication during massive sync batches
    cache_key = (clean.lower(), league_id)
    if cache_key in team_cache:
        return team_cache[cache_key]

    # 1. Exact match within league
    cur.execute("SELECT id FROM teams WHERE name ILIKE %s AND league_id = %s LIMIT 1", (clean, league_id))
    res = cur.fetchone()
    if res:
        team_cache[cache_key] = res["id"]
        return res["id"]

    # 2. Cross-league fallback: team may be registered under a domestic league
    #    but also appear in enrichment data for UCL / Europa etc.
    cur.execute("SELECT id FROM teams WHERE name ILIKE %s LIMIT 1", (clean,))
    res = cur.fetchone()
    if res:
        team_cache[cache_key] = res["id"]
        return res["id"]

    # 3. Genuinely new team — only create if name looks legitimate.
    #    Names shorter than 3 chars are almost certainly garbage from a
    #    malformed CSV row (e.g. empty HomeTeam field parsed as "  ").
    if len(clean) < 3:
        logger.warning("Skipping suspect team name (too short): %r", raw)
        return None

    # 4. Fuzzy match against existing teams in the league to prevent duplicates
    import difflib
    cur.execute("SELECT id, name FROM teams WHERE league_id = %s", (league_id,))
    league_teams = cur.fetchall()
    team_names = [t["name"] for t in league_teams]
    
    matches = difflib.get_close_matches(clean, team_names, n=1, cutoff=0.85)  # raised from 0.60 — reduces false-positive team merges
    if matches:
        matched_name = matches[0]
        # Find the ID for the matched name
        for t in league_teams:
            if t["name"] == matched_name:
                team_cache[cache_key] = t["id"]
                logger.debug("Fuzzy matched enrichment team %r to existing team %r", clean, matched_name)
                return t["id"]

    cur.execute("INSERT INTO teams (name, league_id) VALUES (%s, %s) RETURNING id", (clean, league_id))
    new_id = cur.fetchone()["id"]
    team_cache[cache_key] = new_id
    logger.debug("Created new team: %r (league_id=%s)", clean, league_id)
    return new_id

def _find_match(cur, home_team_id, away_team_id, match_date: str):
    if not match_date:
        # Fallback if no date is found
        cur.execute("""
            SELECT id FROM matches 
            WHERE home_team_id = %s AND away_team_id = %s 
            ORDER BY match_date DESC LIMIT 1
        """, (home_team_id, away_team_id))
    else:
        # Odds files/FBref dates can differ by 1-2 days due to timezone disparities.
        # We search within a generous 3-day window of the declared match date to perfectly align historical matches.
        cur.execute("""
            SELECT id FROM matches 
            WHERE home_team_id = %s AND away_team_id = %s 
              AND match_date >= %s::date - INTERVAL '3 days' 
              AND match_date <= %s::date + INTERVAL '3 days'
            LIMIT 1
        """, (home_team_id, away_team_id, match_date, match_date))
        
    res = cur.fetchone()
    return res["id"] if res else None

def _parse_odds_date(d_str):
    if not d_str: return None
    try:
        if "/" in d_str:
            parts = d_str.split("/")
            if len(parts[2]) == 2: parts[2] = "20" + parts[2]
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return d_str
    except:
        return None

@router.post("/enrichment")
def sync_enrichment(payload: SyncEnrichmentPayload, _admin: dict = Depends(require_admin)):
    conn = get_connection()
    cur = conn.cursor()
    try:
        default_league_id = _get_league(cur, payload.league)
        team_cache = {}
        
        # Helper to determine the correct league dynamically per row.
        # This prevents team-corruption when batch syncing multi-league CSV files (like fixtures.csv)
        def _resolve_league(row_obj):
            row_league = row_obj.get("League") or row_obj.get("league")
            if row_league and row_league != "Unknown League" and row_league != "Global Enrichment Sync":
                return _get_league(cur, row_league)
            return default_league_id
            
        odds_count = 0
        injuries_count = 0
        clubelo_count = 0
        
        # ── 1. Match Odds ───────────────────────────────────────────────────────────
        if payload.odds:
            for row in payload.odds:
                home_name = row.get("HomeTeam") or row.get("Home")
                away_name = row.get("AwayTeam") or row.get("Away")
                if not home_name or not away_name: continue
                
                l_id = _resolve_league(row)
                h_id = _get_team(cur, home_name, l_id, team_cache)
                a_id = _get_team(cur, away_name, l_id, team_cache)
                m_date = _parse_odds_date(row.get("Date"))
                m_id = _find_match(cur, h_id, a_id, m_date)
                
                if m_id:
                    h_odds = safe_num(row.get("B365H"))
                    d_odds = safe_num(row.get("B365D"))
                    a_odds = safe_num(row.get("B365A"))
                    cur.execute("""
                        INSERT INTO match_odds (match_id, b365_home_win, b365_draw, b365_away_win, raw_data)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (match_id) DO UPDATE SET
                            b365_home_win=EXCLUDED.b365_home_win,
                            b365_draw=EXCLUDED.b365_draw,
                            b365_away_win=EXCLUDED.b365_away_win,
                            raw_data=EXCLUDED.raw_data,
                            scraped_at=NOW()
                    """, (m_id, h_odds, d_odds, a_odds, json.dumps(row)))
                    odds_count += 1

        # ── 2. Player Injuries ──────────────────────────────────────────────────────
        if payload.injuries:
            for row in payload.injuries:
                club = row.get("Club")
                player = row.get("Player")
                if not club or not player: continue
                
                l_id = _resolve_league(row)
                t_id = _get_team(cur, club, l_id, team_cache)
                cur.execute("""
                    INSERT INTO player_injuries (team_id, player_name, injury_type, return_date, raw_data)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (team_id, player_name) DO UPDATE SET
                        injury_type=EXCLUDED.injury_type,
                        return_date=EXCLUDED.return_date,
                        raw_data=EXCLUDED.raw_data,
                        scraped_at=NOW()
                """, (t_id, player, row.get("Injury"), row.get("Return_Date"), json.dumps(row)))
                injuries_count += 1

        # ── 3. Team ClubElo ─────────────────────────────────────────────────────────
        if payload.clubelo:
            for row in payload.clubelo:
                raw_data = row.get("raw", row)
                home = raw_data.get("Home") or raw_data.get("HomeTeam") or row.get("home")
                away = raw_data.get("Away") or raw_data.get("AwayTeam") or row.get("away")
                target_date = raw_data.get("Date") or row.get("date")
                
                l_id = _resolve_league(row)
                
                # We may not have exact Elo numbers from the Fixtures endpoint, but we save the predicted Probs!
                for tm, elo_val in [(home, raw_data.get("Home_Elo")), (away, raw_data.get("Away_Elo"))]:
                    if not tm or not target_date: continue
                    t_id = _get_team(cur, tm, l_id, team_cache)
                    cur.execute("""
                        INSERT INTO team_clubelo (team_id, elo_date, elo, raw_data)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (team_id, elo_date) DO UPDATE SET
                            elo=EXCLUDED.elo,
                            raw_data=EXCLUDED.raw_data,
                            scraped_at=NOW()
                    """, (t_id, target_date, safe_num(elo_val), json.dumps(row)))
                    clubelo_count += 1
                
        conn.commit()
        return {
            "success": True, 
            "inserted": {
                "match_odds": odds_count,
                "player_injuries": injuries_count,
                "team_clubelo": clubelo_count
            }
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Enrichment Sync Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()
