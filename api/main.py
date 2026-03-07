import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv


from routes import leagues, teams, matches, standings, squad_stats, player_stats, sync, health, auth, cleanup, predictions, venue_stats


load_dotenv()


# ─── HTTPS redirect middleware ─────────────────────────────────────────────────
class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        proto = request.headers.get("x-forwarded-proto", "")
        host  = request.headers.get("host", "")
        is_local = host.startswith("localhost") or host.startswith("127.0.0.1")
        if proto == "http" and not is_local:
            https_url = str(request.url).replace("http://", "https://", 1)
            return RedirectResponse(url=https_url, status_code=301)
        return await call_next(request)


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Football Analytics API",
    description="Production API for football data scraped from FBref",
    version="1.0.0",
    redirect_slashes=False,
)

# Apply HTTPS redirect before CORS so redirects carry the right headers
app.add_middleware(HTTPSRedirectMiddleware)


# ─── CORS ─────────────────────────────────────────────────────────────────────
# Explicit origins (exact match) — add every known Vercel deployment URL here.
# For preview deployments that change per-commit, the regex below handles them.

CORS_ORIGINS = [
    # ── Local dev ──────────────────────────────────────────────────────────
    "http://localhost:3000",
    "http://localhost:4000",
    "http://localhost:5173",
    "http://localhost:5174",
    "https://localhost:3000",
    "https://localhost:5173",
    # ── Production / stable Vercel deployments ─────────────────────────────
    "https://football-analytics-eight.vercel.app",
    "https://plusone-frontend-mu.vercel.app",
    # ── Current deployment (add new ones here as needed) ───────────────────
    "https://plusone-frontend-7nnahjruz-atugibras-projects.vercel.app",
    "https://plusone-frontend-git-main-atugibras-projects.vercel.app",
]

# Regex catches everything else:
#   - localhost on any port (http or https)          — dev
#   - *.vercel.app (all preview + production URLs)   — Vercel
#   - *.onrender.com                                 — Render
#   - *.railway.app / *.up.railway.app               — Railway
#
# NOTE: Starlette uses re.fullmatch() so the pattern must match the ENTIRE
# origin string (no need for ^ / $ anchors).
CORS_ORIGIN_REGEX = (
    r"https?://localhost(:\d+)?"
    r"|https://[\w-]+(\.[\w-]+)*\.vercel\.app"
    r"|https://[\w-]+\.onrender\.com"
    r"|https://[\w-]+(\.up)?\.railway\.app"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Content-Length"],
    max_age=600,
)


# ─── Routes ───────────────────────────────────────────────────────────────────

app.include_router(health.router,       prefix="/api",             tags=["Health"])
app.include_router(leagues.router,      prefix="/api/leagues",     tags=["Leagues"])
app.include_router(teams.router,        prefix="/api/teams",       tags=["Teams"])
app.include_router(matches.router,      prefix="/api/matches",     tags=["Matches"])
app.include_router(standings.router,    prefix="/api/standings",   tags=["Standings"])
app.include_router(squad_stats.router,  prefix="/api/squad-stats", tags=["Squad Stats"])
app.include_router(venue_stats.router,  prefix="/api/venue-stats", tags=["Venue Stats"])
app.include_router(player_stats.router, prefix="/api/players",     tags=["Players"])
app.include_router(sync.router,         prefix="/api/sync",        tags=["Sync"])
app.include_router(cleanup.router,      prefix="/api/cleanup",     tags=["Cleanup"])
app.include_router(auth.router,                                    tags=["Auth"])
app.include_router(predictions.router,  prefix="/api/predictions", tags=["Predictions"])


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 4000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
