import os
import logging
import psycopg2
import psycopg2.pool
import psycopg2.extensions
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from dotenv import load_dotenv

log = logging.getLogger(__name__)



_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)

DATABASE_URL = os.getenv("DATABASE_URL")

# ─── Connection Pool ─────────────────────────────────────────────────────────
# Supabase free tier allows ~25 simultaneous connections.
# minconn=1 keeps one warm connection; maxconn=10 caps pool usage so
# concurrent requests (auto-consensus, sync, training) don't exhaust the limit.

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _init_pool():
    global _pool
    if _pool is not None:
        return
    p = urlparse(DATABASE_URL)
    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=p.password,
        sslmode="require",
        connect_timeout=10,
        cursor_factory=RealDictCursor,
    )
    log.info("DB connection pool initialised (min=1, max=10).")


def get_connection():
    """Borrow a connection from the pool. Caller MUST call return_connection() when done."""
    global _pool
    if _pool is None:
        _init_pool()
    try:
        return _pool.getconn()
    except psycopg2.pool.PoolError:
        log.warning("DB pool exhausted — falling back to direct connection.")
        p = urlparse(DATABASE_URL)
        return psycopg2.connect(
            host=p.hostname, port=p.port or 5432,
            dbname=p.path.lstrip("/"), user=p.username, password=p.password,
            sslmode="require", connect_timeout=10, cursor_factory=RealDictCursor,
        )


def return_connection(conn):
    """Return a borrowed connection back to the pool."""
    global _pool
    if _pool is not None:
        try:
            _pool.putconn(conn)
        except Exception:
            try: conn.close()
            except: pass
    else:
        try: conn.close()
        except: pass


def get_db():
    """FastAPI dependency — borrows from pool, yields, then returns automatically."""
    conn = get_connection()
    try:
        yield conn
    finally:
        return_connection(conn)


# Initialise pool at import time (non-blocking)
try:
    _init_pool()
except Exception as exc:
    log.warning("DB pool init deferred (will retry on first request): %s", exc)
