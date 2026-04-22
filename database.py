import os
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from dotenv import load_dotenv
# At the top of database.py, after existing imports:
import numpy as np
import psycopg2.extensions

psycopg2.extensions.register_adapter(np.float64, lambda x: psycopg2.extensions.AsIs(float(x)))
psycopg2.extensions.register_adapter(np.float32, lambda x: psycopg2.extensions.AsIs(float(x)))
psycopg2.extensions.register_adapter(np.int64,   lambda x: psycopg2.extensions.AsIs(int(x)))
psycopg2.extensions.register_adapter(np.int32,   lambda x: psycopg2.extensions.AsIs(int(x)))
psycopg2.extensions.register_adapter(np.bool_,   lambda x: psycopg2.extensions.AsIs(bool(x)))
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
        p = urlparse(DATABASE_URL)
        conn = psycopg2.connect(
            host=p.hostname,
            port=p.port or 5432,
            dbname=p.path.lstrip("/"),
            user=p.username,
            password=p.password,
            sslmode="require",
            connect_timeout=10,
            # Prevent any single statement from holding the connection open
            # indefinitely. 60 s is generous for even large upsert batches;
            # runaway queries (deadlock retry loops, lock waits) are killed
            # before Chrome's 6-connection-per-host stall window is reached.
            options="-c statement_timeout=60000",
            cursor_factory=RealDictCursor,
        )
        return conn


def get_db():
        conn = get_connection()
        try:
                    yield conn
        finally:
                conn.close()
