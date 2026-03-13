import os
import psycopg2
from psycopg2.extras import RealDictCursor
from urllib.parse import urlparse
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path, override=True)

DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
        p = urlparse(DATABASE_URL)
        return psycopg2.connect(
            host=p.hostname,
            port=p.port or 5432,
            dbname=p.path.lstrip("/"),
            user=p.username,
            password=p.password,
            sslmode="require",
            connect_timeout=10,
            cursor_factory=RealDictCursor,
        )


def get_db():
        conn = get_connection()
        try:
                    yield conn
        finally:
                conn.close()
