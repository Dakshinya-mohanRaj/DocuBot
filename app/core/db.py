"""Database engine and session management.

The app stores accounts, auth sessions and conversations in a relational
database. Production uses Postgres (Render/Neon) via ``DOCUBOT_DATABASE_URL``;
when that is unset we fall back to a local SQLite file so development and the
test suite run with zero external services.

SQLAlchemy is used purely as a portable layer over the two backends — the ORM
models in :mod:`app.core.models` are the single source of truth for the schema.
"""

import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# SQLite connection string used when no database is configured. Kept inside the
# gitignored ./data directory (see .gitignore).
DEFAULT_SQLITE_URL = "sqlite:///./data/docubot.db"


def _database_url() -> str:
    url = os.environ.get("DOCUBOT_DATABASE_URL", "").strip()
    if not url:
        return DEFAULT_SQLITE_URL
    # Render (and Heroku) hand out ``postgres://`` URLs; SQLAlchemy needs an
    # explicit driver, and we use the psycopg3 driver.
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


_DATABASE_URL = _database_url()
_IS_SQLITE = _DATABASE_URL.startswith("sqlite")

if _IS_SQLITE:
    # Make sure the parent directory of the SQLite file exists before the first
    # connection opens it (otherwise SQLite fails to create the file).
    _path = _DATABASE_URL.split("///", 1)[-1]
    if _path and _path != ":memory:":
        Path(_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    _DATABASE_URL,
    # SQLite + TestClient's threadpool requires cross-thread access; a real
    # server (Postgres) should validate stale connections instead.
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    pool_pre_ping=not _IS_SQLITE,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """Create any missing tables. Safe to call repeatedly on startup."""
    from app.core import models  # noqa: F401  (register models on Base)

    Base.metadata.create_all(bind=engine)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
