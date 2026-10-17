"""Database schema and route wiring.

These guard the two failure modes that produced the production "Not Found":
the auth routers not being mounted, and the database tables not existing.
"""

from sqlalchemy import inspect

from app.core.db import engine, init_db
from app.core.models import AuthSession, Conversation, Message, User
from app.main import app

EXPECTED_TABLES = {"users", "auth_sessions", "conversations", "messages"}


def _route_pairs():
    pairs = set()
    for route in app.routes:
        for method in getattr(route, "methods", set()) or set():
            pairs.add((route.path, method))
    return pairs


def test_auth_routes_are_registered():
    """signup/login/logout/me must all be mounted under /auth."""
    routes = _route_pairs()
    assert ("/auth/signup", "POST") in routes
    assert ("/auth/login", "POST") in routes
    assert ("/auth/logout", "POST") in routes
    assert ("/auth/me", "GET") in routes


def test_no_duplicate_auth_paths():
    """Each auth path is defined exactly once (catches a double-included router)."""
    paths = [route.path for route in app.routes if route.path.startswith("/auth/")]
    assert len(paths) == len(set(paths)), f"duplicate auth routes: {paths}"


def test_init_db_creates_all_required_tables():
    init_db()  # idempotent: safe to call again on an existing database
    tables = set(inspect(engine).get_table_names())
    assert EXPECTED_TABLES.issubset(tables), f"missing tables: {EXPECTED_TABLES - tables}"


def test_model_tablenames_match_expected_schema():
    assert {
        User.__tablename__,
        AuthSession.__tablename__,
        Conversation.__tablename__,
        Message.__tablename__,
    } == EXPECTED_TABLES
