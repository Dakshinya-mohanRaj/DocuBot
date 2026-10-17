"""Password hashing, opaque session tokens, and cookie configuration.

Passwords are hashed with **Argon2id** (the argon2-cffi default) — a memory-hard
KDF that is resistant to GPU cracking. Nothing here ever logs or returns a
password or a raw session token except to the browser at the moment of login.
"""

import hashlib
import os
import secrets

from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

# Argon2id with argon2-cffi's vetted defaults (time_cost, memory_cost, parallelism).
# Explicit ``type=Type.ID`` pins Argon2id rather than relying on a library default.
_hasher = PasswordHasher(type=Type.ID)

# Cookie / header names shared by the backend and the frontend.
SESSION_COOKIE = "docubot_session"
CSRF_COOKIE = "docubot_csrf"
CSRF_HEADER = "x-csrf-token"

# Sessions are opaque server-side records; expire them after this many days.
SESSION_TTL_DAYS = int(os.environ.get("DOCUBOT_SESSION_TTL_DAYS", "14"))

# Cookies are marked Secure by default (correct behind Render's TLS). Local
# http development and the test suite override this via DOCUBOT_COOKIE_SECURE.
COOKIE_SECURE = os.environ.get("DOCUBOT_COOKIE_SECURE", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

MIN_PASSWORD_LENGTH = int(os.environ.get("DOCUBOT_MIN_PASSWORD_LENGTH", "8"))
MAX_PASSWORD_LENGTH = 1024  # bound Argon2 work on absurd inputs


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Constant-time-ish verification; returns False for any malformed hash."""
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def new_token() -> str:
    """A URL-safe, 256-bit opaque token (session or CSRF)."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    """SHA-256 hex digest used as the stored key for a session/CSRF token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    return secrets.compare_digest(a, b)
