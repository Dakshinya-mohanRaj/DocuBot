"""Account, authentication-session, and CSRF persistence.

These functions take an explicit :class:`sqlalchemy.orm.Session` so they are
easy to unit-test and to use as request-scoped dependencies. They never hash,
compare, or expose passwords beyond delegating to :mod:`app.core.security`.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import security
from app.core.models import AuthSession, User, utcnow


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _as_utc(value: datetime) -> datetime:
    # SQLite does not preserve tzinfo; treat stored naive values as UTC so expiry
    # comparisons never blow up (Postgres keeps them aware).
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def get_user(db: Session, user_id: str) -> User | None:
    return db.get(User, user_id)


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == normalize_email(email)))


def create_user(
    db: Session, email: str, password: str, display_name: str | None = None
) -> User:
    """Create an account. Raises ``ValueError`` if the email is already taken."""
    user = User(
        email=normalize_email(email),
        display_name=(display_name or "").strip() or None,
        password_hash=security.hash_password(password),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError("An account with that email already exists.")
    db.refresh(user)
    return user


def authenticate(db: Session, email: str, password: str) -> User | None:
    """Return the user when the credentials are valid, else ``None``.

    The password is always verified against *some* hash — when the email is
    unknown we still run a verify against a dummy hash so response timing does
    not reveal whether an account exists.
    """
    user = get_user_by_email(db, email)
    if user is None:
        security.verify_password(_DUMMY_HASH, password)
        return None
    if not security.verify_password(user.password_hash, password):
        return None
    return user


def create_auth_session(db: Session, user: User) -> tuple[str, str]:
    """Issue a new server-side session. Returns ``(session_token, csrf_token)``.

    Only the SHA-256 hashes of both tokens are stored; the raw values go to the
    browser as cookies and are never logged.
    """
    token = security.new_token()
    csrf = security.new_token()
    now = utcnow()
    row = AuthSession(
        token_hash=security.hash_token(token),
        user_id=user.id,
        csrf_hash=security.hash_token(csrf),
        created_at=now,
        last_seen_at=now,
        expires_at=now + timedelta(days=security.SESSION_TTL_DAYS),
    )
    db.add(row)
    db.commit()
    return token, csrf


def get_auth_session(db: Session, token: str) -> AuthSession | None:
    """Look up a live session by raw token, or ``None`` if missing/expired."""
    if not token:
        return None
    row = db.get(AuthSession, security.hash_token(token))
    if row is None:
        return None
    if _as_utc(row.expires_at) <= datetime.now(timezone.utc):
        db.delete(row)
        db.commit()
        return None
    return row


def touch_session(db: Session, session: AuthSession) -> None:
    session.last_seen_at = utcnow()
    db.commit()


def revoke_session(db: Session, token: str) -> bool:
    """Delete the session for a raw token. Returns whether one existed."""
    if not token:
        return False
    row = db.get(AuthSession, security.hash_token(token))
    if row is None:
        return False
    db.delete(row)
    db.commit()
    return True


def validate_csrf(db: Session, session: AuthSession, csrf_token: str) -> bool:
    """Double-submit check: header token must hash to the session's stored value."""
    if not csrf_token:
        return False
    return security.constant_time_equals(
        security.hash_token(csrf_token), session.csrf_hash
    )


# A valid Argon2id hash of a random throwaway value used to equalise timing for
# unknown-email logins. Generated once at import from an unguessable secret.
_DUMMY_HASH = security.hash_password(security.new_token())
