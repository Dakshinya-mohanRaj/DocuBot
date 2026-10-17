"""FastAPI dependencies for authentication and CSRF enforcement.

``get_current_user`` resolves the signed-in account from the opaque session
cookie. ``require_csrf`` adds the double-submit check for state-changing
requests. Both are request-scoped and share one database session via FastAPI's
dependency cache.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core import accounts, security
from app.core.db import get_db
from app.core.models import User

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated.",
)
_FORBIDDEN_CSRF = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="CSRF validation failed.",
)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


def get_optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    """Resolve the user if a valid session cookie is present, else ``None``."""
    token = request.cookies.get(security.SESSION_COOKIE, "")
    auth_session = accounts.get_auth_session(db, token)
    if auth_session is None:
        return None
    user = accounts.get_user(db, auth_session.user_id)
    if user is None:
        return None
    request.state.auth_session = auth_session
    request.state.session_token = token
    return user


def get_current_user(user: User | None = Depends(get_optional_user)) -> User:
    if user is None:
        raise _UNAUTHORIZED
    return user


def require_csrf(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Enforce double-submit CSRF validation on unsafe methods.

    The browser sends the same token twice: as a readable ``docubot_csrf``
    cookie and as the ``X-CSRF-Token`` header. A cross-site attacker can make
    the browser send the cookie but cannot read it to set the header, so a
    missing/mismatched pair is rejected. The header must also hash to the value
    bound to this session server-side.
    """
    if request.method in _SAFE_METHODS:
        return
    header_token = request.headers.get(security.CSRF_HEADER, "")
    cookie_token = request.cookies.get(security.CSRF_COOKIE, "")
    auth_session = getattr(request.state, "auth_session", None)
    if auth_session is None:
        raise _UNAUTHORIZED
    if not security.constant_time_equals(header_token, cookie_token):
        raise _FORBIDDEN_CSRF
    if not accounts.validate_csrf(db, auth_session, header_token):
        raise _FORBIDDEN_CSRF


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
CsrfProtected = Annotated[None, Depends(require_csrf)]
