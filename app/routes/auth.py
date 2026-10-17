"""Authentication endpoints: signup, login, logout and current-user lookup.

Sessions are server-side and delivered as HttpOnly cookies. A companion
readable CSRF cookie enables the double-submit check on unsafe requests.
"""

import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from app.core import accounts, security
from app.core.db import get_db
from app.core.dependencies import CurrentUser
from app.core.logging_config import log_event
from app.core.models import User

logger = logging.getLogger("docubot.auth")

router = APIRouter(prefix="/auth")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_COOKIE_MAX_AGE = security.SESSION_TTL_DAYS * 24 * 60 * 60


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=security.MAX_PASSWORD_LENGTH)
    display_name: str | None = Field(default=None, max_length=120)

    @field_validator("email")
    @classmethod
    def _valid_email(cls, value: str) -> str:
        value = value.strip()
        if not _EMAIL_RE.match(value):
            raise ValueError("Enter a valid email address.")
        return value


class LoginRequest(BaseModel):
    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=security.MAX_PASSWORD_LENGTH)


def _validate_password(password: str) -> None:
    if len(password) < security.MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Password must be at least {security.MIN_PASSWORD_LENGTH} characters.",
        )


def _user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.display_name,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _set_auth_cookies(response: Response, token: str, csrf: str) -> None:
    # Session cookie: HttpOnly so JS (and therefore XSS) cannot read it.
    response.set_cookie(
        security.SESSION_COOKIE,
        token,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        secure=security.COOKIE_SECURE,
        samesite="lax",
        path="/",
    )
    # CSRF cookie: intentionally readable by the same-origin app for
    # double-submit; it is useless to an attacker who cannot read it cross-site.
    response.set_cookie(
        security.CSRF_COOKIE,
        csrf,
        max_age=_COOKIE_MAX_AGE,
        httponly=False,
        secure=security.COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(security.SESSION_COOKIE, path="/")
    response.delete_cookie(security.CSRF_COOKIE, path="/")


@router.post("/signup", status_code=status.HTTP_201_CREATED)
def signup(payload: SignupRequest, response: Response, db: Session = Depends(get_db)) -> dict:
    _validate_password(payload.password)
    try:
        user = accounts.create_user(db, payload.email, payload.password, payload.display_name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists.",
        )

    token, csrf = accounts.create_auth_session(db, user)
    _set_auth_cookies(response, token, csrf)
    log_event(logger, "signup", user_id=user.id)
    return {"user": _user_dict(user)}


@router.post("/login")
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> dict:
    user = accounts.authenticate(db, payload.email, payload.password)
    if user is None:
        # Deliberately generic so we never reveal which accounts exist.
        log_event(logger, "login_failed", level=logging.WARNING)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token, csrf = accounts.create_auth_session(db, user)
    _set_auth_cookies(response, token, csrf)
    log_event(logger, "login", user_id=user.id)
    return {"user": _user_dict(user)}


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)) -> dict:
    # Idempotent: revoke the server-side session if one is present, then always
    # clear both cookies so the browser cannot keep replaying a stale token.
    token = request.cookies.get(security.SESSION_COOKIE, "")
    if token:
        accounts.revoke_session(db, token)
    _clear_auth_cookies(response)
    return {"ok": True}


@router.get("/me")
def me(user: CurrentUser) -> dict:
    return {"user": _user_dict(user)}
