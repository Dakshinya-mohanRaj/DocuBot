"""Authentication: signup, login, logout, cookies, CSRF and guards."""

import re

from fastapi.testclient import TestClient

from app.main import app

VALID_EMAIL = "alice@example.com"
VALID_PASSWORD = "correct-horse-battery"


def _cookies_named(response, name):
    return [c for c in response.headers.get_list("set-cookie") if name in c]


def test_signup_success_sets_httponly_session_cookie():
    client = TestClient(app)
    response = client.post(
        "/auth/signup",
        json={"email": VALID_EMAIL, "password": VALID_PASSWORD, "display_name": "Alice"},
    )
    assert response.status_code == 201, response.text

    user = response.json()["user"]
    assert user["email"] == VALID_EMAIL
    assert user["display_name"] == "Alice"
    assert "password" not in user
    assert "password_hash" not in response.text

    session_cookie = _cookies_named(response, "docubot_session")
    csrf_cookie = _cookies_named(response, "docubot_csrf")
    assert session_cookie and "HttpOnly" in session_cookie[0]
    assert csrf_cookie and "HttpOnly" not in csrf_cookie[0]  # readable for double-submit


def test_signup_rejects_duplicate_email():
    client = TestClient(app)
    assert client.post(
        "/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD}
    ).status_code == 201
    again = client.post(
        "/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD}
    )
    assert again.status_code == 409


def test_signup_rejects_short_password_and_bad_email():
    client = TestClient(app)
    assert client.post(
        "/auth/signup", json={"email": VALID_EMAIL, "password": "short"}
    ).status_code == 422
    assert client.post(
        "/auth/signup", json={"email": "not-an-email", "password": VALID_PASSWORD}
    ).status_code == 422


def test_me_reflects_signed_in_user(client):
    body = client.get("/auth/me").json()["user"]
    assert body["email"] == "user@example.com"


def test_me_is_401_without_session():
    assert TestClient(app).get("/auth/me").status_code == 401


def test_login_wrong_password_and_unknown_email_are_401():
    client = TestClient(app)
    client.post("/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD})
    fresh = TestClient(app)
    assert fresh.post(
        "/auth/login", json={"email": VALID_EMAIL, "password": "wrong-pass-123"}
    ).status_code == 401
    assert fresh.post(
        "/auth/login", json={"email": "nobody@example.com", "password": VALID_PASSWORD}
    ).status_code == 401


def test_login_success_issues_cookies_and_me_works():
    client = TestClient(app)
    client.post("/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD})
    fresh = TestClient(app)
    response = fresh.post(
        "/auth/login", json={"email": VALID_EMAIL, "password": VALID_PASSWORD}
    )
    assert response.status_code == 200
    assert fresh.get("/auth/me").status_code == 200


def test_logout_clears_cookies_and_revokes_session():
    client = TestClient(app)
    client.post("/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD})
    assert client.get("/auth/me").status_code == 200

    client.headers["X-CSRF-Token"] = client.cookies.get("docubot_csrf", "")
    assert client.post("/auth/logout").status_code == 200

    # Both cookies are gone, so the browser cannot keep using the session.
    assert "docubot_session" not in client.cookies
    assert client.get("/auth/me").status_code == 401


def test_logout_is_idempotent_and_needs_no_csrf():
    client = TestClient(app)
    client.post("/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD})
    client.headers.pop("X-CSRF-Token", None)  # a forced logout cannot be CSRF'd
    assert client.post("/auth/logout").status_code == 200
    assert client.post("/auth/logout").status_code == 200


def test_protected_routes_require_authentication(anon_client):
    assert anon_client.get("/ingest/documents").status_code == 401
    assert anon_client.get("/conversations").status_code == 401
    assert anon_client.get("/auth/me").status_code == 401
    assert anon_client.post("/chat", json={"message": "hi"}).status_code == 401
    assert anon_client.post("/ingest/text", json={"text": "hello"}).status_code == 401
    assert anon_client.post("/conversations").status_code == 401
    assert anon_client.delete("/ingest/a.txt").status_code == 401


def test_state_changing_requests_require_csrf(client):
    # Remove the double-submit header: the token no longer arrives twice.
    client.headers.pop("X-CSRF-Token", None)
    assert client.post("/chat", json={"message": "hi"}).status_code == 403
    assert client.post("/conversations").status_code == 403
    assert client.delete("/ingest/a.txt").status_code == 403

    # A mismatched (forged) token is also rejected.
    client.headers["X-CSRF-Token"] = "a-forged-token"
    assert client.post("/chat", json={"message": "hi"}).status_code == 403


def test_get_requests_do_not_need_csrf(client):
    client.headers.pop("X-CSRF-Token", None)
    assert client.get("/ingest/documents").status_code == 200
    assert client.get("/conversations").status_code == 200
    assert client.get("/auth/me").status_code == 200


def test_email_is_normalised_and_case_insensitive():
    client = TestClient(app)
    client.post("/auth/signup", json={"email": "  MixedCase@Example.COM ", "password": VALID_PASSWORD})
    body = client.get("/auth/me").json()["user"]
    assert body["email"] == "mixedcase@example.com"


def test_password_hash_is_never_exposed():
    client = TestClient(app)
    response = client.post(
        "/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD}
    )
    assert not re.search(r"\$argon2", response.text), "an Argon2 hash must never leave the server"


def test_signup_route_is_registered():
    # A missing router or route shows up as 404; validation/success never do.
    # This is the exact regression that made the deployed app return "Not Found".
    response = TestClient(app).post("/auth/signup", json={})
    assert response.status_code != 404, "POST /auth/signup is not mounted"


def test_login_requires_email_and_password():
    client = TestClient(app)
    assert client.post("/auth/login", json={"email": VALID_EMAIL}).status_code == 422
    assert client.post("/auth/login", json={"password": VALID_PASSWORD}).status_code == 422


def test_signup_returns_503_when_database_is_unavailable(monkeypatch):
    from sqlalchemy.exc import OperationalError

    from app.core import accounts

    def boom(*args, **kwargs):
        raise OperationalError("SELECT 1", {}, Exception("could not connect to server"))

    monkeypatch.setattr(accounts, "create_user", boom)
    response = TestClient(app).post(
        "/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD}
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    # The client gets a useful, generic message...
    assert "database" in detail.lower()
    # ...but never the driver text, which can embed the connection string.
    assert "connect" not in response.text.lower()
    assert "postgres" not in response.text.lower()


def test_unhandled_error_returns_clean_json_500(monkeypatch):
    from app.core import accounts

    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(accounts, "create_user", boom)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/auth/signup", json={"email": VALID_EMAIL, "password": VALID_PASSWORD}
    )
    assert response.status_code == 500
    assert "secret internal detail" not in response.text
