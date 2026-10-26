"""Minimal, dependency-free authentication for the FastAPI service.

The Streamlit client treats access tokens as opaque values.  This module keeps
identity and authorization on the API side, where it belongs, and signs short
lived HMAC tokens without adding a new external service.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import PROJECT_ROOT

TOKEN_TTL_SECONDS = 8 * 60 * 60
_PASSWORD_N = 2**14
_PASSWORD_R = 8
_PASSWORD_P = 1
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class CurrentUser:
    user_id: str
    email: str
    full_name: str


def _utc_timestamp() -> int:
    return int(time.time())


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _token_secret() -> bytes:
    """Read a deployment secret without creating or modifying a secret file."""
    configured = os.getenv("RAG_AUTH_SECRET")
    if configured:
        if len(configured) < 32:
            raise RuntimeError("RAG_AUTH_SECRET must contain at least 32 characters.")
        return configured.encode("utf-8")
    if os.getenv("RAG_ENV", "development").lower() == "production":
        raise RuntimeError("RAG_AUTH_SECRET must be configured in production.")
    # Development-only fallback keeps local tests runnable.  Deployments must
    # always set RAG_AUTH_SECRET, which invalidates any token made with this key.
    return b"development-only-change-me-before-production"


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_PASSWORD_N,
        r=_PASSWORD_R,
        p=_PASSWORD_P,
    )
    return "$".join(
        ("scrypt", str(_PASSWORD_N), str(_PASSWORD_R), str(_PASSWORD_P), _b64encode(salt), _b64encode(digest))
    )


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, n, r, p, salt, digest = stored.split("$")
        if algorithm != "scrypt":
            return False
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(candidate, _b64decode(digest))
    except (TypeError, ValueError):
        return False


def issue_access_token(user: CurrentUser) -> str:
    header = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode("utf-8"))
    payload = _b64encode(
        json.dumps(
            {
                "sub": user.user_id,
                "email": user.email,
                "name": user.full_name,
                "iat": _utc_timestamp(),
                "exp": _utc_timestamp() + TOKEN_TTL_SECONDS,
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signing_input = f"{header}.{payload}".encode("ascii")
    signature = _b64encode(hmac.new(_token_secret(), signing_input, hashlib.sha256).digest())
    return f"{header}.{payload}.{signature}"


def decode_access_token(token: str) -> dict:
    try:
        header, payload, signature = token.split(".")
        signing_input = f"{header}.{payload}".encode("ascii")
        expected = hmac.new(_token_secret(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature)):
            raise ValueError("invalid signature")
        claims = json.loads(_b64decode(payload))
        if not isinstance(claims, dict) or not claims.get("sub") or int(claims.get("exp", 0)) <= _utc_timestamp():
            raise ValueError("expired or malformed token")
        return claims
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Phiên đăng nhập không hợp lệ hoặc đã hết hạn.") from exc


class UserStore:
    """SQLite-backed user directory; it is intentionally separate from Qdrant."""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or PROJECT_ROOT / "data" / "processed" / "users.sqlite3")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )

    @staticmethod
    def _as_user(row: sqlite3.Row | None) -> CurrentUser | None:
        if row is None:
            return None
        return CurrentUser(user_id=row["user_id"], email=row["email"], full_name=row["full_name"])

    def get(self, user_id: str) -> CurrentUser | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, email, full_name FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return self._as_user(row)

    def get_by_email(self, email: str) -> CurrentUser | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, email, full_name FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        return self._as_user(row)

    def register(self, *, email: str, password: str, full_name: str) -> CurrentUser:
        normalized_email = email.strip().lower()
        clean_name = full_name.strip()
        # Keep compatibility with the existing local test account named
        # ``email`` while rejecting empty or whitespace-containing logins.
        # Production deployments can still use normal email addresses.
        if not normalized_email or any(char.isspace() for char in normalized_email) or len(normalized_email) > 320:
            raise ValueError("Email hoặc tên đăng nhập không hợp lệ.")
        if len(password) < 6:
            raise ValueError("Mật khẩu cần có ít nhất 6 ký tự.")
        if not clean_name or len(clean_name) > 200:
            raise ValueError("Họ tên không hợp lệ.")
        user = CurrentUser(user_id=uuid4().hex, email=normalized_email, full_name=clean_name)
        try:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO users (user_id, email, full_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user.user_id, user.email, user.full_name, _hash_password(password), _utc_timestamp()),
                )
        except sqlite3.IntegrityError as exc:
            raise FileExistsError("Email đã tồn tại.") from exc
        return user

    def authenticate(self, *, email: str, password: str) -> CurrentUser | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT user_id, email, full_name, password_hash FROM users WHERE email = ?", (email.strip().lower(),)
            ).fetchone()
        if row is None or not _verify_password(password, row["password_hash"]):
            return None
        return self._as_user(row)


@lru_cache(maxsize=1)
def get_user_store() -> UserStore:
    return UserStore()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    user_store: UserStore = Depends(get_user_store),
) -> CurrentUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Yêu cầu đăng nhập.")
    claims = decode_access_token(credentials.credentials)
    user = user_store.get(str(claims["sub"]))
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Tài khoản không còn tồn tại.")
    return user
