"""Password hashing and signed bearer-token helpers."""

import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from pwdlib import PasswordHash

password_hash = PasswordHash.recommended()
SECRET_KEY = os.getenv("SMARTCART_SECRET_KEY", "development-only-change-me")
ALGORITHM = "HS256"
TOKEN_MINUTES = 60


def hash_password(password: str) -> str:
    return password_hash.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return password_hash.verify(password, hashed)


def create_access_token(user_id: int, role: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_MINUTES)
    return jwt.encode({"sub": str(user_id), "role": role, "exp": expires}, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


__all__ = ["JWTError", "create_access_token", "decode_access_token", "hash_password", "verify_password"]