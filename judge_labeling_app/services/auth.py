from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

PBKDF2_ITERATIONS = 200_000


def _hash_password(password: str, salt_hex: str) -> str:
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return digest.hex()


def _normalize_credentials(raw_data: dict[str, Any]) -> dict[str, dict[str, str]]:
    normalized: dict[str, dict[str, str]] = {}
    for username, record in raw_data.items():
        cleaned_username = str(username).strip()
        if not cleaned_username:
            continue

        # Formato simplificado: "J1": "1234"
        if isinstance(record, str):
            plain_password = record.strip()
            if plain_password:
                normalized[cleaned_username] = {"password": plain_password}
            continue

        if not isinstance(record, dict):
            continue

        # Formato simplificado: "J1": {"password": "1234"}
        plain_password = str(record.get("password", "")).strip()
        if plain_password:
            normalized[cleaned_username] = {"password": plain_password}
            continue

        # Formato anterior (compatibilidade): hash+salt
        salt = str(record.get("salt", "")).strip()
        password_hash = str(record.get("password_hash", "")).strip()
        if salt and password_hash:
            normalized[cleaned_username] = {
                "salt": salt,
                "password_hash": password_hash,
            }
    return normalized


def _load_from_local_file(app_dir: Path) -> dict[str, dict[str, str]]:
    secrets_path = str(st.secrets.get("credentials_path", "")).strip()
    env_path = os.getenv("JUDGE_CREDENTIALS_PATH", "").strip()
    selected_path = env_path or secrets_path or "credentials/juizes.json"
    credentials_path = Path(selected_path)
    if not credentials_path.is_absolute():
        credentials_path = app_dir / credentials_path

    if not credentials_path.exists():
        return {}

    with credentials_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)
    if not isinstance(raw_data, dict):
        return {}
    return _normalize_credentials(raw_data)


def load_credentials(app_dir: Path) -> dict[str, dict[str, str]]:
    raw_direct = st.secrets.get("auth_credentials")
    if isinstance(raw_direct, dict):
        return _normalize_credentials(raw_direct)

    raw_json = st.secrets.get("credentials_json")
    if isinstance(raw_json, str) and raw_json.strip():
        parsed = json.loads(raw_json)
        if isinstance(parsed, dict):
            return _normalize_credentials(parsed)

    return _load_from_local_file(app_dir)


def verify_login(
    username: str,
    password: str,
    credentials: dict[str, dict[str, str]],
) -> bool:
    cleaned_username = username.strip()
    if not cleaned_username or not password:
        return False
    record = credentials.get(cleaned_username)
    if not record:
        return False

    plain_password = str(record.get("password", "")).strip()
    if plain_password:
        return hmac.compare_digest(password, plain_password)

    salt = str(record.get("salt", "")).strip()
    password_hash = str(record.get("password_hash", "")).strip()
    if not salt or not password_hash:
        return False

    computed = _hash_password(password, salt)
    return hmac.compare_digest(computed, password_hash)
