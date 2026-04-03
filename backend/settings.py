from __future__ import annotations

from typing import Any

ALLOWED_APPEARANCE = {"light", "dark", "system"}


def ensure_user_settings(conn, user_id: str) -> dict[str, Any]:
    """
    Ensure the user has a settings row, then return it.
    """
    normalized_user_id = (user_id or "").strip()
    if not normalized_user_id:
        raise ValueError("user_id is required.")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO user_settings (user_id)
            VALUES (%s)
            ON CONFLICT (user_id) DO NOTHING
            """,
            (normalized_user_id,),
        )
        cur.execute(
            """
            SELECT user_id, appearance, created_at, updated_at
            FROM user_settings
            WHERE user_id = %s
            """,
            (normalized_user_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError("Failed to load user settings.")

    return {
        "user_id": row[0],
        "appearance": row[1],
        "created_at": row[2],
        "updated_at": row[3],
    }


def update_appearance(conn, user_id: str, appearance: str) -> dict[str, Any]:
    """
    Update appearance for an existing user settings row.
    """
    normalized_user_id = (user_id or "").strip()
    normalized_appearance = (appearance or "").strip().lower()

    if not normalized_user_id:
        raise ValueError("user_id is required.")
    if normalized_appearance not in ALLOWED_APPEARANCE:
        raise ValueError("appearance must be one of: light, dark, system.")

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE user_settings
            SET appearance = %s,
                updated_at = NOW()
            WHERE user_id = %s
            RETURNING user_id, appearance, created_at, updated_at
            """,
            (normalized_appearance, normalized_user_id),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError("User settings row was not found.")

    return {
        "user_id": row[0],
        "appearance": row[1],
        "created_at": row[2],
        "updated_at": row[3],
    }