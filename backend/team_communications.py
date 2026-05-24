from __future__ import annotations

"""
Team communications API.

Responsibilities:
- Business/Enterprise-only entitlement guard for team communication features
- direct message and group conversation records
- conversation membership
- message persistence
- call-session records for 1-to-1 and group calls
- call participant state
- member presence

Notes:
- LiveKit access tokens are generated server-side from LIVEKIT_API_KEY,
  LIVEKIT_API_SECRET, and LIVEKIT_URL.
- Messaging/calls are organization-scoped and require an active organization
  membership plus an active Business/Enterprise organization entitlement.
"""

from datetime import datetime, timedelta
import os
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, field_validator, model_validator

from backend.auth0_dependencies import AuthenticatedUser, get_current_user
from backend.database import get_db
from backend.subscriptions import get_user_entitlement


router = APIRouter(tags=["team_communications"])

LIVEKIT_API_KEY_ENV = "LIVEKIT_API_KEY"
LIVEKIT_API_SECRET_ENV = "LIVEKIT_API_SECRET"
LIVEKIT_URL_ENV = "LIVEKIT_URL"
LIVEKIT_TOKEN_TTL_MINUTES_ENV = "LIVEKIT_TOKEN_TTL_MINUTES"
DEFAULT_LIVEKIT_TOKEN_TTL_MINUTES = 120

ConversationType = Literal["dm", "group"]
ConversationStatus = Literal["active", "archived"]
ConversationRole = Literal["owner", "admin", "member"]
PresenceStatus = Literal["online", "offline", "in_call"]


class CreateConversationRequest(BaseModel):
    type: ConversationType
    name: str | None = None
    member_user_ids: list[str] = []

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in {"dm", "group"}:
            raise ValueError("type must be one of: dm, group.")
        return normalized

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("member_user_ids")
    @classmethod
    def normalize_member_user_ids(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized_ids: list[str] = []

        for raw_user_id in value or []:
            user_id = normalize_user_id(raw_user_id)
            if user_id in seen:
                continue
            seen.add(user_id)
            normalized_ids.append(user_id)

        return normalized_ids

    @model_validator(mode="after")
    def validate_shape(self):
        if self.type == "dm" and len(self.member_user_ids) != 1:
            raise ValueError("Direct messages require exactly one target member.")
        if self.type == "group":
            if not self.name:
                raise ValueError("Group conversations require a name.")
            if len(self.member_user_ids) < 1:
                raise ValueError("Group conversations require at least one member.")
        return self


class SendMessageRequest(BaseModel):
    body: str

    @field_validator("body")
    @classmethod
    def validate_body(cls, value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError("Message body is required.")
        if len(normalized) > 5000:
            raise ValueError("Message body cannot exceed 5000 characters.")
        return normalized


class UpdatePresenceRequest(BaseModel):
    status: PresenceStatus = "online"

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in {"online", "offline", "in_call"}:
            raise ValueError("status must be one of: online, offline, in_call.")
        return normalized


def normalize_user_id(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise ValueError("user_id is required.")
    return normalized


def entitlement_value(entitlement: Any, key: str, default: Any = None) -> Any:
    if isinstance(entitlement, dict):
        return entitlement.get(key, default)
    return getattr(entitlement, key, default)


def parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def user_public_payload(current_user: AuthenticatedUser) -> dict[str, Any]:
    return {
        "id": current_user.user_id,
        "name": current_user.claims.get("name"),
        "email": current_user.claims.get("email"),
        "picture": current_user.claims.get("picture"),
    }


def get_participant_display_name(current_user: AuthenticatedUser) -> str:
    for key in ("name", "email", "nickname", "preferred_username"):
        value = current_user.claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return current_user.user_id


def get_livekit_config() -> dict[str, str | int]:
    api_key = os.getenv(LIVEKIT_API_KEY_ENV, "").strip()
    api_secret = os.getenv(LIVEKIT_API_SECRET_ENV, "").strip()
    server_url = os.getenv(LIVEKIT_URL_ENV, "").strip()

    if not api_key or not api_secret or not server_url:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "livekit_not_configured",
                "message": (
                    "LiveKit is not configured. Set LIVEKIT_API_KEY, "
                    "LIVEKIT_API_SECRET, and LIVEKIT_URL."
                ),
            },
        )

    try:
        ttl_minutes = int(
            os.getenv(
                LIVEKIT_TOKEN_TTL_MINUTES_ENV,
                str(DEFAULT_LIVEKIT_TOKEN_TTL_MINUTES),
            )
        )
    except ValueError:
        ttl_minutes = DEFAULT_LIVEKIT_TOKEN_TTL_MINUTES

    ttl_minutes = max(5, min(ttl_minutes, 12 * 60))

    return {
        "api_key": api_key,
        "api_secret": api_secret,
        "server_url": server_url,
        "ttl_minutes": ttl_minutes,
    }


def generate_livekit_join_payload(
    *,
    current_user: AuthenticatedUser,
    room_name: str,
) -> dict[str, Any]:
    config = get_livekit_config()

    try:
        from livekit import api as livekit_api
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "livekit_sdk_missing",
                "message": "Install the LiveKit server SDK with: pip install livekit-api",
            },
        ) from exc

    participant_identity = current_user.user_id
    participant_name = get_participant_display_name(current_user)
    ttl = timedelta(minutes=int(config["ttl_minutes"]))

    token = (
        livekit_api.AccessToken(
            str(config["api_key"]),
            str(config["api_secret"]),
        )
        .with_identity(participant_identity)
        .with_name(participant_name)
        .with_ttl(ttl)
        .with_grants(
            livekit_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .to_jwt()
    )

    return {
        "server_url": config["server_url"],
        "room_name": room_name,
        "token": token,
        "token_status": "configured",
        "participant_identity": participant_identity,
        "participant_name": participant_name,
        "expires_in_seconds": int(ttl.total_seconds()),
    }


def row_to_conversation(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "organization_id": row[1],
        "type": row[2],
        "name": row[3],
        "created_by_user_id": row[4],
        "status": row[5],
        "last_message_at": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


def row_to_conversation_member(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "conversation_id": row[1],
        "organization_id": row[2],
        "user_id": row[3],
        "role": row[4],
        "status": row[5],
        "joined_at": row[6],
        "removed_at": row[7],
        "created_at": row[8],
        "updated_at": row[9],
    }


def row_to_message(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "conversation_id": row[1],
        "organization_id": row[2],
        "sender_user_id": row[3],
        "message_type": row[4],
        "body": row[5],
        "metadata": row[6],
        "edited_at": row[7],
        "deleted_at": row[8],
        "created_at": row[9],
        "updated_at": row[10],
    }


def row_to_call_session(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "organization_id": row[1],
        "conversation_id": row[2],
        "type": row[3],
        "status": row[4],
        "created_by_user_id": row[5],
        "livekit_room_name": row[6],
        "started_at": row[7],
        "ended_at": row[8],
        "created_at": row[9],
        "updated_at": row[10],
    }


def row_to_call_participant(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "call_session_id": row[1],
        "organization_id": row[2],
        "user_id": row[3],
        "status": row[4],
        "invited_at": row[5],
        "joined_at": row[6],
        "left_at": row[7],
        "created_at": row[8],
        "updated_at": row[9],
    }


def row_to_presence(row) -> dict[str, Any]:
    return {
        "organization_id": row[0],
        "user_id": row[1],
        "status": row[2],
        "last_seen_at": row[3],
        "updated_at": row[4],
    }


def get_active_organization_membership(
    conn,
    organization_id: int,
    user_id: str,
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, organization_id, user_id, role, status
            FROM organization_members
            WHERE organization_id = %s
              AND user_id = %s
              AND status = 'active'
            """,
            (organization_id, user_id),
        )
        row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": row[0],
        "organization_id": row[1],
        "user_id": row[2],
        "role": row[3],
        "status": row[4],
    }


def require_business_or_enterprise_organization(
    conn,
    organization_id: int,
    current_user: AuthenticatedUser,
) -> dict[str, Any]:
    """
    Require:
    - current user is an active member of this organization
    - current user's entitlement is an active Business/Enterprise org entitlement
    - entitlement organization_id matches the requested organization
    """

    membership = get_active_organization_membership(
        conn,
        organization_id,
        current_user.user_id,
    )

    if membership is None:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_access_denied",
                "message": "You are not an active member of this organization.",
            },
        )

    entitlement = get_user_entitlement(current_user.user_id)
    entitlement_plan = entitlement_value(entitlement, "plan")
    entitlement_status = entitlement_value(entitlement, "status")
    entitlement_source = entitlement_value(entitlement, "source")
    entitlement_org_id = parse_optional_int(
        entitlement_value(entitlement, "organization_id")
    )
    entitlement_is_paid = bool(entitlement_value(entitlement, "is_paid"))

    if (
        not entitlement_is_paid
        or entitlement_source != "organization"
        or entitlement_plan not in {"business", "enterprise"}
        or entitlement_status != "active"
        or entitlement_org_id != organization_id
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "team_communications_unavailable",
                "message": "Team messaging and calls are available only for active Business or Enterprise organizations.",
            },
        )

    return {
        "membership": membership,
        "entitlement": {
            "plan": entitlement_plan,
            "status": entitlement_status,
            "source": entitlement_source,
            "organization_id": entitlement_org_id,
            "is_paid": entitlement_is_paid,
        },
    }


def require_org_admin_or_owner(membership: dict[str, Any]) -> None:
    if membership["role"] not in {"owner", "admin"}:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_admin_required",
                "message": "Only organization owners or admins can perform this action.",
            },
        )


def require_active_org_members(conn, organization_id: int, user_ids: list[str]) -> None:
    if not user_ids:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT user_id
            FROM organization_members
            WHERE organization_id = %s
              AND user_id = ANY(%s)
              AND status = 'active'
            """,
            (organization_id, user_ids),
        )
        active_rows = cur.fetchall()

    active_user_ids = {row[0] for row in active_rows}
    missing_user_ids = [user_id for user_id in user_ids if user_id not in active_user_ids]

    if missing_user_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_conversation_members",
                "message": "All conversation members must be active members of the organization.",
                "invalid_user_ids": missing_user_ids,
            },
        )


def get_conversation(conn, conversation_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, organization_id, type, name, created_by_user_id, status,
                   last_message_at, created_at, updated_at
            FROM organization_conversations
            WHERE id = %s
            """,
            (conversation_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "conversation_not_found",
                "message": "Conversation was not found.",
            },
        )

    return row_to_conversation(row)


def require_active_conversation_member(
    conn,
    conversation_id: int,
    user_id: str,
) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, conversation_id, organization_id, user_id, role, status,
                   joined_at, removed_at, created_at, updated_at
            FROM conversation_members
            WHERE conversation_id = %s
              AND user_id = %s
              AND status = 'active'
            """,
            (conversation_id, user_id),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "conversation_access_denied",
                "message": "You are not an active member of this conversation.",
            },
        )

    return row_to_conversation_member(row)


def fetch_conversation_members(conn, conversation_id: int) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, conversation_id, organization_id, user_id, role, status,
                   joined_at, removed_at, created_at, updated_at
            FROM conversation_members
            WHERE conversation_id = %s
            ORDER BY
                CASE role
                    WHEN 'owner' THEN 1
                    WHEN 'admin' THEN 2
                    ELSE 3
                END,
                created_at ASC,
                id ASC
            """,
            (conversation_id,),
        )
        rows = cur.fetchall()

    return [row_to_conversation_member(row) for row in rows]


def get_existing_dm_conversation(
    conn,
    organization_id: int,
    user_a: str,
    user_b: str,
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT oc.id, oc.organization_id, oc.type, oc.name,
                   oc.created_by_user_id, oc.status, oc.last_message_at,
                   oc.created_at, oc.updated_at
            FROM organization_conversations oc
            JOIN conversation_members cm_a
              ON cm_a.conversation_id = oc.id
             AND cm_a.user_id = %s
             AND cm_a.status = 'active'
            JOIN conversation_members cm_b
              ON cm_b.conversation_id = oc.id
             AND cm_b.user_id = %s
             AND cm_b.status = 'active'
            WHERE oc.organization_id = %s
              AND oc.type = 'dm'
              AND oc.status = 'active'
              AND (
                  SELECT COUNT(*)
                  FROM conversation_members cm_count
                  WHERE cm_count.conversation_id = oc.id
                    AND cm_count.status = 'active'
              ) = 2
            LIMIT 1
            """,
            (user_a, user_b, organization_id),
        )
        row = cur.fetchone()

    return row_to_conversation(row) if row is not None else None


def get_call_session(conn, call_session_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, organization_id, conversation_id, type, status,
                   created_by_user_id, livekit_room_name, started_at, ended_at,
                   created_at, updated_at
            FROM call_sessions
            WHERE id = %s
            """,
            (call_session_id,),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "call_not_found",
                "message": "Call session was not found.",
            },
        )

    return row_to_call_session(row)


def upsert_presence(
    conn,
    organization_id: int,
    user_id: str,
    status: str,
) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO member_presence (
                organization_id,
                user_id,
                status,
                last_seen_at,
                updated_at
            )
            VALUES (%s, %s, %s, NOW(), NOW())
            ON CONFLICT (organization_id, user_id) DO UPDATE SET
                status = EXCLUDED.status,
                last_seen_at = NOW(),
                updated_at = NOW()
            RETURNING organization_id, user_id, status, last_seen_at, updated_at
            """,
            (organization_id, user_id, status),
        )
        row = cur.fetchone()

    return row_to_presence(row)


@router.get("/organizations/{organization_id}/conversations")
def list_conversations(
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            require_business_or_enterprise_organization(
                conn,
                organization_id,
                current_user,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT oc.id, oc.organization_id, oc.type, oc.name,
                           oc.created_by_user_id, oc.status, oc.last_message_at,
                           oc.created_at, oc.updated_at
                    FROM organization_conversations oc
                    JOIN conversation_members cm
                      ON cm.conversation_id = oc.id
                     AND cm.user_id = %s
                     AND cm.status = 'active'
                    WHERE oc.organization_id = %s
                      AND oc.status = 'active'
                    ORDER BY COALESCE(oc.last_message_at, oc.updated_at) DESC,
                             oc.id DESC
                    """,
                    (current_user.user_id, organization_id),
                )
                rows = cur.fetchall()

        return {
            "success": True,
            "conversations": [row_to_conversation(row) for row in rows],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "conversations_load_failed",
                "message": "Could not load conversations.",
            },
        ) from exc


@router.post("/organizations/{organization_id}/conversations")
def create_conversation(
    payload: CreateConversationRequest,
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            access = require_business_or_enterprise_organization(
                conn,
                organization_id,
                current_user,
            )
            org_membership = access["membership"]

            if payload.type == "group":
                require_org_admin_or_owner(org_membership)

            member_user_ids = [
                user_id
                for user_id in payload.member_user_ids
                if user_id != current_user.user_id
            ]

            if payload.type == "dm":
                target_user_id = member_user_ids[0]
                require_active_org_members(conn, organization_id, [target_user_id])

                existing_dm = get_existing_dm_conversation(
                    conn,
                    organization_id,
                    current_user.user_id,
                    target_user_id,
                )

                if existing_dm is not None:
                    return {
                        "success": True,
                        "conversation": existing_dm,
                        "members": fetch_conversation_members(conn, existing_dm["id"]),
                        "already_exists": True,
                    }

                final_member_ids = [current_user.user_id, target_user_id]
                conversation_name = None
            else:
                require_active_org_members(conn, organization_id, member_user_ids)
                final_member_ids = [current_user.user_id, *member_user_ids]
                conversation_name = payload.name

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO organization_conversations (
                        organization_id,
                        type,
                        name,
                        created_by_user_id
                    )
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, organization_id, type, name,
                              created_by_user_id, status, last_message_at,
                              created_at, updated_at
                    """,
                    (
                        organization_id,
                        payload.type,
                        conversation_name,
                        current_user.user_id,
                    ),
                )
                conversation_row = cur.fetchone()
                conversation = row_to_conversation(conversation_row)

                for member_user_id in final_member_ids:
                    role = "owner" if member_user_id == current_user.user_id else "member"
                    cur.execute(
                        """
                        INSERT INTO conversation_members (
                            conversation_id,
                            organization_id,
                            user_id,
                            role,
                            status,
                            joined_at
                        )
                        VALUES (%s, %s, %s, %s, 'active', NOW())
                        ON CONFLICT (conversation_id, user_id) DO UPDATE SET
                            role = EXCLUDED.role,
                            status = 'active',
                            removed_at = NULL,
                            updated_at = NOW()
                        """,
                        (
                            conversation["id"],
                            organization_id,
                            member_user_id,
                            role,
                        ),
                    )

        return {
            "success": True,
            "conversation": conversation,
            "members": fetch_conversation_members(conn, conversation["id"]),
            "already_exists": False,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_conversation",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "conversation_create_failed",
                "message": "Could not create conversation.",
            },
        ) from exc


@router.get("/conversations/{conversation_id}/messages")
def list_messages(
    conversation_id: int = Path(..., ge=1),
    limit: int = Query(50, ge=1, le=100),
    before_message_id: int | None = Query(None, ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            conversation = get_conversation(conn, conversation_id)
            require_business_or_enterprise_organization(
                conn,
                conversation["organization_id"],
                current_user,
            )
            require_active_conversation_member(
                conn,
                conversation_id,
                current_user.user_id,
            )

            with conn.cursor() as cur:
                if before_message_id is not None:
                    cur.execute(
                        """
                        SELECT id, conversation_id, organization_id,
                               sender_user_id, message_type, body, metadata,
                               edited_at, deleted_at, created_at, updated_at
                        FROM conversation_messages
                        WHERE conversation_id = %s
                          AND deleted_at IS NULL
                          AND id < %s
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (conversation_id, before_message_id, limit),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, conversation_id, organization_id,
                               sender_user_id, message_type, body, metadata,
                               edited_at, deleted_at, created_at, updated_at
                        FROM conversation_messages
                        WHERE conversation_id = %s
                          AND deleted_at IS NULL
                        ORDER BY id DESC
                        LIMIT %s
                        """,
                        (conversation_id, limit),
                    )

                rows = cur.fetchall()

        messages = [row_to_message(row) for row in rows]
        messages.reverse()

        return {
            "success": True,
            "conversation": conversation,
            "messages": messages,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "messages_load_failed",
                "message": "Could not load messages.",
            },
        ) from exc


@router.post("/conversations/{conversation_id}/messages")
def send_message(
    payload: SendMessageRequest,
    conversation_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            conversation = get_conversation(conn, conversation_id)
            require_business_or_enterprise_organization(
                conn,
                conversation["organization_id"],
                current_user,
            )
            require_active_conversation_member(
                conn,
                conversation_id,
                current_user.user_id,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_messages (
                        conversation_id,
                        organization_id,
                        sender_user_id,
                        message_type,
                        body
                    )
                    VALUES (%s, %s, %s, 'text', %s)
                    RETURNING id, conversation_id, organization_id,
                              sender_user_id, message_type, body, metadata,
                              edited_at, deleted_at, created_at, updated_at
                    """,
                    (
                        conversation_id,
                        conversation["organization_id"],
                        current_user.user_id,
                        payload.body,
                    ),
                )
                row = cur.fetchone()

        return {
            "success": True,
            "message": row_to_message(row),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_message",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "message_send_failed",
                "message": "Could not send message.",
            },
        ) from exc


@router.post("/conversations/{conversation_id}/calls")
def start_call(
    conversation_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            conversation = get_conversation(conn, conversation_id)
            require_business_or_enterprise_organization(
                conn,
                conversation["organization_id"],
                current_user,
            )
            require_active_conversation_member(
                conn,
                conversation_id,
                current_user.user_id,
            )

            conversation_members = [
                member
                for member in fetch_conversation_members(conn, conversation_id)
                if member["status"] == "active"
            ]

            if len(conversation_members) < 2:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "not_enough_call_participants",
                        "message": "A call requires at least two active conversation members.",
                    },
                )

            call_type = "group" if conversation["type"] == "group" else "one_to_one"
            livekit_room_name = (
                f"org-{conversation['organization_id']}-"
                f"conv-{conversation_id}-"
                f"call-{uuid4().hex}"
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO call_sessions (
                        organization_id,
                        conversation_id,
                        type,
                        status,
                        created_by_user_id,
                        livekit_room_name
                    )
                    VALUES (%s, %s, %s, 'ringing', %s, %s)
                    RETURNING id, organization_id, conversation_id, type, status,
                              created_by_user_id, livekit_room_name, started_at,
                              ended_at, created_at, updated_at
                    """,
                    (
                        conversation["organization_id"],
                        conversation_id,
                        call_type,
                        current_user.user_id,
                        livekit_room_name,
                    ),
                )
                call_row = cur.fetchone()
                call = row_to_call_session(call_row)

                for member in conversation_members:
                    participant_status = (
                        "joined"
                        if member["user_id"] == current_user.user_id
                        else "invited"
                    )
                    cur.execute(
                        """
                        INSERT INTO call_participants (
                            call_session_id,
                            organization_id,
                            user_id,
                            status,
                            joined_at
                        )
                        VALUES (%s, %s, %s, %s, CASE WHEN %s = 'joined' THEN NOW() ELSE NULL END)
                        ON CONFLICT (call_session_id, user_id) DO UPDATE SET
                            status = EXCLUDED.status,
                            joined_at = COALESCE(call_participants.joined_at, EXCLUDED.joined_at),
                            updated_at = NOW()
                        """,
                        (
                            call["id"],
                            conversation["organization_id"],
                            member["user_id"],
                            participant_status,
                            participant_status,
                        ),
                    )

                cur.execute(
                    """
                    UPDATE call_sessions
                    SET status = 'active',
                        started_at = COALESCE(started_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, organization_id, conversation_id, type, status,
                              created_by_user_id, livekit_room_name, started_at,
                              ended_at, created_at, updated_at
                    """,
                    (call["id"],),
                )
                call = row_to_call_session(cur.fetchone())

                cur.execute(
                    """
                    INSERT INTO conversation_messages (
                        conversation_id,
                        organization_id,
                        sender_user_id,
                        message_type,
                        body,
                        metadata
                    )
                    VALUES (%s, %s, %s, 'call_event', %s, %s::jsonb)
                    """,
                    (
                        conversation_id,
                        conversation["organization_id"],
                        current_user.user_id,
                        "Call started.",
                        f'{{"call_session_id": {call["id"]}}}',
                    ),
                )

                upsert_presence(
                    conn,
                    conversation["organization_id"],
                    current_user.user_id,
                    "in_call",
                )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, call_session_id, organization_id, user_id,
                           status, invited_at, joined_at, left_at,
                           created_at, updated_at
                    FROM call_participants
                    WHERE call_session_id = %s
                    ORDER BY id ASC
                    """,
                    (call["id"],),
                )
                participant_rows = cur.fetchall()

        return {
            "success": True,
            "call": call,
            "participants": [
                row_to_call_participant(row) for row in participant_rows
            ],
            "livekit": generate_livekit_join_payload(
                current_user=current_user,
                room_name=call["livekit_room_name"],
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "call_start_failed",
                "message": "Could not start call.",
            },
        ) from exc


@router.post("/calls/{call_session_id}/join")
def join_call(
    call_session_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            call = get_call_session(conn, call_session_id)
            require_business_or_enterprise_organization(
                conn,
                call["organization_id"],
                current_user,
            )

            if call["conversation_id"] is not None:
                require_active_conversation_member(
                    conn,
                    call["conversation_id"],
                    current_user.user_id,
                )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO call_participants (
                        call_session_id,
                        organization_id,
                        user_id,
                        status,
                        joined_at
                    )
                    VALUES (%s, %s, %s, 'joined', NOW())
                    ON CONFLICT (call_session_id, user_id) DO UPDATE SET
                        status = 'joined',
                        joined_at = COALESCE(call_participants.joined_at, NOW()),
                        left_at = NULL,
                        updated_at = NOW()
                    RETURNING id, call_session_id, organization_id, user_id,
                              status, invited_at, joined_at, left_at,
                              created_at, updated_at
                    """,
                    (
                        call_session_id,
                        call["organization_id"],
                        current_user.user_id,
                    ),
                )
                participant = row_to_call_participant(cur.fetchone())

                cur.execute(
                    """
                    UPDATE call_sessions
                    SET status = 'active',
                        started_at = COALESCE(started_at, NOW()),
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, organization_id, conversation_id, type, status,
                              created_by_user_id, livekit_room_name, started_at,
                              ended_at, created_at, updated_at
                    """,
                    (call_session_id,),
                )
                call = row_to_call_session(cur.fetchone())

                presence = upsert_presence(
                    conn,
                    call["organization_id"],
                    current_user.user_id,
                    "in_call",
                )

        return {
            "success": True,
            "call": call,
            "participant": participant,
            "presence": presence,
            "livekit": generate_livekit_join_payload(
                current_user=current_user,
                room_name=call["livekit_room_name"],
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "call_join_failed",
                "message": "Could not join call.",
            },
        ) from exc


@router.post("/calls/{call_session_id}/leave")
def leave_call(
    call_session_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            call = get_call_session(conn, call_session_id)
            require_business_or_enterprise_organization(
                conn,
                call["organization_id"],
                current_user,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE call_participants
                    SET status = 'left',
                        left_at = NOW(),
                        updated_at = NOW()
                    WHERE call_session_id = %s
                      AND user_id = %s
                    RETURNING id, call_session_id, organization_id, user_id,
                              status, invited_at, joined_at, left_at,
                              created_at, updated_at
                    """,
                    (call_session_id, current_user.user_id),
                )
                participant_row = cur.fetchone()

                if participant_row is None:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "call_participant_not_found",
                            "message": "You are not a participant in this call.",
                        },
                    )

                participant = row_to_call_participant(participant_row)

                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM call_participants
                    WHERE call_session_id = %s
                      AND status = 'joined'
                    """,
                    (call_session_id,),
                )
                joined_count = int(cur.fetchone()[0])

                if joined_count == 0:
                    cur.execute(
                        """
                        UPDATE call_sessions
                        SET status = 'ended',
                            ended_at = COALESCE(ended_at, NOW()),
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING id, organization_id, conversation_id, type, status,
                                  created_by_user_id, livekit_room_name, started_at,
                                  ended_at, created_at, updated_at
                        """,
                        (call_session_id,),
                    )
                    call = row_to_call_session(cur.fetchone())

                presence = upsert_presence(
                    conn,
                    call["organization_id"],
                    current_user.user_id,
                    "online",
                )

        return {
            "success": True,
            "call": call,
            "participant": participant,
            "presence": presence,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "call_leave_failed",
                "message": "Could not leave call.",
            },
        ) from exc


@router.post("/calls/{call_session_id}/decline")
def decline_call(
    call_session_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            call = get_call_session(conn, call_session_id)
            require_business_or_enterprise_organization(
                conn,
                call["organization_id"],
                current_user,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE call_participants
                    SET status = 'declined',
                        updated_at = NOW()
                    WHERE call_session_id = %s
                      AND user_id = %s
                    RETURNING id, call_session_id, organization_id, user_id,
                              status, invited_at, joined_at, left_at,
                              created_at, updated_at
                    """,
                    (call_session_id, current_user.user_id),
                )
                row = cur.fetchone()

                if row is None:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "call_participant_not_found",
                            "message": "You are not invited to this call.",
                        },
                    )

        return {
            "success": True,
            "call": call,
            "participant": row_to_call_participant(row),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "call_decline_failed",
                "message": "Could not decline call.",
            },
        ) from exc


@router.get("/organizations/{organization_id}/presence")
def list_presence(
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            require_business_or_enterprise_organization(
                conn,
                organization_id,
                current_user,
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        om.organization_id,
                        om.user_id,
                        COALESCE(mp.status, 'offline') AS status,
                        COALESCE(mp.last_seen_at, om.updated_at) AS last_seen_at,
                        COALESCE(mp.updated_at, om.updated_at) AS updated_at,
                        om.role AS organization_role
                    FROM organization_members om
                    LEFT JOIN member_presence mp
                      ON mp.organization_id = om.organization_id
                     AND mp.user_id = om.user_id
                    WHERE om.organization_id = %s
                      AND om.status = 'active'
                    ORDER BY
                        CASE COALESCE(mp.status, 'offline')
                            WHEN 'in_call' THEN 1
                            WHEN 'online' THEN 2
                            ELSE 3
                        END,
                        om.created_at ASC
                    """,
                    (organization_id,),
                )
                rows = cur.fetchall()

        return {
            "success": True,
            "presence": [
                {
                    "organization_id": row[0],
                    "user_id": row[1],
                    "status": row[2],
                    "last_seen_at": row[3],
                    "updated_at": row[4],
                    "organization_role": row[5],
                }
                for row in rows
            ],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "presence_load_failed",
                "message": "Could not load member presence.",
            },
        ) from exc


@router.post("/organizations/{organization_id}/presence")
def update_presence(
    payload: UpdatePresenceRequest,
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            require_business_or_enterprise_organization(
                conn,
                organization_id,
                current_user,
            )
            presence = upsert_presence(
                conn,
                organization_id,
                current_user.user_id,
                payload.status,
            )

        return {
            "success": True,
            "presence": presence,
            "user": user_public_payload(current_user),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_presence",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "presence_update_failed",
                "message": "Could not update member presence.",
            },
        ) from exc
