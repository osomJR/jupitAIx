from __future__ import annotations

"""
Organization and team-management API.

Responsibilities:
- create organizations
- view organizations the authenticated user belongs to
- view an organization and its subscription
- invite members by email
- accept email invitations
- change member roles
- soft-remove members by setting status = 'removed'

Notes:
- Personal subscriptions remain user-level in user_subscriptions.
- Business/Enterprise subscriptions are organization-level in organization_subscriptions.
- Email invitations are stored as organization_members.user_id = 'invite:{email}' while pending.
- A pending invite grants no entitlement until the invited user accepts it and the row is converted
  to the authenticated Auth0 subject from current_user.user_id.
"""

from typing import Any, Literal
import re

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, field_validator

from backend.auth0_dependencies import AuthenticatedUser, get_current_user
from backend.database import get_db


router = APIRouter(prefix="/organizations", tags=["organizations"])

OrganizationRole = Literal["owner", "admin", "member"]
OrganizationMemberStatus = Literal["active", "invited", "removed"]

VALID_ROLES = {"owner", "admin", "member"}
VALID_MEMBER_STATUSES = {"active", "invited", "removed"}

INVITE_USER_ID_PREFIX = "invite:"
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class CreateOrganizationRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = (value or "").strip()
        if not normalized:
            raise ValueError("Organization name is required.")
        return normalized


class InviteMemberRequest(BaseModel):
    email: str
    role: OrganizationRole = "member"

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        return normalize_email(value)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        return normalize_role(value)


class UpdateMemberRequest(BaseModel):
    role: OrganizationRole | None = None
    status: OrganizationMemberStatus | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_role(value)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_member_status(value)


def normalize_email(value: str) -> str:
    normalized = (value or "").strip().lower()
    if not normalized or not EMAIL_PATTERN.match(normalized):
        raise ValueError("A valid email address is required.")
    return normalized


def normalize_user_id(user_id: str) -> str:
    normalized = (user_id or "").strip()
    if not normalized:
        raise ValueError("user_id is required.")
    return normalized


def normalize_role(role: str) -> OrganizationRole:
    normalized = (role or "").strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError("role must be one of: owner, admin, member.")
    return normalized  # type: ignore[return-value]


def normalize_member_status(status: str) -> OrganizationMemberStatus:
    normalized = (status or "").strip().lower()
    if normalized not in VALID_MEMBER_STATUSES:
        raise ValueError("status must be one of: active, invited, removed.")
    return normalized  # type: ignore[return-value]


def invited_user_id_for_email(email: str) -> str:
    return f"{INVITE_USER_ID_PREFIX}{normalize_email(email)}"


def current_user_email(current_user: AuthenticatedUser) -> str | None:
    email = current_user.claims.get("email")
    if not isinstance(email, str) or not email.strip():
        return None
    return normalize_email(email)


def resolve_member_identifier(member_user_id: str) -> str:
    """
    Resolve a route path value to the stored organization_members.user_id.

    Supports:
    - Auth0 subject values, e.g. auth0|abc123 or google-oauth2|123456
    - raw invited email values, e.g. user@example.com
    - stored invitation IDs, e.g. invite:user@example.com
    """

    raw = (member_user_id or "").strip()
    if not raw:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_member",
                "message": "member_user_id is required.",
            },
        )

    if raw.startswith(INVITE_USER_ID_PREFIX):
        return invited_user_id_for_email(raw.removeprefix(INVITE_USER_ID_PREFIX))

    if "@" in raw:
        return invited_user_id_for_email(raw)

    return raw


def user_public_payload(current_user: AuthenticatedUser) -> dict[str, Any]:
    return {
        "id": current_user.user_id,
        "name": current_user.claims.get("name"),
        "email": current_user.claims.get("email"),
        "picture": current_user.claims.get("picture"),
    }


def row_to_organization_summary(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "name": row[1],
        "owner_user_id": row[2],
        "member": {
            "role": row[3],
            "status": row[4],
            "joined_at": row[5],
        },
        "subscription": {
            "plan": row[6],
            "max_accounts": row[7],
            "status": row[8],
            "active_members": row[9],
        },
        "created_at": row[10],
        "updated_at": row[11],
    }


def row_to_member(row) -> dict[str, Any]:
    return {
        "id": row[0],
        "organization_id": row[1],
        "user_id": row[2],
        "role": row[3],
        "status": row[4],
        "invited_by_user_id": row[5],
        "invited_at": row[6],
        "joined_at": row[7],
        "created_at": row[8],
        "updated_at": row[9],
        "is_email_invitation": isinstance(row[2], str)
        and row[2].startswith(INVITE_USER_ID_PREFIX),
        "email": row[2].removeprefix(INVITE_USER_ID_PREFIX)
        if isinstance(row[2], str) and row[2].startswith(INVITE_USER_ID_PREFIX)
        else None,
    }


def row_to_subscription(row) -> dict[str, Any] | None:
    if row is None:
        return None

    return {
        "id": row[0],
        "organization_id": row[1],
        "plan": row[2],
        "max_accounts": row[3],
        "status": row[4],
        "provider": row[5],
        "provider_customer_id": row[6],
        "provider_subscription_id": row[7],
        "current_period_start": row[8],
        "current_period_end": row[9],
        "active_members": row[10],
        "created_at": row[11],
        "updated_at": row[12],
    }


def get_active_membership(conn, organization_id: int, user_id: str) -> dict[str, Any] | None:
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


def require_active_member(conn, organization_id: int, current_user: AuthenticatedUser) -> dict[str, Any]:
    membership = get_active_membership(conn, organization_id, current_user.user_id)
    if membership is None:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_access_denied",
                "message": "You are not an active member of this organization.",
            },
        )
    return membership


def require_admin_or_owner(conn, organization_id: int, current_user: AuthenticatedUser) -> dict[str, Any]:
    membership = require_active_member(conn, organization_id, current_user)
    if membership["role"] not in {"owner", "admin"}:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_admin_required",
                "message": "Only organization owners or admins can perform this action.",
            },
        )
    return membership


def require_owner(conn, organization_id: int, current_user: AuthenticatedUser) -> dict[str, Any]:
    membership = require_active_member(conn, organization_id, current_user)
    if membership["role"] != "owner":
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_owner_required",
                "message": "Only the organization owner can perform this action.",
            },
        )
    return membership


def get_member_for_update(conn, organization_id: int, member_user_id: str) -> dict[str, Any]:
    resolved_member_user_id = resolve_member_identifier(member_user_id)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, organization_id, user_id, role, status
            FROM organization_members
            WHERE organization_id = %s
              AND user_id = %s
            """,
            (organization_id, resolved_member_user_id),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "member_not_found",
                "message": "Organization member was not found.",
            },
        )

    return {
        "id": row[0],
        "organization_id": row[1],
        "user_id": row[2],
        "role": row[3],
        "status": row[4],
    }


def active_owner_count(conn, organization_id: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM organization_members
            WHERE organization_id = %s
              AND role = 'owner'
              AND status = 'active'
            """,
            (organization_id,),
        )
        row = cur.fetchone()

    return int(row[0] if row else 0)


def assert_can_modify_member(
    *,
    actor_membership: dict[str, Any],
    target_member: dict[str, Any],
    requested_role: str | None = None,
    requested_status: str | None = None,
) -> None:
    actor_role = actor_membership["role"]
    target_role = target_member["role"]

    # Admins can manage regular members, but not owners/admins.
    if actor_role == "admin" and target_role in {"owner", "admin"}:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_owner_required",
                "message": "Only an owner can modify owners or admins.",
            },
        )

    # Only owners can promote someone to admin/owner.
    if actor_role != "owner" and requested_role in {"owner", "admin"}:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "organization_owner_required",
                "message": "Only an owner can assign owner or admin roles.",
            },
        )

    # Invited email placeholders must be accepted by the invited user before becoming active.
    if (
        requested_status == "active"
        and isinstance(target_member["user_id"], str)
        and target_member["user_id"].startswith(INVITE_USER_ID_PREFIX)
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "invitation_acceptance_required",
                "message": "Email invitations must be accepted by the invited user before becoming active.",
            },
        )


@router.post("")
def create_organization(
    payload: CreateOrganizationRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO organizations (name, owner_user_id)
                    VALUES (%s, %s)
                    RETURNING id, name, owner_user_id, created_at, updated_at
                    """,
                    (payload.name, current_user.user_id),
                )
                organization = cur.fetchone()

                if organization is None:
                    raise RuntimeError("Failed to create organization.")

                organization_id = organization[0]

                cur.execute(
                    """
                    INSERT INTO organization_members (
                        organization_id,
                        user_id,
                        role,
                        status,
                        joined_at
                    )
                    VALUES (%s, %s, 'owner', 'active', NOW())
                    ON CONFLICT (organization_id, user_id) DO UPDATE SET
                        role = 'owner',
                        status = 'active',
                        joined_at = COALESCE(organization_members.joined_at, NOW()),
                        updated_at = NOW()
                    RETURNING id, organization_id, user_id, role, status,
                              invited_by_user_id, invited_at, joined_at,
                              created_at, updated_at
                    """,
                    (organization_id, current_user.user_id),
                )
                member = cur.fetchone()

        return {
            "success": True,
            "organization": {
                "id": organization[0],
                "name": organization[1],
                "owner_user_id": organization[2],
                "created_at": organization[3],
                "updated_at": organization[4],
            },
            "member": row_to_member(member),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_organization",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "organization_create_failed",
                "message": "Could not create organization.",
            },
        ) from exc


@router.get("/me")
def list_my_organizations(
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    email = current_user_email(current_user)
    invited_user_id = invited_user_id_for_email(email) if email else None

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        o.id,
                        o.name,
                        o.owner_user_id,
                        om.role,
                        om.status,
                        om.joined_at,
                        os.plan,
                        os.max_accounts,
                        os.status AS subscription_status,
                        (
                            SELECT COUNT(*)
                            FROM organization_members active_om
                            WHERE active_om.organization_id = o.id
                              AND active_om.status = 'active'
                        ) AS active_members,
                        o.created_at,
                        o.updated_at
                    FROM organization_members om
                    JOIN organizations o
                      ON o.id = om.organization_id
                    LEFT JOIN organization_subscriptions os
                      ON os.organization_id = o.id
                    WHERE om.user_id = %s
                      AND om.status = 'active'
                    ORDER BY o.updated_at DESC, o.id DESC
                    """,
                    (current_user.user_id,),
                )
                active_rows = cur.fetchall()

                invite_rows = []
                if invited_user_id is not None:
                    cur.execute(
                        """
                        SELECT
                            o.id,
                            o.name,
                            o.owner_user_id,
                            om.role,
                            om.status,
                            om.joined_at,
                            os.plan,
                            os.max_accounts,
                            os.status AS subscription_status,
                            (
                                SELECT COUNT(*)
                                FROM organization_members active_om
                                WHERE active_om.organization_id = o.id
                                  AND active_om.status = 'active'
                            ) AS active_members,
                            o.created_at,
                            o.updated_at
                        FROM organization_members om
                        JOIN organizations o
                          ON o.id = om.organization_id
                        LEFT JOIN organization_subscriptions os
                          ON os.organization_id = o.id
                        WHERE om.user_id = %s
                          AND om.status = 'invited'
                        ORDER BY om.invited_at DESC NULLS LAST, o.id DESC
                        """,
                        (invited_user_id,),
                    )
                    invite_rows = cur.fetchall()

        return {
            "success": True,
            "user": user_public_payload(current_user),
            "organizations": [row_to_organization_summary(row) for row in active_rows],
            "invitations": [row_to_organization_summary(row) for row in invite_rows],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "organizations_load_failed",
                "message": "Could not load organizations.",
            },
        ) from exc


@router.get("/{organization_id}")
def get_organization(
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            require_active_member(conn, organization_id, current_user)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, owner_user_id, created_at, updated_at
                    FROM organizations
                    WHERE id = %s
                    """,
                    (organization_id,),
                )
                organization = cur.fetchone()

                if organization is None:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "organization_not_found",
                            "message": "Organization was not found.",
                        },
                    )

                cur.execute(
                    """
                    SELECT id, organization_id, user_id, role, status,
                           invited_by_user_id, invited_at, joined_at,
                           created_at, updated_at
                    FROM organization_members
                    WHERE organization_id = %s
                      AND status <> 'removed'
                    ORDER BY
                        CASE role
                            WHEN 'owner' THEN 1
                            WHEN 'admin' THEN 2
                            ELSE 3
                        END,
                        created_at ASC,
                        id ASC
                    """,
                    (organization_id,),
                )
                members = cur.fetchall()

        return {
            "success": True,
            "organization": {
                "id": organization[0],
                "name": organization[1],
                "owner_user_id": organization[2],
                "created_at": organization[3],
                "updated_at": organization[4],
            },
            "members": [row_to_member(row) for row in members],
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "organization_load_failed",
                "message": "Could not load organization.",
            },
        ) from exc


@router.post("/{organization_id}/members")
def invite_member_by_email(
    payload: InviteMemberRequest,
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    invited_user_id = invited_user_id_for_email(payload.email)

    try:
        with get_db() as conn:
            actor_membership = require_admin_or_owner(conn, organization_id, current_user)

            if actor_membership["role"] != "owner" and payload.role in {"owner", "admin"}:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "organization_owner_required",
                        "message": "Only an owner can invite admins or owners.",
                    },
                )

            if payload.email == current_user_email(current_user):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "self_invite_not_allowed",
                        "message": "You are already represented by your authenticated account.",
                    },
                )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM organizations
                    WHERE id = %s
                    """,
                    (organization_id,),
                )
                if cur.fetchone() is None:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "organization_not_found",
                            "message": "Organization was not found.",
                        },
                    )

                cur.execute(
                    """
                    INSERT INTO organization_members (
                        organization_id,
                        user_id,
                        role,
                        status,
                        invited_by_user_id,
                        invited_at
                    )
                    VALUES (%s, %s, %s, 'invited', %s, NOW())
                    ON CONFLICT (organization_id, user_id) DO UPDATE SET
                        role = EXCLUDED.role,
                        status = 'invited',
                        invited_by_user_id = EXCLUDED.invited_by_user_id,
                        invited_at = NOW(),
                        updated_at = NOW()
                    RETURNING id, organization_id, user_id, role, status,
                              invited_by_user_id, invited_at, joined_at,
                              created_at, updated_at
                    """,
                    (
                        organization_id,
                        invited_user_id,
                        payload.role,
                        current_user.user_id,
                    ),
                )
                member = cur.fetchone()

        return {
            "success": True,
            "invitation": row_to_member(member),
            "message": "Invitation recorded. Send an email notification from your mail provider or notification worker.",
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_invitation",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "member_invite_failed",
                "message": "Could not invite organization member.",
            },
        ) from exc


@router.post("/{organization_id}/invitations/accept")
def accept_email_invitation(
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    email = current_user_email(current_user)
    if email is None:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "email_required",
                "message": "Your authenticated profile does not include an email address.",
            },
        )

    invited_user_id = invited_user_id_for_email(email)

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, role
                    FROM organization_members
                    WHERE organization_id = %s
                      AND user_id = %s
                      AND status = 'invited'
                    """,
                    (organization_id, invited_user_id),
                )
                invite = cur.fetchone()

                if invite is None:
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "invitation_not_found",
                            "message": "No pending invitation was found for your email address.",
                        },
                    )

                invite_role = invite[1]

                cur.execute(
                    """
                    SELECT id
                    FROM organization_members
                    WHERE organization_id = %s
                      AND user_id = %s
                    """,
                    (organization_id, current_user.user_id),
                )
                existing_membership = cur.fetchone()

                if existing_membership is not None:
                    cur.execute(
                        """
                        UPDATE organization_members
                        SET role = %s,
                            status = 'active',
                            joined_at = COALESCE(joined_at, NOW()),
                            updated_at = NOW()
                        WHERE organization_id = %s
                          AND user_id = %s
                        RETURNING id, organization_id, user_id, role, status,
                                  invited_by_user_id, invited_at, joined_at,
                                  created_at, updated_at
                        """,
                        (invite_role, organization_id, current_user.user_id),
                    )
                    accepted_member = cur.fetchone()

                    cur.execute(
                        """
                        UPDATE organization_members
                        SET status = 'removed',
                            updated_at = NOW()
                        WHERE organization_id = %s
                          AND user_id = %s
                        """,
                        (organization_id, invited_user_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE organization_members
                        SET user_id = %s,
                            status = 'active',
                            joined_at = NOW(),
                            updated_at = NOW()
                        WHERE organization_id = %s
                          AND user_id = %s
                          AND status = 'invited'
                        RETURNING id, organization_id, user_id, role, status,
                                  invited_by_user_id, invited_at, joined_at,
                                  created_at, updated_at
                        """,
                        (current_user.user_id, organization_id, invited_user_id),
                    )
                    accepted_member = cur.fetchone()

                if accepted_member is None:
                    raise RuntimeError("Failed to accept invitation.")

        return {
            "success": True,
            "member": row_to_member(accepted_member),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "invitation_accept_failed",
                "message": "Could not accept organization invitation.",
            },
        ) from exc


@router.patch("/{organization_id}/members/{member_user_id}")
def update_member(
    payload: UpdateMemberRequest,
    organization_id: int = Path(..., ge=1),
    member_user_id: str = Path(..., min_length=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    if payload.role is None and payload.status is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_member_update",
                "message": "Provide at least one field to update: role or status.",
            },
        )

    try:
        with get_db() as conn:
            actor_membership = require_admin_or_owner(conn, organization_id, current_user)
            target_member = get_member_for_update(conn, organization_id, member_user_id)

            assert_can_modify_member(
                actor_membership=actor_membership,
                target_member=target_member,
                requested_role=payload.role,
                requested_status=payload.status,
            )

            if (
                target_member["role"] == "owner"
                and payload.status == "removed"
                and active_owner_count(conn, organization_id) <= 1
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "last_owner_required",
                        "message": "An organization must keep at least one active owner.",
                    },
                )

            if (
                target_member["role"] == "owner"
                and payload.role is not None
                and payload.role != "owner"
                and active_owner_count(conn, organization_id) <= 1
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "last_owner_required",
                        "message": "Assign another owner before changing the last owner's role.",
                    },
                )

            update_role = payload.role if payload.role is not None else target_member["role"]
            update_status = (
                payload.status if payload.status is not None else target_member["status"]
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE organization_members
                    SET role = %s,
                        status = %s,
                        joined_at = CASE
                            WHEN %s = 'active' AND joined_at IS NULL THEN NOW()
                            ELSE joined_at
                        END,
                        updated_at = NOW()
                    WHERE organization_id = %s
                      AND user_id = %s
                    RETURNING id, organization_id, user_id, role, status,
                              invited_by_user_id, invited_at, joined_at,
                              created_at, updated_at
                    """,
                    (
                        update_role,
                        update_status,
                        update_status,
                        organization_id,
                        target_member["user_id"],
                    ),
                )
                member = cur.fetchone()

                if member is None:
                    raise RuntimeError("Failed to update member.")

        return {
            "success": True,
            "member": row_to_member(member),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "member_update_failed",
                "message": "Could not update organization member.",
            },
        ) from exc


@router.delete("/{organization_id}/members/{member_user_id}")
def remove_member(
    organization_id: int = Path(..., ge=1),
    member_user_id: str = Path(..., min_length=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            actor_membership = require_admin_or_owner(conn, organization_id, current_user)
            target_member = get_member_for_update(conn, organization_id, member_user_id)

            assert_can_modify_member(
                actor_membership=actor_membership,
                target_member=target_member,
                requested_status="removed",
            )

            if (
                target_member["role"] == "owner"
                and active_owner_count(conn, organization_id) <= 1
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "last_owner_required",
                        "message": "An organization must keep at least one active owner.",
                    },
                )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE organization_members
                    SET status = 'removed',
                        updated_at = NOW()
                    WHERE organization_id = %s
                      AND user_id = %s
                    RETURNING id, organization_id, user_id, role, status,
                              invited_by_user_id, invited_at, joined_at,
                              created_at, updated_at
                    """,
                    (organization_id, target_member["user_id"]),
                )
                member = cur.fetchone()

                if member is None:
                    raise RuntimeError("Failed to remove member.")

        return {
            "success": True,
            "member": row_to_member(member),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "member_remove_failed",
                "message": "Could not remove organization member.",
            },
        ) from exc


@router.get("/{organization_id}/subscription")
def get_organization_subscription(
    organization_id: int = Path(..., ge=1),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        with get_db() as conn:
            require_active_member(conn, organization_id, current_user)

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        os.id,
                        os.organization_id,
                        os.plan,
                        os.max_accounts,
                        os.status,
                        os.provider,
                        os.provider_customer_id,
                        os.provider_subscription_id,
                        os.current_period_start,
                        os.current_period_end,
                        (
                            SELECT COUNT(*)
                            FROM organization_members om
                            WHERE om.organization_id = os.organization_id
                              AND om.status = 'active'
                        ) AS active_members,
                        os.created_at,
                        os.updated_at
                    FROM organization_subscriptions os
                    WHERE os.organization_id = %s
                    """,
                    (organization_id,),
                )
                subscription = cur.fetchone()

        return {
            "success": True,
            "subscription": row_to_subscription(subscription),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "subscription_load_failed",
                "message": "Could not load organization subscription.",
            },
        ) from exc
