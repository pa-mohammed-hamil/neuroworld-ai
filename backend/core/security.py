
"""
NeuroWorld AI
backend/app/core/security.py

Security utilities for the NeuroWorld AI platform.

Features:
- Argon2 password hashing
- JWT access token generation
- Token validation through existing dependencies.py
- Role-based access control
- Permission-based access control
- In-memory development rate limiting
- Secure upload validation
- Security response headers
- Restricted file storage
- Human approval for sensitive actions
- Security audit events

Install:
    pip install fastapi PyJWT argon2-cffi python-multipart

Important:
- Configure SECRET_KEY in .env (at least 32 characters).
- In-memory rate limiting is not distributed.
- Use persistent identity and approval storage in production.
- File extension and signature checks do not replace
  malware scanning or content validation.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import threading
import time
import uuid

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import jwt

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from fastapi import (
    Depends,
    HTTPException,
    Request,
    UploadFile,
)
from starlette.datastructures import Headers

from ..config import DATA_DIR, Settings, get_settings
from ..dependencies import (
    CurrentUser,
    decode_access_token,
    get_current_user,
)
from .exceptions import (
    AccessDeniedError,
    FileNotAllowedError,
    HumanApprovalRequiredError,
    InvalidRequestError,
    RateLimitExceededError,
    SecurityViolationError,
)
from .logging import get_logger, log_audit


logger = get_logger("security")


# ============================================================
# CONSTANTS
# ============================================================

JWT_ISSUER = "neuroworld-ai"
JWT_AUDIENCE = "neuroworld-api"

PASSWORD_MIN_LENGTH = 12
PASSWORD_MAX_LENGTH = 1024

UPLOAD_ROOT = DATA_DIR / "uploads"

ALLOWED_ROLES = frozenset({
    "admin",
    "researcher",
    "ml_engineer",
    "operator",
    "reviewer",
    "viewer",
})

ROLE_PERMISSIONS = {
    "admin": frozenset({
        "world:read",
        "world:create",
        "world:update",
        "world:delete",
        "model:read",
        "model:train",
        "model:deploy",
        "simulation:read",
        "simulation:execute",
        "planning:execute",
        "imagination:execute",
        "approval:review",
        "audit:read",
        "user:manage",
    }),
    "researcher": frozenset({
        "world:read",
        "world:create",
        "world:update",
        "model:read",
        "model:train",
        "simulation:read",
        "simulation:execute",
        "planning:execute",
        "imagination:execute",
    }),
    "ml_engineer": frozenset({
        "world:read",
        "model:read",
        "model:train",
        "model:deploy",
        "simulation:read",
        "simulation:execute",
    }),
    "operator": frozenset({
        "world:read",
        "model:read",
        "simulation:read",
        "simulation:execute",
    }),
    "reviewer": frozenset({
        "world:read",
        "model:read",
        "simulation:read",
        "approval:review",
        "audit:read",
    }),
    "viewer": frozenset({
        "world:read",
        "model:read",
        "simulation:read",
    }),
}


# ============================================================
# PASSWORD SECURITY
# ============================================================

_password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def validate_password(password: str) -> None:
    if not isinstance(password, str):
        raise InvalidRequestError(
            "Password must be a string"
        )

    if not (
        PASSWORD_MIN_LENGTH
        <= len(password)
        <= PASSWORD_MAX_LENGTH
    ):
        raise InvalidRequestError(
            "Password length must be between "
            f"{PASSWORD_MIN_LENGTH} and "
            f"{PASSWORD_MAX_LENGTH} characters"
        )

    if "\x00" in password:
        raise InvalidRequestError(
            "Password contains invalid characters"
        )


def hash_password(password: str) -> str:
    validate_password(password)
    return _password_hasher.hash(password)


def verify_password(
    password: str,
    password_hash: str,
) -> bool:
    if not isinstance(password, str):
        return False

    try:
        return _password_hasher.verify(
            password_hash,
            password,
        )
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
        ValueError,
        TypeError,
    ):
        return False


def password_needs_rehash(
    password_hash: str,
) -> bool:
    return _password_hasher.check_needs_rehash(
        password_hash
    )


# ============================================================
# JWT TOKENS
# ============================================================

def get_jwt_secret(
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()

    if settings.secret_key is None:
        raise RuntimeError(
            "SECRET_KEY is not configured"
        )

    secret = settings.secret_key.get_secret_value()

    if len(secret) < 32:
        raise RuntimeError(
            "SECRET_KEY must contain at least 32 characters"
        )

    return secret


def create_access_token(
    *,
    user_id: str,
    roles: list[str] | None = None,
    permissions: list[str] | None = None,
    email: str | None = None,
    expires_minutes: int | None = None,
    settings: Settings | None = None,
) -> str:
    """
    Create a signed JWT compatible with
    dependencies.decode_access_token().

    Roles and permissions must come from a trusted
    identity database, never directly from user input.
    """

    settings = settings or get_settings()

    if not user_id or not user_id.strip():
        raise ValueError("user_id is required")

    selected_roles = roles or []

    if not set(selected_roles).issubset(ALLOWED_ROLES):
        raise ValueError("Unknown role")

    if expires_minutes is None:
        expires_minutes = (
            settings.security.access_token_expire_minutes
        )

    if not 1 <= expires_minutes <= 1440:
        raise ValueError(
            "Token lifetime must be between 1 and 1440 minutes"
        )

    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(
            minutes=expires_minutes
        ),
        "iss": JWT_ISSUER,
        "aud": JWT_AUDIENCE,
        "token_type": "access",
        "jti": uuid.uuid4().hex,
        "roles": selected_roles,
        "permissions": permissions or [],
    }

    if email:
        payload["email"] = email

    return jwt.encode(
        payload,
        get_jwt_secret(settings),
        algorithm=settings.jwt_algorithm,
    )


def verify_access_token(token: str) -> CurrentUser:
    return decode_access_token(token)


# ============================================================
# ROLE AND PERMISSION SECURITY
# ============================================================

def get_role_permissions(
    roles: set[str] | frozenset[str],
) -> frozenset[str]:
    permissions = set()

    for role in roles:
        permissions.update(
            ROLE_PERMISSIONS.get(role, ())
        )

    return frozenset(permissions)


def has_permission(
    user: CurrentUser,
    permission: str,
) -> bool:
    """
    Compare requested permission against trusted claims.

    The token's roles and permissions must have been
    issued from authoritative server-side user records.
    """

    if permission in user.permissions:
        return True

    return permission in get_role_permissions(
        user.roles
    )


def require_security_permission(permission: str):
    async def dependency(
        user: CurrentUser = Depends(get_current_user),
    ) -> CurrentUser:
        if not has_permission(user, permission):
            log_audit(
                action="permission.check",
                resource_type="permission",
                resource_id=permission,
                actor_id=user.user_id,
                outcome="denied",
            )

            raise AccessDeniedError(
                "Permission denied"
            )

        return user

    return dependency


# ============================================================
# RATE LIMITING
# ============================================================

class InMemoryRateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Development/single-process only.

    Production systems should use a shared Redis-backed
    limiter with atomic operations.
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
    ):
        if max_requests < 1 or window_seconds < 1:
            raise ValueError(
                "Rate limit settings must be positive"
            )

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> bool:
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets[identity]

            cutoff = now - self.window_seconds

            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                return False

            bucket.append(now)
            return True


default_rate_limiter = InMemoryRateLimiter(
    max_requests=60,
    window_seconds=60,
)


def require_rate_limit(
    request: Request,
) -> None:
    """
    Limit by client IP.

    For authenticated routes, prefer user ID.
    Do not trust X-Forwarded-For unless the deployment
    has explicitly configured trusted proxy handling.
    """

    identity = (
        request.client.host
        if request.client
        else "unknown"
    )

    if not default_rate_limiter.check(identity):
        raise RateLimitExceededError()


# ============================================================
# SECURE UPLOADS
# ============================================================

FILE_SIGNATURES = {
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".webp": (b"RIFF",),
    ".mp4": (),
    ".webm": (b"\x1a\x45\xdf\xa3",),
    ".json": (),
    ".csv": (),
}


def allowed_upload_extensions() -> set[str]:
    settings = get_settings()

    return {
        (
            extension.lower()
            if extension.startswith(".")
            else "." + extension.lower()
        )
        for extension in (
            settings.security.allowed_upload_extensions
        )
    }


def validate_upload_filename(
    filename: str | None,
) -> str:
    if not filename:
        raise FileNotAllowedError(
            "Uploaded file must have a filename"
        )

    # Do not allow path components from the client.
    normalized = filename.replace("\\", "/")

    if (
        "/" in normalized
        or "\x00" in normalized
        or len(normalized) > 255
    ):
        raise FileNotAllowedError(
            "Invalid upload filename"
        )

    suffix = Path(normalized).suffix.lower()

    if suffix not in allowed_upload_extensions():
        raise FileNotAllowedError(
            "Unsupported upload file type"
        )

    return suffix


def validate_file_signature(
    suffix: str,
    header: bytes,
) -> bool:
    """
    Basic signature validation.

    For MP4, inspect the ISO BMFF 'ftyp' box.
    For WEBP, verify both RIFF and WEBP markers.

    JSON and CSV need separate parser validation.
    """

    if suffix in {".jpg", ".jpeg", ".png", ".webm"}:
        return any(
            header.startswith(signature)
            for signature in FILE_SIGNATURES[suffix]
        )

    if suffix == ".webp":
        return (
            header.startswith(b"RIFF")
            and header[8:12] == b"WEBP"
        )

    if suffix == ".mp4":
        return len(header) >= 12 and header[4:8] == b"ftyp"

    if suffix in {".json", ".csv"}:
        return True

    return False


@dataclass(frozen=True)
class SavedUpload:
    upload_id: str
    original_filename: str
    stored_path: Path
    size_bytes: int
    sha256: str
    extension: str


async def save_upload_securely(
    upload: UploadFile,
    *,
    owner_id: str,
    max_size_mb: int | None = None,
) -> SavedUpload:
    """
    Save an upload under a generated filename.

    The destination is private local storage, not
    the publicly served frontend directory.

    Add content scanning, parsing, authorization,
    retention and quota controls before production.
    """

    if not owner_id or not owner_id.strip():
        raise SecurityViolationError(
            "Authenticated upload owner required"
        )

    settings = get_settings()

    suffix = validate_upload_filename(
        upload.filename
    )

    if max_size_mb is None:
        max_size_mb = (
            settings.security.max_upload_size_mb
        )

    if max_size_mb <= 0:
        raise ValueError(
            "Maximum upload size must be positive"
        )

    max_bytes = max_size_mb * 1024 * 1024

    upload_id = uuid.uuid4().hex

    # Hash owner ID to avoid putting personal identifiers
    # directly into filesystem paths.
    owner_hash = hashlib.sha256(
        owner_id.encode("utf-8")
    ).hexdigest()[:24]

    destination_dir = (
        UPLOAD_ROOT / owner_hash
    ).resolve()

    root = UPLOAD_ROOT.resolve()

    if not destination_dir.is_relative_to(root):
        raise SecurityViolationError(
            "Invalid upload destination"
        )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
        mode=0o700,
    )

    destination = destination_dir / (
        upload_id + suffix
    )

    digest = hashlib.sha256()
    total_size = 0
    first_chunk = True

    try:
        # Exclusive create prevents accidental overwrites.
        with destination.open("xb") as output:
            while True:
                chunk = await upload.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                if first_chunk:
                    first_chunk = False

                    if not validate_file_signature(
                        suffix,
                        chunk[:32],
                    ):
                        raise FileNotAllowedError(
                            "File content does not match extension"
                        )

                total_size += len(chunk)

                if total_size > max_bytes:
                    raise FileNotAllowedError(
                        "Uploaded file exceeds size limit"
                    )

                digest.update(chunk)
                output.write(chunk)

        if total_size == 0:
            raise FileNotAllowedError(
                "Empty uploads are not allowed"
            )

        try:
            destination.chmod(0o600)
        except OSError:
            logger.warning(
                "Could not set restrictive file permissions"
            )

    except Exception:
        destination.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()

    log_audit(
        action="file.upload",
        resource_type="upload",
        resource_id=upload_id,
        actor_id=owner_id,
        outcome="success",
    )

    return SavedUpload(
        upload_id=upload_id,
        original_filename=upload.filename or "",
        stored_path=destination,
        size_bytes=total_size,
        sha256=digest.hexdigest(),
        extension=suffix,
    )


# ============================================================
# SECURITY HEADERS
# ============================================================

class SecurityHeadersMiddleware:
    """
    ASGI middleware for common HTTP security headers.

    Adjust Content-Security-Policy for actual frontend
    requirements before deploying.
    """

    def __init__(
        self,
        app,
        *,
        enable_hsts: bool = False,
    ):
        self.app = app
        self.enable_hsts = enable_hsts

    async def __call__(
        self,
        scope,
        receive,
        send,
    ):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(
                    message.get("headers", [])
                )

                headers.extend([
                    (
                        b"x-content-type-options",
                        b"nosniff",
                    ),
                    (
                        b"x-frame-options",
                        b"DENY",
                    ),
                    (
                        b"referrer-policy",
                        b"strict-origin-when-cross-origin",
                    ),
                    (
                        b"permissions-policy",
                        b"camera=(), microphone=(), geolocation=()",
                    ),
                    (
                        b"content-security-policy",
                        (
                            b"default-src 'self'; "
                            b"base-uri 'self'; "
                            b"object-src 'none'; "
                            b"frame-ancestors 'none'; "
                            b"form-action 'self'; "
                            b"script-src 'self'; "
                            b"style-src 'self'; "
                            b"img-src 'self' data:; "
                            b"connect-src 'self' ws: wss:"
                        ),
                    ),
                ])

                if self.enable_hsts:
                    headers.append((
                        b"strict-transport-security",
                        b"max-age=31536000",
                    ))

                message = {
                    **message,
                    "headers": headers,
                }

            await send(message)

        await self.app(
            scope,
            receive,
            send_wrapper,
        )


# ============================================================
# HUMAN APPROVAL WORKFLOW
# ============================================================

@dataclass
class ApprovalRequest:
    approval_id: str
    action: str
    resource_id: str
    requested_by: str
    created_at: datetime
    expires_at: datetime
    status: Literal[
        "pending",
        "approved",
        "rejected",
        "consumed",
    ] = "pending"
    reviewed_by: str | None = None


class ApprovalManager:
    """
    Development-only in-memory approval manager.

    Approval is bound to an exact action and resource.

    Production systems must persist requests and
    decisions transactionally, enforce object-level
    permissions, and prevent replay across workers.
    """

    def __init__(self):
        self._requests: dict[str, ApprovalRequest] = {}
        self._lock = threading.RLock()

    def create(
        self,
        *,
        action: str,
        resource_id: str,
        requested_by: str,
        ttl_minutes: int = 15,
    ) -> ApprovalRequest:

        if not all([
            action,
            resource_id,
            requested_by,
        ]):
            raise ValueError(
                "Action, resource and requester are required"
            )

        if not 1 <= ttl_minutes <= 1440:
            raise ValueError(
                "Invalid approval lifetime"
            )

        now = datetime.now(timezone.utc)

        approval = ApprovalRequest(
            approval_id=uuid.uuid4().hex,
            action=action,
            resource_id=resource_id,
            requested_by=requested_by,
            created_at=now,
            expires_at=now + timedelta(
                minutes=ttl_minutes
            ),
        )

        with self._lock:
            self._requests[approval.approval_id] = approval

        log_audit(
            action="approval.request",
            resource_type=action,
            resource_id=resource_id,
            actor_id=requested_by,
        )

        return approval

    def review(
        self,
        *,
        approval_id: str,
        reviewer: CurrentUser,
        approve: bool,
    ) -> ApprovalRequest:

        if not has_permission(
            reviewer,
            "approval:review",
        ):
            raise AccessDeniedError(
                "Approval review permission required"
            )

        with self._lock:
            approval = self._requests.get(
                approval_id
            )

            if approval is None:
                raise InvalidRequestError(
                    "Approval request not found"
                )

            if approval.status != "pending":
                raise InvalidRequestError(
                    "Approval request is no longer pending"
                )

            if datetime.now(timezone.utc) >= approval.expires_at:
                raise InvalidRequestError(
                    "Approval request expired"
                )

            if reviewer.user_id == approval.requested_by:
                raise AccessDeniedError(
                    "Requester cannot approve own action"
                )

            approval.status = (
                "approved"
                if approve
                else "rejected"
            )

            approval.reviewed_by = reviewer.user_id

        log_audit(
            action="approval.review",
            resource_type=approval.action,
            resource_id=approval.resource_id,
            actor_id=reviewer.user_id,
            outcome=(
                "success"
                if approve
                else "denied"
            ),
        )

        return approval

    def consume(
        self,
        *,
        approval_id: str,
        action: str,
        resource_id: str,
        requested_by: str,
    ) -> bool:
        """
        Consume a single-use approval.

        Call immediately before a protected operation.
        """

        with self._lock:
            approval = self._requests.get(
                approval_id
            )

            if approval is None:
                raise HumanApprovalRequiredError()

            if approval.status != "approved":
                raise HumanApprovalRequiredError()

            if datetime.now(timezone.utc) >= approval.expires_at:
                raise HumanApprovalRequiredError(
                    "Approval expired"
                )

            if (
                approval.action != action
                or approval.resource_id != resource_id
                or approval.requested_by != requested_by
            ):
                raise SecurityViolationError(
                    "Approval does not match requested action"
                )

            approval.status = "consumed"

        return True


approval_manager = ApprovalManager()


# ============================================================
# SECURITY DIAGNOSTICS
# ============================================================

def security_status() -> dict[str, Any]:
    settings = get_settings()

    return {
        "authentication_enabled": (
            settings.security.enable_authentication
        ),
        "jwt_configured": (
            settings.secret_key is not None
            and len(
                settings.secret_key.get_secret_value()
            ) >= 32
        ),
        "rate_limiting_enabled": (
            settings.security.enable_rate_limiting
        ),
        "upload_limit_mb": (
            settings.security.max_upload_size_mb
        ),
        "allowed_upload_extensions": sorted(
            allowed_upload_extensions()
        ),
        "human_approval_required": (
            settings.security
            .require_human_approval_for_external_actions
        ),
        "approval_storage": "in_memory_development_only",
        "rate_limit_storage": "in_memory_development_only",
    }


# ============================================================
# SELF TEST
# ============================================================

def main():
    print("=" * 60)
    print("NEUROWORLD AI — SECURITY")
    print("=" * 60)

    password = "ExampleStrongPassword2026!"

    encoded = hash_password(password)

    assert verify_password(password, encoded)
    assert not verify_password(
        "incorrect-password",
        encoded,
    )

    print("Password hashing: PASS")

    print("Allowed roles:", sorted(ALLOWED_ROLES))

    print("Upload extensions:", sorted(
        allowed_upload_extensions()
    ))

    print("Security configuration:")
    print(security_status())

    print("=" * 60)


if __name__ == "__main__":
    main()
