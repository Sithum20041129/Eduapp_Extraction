"""
API-key authentication for the AI service.

All AI endpoints depend on :func:`require_api_key`, which checks the ``X-API-Key``
request header against the allowlist in :mod:`config`. Each AI call triggers a
paid Vertex AI invocation, so unauthenticated access is both a data-exposure and
a billing-abuse risk (the audit's HIGH auth finding).

If no keys are configured the dependency allows the request (local-dev fallback)
but a warning is logged once at import time so this is never silently shipped to
production.
"""
from __future__ import annotations

import logging

from fastapi import Header, HTTPException, status

import config

logger = logging.getLogger("ai_service.auth")

if not config.AUTH_ENABLED:
    logger.warning(
        "SERVICE_API_KEY is not set — API authentication is DISABLED. "
        "Set SERVICE_API_KEY (comma-separated for multiple) before deploying."
    )


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency: reject requests without a valid X-API-Key header."""
    if not config.AUTH_ENABLED:
        return
    if not x_api_key or x_api_key not in config.SERVICE_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
