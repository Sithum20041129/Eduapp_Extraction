"""
Centralised configuration for the AI Extraction/Marking service.

Single source of truth for all environment-derived settings. ``load_dotenv()``
is called exactly once here; every other module imports from this module rather
than calling ``os.getenv()`` / ``load_dotenv()`` itself. This removes the
configuration drift the audit flagged (three files, three credential patterns).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env exactly once for the whole process.
load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _csv_env(name: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


# ── GCP / Vertex AI ───────────────────────────────────────────────────────────
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "asia-south1")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-1.5-flash")
MODEL_MAX_RETRIES = _int_env("MODEL_MAX_RETRIES", 3)

# ── Authentication ────────────────────────────────────────────────────────────
# Comma-separated accepted API keys (header: X-API-Key). If none are set, auth
# is DISABLED so local dev still works — but a warning is logged at startup and
# production deployments MUST set SERVICE_API_KEY.
SERVICE_API_KEYS = set(_csv_env("SERVICE_API_KEY"))
AUTH_ENABLED = bool(SERVICE_API_KEYS)

# ── Uploads ───────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = _int_env("MAX_UPLOAD_BYTES", 10 * 1024 * 1024)  # 10 MB default
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_PAPER_TYPES = {"application/pdf"} | ALLOWED_IMAGE_TYPES

# ── CORS ──────────────────────────────────────────────────────────────────────
# Empty = no cross-origin access (same-origin only). Never use "*".
CORS_ALLOWED_ORIGINS = _csv_env("CORS_ALLOWED_ORIGINS")

# ── Input validation ──────────────────────────────────────────────────────────
ALLOWED_SUBJECTS = {"Mathematics", "Physics", "Chemistry", "Biology"}
ALLOWED_DOC_TYPES = {"question", "modelanswer", "handwritten"}
MAX_FORM_FIELD_LEN = _int_env("MAX_FORM_FIELD_LEN", 200)
MAX_BATCH_IMAGES = _int_env("MAX_BATCH_IMAGES", 20)

# ── Token usage logging ───────────────────────────────────────────────────────
# Directory for CSV usage logs. Configurable so it can point outside the app
# root (web-exposed paths were flagged) or be swapped for a remote sink later.
TOKEN_LOG_DIR = os.getenv("TOKEN_LOG_DIR", "")  # "" => default: ./logs beside token_logger.py
