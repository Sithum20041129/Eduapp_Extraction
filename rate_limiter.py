import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Rate limit thresholds — override via environment variables.
# General / health endpoints: generous default.
# AI-invocation endpoints: tighter limits to cap GCP spend.
DEFAULT_RATE_LIMIT = os.getenv("RATE_LIMIT_DEFAULT", "60/minute")
EXTRACT_RATE_LIMIT  = os.getenv("RATE_LIMIT_EXTRACT",  "20/minute")
BATCH_RATE_LIMIT    = os.getenv("RATE_LIMIT_BATCH",    "10/minute")
PAPER_RATE_LIMIT    = os.getenv("RATE_LIMIT_PAPER",    "10/minute")

# Single shared Limiter instance — keyed by client IP.
# Import this object into main.py and attach it to app.state.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[DEFAULT_RATE_LIMIT],
)

# Re-export helpers so main.py only needs one import line.
__all__ = [
    "limiter",
    "_rate_limit_exceeded_handler",
    "RateLimitExceeded",
    "EXTRACT_RATE_LIMIT",
    "BATCH_RATE_LIMIT",
    "PAPER_RATE_LIMIT",
]
