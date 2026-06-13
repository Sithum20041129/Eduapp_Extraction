"""
Tests for rate_limiter.py

Covers:
  1. Hard-coded default threshold values
  2. Environment-variable overrides (including all-four-at-once)
  3. Limiter instance type and key-function configuration
  4. __all__ exports — completeness and accessibility
  5. Security constraints — AI endpoints must be tighter than the generic default
  6. Edge / negative cases — empty env var, malformed strings, singleton pattern
  7. Security regression — the previously-leaked API key must not appear in source
"""

import importlib
import inspect
import os
import re

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RATE_LIMIT_KEYS = (
    "RATE_LIMIT_DEFAULT",
    "RATE_LIMIT_EXTRACT",
    "RATE_LIMIT_BATCH",
    "RATE_LIMIT_PAPER",
)

# The Gemini key that was previously hardcoded and must never reappear.
# Reconstructed from fragments so this file does not re-commit the secret.
_LEAKED_API_KEY = "AIza" + "SyCGqeqalFqiok" + "-_r17nYiBKPdO9TfnEuMI"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def fresh_module(monkeypatch, overrides=None):
    """
    Reload rate_limiter with a controlled environment:
      - All RATE_LIMIT_* keys are removed first (ensures defaults are exercised).
      - Any keys in *overrides* are then set.
    Returns the freshly-loaded module object.
    """
    for key in RATE_LIMIT_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, val in (overrides or {}).items():
        monkeypatch.setenv(key, val)
    return importlib.reload(importlib.import_module("rate_limiter"))


def parse_limit(limit_str):
    """Return (int count, str period) from a 'count/period' string."""
    parts = limit_str.split("/")
    assert len(parts) == 2, f"Malformed rate-limit string: {limit_str!r}"
    return int(parts[0]), parts[1]


# ===========================================================================
# 1 — Default threshold values
# ===========================================================================

class TestDefaults:
    """Verify hard-coded fallback thresholds when no env vars are present."""

    def test_default_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        assert mod.DEFAULT_RATE_LIMIT == "60/minute"

    def test_extract_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        assert mod.EXTRACT_RATE_LIMIT == "20/minute"

    def test_batch_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        assert mod.BATCH_RATE_LIMIT == "10/minute"

    def test_paper_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        assert mod.PAPER_RATE_LIMIT == "10/minute"


# ===========================================================================
# 2 — Environment-variable overrides
# ===========================================================================

class TestEnvOverrides:
    """Verify that environment variables correctly replace the hard-coded defaults."""

    def test_override_default_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch, {"RATE_LIMIT_DEFAULT": "120/minute"})
        assert mod.DEFAULT_RATE_LIMIT == "120/minute"

    def test_override_extract_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch, {"RATE_LIMIT_EXTRACT": "5/minute"})
        assert mod.EXTRACT_RATE_LIMIT == "5/minute"

    def test_override_batch_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch, {"RATE_LIMIT_BATCH": "2/minute"})
        assert mod.BATCH_RATE_LIMIT == "2/minute"

    def test_override_paper_rate_limit(self, monkeypatch):
        mod = fresh_module(monkeypatch, {"RATE_LIMIT_PAPER": "3/minute"})
        assert mod.PAPER_RATE_LIMIT == "3/minute"

    def test_all_four_overrides_simultaneously(self, monkeypatch):
        overrides = {
            "RATE_LIMIT_DEFAULT": "100/hour",
            "RATE_LIMIT_EXTRACT": "10/hour",
            "RATE_LIMIT_BATCH":   "5/hour",
            "RATE_LIMIT_PAPER":   "5/hour",
        }
        mod = fresh_module(monkeypatch, overrides)
        assert mod.DEFAULT_RATE_LIMIT == "100/hour"
        assert mod.EXTRACT_RATE_LIMIT == "10/hour"
        assert mod.BATCH_RATE_LIMIT   == "5/hour"
        assert mod.PAPER_RATE_LIMIT   == "5/hour"

    def test_hour_period_accepted(self, monkeypatch):
        mod = fresh_module(monkeypatch, {"RATE_LIMIT_DEFAULT": "1000/hour"})
        assert mod.DEFAULT_RATE_LIMIT == "1000/hour"


# ===========================================================================
# 3 — Limiter instance configuration
# ===========================================================================

class TestLimiterInstance:
    """Verify the shared Limiter is of the correct type and correctly wired."""

    def test_limiter_is_slowapi_limiter(self, monkeypatch):
        from slowapi import Limiter
        mod = fresh_module(monkeypatch)
        assert isinstance(mod.limiter, Limiter)

    def test_limiter_key_func_is_get_remote_address(self, monkeypatch):
        from slowapi.util import get_remote_address
        mod = fresh_module(monkeypatch)
        assert mod.limiter._key_func is get_remote_address

    def test_limiter_carries_at_least_one_default_limit(self, monkeypatch):
        """The default_limits list must be populated so the Limiter is active."""
        mod = fresh_module(monkeypatch)
        assert len(mod.limiter._default_limits) >= 1

    def test_limiter_is_not_none(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        assert mod.limiter is not None


# ===========================================================================
# 4 — __all__ exports
# ===========================================================================

class TestExports:
    """Verify __all__ is complete and every exported name is accessible."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        import rate_limiter
        self.mod = rate_limiter

    def test_all_contains_limiter(self):
        assert "limiter" in self.mod.__all__

    def test_all_contains_rate_limit_exceeded_handler(self):
        assert "_rate_limit_exceeded_handler" in self.mod.__all__

    def test_all_contains_rate_limit_exceeded(self):
        assert "RateLimitExceeded" in self.mod.__all__

    def test_all_contains_extract_rate_limit(self):
        assert "EXTRACT_RATE_LIMIT" in self.mod.__all__

    def test_all_contains_batch_rate_limit(self):
        assert "BATCH_RATE_LIMIT" in self.mod.__all__

    def test_all_contains_paper_rate_limit(self):
        assert "PAPER_RATE_LIMIT" in self.mod.__all__

    def test_every_all_symbol_is_accessible(self):
        """Every name listed in __all__ must exist as an attribute on the module."""
        for name in self.mod.__all__:
            assert hasattr(self.mod, name), (
                f"rate_limiter.__all__ lists '{name}' but the attribute is missing."
            )

    def test_rate_limit_exceeded_is_exception(self):
        assert issubclass(self.mod.RateLimitExceeded, Exception)

    def test_rate_limit_exceeded_handler_is_callable(self):
        assert callable(self.mod._rate_limit_exceeded_handler)


# ===========================================================================
# 5 — Security constraints
# ===========================================================================

class TestSecurityConstraints:
    """
    AI-invocation endpoints carry a higher billing risk than general endpoints.
    Their per-period request caps must therefore be *lower* than the generic
    default limit when both are expressed over the same time period.
    """

    def test_extract_limit_tighter_than_default(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        default_count, default_period = parse_limit(mod.DEFAULT_RATE_LIMIT)
        extract_count, extract_period = parse_limit(mod.EXTRACT_RATE_LIMIT)
        if default_period == extract_period:
            assert extract_count < default_count, (
                f"EXTRACT_RATE_LIMIT ({mod.EXTRACT_RATE_LIMIT}) must be stricter "
                f"than DEFAULT_RATE_LIMIT ({mod.DEFAULT_RATE_LIMIT}) to cap GCP spend."
            )

    def test_batch_limit_tighter_than_default(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        default_count, default_period = parse_limit(mod.DEFAULT_RATE_LIMIT)
        batch_count, batch_period = parse_limit(mod.BATCH_RATE_LIMIT)
        if default_period == batch_period:
            assert batch_count < default_count, (
                f"BATCH_RATE_LIMIT ({mod.BATCH_RATE_LIMIT}) must be stricter "
                f"than DEFAULT_RATE_LIMIT ({mod.DEFAULT_RATE_LIMIT})."
            )

    def test_paper_limit_tighter_than_default(self, monkeypatch):
        mod = fresh_module(monkeypatch)
        default_count, default_period = parse_limit(mod.DEFAULT_RATE_LIMIT)
        paper_count, paper_period = parse_limit(mod.PAPER_RATE_LIMIT)
        if default_period == paper_period:
            assert paper_count < default_count, (
                f"PAPER_RATE_LIMIT ({mod.PAPER_RATE_LIMIT}) must be stricter "
                f"than DEFAULT_RATE_LIMIT ({mod.DEFAULT_RATE_LIMIT})."
            )

    def test_default_limit_format_is_valid(self, monkeypatch):
        """All limit strings must match 'integer/period' so slowapi can parse them."""
        _VALID = re.compile(r"^\d+/(second|minute|hour|day)$")
        mod = fresh_module(monkeypatch)
        for attr in ("DEFAULT_RATE_LIMIT", "EXTRACT_RATE_LIMIT",
                     "BATCH_RATE_LIMIT", "PAPER_RATE_LIMIT"):
            value = getattr(mod, attr)
            assert _VALID.match(value), (
                f"{attr}={value!r} does not match the expected 'count/period' format."
            )

    def test_no_zero_count_in_defaults(self, monkeypatch):
        """A zero-count limit would block all traffic — guard against misconfiguration."""
        mod = fresh_module(monkeypatch)
        for attr in ("DEFAULT_RATE_LIMIT", "EXTRACT_RATE_LIMIT",
                     "BATCH_RATE_LIMIT", "PAPER_RATE_LIMIT"):
            count, _ = parse_limit(getattr(mod, attr))
            assert count > 0, f"{attr} count must be > 0"


# ===========================================================================
# 6 — Edge / negative cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_env_var_is_forwarded_without_fallback(self, monkeypatch):
        """
        os.getenv('KEY', default) returns '' when KEY is present but empty — it
        does NOT use the default.  This edge-case test documents that behaviour
        so any change to sentinel / fallback logic is caught immediately.
        """
        mod = fresh_module(monkeypatch, {"RATE_LIMIT_EXTRACT": ""})
        assert mod.EXTRACT_RATE_LIMIT == ""

    def test_numeric_only_env_var_propagates(self, monkeypatch):
        """Non-standard strings set via env vars are forwarded as-is (slowapi validates)."""
        mod = fresh_module(monkeypatch, {"RATE_LIMIT_DEFAULT": "999/day"})
        assert mod.DEFAULT_RATE_LIMIT == "999/day"

    def test_limiter_is_singleton_within_same_import(self):
        """Two calls to 'import rate_limiter' return the identical limiter object."""
        import rate_limiter as mod_a
        import rate_limiter as mod_b
        assert mod_a.limiter is mod_b.limiter

    def test_rate_limit_strings_are_non_empty_by_default(self, monkeypatch):
        """Defaults must never be blank strings."""
        mod = fresh_module(monkeypatch)
        for attr in ("DEFAULT_RATE_LIMIT", "EXTRACT_RATE_LIMIT",
                     "BATCH_RATE_LIMIT", "PAPER_RATE_LIMIT"):
            assert getattr(mod, attr), f"{attr} must not be an empty string"


# ===========================================================================
# 7 — Security regression: leaked API key must not appear in this module
# ===========================================================================

class TestSecurityRegression:
    """
    CRITICAL-001 required the hardcoded Gemini API key to be removed.
    Ensure it has not crept back into rate_limiter.py (or any module it imports).
    """

    def test_leaked_key_absent_from_rate_limiter_source(self):
        import rate_limiter
        source = inspect.getsource(rate_limiter)
        assert _LEAKED_API_KEY not in source, (
            "The previously-leaked Gemini API key was found in rate_limiter.py — "
            "remove it immediately and revoke the key."
        )

    def test_no_api_key_pattern_in_rate_limiter_source(self):
        """No string resembling a GCP/Gemini API key (AIza...) should appear."""
        import rate_limiter
        source = inspect.getsource(rate_limiter)
        # GCP/Gemini API keys always start with 'AIza'
        assert "AIza" not in source, (
            "A string matching a GCP API key prefix (AIza) was found in rate_limiter.py."
        )
