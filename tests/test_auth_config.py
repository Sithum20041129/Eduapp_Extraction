"""
Tests for the centralized config and API-key auth dependency.

These cover the security controls added during hardening without needing the
heavy Vertex AI / FastAPI app to import.
"""
import asyncio
import importlib

import pytest


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def _reload_config(monkeypatch, **env):
    import config
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    return importlib.reload(config)


class TestConfig:
    def test_defaults(self, monkeypatch):
        cfg = _reload_config(monkeypatch, SERVICE_API_KEY=None, CORS_ALLOWED_ORIGINS=None,
                             MAX_UPLOAD_BYTES=None)
        assert cfg.MAX_UPLOAD_BYTES == 10 * 1024 * 1024
        assert cfg.AUTH_ENABLED is False
        assert cfg.CORS_ALLOWED_ORIGINS == []
        assert "image/png" in cfg.ALLOWED_IMAGE_TYPES
        assert "application/pdf" in cfg.ALLOWED_PAPER_TYPES

    def test_api_keys_parsed_as_set(self, monkeypatch):
        cfg = _reload_config(monkeypatch, SERVICE_API_KEY="k1, k2 ,k3")
        assert cfg.SERVICE_API_KEYS == {"k1", "k2", "k3"}
        assert cfg.AUTH_ENABLED is True

    def test_max_upload_override(self, monkeypatch):
        cfg = _reload_config(monkeypatch, MAX_UPLOAD_BYTES="1234")
        assert cfg.MAX_UPLOAD_BYTES == 1234

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        cfg = _reload_config(monkeypatch, MAX_UPLOAD_BYTES="not-a-number")
        assert cfg.MAX_UPLOAD_BYTES == 10 * 1024 * 1024

    def test_cors_never_wildcard_by_default(self, monkeypatch):
        cfg = _reload_config(monkeypatch, CORS_ALLOWED_ORIGINS=None)
        assert "*" not in cfg.CORS_ALLOWED_ORIGINS


# ---------------------------------------------------------------------------
# auth dependency
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


class TestAuth:
    def _reload_auth(self, monkeypatch, key_env):
        import config, auth
        if key_env is None:
            monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        else:
            monkeypatch.setenv("SERVICE_API_KEY", key_env)
        importlib.reload(config)
        return importlib.reload(auth)

    def test_disabled_mode_allows_any_request(self, monkeypatch):
        auth = self._reload_auth(monkeypatch, None)
        # Should not raise even with no header
        assert _run(auth.require_api_key(None)) is None

    def test_enabled_rejects_missing_key(self, monkeypatch):
        auth = self._reload_auth(monkeypatch, "secret123")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            _run(auth.require_api_key(None))
        assert ei.value.status_code == 401

    def test_enabled_rejects_wrong_key(self, monkeypatch):
        auth = self._reload_auth(monkeypatch, "secret123")
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as ei:
            _run(auth.require_api_key("wrong"))
        assert ei.value.status_code == 401

    def test_enabled_accepts_correct_key(self, monkeypatch):
        auth = self._reload_auth(monkeypatch, "secret123")
        assert _run(auth.require_api_key("secret123")) is None

    def test_multiple_keys_each_accepted(self, monkeypatch):
        auth = self._reload_auth(monkeypatch, "k1,k2")
        assert _run(auth.require_api_key("k1")) is None
        assert _run(auth.require_api_key("k2")) is None
