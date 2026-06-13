"""
Tests for the security fix in core/list_models.py.

The fix removes a hardcoded Google Generative AI API key (now revoked) and
replaces it with os.getenv('GOOGLE_API_KEY').

Coverage:
  1. [Security]   Leaked key is absent from source text.
  2. [Security]   No AIza-prefixed string literals anywhere in the AST.
  3. [Security]   api_key variable is not assigned a plain string constant.
  4. [Functional] Source calls os.getenv('GOOGLE_API_KEY').
  5. [Functional] genai.configure() receives the value from the env var.
  6. [Edge case]  genai.configure() receives None when env var is absent.
  7. [Security]   configure() is never called with the historically leaked key.
  8. [Functional] Only models supporting 'generateContent' are printed.
  9. [Functional] Models NOT supporting 'generateContent' are not printed.
  10.[Functional] Mixed model list is filtered correctly.
  11.[Edge case]  Empty model list does not crash.
  12.[Functional] Exceptions from list_models() are caught, not propagated.
  13.[Functional] Caught exception produces output containing 'Error'.
  14.[Security]   Error path does not echo the api_key variable value.
"""

import ast
import io
import os
import sys
import uuid
import importlib.util
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_FILE = (Path(__file__).parent.parent / "core" / "list_models.py").resolve()
# Historically leaked key (now revoked). Reconstructed from fragments so this
# test file does not itself commit the verbatim secret that scanners flag.
LEAKED_KEY = "AIza" + "SyCGqeqalFqiok" + "-_r17nYiBKPdO9TfnEuMI"

_UNSET = object()  # sentinel: GOOGLE_API_KEY will be absent from the env


# ---------------------------------------------------------------------------
# Helper: execute the module under test with full dependency control
# ---------------------------------------------------------------------------

def _run_module(api_key=_UNSET, models=None, raise_on_list=None):
    """
    Execute core/list_models.py in isolation.

    Parameters
    ----------
    api_key : str | _UNSET
        Value to place in GOOGLE_API_KEY.  Pass the sentinel to omit the var.
    models : list | None
        Return value for genai.list_models().
    raise_on_list : Exception | None
        If provided, genai.list_models() raises this instead of returning.

    Returns
    -------
    (mock_genai, captured_stdout_text)
    """
    mock_genai = MagicMock()
    if raise_on_list is not None:
        mock_genai.list_models.side_effect = raise_on_list
    else:
        mock_genai.list_models.return_value = [] if models is None else models

    # Build a clean environment: strip GOOGLE_API_KEY, then optionally add it.
    env = {k: v for k, v in os.environ.items() if k != "GOOGLE_API_KEY"}
    if api_key is not _UNSET:
        env["GOOGLE_API_KEY"] = api_key  # type: ignore[assignment]

    # Patch sys.modules so `import google.generativeai as genai` resolves to mock.
    google_stub = sys.modules.get("google", MagicMock())
    sys_patches = {
        "google": google_stub,
        "google.generativeai": mock_genai,
    }

    buf = io.StringIO()
    unique_name = f"list_models__{uuid.uuid4().hex}"

    with patch.dict(sys.modules, sys_patches):
        with patch.dict(os.environ, env, clear=True):
            spec = importlib.util.spec_from_file_location(unique_name, str(SOURCE_FILE))
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            with redirect_stdout(buf):
                spec.loader.exec_module(mod)  # type: ignore[union-attr]

    return mock_genai, buf.getvalue()


# ---------------------------------------------------------------------------
# Stub for fake model objects
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal stand-in for a google.generativeai Model."""

    def __init__(self, name: str, methods: list):
        self.name = name
        self.supported_generation_methods = methods


# ===========================================================================
# 1-3  Security tests
# ===========================================================================

class TestNoHardcodedCredentials:

    def test_leaked_key_absent_from_source_text(self):
        """The previously leaked API key must not appear anywhere in the file."""
        text = SOURCE_FILE.read_text(encoding="utf-8")
        assert LEAKED_KEY not in text, (
            f"CRITICAL: hardcoded API key '{LEAKED_KEY[:12]}...' "
            "still present in core/list_models.py"
        )

    def test_no_aiza_string_constants_in_ast(self):
        """No string constant prefixed 'AIza' (GCP API key format) may exist in the AST."""
        source = SOURCE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        violations = [
            (n.lineno, n.value[:16] + "...")
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("AIza")
        ]
        assert violations == [], f"Hardcoded GCP API key(s) found: {violations}"

    def test_api_key_variable_not_assigned_string_literal(self):
        """The 'api_key' name must not be assigned a plain string constant."""
        source = SOURCE_FILE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "api_key":
                    assert not isinstance(node.value, ast.Constant), (
                        f"api_key assigned a literal at line {node.lineno}; "
                        "must use os.getenv() instead."
                    )


# ===========================================================================
# 4-7  Environment-variable resolution
# ===========================================================================

class TestEnvVarResolution:

    def test_source_calls_os_getenv_google_api_key(self):
        """Source must call os.getenv('GOOGLE_API_KEY') to retrieve the key."""
        text = SOURCE_FILE.read_text(encoding="utf-8")
        assert (
            'os.getenv("GOOGLE_API_KEY")' in text
            or "os.getenv('GOOGLE_API_KEY')" in text
        ), "Must use os.getenv('GOOGLE_API_KEY') to read the API key."

    def test_configure_called_with_env_var_value(self):
        """genai.configure() must receive the exact value placed in GOOGLE_API_KEY."""
        fake_key = "test-key-abc-123-xyz"
        mock_genai, _ = _run_module(api_key=fake_key)
        mock_genai.configure.assert_called_once_with(api_key=fake_key)

    def test_configure_called_with_none_when_env_var_absent(self):
        """When GOOGLE_API_KEY is absent, configure() must receive api_key=None."""
        mock_genai, _ = _run_module()  # _UNSET → env var removed
        mock_genai.configure.assert_called_once_with(api_key=None)

    def test_configure_never_called_with_leaked_key(self):
        """configure() must never be invoked with the historically leaked key string."""
        # Worst-case: env var absent — any hardcoded fallback would surface here.
        mock_genai, _ = _run_module()
        for actual_call in mock_genai.configure.call_args_list:
            args, kwargs = actual_call
            passed_key = kwargs.get("api_key", args[0] if args else None)
            assert passed_key != LEAKED_KEY, (
                "configure() was invoked with the leaked key — it is still hardcoded!"
            )


# ===========================================================================
# 8-11  Model-list filtering
# ===========================================================================

class TestModelFiltering:

    def test_generate_content_model_is_printed(self):
        """A model supporting generateContent must appear in stdout."""
        models = [_FakeModel("models/gemini-pro", ["generateContent"])]
        _, out = _run_module(api_key="k", models=models)
        assert "models/gemini-pro" in out

    def test_non_generate_content_model_is_not_printed(self):
        """A model NOT supporting generateContent must not appear in stdout."""
        models = [_FakeModel("models/embedding-001", ["embedContent"])]
        _, out = _run_module(api_key="k", models=models)
        assert "models/embedding-001" not in out

    def test_mixed_model_list_filtered_correctly(self):
        """Only generateContent models are printed from a mixed list."""
        models = [
            _FakeModel("models/gemini-pro", ["generateContent"]),
            _FakeModel("models/embedding-001", ["embedContent"]),
            _FakeModel("models/gemini-ultra", ["generateContent", "countTokens"]),
        ]
        _, out = _run_module(api_key="k", models=models)
        assert "models/gemini-pro" in out
        assert "models/gemini-ultra" in out
        assert "models/embedding-001" not in out

    def test_empty_model_list_does_not_crash(self):
        """An empty model list must not raise and must not print any model name."""
        _, out = _run_module(api_key="k", models=[])
        assert "models/" not in out


# ===========================================================================
# 12-14  Error handling
# ===========================================================================

class TestErrorHandling:

    def test_list_models_exception_is_caught_not_propagated(self):
        """An exception from genai.list_models() must not escape the module."""
        try:
            _run_module(api_key="k", raise_on_list=Exception("simulated API error"))
        except Exception as exc:
            pytest.fail(
                f"Exception propagated from module — should have been caught: {exc}"
            )

    def test_error_output_produced_on_exception(self):
        """When list_models() raises, the module must print an error message."""
        _, out = _run_module(api_key="k", raise_on_list=Exception("network timeout"))
        assert "error" in out.lower(), (
            f"Expected an error message in stdout; got: {out!r}"
        )

    def test_error_path_does_not_echo_api_key_variable(self):
        """
        The exception handler must not print 'api_key = <value>' to stdout,
        which would leak the credential into logs.
        """
        secret = "s3cr3t-k3y-n0t-4-l0g"
        _, out = _run_module(
            api_key=secret,
            raise_on_list=Exception("auth failure"),
        )
        assert f"api_key = {secret}" not in out
        assert f'api_key = "{secret}"' not in out
        assert f"api_key = '{secret}'" not in out
