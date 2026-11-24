"""Fase 8: garantías estáticas de seguridad (Real bloqueado, sin secretos, capas)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import config
from broker.deriv_auth import OtpUrlProvider, validate_otp_url
from config import AccountEnvironment, RealModeLockedError, Settings, load_settings
from database.database import Database, ForbiddenSettingError

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [p for p in ROOT.rglob("*.py")
           if ".venv" not in p.parts and "tests" not in p.parts]


def test_real_mode_constant_is_false():
    assert config.REAL_MODE_AVAILABLE is False


def test_real_mode_constant_only_defined_in_config():
    """Nadie fuera de config.py asigna REAL_MODE_AVAILABLE (ni lo hace mutable)."""
    offenders = []
    for py in SOURCES:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for t in targets:
                name = getattr(t, "id", None) or getattr(t, "attr", None)
                if name == "REAL_MODE_AVAILABLE" and py.name != "config.py":
                    offenders.append(str(py.relative_to(ROOT)))
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "setattr":
                if any(isinstance(a, ast.Constant) and a.value == "REAL_MODE_AVAILABLE"
                       for a in node.args):
                    offenders.append(str(py.relative_to(ROOT)))
    assert offenders == []


def test_environment_never_loaded_from_env_or_db(monkeypatch, tmp_path):
    for var in ("TRADINGBOT_ENV", "DERIV_ENV", "ENVIRONMENT"):
        monkeypatch.setenv(var, "real")
    db = Database(tmp_path / "x.sqlite3")
    with pytest.raises(ForbiddenSettingError):
        db.set_setting("environment", "real")
    s = load_settings()
    assert s.environment is AccountEnvironment.DEMO
    with pytest.raises(RealModeLockedError):
        Settings().select_environment(AccountEnvironment.REAL)


def test_real_ws_and_otp_paths_refused():
    with pytest.raises(RealModeLockedError):
        validate_otp_url("wss://api.derivws.com/trading/v1/options/ws/real?otp=x")
    with pytest.raises(RealModeLockedError):
        OtpUrlProvider(object(), "X", AccountEnvironment.REAL)


def test_no_hardcoded_secrets_in_sources():
    patterns = [re.compile(p) for p in (
        r"pat_[A-Za-z0-9]{12,}", r"(?i)bearer\s+[A-Za-z0-9._-]{20,}",
        # Asignación literal; se excluyen nombres de variables de entorno (MAYÚSCULAS).
        r"(?i:token|api_key|secret|password)\s*=\s*['\"](?![A-Z0-9_]+['\"])[^'\"]{8,}['\"]",
        r"otp=[A-Za-z0-9]{12,}")]
    hits = []
    for py in SOURCES:
        text = py.read_text(encoding="utf-8")
        for pat in patterns:
            if pat.search(text):
                hits.append(f"{py.relative_to(ROOT)}: {pat.pattern}")
    assert hits == []


def test_secret_scanner_detects_and_ignores_correctly():
    pat = re.compile(r"(?i:token|api_key|secret|password)\s*=\s*['\"](?![A-Z0-9_]+['\"])"
                     r"[^'\"]{8,}['\"]")
    assert pat.search('token = "abcdef123456789"')
    assert pat.search('API_KEY = "Xy9_live_value_123"')
    assert not pat.search('ENV_TOKEN = "DERIV_TOKEN"')


def test_env_example_has_no_values():
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            assert line.split("=", 1)[1].strip() == "", line


def test_gitignore_protects_secrets_and_data():
    gi = (ROOT / ".gitignore").read_text().splitlines()
    for entry in (".env", "*.sqlite3", ".venv/", "models/*"):
        assert entry in gi


def test_gui_never_imports_broker_or_execution():
    """interface/ solo habla con el motor: sin broker, controlador ni riesgo directo."""
    forbidden = ("broker", "core.execution_controller", "core.order_approval")
    offenders = []
    for py in (ROOT / "interface").glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            for n in ([mod] if mod else []) + names:
                if n.startswith(forbidden):
                    offenders.append(f"{py.name}: {n}")
    assert offenders == []
