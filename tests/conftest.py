"""Pytest configuration for repository test runs.

The src path insertion and cli-module eviction happen at module
level so they execute before pytest collects test modules.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

# ---- early path setup (runs at import time, before collection) ----
_project_root = Path(__file__).resolve().parent.parent
_src_str = str(_project_root / "src")
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)
# Evict any cached 'cli' module from an installed third-party package
# so that our project's src/cli is used instead.
for _key in list(sys.modules):
    if _key == "cli" or _key.startswith("cli."):
        _mod = sys.modules[_key]
        _origin = getattr(_mod, "__file__", "") or ""
        if not _origin.startswith(_src_str):
            del sys.modules[_key]
# Clear the path-finder cache so Python re-discovers 'cli' from src/.
importlib.invalidate_caches()
sys.path_importer_cache.clear()
# Pre-load our cli package so no test import picks up the wrong one.
_cli_init = _project_root / "src" / "cli" / "__init__.py"
_spec = importlib.util.spec_from_file_location("cli", str(_cli_init),
                                                submodule_search_locations=[str(_project_root / "src" / "cli")])
if _spec and _spec.loader:
    _cli_mod = importlib.util.module_from_spec(_spec)
    sys.modules["cli"] = _cli_mod
    _spec.loader.exec_module(_cli_mod)
