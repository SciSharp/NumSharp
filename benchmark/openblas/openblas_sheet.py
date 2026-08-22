#!/usr/bin/env python3
"""Compatibility entry point for the unified Managed/OpenBLAS profile harness."""

import runpy
from pathlib import Path

driver = Path(__file__).resolve().parents[1] / "backends" / "backend_profiles.py"
runpy.run_path(str(driver), run_name="__main__")
