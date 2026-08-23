# Tests & Oracle inventory

This directory is the reproducible source for the website's Tests & Oracle Dashboard. It combines
reflection over the three real MSTest assemblies with the committed NumPy/Decimal/index/format
oracle artifacts.

```powershell
python test/inventory/generate_test_inventory.py
python test/inventory/generate_test_inventory.py --check
```

Generated, committed outputs live in `test/inventory/generated/`. The DocFX build copies
`tests-oracle-report.json` to `docs/data/`; the dashboard never runs tests or Python in the browser.

The inventory reports declarations and committed evidence, not a fabricated pass rate. Runtime
pass/fail remains the responsibility of `dotnet test`; the dashboard labels excluded known bugs,
ignored/manual tests, FuzzMatrix gates, and specialized oracle suites directly from attributes.
