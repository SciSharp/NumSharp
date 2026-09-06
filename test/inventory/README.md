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

The inventory reports declarations and committed evidence, not a fabricated pass rate. Its units
are deliberately separate:

- **MSTest declarations** are reflected methods. Declared `DataRow` and `DynamicData` metadata is
  retained separately; the dashboard does not invent or headline an execution count.
- **Oracle test cases** are executable JSONL/index records plus the specialized flags, layout, and
  NPY/NPZ cases. BLAS `*.host.jsonl` provenance records are metadata and are excluded.
- **Oracle operation keys** group corpus schemas for exploration; they are not a public-API count.
- **Dtype links** are non-exclusive because a mixed-type Oracle test case may reference several dtypes.
- Absence of an error test case is neutral unless the corresponding API defines invalid inputs.

Runtime pass/fail remains the responsibility of `dotnet test`; the dashboard labels known-bug,
ignored, manual, and platform gates directly from attributes.
