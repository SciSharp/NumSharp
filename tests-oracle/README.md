# `tests-oracle` — Tests & differential-oracle inventory

A deterministic **inventory** of the test suite and the committed NumPy-oracle corpus. It reflects the
compiled MSTest assemblies (via the `NumSharp.Tools.TestInventory` helper), adds source ownership, and
records the differential-fuzz corpus dimensions. **No tests are executed and no pass rate is invented.**

## Files in a snapshot

| File | What |
|------|------|
| `tests-oracle-report.json` | The inventory the dashboard renders (**`tests-oracle-dashboard.md` fetches this**): projects, areas, classes, source ownership, oracle corpus tiers |
| `tests-oracle-report.csv` | Flat spreadsheet form |
| `tests-oracle-manifest.json` | Schema/generator version + provenance |
| `summary.md` | Human-readable summary (linked under the toc "Generated Summary") |

## Consumed by the docs site

- `tests-oracle-dashboard.md` fetches `tests-oracle-report.json`.
- The toc links `summary.md` under "Unit Tests & Oracle → Generated Summary".

## Regenerate

```bash
python test/inventory/generate_test_inventory.py     # writes test/inventory/generated/ by default
```

Deterministic; builds the `NumSharp.Tools.TestInventory` reflection tool and reads
`test/NumSharp.Tests.Oracle/Fuzz/corpus/`.

## Publish to this branch

```bash
python tools/dashboard_data/publish.py --type tests-oracle \
    --from test/inventory/generated \
    --branch-worktree ../NumSharp-master-code-data \
    --commit
```

(CI does this automatically in `.github/workflows/docs.yml` on pushes to `master`.)
