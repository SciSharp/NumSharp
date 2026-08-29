# `inventory` — NumPy ↔ NumSharp public-API coverage

A deterministic classification of NumPy **2.4.2**'s public API against the compiled NumSharp assembly:
each API is `declared` / `partial` / `unsupported` / `missing` / `extension`, scoped to top-level `np.*`,
public `ndarray` members, and the `np.random` / `np.linalg` / `np.fft` callables. This is the
**"NumPy API Coverage & Support"** dataset (a.k.a. the API inventory) — distinct from `benchmark-coverage`,
which measures benchmark wiring rather than API presence.

## Files in a snapshot

| File | What |
|------|------|
| `coverage.json` | The full per-API classification (**`coverage-support-dashboard.md` fetches this**) |
| `coverage.csv` | Flat spreadsheet form |
| `manifest.json` | Generator version, pinned NumPy version, per-surface totals |
| `summary.md` | Human-readable per-surface summary |

## Consumed by the docs site

- `coverage-support-dashboard.md` fetches `coverage.json`.
- `compliance.md` narrates the same numbers (its snapshot table is hand-maintained).

## Regenerate

```bash
python coverage/generate_coverage.py --output coverage/generated   # needs numpy==2.4.2
```

## Publish to this branch

```bash
python tools/dashboard_data/publish.py --type inventory \
    --from coverage/generated \
    --branch-worktree ../NumSharp-master-code-data \
    --commit
```

(CI does this automatically in `.github/workflows/docs.yml` on pushes to `master`.)
