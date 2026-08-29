# `master-code-data` — NumSharp docs dashboard data

This is an **orphan branch** (no shared history with `master`). It holds **only generated data**
that the documentation website's dashboards consume — no source code. It exists so that large,
frequently-regenerated data blobs live apart from the code branch's history.

## Layout

One folder per **data type**. Inside each, one **snapshot** folder per publish, named
`<date>_<commithash>` (`YYYY-MM-DD_<shortsha>` — the source commit that produced the data;
lexicographically sortable), plus a git symlink `latest` pointing at the newest snapshot.

```
<data_type>/
  README.md                 # what the dataset is, its files, and how to (re)generate it
  2026-08-26_3c21b0d9/      # a snapshot; sortable name, source-commit provenance
    <data files…>
  2026-08-28_6837c918/      # a newer snapshot
    <data files…>
  latest -> 2026-08-28_6837c918   # git symlink (mode 120000), repointed on every publish
```

The four data types:

| Folder | Dataset | Consumed by (on the docs site) |
|--------|---------|-------------------------------|
| [`benchmark/`](benchmark/README.md) | NumSharp-vs-NumPy benchmark report + subsystem sheets | Benchmarks dashboard + "Raw Reports" |
| [`tests-oracle/`](tests-oracle/README.md) | Tests & differential-oracle inventory | "Unit Tests & Oracle" dashboard |
| [`inventory/`](inventory/README.md) | NumPy↔NumSharp public-API coverage / inventory | "NumPy API Coverage & Support" dashboard |
| [`benchmark-coverage/`](benchmark-coverage/README.md) | Benchmark-wiring coverage ledger | "Raw Reports → API Coverage Ledger" |

## How the site chooses data: date priority vs `master` (backwards compatible)

The docs are built from the **code branch** (`master`), which still carries its own committed copies
of every dataset. At docs-build time a resolver compares, **per data type**, the git **commit date**
of master's canonical file against the commit date of this branch's `<type>/latest`, and bakes in
whichever is **newer**. So:

- The site still builds correctly from `master` alone (this branch is purely additive).
- Any newer publish — to `master` *or* to this branch — wins by commit date.
- `latest` is the "which snapshot is newest" pointer; if it is ever missing, consumers fall back to
  the lexicographically-greatest `<date>_<commithash>` folder.

The resolver and publisher live on the **code branch**, not here:

- `tools/dashboard_data/publish.py` — append a snapshot to a data type and repoint `latest`.
- `tools/dashboard_data/resolve.py` — pick the newest of master-vs-branch and stage it for DocFX.
- `tools/dashboard_data/common.py` — the shared per-type file/overlay map.

CI publishes here automatically: `.github/workflows/docs.yml` publishes `inventory` + `tests-oracle`
on pushes to `master`; `.github/workflows/benchmark.yml` publishes `benchmark` + `benchmark-coverage`
after a benchmark run.

### Publish by hand

```bash
# from a checkout of the code branch, with this branch checked out at ../NumSharp-master-code-data
python tools/dashboard_data/publish.py --type inventory \
    --from coverage/generated \
    --branch-worktree ../NumSharp-master-code-data \
    --commit
```

Each per-type README documents the exact regenerate + publish commands for that dataset.
