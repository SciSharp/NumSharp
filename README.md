# `data` — NumSharp docs dashboard data

This is an **orphan branch** (no shared history with `master`). It holds **only generated data**
that the documentation website's dashboards consume — no source code. It is the **single source of
truth** for that data: `master` carries only code and keeps no committed copy.

## Layout

One folder per **data type**. Inside each: one **snapshot** folder per publish, named
`<date>_<commithash>` (`YYYY-MM-DD_<shortsha>` — the source commit that produced the data;
lexicographically sortable), plus a **real `latest/` directory** (a data-only copy of the newest
snapshot, refreshed on every publish). `latest/` deliberately contains **no `README.md`** — the
per-type README is its sibling — so a folder-clone of `latest/` is pure data.

```
<data_type>/
  README.md                 # what the dataset is, its files, and how to (re)generate it
  2026-08-26_3c21b0d9/      # a snapshot; sortable name, source-commit provenance
    <data files…>
  2026-08-29_9b200075/      # a newer snapshot
    <data files…>
  latest/                   # real directory = copy of the newest snapshot (data only, no README)
    <data files…>
```

The four data types:

| Folder | Dataset | Consumed by (on the docs site) |
|--------|---------|-------------------------------|
| [`benchmark/`](benchmark/README.md) | NumSharp-vs-NumPy benchmark report + subsystem sheets | Benchmarks dashboard + "Raw Reports" |
| [`tests-oracle/`](tests-oracle/README.md) | Tests & differential-oracle inventory | "Unit Tests & Oracle" dashboard |
| [`inventory/`](inventory/README.md) | NumPy↔NumSharp public-API coverage / inventory | "NumPy API Coverage & Support" dashboard |
| [`benchmark-coverage/`](benchmark-coverage/README.md) | Benchmark-wiring coverage ledger | "Raw Reports → API Coverage Ledger" |

## How the site consumes this data (submodule, always latest)

The code branch (`master`) mounts this branch as a **git submodule** at `refs/data/`. Its recorded
commit is provenance only and is never bumped. At docs-build time CI clones this branch's tip straight
into that path (`git clone --depth 1 --single-branch --branch data https://github.com/SciSharp/NumSharp.git refs/data`)
and DocFX reads each dataset straight from `refs/data/<type>/latest/`. There is **no** date-priority
resolver and **no** master-side fallback copy — this branch is the only source, and its tip is always
what the site builds from.

`docs/website-src/docfx.json` publishes the reports as `docs/reports/…` and the dashboard files as
`docs/data/…`. DocFX resolves a link by the **source** path of its target, so a docs page or TOC entry
links a file here by its path in the checkout, `../../../refs/data/<type>/latest/<file>` from
`docs/website-src/docs/`, and DocFX writes the published path (`reports/MANIFEST.html`, …). A link to the
published path itself (`reports/MANIFEST.md`) is reported as `InvalidFileLink` and does not work.

Every push to `master` publishes `inventory`, `tests-oracle` and `benchmark-coverage` again, and
`publish.py` always replaces `latest/`, so `latest/` holds what `master`'s own tooling produces from
`master`'s code. A snapshot published from another branch stays in its dated folder but is replaced in
`latest/` by the next `master` push.

The publisher lives on the **code branch**, not here:

- `tools/dashboard_data/publish.py` — append a `<date>_<sha>` snapshot to a data type and refresh its
  real `latest/` directory.
- `tools/dashboard_data/common.py` — the shared per-type file map + snapshot/`latest` helpers.

CI publishes here automatically: `.github/workflows/docs.yml` publishes `inventory` + `tests-oracle`
+ `benchmark-coverage` on pushes to `master`; `.github/workflows/benchmark.yml` publishes `benchmark`
after a benchmark run.

### Publish by hand

```bash
# from a checkout of the code branch, with THIS branch checked out at ../NumSharp-data-wt
python tools/dashboard_data/publish.py --type inventory \
    --from coverage/generated \
    --branch-worktree ../NumSharp-data-wt \
    --commit
```

Each per-type README documents the exact regenerate + publish commands for that dataset.
