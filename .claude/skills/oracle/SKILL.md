---
name: oracle
description: >-
  NumSharp's differential-fuzz pipeline — the NumPy 2.4.2 "oracle" that proves every NDIter-backed
  op is BIT-IDENTICAL to NumPy across the input space. Use this whenever you add or change an np.*
  op and need fuzz coverage, regenerate the committed corpus, wire an op into OpRegistry, understand
  or debug the FuzzMatrix gate, triage a divergence (bit-diff / MisalignedRegistry / OpenBugs), or
  extend dtype/layout/tier coverage. Trigger on: "oracle", "differential fuzz", "fuzz coverage",
  "FuzzMatrix", "gen_oracle", "OpRegistry", "the corpus", "regenerate the corpus", "bit-exact vs
  numpy", "why is the fuzz gate failing / red", "add <op> to the fuzz gate", Char/Decimal/index/npy
  oracle, "shrink a failing case". This is the correctness gate — reach for it before assuming an op
  is done.
---

# NumSharp Differential-Fuzz Oracle

## Mental model (why this exists)

NumPy 2.4.2 is the **oracle**. Python generates a **committed, bytes-exact corpus** of `(inputs → NumPy output)`
cases; the C# harness rebuilds the exact operand bytes, runs NumSharp, and **bit-compares** against the recorded
NumPy output. **No Python runs at test time or in CI** — the corpus is replayed. A green gate means NumSharp is
byte-for-byte NumPy across every layout × dtype the corpus covers.

Three outcomes per case: **bit-exact** (pass), a **documented divergence** in `MisalignedRegistry` (excused, never
silent), or a **failure** (red → real bug, auto-shrunk to a 1-element repro).

The authoritative narrative lives in the project `.claude/CLAUDE.md` → "Differential-Fuzz Pipeline (NumPy oracle)"
and the divergence ledger `test/NumSharp.UnitTest/Fuzz/README.md`. This skill is the **actionable playbook**.

## File map

| Side | File | Role |
|------|------|------|
| Generator | `test/oracle/gen_oracle.py` | Deterministic op matrices (~90 ops) across modes. Writes `Fuzz/corpus/*.jsonl`. |
| Generator | `test/oracle/layout_catalog.py` | The memory-layout builders (the "44 variations") + value pools. |
| Generator | `test/oracle/gen_index_oracle.py` | Advanced-indexing get/set oracle (`index_*` tiers). |
| Generator | `test/oracle/gen_decimal_oracle.cs` | Independent C# oracle for `Decimal` (no NumPy analog). |
| Generator | `test/oracle/gen_npy_oracle.py` | `.npy`/`.npz` format oracle (separate corpus + gate). |
| Generator | `test/oracle/fuzz_random.py` | Seeded random fuzzer (nightly soak). |
| Harness | `test/NumSharp.UnitTest/Fuzz/OpRegistry.cs` | **op-name → NumSharp call.** Pairs 1:1 with `gen_oracle.py`. |
| Harness | `test/NumSharp.UnitTest/Fuzz/FuzzCorpus.cs` | Rebuilds exact NDArray views from `(dtype,shape,strides,offset,bytes)`. |
| Harness | `test/NumSharp.UnitTest/Fuzz/FuzzCorpusTests.cs` | One `[FuzzMatrix]` test per corpus file (`RunCorpus("<tier>.jsonl")`). |
| Harness | `test/NumSharp.UnitTest/Fuzz/{BitDiff,Shrinker}.cs` | Bit-exact compare (NaN tokenized; Decimal by value) / shrink to 1 element. |
| Harness | `test/NumSharp.UnitTest/Fuzz/MisalignedRegistry.cs` | The excused, documented divergences. |
| Corpus | `test/NumSharp.UnitTest/Fuzz/corpus/*.jsonl` | The committed corpus (~68K cases). Copied to test output by the csproj glob. |

## The gate

`dotnet test --filter "TestCategory=FuzzMatrix"` runs three gate classes:
- **`FuzzCorpusTests`** — the op corpus (~40 tiers; `astype/binary/unary/reduce/manip/...`). This is where new-op work lands.
- **`IndexOracleTests`** — advanced-indexing get/set (`index_curated` + `index_dtype` + `index_random`).
- **`MetamorphicTests`** — NumPy-free invariants (round-trips / involutions).

Run one tier while iterating: `dotnet test --no-build -f net10.0 --filter "FullyQualifiedName~FuzzCorpusTests.Manip"`.

> **Known flake:** the full `TestCategory=FuzzMatrix` run may end with "Test host process crashed"
> (an intermittent `AccessViolation`) AFTER all tests report Passed. That's a teardown crash, not a
> failure — re-run the specific `FuzzCorpusTests` class (exit 0, no crash) to confirm green.

## Playbook — add a new op to the oracle

This is the most common task. The full worked example (flip/trim_zeros, plus params/char/OpRegistry patterns)
is in **`references/add-op.md`** — read it when adding an op. In brief:

1. **Pick the tier** in `gen_oracle.py` whose `gen_<mode>` fits your op (shape ops → `gen_manip`, elementwise →
   `gen_binary`/`gen_unary`, reductions → `gen_reduce`, …). Modes are listed in `main()`'s `elif mode == ...`.
2. **Add a job** to that tier's job list: a `(opname, params_dict, lambda v: np.<op>(v, ...))` tuple. Guard by
   `nd`/`sz` where NumPy would raise (the generator's `try/except` skips those and prints a count).
3. **Add the matching case** to `OpRegistry.cs` — `case "<opname>": return np.<op>(ops[0], ...);` — reading params
   with `p["k"].GetInt32()` / `p["trim"].GetString()` / `ParseIntArray(p["axes"])`. Convention: `"axis"` (scalar int)
   vs `"axes"` (int[]) selects the overload.
4. **Regenerate** the corpus (needs `numpy==2.4.2`): `python test/oracle/gen_oracle.py <mode>`.
5. **Build** (the csproj glob copies the corpus to test output) **and run** the tier: `dotnet build` then
   `dotnet test --no-build -f net10.0 --filter "FullyQualifiedName~FuzzCorpusTests.<Tier>"`.
6. **Triage** any red (see below). Char coverage is woven automatically via `char_tier(<mode>)` — no extra wiring.

## Other tasks → where to go

- **Regenerate any/all tiers, or a dtype/layout question** → `references/regenerate.md` (the full command matrix,
  the numpy pin, determinism, the ALL_DTYPES / char / decimal story, how layouts feed every op).
- **A case diverged (red), or you need to excuse an intended difference** → `references/triage.md`
  (bit-diff → shrink → MisalignedRegistry vs OpenBugs; NaN/Decimal comparison rules).
- **Deeper system map** (all generators, the corpus tiers, char/decimal/index/npy oracles, the harness classes)
  → `references/architecture.md`.

## Critical gotchas (learned the hard way)

- **Pin `numpy==2.4.2`.** A different NumPy version can shift bytes and make the committed corpus wrong. Verify
  `python -c "import numpy; print(numpy.__version__)"` before regenerating.
- **The corpus diff is huge but harmless.** Case `id`s carry a global running counter (`{op}/{layout}/{dtype}/{n}`),
  so adding one job renumbers every following id. Expect a large `git diff` on `*.jsonl` — it's renumbering, not
  semantic churn.
- **Char has no NumPy dtype.** It rides the `uint16` proxy: `char_tier("<mode>")` re-runs your `gen_<mode>` with the
  Char pool and relabels `uint16 → char`. Add your op to `gen_<mode>` and Char coverage is automatic.
- **Decimal has no NumPy analog.** It rides the independent C# oracle `gen_decimal_oracle.cs` (naive scalar
  `System.Decimal`), regenerated via `dotnet run test/oracle/gen_decimal_oracle.cs`. If your op needs Decimal
  coverage, add it there too.
- **The generator resolves paths relative to `test/oracle/`** and writes into `test/NumSharp.UnitTest/Fuzz/corpus/`.
  Run it from `test/oracle/` (or with that CWD). CI replays the committed corpus and never runs the generator.
- **OpRegistry's `default:` throws `NotSupportedException(op)`** — so a corpus op with no registered case fails the
  tier loudly. If a tier goes red immediately on a new op, you forgot (or mistyped) the `OpRegistry` case.

## References

- `references/add-op.md` — the detailed add-an-op playbook with a worked example (params, char, OpRegistry patterns).
- `references/regenerate.md` — the full regeneration command matrix + dtype/layout coverage model.
- `references/triage.md` — divergence handling: bit-diff → shrink → MisalignedRegistry vs OpenBugs.
- `references/architecture.md` — the complete system map (generators, tiers, the four oracles, the harness).
