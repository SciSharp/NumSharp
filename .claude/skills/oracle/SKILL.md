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
and the divergence ledger `test/NumSharp.Tests.Oracle/Fuzz/README.md`. This skill is the **actionable playbook**.

## File map

The op oracle lives in **two** directories: generators in `test/oracle/`, the replay harness + committed corpus
in `test/NumSharp.Tests.Oracle/Fuzz/`. (Two *sibling* oracles — flags & layout-parity — live in the main
`test/NumSharp.Tests/` project instead; see "The six oracles" below.)

| Side | File | Role |
|------|------|------|
| Generator | `test/oracle/gen_oracle.py` | Deterministic value/error/kind/artifact matrices (**~7.4K lines**, ~45 modes) across families. Writes one `Fuzz/corpus/<mode>.jsonl` per mode. |
| Generator | `test/oracle/layout_catalog.py` | The memory-layout builders (40 variations: 26 single + 9 pair + 5 where) + value pools. |
| Generator | `test/oracle/gen_nan_oracle.py` | **Standalone** NaN-parity oracle → `nan.jsonl` (complex-unary NaN **sign** bit-exact; float widths value-NaN). Owns its own numbering, like the npy/decimal oracles. |
| Generator | `test/oracle/gen_index_oracle.py` | Advanced-indexing get/set oracle (`index_*` tiers). |
| Generator | `test/oracle/gen_decimal_oracle.cs` | Independent C# oracle for `Decimal` (no NumPy analog) → `decimal_*.jsonl`. |
| Generator | `test/oracle/gen_npy_oracle.py` | `.npy`/`.npz` format oracle (separate corpus + `NpyOracle` gate). |
| Generator | `test/oracle/fuzz_random.py` | Seeded random fuzzer (nightly soak; also the committed `random_smoke.jsonl` batch). |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/OpRegistry{,.Kinds,.Generator}.cs` | **op-name → NumSharp call.** `.cs` = array ops (~365 cases), `.Kinds.cs` = dtype/text/tuple/scalar results, `.Generator.cs` = PCG64/`default_rng` ops. Pairs 1:1 with the generators. |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/FuzzCorpus.cs` | Rebuilds exact NDArray views from `(dtype,shape,strides,offset,bytes)` — the C# side is layout-agnostic. |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/FuzzCorpusTests{,.Kinds}.cs` | One `[FuzzMatrix]` test per corpus file (`RunCorpus`/`RunHostLibmCorpus("<tier>.jsonl")`); `.Kinds.cs` runs the non-array tiers (`iter`/`dtype_text`/`out_where`/`errors_full`/`multioutput`) + the comparators. |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/{BitDiff,Shrinker}.cs` | Bit-exact compare (NaN tokenized *except* the contractual complex-unary NaN sign; Decimal by value) / shrink to 1 element. |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/MisalignedRegistry.cs` | The excused, documented divergences (`Classify(...)` → reason string). |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/{OracleSurfaceCoverageTests,OracleCoverageStrengthTests,Journey3TouchedOracleCoverageTests}.cs` | Coverage gates: public-surface inventory / ≥4-cases-per-op / 186 journey3 callables. |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/{UndisposedIntermediateTests,NativeAllocationChokepointTests,ScopeAudit}.cs` | **Oracle-FREE leak gates that replay the same corpus through `OpRegistry`** — buffer-pool balance + raw-alloc chokepoint. Adding an op enters these too (see gotchas). |
| Harness | `test/NumSharp.Tests.Oracle/Fuzz/{BlasBackendDeltaTests,MatmulParityPin,BlasEngineAutoInstallGuard}.cs` | Managed/OpenBLAS two-pass delta + the host-pin (BLAS-off by default). |
| Corpus | `test/NumSharp.Tests.Oracle/Fuzz/corpus/**/*.jsonl` | The committed corpus (**~118K rows across 68 files** at 2026-09; grows per regeneration). The csproj glob `Fuzz\corpus\**\*.jsonl` copies it — incl. any `regressions/` — to test output. |

## The gate

`dotnet test --filter "TestCategory=FuzzMatrix"` runs the gate classes:
- **`FuzzCorpusTests` (+ `.Kinds`)** — the op corpus: one `[FuzzMatrix]` method per tier. Includes deterministic
  `creation/conversion/multioutput`, value/parity tiers (`specials/precision/products/fft/matmul_parity/
  linalg_parity/poly/einsum/nan/nanscan/random_parity/generator_parity/...`), and result-kind/error tiers
  (`iter/dtype_text/out_where/errors_full`, in `.Kinds`). **This is where new-op work lands.**
  - Two runners: `RunCorpus(file)` (strict everywhere) and **`RunHostLibmCorpus(file)`** — used by
    `unary/nan/precision/fft/numpy_f32_kernels`, hard-gated on Windows and **`Inconclusive` off-Windows** (their
    transcendental / FFT-twiddle / `Vector<T>` cells are win-amd64 CRT-libm and SIMD-width dependent). Every
    *portable* cell in those tiers is a deterministic NumSharp kernel, green on all platforms by construction.
  - **`FuzzRegression()`** enumerates `Fuzz/corpus/regressions/*.jsonl` at runtime and replays each. The subdir is
    created on demand (currently empty) — the nightly soak drops shrunk repros there to pin them forever.
- **`IndexOracleTests`** — advanced-indexing get/set (`index_curated` + `index_dtype` + `index_setter_dtype` + `index_random`).
- **`OracleSurfaceCoverageTests` / `OracleCoverageStrengthTests` / `Journey3TouchedOracleCoverageTests`** — coverage
  gates: every public `np`/`np.linalg`/`np.fft`/`np.random` method is classified; every ordinary op has ≥4 cases with
  ≥1 changing axis; all 186 journey3-touched callables have a direct case. A new unclassified API fails here.
- **`UndisposedIntermediateTests` / `NativeAllocationChokepointTests`** — **oracle-free leak gates** that reuse the
  corpus + `OpRegistry` to assert the buffer pool balances (zero-leak) and that no raw native-alloc site escapes the
  chokepoint allowlist. (`ScopeAudit`/`[TestCategory("ScopeAudit")]` is the shared pool-counter harness.)
- **`BlasBackendDeltaTests`** — replays only the ~1.7K BLAS-affected ordinary cases twice (managed vs OpenBLAS),
  deduplicates identical outcomes, byte-checks the real flips against NumPy on the pinned host.
- **`MetamorphicTests`** — NumPy-free invariants (round-trips / involutions), no oracle needed.
- **`HarnessSelfTests`** — proves the gate has teeth (BitDiff catches value/NaN/-0 diffs; the corpus is non-vacuous; the Shrinker reproduces a planted divergence). A green FuzzMatrix that skipped every case would fail here.
- **`OpRegistryRandomIsolationTests`** — asserts stateful random ops in the corpus don't mutate global `np.random` state.

Run one tier while iterating: `dotnet test --no-build -f net10.0 --filter "FullyQualifiedName~FuzzCorpusTests.Manip"`.

> **Known flake:** the full `TestCategory=FuzzMatrix` run may end with "Test host process crashed"
> (an intermittent `AccessViolation`) AFTER all tests report Passed. That's a teardown crash, not a
> failure — re-run the specific `FuzzCorpusTests` class (exit 0, no crash) to confirm green.

### The six oracles

All share one philosophy (**NumPy — or an independent scalar oracle — is truth; the corpus is committed; no Python
at test time**), but they are separate corpora + gates:

| # | Oracle | Generator → corpus | Gate |
|---|--------|--------------------|------|
| 1 | **Op** (the main one) | `gen_oracle.py` + `gen_nan_oracle.py` → `Fuzz/corpus/*.jsonl` | `FuzzCorpusTests` (`FuzzMatrix`) |
| 2 | **Advanced-indexing** | `gen_index_oracle.py` → `index_*.jsonl` | `IndexOracleTests` (`FuzzMatrix`) |
| 3 | **Decimal** (no NumPy analog) | `gen_decimal_oracle.cs` → `decimal_*.jsonl` | `FuzzCorpusTests.Decimal*` (`FuzzMatrix`) |
| 4 | **`.npy`/`.npz` format** | `gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip` | `IO/NpyOracleTests` (**`NpyOracle`**) |
| 5 | **`ndarray.flags`** | `gen_flags_oracle.py` → `NumSharp.Tests/Backends/corpus/flags_oracle.jsonl` | `Backends/FlagsOracleTests` |
| 6 | **Layout-parity** (numpy-internal view/stride/writeable modelling) | `gen_layout_parity_oracle.py` → `NumSharp.Tests/Backends/corpus/layout_parity_oracle.jsonl` | `Backends/LayoutParityOracleTests` |

Oracles 5 & 6 live in the **main test project** (`test/NumSharp.Tests/Backends/`), not the Oracle project — they gate
flags/view semantics rather than op values, and are NOT part of `TestCategory=FuzzMatrix`.

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
- **Deeper system map** (all generators, the corpus tiers, the six oracles, the leak/coverage gates, the harness classes)
  → `references/architecture.md`.

## Critical gotchas (learned the hard way)

- **Public-surface completeness is gated.** `OracleSurfaceCoverageTests` reflects `np`/`np.linalg`/
  `np.fft`/`np.random` and fails on a public method with no corpus key or explicit classification.
- **Pin `numpy==2.4.2`.** A different NumPy version can shift bytes and make the committed corpus wrong. Verify
  `python -c "import numpy; print(numpy.__version__)"` before regenerating.
- **The corpus diff is huge but harmless.** Case `id`s carry a global running counter (`{op}/{layout}/{dtype}/{n}`),
  so adding one job renumbers every following id. Expect a large `git diff` on `*.jsonl` — it's renumbering, not
  semantic churn.
- **Char has no NumPy dtype.** It rides the `uint16` proxy: `char_tier("<mode>")` re-runs your `gen_<mode>` with the
  Char pool and relabels `uint16 → char`. Add your op to a `gen_<mode>` whose `main()` branch calls `char_tier` (18
  of them — arith/divmod/comparison/unary×2/bitwise/reduce/scan/stat/manip/sort/tail/astype/where/logic/matmul/
  rounding/copyto) and Char coverage is automatic. `creation` and `conversion` append their own proxy
  rows, bringing the committed total to 20 Char-bearing files; a mode with no proxy call (e.g. `modf`, `place`) has none.
- **Decimal has no NumPy analog.** It rides the independent C# oracle `gen_decimal_oracle.cs` (naive scalar
  `System.Decimal`), regenerated via `dotnet run test/oracle/gen_decimal_oracle.cs`. If your op needs Decimal
  coverage, add it there too.
- **The generator resolves paths relative to `test/oracle/`** and writes into `test/NumSharp.Tests.Oracle/Fuzz/corpus/`.
  Run it from `test/oracle/` (or with that CWD). CI replays the committed corpus and never runs the generator.
- **OpRegistry's `default:` throws `NotSupportedException(op)`** — so a corpus op with no registered case fails the
  tier loudly. If a tier goes red immediately on a new op, you forgot (or mistyped) the `OpRegistry` case.
- **A new LAYOUT needs only the Python builder — there is NO `LayoutCatalog.cs`.** `FuzzCorpus.Reconstruct` rebuilds
  any operand view from the serialized `(dtype, shape, element-strides, offset, base-bytes)` descriptor, so the C#
  side is layout-agnostic. Add the `(base, view)` builder to `layout_catalog.py`, regenerate the affected tiers —
  done. (The "mirror it in `LayoutCatalog.cs`" line in `layout_catalog.py`'s own header is stale.)
- **Non-array results have their own wiring.** An op that returns a dtype, a string, a scalar, or a tuple is NOT
  registered in `OpRegistry.Apply`; it goes through `OpRegistry.Kinds.cs` (`ApplyTuple`/`ApplyDtype`/`ApplyText`)
  paired with `gen_oracle.py`'s `gen_dtype_text`/`gen_iter`/`gen_out_where`, and the generator marks it with
  `expected.kind` (`dtype`/`text`/`tuple`/`scalar`) via `_arr_expected(kind=…)`/`_tuple_expected`. See
  `references/architecture.md` → "Result kinds".
- **Registering an op also enters the ZERO-LEAK gate.** `UndisposedIntermediateTests` replays the whole corpus
  through `OpRegistry` with every result disposed and asserts the buffer pool's takes==returns. The `KnownEscapes`
  registry is now **empty** — every op is gated at zero — so an op that strands a pooled intermediate (undisposed
  scratch, an orphaned `AsGeneric`, an operator leftover) turns this gate red **even when its values are bit-exact**.
  Fix the leak (`[NDScoped]`/dispose the intermediate), don't add a `KnownEscapes` entry. `NativeAllocationChokepointTests`
  separately fails if your op adds a raw `NativeMemory.*`/`Marshal.AllocHGlobal` site outside the chokepoint allowlist —
  route scratch through the pools.
- **Some tiers go `Inconclusive` off-Windows, not red.** `unary/nan/precision/fft/numpy_f32_kernels` (via
  `RunHostLibmCorpus`) and the host-pinned `matmul_parity`/`random_parity_host`/`generator_parity_host` record bytes
  reproducible only on win-amd64 (CRT libm, SIMD widths, the exact BLAS build). Off-Windows they assert `Inconclusive`,
  never a failure — expected, not a problem. Regenerate on your host to gate against it (see `references/triage.md`).

## References

- `references/add-op.md` — the detailed add-an-op playbook with a worked example (params, char, OpRegistry patterns).
- `references/regenerate.md` — the full regeneration command matrix + dtype/layout coverage model.
- `references/triage.md` — divergence handling: bit-diff → shrink → MisalignedRegistry vs OpenBugs.
- `references/architecture.md` — the complete system map (generators, tiers, the six oracles, the leak/coverage gates, the harness).
