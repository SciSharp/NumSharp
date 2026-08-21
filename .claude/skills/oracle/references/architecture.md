# Oracle system map

The full narrative is the project `.claude/CLAUDE.md` → "Differential-Fuzz Pipeline (NumPy oracle)". This is the
condensed structural map. Four independent oracles share one philosophy: **NumPy (or an independent scalar oracle)
is the source of truth; the corpus is committed; no Python at test time.**

## 1. The op oracle (the main one)

**Generate** — `test/oracle/gen_oracle.py` (~5,700 lines). Structure:
- Value pools + layout builders imported from `layout_catalog.py`.
- One `gen_<mode>(dtypes, layout_names)` per op family. Each loops `layout × dtype`, builds `(base, view)`, runs a
  `jobs` list of `(opname, params, lambda)`, and appends a case `{id, op, params, operands:[{dtype,shape,strides,
  offset,bufferSize,buffer(hex)}], expected:{dtype,shape,buffer(hex)}, layout, valueclass}`.
- `char_tier(mode)` re-runs the relevant `gen_<mode>` with the Char pool and relabels `uint16 → char`.
- `main()` dispatches `mode → gen_<mode>(...) + char_tier(mode) → write_jsonl(corpus/<mode>.jsonl)`.

**Replay** — `test/NumSharp.Tests.Oracle/Fuzz/`:
- `FuzzCorpus.cs` — parses each JSONL line and rebuilds the EXACT `NDArray` view from `(dtype, shape, strides,
  offset, bytes)`, so C# sees byte-identical operands.
- `OpRegistry.cs` — `Apply(op, params, ops)` maps opname → the NumSharp call. Pairs 1:1 with `gen_oracle.py`.
- `BitDiff.cs` — bit-exact compare (NaN tokenized, Decimal by value). `Shrinker.cs` — minimizes a failure.
- `FuzzCorpusTests.cs` — one `[FuzzMatrix]` `[TestMethod]` per corpus file, each calling `RunCorpus("<tier>.jsonl")`.
  A per-file `MinCases` floor rejects a silently truncated regeneration.
- `MisalignedRegistry.cs` — the excused, documented divergences (`Classify(...)` returns a reason string; scoped
  tight and pinned by `OpenBugs.FuzzGate.cs`).

**Result kinds & error parity** — the corpus is not only single arrays:
- `expected.kind` (`array` default / `scalar` / `dtype` / `text` / `tuple`) selects the comparator in
  `FuzzCorpusTests.Kinds.cs`; non-array ops are dispatched by `OpRegistry.Kinds.cs`
  (`ApplyTuple`/`ApplyDtype`/`ApplyText`), paired with the `gen_iter`/`gen_dtype_text`/`gen_out_where` generators.
  A `tuple` asserts ARITY first then every slot; a `dtype` compares by NumPy dtype name (how the NEP50 promotion
  table itself is gated); `text` is verbatim (the array-printing port).
- Error parity has two strengths: `errors.jsonl` (weak "threw something") and `errors_full.jsonl` /
  `error:{type,text}` (NumPy's exception TYPE + verbatim MESSAGE, via `CheckError`/`ErrorTypeMap`).

## 2. The advanced-indexing oracle

- `test/oracle/gen_index_oracle.py` → `index_curated` / `index_dtype` / `index_setter_dtype` / `index_random` tiers
  (getter/setter over base recipes, portable token encoding). Replayed by `IndexOracleTests.cs` (also `[FuzzMatrix]`)
  — it compares result shape, values, and which-side-raised.

## 3. The Decimal oracle (no NumPy analog)

- `test/oracle/gen_decimal_oracle.cs` — an INDEPENDENT C# oracle using naive scalar `System.Decimal` math, since
  NumPy has no 128-bit decimal. Emits `decimal_{unary,binary,reduce,scan,power,varstd,matmul,astype,stat,where,
  sort,manip}.jsonl`, replayed by the same `FuzzCorpusTests` machinery.

## 4. The `.npy`/`.npz` format oracle (separate corpus + gate)

- `test/oracle/gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip` (REAL `np.save`/`savez` output + a manifest).
- Replayed by `IO/NpyOracleTests.cs` under `TestCategory=NpyOracle` — the claim is stronger: NumSharp's writer must
  be BYTE-IDENTICAL to `np.save`, not merely readable. Reverse interop (NumPy reading NumSharp) is the manual gate
  `python test/oracle/verify_npy_interop.py`.

## 5. Specialized value / parity tiers (still the op corpus, replayed by `FuzzCorpusTests`)

Beyond the elementwise/reduce matrices, `gen_oracle.py` emits several targeted tiers:
- `specials` — IEEE nan/±inf/±0/subnormal/max forced through math/reduce/scan/matmul across float widths.
- `precision` — truthful-vs-precise: each case carries a THIRD buffer `expected.truth` (correctly-rounded,
  mpmath/`Fraction`), consulted ONLY to adjudicate which side lost precision on a NumPy divergence (branches P1–P3).
  Bit-exact-to-NumPy always passes without truth being read.
- `products` — the first value gate for the CBLAS product family (inner/vdot/vecdot/matvec/vecmat/tensordot/
  multi_dot/matrix_power).
- `fft` — the whole `np.fft.*` surface (float64/complex128 bit-exact; float32/float16 the documented complex64
  dtype-only divergence, values verified after up-cast).
- `numpy_f32` — writes `numpy_f32_kernels` + `numpy_f64_kernels`: the exp/log/sin/cos/tanh/rad2deg/deg2rad kernels
  NumSharp ports from NumPy itself, held BIT-EXACT (carved out of the ~ULP excuse).
- `random_parity` / `random_parity_host` — seeded MT19937 stream bytes; `matmul_parity` — np.dot/np.matmul byte
  parity through the optional `NumSharp.Interop.OpenBLAS` engine.

**Host-pinned:** `matmul_parity` and `random_parity_host` record bytes reproducible only on the authoring host
(BLAS build + kernel + thread count, or the win-amd64 CRT libm), so they assert `Inconclusive` — never red —
elsewhere (see `triage.md`).

## Other harness pieces

- `MetamorphicTests.cs` — NumPy-free invariants (round-trips / involutions / identities); catch bugs the oracle
  can't (no reference needed).
- `HarnessSelfTests.cs` — proves the harness has teeth (BitDiff actually detects value/NaN/-0 diffs; the corpus is
  non-vacuous; the Shrinker reproduces a planted divergence). Also a `[FuzzMatrix]` gate class.
- `OpenBugs.FuzzGate.cs` — `MisalignedRegistryTightnessTests` (each excuse branch pinned from both sides — a gross
  regression in the neighbouring cell must NOT be excused) + `FuzzGateRegressionTests` (real bugs the tightening
  exposed, fixed-in-src or pinned `[OpenBugs]`).
- `fuzz_random.py` — the nightly-soak seeded fuzzer (`.github/workflows/fuzz-soak.yml`), ~1M fresh cases/night;
  shrunk failures get pinned under `Fuzz/corpus/regressions/`.

## Where the corpus lives and how it reaches tests

- Generators write to `test/NumSharp.Tests.Oracle/Fuzz/corpus/*.jsonl` (path resolved relative to `test/oracle/`).
- The test `.csproj` has a glob that copies `corpus/*.jsonl` into the build output; `RunCorpus` reads them from
  there. So a regeneration is only "live" after a `dotnet build`.
- The whole point: **CI runs `dotnet test --filter TestCategory=FuzzMatrix` with no Python** — it replays committed
  bytes. Regenerating + committing the `.jsonl` is the entire delivery of new coverage.
