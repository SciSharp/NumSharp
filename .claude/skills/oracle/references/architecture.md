# Oracle system map

The full narrative is the project `.claude/CLAUDE.md` → "Differential-Fuzz Pipeline (NumPy oracle)". This is the
condensed structural map. **Six** independent oracles share one philosophy: **NumPy (or an independent scalar oracle)
is the source of truth; the corpus is committed; no Python at test time.** Four (op, advanced-indexing, Decimal, npy)
plus the two sibling oracles (flags §6, layout-parity §7) that live in the *main* test project. On top of the value
gate, the same op corpus + `OpRegistry` are reused by an **oracle-free leak/allocation gate** (§8).

Rough scale at 2026-09 (drifts per regeneration): **68 corpus files / ~118K rows**, **~371 distinct op keys**,
`gen_oracle.py` **~7,400 lines** across **~45 modes**.

## 1. The op oracle (the main one)

**Generate** — `test/oracle/gen_oracle.py` (~7,400 lines) + the standalone `gen_nan_oracle.py` (→ `nan.jsonl`, its
own numbering). Structure of `gen_oracle.py`:
- Value pools + layout builders imported from `layout_catalog.py`.
- One `gen_<mode>(dtypes, layout_names)` per op family. Each loops `layout × dtype`, builds `(base, view)`, runs a
  `jobs` list of `(opname, params, lambda)`, and appends a case `{id, op, params, operands:[{dtype,shape,strides,
  offset,bufferSize,buffer(hex)}], expected:{dtype,shape,buffer(hex)}, layout, valueclass}`.
- `char_tier(mode)` re-runs the relevant `gen_<mode>` with the Char pool and relabels `uint16 → char`.
- `main()` dispatches `mode → gen_<mode>(...) + char_tier(mode) → write_jsonl(corpus/<mode>.jsonl)`.

**Replay** — `test/NumSharp.Tests.Oracle/Fuzz/`:
- `FuzzCorpus.cs` — parses each JSONL line and rebuilds the EXACT `NDArray` view from `(dtype, shape, strides,
  offset, bytes)`, so C# sees byte-identical operands.
- `OpRegistry{,.Kinds,.Generator}.cs` — `Apply(op, params, ops)` maps opname → the NumSharp call. Pairs 1:1 with
  the generators: `.cs` = array ops (~365 cases), `.Kinds.cs` = dtype/text/tuple/scalar results, `.Generator.cs` =
  PCG64/`default_rng` ops.
- `BitDiff.cs` — bit-exact compare: NaN tokenized (payload/sign non-contractual) EXCEPT the complex-unary ops whose
  NaN **sign** IS contractual (`Compare(nanBitExact)` + `DiffHasSignFlip`, list in `ComplexNanContractOps`); Decimal
  by canonical value. `Shrinker.cs` — minimizes a failure to a 1-element repro.
- `FuzzCorpusTests{,.Kinds}.cs` — one `[FuzzMatrix]` `[TestMethod]` per corpus file, each calling
  `RunCorpus("<tier>.jsonl")` — or **`RunHostLibmCorpus`** for `unary/nan/precision/fft/numpy_f32_kernels` (strict on
  Windows, `Inconclusive` elsewhere; their libm/SIMD-width cells have no cross-platform byte contract). `.Kinds.cs`
  carries the non-array tiers + comparators. A per-file `MinCases` floor rejects a silently truncated regeneration.
  `FuzzRegression()` replays every `corpus/regressions/*.jsonl` (created on demand by the soak; currently empty).
- `OracleSurfaceCoverageTests.cs` — reflects every public `np`/`np.linalg`/`np.fft`/`np.random`
  method and rejects an unclassified surface addition.
- `Journey3TouchedOracleCoverageTests.cs` — pins the conservative 186-callable master→journey3
  production-file inventory and requires a direct committed case for all 186.
- `OracleCoverageStrengthTests.cs` — rejects one-row coverage theatre: every ordinary op key needs
  at least four cases and at least one changing axis (layout/params/dtype/valueclass/outcome); the
  separate indexing corpus keeps explicit matrix floors.
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

Beyond the elementwise/reduce matrices, `gen_oracle.py` (+ `gen_nan_oracle.py`) emits several targeted tiers:
- `creation` / `conversion` / `multioutput` — deterministic zero-operand creators (including
  `empty*` through post-allocation initialization), as*/buffer/text/file-artifact conversions
  (including finite-check errors and verbatim `savetxt`), and full tuple arity/slot coverage.
- `specials` — IEEE nan/±inf/±0/subnormal/max forced through math/reduce/scan/matmul across float widths.
- `nan` (standalone `gen_nan_oracle.py`) — every UNARY op with a reachable NaN output over the full special-value
  grid (finite/±0/±inf/**both NaN signs**): complex128 unary held BIT-EXACT on the NaN sign (`ComplexNanContractOps`),
  float widths held to producing *a* NaN (value, tokenized) + byte-exact non-NaN components. `nanscan` = nancumsum/
  nancumprod; `nanreduce` = the nan-aware reductions.
- `precision` — truthful-vs-precise: each case carries a THIRD buffer `expected.truth` (correctly-rounded,
  mpmath/`Fraction`), consulted ONLY to adjudicate which side lost precision on a NumPy divergence (branches P1–P3).
  Bit-exact-to-NumPy always passes without truth being read.
- `products` — the CBLAS product family (inner/vdot/vecdot/matvec/vecmat/tensordot/multi_dot/matrix_power/cross/
  cov/corrcoef). `poly` — the portable polynomial family (poly/polyval/vander/polyder/polyint/poly{add,sub,mul,div}/
  poly1d). `einsum` — the subscript grammar + integer/small-exact-float contractions + the view path.
- `fft` — the whole `np.fft.*` surface (float64/complex128 bit-exact; float32/float16 the documented complex64
  dtype-only divergence, values verified after up-cast).
- `numpy_f32` — writes `numpy_f32_kernels` + `numpy_f64_kernels`: the exp/log/sin/cos/tanh/rad2deg/deg2rad kernels
  NumSharp ports from NumPy itself, held BIT-EXACT (carved out of the ~ULP excuse).
- `random_parity` (+`_host`) — seeded MT19937 stream bytes; `generator_parity` (+`_host`) — the PCG64
  `default_rng`/`bytes`/`random_integers` streams; `matmul_parity` / `linalg_parity` (+`.host`) — np.dot/np.matmul and
  the LU/eigen/SVD/QR/Cholesky family, byte parity through the optional `NumSharp.Interop.OpenBLAS` engine.
- `BlasBackendDeltaTests` — managed/backend two-pass replay of the ~1.7K BLAS-affected ordinary cases (floor 1400).
  It deduplicates identical outcomes by threw/result state + dtype + shape + bytes and byte-checks only the real
  flips against NumPy on the pinned host (both counts are printed at runtime, not asserted).

**Host-gated (Inconclusive off the authoring host, never red — see `triage.md`):** the pins `matmul_parity` /
`linalg_parity` / `random_parity_host` / `generator_parity_host` (checked via `MatmulParityPin` by BLAS-binary SHA-256
+ core name, or an OS check), AND the broader `RunHostLibmCorpus` tiers `unary` / `nan` / `precision` / `fft` /
`numpy_f32_kernels` (win-amd64 CRT libm + host SIMD reduction widths). Every *portable* cell in the libm tiers is a
deterministic NumSharp kernel, green on every platform by construction; only the libm/SIMD cells are host-gated.

## 6. The `ndarray.flags` oracle (sibling, in the MAIN test project)

- `test/oracle/gen_flags_oracle.py` → `test/NumSharp.Tests/Backends/corpus/flags_oracle.jsonl` (~1100 cases of REAL
  NumPy 2.4.2 flag records over a 13-dtype × layout sweep + a result-flags matrix for order= producers / stacking /
  reductions / creation). Replayed by `Backends/FlagsOracleTests.cs` — a 1:1 twin of the generator's recipe tokens.
  Gates `C/F/O/W/A/X` (contiguity, owndata, writeable, aligned, writebackifcopy) rather than op VALUES, so it is NOT
  in `TestCategory=FuzzMatrix`.

## 7. The layout-parity oracle (sibling, in the MAIN test project)

- `test/oracle/gen_layout_parity_oracle.py` → `test/NumSharp.Tests/Backends/corpus/layout_parity_oracle.jsonl`.
  Replayed by `Backends/LayoutParityOracleTests.cs`. It models numpy-INTERNAL view/stride/writeable/shares-memory
  results (the `w`/`shares`/`samebase` fields) so NumSharp's view semantics stay byte-for-byte NumPy. Also not in
  `FuzzMatrix`.

## 8. The oracle-free leak / allocation gate (reuses the op corpus)

The op corpus + `OpRegistry` are reused for a DIFFERENT claim — no NumPy reference, no value check — under
`test/NumSharp.Tests.Oracle/Fuzz/`:
- `UndisposedIntermediateTests.cs` — a `[FuzzMatrix]` gate that replays the whole corpus through `OpRegistry` with
  every result disposed and asserts the buffer pool's takes==returns. A surplus take is an undisposed intermediate; a
  surplus return is an out-of-pool result buffer. The `KnownEscapes` registry is now **EMPTY** — every op is gated at
  ZERO — so an op that leaks a pooled buffer fails here even when its values are bit-exact. `[DoNotParallelize]`
  (reads process-global pool counters, shared with `ScopeAudit.cs`).
- `NativeAllocationChokepointTests.cs` — the STATIC complement: every raw `NativeMemory.*` / `Marshal.AllocHGlobal` /
  `VirtualAlloc*` site in `NumSharp.Core` must be a known chokepoint (allowlist pins file → site count). A new raw
  allocation fails until it is routed through the pools or consciously allowlisted. Needs the source tree
  (bin-only run → `Inconclusive`).
- `OpRegistryRandomIsolationTests.cs` — asserts the corpus's stateful random ops don't mutate global `np.random` state.

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
- The test `.csproj` has a recursive glob (`Fuzz\corpus\**\*.jsonl`, `PreserveNewest`) that copies the corpus — incl.
  any `regressions/` subdir — into the build output; `RunCorpus` reads them from there. So a regeneration is only
  "live" after a `dotnet build`.
- The whole point: **CI runs `dotnet test --filter TestCategory=FuzzMatrix` with no Python** — it replays committed
  bytes. Regenerating + committing the `.jsonl` is the entire delivery of new coverage.
