# NumPy Differential Fuzzer (Plan A)

Proves every NDIter-backed operation produces **bit-identical** output to NumPy 2.4.2 across the
full input space — caught systematically, not by hand-picked cases. The motivating failure (the
cast saturate-vs-wrap bug, latent in `where`/`copyto`/`concatenate`) must be impossible to ship again.

## How it works

NumPy is the oracle. Python (`test/oracle/`) generates a **committed, bytes-exact corpus**; the C# harness
**replays the operand bytes** and bit-compares — *no Python at test time, none in CI*.

```
test/oracle/                         corpus generators (NumPy 2.4.2)
  layout_catalog.py                  the layout builders (single-array, pairwise, where-triple)
  gen_oracle.py                      deterministic matrices (astype/binary/comparison/unary/reduce/where/place);
                                     per-mode dtype axes widened to ALL_DTYPES; Char WOVEN into every tier
                                     via the uint16 proxy (char_tier) — relabelled uint16->char, bytes intact
  gen_decimal_oracle.cs              INDEPENDENT C# oracle for Decimal (no NumPy analog): naive scalar
                                     System.Decimal math -> decimal_{unary,binary,reduce,scan,power,
                                     varstd,matmul,astype,stat,where,sort,manip}.jsonl (12 tiers, 695 cases)
  fuzz_random.py                     seeded random fuzzer (13 dtypes × unary/binary/comparison/where/
                                     flat-reduce/astype kinds; NumSharp-producible layouts)
test/NumSharp.UnitTest/Fuzz/
  FuzzCorpus.cs                      reconstructs EXACT NDArray views from (dtype,shape,strides,offset,bytes)
  BitDiff.cs                         bit-exact compare; NaN tokenized; Decimal tokenized by canonical VALUE
                                     (scale-insensitive: 1.0m == 1.00m); ULP helpers (documented near-misses)
  OpRegistry.cs                      op-name -> NumSharp call
  MisalignedRegistry.cs              the explicit, documented set of intended/known divergences
  Shrinker.cs                        minimizes a failing element-wise case to a 1-element repro
  FuzzCorpusTests.cs                 one [FuzzMatrix] test per corpus file
  corpus/*.jsonl                     committed, copied to test output
```

A divergence is one of: **bit-exact** (passes), a **documented difference** in `MisalignedRegistry`
(excused + logged, never silent), or a **failure** (any unknown divergence — the gate is red).

### Gate semantics — what is (and is not) asserted

- **Value / dtype / shape parity** (the ordinary case): the replay checks result **dtype**
  (NEP50 promotion), then result **shape** (broadcasting), then the raw result **bytes**
  (bit-exact; NaN tokenized, Decimal compared by canonical value). Any divergence not classified
  by `MisalignedRegistry` fails the tier.
- **Error parity** (`errors.jsonl` and any corpus case flagged `expects_throw`): NumPy raised at
  generation time, so NumSharp must throw **something** — a throw of ANY exception type passes;
  a normal return is the divergence. **Exception type/message parity is NOT asserted** (NumSharp
  does not mirror Python's exception taxonomy). The reverse direction is gated on every ordinary
  case: NumSharp throwing where NumPy returned a result is a `Threw` divergence — a failure
  unless a registry branch excuses it.
- **Index oracle** (`IndexOracleTests`): compares result shape, values, and **which side raised**
  — if both NumPy and NumSharp raise, the case passes regardless of exception type. NumPy's
  exception name is carried in the corpus (`err`) for failure messages only, never for parity.
- **Excused divergences are logged, never silent**: every case a `MisalignedRegistry` branch
  classifies is counted and printed per tier even when the test passes —
  `[<file>] documented Misaligned divergences excused: <n>x <reason>; …` — so growth in an
  excused class stays visible in the test output. Anything unclassified is red.

## Regenerating the corpus

```bash
python test/oracle/gen_oracle.py astype_full      # 13x13 dtypes x 26 layouts
python test/oracle/gen_oracle.py binary           # add/sub/mul/divide x NEP50 pairs x pairwise layouts
python test/oracle/gen_oracle.py divmod_power     # floor_divide/mod (bit-exact, F1) + complex power (Misaligned)
python test/oracle/gen_oracle.py comparison       # ==,!=,<,>,<=,>=
python test/oracle/gen_oracle.py unary            # negate/abs/sqrt/trig/exp/log/...
python test/oracle/gen_oracle.py reduce           # sum/prod/min/max/mean/std/var/argmax/argmin/all/any
python test/oracle/gen_oracle.py where            # np.where(cond,x,y)
python test/oracle/gen_oracle.py place            # np.place(arr,mask,vals)
python test/oracle/gen_oracle.py matmul           # T8 linalg: matmul/dot/outer (gufunc shapes, C/F layouts)
python test/oracle/gen_index_oracle.py            # the four index_* corpora (seed pinned 20240626)
python test/oracle/fuzz_random.py 1234 2000 random_smoke.jsonl
dotnet run test/oracle/gen_decimal_oracle.cs      # Decimal tiers (independent C# System.Decimal oracle)
```

The block shows the most-used modes; the full `gen_oracle.py` mode list (one tier file per mode) is
`smoke astype_full binary divmod_power comparison unary reduce where place matmul rounding bitwise
unary_extra nanreduce scan stat logic modf manip sort tail params aliasing copyto errors groupa`.
Regeneration is deterministic: rerunning an untouched mode must produce a zero corpus diff.

Char rides the applicable NumPy modes automatically (`char_tier` appends uint16-proxy cases
relabelled to `char` into 18 tier files — arith/divmod/comparison/unary×2/bitwise/reduce/scan/stat/
manip/sort/tail/astype/where/logic/matmul/rounding/copyto); there is no separate `char` mode.
Decimal is the one dtype with no NumPy analog, so it has its own C# generator (the last line
above) rather than a `gen_oracle.py` mode.

Then `dotnet build` (copies the corpus to output) and run:

```bash
dotnet test --filter "TestCategory=FuzzMatrix"          # the differential gate (runs every CI)
dotnet test --filter "TestCategory=OpenBugs&ClassName~FuzzCorpusTests"   # known-failing repros
```

The nightly **soak** (`.github/workflows/fuzz-soak.yml`) sweeps seeds for ~1M cases/night; a
divergence prints a shrunk minimal repro — copy it into `corpus/regressions/` so `FuzzRegression`
pins it on every CI thereafter.

## Documented divergence ledger (Misaligned / known bugs)

Two mechanisms, both loud: a **registry excuse** (`MisalignedRegistry` classifies the divergence at
replay time — counted + printed per tier) and a **corpus carve** (the cell is deliberately absent
from the green corpus, with a comment at the carve site and an `[OpenBugs]` pin reproducing the
bug). Every excuse branch is scoped to its exact (op set × dtype × kind) cell — a regression in a
neighbouring cell fails the gate — and `MisalignedRegistryTightnessTests` (OpenBugs.FuzzGate.cs)
pins each scope with paired not-excused/still-excused cases. "Hits" = excused-case count in the
2026-07-07 full-gate sweep (83/83 green, net10.0+net8.0); a 0-hit branch is live code kept as a
guard and a removal candidate once confirmed dead.

### Table 1 — live `MisalignedRegistry` excuse branches

**Intended / algorithmic differences (permanent):**

| Excuse class | Scope | Hits |
|---|---|---|
| NEP50 weak-scalar: 0-D operand promoted weakly | any multi-operand op × Dtype kind, 0-D operand present | 261 |
| unary ~ULP (transcendental/magnitude algorithm difference) | single-operand × Value, every diff ≤2 ULP | 563 |
| complex unary within 3 ULP (full NumPy-algorithm port) | complex unary × Value, ≤3 ULP | 11 |
| complex cos/sin/arccos/sinh/cosh pathological edge (NaN zero-sign / subnormal / overflow boundary) | those 5 ops × complex × Value | 0 |
| complex division ~1 ULP (npy_cdivide vs System.Numerics.Complex) | divide × complex × Value, ≤2 ULP | 17 |
| complex add/subtract within 2 ULP (FMA contraction) | add/subtract × complex × Value, ≤2 ULP | 0 |
| complex multiply cancellation / ~ULP at element magnitude (#12) | multiply × complex × Value, ≤16 element-magnitude ULP | 16 |
| complex power ~ULP / gross inf-NaN edge (Complex.Pow vs npy_cpow) (F5, ledger L6) | power × complex × Value, ≤512 element-magnitude ULP or non-finite | 30 |
| reduction summation/two-pass precision (algorithm order) | sum/mean/std/var/prod × float-family result (Half/Single/Double/Complex) × Value | 401 |
| complex reduction/scan NaN ordering/propagation differs | reduce+cumsum/cumprod × complex × Value, diffs must contain a NaN token | 35 |
| decimal std last digit (independent 28-digit sqrts) (ledger L7) | std × Decimal × Value, ≤1 unit in the 28th significant digit | 4 |

**Known bugs (tracked for fix — remove the branch when fixed):**

| Excuse class | Scope | Hits | Task |
|---|---|---|---|
| floor_divide/mod(float16): NDDivision has no Half path | float16 operand/result × Value/Threw | 38 | |
| power(uint64,int64): NEP50 →float64 not applied; int-power path throws | that dtype pair × Threw | 8 | |
| power(*,float16): result widened past float16 | power × Half-expected × Dtype | 0 | |
| dot(int8): Sum(int8)→int8 IL reduction kernel missing | dot × int8·int8 × Threw | 0 | |
| where(narrow-int) scalar-broadcast: NDExpr zero-push unsupported | where × {i8,u8,i16,u16} operand × Threw | 0 | |
| cumprod(size-1 int): skips NEP50 accumulator widening | cumprod × Dtype, operand element-count ≤1 | 14 | |
| modf(float16/int): no Half kernel, no int→float64 promotion | modf × dtype ∉ {f32,f64} × Threw | 32 | |
| unary hyperbolic/inverse-trig/angle: no Half kernel | sinh…arctan × {bool,i8,u8,f16} (+deg2rad/rad2deg×c128) × Threw | 0 | |
| unary preserve-dtype pending: square/floor/ceil/trunc widen int→float64 | those 4 ops × Dtype | 78 | F3b |
| reduction result dtype differs (NEP50 accumulator / complex→real) | reductions × Dtype | 239 | #10 |
| axis-reduction NaN propagation: axis SIMD min/max skips NaN (flat fixed) | min/max × axis≠null × all-NaN diffs | 8 | #10 |
| bool min/max along axis diverges | min/max × Boolean × Value | 0 | #10 |
| complex 1-D axis reduction throws (NDCoordinatesAxisIncrementor) | (nan)reductions × complex 1-D × Threw | 8 | #10 |
| nan-reduction family: shape ([1] vs scalar, keepdims dropped) / value (masking·count·order) / dtype / nanmedian propagates NaN / empty throws | nan* ops, per-kind branches | 885/526/184/176/4 | #10 |
| median/percentile/quantile: ±inf-NaN interpolation · float interp precision · int-axis gross error | QuantileEngine ops × Value, three branches | 72/40/28 | |
| average: summation-order precision (pairwise vs naive) | average × Value | 30 | |
| isclose: F-contiguous/complex strided pairing | isclose × complex-operand-present × Value | 1 | |
| ops vs raw NumPy stride/offset representation (offset≠0, junk size-1 strides) | corpus-only reconstructions unreachable via the API | n/a | #11 |

### Table 2 — corpus carves (each pinned under `[OpenBugs]`)

| Carve (generator site) | Cell | Pin |
|---|---|---|
| `char_tier` partner filter | Char × {uint8, bool} arithmetic/comparison/bitwise (promote(Char,Byte)→Byte truncation; (Boolean,Char) kernel missing) | `OpenBugsCharTests.Char_Add_Byte_*`, `Char_Compare_Byte_*`, `Char_BitwiseAnd_Bool_KeyNotFound` |
| `_CHAR_UNARY_OPS` / `_CHAR_DIVMOD_OPS` | reciprocal(char)→Double; power(char,·) crash/Double | `Char_Reciprocal_ReturnsDouble`, `Char_Power_Single_ReturnsDouble`, `Char_Power_ScalarChar_Crashes` |
| `char_tier "bitwise"` (invert absent) | invert(char) N≥16 SIMD → NotSupportedException | `Char_Invert_LargeN_NotSupported` |
| `char_tier "matmul"` filters `(dot,(4,))` | dot(char) 1-D·1-D → "Sum not supported for Char" (ledger L9) | `OpenBugsFuzzGapsTests.Dot_Char_1D_Throws` |
| `CLIP_DTYPES` excludes bool | clip(bool) non-contiguous → NotSupportedException (contiguous works) | `OpenBugsDtypeCoverageTests.Clip_Bool_*` |
| `gen_round` dec ∈ {0,1,2} only | round_ dec=-1 (int throws / float wrong) | `Round_NegativeDecimals_Broken` |
| `gen_round` skips float16 dec≥1 | round_(float16) fractional diverges | `Round_Float16_Fractional_Diverges` |
| `ROUND_DTYPES` excludes bool | round_(bool) → Double, NumPy → float16 (ledger L2) | `OpenBugsFuzzGapsTests.Round_Bool_Dtype_Diverges` |
| `gen_round` complex at dec=0 only | round_(complex, dec≠0) is a no-op (ledger L3) | `OpenBugsFuzzGapsTests.Round_Complex_NonzeroDecimals_NoOp` |
| `TRACE_DTYPES` excludes uint8 | trace(unsigned) → Int64, NumPy → uint64 | `Trace_Unsigned_WrongResultDtype` |
| `gen_unary` iscomplex/isreal: real dtypes × contiguous only | complex input ignores imag; strided real garbage | `IsComplex_IgnoresImaginaryPart`, `IsReal_IgnoresImaginaryPart` |
| `gen_unique` contiguous+finite | unique on raw-offset views (#11) + inf/NaN complex ordering | documented at carve site (no pin — unreachable via API) |
| `ALIAS_DTYPES` excludes complex128 | a·a self-multiply catastrophic cancellation (NumSharp matches NumPy *scalar*; NumPy's array ufunc disagrees with itself) | documented non-bug |
| `gen_nanquantile` finite+NaN (no inf) | percentile interpolation across ±inf is ill-defined (inf−inf) | documented out-of-scope |

**FIXED on this branch or before it** (classifier branch/carve removed — the matrix now verifies
these bit-exact): complex→bool imaginary drop · floor_divide/mod integer ÷0/±inf/signed-floor (F1)
· NaN `<=`/`>=` (F2) · transcendental width-based promotion (F3a) · negative(uint) + integer
reciprocal (F4) · bool arithmetic True+True (F6) · size-1 result collapse (F7) · complex np.where
zero-push · maximum/minimum/fmax/fmin direct ufuncs + NaN-propagating clip/out= SIMD ·
exp2 malformed-IL crash (W3-C) · power(float16) scalar-broadcast crash (W1-B) ·
**invert(float/complex/decimal) illegal-instruction crash** (guard @ `Default.Invert.cs`, pinned by
5 always-run tests in `FuzzGateRegressionTests`) · **convolve(complex) discarded the imaginary
dimension** + int64/decimal/bool convolve accumulator (ledger L5, @737c59d6) · **all/any Half+Complex
ignored `Shape.offset`** (ledger L4, @7804b2ad) · **round_(char)→Double** (ledger L8, @1a9cfa9f).

### Decimal (independent oracle — no NumPy analog)

`Decimal` rides an **independent C# oracle** (`gen_decimal_oracle.cs`, naive scalar `System.Decimal`;
`std` is oracled by an independent Newton decimal sqrt, NOT NumSharp's `DecimalMath`) across
unary / binary / reduce (flat + **axis×keepdims** + **empty**) / scan / power (int exponents
**−2…3**) / var / std / matmul / astype (decimal↔int·float·**bool·int16·uint64**) / stat (clip +
median/ptp/percentile/quantile) / where / sort / manip × 13 single + 9 pair layouts — **695 cases
across 12 tiers, all green** except the 15 registry-excused cells (11 argmax/argmin/count_nonzero
result-dtype + 4 std-last-digit, both in Table 1). The one decimal-adjacent finding of the
remediation: `DecimalMath.Pow` matches the exact reciprocal-of-product oracle for negative integer
exponents by value — zero divergence.
