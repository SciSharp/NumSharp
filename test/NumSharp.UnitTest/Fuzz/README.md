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
                                     varstd,matmul,astype,stat,where,sort,manip}.jsonl (12 tiers, 579 cases)
  fuzz_random.py                     seeded random fuzzer (NumSharp-producible layouts)
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
python test/oracle/gen_index_oracle.py            # the three index_* corpora (seed pinned 20240626)
python test/oracle/fuzz_random.py 1234 2000 random_smoke.jsonl
dotnet run test/oracle/gen_decimal_oracle.cs      # Decimal tiers (independent C# System.Decimal oracle)
```

The block shows the most-used modes; the full `gen_oracle.py` mode list (one tier file per mode) is
`smoke astype_full binary divmod_power comparison unary reduce where place matmul rounding bitwise
unary_extra nanreduce scan stat logic modf manip sort tail params aliasing copyto errors groupa`.
Regeneration is deterministic: rerunning an untouched mode must produce a zero corpus diff.

Char rides every NumPy mode automatically (`char_tier` appends uint16-proxy cases relabelled to
`char`); there is no separate `char` mode. Decimal is the one dtype with no NumPy analog, so it has
its own C# generator (the last line above) rather than a `gen_oracle.py` mode.

Then `dotnet build` (copies the corpus to output) and run:

```bash
dotnet test --filter "TestCategory=FuzzMatrix"          # the differential gate (runs every CI)
dotnet test --filter "TestCategory=OpenBugs&ClassName~FuzzCorpusTests"   # known-failing repros
```

The nightly **soak** (`.github/workflows/fuzz-soak.yml`) sweeps seeds for ~1M cases/night; a
divergence prints a shrunk minimal repro — copy it into `corpus/regressions/` so `FuzzRegression`
pins it on every CI thereafter.

## Documented divergence ledger (Misaligned / known bugs)

All are excused + logged by `MisalignedRegistry`; intended differences stay, real bugs are tracked
for fix (remove the classifier branch + corpus tag when fixed).

| Class | Disposition | Task |
|-------|-------------|------|
| `complex -> bool` dropped imaginary part | **FIXED** | — |
| `floor_divide`/`mod`: int ÷0→0, float `//0`→±inf, signed floor, MIN/-1, mixed-precision | **FIXED (F1)** | — |
| NEP50 weak-scalar (0-D operand promoted weakly) | intended (Misaligned) | — |
| complex division / multiply ~1 ULP + cancellation | intended (algorithm) | #12 |
| `power`: complex power ~1 ULP + gross inf/NaN edge | [Misaligned] (complex-binary) | F5 |
| `<=`/`>=` return True for NaN | known bug | #8 |
| unary NEP50 promotion (int->float64 vs width-based); `negative(uint)` throws; `reciprocal` | known bug | #9 |
| reductions: NaN propagation, complex-axis throw, bool min/max, summation precision | known bug | #10 |
| `np.where` complex throws ("Zero-push unsupported") | known bug | — |
| bool arithmetic (`True+True -> 2`), size-1 result collapse | known bug | #12 |
| ops vs raw NumPy stride/offset representation (offset!=0, junk size-1 strides) | unreachable via API | #11 |

### Char & Decimal dtype coverage (the two NumPy-orphan dtypes)

Both are now exercised across the grids (previously excluded). Verified bugs are NOT excused in
`MisalignedRegistry` — each is carved out of the green corpus and reproduced under `[OpenBugs]`
(remove the carve + test when fixed):

| dtype | op / combo | NumSharp | NumPy/value truth | Repro |
|-------|-----------|----------|-------------------|-------|
| Char | `promote(Char, Byte)` → Byte (truncates high byte) — corrupts arithmetic AND comparisons | Byte / wrong bool | uint16 (Char≡uint16) | `OpenBugsCharTests.Char_Add_Byte_*`, `Char_Compare_Byte_*` |
| Char | `reciprocal(char)` | Double | uint16 | `Char_Reciprocal_ReturnsDouble` |
| Char | `power(char, float32)` | Double | float32 | `Char_Power_Single_ReturnsDouble` |
| Char | `power(char, …)` scalar path | InvalidCastException (`Convert.To*(char)`) | computes | `Char_Power_ScalarChar_Crashes` |
| Char | `bitwise_*(bool, char)` | KeyNotFoundException `(Boolean,Char)` | uint16 | `Char_BitwiseAnd_Bool_KeyNotFound` |
| Char | `invert(char)` N≥16 (SIMD path) | NotSupportedException | uint16 | `Char_Invert_LargeN_NotSupported` |
| bool | `clip(bool)` non-contiguous (strided/transposed/F) | NotSupportedException | bool (contiguous works) | `OpenBugsDtypeCoverageTests.Clip_Bool_*` |

`Decimal` rides an **independent C# oracle** (`gen_decimal_oracle.cs`, naive scalar `System.Decimal`;
`std` is oracled by an independent Newton decimal sqrt, NOT NumSharp's `DecimalMath`) across
unary / binary / reduce / scan / power(int-exp) / var / std / matmul / astype(decimal↔int·float) ×
13 single + 7 pair layouts — **302 cases, all green** (no decimal kernel bug found). Two non-bugs were classified and carved (not OpenBugs): the complex self-multiply ULP
(NumSharp matches NumPy *scalar*; NumPy's own array ufunc disagrees on a catastrophic-cancellation
input) and `argsort<bool>/<Complex>` "not supported" (a harness `OpRegistry` gap, now wired → green).
