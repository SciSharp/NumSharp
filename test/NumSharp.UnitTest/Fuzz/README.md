# NumPy Differential Fuzzer (Plan A)

Proves every NpyIter-backed operation produces **bit-identical** output to NumPy 2.4.2 across the
full input space — caught systematically, not by hand-picked cases. The motivating failure (the
cast saturate-vs-wrap bug, latent in `where`/`copyto`/`concatenate`) must be impossible to ship again.

## How it works

NumPy is the oracle. Python (`oracle/`) generates a **committed, bytes-exact corpus**; the C# harness
**replays the operand bytes** and bit-compares — *no Python at test time, none in CI*.

```
oracle/                              corpus generators (NumPy 2.4.2)
  layout_catalog.py                  the layout builders (single-array, pairwise, where-triple)
  gen_oracle.py                      deterministic matrices (astype/binary/comparison/unary/reduce/where/place)
  fuzz_random.py                     seeded random fuzzer (NumSharp-producible layouts)
test/NumSharp.UnitTest/Fuzz/
  FuzzCorpus.cs                      reconstructs EXACT NDArray views from (dtype,shape,strides,offset,bytes)
  BitDiff.cs                         bit-exact compare; NaN tokenized; ULP helpers (for documented near-misses)
  OpRegistry.cs                      op-name -> NumSharp call
  MisalignedRegistry.cs              the explicit, documented set of intended/known divergences
  Shrinker.cs                        minimizes a failing element-wise case to a 1-element repro
  FuzzCorpusTests.cs                 one [FuzzMatrix] test per corpus file
  corpus/*.jsonl                     committed, copied to test output
```

A divergence is one of: **bit-exact** (passes), a **documented difference** in `MisalignedRegistry`
(excused + logged, never silent), or a **failure** (any unknown divergence — the gate is red).

## Regenerating the corpus

```bash
python oracle/gen_oracle.py astype_full      # 13x13 dtypes x 26 layouts
python oracle/gen_oracle.py binary           # add/sub/mul/divide x NEP50 pairs x pairwise layouts
python oracle/gen_oracle.py divmod_power     # floor_divide/mod (bit-exact, F1) + complex power (Misaligned)
python oracle/gen_oracle.py comparison       # ==,!=,<,>,<=,>=
python oracle/gen_oracle.py unary            # negate/abs/sqrt/trig/exp/log/...
python oracle/gen_oracle.py reduce           # sum/prod/min/max/mean/std/var/argmax/argmin/all/any
python oracle/gen_oracle.py where            # np.where(cond,x,y)
python oracle/gen_oracle.py place            # np.place(arr,mask,vals)
python oracle/gen_oracle.py matmul           # T8 linalg: matmul/dot/outer (gufunc shapes, C/F layouts)
python oracle/fuzz_random.py 1234 2000 random_smoke.jsonl
```

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

**Scope:** `Char` and `Decimal` have no NumPy analog and are excluded from the differential corpus
(covered by the `Converts`-oracle cast tests).
