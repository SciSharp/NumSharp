# NumPy Differential Fuzzer (Plan A)

Proves every NDIter-backed operation produces **bit-identical** output to NumPy 2.4.2 across the
full input space — caught systematically, not by hand-picked cases. The motivating failure (the
cast saturate-vs-wrap bug, latent in `where`/`copyto`/`concatenate`) must be impossible to ship again.

### Current measured snapshot (`journey3`, 2026-08-22)

- **116,339 committed JSONL rows / 64 files**: 103,208 ordinary op cases, 12,426 advanced-index
  cases, 703 independent Decimal cases, and two host-pin metadata rows. Char contributes 5,506
  proxy rows across 20 files.
- **363 distinct corpus op keys**. `OracleSurfaceCoverageTests` inventories the public surface
  mechanically (`np` 321 · `np.linalg` 31 · `np.fft` 18 · `np.random` 48) and makes an
  unclassified new API fail `FuzzMatrix`.
- The journey3 receipt inventories **186 touched public callables and requires 186/186 direct
  corpus keys**. The operation-strength gate additionally requires every ordinary op key to have
  at least four non-duplicate-axis cases; advanced indexing keeps explicit 2,000/100/10/10,000
  corpus floors.
- New completeness tiers: `creation.jsonl` (302 deterministic creators), `conversion.jsonl`
  (1,078 value/error/artifact cases), and `multioutput.jsonl` (64 full-tuple/arity cases). The older claim
  that creation and tuple results do not fit the corpus is no longer true.
- Managed/OpenBLAS variation: `BlasBackendDelta` finds 1,775 affected ordinary cases, deduplicates
  1,747 identical managed/backend outcomes, and byte-checks the 28 real backend changes against
  NumPy on the pinned host. Dedup includes threw/result state, dtype, shape, and bytes.
- Full gate: **85/85 green on net8.0 and net10.0**.

## How it works

NumPy is the oracle. Python (`test/oracle/`) generates a **committed, bytes-exact corpus**; the C# harness
**replays the operand bytes** and bit-compares — *no Python at test time, none in CI*.

```
test/oracle/                         corpus generators (NumPy 2.4.2)
  layout_catalog.py                  the layout builders (single-array, pairwise, where-triple)
  gen_oracle.py                      deterministic matrices (astype/binary/comparison/unary/reduce/where/place);
                                     per-mode dtype axes widened to ALL_DTYPES; Char WOVEN into every tier
                                     via the uint16 proxy (char_tier) — relabelled uint16->char, bytes intact
  gen_nan_oracle.py                  STANDALONE NaN-parity oracle -> nan.jsonl (complex-unary NaN sign
                                     BIT-EXACT vs NumPy; float widths value-NaN). Owns its own numbering.
  gen_decimal_oracle.cs              INDEPENDENT C# oracle for Decimal (no NumPy analog): naive scalar
                                     System.Decimal math -> decimal_{unary,binary,reduce,scan,power,
                                     varstd,matmul,astype,stat,where,sort,manip}.jsonl (12 tiers, 703 cases)
  fuzz_random.py                     seeded random fuzzer (13 dtypes × unary/binary/comparison/where/
                                     flat-reduce/astype kinds; NumSharp-producible layouts)
test/NumSharp.Tests.Oracle/Fuzz/
  FuzzCorpus.cs                      reconstructs EXACT NDArray views from (dtype,shape,strides,offset,bytes)
  BitDiff.cs                         bit-exact compare; NaN tokenized (payload/sign non-contractual) EXCEPT
                                     the complex-unary ops, whose NaN sign IS contractual and is raw-byte
                                     compared (Compare nanBitExact + DiffHasSignFlip); Decimal by canonical
                                     VALUE (scale-insensitive 1.0m == 1.00m); ULP helpers (documented near-misses)
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
- **Result kinds** — `expected.kind` selects what is compared, so ops whose result is not a single
  array are gated by the same corpus: `array` (default, the dtype/shape/bytes contract above),
  `scalar` (a C# scalar wrapped 0-d, the `np.allclose` pattern), `dtype` (compared by NumPy dtype
  NAME — this is how the promotion table itself gets gated, not just some binary op's result
  dtype), `text` (printing, compared verbatim), and `tuple` (N slots, **arity asserted first** —
  the older which/piece params gate one slot per case and so structurally cannot catch a wrong
  slot count). Comparators live in `FuzzCorpusTests.Kinds.cs`, callables in `OpRegistry.Kinds.cs`.
- **Error parity** — two tiers, deliberately different in strength:
  - *legacy* (`errors.jsonl`, and any case flagged `expects_throw` without an `error` object):
    NumSharp must throw **something**; a throw of ANY type passes, a normal return is the
    divergence. Exception type/message are not asserted.
  - *message parity* (`errors_full.jsonl`, and any case carrying `error: {type, text}`): NumPy's
    exception class and `str(e)` are recorded at generation time and NumSharp is held to **both** —
    the type via a NumPy-class → .NET-type map (identical names, which NumSharp uses for
    `ValueError`/`TypeError`/`IndexError`/`AxisError`, always match), and the message **verbatim**
    after stripping .NET's `" (Parameter 'x')"` framing. This tier exists because every value
    generator SKIPS the cells where NumPy raises, so those cells previously had no gate at all.
  - The reverse direction is gated on every ordinary case: NumSharp throwing where NumPy returned
    a result is a `Threw` divergence — a failure unless a registry branch excuses it.
- **Index oracle** (`IndexOracleTests`): compares result shape, values, and **which side raised**
  — if both NumPy and NumSharp raise, the case passes regardless of exception type. NumPy's
  exception name is carried in the corpus (`err`) for failure messages only, never for parity.
- **Excused divergences are logged, never silent**: every case a `MisalignedRegistry` branch
  classifies is counted and printed per tier even when the test passes —
  `[<file>] documented Misaligned divergences excused: <n>x <reason>; …` — so growth in an
  excused class stays visible in the test output. Anything unclassified is red.

### Host-dependent values — the one thing the oracle must never assert

A float→integer conversion is **undefined in C** when the value is NaN, ±inf, or outside the
destination's range, and NumPy performs exactly that C cast. The result is the host toolchain's,
not a NumPy contract: glibc/gcc and MSVC disagree, and so do the vectorized and scalar loops of a
*single* numpy build.

```python
np.array([np.nan] * 8).astype(np.uint32)[0]   # 2147483648   (gcc, vectorized loop)
np.float64(np.nan).astype(np.uint32)          # 0            (same build, scalar loop)
```

Two tiers, two rules:

- **Committed corpora are replayed from bytes, never recomputed**, so they may keep the undefined
  edges — `layout_catalog._FLOAT_POOL` front-loads them on purpose. There they pin NumSharp's
  hand-written cast kernels against *themselves*: an internal regression gate, not NumPy parity.
  Regenerating such a tier **on a different OS or CPU rewrites those cells**, and that diff is a
  host difference, not a NumSharp bug — regenerate on the platform the corpus was authored on.
- **`fuzz_random.py` recomputes `expected` on whichever host it runs on**, so it may only emit
  conversions NumPy defines. `_defuse_cast` / `_defuse_integer_reciprocal` rewrite the undefined
  elements before the expectation is taken — keeping the defined truncation and boundary edges, so
  the cell stays covered rather than dropped — and `assert_portable` then audits the serialized
  bytes, so a regression fails generation loudly instead of producing unusable expectations.

This is precisely what broke the nightly soak (run 29722530598): it generates on Ubuntu and replays
against cast kernels that reproduce the MSVC answer, so ~950/200000 cases "diverged" every night on
`astype` float→uint32/uint64 and on `reciprocal(uint64 0)` — which NumPy computes as
`(uint64)(1.0/0)`, the same undefined conversion. No implementation can satisfy both hosts; the fix
was to stop asserting undefined values, not to chase one host's.

## Scope gate — undisposed-intermediate detection (oracle-free)

`UndisposedIntermediateTests` (`[FuzzMatrix]` + `[ScopeAudit]`, `[DoNotParallelize]`) replays the
same corpus through `OpRegistry` with **every result disposed** and asserts the buffer pool's
takes/returns balance via `ScopeAudit` (`SizeBucketedBufferPool`'s public counters; the takes side
includes `ZeroedAllocs` — the calloc path counts there, not in Hits/Misses). In a fully-disposed
region with no GC inside it, `takes − returns` is exactly the number of buffers an op took,
dropped, and left for a future GC + finalizer pass — an **undisposed intermediate** the
[NDScoped] weaver / library scoping failed to cover (the traffic class behind the pre-160ecbba
benchmark collapse). A negative balance is the mirror defect (a result buffer allocated outside
the pool, returned into it). Values are NOT compared here — that is FuzzCorpusTests' job.

Validity rules, each encoded in the harness: a GC inside a region masks escapes (finalizers
Return them mid-window), so regions observing a collection are retried after a settle and a
persistent interference is inconclusive, never red; a **non-zero screen is re-measured after a
settle** because the sweep runs over a library with known leaks — escaped buffers accumulate,
GCs collect them, and the finalizer thread drains returns *asynchronously* across later regions,
invisible to GC-count detection (first landing showed phantom escapes of −262 from exactly this);
each case gets one un-measured **warm invocation** so one-time caches (FFT plans, emitted
kernels) don't read as escapes; and the teeth tests (`Harness_Detects_*`) prove the detector
fires in both directions, so a counters-accounting bug cannot read as "everything clean".

**Landing inventory (2026-08-26):** the first full sweep found pre-existing leaks in **91 ops /
~9,900 of 102,785 measured cases** — the axis/nan/cumulative reduction family, the product family
(matmul/dot/vecdot/matvec/vecmat/vdot), fft, the tri/tril/triu/diag* family, `trim_zeros` (up to
19 buffers per call), `np.empty` itself, ufunc `out=` paths, and the NEP50 scalar-operand binary
cells (the engine's `Cast(rhs, resultType, copy: true)` parameter-reassign drop). These are
documented in `KnownEscapes` — surfaced green with a per-op **ceiling** (an op leaking more than
its recorded worst still fails) — and held red by the `KnownEscapeFamilies_AreFixed`
`[OpenBugs]` pin plus the mechanism pin `BinaryScalarCastTemp_IsDisposed`. Working the list down:
fix an op, remove its `KnownEscapes` entry (the sweep then gates it at zero forever); when the
registry empties, delete the pin. Every op NOT in the registry is gated at zero from day one.

**Outside-pool allocation detection** rides the same sweep plus a static gate, because the two
halves of the bypass class need different instruments. The RUNTIME half: a result that is fresh
(not an operand instance, its data pointer outside every operand's base-buffer byte range, larger
than a `StackedMemoryPool` scalar slot — 16 B, the second pool these counters cannot see) while
the region shows **zero** bucketed-pool takes was allocated AND freed outside the pool — paying a
cold alloc + first-touch faults per call with no warm reuse. That verdict needs no drain-confirm
(drain adds returns, which turns the balance negative and routes down the escape path; takes==0
with escaped==0 is arithmetically drain-free), and its teeth is `np.frombuffer` — a bypass BY
DESIGN (zero-copy wrap), which is also what the landing sweep found: only the I/O wrap class
(`frombuffer`/`fromfile`/`loadtxt`, results wrapping caller memory or parsed managed arrays — no
native alloc exists to route), recorded in `KnownBypassByDesign` (documented, never pin-tracked)
vs `KnownBypassDebt` (pin-tracked; empty at landing). The STATIC half —
`NativeAllocationChokepointTests` — covers what the runtime check cannot see by construction: an
internal scratch buffer allocated raw and freed raw inside an op never reaches a result. It scans
`src/NumSharp.Core` for raw `NativeMemory.Alloc*`/`Marshal.AllocHGlobal`/`VirtualAlloc*` call
sites (comment lines excluded) and pins an exact file→count allowlist: the two pools + the
guard-page allocator ARE the chokepoints, NDIter's buffered-mode scratch (7 sites) and
`np.bincount`'s counting table (1) are carried as routed-through-the-pool audit debt, and ANY new
raw site — new file or count growth in an allowed file — is red until pooled or consciously
allowlisted. Inconclusive (never false-green) without a source checkout.

## Regenerating the corpus

```bash
python test/oracle/gen_oracle.py astype_full      # 13x13 dtypes x 26 layouts (host-sensitive, see above)
python test/oracle/gen_oracle.py binary           # add/sub/mul/divide x NEP50 pairs x pairwise layouts
python test/oracle/gen_oracle.py divmod_power     # floor_divide/mod (bit-exact, F1) + complex power (Misaligned)
python test/oracle/gen_oracle.py comparison       # ==,!=,<,>,<=,>=
python test/oracle/gen_oracle.py unary            # negate/abs/sqrt/trig/exp/log/...
python test/oracle/gen_oracle.py reduce           # sum/prod/min/max/mean/std/var/argmax/argmin/all/any
python test/oracle/gen_oracle.py where            # np.where(cond,x,y)
python test/oracle/gen_oracle.py creation         # deterministic zero-operand creators + Char proxy
python test/oracle/gen_oracle.py conversion       # array/as*/require/frombuffer/fromstring + errors
python test/oracle/gen_oracle.py multioutput      # full tuple arity + every result slot
python test/oracle/gen_oracle.py iter             # ndindex/ndenumerate/nditer/broadcast TRACES (order gate)
python test/oracle/gen_oracle.py dtype_text       # dtype/scalar/text/tuple result kinds
python test/oracle/gen_oracle.py errors_full      # the raising cells every value generator skips
python test/oracle/gen_oracle.py out_where        # ufunc out=/where= x out layout x mask layout
python test/oracle/gen_oracle.py place            # np.place(arr,mask,vals)
python test/oracle/gen_oracle.py matmul           # T8 linalg: matmul/dot/outer (gufunc shapes, C/F layouts)
python test/oracle/gen_oracle.py specials         # IEEE special-value parity (nan/±inf/±0/subnormal/max)
python test/oracle/gen_oracle.py precision        # truthful-vs-precise (adversarial accumulation; needs mpmath)
python test/oracle/gen_oracle.py products         # CBLAS product family values (inner/vdot/vecdot/matvec/...)
python test/oracle/gen_oracle.py fft              # np.fft.* — 1-D/N-D/hermitian transforms + freq/shift helpers
python test/oracle/gen_oracle.py random_parity    # seeded np.random stream bytes (portable + host-libm files)
python test/oracle/gen_index_oracle.py            # the four index_* corpora (seed pinned 20240626)
python test/oracle/gen_nan_oracle.py              # nan.jsonl — NaN parity grid (standalone; complex bit-exact)
python test/oracle/fuzz_random.py 1234 2000 random_smoke.jsonl
dotnet run test/oracle/gen_decimal_oracle.cs      # Decimal tiers (independent C# System.Decimal oracle)
```

The authoritative full mode list is the unknown-mode message in `gen_oracle.py`; it includes
`conversion creation multioutput iter dtype_text out_where errors_full` plus every value/parity
mode shown here.
Regeneration is deterministic: rerunning an untouched mode must produce a zero corpus diff.

The **`fft` tier** (`fft.jsonl`, 2,000 cases) gates the managed pocketfft engine
(`src/NumSharp.Core/Fourier/`): the 1-D core (`fft`/`ifft`/`rfft`/`irfft`) + hermitian
(`hfft`/`ihfft`) + the N-D forms (`fft2`/`ifft2`/`fftn`/`ifftn`/`rfft2`/`irfft2`/`rfftn`/`irfftn`)
+ helpers (`fftfreq`/`rfftfreq`/`fftshift`/`ifftshift`), swept over dtype, `n`/`s`
{default/truncate/zero-pad/prime-13 Bluestein}, `norm` {backward/ortho/forward}, `axis`/`axes`, and the
memory layouts {C, F, strided, reversed, transposed, broadcast-read}. **float64/complex128/int/bool are
bit-exact** with NumPy 2.4.2; **float32/float16 are the one documented divergence** (F1 above) —
NumSharp has no complex64, so it promotes to double (values = the correctly-rounded double result).
A generator note that bit: NumPy's 2-D forms default `axes` to `(-2,-1)` but treat an *explicit*
`axes=None` as all-axes (fftn), so the generator OMITS a `None` `s`/`axes`/`norm` to exercise each op's
real default — which is exactly what NumSharp's null-coalescing (`axes ?? {-2,-1}`) mirrors.

Char rides the applicable NumPy modes automatically (`char_tier` appends uint16-proxy cases
relabelled to `char` into 18 ordinary tier files — arith/divmod/comparison/unary×2/bitwise/reduce/
scan/stat/manip/sort/tail/astype/where/logic/matmul/rounding/copyto); creation and conversion append
their own proxy rows, for 20 Char-bearing files total. There is no separate `char` mode.
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

### Table 0 — divergences found by the result-kind / error / iterator tiers

Added with `iter.jsonl` (4,611), `dtype_text.jsonl` (2,618), `multioutput.jsonl` (64) and
`errors_full.jsonl` (688). Every
row is scoped in `MisalignedRegistry` branches K1–K9, so each is counted and printed on every run.

| Finding | Where | Status |
|---|---|---|
| `np.nditer(0-d, external_loop)` → `it[0]` **kills the process** (AccessViolation): `GetInnerLoopSizePtr` reads `Shape[-1]` | `np.nditer.cs` indexer | **FIXED** — 0-d answered directly, as `NDIterTyped.ReadInnerLoop` already did |
| `order='A'` over a transposed 3-D operand picks a different axis ordering than NumPy | K1 | known bug |
| `external_loop` coalesces fewer dimensions → more/shorter chunks (values agree, chunk lengths do not) | K1 | known bug |
| `isscalar(0-d array)` → True, NumPy False | K2 | known bug |
| `nonzero(0-d)` returns a tuple, NumPy raises | K3 | known bug |
| complex-input ufunc rejection: same refusal, NumSharp's own wording/type | K4 | known gap |
| …and the rejection is **skipped entirely** on a zero-size complex operand (NumPy validates the loop, not the data) | K5 | known bug |
| `power(bool, negative int)` misses the integer-power guard | K6 | known bug |
| `power(int, negative int)` trips `Debug.Fail("index < Count, Memory corruption expected")` instead of NumPy's ValueError — in Release that path has no assert | K8 | known bug (memory safety) |
| `result_type(mixed signed/unsigned, 0-D operand)` throws instead of resolving | K9 | known bug |
| NEP50 weak-scalar reached via the error path (int64+uint64 succeeds where NumPy refuses) | K7 | intended |
| ufunc `out=` on a read-only **broadcast** view: NumSharp writes through it (587 cases), contradicting its own `Shape.IsWriteable == false` rule | K10 | known bug |
| `isnan` into a **strided bool `out`**: results land on the wrong elements (contiguous out is correct) | K12 | known bug |
| `exp(1.0f)` in a (4,5) float32 array returns `0x402df854` from `np.exp(x)`, `np.exp(x, out)` and `np.exp(x, out, where)` alike, while NumPy **and the committed `unary.jsonl` expectation for the same values/shape/dtype** say `0x402df855` — the unary tier is green, so the same op on the same data disagrees depending on how the array was built | K11 | **open question** |

`out=`/`where=` results that are *clean*: no out-of-window writes in any layout (strided / offset /
negstride / F / transposed all keep the bytes outside the view intact), and masked-off slots retain
their prior contents in every one of the 882 `where=all_false` cases.

**Findings from the 2026-08-21 surface-completeness expansion:** all newly exposed product defects
were fixed with their corpus cells retained as hard regression proofs: `angle(deg=True)` 0-D
float-tier widening · `full_like` selecting the fill value's CLR dtype · integer `linspace`
truncating instead of flooring · Char ones/eye/identity writing `'1'` (0x31) ·
`ascontiguousarray`/`asfortranarray` failing NumPy's ndim≥1 scalar contract · einsum's internal
order materialization leaking that public scalar promotion into a `()` contraction result. The one
algorithmic remainder is complex `corrcoef`: its two normalization divides inherit the existing
`npy_cdivide` versus `System.Numerics.Complex` 1-ULP difference; it now has its own ≤2-ULP branch
and paired tightness pins instead of hiding under the broad complex-unary envelope.

### Table 1 — live `MisalignedRegistry` excuse branches

**Intended / algorithmic differences (permanent):**

| Excuse class | Scope | Hits |
|---|---|---|
| NEP50 weak-scalar: 0-D operand promoted weakly | any multi-operand op × Dtype kind, 0-D operand present | 261 |
| **F1** `np.fft(float32/float16)`: NumPy returns complex64/float32/float16, NumSharp complex128/float64 — a dtype-ONLY divergence (values = the correctly-rounded double result; bit-verified `fft(x32) == fft(x32.astype(f8))`). NumSharp has one complex type. The unnameable complex64 is routed to a Dtype divergence by `CompareArray`. Contractual dtypes (float64/complex128/int/bool) are bit-exact; the helpers never diverge | 14 fft transforms × {float32,float16} input × Dtype kind | 516 |
| unary ~ULP (transcendental/magnitude algorithm difference) | single-operand × Value, every diff ≤2 ULP — **EXCEPT exp/log/sin/cos/rad2deg/deg2rad at a float32 result, which are gated bit-exact** (see below) | 563 |
| complex unary within 3 ULP (full NumPy-algorithm port) — FINITE interior only; the NaN SIGN of these ops is now compared **raw-byte** (`ComplexNanContractOps`, `BitDiff.Compare(nanBitExact:true)`) and a pure NaN-sign / signed-zero flip HARD-FAILS via `DiffHasSignFlip` before any ULP excuse runs (NumSharp reproduces NumPy 2.4.2 win-amd64 / MSVC UCRT NaN signs bit-for-bit) | complex unary × Value, ≤3 ULP, no sign flip | 11 |
| complex arccos/arccosh/sinh/cosh (+ sin/cos routing through sinh/cosh) pathological FINITE edge (sub-DBL_MIN denormal-real flush / \|x\|∈[710,710.13] overflow boundary) — the former "cos/sin NaN zero-sign" regime is GONE (now byte-exact) | those ops × complex × Value, finite | 0 |
| complex division ~1 ULP (npy_cdivide vs System.Numerics.Complex) | divide × complex × Value, ≤2 ULP | 17 |
| complex corrcoef normalization inherits complex-division rounding | corrcoef × complex input/result × Value, ≤2 ULP | 1 |
| complex add/subtract within 2 ULP (FMA contraction) | add/subtract × complex × Value, ≤2 ULP | 0 |
| complex multiply cancellation / ~ULP at element magnitude (#12) | multiply × complex × Value, ≤16 element-magnitude ULP | 16 |
| complex power ~ULP / gross inf-NaN edge (Complex.Pow vs npy_cpow) (F5, ledger L6) | power × complex × Value, ≤512 element-magnitude ULP or non-finite | 30 |
| reduction summation/two-pass precision (algorithm order) | sum/mean/std/var/prod × float-family result (Half/Single/Double/Complex) × Value | 401 |
| complex reduction/scan NaN ordering/propagation differs | reduce+cumsum/cumprod × complex × Value, diffs must contain a NaN token | 35 |
| decimal std last digit (independent 28-digit sqrts) (ledger L7) | std × Decimal × Value, ≤1 unit in the 28th significant digit | 4 |
| **S1** expm1/log1p small-\|x\| precision loss / -0 / subnormal flush (`Exp(x)-1`, `Log(1+x)`; NumPy calls the CRT, non-portable) | expm1/log1p × Value, every diff ≤2 ULP **or** ≤~ulp(1) abs | 20 |
| **S2** fmax/fmin ±0-tie sign on a reversed float32 view (NumPy's OWN fmax ±0 sign is SIMD-path-dependent — array returns operand 2, scalar returns +0) | fmax/fmin × Single × Value, both tokens a signed zero | 2 |
| **S3** complex matmul/dot/outer infinite-operand product: C99-unspecified complex-inf (zgemm `(nan,nan)` inf-recovery vs managed `(inf,nan)`) | matmul/dot/outer × complex × Value, every diff non-finite | 9 |
| **P1** prefer-precise: diverges from NumPy TOWARD the correctly-rounded truth — parity debt (port NumPy's algorithm), never a win | truth-bearing × Value, all diffs not-less-truthful, some strictly closer | 15 |
| **P2** prefer-precise: diverges within truth-equivalence slack (neither side less accurate) | truth-bearing × Value, all diffs ≤ max(4×dNPY, dNPY+8) ULP-vs-truth | 19 |
| **R1** rnd transform ~ULP: chisquare/wald/noncentral_f/dirichlet compose their draws with a slightly different arithmetic ordering than NumPy's C (stream bit-identical — a stream slip is gross and still fails) | rnd × those 4 dists × Value, ≤8 ULP (32 for wald) | 12 |

**Known bugs (tracked for fix — remove the branch when fixed), truth-adjudicated:**

| Excuse class | Scope | Hits |
|---|---|---|
| **P3** precision-loss (known): f32 var/std accumulation (55/26 ULP vs truth where NumPy sits at 3/2) + negative-stride reduction accumulation (11–32 ULP where NumPy is EXACT on the same reversed view) + f32 deep product contraction (inner/tensordot K=2049, 1–2 ULP past the prefer-precise slack vs BLAS sgemm) | truth-bearing × Value × (negstride sum/mean/var/std, or Single var/std, or Single product family), every diff ≤256 ULP-vs-truth | 9 |

**Narrowed: the NumPy-ported float kernels are no longer excused.** `NDFloatMath` ports the kernels NumPy
2.4.2 actually runs — `simd_exp_FLOAT`, `simd_log_FLOAT`, `simd_sincos_f32`, `simd_tanh_f32`/`simd_tanh_f64` — and
`rad2deg`/`deg2rad` now form their constant at float precision like NumPy's macros. Each agrees with NumPy 2.4.2 on
**all 2³² float32 inputs** (verified by a chunked-checksum sweep over the entire bit space, through both the SIMD and
scalar paths), so the blanket unary-ULP branch now skips `exp`/`log`/`sin`/`cos`/`tanh`/`rad2deg`/`deg2rad` at a
float32 result and any divergence there fails the gate. The `numpy_f32_kernels.jsonl` tier (140 cases) feeds each kernel the inputs that
discriminate — every NaN spelling, exp's saturation boundaries ±1 ULP and its subnormal-output band, log's
1/sqrt(2) mantissa split and 2^100 subnormal rescale, the quadrant seams and BOTH Cody-Waite libc cutoffs for the
trig pair, tanh's 32 subinterval seams ±1 ULP and its saturation cut, and each NumPy-documented worst-error
input — and `OpenBugs.FuzzGate.cs`'s B8 tests pin the carve-out from both sides (the ported ops must NOT be excused;
expm1/log1p/exp2/arctan still must be). Deliberately still excused: every float16 loop (NumPy's separate
`loops_half` kernels) and float64 exp/log/sin/cos (the platform's scalar `npy_*`), which already agree bit-for-bit
here anyway.

**`tanh` is carved out at float64 too — the only op that is.** NumPy ships its own table-driven tanh at BOTH widths
(`loops_hyperbolic`), so `Math.Tanh` diverged on 8.1% of f8 inputs where the other kernels' f8 loops already agreed.
`NumPyPortedFloat64Kernels` holds that one name, and the **`numpy_f64_kernels.jsonl`** tier (24 cases) sweeps the
same layouts over the f8 subinterval seams, the saturation cut, ±0/±inf/NaN and the int32-and-wider dtypes that
promote INTO this loop. float64 tanh is verified over 4.83 billion values — 2³² covering every sign × exponent ×
2²⁰ mantissa prefix, plus 5.4×10⁸ full-width splitmix64 patterns — not exhaustively, which f8 does not admit.

**Measured, still divergent, NOT ported** (so the envelope still covers them): `exp2` float32 0.04% / float64
0.02% of inputs, 1 ULP; and `expm1`/`log1p` (31.1% / 30.7% and 33.6% / 15.1%), which NumSharp computes as
`Exp(x)-1` / `Log(1+x)` — not merely a ULP difference but catastrophic for small |x| (`expm1(1e-8)` returns 0 where
NumPy returns 1e-8) plus a signed-zero bug (`expm1(-0.0)` → `+0.0`). NumPy calls the CRT for all three, so bit-parity
is NOT reachable from managed code the way it was for exp/log/sin/cos/tanh; the accuracy bug is worth fixing on its
own terms, and note .NET's own `float.ExpM1`/`double.ExpM1`/`LogP1` are themselves just `Exp(x)-1`/`Log(1+x)` and do
not help. Also still excused: the float16 loops, 17 differing values across all 65,536. The **`specials` tier**
now DRIVES expm1/log1p into the catastrophic small-|x| band the ordinary pools never reach (subnormal / tiny / -0
inputs) and gates the divergence with the `S1` branch — bounded to ≤2 ULP **or** a ~ulp(1) absolute envelope, so a
gross regression (wrong magnitude/sign at a non-tiny result) still fails while the documented precision loss is
excused, and a future dedicated small-|x| kernel flips those 20 cells straight to bit-exact.

**tanh's FMA is NOT part of the host pin below.** exp/log/sincos had to reproduce MSVC's contraction of a separate
multiply and add; `simd_tanh_*` spells its Horner steps as `hn::MulAdd`, an explicit fused multiply-add, so the port
transcribes it literally rather than betting on a compiler. A negative control confirms the sweep is not vacuous: the
pre-port `MathF.Tanh` differs from NumPy in 34 of the 512 chunks (the rest of the f32 space saturates to ±1 or is
NaN, where libm already agreed), while the ported kernel matches in all 512.

**Host-pinned, like the cast kernels.** The port fuses the quadrant's `mul`+`add` because MSVC 19.44 — the
compiler of the pinned `numpy==2.4.2` win-amd64 wheel — contracted that intrinsic pair into a `vfmadd`. It is
observable at exactly one probed input, `x = 0xc26d0e6c`, where `x·log2(e)` is the exact tie `-85.5`: fused
rounds the quadrant to -85, unfused to -86, and the results differ by 1 ULP. A NumPy built by a toolchain that
does not contract there would differ at such ties, so this is a NumSharp regression pin against *this* wheel
rather than portable IEEE parity — the same status as the "Host-dependent values" cast cells below.

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
| `gen_random_parity` carve list | 8 samplers whose STREAM diverges (different algorithm / accept-reject boundary): gamma(shape<1 via 2-arg), f, pareto, standard_cauchy, binomial (both branches), negative_binomial, multinomial, multivariate_normal | `OpenBugsRandom.RandomParity_*` (8 pins) |

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

The 2026-08-21 completeness pass additionally fixed: `angle(deg=True)` scalar dtype ·
`full_like` dtype selection · integer `linspace` floor semantics · Char numeric-one creation ·
0-D `ascontiguousarray`/`asfortranarray` rank · einsum scalar-result rank after order resolution.

### IEEE special-value parity (`specials` tier)

`specials.jsonl` (2,393 cases, `gen_oracle.py specials`) FORCES nan / ±inf / ±0 / smallest-subnormal /
±max / ±tiny through the elementwise-math (unary + binary), reduction (incl. the `nan*` family), scan
and matmul/dot/outer ops across float16/float32/float64/complex128 and every layout (contiguous /
2-D / F-contiguous / step-2 strided / negative-stride). It closes three gaps the *incidental*
front-loading of specials in `layout_catalog._FLOAT_POOL` leaves:

1. **Cross-operand interactions.** The ordinary pair layouts align `A[i]` with `B[i]` from the SAME
   pool in the SAME order, so the interactions that ARE IEEE arithmetic never occur. The tier builds
   explicit aligned pairs that force `inf+(-inf)=nan`, `0*inf=nan`, `0/0=nan`, `inf/inf=nan`,
   `1**inf`, `max*max→inf`, and the signed-zero / subnormal / tiny boundaries.
2. **NaN/inf propagation through the managed GEMM.** `matmul`/`dot`/`outer` draw from `_mm_fill`
   (clean integer/half ramps), so propagation through the ONE product path a plain test run takes —
   NumSharp.Core ships no BLAS — was never gated. The operands are built so every output cell is
   order-independent (a NaN anywhere → NaN; an inf against all-positive-finite → +inf; no inf−inf
   cancellation inside a dot), isolating propagation from summation reassociation.
3. **Per-dtype extremes and the smallest subnormal**, absent from the pools entirely.

`BitDiff` tokenizes NaN (any payload → `"NaN"`) and bit-compares ±0.0 / ±inf, so the tier asserts the
CONTRACTUAL part of IEEE parity — is-NaN, the sign of a zero, the sign of an infinity.
**2,118 / 2,393 cases are bit-exact with NumPy 2.4.2**; the 275 excused are all scoped registry
branches surfaced anew by the denser inputs (chiefly the known `nan*`-family and complex-reduction
divergences) PLUS the three the tier discovered — `S1`/`S2`/`S3` in Table 1. **Headline result: every
real-dtype (f16/f32/f64) matmul/dot/outer specials case is bit-exact** — the managed float GEMM
propagates NaN/inf exactly like NumPy's BLAS on these operands; only C99-unspecified complex-infinity
arithmetic (`S3`) diverges.

### NaN parity (`nan` tier) — "do our functions produce NumPy's NaN?"

`nan.jsonl` (120 cases, `gen_nan_oracle.py` — a STANDALONE generator like the npy/decimal oracles,
so it never renumbers the shared corpus) is the dedicated NaN oracle. It runs every UNARY op for
which a NaN output is reachable over the FULL special-value grid — finite / ±0 / ±inf / **BOTH NaN
signs (+NaN `0x7ff8…` AND −NaN `0xfff8…`)** — as complex128 (the 64-element re×im cross-product) plus
float16/32/64 lines, recording NumPy 2.4.2's exact output bytes. `CompareArray` applies the
NaN-contract policy verbatim:

- **complex128 unary ops** (`ComplexNanContractOps`: sqrt/log/log2/log10/log1p/exp/exp2/expm1/square/
  reciprocal/sin/cos/tan/sinh/cosh/tanh/arc*/conjugate/negative/positive/**sign**/**abs**) are compared
  **BIT-EXACT** on the NaN sign — NumSharp reproduces NumPy's MSVC-UCRT per-path sign (produce-a-NaN
  slots → +NaN; csqrt/clog/cexp/ctanh propagate; csinh/ccosh canonicalise so csin/ccos follow the
  transform negate). A pure sign flip HARD-FAILS via `DiffHasSignFlip` before any ULP excuse. `abs`
  returns float64, so the trigger keys on a complex **operand**, not the result dtype.
- **float16/32/64** stay tokenized, so the tier still gates that a NaN is produced (correct VALUE)
  exactly where NumPy does and the non-NaN outputs are byte-exact, without false-failing the
  non-contractual float NaN SIGN (order/algorithm/platform-dependent — see the float32-sum note).

Green on both frameworks; the only excused entries are ~18 `unary ~ULP` on the non-portable float
`expm1`/`log1p` finite outputs. Teeth verified: reverting any per-path fix turns the tier red with an
explicit "NaN-sign/signed-zero contract violation". This is the permanent form of the one-off
103,229-case raw-byte audit that found the 73 detectable NaN-bit divergences (2 fixed — complex
`abs`/`sign`; the other 71 non-contractual/documented — float reductions, sort, log1p, complex
binary/reductions, floor_divide/mod-f16).

### Truthful vs precise (`precision` tier)

The vision is **byte-identical parity to NumPy**, which fixes the gate hierarchy: **"precise"
(bit-exact to NumPy) always passes, without ever consulting mathematical truth** — matching NumPy's
documented 2.52-ulp-wrong f32 exp IS the contract, and truth structurally cannot turn a
NumPy-matching result red (the comparator returns before truth is read). But when a case DIVERGES
from NumPy, the parity bytes alone cannot say which side lost precision — and the summation-order
excuses were UNBOUNDED, so any magnitude of loss was excused with the same label as a 1-ULP
reassociation.

`precision.jsonl` (72 cases, `gen_oracle.py precision`, needs `mpmath` at generation time only)
closes that. Each case carries a THIRD buffer, `expected.truth`: the correctly-rounded mathematical
reference (exact `Fraction` arithmetic for sum/mean/var/cumsum/prod; 200-bit mpmath for std's sqrt
and expm1/log1p). The inputs are precision-ADVERSARIAL — the ordinary pools cannot stress
accumulation (at 8–36 elements a f32 sum sits ≤1 ULP from exact; at N=2049 a naive loop is 512 ULP
off while NumPy's pairwise is ~2): wide-magnitude sums whose unit elements straddle ulp/2 of the
big element, cancellation triples, mixed-magnitude pseudo-noise, large-mean variance, near-1
products, the expm1/log1p small-|x| band — × contig/negstride × f32/f64.

On a divergence the registry adjudicates by ULP distance to truth (branches P1–P3):

- **not-less-truthful** than NumPy (within 4×/+8 ULP slack of NumPy's own distance) → excused as
  **prefer-precise parity debt**, in two logged flavors: *toward truth* (NumSharp strictly closer —
  still a divergence to close by porting NumPy's algorithm, exactly as exp/log/sin/cos/tanh were
  ported; being more accurate than NumPy is never a win) and *equally truthful* (reassociation
  noise). The slack absorbs cross-host SIMD lane-count variation.
- **less truthful** (beyond slack) → genuine precision LOSS: falls through to the tightly-scoped
  known-bug branches (S1, P3) or FAILS, with the loss quantified in the failure line
  (`truth-ulp NS=… NPY=…`).

The unbounded reduction blanket is gated on `truth == null` so a truth-bearing loss cannot hide in
it; truthless tiers keep it unchanged. **Findings on arrival** (now P3, bounded ≤256): f32 var/std
accumulation loses 55/26 ULP where NumPy's two-pass pairwise sits at 3/2, and the negative-stride
reduce path loses 11–32 ULP where NumPy is EXACT on the same reversed view. 59/100 cases are
bit-exact; scope pins in `MisalignedRegistryTightnessTests` (`P_*`) hold the branch tight from both
sides.

**The AXIS dimension** (28 cases): the flat cases never touch the axis kernels
(`Reduction.Axis.*`), which are different code AND different NumPy behavior — NumPy's axis-0
(outer-axis) reduction is a NAIVE sequential accumulation per column (it loses ALL 2048 unit
elements of the wide-magnitude input, 1024 ULP from truth) while axis-1 runs pairwise (~8 ULP).
Probed and now pinned: **NumSharp bit-matches both**, including reproducing the naive axis-0
order, across sum/mean/var/std × axis 0/1 × keepdims, a transposed (strided-source) view, and
cumsum along each axis. This is the prefer-precise policy protecting itself: a future "improved"
axis accumulation would diverge from NumPy and surface as P1 parity debt instead of passing
silently.

### CBLAS product family (`products` tier)

`products.jsonl` (408 cases, `gen_oracle.py products`) is the FIRST value gate for `inner`,
`vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`, `linalg.multi_dot` and `linalg.matrix_power` —
previously only their error contracts were tested, yet they carry the cells that regress silently:
vdot/vecdot conjugate the FIRST operand (complex), vecdot reduces in the LOOP dtype (int32 stays
int32, not NEP50's int64), tensordot's int/pair axes forms, matrix_power's binary exponentiation.
Two value classes: the SMALL-EXACT bulk (contraction depth ≤4 over `_mm_fill` values — float sums
exact, hence order-independent and bit-comparable even against NumPy's BLAS-backed dot/inner/vdot)
across all 13 dtypes × the call-form matrix, and DEEP-TRUTH f32/f64 cases (K=2049 mixed-magnitude)
carrying `expected.truth`, adjudicated by the prefer-precise branches since NumPy routes those
through BLAS. **397/408 bit-exact**; 8 deep cases prefer-precise-excused (NumSharp CLOSER to truth
than BLAS), 2 f32 deep contractions in P3's bounded known-loss scope, and one complex corrcoef
normalization cell in the explicit ≤2-ULP npy_cdivide envelope. The tier also caught its own
harness trap on arrival: a positional `axis` int to `np.vecdot` silently binds `out=` via the
int→NDArray implicit conversion — the registry passes it BY NAME (documented at the call).

### LAPACK factorisation family (`linalg_parity` tier)

`linalg_parity.jsonl` (366 cases, `gen_oracle.py linalg_parity`) is the FIRST *corpus* value gate for the
LAPACK factorisations — the eigen/SVD/QR/Cholesky family `cholesky`, `eig`, `eigvals`, `eigh`, `eigvalsh`,
`svd`, `svdvals`, `pinv`, `matrix_rank`, `cond`, `lstsq`, `qr`, `norm{2,-2,'nuc'}`, and the **LU family**
`solve`, `inv`, `det`, `slogdet`, `tensorinv`, `tensorsolve` and `matrix_power(n<0)` (added 2026-08-21) —
previously gated only by the interop live-parity suite. **HOST-PINNED exactly like `matmul_parity`** (`linalg_parity.host.jsonl`, same
`MatmulParityPin` shape), because `NumSharp.Core` ships NO managed LU/QR/SVD/eigensolver: these compute
ONLY through the opt-in `NumSharp.Interop.OpenBLAS` backend, and the result bytes come out of a specific
LAPACK build dispatched to a specific CPU kernel. The gate enables that backend before replay and
`Disable()`s after; a host that cannot load NumPy 2.4.2's pinned scipy-openblas (matched by CONTENT
sha256, not file name) goes **Inconclusive, never red**. Pinned at **threads=1** — the deterministic
config the interop suite proves — rather than `matmul_parity`'s ambient max; `gen_linalg_parity()` forces
single-thread via ctypes so the recorded bytes are threading-independent. NumPy's `linalg` is "lite" (it
factorises every operand in double/cdouble and rounds back once, `_commonType`), so **float32 results are
byte-identical too** and int/bool operands widen to float64 — verified across dtypes/layouts/shapes/batched/
degenerate. **366/366 bit-exact** on the pinned host (empirically probed 25/26 before wiring; the sign/phase
freedom of eigenvectors/U/Vh/Q/R is resolved identically because the SAME LAPACK routine runs on both sides).

Tuple results (`svd`→(U,S,Vh), `eig`/`eigh`→(w,v), `qr`→(Q,R)/(h,τ), `lstsq`→(x,res,rank,s)) ride the
`kind:"tuple"` comparator (arity asserted); the array siblings (incl. `svd(compute_uv=False)`→S and
`qr(mode='r')`→R) ride the ordinary bytes contract. Only the **byte-reproducible** surface is recorded —
three outputs are NOT and are deliberately EXCLUDED (covered by the interop suite's reconstruction/tolerance
checks instead, and listed in Table 1):

- **complex-Hermitian `eigh` EIGENVECTORS** — `heevd` does not canonicalize the phase and it is not
  reproducible across processes; the eigenVALUES (`eigvalsh`, and `eigh`'s `[0]` slot for REAL-symmetric
  input) are recorded, but complex-Hermitian cases contribute VALUES ONLY (via `eigvalsh`).
- **float32 `eig`/`eigvals` with COMPLEX eigenvalues** — NumPy yields complex64, NumSharp complex128 (no
  complex64 dtype); float32 eig is recorded only for all-REAL-eigenvalue matrices.
- **`cond`/`norm` orders that are NOT SVD-based** (`fro`/1/-1/±inf) — they compose an elementwise reduction
  whose summation order rounds 1 ULP off NumPy (measured: `cond(a,'fro')` differs in the last byte); only
  the SVD-based orders (`cond` None/2/-2, `norm` 2/-2/'nuc' — exactly the task scope) are recorded.

The **LU family** (`solve`/`inv`/`det`/`slogdet`/`tensorinv`/`tensorsolve` + `matrix_power(n<0)`, added
2026-08-21) reaches `getrf`/`gesv`. Unlike the eigen/SVD factorisations it has **no sign/phase ambiguity** —
LU with partial pivoting is a deterministic function of the input — so **every** output is byte-reproducible
and NOTHING here is excluded (probed 2/2 per case, cross-process). `det` of one matrix is a 0-D scalar and of
a stack is 1-D; `slogdet` is a `kind:"tuple"` `(sign, logabsdet)` where a complex operand's `sign` is a
unit-modulus complex and `logabsdet` stays real; a singular operand gives `det`→0 and `slogdet`→`(0,-inf)`
exactly (same LU product on both sides). `solve` covers NumPy 2.0's b-is-a-vector-iff-1-D rule (vector /
matrix / batched-matrix / broadcast-vector RHS); `tensorinv`/`tensorsolve` are reshape→`inv`/`solve`→reshape;
`matrix_power(n<0)` is `inv(a)**|n|` (portable positive/zero `n` stays in `products.jsonl`). float32 upcasts
to double and rounds back, so it is byte-identical too; int/bool widen to float64 — verified across
dtypes/layouts/shapes/batched/degenerate.

`roots`, `polyfit` and `poly` of a **2-D matrix** also ride this host-pinned tier (added 2026-08-21): they
reach the LAPACK seam (companion-matrix `eigvals` / `lstsq`) and THROW without the backend, so they cannot
be portable. Small operands, threads=1, byte-exact.

### Polynomial family (`poly` tier)

`poly.jsonl` (`gen_oracle.py poly`) is the FIRST value gate for the PORTABLE polynomial family — `poly`
(1-D roots→coefficients), `polyval` (Horner), `vander`, `polyder`, `polyint`, `polyadd`/`polysub`/`polymul`,
`polydiv` (quotient+remainder tuple), and `poly1d` (leading-zero normalisation + construction from roots).
These are pure array arithmetic / convolution / Horner with NO backend and NO long reduction, so they are
**bit-exact vs NumPy everywhere** — probed: Horner evaluation order, leading-zero normalisation and
polynomial long division all match byte-for-byte, across float64/float32/complex128 + small-exact int64 and
strided/reversed reads. The three BACKEND polynomial ops (`roots`/`polyfit`/`poly`-of-a-matrix) ride
`linalg_parity` instead (above). `poly` and `polyint` on int64 return **float64** (NumPy floats them), which
the corpus records; `polyder`/`vander` preserve int64. The pure single-operand ops (poly/polyder/polyint/
vander/poly1d/cov/corrcoef) are **carved out of the blanket "unary ~ULP" excuse** in `MisalignedRegistry`
(they are arithmetic, not transcendental-libm, so a ≤2-ULP drift is a regression — the same narrowing the
ported float32 kernels get).

### einsum (`einsum` tier)

`einsum.jsonl` (`gen_oracle.py einsum`) gates `np.einsum` + `np.einsum_path`. **einsum** is byte-exact vs
NumPy for INTEGER/complex-integer contractions (order-independent) and for SMALL-EXACT float contractions
(short exact sums), plus the whole VIEW path (transpose/diagonal/no-sum/copy) — probed across
matmul/transpose/diag/trace/row-sum/col-sum/full-sum/dot/outer/hadamard/frobenius/batched. **Operands are
NONZERO on purpose:** a signed zero diverges in the outer/hadamard einsums (NumPy's `sop` accumulator, seeded
`+0.0`, absorbs a `-x*0 = -0.0` term into `+0.0` while NumSharp's element-wise multiply keeps the raw
`-0.0`), so zero operands are avoided — signed-zero and larger-float-contraction handling are out of scope
(the latter routes through matmul; NumPy's default einsum uses its own C iterator, so it is NOT byte-exact
and stays small-exact here). **einsum_path** returns the contraction planner's info STRING (`text` kind); it
is shape-derived (operand values irrelevant) and byte-identical to NumPy for non-ellipsis subscripts (the
ellipsis placeholder letters are NumPy's one hash-randomised divergence and are avoided).

### cross / cov / corrcoef (in the `products` tier)

The `products` tier gained the lone product-family gap and the covariance pair (2026-08-21). **`cross`** is
multiply-subtract (`a1*b2 - a2*b1`, …) with NO reduction, so it is bit-exact for float64/float32/complex128/
int64 at every value and layout (int32-and-narrower widen to int64 in NumPy 2.x's cross — a dtype divergence
left out). **`cov`/`corrcoef`** are normalized dot products, byte-exact for the SMALL observation counts
here (the dot is an exact short float sum) across rowvar/bias/ddof/y/complex/int-widen; the WEIGHTED cov path
(`fweights`/`aweights`) rounds 1 ULP off in the `fact` normalisation (measured) and is left to cov's
tolerance battle-tests.

### np.random byte-parity (`random_parity` tiers)

The documented claim — MT19937 with 1-to-1 seed/state parity, "byte-identical sequences" — was
guarded only by STATISTICAL tests (`normal` asserts the mean within 0.01, which a completely
different generator would pass). These tiers pin the actual seeded streams: seed → draw → compare
raw bytes, including `draws: 2` cases that pin stream ADVANCEMENT. Two files because CI replays on
three OSes:

- **`random_parity.jsonl`** (38) — the PORTABLE subset: pure MT19937 bit manipulation +
  exactly-rounded arithmetic (uniform/rand/random_sample/randint/permutation/shuffle/choice incl.
  weighted `p`). Hard-gated everywhere. **All bit-exact.**
- **`random_parity_host.jsonl`** (108) — every transform/rejection sampler consuming libm (gauss
  polar log/sqrt, exponential inversion, gamma/poisson rejection loops, where a 1-ulp libm
  difference flips an accept/reject decision and shifts the whole stream). NumPy calls the CRT and
  NumSharp calls `Math.*` — the same CRT on Windows, glibc elsewhere — so the corpus is
  win-amd64-authored: hard on Windows, Inconclusive elsewhere (the matmul_parity pattern).
  96 bit-exact + 12 under the R1 ≤8-ULP envelope (chisquare/wald/noncentral_f/dirichlet —
  arithmetic-ordering noise on an IDENTICAL stream).

**Findings on arrival — the 1-to-1 claim does NOT hold for 8 samplers** (carved + pinned under
`OpenBugsRandom.RandomParity_*`): gamma(shape<1 via the two-arg API — while `standard_gamma`(any
shape) and gamma(shape≥1, any scale) match byte-for-byte), f, pareto, standard_cauchy, binomial
(both internal algorithms), negative_binomial, multinomial, multivariate_normal. Also documented:
legacy RandomState's int outputs are C-long (int32 on win-amd64, int64 on Linux) while NumSharp
fixes int64 — the corpus records those streams WIDENED to int64 so the VALUES stay hard-gated
(randint and plain `choice` are the exceptions: NumSharp returns int32 there, matching win-amd64;
`choice` WITH `p` returns int64 — an internal inconsistency noted in the generator).

### Decimal (independent oracle — no NumPy analog)

`Decimal` rides an **independent C# oracle** (`gen_decimal_oracle.cs`, naive scalar `System.Decimal`;
`std` is oracled by an independent Newton decimal sqrt, NOT NumSharp's `DecimalMath`) across
unary / binary / reduce (flat + **axis×keepdims** + **empty**) / scan / power (int exponents
**−2…3**) / var / std / matmul / astype (decimal↔int·float·**bool·int16·uint64**) / stat (clip +
median/ptp/percentile/quantile) / where / sort / manip × 13 single + 9 pair layouts — **703 cases
across 12 tiers, all green** except the 15 registry-excused cells (11 argmax/argmin/count_nonzero
result-dtype + 4 std-last-digit, both in Table 1). The one decimal-adjacent finding of the
remediation: `DecimalMath.Pow` matches the exact reciprocal-of-product oracle for negative integer
exponents by value — zero divergence.
