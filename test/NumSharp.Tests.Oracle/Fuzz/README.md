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
  gen_random_oracle.py               the random-API oracle: every overload of the random world by exact C#
                                     signature -> random_api{,_host,_mvn,_lp64}.jsonl, against the inventory
                                     random_surface.json; writes random_numpy_signatures.json (see "The
                                     random-API oracle" below)
test/NumSharp.Tests.Oracle/Fuzz/
  FuzzCorpus.cs                      reconstructs EXACT NDArray views from (dtype,shape,strides,offset,bytes)
  CorpusFile.cs                      one tier as pooled UTF-8: case count up front, cases parsed ONE AT A TIME
                                     (replays stream it — holding a tier's parsed list cost more in GC promotion
                                     than the parse itself)
  CorpusSurvey.cs                    header-only Utf8JsonReader scan of the whole corpus, built once per process
                                     and shared by the coverage gates (surface/strength/journey3/applicability),
                                     which used to parse all ~170 MB eight times; CorpusSurveyTests pins it to
                                     the full parser
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

### Host-dependent values — what the oracle must never assert

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

**The second class: complex results of the platform's C99 complex library.** NumPy computes the
complex128 `sqrt`/`log`/`exp`/`sin`/`cos`/`tan` of `fuzz_random.py`'s pool by calling the platform
library: glibc on Linux, UCRT's `cexp`/`csin`/`ccos`/`ctan` plus NumPy's own msun ports of
`csqrt`/`clog` on win-amd64 (`npy_config.h` blocklists those under `_MSC_VER`). C99 Annex G leaves
the sign of many NaN and infinite results of a non-finite argument unspecified, so the two hosts
disagree, while NumSharp reproduces the win-amd64 bits and the harness holds complex NaN signs
byte-exact on x64 (`ComplexNanContractOps`). When that contract reached master (PR #628,
2026-09-06), the soak went red every night with ~650/200000 "NaN-sign/signed-zero contract
violations", all six ops, all with a NaN or inf input component. One seed generated on both hosts
had identical operands and differed in NaN signs, infinity signs (`csin` → `(+nan, -inf)` vs
`(+nan, +inf)`) and last-bit rounding. `_defuse_complex_nonfinite` now replaces only the non-finite
COMPONENTS of those ops' complex inputs with finite pool values; finite arguments are fully
specified, so what stays is ULP-level rounding the complex-unary excuse already bounds.
`assert_portable` audits it like the casts. The committed `unary`/`specials`/`nan` tiers keep the
non-finite complex edges as win-amd64 bytes, so the NaN-sign contract stays gated there. Measured
on WSL Ubuntu with a real Linux NumPy 2.4.2: that night's 10 seeds went from ~650 divergences each
to 0.

**The runner's ISA is a host, too.** On AVX512_SKX hardware the Linux wheel links Intel SVML
(float64 `exp`/`log` and ~20 more ufuncs) and dispatches its AVX512F float32 `exp`/`log` kernels,
none of which the win-amd64 reference wheel ever runs. Hosted `ubuntu-latest` runners are a mix of
AVX2 and AVX-512 machines, so the soak job (like `interop-test`'s ubuntu leg) pins
`NPY_DISABLE_CPU_FEATURES: X86_V4 AVX512_ICL AVX512_SPR` and logs the runner's raw `AVX512F` flag
next to NumPy's effective `X86_V4` dispatch. To regenerate a corpus on Linux locally, set the same
variable, and use a real Linux interpreter: a WSL `python3` can be a shim to the Windows one.

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
cells (the engine's `Cast(rhs, resultType, copy: true)` parameter-reassign drop). They were
documented in `KnownEscapes` — surfaced green with a per-op **ceiling** (an op leaking more than
its recorded worst still failed) — and worked down to **zero** across four fix waves. The registry
is now EMPTY and its `KnownEscapeFamilies_AreFixed` tracking pin is RETIRED: every op is gated at
zero. The mechanism pin `BinaryScalarCastTemp_IsDisposed` stays. A regression is fixed, not
registered — `KnownEscapes` exists only as the documented escape hatch for a leak that must land
before its fix.

### Coverage completion — every public member is leak-measured (2026-09-23)

An inventory cross-reference (the `coverage/NumSharp.Tools.ApiInventory` surface against the op
keys the sweep actually MEASURED) found 265 of the 961 `[ModuleName]`-module members never
leak-measured: the sweep replayed only the ordinary op tiers. Five layers now close that, all
`[ScopeAudit]` + `[FuzzMatrix]`, all reading process-wide pool counters under the same protocol
(warm invocation, `ScopeAudit.MeasureConfirmedTraffic`, the same escape/bypass verdict):

| Layer | What it measures | Counts (2026-09-23, identical net10.0 / net8.0) |
|---|---|---|
| `Corpus_AllOps_…` (`SharedSweep`) | three corpus families — ordinary op tiers, masked-array tiers (`ma_*`, `NDMaskedArray` operands through `OpRegistry.ApplyMasked`), index tiers (`index_*`, the indexer get/set) — and EVERY error path (a NumPy-raising case must throw without stranding a buffer) | 213,264 success (ordinary 138,660 / masked 68,860 / index 5,744) + 9,123 error paths |
| `Corpus_BackendOps_…` (`SharedBackendSweep`) | the ordinary tiers replayed with OpenBLAS installed (threads=1): the LAPACK family Core cannot compute without a backend, the BLAS product seams, the Interop glue. `Inconclusive` where no library loads | 138,936 + 2,441 error paths |
| `Catalogue_EveryEntry_…` (`SharedCatalogue`) | `LeakCatalogue*.cs` — one direct invocation per member NO corpus row reaches (np.* conveniences, the `np.linalg` Array-API forms, `ndarray`/`NDArray<T>`/`NDMaskedArray` methods, every operator, object surfaces and indexers). Backend-only entries run under OpenBLAS; `T(...)` entries (members that always raise) are measured as error paths; an entry that cannot run is a HARNESS ERROR and fails | 474 of 482 entries measured (8 always-raise error paths) |
| `EveryPropertyAndField_Read_…` (`SharedPropertyReads`) | every surface property/field read — settable ones round-trip written — on targets built INSIDE the measured region, so a getter caching an allocation on its owner is caught when the owner is released | 539 reads + 15 error paths |
| `EveryInventoryMember_IsLeakAudited` | the COMPLETENESS gate (`LeakSurfaceCoverageTests`): every member of `LeakSurface` — the `[ModuleName]` modules, the operators of `ndarray`/`NDArray<T>`/`NDMaskedArray` and of every object owner that declares any (`DType`, `NDArrayFlags`, `poly1d`), the mapped object surfaces — must be credited by a MEASUREMENT from one of the four runs above | 1,348 / 1,348 members |

Credit follows measurement, never declaration: a catalogue entry or property read counts only when
its run actually exercised the member (`DirectRunResult.Attempted`), a LAPACK-only op only when the
backend pass measured it, and a corpus key only on a SUCCESS path (an op whose every case raises has
an unaudited success path). Two environmental outcomes are credited because they are never red
elsewhere either: a GC-inconclusive measurement (the member ran; only the host withheld the verdict)
and — only on a host where no CBLAS/LAPACK library loads — a backend-skipped one. The gate also
fails when an object owner declares an operator outside `LeakSurface.OperatorOwners`, when a
catalogue/alias-map id names no live member, or when a mapping went stale. Non-vacuity floors sit
5% under the counts above. **To audit a new public API:** give it corpus rows (`gen_oracle.py` +
`OpRegistry`) or a catalogue entry (`E(...)`/`T(...)` in `LeakCatalogue*.cs`); a new property is read
automatically unless its owner needs a new read target (`ReadTargetsFor`).

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
guard-page allocator ARE the chokepoints; NDIter's state block and buffered-mode scratch (6 sites —
`NDIter.cs` tightened 2 → 1 on 2026-09-23), the privatized counting tables of `np.bincount` and the
fused histogram kernel, the managed LU's scratch, and the >int.MaxValue-capable sort/partition line
buffers (`AxisSort`/`AxisPartition`) are carried as audit debt (each an alloc+free pair inside one
call); and ANY new raw site — new file or count growth in an allowed file — is red until pooled or
consciously allowlisted. A file whose count DROPS below its pin prints a "tighten the allowlist"
note. Inconclusive (never false-green) without a source checkout.

## Regenerating the corpus

```bash
python test/oracle/gen_oracle.py astype_full      # 13x13 dtypes x 26 layouts (host-sensitive, see above)
python test/oracle/gen_oracle.py binary           # add/sub/mul/divide x NEP50 pairs x pairwise layouts
python test/oracle/gen_oracle.py divmod_power     # floor_divide/mod (bit-exact, F1) + complex power (integer exp bit-exact; non-integer exp Misaligned)
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
python test/oracle/gen_oracle.py polyeval         # np.polynomial.* evaluation family (module-qualified keys)
python test/oracle/gen_oracle.py polyseries       # np.polynomial.* additive family + polyutils + the module constants
python test/oracle/gen_oracle.py polycalc         # np.polynomial.* calculus family ({p}der / {p}int, every parameter)
python test/oracle/gen_oracle.py polyalgebra      # np.polynomial.* series algebra (+ the host-pinned polyalgebra_parity tier)
python test/oracle/gen_oracle.py polyvander       # np.polynomial.* Vandermonde family ({p}vander / {p}vander2d / {p}vander3d)
python test/oracle/gen_oracle.py polyroots        # np.polynomial.* companion / roots (+ the host-pinned polyroots_parity tier; needs OPENBLAS_NUM_THREADS=1)
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

The nightly **soak** (`.github/workflows/fuzz-soak.yml`) sweeps one fixed seed plus nine fresh random
seeds, 200K cases each (~2M cases/night, ~1.8M of them new draws). The generator is deterministic, so
the fixed seed replays an identical corpus every night: a deterministic canary, and a `source_sha256`
in the uploaded evidence that should repeat until the generator or the NumPy pin changes (a new value
means NumPy answered differently on that runner). A divergence prints a shrunk minimal repro — copy it
into `corpus/regressions/` so `FuzzRegression` pins it on every CI thereafter.

### Coverage join — per-API parity evidence

The API coverage tool (`coverage/generate_coverage.py`, the "NumPy API Coverage & Support" dashboard)
joins every committed contract of this corpus — plus the `.npy`, flags and layout oracles in
`test/NumSharp.Tests/` — onto the catalog row of the API it exercises, so each NumPy API reads
**available** (the compiled surface has it) and, separately, **oracle-verified** (committed NumPy output
replays against it here). The join lives in `coverage/oracle_evidence.py`; the reviewed table is
`coverage/oracle_map.json`. It is STRICT, and that has two consequences for anyone adding corpus cases:

* **A new op key must resolve.** Bare keys resolve to `np.<key>` first (then `np.linalg`/`np.fft`/
  `np.random`, then a NumSharp-only `np` row); dotted keys by namespace (`ndarray.`, `emath.`, `ma.`, the six
  polynomial bases). Anything else — a variant suffix (`std_ddof`), a parameter-driven key
  (`out_binary` + `params.ufunc`), a protocol (`get`/`set`) — needs a `fuzz.keys` entry, or the coverage
  generator fails naming the key. Entries that go unused, or merely repeat the automatic rules, fail too.
* **A new corpus file must be replayed.** A `*.jsonl` no replay suite (`FuzzCorpusTests*.cs`,
  `IndexOracleTests.cs`) names as a string literal fails the join: its contracts would prove nothing.
  Files run through `RunHostLibmCorpus(...)` are discovered as host-pinned automatically; other pinned
  tiers are declared under `host_pins`.

Aliases mirror `OracleSurfaceCoverageTests`' `EquivalentAliases`/`NdarrayAliases`/`MaAliases` (a Python
test fails if a C# alias has no counterpart), plus an automatic identity rule: a row credited to the SAME
NumSharp member as a gated row, whose NumPy object is the SAME object (`np.acos is np.arccos`), shares its
contracts. Gate: `python -m unittest discover -s coverage -p 'test_oracle_evidence.py'`.

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
| complex-input ufunc rejection (cbrt/floor/ceil/trunc/deg2rad/rad2deg/floor_divide/mod) | K4 | **FIXED** — each `Default.<Op>` guard now raises NumPy's exact `TypeError("ufunc '<name>' not supported for the input types…")` (mod→'remainder') instead of a kernel `NotSupportedException`; excuse deleted |
| …and the rejection on a **zero-size** complex operand | K5 | **FIXED** — the K4 guard keys off the input DTYPE, not the data, so a zero-size complex operand is rejected too (NumPy validates the loop, not the data); excuse deleted |
| `power(bool, negative int)` misses the integer-power guard | K6 | known bug |
| `power(int, negative int)` trips `Debug.Fail("index < Count, Memory corruption expected")` instead of NumPy's ValueError — in Release that path has no assert | K8 | known bug (memory safety) |
| `result_type(mixed signed/unsigned, 0-D operand)` throws instead of resolving | K9 | known bug |
| NEP50 weak-scalar reached via the error path (int64+uint64 succeeds where NumPy refuses) | K7 | intended |
| ufunc `out=` on a read-only **broadcast** view (983 cases) | K10 | **FIXED** — `ThrowReadOnly` now raises NumPy's exact `ValueError("output array is read-only")` (was `NumSharpException`, same text) so these pass without an excuse; the write-through was already prevented by `ThrowIfNotWriteable` |
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
algorithmic remainder is complex `corrcoef`: the complex true-division itself is now BIT-EXACT
(`ComplexDivideNumPy` ports NumPy's `CDOUBLE_divide` Smith's algorithm), so corrcoef's residual comes
solely from `cov`'s managed complex GEMM (`np.dot`) — it has its own ≤2-ULP branch scoped to that GEMM
instead of hiding under the broad complex-unary envelope.

### The 2026-09-18 coverage expansion (docs/plans/oracle-coverage-expansion.md, phases P0–P3)

The assertion-kind and parameter axes were systematically widened; every cell below is now GATED
(surface + applicability), not just present.

**New tiers.** `instance.jsonl` (5,410 — the whole `ndarray.*` INSTANCE surface, row G0: dual-form
methods through their instance defaults, instance-only members `item`/`tobytes`/`view`/`getfield`/
`byteswap`/`__len__`, the D4 property reads `T`/`mT`/`real`/`imag`/`flat`/`nbytes`/`itemsize`/
`ndim`/`size`/`strides`, `nonzero`'s tuple, and the IN-PLACE mutators `sort`/`fill`/`put`/`resize`
compared as **[post-call view, post-call whole base buffer]** — the out_where two-slot contract, so
a mutator writing outside a strided view's window is caught; op keys carry the `ndarray.` prefix and
`MisalignedRegistry` strips it so the shared excuses apply) and `emath.jsonl` (327 — the scimath
module promoted into the corpus: the real→complex promotion DECISION over the complex128/float64
lanes; int8/16/uint16/float32/float16 promote to NumPy's complex64 and stay sibling-owned, #569).

**Widened tiers.** `out_where.jsonl` +240: `out=` beyond the elementwise ufuncs — `out_scan`
(cumsum/cumprod; complex cumPROD carved exactly as nanscan carves it), `out_round`, `out_clip`
(scalar 0-D bounds; the read-only broadcast out refusal included), `out_nanarg` (nanargmax/
nanargmin). `params.jsonl` +288: the §C1 multi-axis cells — `axes` int[] for the reductions with a
tuple-axis overload (median/average/nanmedian; sum/prod/min/max have NO int[] overload yet — a
tracked feature gap the applicability matrix warns on, deliberately NOT generated). `errors_full.jsonl`
714→801 over 22→50 distinct messages: curated §B1 recipes (reshape rejection family, expand_dims OOB
both signs, flip axis/repeated-axis, matrix_transpose ndim, take/put OOB + float-index dtype
rejection, partition kth OOB, percentile/quantile q-range, linalg 1-D/non-square/float16 validation,
fft n-guard). `nan.jsonl` 120→176: the §B3 BINARY special-pair CROSS grid (FLOAT_VALS × FLOAT_VALS,
both NaN signs) over 19 f64/f32 ops + a 13-op f16 subset (the widen-compute-narrow/bit-level lanes)
+ 5 complex128 binaries against a rolled grid.

**Schema addition — `operand.writeable`.** `layout_catalog.describe()` now serializes
`"writeable": false` for a read-only view whose read-onlyness the strides CANNOT convey — a
SAME-SHAPE `np.broadcast_to` keeps ordinary strides — and `FuzzCorpus.Reconstruct` clears the flag
via the public `setflags(write: false)` route. This closed a REAL gate hole: since the K10 excuse
deletion (60024b44), the 293 one-D `out=broadcast` refusal cells reconstructed WRITEABLE, NumSharp
computed a result where NumPy raises, and the OutWhere tier had been silently red. Emitted only when
False, so every pre-flag corpus row stays byte-identical.

**Real bugs found + fixed by the new coverage** (each with its corpus cells retained as regression
proof): `np.trace`/`ndarray.trace` promoted **unsigned** narrow lanes (uint8/uint16/Char) to int64
where NumPy's add.reduce rule gives **uint64** — the pre-existing trace corpus covered only signed/
float lanes, so the wrong dtype AND wrong wrap point were invisible (`DirectILKernelGenerator.Trace`
both maps fixed); `ndarray.flat` over a **0-d** array returned 0-d where NumPy yields shape (1,);
`np.fmod` with a complex operand leaked the kernel `NotSupportedException` instead of NumPy's
verbatim no-loop `TypeError` (fmod landed after the K4/K5 sweep and missed the guard); `np.take`
mode='raise' OOB said "for axis with size" — NumPy says "for axis {n} with size" on an axis take
and drops the clause entirely on a flat one (both spellings now exact); `ndarray.item()` on size>1
and `__len__()` on 0-d now raise NumPy's verbatim texts ("can only convert an array of size 1 to a
Python scalar" / "len() of unsized object").

**New/extended excuse branches** (Table 1 additions): *argsort tie order* — NumPy's default
introsort is UNSTABLE, so ±0.0/duplicate ties resolve arbitrarily while NumSharp's radix argsort is
stable; excused ONLY when `ArgsortPermutationsEquivalent` re-derives from the operand bytes that
both index vectors are in-range permutations selecting pairwise IEEE-equal values (the main sort
tier uses distinct values and never reaches it). *K13 extended to `out_clip`* — clip IS
maximum(minimum(x, hi), lo), so a ±0-sign result inherits the same non-contractual lane-dependence;
still guarded by every-diff-is-a-pure-±0-flip. The `ndarray.`/`emath.` prefixes are stripped at
`Classify` entry (the `ma.` convention), so instance/emath spellings ride the shared branches.

**New gates.** `OracleSurfaceCoverageTests` now reflects ALL 8 `[ModuleName]` facades: `ndarray`
(lowercase = NumPy-named and must be corpus/`ndarray.*`-covered or classified; PascalCase = C#
infra by convention; the property row included), `np.emath` (100% corpus-covered — no
classifications allowed), `np.ma` (`ma.*` keys + aliases + sibling ledger), `np.dtypes` (the 29
class properties, DTypes-suite-owned), each with the stale-classification self-retirement rule.
`OracleApplicabilityTests` is the plan's §M1–M3 mechanization: a 22-row declarative manifest of
(group × assertion-kind) cells — Enforced cells assert op-count floors (MinCases-style ratchets),
Warn cells PRINT their missing ops (the driven checklist for the next session), NotApplicable
requires a reason and fails if coverage appears anyway; kinds a row omits are covered by its
one-line `InapplicableNote`. Applied kinds are computed from the corpus itself (out_* vehicle keys
credit the ufunc named in their params; masked kinds normalize to array/tuple).

### The dtype-spread validation (2026-09-18, follow-up pass) — "do all oracles cover all 15 dtypes?"

A full per-op × dtype scan of every committed corpus (612 op keys) answered the question and made
the answer a GATE. The model: the NumPy corpus expresses 13 dtypes directly, **Char** rides the
uint16 proxy weave, **Decimal** rides the independent C# oracle (`decimal_*.jsonl`) — and a dtype
counts as covered for an op when it appears as an operand of a value OR error cell (a gated
rejection IS coverage of the combination).

**Axes widened by the scan** (all bit-exact on regeneration — no new NumSharp bugs, which is
itself a result: the char instance weave alone re-proved Char ≡ uint16 across ~50 instance
methods, in-place mutators included):
`instance.jsonl` 5,414→9,463 (8→13 dtypes + `char_tier("instance")` — the whole tier re-runs on
the proxy, dedicated dot/searchsorted/choose + resize jobs included; `_relabel_dtype` now recurses
into tuple `slots`, without which every char in-place mutator kept "uint16" slot dtypes and failed);
`modf.jsonl` 128→208 (the 59f99320 per-width promotion made every non-complex lane computable —
the old 4-dtype list predated it; + `char_tier("modf")`); `nanreduce.jsonl` +4,150
(nanpercentile/nanquantile integer/bool lanes — legal NumPy, degenerates to percentile; the
generator's NaN-laced float pool would RAISE at construction for unsigned/bool, so integer lanes
get an int64-built modular-astype pool — a float→uint astype is C-undefined and must never seed an
oracle); `emath.jsonl` 329→482 (unsigned lanes — no negatives means the `any(x<0)` trigger never
promotes, and `|x|>1` cells that would land complex64 auto-skip); `out_where.jsonl` +168
(float16/uint8 lanes through out_scan/out_round/out_clip/out_nanarg); `ndarray.view` gained the
int16/uint16/uint32/uint64 same-size reinterpret pairs.

**The gate** — `OracleCoverageStrengthTests.EveryOrdinaryOp_MeetsItsDtypeSpreadFloor` (same file
skips as the case-count gate: host pins, `index_`, `ma_`): every op must reach **≥ 4 distinct
dtypes** or carry a one-line entry in `FixedDtypeOps` (61 reviewed reasons in five classes:
fixed-output generators — windows/fftfreq/index-coordinate builders; dtype-axis-in-PARAMS —
can_cast/promote_types/min_scalar_type sweep dtype pairs the operand metric cannot see; PRNG
protocol; the host-pinned LAPACK/CBLAS f32/f64/c128 family; variant keys whose primary op carries
the axis — modf-tuple/std_ddof/average_returned; plus per-op semantics like einsum's small-exact
lanes and unique_values' [Misaligned] hash-order carve). The ledger is self-retiring BOTH ways: an
entry whose op vanished or whose spread grew past the floor fails. Rule 2 is 15 per-dtype GLOBAL
op-count floors (bool 298 … char 216 … decimal 117 … complex128 369 — ~95 % of the post-widening
spread), so a regeneration that silently drops a dtype axis turns the gate red.

**The sibling oracles' posture** (validated, no changes needed): the **.npy format oracle** covers
every expressible dtype map entry incl. `<U1`/Char, big-endian variants and the `<c8` widening,
with Decimal's rejection itself gated; the **advanced-indexing oracle** carries a dedicated
13-dtype tier (`index_dtype.jsonl`, 8 cases each) + a setter-dtype tier; the **flags** and
**layout-parity** oracles spot-sweep all 13 NumPy dtypes (×6 / ×1-6) around an int64/f64 bulk —
correct for their contract, since flags/view semantics depend on itemsize, not lane type (Char and
Decimal behave as uint16/16-byte lanes; the flags oracle is under active work in a parallel
session). **Decimal** remains gated exclusively by `decimal_*.jsonl` (124 op keys) — by design,
per the decimal-coverage expansion; ops absent there raise or ride shared engine paths, and
widening that set is C#-oracle work tracked in its own memory topic.

### Table 1 — live `MisalignedRegistry` excuse branches

**Intended / algorithmic differences (permanent):**

| Excuse class | Scope | Hits |
|---|---|---|
| NEP50 weak-scalar: 0-D operand promoted weakly | any multi-operand op × Dtype kind, 0-D operand present | 261 |
| **F1** `np.fft(float32/float16)`: NumPy returns complex64/float32/float16, NumSharp complex128/float64 — a dtype-ONLY divergence (values = the correctly-rounded double result; bit-verified `fft(x32) == fft(x32.astype(f8))`). NumSharp has one complex type. The unnameable complex64 is routed to a Dtype divergence by `CompareArray`. Contractual dtypes (float64/complex128/int/bool) are bit-exact; the helpers never diverge | 14 fft transforms × {float32,float16} input × Dtype kind | 516 |
| unary ~ULP (transcendental/magnitude algorithm difference) | single-operand × Value, every diff ≤2 ULP — **EXCEPT exp/log/sin/cos/rad2deg/deg2rad at a float32 result, which are gated bit-exact** (see below) | 563 |
| complex unary within 3 ULP (full NumPy-algorithm port) — FINITE interior only; the NaN SIGN of these ops is now compared **raw-byte** (`ComplexNanContractOps`, `BitDiff.Compare(nanBitExact:true)`) and a pure NaN-sign / signed-zero flip HARD-FAILS via `DiffHasSignFlip` before any ULP excuse runs (NumSharp reproduces NumPy 2.4.2 win-amd64 / MSVC UCRT NaN signs bit-for-bit) | complex unary × Value, ≤3 ULP, no sign flip | 11 |
| complex arccos/arccosh/sinh/cosh (+ sin/cos routing through sinh/cosh) pathological FINITE edge (sub-DBL_MIN denormal-real flush / \|x\|∈[710,710.13] overflow boundary) — the former "cos/sin NaN zero-sign" regime is GONE (now byte-exact) | those ops × complex × Value, finite | 0 |
| complex corrcoef: `cov`'s managed complex GEMM vs zgemm (division itself is now bit-exact) | corrcoef × complex input/result × Value, ≤2 ULP | 1 |
| complex add/subtract within 2 ULP (FMA contraction) | add/subtract × complex × Value, ≤2 ULP | 0 |
| complex power, NON-integer/complex exponent ~ULP / gross inf-NaN edge (Complex.Pow vs npy_cpow's host cpow) (F5, ledger L6) — INTEGER exponents are now BIT-EXACT (ComplexPowNumPy ports npy_cpow's exact repeated-multiplication branch), so this is scoped OFF integer exponents | power × complex × non-integer exponent × Value, ≤512 element-magnitude ULP or non-finite | 30 |
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
| cumprod(size-1 int) NEP50 widening — FIXED (ReduceCumMul casts to GetAccumulatingType; excuse removed, now regression-guarded) | cumprod × Dtype, element-count ≤1 | 0 | |
| modf(float16/int): no Half kernel, no int→float64 promotion | modf × dtype ∉ {f32,f64} × Threw | 32 | |
| unary hyperbolic/inverse-trig/angle: no Half kernel | sinh…arctan × {bool,i8,u8,f16} (+deg2rad/rad2deg×c128) × Threw | 0 | |
| unary preserve-dtype pending: square/floor/ceil/trunc widen int→float64 | those 4 ops × Dtype | 78 | F3b |
| reduction result dtype differs (NEP50 accumulator / complex→real) — sum/prod/cumprod NOW bit-exact on EVERY size incl. flat 0-d/size-1 (fixed 2026-09-13) & EXCLUDED from the excuse (regression-guarded); residual = std/var complex→real (+decimal) | reductions × Dtype (excl. sum/prod/cumprod) | 85 | #10 |
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
| `gen_random_parity` carve list | `multivariate_normal` only: byte-identical only through a LAPACK backend's `gesdd` (the managed Jacobi SVD flips a singular vector's sign). The other seven samplers (gamma shape<1, f, pareto, standard_cauchy, binomial, negative_binomial, multinomial) were uncarved 2026-09-25 — legacy-distributions.c ports — and every sampler is also covered by the `random_api` tiers | `OpenBugsRandom.RandomParity_MultivariateNormal_Seed42_ShouldMatchNumPy` (the seven former pins are ordinary tests) |

**FIXED on this branch or before it** (classifier branch/carve removed — the matrix now verifies
these bit-exact): complex→bool imaginary drop · floor_divide/mod integer ÷0/±inf/signed-floor (F1)
· NaN `<=`/`>=` (F2) · transcendental width-based promotion (F3a) · negative(uint) + integer
reciprocal (F4) · bool arithmetic True+True (F6) · size-1 result collapse (F7) · complex np.where
zero-push · maximum/minimum/fmax/fmin direct ufuncs + NaN-propagating clip/out= SIMD ·
exp2 malformed-IL crash (W3-C) · power(float16) scalar-broadcast crash (W1-B) ·
**invert(float/complex/decimal) illegal-instruction crash** (guard @ `Default.Invert.cs`, pinned by
5 always-run tests in `FuzzGateRegressionTests`) · **convolve(complex) discarded the imaginary
dimension** + int64/decimal/bool convolve accumulator (ledger L5, @737c59d6) · **all/any Half+Complex
ignored `Shape.offset`** (ledger L4, @7804b2ad) · **round_(char)→Double** (ledger L8, @1a9cfa9f) ·
**complex convolve / correlate with infinities or NaNs** (zdotu's C99 result construction turns a non-finite
imaginary part's `im*0` into a NaN real part, while a one-element operand with a non-positive stride takes
CDOUBLE_dot's plain loop; the `groupa` tier's complex block, the U2 wholeness pass).

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
normalization cell in the explicit ≤2-ULP managed-complex-GEMM envelope (its complex division is now
bit-exact — `ComplexDivideNumPy` ports `CDOUBLE_divide`). The tier also caught its own
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

### numpy.polynomial evaluation family (`polyeval` tier)

`polyeval.jsonl` (`gen_oracle.py polyeval`, 19,216 cases, floor 19,100) gates the PACKAGE evaluation
family — `{p}val`/`{p}val2d`/`{p}val3d`/`{p}grid2d`/`{p}grid3d` for the six bases (`polynomial`,
`chebyshev`, `legendre`, `laguerre`, `hermite`, `hermite_e`) plus NumSharp's `{p}valnd` twins — **bit-exact,
0 excused**. Op keys are MODULE-QUALIFIED (`chebyshev.chebval`) because the package reuses the legacy
`np.polyval`/`polyder` names with the OPPOSITE coefficient order: a bare key would credit one family with
the other's corpus (the coverage join and `OracleSurfaceCoverageTests.PolynomialSurface_*` both key on the
qualified name). `params.xs` records each x position's form — an ndarray operand (0-d arrays included,
STRONG) or a weak Python int/float/complex/bool (NEP 50: `chebval(2.0, float32_c)` is float32, and CPython
computes `2*x`/`x*0`/`(2*nd-1) - x` before NumPy sees them; ints may exceed 64 bits). The matrix: every x
dtype × 8 series dtypes × coefficient counts 1/2/3/4/7/12, 26 x layouts, N-D series (tensor True/False,
transposed/reversed/broadcast), Python-scalar and 0-d x, NaN/±inf/-0 coefficients, every vector-lane width
at 45 points (1-D and per-point series, at the dtype extremes, and column-strided), and the error cells
(incl. `lagval`'s OverflowError); Char rides the uint16 proxy. Section (J), 720 cells from the 2026-09-29 wholeness
pass, covers the x forms a C# caller spells differently: a Python-sequence x (tuples, nested lists and tuples, mixed
nesting, empty, complex and bool items, a float16 0-d item, ragged → NumPy's inhomogeneous ValueError), coerced by
`NDPolySequence`'s port of `np.array`, and a Python int x past int64 (a C# `BigInteger`, WEAK: CPython's int
arithmetic on the x-only terms, then NumPy's conversion — inf or nan, and OverflowError past float64). Section (K),
1,890 cells from the same pass, gives c and the ordinates as Python values too (`params["c"]` holds c's spec when it is
not an operand): a list / tuple / nested / scalar / bool / complex / NumPy-scalar-item / empty / ragged / uint64 c for
`{p}val` at a weak, an array and a list x (`tensor=False` on the N-D series); list / tuple / Python-scalar / mixed /
nested / empty ordinates for `{p}val2d`/`{p}val3d`/`{p}grid2d`/`{p}grid3d`/`valnd` against an array, a float32 array
(a strong int64 ordinate makes it float64, a weak grid scalar keeps float32), a list and a ragged c; and the three
error orders — a ragged c before a ragged x (the two ragged at DIFFERENT depths, so the texts differ), a ragged
ordinate before the shape check, the shape check before a ragged c. Section (I), 4,392 cells, covers single-element
broadcasts: an N-D series at a per-point x whose ONE-element result comes from operands of different ndim.
NumPy runs those complex products on NpyIter's stride-0 loop, i.e. `CDOUBLE_multiply`'s MSVC-contracted
fallback, not `simd_cmul`. The cells cover:
- every basis at 8 shape configurations x 4 dtype pairs x nc 1-7;
- the 2-D/3-D/N-D compositions whose passes reach it;
- the more-than-one-element near misses that must stay `simd_cmul`.

Their values are seeded random FULL-MANTISSA draws, because this tier's moderate values make the forms
coincide. Planted-bug check: 12 mutants, 11 killed, and the survivor (the complex-loop gate) is value-equivalent
by design. Design and measurements: `docs/plans/numpy-polynomial-review.md`. Cells
whose NumPy result is complex64 are skipped (one complex width, #569). `OpRegistry.Polynomial.cs` replays it;
`MisalignedRegistry`'s generic unary/complex ULP branches are carved out for these ops, so any drift fails.

### numpy.polynomial additive family + polyutils (`polyseries` tier)

`polyseries.jsonl` (`gen_oracle.py polyseries`, 18,698 cases, floor 18,690) gates plan unit U1 — `{p}add`,
`{p}sub`, `{p}trim`, `{p}line` for the six bases, the 24 module constants `{p}domain/zero/one/x`, and
`polyutils.as_series/trimseq/trimcoef/getdomain/mapparms/mapdomain` — **bit-exact, 0 excused**. Keys are
module-qualified like `polyeval`'s. Arguments are NAMED (`c1`, `c2`, `c`, `tol`, `off`, `scl`, `old`, `new`, `x`,
`alist`, `seq`) and each is `"a"` (the next operand) or a Python-typed spec (int/float/complex/bool/list/tuple/str),
so the facade sees the same KIND of argument NumPy saw — the kind decides whether CPython, NumPy's scalarmath or a
ufunc computes `mapparms`/`mapdomain`/`{p}line`. The matrix: the full dtype-pair matrix of add/sub on the power
basis (a subset on the others) × trailing-zero/all-zero/NaN-tail patterns × layouts, 0-d and broadcast operands,
Python lists; trimcoef tolerances (weak Python vs strong NumPy, NaN, negative, complex, huge ints); trimseq views;
as_series (tuple kind, arity asserted); getdomain (complex corners, signed zeros, strided); mapparms over Python
tuples/lists of ints/floats/complexes (ints past 2^53, 2^63 and 2^64; `long` overflow inside a difference or
product; `0 / -5 == -0.0`; both ZeroDivisionError texts), same-dtype 1-D array domains of every dtype (wrapping
int intermediates, float specials, strided/reversed views), mixed Python/NumPy kinds, `pycomplex op np.float64`,
2-D domains; mapdomain for every x dtype × domain form; `{p}line` over Python and NumPy scalars of every dtype; and
four FACETS of each constant (value, identity, writeable, owndata — a constant has no argument to vary).

Four sections were added by the U1 parity audit (2026-09-27/28), because the original matrix never went past 12
coefficients and so never reached the house-kernel routes a long series takes (every contiguous/strided series of
64+ elements, mapdomain's block-converting affine route): **(K) long series** — add/sub/legsub across the 64-element
threshold (63/64/65, 70/64, 257/100 … 1,000/1,000, 1,003/5) in float64/float32/float16/complex128, twelve mixed dtype
pairs, contiguous/stride-2/reversed operand pairs, trailing zeros trimmed below the threshold, a cancelling tail, and
specials (NaN, ±inf, ±0, subnormals, overflowing sums) in the updated prefix and the untouched tail; as_series/
trimcoef/getdomain/mapdomain at 100 and 1,000 points over every dtype and layout (1,458 cases). **(M) block and
staging boundaries** (45 cases, appended after L) — the 1,024-element blocks the in-place combine materializes a
converted / strided / negated operand in (add/sub of 2,100 against 2,150 coefficients: int32+float64 both ways,
float64 negated, float32 stride-2, int16 reversed, float16+float32), as_series' blocked strided cast (int16 stride-2,
int64 reversed, uint32 stride-2, float32 reversed at 2,100), mapdomain's affine route over whole blocks and a partial
last one (contiguous 8/16/64-bit integers at 2,100 points into a float64 loop, int8/uint8 at 1,030 into a float32
loop, float16 at 600; int64 reversed, int16/uint8/int32/float32 stride-2 at 2,100), and the fused pass's staging of a
transposed 2-D x (96×90 = 8,640 points, past the 8,192 threshold). **(L) complex64
ARRAY loops of mapdomain** (552 cases) — a complex64 off/scl (mapparms of a float16/float32 domain and a Python
complex tuple) with every narrow x, and a Python complex tuple domain with a float16/float32 x, at lengths 1
(NumPy's trivially-iterable single-element call still runs `simd_cmul`) through vector-body lengths and strided/
reversed layouts. NumSharp has no complex64 dtype (#569): those loops run on its float32 kernels with NumPy's
product forms, the result is carried as complex128 holding the float32-exact values, and the replay VALUE-compares
the up-cast pairs (`Complex64ValuesMatch`) — 752 complex64-result cases, 0 excused. **(N) getdomain across the
blocked pass's windows** (72 cases, appended after M) — getdomain reduces NumPy's copy 8 KB at a time (in place, or
packed into an L1 scratch) and carries its min/max accumulators across windows, so every dtype the pass serves
crosses a window boundary (window + 37 elements) contiguous / stride-2 / reversed, plus stride 3 for int64 / float64 /
uint8 (the AVX2 gather and the scalar sub-word copy); and the schedule facts the carried accumulators must keep are
pinned at float64 and float32 over two windows with no scalar tail: a -0.0 early in the HIGHEST lane against a +0.0
late in lane 0 (NumPy returns the early zero — the cascade lets the higher lane win the tie; a sequential fold would
return the late one — and the mirror for min), and one NaN with a payload that comes back canonical from a window
boundary inside the vector section and with its payload from the scalar tail. **(O) Python tuples and nested
sequences** (184 cases, appended after N, from the 2026-09-29 wholeness pass) — the C# replay's `ValueTuple` (a Python
tuple) and nested `object[]` (lists), coerced by `NDPolySequence`'s port of `np.array`: tuple operands of
`polyadd`/`polysub`/`chebadd`/`chebsub` on either side (ints, floats, mixed with a complex, nested → "not 1-d", tuples
of float32 arrays, the empty tuple, ragged → NumPy's inhomogeneous ValueError), `as_series` iterating a tuple (of
arrays, tuples, lists, scalars; nested and ragged items; trim on and off), and tuple / nested / ragged / empty inputs
of `trimcoef`, `getdomain` and `mapdomain`. **(P) `trimseq` of Python sequences** (26 cases) — NumPy returns the KIND
it was given (the list / tuple itself or its slice); the corpus records `np.asarray` of the result and the replay
compares the same coercion of what the facade returned: lists and tuples (trailing zeros, none, all, empty, a
10-item tuple past C#'s 7-item flat ValueTuple), NaN / -0.0 / complex / bool items, float16 0-d items, one-element
array items (by their truth value), and NumPy's truth-value errors for a tested array item of two elements or none.
**(Q) the deferred object / str refusal** (51 cases, from the U2 wholeness pass) — a None / str / oversized-int series
makes NumPy's array an object or str one, which as_series still checks for size and dims (every argument) before its
common type fails (str, bool) or it computes with Python objects (NumSharp refuses there — not recorded): add / sub of
the power and Chebyshev bases with the bad series on either side, as_series of mixed lists, getdomain / trimcoef (and
trimcoef's `tol < 0` check, which runs first), filtered by NumPy's outcome (`_pa_object_land`).

`OpRegistry.PolySeries.cs` replays it. **Every `mapparms`/`mapdomain` case runs three routes** — the object
overload, the generic tuple overload (tuples rebuilt element-typed by reflection, so a Python int is a `long`) and
`NDPolySeries.MapParmsGeneral` (the exact `BigInteger` reference) — which must agree to the byte (or raise the same
type and text) before the result is compared with NumPy; this is what proves the CPython machine-number lane and
the fused same-dtype kernel. Planted-bug checks: dropping the lane's product-overflow bail (27 red), corrupting the
list lane's `long` read (205 red), feeding the fused kernel the wrong numerator (129 red). Cells whose NumPy result
is a complex64 ARRAY outside mapdomain (a `{p}add` of float32 coefficients and a Python complex list, …) or an
object array are skipped (109) — every complex64 SCALAR result and every mapdomain complex64 array is recorded.

### numpy.polynomial calculus family (`polycalc` tier)

`polycalc.jsonl` (`gen_oracle.py polycalc`, 27,526 cases, floor 26,400) gates plan unit U4 — `{p}der` and `{p}int`
for the six bases, with every parameter (`m`, `k`, `lbnd`, `scl`, `axis`) — **bit-exact, 0 excused**. Keys are
module-qualified like `polyeval`'s (`chebyshev.chebint`). Arguments are NAMED (`c`, `m`, `k`, `lbnd`, `scl`, `axis`);
`c`/`k`/`lbnd`/`scl` are `"a"` (the next operand) or a Python-typed spec, so a weak Python `scl`/`lbnd`/constant and
a strong 0-d array reach the facade as the same KIND NumPy saw — the kind decides the NEP 50 dtype of `c *= scl`
and of the integral's correction `tmp[0] += k[i] - {p}val(lbnd, tmp)`. The sections of `gen_polycalc`:
- **(A)** every dtype × length 0–13 × order 0 … n+1 at the defaults (integer/bool series become float64);
- **(B)** scl kinds on 1-D and N-D series — weak ints/bools/floats/complexes (signed zero, NaN, inf) and strong 0-d
  arrays of every width (a 0-d float64 scl multiplies a float32 series in float64 and casts back);
- **(C)** ARRAY scl on derivatives (it broadcasts IN PLACE against the series, per order; stretching or
  non-broadcasting shapes raise NumPy's texts; the integral rejects any array scl);
- **(D)** integration constants: weak scalars, Python lists (short/long/too many), typed arrays of every width, 0-d
  arrays, lists of arrays, and N-D rows (broadcast, 2-D, complex into a real series — the in-place add's error);
- **(E)** lbnd kinds × 1-D / N-D × float dtypes: a complex value into a 1-D real series keeps its real part
  (setitem's ComplexWarning), into an N-D one raises the add's UFuncTypeError — naming NumPy's **complex64** for a
  float16/float32 series, whose 1-D `{p}val` also runs in complex64 scalarmath (reproduced exactly);
- **(F)** N-D series at every axis × memory layouts (C, F, transposed 3-D, column-strided, reversed; int32 on C/F for
  the converting direct load) — values AND a `"facet": "flags"` case recording `[C_CONTIGUOUS, F_CONTIGUOUS,
  OWNDATA]`, because NumPy's result objects differ: moveaxis views of fresh C-order buffers, the m == 0 K-order
  copy, `c[:1]*0` (hermeder alone returns it unmoved), and the n == 1 zero branch's view of the moved copy;
  broadcast and 0-d series;
- **(G)** special values (NaN, ±inf, ±0, subnormals) on 1-D (scalarmath) and N-D (ufunc) series;
- **(H)** full-mantissa complex values: a 1-D series runs NumPy scalarmath (the NAIVE complex product), an N-D one
  ufuncs (`simd_cmul`) — the integral's lbnd correction differs in the last bits between the two;
- **(I)** long 1-D series (64/257/1,000 coefficients) and widths around every lane count (1/3/7/8/9/45 columns), plus
  wider-than-one-block series (4,200 columns, float32 and complex128) for the kernel's block loop;
- **(J)** argument errors in NumPy's check order (order sign before the axis, constants before lbnd before scl before
  the axis, a complex scl caught at the first `c *= scl` even for an empty series);
- **(K)** Python-list series (ints, mixed, complex, bools, nested, empty, scalars);
- **(L)** the integral's n == 1 branch (`np.all(c[0] == 0)` keeps the series one coefficient long until a nonzero
  constant lands): zero, -0.0, NaN, complex and float16 zeros, N-D rows, with the result flags;
- float16's weak-int constants past its exact integers (2,100 coefficients: `2*(j+1)` rounds to nearest-even) and
  past its range (hermder's `2*j` and polyint's `j + 1` become inf) — 33,000 and 66,000 coefficients.

Three sections were added by the 2026-09-29 wholeness pass (5,880 cases), after a replay of every C# boundary kind
against NumPy found the sequence conversions wrong (reverting the fix turns 1,278 of them red):
- **(M) argument kinds** (3,480) — the C# replay turns a Python tuple into a `ValueTuple`, a list into `object[]`, a
  big Python int into `ulong`/`BigInteger` and a NumPy scalar item into a 0-d array, so these cells drive
  `NDPolySequence`'s port of `np.array`'s coercion: c as tuples, nested lists/tuples to depth 3, rows of arrays,
  float16 0-d items, `[2**64-1, 1]` (uint64) and `[2**63, -1]` (float64), empty sequences (`[]`, `()`, `[[], []]`, deep
  empties, an empty array leaf), and ragged input along every detection path (NumPy's texts, with the shape the walk
  still agreed on); scl as tuples/lists/nested columns (the `scl must be a scalar.` ndim check), ragged, and Python
  ints 2^20 / 2^200 / −2^70 / 2^64−1 / 2^1030 (inf in float32/float16, OverflowError past float64); k as tuples,
  lists of tuples, nested rows, big ints, strs (their characters) and ragged items — converted only when their order
  runs, AFTER that order's `{p}val`, so an overflowing lbnd wins; lbnd as tuples/lists/ragged/big ints, and a str
  lbnd, which fails only where NumPy evaluates at it (not for m == 0, not in the one-coefficient zero branch, not before
  an array scl's or a bad axis' own error);
- **(N) zero-size, 5-D and extreme-int series** (1,728) — shapes (3,0) (0,3) (1,0) (3,0,2) (2,0,0) at every axis with
  m 1/2 (values + flags, integrals with k), a (2,3,1,2,3) float64/complex128 series at every axis, and the extremes
  of every integer width (min, max, ±1 from them, ⅓ of them) through the converting direct load's vector lanes and
  scalar tail;
- **(O) the widened `c *= scl`** (672) — a strong scale that promotes the series runs NumPy's wider multiply loop and
  casts back: float16 products just past a float16 tie (`_pc_f16_hazards`; NumPy rounds float64 → float16 ONCE), NaN
  and inf coefficients and scales (the series' NaN wins), a float16 series' float32 loop (int16 2049, float32 0.1,
  int32 70000 — the exact float32 scale, not a float16 one) and a float32 series' float64 loop (1e300 → inf,
  2^40+1); 1-D series run the scalar tails, (3, n) series the vector lanes. Planted-bug check: 4 widened-lane mutants
  (an f64 → f32 → f16 chain in the vector and the scalar helper, a float16-rounded scale, a float32 multiply), 44–73
  red cases each.

Char rides the uint16 proxy (section A). Cells whose NumPy result is complex64 are skipped (#569). `OpRegistry.
PolySeries.cs` replays it; a flags case disposes the result it replaces (the leak gate reads an undisposed result
as an escaped buffer). Design and measurements: `docs/plans/numpy-polynomial.md` (U4).

### numpy.polynomial series algebra (`polyalgebra` + `polyalgebra_parity` tiers)

`polyalgebra.jsonl` (`gen_oracle.py polyalgebra`, 28,144 cases, floor 28,100; 26,163 at delivery) gates plan unit U2: `{p}mulx`, `{p}mul`,
`{p}div` (a `tuple` kind: quo, rem), `{p}pow`, `{p}fromroots` for the six bases, and `X2poly`/`poly2X` for the five
non-power ones. It is **bit-exact, 0 excused**. Keys are module-qualified (`chebyshev.chebmul`). Arguments are NAMED
(`c`, `c1`, `c2`, `roots`, `pol`, `pow`, `maxpower`): `"a"` (the next operand) or a Python-typed spec (a weak
int/float/complex, a list, a tuple, a big int, a str, `{"kind": "none"}` for None, `{"kind": "npscalar"}` for
np.float16). A `maxpower` is a raw JSON int or `null` (NumPy's None) where the facades' `int?` parameter takes it, a
spec for any other kind — which, like a power the int / double overloads cannot take, binds the object-typed
overloads (`PolyPow` binds as C# source would).

**The split.** NumPy's power- and Chebyshev-basis products are `np.convolve`, whose `?dot` switches to OpenBLAS's
VECTOR kernel once a dot reaches ddot's 16 / sdot's 32 / zdotu's 8 terms. No managed engine reproduces that summation
order. (A complex product with infinities / NaNs used to count too — zdotu's C99 result construction was not
modelled; it is now, with CDOUBLE_dot's plain loop for a one-element kernel, so 43 such cases moved to the portable
file.) The generator therefore patches `np.convolve` while it runs a case
(`_PAConvRecorder`: operand lengths, dtype, finiteness) and routes every case with a BLAS-bound product to
`polyalgebra_parity.jsonl` (139 cases; 186 at delivery). That file is **host-pinned** exactly like `matmul_parity`
(`polyalgebra_parity.host.jsonl`, the same `MatmulParityPin`): it is replayed with the OpenBLAS backend at threads=1,
byte-exact on NumPy's own pinned scipy-openblas, and Inconclusive off the pinned host. Everything else, the scalar
regimes included, stays in the portable file. `BlasBackendDelta` replays the portable tier's six convolving ops
backend-on (`polymul`/`polypow`/`polyfromroots`, `chebmul`/`chebpow`/`chebfromroots`): 6,522 affected, 6,480
identical, 42 flips at delivery, every one byte-checked against NumPy (14 of the 42 differ only in a NaN's payload,
which no gate compares); 7,064 / 7,001 / 63 with the wholeness pass's cases — the new non-finite complex products and
the `groupa` complex convolve / correlate block, whose one-element-operand cases pin that the backend is bypassed
where NumPy's CDOUBLE_dot takes its plain loop.

The sections of `gen_polyalgebra`:
- **(A)** every dtype × length at the defaults: mulx, the conversions, pow 0–4, fromroots of 0–12 roots, and mul / div
  over every dtype PAIR (the full matrix for the power, Chebyshev and Legendre modules; a subset for the rest);
- **(B)** full-mantissa values (float64/float32/float16/complex128): the product forms and dot schedules differ only
  on such operands, and lengths straddle every scalar/vector boundary;
- **(C)** trim / special patterns (trailing zeros with a −0.0, all-zero, zeros after the first term, NaN/±inf/±0,
  a trailing NaN) on each operand and both;
- **(D)** layouts: contiguous, strided, reversed and offset views on either operand, 0-d arrays (a length-1 series),
  stride-0 broadcasts;
- **(E)** Python-typed arguments: scalars, lists, tuples, `[1, 2**63]` (uint64 / float64 coercion), `[2**64-1]`,
  nested and bool lists, on each side of mul / div; fromroots of lists, tuples, scalars, big ints, a str, 2-D and
  empty sequences;
- **(F)** errors in NumPy's check order: empty / 2-D / bool / str / ragged series on either side, the zero divisor
  (`ZeroDivisionError`, empty message, checked before the lengths), fromroots' `len()` TypeError before as_series;
- **(G)** pow and maxpower kinds: float powers (2.0 binds, 2.5 fails), −1, −0.0, NaN, ±inf, 1e300, 2^40 (a huge valid
  power passes `maxpower=16` explicitly: polypow's default None would run forever), and maxpower None / 0 / 3 / −1 / 16;
- **(H)** long series across the managed / BLAS boundary (power and Chebyshev at 15/16, 31/32, 7/8 terms and far past
  them; the recurrence bases at 20–40 terms), long mulx (257 / 1,000 terms), long conversions and divisions;
- **(I)** result flags (`"facet": "flags"`: C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA): a trimseq slice is a VIEW, polydiv's
  remainder a view of its working copy;
- **(J)** fromroots' root kinds: NaN, ±inf, repeats, all −0.0, unsorted, huge, subnormal, conjugate pairs,
  lexicographic complex order. Only ONE zero per list: NumPy's SIMD sort orders +0.0/−0.0 by CPU (a unit-pinned
  `[Misaligned]`);
- **(K)** underflow and overflow in pow / mul / div / mulx / the conversions;
- **(L)** chebmulx's fused kernel through its vector stage: the special pattern at 40 / 101 terms per float dtype;
  every float16 bit pattern below 2^-12 plus the infinities, NaNs and extremes, once as c[j−1] and once as c[j+1] of
  an output; float32 / float64 subnormals. Planted-bug check: dropping the float16 grid rounding turns this
  section's 2 f16 cases red (nothing else in the corpus reaches it), a plain complex multiply 5 cases;
- **(M)** the wholeness pass (1,934 cases): every pow kind the object overload takes (bool, np.float16, str, None,
  list, tuple, complex, 0-d arrays of every kind, 1-D / 2-D / empty arrays, ints past int64) and every maxpower kind
  (float, NaN, ±inf, big int, bool, np.float16, complex, str, list, tuple, 0-d / one-element / two-element / empty
  arrays, uint64 / int8 / float16 / float32 0-d) with NumPy's texts; the comparison-dtype edges (2049 vs float16 2048
  computes, 70000 vs float16 65504 raises, 2**1100 against float / complex / bool / uint64 / int8 limits); the DEFERRED
  object / str refusal — None / str / oversized-int series checked by as_series, np.array's raggedness, div's zero
  divisor and pow's checks in NumPy's order, filtered to outcomes NumPy reaches before any object arithmetic
  (`_pa_object_land`); and complex series with infinities / NaNs through every function, one-term factors (the plain
  loop) included.

Char rides the uint16 proxy. Cells whose NumPy result is complex64 or an object array are skipped (#569).
`OpRegistry.PolyAlgebra.cs` replays both files. `{p}pow` binds NumPy's argument kinds: an in-range int → the int
overload, a float / huge int → the double one, and maxpower only when the case has it. Design and measurements:
`docs/plans/numpy-polynomial.md` (U2).

### numpy.polynomial Vandermonde family (`polyvander` tier)

`polyvander.jsonl` (`gen_oracle.py polyvander`, 31,180 cases, floor 31,000; 16,612 at delivery) gates plan unit U5 —
`{p}vander`, `{p}vander2d` and `{p}vander3d` for the six bases — **bit-exact, 0 excused**. Keys are
module-qualified like `polyeval`'s (`legendre.legvander2d`). Arguments are NAMED (`x`, `y`, `z`, `deg`) with the
polyseries encoding (`_ps_enc`), decoded in that fixed order; `deg` keeps its Python KIND (an int, a bool, a 0-d array,
a float, a str, a list / tuple / ndarray container), because the kind decides `operator.index`'s answer and NumPy's
error. 846 cases record NumPy's error type and text (390 ValueError, 456 TypeError); 840 are `"facet": "flags"` cases
recording `[C_CONTIGUOUS, F_CONTIGUOUS, OWNDATA]` of the result — NumPy returns `np.moveaxis` VIEWS (OWNDATA false:
F-contiguous for 1-D points, C and F for a scalar / degree 0 / empty points, neither for N-D points), and which one
comes back is part of the contract. The sections of `gen_polyvander`:
- **(A)** every dtype × length 0–33 × degree 0–14 on a contiguous 1-D x (bool / integers / char become float64; the
  `+ 0.0` turns -0.0 into +0.0);
- **(B)** full-mantissa random values around every lane width (2 complex, 4 float64, 8 float32 / float16);
- **(C)** special values (`_pv_special`: signalling and negative-payload NaNs — quieted, payload kept — ±inf, ±0,
  subnormals, max bit patterns; complex built through view bit copies);
- **(D)** memory layouts of x (strided, reversed, offset, F and transposed N-D views, broadcast, 0-d) with values
  AND flags, N-D point shapes, and the zero-size shapes (`(0,)`, `(0, 3)`, `(2, 0, 2)`);
- **(E)** float16's Python-int constants past its exact integers (degree 2,100: `2*i - 1` rounds to nearest even) and
  past its range (degree 33,000 / 65,600: the constant becomes inf), and degree-300 recurrences at float32 / float64 /
  complex128;
- **(F)** 2-D / 3-D at every dtype × point shape × degree pair / triple, 14 MIXED dtype pairs (the stack's strong
  promotion: float32+int32 is float64, float16+int8 float16), special and full-mantissa values in every coordinate
  (the outer product is where two independently sourced NaNs meet: NumPy's multiply keeps the SECOND operand's NaN),
  point layouts per coordinate, the result flags, and Python-typed points (floats, ints, bools, complexes, lists,
  tuples, nested lists, np.float16 items, uint64-range ints, empty lists, lists of 0-d arrays) stacked by
  `np.asarray`'s coercion;
- **(G)** argument kinds and errors in NumPy's order: 54 degree kinds for the 1-D form (bool is an int, a 0-d integer
  array converts, a 0-d bool does not; floats, NaN, complex, str, None, lists and huge ints raise `deg must be an
  integer, received {format(deg,'')}` — NumPy formats a 0-d array through its scalar, a list through its items' repr);
  `deg must be non-negative`; `Maximum allowed dimension exceeded` at ≥ 2^63 - 1 rows and AllocationGuard's `array is
  too big` below it — the degree checked BEFORE x is converted (only where NumPy's check comes first: a valid huge
  degree over an object x lets NumPy compute `x + 0.0` with Python objects, which NumSharp refuses); ragged x;
  20 x kinds; 37 / 14 degree containers for the 2-D / 3-D forms (`object of type 'int' has no len()`, `len() of unsized object`,
  `Expected 2 dimensions of degrees, got 3`, str containers read per character, items checked dimension by dimension
  with the per-dimension allocation between them); and points' errors in NumPy's order (a ragged stack's inhomogeneous
  text, the empty stack's reshape error `cannot reshape array of size 0 into shape (0,newaxis)`);
- **(I)** inputs longer than one kernel block (cheb and lag carry it: 1,700–9,000 points, strided included);
- **(H)** Char: the uint16 section relabelled (600 cases);
- **(J)** the OBJECT stack of scalars (the 2026-09-30 wholeness pass, 14,568 cases, in its own pass after A–I so every
  earlier id is unchanged): a Python int past uint64 among scalar points makes `np.asarray((x, y[, z]))` an object
  array, and NumPy's per-element `+ 0.0` then hands every dimension its OWN number — `float(int)`, correctly rounded
  (`2**64 + 2**11 + 1` rounds up, `2**1024 - 2**970 - 1` down to the largest double) or CPython's OverflowError past
  the float range (the tie `2**1024 - 2**970` rounds up to it), a Python float / bool / complex, an np.float16 or a
  0-d array keeping NEP 50's dtype — so the dimensions' matrices can have different dtypes, promoted by the outer
  product's multiply. Twelve huge ints × 23 partner kinds (NaN, -0.0, complex NaN, np.float16 NaN / -0.0, 0-d float32 /
  int8 / bool / float16 / complex / uint64-max / NaN arrays, another huge int), both argument orders, three degree pairs,
  three 3-D arrangements, and the degree errors after the stack. A None / str partner is skipped (`_pv_refused`: NumPy
  raises TypeError at that element; NumSharp refuses the object stack there, as everywhere). Planted-bug check: a
  truncating int -> float conversion turns 3,132 cases red.

MemoryError texts (machine-dependent) and the str / object refusals are unit-test-pinned instead
(`Polynomial/PolynomialVanderTests.cs`; the C#-only argument kinds — typed arrays, `List<T>`, LINQ, `System.Tuple`,
`object[,]`, `Memory<T>`, `short` / `char` degrees binding the int overload, the typed-collection coercion — in
`Polynomial/PolynomialVanderArgumentKindsTests.cs`), and so are results past 32 MiB, whose product rows stream through
non-temporal stores (forced on and off at every alignment there; the corpus' matrices are all small). NaN payloads
through a complex product are tokenized (the house `simd_cmul` picks a different NaN operand than NumPy's; the value
is NaN either way). Planted-bug check at delivery: two kernel mutants turned 2,403 and 117 cases red. `OpRegistry.PolyVander.cs` replays it: a 1-D degree that is a C# long in int range
binds the int overload, every other kind the object one. Design and measurements: `docs/plans/numpy-polynomial.md` (U5).

### numpy.polynomial companion matrices and roots (`polyroots` + `polyroots_parity` tiers)

`polyroots.jsonl` (`gen_oracle.py polyroots`, 8,195 cases, floor 8,150; 6,681 at delivery) gates plan unit U7 — `{p}companion` and
`{p}roots` for the six bases — **bit-exact, 0 excused**. Keys are module-qualified (`laguerre.lagcompanion`); the one
argument `c` uses the polyseries encoding (`_ps_enc`). The companion is NumPy's statements over the as_series copy (the
zero matrix, diagonals assigned through `mat.reshape(-1)[k::n+1]`, one in-place update of the last column) — `+ - * /`,
`sqrt` and `cumprod` only — so every companion case is portable; so is every `{p}roots` call that never reaches LAPACK
(a constant series' empty array in its dtype, a linear series' scalarmath root, float16's linalg TypeError and a
non-finite companion's LinAlgError). The roots that run geev go to `polyroots_parity.jsonl` (1,354 cases, floor 1,350; 1,242 at delivery),
**host-pinned** exactly like `linalg_parity` (`polyroots_parity.host.jsonl`, the same `MatmulParityPin`, threads=1).
The generator decides the split by replacing `np.linalg.eigvals` while it runs a case (`_PREigRecorder` — the polynomial
modules look it up at call time): geev ran exactly when eigvals received a finite float32 / float64 / complex128 matrix.
The sections of `gen_polyroots`:
- **(A)** every dtype × length 1–13 (the power, Chebyshev and Legendre modules over all 13 dtypes, a subset for the
  rest; bool is as_series' no-common-type ValueError, integers / char compute in float64);
- **(B)** full-mantissa values, and the same series with a 1000× smaller leading coefficient (float16's quotient
  overflows to inf, so its roots are the LinAlgError);
- **(C)** trim / special patterns (trailing zeros with a −0.0, all-zero, zeros after the first term, NaN / ±inf / ±0, a
  trailing NaN — never trimmed);
- **(D)** layouts: strided, reversed, offset views, a 0-d array (a one-term series) and a stride-0 broadcast;
- **(E)** Python-typed series (scalars, lists, tuples, uint64-range ints, np.float16 items, empty tuples) and the OBJECT
  series NumPy refuses before it computes with Python objects (`_pa_object_land` filters the rest): the length check
  runs on the TRIMMED object array, so `[None, 0]` and `[2**70, 0]` are companion's ValueError; a str series is
  as_series' no-common-type ValueError;
- **(F)** errors in NumPy's order (empty, 2-D, bool, str, ragged, nested-empty, a float16 NaN / inf reaching eigvals'
  finiteness check before its dtype check);
- **(G)** roots of known polynomials built by `{p}fromroots` (distinct, repeated, clustered, wide, Wilkinson's 10,
  conjugate pairs, lexicographic complex ties) and float32 series with complex roots — NumPy's complex64, which the
  replay compares by VALUE up-cast (`np.linalg.eigvals` rounds a float32 operand's complex result to float32
  components, exactly);
- **(H)** result flags (`"facet": "flags"`): the companion is a fresh C-contiguous owning matrix; a float64 series' real
  roots are a strided VIEW of eigvals' complex result (OWNDATA false), a float32 series' a fresh cast;
- **(I)** long series (companion of 40 / 65 terms, roots of 25 / 51 terms);
- **(J)** the linear roots' scalarmath over every special-value pair (float64 / float32 / float16, and complex);
- **(K)** the wholeness pass (its own pass, so A–J ids hold): two-term OBJECT series NumPy still computes — a Python
  int past uint64 makes as_series' copy an object array whose items keep their kinds (Python numbers, NumPy scalars,
  0-d ARRAYS), and the linear statement runs on the items (CPython / scalarmath / ufunc) into a NUMERIC array;
  trailing 0-d zeros are trimmed (`item != 0`); CPython's int/int OverflowError is kept, object results and
  object-arithmetic TypeErrors are filtered (`_pa_object_land`) — plus edge numeric series (±inf / NaN leading or
  interior terms, tiny / huge leading terms, subnormals, signed-zero trims, int64 / uint64 / int8 extremes, complex
  non-finite components): 1,142 portable + 112 host-pinned cases;
- **(L)** object series of THREE or more terms (its own pass after K): NumPy computes the object companion matrix, so
  `{p}companion`'s result is an object array (filtered) and `{p}roots` raises eigvals' `ufunc 'isfinite' not supported`
  TypeError before LAPACK — unless the last-column arithmetic raises first, a Python int too large for a float:
  `integer division result too large for a float` when divided by a Python int (the divisor `c[-1]` is CAST to the
  object dtype first, so a 0-d integer divisor is one there) or `int too large to convert to float` when it meets
  anything else (herm / herme multiply by their float64 helper first). 372 cases, `obj3_object_land`-filtered.
Char is the uint16 weave in both files. A result whose sorted order is not fixed by its values — two elements equal by
value but different in bits (+0.0 / −0.0) — is skipped: NumPy's SIMD sort orders such ties by CPU (none occurred in
the 7,923 generated cases). Unit tests (`Polynomial/PolynomialRootsTests.cs`) pin what the corpus cannot: the object
roots NumPy answers with an object array (`[Misaligned]`), the object series' C# spellings (BigInteger, Half, 0-d
arrays), decimal, the backend-missing path, real roots surviving a later call (the reinterpreting-alias
use-after-free, `ArcLifecycleTests.ReinterpretingAlias_*`), both write-once allocation paths and the ramp kernel. Design and measurements: `docs/plans/numpy-polynomial.md` (U7).

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

### The random-API oracle (`random_api` tiers)

Plan and state: `docs/plans/random-oracle-coverage.md`. The `random_parity` tiers above pin streams through a
hand-written dispatch (one C# call per NumPy call pattern); this family covers the WHOLE random world by exact
overload: every public member of `NumPyRandom`, `NumPyRandom.State`, `NativeRandomState`, `Generator`, `BitGenerator`,
the five engines (`MT19937`, `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`) and their `State` classes, `SeedSequence`,
`SeedlessSeedSequence` and the two seed-sequence interfaces — 493 members in `test/oracle/random_surface.json` (439 at
the plan's P7; the wholeness pass added the 50 `np.random.<Class>(...)` factories and the `NDArray`/`BigInteger` bounds
of `integers` and `randint`), which the G1 gate keeps equal to reflection.

- **Signature-addressed cases.** Each case names the canonical C# signature it exercises (`params.sig`); the harness
  (`RandomApi/RandomApiHarness.cs`) resolves it by reflection and invokes exactly that member — an omitted optional
  argument gets the declared default, as a compiled call would, while NumPy is called without it, so a default that
  differs from NumPy's is a divergence. `params.member` (`<Owner>.<member>`) is the coverage join's key.
- **Call shapes are not the replay's job.** Because the harness reflects, it proves what each overload DOES, never that
  NumPy's call spelled verbatim in C# compiles and binds that overload. The wholeness pass (plan §9) compiled 1,793
  NumPy spellings against NumPy 2.4.2 and fixed what did not bind: the size-only call of nine legacy samplers, the
  `np.random.<Class>(...)` spellings, a bare `null` seed, `size: default` against the size shims (it bound
  `multivariate_normal`'s `long` shim: size 0, silently), plus a typed null seed ARRAY that seeded the empty list — hidden
  here because entropy cases mask the state on both sides. `RandomSampling/RandomApiWholeness.Test.cs` pins each: a
  compile-time proof of the NumPy spelling, then NumPy's values. Add a pin there when a new overload could change what
  an existing NumPy spelling binds.
- **Receivers.** Legacy members run on the legacy-seeded MT19937 and on `RandomState(ENGINE(seed))` for every engine;
  `Generator` members on `Generator(ENGINE(seed))` for every engine; bit-generator members on every engine; state
  objects, seed sequences and constructors on their own receivers. Primings (`u32` buffered half, `gauss` cached
  Gaussian, `raw3` raw words) and repeated calls reach the non-trivial stream states.
- **Observations** (`RandomApi/RandomApiObservation.cs`, mirror of `obs` in the generator): arrays by dtype, shape and
  bytes; Python scalars by value; engines, Generators and RandomStates by type and canonical state text (plus the seed
  sequence); seed sequences by pool hash and repr. Every answered case on a stateful receiver ALSO records the
  receiver's full state after the call, so over- and under-drawing is caught, not only wrong values. OS-entropy words
  are masked identically on both sides.
- **Tiers.** `random_api.jsonl` (portable, 9,820), `random_api_host.jsonl` (samplers consuming libm, win-amd64
  authored, 15,795), `random_api_mvn.jsonl` (`multivariate_normal`, byte-exact only through NumPy's own
  scipy-openblas at one thread, 730) and `random_api_lp64.jsonl` (answers that depend on C long being 64-bit, from
  Linux NumPy through the generator's WSL sub-run, 234; hard-gated on x64 Windows and Linux). 10 fixed seeds; committed
  floors in `FuzzCorpusTests.RandomApi.RandomApiMinCases`. A case moves to the LP64 tier when Linux's answer leaves
  int32, when the platforms disagree on raising, or when a PORTABLE case's value differs between them (no libm is
  involved there, so the difference is the C long: `randint`'s float bounds in a non-C layout take the 64-bit
  element-wise conversion on Linux only); a Windows row is dropped when Linux NumPy raised at an earlier receiver of the
  same signature and variant.
- **Gates** (`RandomApi/RandomApiCoverageTests.cs`, corpus-only): G2 every overload has a case or a reasoned
  exemption; G3 every parameter omitted and non-default, every required one two values, every nullable one null and
  non-null, `params` arrays 0 and 2+ elements, every enumerated value NumPy accepts plus one it rejects, and NumPy's
  parameter NAMES in NumPy's order (against `test/oracle/random_numpy_signatures.json`); G4 every engine; G5 all ten
  seeds per receiver engine; G6 the post-call state on every stateful case. All exemption tables are self-retiring.
- **Soak.** `fuzz-soak.yml`'s `random-api-lp64` (Linux NumPy's LP64 rows) + `random-api-soak` (windows-latest:
  generation with `--lp64-rows`, replay on net10.0 and net8.0 through `RandomApiSoak`) regenerate every family under
  10 fresh seeds every night.
- **Intended divergences** (`RandomApi/RandomApiDivergences.cs`, each keyed on a condition and printed per replay):
  `Generator.pareto`/`power` within the in-band `expm1` ULP bound; `get_state()`'s tuple overload refusing a
  non-MT19937 engine (NumPy warns and returns the dict — the dict overload is that); MT19937 positions outside
  `[0, 624]` and a negative Philox `buffer_pos` refused where NumPy stores them and then reads outside the array; a
  string / nested / ndarray `spawn_key` stored as its flattened ints (identical pool; only the repr line differs).

Regenerate: `python test/oracle/gen_random_oracle.py` (NumPy 2.4.2; WSL with Linux NumPy at `~/np242/bin/python` for
the LP64 tier), then rebuild. After a surface change: `NUMSHARP_WRITE_RANDOM_SURFACE=1` on the G1 test first. When the
member set changes, `python test/oracle/random_api_oracle_map.py` rebuilds the `random_api` key of
`coverage/oracle_map.json` (the strict coverage join) from the corpus and the coverage catalog.

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
