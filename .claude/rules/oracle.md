---
paths:
  - "test/oracle/**"
  - "test/NumSharp.Tests.Oracle/**"
  - "test/NumSharp.Tests/IO/{NpyOracle*,corpus/**}"
  - "coverage/**"
---

# The differential-fuzz pipeline (NumPy oracle)

Every NDIter-backed op is proven bit-identical to NumPy 2.4.2 across the input space. NumPy is the oracle: Python
generates a committed, bytes-exact corpus; the C# harness replays the operand bytes and bit-compares, with no Python at
test time or in CI. Playbooks (adding an op, regenerating, triage): the `oracle` skill. The divergence ledger:
`test/NumSharp.Tests.Oracle/Fuzz/README.md`.

```
test/oracle/                          corpus generators (NumPy 2.4.2; run by hand and by the nightly soak)
  layout_catalog.py                   memory-layout builders (single-operand, pair and where= layouts)
  gen_oracle.py                       deterministic op matrices across value/error/kind/artifact tiers; Char woven
                                      into the tiers through the uint16 proxy (char_tier, relabelled uint16 -> char)
  gen_decimal_oracle.cs               an INDEPENDENT C# oracle for Decimal (no NumPy analog): naive scalar
                                      System.Decimal math -> decimal_{unary,binary,reduce,scan,power,varstd,matmul,
                                      astype,stat,where,sort,manip}.jsonl
  gen_index_oracle.py                 getter/setter index oracle (a token index over base recipes)
  fuzz_random.py                      seeded random fuzzer (imports the other two)
  gen_npy_oracle.py                   .npy/.npz format oracle: real np.save/savez output + a manifest
                                      -> NumSharp.Tests/IO/corpus/npy_oracle.zip
  interop_write.cs                    the reverse direction: NumSharp writes, verify_npy_interop.py reads
  verify_npy_interop.py               manual gate: real NumPy reads NumSharp's output (needs Python + the SDK)
test/NumSharp.Tests.Oracle/Fuzz/      C# replay harness (no Python)
  FuzzCorpus.cs                       rebuilds EXACT NDArray views from (dtype, shape, strides, offset, bytes)
  OpRegistry*.cs                      op name -> NumSharp call (pairs 1:1 with the generators)
  BitDiff.cs / Shrinker.cs            bit-exact compare (NaN tokenized; Decimal by canonical value, 1.0m ≡ 1.00m) /
                                      shrink a failure to one element
  MisalignedRegistry.cs               the documented, excused divergences
  FuzzCorpusTests*.cs                 one [FuzzMatrix] test per corpus file (dtype + shape + bytes + error parity)
  IndexOracleTests.cs                 index get/set differential gate
  MetamorphicTests.cs                 oracle-free invariants (round-trips, involutions, identities)
  HarnessSelfTests.cs                 proves the harness detects value / NaN / -0 differences and is non-vacuous
  corpus/*.jsonl                      the committed corpus, copied to the test output by the csproj glob
```

## Gate semantics

- Generators write into `test/NumSharp.Tests.Oracle/Fuzz/corpus/` (the path resolved relative to `test/oracle/`); CI
  replays the committed corpus, never the generators.
- Run: `dotnet test --filter "TestCategory=FuzzMatrix"`. Each case is bit-exact (pass), a documented difference in
  `MisalignedRegistry` (excused, never silent), or a failure (red). A failing value case auto-shrinks.
- Result kinds (`expected.kind`): `array` (the default: dtype + shape + bytes), `scalar` (a C# scalar wrapped 0-d), `dtype`
  (compared by NumPy dtype NAME: it gates the promotion table), `text` (printing, verbatim), `tuple` (N slots, arity
  asserted). `error: {type, text}` records NumPy's exception verbatim, so error cases assert the MESSAGE (cases with only
  `expects_throw` assert that something was thrown). Comparators `FuzzCorpusTests.Kinds.cs`, callables
  `OpRegistry.Kinds.cs`, ledger `Fuzz/README.md` → Table 0.
- `FuzzMatrix` layers: `FuzzCorpusTests` (the ordinary managed kernels, the host-pinned backend tiers and the
  deduplicated backend delta), `IndexOracleTests`, `MetamorphicTests`, the harness self-tests, and
  `OracleSurfaceCoverageTests` (the public-surface guard inventories `np`, `np.linalg`, `np.fft` and `np.random` and
  rejects an unclassified new API).
- Dtype coverage: the per-mode dtype axes span `ALL_DTYPES`; Char (no NumPy dtype) through uint16-proxy rows; Decimal
  through the independent C# scalar oracle. Real product bugs are carved and pinned (`[OpenBugs]`) or fixed, never
  excused in the registry.
- Regenerate (deterministic; needs `numpy==2.4.2`): `python test/oracle/gen_oracle.py <mode>` (modes include `conversion
  creation multioutput iter dtype_text out_where errors_full smoke astype_full binary divmod_power comparison unary
  reduce where place matmul rounding bitwise unary_extra nanreduce scan stat logic modf manip sort tail params aliasing
  copyto copyto_full power_fast weak_scalar errors groupa numpy_f32 matmul_parity linalg_parity poly einsum specials
  precision products random_parity fft`; `precision` also needs `mpmath`) and the index / Decimal / NPY generators, then
  `dotnet build` to copy the corpus assets.
- Managed and OpenBLAS without duplicate noise: the ordinary tiers always run Core managed; the host-pinned
  `matmul_parity`/`linalg_parity` tiers run with the backend; `BlasBackendDelta` replays only the ordinary cases the
  backend can affect: identical outcomes are deduplicated, and the flips are byte-compared with NumPy on the pinned host.
  Outcome equality includes threw/result, dtype, shape and bytes; a backend change on a non-CBLAS dtype is a hard failure.
- Per-API parity in the coverage artifact: `coverage/generate_coverage.py` joins every committed contract of the corpus,
  plus the `.npy`/flags/layout oracles and the string oracle's `numpy.strings` calls, onto its API row
  (`coverage/oracle_evidence.py` + the reviewed `coverage/oracle_map.json`): each row reads available AND
  oracle-verified (direct contracts, an identity alias (the same NumPy object mapped to the same NumSharp member) or a
  reviewed delegation alias). The join is strict: a new op key must resolve (bare keys go to `np.*` first, dotted keys
  by namespace, anything else needs a `fuzz.keys` entry), a new corpus file must be named by a replay suite, and stale
  or redundant map entries fail; otherwise the coverage generator (docs CI) fails, naming the key. Gate
  `coverage/test_oracle_evidence.py` (including a cross-check that every C#
  `EquivalentAliases`/`NdarrayAliases`/`MaAliases` entry has its reviewed alias).
- Signature parity in the coverage artifact: availability credits a NumPy API by NAME, which says nothing about the call
  (a member of that name with no `out=` reads "available / exact"). The generator runs the gap map's tools
  (`tools/api-parameter-parity`: `dump_modules.cs`, `binding_probe.py`, `signature_diff.py`, through
  `coverage/signature_gap.py`; it needs the .NET 10 SDK), and `coverage/signature_parity.py` joins each C# spelling onto
  the catalog row of the NumPy API it spells (module functions and members, a `(constructor)` onto the class row,
  `NDArray<T>` onto ndarray's, extension methods; an alias or inherited member adopts the spelling of the row with the
  same C# member AND the same NumPy object); a credited row the gap map does not reach keeps the inventory's own
  comparison (`source: inventory`, never compile-probed). Row `signature`: `identical` / `differs` / `not-compared`, the
  gap map's symbol per mark, every spelling. Marks: M (a NumPy parameter no overload offers; a rename counts), X (a
  parameter NumPy lacks), O (more than one overload), T (a dtype as `NPTypeCode` / `Type`), B (a NumPy-style call that
  does not compile), L (`null` refused for None), Q (an integer sequence refused), N (credited through another name).
  A row turns `identical` by itself once its API is fixed. Strict: a credited call the index cannot resolve, a gap
  spelling that joins no row or an uncredited row, and a stale `GAP_JOIN_EXEMPT` entry fail the generator. Gates
  `coverage/test_signature_parity.py` (with the committed `docs/plans/numpy-signature-diff.csv`),
  `test_object_surfaces.py`, `test_dashboard.cjs`.

## Tiers

- `iter.jsonl`: materialized TRACES of `ndindex`/`ndenumerate`/`nditer`/`nested_iters`/`broadcast` (the value stream,
  multi_index, c_index/f_index and external_loop chunk lengths across order C/F/A/K × layouts); traversal order has no
  other gate. `dtype_text.jsonl`: promotion/dtype predicates, printing, scalar predicates and `nonzero` at any rank.
  `multioutput.jsonl`: full tuple arity and every slot for split/unstack/broadcast/meshgrid/modf/average/unique.
  `creation.jsonl`, `conversion.jsonl`, `errors_full.jsonl` (the raising cells every value generator skips).
- `out_where.jsonl`: ufunc `out=`/`where=` over out layouts × mask layouts × ufuncs; each case records the returned view
  AND the entire base buffer behind `out`, so masked-off slots keep their contents and a kernel writing outside a
  strided/offset/negstride window is caught.
- `power_fast.jsonl` (`gen_oracle.py power_fast`): np.power's DOUBLE_power / FLOAT_power / HALF_power and np.float_power
  over every float base layout × 0-d / one-element / (1,1) / broadcast / Python-scalar exponents, integer / bool / complex
  bases, the trivial loop's cast copies, NpyIter's buffering threshold at `np.setbufsize(256)` and the default, varying
  exponents, `out=`/`where=` layouts and `dtype=`; bases hold values on which the fast path and pow differ. Zero excuses.
  Not generated: a 0-d float32/float16 base with a Python-scalar exponent (the NEP 50 0-d divergence), complex bases with
  non-integer exponents (F5), unsigned bases with -1, an `out=` that does not broadcast (its text is NumPy's, pinned by
  `UfuncOutWhereTests`).
- `weak_scalar.jsonl` (`gen_oracle.py weak_scalar`): NEP 50 weak Python scalars: the six comparisons and the binary ufuncs
  with one operand the C# literal NumPy's Python int / float / complex / bool maps to, every dtype against the scalars
  where the precision decides, both sides, np.* and both operator overloads, `out=` / `where=`, the iterator layouts, ints
  past 2^53 into float16 / float32 loops (two roundings), weak ints into integer loops (NumPy's OverflowError), long
  arrays through the scalar-broadcast kernels' packed vector loops (section W), empty broadcasts, and section S: the
  literal STORED at every store entry point (the indexer routes, fill, full, full_like, put, putmask, place,
  put_along_axis, insert, fill_diagonal, pad, the flat iterator, asarray, copyto), Python scalars and the NumPy scalars
  Half / char. Every other tier passes arrays, which NumPy treats as strong. Zero excuses. Generator drops: a where=
  comparison of an integer array with an out-of-range Python int (NumPy 2.4.2 segfaults there), a bool array against an
  int past int64, a store error naming the platform C long (the unit tests pin the LP64 texts), np.copyto's complex64
  cast text.
- `precision.jsonl`: each case carries `expected.truth`, the correctly rounded mathematical reference (exact `Fraction` /
  200-bit mpmath, generator-side), over precision-adversarial inputs (wide-magnitude and cancellation sums, large-mean
  variance, near-1 products, expm1/log1p at small |x|). Bit-exact to NumPy passes without truth being read; truth only
  adjudicates a divergence. Not less truthful than NumPy → excused "prefer-precise" parity debt (more accurate than NumPy
  is still a divergence, closed by porting NumPy's algorithm); less truthful beyond the 4×/+8 ULP slack → a precision
  LOSS, red unless a bounded known-bug branch covers it (`MisalignedRegistry` P1–P3; the summation blanket applies only
  without truth). P3 covers the negative-stride sum/mean path and float32 deep product contractions (inner/tensordot at
  K = 2049), bounded at 256 ULP from truth.
- `reduce_start.jsonl`: every reduction, scan and product over all-`-0.0` operands (size 1, length-1 axes, C / transposed
  / reversed / stepped, f64/f32/f16/c128), so the SIGN of a zero result shows whether NumSharp starts the way NumPy does
  (`add.reduce` from the `+0.0` identity, accumulate copying element 0, complex `prod` times `1+0j`,
  `cblas_matrixproduct`'s size-1 scalar branch, the default einsum's zero-filled output). Zero excuses (the registry
  returns null for its `negzero` valueclass). The cells NumSharp gets wrong are the generator's carve (`_rs_carved`, one
  rule per root cause, each naming its `[OpenBugs]` pin in `OpenBugs.Reductions.cs`); drop a rule and regenerate when its
  path is fixed.
- `varstd.jsonl`, `nanvarstd.jsonl`, `mean_loops.jsonl`, `nanmean.jsonl`, `sum_identity.jsonl`, `minmax_kwargs.jsonl`,
  `half_minmax`: the reduction ports (`reductions.md`), each with zero excuses and its carve of the open np.sum residuals.
  `VarStdChunked` replays the var/std tier through the chunked deviation sums.
- `ma_meanvar.jsonl` (`gen_ma_oracle.py meanvar`): numpy.ma's mean/var/std/anom statements over every layout × dtypes ×
  nomask / all-False / some masks, the mask classes, keepdims and ddof, `dtype=`, the `mean=` forms, special values and
  NumPy's errors; each case also records `raw` (the data at every slot) and `form` (singleton / scalar / nomask / mask),
  which `FuzzCorpusTests.Ma` compares. Zero excuses (valueclass `mameanvar`) and no carve: the cells whose mean sums the
  operand view on a layout np.sum used to order differently (reversed axes, broadcast operands) are in the gate since
  that sum follows NumPy's schedule.
- `copyto_full.jsonl`: np.copyto over dtypes × destination and source layouts × mask forms, long runs at every element
  width, cast pairs, overlapping views of one buffer (masked and unmasked), C# literal sources as Python scalars, 0-d and
  empty destinations, and the error jobs that gate NumPy's check ORDER; every case records the destination's whole base
  buffer. Zero excuses.
- Ownership, the deterministic over-disposal (`Fuzz/Ownership/` +
  `UndisposedIntermediateTests.{Ownership,ForeignMemory}.cs`; README "Ownership gate"): no call may dispose, release,
  under-count, swap or pool memory it does not own (its caller's operands, masks, index and value arrays, `out=`
  targets, the leak fixtures, the process-wide shared arrays: `np.ma.nomask`, the masked constant, the numpy.polynomial
  module constants and class attributes). The leak sweep's pool balance CANNOT see it: `NDArray.Dispose` is idempotent
  and every sweep family runs an unmeasured warm call first, so an op that disposes its operand returns the buffer there
  and the measured window balances (`PoolBalance_CannotSeeADisposedOperand`). `OwnershipLedger` snapshots each array
  (its block's ARC kind, counted references and freed flag through `ArcProbe`'s compiled accessors on the private
  `Disposer`) and checks it after every region of every case in all four leak runners, plus a `PoolScan` of the 64 most
  recent buffers of the caller memory's bucket sizes. `Corpus_ForeignOperands_StayTheCallers` replays one case per (op,
  layout, dtype) with every operand rebuilt over six `ForeignKind`s (pooled, a pinned caller `T[]`, `np.ndarray(buffer:
  byte[])`, `np.frombuffer`, a non-owning raw pointer, caller memory with a release action). `ReferenceRetained` (a
  reference to the caller's memory outlived the call) is printed, not judged.
- Lifetime, lost GC anchors (`Fuzz/Lifetime/`, `test/oracle/gen_lifetime_oracle.py`; README "Lifetime gate" and "Scaled
  lifetime tier"): a method that reads an operand's raw pointer and runs its kernel with no later use of the operand lets
  optimized code drop the operand at the pointer read; when the caller passed a temporary, a collection during the kernel
  finalizes its block and the buffer is pooled or freed under the kernel (silently wrong values, spurious IndexErrors,
  fatal access violations). No other tier can see it: they hold their operands.
  - Cases carry operand RECIPES (NumPy's answer as dtype / shape / SHA-256) in `Fuzz/Lifetime/corpus/`, outside the
    generic sweeps. The replay runs the Oracle assembly as a CHILD process with `DOTNET_TieredCompilation=0` (tier-0 keeps
    every local alive and hides the bug; a crash ends only the child, which resumes after the crashed case; entry point
    `LifetimeChildProgram.Main`, so the csproj sets `GenerateProgramFile=false`), builds each temporary operand inside the
    call expression, and a stress thread collects, drains finalizers and poisons the buffers they return. A control must
    be caught or the run is inconclusive. Release only: a Debug build keeps every local live, so the tests are
    inconclusive there (`dotnet test -c Release`; CI does).
  - Gate `LifetimeOracleTests.Lifetime` (`[FuzzMatrix]`): its cases PIN the fixed gaps LT01–LT30
    (`Lifetime_CorpusContract`; a red pinned case reads `FIXED gaps regressed: LTnn`); open gaps (none) are carved into
    `lifetime_gaps.jsonl` (`GAPS` in the generator, `FIXED` the rest). The scaled tier
    (`Fuzz/Lifetime/LifetimeScaled.cs`, gate `LifetimeScaled_EveryOpKey`) runs every op key of the ordinary and
    masked-array corpus with its operands enlarged into ~2 MiB temporaries against the same call over held operands (no
    NumPy answer needed); `KnownScaledGaps` (empty) maps an op key to the open lifetime gap whose site it reaches;
    `NUMSHARP_LIFETIME_SCALED_REPS` sets the depth (the nightly `lifetime-scaled` job runs 32).
  - The fix is the weaver's OPERAND ROOTS, never a hand-placed `GC.KeepAlive` (a scope roots only arrays built while it is
    open, never the caller's operands): every method that opens an NDScope copies its NDArray-reaching parameters (and
    `this` of a class) into `object` shadow locals at entry and passes them to `NDScope.Anchor` before every `ret`
    (`ScopeWeaver.InjectParameterAnchors`); an unscoped entry point that hands an operand's address to a kernel carries
    `[NDAnchored]` (roots only, no scope, not inherited; NDW019 refuses a body-less / async / iterator / tail-call /
    also-scoped / nothing-to-root method; NDW013 and the MSBuild fallback scan cover it). A new site: scope or anchor its
    entry point. Deterministic guards: `test/NumSharp.Tests/Lifetime/OperandRootsTests.cs` reads the woven IL, and
    `WeaverOperandRootTests` (Build.Analyzer) pins the transform.
  - Traps: JIT liveness is per call site over every later path (`astype` keeps its receiver because a later branch reads
    `this.Shape`; the same cast through `TensorEngine.Cast` loses it); a pass is not proof (short windows go uncaught for
    many runs); the child watchdog is per case (`CaseTimeout`); detection is a race, never a false positive, so a red gate
    is a real lost anchor (carve it under a new LT id).
- The array-object oracle (`test/oracle/gen_viewflags_oracle.py`; README "The array-object oracle") gates the OBJECT an
  API returns: values plus a `mem` facet (C / F / OWNDATA / WRITEABLE / ALIGNED flags, byte strides, view-of / IS each
  operand, tuple slot sharing). `view_flags.jsonl` (APIs × source layouts), `out_flags.jsonl` (out-capable APIs × out
  layouts: it returns the out ITSELF and writes only its slots), `shape_args.jsonl` (the same dims as `Shape` / `long[]`
  / `int[]` / `int` / a view's Shape), `array_args.jsonl` (secondary operand positions × layouts). Operands rebuild as
  owners (`own: true`) or read-only (`writeable: false`). Zero excuses; every known divergence is CARVED by `_gap_of`
  into `corpus/gaps/<tier>_gaps.jsonl` (exact: each carved case diverges, no kept case does), replayed by the
  `*_KnownGaps` `[OpenBugs]` tests, with repros in `NumSharp.Tests/OpenBugs.ViewFlags.cs`. The worst class found:
  C-contiguous views at a non-zero offset (`np.split` / `np.unstack` / `np.array_split` children) read from the parent's
  start (silent wrong values); still open in `np.where`, `np.ldexp` and the flat `np.nansum` / `np.nanprod`, fixed in
  `np.repeat`, `np.unique`, the set routines and `np.searchsorted`; `a[:, idx]` on such a view throws. Coverage join:
  `oracle_map.json` `viewflags` / `outflags` / `shapearg` param keys (`params.fn`); the leak sweep labels these keys'
  families `op:fn`, and the leaks they found sit in `KnownEscapes` with ceilings. Portability: libm-backed `out=` rows
  take zeros; FFT lengths stay in {2, 3, 4, 6}.
- The relayout tiers (`test/oracle/gen_relayout_oracle.py`; README "The relayout tiers"; run it AFTER every other
  generator, its sources are their files): every (op key, operand position, dtype class) cell is surveyed for 19 ndarray
  KINDS (C / F / permuted / stepped, each also at an extra offset; reversed; stepped and reversed; broadcast and
  one-element broadcast, also at an offset; a 0-d owner; a 0-d view at an offset; a one-element 1-D; empty; an empty view
  inside a real buffer; plus read-only), and each missing cell is filled by re-emitting a source case that a plain NumPy
  call reproduces byte for byte, one operand moved into the kind, NumPy's answer recomputed. `relayout.jsonl` and
  `relayout_host.jsonl` (via `RunHostLibmCorpus`). A variant inherits its source tier's excuses, so
  `RelayoutLayoutInvariance` holds, with no registry, that NumSharp's outcome equals its own outcome on a C-ordered copy
  (`FuzzCorpus.ReconstructCanonical`) wherever NumPy's does not change; the relayout tiers are exempt from the NEP 50 0-d
  branches (`MisalignedRegistry.IsRelayoutCase`). Carved gaps: the open ids of RL01–RL07 (`gaps/relayout_gaps.jsonl`,
  repros in `NumSharp.Tests/OpenBugs.Relayout.cs`). It and the matrix-operand generator skip each other's files, so their
  order does not matter. An empty operand descriptor with a buffer rebuilds as an empty VIEW over it.
- The surface tier (`test/oracle/gen_surface_oracle.py`; README "The surface tier"): contracts for APIs the coverage
  inventory lists as available (the Array-API linalg forms, the histogram family, array_equiv, shares_memory /
  may_share_memory, broadcast_shapes, apply_along_axis / apply_over_axes, bmat, the text formatters, ndarray members,
  finfo / iinfo / dtype / ndarray.flags / poly1d attributes, the np.broadcast / flatiter / np.nditer members after k steps
  and their iteration protocol, the print and buffer options, dtype.type, frompyfunc, the inventory's aliases). Inputs
  the operand corpus cannot express ride params: a FUNCTION is a name from a fixed menu both sides map to one
  computation, two VIEWS of one buffer are step recipes applied to one owner, global options are set and restored inside
  the case. Op keys are the API names (`coverage/oracle_map.json` prefix rules). `surface.jsonl` (valueclass `surface`,
  zero excuses; its section dragon4 holds the float formatters) and `gaps/surface_gaps.jsonl` (the open ids of
  SU01…SU42, carved exactly by modelling each bug; repros in `NumSharp.Tests/OpenBugs.Surface.cs`, a fixed id's repro
  kept as an untagged regression pin). The relayout generator skips this tier. Not provable by design:
  numpy.ma.polyfit (needs a LAPACK backend), numpy.nditer.debug_print (a C-level stdout dump) and the `BitGenerator` /
  engine `.lock` rows (a `threading.Lock`).
- The numpy.ma surface tier (`test/oracle/gen_ma_surface_oracle.py`): contracts for the numpy.ma functions (creation
  wrappers, stacking, fill-value helpers, mask predicates, harden/soften, masked_object, putmask,
  convolve/correlate/cov/corrcoef, the edge finders, compress_nd / compress_rowcols, ndenumerate, apply_along_axis /
  apply_over_axes) over masked operands in every relayout data kind × eight mask patterns. The ma harness has `tuple` and
  `text` result kinds (a Python object as its normalized repr) and `RunMaGapCorpus`; masked results record the FORM and,
  where NumPy defines it, the RAW data at masked slots. `ma_surface.jsonl` (valueclass `masurface`, zero excuses) and
  `gaps/ma_surface_gaps.jsonl` (the open ids of MS1–MS37; repros in `NumSharp.Tests/OpenBugs.MaSurface.cs`, a fixed id's
  repro kept as an untagged regression pin).
- The matrix tier: `matrix.md`. The matrix-operand tier (`test/oracle/gen_matrix_operand_oracle.py`; run it AFTER every
  other generator): corpus cases that a plain NumPy call reproduces, re-emitted with one 2-D operand typed as a matrix
  (`"matrix": true`, layout `matrix@<position>`), NumPy's answer recomputed (class, dtype, shape, values, or error type
  and text). `matrix_operand.jsonl` + `matrix_operand_host.jsonl` (via `RunHostLibmCorpus`) + `gaps/matrix_operand{,_host}_gaps.jsonl`
  (MO01–MO05, carved exactly; `MatrixOperand*_KnownGaps`). The CLASS is compared without a registry call, and
  `MisalignedRegistry` returns null for a shape / threw / error-text / expected-error divergence of a `matrix@` case (the
  source tier's value excuses stay; the NEP 50 0-d blankets are off). The registry calls the function form for operator
  keys (`OpRegistry.MatrixOperand.cs`). NumPy 2.4.2's `np.stack` / `np.dstack` with a matrix operand raise and corrupt
  NumPy's heap, so the generator makes every matrix call in a worker process it retires after such an error.
- `products.jsonl`: the CBLAS family, tensor products, cross/cov/corrcoef and the Array-API vector/matrix norms.
  `random_parity.jsonl` (portable) and `random_parity_host.jsonl` (host libm) pin the stream methods and distributions;
  managed `multivariate_normal` (no LAPACK backend) stays carved and `[OpenBugs]`-pinned.
- The random-API oracle (`test/oracle/gen_random_oracle.py`; README "The random-API oracle"): every public member of the
  random world (`test/oracle/random_surface.json`, G1-pinned to reflection) replayed by its EXACT C# signature
  (`params.sig`, reflection-invoked; an omitted optional is the declared default, and NumPy is called without it) on
  every engine × the fixed seeds, recording NumPy's result and the receiver's full post-call state. Tiers `random_api`
  (portable) / `_host` (win-amd64 libm) / `_mvn` (NumPy's own scipy-openblas) / `_lp64` (Linux NumPy through WSL). Call
  shapes are pinned at compile time instead (`RandomSampling/RandomApiWholeness.Test.cs`). Gates `RandomApiCoverageTests`
  (corpus-only): G2 overloads, G3 parameters (omitted and non-default, two values, null and non-null, `params` 0 and 2+,
  every accepted enumerated value plus one rejection, NumPy's parameter names and order against
  `test/oracle/random_numpy_signatures.json`), G4 engines, G5 seeds, G6 state; exemptions are reasoned and retire
  themselves. The nightly `fuzz-soak.yml` jobs `random-api-lp64` and `random-api-soak` replay the families under fresh
  seeds (`RandomApiSoak`, `NUMSHARP_RANDOM_SOAK_DIR`). After a surface change: `NUMSHARP_WRITE_RANDOM_SURFACE=1` on the G1
  test, then regenerate.
- `linalg_parity.jsonl` (host-pinned like `matmul_parity`: `linalg_parity.host.jsonl`, `MatmulParityPin`):
  `cholesky`/`eig`/`eigvals`/`eigh`/`eigvalsh`/`svd`/`svdvals`/`pinv`/`matrix_rank`/`cond{None,2,-2}`/`lstsq`/`qr` and
  `norm{2,-2,nuc}`, plus the LU family `solve`/`inv`/`det`/`slogdet`/`tensorinv`/`tensorsolve` and `matrix_power(n<0)`,
  byte-exact on NumPy's pinned scipy-openblas at threads = 1, Inconclusive off the pinned host; tuple results and array
  siblings; float32 byte-exact too (NumPy's lite double-compute-round). The LU family is `getrf`/`gesv` (deterministic
  pivoting), so every output is byte-reproducible (`det` 0-D/1-D, a complex `slogdet` sign of unit modulus, singular →
  `(0,-inf)`). Excluded: complex-Hermitian `eigh` eigenvectors (the heevd phase varies across processes; covered by
  `eigvalsh`), float32 `eig` with complex eigenvalues (complex64 against complex128), and the non-SVD `cond`/`norm`
  orders (a 1-ULP reduction order).
- `poly.jsonl` (`gen_oracle.py poly`): the portable legacy polynomial family (`poly` 1-D, `polyval`, `vander`, `polyder`,
  `polyint`, `polyadd`/`polysub`/`polymul`, the `polydiv` tuple, `poly1d`). `einsum.jsonl`: integer and small-exact-float
  contractions plus the view path, over NONZERO operands (a signed zero diverges through NumPy's `+0.0`-seeded `sop`
  accumulator), and `einsum_path` (the info string, `text` kind, non-ellipsis). The products tier also holds `cross`
  (bit-exact f64/f32/c128/i64) and `cov`/`corrcoef` (byte-exact small and unweighted; the weighted form is 1 ULP off and
  covered by battle tests). The backend polynomial ops (`roots`/`polyfit`/`poly` of a matrix, through eigvals/lstsq) ride
  `linalg_parity`. The pure single-operand ops are carved out of the "unary ~ULP" blanket, so a ≤ 2-ULP drift fails.

## The `.npy` / `.npz` format oracle

The same philosophy, a separate corpus, and a stronger claim: NumSharp's writer must be byte-identical to `np.save`.
- Regenerate: `python test/oracle/gen_npy_oracle.py` (deterministic; `numpy==2.4.2`), then `dotnet build`.
- Run: `dotnet test --filter "TestCategory=NpyOracle"`: every dtype × {0-d, empty, 1-D, 2-D, 3-D, unit, empty 2-D/3-D}
  × {C, F, strided, reversed, offset, broadcast, transposed} × versions {1.0, 2.0, 3.0} × {little, big, native} endian,
  value fidelity (NaN payloads including sNaN, subnormals, signed zero, integer extremes, BMP seams) and malformed /
  unsupported / hostile files whose messages must match NumPy's.
- `kind: "header"` cases drive the writer through the header dict, because two writer branches are unreachable from a
  real array: the growth padding is normally invisible (shrink the body by 5 characters and the alignment padding grows
  by 5), so a wrong growth axis (`shape[0]` against `shape[-1]`) shows only on shapes that tip the header across a
  64-byte bucket (10¹⁷ elements); the v1.0 → v2.0 auto-selection boundary sits at 21,817 dimensions.
- Reverse interop (manual; Python + the SDK): `python test/oracle/verify_npy_interop.py` has NumSharp write files and
  real NumPy read them, the only proof for `.npz` output, whose ZIP framing differs from Python's `zipfile`.

## The text I/O oracle

`test/oracle/gen_textio_oracle.py` → `test/NumSharp.Tests.Oracle/TextIO/corpus/textio_<section>.jsonl`, replayed by
`TextIO/TextIOOracleTests.cs` (category `TextIOOracle`, one test per section, zero excuses): `np.fromstring`,
`np.fromfile`, `np.loadtxt`, `np.savetxt` and `ndarray.tofile` over every dtype, StringDType included (`text-io.md`).
- Sections: `fromtext` (random `fromstring` / text `fromfile` calls: separators, counts, offsets, the file position
  afterwards, texts past the C `FILE` model's 64 KB chunk), `frombin` (binary reads), `fromedges` (a FIXED grid: every
  scanner token alone, after a clean element and before one; every separator in seven placements; bytes at or past
  0x80), `loadtxt` (every option and source), `ltfields` (numeric fields, every refused one alone), `lttokens` (tokenizer
  edges), `savetxt`, `tofile`.
- Regenerate with LINUX NumPy (the host check refuses anything else: NumSharp pins a 64-bit C `long` and glibc's
  `fscanf`): `wsl -e bash -lc 'cd ~ && ~/np242/bin/python /mnt/k/source/NumSharp/test/oracle/gen_textio_oracle.py'` (from
  `~`: `refs/numpy`'s sources would shadow the package), `--sections a,b` for a subset, then `dotnet build`.
  Deterministic (one seed per section).
- Newlines: a file NAME written by `np.savetxt` goes through Python's text mode (`\n` → `os.linesep`), and so does
  NumSharp. Each such case carries a `crlf` twin recorded through `io.TextIOWrapper(io.BytesIO(), newline="\r\n")`, which
  the generator first proves equal to the real file with `newline="\n"`; the replay picks it on a Windows host. Gated on
  Windows AND Linux (WSL: `~/.dotnet/dotnet test <copied output>/NumSharp.Tests.Oracle.dll --filter
  TestCategory=TextIOOracle`), both TFMs.
- Left out on purpose (documented divergences, unit-tested): results NumPy builds with zero-width string elements; a
  non-native dtype's TEXT read (the case records NumPy's answer for the native dtype); `tofile` of a non-ASCII element
  (crashes NumPy 2.4.2).
- The `fromedges` grid exists because a random text fails on an early token before an overflowing value is read, so two
  planted scanner bugs (fromfile's saturation by sign, the `0xFF` a signed `char` loses) survived the random sections.
  Error classes: only `OverflowError` → `OverflowException` is aliased; every other error the corpus records is raised
  under NumPy's own class name.
- Trap: a temp file costs ~2.4 ms in the user's `%TEMP%` (~6 ms inside the test host: the antivirus scans a new file on
  its first open) against ~0.6 ms in the build output on K:, so the replay writes beside the test assembly
  (`TextIO/tmp_<pid>`).
- Trap: `cond ? long : double` is typed `double` in C#: the replay's JSON decoder needs `(object)l`, or a Python int in a
  parameter list (`comments=[1]`) reaches NumSharp as `1.0` ("not float" where NumPy says "not int").

## Traps

- `describe()` serializes an operand's BASE in C order, so a layout is a C-contiguous base plus a view (an F series is the
  `.T` of a C buffer holding `base.T`).
- A facet case (flags, layout) REPLACES the result, so the registry must dispose the result it drops, or the leak gate
  (`UndisposedIntermediateTests`) counts one escaped buffer per case.
- A corpus regenerated with `--uncarved` and copied into `bin/.../Fuzz/corpus` stays there after the carved regeneration
  (copy-if-newer keeps the newer bin file), so the gate replays the uncarved file: copy the carved corpus and gap files
  into both `bin/Release/{net10.0,net8.0}` (or rebuild) after an uncarved replay.
- A carve rule written from a failure summary over-carves error cases (NumSharp raises NumPy's text before it reaches the
  bug): check the rule set against an uncarved replay's failure dump in both directions (failing but not carved; carved
  but passing).
- `CheckError` reports "NumPy raises but NumSharp produced a result" as a VALUE divergence, so a source tier's value
  excuse can hide error-parity gaps; expected-error cases are strict in the matrix-operand tier for that reason.
- A carve on the result FORM hides value/mask divergences behind it (the form is compared first):
  `NUMSHARP_MA_SKIP_FORM=1` replays without the form check.
- An uncommon value class needs an explicit edge set: random operands never reach a vector stage's rare branches, never
  pair two NaNs, and never hit a float16 double-rounding hazard.
