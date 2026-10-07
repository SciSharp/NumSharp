# `np.ma` differential-fuzz oracle — design & coverage map

**Goal:** bring `numpy.ma` (masked arrays) under the committed NumPy 2.4.2 differential corpus, the same
way every NDIter-backed `np.*` op already is — so a masked-array regression turns the `FuzzMatrix` gate red
with no Python at test time. Until now `np.ma` was byte-validated only by **throwaway** dotnet-run
differentials (the pass-13 6055-case run, `docs/MA_MODULE_AUDIT.md`) plus 90 `MaskedArrayTests`; nothing
committed replays NumPy bytes for it in CI.

Source of truth: NumPy 2.4.2. Implementation under test: `src/NumSharp.Core/Ma/NDMaskedArray.cs`. This doc is
the map of **what** masked op goes in **which** tier and **how** a masked case is serialized/compared.

> **Renamed 2026-09-23:** NumSharp's port of `numpy.ma.MaskedArray` is the class `NDMaskedArray` (the `ND*`
> convention of `NDArray`), its constant type `NDMaskedConstant`, and the gate `NDMaskedArrayTests` — formerly
> `MaskedArray` / `MaskedConstant` / `MaskedArrayTests`. The masked constant is `np.ma.NDMasked`, with NumPy's
> `np.ma.masked` / `np.ma.masked_singleton` kept as aliases of the same instance. Historical notes below keep
> the names in use at the time; `MaskedArray` qualified by NumPy (`numpy.ma.MaskedArray`) is NumPy's class.

## Why a NEW corpus family (not a fold into the existing tiers)

An `NDMaskedArray` is `NDArray _data` + an optional bool `NDArray _mask` (`_mask == null` IS NumPy's `nomask`
fast path). The existing corpus can express only ONE buffer per operand and ONE buffer per result, so it
cannot carry a mask. The masked corpus therefore extends the schema, minimally and additively:

### Operand: data + optional mask

```
operand = { dtype, shape, strides, offset, bufferSize, buffer,   # the DATA view (unchanged)
            mask? }                                               # NEW: hex of a C-contiguous bool
```

- `buffer` is the data exactly as `layout_catalog.describe()` serializes any NDArray operand (any layout:
  C/F/strided/transposed/negstride/broadcast/offset).
- `mask` (optional) is a **C-contiguous bool buffer of `product(shape)` bytes** — the mask at the view's
  logical C-order positions. Absent ⇒ `nomask` (the fast path). The harness rebuilds
  `np.ma.masked_array(Reconstruct(data), maskArr)`, matching how the generator builds `np.ma.array(view, mask=…)`.
- The **masked-slot data does not affect the result comparison** (see below), but the real data is still
  serialized so NumSharp computes on byte-identical inputs.

### Result: `kind: "masked"` = filled(0) + mask

```
expected = { kind:"masked", dtype, shape,
             buffer,   # filled(0) bytes  (masked slots -> 0, unmasked -> data), C-contiguous
             mask }    # getmaskarray bytes (C-contiguous bool)
```

The comparator asserts **data dtype + shape + `filled(0)` bytes (BitDiff, NaN-aware) + `getmaskarray` bytes
(bit-exact bool)**. This is exactly the pass-13 "`M` token + hex" scheme expressed in the committed corpus:

- **Masked slots** compare as `0 == 0` on both sides — so the underlying data at a masked slot (arbitrary
  for `unique`/set-ops, input-restored for ufuncs) is a wildcard, never a false failure.
- **Unmasked slots** compare bitwise (NaN tokenized) — every real value is gated.
- **The mask itself** is gated bit-for-bit separately — so a mask that is one bit wrong fails even when the
  visible values coincide.
- The all-masked reduction (`np.ma.masked` singleton) is uniform: `filled(0)` → `0`, `getmaskarray` → `True`,
  shape `()`. NumSharp's 0-d fully-masked MaskedArray materializes identically. No separate "masked scalar"
  kind is needed.
- A partially-masked flat reduction where NumPy returns a **bare numpy scalar** (NumSharp returns a 0-d
  MaskedArray, the one documented type divergence, `MA_MODULE_AUDIT §5.1`) is compared by VALUE (filled+mask),
  so the wrapper-type difference is invisible and correct.

Other result kinds a masked op can return, reusing existing comparators:
- `array` — bare `NDArray` (`compressed`/`count`/`getdata`/`getmaskarray`/`filled`/`argsort`/`trace`/`vander`
  /`make_mask`/`flatten_mask`/…). Compared exactly like any `np.*` array result.
- `masked_tuple` — `MaskedArray[]` (`hsplit`). Arity + each slot via the masked comparator.
- `scalar` — a boxed bool/int (`is_masked`/`is_mask`/`isMaskedArray`/`allclose`/`allequal`/`ndim`/`size`),
  wrapped in a 0-d NDArray and compared as an array.
- `dtype` — `make_mask_descr` (always bool). Compared by NumPy dtype name.

## Op-name namespace

Every masked op key is prefixed **`ma.`** (`ma.add`, `ma.sum`, `ma.masked_where`) so it never collides with
the `np.*` key of the same name (`sum` exists in both registries). `OpRegistry.Ma.ApplyMasked(op, params,
MaskedArray[])` dispatches the prefixed name and returns `object` (MaskedArray / MaskedArray[] / NDArray /
boxed scalar / DType). Divergences route through the shared `MisalignedRegistry` with the `ma.` prefix
STRIPPED — ma delegates to the identical `np.*` op on the data, so the same excuse ledger applies — plus the
ma-local excuse branches of `FuzzCorpusTests.Ma` (the complex-unary ≤3-ULP envelope and the known gaps below),
matching the honest boundary in `MA_MODULE_AUDIT §5.6/§5.8`. Since 2026-10-06 none of them takes `mean` / `var` /
`std` / `anom`, which are numpy.ma's statements (plan `docs/plans/api-parameter-parity.md` A6, Pass 3 below).

## Tier map (what goes where)

| Tier file | `gen_ma_*` | Ops | Result kinds | Host-pin |
|-----------|-----------|-----|--------------|----------|
| `ma_unary.jsonl` | `gen_ma_unary` | abs/absolute/negative/conjugate/sqrt/exp/log/log2/log10/sin/cos/tan/sinh/cosh/tanh/arcsin/arccos/arctan/arcsinh/arccosh/arctanh/floor/ceil/fabs/around/angle/logical_not | masked | yes (libm/complex) |
| `ma_binary.jsonl` | `gen_ma_binary` | add/subtract/multiply/divide/true_divide/floor_divide/mod/remainder/fmod/power/arctan2/hypot + equal/not_equal/less/less_equal/greater/greater_equal + logical_and/or/xor + bitwise_and/or/xor/left_shift/right_shift + maximum/minimum | masked | partial (power/libm) |
| `ma_reduce.jsonl` | `gen_ma_reduce` | sum/prod/product/mean/min/max/amin/amax/ptp/var/std/all/any/anom/median/average + count/count_masked/argmin/argmax(→array) | masked, array | no (mean/var/std/anom strict but finding 5) |
| `ma_meanvar.jsonl` | `gen_ma_meanvar` | mean/var/std/anom with every layout, mask class, keepdims, ddof, dtype= loop and mean= form, NumPy's errors (Pass 3) | masked (+ `raw`, `form`), errors | no; zero excuses |
| `ma_scan.jsonl` | `gen_ma_scan` | cumsum/cumprod/diff/ediff1d | masked | no |
| `ma_manip.jsonl` | `gen_ma_manip` | ravel/flatten/reshape/transpose/swapaxes/moveaxis/squeeze/expand_dims/repeat/concatenate/stack/hstack/vstack/dstack/column_stack/atleast_1d/2d/3d/diag/diagflat/diagonal/append/resize + hsplit(→masked_tuple) + compressed/getdata/getmask/getmaskarray/filled(→array) | masked, array, masked_tuple | no |
| `ma_construct.jsonl` | `gen_ma_construct` | masked_where/masked_equal/masked_greater/…/masked_values/masked_inside/masked_outside/masked_invalid/fix_invalid/masked_all/masked_all_like + array/masked_array | masked | no |
| `ma_select.jsonl` | `gen_ma_select` | take/choose/compress/clip/where/put/putmask(→mutated masked)/nonzero(→tuple) | masked, array | no |
| `ma_sortsetops.jsonl` | `gen_ma_sortsetops` | sort/argsort(→array)/unique + intersect1d/union1d/setxor1d/setdiff1d/isin/in1d | masked, array | no |
| `ma_setops.jsonl` | `gen_ma_setops` | unique (+ return_index / return_inverse → masked_tuple) / in1d / isin / intersect1d / setxor1d / union1d / setdiff1d (+ assume_unique / invert) over 13 dtypes, value pools, mask classes, hard masks, plain operands, layouts, mixed dtypes and 4,500-element integer operands, NumPy's errors (2026-10-08) | masked (+ `form`, in1d/isin `raw`), masked_tuple, errors | no; zero excuses |
| `ma_extras.jsonl` | `gen_ma_extras` | dot/inner/outer(small, exact) + trace/vander(→array) + mask_rows/mask_cols/mask_rowcols + compress_rows/compress_cols(→array) + predicates is_mask/is_masked/isMaskedArray/allclose/allequal(→scalar) + make_mask/make_mask_none/mask_or/flatten_mask(→array) + make_mask_descr(→dtype) | masked, array, scalar, dtype | no |

Deliberately EXCLUDED from the byte corpus (unit-test-only, same policy as the NDArray corpus):
- **`cov`/`corrcoef`** and large `dot` — GEMM-bound, allclose-only at scale (byte-exact only for tiny shapes,
  and their mask logic is already exercised by small `dot`/`outer`/`inner`).
- **`convolve`/`correlate`** — long-kernel float divergence (the `NumSharp.Interop.OpenBLAS` sliding-dot class).
- **`apply_along_axis`/`apply_over_axes`/`fromfunction`** — take a C# delegate (not corpus-serializable).
- **`ndenumerate`** (IEnumerable), `put`/`putmask` non-mutated-return forms, `notmasked_*`/`clump_*`
  (polymorphic Slice/object returns), `masked_object`/`mr_` (index-expr DSL / object path), `frombuffer`.

## Mask patterns generated per operand

Each op × dtype × layout is emitted with three mask patterns to exercise every branch:
- `nomask` — `_mask == null` fast path.
- `some` — a deterministic checker/edge bit pattern (exercises OR-propagation, restore-at-slot, domain masks).
- `all` — fully masked (exercises the `masked` singleton / all-masked-slice identity fill).

## Wiring

- Generator: `test/oracle/gen_oracle.py` — `gen_ma_*` + `main()` `elif mode.startswith("ma_")`, using a new
  `ma_operand()` (data descriptor + mask hex) and `_masked_expected()` (filled(0)+mask) helper. Char/Decimal
  are NOT woven (no NumPy analog for those dtypes under `np.ma`; the 13 NumPy dtypes carry the coverage).
- Harness: `test/NumSharp.Tests.Oracle/Fuzz/OpRegistry.Ma.cs` (`ApplyMasked`) +
  `FuzzCorpusTests.Ma.cs` (`RunMaCorpus` + `CompareMasked`/`CompareMaskedTuple` + tier `[FuzzMatrix]` methods)
  + `FuzzCorpus.Operand.Mask` / `FuzzCorpus.Expected.Mask` fields + `MinCases` floors.
- Regenerate: `python test/oracle/gen_ma_oracle.py <tier|all>`, then `dotnet build`, then run the tier
  (`dotnet test --filter FullyQualifiedName~FuzzCorpusTests.Ma`). It is a STANDALONE generator (like
  `gen_nan_oracle.py`), not a mode of `gen_oracle.py`.
- The corpus-replay leak/coverage gates (`UndisposedIntermediateTests`, `OracleCoverageStrengthTests`)
  SKIP `ma_*.jsonl`: those drive `OpRegistry.Apply` over the ordinary-op model, which ma bypasses via
  `ApplyMasked` + `RunMaCorpus`. The name-collecting gates (`Journey3Touched`, `OracleSurfaceCoverage`)
  ignore the `ma.`-prefixed keys harmlessly.

## Corpus stats (2026-09-18)

**26,586 committed cases** across 9 tiers (`ma_unary` 4452, `ma_binary` 4360, `ma_reduce` 11234,
`ma_scan` 1035, `ma_manip` 2536, `ma_construct` 1916, `ma_select` 516, `ma_sortsetops` 464,
`ma_extras` 73). All 9 tiers GREEN on net8.0 + net10.0.

## Findings — np.ma is NOT byte-identical everywhere (the corpus corrected the premise)

The prior "byte-identical" claim rested on a throwaway 6055-case differential that used `.copy()` before
reading, materializing away the layout-sensitive bugs. This committed corpus tests strided/non-contiguous
views and cross-dtype combos directly and surfaced genuine divergences. One was a real bug worth fixing on
the spot; the rest are documented gaps (excused with `[known gap]` labels in `FuzzCorpusTests.Ma`, so the
gate is green AND every divergence is catalogued and printed at run time).

### FIXED in this pass (real bug, `src/NumSharp.Core/Ma/MaskedArray.cs`)

- **Mask/scratch allocation from a non-contiguous `Shape` produced GARBAGE.** `np.zeros(a.Shape)` for a
  strided/offset/transposed `a` builds a STRIDED zeros over the base buffer and materializes the
  uninitialized tail as non-zero bytes — so `getmaskarray` of a nomask `[::2]` view returned `01`-bytes,
  and every manip/constructor op (transpose/squeeze/reshape/diagonal/masked_invalid/masked_array/
  masked_all_like/…) plus `count` inherited a corrupt mask/count on non-contiguous inputs. Root-caused with
  a native (non-reconstructed) strided view, so it is a genuine library bug, not a harness artifact. Fixed
  by sizing every mask/scratch allocation by the DIMENSIONS (`DimsOf(a)` / `new Shape(a.shape)` /
  `CountUnmasked(…, DimsOf(…))`), matching NumPy's `make_mask_none(a.shape)`. Verified: the whole
  `ma_manip`/`ma_construct` strided class went green and the 90 `MaskedArrayTests` still pass.

### Documented KNOWN GAPS (excused, tracked for a future fix pass)

| Area | Divergence | Disposition |
|------|-----------|-------------|
| `argsort` (ndim≥2) | NumSharp has no `axis=None` flatten; uses -1 (numpy's DOCUMENTED FUTURE default — numpy 2.4.2 emits a FutureWarning that its current `None` is deprecated) | generator restricts argsort to 1-D nomask |
| `argsort` (masked / negstride float) | masked/tie INDEX ordering differs (fill-then-sort vs stable radix) | restricted to nomask non-negstride |
| ~~`unique` (masked)~~ | FIXED 2026-10-08: numpy.ma's arraysetops are NumPy's statements (`Ma/NDMaskedArray.SetOps.cs`); a masked slot sorts by the fill key, so it can sit among unmasked values equal to that key (NumPy's "distinct masked value" quirk is this). `ma_setops.jsonl` covers masked unique; only the cases whose answer depends on NumPy's unstable quicksort tie order are dropped by its generator | — |
| ~~`unique` (complex128 NaN)~~ | FIXED 2026-10-08: `np.unique`'s complex path is NumPy's hash (`unique.cpp`): each class keeps its first element, all NaN-part values one class | — |
| `all`/`any`/`median` (axis=None, keepdims=True) | flat keepdims returns 0-d, numpy returns (1,)*ndim | generator skips that one combo |
| `average` (was also `var`/`std`/`mean`/`anom`: FIXED 2026-10-06, Pass 3) | mask a NON-FINITE (nan/inf) reduction result where NumSharp leaves it unmasked (the masked-op composition differs) | `[known gap]` mask/value excuse (float/complex) |
| `average`/`median` (was also `anom`: FIXED 2026-10-06) | return-dtype widens narrow-float → float64 where numpy keeps the width | `[known gap]` dtype excuse |
| `median` (was also `anom`: FIXED 2026-10-06) | VALUE diverges on a non-contiguous 1-D input (its internal sort over a strided view) | `[known gap]` value excuse |
| `sort` (NaN in unmasked) | masked-slot position vs NaN-ordering differs | `[known gap]` mask excuse |
| ~~`isin`/`in1d`~~ | FIXED 2026-10-08: NumPy's statements — a flag whose sorted neighbour is masked is masked (the old composition never masked a flag; all 8 cases failed once the excuse was removed) | — |
| `arctanh`/complex-unary at nan/inf | inherited complex-unary special-value FINITENESS edge (msun vs System.Numerics) flips the `~isfinite` mask bit | `[documented]` complex-edge excuse (MisalignedRegistry branch 7 sibling) |
| `divide`/`mod` domained → ±0 | signed-zero of a division-to-zero (IEEE non-contractual) | `[documented]` signed-zero excuse |
| `power` complex | result FINITENESS at a pathological input differs (F5 class) | `[documented]` excuse |
| ~~`var`/`std` accumulation~~ | FIXED 2026-10-06 (Pass 3): numpy.ma's own statements, bit-exact; the shared summation blanket no longer takes them | — |

**Not included** (unit-test-only, same policy as the NDArray corpus): `cov`/`corrcoef`/large `dot`
(GEMM-bound, allclose), `convolve`/`correlate` (long-kernel float), `apply_along_axis`/`apply_over_axes`/
`fromfunction` (delegate-taking), `ndenumerate`, `notmasked_*`/`clump_*`/`mr_`/`masked_object` (polymorphic
/ index-DSL returns). These remain covered by `MaskedArrayTests`.

## Pass 2 (2026-09-18) — deeper assertions, more discrepancies

Broadened the corpus from **26,586 → 68,860 cases** to hit the paths pass 1 did not:
**edge layouts** (`c_contiguous_3d`/`transposed_3d`/`f_contiguous_3d`/`broadcast_1d_to_2d`/`scalar_0d`/
`one_element_1d`/`empty_2d`/`negstride_2d_offset`/`strided_2d_cols`/`reshape_view_2d`) across
unary/reduce/scan/manip/construct/sortsetops; **scalar + broadcast pair layouts** (`pp_broadcast_col`,
`pp_scalar_right`/`left`) and **6 more dtype pairs** in binary; the skipped ops **`choose`** and **`take`
mode=wrap/clip**; and parameter variations (`diagonal` offset, `diag` k, `moveaxis`, `diff` n/axis,
`cumsum` axis, second `expand_dims` axis). It surfaced:

### FIXED (real bugs)

- **`ma.median` on an EMPTY array — intermittent HEAP-CORRUPTION CRASH** (fatal, uncatchable
  `AccessViolation`, same class as the pass-13 `arctan2(complex)` OOB). `MedianAxisMasked` guarded only the
  *reduced* axis being 0-length, so `median(axis=1)` of a `(0,3)` array (0 rows, reduced axis length 3)
  slipped past and ran `sort` / `take_along_axis` / middle-averaging on 0-row data, writing out of bounds —
  cumulative corruption that faulted a later allocation (needed op diversity + ~2000 complex128 reduces to
  surface; localized by op-exclusion bisection to `median`, then to the `empty_2d` layout). Fixed with a top
  guard: **any** empty dimension short-circuits to `mean` (NumPy's `_median` empty-slice value) BEFORE the
  OOB-prone path. `src/NumSharp.Core/Ma/MaskedArray.cs` `MedianAxisMasked`.
- **`ma.around` dispatch (harness)** — NumPy's `ma.around` IS the round ufunc, which collapses a 0-d
  all-masked input to the float64 `masked` singleton; the registry called the two-arg `around(a, decimals)`
  = round path, which preserves the input dtype. Routed decimals=0 to the one-arg ufunc (`OpRegistry.Ma.cs`).
  The two-arg `ma.round`/`around(a, decimals)` NOT collapsing a 0-d all-masked to the singleton is a latent
  library inconsistency (only the ufunc form is corpus-tested; noted here for a future fix).

### Documented KNOWN GAPS added

| Area | Divergence | Disposition |
|------|-----------|-------------|
| `median` value on 2-D non-contiguous (`strided_2d_cols`/`negstride_2d_offset`); `anom` FIXED 2026-10-06 (Pass 3) | internal sort over a strided view reads the wrong stride | extended the pass-1 median non-contiguous `[known gap]` excuse |
| complex128 `cumprod` | ~ULP per component — host-FMA-contracted `npy_cmul` vs .NET `Complex operator*` (bounded ≤256 ULP/component) | `[documented]` the nanscan complex-cumprod carve-out class |

### Noted (identified by code-read, not corpus-plumbed)

- **`minimum`/`maximum`/`default_fill_value` widen a FLOAT fill to `double`** (`float16`/`float32` inputs)
  where NumPy returns the input float width. A minor scalar-dtype gap; the complex fills (a real pass-1-era
  bug) are correct. Left for a future pass (boxed-scalar wrapping + a dtype excuse).

### Clean (broadened with NO new divergence)

manip/construct/select/sortsetops/binary all stayed byte-exact across the edge layouts, scalar/broadcast
operands and new params — notably the pass-12 strong-scalar-widening fix holds across the 6 added dtype
pairs and the 0-d scalar-operand layouts, and complex128 `sort`/`unique` on empty/edge shapes did NOT crash
(only `median` did).

## Pass 3 (2026-10-06) — mean / var / std / anom as numpy.ma's statements

Plan `docs/plans/api-parameter-parity.md` A6. The four functions computed a masked reduction of their own: a slice
NumPy masks (all masked, `count - ddof <= 0`) held NaN instead of the masked 0, an unmasked inf stayed where NumPy
masks it into 0.0, the variance kept the divide's mask, and the masked mean of a float32 stayed float32. `ma_reduce`
excused them under the shared summation and dtype blankets (`IsMaVarStd` / `IsMaMean`, 254 cases).

**The port** (`src/NumSharp.Core/Ma/NDMaskedArray.cs`) runs NumPy's statements: without a mask, ndarray's `_mean` /
`_var` on the masked array (their `true_divide(out=...)` meets `MaskedArray.__array_wrap__`'s divide domain: a quotient
with `|q| * tiny >= count` becomes 1.0, masked); with a mask, `dsum * 1. / cnt` through ma.multiply and
ma.true_divide, `self - mean` through ma.subtract, the square in place (complex `abs(danom) ** 2` through ma.power),
the division by `count - ddof` and the replaced mask `mask_or(mask.all(axis), cnt <= 0)`; std is ma.sqrt of the
variance and ignores `mean=`; anom is `self - mean`. Each intermediate a later sum walks takes NumPy's ufunc layout.

**The tier** `ma_meanvar.jsonl` (26,379 cases) extends this corpus's contract with two fields, `raw` (the data at every
slot, masked ones included) and `form` (`masked` / `scalar` / `nomask` / `mask`), and with NumPy's errors
(`expects_throw`, type + text, checked by `CheckMaError`). It is held to zero excuses; the cells whose mean sums the
operand view itself on a layout np.sum still orders differently (finding 5, finding 10 / X12) are carved by
`_mv_carved` and pinned in `OpenBugs.Reductions.cs`. `ma_reduce` holds the four strict too, but for the finding-5
branch (3 cases). Gates: `FuzzCorpusTests.MaMeanVar`, `MaReduce`, `Ma/np.ma.meanvar.NumPyPort.Test.cs`, and the
tightness test `MaMeanVarStd_NoDivergenceIsExcused`.

**Still open** (plan items): the masked copy behind `filled` is slow under a random mask (X11, most of a masked
reduction's time); float16 subtraction with a column-broadcast operand (X15); `out=` of the four; the public
`NDMaskedArray.filled()` returns a C-order copy where NumPy returns the data itself or a K-order copy.
