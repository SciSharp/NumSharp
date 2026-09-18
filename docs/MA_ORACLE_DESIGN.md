# `np.ma` differential-fuzz oracle — design & coverage map

**Goal:** bring `numpy.ma` (masked arrays) under the committed NumPy 2.4.2 differential corpus, the same
way every NDIter-backed `np.*` op already is — so a masked-array regression turns the `FuzzMatrix` gate red
with no Python at test time. Until now `np.ma` was byte-validated only by **throwaway** dotnet-run
differentials (the pass-13 6055-case run, `docs/MA_MODULE_AUDIT.md`) plus 90 `MaskedArrayTests`; nothing
committed replays NumPy bytes for it in CI.

Source of truth: NumPy 2.4.2. Implementation under test: `src/NumSharp.Core/Ma/MaskedArray.cs`. This doc is
the map of **what** masked op goes in **which** tier and **how** a masked case is serialized/compared.

## Why a NEW corpus family (not a fold into the existing tiers)

A `MaskedArray` is `NDArray _data` + an optional bool `NDArray _mask` (`_mask == null` IS NumPy's `nomask`
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
STRIPPED — ma delegates to the identical `np.*` op on the data, so the same excuse ledger applies — plus two
ma-local excuse branches (`var`/`std` accumulation-order ≤ small ULP, and the complex-unary ≤3-ULP envelope),
matching the honest boundary in `MA_MODULE_AUDIT §5.6/§5.8`.

## Tier map (what goes where)

| Tier file | `gen_ma_*` | Ops | Result kinds | Host-pin |
|-----------|-----------|-----|--------------|----------|
| `ma_unary.jsonl` | `gen_ma_unary` | abs/absolute/negative/conjugate/sqrt/exp/log/log2/log10/sin/cos/tan/sinh/cosh/tanh/arcsin/arccos/arctan/arcsinh/arccosh/arctanh/floor/ceil/fabs/around/angle/logical_not | masked | yes (libm/complex) |
| `ma_binary.jsonl` | `gen_ma_binary` | add/subtract/multiply/divide/true_divide/floor_divide/mod/remainder/fmod/power/arctan2/hypot + equal/not_equal/less/less_equal/greater/greater_equal + logical_and/or/xor + bitwise_and/or/xor/left_shift/right_shift + maximum/minimum | masked | partial (power/libm) |
| `ma_reduce.jsonl` | `gen_ma_reduce` | sum/prod/product/mean/min/max/amin/amax/ptp/var/std/all/any/anom/median/average + count/count_masked/argmin/argmax(→array) | masked, array | var/std ma-excused |
| `ma_scan.jsonl` | `gen_ma_scan` | cumsum/cumprod/diff/ediff1d | masked | no |
| `ma_manip.jsonl` | `gen_ma_manip` | ravel/flatten/reshape/transpose/swapaxes/moveaxis/squeeze/expand_dims/repeat/concatenate/stack/hstack/vstack/dstack/column_stack/atleast_1d/2d/3d/diag/diagflat/diagonal/append/resize + hsplit(→masked_tuple) + compressed/getdata/getmask/getmaskarray/filled(→array) | masked, array, masked_tuple | no |
| `ma_construct.jsonl` | `gen_ma_construct` | masked_where/masked_equal/masked_greater/…/masked_values/masked_inside/masked_outside/masked_invalid/fix_invalid/masked_all/masked_all_like + array/masked_array | masked | no |
| `ma_select.jsonl` | `gen_ma_select` | take/choose/compress/clip/where/put/putmask(→mutated masked)/nonzero(→tuple) | masked, array | no |
| `ma_sortsetops.jsonl` | `gen_ma_sortsetops` | sort/argsort(→array)/unique + intersect1d/union1d/setxor1d/setdiff1d/isin/in1d | masked, array | no |
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
| `unique` (masked) | numpy treats each MASKED element as a distinct value (ill-specified quirk); nomask is byte-exact | generator restricts unique to nomask |
| `unique` (complex128 NaN) | the NaN element's imaginary lane is non-contractual | `[known gap]` excuse |
| `all`/`any`/`median` (axis=None, keepdims=True) | flat keepdims returns 0-d, numpy returns (1,)*ndim | generator skips that one combo |
| `var`/`std`/`mean`/`average`/`anom` | mask a NON-FINITE (nan/inf) reduction result where NumSharp leaves it unmasked (the masked-op composition differs) | `[known gap]` mask/value excuse (float/complex) |
| `average`/`anom`/`median` | return-dtype widens narrow-float → float64 where numpy keeps the width | `[known gap]` dtype excuse |
| `anom`/`median` | VALUE diverges on a non-contiguous 1-D input (their internal mean/sort over a strided view) | `[known gap]` value excuse |
| `sort` (NaN in unmasked) | masked-slot position vs NaN-ordering differs | `[known gap]` mask excuse |
| `isin`/`in1d` | do NOT propagate the element operand's mask to the bool result | `[known gap]` mask excuse |
| `arctanh`/complex-unary at nan/inf | inherited complex-unary special-value FINITENESS edge (msun vs System.Numerics) flips the `~isfinite` mask bit | `[documented]` complex-edge excuse (MisalignedRegistry branch 7 sibling) |
| `divide`/`mod` domained → ±0 | signed-zero of a division-to-zero (IEEE non-contractual) | `[documented]` signed-zero excuse |
| `power` complex | result FINITENESS at a pathological input differs (F5 class) | `[documented]` excuse |
| `var`/`std` accumulation | ~1 ULP order-dependence (the shared reduce-accumulation class) | reused shared `MisalignedRegistry` excuse (prefix-stripped) |

**Not included** (unit-test-only, same policy as the NDArray corpus): `cov`/`corrcoef`/large `dot`
(GEMM-bound, allclose), `convolve`/`correlate` (long-kernel float), `apply_along_axis`/`apply_over_axes`/
`fromfunction` (delegate-taking), `ndenumerate`, `notmasked_*`/`clump_*`/`mr_`/`masked_object` (polymorphic
/ index-DSL returns). These remain covered by `MaskedArrayTests`.
