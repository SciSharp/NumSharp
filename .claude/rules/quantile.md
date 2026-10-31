---
paths:
  - "src/NumSharp.Core/Statistics/{QuantileIndexes,QuantileText,QuantileWeighted}.cs"
  - "src/NumSharp.Core/Statistics/np.{quantile,percentile,median,nanquantile,nanpercentile,nanmedian}.cs"
  - "src/NumSharp.Core/Backends/Default/Statistics/Default.Quantile*.cs"
  - "src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.Quantile.cs"
  - "src/NumSharp.Core/Utilities/QuickSelect.cs"
  - "test/NumSharp.Tests/Statistics/np.{quantile_family,nanquantile_family,percentile,median}*.cs"
---

# The quantile family

`np.quantile` / `np.percentile` / `np.median` and their nan forms run NumPy 2.4.2's statements: `_ureduce`
(`Backends/Default/Statistics/Default.Quantile.UReduce.cs`), `_quantile` and its helpers
(`Default.Quantile{,.Functions,.Median}.cs`), byte for byte for every dtype; text data: `strings-compute.md`.

## `q`, the indexes and the lerp

NumPy computes the virtual index, the neighbours, the weight and `1 - t` in `q`'s dtype (the Python ints `n` / `n - 1`
and the methods' constants are weak under NEP 50). `Statistics/QuantileIndexes.cs` computes each operation in double and
rounds it to `q`'s dtype (exact for float16 / float32: a double carries at least 2p + 2 bits); a call's plan comes from
one builder per precision (`DirectILKernelGenerator.QuantilePlan`, cached per dtype), and the text paths use it too.
- `q = np.asanyarray(q)` keeps a Python int an integer, so an integer q TAKES the element over every dtype
  (`np.quantile([1., inf], 1)` is inf, where the float q 1.0 interpolates to NaN): the `long q` overloads and the array-q
  integer route. `weak_q = type(q) in (int, float)`: only a C# number keeps float16 / float32 data's dtype; a 0-d array q
  is strong (`DefaultEngine.QuantileCompute(weakQ:)`). A 2-d `q` is kept (rank 3+ raises after the axis checks); an
  unknown method is NumPy's verbatim ValueError (`'x' is not a valid method. Use one of: dict_keys([...])`), checked where
  `_quantile` reads it.
- float16 / float32 `q`: `(n - 1) * q` rounds (float16(0.999) over 2053 elements gives 2050.0, the same `q` as float64
  2049.99609375; from 65,520 elements `n - 1` is inf). `.astype(np.intp)` is x86's (NaN, ±inf and anything outside int64
  become INT64_MIN), and `partition_prep_kth_array`'s check runs in NumPy's order with NumPy's text (`kth(=2052) out of
  bounds (2052)` for 'lower' over 2052 elements), skipped for a size-0 array. np.percentile divides `q / 100` in `q`'s
  dtype and checks the quotient's range.
- Integer `q` arrays compute in their integer dtype: wrapping arithmetic, NEP 50's conversion of `n` (`Python integer 300
  out of bounds for uint8`), weibull's clipped -1 stored into an unsigned index array (`Python integer -1 out of bounds
  for uint8`); 'linear' takes the element, a bool `q` computes in int64, a uint64 'linear' index cannot partition
  (`Partition index must be integer`). weibull / averaged_inverted_cdf / interpolated_inverted_cdf lerp with an int64
  weight (integer data gives int64).
- The lerp runs in `result_type(data, q)` for a `q` ARRAY (a float16 `q` over float32 data: float32; over integer data:
  float64). An `out=` of another float / complex dtype receives the t >= 0.5 branch computed in ITS dtype
  (`subtract(..., dtype=type(out.dtype))`), not the float64 answer cast. Large uint64 data widens through
  `Converts.ToDouble` / `Converts.ToSingle` (rounded once; .NET 8's implicit uint64 → double conversion rounds twice from
  2^63 on).

## `out=`, empty data, the nan forms

- `out=` follows NumPy's routes: the take (exact shape, the safe out-to-data cast check, `WRITEBACKIFCOPY base is
  read-only`, an unsafe store, the NaN slices' same_kind refusal), the lerp (the add's same_kind cast, broadcast, `output
  array is read-only`), the nan forms' `out[...] = result` (an array by the unsafe cast; a NumPy SCALAR by `PyArray_Pack`,
  `ReductionInitial.PackNumPyScalar`: only a SIGNED integer out converts with `int()`, so NaN, infinity and 400 into int8
  raise there, while an unsigned / bool / float out casts: NaN is 0 in uint8 and 2^63 in uint64), keepdims' out index.
- All-NaN data answers `np.full(q.shape, nan)` in the data's dtype, and an all-NaN FIRST slice makes the result the data's
  dtype. `np.nanquantile` takes the element for an integer `q` and reads an array `q` as a strong float64 weight. Empty data
  raises NumPy's IndexError (`index -1 ...`; `index 0 ...` for integer data).
- The nan forms along an axis return `np.apply_along_axis`'s object: a view of its C buffer with `q`'s axes moved first
  (F-ordered, OWNDATA false). An all-NaN slice is Python's `np.nan` (`0x7FF8...`), a plain form's NaN slice copies the
  data's own NaN. The nan forms' q-axis order over merged axes follows NumPy, which moves q's axes from the UNnormalized
  -1 (`UReduced.NumPyAxis`); over all-NaN data they write a positive NaN through views and the layout NumPy's broadcast
  store gives (`FillWithNumPyNaN`).

## `weights=`

`NDArray weights = null` on all of `np.quantile` / `np.percentile` / `np.nanquantile` / `np.nanpercentile` runs NumPy
2.4.2's weighted branch (`inverted_cdf` only) for every dtype, text included (`Statistics/QuantileWeighted.cs`):
- NumPy's check order and texts (the method, the axis with argname `axis`, `_weights_are_valid`, `np.any(weights < 0)`),
  `_ureduce` merging data AND weights, argsort + gathers + `cumsum / last` + the -1 rule + searchsorted-left, NaN slices,
  the nan forms' `_nanquantile_1d` per position (NumPy 2.4.2's own IndexError for reduced-shape weights over a kept axis
  longer than 1, reproduced).
- Routes: one line never gathers the whole sorted data (the last element and the picks go through the argsort) and, for
  a float64 search, divides only the cumulative sums its bisection probes (`SearchLazily`; float16 / float32 `q` cast the
  cdf first, `SearchEagerly`); rows count the cdf entries below each `q` in one vectorized pass, or search row by row past
  `CountingMaxWork`. The thread-static hooks `ForceRowByRowTestHook` / `ForceEagerSearchTestHook` and two oracle replays
  keep both alternate routes gated.
- The computation runs in a plain region (`NDMatrixBoundary.PlainFor`): a chararray `a` keeps its class in the plain
  forms and gets the plain text's `isnan` refusal in the nan forms; chararray WEIGHTS refuse at `weights < 0` with
  compare_chararrays' `comparison of non-string arrays`.
- Known divergence: TIED numbers add their weights in NumPy's unstable argsort order (introsort / x86-simd-sort,
  host-dependent); NumSharp's argsort is stable; text is exact. The float64 flat cells are bound by the argsort.

## The partition

`Utilities/QuickSelect.cs`, shared with np.partition / np.argpartition / np.median, takes NumPy's `introselect_` shortcuts:
dumb_select for `kth - low < 3`, and the float32 / float64 maximum pass for `kth == n - 1` (`npy_half` is a `uint16_t`,
so float16 takes the quickselect, as in NumPy). Both are vector passes that keep NumPy's choice among ties (the first
minimum, the last maximum: the zero sign and the argpartition index). A window of up to 16 elements is insertion-sorted
before them, and the kernel's int entry sorts a line of up to 16 whole. float16 compares through `HalfOrderKey` (the
magnitude, negated for a negative value: IEEE `<` without branches, checked over all 63,490 non-NaN halves). The nan path
plans each NaN-free length once per call (`planLength` / `planCount`, locals of the emitted kernel). TRAP: on .NET 9+
`Vector.Min` / `Vector.Max` of floats is the IEEE 754-2019 minimum, a compare-and-blend sequence (chained through one
accumulator it read 100,000 doubles in ~110 µs; `Vector.MinNative` with four accumulators takes ~20 µs).

## Gates

The oracle sections `qfamily` (the weighted branch included) and `qprec` (the arithmetic in `q`'s dtype), zero excuses;
the soak's `quant` family with weights; unit `Statistics/np.quantile_family.{Weights,Precision,Prologue,QKinds}.Tests.cs`,
`np.nanquantile_family.BattleTests.cs`, `np.percentile.BattleTests.cs`, `np.median.BattleTests.cs`, and two partition tie
tests.
