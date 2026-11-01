---
paths:
  - "src/NumSharp.Core/Backends/Default/Math/Reduction/**"
  - "src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction*.cs"
  - "src/NumSharp.Core/Statistics/**"
  - "src/NumSharp.Core/Sorting_Searching_Counting/np.{max,min,amax,amin}*.cs"
  - "src/NumSharp.Core/Ma/NDMaskedArray.cs"
  - "test/NumSharp.Tests/Statistics/**"
  - "test/NumSharp.Tests/Math/np.{sum,mean,var,nanvar,nanmean}*.cs"
---

# Reductions

## The reduce prologue

Every `ufunc.reduce`-based reduction checks its arguments in NumPy's order before it reads an element
(`Backends/Default/Math/Reduction/ReductionArguments.cs`, ported from `PyUFunc_GenericReduction`,
`PyUFunc_ReduceWrapper` and `npyiter_fill_axisdata`, with NumPy's texts): a read-only `out=` first ("output array is
read-only"), then a non-bool `where=` ("Cannot cast array data|scalar from dtype(…) to dtype('bool') according to the
rule 'safe'"), the axis, a repeated axis, `out=`'s rank, then the reduction iterator's shape checks over
`[out, input, where]`: the input is never stretched by the mask, the mask may not have more axes than the input, a kept
reduced axis of `out=` must be 1 long, and an `out=` axis of length 1 cannot broadcast. `np.all`/`np.any` (when `out=`
or `where=` is given) and the max/min family run it. Gates: the `reduce_kwargs` tier (zero excuses) and
`Logic/np.all_any.tuple_out_where.Test.cs`.

## `np.sum` / `np.mean`: the add identity and NumPy's add.reduce

`Backends/Default/Math/Reduction/Default.Reduction.Add{,.Exact,.Identity}.cs`.
- The identity: NumPy's `PyUFunc_ReduceWrapper` assigns the loop's identity (`+0.0` for every inexact loop) and only
  then runs the add loop, for one element and a length-1 axis too, so a sum of only `-0.0` is `+0.0` and a signalling NaN
  comes back quiet. `SumOfOneElement` / `SumOverLengthOneAxis` add the lone element(s) to a 0-d identity with the binary
  add kernel; `amin`/`amax`/`nanmin`/`nanmax` (no identity) copy through `HandleScalarReduction` /
  `HandleTrivialAxisReduction`. Both run before the cast-loop branch (the identity add casts the element itself).
  Inherited by `np.mean` (count division; complex divides by `n+0j` with NumPy's complex loop), `np.average`,
  `np.nansum`/`np.nanprod` (one element: the fused NaN-skipping kernels, which start from the identity), `np.nanmean`,
  `np.trapezoid`, `np.einsum`'s sums and `np.evaluate`'s Sum / Mean / NanSum / NanMean.
- `dtype=` loops: a loop that is neither the input nor its NEP 50 accumulator casts every element first
  (`ReduceAddInCastLoop`: float→int truncates, complex→real keeps the real part, `dtype=bool` is logical OR). An AXIS cast
  loop takes the axis kernel's own float32→float64 widening only where that is NumPy's sequential order (the reduced axis
  not the innermost, `IsSequentialWideningAxisSum`), else casts in blocks of `CastLoopBlockElements` (128K) along the
  outermost kept axis. A FLAT integer / bool cast loop (order-free: modular addition, OR) casts and sums in blocks along
  the outermost axis in memory.
- The flat engine (`NumPyAddReduce`; float16 / float32 / float64 / complex128 loops, the input already the loop dtype or
  cast into it) follows NpyIter's call schedule: axes sorted by |stride| and coalesced signed
  (`NumPyMinMaxReduce.FlatIteratorAxes`), then `npyiter_find_buffering_setup`'s cost model (`FlatBufferingDim`; a cast
  operand starts at cost 2). One run is one call in place (any stride; a reversed run in logical order); several runs
  whose inner run exceeds half the buffer are one call each; otherwise the operand is buffered in fills of
  `core·⌊bufsize/core⌋` that never cross an outer block (a cast run: fills of bufsize). Every call is
  `*out += pairwise_sum(call)` (real: 8 accumulators per 128-element leaf, split `n/2` down to a multiple of 8; complex
  `CDOUBLE_pairwise_sum`, 4 accumulators per part; float16's HALF_add narrows the running float32 sum per call). Fills
  follow `np.getbufsize()`. A fill is streamed: ≤ 1024-element sub-ranges of its pairwise recursion, each folded in place
  when it lies inside one run and the fold reads that stride with SIMD loads (the real part of a complex128 array is a
  stride-2 float64 run), else copied/cast into an L1 block (the same bits, no 64 KB buffer).
- `np.mean` of the flat engine is `_mean`: int/bool → the float64 loop, float16 → the float32 loop with a float16 result
  (keepdims rounds twice, float64 → float32 → float16), the quotient by the count in float64; complex through Smith's
  division.
- The folds (`ILKernelGenerator.Reduction.Pairwise{,.Runs}.cs`, `.Half.cs`): run kinds contiguous, reversed (a load plus
  a lane reversal) and stride 2 (two overlapping loads plus blend/permute, never past the last element), a two-leaf
  interleave for nodes whose halves are both leaves, complex two per register at any stride, float16 runs and an AVX2
  gather; `PairwiseFoldIsVectorized` says which strides fold with SIMD.
- Reversed and broadcast operands: an inexact loop over an operand with a stride <= 0 on an axis longer than 1
  (`HasReversedOrBroadcastAxis`) that the exact flat engine and the one-output axis route decline (a broadcast operand
  flat; such an operand along an axis with several outputs; `out=` too) runs `ReduceAddThroughIterator`: the port of
  NumPy's reduction NpyIter (`NumPyReduceIterator`: `DONT_NEGATE_STRIDES`, the axis sort with a stride-0 operand
  abstaining, coalescing, a cast operand buffered in `np.getbufsize()` fills, the reduce double loop) with NumPy's add
  loops (`NumPyAddLoops`, `Default.Reduction.Add.Axes.cs`). So a reversed axis is added in its logical order and a
  broadcast operand is one add-loop call per broadcast row. The loops: the elementwise form vectorized for a contiguous
  input (`(a & (b == b)) + b`, the second operand's NaN), a broadcast value (NumPy's region rule: the broadcast value's NaN
  in the body of a call longer than two vectors, the array operand's in its tail) and a reversed one (loaded from below,
  reversed in the register); the several-rows form `BinaryRows`; `PairwiseOfCopies` (NumPy's pairwise sum of n copies of
  one value, a stride-0 run); HALF_add through `DirectILKernelGenerator.HalfArithRun` in place; a stride-0 buffer fill
  converts one element and replicates it. A decimal input and a cast without dynamic code keep their routes; integer /
  bool loops keep their kernels (exact in any order). Floor: float64 reversed rows along axis 0, memory-bound on both
  sides. Gate `Math/np.sum.ReversedBroadcast.Test.cs`.
- `out=` of another dtype (the engine's `ReduceAdd(..., out)`): the sum is computed in the loop dtype and cast into `out`
  (`DeliverReduction`). Residual: past one buffer fill NumPy carries the partial sum through out's dtype between fills;
  this route rounds into out once. Gate `Math/np.sum.OutDtype.Test.cs`.
- An EMPTY reduction without an identity raises NumPy's `ValueError` on every route (`zero-size array to reduction
  operation maximum which has no identity` from `np.max` / `np.min` / `np.nanmax` / `np.nanmin` / `np.ptp` and
  np.evaluate's `Min` / `Max`; `attempt to get argmax of an empty sequence` from `np.argmax` / `np.argmin`); `ValueError`
  derives from `ArgumentException`, so a handler for that still catches it.
- Open: a sum of SEVERAL NaNs keeps another NaN's payload (MSVC's `pairwise_sum` operand order; an `[OpenBugs]` pin; a
  complex128 short run chooses its NaN per part, group 8); the plain C-contiguous sum along an OUTER axis keeps a
  column's FIRST NaN where NumPy's elementwise add keeps the later row's (the axis kernel's slab fold, `[OpenBugs]` group
  15); the reduce step `*acc += pairwise_sum(call)` (`AccumulateDouble` / `AccumulateSingle`) leaves the NaN of NaN + NaN
  to the JIT, so a Debug build keeps the pairwise sum's NaN where NumPy and a Release build keep the accumulator's (NL8,
  `OpenBugs.NpLayer.cs`); the multi-element nan-reductions keep their own folds (the W4-B excuse); an axis reduction in a
  cast loop whose reduced runs exceed the buffer splits differently from NumPy
  (`OpenBugsReductionTests.Sum_CastLoopAxis_RowLongerThanTheBuffer_SplitsAcrossFills`; `np.var`/`np.std` with `dtype=`
  inherit it), except when the axis leaves ONE output (a 1-D operand's only axis, or siblings all of extent 1), which
  NumPy runs as the flat schedule and so does NumSharp (`TryExactOneOutputAxisSum`), a one-output reversed axis included.
- Gates: `sum_identity.jsonl` (zero excuses; section "s" holds order-sensitive flat schedules under `np.setbufsize`
  16 / 32, section "o" one-output axis schedules), `reduce_start.jsonl` (zero excuses),
  `Math/np.sum.{Identity,FlatSchedule,CastLoopBlocks}.Test.cs`, `Backends/Kernels/{PairwiseFoldRunKind,CopyRunSubword}Tests.cs`.

## `np.var` / `np.std`: NumPy's `_var` / `_std`

`Backends/Default/Math/Reduction/Default.Reduction.VarStd.cs`, `VarStdNumPy`, statement for statement, reusing the
engine's exact pieces:
- `rcount` after the axis is normalized (an axis on a 0-d array is NumPy's AxisError). The loop is `dtype=`, else float64
  for an integer / bool / Char input, else the input's dtype. `arrmean = umr_sum(keepdims=True)` through `ReduceAdd`
  (np.sum's exact schedules), divided by the count in place as NumPy does (`DivideByCountLikeNumPy`: float16 / float32 in
  float64, complex by Smith's division, an integer loop truncates). `x = arr - arrmean` in NpyIter's layout
  (`DeviationLayout`: `Shape.NpyIterOutputShape` over `arr` and the mean's K-order keepdims layout). The square in
  NumPy's three forms (`x*x`; `real² + imag²` of a complex `x` from a complex or bool input; `(x*conj(x)).real`).
  `ret = ReduceAdd(x², axis, dtype) / max(rcount - ddof, 0)`: NaN or `+inf`, never negative. std is `sqrt(out=ret)`: an
  integer loop raises NumPy's `Cannot cast ufunc 'sqrt' output from dtype('float64') to dtype('int32') with casting rule
  'same_kind'` for an array result and truncates a scalar one.
- No x² buffer where NumPy's order allows it: a flat reduction over one memory run with no cast streams
  `Sum(Square(arr - mean))` through `np.evaluate` (the streamed pairwise sum, bit-identical). Otherwise one `np.evaluate`
  pass fills the squared deviations (`SquaredDeviationExpr`), or the eager ufuncs do for float16 and decimal. Above
  `VarStdChunkMinBytes` (8 MiB of x²) the deviations are computed and summed in chunks, each route byte-identical to the
  one-buffer route: output chunks along the outermost kept axis (`ChunkedDeviationSum`); row blocks with a running
  `np.add(acc, slab, out: acc)` per slab when the reduced axis is outermost (`RowBlockDeviationSum`, NumPy's slab order);
  fills of `np.getbufsize()` folded by `NumPyAddReduce.TryReduceFlat` for a flat cast loop (`FlatFillDeviationSum`).
  `DefaultEngine.VarStdChunkBytesTestOverride` forces them; the oracle's `VarStdChunked` replays the tier through them.
- Decimal: exact decimal arithmetic in np.sum's order (sequential, memory order) and `DecimalMath.Sqrt`; a zero divisor
  raises `DivideByZeroException` (decimal has no NaN).
- The moment kernels `DirectILKernelGenerator.Reduction.Axis.VarStd.cs` and `.Masking.VarStd.cs` have no caller in
  var/std.
- Gates: `varstd.jsonl` (zero excuses: layouts × dtypes × axis × keepdims × ddof, `ddof >= n`, `dtype=` over ten loops,
  special values, `np.setbufsize` fills, order-sensitive sizes) with `VarStdChunked`, `decimal_varstd.jsonl`,
  `Math/np.var.NumPyPort.Test.cs`.

## `np.mean`: NumPy's `_mean`

`Backends/Default/Math/Reduction/Default.Reduction.Mean.cs`, `MeanLikeNumPy`.
- `rcount` is an `np.intp` (a strong int64); np.mean normalizes the axis itself, so a 0-d input refuses every axis. The
  loop is `dtype=`, else float64 for bool/integers, float32 for float16 with a float16 result. `ret = ReduceAdd(arr, axis,
  keepdims, loop)` (np.sum's exact schedules: a cast loop casts each element, an integer loop wraps, the bool loop is OR).
  The division in place: the engine's dense-block division where it is NumPy's quotient (float64, complex by Smith's
  division, float32 while `n <= 2^24`), else `DivideByCountLikeNumPy` (the float64 loop and an unsafe cast back: an empty
  slice's `0 / 0` is INT_MIN, or True). A float16 result rounds twice for an array (float64 → float32 sum → float16) and
  once for a scalar, NumPy's two branches.
- Routes: every float/complex loop along an axis but a length-1 one (its identity add is exact) and every integer/bool
  loop take `MeanLikeNumPy`; the flat float loops take `TryExactFlatMean`. A flat operand that engine declines (a broadcast
  one) keeps the elementwise fold only where it is NumPy's value: the float64 loop over a narrow integer / bool / char
  input whose sum `size × max|x|` stays within 2^53 (int64/uint64 inputs, larger sums, other loops and decimal take
  `MeanLikeNumPy`; the fold costs no memory on a huge broadcast). A broadcast float16 / float32 / float64 / complex128
  operand takes `MeanLikeNumPy` with the `dtype=` loop, its sum np.sum's iterator route.
- np.sum's integer/bool axis routes: an integer → float64 axis sum runs the widening kernel when the sum is NumPy's in
  every order (32 bits or less within the 2^53 bound) or the reduced axis is not the innermost (int64/uint64: sequential
  per output, NumPy's slab order), `IsExactIntegerWideningAxisSum`; a bool axis sum (int64 or float64 loop) counts
  nonzero elements with np.evaluate's CountNonzero axis stream for C/F operands (`TryCountBoolAxisSum`).
- Open: np.sum's cast-loop rows above; the flags of an empty input's mean (over an empty (0, 9, 9) along axis 0 NumPy's
  result is F-ordered, NumSharp's C: the empty-strides gap). The oracle holds np.mean strict in every tier
  (`ByteExactArithmeticUnaryOps`, out of the reduction blankets; `ma.mean` too), without exception.
- Gates: `mean_loops.jsonl` (zero excuses), `Math/np.mean.NumPyPort.Test.cs`, the tightness tests
  `Mean_NoDivergenceIsExcused` / `Mean_ReversedAndBroadcastLayouts_AreNotExcused`.

## `np.any` / `np.all` / `np.count_nonzero` along an axis

Laid out like NumPy's reduction: NumPy allocates the result with `npyiter_new_temp_array` over the reduce iterator, so an
F input gives an F result and a permuted one keeps its memory order. The logical reductions allocate with
`Shape.ReductionOutputShape` (`DefaultEngine.LogicalReductionShape`), `NDAxisIter.ReduceBool` writes each output through
the result's own strides, keepdims re-inserts the axis with `ExpandDimension`, and the tuple-axis chain drops its kept
axes with their strides. `np.count_nonzero(a, axis)` is NumPy's own `a.astype(bool, copy=False).sum(axis, dtype=intp)`;
np.sum's bool count route (`TryCountBoolAxisSum`) returns an owning result (never np.evaluate's reshaped keepdims view).
A bool C/F operand of at least 16K elements reduced to rank >= 1 runs np.evaluate's Any / All axis stream
(`TryStreamLogicalAxis`; long contiguous rows of 4096 or more stay on the engine's per-row scan). Floor: the "read every
element" F-slab cells at 1M stream from L2 on both sides. Gate: `Logic/np.all_any.ResultLayout.Test.cs` (probed strides
for bool / int32 / U inputs, F, permuted, stepped, tuple axes) and the string oracle's `strcomp` section.

## `np.average` / `np.average_returned`: NumPy's statements

`TensorEngine.Average` (`TensorEngine.FunctionBase.cs`), whose `DefaultEngine` override holds NumPy's
`numpy/lib/_function_base_impl.py` statements (`Backends/Default/Statistics/Default.Average.cs`); the facades keep the
numpy.matrix boundary and the null check.
- `_weights_are_valid`: weights of another shape raise `TypeError("Axis must be specified when shapes of a and weights
  differ.")` without an axis and `ValueError("Shape of weights must be consistent with shape of a along specified
  axis.")` along one; weights along the axes are transposed and reshaped into the operand's frame. `result_dtype` is
  `np.result_type` over the descriptors (`'f8'` joins for an integer / bool operand), so text weights meet NumPy's
  refusals in the weight sum. Without weights `avg = a.mean(axis)` and the count is `avg.dtype.type(a.size / avg.size)`:
  NumPy's scalar beside a 0-d mean, the broadcast copy beside one with dimensions. With weights `scl = wgt.sum(axis,
  dtype=result_dtype)`, the zero check (`DivideByZeroException` "Weights sum to zero, can't be normalized"),
  `np.multiply(a, wgt, dtype=result_dtype).sum(axis) / scl` (`SumOfProducts`, `Quotient`), and `returned` broadcasts and
  copies `scl` when its shape differs.
- Axis tuples: one axis is np.mean's / np.sum's axis route (tested first, so a 1-D operand's `(0,)` takes the one-output
  axis route), every axis their flat route, any other set (two axes or more, or the empty tuple) ONE reduction over the
  set: `DefaultEngine.ReduceAddOverAxes` / `MeanOverAxes` (`Default.Reduction.Add.Axes.cs`) run `NumPyReduceIterator`
  with `NumPyAddLoops`, the result owning its data in the iterator's layout.
- The products' sum streams where that is exact: with the same dimensions and one memory order (both C- or both
  F-contiguous), a float32 / float64 / complex128 result and one axis or every axis, `np.evaluate(Sum(Cast(a) * Cast(w),
  axis))` replaces the product buffer (the same products rounded in the result dtype, added in np.sum's order).
  Everything else forms the product in NumPy's ufunc layout (`Shape.NpyIterOutputShape`) and sums it. The fused
  weighted-sum kernel (`DirectILKernelGenerator.WeightedSum.cs`, public API shipped in 0.70.0) is no longer called: it
  added both sums in its own SIMD order, a few ULP off NumPy's pairwise sums.
- Residuals: a decimal operand with NaN weights raises np.sum's `OverflowException` (NaN into decimal); an unweighted
  complex128 mean over several NaNs keeps np.sum's NaN per part, not NumPy's (group 8); broadcast weights form the
  product.
- Gates: `Statistics/np.average.NumPyPort.Test.cs`, `Statistics/np.average.TupleAxes.Test.cs`,
  `Backends/Iterators/NDIterEmptyIterationTests.cs` (an empty iteration runs no inner loop); the oracle holds average
  strict (`ByteExactArithmeticUnaryOps`, `Average_NoDivergenceIsExcused`).

## `np.ma.mean` / `ma.var` / `ma.std` / `ma.anom`: numpy.ma's statements

`src/NumSharp.Core/Ma/NDMaskedArray.cs`.
- Without a mask: ndarray's `_mean` / `_var` run on the masked array, so their `true_divide(out=...)` meets
  `MaskedArray.__array_wrap__`'s divide domain (`ArrayWrapDivideDomainFill`: a quotient with `|q| * tiny >= count`
  becomes 1.0, masked; a masked 0-d result is `masked`). `mean=` passes to `_var` as given: a C# number is weak there.
- With a mask (an all-False one too): the mean is `dsum * 1. / cnt`, the sum of numpy.ma's K-order `filled(0)`
  (`FilledForReduction`; the data itself under an all-False mask), times a strong float64 one, divided through
  ma.true_divide (non-finite and domain quotients masked, the dividend put back); a 0-d sum is scalar arithmetic, a 0-d
  count of zero `masked`, a float16 result `np.float16(...)`. The variance subtracts through ma.subtract (a C# number is
  strong through `getdata`), squares in place (complex: `abs(danom) ** 2` through ma.power, a non-finite square masked),
  divides by `count - ddof`, and replaces the mask with `mask_or(mask.all(axis), cnt <= 0)` (shrunk to nomask); a flat
  `count - ddof < 0` is a plain negative scalar. std is ma.sqrt of the variance and ignores `mean=`. anom is
  `self - mean`, the mean expanded only for a nonzero axis.
- Layout: every intermediate a later sum walks is allocated in NumPy's ufunc layout (`UfuncResultBuffer`,
  `Shape.NpyIterOutputShape`) and written in place, because NumSharp's eager ufuncs write C order for a permuted input,
  which adds in another order. np.power's result is computed by the plain call and copied.
- Gates: `ma_meanvar.jsonl` (zero excuses, valueclass `mameanvar`: raw data at every slot and the result's form —
  singleton / scalar / nomask / mask array — beside `filled(0)` and `getmaskarray`; NumPy's errors; no carve: the
  reversed and broadcast layouts are in the gate); `ma_reduce` holds the four strict; `Ma/np.ma.meanvar.NumPyPort.Test.cs`.
  The rest of numpy.ma: `masked-arrays.md`.

## `np.max` / `np.min` / `np.ptp`: NumPy's reduction schedules

`Backends/Default/Math/Reduction/Default.Reduction.MinMax.Exact.cs`, `NumPyMinMaxReduce`, hooked first in
`Default.Reduction.AMax/AMin` (the IL kernels run only on a decline). For floats the ORDER is observable (which zero sign
survives a ±0 tie; whether a NaN comes back canonical after a lane reduce or with its payload), so the schedules are
NumPy's:
- Axis: NumPy copies each output's first reduced element, then walks a K-order `NpyIter` with `DONT_NEGATE_STRIDES`; the
  innermost axis is the smallest-|stride| axis of extent > 1 (a tie → the later axis). Reduced axis innermost → ROW mode
  (a contiguous row is `simd_reduce_c` after the copied element, a strided row the scalar 8-accumulator unroll);
  otherwise SLAB mode (the first slab copied, then `o = N(o, x_k)` per reduced index; contiguous slab runs fold eight
  indices per pass through `CombineRun8`, evaluated as the balanced tree `N(o, N(N(N(x0,x1),N(x2,x3)),N(N(x4,x5),N(x6,x7))))`,
  bit-identical because `N` is associative).
- Flat: the 0-d output never votes, so axes coalesce signed into runs; one run = one call; several runs →
  `npyiter_find_buffering_setup`'s cost model (its `size >= maximum_size` stop included): an innermost run longer than
  half the buffer → per-run calls, else the input is buffered (copied in iteration order into fills of
  `coresize·⌊bufsize/coresize⌋`, never across the outer block's end), each fill one contiguous `simd_reduce_c` call, so
  fill boundaries decide ties and re-canonicalize a running NaN. The fill follows `np.getbufsize()`.
- `N(a, b) = isnan(a) ? a : (a > b ? a : b)` is associative, so the strided unroll is "the latest of x[0], lane 0…7, the
  tail wins a tie", and its eight accumulators run as VECTOR lanes (a stride of -1 is one contiguous load plus a lane
  reverse). A NaN-free fast fold (plain `vmaxp` on groups proven NaN-free by `vcmpunordp`, switching to NumPy's blend at
  the first NaN group) keeps the contiguous path fast.
- Declines: broadcast operands and complex128 go to the keyword path's reduction iterator (NumPy's bits);
  Decimal/Bool/Char use the IL kernels (value-exact; bool along its innermost axis → the keyword path's early-exit
  `logical_or` loop); `dtype=` casts; rank > 64. float16 has its own route (below). Hooks `NDExpr.DisableExactMinMax` /
  `ExactMinMaxRuns` (shared with np.evaluate).
- Open: `np.max(x * 1, axis)` over a permuted input differs, because NumSharp's eager elementwise writes C order where
  NumPy's ufunc keeps the K-order layout.
- Gates: `Backends/Kernels/MinMaxExactScheduleTests.cs` (NumPy-probed literals, including the every-lane-pair tie test and
  a ±0 tie / NaN pair on every node of the slab tree) and three NumPy replay oracles.

## max / min with `out=`, `initial=`, `where=`

`np.max`/`np.min`/`np.amax`/`np.amin`/`np.nanmax`/`np.nanmin`/`ndarray.max`/`ndarray.min`
(`Sorting_Searching_Counting/np.max.Keywords.cs`, `Statistics/NDArray.max.Keywords.cs`, engine
`DefaultEngine.MaxMinNumPy` in `Default.Reduction.MinMax.Keywords.cs`). Each is a ufunc reduction (`np.maximum.reduce`;
np.nanmax of an ndarray is `np.fmax.reduce`), so the keywords are `PyUFunc_ReduceWrapper`'s, in its order:
- the reduce prologue; the loop dtype `promote_types(out, a)`;
- `initial=` packed into the loop by a port of `PyArray_Pack` (`ReductionInitial.cs`): a C# integer literal is a Python
  int (range-checked with the LP64 texts, never wrapped), a `double` a Python float truncated into an integer loop,
  `Half`/`char` NumPy scalars `int()`-converted by a signed integer loop and cast by every other, a 0-d array cast, a str
  parsed, a sequence refused; without it a mask raises "... does not have an identity, so to use a where mask one has to
  specify 'initial'" and an empty axis "zero-size array to reduction operation ...";
- the result starts as the initial value or each output's first element (an unsafe cast).
- ONE overload per function: NumPy's `(a, axis, out, keepdims, initial, where)` and `ndarray.max/min(axis, out,
  keepdims, initial, where)`, nothing else (no positional `(a, axis, keepdims)` shim, no `dtype=` extension, no generic
  `np.amax<T>` / `a.max<T>()`, no `NDArray.amax` / `NDArray.amin`: they resolved only through
  `[OverloadResolutionPriority]`, which C# 12 ignores). A positional third argument is `out`, as in NumPy:
  `np.max(a, 1, true)` raises NumPy's `TypeError: return arrays must be of ArrayType` (the reduce prologue reads
  `NDArray.WasScalarArgument`, the implicit scalar conversions' mark), so keepdims goes by name. A typed reduction,
  NumPy's `np.maximum.reduce(a, dtype=...)`, is the engine's `TensorEngine.ReduceAMax/ReduceAMin(a, axis, keepdims,
  dtype)`. A keyword-less call keeps the engine route above. The C# literal `true` for `where=` is NumPy's `where=True`
  (no mask), and so is a 0-d bool ARRAY holding true, where NumPy treats `np.array(True)` as a mask needing `initial=`
  (`[Misaligned]`).
- Four routes: (1) a broadcast input whose answer cannot depend on how often an element recurs reduces one copy
  (`MaxMinBroadcastShortcut`: every loop but float32/float64; those only when every NumPy call is elementwise, since each
  reduce-form `simd_reduce_c` call re-canonicalizes a NaN); (2) an integer input under a wider loop reduces in its own
  dtype, then casts (`max` commutes with a monotone cast); (3) same dtype, no `where=` → the exact engines above, seeded
  with the initial; (4) everything else → `NumPyReduceIterator` (`Backends/Default/Math/Reduction/NumPyReduceIterator.cs`),
  a port of NumPy's reduction NpyIter: the axis sort (C wins a tie), coalescing, `npyiter_find_buffering_setup`, fills,
  the reduce double loop, the first-visit skip, masked runs (`generic_masked_strided_loop`), and a cast out's round trip
  through its own dtype at every fill unless the buffer is reused (`NPY_OP_ITFLAG_BUF_REUSABLE`: only from the reduce
  double loop's own iternext with the out's pointer unchanged, which is why `np.max(int16 [300, 100…], out=int8)` is 100
  past 8,192 elements).
- Its loops (`NumPyMinMaxLoops`): the SIMD rules for float32/float64/integers, `logical_or`/`logical_and` for bool
  (early exit), the sequential BINARY_LOOPs for float16/char/decimal, and complex128 vectorized (each rule is an
  associative SELECTION, so a 1024-element block folds in two vector passes: the extreme value and whether a NaN occurs,
  then the first element equal to it / the first NaN). Beyond NumPy's two loop forms: a WHOLE masked call (order-free
  loops fold the selected elements in one pass; the float SIMD loops make NumPy's per-run calls but compose each run
  through a closed form of `simd_reduce_c`, `NRunContiguous`/`PRunContiguous`) and SEVERAL ROWS per call where the out
  does not move (`Loop.BinaryRows`).
- A fresh axis result of every reduction that allocates through `AllocateReductionResult` (sum / mean / var / std / the
  nan family / min / max) is laid out like NumPy's (`Shape.ReductionOutputShape`, `npyiter_new_temp_array`: the input's
  axis order, not C order).
- Gates: `minmax_kwargs.jsonl` (zero excuses), `Selection/np.max.Keywords.Test.cs`.

## float16 max / min / nanmax / nanmin: NumPy's sequential HALF loops

`Backends/Default/Math/Reduction/Default.Reduction.MinMax.Half.cs`, `DefaultEngine.HalfMinMaxReduce`, taken before any
other route, flat and axis, every layout including broadcast. NumPy has no SIMD float16 compare loop: `HALF_maximum` is
`(ge(acc,x) || isnan(acc)) ? acc : x` (fmax: `|| isnan(x)`), a SELECTION, so the answer is the FIRST NaN verbatim
(payload, sign, a signalling NaN unquieted), else the first extreme element (fmax: NaN skipped, an all-NaN slice → its
first element), a ±0 tie keeping the earlier zero, "first" in NumPy's visiting order: each axis in increasing logical
index (`DONT_NEGATE_STRIDES`: a reversed run is visited from its high address down), a flat reduction in K order
(`NumPySequentialReduce.FlatVisitOrder`).
- The selection is associative, so the walks run SIMD and stitch run winners in order. FLAT drops broadcast axes,
  coalesces runs, sends ±1-stride runs ≥ 32 to the bit-level cores (`HalfRunWinnerBits`; reversed runs rescan from the
  top) and folds the rest by a branch-light PRIORITY KEY (`HalfPriorityKey`: the order key with ±0 merged, inverted for
  min, NaN → 0xFFFF or 0, so "take iff strictly greater" is the fold step for all four rules). AXIS picks ROW (a SIMD core
  per ±1-stride row, a branch-free masked fold for rows ≤ 16, the out-of-line branchy fold otherwise) or SLAB (the
  elementwise pair-select over 4096-lane blocks, a stepped inner run gathered first).
- np.evaluate's flat `Max`/`Min` over a float16 child use the same walks (`TryExactFlatMinMaxHalf`: a leaf in place, a
  computed child in NumPy's layout); the `dtype=float16` flat path casts first; the `(Half, Half)` axis kernel key is an
  adapter onto the walk.
- `np.ptp` (NumPy's `subtract(maximum.reduce, minimum.reduce)`) inherits the walks: its NaN is the same first NaN,
  quieted by HALF_subtract's float32 arithmetic.
- Gates: `Math/HalfMinMaxLayoutTests.cs` (NumPy-probed bit literals), the raw-byte `half_minmax` tier, the float16
  SELECTION CONTRACT (`FuzzCorpusTests.Kinds.HalfSelectionContractOps`: these four ops on a float16 operand are compared by
  raw bytes in every tier, and any difference is a hard failure with no registry excuse), and the float16 ptp contract
  (the same, on x86 hosts only: `inf - inf` makes the CPU's default NaN, negative on x86 and positive on arm64).

## `np.cov`

`cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None)` is a pure composition over
`average`/`dot`/`concatenate`/`atleast_2d`/`conjugate`/`squeeze`, a line-for-line port of NumPy's
`numpy/lib/_function_base_impl.cov` (`TensorEngine.Cov`, run by `Backends/Default/Statistics/Default.Cov.cs`; the facade
`Statistics/np.cov.cs`).
- `np.corrcoef(x, y=None, rowvar=True, dtype=None)` is NumPy 2.x's thin wrapper over cov (`TensorEngine.Corrcoef`,
  `Backends/Default/Statistics/Default.Corrcoef.cs`): `c` divided in place by `sqrt(diag(c).real)` along the rows and then
  the columns, then its real lane (and the imaginary one of a complex result) clipped to [-1, 1] in place with bounds in
  the lane's dtype; one variable (cov's 0-d result, which has no diagonal) gives NumPy's `c / c`. No `bias` / `ddof`
  (removed in NumPy 2.x).
- `m`/`y` are 1-D or 2-D (more → `ValueError`); `rowvar=False` transposes the variable/observation relationship; `ddof`
  overrides `bias` (defaults: `bias` → N, else N-1). The result dtype is `result_type(m[, y], float64)` (int/bool →
  float64, complex128 stays complex) unless `dtype` is given, and it is not force-cast at the end, so `dtype=float32`
  with any weights still yields float64 (the float64 weights promote `avg` → `dot`), as in NumPy.
- `fweights` (integer frequencies) / `aweights` (relative) combine to `w = fw*aw`; the `fact = w_sum - ddof*sum(w*aw)/w_sum`
  normalization, the `fact <= 0 → +inf` path, and the empty `(0,0)` float64 / single-observation `nan` / scalar `nan`
  edges are NumPy's.
- Errors: `ValueError` (`m has more than 2 dimensions`, `{f,a}weights cannot be negative`), `TypeError`
  (`fweights must be integer`), `RuntimeError` (`cannot handle multidimensional {f,a}weights`, `incompatible numbers of
  samples and {f,a}weights`); weights summing to zero raise `DivideByZeroException` (NumPy's `ZeroDivisionError`).
- Two NumPy details ported on purpose: `X -= avg[:, None]` runs in the operand's dtype (so `dtype=float32` without
  weights keeps float32), and for complex input `fact` is complex128, so NumPy's `.../w_sum` is Smith's
  `numer*(1/w_sum)`, reproduced as a reciprocal multiply (it rounds 1 ULP off real division and decides finite versus inf
  when `fact ~ 0`).
- cov is GEMM-bound: byte-identical to NumPy with `NumSharp.Interop.OpenBLAS`, within managed-GEMM tolerance without it.
  `TryGramCov`, the managed Gram (syrk) path for ≤ 16 variables, runs only when `TensorEngine.Blas is null`: with a backend
  the product must take `np.dot`, as NumPy computes cov through BLAS.
- Gate: `Statistics/np.cov.BattleTests.cs`.

## NaN-aware reductions

### `np.nanmax` / `np.nanmin`: `np.fmax.reduce` / `np.fmin.reduce`

The ±0 sign of a tie and which NaN an all-NaN slice returns (payload intact, never canonical) match NumPy 2.4.2 on every
non-broadcast float layout (np.evaluate's `NanMax`/`NanMin` too).
- fmax's loop has two per-element operations (the "P rules", header of `Default.Reduction.MinMax.Exact.cs`): the VECTOR
  op `npyv_maxp` (a tie keeps the later operand, two NaNs keep the FIRST) inside each inner-loop call's vector section,
  and the MSVC CRT `fmax` (a tie → bitwise AND, i.e. `+0`; fmin → OR, `-0`; two NaNs → the LAST) for the call's scalar
  tail and strided rows; `npyv_reduce_maxp` returns lane 0 raw when every lane is NaN. So the answer depends on where
  NumPy's calls cut the reduction: `SlabCallTiling.Build` ports that cut (`npyiter_find_buffering_setup`,
  `np.getbufsize()` honoured; a float32 in-place negative-stride input runs every call scalar), and the flat/row paths
  reuse `NumPyMinMaxReduce` with `TLane.PropagatesNaN == false`.
- float16 and complex128 run NumPy's sequential BINARY_LOOPs (`HALF_fmax`/`CDOUBLE_fmax`) in NpyIter's visiting order:
  NaN skipped (complex: a NaN in either part), the earlier of a tie kept, complex lexicographic, an all-NaN slice → its
  first NaN verbatim. float16 uses the bit-level SIMD walks above; complex128 the scalar fold of
  `Default.Reduction.Nan.Sequential.cs`. Integers, bool, char and decimal are `np.amax`/`np.amin`.
- An empty reduction raises `zero-size array to reduction operation fmax/fmin which has no identity`.
- The P rules' vector section is ONE order-free fold (`NumPyMinMaxReduce.FoldVectorsP`, `DefaultEngine.Evaluate.MinMax.cs`):
  `maxp` is an associative selection ("the LAST largest non-NaN, else the FIRST NaN"), so NumPy's group tree plus its
  single-vector loop equal, per lane, a BACKWARD scan with the plain `vmaxp(data, suffix)` from `∓inf` (the suffix never
  NaN, a NaN data lane skipped, a tie keeping the later element), read through the 32-byte-aligned vectors covering
  NumPy's `x[1:]` section, with one `vpermd` rotating positions back to NumPy's lanes and a vectorized rescan of the rare
  NaN-accumulator lane the fill cannot decide. It is used only for stretches of at least `FoldVectorsPMinElements` = 768
  elements (its fixed cost is paid per ROW), so shorter stretches keep the literal tree. `Finish` dispatches to two
  separate methods (`FinishTree`, `FinishFoldP`), because tier-1 PGO compiles one body from its early profile and a
  shared body lays the tree's loops out cold when long reductions run first.
- complex128 axis and non-C cells use the scalar fold; a broadcast float32/float64 input runs the keyword path's
  reduction iterator (NumPy's bits).
- Gates: `Backends/Kernels/NanMinMaxExactScheduleTests.cs` (NumPy-probed literals), NumPy replay oracles, and the
  `nanreduce`/`specials` tiers, where nanmax/nanmin are not excused (`MisalignedRegistry.NanReduceOps` leaves them out).

### `np.nanvar` / `np.nanstd`: NumPy's `nanvar` / `nanstd`

`Backends/Default/Math/Reduction/Default.Reduction.NanVarStd.cs`, `NanVarStdNumPy`; `np.nanvar` / `np.nanstd` take
`dtype=`. NumPy's `numpy/lib/_nanfunctions_impl.py`, statement for statement:
- An input that cannot hold NaN (integer, bool, char and NumSharp's decimal) is `np.var` / `np.std`. For float16 /
  float32 / float64 / complex128: a non-inexact `dtype=` raises "If a is inexact, then dtype must be inexact" before the
  axis is read; then the axis follows ufunc.reduce's rule, because nanvar's first axis consumer is `np.sum` (on a 0-d
  input axis 0 / -1 reduce flat, where np.var raises).
- Then: the non-NaN count (int64, keepdims); the mean of the NaN-zeroed K-order copy (`Shape.KeepOrder()`) divided by it
  (the float64 loop, Smith's division for complex); the deviations rounded into the INPUT's dtype
  (`subtract(out=arr, casting='unsafe')`: with `dtype=float64` a float32 input still squares float32 deviations), zeroed at
  the NaNs, squared (complex: the real part of NumPy's `simd_cmul` `x * conj(x)`, one rounding of `re*re + round(im*im)`,
  not np.var's `re² + im²`), summed, divided by `cnt - ddof`, and `np.nan` (the positive quiet NaN) wherever
  `cnt - ddof <= 0`. np.var's `+inf` at `ddof == n` is NaN here, and an inf element is NaN (`inf - inf`).
- Engines: both sums run over NumPy's K-order copy. A dense input without a cast (one memory run flat, C/F along an axis)
  already is that copy: the mean's sum and the count stream over the input, and a flat variance streams the squared
  deviations too (three reads, no buffer). Otherwise the copy is materialized for `ReduceAdd`, and one fused
  `np.evaluate` pass writes the squared deviations into a K-order buffer. float16 and hosts without dynamic code run
  NumPy's eager statements. Because the sums never walk the view, np.sum's view divergences do not reach nanvar.
- Open: np.sum's cast-loop rows longer than the buffer reach `dtype=`; `out=` / `where=` / `mean=` / `correction=` and
  NumPy's parameter order are not implemented.
- Gates: `nanvarstd.jsonl` (zero excuses), the `nanreduce` / `specials` / `reduce_start` tiers (nanvar/nanstd out of the
  nan-family blankets and in `ByteExactArithmeticUnaryOps`), `Math/np.nanvar.NumPyPort.Test.cs`.

### `np.nanmean`: NumPy's `nanmean`

`Backends/Default/Math/Reduction/Default.Reduction.NanMean.cs`, `NanMeanNumPy`; `np.nanmean` takes `dtype=`. An input
that is not inexact (and decimal) is `np.mean` with the same keywords. An inexact input: the dtype TypeError,
ufunc.reduce's axis rule (a 0-d input takes 0 / -1 as flat), the non-NaN count (`keepdims=keepdims`), the sum of the
NaN-zeroed K-order copy in its loop, and the in-place division (`DivideByCountsLikeNumPy`: the float64 loop, Smith's
complex division; an all-NaN slice keeps `0 / 0`). Routes as nanvar's; the zero fill (`ZeroWhereMask`, shared with
nanvar) is NumPy's `np.copyto(arr, 0, where=mask)`. Gates: `nanmean.jsonl` (zero excuses; nanmean is out of
`NanReduceOps` and in `ByteExactArithmeticUnaryOps`), `Math/np.nanmean.NumPyPort.Test.cs`.
