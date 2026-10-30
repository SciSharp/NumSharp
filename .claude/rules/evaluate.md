---
paths:
  - "src/NumSharp.Core/Backends/Iterators/NDExpr*.cs"
  - "src/NumSharp.Core/Backends/Default/Math/DefaultEngine.Evaluate*.cs"
  - "test/NumSharp.Tests/**/NDEvaluate*.cs"
  - "benchmark/fusion/**"
---

# `np.evaluate`: fused expressions (NumSharp extension)

`np.evaluate(expr[, operands][, out])` compiles an `NDExpr` tree to one NDIter pass: every elementwise node runs inside a
single inner-loop kernel, so chained expressions allocate no intermediates and read each operand once (numexpr's
`evaluate` role). Benchmarks: the `benchmark/fusion` subsystem of `benchmark/run_benchmark.py`.

```csharp
NDArray r = np.evaluate((NDExpr)a * b + 2);                          // fused a*b+2
NDArray d = np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b)); // a, b dedup → 3 operand streams
NDArray s = np.evaluate(NDExpr.Sum((NDExpr)a * b), @out: x);        // one-pass sum(a*b), no temp
```

Sources: `Backends/Iterators/NDExpr.cs` (nodes and emission), `NDExpr.Typing.cs` (the per-node result_type pass),
`NDExpr.Evaluate.cs` (array leaves, binding, reductions, operators, `NDExprReduceKind`, `WeightedAverageNode`, the
factories and their `keepdims` overloads), `NDExpr.{Program,Structure,Params}.cs` (programs, the structural cache,
parameters), and the host `Backends/Default/Math/DefaultEngine.Evaluate*.cs` (`EvaluateReduce`, `EvaluateAxisReduce`,
`EvaluateDelegatingReduce`, `EvaluateStatReduce`, `EvaluateWeightedAverage`, `KeepdimsFlat`, `FlatReduceShape`,
`ValidateFlatReduceOut`). Semantics are pinned in `NDEvaluateTests.cs`.

## Typing

Dtypes follow NumPy's result_type per node (`NDExpr.Typing.cs`): NEP 50 strong-strong including the int/float tier
crossing (`i4+f4 → f8`); weak Python-scalar literals (`i4+2 → i4`, `f2+2.5 → f2`, `bool+2 → i64`, out of range →
OverflowError); `true_divide` of ints → f64; `arctan2` tier floats; `power`/`remainder`/`floor_divide` bool → i8; the
unary math tiers (bool/i8 → f16, i16 → f32, i32+ → f64); comparisons → bool. `(i4*i4)+f8` wraps the multiply in int32
before promoting, as NumPy's unfused sequence does.

## Reductions (root only)

- `NDExpr.Sum/Prod/Min/Max/Mean(expr)` with NumPy's reduce dtypes; empty: sum = 0, prod = 1, mean = NaN, min/max raise.
- Flat float32/float64/complex `Sum`/`Mean`, float `Prod`, and their axis forms over a C-contiguous child run NumPy's own
  schedule (the pairwise `np.add.reduce`, the sequential `np.multiply.reduce`). Over contiguous inputs in one shared
  order the host streams that schedule: it drives the pairwise recursion itself, each ≤ 8 KB leaf evaluated by the fused
  child kernel into L1 scratch and folded by the same cached fold, so no intermediate is allocated and the result is
  bit-identical; strided/broadcast/mixed-order inputs are materialized first. All-F operands stream as the C reduction of
  axis `nd-1-k` over the reversed dims (`TryStreamAxisReduce`), NumPy's memory-order schedule for the F child it
  materializes. float16 at any axis and complex `Prod` (the npy_cmul FMA gap) run the 4-accumulator fold
  (`MisalignedRegistry` E1).
- `Any`/`All` (logical OR/AND; identities False/True, so an empty input gives False/True) and
  `CountNonzero`/`NanSum`/`NanProd` (the factories rewrite `Sum(x != 0)`, `Sum(Where(IsNaN(x), 0, x))`,
  `Prod(Where(IsNaN(x), 1, x))`).
- `Ptp`/`NanMin`/`NanMax` and `ArgMax`/`ArgMin` delegate (`EvaluateDelegatingReduce`: the child is materialized once,
  then `np.ptp`/`np.nanmin`/`np.nanmax`/`np.argmax`/`np.argmin` run over it), inheriting NumPy's type rules (an int child
  → plain min/max; `Ptp` keeps the dtype, so integer overflow wraps; `NanMin`/`NanMax` skip NaN, all-NaN → NaN; `Ptp`
  propagates NaN). `NanMin`/`NanMax` are bit-exact (`TryExactNanMinMaxEval` runs NumPy's fmax/fmin schedules over the
  buffer NumPy reduces); `Ptp`'s bits over a permuted computed child are not pinned. `ArgMax`/`ArgMin` return int64 and
  follow the C-order first-tie / first-NaN rule; an empty child raises.
- `NanMean`/`Var`/`Std` are host-computed (`EvaluateStatReduce`) over the materialized child, NumPy's `nanmean` / `_var`
  op for op over the same pairwise/axis sums: `NanMean = pairwise_sum(NaN → 0) / count_of_non_NaN`,
  `Var = pairwise_sum((x − mean)²) / max(N − ddof, 0)` with `mean = pairwise_sum(x) / N`, `Std = sqrt(Var)`. `ddof` is
  part of the program identity (hash, `StructureEquals`, string key). `Var`/`Std` of complex128 yield a real float64
  (`|x − mean|²` computed as `re² + im²`, not an FMA). FLAT folds the child buffer in memory order (bit-exact for a C or F
  child); AXIS rides the engine's C-contiguous axis add.reduce. float16 and decimal result dtypes are refused
  (`NotSupportedException`: NumPy accumulates a float16 variance in float16).
- The weighted `Average` = `Σ(values·weights) / Σ(weights)` (`NDExpr.Average(values, weights[, axis, keepdims])`) has its
  own root node, `WeightedAverageNode`. Both children are cast to one result dtype and reduced with the pairwise sum
  (`ExactSumArray`); both sums stream (`TryStreamWeightedAverage` over `AvgNumeratorProgram`/`AvgDenominatorProgram`, no
  temporaries). The result dtype is keyed on the VALUES tree (int/bool → `result_type(values, weights, float64)`, else
  `result_type(values, weights)`; always float or complex, so an int·int product is float64). A zero total weight (an
  empty input included) raises `DivideByZeroException("Weights sum to zero, can't be normalized")`; a NaN weight gives
  NaN. float16 and decimal results are refused.
- `axis=None` + `keepdims` exists for every reduction kind: each flat factory has a `keepdims` overload (a REQUIRED
  `bool`, so `Sum(x)` stays 0-d and `Sum(x, keepdims: true)` is `(1,)*ndim`), reshaping the 0-d result (a 0-d child stays
  0-d; a direct `out=` is validated for the keepdims shape).
- A tuple `axis` is not supported (every NumSharp reduction is single-axis).
- Flat `Any`/`All`/`CountNonzero` stream rather than fold (`DefaultEngine.TryStreamBoolFold`): the bool child is
  evaluated block by block into L1 scratch and folded with vectorized span scans (`IndexOfAnyExcept`/`IndexOf`/`Count`);
  `Any`/`All` stop at the first deciding block. The factories' `x != 0` over a Boolean `x` is typed by NEP 50 as an int64
  compare (a scalar kernel), so `NDExprProgram.NonzeroBoolOperandProgram` streams `x` itself through its SIMD kernel; a
  bare bool leaf is scanned in place (`Address + offset`). The axis forms stream too (`DefaultEngine.TryStreamAxisBoolFold`,
  all-C or all-F operands; an F walk is the C walk of the reversed dims with an F-contiguous result): a contiguous reduced
  axis folds whole rows with the span scans; otherwise each output's slab is one contiguous run, combined into an
  accumulator row a vector at a time (`nz = ~(v == 0) & 1`; `Any`/`All` stop once every output of the current outer index
  is decided; the count runs in byte counters flushed into the int64 result every 255 slabs, the result row cleared
  first because the allocation is uninitialized). Other kinds, an empty axis/output and strided/broadcast/mixed-order
  operands decline to the seeded fold.
- Flat `Min`/`Max` are NumPy's own schedule, bit for bit (`DefaultEngine.Evaluate.MinMax.cs`, `NumPyMinMaxReduce`): a
  lane-exact port of `simd_reduce_c_{max,min}` (AVX2 build: `splat(x[0])`, 8-vector `maxn` groups, a canonical +NaN
  horizontal step, an SSE scalar tail; the second operand wins a tie), so the sign of a ±0 result and a NaN's payload
  match NumPy. It is shaped as `Vector256<T>` on every host (the lanes are the contract), the float lanes NumPy's
  `vmaxp`+`vcmpordp`+`vblendvp` under `Avx.IsSupported`. Routes: a dense bare leaf in place (memory order), a streamed
  computed child, a materialized non-streamable float child (in the layout NumPy's own ufunc would allocate); a non-dense
  bare leaf runs NumPy's iterator schedule (the engine's flat walker for `np.max`); only a broadcast leaf keeps the fold.
  The axis forms are exact too: a bare leaf reduces in place through `ReduceAxis`, a computed child over all-C/all-F
  operands streams (`DefaultEngine.Evaluate.MinMaxAxis.cs`), anything else is materialized in NumPy's layout first.
  Corpus block C9 (signed zeros kept) gates it. `MisalignedRegistry`'s generic unary ~ULP excuse does not apply to
  evaluate trees.
- Flat reductions over a shared dense axis permutation (a transposed block) stream in memory order
  (`CanStreamChild(allowPermuted)` / `IsSharedDensePermutation`, NumPy's K-order coalescing), and a flat `Mean` of an
  integer/bool child is NumPy's buffered-cast float64 sum (a pairwise sum per `np.getbufsize()` chunk; `np.setbufsize`
  changes the last bits in NumPy too).
- `ArgMax`/`ArgMin` stream over a fast engine axis argmax (`DefaultEngine.Evaluate.ArgStream.cs`, `TryStreamArgReduce`,
  tried first in `EvaluateDelegatingReduce`): a bare leaf goes straight to the engine argmax (every layout, no copy); a
  computed child over all-C operands (`CanStreamChild(allowF: false)`: only there is memory index k the logical C index
  NumPy's argmax visits) is produced block by block into L1 scratch. ROWS (the reduced axis innermost, or flat): the flat
  SIMD row kernel per row; a row longer than a block is taken chunk by chunk, the chunk winners combined in increasing
  order by the same strict rule (a later tie never replaces; stop once decided). SLABS (an outer axis): the engine's
  `SeedSlabChunk`/`FoldSlabRows` fed with the produced rows. F/strided/broadcast/mixed children, Half/Decimal/Complex,
  empty children and a bad axis decline to the materialize route (NumPy's errors kept). Hooks
  `NDExpr.DisableStreamingReduce` / `StreamingReductions`; gates `NDEvaluateStreamingTests` `ArgReduce_*`,
  `AxisArgFastPathTests`.

## Execution

- Repeated NDArray references deduplicate to one iterator operand; `out=` follows the ufunc rules; `ExecuteExpression`
  (Tier 3C) throws without EXTERNAL_LOOP (a per-element walk would be about 40× slower); `np.evaluate` configures the
  iterator itself.
- Trivially iterable calls skip NDIter (`DefaultEngine.Evaluate.Trivial.cs`, a port of NumPy's
  `try_trivial_single_output_loop`): identical dims on every non-0-d operand, each 1-D (any stride) or contiguous in one
  shared C/F order, no cast / `dtype=` / `where=`, and a provided `out` overlapping an input only exactly → the fused
  kernel runs once, the same call NDIter makes after coalescing (byte-identical). Everything else declines (never
  throws) to the NDIter pass. Hooks `NDExpr.DisableTrivialLoop` / `TrivialLoopRuns`; gate `NDEvaluateTrivialLoopTests`.

## Programs, parameters and the cache

- Compiled once per STRUCTURE, not per call (`NDExprProgram` + `NDExprProgramCache`): binding, the NEP 50 typing pass,
  the vector plan and the kernel lookup run once per (tree structure, input dtype signature, 0-d mask) and are cached
  twice: on the root instance (a hoisted or `Compile()`d tree: zero walks) and in a process-wide structural cache keyed
  by a 64-bit order-sensitive structural hash and VERIFIED node by node against the cached bound tree before use (a
  hash-only hit would run the wrong kernel; one entry per hash; a 4096-entry cap with a wholesale clear when full; the
  kernels stay in the kernel cache). So the natural spelling `np.evaluate((NDExpr)a * b + c)` rebuilt inside a loop is
  cheap, and the library's consumers (`np.sinc`, the windows, `trapezoid`, `gradient`) spell their trees inline.
- A 0-d input is a hoisted PARAMETER, not a stride-0 operand (`NDExpr.Params.cs`): its single element is copied into the
  kernel's aux block once per call and loaded into a local by the kernel prologue (the scalar, or its broadcast vector /
  lane mask in the SIMD body), so it costs nothing per element and keeps one kernel per structure (`np.hanning(M)` over
  many M compiles no new kernel). Typing is unchanged (a 0-d array is strong: `int32 * 0-d int64 → int64`, where a C#
  literal is weak). The value is read before the pass (a parameter aliasing `out` keeps its original value, as NumPy's
  COPY_IF_OVERLAP gives it). When every input is 0-d nothing is hoisted (a 0-d result, the iterator path). The mask is
  part of the program identity and re-validated per call (`ndarray.resize` of a 0-d array re-resolves; a `Compile()`d
  handle refuses with a clear message); a dtype-pinned positional `Compile(types)` handle knows no shapes and streams
  everything.
- Prefer a 0-d array over a literal for any scalar that varies at run time: a literal is baked into the IL and compiles
  one kernel per value (a JIT each and a kernel-cache entry each; the structural cache clears at 4096).
- `expr.Compile()` / `expr.Compile(params NPTypeCode[] inputTypes)` → `CompiledExpression`, the explicit handle
  (numexpr's `NumExpr(expr)`): eager JIT, `Evaluate(out)` for embedded arrays / `Evaluate(operands, out)` for positional
  trees, `IsPositional`/`IsReduction`/`InputTypes`/`ResultType`. A positional handle PINS its dtype signature (a
  `TypeError` naming both signatures on a mismatch, never a silent recompile). It is not the Tier-3C
  `Compile(inputTypes, outputType, cacheKey)`, which emits a raw kernel computed at one dtype.
- The inputs' broadcast is one N-ary single-allocation pass (`BroadcastInputDims`) whose mismatch lists every operand
  shape like NumPy (`operands could not be broadcast together with shapes (2,3) (2,3) (4,) `, as
  `IncorrectShapeException`); `out=` joins through `ResolveUfuncIterationShape`.
- `Call` nodes: the delegate/target SLOT is part of the kernel identity (two closures over one lambda body get separate
  kernels), and `DelegateSlots` deduplicates registration by delegate identity (target reference + method) / target
  reference, so a delegate held in a field and rebuilt into the tree per call keeps one slot and one kernel. A closure
  allocated per call is a new identity (a new slot and a new kernel JIT): hold delegates in fields.
- Gates: `NDEvaluateProgramTests`, `NDEvaluateParamTests`, the `evaluate.jsonl` tier (its `pp_scalar_*` / `scalar_0d`
  layouts drive the parameter path bit-exact), `windows.jsonl`.
