# NumSharp Release Notes — `nditer` branch

## TL;DR

This release lands a **full NumPy `nditer` port** (`NpyIter`), a **composable expression DSL** (`NpyExpr`) with a three-tier custom-op API, **multi-order memory layout** (C/F/A/K) wired through the whole API surface, **stride-native matmul** for all 12 dtypes (eliminates a 100× slowdown on transposed inputs), a new **`Char8`** dtype (1-byte NumPy `S1` equivalent with 100% Python `bytes` parity), and a complete **trainable MNIST MLP example** that fuses bias+activation passes into single iterator invocations.

- **+50,426 / −1,188 lines across 156 files**
- **6,710 tests passing** on net8.0 + net10.0 (zero regressions)
- **566/566 NumPy 2.4.2 nditer parity scenarios** verified byte-for-byte
- **MLP training: 100s → 3s** (5 epochs) and ultimately **1ms per `np.dot` on transposed views** (down from 240ms)

---

## Headline Features

### 1. `NpyIter` — full NumPy `nditer` port

A from-scratch C# port of NumPy 2.4.2's `nditer` machinery, located under `src/NumSharp.Core/Backends/Iterators/`. Implements virtually the entire NumPy nditer surface (32+ APIs) with byte-for-byte semantic parity.

| Capability | Notes |
|---|---|
| Iteration orders | C, F, A, K (with NEGPERM for negative-stride memory-order traversal) |
| Indexing modes | `MULTI_INDEX`, `C_INDEX`, `F_INDEX`, `RANGE` (parallel chunking) |
| Buffering | Type conversion during buffered iteration; full casting rules (`no`/`equiv`/`safe`/`same_kind`/`unsafe`) |
| Reduction | `op_axes` with `-1` reduction axes; `REDUCE_OK`, `IsFirstVisit`; **buffered-reduce double-loop** including `bufferSize < coreSize` |
| Multi-operand | **Unlimited operands** (NumPy's `NPY_MAXARGS=64` parity, dynamic allocation) |
| Dimensions | **Unlimited dimensions** (NumSharp divergence; replaces NumPy's fixed `NPY_MAXDIMS=64`) |
| Masking | `WRITEMASKED` + `ARRAYMASK` with reduction safety check |
| APIs ported | `Copy`, `GotoIndex`, `GotoMultiIndex`, `RemoveAxis`, `RemoveMultiIndex`, `ResetBasePointers`, `GetMultiIndexFunc`, `GetInnerFixedStrideArray`, `GetAxisStrideArray`, `CreateCompatibleStrides`, `DebugPrint`, `GetIterView`, `IterRange`, `Iternext`, `GetValue<T>`/`SetValue<T>`, `Finished`, `Shape`, `OVERLAP_ASSUME_ELEMENTWISE`, `TRANSFERFLAGS`, reduction-axis encoding (`axis + (1<<30)`), and more |
| Battletest | 491-scenario random fuzz (seed 42) + 75 structured scenarios — all match NumPy 2.4.2 |

### 2. `NpyExpr` DSL + three-tier custom-op API

User-extensible kernel layer on top of `NpyIter`, with three tiers of escalating control:

- **Tier 3A — `ExecuteRawIL(body, key, aux)`**: raw IL against the NumPy ufunc signature.
- **Tier 3B — `ExecuteElementWise(scalar, vector, ...)`**: per-element IL + 4×-unrolled SIMD shell with scalar tail and strided fallback.
- **Tier 3C — `ExecuteExpression(expr, inputTypes, outputType)`**: compose `NpyExpr` trees, no IL exposure, auto-derived cache key.

**DSL coverage:** `Add Sub Mul Div Mod Power FloorDiv ATan2`, bitwise (`& | ^ ~`), unary math (`Abs Sign Sqrt Cbrt Square Reciprocal Floor Ceil Round Truncate Exp Exp2 Expm1 Log Log2 Log10 Log1p Sin Cos Tan Sinh Cosh Tanh ASin ACos ATan Deg2Rad Rad2Deg`), predicates (`IsNaN IsFinite IsInf LogicalNot`), comparisons (`Equal NotEqual Less Greater LessEqual GreaterEqual`), combinators (`Min Max Clamp Where`), plus full operator overloads (`+ - * / % & | ^ ~ !`).

**`Call(...)` escape hatch (commit `8da3e693`)**: invoke any `Func<...>`, `Delegate`, or `MethodInfo` per element — fuses arbitrary .NET methods into the surrounding expression with auto-conversion at the call boundary. Three dispatch paths (static / bound-instance / captured-delegate) chosen at construction; static calls are zero-indirection (JIT-inlinable).

**Bugs caught and fixed during DSL battletest:**
- Predicate ops (`IsNaN`/`IsFinite`/`IsInf`) silently wrote I4 0/1 into double slots (denormals instead of 1.0)
- `LogicalNot` broken for Int64/Single/Double/Decimal (`Ldc_I4_0+Ceq` only valid for I4 operands)
- `WhereNode` prelude was unfinished (threw at compile time)
- `MinMaxNode` didn't propagate NaN — rerouted through `Math.Min/Max` (matches `np.minimum`)
- `Vector256.Round/Truncate` are .NET 9+ only — excluded from SIMD path on net8.0

### 3. Multi-order memory layout (C / F / A / K)

NumSharp now correctly tracks and preserves Fortran-contiguous (column-major) layout throughout the API:

- **`Shape`** — added `IsFContiguous` (O(1) flag check), `ComputeFContiguousStrides`, `Shape(dims, char order)` ctor; aligned contiguity computation with NumPy's `_UpdateContiguousFlags` (single-pass `(isC, isF)` tuple); **fixed empty-array semantics** (any `dim==0` is both C- and F-contig per NumPy).
- **`OrderResolver`** — centralizes C/F/A/K → C/F mapping.
- **API surface wiring** — `np.copy`, `np.array`, `np.asarray`, `np.asanyarray`, `np.asfortranarray` (new), `np.ascontiguousarray` (new), `*_like` (`empty_like`/`zeros_like`/`ones_like`/`full_like`), `astype`, `flatten`, `ravel`, `reshape`, `eye`, `concatenate`, `vstack`, `hstack`, `cumsum`, `argsort` all accept and respect `order=`.
- **Post-hoc F-contig preservation across ILKernel dispatch** — instead of refactoring 27 partial files (~21K lines) of IL emitters to accept arbitrary output strides, a cheap `.copy('F')` relays results to F-contig at the central dispatchers (`ExecuteBinaryOp`, `ExecuteUnaryOp`, `ExecuteComparisonOp`) when every non-scalar operand is F-contig. Fixes 41 element-wise layout bugs.
- **`np.modf`, `np.clip`, `np.negative`, `np.maximum/minimum`** — updated for F-contig preservation.

**51 sections of TDD coverage** added in `OrderSupport.OpenBugs.Tests.cs` (3,005 lines), each driven by side-by-side Python/NumPy 2.4.2 output. Remaining `[OpenBugs]` are minimal API gaps (`np.tile`, `np.flip`, `np.where`, `np.sort`).

### 4. Stride-native GEMM for matmul (perf)

`np.dot` / `np.matmul` previously fell into a ~100× slower fallback whenever an operand was non-contiguous (transposed view, slice, etc.). This release ships **stride-native paths for all 12 dtypes**:

- **`SimdMatMul.Strided.cs`** — generalized 8×16 Vector256 FMA micro-kernel for `float`; new packers (`PackAPanelsStrided`, `PackBPanelsStrided`) absorb arbitrary strides with fast paths for transposed-contig and row-contig.
- **`SimdMatMul.Double.cs`** — stride-aware IKJ Vector256<double> kernel (4 FMAs).
- **`Default.MatMul.Strided.cs`** — `MatMulStridedSame<T> where T : INumber<T>` (JIT specializes per type with auto-vectorization), plus `MatMulStridedBool`, `MatMulStridedMixed<TResult>`. Replaces the old `GetValue(coords)`-based mixed-type path (no more boxing in the inner loop).
- **Dead code removed**: `MatMulGeneric`, `MatMulCore`, `MatMulSameType`, four `MatMulContiguous` overloads, `MatMulMixedType` — ~165 lines.

**Measured impact (MLP backward shapes):**

| Op | Before | After |
|---|---|---|
| `dot(x.T, grad)` 784×64 @ 64×128 | 240 ms | **1 ms** |
| `dot(grad, W.T)` 64×128 @ 128×784 | 226 ms | **1 ms** |
| Lt(400,500) @ L(500,400) blocked | 12 ms | **8 ms** (skips copy) |

28 new `MatMulStridedTests` cover all 4 BLAS transpose cases × float/double, per-dtype stride-native (byte/int16/uint16/int32/uint32/int64/uint64/char/decimal/bool), sliced views with `Shape.offset > 0`, mixed-type, and the exact MLP shapes.

### 5. Trainable MNIST MLP example

`examples/NeuralNetwork.NumSharp/MnistMlp/` — a runnable end-to-end classifier demonstrating fusion:

- **Architecture**: 784 → 128 (ReLU) → 10, float32, He-init, Adam optimizer.
- **Forward fusion**: post-matmul `bias + ReLU` collapses into one `NpyIter` per layer (`NpyExpr.Max(Input(0) + Input(1), 0)`).
- **Backward fusion**: `gradOut * (y > 0)` ReLU mask fused in one iter.
- **Loss**: `SoftmaxCrossEntropy` (combined, numerically stable, max-subtracted).
- **Trainer**: `MlpTrainer.cs` with periodic test eval (every `min(5, epochs)` epochs).

**Results** (6000 train / 1000 test, batch 128, Adam lr=1e-3):

| Phase | Total time | Final test acc |
|---|---|---|
| Pre-stride-native dot | 100.7 s (5 epochs) | 100% |
| Post-`copy()` workaround | 3.2 s (5 epochs) | 100% |
| 100-epoch demo | ~42 s | **99.89%** |

**NN scaffolding fixes** (`examples/NeuralNetwork.NumSharp/`): `Softmax` had empty `Forward` and a wrong (sigmoid-derivative) `Backward`; `Sigmoid.Forward` was empty; `CategoricalCrossentropy` had no clipping and the wrong backward formula; `BinaryCrossEntropy` didn't divide by N to match its mean reduction; `Accuracy` collapsed both `argmax` calls to a scalar (no axis); `BinaryAccuacy` returned null; `FullyConnected` had no bias and used `np.random.normal(0.5, 1, ...)` (skewed mean, wrong dtype); `NeuralNet.Train` used 2-index integer selection where slicing was intended (silently trained on a single element); Adam optimizer's `ms`/`vs` init was commented out (KeyNotFoundException on first step); `SGD` optimizer didn't exist. All fixed and verified against analytical references with finite-difference grad checks (29/29 pass).

### 6. `Char8` — 1-byte NumPy `S1` equivalent

New `NumSharp.Char8` type (`[StructLayout(Sequential, Size=1)]` readonly struct), the NumPy `dtype('S1')` / Python `bytes` of length 1 analogue. Five partial files (~1,450 lines): `Char8.cs` (core), `.Operators.cs` (mixed-type ops), `.Conversions.cs` (dtype interop), `.Spans.cs` (span primitives + UTF-8 classification), `.PyBytes.cs` (Python `bytes` array methods).

- Adapted from .NET `System.Char` (Latin1CharInfo table copied verbatim).
- Full Python `bytes` parity: `Strip`, `Split`, `SplitLines` (bytes-only — only `\n`/`\r`/`\r\n`), `Partition`, `Replace` (with empty-pattern handling), `Center` (CPython's odd-padding-on-the-left formula), `ZFill`, predicates (`IsDigits`/`IsAlphas`/etc.).
- `Converts.Char8.cs` (324 lines) — parallel to `Converts.Native.cs` for all 12 dtypes; throws on overflow/NaN per existing convention.
- `src/dotnet/` — fetched System.Char dependency tree (`Char.cs`, `Latin1Utility`, `Ascii.*`, `Rune`, `UnicodeUtility`, `HexConverter`, `Number.Parsing`, etc.) into a reference library. Indexed in `INDEX.md`.
- 250-line Python `bytes` oracle diff (identical) + 270+ C# edge assertions.
- **Standalone for now** — not yet wired into `NPTypeCode` enum (would touch ~50 switch statements; deferred).

### 7. Bug fixes (NPTypeCode + dispatch)

- **`NPTypeCode.Char.SizeOf()` returned 1, real is 2** (UTF-16). Affected `NpyIter.SetOpDType` (`ElementSizes[op]` × stride in 8 places), 8 cast sites, `np.frombuffer`, `np.dtype(char).itemsize`, axis reductions. Survived without test failures because NumPy has no native char dtype and ASCII reads accidentally land on the right byte.
- **`GetPriority(Decimal) = 5*10*32` was stale** after the prior Decimal SizeOf fix — corrected to `5*10*16=800` (no behavioral change; relative ordering preserved).
- **`DefaultEngine.IsInf` was stubbed to return null** (NRE on any `IsInf` call). Now wired through `ExecuteUnaryOp` with the existing IL kernel.
- **`NDArray.Copy.cs` share-by-reference bug** — `new Shape(this.Shape.dimensions, 'F')` aliased the source `int[]`; cloned now.
- **`NDArray.argsort`** — copies non-C-contig input to C-contig first (matches NumPy's invariant that argsort always produces C-contig output).

### 8. Documentation

- **`docs/website-src/docs/NDIter.md`** (1,934 lines) — comprehensive NpyIter reference: 7-technique quick reference, decision tree, full Tier C node catalog with NumPy-equivalent column, type discipline, SIMD coverage rules, caching/auto-keys, validation, gotchas, debugging, memory model + lifetime, 19 worked examples (Swish, GELU, Heaviside, Horner polynomial, fused sigmoid, NaN replacement, etc.).
- **`docs/website-src/docs/ndarray.md`** (537 lines) — NDArray reference: anatomy, creation helpers, indexing/slicing, views vs copies, operator quirks, dtype conversion, 0-d scalars, generic `NDArray<T>`, save/load, memory layout, equality, troubleshooting.
- **`docs/NPYITER_AUDIT.md`**, **`NPYITER_DEEP_AUDIT.md`**, **`NPYITER_NUMPY_DIFFERENCES.md`**, **`NPYITER_BUFFERED_REDUCE_ANALYSIS.md`** — implementation audit reports.
- Tier names renamed `A/B/C → 3A/3B/3C` to make the layer-3 sub-tier relationship explicit (100 references across 6 files).

---

## Behavioral Changes / Notes

| Area | Change | Migration |
|---|---|---|
| `np.copy` default order | `'C'` → `'K'` | No behavioral change for C-contig input (K preserves layout) |
| `MaxOperands=8` removed | Now unlimited (dynamic alloc) | Drop-in; `ManyOperands_Works` test added |
| `MaxDims=64` removed | Now unlimited (~300K dims, stackalloc-bound) | Drop-in |
| F-order iteration | Now produces `[0,3,1,4,2,5]` for 2×3 C-contig (was `[0,1,2,3,4,5]`) | Matches NumPy |
| K-order on broadcast / non-contig | Falls back to C-order (was stride-sort, broken with `stride=0`) | Matches NumPy |
| Negative strides | Only flipped for K-order (per NumPy's `FORCEDORDER` rule) | Matches NumPy |
| Empty arrays | `IsContiguous` and `IsFContiguous` both `true` (was both `false`) | Matches NumPy |
| `Shape.Order` | Now derives from contiguity flags (transpose of C reports `'F'`) | Was hardcoded to `'C'` |

---

## Test Suite

- **6,710 tests** pass on net8.0 + net10.0 (CI filter: `TestCategory!=OpenBugs&TestCategory!=HighMemory`); zero regressions.
- **+566 NumPy 2.4.2 nditer parity scenarios** (491 random fuzz, 75 structured) — element sequences, stride arrays, multi-indices, reduction outputs all byte-equivalent to Python NumPy.
- **+264 NpyExpr + custom-op tests** (`NpyIterCustomOpTests`, `NpyIterCustomOpEdgeCaseTests`, `NpyExprExtensiveTests`, `NpyExprCallTests`).
- **+94 nditer API parity tests** (`NpyIterAxisStrideArrayTests`, `NpyIterCreateCompatibleStridesTests`, etc.).
- **+28 `MatMulStridedTests`**.
- **+69 `Char8` cases** (source-generated discovery).
- **+150 OrderSupport TDD tests** across 51 sections.
- **+24 `Shape.Order.Tests`**.
