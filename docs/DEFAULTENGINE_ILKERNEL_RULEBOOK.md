# DefaultEngine + ILKernelGenerator Rulebook

This document captures the implicit implementation rules currently used across `DefaultEngine` and `ILKernelGenerator`.

Scope:
- `src/NumSharp.Core/Backends/Default/*`
- `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator*.cs`
- `src/NumSharp.Core/View/Shape*.cs`

## 1) Ownership and call boundaries

- `ILKernelGenerator` is backend infrastructure; access should flow through `TensorEngine` / `DefaultEngine`, not directly from top-level APIs.
  - See: `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.cs` (class summary and architecture comments).
- `DefaultEngine` owns high-level semantics (dtype rules, shape/broadcast behavior, keepdims, edge cases); kernels own tight loops.

## 2) Standard dispatch pipeline (elementwise ops)

For binary/unary/comparison operations, the repeated flow is:

1. Resolve dtype semantics first.
2. Handle scalar/scalar fast path.
3. Broadcast or normalize shapes.
4. Allocate contiguous output shape (`Shape.Clean()` / fresh `Shape` from dims).
5. Classify execution path (contiguous / scalar-broadcast / chunk / general).
6. Build kernel key.
7. Get-or-generate kernel from cache.
8. Execute kernel with pointer + strides + shape.
9. Use fallback path or throw explicit `NotSupportedException` if kernel unavailable.

Primary references:
- `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.BinaryOp.cs`
- `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.UnaryOp.cs`
- `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.CompareOp.cs`

## 3) Dtype rules are explicit and front-loaded

- Binary ops use `np._FindCommonType(lhs, rhs)` as the baseline promotion.
  - `DefaultEngine.BinaryOp.cs`
- True division on non-float common types is forced to `float64` (`NPTypeCode.Double`).
  - `DefaultEngine.BinaryOp.cs`
- Unary math promotion goes through `ResolveUnaryReturnType` / `GetComputingType`, while selected ops intentionally preserve input type (`Negate`, `Abs`, `LogicalNot`).
  - `DefaultEngine.UnaryOp.cs`
  - `DefaultEngine.ResolveUnaryReturnType.cs`
- Reductions use accumulator type decisions up front (for example `GetAccumulatingType`, std/var double output path in axis kernels).
  - `DefaultEngine.ReductionOp.cs`
  - `Default.Reduction.Var.cs`
  - `Default.Reduction.Std.cs`

## 4) Shape/offset correctness is non-negotiable

- Kernel inputs must include shape-offset-adjusted base pointers for sliced views:
  - `base = Address + shape.offset * dtypesize`
  - `DefaultEngine.BinaryOp.cs`
  - `DefaultEngine.UnaryOp.cs`
  - `DefaultEngine.ReductionOp.cs`
- Output arrays are usually allocated as contiguous clean shapes.
- Broadcast semantics rely on stride-0 dimensions and read-only protection at shape-level flags.
  - `src/NumSharp.Core/View/Shape.cs`
  - `src/NumSharp.Core/View/Shape.Broadcasting.cs`

## 5) Execution-path model

The core path taxonomy is:
- `SimdFull`: fully contiguous
- `SimdScalarRight` / `SimdScalarLeft`: one operand broadcast scalar
- `SimdChunk`: inner dimension contiguous/broadcast
- `General`: arbitrary strides

References:
- `src/NumSharp.Core/Backends/Kernels/StrideDetector.cs`
- `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.BinaryOp.cs`
- `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.MixedType.cs`

Important current caveat:
- `MixedType` `SimdChunk` currently emits the general loop (`TODO` placeholder), not true chunked SIMD.
  - `ILKernelGenerator.MixedType.cs`
- Comparison `SimdChunk` intentionally falls through to general path.
  - `ILKernelGenerator.Comparison.cs`

## 6) Kernel-key and cache conventions

- Kernels are cached by keys that encode everything affecting generated IL (types, op, path, contiguity).
- Caches are `ConcurrentDictionary<key, delegate>`.
- Standard retrieval API: `Get*Kernel` and `TryGet*Kernel`.
- `TryGet*` methods are intentionally catch-all and return `null` to allow graceful fallback.

References:
- `ILKernelGenerator.cs` (exception-handling design notes)
- `ILKernelGenerator.MixedType.cs`
- `ILKernelGenerator.Unary.cs`
- `ILKernelGenerator.Comparison.cs`
- `ILKernelGenerator.Reduction.cs`

## 7) SIMD policy and loop shape

- SIMD is only enabled for explicitly supported type/op combinations.
  - `CanUseSimd(NPTypeCode)` excludes `Boolean`, `Char`, `Decimal`.
  - `ILKernelGenerator.cs`
- Mixed-type SIMD requires additional constraints (often same-type for vectorized path or no per-element conversion).
  - `ILKernelGenerator.MixedType.cs`
- Typical contiguous loop form:
  - 4x unrolled SIMD block
  - remainder SIMD block
  - scalar tail
  - `ILKernelGenerator.Binary.cs`
  - `ILKernelGenerator.Unary.cs`
  - `ILKernelGenerator.Reduction.cs`

## 8) Scalar fast paths avoid boxing

- Scalar-scalar ops dispatch through typed delegates with exhaustive NPTypeCode switches.
- Pattern is nested type dispatch (lhs -> rhs -> result) rather than object/boxed conversion.

References:
- `DefaultEngine.BinaryOp.cs`
- `DefaultEngine.UnaryOp.cs`
- `DefaultEngine.CompareOp.cs`

## 9) Reduction-specific conventions

- Elementwise reductions:
  - empty input returns op identity (or op-specific behavior at higher level),
  - scalar short-circuit,
  - contiguous kernel path, strided fallback.
  - `DefaultEngine.ReductionOp.cs`
- Axis reductions:
  - output dims computed by removing axis,
  - SIMD path usually constrained to inner-contiguous axis for fast case,
  - keepdims reshapes handled at engine level after reduction.
  - `DefaultEngine.ReductionOp.cs`
- `var` / `std` axis kernels compute ddof=0 baseline, then apply ddof correction in engine.
  - `Default.Reduction.Var.cs`
  - `Default.Reduction.Std.cs`

## 10) NaN-aware behavior uses dedicated logic

- NaN reductions are float/double-specific; non-float types delegate to regular reductions.
- For contiguous float/double inputs, dedicated NaN SIMD helpers are used; scalar iterator fallback otherwise.
- keepdims reshaping is handled explicitly after scalar/elementwise NaN reductions.

Reference:
- `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Nan.cs`

## 11) General-path philosophy

- General path prioritizes correctness for non-contiguous, sliced, and broadcast layouts.
- Coordinate-based offset computation is acceptable when required by arbitrary strides.
- For complex cases (broadcast + views + type conversion), correctness path should remain available even when fast path exists.

Representative references:
- `ILKernelGenerator.MixedType.cs` (`EmitGeneralLoop`)
- `Default.ClipNDArray.cs` (contiguous fast path + general path split)
- `Default.Reduction.CumAdd.cs`
- `Default.Reduction.CumMul.cs`

## 12) Practical checklist for adding a new core operation

Before merge, verify all of the following:

- NumPy behavior matrix captured first (dtype promotion + edge cases).
- Scalar-scalar behavior implemented and tested.
- Contiguous fast path exists where meaningful.
- Non-contiguous and sliced views work (`shape.offset`, strides).
- Broadcast dimensions (stride=0) are handled correctly.
- Output shape/layout rules match NumPy behavior.
- All supported NumSharp dtypes are either implemented or explicitly rejected.
- Keepdims / axis / negative-axis behavior is explicitly tested.
- Empty-array behavior is explicit (identity / NaN / exception, as appropriate).
- Kernel key includes all generation-sensitive dimensions (types/op/path/flags).
- `TryGet*` fallback behavior is deterministic and test-covered.
- Tests use actual NumPy output as source of truth.

## 13) Current technical debt markers (worth tracking)

- True chunked SIMD emission for mixed-type `SimdChunk` path is not implemented yet.
- Comparison `SimdChunk` currently routes to general kernel.
- Some comments indicate ownership/history items (for example cache-clear ownership) that should be periodically validated against current code.
