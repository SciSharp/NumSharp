# DefaultEngine and ILKernelGenerator Playbook

This document captures the implementation rules that are already implicit in the current `DefaultEngine`, `ILKernelGenerator`, and test suite.

It is not a NumPy spec. NumPy remains the source of truth for behavior. This is the "how we implement NumPy-compatible functionality in NumSharp" guide.

Representative source files:

- `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.BinaryOp.cs`
- `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.UnaryOp.cs`
- `src/NumSharp.Core/Backends/Default/Math/DefaultEngine.ReductionOp.cs`
- `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Add.cs`
- `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.MixedType.cs`
- `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Unary.cs`
- `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.Axis.cs`
- `src/NumSharp.Core/Backends/Kernels/StrideDetector.cs`

## Mental Model

A good `DefaultEngine` implementation has three layers:

1. Public override: thin API surface, almost no logic.
2. Dispatch helper: resolves NumPy semantics, shapes, dtypes, edge cases, and execution path.
3. Kernel/helper layer: contiguous SIMD fast path plus a correct general path for strided, sliced, and broadcast inputs.

The consistent pattern is:

- decide behavior first
- decide dtype first
- decide shape first
- only then optimize execution

Performance is never allowed to define semantics.

## Rules for Good DefaultEngine Functions

### 1. Keep public overrides thin

Most good overrides are one-line wrappers into a shared dispatcher:

- binary ops call `ExecuteBinaryOp`
- unary ops call `ExecuteUnaryOp`
- comparisons call `ExecuteComparisonOp`
- reductions call a dedicated reduction dispatcher

Examples:

- `Default.Add.cs`
- `Default.Sqrt.cs`
- `Default.Sum.cs`

If a method grows large, it usually means it needs a shared helper or it is genuinely a special-case operation such as `ATan2`, `ModF`, `ClipNDArray`, `Var`, or `Std`.

### 2. Resolve NumPy semantics before choosing a fast path

The dispatch layer should answer these questions before touching the kernel:

- What is the result dtype?
- What is the broadcasted shape?
- Does the operation have NumPy-specific promotion rules?
- What happens for empty arrays?
- What happens for scalars?
- What happens for negative axes?
- What happens for `keepdims`?
- What happens for `out`?

Examples already in the code:

- true division promotes integer inputs to `float64`
- `power` has custom promotion rules
- `argmax`/`argmin` always return `int64`
- reductions use accumulating or computing dtypes rather than ad hoc casts
- `ATan2` has bespoke output rules

### 3. Handle structural edge cases up front

The current engine repeatedly uses this order:

1. empty array
2. scalar
3. `axis == null` element-wise reduction
4. trivial axis cases such as `shape[axis] == 1`
5. general kernel path

This keeps the hot path simple and prevents subtle bugs in reshaping, aliasing, and identity handling.

Important examples:

- empty reductions do not all behave the same
- `min`/`max` on empty inputs can raise while `sum`/`prod` can return identities
- reducing an axis of size `1` must return an independent result, not a view into the source

The memory-independence rule is enforced by `AxisReductionMemoryTests`.

### 4. Treat non-contiguous, sliced, and broadcast arrays as first-class inputs

A function is not done when the contiguous case passes.

Good implementations always account for:

- `shape.offset` for sliced views
- non-unit strides for strided/transposed views
- stride `0` for broadcast dimensions
- read-only broadcast inputs

The common pattern is:

- compute the base address as `Address + shape.offset * elemSize`
- pass strides in element units to kernels
- use coordinate-based iteration for the general path

Do not assume `Address` already points at the logical first element of the view.

### 5. The result is usually a fresh contiguous array

Input layout affects execution strategy, not output layout.

The engine usually:

- broadcasts input shapes
- calls `Clean()` on the result shape
- allocates a new contiguous output array
- reshapes afterward for `keepdims`

This is simpler, faster, and avoids accidentally leaking view semantics into operations that NumPy materializes.

### 6. Use the 12-type outer switch, then move into generic code

The project convention is:

- outer `switch` on `NPTypeCode`
- then call a typed generic helper

This avoids reflection, avoids boxing, and makes unsupported cases explicit.

Do not hide dtype coverage inside weakly typed helper code unless the operation truly requires runtime conversion fallback.

### 7. Use the right dtype helper instead of inventing local promotion rules

The existing code already encodes policy:

- `_FindCommonType` for binary promotion
- `GetAccumulatingType()` for `sum`/`prod`/`cumsum`
- `GetComputingType()` for many unary math functions
- explicit op-specific overrides when NumPy requires them

If you find yourself sprinkling `Convert.ToDouble` everywhere, the design is probably drifting away from the engine conventions.

### 8. Normalize axes once

Axis normalization is centralized for a reason:

- negative axes are valid
- out-of-range axes must raise NumPy-style errors

Use `NormalizeAxis` and keep the rest of the function working with normalized non-negative axes.

### 9. Apply `keepdims` after computation when possible

The common pattern is:

- compute the reduced result using the natural reduced shape
- reshape the result afterward to inject size-`1` dimensions

This keeps kernels simpler and matches how many current reduction helpers are structured.

### 10. Only write bespoke engine logic when the generic dispatch model is not expressive enough

Special-case functions in the current codebase exist for real reasons:

- `ATan2` has unique type rules and scalar conversion behavior
- `ModF` returns two arrays
- `ClipNDArray` handles broadcasted array bounds and `out`
- `Var` and `Std` need two-pass statistics and `ddof`
- NaN-aware reductions need masking/counting behavior

The rule is not "avoid bespoke code". The rule is "do not bypass the shared dispatch structure unless the operation genuinely needs different semantics."

## Rules for Good ILKernelGenerator Kernels

### 1. Cache by the full behavioral key

Good kernel keys include every detail that changes generated code:

- input type
- output type or accumulator type
- operation
- execution path
- contiguity flag when relevant

This is why the code has separate keys such as:

- `MixedTypeKernelKey`
- `UnaryKernelKey`
- `ElementReductionKernelKey`
- `AxisReductionKernelKey`

### 2. `TryGet*Kernel` must fail safely

The generator is designed for graceful degradation:

- `Get*Kernel` is the strict path
- `TryGet*Kernel` returns `null` on unsupported generation or IL failure

This is a deliberate contract. A good engine caller either:

- falls back to a scalar/general implementation, or
- throws a precise `NotSupportedException` if no correct fallback exists

Do not let kernel-generation failure silently corrupt behavior.

### 3. Execution path selection is stride-driven

The current path hierarchy is stable:

1. `SimdFull`
2. `SimdScalarRight`
3. `SimdScalarLeft`
4. `SimdChunk`
5. `General`

Fast path selection is based on memory layout, not just dtype.

Key rule:

- contiguous and scalar-broadcast cases deserve distinct kernels
- arbitrary strided layouts must still be correct through a general coordinate-based path

### 4. SIMD gating is conservative on purpose

The generator only uses SIMD when all of these are true:

- the operation is supported
- the dtype is supported
- the path shape can actually use vector loads efficiently
- per-element conversion is not required in the vector loop

This is why many paths intentionally fall back to scalar code for:

- `decimal`
- `char`
- some boolean behavior
- mixed-type cases with conversion
- operations with no vector intrinsic equivalent

Do not force SIMD into cases that require per-lane conversions or semantics it cannot express cleanly.

### 5. Every fast path needs a correct general path

A kernel is not complete when `SimdFull` works.

Good kernel work means covering:

- contiguous arrays
- scalar broadcast
- chunkable inner-contiguous views
- arbitrary strided views

If you add a fast path but skip the general path, the feature is incomplete for NumSharp's view/broadcast model.

### 6. Offsets and strides must be handled exactly

There is a recurring subtle contract in the engine:

- base pointer already includes `shape.offset * elemSize`
- stride arrays are in element units
- load/store address arithmetic inside the kernel multiplies stride by element size when needed

Do not mix byte strides and element strides in the same layer.

### 7. Prefer unrolled vector loops plus scalar tails

The generator already follows a house style:

- vector loop for the bulk of the work
- often 4x unrolled for better ILP
- scalar remainder/tail

That pattern shows up in unary, binary, and reduction code because it is the stable performance baseline.

### 8. The general path should use explicit coordinate math

Kernel-level general loops typically compute:

- output coordinates from a linear index
- input base offsets from those coordinates
- axis offsets or per-operand offsets from strides

That is preferred over trying to bolt iterator objects into generated IL.

Outside the kernel generator, iterator-based code is still fine when it keeps special-case logic simpler.

### 9. Numeric semantics stay explicit in the kernel

Examples from the current code:

- NaN-aware reductions use explicit NaN masking and count tracking
- `Var`/`Std` use dedicated two-pass logic
- mean is implemented as sum plus count division, not a magical special vector op
- arg reductions must preserve index semantics, including first-occurrence behavior

The rule is to encode the semantic invariant directly, then optimize it.

## Common Design Patterns Already Used Successfully

### Thin override + shared dispatcher

Use for:

- add/subtract/multiply/divide/mod
- unary math
- comparisons

This is the default pattern.

### Specialized dispatcher with familiar structure

Use when the operation does not fit standard unary or binary semantics.

Good examples:

- `Default.ATan2.cs`
- `Default.Modf.cs`
- `Default.ClipNDArray.cs`

Even these specialized files still follow the same broad structure:

- validate semantics
- resolve dtype
- resolve shapes
- branch on scalar/empty/contiguous/general
- call kernel or helper

### Axis reduction helper + keepdims reshape

Use for reductions where:

- output shape is input shape minus one axis
- `keepdims` only changes the visible shape, not the computation

This is the standard pattern for `sum`, `prod`, `min`, `max`, `mean`, and count-style reductions.

## Testing Rules Implied by the Existing Suite

A "good" engine implementation is expected to have tests for more than value correctness.

Minimum matrix:

- NumPy-derived expected output
- contiguous input
- non-contiguous or transposed input
- sliced input with non-zero offset
- broadcast input with stride `0`
- scalar input
- empty input
- negative axis
- `keepdims`
- dtype promotion
- `out` handling when supported
- alias-safety where NumPy materializes instead of returning a view
- NaN behavior for floating-point operations

The current suite also uses two important categories:

- `OpenBugs` for known failures that should become passing tests later
- `Misaligned` for documented NumSharp-vs-NumPy behavior gaps

Do not "normalize" a failing NumPy mismatch into a regular passing test. Mark it accurately.

## Current Caution Points

A few current tests show where you should be careful not to infer the wrong rule from the current implementation:

- `mean(float32)` still returns `float64` in NumSharp, even though NumPy 2.x uses `float32`
- `var/std(float32)` still have open alignment gaps
- `reciprocal(int)` is documented as misaligned
- empty `bool` product still has an open dtype issue

These are not design targets. They are warnings that the implementation still has rough edges in some areas.

## Checklist for Adding or Refactoring a DefaultEngine Function

1. Run the equivalent NumPy code first and write down dtype, shape, empty, NaN, broadcasting, and axis behavior.
2. Decide whether the function fits an existing shared dispatcher.
3. If it does not, create a specialized dispatcher that still follows the same shape: validate, normalize, classify, execute.
4. Handle empty, scalar, axis, and trivial-axis cases before the hot loop.
5. Make the result dtype explicit using existing promotion helpers or an operation-specific rule.
6. Ensure sliced, strided, and broadcast inputs work by honoring offsets and strides.
7. Add or reuse an IL kernel only when it has both a real fast path and a correct general path.
8. Keep the public override thin.
9. Add NumPy-based tests for contiguous, strided, broadcast, empty, scalar, and dtype cases.
10. If behavior is still intentionally wrong, mark it `OpenBugs` or `Misaligned` instead of hiding it.

## Short Version

The house style is:

- thin API method
- semantics first
- dtype first
- shape first
- empty/scalar/trivial cases first
- contiguous SIMD fast path when layout allows
- correct general path for everything else
- tests that prove NumPy parity across layout and dtype edge cases

That is what the best current `DefaultEngine` and `ILKernelGenerator` code is already doing.
