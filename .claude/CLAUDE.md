# NumSharp Project Instructions

NumSharp is a .NET port of Python's NumPy library targeting **1-to-1 API and behavioral compatibility with NumPy 2.x**.

## NumPy Reference Source

A full clone of the NumPy repository is available at `src/numpy/`. Use this as the authoritative reference for API behavior, edge cases, and implementation details when implementing or verifying NumSharp functions.

## Core Principles

1. **Match NumPy Exactly**: Run actual Python/NumPy code first, observe behavior, replicate in C#
2. **Match NumPy Implementation Patterns**: Don't just match behavior - match NumPy's implementation structure. If NumPy has a clean approach and NumSharp has spaghetti code, refactor to match NumPy's design
3. **Edge Cases Matter**: NaN handling, empty arrays, type promotion, broadcasting, negative axis
4. **Breaking Changes OK**: Breaking changes are acceptable to match NumPy
5. **Test From NumPy Output**: Tests should be based on running actual NumPy code

**When fixing bugs:** Don't just patch symptoms. Check `src/numpy/` for how NumPy implements the same functionality, then refactor NumSharp to match NumPy's structure.

## Definition of Done (DOD) - Operations

Every np.* function and DefaultEngine operation MUST satisfy these criteria:

### Memory Layout Support
- **Contiguous arrays**: Works correctly with C-contiguous memory (SIMD fast path)
- **Non-contiguous arrays**: Works correctly with sliced/strided/transposed views
- **Broadcast arrays**: Works correctly with stride=0 dimensions (read-only)
- **Sliced views**: Correctly handles Shape.offset for base address calculation

### Dtype Support
All 15 NumSharp types must be handled (or explicitly documented as unsupported):
Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Half, Single, Double, Decimal, Complex

### NumPy API Parity
- Function signature matches NumPy (parameter names, order, defaults)
- Type promotion matches NumPy 2.x (NEP50)
- Edge cases match NumPy (empty arrays, scalars, NaN handling, broadcasting)
- Return dtype matches NumPy exactly

### Testing
- Unit tests based on actual NumPy output
- Edge case tests (empty, scalar, broadcast, strided)
- Dtype coverage tests

**Full audit tracking:** See `docs/KERNEL_API_AUDIT.md`

## Supported Types (15)

| NPTypeCode | C# Type | NPTypeCode | C# Type |
|------------|---------|------------|---------|
| Boolean | bool | UInt64 | ulong |
| Byte | byte | Char | char |
| SByte | sbyte | Half | System.Half |
| Int16 | short | Single | float |
| UInt16 | ushort | Double | double |
| Int32 | int | Decimal | decimal |
| UInt32 | uint | Complex | System.Numerics.Complex |
| Int64 | long |  |  |

All operations must handle all 15 types via type switch pattern.

**Perf notes:**
- SByte / Byte / Int*/UInt* / Single / Double — full SIMD via the mixed-type kernel's `SimdFull` execution path (V128/V256/V512 detected at startup).
- Half — scalar path (no `Vector<Half>` arithmetic in .NET BCL). Routes through `Half→double→Math.Pow→Half` for `np.power`; ~2× slower than NumPy.
- Complex — scalar path via `System.Numerics.Complex` operators / `Complex.Pow`. ~2× slower than NumPy.
- Decimal — scalar path via `DecimalMath.Pow`. Highest precision, slowest.

## Architecture

```
NDArray           Main class (like numpy.ndarray)
├── Storage       UnmanagedStorage (raw pointers, not managed arrays)
├── Shape         Dimensions, strides, offset calculation
└── TensorEngine  Computation backend (DefaultEngine = pure C#)

np                Static API class (like `import numpy as np`)
├── np.random     NumPyRandom (1-to-1 seed/state with NumPy)
└── np.*          Functions in Creation/, Math/, Statistics/, Logic/, etc.
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Unmanaged memory | Benchmarked fastest; optimized for performance |
| Order-aware layout | Row-major (C-order) remains the default. `Shape` also tracks F-contiguity, and APIs with an `order` parameter resolve NumPy `C`/`F`/`A`/`K` modes through `OrderResolver`. |
| Regen templating | Type-specific code generation (legacy, mostly replaced by ILKernel) |
| TensorEngine abstract | Future GPU/SIMD backends possible |
| View semantics | Slicing returns views (shared memory), not copies |
| Shape readonly struct | Immutable after construction (NumPy-aligned). Contains `ArrayFlags` for cached O(1) property access |
| Broadcast write protection | Broadcast views are read-only (`IsWriteable = false`), matching NumPy behavior |
| ILKernelGenerator + DirectILKernelGenerator | Runtime IL emission with SIMD V128/V256/V512 (~36K lines). Two classes split by kernel-driving contract: `ILKernelGenerator` (per-chunk kernels driven by NpyIter — `np.where`, flat/pairwise reductions) and `DirectILKernelGenerator` (whole-array kernels, 59 partials in `Direct/`). Supersedes Regen templates. |

## ILKernelGenerator + DirectILKernelGenerator

Runtime IL generation via `System.Reflection.Emit.DynamicMethod` for high-performance kernels. **Two physically distinct classes**, each encoding its kernel-driving contract in the type name:

### `DirectILKernelGenerator` — whole-array kernels (59 partials)

**Location:** `src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.*.cs`

**Contract:** A single kernel call processes the **whole array**. The kernel itself walks dimensions/strides; the iterator (if any) is only a setup helper. Signature shape varies per kernel family but typically:

```csharp
delegate void DirectKernel(
    void* input, void* output,
    long* strides, long* shape, int ndim,
    long iterSize);
```

Carries the bulk of NumSharp's elementwise, reduction, scan, cast, and selection kernels.

**Partial files in `Direct/` (59):**
| Category | Files |
|----------|-------|
| Core | `DirectILKernelGenerator.cs` (type mapping, SIMD detection), `.Scalar.cs` |
| Binary | `.Binary.cs`, `.MixedType.cs`, `.Shift.cs` |
| Unary | `.Unary.cs`, `.Unary.Math.cs`, `.Unary.Decimal.cs`, `.Unary.Vector.cs`, `.Unary.Predicate.cs`, `.Unary.Strided.cs` |
| Comparison | `.Comparison.cs` |
| Reduction (flat) | `.Reduction.cs`, `.Reduction.Arg.cs`, `.Reduction.Boolean.cs`, `.Reduction.NaN.cs` |
| Reduction (axis) | `.Reduction.Axis.cs`, `.Reduction.Axis.Arg.cs`, `.Reduction.Axis.Boolean.cs`, `.Reduction.Axis.Simd.cs`, `.Reduction.Axis.NaN.cs`, `.Reduction.Axis.VarStd.cs`, `.Reduction.Axis.Widening.cs` |
| Scan | `.Scan.cs` (CumSum, CumProd) |
| Masking | `.Masking.Boolean.cs`, `.Masking.NaN.cs`, `.Masking.VarStd.cs` |
| Cast & Copy | `.Cast.cs`, `.Cast.Masked.cs`, `.Cast.Scalar.cs`, `.Cast.Half.cs`, `.Cast.ToHalf.cs`, `.Cast.FloatNarrow.cs`, `.Cast.FloatWideInt.cs`, `.Cast.IntNarrow.cs`, `.Cast.ToBool.cs`, `.Cast.Complex.cs`, `.Copy.cs` |
| Selection | `.Where.cs`, `.Where.Scalar.cs`, `.Place.cs`, `.Put.cs`, `.Take.cs`, `.NonZero.cs`, `.Argwhere.cs`, `.Indices.cs`, `.Filter.cs`, `.Search.cs` |
| Linear algebra | `.MatMul.cs`, `.Trace.cs` |
| Other | `.Clip.cs`, `.Modf.cs`, `.Repeat.cs`, `.Quantile.cs`, `.WeightedSum.cs`, `.RavelMultiIndex.cs`, `.UnravelIndex.cs`, `.InnerLoop.cs`, `.StorageAlias.cs` |

### `ILKernelGenerator` — NpyIter-driven per-chunk kernels

**Location:** `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator*.cs` (root, not in `Direct/`)

**Contract:** Kernel processes **one inner-loop chunk** per call. The iterator (`NpyIterRef`) drives the loop and advances pointers between calls. Matches NumPy's `PyUFuncGenericFunction` model:

```csharp
unsafe delegate void NpyInnerLoopFunc(
    void** dataptrs,   // [nop] current operand pointers
    long*  strides,    // [nop] per-operand byte stride for inner loop
    long   count,      // number of elements this chunk
    void*  auxdata);   // op-specific extras (e.g. axis index)
```

Emits the per-chunk kernels run as an NpyIter inner loop — currently `np.where` and the flat/pairwise reductions (partials `ILKernelGenerator.{Where,Reduction,Reduction.Pairwise}.cs`). Driven via `NpyIterRef.Execute(kernelKey)`.

### Shared infrastructure

Both classes use these helpers at `src/NumSharp.Core/Backends/Kernels/` (root, outside `Direct/`):
- `VectorMethodCache.cs`, `ScalarMethodCache.cs` — reflection caches (Vector{128,256,512}, Math/MathF)
- `KernelOp.cs` — `BinaryOp`, `UnaryOp`, `ReductionOp`, `ComparisonOp`, `ExecutionPath` enums
- `BinaryKernel.cs`, `CopyKernel.cs`, `ReductionKernel.cs`, `ScalarKernel.cs` — kernel key structs and delegate types
- `StrideDetector.cs` — layout classification (Contiguous / Strided / Broadcast)
- `SimdMatMul.*.cs` — matmul SIMD primitives

**Execution Paths (DirectILKernelGenerator):**
1. **SimdFull** - Both operands contiguous, SIMD-capable dtype → Vector loop + scalar tail
2. **ScalarFull** - Both contiguous, non-SIMD dtype (Decimal) → Scalar loop
3. **General** - Strided/broadcast → Coordinate-based iteration

**NEP50 Dtype Alignment (NumPy 2.x):**
| Operation | Returns |
|-----------|---------|
| `sum(int32)` | `int64` |
| `prod(int32)` | `int64` |
| `cumsum(int32)` | `int64` |
| `abs(int32)` | `int32` (preserves) |
| `sign(int32)` | `int32` (preserves) |
| `power(int32, float)` | `float64` |

**ILKernel Coverage:**
| Category | Operations |
|----------|------------|
| Binary | Add, Sub, Mul, Div, Power, FloorDivide, BitwiseAnd/Or/Xor |
| Shift | LeftShift, RightShift (SIMD for scalar, scalar loop for array) |
| Unary | Negate, Abs, Sign, Sqrt, Cbrt, Square, Reciprocal, Floor, Ceil, Truncate, Trig, Exp, Log, BitwiseNot |
| Reduction | Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any, Std, Var |
| Scan | CumSum, CumProd (element-wise SIMD + axis support) |
| Comparison | Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual |
| Clip/Modf | Clip, Modf (SIMD helpers) |
| Axis reductions | Sum, Prod, Min, Max, Mean, Std, Var (iterator path) |

## Shape Architecture (NumPy-Aligned)

Shape is a `readonly struct` with cached `ArrayFlags` computed at construction:

```csharp
public readonly partial struct Shape
{
    internal readonly int _flags;        // Cached ArrayFlags bitmask
    internal readonly int _hashCode;     // Precomputed hash code
    internal readonly int size;          // Total element count
    internal readonly int[] dimensions;  // Dimension sizes
    internal readonly int[] strides;     // Stride values (0 = broadcast dimension)
    internal readonly int bufferSize;    // Size of underlying buffer
    internal readonly int offset;        // Base offset into storage
}
```

**ArrayFlags enum** (matches NumPy's `ndarraytypes.h`):
| Flag | Value | Meaning |
|------|-------|---------|
| `C_CONTIGUOUS` | 0x0001 | Data is row-major contiguous |
| `F_CONTIGUOUS` | 0x0002 | Data is column-major contiguous |
| `OWNDATA` | 0x0004 | Array owns its data buffer |
| `ALIGNED` | 0x0100 | Always true for managed allocations |
| `WRITEABLE` | 0x0400 | False for broadcast views |
| `BROADCASTED` | 0x1000 | Has stride=0 with dim > 1 |

**Key Shape properties:**
- `IsContiguous` — O(1) check via `C_CONTIGUOUS` flag
- `IsFContiguous` — O(1) check via `F_CONTIGUOUS` flag
- `IsBroadcasted` — O(1) check via `BROADCASTED` flag
- `IsWriteable` — False for broadcast views (prevents corruption)
- `IsSliced` — True if offset != 0, different size, or non-contiguous
- `IsSimpleSlice` — IsSliced && !IsBroadcasted (fast offset path)

## Critical: View Semantics

**Slicing returns views, not copies!** Memory is shared:
```csharp
var view = original["2:5"];  // Shares memory with original
view[0] = 999;               // Modifies original[2]!
var copy = original["2:5"].copy();  // Explicit copy
```

## Slicing Syntax

```csharp
nd[":"]           // All elements
nd["1:5"]         // Elements 1-4 (stop exclusive)
nd["::2"]         // Every 2nd element
nd["-1"]          // Last element (reduces dimension)
nd["::-1"]        // Reversed
nd[":, 0"]        // All rows, first column
nd["..., -1"]     // Ellipsis fills dimensions
```

---

## Missing Functions (8)

These NumPy functions are **not implemented**:

| Category | Functions |
|----------|-----------|
| Sorting | `np.sort` |
| Manipulation | `np.flip`, `np.fliplr`, `np.flipud`, `np.rot90` |
| Diagonal | `np.diag` |
| Cumulative | `np.gradient` |
| Rounding | `np.round` (use `np.round_` or `np.around`) |

---

## Supported np.* APIs

Tested against NumPy 2.x.

### Array Creation
`arange`, `array`, `asanyarray`, `asarray`, `ascontiguousarray`, `asfortranarray`, `copy`, `empty`, `empty_like`, `eye`, `frombuffer`, `full`, `full_like`, `identity`, `linspace`, `meshgrid`, `mgrid`, `ones`, `ones_like`, `zeros`, `zeros_like`

### Shape Manipulation
`append`, `array_split`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `concatenate`, `delete`, `dsplit`, `dstack`, `expand_dims`, `flatten`, `hsplit`, `hstack`, `insert`, `moveaxis`, `pad`, `ravel`, `repeat`, `reshape`, `roll`, `rollaxis`, `split`, `squeeze`, `stack`, `swapaxes`, `tile`, `transpose`, `unique`, `vsplit`, `vstack`

### Broadcasting
`are_broadcastable`, `broadcast`, `broadcast_arrays`, `broadcast_to`

### Math — Arithmetic
`abs`, `absolute`, `add`, `arccos`, `arcsin`, `arctan`, `arctan2`, `cbrt`, `ceil`, `clip`, `convolve`, `cos`, `cosh`, `deg2rad`, `degrees`, `divide`, `exp`, `exp2`, `expm1`, `floor`, `floor_divide`, `log`, `log10`, `log1p`, `log2`, `mod`, `modf`, `multiply`, `negative`, `positive`, `power`, `rad2deg`, `radians`, `reciprocal`, `sign`, `sin`, `sinh`, `sqrt`, `square`, `subtract`, `tan`, `tanh`, `true_divide`, `trunc`

**ufunc `out=` / `where=` parameters** are supported on the elementwise core (NumPy semantics, probed against 2.4.2): binary `add`/`subtract`/`multiply`/`divide`/`true_divide`/`mod`/`power`/`floor_divide`/`arctan2`/`bitwise_and`/`bitwise_or`/`bitwise_xor`, unary `sqrt`/`exp`/`log`/`sin`/`cos`/`tan`/`abs`/`absolute`/`negative`/`square`/`log2`/`log10`/`log1p`/`exp2`/`expm1`/`cbrt`/`sign`/`floor`/`ceil`/`trunc`/`reciprocal`/`sinh`/`cosh`/`tanh`/`arcsin`/`arccos`/`arctan`/`deg2rad`(`radians`)/`rad2deg`(`degrees`)/`invert`(`bitwise_not`). `round_`/`around` take `out=` ONLY (np.round is a function, not a ufunc — no where/dtype; decimals≠0 cast errors name ufunc 'multiply' per NumPy's composition). floor/ceil/trunc have IDENTITY loops on every bool/int dtype (dtype preserved; np.round's int path is an identity copy); the loop dtype comes from the input tier (`sinh(i1, out=f8)` stores float16-precision values); reciprocal int 1/0 → signed MinValue (NumPy 2.4.2); sign/positive reject bool with the verbatim no-loop UFuncTypeError; bitwise/invert raise the no-loop TypeError for float inputs (probed order: bad where → no-loop → out-cast → shape). `out` joins the broadcast but is never stretched, requires a same_kind cast from the loop dtype (resolved from inputs), returns the same instance, and may alias an input (overlap-safe via COPY_IF_OVERLAP). `where` must be bool, broadcasts and joins the output shape; masked-off `out` slots keep prior contents. Engine plumbing: `Backends/Default/Math/DefaultEngine.UfuncOut.cs`.

**Each of those ufuncs exposes ONE NumPy-shaped overload** — `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)` mirroring NumPy's `f(x1[, x2], /, out=None, *, where=True, dtype=None)`: `out` is the second/third positional slot exactly like NumPy, `where`/`dtype` are reachable by name without `out`. `dtype=` selects the LOOP (NumPy loop-signature semantics, probed): computation runs at that precision even with `out=` (`sqrt([2.], out=f64, dtype=f32)` stores the f32-rounded value; `add(0.1, 0.2, out=f64, dtype=f32)` stores `0.30000001…`), `power(2, -1, dtype: f64) = 0.5` (the negative-int-exponent ValueError applies only to integer loops), `power(10, 11, dtype: f64) = 1e11` exactly (no compute-then-cast), `add(True, True, dtype: i32) = 2` (the bool→logical-OR remap keys off the FINAL loop dtype), `negative(bool, dtype: f64)` is legal, and inputs must reach the loop via same_kind casts (verbatim `Cannot cast ufunc '<name>' input [N] from ...` errors; binary errors are indexed, unary are not). Loop-existence gates raise `No loop matching ... ufunc <name>`: float-only ufuncs (sqrt/exp/log/trig + `divide`/`true_divide`) reject int/bool dtype; bitwise rejects float/complex/decimal dtype. `positive` is a full ufunc (identity loops at every dtype EXCEPT bool: plain `positive(bool)` and `dtype: bool` raise the verbatim `did not contain a loop with signature matching types <class 'numpy.dtypes.XDType'> -> ...` texts; `positive(bool, dtype: f64)` works). `round_`/`around` follow NumPy's non-ufunc shape `round(a, int decimals = 0, NDArray out = null)` (2nd positional is decimals, NOT out). Legacy positional-dtype overloads (`np.sqrt(x, NPTypeCode.Single)`, `(x, Type)`) remain for source compat but are non-NumPy call forms (NumPy's 2nd positional is `out`). Tests: `Math/UfuncDtypeOverloadTests.cs`.

### Math — Reductions
`all`, `amax`, `amin`, `any`, `argmax`, `argmin`, `average`, `average_returned`, `count_nonzero`, `cumprod`, `cumsum`, `diff`, `ediff1d`, `max`, `mean`, `median`, `min`, `percentile`, `prod`, `ptp`, `quantile`, `std`, `sum`, `var`

### Math — NaN-Aware
`nanmax`, `nanmean`, `nanmedian`, `nanmin`, `nanpercentile`, `nanprod`, `nanquantile`, `nanstd`, `nansum`, `nanvar`

### Bitwise
`bitwise_and`, `bitwise_or`, `bitwise_xor` (ufunc `out=`/`where=`/`dtype=` supported; float/complex/decimal INPUTS raise NumPy's coercion TypeError while a float/complex/decimal `dtype=` raises the no-loop text — distinct messages, both probed; probed order: bad `where` → no-loop → out-cast → shape), `invert`, `left_shift`, `right_shift`

### Comparison & Logic
`all`, `allclose`, `any`, `array_equal`, `equal`, `fmax`, `fmin`, `greater`, `greater_equal`, `isclose`, `iscomplex`, `iscomplexobj`, `isfinite`, `isinf`, `isnan`, `isreal`, `isrealobj`, `isscalar`, `less`, `less_equal`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `maximum`, `minimum`, `not_equal`

The six comparisons and `isnan`/`isfinite`/`isinf` expose **ONE NumPy-shaped overload each** — `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)` (no bare/out split). It returns plain `NDArray` — NumPy's `np.less(a, b, out=f64)` returns the f64 out itself; `True→1` at any numeric out dtype since bool casts same_kind to all of them. A plain call still returns an `NDArray<bool>` *instance* (TensorEngine contract), so the typed wrapper is one zero-alloc cast away and the C# comparison operators (`==`, `<`, …) keep the `NDArray<bool>` static type via `AsGeneric<bool>()`. `dtype=` is validate-only (probed 2.4.2): bool loops only — `dtype: Boolean` is a no-op, anything else raises `No loop matching the specified signature and casting was found for ufunc <name>`. Comparisons compare at `result_type(lhs, rhs)` inside the kernel (probed: `greater(i8 2^53+1, f8 2^53)` → False, `equal` → True). Engine members follow the house order `(inputs, typeCode, out, where)`.

### Type Promotion & Dtype
`can_cast`, `common_type`, `find_common_type`, `finfo`, `iinfo`, `issubdtype`, `min_scalar_type`, `mintypecode`, `promote_types`, `result_type`

### Selection
`compress`, `extract`, `indices`, `place`, `put`, `ravel_multi_index`, `take`, `unravel_index`, `where`

### Fused Expressions (NumSharp extension)
`evaluate` — `np.evaluate(expr[, operands][, out])` compiles an `NpyExpr` tree to ONE NpyIter pass: every elementwise node runs inside a single inner-loop kernel, so chained expressions allocate no intermediates and read each operand once (NumPy-ecosystem equivalent: `numexpr.evaluate`; measured 3.2–6.1× faster than NumPy 2.4.2 on 4M chains, 1.2–4× over NumSharp's own unfused chains — gate: `benchmark/poc/evaluate_bench.{cs,py}`).

```csharp
NDArray r = np.evaluate((NpyExpr)a * b + 2);                          // fused a*b+2
NDArray d = np.evaluate((NpyExpr.Arr(a) - b) / (NpyExpr.Arr(a) + b)); // a,b dedup → 3 operand streams
NDArray s = np.evaluate(NpyExpr.Sum((NpyExpr)a * b), @out: x);        // one-pass sum(a*b), no temp
```

Semantics (all probed against NumPy 2.4.2, pinned in `NpyEvaluateTests.cs`):
- **Dtypes follow NumPy result_type PER NODE** (`NpyExpr.Typing.cs`): NEP50 strong-strong incl. the int/float tier crossing (`i4+f4→f8`); weak python-scalar literals (`i4+2→i4`, `f2+2.5→f2`, `bool+2→i64`, out-of-range → OverflowError); `true_divide` ints→f64; `arctan2` tier floats; `power`/`remainder`/`floor_divide` bool→i8; unary math tiers (`bool/i8→f16`, `i16→f32`, `i32+→f64`); comparisons→bool. `(i4*i4)+f8` wraps the multiply in int32 before promoting — bit-compatible with the unfused NumPy sequence.
- **Reductions are root-only**: `NpyExpr.Sum/Prod/Min/Max/Mean(expr)` run a one-pass accumulating kernel over the inputs (NumPy reduce dtypes; f16/f32 sums accumulate in f64 and cast back — documented divergence from NumPy's pairwise, usually more accurate; min/max NaN-propagate; empty: sum=0, prod=1, mean=NaN, min/max raise).
- Repeated NDArray references deduplicate to one iterator operand; `out=` follows the ufunc rules above; `ExecuteExpression` (Tier 3C) throws without `EXTERNAL_LOOP` (the ~40× per-element foot-gun) — `np.evaluate` configures the iterator itself.

### Sorting & Searching
`argmax`, `argmin`, `argsort`, `argwhere`, `flatnonzero`, `nonzero`, `searchsorted`

### Linear Algebra
`diagonal`, `dot`, `matmul`, `outer`, `trace`

### Random (`np.random.*`)
`bernoulli`, `beta`, `binomial`, `chisquare`, `choice`, `dirichlet`, `exponential`, `f`, `gamma`, `geometric`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `logseries`, `multinomial`, `multivariate_normal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `permutation`, `poisson`, `power`, `rand`, `randint`, `randn`, `random_sample`, `rayleigh`, `seed`, `shuffle`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`, `uniform`, `vonmises`, `wald`, `weibull`, `zipf`

### File I/O
`fromfile`, `load`, `save`, `tofile`

### Other
`around`, `asscalar`, `copyto`, `round_`, `size`

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`, unary `-`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&`, `|`, `!`

### Indexing
- Integer and slice indexing (`nd[0]`, `nd[1:3]`, `nd[::-1]`)
- Boolean masking (`nd[mask]`) — read-only
- Fancy indexing (`nd[indices]`)

---

## Core Source Files

| Component | Location |
|-----------|----------|
| NDArray | `Backends/NDArray.cs` |
| UnmanagedStorage | `Backends/Unmanaged/UnmanagedStorage.cs` |
| Shape | `View/Shape.cs` |
| Slice | `View/Slice.cs` |
| TensorEngine | `Backends/TensorEngine.cs` |
| DefaultEngine | `Backends/Default/DefaultEngine.*.cs` |
| np API | `APIs/np.cs` |
| Iterators | `Backends/Iterators/NpyIter.cs` |
| Expression DSL (np.evaluate) | `Backends/Iterators/NpyExpr.cs` (nodes + emission), `NpyExpr.Typing.cs` (per-node NumPy result_type pass), `NpyExpr.Evaluate.cs` (array leaves, binding, reductions, operators), `Backends/Default/Math/DefaultEngine.Evaluate.cs` (host) |
| ILKernelGenerator | `Backends/Kernels/ILKernelGenerator*.cs` (per-chunk, NpyIter-driven) |
| DirectILKernelGenerator | `Backends/Kernels/Direct/DirectILKernelGenerator.*.cs` (whole-array, 59 partials) |
| Kernel shared infra | `Backends/Kernels/{VectorMethodCache,ScalarMethodCache,KernelOp,StrideDetector,SimdMatMul.*}.cs` |
| Type info | `Utilities/InfoOf.cs` |
| Generic NDArray | `Generics/NDArray\`1.cs` |

---

## Implementation Patterns

### Pattern 1: Compose existing np functions
```csharp
public static NDArray std(NDArray a, int? axis = null, ...)
{
    var mean_val = np.mean(a, axis, keepdims: true);
    return np.sqrt(np.mean(np.power(a - mean_val, 2), axis));
}
```

### Pattern 2: Delegate to TensorEngine
```csharp
public static NDArray sum(NDArray a, int? axis = null, ...)
{
    return a.TensorEngine.Sum(a, axis, typeCode, keepdims);
}
```
Use Pattern 2 when low-level optimization is needed.

## Type Switch Pattern

```csharp
switch (nd.typecode)
{
    case NPTypeCode.Boolean: return Process<bool>(nd);
    case NPTypeCode.Byte: return Process<byte>(nd);
    case NPTypeCode.SByte: return Process<sbyte>(nd);
    case NPTypeCode.Int16: return Process<short>(nd);
    case NPTypeCode.UInt16: return Process<ushort>(nd);
    case NPTypeCode.Int32: return Process<int>(nd);
    case NPTypeCode.UInt32: return Process<uint>(nd);
    case NPTypeCode.Int64: return Process<long>(nd);
    case NPTypeCode.UInt64: return Process<ulong>(nd);
    case NPTypeCode.Char: return Process<char>(nd);
    case NPTypeCode.Half: return Process<Half>(nd);
    case NPTypeCode.Single: return Process<float>(nd);
    case NPTypeCode.Double: return Process<double>(nd);
    case NPTypeCode.Decimal: return Process<decimal>(nd);
    case NPTypeCode.Complex: return Process<Complex>(nd);
    default: throw new NotSupportedException();
}
```

## GitHub Issues

Create issues on `SciSharp/NumSharp` via `gh issue create`. `GH_TOKEN` is available via the `env-tokens` skill.

### Feature / Enhancement

- **Overview**: 1-2 sentence summary of what and why
- **Problem**: What's broken or missing, why it matters
- **Proposal**: What to change, with a task checklist (`- [ ]`)
- **Evidence**: Data, benchmarks, or references supporting the proposal
- **Scope / Non-goals**: What this issue does NOT cover (prevent scope creep)
- **Benchmark / Performance** (if applicable): Before/after numbers, methodology, what to measure
- **Breaking changes** table (if any): Change | Impact | Migration
- **Related issues**: Link dependencies

### Bug Report

- **Overview**: 1-2 sentence summary of the bug and its impact
- **Reproduction**: Minimal code to trigger the bug
- **Expected**: Correct behavior (include NumPy output as source of truth)
- **Actual**: What NumSharp does instead (error message, wrong output, crash)
- **Workaround** (if any): How users can avoid the bug today
- **Root cause** (if known): File, line, why it happens
- **Related issues**: Link duplicates or upstream causes

## Performance Convention

All NumSharp-vs-NumPy benchmark ratios are reported as **NPY/NS**:

> **ratio = NumPy_ms / NumSharp_ms** — **`>1` = NumSharp FASTER, `<1` = NumSharp slower, `=1` = parity.**

**Higher is better.** Equivalently `speedup = NumPy_time / NumSharp_time`. A cell of
`0.5` means NumSharp takes 2× NumPy's time; `2.0` means NumSharp is 2× faster. Use this
direction **everywhere** — matrices, geomeans, commit messages, the `npyiter` sheet, and
the `benchmark/poc/*_merge.py` outputs — so one glance answers "are we faster?".

- Icons used in the matrices: ✅ `≥1.0` · 🟡 `≥0.5` · 🟠 `≥0.2` · 🔴 `<0.2`.
- Timing scripts MUST run `dotnet run -c Release - < script.cs` (Debug taints hand-written kernels ~2×; see `benchmark/CLAUDE.md`).
- best-of-rounds (take the min), warmup excluded, correctness checked before every timed row.
- The canonical harness is `benchmark/npyiter/` (NpyIter aspects) + `benchmark/poc/reduce_layout_*` (reduction × layout × dtype). The legacy `run-benchmarks.ps1` "Status Icons" table reports the *inverse* (NS/NPY, lower = better) — prefer this NPY/NS convention.

## Build & Test

```bash
# Build (silent, errors only)
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
```

### Running Tests

Tests use **MSTest v3** framework with source-generated test discovery.

```bash
# Run from test directory
cd test/NumSharp.UnitTest

# All tests (includes OpenBugs - expected failures)
dotnet test --no-build

# Exclude OpenBugs (CI-style - only real failures)
dotnet test --no-build --filter "TestCategory!=OpenBugs"

# Run ONLY OpenBugs tests
dotnet test --no-build --filter "TestCategory=OpenBugs"

# Exclude multiple categories
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"

# Run specific test class
dotnet test --no-build --filter "ClassName~BinaryOpTests"

# Run specific test method
dotnet test --no-build --filter "Name~Add_Int32_SameType"
```

### Output Formatting

```bash
# Results only (no messages, no stack traces)
dotnet test --no-build 2>&1 | grep -E "^(failed|skipped|Test run|  total:|  failed:|  succeeded:|  skipped:|  duration:)"

# Results with messages (no stack traces)
dotnet test --no-build 2>&1 | grep -v "^    at " | grep -v "^     at " | grep -v "^    ---" | grep -v "^  from K:" | sed 's/AssertFailedException: //'

# Verbose output (shows passed tests too)
dotnet test --no-build -v normal
```

## Test Categories

Tests use typed category attributes defined in `TestCategory.cs`. Adding new bug reproductions or platform-specific tests only requires the right attribute — no CI workflow changes.

| Category | Attribute | Purpose | CI Behavior |
|----------|-----------|---------|-------------|
| `OpenBugs` | `[OpenBugs]` | Known-failing bug reproductions. Remove when fixed. | **EXCLUDED** via filter |
| `Misaligned` | `[Misaligned]` | Documents NumSharp vs NumPy behavioral differences. | Runs (tests pass) |
| `WindowsOnly` | `[WindowsOnly]` | Requires GDI+/System.Drawing.Common | Excluded on non-Windows |
| `LongIndexing` | `[LongIndexing]` | Arrays with size > int.MaxValue (>2B elements) | Runs (excluded only if also HighMemory) |
| `HighMemory` | `[HighMemory]` | Requires 8GB+ RAM | **EXCLUDED** via filter |
| `LargeMemoryTest` | `[LargeMemoryTest]` | Memory-heavy non-bugs (combines OpenBugs+HighMemory) | **EXCLUDED** via filter |

### How CI Excludes Categories

The CI pipeline (`.github/workflows/build-and-release.yml`) uses MSTest's `--filter` to exclude categories:

```yaml
- name: Test (net10.0)
  run: |
    dotnet test test/NumSharp.UnitTest/NumSharp.UnitTest.csproj \
      --configuration Release --no-build --framework net10.0 \
      --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
```

This filter excludes all tests with `[OpenBugs]` or `[HighMemory]` attributes from CI runs. Tests pass locally when the bug is fixed — then remove the `[OpenBugs]` attribute.

### Usage

```csharp
// Class-level (all tests in class)
[TestClass]
[OpenBugs]
public class BroadcastBugTests { ... }

// Method-level
[TestMethod]
[OpenBugs]
public void BroadcastWriteCorruptsData() { ... }

// Documenting behavioral differences (NOT excluded from CI)
[TestMethod]
[Misaligned]
public void BroadcastSlice_MaterializesInNumSharp() { ... }
```

### Local Filtering

```bash
# Exclude OpenBugs (same as CI)
dotnet test --filter "TestCategory!=OpenBugs"

# Run ONLY OpenBugs tests (to verify fixes)
dotnet test --filter "TestCategory=OpenBugs"

# Run ONLY Misaligned tests
dotnet test --filter "TestCategory=Misaligned"

# Combine multiple exclusions
dotnet test --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=WindowsOnly"
```

**OpenBugs files**: `OpenBugs.cs` (general), `OpenBugs.Bitmap.cs` (bitmap), `OpenBugs.ApiAudit.cs` (API audit), `OpenBugs.ILKernelBattle.cs` (IL kernel), `OpenBugs.Random.cs` (random), `OpenBugs.BroadcastReduce.cs` (broadcast reduce).

## CI Pipeline

`.github/workflows/build-and-release.yml` — test on 3 OSes (Windows/Ubuntu/macOS), build NuGet on tag push, create GitHub Release, publish to nuget.org.

## Scripting with `dotnet run` (.NET 10 file-based apps)

### Accessing Internal Members

NumSharp has many key types/fields/methods marked `internal` (Shape.dimensions, Shape.strides, NDArray.Storage, np._FindCommonType, etc.). To access them from a `dotnet run` script, override the assembly name to match an existing `InternalsVisibleTo` entry:

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
```

**How it works:** NumSharp declares `[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript")]` in `src/NumSharp.Core/Assembly/Properties.cs`. The `#:property AssemblyName=NumSharp.DotNetRunScript` directive overrides the script's assembly name (which normally derives from the filename) to match, granting full access to all `internal` and `protected internal` members.

### Accessing Unsafe code
NumSharp uses unsafe in many places, hence include `#:property AllowUnsafeBlocks=true` in scripts.

### Script Template (copy-paste ready)

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
```

### Key Internal Members Available

| Member | What it exposes |
|--------|----------------|
| `shape.dimensions` | Raw int[] of dimension sizes |
| `shape.strides` | Raw int[] of stride values |
| `shape.size` | Internal field: total element count |
| `shape.offset` | Base offset into storage (NumPy-aligned) |
| `shape.bufferSize` | Size of underlying buffer |
| `shape._flags` | Cached ArrayFlags bitmask |
| `shape.IsWriteable` | False for broadcast views (NumPy behavior) |
| `shape.IsBroadcasted` | Has any stride=0 with dimension > 1 |
| `shape.IsSimpleSlice` | IsSliced && !IsBroadcasted |
| `shape.OriginalSize` | Product of non-broadcast dimensions |
| `arr.Storage` | Underlying `UnmanagedStorage` |
| `arr.GetTypeCode` | `NPTypeCode` of the array |
| `arr.Array` | `IArraySlice` — raw data access |
| `np._FindCommonType(...)` | Type promotion logic |
| `np.powerOrder` | Type promotion ordering |
| `NPTypeCode.GetGroup()` | Type category (int/uint/float/etc.) |
| `NPTypeCode.GetPriority()` | Type priority for promotion |
| `NPTypeCode.AsNumpyDtypeName()` | NumPy dtype name (e.g. "int32") |
| `Shape.NewScalar()` | Create scalar shapes |

### Common Public NDArray Properties

| Property | Description |
|----------|-------------|
| `nd.shape` | Dimensions as `int[]` |
| `nd.ndim` | Number of dimensions |
| `nd.size` | Total element count |
| `nd.dtype` | Element type as `Type` |
| `nd.typecode` | Element type as `NPTypeCode` |
| `nd.T` | Transpose (swaps axes) |
| `nd.flat` | 1D iterator over elements |

## Adding New Features

1. Read NumPy docs for the function
2. **Run actual Python code** to observe exact behavior and fuzzy all possible inputs to define a behavior matrix.
3. Check existing similar implementations
4. Implement behavior matching exactly that of numpy.
5. Write tests based on observed NumPy output
6. Handle all 15 dtypes

---

## Q&A - Design & Architecture

**Q: Why unmanaged memory instead of Span<T>/Memory<T>?**
A: Benchmarking showed unmanaged memory was fastest. NDArray is self-managed memory allocation optimized for performance.

**Q: Why Regen templating instead of T4 or source generators?**
A: Original needs felt too complicated for alternatives. Regen is mostly replaced by ILKernelGenerator/DirectILKernelGenerator which use runtime IL emission.

**Q: Why are there TWO classes — `ILKernelGenerator` AND `DirectILKernelGenerator`?**
A: They encode two different kernel-driving contracts. `DirectILKernelGenerator` (59 partials in `Direct/`) emits whole-array kernels: one call processes the entire array; the kernel walks dimensions/strides itself. `ILKernelGenerator` (root) emits per-chunk kernels matching NumPy's `PyUFuncGenericFunction` contract: the iterator (`NpyIterRef`) drives the loop and the kernel only processes one chunk. The split makes the contract explicit in the type name. A kernel lives in whichever class matches how it is driven — `ILKernelGenerator` for kernels run as an NpyIter inner loop (`np.where`, flat/pairwise reductions), `DirectILKernelGenerator` for kernels that own their full-array traversal.

**Q: When should I write kernels in `ILKernelGenerator` vs `DirectILKernelGenerator`?**
A: Match the driving contract. A kernel that runs as the inner loop of an NpyIter pass (the iterator advances the operand pointers, the kernel handles one chunk) belongs in `ILKernelGenerator`. A kernel that takes the whole array plus its shape/strides and walks it itself belongs in `DirectILKernelGenerator`.

**Q: Why is TensorEngine abstracted?**
A: To support potential future backends (GPU/CUDA, SIMD intrinsics, MKL/BLAS). Not implemented yet, but the architecture allows it.

**Q: How closely does the API match NumPy?**
A: Goal is as close as possible - all edge cases included (NaN handling, multi-type operations, broadcasting). Target is NumPy 2.x.

**Q: Does np.random match NumPy's random state/seed behavior?**
A: Yes, 1-to-1 matching.

**Q: What are the primary use cases?**
A: Anything that can use the capabilities - porting Python ML code, standalone .NET scientific computing, integration with TensorFlow.NET/ML.NET.

**Q: Are there areas of known fragility?**
A: Slicing/broadcasting system is complex — offset/stride calculations with contiguity detection require careful handling. The `readonly struct Shape` with `ArrayFlags` simplifies this but edge cases remain.

**Q: How is NumPy compatibility validated?**
A: Written by hand based on NumPy docs and original tests. Testing philosophy: run actual NumPy code, observe output, replicate 1-to-1 in C#.

**Q: What's the pattern for adding new np.* functions?**
A: Sometimes uses other np functions (no DefaultEngine needed). Sometimes requires DefaultEngine for optimization. Tests should be based on actually running NumPy code and imitating the outcome.

**Q: Are breaking changes acceptable?**
A: Yes - breaking changes are accepted to align with NumPy 2.x behavior.

**Q: What needs the most work?**
A: Implementations that differ from NumPy 2.x behavior. See the Missing Functions section.

---

## Q&A - Core Components

**Q: What are the three pillars of NumSharp?**
A: `NDArray` (user-facing API), `UnmanagedStorage` (raw memory management), and `Shape` (dimensions, strides, coordinate translation). They work together: NDArray wraps Storage which uses Shape for offset calculations.

**Q: What is Shape responsible for?**
A: Shape is a `readonly struct` containing dimensions, strides, offset, bufferSize, and cached `ArrayFlags`. Key properties: `IsScalar`, `IsContiguous`, `IsSliced`, `IsBroadcasted`, `IsWriteable`, `IsSimpleSlice`. Methods: `GetOffset(coords)`, `GetCoordinates(offset)`. NumPy-aligned: broadcast views are read-only (`IsWriteable = false`).

**Q: How does slicing work internally?**
A: The `Slice` class parses Python notation (e.g., "1:5:2") into `Start`, `Stop`, `Step`. It converts to `SliceDef` (absolute indices) for computation. `SliceDef.Merge()` handles recursive slicing (slice of a slice).

**Q: What are the special Slice instances?**
A: `Slice.All` (`:` - all elements), `Slice.Ellipsis` (`...` - fill dimensions), `Slice.NewAxis` (insert dimension), `Slice.Index(n)` (single element, reduces dimensionality).

**Q: What is NpyIter?**
A: The NumPy-aligned multi-operand iterator. It handles C/F/A/K order, broadcasting, external loops, buffering, casting, masks, reductions, and synchronized traversal for copy and elementwise kernels. Copy and multi-operand execution go through `NpyIter.Copy` and the multi-operand iterator.

**Q: How does broadcasting work?**
A: Shapes align from the right. Dimensions must be equal OR one must be 1. Dimension of 1 "stretches" to match. Implemented via `DefaultEngine.Broadcast()` which resolves compatible shapes.

**Q: What is InfoOf<T>?**
A: Static type information cache to avoid runtime reflection. Provides `InfoOf<T>.Size` (bytes), `InfoOf<T>.NPTypeCode`, `InfoOf<T>.Zero`, `InfoOf<T>.MaxValue/MinValue`.

**Q: What is NDArray<T>?**
A: Generic typed wrapper providing type-safe access. Returns `T` from indexer instead of NDArray. Has typed `Address` pointer (`T*`) and `Array` property (`ArraySlice<T>`).

**Q: When does DefaultEngine use parallelization?**
A: Parallelization is minimal. Most operations use SIMD vectorization instead for performance.

---

## Q&A - Operations & Operators

**Q: How do arithmetic operators work?**
A: All operators (`+`, `-`, `*`, `/`, `%`, unary `-`) are defined in `NDArray.Primitive.cs`. They delegate to `TensorEngine.Add()`, `Subtract()`, etc. Scalar operands are wrapped via `NDArray.Scalar()`.

**Q: How do comparison operators work?**
A: Element-wise comparisons (`==`, `!=`, `>`, `<`, etc.) return `NDArray<bool>`. Defined in `NDArray.Equals.cs`, `NDArray.Greater.cs`, etc. Support broadcasting.

**Q: What indexing modes are supported?**
A: Integer indices, string slices (`"1:3, :"`), Slice objects, boolean masks, fancy indexing (NDArray<int> indices), and mixed combinations. All in `Selection/NDArray.Indexing*.cs`.

**Q: How is linear algebra implemented?**
A: Core ops (`dot`, `matmul`, `outer`) in `LinearAlgebra/`; `trace`/`diagonal` in `Indexing/`. Advanced decompositions (`inv`, `qr`, `svd`, `lstsq`) are stub methods that return null/default — there is no LAPACK backend.

---

## Q&A - Development

**Q: What's in the test suite?**
A: MSTest v3 framework in `test/NumSharp.UnitTest/`. Many tests adapted from NumPy's own test suite. Decent coverage but gaps in edge cases. Uses source-generated test discovery (no special flags needed).

**Q: What .NET version is targeted?**
A: Library multi-targets `net8.0` and `net10.0`. Tests also multi-target both frameworks.

**Q: What are the main dependencies?**
A: No external runtime dependencies. `System.Memory` and `System.Runtime.CompilerServices.Unsafe` (previously NuGet packages) are built into the .NET 8+ runtime.

**Q: What projects use NumSharp?**
A: TensorFlow.NET, ML.NET integrations, Gym.NET, Pandas.NET, and various scientific computing projects.

**Q: Can I save/load NumPy files?**
A: Yes. `np.save()` writes `.npy` files, `np.load()` reads both `.npy` and `.npz` archives. Compatible with Python NumPy files.

**Q: What random distributions are supported?**
A: An extensive NumPy-matching set (40+ distributions and samplers — see the Supported np.* APIs → Random list), all on the `NumPyRandom` class in `RandomSampling/`, with 1-to-1 seed/state parity to NumPy.

---

## Detailed Documentation

@ARCHITECTURE.md for comprehensive technical details and @CONTRIBUTING.md for development workflow.
