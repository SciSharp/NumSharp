# NumSharp Project Instructions

NumSharp is a .NET port of Python's NumPy library targeting **1-to-1 API and behavioral compatibility with NumPy 2.x (latest)**.

## NumPy Reference Source

A full clone of the NumPy repository is available at `src/numpy/`, checked out to **v2.4.2** (latest stable release). Use this as the authoritative reference for API behavior, edge cases, and implementation details when implementing or verifying NumSharp functions.

## Core Principles

1. **Match NumPy Exactly**: Run actual Python/NumPy code first, observe behavior, replicate in C#
2. **Match NumPy Implementation Patterns**: Don't just match behavior - match NumPy's implementation structure. If NumPy has a clean approach and NumSharp has spaghetti code, refactor to match NumPy's design
3. **Edge Cases Matter**: NaN handling, empty arrays, type promotion, broadcasting, negative axis
4. **Breaking Changes OK**: Library was dormant; API stability is not a constraint
5. **Test From NumPy Output**: Tests should be based on running actual NumPy code

**When fixing bugs:** Don't just patch symptoms. Check `src/numpy/` (v2.4.2) for how NumPy implements the same functionality, then refactor NumSharp to match NumPy's structure.

## Definition of Done (DOD) - Operations

Every np.* function and DefaultEngine operation MUST satisfy these criteria:

### Memory Layout Support
- **Contiguous arrays**: Works correctly with C-contiguous memory (SIMD fast path)
- **Non-contiguous arrays**: Works correctly with sliced/strided/transposed views
- **Broadcast arrays**: Works correctly with stride=0 dimensions (read-only)
- **Sliced views**: Correctly handles Shape.offset for base address calculation

### Dtype Support
All 12 NumSharp types must be handled (or explicitly documented as unsupported):
Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Single, Double, Decimal

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

## Supported Types (12)

| NPTypeCode | C# Type | NPTypeCode | C# Type |
|------------|---------|------------|---------|
| Boolean | bool | Int64 | long |
| Byte | byte | UInt64 | ulong |
| Int16 | short | Char | char |
| UInt16 | ushort | Single | float |
| Int32 | int | Double | double |
| UInt32 | uint | Decimal | decimal |

All operations must handle all 12 types via type switch pattern.

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
| Unmanaged memory | Benchmarked fastest ~5y ago; Span/Memory immature then |
| C-order only | Only row-major (C-order) memory layout. Uses `ArrayFlags.C_CONTIGUOUS` flag. No F-order/column-major support. The `order` parameter on `ravel`, `flatten`, `copy`, `reshape` is accepted but ignored. |
| Regen templating | ~200K lines generated for type-specific code |
| TensorEngine abstract | Future GPU/SIMD backends possible |
| View semantics | Slicing returns views (shared memory), not copies |
| Shape readonly struct | Immutable after construction (NumPy-aligned). Contains `ArrayFlags` for cached O(1) property access |
| Broadcast write protection | Broadcast views are read-only (`IsWriteable = false`), matching NumPy behavior |
| ILKernelGenerator | Runtime IL emission replacing ~500K lines of Regen templates; SIMD V128/V256/V512 |

## ILKernelGenerator (0.41.x)

Runtime IL generation via `System.Reflection.Emit.DynamicMethod` for high-performance kernels.

**Partial Class Structure:**
| File | Responsibility |
|------|----------------|
| `ILKernelGenerator.cs` | Core: type mapping, SIMD detection (VectorBits), shared IL primitives |
| `.Binary.cs` | Same-type binary ops (Add, Sub, Mul, Div, BitwiseAnd/Or/Xor) |
| `.MixedType.cs` | Mixed-type binary ops with promotion; owns `ClearAll()` |
| `.Unary.cs` | Math functions (Negate, Abs, Sqrt, Sin, Cos, Exp, Log, Sign, etc.) |
| `.Comparison.cs` | Comparisons (==, !=, <, >, <=, >=) returning bool arrays |
| `.Reduction.cs` | Reductions (Sum, Prod, Min, Max, ArgMax, ArgMin, All, Any) |

**Execution Paths:**
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

**ILKernel Status (0.41.x):**
| Category | Implemented | Pending |
|----------|-------------|---------|
| Binary | Add, Sub, Mul, Div, Power, FloorDivide, BitwiseAnd/Or/Xor | LeftShift, RightShift (use Default.Shift.cs) |
| Unary | Negate, Abs, Sign, Sqrt, Cbrt, Square, Reciprocal, Floor, Ceil, Truncate, Trig, Exp, Log, BitwiseNot | Deg2Rad, Rad2Deg (use DefaultEngine) |
| Reduction | Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any, CumSum | Std, Var (use Regen templates) |
| NaN Reduction | — | NanSum, NanProd, NanMin, NanMax (Task #88) |
| Comparison | Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual | — |
| Clip/Modf | Clip, Modf (SIMD helpers) | — |
| Axis reductions | Uses iterator path (no SIMD) | SIMD axis kernels (Task #89) |

**DefaultEngine ops needing IL migration:**
- High impact: `MatMul`, `Dot` (complex - consider BLAS integration)

## Shape Architecture (NumPy-Aligned)

Shape is a `readonly struct` with cached `ArrayFlags` computed at construction:

```csharp
public readonly partial struct Shape
{
    internal readonly int[] dimensions;  // Dimension sizes
    internal readonly int[] strides;     // Stride values (0 = broadcast dimension)
    internal readonly int offset;        // Base offset into storage
    internal readonly int bufferSize;    // Size of underlying buffer
    internal readonly int _flags;        // Cached ArrayFlags bitmask
}
```

**ArrayFlags enum** (matches NumPy's `ndarraytypes.h`):
| Flag | Value | Meaning |
|------|-------|---------|
| `C_CONTIGUOUS` | 0x0001 | Data is row-major contiguous |
| `F_CONTIGUOUS` | 0x0002 | Reserved (always false for NumSharp) |
| `OWNDATA` | 0x0004 | Array owns its data buffer |
| `ALIGNED` | 0x0100 | Always true for managed allocations |
| `WRITEABLE` | 0x0400 | False for broadcast views |
| `BROADCASTED` | 0x1000 | Has stride=0 with dim > 1 |

**Key Shape properties:**
- `IsContiguous` — O(1) check via `C_CONTIGUOUS` flag
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

## Known Issues & Bugs

### Critical Bugs (Fundamentally Wrong Output)

| Bug | Function | Issue | Workaround |
|-----|----------|-------|------------|
| BUG-21 | `np.arange`/`np.sum` | int32 default dtype, no overflow protection | Specify `dtype: np.int64` |

### Medium Severity Bugs

| Bug | Function | Issue |
|-----|----------|-------|
| BUG-12 | `np.searchsorted` | Scalar input throws IndexOutOfRangeException |
| BUG-16 | `np.moveaxis` | Returns unchanged shape |
| BUG-17 | `nd.astype()` | Uses rounding instead of truncation for float->int |

### Low Severity (Behavioral Differences)

| Bug | Function | Issue |
|-----|----------|-------|
| BUG-14 | `np.unique` | Doesn't sort results (NumPy sorts) |
| BUG-4 | `np.std`/`np.var` | ddof parameter ignored |
| BUG-7 | sbyte (int8) | Type not supported |

### Fixed Bugs (0.41.x)

These bugs were fixed in recent commits:

| Bug | Function | Fix Commit |
|-----|----------|------------|
| BUG-19 | `np.negative` | `0857d109` — was applying abs() then negating |
| BUG-20 | `np.positive` | `0857d109` — was applying abs() instead of identity |
| BUG-18 | `np.convolve` | `0857d109` — NullReferenceException fixed |
| BUG-15 | `np.abs` | `0857d109` — int dtype preserved (no longer converts to Double) |
| BUG-13 | `np.linspace` | `0857d109` — returns float64 (was float32) |

### Dead Code (Returns null/default)

These functions exist but are non-functional:

| Function | Issue |
|----------|-------|
| `np.isnan`, `np.isfinite`, `np.isclose` | DefaultEngine returns null |
| `np.allclose` | Depends on np.isclose |
| `np.linalg.inv`, `qr`, `svd`, `lstsq` | LAPACK bindings removed |
| `nd.delete()`, `nd.multi_dot()` | Return null |
| `operator &`, `operator \|` | Fixed in 0.41.x via ILKernel |

---

## Missing Functions (22)

These NumPy functions are **not implemented**:

| Category | Functions |
|----------|-----------|
| Sorting | `np.sort` |
| Selection | `np.where` |
| Manipulation | `np.flip`, `np.fliplr`, `np.flipud`, `np.rot90`, `np.tile`, `np.pad` |
| Splitting | `np.split`, `np.array_split`, `np.hsplit`, `np.vsplit`, `np.dsplit` |
| Diagonal | `np.diag`, `np.diagonal`, `np.trace` |
| Cumulative | `np.cumprod`, `np.diff`, `np.gradient`, `np.ediff1d` |
| Counting | `np.count_nonzero` |
| Rounding | `np.round` (use `np.around` instead) |

---

## Verified Working Functions (75+)

Tested against NumPy 2.4.2 (2703 tests, 97.5% pass rate):

**Array Creation:** `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `eye`, `identity`, `meshgrid`, `copy`, `zeros_like`, `ones_like`, `empty_like`, `full_like`, `array`, `asarray`

**Shape Manipulation:** `reshape`, `transpose`, `ravel`, `flatten`, `squeeze`, `expand_dims`, `swapaxes`, `rollaxis`, `atleast_1d/2d/3d`, `stack`, `vstack`, `hstack`, `dstack`, `concatenate`

**Math Operations:** `add`, `subtract`, `multiply`, `divide`, `mod`, `power`, `sqrt`, `abs`, `sign`, `floor`, `ceil`, `sin`, `cos`, `tan`, `exp`, `log`, `log2`, `log10`, `log1p`, `expm1`, `exp2`, `around`

**Reductions:** `sum`, `prod`, `mean`, `std`, `var`, `max`, `min`, `argmax`, `argmin`, `all`, `any`, `cumsum`

**Comparisons:** `==`, `!=`, `<`, `>`, `<=`, `>=`, `array_equal`

**Linear Algebra:** `dot`, `matmul`, `outer`

**Random:** `seed`, `rand`, `randn`, `randint`, `uniform`, `choice`, `shuffle`, `permutation`

**File I/O:** `save`/`load` (.npy), `tofile`/`fromfile`

**Other:** `clip`, `roll`, `repeat`, `argsort`, `copyto`, `broadcast_to`, `modf`, `nonzero`, `unique`

---

## Capabilities Reference

### Array Creation (`Creation/`)
| Function | File |
|----------|------|
| `np.array` | `np.array.cs` |
| `np.zeros`, `np.zeros_like` | `np.zeros.cs`, `np.zeros_like.cs` |
| `np.ones`, `np.ones_like` | `np.ones.cs`, `np.ones_like.cs` |
| `np.empty`, `np.empty_like` | `np.empty.cs`, `np.empty_like.cs` |
| `np.full`, `np.full_like` | `np.full.cs`, `np.full_like.cs` |
| `np.arange` | `np.arange.cs` |
| `np.linspace` | `np.linspace.cs` |
| `np.eye` | `np.eye.cs` |
| `np.meshgrid`, `np.mgrid` | `np.meshgrid.cs`, `np.mgrid.cs` |
| `np.copy` | `np.copy.cs` |
| `np.asarray`, `np.asanyarray` | `np.asarray.cs`, `np.asanyarray.cs` |
| `np.frombuffer` | `np.frombuffer.cs` |

### Stacking & Joining (`Creation/`)
| Function | File |
|----------|------|
| `np.concatenate` | `np.concatenate.cs` |
| `np.stack` | `np.stack.cs` |
| `np.hstack` | `np.hstack.cs` |
| `np.vstack` | `np.vstack.cs` |
| `np.dstack` | `np.dstack.cs` |

### Broadcasting (`Creation/`)
| Function | File |
|----------|------|
| `np.broadcast` | `np.broadcast.cs` |
| `np.broadcast_to` | `np.broadcast_to.cs` |
| `np.broadcast_arrays` | `np.broadcast_arrays.cs` |
| `np.are_broadcastable` | `np.are_broadcastable.cs` |

### Math Functions (`Math/`)
| Function | File |
|----------|------|
| `np.add`, `np.subtract`, `np.multiply`, `np.divide` | `np.math.cs` |
| `np.mod`, `np.true_divide` | `np.math.cs` |
| `np.positive`, `np.negative`, `np.convolve` | `np.math.cs` |
| `np.sum` | `np.sum.cs` |
| `np.prod`, `nd.prod()` | `np.math.cs`, `NDArray.prod.cs` |
| `np.cumsum`, `nd.cumsum()` | `APIs/np.cumsum.cs`, `Math/NDArray.cumsum.cs` |
| `np.power` | `np.power.cs` |
| `np.sqrt` | `np.sqrt.cs` |
| `np.abs`, `np.absolute` | `np.absolute.cs` |
| `np.sign` | `np.sign.cs` |
| `np.floor`, `np.ceil` | `np.floor.cs`, `np.ceil.cs` |
| `np.round` | `np.round.cs` |
| `np.clip` | `np.clip.cs` |
| `np.modf` | `np.modf.cs` |
| `np.maximum`, `np.minimum` | `np.maximum.cs`, `np.minimum.cs` |
| `np.log`, `np.log2`, `np.log10`, `np.log1p` | `np.log.cs` |
| `np.exp`, `np.exp2`, `np.expm1` | `Statistics/np.exp.cs` |
| `np.sin`, `np.cos`, `np.tan` | `np.sin.cs`, `np.cos.cs`, `np.tan.cs` |

### Statistics (`Statistics/`)
| Function | File |
|----------|------|
| `np.mean`, `nd.mean()` | `np.mean.cs`, `NDArray.mean.cs` |
| `np.std`, `nd.std()` | `np.std.cs`, `NDArray.std.cs` |
| `np.var`, `nd.var()` | `np.var.cs`, `NDArray.var.cs` |
| `np.amax`, `nd.amax()` | `Sorting_Searching_Counting/np.amax.cs`, `NDArray.amax.cs` |
| `np.amin`, `nd.amin()` | `Sorting_Searching_Counting/np.min.cs`, `NDArray.amin.cs` |
| `np.argmax`, `nd.argmax()` | `Sorting_Searching_Counting/np.argmax.cs`, `NDArray.argmax.cs` |
| `np.argmin`, `nd.argmin()` | `Sorting_Searching_Counting/np.argmax.cs`, `NDArray.argmin.cs` |

### Sorting & Searching (`Sorting_Searching_Counting/`)
| Function | File |
|----------|------|
| `np.argsort`, `nd.argsort()` | `np.argsort.cs`, `ndarray.argsort.cs` |
| `np.searchsorted` | `np.searchsorted.cs` |

### Linear Algebra (`LinearAlgebra/`)
| Function | File |
|----------|------|
| `np.dot`, `nd.dot()` | `np.dot.cs`, `NDArray.dot.cs` |
| `np.matmul` | `np.matmul.cs` |
| `np.outer` | `np.outer.cs` |
| ~~`np.linalg.norm`~~ | `np.linalg.norm.cs` | **DEAD CODE**: declared `private static` — not accessible |
| `nd.matrix_power()` | `NDArray.matrix_power.cs` | |
| ~~`nd.inv()`~~ | `NdArray.Inv.cs` | **DEAD CODE**: returns null |
| ~~`nd.qr()`~~ | `NdArray.QR.cs` | **DEAD CODE**: returns default |
| ~~`nd.svd()`~~ | `NdArray.SVD.cs` | **DEAD CODE**: returns default |
| ~~`nd.lstsq()`~~ | `NdArray.LstSq.cs` | **DEAD CODE**: named `lstqr`, returns null |
| ~~`nd.multi_dot()`~~ | `NdArray.multi_dot.cs` | **DEAD CODE**: returns null |

### Shape Manipulation (`Manipulation/`)
| Function | File |
|----------|------|
| `np.reshape`, `nd.reshape()` | `np.reshape.cs` |
| `np.transpose`, `nd.T` | `np.transpose.cs`, `NdArray.Transpose.cs` |
| `np.ravel`, `nd.ravel()` | `np.ravel.cs`, `NDArray.ravel.cs` |
| `nd.flatten()` | `NDArray.flatten.cs` |
| `np.squeeze` | `np.squeeze.cs` |
| `np.expand_dims` | `np.expand_dims.cs` |
| `np.swapaxes` | `np.swapaxes.cs`, `NdArray.swapaxes.cs` |
| `np.moveaxis` | `np.moveaxis.cs` |
| `np.rollaxis` | `np.rollaxis.cs` |
| `np.roll`, `nd.roll()` | `np.roll.cs`, `NDArray.roll.cs` | Fully implemented (all dtypes, with/without axis) |
| `np.atleast_1d/2d/3d` | `np.atleastd.cs` |
| `np.unique`, `nd.unique()` | `np.unique.cs`, `NDArray.unique.cs` |
| `np.repeat` | `np.repeat.cs` |
| ~~`nd.delete()`~~ | `NdArray.delete.cs` | **DEAD CODE**: returns null |
| `np.copyto` | `np.copyto.cs` |

### Logic Functions (`Logic/`)
| Function | File | Notes |
|----------|------|-------|
| `np.all` | `np.all.cs` | All dtypes; SIMD optimized with early-exit |
| `np.any` | `np.any.cs` | All dtypes; SIMD optimized with early-exit |
| ~~`np.allclose`~~ | `np.allclose.cs` | **DEAD CODE**: depends on `np.isclose` which returns null |
| `np.array_equal` | `np.array_equal.cs` | |
| `np.isscalar` | `np.is.cs` | |
| ~~`np.isnan`~~ | `np.is.cs` | **DEAD CODE**: `DefaultEngine.IsNan` returns null |
| ~~`np.isfinite`~~ | `np.is.cs` | **DEAD CODE**: `DefaultEngine.IsFinite` returns null |
| ~~`np.isclose`~~ | `np.is.cs` | **DEAD CODE**: `DefaultEngine.IsClose` returns null |
| `np.find_common_type` | `np.find_common_type.cs` | |

### Comparison Operators (`Operations/Elementwise/`)
| Operator | File | Notes |
|----------|------|-------|
| `==` (element-wise) | `NDArray.Equals.cs` | SIMD optimized |
| `!=` | `NDArray.NotEquals.cs` | SIMD optimized |
| `>`, `>=` | `NDArray.Greater.cs` | SIMD optimized |
| `<`, `<=` | `NDArray.Lower.cs` | SIMD optimized |
| `&` (AND) | `NDArray.AND.cs` | Fixed in 0.41.x via ILKernel |
| `\|` (OR) | `NDArray.OR.cs` | Fixed in 0.41.x via ILKernel |
| `!` (NOT) | `NDArray.NOT.cs` | |

### Arithmetic Operators (`Operations/Elementwise/`)
| Operator | File |
|----------|------|
| `+`, `-`, `*`, `/`, `%`, unary `-` | `NDArray.Primitive.cs` |

### Indexing & Selection (`Selection/`)
| Feature | File |
|---------|------|
| Integer/slice indexing | `NDArray.Indexing.cs` |
| Boolean masking | `NDArray.Indexing.Masking.cs` | Read works; setter throws NotImplementedException |
| Fancy indexing (NDArray indices) | `NDArray.Indexing.Selection.cs` |
| `np.nonzero` | `Indexing/np.nonzero.cs` |

### Random Sampling (`RandomSampling/`)
| Function | File |
|----------|------|
| `np.random.rand` | `np.random.rand.cs` |
| `np.random.randn` | `np.random.randn.cs` |
| `np.random.randint` | `np.random.randint.cs` |
| `np.random.uniform` | `np.random.uniform.cs` |
| `np.random.choice` | `np.random.choice.cs` |
| `np.random.shuffle` | `np.random.shuffle.cs` |
| `np.random.permutation` | `np.random.permutation.cs` |
| `np.random.beta` | `np.random.beta.cs` |
| `np.random.binomial` | `np.random.binomial.cs` |
| `np.random.gamma` | `np.random.gamma.cs` |
| `np.random.poisson` | `np.random.poisson.cs` |
| `np.random.exponential` | `np.random.exponential.cs` |
| `np.random.geometric` | `np.random.geometric.cs` |
| `np.random.lognormal` | `np.random.lognormal.cs` |
| `np.random.chisquare` | `np.random.chisquare.cs` |
| `np.random.bernoulli` | `np.random.bernoulli.cs` |

### File I/O (`APIs/`)
| Function | File |
|----------|------|
| `np.save` (`.npy`) | `np.save.cs` |
| `np.load` (`.npy`, `.npz`) | `np.load.cs` |
| `np.fromfile` | `np.fromfile.cs` |
| `nd.tofile()` | `np.tofile.cs` |

### Other APIs (`APIs/`)
| Function | File |
|----------|------|
| `np.size` | `np.size.cs` |

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
| Iterators | `Backends/Iterators/NDIterator.cs`, `MultiIterator.cs` |
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
    case NPTypeCode.Int16: return Process<short>(nd);
    case NPTypeCode.UInt16: return Process<ushort>(nd);
    case NPTypeCode.Int32: return Process<int>(nd);
    case NPTypeCode.UInt32: return Process<uint>(nd);
    case NPTypeCode.Int64: return Process<long>(nd);
    case NPTypeCode.UInt64: return Process<ulong>(nd);
    case NPTypeCode.Char: return Process<char>(nd);
    case NPTypeCode.Double: return Process<double>(nd);
    case NPTypeCode.Single: return Process<float>(nd);
    case NPTypeCode.Decimal: return Process<decimal>(nd);
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

## Build & Test

```bash
# Build (silent, errors only)
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
```

### Running Tests

Tests use **TUnit** framework with source-generated test discovery.

```bash
# Run from test directory
cd test/NumSharp.UnitTest

# All tests (includes OpenBugs - expected failures)
dotnet test --no-build

# Exclude OpenBugs (CI-style - only real failures)
dotnet test --no-build -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"

# Run ONLY OpenBugs tests
dotnet test --no-build -- --treenode-filter "/*/*/*/*[Category=OpenBugs]"
```

### Output Formatting

```bash
# Results only (no messages, no stack traces)
dotnet test --no-build 2>&1 | grep -E "^(failed|skipped|Test run|  total:|  failed:|  succeeded:|  skipped:|  duration:)"

# Results with messages (no stack traces)
dotnet test --no-build 2>&1 | grep -v "^    at " | grep -v "^     at " | grep -v "^    ---" | grep -v "^  from K:" | sed 's/TUnit.Engine.Exceptions.TestFailedException: //' | sed 's/AssertFailedException: //'

# Detailed output (shows passed tests too)
dotnet test --no-build -- --output Detailed
```

## Test Categories

Tests use typed category attributes defined in `TestCategory.cs`. Adding new bug reproductions or platform-specific tests only requires the right attribute — no CI workflow changes.

| Category | Attribute | Purpose | CI Behavior |
|----------|-----------|---------|-------------|
| `OpenBugs` | `[OpenBugs]` | Known-failing bug reproductions. Remove when fixed. | **EXCLUDED** via filter |
| `Misaligned` | `[Misaligned]` | Documents NumSharp vs NumPy behavioral differences. | Runs (tests pass) |
| `WindowsOnly` | `[WindowsOnly]` | Requires GDI+/System.Drawing.Common | Runtime platform check |

### How CI Excludes OpenBugs

The CI pipeline (`.github/workflows/build-and-release.yml`) uses TUnit's `--treenode-filter` to exclude `OpenBugs`:

```yaml
env:
  TEST_FILTER: '/*/*/*/*[Category!=OpenBugs]'

- name: Test
  run: dotnet run ... -- --treenode-filter "${{ env.TEST_FILTER }}"
```

This filter excludes all tests with `[OpenBugs]` or `[Category("OpenBugs")]` from CI runs. Tests pass locally when the bug is fixed — then remove the `[OpenBugs]` attribute.

### Usage

```csharp
// Class-level (all tests in class)
[OpenBugs]
public class BroadcastBugTests { ... }

// Method-level
[Test]
[OpenBugs]
public async Task BroadcastWriteCorruptsData() { ... }

// Documenting behavioral differences (NOT excluded from CI)
[Test]
[Misaligned]
public void BroadcastSlice_MaterializesInNumSharp() { ... }
```

### Local Filtering

```bash
# Exclude OpenBugs (same as CI)
dotnet test -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"

# Run ONLY OpenBugs tests (to verify fixes)
dotnet test -- --treenode-filter "/*/*/*/*[Category=OpenBugs]"

# Run ONLY Misaligned tests
dotnet test -- --treenode-filter "/*/*/*/*[Category=Misaligned]"
```

**OpenBugs files**: `OpenBugs.cs` (general bugs), `OpenBugs.Bitmap.cs` (bitmap bugs), `OpenBugs.ApiAudit.cs` (API audit failures).

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
6. Handle all 12 dtypes

---

## Q&A - Design & Architecture

**Q: Why unmanaged memory instead of Span<T>/Memory<T>?**
A: Extensive benchmarking ~5 years ago showed unmanaged memory was fastest. Span/Memory weren't mature across the .NET ecosystem then. NDArray is self-managed memory allocation optimized for performance.

**Q: Why Regen templating instead of T4 or source generators?**
A: Original needs felt too complicated for alternatives. Migration to T4 is possible but not prioritized. The ~200K lines of generated code is acceptable if it works correctly.

**Q: Why is TensorEngine abstracted?**
A: To support potential future backends (GPU/CUDA, SIMD intrinsics, MKL/BLAS). Not implemented yet, but the architecture allows it.

**Q: How closely does the API match NumPy?**
A: Goal is as close as possible - all edge cases included (NaN handling, multi-type operations, broadcasting). Target is NumPy 2.x (latest version), upgraded from original 1.x target.

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
A: Implementations that differ from original NumPy 2.x behavior. A comprehensive API mapping expedition (NumSharp vs NumPy 2.x) is planned to identify: what exists, what's missing, what has behavioral differences.

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

**Q: What is NDIterator used for?**
A: Traversing arrays with different memory layouts. Handles contiguous (fast pointer increment) and sliced (uses GetOffset) arrays. Has `MoveNext()`, `HasNext()`, `Reset()`. AutoReset mode for broadcasting smaller arrays.

**Q: What is MultiIterator?**
A: Handles paired iteration for broadcasting. `MultiIterator.Assign(lhs, rhs)` copies with broadcasting. `GetIterators(lhs, rhs, broadcast)` creates synchronized iterators.

**Q: How does broadcasting work?**
A: Shapes align from the right. Dimensions must be equal OR one must be 1. Dimension of 1 "stretches" to match. Implemented via `DefaultEngine.Broadcast()` which resolves compatible shapes.

**Q: What is InfoOf<T>?**
A: Static type information cache to avoid runtime reflection. Provides `InfoOf<T>.Size` (bytes), `InfoOf<T>.NPTypeCode`, `InfoOf<T>.Zero`, `InfoOf<T>.MaxValue/MinValue`.

**Q: What is NDArray<T>?**
A: Generic typed wrapper providing type-safe access. Returns `T` from indexer instead of NDArray. Has typed `Address` pointer (`T*`) and `Array` property (`ArraySlice<T>`).

**Q: When does DefaultEngine use parallelization?**
A: For arrays exceeding 85,000 elements (`ParallelAbove = 84999`). Uses `Parallel.For` for large arrays, sequential loop for smaller ones.

---

## Q&A - Operations & Operators

**Q: How do arithmetic operators work?**
A: All operators (`+`, `-`, `*`, `/`, `%`, unary `-`) are defined in `NDArray.Primitive.cs`. They delegate to `TensorEngine.Add()`, `Subtract()`, etc. Scalar operands are wrapped via `NDArray.Scalar()`.

**Q: How do comparison operators work?**
A: Element-wise comparisons (`==`, `!=`, `>`, `<`, etc.) return `NDArray<bool>`. Defined in `NDArray.Equals.cs`, `NDArray.Greater.cs`, etc. Support broadcasting.

**Q: What indexing modes are supported?**
A: Integer indices, string slices (`"1:3, :"`), Slice objects, boolean masks, fancy indexing (NDArray<int> indices), and mixed combinations. All in `Selection/NDArray.Indexing*.cs`.

**Q: How is linear algebra implemented?**
A: Core ops (`dot`, `matmul`) in `LinearAlgebra/`. Advanced decompositions (`inv`, `qr`, `svd`, `lstsq`) are stub methods that return null/default — the LAPACK native bindings they depended on have been removed.

---

## Q&A - Development

**Q: What's in the test suite?**
A: TUnit framework in `test/NumSharp.UnitTest/`. Many tests adapted from NumPy's own test suite. Decent coverage but gaps in edge cases. Uses source-generated test discovery (no special flags needed).

**Q: What .NET version is targeted?**
A: Library multi-targets `net8.0` and `net10.0`. Tests currently target `net10.0` only (TUnit requires .NET 9+ runtime). Dropped `netstandard2.0` in the dotnet810 branch upgrade.

**Q: What are the main dependencies?**
A: No external runtime dependencies. `System.Memory` and `System.Runtime.CompilerServices.Unsafe` (previously NuGet packages) are built into the .NET 8+ runtime.

**Q: What projects use NumSharp?**
A: TensorFlow.NET, ML.NET integrations, Gym.NET, Pandas.NET, and various scientific computing projects.

**Q: Can I save/load NumPy files?**
A: Yes. `np.save()` writes `.npy` files, `np.load()` reads both `.npy` and `.npz` archives. Compatible with Python NumPy files.

**Q: What random distributions are supported?**
A: Uniform, normal (randn), integers, beta, binomial, gamma, poisson, exponential, geometric, lognormal, chi-square, bernoulli. All in `RandomSampling/`.

---

## Detailed Documentation

@ARCHITECTURE.md for comprehensive technical details and @CONTRIBUTING.md for development workflow.
