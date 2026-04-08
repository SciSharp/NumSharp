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
| Unmanaged memory | Benchmarked fastest; optimized for performance |
| C-order only | Only row-major (C-order) memory layout. Uses `ArrayFlags.C_CONTIGUOUS` flag. No F-order/column-major support. The `order` parameter on `ravel`, `flatten`, `copy`, `reshape` is accepted but ignored. |
| Regen templating | Type-specific code generation (legacy, mostly replaced by ILKernel) |
| TensorEngine abstract | Future GPU/SIMD backends possible |
| View semantics | Slicing returns views (shared memory), not copies |
| Shape readonly struct | Immutable after construction (NumPy-aligned). Contains `ArrayFlags` for cached O(1) property access |
| Broadcast write protection | Broadcast views are read-only (`IsWriteable = false`), matching NumPy behavior |
| ILKernelGenerator | Runtime IL emission (~18K lines) with SIMD V128/V256/V512; replaces Regen templates |

## ILKernelGenerator

Runtime IL generation via `System.Reflection.Emit.DynamicMethod` for high-performance kernels.

**Partial Class Structure (27 files):**
| Category | Files |
|----------|-------|
| Core | `ILKernelGenerator.cs` (type mapping, SIMD detection), `.Scalar.cs` |
| Binary | `.Binary.cs`, `.MixedType.cs`, `.Shift.cs` |
| Unary | `.Unary.cs`, `.Unary.Math.cs`, `.Unary.Decimal.cs`, `.Unary.Vector.cs`, `.Unary.Predicate.cs` |
| Comparison | `.Comparison.cs` |
| Reduction | `.Reduction.cs`, `.Reduction.Arg.cs`, `.Reduction.Boolean.cs`, `.Reduction.Axis.cs`, `.Reduction.Axis.Arg.cs`, `.Reduction.Axis.Simd.cs`, `.Reduction.Axis.NaN.cs`, `.Reduction.Axis.VarStd.cs` |
| Scan | `.Scan.cs` (CumSum, CumProd) |
| Masking | `.Masking.cs`, `.Masking.Boolean.cs`, `.Masking.NaN.cs`, `.Masking.VarStd.cs` |
| Other | `.Clip.cs`, `.Modf.cs`, `.MatMul.cs` |

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

## Missing Functions (20)

These NumPy functions are **not implemented**:

| Category | Functions |
|----------|-----------|
| Sorting | `np.sort` |
| Selection | `np.where` |
| Manipulation | `np.flip`, `np.fliplr`, `np.flipud`, `np.rot90`, `np.tile`, `np.pad` |
| Splitting | `np.split`, `np.array_split`, `np.hsplit`, `np.vsplit`, `np.dsplit` |
| Diagonal | `np.diag`, `np.diagonal`, `np.trace` |
| Cumulative | `np.diff`, `np.gradient`, `np.ediff1d` |
| Rounding | `np.round` (use `np.round_` or `np.around`) |

---

## Supported np.* APIs

Tested against NumPy 2.x.

### Array Creation
`arange`, `array`, `asanyarray`, `asarray`, `copy`, `empty`, `empty_like`, `eye`, `frombuffer`, `full`, `full_like`, `identity`, `linspace`, `meshgrid`, `mgrid`, `ones`, `ones_like`, `zeros`, `zeros_like`

### Shape Manipulation
`atleast_1d`, `atleast_2d`, `atleast_3d`, `concatenate`, `dstack`, `expand_dims`, `flatten`, `hstack`, `moveaxis`, `ravel`, `repeat`, `reshape`, `roll`, `rollaxis`, `squeeze`, `stack`, `swapaxes`, `transpose`, `unique`, `vstack`

### Broadcasting
`are_broadcastable`, `broadcast`, `broadcast_arrays`, `broadcast_to`

### Math — Arithmetic
`abs`, `absolute`, `add`, `cbrt`, `ceil`, `clip`, `convolve`, `divide`, `exp`, `exp2`, `expm1`, `floor`, `floor_divide`, `log`, `log10`, `log1p`, `log2`, `mod`, `modf`, `multiply`, `negative`, `positive`, `power`, `reciprocal`, `sign`, `sin`, `cos`, `tan`, `sqrt`, `square`, `subtract`, `true_divide`, `trunc`

### Math — Reductions
`all`, `amax`, `amin`, `any`, `argmax`, `argmin`, `count_nonzero`, `cumprod`, `cumsum`, `max`, `mean`, `min`, `prod`, `std`, `sum`, `var`

### Math — NaN-Aware
`nanmax`, `nanmean`, `nanmin`, `nanprod`, `nanstd`, `nansum`, `nanvar`

### Bitwise
`bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert`, `left_shift`, `right_shift`

### Comparison & Logic
`all`, `allclose`, `any`, `array_equal`, `find_common_type`, `isclose`, `isfinite`, `isinf`, `isnan`, `isscalar`, `maximum`, `minimum`

### Sorting & Searching
`argmax`, `argmin`, `argsort`, `nonzero`, `searchsorted`

### Linear Algebra
`dot`, `matmul`, `outer`

### Random (`np.random.*`)
`bernoulli`, `beta`, `binomial`, `chisquare`, `choice`, `exponential`, `gamma`, `geometric`, `lognormal`, `normal`, `permutation`, `poisson`, `rand`, `randint`, `randn`, `seed`, `shuffle`, `standard_normal`, `uniform`

### File I/O
`fromfile`, `load`, `save`, `tofile`

### Other
`around`, `copyto`, `round_`, `size`

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
- `dotnet_test_tunit --filter "..."`: MSTest-style filter for TUnit (Category=, Name~, ClassName~, FullyQualifiedName~)

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
- name: Test (net10.0)
  run: |
    dotnet run --project test/NumSharp.UnitTest/NumSharp.UnitTest.csproj \
      --configuration Release --no-build --framework net10.0 \
      -- --treenode-filter '/*/*/*/*[Category!=OpenBugs]'
```

This filter excludes all tests with `[OpenBugs]` attribute from CI runs. Tests pass locally when the bug is fixed — then remove the `[OpenBugs]` attribute.

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

**OpenBugs files**: `OpenBugs.cs` (general), `OpenBugs.Bitmap.cs` (bitmap), `OpenBugs.ApiAudit.cs` (API audit), `OpenBugs.ILKernelBattle.cs` (IL kernel).

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
A: Benchmarking showed unmanaged memory was fastest. NDArray is self-managed memory allocation optimized for performance.

**Q: Why Regen templating instead of T4 or source generators?**
A: Original needs felt too complicated for alternatives. Regen is mostly replaced by ILKernelGenerator which uses runtime IL emission.

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
A: Core ops (`dot`, `matmul`) in `LinearAlgebra/`. Advanced decompositions (`inv`, `qr`, `svd`, `lstsq`) are stub methods that return null/default — the LAPACK native bindings they depended on have been removed.

---

## Q&A - Development

**Q: What's in the test suite?**
A: TUnit framework in `test/NumSharp.UnitTest/`. Many tests adapted from NumPy's own test suite. Decent coverage but gaps in edge cases. Uses source-generated test discovery (no special flags needed).

**Q: What .NET version is targeted?**
A: Library multi-targets `net8.0` and `net10.0`. Tests require .NET 9+ runtime (TUnit requirement).

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
