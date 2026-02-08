# NumSharp Project Instructions

NumSharp is a .NET port of Python's NumPy library targeting **1-to-1 API and behavioral compatibility with NumPy 2.x (latest)**.

## NumPy Reference Source

A full clone of the NumPy repository is available at `src/numpy/`, checked out to **v2.4.2** (latest stable release). Use this as the authoritative reference for API behavior, edge cases, and implementation details when implementing or verifying NumSharp functions.

## Core Principles

1. **Match NumPy Exactly**: Run actual Python/NumPy code first, observe behavior, replicate in C#
2. **Edge Cases Matter**: NaN handling, empty arrays, type promotion, broadcasting, negative axis
3. **Breaking Changes OK**: Library was dormant; API stability is not a constraint
4. **Test From NumPy Output**: Tests should be based on running actual NumPy code

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
| Regen templating | ~200K lines generated for type-specific code |
| TensorEngine abstract | Future GPU/SIMD backends possible |
| View semantics | Slicing returns views (shared memory), not copies |

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
| `np.sum` | `np.sum.cs` |
| `np.prod` | `NDArray.prod.cs` |
| `np.cumsum` | `NDArray.cumsum.cs` |
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
| `np.amax`, `nd.amax()` | `Sorting/np.amax.cs`, `NDArray.amax.cs` |
| `np.amin`, `nd.amin()` | `Sorting/np.min.cs`, `NDArray.amin.cs` |
| `np.argmax`, `nd.argmax()` | `Sorting/np.argmax.cs`, `NDArray.argmax.cs` |
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
| `nd.roll()` | `NDArray.roll.cs` | Partial: only Int32/Single/Double with axis; no-axis returns null |
| `np.atleast_1d/2d/3d` | `np.atleastd.cs` |
| `np.unique`, `nd.unique()` | `np.unique.cs`, `NDArray.unique.cs` |
| `np.repeat` | `np.repeat.cs` |
| ~~`nd.delete()`~~ | `NdArray.delete.cs` | **DEAD CODE**: returns null |
| `np.copyto` | `np.copyto.cs` |

### Logic Functions (`Logic/`)
| Function | File |
|----------|------|
| `np.all` | `np.all.cs` | All dtypes; with-axis works |
| `np.any` | `np.any.cs` | All dtypes; with-axis **BUGGY** (always throws) |
| ~~`np.allclose`~~ | `np.allclose.cs` | **DEAD CODE**: depends on `np.isclose` which returns null |
| `np.array_equal` | `np.array_equal.cs` | |
| `np.isscalar` | `np.is.cs` | |
| ~~`np.isnan`~~ | `np.is.cs` | **DEAD CODE**: `DefaultEngine.IsNan` returns null |
| ~~`np.isfinite`~~ | `np.is.cs` | **DEAD CODE**: `DefaultEngine.IsFinite` returns null |
| ~~`np.isclose`~~ | `np.is.cs` | **DEAD CODE**: `DefaultEngine.IsClose` returns null |
| `np.find_common_type` | `np.find_common_type.cs` | |

### Comparison Operators (`Operations/Elementwise/`)
| Operator | File |
|----------|------|
| `==` (element-wise) | `NDArray.Equals.cs` |
| `!=` | `NDArray.NotEquals.cs` |
| `>`, `>=` | `NDArray.Greater.cs` |
| `<`, `<=` | `NDArray.Lower.cs` |
| ~~`&` (AND)~~ | `NDArray.AND.cs` | **DEAD CODE**: returns null |
| ~~`\|` (OR)~~ | `NDArray.OR.cs` | **DEAD CODE**: returns null |
| `!` (NOT) | `NDArray.NOT.cs` |

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
| `np.cumsum` | `np.cumsum.cs` |

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
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
dotnet test -v q --nologo "-clp:ErrorsOnly" test/NumSharp.UnitTest/NumSharp.UnitTest.csproj
```

## Test Categories

Tests are filtered by `[TestCategory]` attributes. Adding new bug reproductions or platform-specific tests only requires the right attribute — no CI workflow changes.

| Category | Purpose | CI filter |
|----------|---------|-----------|
| `OpenBugs` | Known-failing bug reproductions. Remove category when fixed. | `TestCategory!=OpenBugs` (all platforms) |
| `WindowsOnly` | Requires GDI+/System.Drawing.Common | `TestCategory!=WindowsOnly` (Linux/macOS) |

Apply at class level (`[TestClass][TestCategory("OpenBugs")]`) or individual method level (`[TestMethod][TestCategory("OpenBugs")]`).

**OpenBugs files**: `OpenBugs.cs` (broadcast bugs), `OpenBugs.Bitmap.cs` (bitmap bugs). When a bug is fixed, the test starts passing — remove the `OpenBugs` category and move to a permanent test class.

## CI Pipeline

`.github/workflows/build-and-release.yml` — test on 3 OSes (Windows/Ubuntu/macOS), build NuGet on tag push, create GitHub Release, publish to nuget.org.
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
| `shape.size` | Total element count |
| `shape.ViewInfo` | Slice/view metadata (null if not a view) |
| `shape.BroadcastInfo` | Broadcast metadata (null if not broadcast) |
| `arr.Storage` | Underlying `UnmanagedStorage` |
| `arr.GetTypeCode` | `NPTypeCode` of the array |
| `arr.Array` | `IArraySlice` — raw data access |
| `np._FindCommonType(...)` | Type promotion logic |
| `np.powerOrder` | Type promotion ordering |
| `NPTypeCode.GetGroup()` | Type category (int/uint/float/etc.) |
| `NPTypeCode.GetPriority()` | Type priority for promotion |
| `NPTypeCode.AsNumpyDtypeName()` | NumPy dtype name (e.g. "int32") |
| `Shape.NewScalar()` | Create scalar shapes |
| `Shape.ComputeHashcode()` | Recalculate shape hash |

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
A: Slicing/broadcasting system is complex with ViewInfo and BroadcastInfo interactions - fragile but working.

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
A: Dimensions, strides, coordinate-to-offset translation, contiguity tracking, and slice/broadcast info. Key properties: `IsScalar`, `IsContiguous`, `IsSliced`, `IsBroadcasted`. Methods: `GetOffset(coords)`, `GetCoordinates(offset)`.

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
A: MSTest framework in `test/NumSharp.UnitTest/`. Many tests adapted from NumPy's own test suite. Decent coverage but gaps in edge cases.

**Q: What .NET version is targeted?**
A: Library and tests multi-target `net8.0` and `net10.0`. Dropped `netstandard2.0` in the dotnet810 branch upgrade.

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
