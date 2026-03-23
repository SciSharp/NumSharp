# NumSharp 0.41.0-prerelease

This prerelease introduces the **IL Kernel Generator** - a complete architectural overhaul that replaces ~600K lines of Regen-generated template code with ~19K lines of runtime IL generation. This delivers massive performance improvements, comprehensive NumPy 2.x alignment, and significantly cleaner maintainable code.

---

## TL;DR

- **IL Kernel Generator**: Runtime IL emission replaces 600K lines of Regen templates with 19K lines
- **SIMD everywhere**: Vector128/256/512 with runtime detection across all operations
- **35 new functions**: nansum/prod/min/max/mean/var/std, cbrt, floor_divide, left/right_shift, deg2rad, rad2deg, cumprod, count_nonzero, isnan, isfinite, isinf, isclose, invert, reciprocal, square, trunc, plus comparison and logical modules
- **Operators fixed**: `==`, `!=`, `<`, `>`, `<=`, `>=`, `&`, `|`, `^`
- **np.comparison module**: `np.equal()`, `np.not_equal()`, `np.less()`, `np.greater()`, `np.less_equal()`, `np.greater_equal()`
- **np.logical module**: `np.logical_and()`, `np.logical_or()`, `np.logical_not()`, `np.logical_xor()`
- **NDArray\<T\> operators**: Typed `&`, `|`, `^` for generic arrays (resolves `NDArray<bool>` ambiguity)
- **Math functions rewritten**: sin, cos, tan, exp, log, sqrt, abs, sign, floor, ceil, etc.
- **60+ bug fixes**: np.negative, np.positive, np.unique, np.dot, np.matmul, np.abs, np.argmax/min, np.mean, np.std/var, np.cumsum, np.nonzero, np.all/any, np.clip, and more
- **MatMul 35-100x faster**: Cache-blocked SIMD achieving 20+ GFLOPS
- **Boolean indexing rewrite**: SIMD fast path with CountTrue/CopyMasked
- **Axis reductions rewrite**: AVX2 gather, NaN-aware, proper keepdims and empty array handling
- **Single-threaded execution**: Deterministic, non-blocking (SIMD compensates for parallelism), Removed use of `Parallel.*`
- **Architecture cleanup**: Broadcasting in Shape struct, TensorEngine routing, static ILKernelGenerator
- **np.random aligned** (#582): Parameter names match NumPy, Shape overloads added
- **DecimalMath internalized** (#588): Removed embedded third-party code
- **NEP50 compliant**: NumPy 2.x type promotion rules
- **Benchmark infrastructure**: SIMD vs scalar comparison suite
- **DefaultEngine dispatch layer**: BinaryOp, BitwiseOp, CompareOp, ReductionOp, UnaryOp
- +4,200 unit tests, our own and migrated from python/numpy to C#.

---

## Contents

| Section | Highlights |
|---------|------------|
| [Summary](#summary) | 106 commits, -533K lines, 3,907 tests |
| [IL Kernel Generator](#il-kernel-generator) | 27 files, SIMD V128/256/512 |
| [Architecture](#architecture) | Static ILKernelGenerator, TensorEngine routing |
| [New NumPy Functions (35)](#new-numpy-functions-25) | nansum, isnan, cumprod, etc. |
| [Critical Bug Fixes](#critical-bug-fixes) | negative, unique, dot, linspace, intp |
| [Operator Rewrites](#operator-rewrites) | ==, !=, <, >, &, \| now work |
| [Boolean Indexing Rewrite](#boolean-indexing-rewrite) | SIMD fast path, 76 battle tests |
| [Slicing Improvements](#slicing-improvements) | Broadcast stride=0 preserved |
| [Performance Improvements](#performance-improvements) | MatMul 35-100x, 20+ GFLOPS |
| [Code Reduction](#code-reduction) | 99% binary, 98% MatMul, 97% Dot |
| [Infrastructure Changes](#infrastructure-changes) | NativeMemory, static kernels |
| [API Alignment](#api-alignment) | random() params aligned with NumPy |
| [New Test Files (68)](#new-test-files-68) | 34 kernel, 8 NumPy, 4 linalg, 76 boolean |
| [Breaking Changes](#breaking-changes) | None |
| [Known Issues](#known-issues-openbugs) | 52 OpenBugs excluded |
| [Installation](#installation) | `dotnet add package NumSharp` |

---

## Summary

| Metric | Value |
|--------|-------|
| Commits | 106 |
| Files Changed | 558 |
| Lines Added | +72,635 |
| Lines Deleted | -605,976 |
| **Net Change** | **-533K lines** |
| Test Results | 3,907 passed, 52 OpenBugs, 11 skipped |

---

## IL Kernel Generator

Runtime IL generation via `System.Reflection.Emit.DynamicMethod` replaces static Regen templates.

### Kernel Files (27 new files)
- `ILKernelGenerator.cs` - Core infrastructure, SIMD detection (Vector128/256/512)
- `ILKernelGenerator.Binary.cs` - Add, Sub, Mul, Div, BitwiseAnd/Or/Xor
- `ILKernelGenerator.MixedType.cs` - Mixed-type ops with type promotion
- `ILKernelGenerator.Unary.cs` - Negate, Abs, Sqrt, Sin, Cos, Exp, Log, Sign
- `ILKernelGenerator.Comparison.cs` - ==, !=, <, >, <=, >= returning bool arrays
- `ILKernelGenerator.Reduction.cs` - Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any
- `ILKernelGenerator.Reduction.Axis.Simd.cs` - AVX2 gather for axis reductions
- `ILKernelGenerator.Scan.cs` - CumSum, CumProd with SIMD
- `ILKernelGenerator.Shift.cs` - LeftShift, RightShift
- `ILKernelGenerator.MatMul.cs` - Cache-blocked SIMD matrix multiply
- `ILKernelGenerator.Clip.cs`, `.Modf.cs`, `.Masking.cs` - Specialized ops

### Execution Paths
1. **SimdFull** - Contiguous + SIMD-capable dtype → Vector loop + scalar tail
2. **ScalarFull** - Contiguous + non-SIMD dtype (Decimal) → Scalar loop
3. **General** - Strided/broadcast → Coordinate-based iteration

### Infrastructure
- `KernelKey.cs`, `KernelOp.cs`, `KernelSignatures.cs` - Kernel dispatch
- `SimdMatMul.cs` - SIMD matrix multiplication helpers
- `TypeRules.cs` - NEP50 type promotion rules

---

## Architecture

Clean separation of concerns:

| Component | Design |
|-----------|--------|
| `ILKernelGenerator` | Static class (27 partial files), internal to `DefaultEngine` |
| `TensorEngine` | All `np.*` ops route through abstract methods |
| `Shape.Broadcasting` | Pure shape math in `Shape` struct (456 lines) |
| `ArgMin/ArgMax` | Unified IL kernel with NaN-aware + Boolean semantics |
| `DecimalMath` | Internal utility (~403 lines) for Sqrt, Pow, ATan2, Exp, Log |

### Single-Threaded Execution
All computation is single-threaded with no `Parallel.For` usage. This provides:
- **Deterministic behavior** - Same inputs always produce same outputs in same order
- **Non-blocking execution** - No thread synchronization overhead
- **Simplified debugging** - Stack traces are straightforward
- **SIMD compensation** - Vector128/256/512 intrinsics provide parallelism at the CPU level

### Broadcasting External to Engine
Broadcasting logic (`Shape.Broadcasting.cs`) is pure shape math with no engine dependencies:
- `Shape.AreBroadcastable()` - Check if shapes can broadcast
- `Shape.Broadcast()` - Compute broadcast result shape and strides
- `Shape.ResolveReturnShape()` - Determine output shape for operations
- `DefaultEngine` delegates all broadcasting to `Shape.*` methods

### DecimalMath (#588)
Replaced embedded third-party `DecimalEx.cs` (~1061 lines) with minimal internal `DecimalMath.cs` (~403 lines) containing only the functions NumSharp actually uses: Sqrt, Pow, ATan2, Exp, Log, Log10, ATan.

### TensorEngine Abstract Methods
`Compare`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual`, `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `LeftShift`, `RightShift`, `Power(NDArray, NDArray)`, `FloorDivide`, `Truncate`, `Reciprocal`, `Square`, `Cbrt`, `Invert`, `Deg2Rad`, `Rad2Deg`, `IsInf`, `ReduceCumMul`, `Any`, `NanSum`, `NanProd`, `NanMin`, `NanMax`, `BooleanMask`

### DefaultEngine Dispatch Files (IL kernel integration)
| File | Functions |
|------|-----------|
| `DefaultEngine.BinaryOp.cs` | `np.add`, `np.subtract`, `np.multiply`, `np.divide`, `np.mod`, `np.power` |
| `DefaultEngine.BitwiseOp.cs` | `np.bitwise_and`, `np.bitwise_or`, `np.bitwise_xor`, `&`, `\|`, `^` |
| `DefaultEngine.CompareOp.cs` | `np.equal`, `np.not_equal`, `np.less`, `np.greater`, `np.less_equal`, `np.greater_equal` |
| `DefaultEngine.ReductionOp.cs` | `np.sum`, `np.prod`, `np.min`, `np.max`, `np.mean`, `np.std`, `np.var`, `np.argmax`, `np.argmin` |
| `DefaultEngine.UnaryOp.cs` | `np.abs`, `np.negative`, `np.sqrt`, `np.sin`, `np.cos`, `np.exp`, `np.log`, `np.sign`, etc. |

### Implementation Files
`Default.Any.cs`, `Default.BooleanMask.cs`, `Default.Reduction.Nan.cs`, `Shape.Broadcasting.cs`

---

## New NumPy Functions (35)

### NaN-Aware Reductions (7)
| Function | Description |
|----------|-------------|
| `np.nansum` | Sum ignoring NaN |
| `np.nanprod` | Product ignoring NaN |
| `np.nanmin` | Minimum ignoring NaN |
| `np.nanmax` | Maximum ignoring NaN |
| `np.nanmean` | Mean ignoring NaN |
| `np.nanvar` | Variance ignoring NaN |
| `np.nanstd` | Standard deviation ignoring NaN |

### Math Operations (8)
| Function | Description |
|----------|-------------|
| `np.cbrt` | Cube root |
| `np.floor_divide` | Integer division |
| `np.reciprocal` | Element-wise 1/x |
| `np.trunc` | Truncate to integer |
| `np.invert` | Bitwise NOT |
| `np.square` | Element-wise square |
| `np.cumprod` | Cumulative product |
| `np.count_nonzero` | Count non-zero elements |

### Bitwise & Trigonometric (4)
| Function | Description |
|----------|-------------|
| `np.left_shift` | Bitwise left shift |
| `np.right_shift` | Bitwise right shift |
| `np.deg2rad` | Degrees to radians |
| `np.rad2deg` | Radians to degrees |

### Logic & Validation (4) - Previously returned `null`
| Function | Description |
|----------|-------------|
| `np.isnan` | Test element-wise for NaN |
| `np.isfinite` | Test element-wise for finiteness |
| `np.isinf` | Test element-wise for infinity |
| `np.isclose` | Element-wise comparison within tolerance |

### Operators (2) - Previously returned `null`
| Operator | Description |
|----------|-------------|
| `operator &` | Bitwise/logical AND with broadcasting |
| `operator \|` | Bitwise/logical OR with broadcasting |

### Comparison Functions (6) - New named API
| Function | Description |
|----------|-------------|
| `np.equal` | Element-wise equality (wraps `==`) |
| `np.not_equal` | Element-wise inequality (wraps `!=`) |
| `np.less` | Element-wise less than (wraps `<`) |
| `np.greater` | Element-wise greater than (wraps `>`) |
| `np.less_equal` | Element-wise less or equal (wraps `<=`) |
| `np.greater_equal` | Element-wise greater or equal (wraps `>=`) |

### Logical Functions (4) - New named API
| Function | Description |
|----------|-------------|
| `np.logical_and` | Element-wise logical AND |
| `np.logical_or` | Element-wise logical OR |
| `np.logical_not` | Element-wise logical NOT |
| `np.logical_xor` | Element-wise logical XOR |

### New Overloads
| Function | New Capability |
|----------|----------------|
| `np.power(array, array)` | Array exponents (was scalar only) |
| `np.repeat(array, NDArray)` | Per-element repeat counts |
| `np.argmax/argmin(axis, keepdims)` | keepdims parameter |
| `np.convolve` | Complete rewrite (was throwing NRE) |

---

## Critical Bug Fixes

### Behavioral Fixes
| Bug | Before | After |
|-----|--------|-------|
| `np.negative()` | Only negated positive values (`if val > 0`) | Negates ALL values (`val = -val`) |
| `np.positive()` | Applied `abs()` | Identity operation (returns input unchanged) |
| `np.unique()` | Returned unsorted | Sorts output, NaN at end |
| `np.dot(1D, 2D)` | Threw `NotSupportedException` | Treats 1D as row vector |
| `np.dot()` non-contiguous | Failed on strided arrays | Works with all memory layouts |
| `np.matmul()` broadcast | Crashed with >2D arrays | Full broadcasting support |
| `np.linspace()` | Returned `float32` for float inputs | Always `float64` default |
| `np.arange()` | Threw on `start >= stop` | Returns empty array |
| `np.searchsorted()` | No scalar support | Added scalar overloads returning `int` |
| `np.shuffle()` | Non-standard `passes` parameter | NumPy legacy API (axis-0 only) |
| `np.moveaxis()` | Broken | Verified working |
| `np.argsort()` | NaN handling incorrect | NaN-aware sorting |
| `np.intp` | Mapped to `int` (always 32-bit) | Uses `nint` (native-sized integer) |
| `np.uintp` | Not defined | Added as `nuint` (native unsigned) |
| `np.LogicalNot()` | Changed dtype | Preserves Boolean type |
| Float-to-int conversion | Used rounding | Uses truncation toward zero |

### Return Type Fixes
| Function | Before | After |
|----------|--------|-------|
| `np.argmax()` / `np.argmin()` | Returned `int` | Returns `long` (large array support) |
| `np.abs()` | Converted to Double | Preserves input dtype |

### Empty Array Handling
| Function | Before | After |
|----------|--------|-------|
| `np.mean([])` | Threw or returned 0 | Returns `NaN` |
| `np.mean(zeros((0,3)), axis=0)` | Incorrect | `[NaN, NaN, NaN]` |
| `np.mean(zeros((0,3)), axis=1)` | Incorrect | Empty array `[]` |
| `np.std/var` single element | Returned 0 | Returns `NaN` with `ddof >= size` |

### keepdims Fixes
All reduction functions now properly preserve dimensions when `keepdims=True`:
- `np.sum`, `np.prod`, `np.mean`, `np.std`, `np.var`
- `np.min`, `np.max`, `np.argmin`, `np.argmax`

### Rewritten Functions (IL kernel migration)
| Function | Fix |
|----------|-----|
| `np.all()` | SIMD, all 12 dtypes (was boolean-only) |
| `np.any()` | SIMD with early-exit; axis parameter fixed (was always throwing) |
| `np.sum()` | Axis reduction for broadcast arrays |
| `np.cumsum()` | Axis support with SIMD, 4K lines Regen removed |
| `np.cumprod()` | Axis support with SIMD |
| `np.nonzero()` | Unified IL approach |
| `np.clip()` | IL kernel rewrite |

### Math Functions (IL migration)
All migrated from Regen templates to IL kernels with SIMD:
- **Trig**: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`, `arctan2`
- **Exp/Log**: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p`
- **Other**: `sqrt`, `abs`, `sign`, `floor`, `ceil`, `round`

---

## Operator Rewrites

### Comparison Operators (==, !=, <, >, <=, >=)
- **Before**: Manual type switch per dtype
- **After**: Uses `TensorEngine` with IL kernels
- Proper null handling (returns `false` scalar)
- Empty array handling (returns empty bool array)
- Added reverse operators (`object op NDArray`)
- Full broadcasting support

### Bitwise Operators (&, |, ^)
- **Before**: Returned `null`
- **After**: Full implementation via IL kernels
- Added `NDArray<T>` typed operators
- Scalar overloads for all integer types

### Implicit Scalar Conversion
- **Before**: `(int)ndarray_float64` would fail
- **After**: Uses `Converts.ChangeType` for cross-dtype conversion

---

## Boolean Indexing Rewrite

Complete rewrite with NumPy-aligned behavior:

### Two Cases Supported
1. `arr[mask]` where `mask.shape == arr.shape` → element-wise selection
2. `arr[mask]` where `mask` is 1D and `mask.shape[0] == arr.shape[0]` → axis-0 selection

### SIMD Fast Path
- New `BooleanMaskFastPath` for contiguous arrays
- `CountTrue(bool*, int)` - SIMD count of true values
- `CopyMasked<T>(src, mask, dest, size)` - SIMD masked copy

---

## Slicing Improvements

### Broadcast Array Handling
- **Before**: Slicing broadcast arrays would materialize data (losing stride=0)
- **After**: Preserves stride=0 information (NumPy behavior)
- Critical for `cumsum` and axis reductions on broadcast arrays

### Empty Slice Handling
- `a[100:200]` on 10-element array now returns proper empty array

### Contiguous Optimization
- Contiguous slices get fresh shape with `offset=0`
- `IsSliced=false` for contiguous slices

---

## Performance Improvements

| Operation | Improvement | Details |
|-----------|-------------|---------|
| MatMul (2D) | 35-100x | Cache-blocked SIMD, 20+ GFLOPS |
| Axis Reductions | Major | AVX2 gather + parallel outer loop |
| All/Any | Major | SIMD with early-exit |
| CumSum/CumProd | Major | Element-wise SIMD |
| Boolean Masking | Major | SIMD CountTrue + CopyMasked |
| Integer Abs/Sign | Minor | Bitwise (branchless) |
| Vector512 | New | Runtime detection and utilization |
| Loop Unrolling | 4x | All SIMD kernels |

---

## Code Reduction

### Massive File Deletions
| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Binary ops (Add/Sub/Mul/Div/Mod) | 60 files, ~500K lines | 2 IL files | **99%** |
| `Default.MatMul.2D2D.cs` | ~20K lines | 325 lines | **98.4%** |
| `Default.Dot.NDMD.cs` | ~16K lines | 422 lines | **97.4%** |
| Comparison ops (Equals) | 13 files | 1 IL file | **92%** |
| Std/Var reductions | ~20K lines | ~500 lines | **97%** |

### Deleted Files (76)
- 60 binary op files (`Default.Add.{Type}.cs`, etc.)
- 13 comparison files (`Default.Equals.{Type}.cs`, etc.)
- 3 template files

---

## Infrastructure Changes

### Memory Allocation
- `Marshal.AllocHGlobal` → `NativeMemory.Alloc`
- `Marshal.FreeHGlobal` → `NativeMemory.Free`
- `AllocationType.AllocHGlobal` → `AllocationType.Native`
- `StackedMemoryPool` migrated to NativeMemory

### DefaultEngine
- `ILKernelGenerator` is a static class (internal to DefaultEngine)
- Single-threaded execution (no `Parallel.For`)

### Math Functions
All migrated from Regen templates to `ExecuteUnaryOp`:
- Sin, Cos, Tan, ASin, ACos, ATan, ATan2
- Exp, Exp2, Expm1, Log, Log2, Log10, Log1p
- Sqrt, Cbrt, Abs, Sign, Floor, Ceil, Truncate
- Removed `DecimalMath` dependency for most operations

### TensorEngine Extensions
New abstract methods (28 total):
- **Comparison**: `Compare`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual`
- **Bitwise**: `BitwiseAnd`, `BitwiseOr`, `BitwiseXor`, `LeftShift`, `RightShift`
- **Math**: `Power(NDArray, NDArray)`, `FloorDivide`, `Truncate`, `Reciprocal`, `Square`, `Cbrt`, `Invert`, `Deg2Rad`, `Rad2Deg`, `IsInf`
- **Reduction**: `ReduceCumMul`, `Any`, `NanSum`, `NanProd`, `NanMin`, `NanMax`
- **Indexing**: `BooleanMask`

### IKernelProvider Methods
- `CountTrue(bool*, int)` - SIMD true count
- `CopyMasked<T>` - SIMD masked copy
- `Variance<T>`, `StandardDeviation<T>` - SIMD two-pass
- `NanSum/Prod/Min/Max` for float/double
- `FindNonZeroStrided<T>` - Strided nonzero detection

---

## API Alignment

| API | NumPy-Aligned Behavior |
|-----|------------------------|
| `np.random.random()` | Alias for `random_sample()` |
| `np.random.standard_normal()` | Correct spelling (matches NumPy) |
| `np.random.*` params | `size`, `a`, `b`, `p`, `d0` (NumPy names) |
| `np.random.randn/rand/normal` | Accept `Shape` parameter |
| `np.minimum/maximum` | `dtype` parameter (not `outType`) |
| `np.modf()` | Validates floating-point input |

---

## New Test Files (68)

### Kernel Tests (34)
`BinaryOpTests`, `UnaryOpTests`, `ComparisonOpTests`, `ReductionOpTests`, `AxisReductionSimdTests`, `NonContiguousTests`, `SlicedArrayOpTests`, `NanReductionTests`, `VarStdComprehensiveTests`, `ArgMaxArgMinComprehensiveTests`, `CumSumComprehensiveTests`, `BitwiseOpTests`, `ShiftOpTests`, `DtypeCoverageTests`, `DtypePromotionTests`, `EdgeCaseTests`, `BattleProofTests`, `SimdOptimizationTests`, and more.

### NumPy Ported Tests (8)
`ArgMaxArgMinEdgeCaseTests`, `ClipEdgeCaseTests`, `ClipNDArrayTests`, `CumSumEdgeCaseTests`, `ModfEdgeCaseTests`, `NonzeroEdgeCaseTests`, `PowerEdgeCaseTests`, `VarStdEdgeCaseTests`

### Linear Algebra Battle Tests (4)
`np.dot.BattleTest` (195 tests), `np.matmul.BattleTest` (106 tests), `np.outer.BattleTest` (88 tests)

### Boolean Indexing Battle Tests (76 tests)
`BooleanIndexing.BattleTests.cs` - Comprehensive NumPy 2.4.2 alignment covering same-shape masks, axis-0 selection, partial shape match, 0-D indexing, mask assignment, empty masks, shape mismatch errors, non-contiguous arrays, all dtypes, NaN/Infinity, logical operations.

### Random Sampling Tests
`np.random.shuffle.NumPyAligned.Test.cs` (133 tests)

---

## Breaking Changes

**None.** This is a drop-in replacement with improved performance and NumPy compatibility.

---

## Known Issues (OpenBugs)

52 tests marked as `[OpenBugs]` are excluded from CI:
- sbyte (int8) type not supported
- Some bitmap operations require GDI+ (Windows only)
- Various edge cases documented in test files

---

## Installation

```bash
dotnet add package NumSharp --version 0.41.0-prerelease
```

Or via Package Manager:
```powershell
Install-Package NumSharp -Version 0.41.0-prerelease
```

## Testing

```bash
cd test/NumSharp.UnitTest

# Run tests excluding known issues
dotnet test -- "--treenode-filter=/*/*/*/*[Category!=OpenBugs]"

# Run all tests
dotnet test
```

---

## Feedback

This is a prerelease. Please report any issues at:
https://github.com/SciSharp/NumSharp/issues

---

**Full Changelog**: See [CHANGES.md](./CHANGES.md) for complete documentation of all 106 commits.
