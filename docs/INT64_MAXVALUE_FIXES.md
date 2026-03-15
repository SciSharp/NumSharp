# Int64 MaxValue Boundary Fixes

Tracking document for fixing `int.MaxValue` boundary checks that violate the int64 migration guide.

---

## Task List

| # | File | Status | Issue | Fix Applied |
|---|------|--------|-------|-------------|
| 1 | `ndarray.argsort.cs` | DONE | Enumerable.Range limit throws | Already fixed - uses LongRange helper |
| 2 | `SimdMatMul.cs` | DONE | Dimension > int.MaxValue throws | Already fixed - uses long params, graceful fallback |
| 3 | `Default.MatMul.2D2D.cs` | DONE | SIMD path condition uses int.MaxValue | Already fixed - uses long throughout |
| 4 | `Default.NonZero.cs` | DONE | List capacity with int cast | Uses LongIndexBuffer, long[] shape |
| 5 | `ILKernelGenerator.Masking.cs` | DONE | List capacity with int cast | Uses LongIndexBuffer, long[] shape |
| 6 | `IKernelProvider.cs` | DONE | Interface uses List<long>, int[] shape | Uses LongIndexBuffer, long[] shape |
| 7 | `ILKernelGenerator.Reduction.Axis.Simd.cs` | DONE | Had Parallel.For + int types | Removed Parallel.For, using long types |
| 8 | `ILKernelGenerator.Reduction.Axis.VarStd.cs` | DONE | Had Parallel.For + int types | Removed Parallel.For, using long types |

---

## Session 3: Post-Merge Fixes (ilkernel integration)

### Context

After merging `ilkernel` branch (which removed Parallel.For) into `longindexing` branch, there were conflicts between:
- ilkernel: `int` types, no Parallel.For
- longindexing: `long` types, with Parallel.For

### Resolution

Restored longindexing versions (with `long` types) and manually removed Parallel.For to match ilkernel's single-threaded execution approach.

### Files Fixed

1. **ILKernelGenerator.Reduction.Axis.Simd.cs**
   - Removed `using System.Threading.Tasks;`
   - Removed `AxisReductionParallelThreshold` constant
   - Removed Parallel.For branch in `AxisReductionSimdHelper<T>`
   - Removed `ReduceAxisElement<T>` helper method (was only used by Parallel.For)
   - Kept `long*` pointer parameters and `long` loop variables
   - Updated IKernelProvider interface implementations to match new signatures:
     - `FindNonZero<T>` now takes `ref LongIndexBuffer`
     - `ConvertFlatToCoordinates` now takes `ref LongIndexBuffer, long[]`
     - `FindNonZeroStrided<T>` now takes `long[]` shape

2. **ILKernelGenerator.Reduction.Axis.VarStd.cs**
   - Removed `using System.Threading.Tasks;`
   - Removed `parallelThreshold` constant
   - Removed Parallel.For branch in `AxisVarStdSimdHelper<TInput>`
   - Removed `ComputeVarStdElement<TInput>` helper method
   - Kept `long*` pointer parameters and `long` loop variables

### Build Status

**BUILD SUCCEEDED** - 0 errors, 19 warnings

---

## Previous Sessions Summary

**Session 1:**
- Tasks 1-3: Already fixed in previous commits
- Tasks 4-6: Replaced List<T> with LongIndexBuffer, int[] shape with long[]
- Task 7 (new): Created LongIndexBuffer helper struct

**Session 2:**
- Fixed loop counters in: NDArray.unique, np.random.uniform, np.repeat, Reduction Mean/Var/Std, np.random.randint, np.mgrid, np.eye

---

## Verification Checklist

- [x] Build succeeds
- [ ] Tests pass
- [x] No Parallel.For in axis reduction files
- [x] All IKernelProvider interface implementations match signatures
- [x] Long types used for indices/sizes/strides/offsets
