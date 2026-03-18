# Int32 Cast Landmines - Audit Complete

**Audit Date:** 2026-03-18
**Status:** All critical issues fixed or verified as protected

Arrays > 2 billion elements could overflow silently at these locations.

**Related:** See `INT64_DEVELOPER_GUIDE.md` for migration patterns.

---

## Summary

| Category | Found | Fixed | Already Protected | Known Limitation |
|----------|-------|-------|-------------------|------------------|
| Critical Storage | 5 | 2 | 3 | 0 |
| High Array/Shape | 10 | 6 | 4 | 0 |
| Medium String/Span | 7 | 0 | 7 | 0 |
| Collections | 15 | 0 | 8 | 7 (.NET Array limits) |
| Coordinate/Axis | 9 | 1 | 8 | 0 |
| Value Conversions | ~80 | 0 | N/A | N/A (not index-related) |

**Total Fixed:** 9 locations
**Already Protected:** 30+ locations
**Known Limitations:** 7 (inherent .NET Array int-indexing)

---

## FIXED - Category 1: Critical Storage/Allocation

| File | Line | Fix |
|------|------|-----|
| `UnmanagedStorage.cs` | 1013 | Removed `(int)` - long overload exists |
| `UnmanagedStorage.cs` | 1026 | Removed `(int)` - long overload exists |

**Already Protected:**
- `UnmanagedStorage.cs:188` - Has guard at line 186-187
- `ArraySlice\`1.cs:333` - Has guard at line 331-332
- `SimdMatMul.cs:48` - Has guard + chunked fallback at line 47-54

---

## FIXED - Category 2: Shape/Reshape Operations

| File | Line | Fix |
|------|------|-----|
| `np.meshgrid.cs` | 25 | Use `reshape(x1.size, 1)` - long overload |
| `np.meshgrid.cs` | 30 | Use `reshape(x2.size, 1)` - long overload |
| `np.nanvar.cs` | 178 | `outputShapeList.Add(inputShape[i])` - List<long> |
| `np.nanstd.cs` | 178 | `outputShapeList.Add(inputShape[i])` - List<long> |
| `np.nanmean.cs` | 125 | `outputShapeList.Add(inputShape[i])` - List<long> |

**Already Protected:**
- `np.nanvar.cs:198,272` - Guard at line 189-190
- `np.nanstd.cs:198,272` - Guard at line 189-190
- `np.nanmean.cs:145,191` - Guard at line 136-137

---

## FIXED - Category 3: Array Conversion

| File | Line | Fix |
|------|------|-----|
| `NdArrayToMultiDimArray.cs` | 34 | Added dimension overflow check before conversion |

---

## ALREADY PROTECTED - Category 4: String Operations

All string operations have proper guards:

| File | Line | Guard |
|------|------|-------|
| `NDArray.String.cs` | 33 | Guard at line 31-32 |
| `NDArray.String.cs` | 87 | Guard at line 85-86 |
| `NDArray.String.cs` | 100 | Guard at line 98-99 |

---

## ALREADY PROTECTED - Category 5: SIMD Operations

| File | Line | Guard |
|------|------|-------|
| `ILKernelGenerator.Reduction.Axis.Simd.cs` | 396 | Guard at 391-394, fallback to scalar |
| `ILKernelGenerator.Reduction.Axis.Simd.cs` | 440 | Guard at 434-438, fallback to scalar |
| `SimdMatMul.cs` | 128,136,149,230 | Block sizes bounded by constants |

---

## ALREADY PROTECTED - Category 6: Shape Explicit Operators

All use `checked` or have overflow guards:

| File | Line | Protection |
|------|------|------------|
| `Shape.cs` | 927 | Uses `checked((int)dims[i])` - throws on overflow |
| `Shape.cs` | 1076-1086 | Has overflow check at line 1081-1082 |
| `Shape.cs` | 1097-1102 | Has overflow check at line 1099-1100 |
| `Slice.cs` | 296 | Has guard at line 293-294 |

---

## KNOWN LIMITATION - Category 7: Collections (.NET Array Limits)

LongList and Hashset wrap .NET Array methods which are int-indexed:

| File | Lines | Status |
|------|-------|--------|
| `LongList\`1.cs` | 294, 368 | ICollection.CopyTo takes int |
| `LongList\`1.cs` | 548, 562 | Array.IndexOf takes int |
| `Hashset\`1.cs` | 238-252 | Has chunking for Clear() |
| `Hashset\`1.cs` | 367 | Throws if count > int.MaxValue |

**Note:** Proper fix requires custom implementations not using Array.* methods.

---

## KNOWN LIMITATION - Category 8: .NET Multi-Dim Arrays

.NET multi-dimensional arrays are inherently int-indexed:

| File | Lines | Status |
|------|-------|--------|
| `NdArrayToMultiDimArray.cs` | 34, 48 | Now has overflow check |
| `np.save.cs` | 55, 74, etc. | Working with .NET Arrays |
| `np.load.cs` | 30 | Working with .NET Arrays |
| `ArrayConvert.cs` | 43-46 | GetLength() returns int |

---

## NOT INDEX-RELATED - Value Conversions

These are value conversions (double->int for computation results), not indices:

- `Default.Reduction.CumAdd.cs:225` - `(int)sum`
- `Default.Reduction.CumMul.cs:210` - `(int)product`
- `NdArray.Convolve.cs:156` - `(int)sum`
- `Default.Shift.cs:153` - Shift amounts < 64 bits
- `Randomizer.cs:264,268` - Has range guards
- `Operator.cs` - Arithmetic type promotion

---

## NOT INDEX-RELATED - Type Dispatch

IL code generation type checking patterns:

- `typeof(T) == typeof(int)` (~50 occurrences)
- `il.DeclareLocal(typeof(int))` (~10 occurrences)
- `il.Emit(OpCodes.Ldc_I4, ...)` (~15 occurrences)

---

## NOT INDEX-RELATED - Axis Operations

Axis indices are bounded by ndim (which is int, max ~32):

| File | Lines | Reason |
|------|-------|--------|
| `Default.Transpose.cs` | 71-72, 152 | Axis < ndim |
| `Shape.cs` | 1310-1335 | Backward-compat int[] methods |

---

## Verification

Build and tests pass:
```bash
dotnet build src/NumSharp.Core  # 0 errors
dotnet test -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"  # 3887 passed
```
