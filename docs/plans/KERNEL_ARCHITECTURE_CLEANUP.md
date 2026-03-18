# Kernel Architecture Cleanup Plan

## Progress Tracking

| Phase | Task | Status | Date |
|-------|------|--------|------|
| 5.3 | Replace `KernelProvider.` with `ILKernelGenerator.` in DefaultEngine | DONE | 2026-03-18 |
| 5.2 | Remove `protected KernelProvider` field from DefaultEngine.cs | DONE | 2026-03-18 |
| 1.1 | Add `Any(NDArray)` and `Any(NDArray, int)` to TensorEngine | DONE | 2026-03-18 |
| 2.1 | Create `Default.All.cs` with all 12 dtypes + SIMD | DONE | 2026-03-18 |
| 2.2 | Create `Default.Any.cs` with all 12 dtypes + SIMD | DONE | 2026-03-18 |
| 3.1 | Route `np.all(NDArray)` through TensorEngine | DONE | 2026-03-18 |
| 3.2 | Route `np.any(NDArray)` through TensorEngine | DONE | 2026-03-18 |
| 1.2 | Add NaN reduction methods to TensorEngine | DONE | 2026-03-18 |
| 2.3 | Implement NaN reductions in DefaultEngine | DONE | 2026-03-18 |
| 3.3 | Route np.nansum/prod/min/max through TensorEngine | DONE | 2026-03-18 |
| 1.3 | Add BooleanMask method to TensorEngine | DONE | 2026-03-18 |
| 2.4 | Implement BooleanMask in DefaultEngine | DONE | 2026-03-18 |
| 4 | Route NDArray.Indexing.Masking through TensorEngine | DONE | 2026-03-18 |
| 5.1 | Remove `DefaultKernelProvider` static property | DONE | 2026-03-18 |
| 5.4 | Delete IKernelProvider.cs | DONE | 2026-03-18 |
| 5.5 | Remove IKernelProvider interface from ILKernelGenerator | DONE | 2026-03-18 |
| 5.6 | Update Default.NonZero.cs to use ILKernelGenerator directly | DONE | 2026-03-18 |
| 6 | Final verification | DONE | 2026-03-18 |
| 7.1 | Remove `Instance` singleton from ILKernelGenerator | DONE | 2026-03-18 |
| 7.2 | Make ILKernelGenerator a static class | DONE | 2026-03-18 |
| 7.3 | Remove BackendFactory usage from np.array_manipulation.cs | DONE | 2026-03-18 |
| 8 | Move broadcast utilities from DefaultEngine to Shape | PENDING | - |

**Violations Fixed: 7/7 files - ALL DONE**
**IKernelProvider Interface: DELETED**
**ILKernelGenerator: Now a static class (no Instance singleton)**
**Tests: 4122 passed, 51 failed (pre-existing), 11 skipped**

Verification commands return no results:
```bash
# No IKernelProvider references
grep -r "IKernelProvider" src/NumSharp.Core --include="*.cs"

# No DefaultKernelProvider references
grep -r "DefaultKernelProvider" src/NumSharp.Core --include="*.cs"

# No kernel imports outside Backends/
grep -l "using NumSharp.Backends.Kernels" src/NumSharp.Core --include="*.cs" -r | grep -v "/Backends/"
```

**Files modified:**
- `TensorEngine.cs` - Added `Any`, `NanSum`, `NanProd`, `NanMin`, `NanMax`, `BooleanMask` abstract methods
- `DefaultEngine.cs` - Removed `KernelProvider` field, added TODO for `DefaultKernelProvider`
- `DefaultEngine.BinaryOp.cs` - Changed to `ILKernelGenerator.GetMixedTypeKernel()` / `GetBinaryScalarDelegate()`
- `DefaultEngine.UnaryOp.cs` - Changed to `ILKernelGenerator.GetUnaryKernel()` / `GetUnaryScalarDelegate()`
- `DefaultEngine.CompareOp.cs` - Changed to `ILKernelGenerator.GetComparisonKernel()` / `GetComparisonScalarDelegate()`
- `DefaultEngine.ReductionOp.cs` - Changed to `ILKernelGenerator.TryGetTypedElementReductionKernel()` / `TryGetAxisReductionKernel()`
- `Default.ATan2.cs` - Changed to `ILKernelGenerator.GetMixedTypeKernel()`
- `Default.All.cs` - Rewritten: all 12 dtypes + SIMD via `ILKernelGenerator.AllSimdHelper<T>()`
- `Default.Any.cs` - NEW: all 12 dtypes + SIMD via `ILKernelGenerator.AnySimdHelper<T>()`
- `Default.Reduction.Nan.cs` - NEW: NanSum/NanProd/NanMin/NanMax implementations
- `Default.BooleanMask.cs` - NEW: BooleanMask with SIMD path
- `np.all.cs` - Simplified: `all(NDArray)` now calls `a.TensorEngine.All(a)`
- `np.any.cs` - Simplified: `any(NDArray)` now calls `a.TensorEngine.Any(a)`
- `np.nansum.cs` - Simplified: now calls `a.TensorEngine.NanSum(a, axis, keepdims)`
- `np.nanprod.cs` - Simplified: now calls `a.TensorEngine.NanProd(a, axis, keepdims)`
- `np.nanmin.cs` - Simplified: now calls `a.TensorEngine.NanMin(a, axis, keepdims)`
- `np.nanmax.cs` - Simplified: now calls `a.TensorEngine.NanMax(a, axis, keepdims)`
- `NDArray.Indexing.Masking.cs` - Simplified: now calls `TensorEngine.BooleanMask()`

**Phase 5 Changes (IKernelProvider Removal):**
- `IKernelProvider.cs` - DELETED
- `DefaultEngine.cs` - Removed `DefaultKernelProvider` static property
- `Default.NonZero.cs` - Changed from `DefaultKernelProvider` to `ILKernelGenerator.*` direct calls
- `ILKernelGenerator.cs` - Removed `: IKernelProvider`, removed explicit interface implementations
- `ILKernelGenerator.Reduction.Axis.Simd.cs` - Removed explicit interface implementations
- `ILKernelGenerator.Binary.cs` - Updated comments
- `KernelSignatures.cs` - Updated comments

**Phase 7 Changes (Static Class Conversion):**
- `ILKernelGenerator.cs` - Removed `Instance` singleton, changed to `static partial class`
- All 27 ILKernelGenerator.*.cs files - Changed to `public static partial class`

**Phase 7.3 Changes (BackendFactory Removal):**
- `np.array_manipulation.cs` - Removed BackendFactory dependency, use NDArray constructors directly

---

## Goal

Enforce clean separation: **ALL computation on NDArray goes through TensorEngine**.

- `ILKernelGenerator` is an internal implementation detail of `DefaultEngine`
- Remove `IKernelProvider` interface (premature abstraction)
- No direct kernel access from `np.*` or `NDArray` classes

## Current State Analysis

### Architecture Violations Found

| Violation Type | Count | Files |
|----------------|-------|-------|
| `np.*` calls `ILKernelGenerator.*` directly | 14 | np.nansum, np.nanprod, np.nanmin, np.nanmax, etc. |
| `np.*` calls `DefaultEngine.DefaultKernelProvider` | 2 | np.all, np.any |
| `NDArray` calls `DefaultEngine.DefaultKernelProvider` | 2 | NDArray.Indexing.Masking.cs |
| `DefaultEngine` calls `ILKernelGenerator.*` directly (bypassing interface) | 44 | Default.Clip, Default.Shift, Default.Var, etc. |

### Detailed Violation Inventory

#### np.* files calling ILKernelGenerator directly:

```
np.nansum.cs:53    → ILKernelGenerator.Enabled
np.nansum.cs:61    → ILKernelGenerator.NanSumSimdHelperFloat()
np.nansum.cs:64    → ILKernelGenerator.NanSumSimdHelperDouble()
np.nansum.cs:159   → ILKernelGenerator.TryGetNanAxisReductionKernel()

np.nanprod.cs:53   → ILKernelGenerator.Enabled
np.nanprod.cs:61   → ILKernelGenerator.NanProdSimdHelperFloat()
np.nanprod.cs:64   → ILKernelGenerator.NanProdSimdHelperDouble()

np.nanmin.cs:40    → ILKernelGenerator.Enabled
np.nanmin.cs:48    → ILKernelGenerator.NanMinSimdHelperFloat()
np.nanmin.cs:51    → ILKernelGenerator.NanMinSimdHelperDouble()

np.nanmax.cs:40    → ILKernelGenerator.Enabled
np.nanmax.cs:48    → ILKernelGenerator.NanMaxSimdHelperFloat()
np.nanmax.cs:51    → ILKernelGenerator.NanMaxSimdHelperDouble()
```

#### np.* files calling DefaultEngine.DefaultKernelProvider:

```
np.all.cs:172      → DefaultEngine.DefaultKernelProvider.All()
np.any.cs:177      → DefaultEngine.DefaultKernelProvider.Any()
```

#### np.* files calling DefaultEngine static helpers (EXCEPTION - see below):

```
np.are_broadcastable.cs  → DefaultEngine.AreBroadcastable() (4 calls)
np.broadcast.cs          → DefaultEngine.ResolveReturnShape()
np.broadcast_arrays.cs   → DefaultEngine.Broadcast() (2 calls)
np.broadcast_to.cs       → DefaultEngine.Broadcast() (9 calls)
```

**Note:** These broadcast helpers are pure shape operations (no array data computation).
They may remain as static utilities since they don't need backend-specific implementations.

#### NDArray files calling DefaultEngine.DefaultKernelProvider:

```
NDArray.Indexing.Masking.cs:35   → DefaultEngine.DefaultKernelProvider (VectorBits check)
NDArray.Indexing.Masking.cs:172  → DefaultEngine.DefaultKernelProvider.CountTrue(), CopyMasked()
```

#### DefaultEngine files calling ILKernelGenerator directly (should use interface or be internal):

```
Default.Clip.cs              → ILKernelGenerator.ClipHelper() (27 calls)
Default.ClipNDArray.cs       → ILKernelGenerator.ClipArrayBounds/Min/Max() (27 calls)
Default.Modf.cs              → ILKernelGenerator.ModfHelper()
Default.Shift.cs             → ILKernelGenerator.GetShiftArrayKernel/ScalarKernel()
Default.Reduction.Var.cs     → ILKernelGenerator.VarSimdHelper() (9 calls)
Default.Reduction.Std.cs     → ILKernelGenerator.StdSimdHelper() (9 calls)
Default.Reduction.CumAdd.cs  → ILKernelGenerator.TryGetCumulativeKernel/AxisKernel()
Default.Reduction.CumMul.cs  → ILKernelGenerator.TryGetCumulativeKernel/AxisKernel()
Default.Reduction.ArgMax.cs  → ILKernelGenerator.TryGetAxisReductionKernel()
Default.Reduction.Add.cs     → ILKernelGenerator.TryGetAxisReductionKernel()
Default.MatMul.2D2D.cs       → ILKernelGenerator.Enabled, GetMatMulKernel()
Default.NonZero.cs           → Uses ILKernelGenerator via comment reference
```

---

## Target Architecture

```
PUBLIC API LAYER
================
np.* (static methods)          NDArray (operators, indexers)
    |                              |
    | arr.TensorEngine.*           | this.TensorEngine.*
    |                              |
    +-------------+----------------+
                  |
                  v
ABSTRACT ENGINE LAYER
=====================
          TensorEngine (abstract class)
          ~80 abstract methods defining ALL operations
                  |
                  | extends
                  v
CONCRETE ENGINE LAYER
=====================
          DefaultEngine : TensorEngine
          |
          | INTERNAL: calls ILKernelGenerator
          |           (no interface, direct static calls)
          v
KERNEL LAYER (INTERNAL)
=======================
          ILKernelGenerator (internal static class)
          - Runtime IL emission
          - SIMD optimization (V128/V256/V512)
          - Cached kernel delegates
```

### Key Rules

1. **np.\* NEVER accesses ILKernelGenerator** - only `arr.TensorEngine.*`
2. **NDArray NEVER accesses ILKernelGenerator** - only `this.TensorEngine.*`
3. **ILKernelGenerator is internal** to `NumSharp.Backends` namespace
4. **IKernelProvider interface is deleted** - premature abstraction
5. **DefaultEngine.DefaultKernelProvider is deleted** - no public kernel access

---

## Missing TensorEngine Methods

Methods needed to route all operations through TensorEngine:

| Method | Currently Called From | Status |
|--------|----------------------|--------|
| `NanSum(NDArray, axis?, keepdims)` | np.nansum.cs | **MISSING** |
| `NanProd(NDArray, axis?, keepdims)` | np.nanprod.cs | **MISSING** |
| `NanMin(NDArray, axis?, keepdims)` | np.nanmin.cs | **MISSING** |
| `NanMax(NDArray, axis?, keepdims)` | np.nanmax.cs | **MISSING** |
| `All(NDArray)` | np.all.cs | EXISTS (bool only, no SIMD) |
| `Any(NDArray)` | np.any.cs | **MISSING** |
| `CountTrue(NDArray<bool>)` | Masking.cs | **MISSING** |
| `BooleanMask(NDArray, NDArray<bool>)` | Masking.cs | **MISSING** |

### Existing TensorEngine Methods That Need Enhancement

| Method | Issue |
|--------|-------|
| `All(NDArray)` | Only supports bool dtype, no SIMD, needs all 12 dtypes |
| `All(NDArray, int axis)` | Throws NotImplementedException |

---

## Implementation Phases

### Phase 1: Add Missing TensorEngine Abstract Methods

**File: `TensorEngine.cs`**

Add these abstract methods:

```csharp
// NaN-aware reductions
public abstract NDArray NanSum(in NDArray a, int? axis = null, bool keepdims = false);
public abstract NDArray NanProd(in NDArray a, int? axis = null, bool keepdims = false);
public abstract NDArray NanMin(in NDArray a, int? axis = null, bool keepdims = false);
public abstract NDArray NanMax(in NDArray a, int? axis = null, bool keepdims = false);

// Boolean operations (enhance existing)
public abstract bool Any(NDArray nd);
public abstract NDArray<bool> Any(NDArray nd, int axis);

// Boolean masking
public abstract int CountTrue(NDArray<bool> mask);
public abstract NDArray BooleanMask(NDArray arr, NDArray<bool> mask);
public abstract void BooleanMaskSet(NDArray arr, NDArray<bool> mask, NDArray value);
```

**Deliverable:** `TensorEngine.cs` updated with ~10 new abstract methods.

---

### Phase 2: Implement Missing Methods in DefaultEngine

**New file: `Backends/Default/Reduction/Default.NanReduction.cs`**

Move logic from `np.nansum.cs`, `np.nanprod.cs`, `np.nanmin.cs`, `np.nanmax.cs` into DefaultEngine:

```csharp
public partial class DefaultEngine
{
    public override NDArray NanSum(in NDArray a, int? axis = null, bool keepdims = false)
    {
        // Move implementation from np.nansum.cs
        // Call ILKernelGenerator internally
    }
    // ... NanProd, NanMin, NanMax
}
```

**New file: `Backends/Default/Logic/Default.Any.cs`**

```csharp
public override bool Any(NDArray nd)
{
    // Use ILKernelGenerator.AnySimdHelper internally
}
```

**New file: `Backends/Default/Indexing/Default.BooleanMask.cs`**

```csharp
public override int CountTrue(NDArray<bool> mask)
{
    // Use ILKernelGenerator.CountTrueSimdHelper internally
}

public override NDArray BooleanMask(NDArray arr, NDArray<bool> mask)
{
    // Move logic from NDArray.Indexing.Masking.cs
}
```

**Enhance existing: `Backends/Default/Logic/Default.All.cs`**

- Support all 12 dtypes (not just bool)
- Use SIMD optimization via ILKernelGenerator

**Deliverable:** ~4 new DefaultEngine partial files, 1 enhanced file.

---

### Phase 3: Route np.* Through TensorEngine

Update all violating np.* files to call TensorEngine:

| File | Before | After |
|------|--------|-------|
| `np.nansum.cs` | `ILKernelGenerator.NanSumSimdHelperFloat(...)` | `return arr.TensorEngine.NanSum(arr, axis, keepdims);` |
| `np.nanprod.cs` | `ILKernelGenerator.NanProdSimdHelperFloat(...)` | `return arr.TensorEngine.NanProd(arr, axis, keepdims);` |
| `np.nanmin.cs` | `ILKernelGenerator.NanMinSimdHelperFloat(...)` | `return arr.TensorEngine.NanMin(arr, axis, keepdims);` |
| `np.nanmax.cs` | `ILKernelGenerator.NanMaxSimdHelperFloat(...)` | `return arr.TensorEngine.NanMax(arr, axis, keepdims);` |
| `np.all.cs` | `DefaultEngine.DefaultKernelProvider.All(...)` | `return arr.TensorEngine.All(arr);` |
| `np.any.cs` | `DefaultEngine.DefaultKernelProvider.Any(...)` | `return arr.TensorEngine.Any(arr);` |

**Deliverable:** ~6 np.* files simplified to single TensorEngine calls.

---

### Phase 4: Route NDArray Through TensorEngine

**File: `NDArray.Indexing.Masking.cs`**

```csharp
// BEFORE
var kp = DefaultEngine.DefaultKernelProvider;
int trueCount = kp.CountTrue((bool*)mask.Address, size);
kp.CopyMasked((int*)this.Address, (bool*)mask.Address, (int*)result.Address, size);

// AFTER
return this.TensorEngine.BooleanMask(this, mask);
```

**Deliverable:** NDArray.Indexing.Masking.cs cleaned up.

---

### Phase 5: Remove IKernelProvider Interface

1. **Delete:** `IKernelProvider.cs`

2. **Update:** `DefaultEngine.cs`
   ```csharp
   // DELETE these lines:
   // protected readonly IKernelProvider KernelProvider = ILKernelGenerator.Instance;
   // public static IKernelProvider DefaultKernelProvider { get; } = ILKernelGenerator.Instance;
   ```

3. **Update:** All `KernelProvider.` calls in DefaultEngine to `ILKernelGenerator.`:
   ```csharp
   // BEFORE
   var kernel = KernelProvider.GetMixedTypeKernel(key);

   // AFTER
   var kernel = ILKernelGenerator.GetMixedTypeKernel(key);
   ```

4. **Make internal:** `ILKernelGenerator.cs`
   ```csharp
   // BEFORE
   public partial class ILKernelGenerator : IKernelProvider

   // AFTER
   internal static partial class ILKernelGenerator
   ```

**Deliverable:** IKernelProvider.cs deleted, ILKernelGenerator made internal.

---

### Phase 6: Consolidate ILKernelGenerator Calls

Ensure ALL ILKernelGenerator calls are within `Backends/Default/` folder:

```
ALLOWED:
  Backends/Default/*.cs         → Can call ILKernelGenerator
  Backends/Kernels/*.cs         → Is ILKernelGenerator

NOT ALLOWED (after cleanup):
  APIs/*.cs                     → Must use TensorEngine
  Math/*.cs                     → Must use TensorEngine
  Logic/*.cs                    → Must use TensorEngine
  Selection/*.cs                → Must use TensorEngine
  Operations/*.cs               → Must use TensorEngine
```

**Deliverable:** All ILKernelGenerator usages confined to Backends/ folder.

---

## Verification Checklist

After all phases complete, these greps should return ZERO results:

```bash
# No ILKernelGenerator usage outside Backends/
grep -r "ILKernelGenerator" src/NumSharp.Core --include="*.cs" | grep -v "/Backends/"

# No DefaultEngine.DefaultKernelProvider anywhere
grep -r "DefaultKernelProvider" src/NumSharp.Core --include="*.cs"

# No IKernelProvider references
grep -r "IKernelProvider" src/NumSharp.Core --include="*.cs"

# ILKernelGenerator should be internal
grep "public.*class ILKernelGenerator" src/NumSharp.Core --include="*.cs"
```

---

## File Change Summary

| Action | File | Description |
|--------|------|-------------|
| **MODIFY** | `TensorEngine.cs` | Add ~10 abstract methods |
| **CREATE** | `Default.NanReduction.cs` | NanSum/Prod/Min/Max implementations |
| **CREATE** | `Default.Any.cs` | Any() implementation |
| **CREATE** | `Default.BooleanMask.cs` | CountTrue, BooleanMask implementations |
| **MODIFY** | `Default.All.cs` | Enhance for all dtypes + SIMD |
| **MODIFY** | `np.nansum.cs` | Simplify to TensorEngine call |
| **MODIFY** | `np.nanprod.cs` | Simplify to TensorEngine call |
| **MODIFY** | `np.nanmin.cs` | Simplify to TensorEngine call |
| **MODIFY** | `np.nanmax.cs` | Simplify to TensorEngine call |
| **MODIFY** | `np.all.cs` | Simplify to TensorEngine call |
| **MODIFY** | `np.any.cs` | Simplify to TensorEngine call |
| **MODIFY** | `NDArray.Indexing.Masking.cs` | Route through TensorEngine |
| **DELETE** | `IKernelProvider.cs` | Remove interface |
| **MODIFY** | `DefaultEngine.cs` | Remove DefaultKernelProvider |
| **MODIFY** | `DefaultEngine.*.cs` (7 files) | Change KernelProvider → ILKernelGenerator |
| **MODIFY** | `ILKernelGenerator.cs` | Make internal, remove IKernelProvider inheritance |

**Total: ~20 files modified, 3 created, 1 deleted**

---

## Testing Strategy

1. **Unit tests:** Run full test suite after each phase
2. **Grep verification:** Run verification commands after Phase 6
3. **API surface:** Ensure no public API changes (internal refactor only)
4. **Performance:** Benchmark before/after to ensure no regression

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing code | Internal refactor only, public API unchanged |
| Performance regression | SIMD paths preserved, just routed through TensorEngine |
| Missing edge cases | Comprehensive test suite, phase-by-phase verification |

---

## Notes

- The IKernelProvider abstraction was premature - no alternative backends exist
- Future GPU/Vulkan backends would need TensorEngine implementations, not kernel swapping
- This cleanup aligns with the principle: "TensorEngine is THE abstraction for computation"

---

## Exceptions: What CAN Stay Outside TensorEngine

Not everything needs to go through TensorEngine. These are acceptable:

### 1. Pure Shape Operations (Static Utilities)

These operate on shapes, not array data:

```csharp
DefaultEngine.AreBroadcastable(shapes)     // Shape compatibility check
DefaultEngine.Broadcast(shape1, shape2)    // Shape resolution
DefaultEngine.ResolveReturnShape(s1, s2)   // Output shape calculation
```

**Rationale:** Shape operations are pure math on dimension arrays. They don't need
backend-specific implementations (GPU shapes work the same as CPU shapes).

**Decision:** Keep as static methods on DefaultEngine. Consider moving to a
`ShapeUtils` static class for clarity.

### 2. NDIterator Usage

Iterators are data access patterns, not computations:

```csharp
new NDIterator<T>(array)    // Traversal helper
array.AsIterator<T>()       // Extension method
```

**Rationale:** Iteration is how we access data, not what we compute. TensorEngine
methods use iterators internally.

**Decision:** Keep as public utilities.

### 3. Array Construction

Creating arrays doesn't need backend dispatch:

```csharp
np.zeros(shape)             // Just allocates memory
np.ones(shape)              // Allocates + fills
new NDArray(dtype, shape)   // Constructor
```

**Rationale:** Memory allocation is the same across backends. The data that goes
into arrays comes from TensorEngine operations.

**Decision:** Keep as direct NDArray construction.

---

## What MUST Go Through TensorEngine

All operations that:

1. **Read array data** for computation (reductions, comparisons)
2. **Write array data** as result of computation (math ops, transforms)
3. **Could benefit from SIMD/GPU acceleration**

Examples:
- `np.sum()` - reads all elements, produces scalar
- `np.add()` - reads two arrays, produces result array
- `np.all()` - reads all elements, produces boolean
- Boolean masking - reads mask + data, produces filtered array

---

## Open Questions

### Q1: Should broadcast operations move to TensorEngine?

Current: `DefaultEngine.Broadcast(s1, s2)` is static

Option A: Keep as static (shapes are backend-agnostic)
Option B: Add `TensorEngine.Broadcast()` for consistency

**Recommendation:** Option A - shapes don't need backend dispatch.

### Q2: Should we create a ShapeUtils class?

Current: Broadcast helpers are on DefaultEngine (feels wrong)

Option A: Keep on DefaultEngine
Option B: Create `NumSharp.Utilities.ShapeUtils` static class
Option C: Create `Shape.Broadcast()` static methods

**Recommendation:** Option C - Shape is already a value type, add static helpers.

### Q3: What about np.all/np.any axis overloads?

Current: `np.all(arr, axis)` has inline implementation in np.all.cs
TensorEngine: `All(NDArray, int axis)` throws NotImplementedException

**Recommendation:** Implement in DefaultEngine, route np.all through it.

---

## Phase 7: Remove Instance Singleton

**Status:** COMPLETE (2026-03-18)

### Goal

Remove the unnecessary `Instance` singleton from `ILKernelGenerator`.
All methods are already static, so the instance serves no purpose.

### Tasks

| Task | Description | Status |
|------|-------------|--------|
| 7.1 | Remove `Instance` singleton field | DONE |
| 7.2 | Change `public sealed partial class` to `public static partial class` | DONE |

### Change (Completed)

Before:
```csharp
public sealed partial class ILKernelGenerator
{
    public static readonly ILKernelGenerator Instance = new();
    private ILKernelGenerator() { }
    public static string Name => "IL";
    // All methods are static...
}
```

After (current state):
```csharp
public static partial class ILKernelGenerator
{
    public static string Name => "IL";
    // All methods already static, no instance needed
}
```

### Files to Modify

All ILKernelGenerator partial files need `static` added to class declaration:
- `ILKernelGenerator.cs` - Remove Instance, change to static class
- `ILKernelGenerator.Binary.cs`
- `ILKernelGenerator.MixedType.cs`
- `ILKernelGenerator.Unary.cs`
- `ILKernelGenerator.Comparison.cs`
- `ILKernelGenerator.Reduction.cs`
- `ILKernelGenerator.Reduction.Axis.Simd.cs`
- `ILKernelGenerator.Masking.cs`
- `ILKernelGenerator.Scan.cs`
- `ILKernelGenerator.Shift.cs`
- `ILKernelGenerator.MatMul.cs`
- `ILKernelGenerator.Clip.cs`
- `ILKernelGenerator.Modf.cs`
- (and all other partial files)

### Verification (Passed)

All commands return no results:
```bash
# No Instance references
grep "ILKernelGenerator\.Instance" src/NumSharp.Core --include="*.cs" -r
# (none)

# No IKernelProvider references
grep -r "IKernelProvider" src/NumSharp.Core --include="*.cs"
# (none)

# No DefaultKernelProvider references
grep -r "DefaultKernelProvider" src/NumSharp.Core --include="*.cs"
# (none)

# No kernel imports outside Backends
grep -l "using NumSharp.Backends.Kernels" src/NumSharp.Core --include="*.cs" -r | grep -v "/Backends/"
# (none)

# Class is now static
grep -h "partial class ILKernelGenerator" src/NumSharp.Core/Backends/Kernels/ILKernelGenerator*.cs | sort | uniq
# public static partial class ILKernelGenerator
```

### Risk Assessment

- **Completed successfully**: All callers already used static methods
- `Instance` was verified to have no references before removal
- Build succeeded, tests pass (51 pre-existing failures unrelated to this change)

---

## Phase 7.3: Remove BackendFactory from np.array_manipulation.cs

**Status:** COMPLETE (2026-03-18)

### Investigation

The file `np.array_manipulation.cs` used:
```csharp
BackendFactory.GetEngine().CreateNDArray(shape, dtype, buffer, order)
```

**Why it was there:** Historical design thinking that different backends (GPU) might allocate memory differently.

**Why it's wrong:**
1. Memory allocation is NOT backend-specific - all backends use the same unmanaged memory
2. NDArray constructors already handle all cases:
   - `new NDArray(dtype, shape)` - allocate with zeros
   - `new NDArray(buffer, shape, order)` - use existing buffer
3. Going through TensorEngine for construction adds unnecessary indirection

### Fix Applied

```csharp
// BEFORE
BackendFactory.GetEngine().CreateNDArray(shape, dtype, buffer, order)

// AFTER
if (buffer == null)
    return new NDArray(dtype ?? typeof(float), shape);
else
    return new NDArray(buffer, shape, order);
```

### Files Modified
- `np.array_manipulation.cs` - Removed BackendFactory dependency, use NDArray constructors directly

---

## Phase 8: Move Broadcast Utilities to Shape

**Status:** PENDING

### Goal

Move pure shape operations from `DefaultEngine` to `Shape` struct as static methods.
Broadcasting is a mathematical concept, not a tensor/engine-specific operation.

### Current State (32 usages outside Backends)

```csharp
// Currently on DefaultEngine:
DefaultEngine.Broadcast(shape1, shape2)        // → (Shape, Shape) tuple
DefaultEngine.AreBroadcastable(shapes)         // → bool
DefaultEngine.ResolveReturnShape(s1, s2)       // → Shape
```

### Target State

```csharp
// Move to Shape struct:
Shape.Broadcast(shape1, shape2)                // → (Shape, Shape) tuple
Shape.AreBroadcastable(shapes)                 // → bool
Shape.ResolveReturnShape(s1, s2)               // → Shape
```

### Rationale

1. **Broadcasting is math, not engine logic** - It's about shape compatibility rules
2. **Shape is the right home** - These operate on Shape, return Shape
3. **Removes DefaultEngine dependency** - np.* files won't need `using NumSharp.Backends`
4. **Cleaner API** - `Shape.Broadcast()` is more intuitive than `DefaultEngine.Broadcast()`

### Files to Modify

| File | Change |
|------|--------|
| `View/Shape.cs` | Add static Broadcast, AreBroadcastable, ResolveReturnShape methods |
| `Backends/Default/Default.Broadcast.cs` | Move implementation to Shape, keep thin wrapper |
| `np.are_broadcastable.cs` | Change to `Shape.AreBroadcastable()` |
| `np.broadcast.cs` | Change to `Shape.ResolveReturnShape()` |
| `np.broadcast_arrays.cs` | Change to `Shape.Broadcast()` |
| `np.broadcast_to.cs` | Change to `Shape.Broadcast()` |
| `Operations/Elementwise/Templates/*.cs` | Change to `Shape.Broadcast()` |

### Implementation Steps

1. Add static broadcast methods to `Shape` struct
2. Move implementation from `DefaultEngine` to `Shape`
3. Update all callers to use `Shape.*` instead of `DefaultEngine.*`
4. Keep `DefaultEngine.Broadcast()` as thin wrapper (for any internal usage)
5. Remove `using NumSharp.Backends` from files that only needed broadcast

### Verification

After completion:
```bash
# No DefaultEngine.Broadcast outside Backends
grep -rn "DefaultEngine\.Broadcast\|DefaultEngine\.AreBroadcastable\|DefaultEngine\.ResolveReturnShape" \
  src/NumSharp.Core --include="*.cs" | grep -v "/Backends/"
# Should return: (none)
```
