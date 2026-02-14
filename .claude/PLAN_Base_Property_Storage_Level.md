# Plan: NumPy-compatible `.base` Property (Storage-Level Tracking)

## Overview

Implement NumPy's `.base` property by tracking view ownership at the `UnmanagedStorage` level, not `NDArray` level. This is architecturally cleaner because `UnmanagedStorage` is where data sharing actually occurs.

## Design

### Core Concept

```
NumPy:     ndarray.base → the ndarray that owns the data (or None)
NumSharp:  UnmanagedStorage._baseStorage → the storage that owns the data
           NDArray.@base → computed property that wraps _baseStorage
```

### Architecture

```
Original Array (owns data):
  NDArray a
    └── Storage A (_baseStorage = null, InternalArray = DATA)

View (shares data):
  NDArray b = a[2:5]
    └── Storage B (_baseStorage = A, InternalArray = DATA)  ← same DATA

View of View (chains to original):
  NDArray c = b[1:2]
    └── Storage C (_baseStorage = A, InternalArray = DATA)  ← still points to A
```

## Implementation

### Step 1: Add `_baseStorage` field to UnmanagedStorage

**File:** `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.cs`

```csharp
public partial class UnmanagedStorage : ICloneable
{
    // ... existing fields ...

    /// <summary>
    /// The original storage this is a view of, or null if this storage owns its data.
    /// Always points to the ultimate owner (not intermediate views).
    /// </summary>
    internal UnmanagedStorage? _baseStorage;

    // ... rest of class ...
}
```

### Step 2: Update Alias methods to set `_baseStorage`

**File:** `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Cloning.cs`

```csharp
public UnmanagedStorage Alias()
{
    var r = new UnmanagedStorage();
    r._shape = _shape;
    r._typecode = _typecode;
    r._dtype = _dtype;
    if (InternalArray != null)
        r.SetInternalArray(InternalArray);
    r.Count = _shape.size;
    r._baseStorage = _baseStorage ?? this;  // ← ADD: chain to original
    return r;
}

public UnmanagedStorage Alias(Shape shape)
{
    var r = new UnmanagedStorage();
    r._typecode = _typecode;
    r._dtype = _dtype;
    if (InternalArray != null)
        r.SetInternalArray(InternalArray);
    r._shape = shape;
    r.Count = shape.size;
    r._baseStorage = _baseStorage ?? this;  // ← ADD: chain to original
    return r;
}

public UnmanagedStorage Alias(ref Shape shape)
{
    var r = new UnmanagedStorage();
    r._shape = shape;
    r._typecode = _typecode;
    r._dtype = _dtype;
    if (InternalArray != null)
        r.SetInternalArray(InternalArray);
    r.Count = shape.size;
    r._baseStorage = _baseStorage ?? this;  // ← ADD: chain to original
    return r;
}
```

### Step 3: Update GetView to propagate `_baseStorage`

**File:** `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Slicing.cs`

Review `GetView()` - it calls `Alias()` internally, so should automatically work. Need to verify.

### Step 4: Update CreateBroadcastedUnsafe

**File:** `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.cs` lines 148-153

Only the overload that takes `UnmanagedStorage` needs updating (the one taking `IArraySlice` creates owned data):

```csharp
public static UnmanagedStorage CreateBroadcastedUnsafe(UnmanagedStorage storage, Shape shape)
{
    var ret = new UnmanagedStorage();
    ret._Allocate(shape, storage.InternalArray);
    ret._baseStorage = storage._baseStorage ?? storage;  // ← ADD: track original
    return ret;
}
```

The `IArraySlice` overload (line 135) stays unchanged - it creates owned data.

### Step 5: Add computed `@base` property to NDArray

**File:** `src/NumSharp.Core/Backends/NDArray.cs`

```csharp
/// <summary>
/// NumPy-compatible: The array owning the memory, or null if this array owns its data.
/// </summary>
/// <remarks>
/// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.base.html
/// Note: Unlike NumPy, this returns a new NDArray wrapper each time.
/// For checking view status, use: arr.@base != null
/// </remarks>
public NDArray? @base => Storage._baseStorage is { } bs
    ? WrapOwned(bs)
    : null;
```

### Step 6: Add WrapOwned factory (if not exists)

**File:** `src/NumSharp.Core/Backends/NDArray.cs`

```csharp
/// <summary>
/// Wrap an UnmanagedStorage in an NDArray. Used for Clone, Scalar, computed .base, etc.
/// </summary>
internal static NDArray WrapOwned(UnmanagedStorage storage)
    => new NDArray(storage);
```

### Step 7: Remove manual @base assignments

Remove all the manual `view.@base = source.@base ?? source` assignments we added earlier:

- `NDArray.Indexing.cs`
- `NDArray.Indexing.Selection.Getter.cs`
- `Default.Transpose.cs`
- `np.expand_dims.cs`
- `np.broadcast_to.cs`
- `NdArray.ReShape.cs`
- `NDArray.cs` (view method)
- `Default.Reduction.*.cs`

These are now handled automatically by `Alias()`.

### Step 8: Remove `@base` field from NDArray

The field we added:
```csharp
public NDArray? @base;  // ← REMOVE this field
```

Replace with computed property from Step 5.

## Files to Modify

| File | Change |
|------|--------|
| `Backends/Unmanaged/UnmanagedStorage.cs` | Add `_baseStorage` field, update `CreateBroadcastedUnsafe(storage, shape)` |
| `Backends/Unmanaged/UnmanagedStorage.Cloning.cs` | Set `_baseStorage` in all 3 `Alias()` methods |
| `Backends/NDArray.cs` | Replace `@base` field with computed property |

### Files to Revert (remove manual @base assignments)

| File | Lines to Revert |
|------|-----------------|
| `Selection/NDArray.Indexing.cs` | Lines 63-66, 80-83 |
| `Selection/NDArray.Indexing.Selection.Getter.cs` | Lines 52-56, 117-121 |
| `Backends/Default/ArrayManipulation/Default.Transpose.cs` | Lines 192-194 |
| `Manipulation/np.expand_dims.cs` | Lines 11-13 |
| `Creation/np.broadcast_to.cs` | Lines 70-73, 111-114, 153-156 |
| `Creation/NdArray.ReShape.cs` | Lines 37-39, 63-65, 89-91, 107-109 |
| `Backends/NDArray.cs` | Lines 476-479 (view method) |
| `Backends/Default/Math/Reduction/*.cs` | Various (7 files) |

## Investigation Results

### 1. CreateBroadcastedUnsafe Location

**File:** `UnmanagedStorage.cs` lines 135-153

Two overloads:
```csharp
// Creates new storage from raw slice - OWNS data, no _baseStorage
public static UnmanagedStorage CreateBroadcastedUnsafe(IArraySlice arraySlice, Shape shape)

// Creates view of existing storage - SHARES data, needs _baseStorage
public static UnmanagedStorage CreateBroadcastedUnsafe(UnmanagedStorage storage, Shape shape)
```

**Action:** Only update the second overload (line 148) to propagate `_baseStorage`.

### 2. GetView() Path Analysis

**File:** `UnmanagedStorage.Slicing.cs`

All paths go through `Alias()`:
- Line 49: `view.Alias(view.Shape.ExpandDimension(axis))`
- Line 98: `return Alias()`
- Line 116: `return Alias(slicedShape)`

**Exception:** Line 108 creates `new UnmanagedStorage(clonedData, cleanShape)` for broadcast materialization - this is a COPY (owns data), correctly should NOT set `_baseStorage`.

**Action:** No changes needed - Alias() handles it.

### 3. Clone() Behavior

**File:** `UnmanagedStorage.Cloning.cs` line 200

```csharp
public UnmanagedStorage Clone() => new UnmanagedStorage(CloneData(), _shape.Clone(...));
```

Uses constructor, not Alias(). Creates owned data.

**Action:** No changes needed - Clone() should NOT set `_baseStorage`.

### 4. np.* Functions

All use constructors like `new NDArray(dtype, shape)` or `new UnmanagedStorage(...)`.
These create owned data, `_baseStorage` stays null by default.

**Action:** No changes needed.

### 5. NDArray<T> Generic Class

Inherits from NDArray. Uses `base(storage)` constructors.

**Action:** No changes needed - inherits behavior from NDArray.

## Behavior Comparison

| Scenario | NumPy | NumSharp (this plan) |
|----------|-------|---------------------|
| `a = np.arange(10)` | `a.base is None` | `a.@base == null` |
| `b = a[2:5]` | `b.base is a` | `b.@base != null` (wraps a's storage) |
| `c = b[1:2]` | `c.base is a` | `c.@base` wraps a's storage |
| `d = a.copy()` | `d.base is None` | `d.@base == null` |
| `c.base is a` | `True` | `False` (different wrapper) |

**Note:** Object identity (`is`) won't match, but semantic equivalence does.

## Testing

```csharp
[Test]
public void Base_OriginalArray_IsNull()
{
    var a = np.arange(10);
    Assert.That(a.@base, Is.Null);
}

[Test]
public void Base_SliceOfOriginal_PointsToOriginalStorage()
{
    var a = np.arange(10);
    var b = a["2:5"];
    Assert.That(b.@base, Is.Not.Null);
    Assert.That(b.@base!.Storage, Is.SameAs(a.Storage));
}

[Test]
public void Base_SliceOfSlice_PointsToOriginalStorage()
{
    var a = np.arange(10);
    var b = a["2:7"];
    var c = b["1:3"];
    Assert.That(c.@base, Is.Not.Null);
    Assert.That(c.@base!.Storage, Is.SameAs(a.Storage));
}

[Test]
public void Base_Copy_IsNull()
{
    var a = np.arange(10);
    var b = np.copy(a);
    Assert.That(b.@base, Is.Null);
}

[Test]
public void Base_Reshape_PointsToOriginalStorage()
{
    var a = np.arange(12);
    var b = a.reshape(3, 4);
    Assert.That(b.@base, Is.Not.Null);
    Assert.That(b.@base!.Storage, Is.SameAs(a.Storage));
}

[Test]
public void Base_Transpose_PointsToOriginalStorage()
{
    var a = np.arange(12).reshape(3, 4);
    var b = a.T;
    // b.@base points to reshape's storage, which points to a's storage
    Assert.That(b.@base, Is.Not.Null);
}
```

## Rollback Plan

If issues arise:
1. Remove `_baseStorage` field from UnmanagedStorage
2. Revert Alias() methods
3. Go back to NDArray-level tracking with manual assignments

## Open Questions

1. Should `@base` cache the created NDArray to avoid repeated allocations?
2. Should we add a `bool OwnsData` property for cleaner checks?
3. Impact on memory/GC - does `_baseStorage` reference prevent collection?
