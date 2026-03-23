# NumSharp Exception Redesign

## Goals

1. **NumPy message compatibility**: First line matches NumPy exactly
2. **Rich context**: Second line provides NumSharp-specific debugging info
3. **Catchable hierarchy**: Exceptions grouped by category
4. **Replace blank throws**: 158+ throws with no message get proper context

---

## Document Status

| Section | Status |
|---------|--------|
| Current exceptions inventory | COMPLETE |
| NumPy message verification | COMPLETE (10/10 verified from source) |
| New hierarchy design | COMPLETE |
| Migration mapping | COMPLETE (382 throws categorized) |
| Throw helper design | COMPLETE |
| Implementation | NOT STARTED |

**Last Updated:** Based on NumPy v2.4.2 source at `src/numpy/`

---

## Part 1: Current NumSharp Exceptions

### Existing Exception Classes

| Class | File | Inherits | Current Usage |
|-------|------|----------|---------------|
| `NumSharpException` | `Exceptions/NumSharpException.cs` | `Exception` | Base class, has `ThrowIfNotWriteable` helper |
| `IncorrectShapeException` | `Exceptions/IncorrectShapeException.cs` | `NumSharpException` | ~40 throws |
| `IncorrectTypeException` | `Exceptions/IncorrectTypeException.cs` | `NumSharpException` | ~5 throws |
| `IncorrectSizeException` | `Exceptions/IncorrectSizeException.cs` | `NumSharpException` | ~19 throws |
| `AxisOutOfRangeException` | `Exceptions/AxisOutOfRangeException.cs` | `ArgumentOutOfRangeException` | ~2 throws |

### Current Throw Patterns (501 total)

| Pattern | Count | Quality | Target Exception |
|---------|-------|---------|------------------|
| `NotSupportedException()` | 156 | BAD - no message | `UnsupportedDTypeException` |
| `NotSupportedException("...")` | 74 | MIXED | various |
| `ArgumentNullException(nameof(...))` | 79 | OK | keep |
| `ArgumentException("...")` | 56 | OK | keep or migrate |
| `ArgumentOutOfRangeException()` | 33 | BAD - no message | `AxisOutOfRangeException` |
| `ArgumentOutOfRangeException("...")` | 14 | OK | review |
| `IncorrectShapeException()` | 18 | BAD - no message | `BroadcastException`/`ReshapeException` |
| `IncorrectShapeException("...")` | 22 | MIXED | review |
| `IncorrectSizeException("...")` | 19 | OK | keep or `ScalarConversionException` |
| `IncorrectTypeException()` | 3 | BAD - no message | `UnsupportedDTypeException` |
| `Exception("...")` | 13 | BAD - generic | `InternalException` or specific |
| `IndexOutOfRangeException("...")` | 7 | OK | `IndexException` |
| `InvalidOperationException("...")` | 5 | MIXED | various |
| `ReadOnlyException()` | 3 | BAD - System.Data | `ReadOnlyArrayException` |

---

## Part 2: NumPy Error Messages (Source Verified)

### Shape Errors

| Error Type | NumPy Message | NumPy Source Location | Verified |
|------------|---------------|----------------------|----------|
| Broadcast failure (simple) | `operands could not be broadcast together with shapes (2,3) (4,)` | `_core/src/multiarray/nditer_constr.c:1740` | YES |
| Broadcast failure (with requested) | `operands could not be broadcast together with shapes (2,3) and requested shape (5,)` | `_core/src/multiarray/nditer_constr.c:1752` | YES |
| Reshape size mismatch | `cannot reshape array of size 24 into shape (3,6)` | `_core/src/multiarray/shape.c:467` | YES |
| Dimension mismatch | *(varies by context - no single standard message)* | various | NO |

### Axis Errors

| Error Type | NumPy Message | NumPy Source Location | Verified |
|------------|---------------|----------------------|----------|
| Axis out of range | `axis {axis} is out of bounds for array of dimension {ndim}` | `numpy/exceptions.py:193` | YES |
| With prefix | `{prefix}: axis {axis} is out of bounds for array of dimension {ndim}` | `numpy/exceptions.py:195` | YES |

**Note:** NumPy's `AxisError` inherits from BOTH `ValueError` AND `IndexError` for backwards compatibility.

### Index Errors

| Error Type | NumPy Message | NumPy Source Location | Verified |
|------------|---------------|----------------------|----------|
| Index out of bounds | `index {obj} is out of bounds for axis {axis} with size {size}` | `lib/_function_base_impl.py:5354` | YES |

### Type/DType Errors

| Error Type | NumPy Message | NumPy Source Location | Verified |
|------------|---------------|----------------------|----------|
| Ufunc not supported | `ufunc '{name}' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule '{rule}'` | `_core/src/umath/ufunc_type_resolution.c:1997` | YES |
| Loop not supported | `loop of ufunc does not support argument {i} of type {type} which has no callable {method} method` | `_core/src/umath/loops.c.src:233` | YES |
| Type promotion failure | `The DType <class 'numpy.dtype[X]'> could not be promoted by <class 'numpy.dtype[Y]'>...` | `numpy/exceptions.py:227` (DTypePromotionError) | YES |

**Note:** NumPy's `DTypePromotionError` inherits from `TypeError`.

### Size/Empty Errors

| Error Type | NumPy Message | NumPy Source Location | Verified |
|------------|---------------|----------------------|----------|
| Empty array reduction | `zero-size array to reduction operation {funcname} which has no identity` | `_core/src/umath/reduction.c:101` | YES |
| Empty argmax/argmin | `attempt to get {func_name} of an empty sequence` | `_core/src/multiarray/calculation.c:142` | YES |

### Read-Only Errors

| Error Type | NumPy Message | NumPy Source Location | Verified |
|------------|---------------|----------------------|----------|
| Assignment to read-only | `{name} is read-only` | `_core/src/multiarray/arrayobject.c:560` | YES |

**Note:** The `{name}` parameter is typically "assignment destination", "output array", or "array".

---

## Part 3: New Exception Hierarchy

```
NumSharpException (base)
|
+-- ShapeException
|   +-- BroadcastException
|   +-- ReshapeException
|   +-- DimensionMismatchException
|
+-- DTypeException
|   +-- UnsupportedDTypeException
|   +-- TypeMismatchException
|
+-- AxisOutOfRangeException (existing, enhanced)
|
+-- IndexException
|
+-- SizeException
|   +-- EmptyArrayException
|   +-- ScalarConversionException
|
+-- MemoryLayoutException
|   +-- NonContiguousException
|   +-- ReadOnlyArrayException
|
+-- InternalException
```

---

## Part 4: Detailed Migration Mapping

### 4.1 NotSupportedException() - 156 cases

**By File (top contributors):**

| File | Count | Context | New Exception |
|------|-------|---------|---------------|
| `Backends/Unmanaged/UnmanagedMemoryBlock.Casting.cs` | 34 | Type switch default | `UnsupportedDTypeException` |
| `Utilities/Converts.cs` | 30 | Type switch default | `UnsupportedDTypeException` |
| `Backends/Unmanaged/ArraySlice.cs` | 20 | Type switch default | `UnsupportedDTypeException` |
| `Backends/NPTypeCode.cs` | 18 | Type switch default | `UnsupportedDTypeException` |
| `Backends/Iterators/NDIteratorExtensions.cs` | 10 | Type switch default | `UnsupportedDTypeException` |
| `Backends/Iterators/MultiIterator.cs` | 10 | Type switch default | `UnsupportedDTypeException` |
| `Backends/Unmanaged/UnmanagedStorage.cs` | 8 | Type switch default | `UnsupportedDTypeException` |
| `Casting/Implicit/NdArray.Implicit.Array.cs` | 6 | Type switch default | `UnsupportedDTypeException` |
| `Backends/Unmanaged/UnmanagedMemoryBlock.cs` | 6 | Type switch default | `UnsupportedDTypeException` |
| Other files | 14 | Various | Review individually |

**Pattern:** Most are `default:` cases in `switch (typeCode)` statements. All should become:
```csharp
default: throw new UnsupportedDTypeException(typeCode, "OperationName");
// or using helper:
default: Throw.UnsupportedDType(typeCode);  // auto-captures method name
```

**Sample migration:**
```csharp
// BEFORE (no context):
default:
    throw new NotSupportedException();

// AFTER (NumPy-style + context):
default:
    throw new UnsupportedDTypeException(typeCode, "FromMultiDimArray");
    // Message: "FromMultiDimArray() does not support dtype 'complex64'"
```

### 4.2 ArgumentOutOfRangeException() - 72 blank cases

**IMPORTANT:** These are NOT axis errors! They are type switch defaults.

**By File (top contributors):**

| File | Count | Context | New Exception |
|------|-------|---------|---------------|
| `Backends/Iterators/NDIterator.Cast.*.cs` | 48 | Type switch default | `UnsupportedDTypeException` |
| `Utilities/ArrayConvert.cs` | 15 | Type switch default | `UnsupportedDTypeException` |
| `Backends/Iterators/NDIterator.cs` | 4 | Type switch default | `UnsupportedDTypeException` |
| `Backends/Iterators/NDIterator.template.cs` | 4 | Template file | `UnsupportedDTypeException` |
| `Backends/Unmanaged/UnmanagedMemoryBlock.cs` | 1 | Type switch default | `UnsupportedDTypeException` |

### 4.2b ArgumentOutOfRangeException(nameof(axis)) - 25 cases

**These ARE axis errors and should use `AxisOutOfRangeException`:**

| File | Line | Current | New Message |
|------|------|---------|-------------|
| `np.any.cs` | 74 | `nameof(axis)` only | `axis {axis} is out of bounds for array of dimension {ndim}` |
| `np.all.cs` | 73 | `nameof(axis)` only | `axis {axis} is out of bounds for array of dimension {ndim}` |
| `NDArray.cs` | 557 | `nameof(axis)` only | `axis {axis} is out of bounds for array of dimension {ndim}` |
| `np.squeeze.cs` | 35 | `nameof(axis)` only | `axis {axis} is out of bounds for array of dimension {ndim}` |
| `np.nansum.cs` | 154 | `nameof(axis)` only | `axis {axis} is out of bounds for array of dimension {ndim}` |
| `np.nanvar.cs` | 164 | Has message already | KEEP (already NumPy format) |
| `np.nanstd.cs` | 164 | Has message already | KEEP (already NumPy format) |
| `np.nanmean.cs` | 111 | Has message already | KEEP (already NumPy format) |
| `np.count_nonzero.cs` | 39 | `nameof(axis)` only | `axis {axis} is out of bounds for array of dimension {ndim}` |
| Various reduction files | ~15 | `nameof(axis)` or `nameof(axis_)` | `axis {axis} is out of bounds for array of dimension {ndim}` |

**Migration:** Replace with `throw new AxisOutOfRangeException(ndim, axis);`

### 4.3 IncorrectShapeException() - 18 cases

**By context:**

| File | Count | Context | New Exception | New Message |
|------|-------|---------|---------------|-------------|
| `Casting/Implicit/NdArray.Implicit.ValueTypes.cs` | 13 | `if (nd.ndim != 0)` - scalar conversion | `ScalarConversionException` | `cannot convert {ndim}D array to scalar` |
| `RandomSampling/np.random.uniform.cs` | 1 | Shape validation | `BroadcastException` | Review context |
| `Creation/np.mgrid.cs` | 1 | Invalid input | `ShapeException` | Review context |
| `Operations/Elementwise/NdArray.DetermineEmptyResult.cs` | 1 | Shape mismatch | `BroadcastException` | NumPy msg |
| `LinearAlgebra/NdArray.multi_dot.cs` | 2 | Commented out | N/A | Dead code |

**Note:** The scalar conversion cases (13 of 18) should use a new `ScalarConversionException`:
```csharp
// BEFORE:
if (nd.ndim != 0)
    throw new IncorrectShapeException();

// AFTER:
if (nd.ndim != 0)
    throw new ScalarConversionException(nd.ndim);
    // Message: "cannot convert 2D array to scalar (must be 0D)"
```

### 4.4 Exception("...") - 13 cases

| File | Line | Current Message | New Exception | New Message |
|------|------|-----------------|---------------|-------------|
| `np.save.cs` | 185 | `""` | `InternalException` | TBD |
| `np.vstack.cs` | 14 | `"Input arrays can not be empty"` | `EmptyArrayException` | NumPy msg TBD |
| `np.vstack.cs` | 20 | `"Arrays mush have same shapes"` | `BroadcastException` | NumPy msg TBD |
| `NDArray.Indexing.Selection.Setter.cs` | 194 | `"if (nd.typecode == NPTypeCode.Boolean)"` | `InternalException` | internal error |
| `np.load.cs` | 314, 322, 383 | `""` | `InternalException` | TBD |
| `NdArrayFromJaggedArr.cs` | 33 | `"Multi dim arrays are not allowed here!"` | `ArgumentException` | TBD |
| `NDArray.matrix_power.cs` | 10 | `"matrix_power just work with int >= 0"` | `ArgumentOutOfRangeException` | TBD |
| `Default.Transpose.cs` | 72 | `"source and destination arguments must have..."` | `IncorrectShapeException` | TBD |
| `Default.Transpose.cs` | 106 | `"start arg requires start <= n + 1..."` | `ArgumentOutOfRangeException` | TBD |
| `Default.Transpose.cs` | 141 | `"axes don't match array"` | `AxisOutOfRangeException` | TBD |
| `Default.Transpose.cs` | 150 | `"repeated axis in transpose"` | `ArgumentException` | TBD |

### 4.5 ReadOnlyException() - 3 cases

**Source:** `System.Data.ReadOnlyException` - wrong namespace for a math library!

| File | Line | Context | New Exception | New Message |
|------|------|---------|---------------|-------------|
| `Utilities/NpzDictionary.cs` | 143 | `Add()` method | `NotSupportedException` | `NpzDictionary is read-only` |
| `Utilities/NpzDictionary.cs` | 148 | `Clear()` method | `NotSupportedException` | `NpzDictionary is read-only` |
| `Utilities/NpzDictionary.cs` | 161 | `Remove()` method | `NotSupportedException` | `NpzDictionary is read-only` |

**Note:** These are collection operations, not array operations. Use standard `NotSupportedException` with message.

### 4.6 NumSharpException.ThrowReadOnly() - existing helper

Already exists in `NumSharpException.cs:50` with NumPy-compatible format:
```csharp
throw new NumSharpException($"{name} is read-only");
// Default name: "assignment destination"
```

**Decision:** Move this to `Throw.ReadOnly()` helper and create `ReadOnlyArrayException` for array-specific cases.

---

## Part 5: New Exception Class Specifications

### 5.1 BroadcastException

```csharp
/// <summary>Shapes cannot be broadcast together.</summary>
public class BroadcastException : ShapeException
{
    public int[] ShapeA { get; }
    public int[] ShapeB { get; }

    // NumPy message (verified): "operands could not be broadcast together with shapes (2,3) (4,)"
    public BroadcastException(int[] a, int[] b)
        : base($"operands could not be broadcast together with shapes {Fmt(a)} {Fmt(b)}") { ... }

    // NumPy message with requested shape
    public BroadcastException(int[] a, int[] b, int[] requested)
        : base($"operands could not be broadcast together with shapes {Fmt(a)} and requested shape {Fmt(requested)}") { ... }
}
```

### 5.2 ReshapeException

```csharp
/// <summary>Array cannot be reshaped to target shape.</summary>
public class ReshapeException : ShapeException
{
    public int SourceSize { get; }
    public int[] TargetShape { get; }

    // NumPy message (verified): "cannot reshape array of size 24 into shape (3,6)"
    public ReshapeException(int sourceSize, int[] targetShape)
        : base($"cannot reshape array of size {sourceSize} into shape {Fmt(targetShape)}") { ... }
}
```

### 5.3 AxisOutOfRangeException (existing - enhanced)

```csharp
/// <summary>Axis is out of bounds.</summary>
/// <remarks>Inherits from both ValueError and IndexError like NumPy's AxisError.</remarks>
public class AxisOutOfRangeException : ArgumentOutOfRangeException, INumSharpException
{
    public int Axis { get; }
    public int NDim { get; }
    public int[]? Shape { get; }  // NEW: optional shape context

    // NumPy message (verified): "axis {axis} is out of bounds for array of dimension {ndim}"
    public AxisOutOfRangeException(int ndim, int axis)
        : base("axis", $"axis {axis} is out of bounds for array of dimension {ndim}") { ... }

    // With shape context (NumSharp enhancement)
    public AxisOutOfRangeException(int ndim, int axis, int[] shape)
        : base("axis", $"axis {axis} is out of bounds for array of dimension {ndim}\n  -> array.shape={Fmt(shape)}") { ... }
}
```

### 5.4 IndexException

```csharp
/// <summary>Index is out of bounds.</summary>
public class IndexException : NumSharpException
{
    public int Index { get; }
    public int Axis { get; }
    public int Size { get; }

    // NumPy message (verified): "index {obj} is out of bounds for axis {axis} with size {size}"
    public IndexException(int index, int axis, int size)
        : base($"index {index} is out of bounds for axis {axis} with size {size}") { ... }
}
```

### 5.5 UnsupportedDTypeException

```csharp
/// <summary>Operation does not support the given dtype.</summary>
public class UnsupportedDTypeException : DTypeException
{
    public NPTypeCode TypeCode { get; }
    public string? Operation { get; }

    // NumPy message (verified): "ufunc '{name}' not supported for the input types..."
    // Simplified for NumSharp: "{operation}() does not support dtype '{dtype}'"
    public UnsupportedDTypeException(NPTypeCode typeCode, string operation)
        : base($"{operation}() does not support dtype '{typeCode}'") { ... }

    // For switch default (auto-captures caller)
    public static UnsupportedDTypeException ForSwitch(
        NPTypeCode typeCode,
        [CallerMemberName] string? operation = null)
        => new(typeCode, operation ?? "unknown");
}
```

### 5.6 EmptyArrayException

```csharp
/// <summary>Operation cannot be performed on an empty array.</summary>
public class EmptyArrayException : SizeException
{
    public string? Operation { get; }

    // NumPy message (verified): "zero-size array to reduction operation {funcname} which has no identity"
    public static EmptyArrayException NoIdentity(string operation)
        => new($"zero-size array to reduction operation {operation} which has no identity");

    // NumPy message (verified): "attempt to get {func_name} of an empty sequence"
    public static EmptyArrayException EmptySequence(string operation)
        => new($"attempt to get {operation} of an empty sequence");
}
```

### 5.7 ScalarConversionException

```csharp
/// <summary>Array cannot be converted to scalar (not 0-dimensional).</summary>
public class ScalarConversionException : SizeException
{
    public int NDim { get; }

    // NumSharp-specific (NumPy uses different mechanism via .item())
    public ScalarConversionException(int ndim)
        : base($"cannot convert {ndim}D array to scalar (must be 0D)") { ... }
}
```

### 5.8 ReadOnlyArrayException

```csharp
/// <summary>Array is read-only and cannot be modified.</summary>
public class ReadOnlyArrayException : MemoryLayoutException
{
    // NumPy message (verified): "{name} is read-only"
    public ReadOnlyArrayException(string name = "assignment destination")
        : base($"{name} is read-only") { }

    // With context (NumSharp enhancement)
    public static ReadOnlyArrayException BroadcastView()
        => new("assignment destination is read-only\n  -> array is a broadcast view (stride=0 dimension)");
}
```

### 5.9 NonContiguousException

```csharp
/// <summary>Operation requires contiguous memory layout.</summary>
public class NonContiguousException : MemoryLayoutException
{
    // NumSharp-specific (NumPy handles this differently)
    public static NonContiguousException CannotPin()
        => new("cannot pin reference when array is sliced or broadcasted");

    public static NonContiguousException CannotSpan()
        => new("cannot create Span over non-contiguous storage");
}
```

### 5.10 InternalException

```csharp
/// <summary>Internal error that should not occur in normal operation.</summary>
public class InternalException : NumSharpException
{
    // For "this shouldn't happen" cases - replaces generic Exception throws
    public InternalException(string message)
        : base($"internal error: {message} (please report this bug)") { }
}
```

---

## Part 6: Throw Helper Specifications

```csharp
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace NumSharp;

/// <summary>
/// Centralized throw helpers with context capture.
/// All methods are [DoesNotReturn] to help flow analysis.
/// </summary>
public static class Throw
{
    // ===== Shape Errors =====

    [DoesNotReturn]
    public static void CannotBroadcast(int[] a, int[] b)
        => throw new BroadcastException(a, b);

    [DoesNotReturn]
    public static void CannotBroadcast(Shape a, Shape b)
        => throw new BroadcastException(a.Dimensions, b.Dimensions);

    [DoesNotReturn]
    public static void CannotReshape(int size, int[] target)
        => throw new ReshapeException(size, target);

    // ===== DType Errors =====

    /// <summary>
    /// For type switch default cases. Auto-captures method name.
    /// </summary>
    [DoesNotReturn]
    public static void UnsupportedDType(NPTypeCode typeCode, [CallerMemberName] string? op = null)
        => throw UnsupportedDTypeException.ForSwitch(typeCode, op);

    // ===== Axis Errors =====

    [DoesNotReturn]
    public static void AxisOutOfRange(int axis, int ndim)
        => throw new AxisOutOfRangeException(ndim, axis);

    [DoesNotReturn]
    public static void AxisOutOfRange(int axis, int ndim, int[] shape)
        => throw new AxisOutOfRangeException(ndim, axis, shape);

    /// <summary>
    /// Normalize axis (handle negative) and throw if out of range.
    /// </summary>
    public static int NormalizeAxis(int axis, int ndim)
    {
        if (axis < -ndim || axis >= ndim)
            throw new AxisOutOfRangeException(ndim, axis);
        return axis < 0 ? ndim + axis : axis;
    }

    // ===== Index Errors =====

    [DoesNotReturn]
    public static void IndexOutOfBounds(int index, int axis, int size)
        => throw new IndexException(index, axis, size);

    // ===== Size Errors =====

    [DoesNotReturn]
    public static void EmptyArray([CallerMemberName] string? op = null)
        => throw EmptyArrayException.NoIdentity(op ?? "unknown");

    [DoesNotReturn]
    public static void EmptySequence([CallerMemberName] string? op = null)
        => throw EmptyArrayException.EmptySequence(op ?? "unknown");

    [DoesNotReturn]
    public static void CannotConvertToScalar(int ndim)
        => throw new ScalarConversionException(ndim);

    // ===== Memory Layout Errors =====

    [DoesNotReturn]
    public static void CannotPin()
        => throw NonContiguousException.CannotPin();

    [DoesNotReturn]
    public static void ReadOnly(string name = "assignment destination")
        => throw new ReadOnlyArrayException(name);

    /// <summary>
    /// Check if array is writeable, throw if not.
    /// Equivalent to NumPy's PyArray_FailUnlessWriteable.
    /// </summary>
    public static void IfNotWriteable(in Shape shape, string name = "assignment destination")
    {
        if (!shape.IsWriteable)
            throw new ReadOnlyArrayException(name);
    }

    // ===== Null Checks =====

    public static void IfNull<T>([NotNull] T? value, [CallerArgumentExpression(nameof(value))] string? name = null)
        where T : class
    {
        if (value is null)
            throw new ArgumentNullException(name);
    }

    // ===== Internal Errors =====

    [DoesNotReturn]
    public static void Internal(string message)
        => throw new InternalException(message);
}
```

### Usage Examples

```csharp
// BEFORE (no context):
default:
    throw new NotSupportedException();

// AFTER (auto-captures method name):
default:
    Throw.UnsupportedDType(typeCode);  // -> "MethodName() does not support dtype 'int8'"

// BEFORE:
if (axis < 0 || axis >= ndim)
    throw new ArgumentOutOfRangeException(nameof(axis));

// AFTER:
axis = Throw.NormalizeAxis(axis, ndim);  // handles negative, throws with NumPy message

// BEFORE:
if (!shape.IsWriteable)
    throw new ReadOnlyException();

// AFTER:
Throw.IfNotWriteable(shape);  // -> "assignment destination is read-only"
```

---

## Part 7: Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `Exceptions/ShapeException.cs` | Base + BroadcastException + ReshapeException + DimensionMismatchException |
| `Exceptions/DTypeException.cs` | Base + UnsupportedDTypeException + TypeMismatchException |
| `Exceptions/IndexException.cs` | Index out of bounds |
| `Exceptions/SizeException.cs` | Base + EmptyArrayException + ScalarConversionException |
| `Exceptions/MemoryLayoutException.cs` | Base + NonContiguousException + ReadOnlyArrayException |
| `Exceptions/InternalException.cs` | For "should not happen" cases |
| `Exceptions/Throw.cs` | Static throw helpers |

### Modified Files

| File | Changes |
|------|---------|
| `Exceptions/NumSharpException.cs` | Keep as base, remove `ThrowIfNotWriteable` (moved to `Throw`) |
| `Exceptions/IncorrectShapeException.cs` | DEPRECATE - redirect to ShapeException |
| `Exceptions/IncorrectTypeException.cs` | DEPRECATE - redirect to DTypeException |
| `Exceptions/IncorrectSizeException.cs` | DEPRECATE - redirect to SizeException |
| `Exceptions/AxisOutOfRangeException.cs` | ENHANCE - add optional shape context |

---

## Appendix A: Throw Count Summary

| Pattern | Count | Target |
|---------|-------|--------|
| `NotSupportedException()` blank | 156 | `UnsupportedDTypeException` |
| `NotSupportedException("...")` with message | 74 | Review - keep good ones |
| `ArgumentOutOfRangeException()` blank | 72 | `UnsupportedDTypeException` (type switches) |
| `ArgumentOutOfRangeException(nameof(axis))` | 25 | `AxisOutOfRangeException` |
| `IncorrectShapeException()` blank | 18 | Various (13 are `ScalarConversionException`) |
| `IncorrectShapeException("...")` with message | 22 | Review - align with NumPy messages |
| `IncorrectSizeException("...")` | 19 | Keep or `ScalarConversionException` |
| `Exception("...")` generic | 13 | `InternalException` or specific type |
| `ReadOnlyException()` (System.Data!) | 3 | `NotSupportedException` (collection) |
| **Total throws to migrate** | ~382 | |

## Appendix B: NumPy Source References (v2.4.2)

All paths relative to `src/numpy/numpy/`:

| Error Type | Source File | Line | Verified Message |
|------------|-------------|------|------------------|
| Broadcast | `_core/src/multiarray/nditer_constr.c` | 1740 | `operands could not be broadcast together with shapes %S` |
| Broadcast+requested | `_core/src/multiarray/nditer_constr.c` | 1752 | `...with shapes %S and requested shape %S` |
| Reshape | `_core/src/multiarray/shape.c` | 467 | `cannot reshape array of size %zd into shape %S` |
| AxisError | `exceptions.py` | 193 | `axis {axis} is out of bounds for array of dimension {ndim}` |
| AxisError (class) | `exceptions.py` | 108-196 | Inherits from `ValueError, IndexError` |
| IndexError | `lib/_function_base_impl.py` | 5354 | `index {obj} is out of bounds for axis {axis} with size {size}` |
| Empty reduction | `_core/src/umath/reduction.c` | 101 | `zero-size array to reduction operation %s which has no identity` |
| Empty argmax | `_core/src/multiarray/calculation.c` | 142 | `attempt to get %s of an empty sequence` |
| Read-only | `_core/src/multiarray/arrayobject.c` | 560 | `%s is read-only` |
| Ufunc not supported | `_core/src/umath/ufunc_type_resolution.c` | 1997 | `ufunc '%s' not supported for the input types...` |
| DTypePromotionError | `exceptions.py` | 199-246 | Inherits from `TypeError` |

## Appendix C: Files to Modify (Migration Plan)

### Phase 1: Create New Exception Classes (0 breaking changes)

| File | Action |
|------|--------|
| `Exceptions/ShapeException.cs` | CREATE: Base + BroadcastException + ReshapeException |
| `Exceptions/DTypeException.cs` | CREATE: Base + UnsupportedDTypeException + TypeMismatchException |
| `Exceptions/IndexException.cs` | CREATE |
| `Exceptions/SizeException.cs` | CREATE: Base + EmptyArrayException + ScalarConversionException |
| `Exceptions/MemoryLayoutException.cs` | CREATE: Base + NonContiguousException + ReadOnlyArrayException |
| `Exceptions/InternalException.cs` | CREATE |
| `Exceptions/Throw.cs` | CREATE: Static helpers |
| `Exceptions/AxisOutOfRangeException.cs` | MODIFY: Add Shape property |

### Phase 2: Migrate High-Impact Throws

| File Pattern | Count | Change |
|--------------|-------|--------|
| `*/default: throw new NotSupportedException();` | 156 | `Throw.UnsupportedDType(typeCode)` |
| `*/default: throw new ArgumentOutOfRangeException();` | 72 | `Throw.UnsupportedDType(typeCode)` |
| `throw new ArgumentOutOfRangeException(nameof(axis))` | 25 | `throw new AxisOutOfRangeException(ndim, axis)` |

### Phase 3: Deprecate Old Exceptions

| Old Class | New Class | Migration |
|-----------|-----------|-----------|
| `IncorrectShapeException` | `ShapeException` / `BroadcastException` / `ReshapeException` | Add `[Obsolete]`, inherit from new base |
| `IncorrectTypeException` | `DTypeException` / `UnsupportedDTypeException` | Add `[Obsolete]`, inherit from new base |
| `IncorrectSizeException` | `SizeException` / `ScalarConversionException` | Add `[Obsolete]`, inherit from new base |
