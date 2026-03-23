# NumSharp Exceptions

NumSharp provides a structured exception hierarchy designed for NumPy compatibility and clear error messages. All exceptions inherit from `NumSharpException` and follow NumPy's error message format.

## Design Principles

1. **NumPy message compatibility** - The first line of each error message matches NumPy exactly
2. **Rich context** - Additional lines provide NumSharp-specific debugging information
3. **Catchable hierarchy** - Exceptions are grouped by category for flexible error handling
4. **Self-documenting** - Exception properties expose the values that caused the error

---

## Exception Hierarchy

```
NumSharpException (base)
|
+-- ShapeException
|   +-- BroadcastException
|   +-- ReshapeException
|
+-- DTypeException
|   +-- UnsupportedDTypeException
|
+-- AxisOutOfRangeException
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

## Shape Exceptions

Shape exceptions occur when array dimensions are incompatible with the requested operation.

### BroadcastException

Raised when shapes cannot be broadcast together.

```csharp
var a = np.ones(3, 4);
var b = np.ones(5);
var c = a + b;  // BroadcastException
```

**Message format:**
```
operands could not be broadcast together with shapes (3, 4) (5,)
```

**Properties:**
- `ShapeA` - First operand's dimensions
- `ShapeB` - Second operand's dimensions

### ReshapeException

Raised when an array cannot be reshaped to the target shape.

```csharp
var a = np.arange(12);
var b = a.reshape(3, 5);  // ReshapeException: 12 elements cannot fit in 3x5=15
```

**Message format:**
```
cannot reshape array of size 12 into shape (3, 5)
```

**Properties:**
- `SourceSize` - Original array size
- `TargetShape` - Requested shape dimensions

---

## DType Exceptions

DType exceptions occur when an operation doesn't support the array's data type.

### UnsupportedDTypeException

Raised when an operation cannot handle the given dtype.

```csharp
var a = np.array(new[] { 1m, 2m, 3m });  // decimal
var b = np.sin(a);  // UnsupportedDTypeException: sin() doesn't support decimal
```

**Message format:**
```
sin() does not support dtype 'decimal'
```

**Properties:**
- `TypeCode` - The unsupported `NPTypeCode`
- `Operation` - Name of the operation that failed

**Usage in switch statements:**

```csharp
switch (arr.typecode)
{
    case NPTypeCode.Double: return ProcessDouble(arr);
    case NPTypeCode.Single: return ProcessSingle(arr);
    // ... other types
    default:
        throw new UnsupportedDTypeException(arr.typecode, nameof(MyOperation));
}
```

---

## Axis Exceptions

### AxisOutOfRangeException

Raised when an axis parameter is outside the valid range for the array's dimensions.

```csharp
var a = np.zeros(3, 4);  // 2D array
var b = np.sum(a, axis: 5);  // AxisOutOfRangeException
```

**Message format:**
```
axis 5 is out of bounds for array of dimension 2
```

With shape context:
```
axis 5 is out of bounds for array of dimension 2
  -> array.shape=(3, 4)
```

**Properties:**
- `Axis` - The invalid axis value
- `NDim` - Number of dimensions in the array
- `Shape` - Optional array dimensions for context

**Negative axis handling:**

NumSharp normalizes negative axes like NumPy. Use `Throw.NormalizeAxis()`:

```csharp
// axis=-1 on a 3D array becomes axis=2
int normalizedAxis = Throw.NormalizeAxis(axis, arr.ndim);
```

---

## Index Exceptions

### IndexException

Raised when an index is out of bounds for a dimension.

```csharp
var a = np.arange(5);
var b = a[10];  // IndexException
```

**Message format:**
```
index 10 is out of bounds for axis 0 with size 5
```

**Properties:**
- `Index` - The invalid index value
- `Axis` - Which axis the index applies to
- `Size` - Size of that axis

---

## Size Exceptions

Size exceptions occur when array size is incompatible with the operation.

### EmptyArrayException

Raised when an operation cannot be performed on an empty array.

```csharp
var a = np.array(new double[0]);
var b = np.argmax(a);  // EmptyArrayException
```

**Message formats:**

For reductions without identity:
```
zero-size array to reduction operation add which has no identity
```

For argmax/argmin:
```
attempt to get argmax of an empty sequence
```

**Properties:**
- `Operation` - Name of the operation that failed

### ScalarConversionException

Raised when converting a non-scalar array to a scalar value.

```csharp
var a = np.array(new[] { 1, 2, 3 });
int x = (int)a;  // ScalarConversionException
```

**Message format:**
```
cannot convert 1D array to scalar (must be 0D)
```

**Properties:**
- `NDim` - Actual number of dimensions

---

## Memory Layout Exceptions

Memory layout exceptions occur when an operation requires a specific memory layout.

### NonContiguousException

Raised when an operation requires contiguous memory but the array is sliced or strided.

```csharp
var a = np.arange(20).reshape(4, 5);
var view = a["::2"];  // Non-contiguous view
var ptr = view.GetPinnableReference();  // NonContiguousException
```

**Message formats:**
```
cannot pin reference when array is sliced or broadcasted
```

```
cannot create Span over non-contiguous storage
```

### ReadOnlyArrayException

Raised when attempting to modify a read-only array (such as a broadcast view).

```csharp
var a = np.array(new[] { 1, 2, 3 });
var b = np.broadcast_to(a, new Shape(3, 3));  // b is read-only
b[0, 0] = 999;  // ReadOnlyArrayException
```

**Message format:**
```
assignment destination is read-only
  -> array is a broadcast view (stride=0 dimension)
```

---

## Internal Exceptions

### InternalException

Raised for errors that should not occur during normal operation. These indicate bugs in NumSharp.

**Message format:**
```
internal error: {description} (please report this bug)
```

If you encounter an `InternalException`, please report it at https://github.com/SciSharp/NumSharp/issues

---

## Throw Helpers

NumSharp provides static helpers in the `Throw` class for common validation patterns.

### Shape Validation

```csharp
// Throws BroadcastException if shapes are incompatible
Throw.CannotBroadcast(a.Shape, b.Shape);

// Throws ReshapeException if sizes don't match
Throw.CannotReshape(arr.size, targetShape);
```

### Axis Validation

```csharp
// Throws AxisOutOfRangeException if invalid
Throw.AxisOutOfRange(axis, arr.ndim);

// Normalize negative axis and validate (returns normalized value)
int normalizedAxis = Throw.NormalizeAxis(axis, arr.ndim);
```

### DType Validation

```csharp
// For switch statement defaults - auto-captures method name
default:
    Throw.UnsupportedDType(arr.typecode);
    // Message: "MethodName() does not support dtype 'complex64'"
```

### Size Validation

```csharp
// Throws EmptyArrayException with appropriate message
if (arr.size == 0)
    Throw.EmptyArray();  // For reductions

if (arr.size == 0)
    Throw.EmptySequence();  // For argmax/argmin

// Throws ScalarConversionException
if (arr.ndim != 0)
    Throw.CannotConvertToScalar(arr.ndim);
```

### Memory Layout Validation

```csharp
// Throws NonContiguousException if array cannot be pinned
if (!arr.Shape.IsContiguous)
    Throw.CannotPin();

// Throws ReadOnlyArrayException if array is not writeable
Throw.IfNotWriteable(arr.Shape);
```

### Null Validation

```csharp
// Throws ArgumentNullException with parameter name
Throw.IfNull(array);  // Auto-captures "array" as parameter name
```

---

## Catching Exceptions

The hierarchy allows catching at different granularities:

```csharp
try
{
    var result = np.sum(a, axis: 5);
}
catch (AxisOutOfRangeException ex)
{
    // Handle specifically axis errors
    Console.WriteLine($"Invalid axis {ex.Axis} for {ex.NDim}D array");
}
catch (ShapeException ex)
{
    // Handle any shape-related error (broadcast, reshape, etc.)
}
catch (NumSharpException ex)
{
    // Handle any NumSharp error
}
```

---

## NumPy Compatibility Reference

| NumSharp Exception | NumPy Equivalent | NumPy Message Format |
|--------------------|------------------|----------------------|
| `BroadcastException` | `ValueError` | `operands could not be broadcast together with shapes {shapes}` |
| `ReshapeException` | `ValueError` | `cannot reshape array of size {n} into shape {shape}` |
| `AxisOutOfRangeException` | `AxisError` | `axis {axis} is out of bounds for array of dimension {ndim}` |
| `IndexException` | `IndexError` | `index {i} is out of bounds for axis {ax} with size {sz}` |
| `EmptyArrayException` | `ValueError` | `zero-size array to reduction operation {op} which has no identity` |
| `ReadOnlyArrayException` | `ValueError` | `{name} is read-only` |
| `UnsupportedDTypeException` | `TypeError` | `ufunc '{name}' not supported for the input types...` |

---

## Migration from Legacy Exceptions

The following legacy exceptions are deprecated:

| Deprecated | Replacement |
|------------|-------------|
| `IncorrectShapeException` | `ShapeException`, `BroadcastException`, or `ReshapeException` |
| `IncorrectTypeException` | `DTypeException` or `UnsupportedDTypeException` |
| `IncorrectSizeException` | `SizeException`, `EmptyArrayException`, or `ScalarConversionException` |

The deprecated exceptions still work but will be removed in a future version.
