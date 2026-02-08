# Contributing to NumSharp

This guide covers the practical aspects of contributing to NumSharp.

## Development Philosophy

### The Golden Rule: Match NumPy Exactly

NumSharp aims for **1-to-1 behavioral compatibility** with NumPy 2.x. This means:

1. **Run actual NumPy code first**
2. **Observe and document the exact output**
3. **Replicate that behavior precisely in C#**

```python
# Step 1: Run in Python
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.sum(a, axis=0, keepdims=True)
print(b)        # [[4 6]]
print(b.shape)  # (1, 2)
print(b.dtype)  # int64
```

```csharp
// Step 2: Match in C#
[TestMethod]
public void sum_Axis0_Keepdims()
{
    var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
    var b = np.sum(a, axis: 0, keepdims: true);

    Assert.AreEqual("[[4, 6]]", b.ToString());
    CollectionAssert.AreEqual(new[] { 1, 2 }, b.shape);
    Assert.AreEqual(typeof(int), b.dtype);
}
```

### Edge Cases Matter

NumPy has specific behavior for edge cases. Always test:

- Empty arrays: `np.array([])`
- Scalars vs 0-d arrays: `np.array(5)` vs `5`
- NaN/Inf handling: `np.array([1, np.nan, 3])`
- Type promotion: `np.array([1, 2]) + np.array([1.5])` → float64
- Broadcasting edge cases: shapes like `(3, 1)` + `(1, 4)`
- Negative axis values: `axis=-1`
- Out-of-bounds: What errors does NumPy raise?

### Breaking Changes Are Acceptable

We accept breaking changes to align with NumPy 2.x behavior.

---

## Project Structure for Contributors

### Where to Add New Functions

| Category | Location | Example |
|----------|----------|---------|
| Creation | `src/NumSharp.Core/Creation/` | np.zeros, np.arange, np.concatenate |
| Math | `src/NumSharp.Core/Math/` | np.sum, np.sin, np.power |
| Statistics | `src/NumSharp.Core/Statistics/` | np.mean, np.std, np.var |
| Logic | `src/NumSharp.Core/Logic/` | np.all, np.any, np.allclose |
| Manipulation | `src/NumSharp.Core/Manipulation/` | np.reshape, np.transpose, np.squeeze |
| LinearAlgebra | `src/NumSharp.Core/LinearAlgebra/` | np.dot, np.matmul, np.linalg.* |
| Selection | `src/NumSharp.Core/Selection/` | Indexing, masking |
| Indexing | `src/NumSharp.Core/Indexing/` | np.nonzero |
| Sorting | `src/NumSharp.Core/Sorting_Searching_Counting/` | np.argsort, np.argmax |
| Random | `src/NumSharp.Core/RandomSampling/` | np.random.* |
| File I/O | `src/NumSharp.Core/APIs/` | np.save, np.load |

### File Naming Convention

```
np.{function_name}.cs
```

Examples:
- `np.sum.cs`
- `np.arange.cs`
- `np.concatenate.cs`

### Test Location

```
test/NumSharp.UnitTest/{Category}/{FunctionName}Tests.cs
```

---

## Implementation Patterns

### Pattern 1: Simple np.* Function (Composing Others)

When the function can be built from existing operations:

```csharp
// File: src/NumSharp.Core/Statistics/np.std.cs
namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Compute the standard deviation along the specified axis.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.std.html</remarks>
        public static NDArray std(NDArray a, int? axis = null, NPTypeCode? dtype = null, bool keepdims = false)
        {
            var mean_val = np.mean(a, axis: axis, keepdims: true);
            var diff = a - mean_val;
            var sq = np.power(diff, 2);
            var variance = np.mean(sq, axis: axis, keepdims: keepdims);
            return np.sqrt(variance);
        }
    }
}
```

### Pattern 2: Function Requiring TensorEngine

When you need low-level type-specific optimization:

**Step 1: Add to TensorEngine abstract class**
```csharp
// Backends/TensorEngine.cs
public abstract NDArray NewOperation(in NDArray nd, int? axis = null);
```

**Step 2: Implement in DefaultEngine**
```csharp
// Backends/Default/DefaultEngine.NewOperation.cs
public partial class DefaultEngine
{
    public override NDArray NewOperation(in NDArray nd, int? axis = null)
    {
        // Type switch using Regen pattern
        switch (nd.GetTypeCode)
        {
            case NPTypeCode.Double: return NewOperation_Double(nd, axis);
            case NPTypeCode.Single: return NewOperation_Single(nd, axis);
            // ... all types
        }
    }

    private NDArray NewOperation_Double(in NDArray nd, int? axis)
    {
        // Actual implementation
    }
}
```

**Step 3: Add np.* wrapper**
```csharp
// Math/np.newoperation.cs
public static partial class np
{
    public static NDArray newoperation(NDArray a, int? axis = null)
    {
        return a.TensorEngine.NewOperation(a, axis);
    }
}
```

### Pattern 3: Type Switch (Regen Style)

For operations that need type-specific code:

```csharp
#if _REGEN
    switch (arr.typecode)
    {
        %foreach supported_dtypes,supported_dtypes_lowercase%
        case NPTypeCode.#1: return Process<#2>(arr);
        %
        default:
            throw new NotSupportedException();
    }
#else
    switch (arr.typecode)
    {
        case NPTypeCode.Boolean: return Process<bool>(arr);
        case NPTypeCode.Byte: return Process<byte>(arr);
        case NPTypeCode.Int16: return Process<short>(arr);
        case NPTypeCode.UInt16: return Process<ushort>(arr);
        case NPTypeCode.Int32: return Process<int>(arr);
        case NPTypeCode.UInt32: return Process<uint>(arr);
        case NPTypeCode.Int64: return Process<long>(arr);
        case NPTypeCode.UInt64: return Process<ulong>(arr);
        case NPTypeCode.Char: return Process<char>(arr);
        case NPTypeCode.Double: return Process<double>(arr);
        case NPTypeCode.Single: return Process<float>(arr);
        case NPTypeCode.Decimal: return Process<decimal>(arr);
        default:
            throw new NotSupportedException();
    }
#endif
```

---

## Working with Core Types

### Creating NDArrays in Code

```csharp
// From shape (zeros)
var a = new NDArray(NPTypeCode.Double, new Shape(3, 4));

// From .NET array
var b = new NDArray(new double[] { 1, 2, 3, 4 });
var c = new NDArray(new double[,] { { 1, 2 }, { 3, 4 } });

// Using np.* functions
var d = np.zeros(3, 4);
var e = np.arange(10);
var f = np.array(new[] { 1.0, 2.0, 3.0 });
```

### Accessing Data

```csharp
// Typed access (preferred for performance)
unsafe
{
    double* ptr = (double*)nd.Address;
    for (int i = 0; i < nd.size; i++)
        ptr[i] = i * 2.0;
}

// Via ArraySlice
ArraySlice<double> slice = nd.Storage.GetData<double>();
Span<double> span = slice.AsSpan();

// Via iterator (handles sliced arrays correctly)
using var iter = new NDIterator<double>(nd);
while (iter.HasNext())
{
    double val = iter.MoveNext();
}
```

### Working with Shape

```csharp
// Create shape
Shape shape = new Shape(3, 4, 5);
Shape scalar = Shape.Scalar;
Shape vector = new Shape(10);

// Properties
int ndim = shape.NDim;           // 3
int size = shape.size;           // 60
int[] dims = shape.Dimensions;   // [3, 4, 5]
int[] strides = shape.Strides;   // [20, 5, 1]

// Checks
bool isScalar = shape.IsScalar;
bool isContiguous = shape.IsContiguous;
bool isSliced = shape.IsSliced;

// Coordinate ↔ Offset
int offset = shape.GetOffset(1, 2, 3);  // Linear index
int[] coords = shape.GetCoordinates(offset);
```

### Working with Slices

```csharp
// Parse string notation
Slice[] slices = Slice.ParseSlices("1:3, :, -1");

// Programmatic slicing
var slice = new Slice(start: 1, stop: 5, step: 2);
var index = Slice.Index(3);  // Single element

// Apply to NDArray
NDArray view = nd[slices];
NDArray view2 = nd[1, Slice.All, Slice.Index(-1)];
```

---

## Testing Guidelines

### Test File Structure

```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Math
{
    [TestClass]
    public class np_sum_Tests
    {
        [TestMethod]
        public void sum_1DArray_ReturnsScalar()
        {
            // Arrange
            var a = np.array(new[] { 1, 2, 3, 4, 5 });

            // Act
            var result = np.sum(a);

            // Assert
            Assert.AreEqual(15, result.GetInt32());
        }

        [TestMethod]
        public void sum_2DArray_Axis0_SumsColumns()
        {
            var a = np.array(new int[,] { { 1, 2 }, { 3, 4 } });
            var result = np.sum(a, axis: 0);

            CollectionAssert.AreEqual(new[] { 4, 6 }, result.ToArray<int>());
        }

        [TestMethod]
        public void sum_EmptyArray_ReturnsZero()
        {
            var a = np.array(new double[0]);
            var result = np.sum(a);

            Assert.AreEqual(0.0, result.GetDouble());
        }

        [TestMethod]
        public void sum_WithNaN_PropagatesNaN()
        {
            var a = np.array(new[] { 1.0, double.NaN, 3.0 });
            var result = np.sum(a);

            Assert.IsTrue(double.IsNaN(result.GetDouble()));
        }
    }
}
```

### What to Test

1. **Basic functionality** - Normal use case
2. **All relevant dtypes** - int, float, double at minimum
3. **Different shapes** - 1D, 2D, 3D+
4. **Axis parameter** - All valid axes, negative axes
5. **keepdims parameter** - true and false
6. **Edge cases**:
   - Empty arrays
   - Scalar inputs
   - Single-element arrays
   - NaN/Inf values
   - Type promotion

### Running Tests

```bash
cd test/NumSharp.UnitTest
dotnet test
```

---

## Performance Considerations

### Contiguous vs Sliced Arrays

```csharp
// Fast path: contiguous memory
if (nd.Shape.IsContiguous)
{
    unsafe
    {
        double* ptr = (double*)nd.Address;
        // Direct pointer arithmetic
    }
}
else
{
    // Slow path: use iterator or shape.GetOffset
    using var iter = new NDIterator<double>(nd);
    // ...
}
```

### Parallel Threshold

DefaultEngine uses `Parallel.For` for arrays > 85,000 elements:

```csharp
public const int ParallelAbove = 84999;

if (size > ParallelAbove)
{
    Parallel.For(0, size, i => { /* ... */ });
}
else
{
    for (int i = 0; i < size; i++) { /* ... */ }
}
```

### Avoid Allocations in Hot Paths

```csharp
// Bad: allocates array each call
int[] indices = new int[ndim];

// Good: stackalloc for small, fixed-size
Span<int> indices = stackalloc int[ndim];

// Good: reuse iterator
using var iter = new NDIterator<double>(nd);
var moveNext = iter.MoveNext;  // Cache delegate
while (iter.HasNext())
    moveNext();
```

---

## Common Pitfalls

### 1. Forgetting View Semantics (CRITICAL)

```csharp
// WRONG: modifies original
var view = original["1:3"];
view[0] = 999;  // original[1] is now 999!

// RIGHT: explicit copy if needed
var copy = original["1:3"].copy();
copy[0] = 999;  // original unchanged
```

### 2. Type Assumptions

```csharp
// WRONG: assumes type
double val = nd.GetDouble();  // Throws if not double

// RIGHT: check or convert
if (nd.dtype == typeof(double))
    double val = nd.GetDouble();
// OR
var converted = nd.astype(np.float64);
```

### 3. Axis Handling

```csharp
// WRONG: forgetting negative axis
if (axis >= nd.ndim) throw new ArgumentOutOfRangeException();

// RIGHT: normalize negative axis first
if (axis < 0) axis = nd.ndim + axis;
if (axis < 0 || axis >= nd.ndim) throw new ArgumentOutOfRangeException();
```

### 4. Empty Array Edge Cases

```csharp
// WRONG: crashes on empty
var result = arr[0];  // IndexOutOfRange if arr.size == 0

// RIGHT: check first
if (arr.size == 0)
    return np.array(Array.Empty<double>());
```

---

## Documentation Standards

### XML Documentation

```csharp
/// <summary>
/// Return the sum of array elements over a given axis.
/// </summary>
/// <param name="a">Input array.</param>
/// <param name="axis">Axis or axes along which a sum is performed.
/// The default, axis=None, will sum all of the elements of the input array.</param>
/// <param name="dtype">The type of the returned array and of the accumulator
/// in which the elements are summed.</param>
/// <param name="keepdims">If this is set to True, the axes which are reduced
/// are left in the result as dimensions with size one.</param>
/// <returns>An array with the same shape as a, with the specified axis removed.</returns>
/// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.sum.html</remarks>
public static NDArray sum(NDArray a, int? axis = null, NPTypeCode? dtype = null, bool keepdims = false)
```

### Always Include NumPy Reference Link

```csharp
/// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.{function}.html</remarks>
```

---

## Supported Types Reference

All operations must handle these 12 types:

| NPTypeCode | C# Type | NPTypeCode | C# Type |
|------------|---------|------------|---------|
| Boolean | bool | Int64 | long |
| Byte | byte | UInt64 | ulong |
| Int16 | short | Char | char |
| UInt16 | ushort | Single | float |
| Int32 | int | Double | double |
| UInt32 | uint | Decimal | decimal |

Use `InfoOf<T>` for type information without reflection:
```csharp
var size = InfoOf<double>.Size;      // 8
var code = InfoOf<float>.NPTypeCode; // NPTypeCode.Single
var zero = InfoOf<int>.Zero;         // 0
```

---

## Questions?

If you're unsure about implementation details:

1. Check existing similar functions in the codebase
2. Run the Python equivalent and document behavior
3. Look at NumPy source code for complex edge cases
4. Open an issue for discussion before large changes
