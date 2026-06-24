# NumSharp Architecture Guide

This document provides an in-depth technical overview of the NumSharp library internals, design decisions, and development practices.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [Memory Management](#memory-management)
4. [Type System](#type-system)
5. [API Layer](#api-layer)
6. [Slicing and Views](#slicing-and-views)
7. [Broadcasting](#broadcasting)
8. [Iterator System](#iterator-system)
9. [Code Generation](#code-generation)
10. [Development Workflow](#development-workflow)
11. [Technical Debt & Known Issues](#technical-debt--known-issues)
12. [Future Roadmap](#future-roadmap)

---

## Project Overview

NumSharp is a .NET port of Python's NumPy library, providing n-dimensional array operations for scientific computing in C#. The library aims to match NumPy's API as closely as possible, including edge cases like NaN handling, multi-type operations, and broadcasting semantics.

### Goals

- **API Compatibility**: Match NumPy 2.x API (upgraded from original 1.x target)
- **1-to-1 Behavior**: Replicate NumPy behavior exactly, including random state/seed
- **Performance**: Achieve competitive performance through unmanaged memory and unsafe code
- **Ecosystem Integration**: Support TensorFlow.NET, ML.NET, and other .NET ML frameworks

### Project Structure

```
NumSharp/
├── src/
│   └── NumSharp.Core/           # Main library
│       ├── APIs/                # np.* static entry points
│       ├── Backends/            # TensorEngine, Storage, Iterators
│       │   ├── Default/         # Pure C# engine implementation
│       │   ├── Unmanaged/       # Memory management
│       │   ├── Iterators/       # NDIterator system
│       │   └── LAPACK/          # Linear algebra bindings
│       ├── Creation/            # np.zeros, np.arange, np.ones, etc.
│       ├── Math/                # np.sum, np.sin, np.log, etc.
│       ├── LinearAlgebra/       # np.dot, np.matmul, np.linalg.*
│       ├── Manipulation/        # reshape, transpose, flatten
│       ├── RandomSampling/      # np.random.*
│       ├── Statistics/          # mean, std, var, argmax
│       ├── Logic/               # np.all, np.any, np.allclose
│       ├── Selection/           # Indexing, slicing, masking
│       ├── Generics/            # NDArray<T> typed wrapper
│       ├── View/                # Shape, Slice, ViewInfo
│       ├── Operations/          # Operator overloads
│       └── Utilities/           # Type helpers, converters
├── test/
│   └── NumSharp.UnitTest/       # MSTest unit tests
├── examples/
│   └── NeuralNetwork.NumSharp/  # Neural network example
└── docs/                        # Documentation assets
```

---

## Core Architecture

### Three Pillars: NDArray, UnmanagedStorage, Shape

The library is built on three fundamental classes that work together:

```
┌─────────────────────────────────────────────────────────┐
│                        NDArray                          │
│  - Public API surface                                   │
│  - Operator overloads (+, -, *, /, indexing)           │
│  - References TensorEngine for computations             │
├─────────────────────────────────────────────────────────┤
│                    UnmanagedStorage                     │
│  - Holds raw data in unmanaged memory                   │
│  - Manages ArraySlice<T> for each dtype                 │
│  - Handles allocation, slicing views, data access       │
├─────────────────────────────────────────────────────────┤
│                         Shape                           │
│  - Dimensions and strides                               │
│  - Coordinate ↔ offset translation                      │
│  - Slicing, broadcasting, contiguity tracking           │
└─────────────────────────────────────────────────────────┘
```

### NDArray

`NDArray` is the primary user-facing class, analogous to `numpy.ndarray`:

```csharp
// Key properties
public Shape Shape { get; }           // Dimensions
public Type dtype { get; }            // Element type
public int ndim { get; }              // Number of dimensions
public int size { get; }              // Total element count
public int[] strides { get; }         // Byte strides per dimension
public UnmanagedStorage Storage { get; }
public TensorEngine TensorEngine { get; }

// Key operations
public NDArray this[string slice] { get; set; }  // "1:3, :, -1"
public NDArray reshape(params int[] shape);
public NDArray T { get; }                         // Transpose
public NDArray astype(Type dtype);
```

### TensorEngine

`TensorEngine` is an abstract class defining all computational operations. This abstraction exists to potentially support alternative backends (GPU, SIMD, MKL) in the future.

```csharp
public abstract class TensorEngine
{
    // Allocation
    public abstract UnmanagedStorage GetStorage(NPTypeCode typeCode);

    // Arithmetic
    public abstract NDArray Add(in NDArray lhs, in NDArray rhs);
    public abstract NDArray Subtract(in NDArray lhs, in NDArray rhs);
    public abstract NDArray Multiply(NDArray lhs, NDArray rhs);
    public abstract NDArray Divide(in NDArray lhs, in NDArray rhs);

    // Reduction
    public abstract NDArray ReduceAdd(in NDArray arr, int? axis_, bool keepdims, ...);
    public abstract NDArray ReduceArgMax(NDArray arr, int? axis_);

    // Unary functions
    public abstract NDArray Sqrt(in NDArray nd, NPTypeCode? typeCode);
    public abstract NDArray Log(in NDArray nd, NPTypeCode? typeCode);
    public abstract NDArray Exp(in NDArray nd, NPTypeCode? typeCode);
    // ... 30+ more operations

    // Linear algebra
    public abstract NDArray Dot(in NDArray x, in NDArray y);
    public abstract NDArray Matmul(NDArray lhs, NDArray rhs);
}
```

The `DefaultEngine` is the current implementation - pure micro-optimized C# that uses `Parallel.For` for arrays exceeding 85,000 elements.

---

## Memory Management

### Why Unmanaged Memory?

NumSharp uses unmanaged memory (raw pointers) rather than managed arrays or `Span<T>`/`Memory<T>`. This decision was made ~5 years ago based on extensive benchmarking when Span/Memory were not yet properly supported across the .NET ecosystem.

**Benefits:**
- Zero-copy slicing (views share underlying memory)
- Direct pointer arithmetic for maximum performance
- No GC pressure for large arrays
- Interop-friendly for native libraries

**Trade-offs:**
- Requires careful memory management
- Must use `unsafe` code blocks
- Manual disposal considerations

### UnmanagedStorage

```csharp
public class UnmanagedStorage
{
    internal IArraySlice InternalArray;  // Type-erased ArraySlice<T>

    public Shape Shape { get; }
    public Type DType { get; }
    public NPTypeCode TypeCode { get; }
    public unsafe void* Address { get; }  // Raw pointer to data

    // Get typed view
    public ArraySlice<T> GetData<T>() where T : unmanaged;

    // Slicing returns views, not copies
    public UnmanagedStorage GetView(params Slice[] slices);
}
```

### ArraySlice<T>

The generic `ArraySlice<T>` wraps unmanaged memory with type safety:

```csharp
public readonly struct ArraySlice<T> : IArraySlice where T : unmanaged
{
    public readonly unsafe T* Address;
    public readonly int Count;

    public ref T this[int index] { get; }
    public Span<T> AsSpan();
}
```

---

## Type System

### Supported Types (NPTypeCode)

NumSharp supports 12 primitive types:

| NPTypeCode | C# Type | Size (bytes) |
|------------|---------|--------------|
| Boolean    | bool    | 1 |
| Byte       | byte    | 1 |
| Int16      | short   | 2 |
| UInt16     | ushort  | 2 |
| Int32      | int     | 4 |
| UInt32     | uint    | 4 |
| Int64      | long    | 8 |
| UInt64     | ulong   | 8 |
| Char       | char    | 2 |
| Single     | float   | 4 |
| Double     | double  | 8 |
| Decimal    | decimal | 16 |

### InfoOf<T> - Type Information Cache

To avoid runtime reflection costs, type information is cached statically:

```csharp
public class InfoOf<T>
{
    public static readonly int Size;           // Byte size
    public static readonly NPTypeCode NPTypeCode;
    public static readonly T Zero;             // default(T)
    public static readonly T MaxValue;
    public static readonly T MinValue;
}

// Usage
var size = InfoOf<double>.Size;  // 8
var code = InfoOf<float>.NPTypeCode;  // NPTypeCode.Single
```

### NDArray<T> - Generic Typed Wrapper

For type-safe operations, `NDArray<T>` provides a generic wrapper:

```csharp
public class NDArray<T> : NDArray where T : unmanaged
{
    public new T this[params int[] indices] { get; set; }
    public new ArraySlice<T> Array { get; }
    public new unsafe T* Address { get; }
    public new NDArray<T> this[string slice] { get; }
}

// Usage
NDArray<float> arr = np.zeros<float>(3, 4);
float val = arr[1, 2];  // Direct typed access
```

---

## API Layer

### The `np` Static Class

The `np` class is the primary entry point, mirroring Python's `import numpy as np`:

```csharp
public static partial class np
{
    // Type aliases (matching NumPy)
    public static readonly Type float64 = typeof(double);
    public static readonly Type float32 = typeof(float);
    public static readonly Type int32 = typeof(int);
    public static readonly Type int64 = typeof(long);
    public static readonly Type bool_ = typeof(bool);

    // Constants
    public const double pi = Math.PI;
    public const double e = Math.E;
    public static readonly double nan = double.NaN;
    public static readonly double inf = double.PositiveInfinity;

    // Random module
    public static NumPyRandom random { get; }

    // Creation: np.zeros, np.ones, np.arange, np.linspace, etc.
    // Math: np.sum, np.mean, np.sin, np.cos, np.exp, np.log, etc.
    // Linear algebra: np.dot, np.matmul, np.linalg.*
    // Manipulation: np.reshape, np.transpose, np.concatenate, etc.
}
```

### Function Implementation Patterns

There are two patterns for implementing `np.*` functions:

**Pattern 1: Delegating to TensorEngine**
```csharp
// np.sum delegates to engine
public static NDArray sum(NDArray a, int? axis = null, ...)
{
    return a.TensorEngine.Sum(a, axis, typeCode, keepdims);
}
```

**Pattern 2: Composing other np functions**
```csharp
// np.std composes np.mean and other operations
public static NDArray std(NDArray a, int? axis = null, ...)
{
    var mean = np.mean(a, axis, keepdims: true);
    var diff = a - mean;
    var sq = np.power(diff, 2);
    return np.sqrt(np.mean(sq, axis, keepdims: keepdims));
}
```

---

## Slicing and Views

### Slice Class

The `Slice` class parses and represents Python-style slice notation:

```csharp
public class Slice
{
    public int? Start;      // null = from beginning
    public int? Stop;       // null = to end
    public int Step;        // default 1
    public bool IsIndex;    // Single element, reduces dimension
    public bool IsEllipsis; // ... fills remaining dimensions
    public bool IsNewAxis;  // np.newaxis inserts dimension

    // Special instances
    public static readonly Slice All;      // ":"
    public static readonly Slice None;     // "0:0"
    public static readonly Slice Ellipsis; // "..."
    public static readonly Slice NewAxis;  // "np.newaxis"

    // Parsing
    public static Slice[] ParseSlices(string notation);  // "1:3, :, -1"
}
```

### Slice Examples

```csharp
nd[":"]           // All elements
nd["1:5"]         // Elements 1-4
nd["::2"]         // Every other element
nd["-1"]          // Last element (reduces dimension)
nd["1::-1"]       // Reverse from index 1
nd[":, 0"]        // All rows, first column
nd["..., -1"]     // Last element of last dimension
```

### View Semantics

**Critical**: Slicing returns views, not copies. The view shares memory with the original:

```csharp
var original = np.arange(10);
var view = original["2:5"];  // View, shares memory
view[0] = 999;               // Modifies original[2]!

var copy = original["2:5"].copy();  // Explicit copy
```

### SliceDef - Internal Representation

For efficient computation, slices are converted to `SliceDef`:

```csharp
public struct SliceDef
{
    public int Start;  // Absolute start index
    public int Step;   // Step size (can be negative)
    public int Count;  // Number of elements (-1 = single index)

    // Merge handles recursive slicing
    public SliceDef Merge(SliceDef other);
}
```

---

## Broadcasting

Broadcasting allows operations between arrays of different shapes by virtually expanding dimensions:

```csharp
var a = np.ones(3, 4);      // Shape: (3, 4)
var b = np.ones(4);         // Shape: (4,)
var c = a + b;              // Broadcasting: b treated as (1, 4) → (3, 4)
```

### Broadcast Resolution

```csharp
public static (Shape, Shape) Broadcast(Shape left, Shape right)
{
    // Rules:
    // 1. Align shapes from the right
    // 2. Dimensions must be equal OR one must be 1
    // 3. Dimension of 1 is "stretched" to match
}
```

### MultiIterator

For element-wise operations with broadcasting:

```csharp
public static class MultiIterator
{
    // Creates paired iterators with broadcasting
    public static (NDIterator, NDIterator) GetIterators(
        UnmanagedStorage lhs,
        UnmanagedStorage rhs,
        bool broadcast);

    // Assignment with broadcasting
    public static void Assign(NDArray lhs, NDArray rhs);
}
```

---

## Iterator System

### NDIterator<T>

The iterator system handles traversal of arrays with different memory layouts:

```csharp
public class NDIterator<T> where T : unmanaged
{
    public Func<T> MoveNext;                    // Get next value
    public MoveNextReferencedDelegate<T> MoveNextReference;  // Get reference
    public Func<bool> HasNext;                  // Check if more elements
    public Action Reset;                        // Reset to beginning

    public bool AutoReset;  // For broadcasting (smaller array loops)
    public IteratorType Type;  // Scalar, Vector, Matrix, Tensor
}
```

### Iterator Types

```csharp
public enum IteratorType
{
    Scalar,  // Single element
    Vector,  // 1D array
    Matrix,  // 2D array
    Tensor   // 3D+ array
}
```

### Optimization Paths

The iterator chooses different code paths based on:

1. **Contiguous arrays**: Direct pointer increment
2. **Sliced arrays**: Coordinate-to-offset calculation
3. **Auto-reset mode**: For broadcasting smaller arrays

```csharp
// Contiguous: fast path
MoveNext = () => *((T*)Address + index++);

// Sliced: uses shape.GetOffset
MoveNext = () => *((T*)Address + shape.GetOffset(index++));
```

---

## Code Generation

### Regen Templating

NumSharp uses Regen (a custom templating engine) to generate type-specific code. This results in approximately **200,000 lines of generated code**.

The pattern appears in many files:

```csharp
#if _REGEN
    #region Compute
    switch (typeCode)
    {
        %foreach supported_dtypes,supported_dtypes_lowercase%
        case NPTypeCode.#1: return DoOperation<#2>(arr);
        %
        default:
            throw new NotSupportedException();
    }
    #endregion
#else
    // Generated code follows...
    switch (typeCode)
    {
        case NPTypeCode.Boolean: return DoOperation<bool>(arr);
        case NPTypeCode.Byte: return DoOperation<byte>(arr);
        case NPTypeCode.Int16: return DoOperation<short>(arr);
        // ... all 12 types
    }
#endif
```

### Why Code Generation?

- **Performance**: Avoids boxing and virtual dispatch
- **Type safety**: Compile-time checks for each type
- **NumPy compatibility**: Exact type handling behavior

### Trade-offs

- **Heavy codebase**: 200K lines of generated code
- **Maintenance burden**: Changes require regeneration
- **Compile time**: Longer builds

> **Note**: Migration to T4 templates or C# source generators is possible but not currently prioritized.

---

## Development Workflow

### Adding a New np.* Function

1. **Research NumPy behavior**:
   - Read NumPy documentation
   - Run actual Python/NumPy code
   - Document edge cases (NaN, empty arrays, broadcasting)

2. **Choose implementation pattern**:
   - If needs low-level optimization → Add to `DefaultEngine`
   - If can compose existing functions → Implement directly in `np.*`

3. **Implement**:
   ```csharp
   // In np.newfunction.cs
   public static partial class np
   {
       public static NDArray newfunction(NDArray a, int axis = -1)
       {
           // Implementation
       }
   }
   ```

4. **Write tests**:
   - Run NumPy code, capture exact outputs
   - Replicate 1-to-1 in C# tests
   - Include edge cases

### Testing Philosophy

Tests should be based on **actual NumPy execution**:

```python
# Python
import numpy as np
a = np.array([1, 2, np.nan, 4])
result = np.nanmean(a)
print(result)  # 2.3333...
```

```csharp
// C# test
[TestMethod]
public void nanmean_WithNaN_IgnoresNaN()
{
    var a = np.array(new double[] { 1, 2, double.NaN, 4 });
    var result = np.nanmean(a);
    Assert.AreEqual(2.333333, result.GetDouble(), 0.0001);
}
```

### Test Coverage

- Tests use MSTest framework
- Many tests were adapted from NumPy's own test suite
- Coverage is decent but has gaps in edge cases

---

## Implemented Capabilities

NumSharp provides extensive NumPy-compatible functionality across multiple domains:

### Array Creation
`np.array`, `np.zeros`, `np.ones`, `np.empty`, `np.full`, `np.arange`, `np.linspace`, `np.eye`, `np.meshgrid`, `np.mgrid`, `np.copy`, `np.asarray`, `np.frombuffer`, `np.zeros_like`, `np.ones_like`, `np.empty_like`, `np.full_like`

### Stacking & Joining
`np.concatenate`, `np.stack`, `np.hstack`, `np.vstack`, `np.dstack`

### Broadcasting
`np.broadcast`, `np.broadcast_to`, `np.broadcast_arrays`, `np.are_broadcastable`

### Mathematical Functions
`np.sum`, `np.prod`, `np.cumsum`, `np.power`, `np.sqrt`, `np.abs`, `np.sign`, `np.floor`, `np.ceil`, `np.round`, `np.clip`, `np.modf`, `np.maximum`, `np.minimum`, `np.log`, `np.log2`, `np.log10`, `np.log1p`, `np.exp`, `np.exp2`, `np.expm1`, `np.sin`, `np.cos`, `np.tan`

### Statistics
`np.mean`, `np.std`, `np.var`, `np.amax`, `np.amin`, `np.argmax`, `np.argmin`

### Sorting & Searching
`np.argsort`, `np.searchsorted`

### Linear Algebra
`np.dot`, `np.matmul`, `np.outer`, `np.linalg.norm`, `nd.inv()`, `nd.qr()`, `nd.svd()`, `nd.lstsq()`, `nd.multi_dot()`, `nd.matrix_power()`

### Shape Manipulation
`np.reshape`, `np.transpose`, `np.ravel`, `np.squeeze`, `np.expand_dims`, `np.swapaxes`, `np.moveaxis`, `np.rollaxis`, `np.atleast_1d/2d/3d`, `np.unique`, `np.repeat`, `np.copyto`, `nd.flatten()`, `nd.roll()`, `nd.delete()`

### Logic Functions
`np.all`, `np.any`, `np.allclose`, `np.array_equal`, `np.isnan`, `np.isinf`, `np.isfinite`, `np.find_common_type`

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`, unary `-`
- Comparison: `==`, `!=`, `>`, `>=`, `<`, `<=`
- Logical: `&`, `|`, `!`

### Indexing & Selection
Integer indexing, string slice notation, Slice objects, boolean masking, fancy indexing (NDArray indices), `np.nonzero`

### Random Sampling
`np.random.rand`, `np.random.randn`, `np.random.randint`, `np.random.uniform`, `np.random.choice`, `np.random.shuffle`, `np.random.permutation`, `np.random.beta`, `np.random.binomial`, `np.random.gamma`, `np.random.poisson`, `np.random.exponential`, `np.random.geometric`, `np.random.lognormal`, `np.random.chisquare`, `np.random.bernoulli`

### File I/O
`np.save` (`.npy`), `np.load` (`.npy`, `.npz`), `np.fromfile`, `nd.tofile()`

---

## Future Roadmap

### Near-term Goals

1. **NumPy 2.x API Mapping**: Comprehensive audit of:
   - Existing functions
   - Missing functions
   - Behavioral discrepancies

2. **Behavioral Corrections**: Fix implementations that diverge from NumPy

3. **Documentation**: API documentation and examples

### Potential Future Directions

1. **Alternative Backends**: GPU (CUDA), SIMD intrinsics, MKL/BLAS
2. **Source Generator Migration**: Replace Regen with C# source generators
3. **Span<T>/Memory<T> Integration**: Where beneficial without breaking changes

### Breaking Changes

The library accepts breaking changes - it was deprecated for an extended period and is being revitalized. API stability is not a constraint.

---

## Appendix: Key Files Reference

| Component | Primary Files |
|-----------|--------------|
| NDArray | `Backends/NDArray.cs`, `Backends/NDArray.*.cs` |
| Storage | `Backends/Unmanaged/UnmanagedStorage.cs` |
| Shape | `View/Shape.cs` |
| Slicing | `View/Slice.cs` |
| TensorEngine | `Backends/TensorEngine.cs`, `Backends/Default/DefaultEngine.*.cs` |
| Iterators | `Backends/Iterators/NDIterator.cs`, `MultiIterator.cs` |
| np API | `APIs/np.cs`, individual `np.*.cs` files |
| Operators | `Operations/Elementwise/NDArray.Primitive.cs` |
| Type Info | `Utilities/InfoOf.cs`, `Backends/NPTypeCode.cs` |
| Random | `RandomSampling/np.random.cs`, `NumPyRandom.cs` |
| Generic | `Generics/NDArray\`1.cs` |

---

## Contributing

When contributing to NumSharp:

1. **Match NumPy exactly** - Run Python code, observe behavior, replicate
2. **Write tests first** - Based on actual NumPy output
3. **Handle all types** - Use Regen patterns or switch statements for all 12 dtypes
4. **Consider edge cases** - NaN, empty arrays, scalar vs array, broadcasting
5. **Document behavior** - Reference NumPy docs in comments

See the test suite for examples of expected behavior patterns.
