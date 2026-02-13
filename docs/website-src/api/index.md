---
uid: api-index
---

# NumSharp API Reference

NumSharp is a .NET port of Python's NumPy library. This API reference is organized by functionality, matching NumPy's documentation structure.

---

## Core Types

The essential types for working with NumSharp.

| Type | Description |
|------|-------------|
| @NumSharp.NDArray | The main n-dimensional array type |
| @NumSharp.np | Static API class (like `import numpy as np` in Python) |
| @NumSharp.Shape | Array dimensions and strides |
| @NumSharp.Slice | Slice specification for array indexing |
| @NumSharp.Generic.NDArray`1 | Generic typed wrapper for type-safe access |

### Quick Example

```csharp
using NumSharp;

// Create arrays
var a = np.array(new[] { 1, 2, 3, 4, 5 });
var b = np.zeros((3, 4));
var c = np.arange(10);

// Operations
var sum = np.sum(a);
var reshaped = a.reshape(5, 1);
var sliced = a["1:4"];  // Elements 1, 2, 3
```

---

## Array Creation

Functions for creating new arrays.

| Function | Description |
|----------|-------------|
| `np.array(data)` | Create array from existing data |
| `np.zeros(shape)` | Array filled with zeros |
| `np.zeros_like(a)` | Array of zeros with same shape as `a` |
| `np.ones(shape)` | Array filled with ones |
| `np.ones_like(a)` | Array of ones with same shape as `a` |
| `np.empty(shape)` | Uninitialized array |
| `np.full(shape, value)` | Array filled with a constant value |
| `np.eye(N)` | Identity matrix |
| `np.arange(start, stop, step)` | Evenly spaced values within interval |
| `np.linspace(start, stop, num)` | Evenly spaced values (specify count) |
| `np.meshgrid(x, y)` | Coordinate matrices from vectors |
| `np.copy(a)` | Return a copy of the array |
| `np.asarray(a)` | Convert input to array |
| `np.frombuffer(buffer)` | Create array from buffer |

---

## Stacking & Joining

Functions for combining multiple arrays.

| Function | Description |
|----------|-------------|
| `np.concatenate(arrays, axis)` | Join arrays along an existing axis |
| `np.stack(arrays, axis)` | Join arrays along a new axis |
| `np.vstack(arrays)` | Stack arrays vertically (row-wise) |
| `np.hstack(arrays)` | Stack arrays horizontally (column-wise) |
| `np.dstack(arrays)` | Stack arrays depth-wise (along 3rd axis) |

---

## Math Operations

Arithmetic and mathematical functions.

### Arithmetic Operators

| Operator | Description |
|----------|-------------|
| `a + b` | Element-wise addition |
| `a - b` | Element-wise subtraction |
| `a * b` | Element-wise multiplication |
| `a / b` | Element-wise division |
| `a % b` | Element-wise modulo |
| `-a` | Element-wise negation |

### Math Functions

| Function | Description |
|----------|-------------|
| `np.sum(a, axis)` | Sum of array elements |
| `np.prod(a)` | Product of array elements |
| `np.cumsum(a)` | Cumulative sum |
| `np.sqrt(a)` | Element-wise square root |
| `np.power(a, n)` | Element-wise power |
| `np.abs(a)` | Element-wise absolute value |
| `np.sign(a)` | Element-wise sign |
| `np.floor(a)` | Element-wise floor |
| `np.ceil(a)` | Element-wise ceiling |
| `np.round(a)` | Element-wise rounding |
| `np.clip(a, min, max)` | Clip values to range |
| `np.maximum(a, b)` | Element-wise maximum |
| `np.minimum(a, b)` | Element-wise minimum |

### Exponentials & Logarithms

| Function | Description |
|----------|-------------|
| `np.exp(a)` | Element-wise exponential |
| `np.exp2(a)` | Element-wise 2^x |
| `np.expm1(a)` | exp(x) - 1 |
| `np.log(a)` | Natural logarithm |
| `np.log2(a)` | Base-2 logarithm |
| `np.log10(a)` | Base-10 logarithm |
| `np.log1p(a)` | log(1 + x) |

### Trigonometric Functions

| Function | Description |
|----------|-------------|
| `np.sin(a)` | Element-wise sine |
| `np.cos(a)` | Element-wise cosine |
| `np.tan(a)` | Element-wise tangent |

---

## Statistics

Statistical functions.

| Function | Description |
|----------|-------------|
| `np.mean(a, axis)` | Arithmetic mean |
| `np.std(a, axis)` | Standard deviation |
| `np.var(a, axis)` | Variance |
| `np.amax(a, axis)` | Maximum value |
| `np.amin(a, axis)` | Minimum value |

---

## Sorting & Searching

Functions for sorting arrays and finding elements.

| Function | Description |
|----------|-------------|
| `np.argsort(a)` | Indices that would sort an array |
| `np.argmax(a, axis)` | Index of maximum value |
| `np.argmin(a, axis)` | Index of minimum value |
| `np.searchsorted(a, v)` | Find indices for inserting values |
| `np.nonzero(a)` | Indices of non-zero elements |

---

## Linear Algebra

Matrix and vector operations.

| Function | Description |
|----------|-------------|
| `np.dot(a, b)` | Dot product / matrix multiplication |
| `np.matmul(a, b)` | Matrix product (@ operator) |
| `np.outer(a, b)` | Outer product of two vectors |

---

## Shape Manipulation

Functions for changing array shape and dimensions.

| Function | Description |
|----------|-------------|
| `np.reshape(a, shape)` | Reshape without changing data |
| `a.reshape(shape)` | Instance method for reshape |
| `np.transpose(a)` | Permute array dimensions |
| `a.T` | Transpose property |
| `np.ravel(a)` | Flatten to 1-D array (returns view) |
| `a.flatten()` | Flatten to 1-D array (returns copy) |
| `np.squeeze(a)` | Remove axes of length 1 |
| `np.expand_dims(a, axis)` | Insert a new axis |
| `np.swapaxes(a, ax1, ax2)` | Swap two axes |
| `np.moveaxis(a, src, dst)` | Move axes to new positions |
| `np.rollaxis(a, axis)` | Roll axis backwards |
| `np.atleast_1d(a)` | Convert to at least 1-D |
| `np.atleast_2d(a)` | Convert to at least 2-D |
| `np.atleast_3d(a)` | Convert to at least 3-D |

---

## Indexing & Slicing

NumSharp supports Python-style array indexing and slicing.

### Slice Syntax

```csharp
a[":"]           // All elements
a["1:5"]         // Elements 1-4 (stop exclusive)
a["::2"]         // Every 2nd element
a["-1"]          // Last element (reduces dimension)
a["::-1"]        // Reversed
a[":, 0"]        // All rows, first column
a["..., -1"]     // Ellipsis fills dimensions
```

### Special Slice Constants

| Constant | Description |
|----------|-------------|
| `Slice.All` | All elements (`:`) |
| `Slice.Ellipsis` | Fill remaining dimensions (`...`) |
| `Slice.NewAxis` | Insert new dimension |
| `Slice.Index(n)` | Single element selection |

### Boolean Masking

```csharp
var a = np.array(new[] { 1, 2, 3, 4, 5 });
var mask = a > 2;        // [false, false, true, true, true]
var filtered = a[mask];  // [3, 4, 5]
```

---

## Logic Functions

Boolean operations and comparisons.

### Comparison Operators

| Operator | Description |
|----------|-------------|
| `a == b` | Element-wise equality |
| `a != b` | Element-wise inequality |
| `a > b` | Element-wise greater than |
| `a >= b` | Element-wise greater or equal |
| `a < b` | Element-wise less than |
| `a <= b` | Element-wise less or equal |
| `!a` | Element-wise NOT (boolean arrays) |

### Logic Functions

| Function | Description |
|----------|-------------|
| `np.all(a, axis)` | Test if all elements are true |
| `np.any(a, axis)` | Test if any element is true |
| `np.array_equal(a, b)` | Test if arrays are equal |
| `np.isscalar(a)` | Test if input is scalar |

---

## Random Sampling

Random number generation. Access via @NumSharp.NumPyRandom.

| Function | Description |
|----------|-------------|
| `np.random.rand(d0, d1, ...)` | Random values in [0, 1) |
| `np.random.randn(d0, d1, ...)` | Standard normal distribution |
| `np.random.randint(low, high, size)` | Random integers |
| `np.random.uniform(low, high, size)` | Uniform distribution |
| `np.random.choice(a, size, replace)` | Random sample from array |
| `np.random.shuffle(a)` | Shuffle array in-place |
| `np.random.permutation(a)` | Random permutation |

### Distributions

| Function | Description |
|----------|-------------|
| `np.random.beta(a, b, size)` | Beta distribution |
| `np.random.binomial(n, p, size)` | Binomial distribution |
| `np.random.gamma(shape, scale, size)` | Gamma distribution |
| `np.random.poisson(lam, size)` | Poisson distribution |
| `np.random.exponential(scale, size)` | Exponential distribution |
| `np.random.geometric(p, size)` | Geometric distribution |
| `np.random.lognormal(mean, sigma, size)` | Log-normal distribution |
| `np.random.chisquare(df, size)` | Chi-square distribution |
| `np.random.bernoulli(p, size)` | Bernoulli distribution |

---

## Broadcasting

Functions for array broadcasting.

| Function | Description |
|----------|-------------|
| `np.broadcast_to(a, shape)` | Broadcast array to new shape |
| `np.broadcast_arrays(a, b, ...)` | Broadcast arrays against each other |

Broadcasting automatically aligns array shapes for operations:
- Shapes align from the right
- Dimensions must be equal OR one must be 1
- Dimension of 1 "stretches" to match

---

## File I/O

Functions for saving and loading arrays.

| Function | Description |
|----------|-------------|
| `np.save(file, arr)` | Save array to `.npy` file |
| `np.load(file)` | Load `.npy` or `.npz` file |
| `np.fromfile(file, dtype)` | Load array from binary file |
| `arr.tofile(file)` | Write array to binary file |

---

## Unique & Set Operations

| Function | Description |
|----------|-------------|
| `np.unique(a)` | Find unique elements |
| `np.repeat(a, repeats)` | Repeat elements of array |

---

## Internals & Advanced

These types are for advanced users extending NumSharp or understanding its internals.

### Storage & Backends

| Type | Description |
|------|-------------|
| @NumSharp.Backends.UnmanagedStorage | Raw unmanaged memory management |
| @NumSharp.TensorEngine | Abstract computation backend interface |
| @NumSharp.Backends.DefaultEngine | Pure C# implementation of TensorEngine |

### Iteration

| Type | Description |
|------|-------------|
| @NumSharp.NDIterator | Traverses arrays with different memory layouts |
| @NumSharp.MultiIterator | Paired iteration for broadcasting |

### Memory Management

| Type | Description |
|------|-------------|
| @NumSharp.Backends.Unmanaged.ArraySlice`1 | Typed memory slice |
| @NumSharp.Backends.Unmanaged.IMemoryBlock | Memory block interface |

### Utilities

| Type | Description |
|------|-------------|
| @NumSharp.Utilities.InfoOf`1 | Static type information cache |
| @NumSharp.NPTypeCode | Enum of supported data types |

---

## Supported Data Types

NumSharp supports 12 numeric data types:

| NPTypeCode | C# Type | NumPy Equivalent |
|------------|---------|------------------|
| Boolean | `bool` | `np.bool_` |
| Byte | `byte` | `np.uint8` |
| Int16 | `short` | `np.int16` |
| UInt16 | `ushort` | `np.uint16` |
| Int32 | `int` | `np.int32` |
| UInt32 | `uint` | `np.uint32` |
| Int64 | `long` | `np.int64` |
| UInt64 | `ulong` | `np.uint64` |
| Char | `char` | (no equivalent) |
| Single | `float` | `np.float32` |
| Double | `double` | `np.float64` |
| Decimal | `decimal` | (no equivalent) |

---

## See Also

- [User Documentation](../docs/intro.md) - Tutorials and guides
- [GitHub Repository](https://github.com/SciSharp/NumSharp) - Source code and issues
