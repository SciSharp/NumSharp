# NumPy Compliance & Compatibility

NumSharp exists for one reason: to let you write NumPy-style code in C#. But "NumPy-style" isn't just about having similar function names—it's about behaving the same way. When you add a scalar to an array, when you slice with negative indices, when you broadcast two arrays together, NumSharp should do exactly what NumPy does.

This page explains where we are on that journey, what challenges we face, and how you can help.

---

## Why Compatibility Matters

If you're porting Python ML code to C#, the last thing you want is subtle behavioral differences causing bugs. Consider this Python code:

```python
import numpy as np
a = np.array([1, 2, 3], dtype=np.uint8)
b = a + 255
print(b)  # [0, 1, 2] - overflow wraps around
```

What should NumSharp do here? In NumPy 1.x, this would silently upcast to int16 to avoid overflow. In NumPy 2.x, it wraps with a warning. These differences matter when you're debugging why your neural network produces different results in C#.

Our goal is **1-to-1 behavioral compatibility with NumPy 2.x** (currently targeting 2.4.2). We also aim to comply with the **Python Array API Standard**, which defines portable array operations across NumPy, PyTorch, JAX, and other libraries.

---

## The Big Picture: Three Compliance Tracks

We're tracking compliance across three related but distinct standards:

### 1. NumPy 2.x Compatibility

NumPy 2.0 (released April 2024) was a major breaking release. It changed how types are promoted, removed deprecated functions, and added new APIs. If you learned NumPy before 2024, some of your intuitions might be wrong now.

**Tracking:** [NumPy 2.x Compliance Milestone](https://github.com/SciSharp/NumSharp/milestone/9)

### 2. Array API Standard

The Python Array API Standard is an industry consortium effort to define a common API that works across array libraries. Write code against the Array API, and it runs on NumPy, PyTorch, JAX, CuPy, or Dask without changes. NumPy adopted it in version 2.0.

**Deep Dive:** [Array API Standard](array-api-standard.md) — Our dedicated page with full specification details

**Tracking:** [Array API Standard Milestone](https://github.com/SciSharp/NumSharp/milestone/6)

### 3. NumPy Enhancement Proposals (NEPs)

NEPs are the design documents that define NumPy's behavior. When we say "NumPy does X," there's usually a NEP that specifies exactly what X means. We track the NEPs most relevant to NumSharp.

**Tracking:** [NEP Compliance Milestone](https://github.com/SciSharp/NumSharp/milestone/7)

---

## Type Promotion: The Biggest Change in NumPy 2.0

If there's one thing you need to understand about NumPy 2.x compatibility, it's **NEP 50: Promotion Rules for Python Scalars**.

### The Old Way (NumPy 1.x)

NumPy 1.x used "value-based" promotion. It would inspect the actual value of a scalar to decide the output type:

```python
# NumPy 1.x behavior
np.result_type(np.int8, 1)    # → int8 (1 fits in int8)
np.result_type(np.int8, 255)  # → int16 (255 doesn't fit, upcast!)
```

This was convenient—you rarely got overflow errors. But it was also unpredictable. The same code could produce different types depending on the runtime values, making optimization and type inference nearly impossible.

### The New Way (NumPy 2.x)

NumPy 2.x uses "weak scalar" promotion. Python scalars defer to the array's dtype:

```python
# NumPy 2.x behavior
np.uint8(1) + 2    # → uint8(3)
np.uint8(1) + 255  # → uint8(0) with overflow warning!
```

The scalar `2` is "weak"—it takes on whatever type the array has. This is more predictable and enables better optimization, but it can cause overflow where NumPy 1.x would have silently upcasted.

### Where NumSharp Stands

NumSharp currently has mixed behavior. Some operations follow the old value-based rules, others follow NEP 50. We're working on consistent NEP 50 compliance.

**Key Issue:** [#529 - Type promotion diverges from NumPy 2.x](https://github.com/SciSharp/NumSharp/issues/529)

**What you might see:** If you're porting NumPy code and get unexpected results with mixed types (especially unsigned + signed), this is likely why.

---

## API Changes: What Got Removed and Added

### Removed in NumPy 2.0 (NEP 52)

NumPy 2.0 cleaned house, removing ~100 deprecated functions and aliases. If you're porting old NumPy code, you might need to update these:

| Don't Use | Use Instead | Why It Changed |
|-----------|-------------|----------------|
| `np.round_` | `np.round` | Underscore was to avoid Python keyword conflict (no longer needed) |
| `np.product` | `np.prod` | Consistency with `sum` → `prod` |
| `np.sometrue` | `np.any` | Clearer naming |
| `np.alltrue` | `np.all` | Clearer naming |
| `np.rank` | `np.ndim` | `rank` was confusing (matrix rank vs array rank) |

NumSharp supports the canonical names. We never implemented most deprecated aliases, so this is actually an advantage—less legacy baggage.

### Added in NumPy 2.0 (NEP 56)

NumPy 2.0 added Array API Standard functions. These are mostly aliases for existing functions, but some are genuinely new:

**New Aliases** (for Array API compatibility):
- `np.acos`, `np.asin`, `np.atan` → aliases for `arccos`, `arcsin`, `arctan`
- `np.concat` → alias for `concatenate`
- `np.permute_dims` → alias for `transpose`
- `np.pow` → alias for `power`

**Genuinely New:**
- `np.isdtype(dtype, kind)` — Check if dtype belongs to a category
- `np.unique_values()`, `np.unique_counts()`, `np.unique_inverse()`, `np.unique_all()` — Split the overloaded `np.unique()` into focused functions
- `ndarray.mT` — Matrix transpose (transposes last two dimensions only)
- `ndarray.device` — Returns the device (CPU for NumSharp)

**NumSharp Status:** We have most aliases but are missing `isdtype()`, the `unique_*` family, `.mT`, and `.device`.

---

## Data Types: What We Support (and Don't)

NumSharp supports 12 numeric types—more than most users need, but not everything NumPy offers.

### Fully Supported

| NumSharp Type | C# Type | NumPy Type | Notes |
|---------------|---------|------------|-------|
| Boolean | `bool` | `bool_` | |
| Byte | `byte` | `uint8` | |
| Int16 | `short` | `int16` | |
| UInt16 | `ushort` | `uint16` | |
| Int32 | `int` | `int32` | Default integer type |
| UInt32 | `uint` | `uint32` | |
| Int64 | `long` | `int64` | |
| UInt64 | `ulong` | `uint64` | |
| Single | `float` | `float32` | |
| Double | `double` | `float64` | Default float type |
| Char | `char` | — | C#-specific, no NumPy equivalent |
| Decimal | `decimal` | — | C#-specific, 128-bit decimal |

### Not Yet Supported

**Complex Numbers** (`complex64`, `complex128`)

This is our biggest gap. Complex numbers are required by the Array API Standard and essential for signal processing, FFT, and many scientific applications. They're also tricky to implement efficiently in C#.

**Why it's hard:** C# has `System.Numerics.Complex`, but it's always 128-bit (complex128). There's no native complex64. We'd need to implement our own struct for float-based complex numbers.

**DateTime Types** (`datetime64`, `timedelta64`)

NumPy's datetime types (NEP 7) are powerful for time series analysis. We haven't implemented them.

**Why it's hard:** NumPy datetime64 has multiple resolutions (nanoseconds to years) stored in the dtype. C# has `DateTime` and `TimeSpan`, but they don't map cleanly to NumPy's model.

**Variable-Width Strings** (`StringDType`)

NumPy 2.0 added a new UTF-8 variable-width string type (NEP 55). The old fixed-width strings (`S10`, `U10`) wasted memory. We don't support either.

---

## Memory Layout: C-Order Only

Here's a limitation that might surprise NumPy users: **NumSharp only supports C-order (row-major) memory layout.**

### What This Means

NumPy arrays can be stored in two layouts:
- **C-order (row-major):** Last index varies fastest. Default in NumPy.
- **F-order (column-major):** First index varies fastest. Default in Fortran, MATLAB.

```python
# NumPy can do both
c_array = np.zeros((3, 4), order='C')  # Row-major
f_array = np.zeros((3, 4), order='F')  # Column-major
```

NumSharp always uses C-order. The `order` parameter exists on functions like `reshape`, `ravel`, and `flatten`, but it's ignored—we always use C-order.

### When This Matters

Most of the time, you won't notice. But if you're:
- Interfacing with Fortran libraries (LAPACK, BLAS)
- Reading data written by MATLAB
- Optimizing cache access patterns for column-wise operations

...you might hit issues. See [#546](https://github.com/SciSharp/NumSharp/issues/546) for F-order support tracking.

---

## Array API Standard

The Array API Standard specifies 133 core functions, 14 data types, and strict type promotion rules. NumSharp currently implements about **74%** of the core specification.

| Category | Required | NumSharp | Coverage |
|----------|----------|----------|----------|
| Creation | 16 | 13 | 81% |
| Element-wise | 67 | ~50 | 75% |
| Statistical | 9 | 8 | 89% |
| Manipulation | 14 | 11 | 79% |
| Set | 4 | 1 | 25% |
| Other | 23 | ~15 | ~65% |

**Biggest Gaps:**
- Complex number types (`complex64`, `complex128`) — blocks FFT and many math functions
- Set functions (`unique_all`, `unique_counts`, `unique_inverse`)
- Array properties (`.device`, `.mT`)

For the complete specification details, function lists, type promotion rules, and extension coverage, see our dedicated **[Array API Standard](array-api-standard.md)** page.

---

## Random Number Generation

Good news: NumSharp's `np.random` module provides **1-to-1 seed matching** with NumPy.

```csharp
// NumSharp
np.random.seed(42);
var a = np.random.rand(5);
// Produces: [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]

// Equivalent Python
np.random.seed(42)
a = np.random.rand(5)
# Produces: [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]
```

This is critical for reproducibility. If you're porting ML code that depends on specific random sequences (for testing, debugging, or reproducible experiments), you'll get identical results.

### Supported Distributions

- **Uniform:** `rand`, `uniform`, `randint`
- **Normal:** `randn`, `normal`
- **Other:** `beta`, `binomial`, `gamma`, `poisson`, `exponential`, `geometric`, `lognormal`, `chisquare`, `bernoulli`
- **Utilities:** `seed`, `shuffle`, `permutation`, `choice`

---

## File Format Interoperability

NumSharp can read and write NumPy's `.npy` file format. This means you can:

1. Create arrays in Python, save with `np.save()`, load in NumSharp
2. Create arrays in NumSharp, save with `np.save()`, load in Python
3. Share data files between Python and C# applications

```csharp
// Save
var arr = np.arange(100).reshape(10, 10);
np.save("mydata.npy", arr);

// Load
var loaded = np.load("mydata.npy");
```

### .npz Archives

NumPy's `.npz` format stores multiple arrays in a ZIP archive. NumSharp can **read** `.npz` files but not write them yet.

```csharp
// Load multiple arrays from .npz
var archive = np.load("data.npz") as NpzDictionary;
var weights = archive["weights"];
var biases = archive["biases"];
```

---

## Linear Algebra: Partial Support

NumSharp has basic linear algebra operations, but advanced decompositions are incomplete.

### Working

| Function | Notes |
|----------|-------|
| `np.dot` | Matrix multiplication |
| `np.matmul` | Matrix multiplication (equivalent to `@` in Python) |
| `np.outer` | Outer product |
| `ndarray.T` | Transpose |

### Stubs (Return null/default)

These functions exist but don't work:
- `np.linalg.inv` — Matrix inverse
- `np.linalg.qr` — QR decomposition
- `np.linalg.svd` — Singular value decomposition
- `np.linalg.lstsq` — Least squares

**Why?** These originally used native LAPACK bindings that have been removed. Implementing them in pure C# is possible but significant work.

---

## What's Next: Implementation Roadmap

### Phase 1: Core Compatibility (Current Focus)

- Fix type promotion to match NEP 50
- Add Array API function aliases
- Implement `isdtype()`, `unique_*` family
- Add `.mT` and `.device` properties

### Phase 2: Feature Completeness

- Complex number support (`complex64`, `complex128`)
- `datetime64` / `timedelta64` types
- Complete missing Array API functions

### Phase 3: Linear Algebra

- Implement matrix decompositions (QR, SVD, etc.)
- Either pure C# or via Math.NET Numerics integration

### Phase 4: Performance

- SIMD optimization for element-wise operations
- Iterator optimization for non-contiguous arrays

---

## How You Can Help

NumSharp is open source. Here's how to contribute:

1. **Report incompatibilities.** If NumSharp behaves differently from NumPy, file an issue with both code snippets.

2. **Add tests.** Write tests that verify NumPy behavior, then make them pass in NumSharp.

3. **Implement missing functions.** Check the milestones for prioritized work.

### GitHub Milestones

- [NumPy 2.x Compliance](https://github.com/SciSharp/NumSharp/milestone/9) — 7 open issues
- [Array API Standard](https://github.com/SciSharp/NumSharp/milestone/6) — 1 open issue
- [NEP Compliance](https://github.com/SciSharp/NumSharp/milestone/7) — 9 open issues

---

## References

- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html) — What changed in NumPy 2.0
- [Python Array API Standard](https://data-apis.org/array-api/latest/) — The specification we're implementing
- [NumPy Enhancement Proposals](https://numpy.org/neps/) — Design documents for NumPy behavior
- [NumPy Source (v2.4.2)](https://github.com/numpy/numpy/tree/v2.4.2) — Reference implementation (also at `src/numpy/` in our repo)
