# Python Array API Standard

If you've ever tried to write code that works with NumPy, PyTorch, JAX, and CuPy, you know the pain. They all do similar things, but the APIs are just different enough that your code breaks when you switch libraries. The **Python Array API Standard** exists to fix this.

NumSharp is working toward Array API compliance because it means your code can be more portable—not just between Python libraries, but between Python and C#.

---

## What Is the Array API Standard?

The Array API Standard is a specification developed by the [Consortium for Python Data API Standards](https://data-apis.org/). It defines a common interface for array operations that any library can implement.

Think of it like USB for arrays. Before USB, every device had its own connector. Now they all use the same port. The Array API does the same thing for array libraries.

### The Problem It Solves

By 2020, Python had accumulated a zoo of array libraries:

- **NumPy** — The original, CPU-only
- **PyTorch** — Deep learning, GPU support
- **TensorFlow** — Deep learning, different API
- **JAX** — Functional, JIT compilation
- **CuPy** — NumPy clone for NVIDIA GPUs
- **Dask** — Distributed/parallel arrays
- **MXNet**, **PaddlePaddle**, and more...

Each library evolved independently. They all have `reshape()`, but the parameters are slightly different. They all have `sum()`, but the axis handling varies. Code written for NumPy rarely works on PyTorch without modification.

The Array API Standard says: "Here's exactly what `reshape()` should look like. Here's exactly how `sum()` should behave. Implement these, and code becomes portable."

### Who's Adopting It?

As of 2024, these libraries have adopted or are adopting the Array API:

| Library | Status |
|---------|--------|
| NumPy 2.0+ | Full support in main namespace |
| PyTorch 2.0+ | `torch` namespace is mostly compliant |
| JAX | Compliant (with some extras) |
| CuPy | Compliant |
| Dask | Compliant |
| ndonnx | Compliant |

NumSharp aims to join this list.

---

## Why Should You Care?

### For NumPy Users Moving to C#

If you're porting Python ML code to C#, Array API compliance means fewer surprises. When NumSharp follows the same specification as NumPy 2.x, the behavior matches.

### For Library Authors

If you're building a C# library that consumes arrays, coding against the Array API subset means your library works with any compliant array type—not just NumSharp.

### For Cross-Platform Development

Write once, run anywhere. The same algorithms can work on NumPy in Python and NumSharp in C#, producing identical results.

---

## The Specification: What's Required?

The Array API Standard (version 2024.12) specifies:

- **14 data types**
- **5 constants**
- **133 core functions**
- **7 array attributes**
- **Full set of operators**
- **2 optional extensions** (linear algebra, FFT)

Let's break these down.

---

## Data Types: 14 Required

The standard mandates support for exactly these types:

### Integer Types

| Type | Bits | Range | C# Equivalent |
|------|------|-------|---------------|
| `int8` | 8 | -128 to 127 | `sbyte` |
| `int16` | 16 | -32,768 to 32,767 | `short` |
| `int32` | 32 | -2B to 2B | `int` |
| `int64` | 64 | -9Q to 9Q | `long` |
| `uint8` | 8 | 0 to 255 | `byte` |
| `uint16` | 16 | 0 to 65,535 | `ushort` |
| `uint32` | 32 | 0 to 4B | `uint` |
| `uint64` | 64 | 0 to 18Q | `ulong` |

### Floating-Point Types

| Type | Bits | Precision | C# Equivalent |
|------|------|-----------|---------------|
| `float32` | 32 | ~7 digits | `float` |
| `float64` | 64 | ~16 digits | `double` |

### Complex Types

| Type | Bits | Components | C# Equivalent |
|------|------|------------|---------------|
| `complex64` | 64 | Two float32 | Custom struct needed |
| `complex128` | 128 | Two float64 | `System.Numerics.Complex` |

### Boolean

| Type | Bits | C# Equivalent |
|------|------|---------------|
| `bool` | 1 | `bool` |

### NumSharp Status

We support 12 of 14 types. **Missing:** `complex64` and `complex128`.

Complex numbers are our biggest gap. C# has `System.Numerics.Complex`, but it's always 128-bit. For `complex64`, we'd need to implement our own struct with two `float` components.

---

## Constants: 5 Required

| Constant | Value | NumSharp |
|----------|-------|----------|
| `e` | 2.71828... | ✅ |
| `inf` | Positive infinity | ✅ |
| `nan` | Not a Number | ✅ |
| `newaxis` | None (for dimension expansion) | ✅ |
| `pi` | 3.14159... | ✅ |

Full compliance here.

---

## Array Attributes: 7 Required

Every array object must have these properties:

| Attribute | Description | NumSharp |
|-----------|-------------|----------|
| `dtype` | Data type of elements | ✅ |
| `device` | Hardware location (CPU/GPU) | ❌ |
| `mT` | Matrix transpose (last 2 axes) | ❌ |
| `ndim` | Number of dimensions | ✅ |
| `shape` | Tuple of dimension sizes | ✅ |
| `size` | Total number of elements | ✅ |
| `T` | Full transpose | ✅ |

### What's `device`?

The `device` attribute tells you where the array lives—CPU, GPU, TPU, etc. For NumSharp (CPU-only), this would always return a CPU device object. We need to implement this for compliance, even though we only support one device.

### What's `mT`?

The `mT` property is "matrix transpose"—it only transposes the last two dimensions. This matters for batched matrix operations:

```python
# x has shape (batch, rows, cols)
x.T    # Transposes ALL dimensions → (cols, rows, batch) — usually wrong!
x.mT   # Transposes last two only → (batch, cols, rows) — what you want
```

NumPy 2.0 added `mT` for Array API compliance. NumSharp needs it too.

---

## Operators: Complete Set Required

Arrays must support these operators with proper semantics:

### Arithmetic
`+`, `-`, `*`, `/`, `//` (floor division), `%`, `**` (power), unary `-`, unary `+`

### Comparison
`<`, `<=`, `>`, `>=`, `==`, `!=`

### Bitwise
`~` (NOT), `&` (AND), `|` (OR), `^` (XOR), `<<`, `>>`

### Matrix
`@` (matrix multiplication)

### In-place
`+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `&=`, `|=`, `^=`, `<<=`, `>>=`, `@=`

NumSharp implements most of these. We're missing the bitwise operators as named functions (though the operators themselves work) and `@` (we have `np.matmul()` instead).

---

## Core Functions: 133 Required

The specification groups functions into categories. Here's where NumSharp stands:

### Creation Functions (16)

These create new arrays from scratch or from existing data.

| Function | Description | NumSharp |
|----------|-------------|----------|
| `arange` | Evenly spaced values in interval | ✅ |
| `asarray` | Convert to array | ✅ |
| `empty` | Uninitialized array | ✅ |
| `empty_like` | Same shape, uninitialized | ✅ |
| `eye` | Identity matrix | ✅ |
| `from_dlpack` | From DLPack capsule | ❌ |
| `full` | Filled with constant | ✅ |
| `full_like` | Same shape, filled | ✅ |
| `linspace` | Evenly spaced (by count) | ✅ |
| `meshgrid` | Coordinate matrices | ✅ |
| `ones` | Filled with ones | ✅ |
| `ones_like` | Same shape, ones | ✅ |
| `tril` | Lower triangle | ❌ |
| `triu` | Upper triangle | ❌ |
| `zeros` | Filled with zeros | ✅ |
| `zeros_like` | Same shape, zeros | ✅ |

**Coverage: 81%** — Missing `tril`, `triu`, `from_dlpack`

### Element-wise Functions (67)

The largest category. Mathematical operations applied to each element.

**Arithmetic:** `add`, `subtract`, `multiply`, `divide`, `floor_divide`, `remainder`, `pow`, `negative`, `positive`, `abs`, `sign`

**Rounding:** `ceil`, `floor`, `trunc`, `round`

**Exponential/Log:** `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`

**Trigonometric:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

**Comparison:** `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`, `maximum`, `minimum`

**Logical:** `logical_and`, `logical_or`, `logical_xor`, `logical_not`

**Bitwise:** `bitwise_and`, `bitwise_or`, `bitwise_xor`, `bitwise_invert`, `bitwise_left_shift`, `bitwise_right_shift`

**Type checking:** `isfinite`, `isinf`, `isnan`

**Other:** `sqrt`, `square`, `clip`, `copysign`, `hypot`, `logaddexp`, `nextafter`, `signbit`, `conj`, `imag`, `real`

**NumSharp Coverage: ~75%**

We're missing:
- `copysign`, `hypot`, `logaddexp` (math functions)
- `nextafter`, `signbit` (floating-point utilities)
- `conj`, `imag`, `real` (complex number functions—blocked on complex type support)
- Named bitwise functions (we have the operators, not the functions)

### Statistical Functions (9)

| Function | Description | NumSharp |
|----------|-------------|----------|
| `max` | Maximum value | ✅ (`amax`) |
| `mean` | Arithmetic mean | ✅ |
| `min` | Minimum value | ✅ (`amin`) |
| `prod` | Product of elements | ✅ |
| `std` | Standard deviation | ✅ |
| `sum` | Sum of elements | ✅ |
| `var` | Variance | ✅ |
| `cumulative_sum` | Cumulative sum | ✅ (`cumsum`) |
| `cumulative_prod` | Cumulative product | ❌ |

**Coverage: 89%** — Missing `cumulative_prod`

**Note:** The Array API uses a `correction` parameter for `std`/`var`:
```python
# Array API
std(x, correction=1)   # Sample standard deviation

# NumPy (and NumSharp)
np.std(x, ddof=1)      # Same thing, different name
```

### Manipulation Functions (14)

| Function | Description | NumSharp |
|----------|-------------|----------|
| `broadcast_arrays` | Broadcast shapes | ✅ |
| `broadcast_to` | Broadcast to shape | ✅ |
| `concat` | Join along axis | ✅ (`concatenate`) |
| `expand_dims` | Add dimension | ✅ |
| `flip` | Reverse along axis | ✅ |
| `moveaxis` | Move axis position | ✅ |
| `permute_dims` | Permute dimensions | ✅ (`transpose`) |
| `repeat` | Repeat elements | ✅ |
| `reshape` | Change shape | ✅ |
| `roll` | Shift elements | Partial |
| `squeeze` | Remove size-1 dimensions | ✅ |
| `stack` | Join along new axis | ✅ |
| `tile` | Repeat whole array | ❌ |
| `unstack` | Split along axis | ❌ |

**Coverage: ~79%** — Missing `tile`, `unstack`; `roll` is partial

### Set Functions (4)

This is our weakest area.

| Function | Description | NumSharp |
|----------|-------------|----------|
| `unique_all` | Values + indices + inverse + counts | ❌ |
| `unique_counts` | Values + counts | ❌ |
| `unique_inverse` | Values + inverse indices | ❌ |
| `unique_values` | Just unique values | ✅ (`np.unique`) |

**Coverage: 25%**

The Array API split NumPy's `np.unique(return_counts=True, return_inverse=True)` into four focused functions. We only have the basic version.

### Other Categories

| Category | Required | NumSharp | Coverage |
|----------|----------|----------|----------|
| Searching | 6 | ~4 | ~67% |
| Sorting | 2 | 2 | 100% |
| Linear Algebra (core) | 4 | 4 | 100% |
| Indexing | 2 | 0 | 0% |
| Data Types | 6 | ~3 | ~50% |
| Utility | 3 | 2 | ~67% |

---

## Type Promotion Rules

The Array API specifies strict rules for what happens when you combine different types.

### Same-Kind Promotion

Within a type category, smaller types promote to larger:

```
int8 + int16 → int16
int16 + int32 → int32
float32 + float64 → float64
```

### Cross-Kind: Undefined!

Here's the crucial difference from NumPy 1.x: **mixing integers and floats is undefined** in the Array API.

```python
# Array API says: DON'T DO THIS
int32_array + float32_array  # Undefined behavior!
```

NumPy 2.x still allows it (promoting to float), but the Array API deliberately leaves this unspecified so libraries can make their own choices.

### Scalar Promotion

When you mix a Python scalar with an array, the scalar is "weak"—it adopts the array's type:

```python
uint8_array + 2  → uint8_array  # Scalar becomes uint8
float32_array + 1.5 → float32_array  # Scalar becomes float32
```

This is consistent with NEP 50 in NumPy 2.x.

---

## Extensions: Optional but Defined

The Array API defines two optional extensions. If a library implements an extension, it must implement all functions in that extension.

### Linear Algebra Extension (23 functions)

Accessible via `linalg` namespace.

| Function | Description |
|----------|-------------|
| `cholesky` | Cholesky decomposition |
| `cross` | Cross product |
| `det` | Determinant |
| `diagonal` | Extract diagonal |
| `eigh` | Eigenvalues/vectors (Hermitian) |
| `eigvalsh` | Eigenvalues only (Hermitian) |
| `inv` | Matrix inverse |
| `matmul` | Matrix multiplication |
| `matrix_norm` | Matrix norm |
| `matrix_power` | Matrix to integer power |
| `matrix_rank` | Numerical rank |
| `matrix_transpose` | Transpose last 2 dims |
| `outer` | Outer product |
| `pinv` | Pseudo-inverse |
| `qr` | QR decomposition |
| `slogdet` | Sign and log-determinant |
| `solve` | Solve linear system |
| `svd` | Singular value decomposition |
| `svdvals` | Singular values only |
| `tensordot` | Tensor contraction |
| `trace` | Sum of diagonal |
| `vecdot` | Vector dot product |
| `vector_norm` | Vector norm |

**NumSharp Status:** We have `matmul`, `outer`, `trace`, and basic operations. The decompositions (`qr`, `svd`, `eigh`, `cholesky`, `inv`) are stubs that return null.

### FFT Extension (14 functions)

Accessible via `fft` namespace.

| Function | Description |
|----------|-------------|
| `fft` | 1-D discrete Fourier transform |
| `ifft` | Inverse of fft |
| `fftn` | N-D DFT |
| `ifftn` | Inverse of fftn |
| `rfft` | 1-D DFT for real input |
| `irfft` | Inverse of rfft |
| `rfftn` | N-D DFT for real input |
| `irfftn` | Inverse of rfftn |
| `hfft` | 1-D DFT for Hermitian input |
| `ihfft` | Inverse of hfft |
| `fftfreq` | DFT sample frequencies |
| `rfftfreq` | Sample frequencies for rfft |
| `fftshift` | Shift zero-frequency to center |
| `ifftshift` | Inverse of fftshift |

**NumSharp Status:** Not implemented. FFT requires complex number support.

---

## What's NOT in the Standard

The Array API deliberately excludes some things to remain implementable across diverse libraries:

### Out of Scope

- **I/O operations** — No `save`, `load`, `fromfile`
- **String dtypes** — No `StringDType` or fixed-width strings
- **Datetime dtypes** — No `datetime64`, `timedelta64`
- **Object dtype** — No arrays of arbitrary Python objects
- **Specific error types** — Error handling is implementation-defined
- **C API** — Only Python-level interface specified
- **Execution semantics** — Eager vs. lazy, parallelization, etc.

This means NumSharp can have these features (and we do—`np.save`, `np.load` work), they're just outside the Array API specification.

---

## Real-World Use Cases

The specification documents several motivating use cases:

### SciPy Without Dependencies

SciPy's signal processing functions are pure Python but tied to NumPy. With Array API compliance, `scipy.signal.welch(x)` could work on GPU arrays (CuPy), distributed arrays (Dask), or NumSharp arrays—without SciPy depending on any of them.

### einops Without Backend Code

The [einops](https://github.com/arogozhnikov/einops) library maintains ~550 lines of glue code to support multiple backends. Array API compliance would eliminate this entirely.

### JIT Compilation

Numba and other JIT compilers struggle with NumPy's value-dependent type rules. The Array API's strict type-based promotion makes JIT compilation predictable.

---

## NumSharp's Path to Compliance

### Current Coverage

| Category | Functions | NumSharp | % |
|----------|-----------|----------|---|
| Creation | 16 | 13 | 81% |
| Element-wise | 67 | ~50 | 75% |
| Statistical | 9 | 8 | 89% |
| Manipulation | 14 | 11 | 79% |
| Set | 4 | 1 | 25% |
| Searching | 6 | 4 | 67% |
| Sorting | 2 | 2 | 100% |
| Linear Algebra | 4 | 4 | 100% |
| Indexing | 2 | 0 | 0% |
| Data Types | 6 | 3 | 50% |
| Utility | 3 | 2 | 67% |
| **Total Core** | **133** | **~98** | **~74%** |

### Priority Items

1. **Complex number types** — Blocks FFT extension and many math functions
2. **`device` and `mT` properties** — Simple to add
3. **Set functions** (`unique_*` family) — Moderate effort
4. **Missing element-wise functions** — Incremental work
5. **Indexing functions** (`take`, `take_along_axis`) — Moderate effort

### Tracking

See [Array API Standard Milestone](https://github.com/SciSharp/NumSharp/milestone/6) for detailed issue tracking.

---

## References

- [Array API Standard Specification](https://data-apis.org/array-api/latest/) — The full specification
- [Type Promotion Rules](https://data-apis.org/array-api/latest/API_specification/type_promotion.html) — How types combine
- [Linear Algebra Extension](https://data-apis.org/array-api/latest/extensions/linear_algebra_functions.html) — All linalg functions
- [FFT Extension](https://data-apis.org/array-api/latest/extensions/fourier_transform_functions.html) — All FFT functions
- [Consortium for Python Data API Standards](https://data-apis.org/) — The organization behind the standard
- [NumPy Array API Support](https://numpy.org/doc/stable/reference/array_api.html) — NumPy's implementation notes
