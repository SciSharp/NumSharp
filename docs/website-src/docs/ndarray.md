# NumSharp's ndarray is NDArray!

NumPy's central type is `numpy.ndarray`. NumSharp's is `NDArray`. If you know one, you know the other — same concept, same memory model, same semantics, same operator behavior, ported to .NET idioms. This page is the quick tour: what `NDArray` is, how to make one, how to read and modify it, how it compares to `numpy.ndarray`, and where the two diverge because C# is not Python.

---

## Anatomy

An `NDArray` is three things glued together:

```
NDArray              ← user-facing handle (the type you work with)
├── Storage          ← UnmanagedStorage: raw pointer to native memory
├── Shape            ← dimensions, strides, offset, flags
└── TensorEngine     ← dispatches operations (DefaultEngine by default)
```

- **Storage** holds the actual bytes in unmanaged memory (not GC-allocated). Benchmarked fastest; optimized for SIMD and interop.
- **Shape** is a `readonly struct` describing how the 1-D byte block is viewed as N-D. It knows dimensions, strides, offset, and precomputed `ArrayFlags` (contiguous, broadcasted, writeable, owns-data).
- **TensorEngine** is where `+`, `-`, `sum`, `matmul`, etc. actually run. Different engines can plug in (GPU/SIMD/BLAS); the default is pure C# with IL-generated kernels.

You rarely touch Storage or TensorEngine directly — `NDArray` exposes everything.

---

## Creating an NDArray

The usual ways, with their `numpy` counterparts:

```csharp
np.array(new[] {1, 2, 3});                 // np.array([1, 2, 3])
np.array(new int[,] {{1, 2}, {3, 4}});     // np.array([[1, 2], [3, 4]])

np.zeros((3, 4));                          // np.zeros((3, 4))
np.ones(5);                                // np.ones(5)
np.full(new Shape(2, 2), 7);               // np.full((2, 2), 7)
np.full((2, 2), 7);                        // np.full((2, 2), 7)
np.empty(new Shape(3, 3));                 // np.empty((3, 3))
np.eye(4);                                 // np.eye(4)
np.identity(4);                            // np.identity(4)

np.arange(10);                             // np.arange(10)
np.arange(0, 1, 0.1);                      // np.arange(0, 1, 0.1)
np.linspace(0, 1, 11);                     // np.linspace(0, 1, 11)

np.random.rand(3, 4);                      // np.random.rand(3, 4)
np.random.randn(100);                      // np.random.randn(100)
```

> **Where `(3, 4)` comes from.** NumSharp's `Shape` struct defines implicit conversions from `int`, `long`, `int[]`, `long[]`, and value tuples of 2–6 dimensions: `(int, int)`, `(int, int, int)`, … So `np.zeros((3, 4))`, `np.zeros(new[] {3, 4})`, `np.zeros(new Shape(3, 4))`, and `np.zeros(new Shape(3L, 4L))` all produce the same array. A bare `np.zeros(5)` creates a 1-D length-5 array (the `int shape` overload).

Scalars (0-d arrays) flow in implicitly:

```csharp
NDArray a = 42;                          // 0-d int32
NDArray b = 3.14;                        // 0-d double
NDArray c = Half.One;                    // 0-d float16
NDArray d = NDArray.Scalar(100.123m);    // 0-d decimal
NDArray e = NDArray.Scalar<long>(1);     // 0-d with explicit dtype
```

Implicit scalar → NDArray exists for all 15 dtypes (`bool, sbyte, byte, short, ushort, int, uint, long, ulong, char, Half, float, double, decimal, Complex`). Use `NDArray.Scalar<T>(value)` when you want to force a specific dtype that the C# literal wouldn't pick (e.g. `short` vs `int`).

See also: [Dtypes](dtypes.md) for how to pick element types, [Broadcasting](broadcasting.md) for shape rules.

---

## Core Properties

| Property | Type | NumPy equivalent | Description |
|----------|------|------------------|-------------|
| `shape` | `long[]` | `ndarray.shape` | Dimensions |
| `ndim` | `int` | `ndarray.ndim` | Number of dimensions |
| `size` | `long` | `ndarray.size` | Total element count |
| `dtype` | `Type` | `ndarray.dtype` | C# element type |
| `typecode` | `NPTypeCode` | — | Compact enum form of dtype |
| `strides` | `long[]` | `ndarray.strides` | Byte stride per dimension |
| `T` | `NDArray` | `ndarray.T` | Transpose (view) |
| `flat` | `NDArray` | `ndarray.flat` | 1-D iterator view |
| `Shape` | `Shape` | — | Full shape object (dimensions + strides + flags) |
| `@base` | `NDArray?` | `ndarray.base` | Owner array if this is a view, else `null` |

```csharp
var a = np.arange(12).reshape(3, 4);
a.shape;       // [3, 4]
a.ndim;        // 2
a.size;        // 12
a.dtype;       // typeof(int)
a.typecode;    // NPTypeCode.Int32
a.T.shape;     // [4, 3]
a.@base;       // null means arange owns its data
var b = a["1:, :2"];
b.@base;       // wraps a's Storage (b is a view)
```

---

## Indexing & Slicing

Python's slice notation is accepted as a string:

```csharp
var a = np.arange(20).reshape(4, 5);

a[0];              // first row — reduces dim, returns (5,)
a[-1];             // last row
a[1, 2];           // single element at row 1, col 2
a["1:3"];          // rows 1-2 — keeps dim, returns (2, 5)
a["1:3, :2"];      // rows 1-2, first two cols → (2, 2)
a["::2"];          // every other row
a["::-1"];         // reversed first axis
a["..., -1"];      // ellipsis + last column
```

Boolean and fancy indexing work like NumPy:

```csharp
var arr = np.array(new[] {10, 20, 30, 40, 50});

var mask = arr > 20;           // NDArray<bool>
arr[mask];                     // [30, 40, 50]

var idx = np.array(new[] {0, 2, 4});
arr[idx];                      // [10, 30, 50] — fancy indexing
```

Assignment follows the same rules:

```csharp
a[1, 2] = 99;               // scalar write
a["0"] = np.zeros(5);       // row write
a[a > 10] = -1;             // masked write
```

> **Note:** Boolean-mask results are read-only copies in NumSharp; fancy-indexed slices and plain slices are writeable views.

---

## Views vs Copies — Most Important Rule

**Slicing returns a view, not a copy.** The view shares memory with the parent. This matches NumPy and is the source of most "why did my array change?" questions.

```csharp
var a = np.arange(10);
var v = a["2:5"];            // view — shares memory with a
v[0] = 999;                  // mutates a[2] as well!
a[2];                        // 999

var c = a["2:5"].copy();     // explicit copy — independent memory
c[0] = 0;
a[2];                        // still 999
```

Detect views with `arr.@base != null` or `arr.Storage.IsView`. Force a copy with `.copy()` or `np.copy(arr)`.

Broadcasted arrays are a special case: they're views with stride=0 dimensions, and they're **read-only** (`Shape.IsWriteable == false`) to prevent cross-row corruption. See [Broadcasting](broadcasting.md#memory-behavior).

---

## Operators

Every NumPy operator that C# can express is defined on `NDArray` with matching semantics.

### Arithmetic

| NumPy | NumSharp | Broadcasts? |
|-------|----------|-------------|
| `a + b` | `a + b` | yes |
| `a - b` | `a - b` | yes |
| `a * b` | `a * b` | yes |
| `a / b` | `a / b` | yes — returns float dtype for int inputs |
| `a % b` | `a % b` | yes — result sign follows divisor (Python/NumPy convention) |
| `-a` | `-a` | — |
| `+a` | `+a` | returns a copy |

Each takes `NDArray × NDArray`, `NDArray × object`, and `object × NDArray` — so `10 - arr` works just like `arr - 10`.

### Bitwise & shift

| NumPy | NumSharp | Notes |
|-------|----------|-------|
| `a & b` | `a & b` | bool arrays: logical AND |
| `a \| b` | `a \| b` | bool arrays: logical OR |
| `a ^ b` | `a ^ b` | — |
| `~a` | `~a` | — |
| `a << b` | `a << b` | integer dtypes only |
| `a >> b` | `a >> b` | integer dtypes only |

### Comparison

| NumPy | NumSharp | Returns |
|-------|----------|---------|
| `a == b` | `a == b` | `NDArray<bool>` |
| `a != b` | `a != b` | `NDArray<bool>` |
| `a < b` | `a < b` | `NDArray<bool>` |
| `a <= b` | `a <= b` | `NDArray<bool>` |
| `a > b` | `a > b` | `NDArray<bool>` |
| `a >= b` | `a >= b` | `NDArray<bool>` |

Comparisons with `NaN` return `False` (IEEE 754), just like NumPy.

### Logical

| NumPy | NumSharp | Notes |
|-------|----------|-------|
| `np.logical_not(a)` | `!a` | `NDArray<bool>` only |

### Operators NumPy has that C# doesn't

C# has no `**`, `//`, `@` operators, and no `__abs__`/`__divmod__` protocol. Use the functions:

| NumPy | NumSharp |
|-------|----------|
| `a ** b` | `np.power(a, b)` |
| `a // b` | `np.floor_divide(a, b)` |
| `a @ b` | `np.matmul(a, b)` or `np.dot(a, b)` |
| `abs(a)` | `np.abs(a)` |
| `divmod(a, b)` | `(np.floor_divide(a, b), a % b)` |

### C# shift-operator quirk

C# requires the declaring type on the left of `<<` / `>>`, so `object << NDArray` is a compile error. Use the named form:

```csharp
arr << 2;                     // OK
arr << someObject;            // OK (object RHS supported)
2 << arr;                     // compile error
np.left_shift(2, arr);        // use the function
```

### Compound assignment

`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=` all work. **But**: C# synthesizes them as `a = a op b` — they produce a new array and reassign the variable. They are **not in-place** like NumPy's compound operators. Other references to the original array do not see the change:

```csharp
var x = np.array(new[] {1, 2, 3});
var ref_ = x;
x += 10;                 // x -> new array [11, 12, 13]
ref_;                    // still [1, 2, 3] — different from NumPy!

y = x + 10;           // this way x stays the same and so does _ref and out is y.
```

This is a C# language constraint (compound operators on reference types cannot mutate independently of `op`) — not a NumSharp choice.

---

## Dtype Conversion

Three ways to change an array's type:

```csharp
var a = np.array(new[] {1, 2, 3});

// astype — allocates a new array (default) or rewrites in place (copy: false)
var b = a.astype(np.float64);
var c = a.astype(NPTypeCode.Int64);

// explicit cast on 0-d arrays — matches NumPy's int(arr), float(arr), complex(arr)
var scalar = np.array(new[] {42}).reshape();  // 0-d
int i = (int)scalar;
double d = (double)scalar;
Half h = (Half)scalar;
Complex cx = (Complex)scalar;
```

Rules (match NumPy 2.x):

- 0-d required. Casting an N-d array to a scalar throws `ScalarConversionException`.
- Complex → non-complex throws `TypeError` (mirroring Python's `int(1+2j)` error). Use `np.real(arr)` first.
- Numeric → numeric follows NEP 50 promotion: `int32 + float64 → float64`, `int32 * 1.0 → float64`, etc.

See [Dtypes](dtypes.md) for the full type table and conversion rules.

---

## Scalars (0-d Arrays)

A 0-d array has no dimensions — `ndim == 0`, `shape == []`, `size == 1`. Create one with `NDArray.Scalar<T>(value)` or implicit scalar conversion:

```csharp
var s1 = NDArray.Scalar(42);       // explicit
NDArray s2 = 42;                   // implicit (same result)

s1.ndim;                           // 0
s1.size;                           // 1
(int)s1;                           // 42 — explicit cast out
```

Indexing a 1-d array with a single integer returns a 0-d array (NumPy 2.x behavior). Further `(int)` casts recover the scalar.

---

## Reading & Writing Elements

Five ways to touch individual elements, picked based on how many indices you have and whether you already know the dtype:

```csharp
var a = np.arange(12).reshape(3, 4);

// 1. Indexer — returns NDArray (0-d for a single element)
NDArray elem = a[1, 2];
int v = (int)elem;             // explicit cast to scalar

// 2. .item() — direct scalar extraction (NumPy parity)
int v2 = a.item<int>(6);       // flat index 6 → row 1, col 2
object box = a.item(6);        // untyped form (returns object)

// 3. GetValue<T> — N-D coordinates, typed
int v3 = a.GetValue<int>(1, 2);

// 4. GetAtIndex<T> — flat index, typed (bypasses Shape calculation — fastest)
int v4 = a.GetAtIndex<int>(6);

// Writes mirror the reads:
a[1, 2] = 99;                           // indexer assignment
a.SetValue(99, 1, 2);                   // N-D coordinates
a.SetAtIndex(99, 6);                    // flat index
```

**Rule of thumb:** use `.item<T>()` when porting NumPy code, `GetAtIndex<T>` on a hot loop, and the indexer (`a[i, j]`) when you want NumPy-like ergonomics and don't mind the 0-d NDArray detour.

> `.item()` without arguments works on any size-1 array (0-d, 1-element 1-d, 1×1 2-d) and throws `IncorrectSizeException` otherwise — the NumPy 2.x replacement for the removed `np.asscalar()`.

---

## Iterating (foreach)

`NDArray` implements `IEnumerable`, so `foreach` works — and it iterates along **axis 0**, matching NumPy:

```csharp
var m = np.arange(6).reshape(2, 3);
foreach (NDArray row in m)
{
    Console.WriteLine(row);   // each `row` is shape (3,), a view of m
}
```

For a 1-D array, `foreach` yields individual elements (boxed). For higher-D arrays, each iteration yields a view of the subarray at that axis-0 index.

To iterate all elements flat, use `.flat` or index into `.ravel()`:

```csharp
foreach (var x in m.flat) { ... }
```

---

## Common Patterns

### Flatten to 1-D (view if possible)

```csharp
a.ravel();        // view if contiguous, copy if not
a.flatten();      // always a copy
```

### Reshape

```csharp
a.reshape(3, 4);               // explicit dims
a.reshape(-1);                 // auto-size one dim (here: 1-D flatten as view)
a.reshape(-1, 4);              // infer first dim
```

### Transpose / axis shuffle

```csharp
a.T;                           // full transpose (view)
a.transpose(new[] {1, 0, 2});  // permute axes
np.swapaxes(a, 0, 1);
np.moveaxis(a, 0, -1);
```

### Copy semantics at a glance

| Operation | Result |
|-----------|--------|
| `a["1:3"]` | view |
| `a.T` | view |
| `a.reshape(...)` | view if possible, else copy |
| `a.ravel()` | view if contiguous, else copy |
| `a.flatten()` | always copy |
| `a.copy()` | always copy |
| `a + b` | always new array |
| `a[mask]` with bool mask | copy |
| `a[idx]` with int indices | copy |

---

## Generic `NDArray<T>`

For type-safe element access, use `NDArray<T>`:

```csharp
NDArray<double> a = np.zeros(10).MakeGeneric<double>();
double first = a[0];                  // T, not NDArray
a[0] = 3.14;
```

Three ways to get a typed wrapper:

| Method | Allocates? | When to use |
|--------|------------|-------------|
| `MakeGeneric<T>()` | never (same storage) | You know the dtype matches |
| `AsGeneric<T>()` | never; throws if dtype mismatch | Defensive typing |
| `AsOrMakeGeneric<T>()` | only if dtype differs (then `astype`) | Accept any dtype, convert if needed |

`NDArray<T>` wraps the same storage; use the untyped `NDArray` when dtype is dynamic.

---

## Memory Layout

NumSharp is **C-contiguous** — row-major storage, like NumPy's default. The `order` parameter on `reshape`, `ravel`, `flatten`, and `copy` is accepted for API compatibility but ignored (there is no F-order path).

This means:

- `arr.shape = [3, 4]` → element `[i, j]` is at flat offset `i * 4 + j`.
- `arr.strides` reports byte strides, not element strides.
- For higher dimensions, the last axis varies fastest (element `[i, j, k]` is at `i * stride[0] + j * stride[1] + k * stride[2]` bytes from `Storage.Address`).

Views can be non-contiguous (sliced, transposed, broadcasted). Use `arr.Shape.IsContiguous` to detect; use `arr.copy()` to materialize contiguous memory when a kernel needs it.

---

## When Two Arrays Are "The Same"

| Comparison | Returns | Meaning |
|------------|---------|---------|
| `a == b` | `NDArray<bool>` | element-wise equality (broadcasts) |
| `np.array_equal(a, b)` | `bool` | same shape AND all elements equal |
| `np.allclose(a, b)` | `bool` | same shape AND all elements within tolerance (good for floats) |
| `ReferenceEquals(a, b)` | `bool` | same C# object (rare to want this) |
| `a.Storage == b.Storage` | `bool` | share underlying memory (i.e. views of the same data) |

---

## Troubleshooting

### "My array changed when I modified a slice!"

That's views. `a["1:3"]` shares memory with `a`. Force a copy: `a["1:3"].copy()`.

### "ReadOnlyArrayException writing to my slice"

You're writing to a broadcasted view (stride=0 dimension). Copy first: `b.copy()[...] = value`.

### "ScalarConversionException on `(int)arr`"

The array isn't 0-d. `(int)` casts only work on scalars. Use `arr.GetAtIndex<int>(0)` or index first: `(int)arr[0]`.

### "10 << arr doesn't compile"

C# requires the declaring type on the left of shift operators. Use `np.left_shift(10, arr)`.

### "a += 1 didn't update another reference"

C# compound assignment reassigns the variable; it doesn't mutate. See [Compound assignment](#compound-assignment) above. For in-place modification, write directly: `a[...] = a + 1`.

---

## API Reference

### Properties

| Member | Type | Description |
|--------|------|-------------|
| `shape` | `long[]` | Dimensions |
| `ndim` | `int` | Rank |
| `size` | `long` | Total elements |
| `dtype` | `Type` | Element `Type` |
| `typecode` | `NPTypeCode` | Element type enum |
| `strides` | `long[]` | Byte strides |
| `T` | `NDArray` | Transpose (view) |
| `flat` | `NDArray` | 1-D view |
| `Shape` | `Shape` | Full shape struct |
| `@base` | `NDArray?` | Owning array if view, else `null` |
| `Storage` | `UnmanagedStorage` | Raw memory handle (internal) |
| `TensorEngine` | `TensorEngine` | Operation dispatcher |

### Instance Methods

| Method | Description |
|--------|-------------|
| `astype(type, copy)` | Cast to different dtype (copy by default) |
| `copy()` | Deep copy |
| `Clone()` | Same as `copy()` (ICloneable) |
| `reshape(...)` | Reshape (view if possible) |
| `ravel()` | Flatten to 1-D (view if contiguous) |
| `flatten()` | Flatten to 1-D (always copy) |
| `transpose(...)` | Permute axes |
| `view(dtype)` | Reinterpret bytes as a different dtype (no copy) |
| `item()` / `item<T>()` | Extract size-1 array as scalar |
| `item(index)` / `item<T>(index)` | Extract element at flat index as scalar |
| `GetAtIndex<T>(i)` | Read element at flat index (typed, fastest) |
| `SetAtIndex<T>(value, i)` | Write element at flat index |
| `GetValue<T>(indices)` | Read at N-D coordinates |
| `SetValue<T>(value, indices)` | Write at N-D coordinates |
| `MakeGeneric<T>()` | Wrap as `NDArray<T>` (same storage) |
| `AsGeneric<T>()` | Wrap as `NDArray<T>`; throws if dtype mismatch |
| `AsOrMakeGeneric<T>()` | Wrap as `NDArray<T>`; `astype` if dtype differs |
| `Data<T>()` | Get the underlying `ArraySlice<T>` handle |
| `ToMuliDimArray<T>()` | Copy to a rank-N .NET array |
| `ToJaggedArray<T>()` | Copy to a jagged .NET array |
| `tofile(path)` | Write raw bytes to file |

### Operators

| Operator | Overloads |
|----------|-----------|
| `+`, `-`, `*`, `/`, `%` | `(NDArray, NDArray)`, `(NDArray, object)`, `(object, NDArray)` |
| unary `-`, unary `+` | `(NDArray)` |
| `&`, `\|`, `^` | `(NDArray, NDArray)`, `(NDArray, object)`, `(object, NDArray)` |
| `~`, `!` | `(NDArray)`, `(NDArray<bool>)` |
| `<<`, `>>` | `(NDArray, NDArray)`, `(NDArray, object)` — RHS only |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | `(NDArray, NDArray)`, `(NDArray, object)`, `(object, NDArray)` |

### Conversions

| Direction | Kind | Notes |
|-----------|------|-------|
| scalar → `NDArray` | implicit | `bool, sbyte, byte, short, ushort, int, uint, long, ulong, char, Half, float, double, decimal, Complex` |
| `NDArray` → scalar | explicit | same 15 types + `string` — 0-d required; complex → non-complex throws `TypeError` |

### Persistence

| Call | Format | Notes |
|------|--------|-------|
| `np.save(path, arr)` | `.npy` | NumPy-compatible; writes header + data |
| `np.load(path)` | `.npy` / `.npz` | Also accepts a `Stream` |
| `arr.tofile(path)` | raw | Element bytes only, no header |
| `np.fromfile(path, dtype)` | raw | Pair with `tofile` |

---

See also: [Dtypes](dtypes.md), [Broadcasting](broadcasting.md), [Exceptions](exceptions.md), [NumPy Compliance](compliance.md).
