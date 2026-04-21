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

- **Storage** holds the actual bytes in unmanaged memory (not GC-allocated). This beat every managed alternative in benchmarking and is what makes SIMD and zero-copy interop practical.
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
np.full((2, 2), 7);                        // np.full((2, 2), 7)
np.full(new Shape(2, 2), 7);               // same thing, explicit Shape form
np.empty((3, 3));                          // np.empty((3, 3))
np.eye(4);                                 // np.eye(4)
np.identity(4);                            // np.identity(4)

np.arange(10);                             // np.arange(10)
np.arange(0, 1, 0.1);                      // np.arange(0, 1, 0.1)
np.linspace(0, 1, 11);                     // np.linspace(0, 1, 11)

np.random.rand(3, 4);                      // np.random.rand(3, 4)
np.random.randn(100);                      // np.random.randn(100)
```

> **Where `(3, 4)` comes from.** NumSharp's `Shape` struct has implicit conversions from `int`, `long`, `int[]`, `long[]`, and value tuples of 2–6 dimensions. So these four calls all produce the same (3, 4) array:
>
> ```csharp
> np.zeros((3, 4));              // tuple → Shape
> np.zeros(new[] {3, 4});        // int[] → Shape
> np.zeros(new Shape(3, 4));     // explicit Shape
> np.zeros(new Shape(new[] {3L, 4L}));
> ```
>
> A bare `np.zeros(5)` creates a 1-D length-5 array — it hits the `int shape` overload, not a tuple.

Scalars (0-d arrays) flow in implicitly:

```csharp
NDArray a = 42;                          // 0-d int32
NDArray b = 3.14;                        // 0-d double
NDArray c = Half.One;                    // 0-d float16
NDArray d = NDArray.Scalar(100.123m);    // 0-d decimal
NDArray e = NDArray.Scalar<long>(1);     // 0-d with explicit dtype
```

Implicit scalar → NDArray exists for all 15 dtypes (`bool, sbyte, byte, short, ushort, int, uint, long, ulong, char, Half, float, double, decimal, Complex`). Use `NDArray.Scalar<T>(value)` to force a specific dtype the C# literal wouldn't pick — e.g. `NDArray.Scalar<short>(1)` instead of `NDArray x = 1;` (which would be int32).

See also: [Dtypes](dtypes.md) for how to pick element types, [Broadcasting](broadcasting.md) for shape rules.

---

## Wrapping Existing Buffers — `np.frombuffer`

When you already have memory — a `byte[]` read from a file, a network packet, a pointer from a native library, or even a typed `T[]` you want to reinterpret — `np.frombuffer` wraps it as an NDArray **without copying** whenever possible. Same contract as NumPy's `numpy.frombuffer`.

```csharp
// From a byte[] — creates a view (pins the array)
byte[] buffer = File.ReadAllBytes("sensor_data.bin");
var readings = np.frombuffer(buffer, typeof(float));

// Skip a header
var data = np.frombuffer(buffer, typeof(float), offset: 16);

// Read only part of the buffer
var subset = np.frombuffer(buffer, typeof(float), count: 1000, offset: 16);

// Reinterpret a typed array as a different dtype (view)
int[] ints = { 1, 2, 3, 4 };
var bytes = np.frombuffer<int>(ints, typeof(byte));   // 16 bytes: [1,0,0,0, 2,0,0,0, ...]

// From .NET buffer types
var fromSegment = np.frombuffer(new ArraySegment<byte>(buffer, 0, 128), typeof(int));
var fromMemory  = np.frombuffer((Memory<byte>)buffer, typeof(float));
// ReadOnlySpan<byte> always copies (spans can't be pinned)
ReadOnlySpan<byte> span = stackalloc byte[16];
var fromSpan = np.frombuffer(span, typeof(int));

// From native memory — NumSharp takes ownership and frees on GC
IntPtr owned = Marshal.AllocHGlobal(1024);
var arr1 = np.frombuffer(owned, 1024, typeof(float),
    dispose: () => Marshal.FreeHGlobal(owned));

// Or just borrow — caller must keep it alive and free it later
IntPtr borrowed = NativeLib.GetData(out int size);
var arr2 = np.frombuffer(borrowed, size, typeof(float));
// ... use arr2 ...
NativeLib.FreeData(borrowed);                       // after arr2 is done

// Endianness via dtype strings (big-endian triggers a copy)
byte[] networkData = ReceivePacket();
var be = np.frombuffer(networkData, ">i4");         // big-endian int32 (copy)
var le = np.frombuffer(networkData, "<i4");         // little-endian int32 (view on x86/x64)
```

### View or copy?

| Source | Behavior |
|--------|----------|
| `byte[]`, `ArraySegment<byte>`, array-backed `Memory<byte>` | view (array is pinned) |
| `T[]` via `frombuffer<T>(T[], …)` | view (reinterpret bytes) |
| `IntPtr` | view (optionally with `dispose` callback for ownership transfer) |
| `ReadOnlySpan<byte>` | copy (spans can't be pinned) |
| `Memory<byte>` not backed by an array | copy |
| Big-endian dtype string on a little-endian CPU | copy (must swap bytes) |

### Key rules (same as NumPy)

- **`offset` is in bytes, `count` is in elements.** A `float` buffer with `offset: 4, count: 10` reads 40 bytes starting at byte 4.
- **Buffer length (minus offset) must be a multiple of the element size**, or NumSharp throws.
- **Views couple lifetimes.** If you return an NDArray wrapping a local `byte[]`, the array can be GC'd out from under the view. Either `.copy()` before returning, or allocate through NumSharp (`np.zeros`, `np.empty`).
- **Native memory without `dispose` is borrowed** — the caller must keep the memory alive and free it after all viewing NDArrays are gone.

See the [Buffering & Memory](buffering.md) page for the full story: memory architecture, ownership patterns (ArrayPool, COM, P/Invoke), endianness, and troubleshooting.

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
a.@base;       // null (arange owns its data)
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
a[0] = np.zeros(5);         // row write (assign a full row)
a[a > 10] = -1;             // masked write
```

> **View / copy summary for indexing:**
> - Plain slices (`a["1:3"]`, `a[0]`, `a[..., -1]`): **writeable view** — shares memory with the parent.
> - Fancy indexing (`a[indexArray]`): **writeable copy** — independent memory (matches NumPy).
> - Boolean masking (`a[mask]`): **read-only copy** — independent memory; mutation via `a[mask] = value` still works as an *assignment* because it goes through the setter, not by writing into the returned array.

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

Detect views with `arr.@base != null`. Force a copy with `.copy()` or `np.copy(arr)`.

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
object rhs = 2;
arr << 2;                     // OK — int RHS
arr << rhs;                   // OK — object RHS supported
2 << arr;                     // compile error
np.left_shift(2, arr);        // use the function instead
```

### Compound assignment

`+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=` all work. **But**: C# synthesizes them as `a = a op b` — they produce a new array and reassign the variable. They are **not in-place** like NumPy's compound operators. Other references to the original array do not see the change:

```csharp
var x = np.array(new[] {1, 2, 3});
var alias = x;
x += 10;                 // x  →  new array [11, 12, 13]
// alias                 // still [1, 2, 3] — different from NumPy!
```

This is a C# language constraint — compound operators on reference types cannot be defined independently of the binary operator — not a NumSharp choice.

---

## Dtype Conversion

Three ways to change an array's type:

```csharp
var a = np.array(new[] {1, 2, 3});

// astype — allocates a new array (default) or rewrites in place (copy: false)
var b = a.astype(np.float64);
var c = a.astype(NPTypeCode.Int64);

// explicit cast on 0-d arrays — matches NumPy's int(arr), float(arr), complex(arr)
NDArray scalar = NDArray.Scalar(42);        // 0-d
int i = (int)scalar;                        // 42
double d = (double)scalar;                  // 42.0
Half h = (Half)scalar;                      // (Half)42
Complex cx = (Complex)scalar;               // 42 + 0i
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

Integer indexing always reduces one dimension:

- 1-D `a[i]` → 0-d NDArray (single element, still wrapped as an array — matches NumPy 2.x)
- 2-D `a[i]` → 1-D NDArray (a row view)
- 3-D `a[i]` → 2-D NDArray (a slab view)

To unwrap a 0-d result to a raw C# scalar, cast: `(int)a[i]` or `a.item<int>(i)`.

---

## Reading & Writing Elements

Four ways to touch individual elements, picked based on how many indices you have and whether you already know the dtype:

```csharp
var a = np.arange(12).reshape(3, 4);

// 1. Indexer — returns NDArray (0-d for a single element)
NDArray elem = a[1, 2];
int v = (int)elem;                      // explicit cast to scalar

// 2. .item<T>() — direct scalar extraction (NumPy parity)
int v2 = a.item<int>(6);                // flat index 6 → row 1, col 2
object box = a.item(6);                 // untyped form returns object

// 3. GetValue<T> — N-D coordinates, typed
int v3 = a.GetValue<int>(1, 2);

// 4. GetAtIndex<T> — flat index, typed, no Shape math (fastest)
int v4 = a.GetAtIndex<int>(6);

// Writes mirror the reads:
a[1, 2] = 99;                           // indexer assignment
a.SetValue(99, 1, 2);                   // N-D coordinates
a.SetAtIndex(99, 6);                    // flat index
```

**Rule of thumb:** use `.item<T>()` when porting NumPy code, `GetAtIndex<T>` in a hot loop, and the indexer (`a[i, j]`) when you want NumPy-like ergonomics and don't mind the 0-d NDArray detour.

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
a.reshape(-1);                 // auto-size one dim → 1-D flatten
a.reshape(-1, 4);              // infer first dim, second is 4
```

All three return a view when the source is contiguous and a copy otherwise.

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

## Saving, Loading, and Interop

NumSharp reads and writes NumPy's `.npy` / `.npz` formats and raw binary — files saved in Python open in NumSharp, and vice versa. To wrap an existing in-memory byte buffer (file bytes, a network packet, a native pointer) see [`np.frombuffer`](#wrapping-existing-buffers--npfrombuffer) above.

```csharp
// .npy round-trip
np.save("arr.npy", arr);
var loaded = np.load("arr.npy");           // also handles .npz archives

// Raw binary
arr.tofile("data.bin");
var raw = np.fromfile("data.bin", np.float64);
```

Interop with standard .NET arrays:

```csharp
var arr = np.array(new[,] {{1, 2}, {3, 4}});

// To multi-dim array (preserves shape). Note the method name is "Muli", not "Multi" —
// a longstanding API typo preserved for backwards compatibility.
int[,] md = (int[,])arr.ToMuliDimArray<int>();

// To jagged array
int[][] jag = (int[][])arr.ToJaggedArray<int>();

// From .NET array back (np.array accepts any rank)
NDArray fromMd = np.array(md);
```

For unsafe interop with native code, use `arr.Data<T>()` (gets the `ArraySlice<T>` handle) or the underlying `arr.Storage.Address` pointer. Contiguous-only; check `arr.Shape.IsContiguous` first or copy with `arr.copy()`.

---

## Memory Layout

NumSharp is **C-contiguous only** — row-major storage, like NumPy's default. The `order` parameter on `reshape`, `ravel`, `flatten`, and `copy` is accepted for API compatibility but ignored (there is no F-order path).

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
| `ReferenceEquals(a, b)` | `bool` | same C# object (rarely what you want) |
| `a.@base != null` | `bool` | `a` is a view (shares memory with some owner) |

> Caveat: NumSharp does not expose a direct "do these two arrays share memory?" check from user code. `a.@base` returns a fresh wrapper on every call and the underlying `Storage` is `protected internal`, so strict memory-identity testing is only available inside the assembly.

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

### Persistence & Buffers

| Call | Format | View / copy | Notes |
|------|--------|-------------|-------|
| `np.save(path, arr)` | `.npy` | — | NumPy-compatible; writes header + data |
| `np.load(path)` | `.npy` / `.npz` | — | Also accepts a `Stream` |
| `arr.tofile(path)` | raw | — | Element bytes only, no header |
| `np.fromfile(path, dtype)` | raw | copy | Pair with `tofile` |
| `np.frombuffer(byte[], …)` | in-memory | view (pins array) | Endian-prefix dtype strings trigger a copy |
| `np.frombuffer(ArraySegment<byte>, …)` | in-memory | view | Uses segment's offset |
| `np.frombuffer(Memory<byte>, …)` | in-memory | view if array-backed, else copy | |
| `np.frombuffer(ReadOnlySpan<byte>, …)` | in-memory | copy | Spans can't be pinned |
| `np.frombuffer(IntPtr, byteLength, …, dispose)` | native | view (optional ownership) | Pass `dispose` to transfer ownership |
| `np.frombuffer<T>(T[], …)` | in-memory | view | Reinterpret typed array as different dtype |

---

See also: [Dtypes](dtypes.md), [Broadcasting](broadcasting.md), [Exceptions](exceptions.md), [NumPy Compliance](compliance.md).
