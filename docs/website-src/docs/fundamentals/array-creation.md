# Array creation

There are six general ways to create an `NDArray` in NumSharp — the same six NumPy documents, adapted to C#:

1. Conversion from .NET sequences (arrays, jagged arrays, spans)
2. Intrinsic creation functions (`arange`, `zeros`, `ones`, `eye`, …)
3. Replicating, joining, or mutating existing arrays
4. Reading arrays from disk (`.npy`/`.npz`, text, raw binary)
5. Creating arrays from raw bytes (`frombuffer`, `fromstring`, `fromfile`)
6. Special library functions (`np.random`, …)

This page covers the general mechanisms. For element types and how they are inferred, see [Data types](../dtypes.md); for the memory-sharing rules that govern step 3, see [Copies and views](copies-and-views.md).

---

## 1. Converting .NET sequences to NDArrays

NumSharp reads .NET arrays the way NumPy reads Python lists. A 1-D array becomes a vector, a rectangular 2-D array a matrix, and so on:

```csharp
var a1D = np.array(new[] { 1, 2, 3, 4 });
var a2D = np.array(new[,] { { 1, 2 }, { 3, 4 } });
var a3D = np.array(new[,,] { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } });
```

The **dtype is inferred from the .NET element type** — this is the first thing to internalize, because it differs from NumPy in one important way:

```csharp
np.array(new[] { 1, 2, 3 });                 // int32  (from int[])
np.array(new[] { 1.0, 2.0 });                // float64 (from double[])
np.array(new sbyte[] { -1, 0, 1 });          // int8
np.array(new[] { (Half)1, (Half)2 });        // float16
np.array(new[] { new Complex(1, 2) });       // complex128
```

To pin a dtype explicitly, pass it as the second argument:

```csharp
np.array(new[] { 1, 2, 3 }, np.float64);     // [1. 2. 3.] as float64
np.array(new[] { 1, 2, 3 }, np.int8);        // int8
```

### The int32-vs-int64 gotcha

`np.array(int[])` produces **int32** (it follows the .NET `int` type), but `np.arange` and most intrinsic integer functions produce **int64** (NumPy 2.x's integer default):

```csharp
np.array(new[] { 0, 1, 2 }).dtype;   // int32   ← from int[]
np.arange(3).dtype;                  // int64   ← NumPy 2.x integer default
```

If a downstream operation is dtype-sensitive, make the intent explicit rather than relying on which constructor you happened to use.

### Weak scalars vs strong arrays on downcast

NumPy raises when a Python *list* literal overflows the requested dtype (`np.array([127, 128, 129], dtype=np.int8)` → `OverflowError`). In NumSharp a C# **`int[]` is a strong array**, so a downcast **wraps** instead of raising — the weak/strong distinction that governs NEP 50:

```csharp
np.array(new[] { 127, 128, 129 }, np.int8);  // [127, -128, -127]  ← wraps (strong array)
```

A *weak* C# scalar assigned into an integer array **does** range-check and raise (`OverflowException`). The full rules are in [Getting & Setting Values → Value coercion](../getting-and-setting-values.md#value-coercion-on-assignment-nep50).

---

## 2. Intrinsic creation functions

### 1-D range and spacing

| Function | Produces |
|----------|----------|
| `np.arange(stop)` / `np.arange(start, stop, step)` | regularly incrementing values (stop **exclusive**); integer overloads are **int64** |
| `np.linspace(start, stop, num)` | `num` values, endpoints included |
| `np.logspace(start, stop, num, base)` | `base ** linspace(...)` — log-scaled |
| `np.geomspace(start, stop, num)` | geometric progression between endpoints |

```csharp
np.arange(10);                 // [0 1 2 3 4 5 6 7 8 9]  (int64)
np.arange(2, 10, dtype: np.float64);   // [2. 3. 4. 5. 6. 7. 8. 9.]
np.arange(2.0, 3.0, 0.1);      // [2. 2.1 ... 2.9]  (roundoff can include stop)
np.linspace(1.0, 4.0, 6);      // [1. 1.6 2.2 2.8 3.4 4.]
np.logspace(0, 2, 3);          // [1. 10. 100.]   (base 10)
np.geomspace(1, 1000, 4);      // [1. 10. 100. 1000.]
```

As in NumPy, prefer `linspace` when you need a guaranteed element count and both endpoints; `arange` with a float step can miss or include `stop` from roundoff.

### 2-D / matrix constructors

| Function | Produces |
|----------|----------|
| `np.eye(n[, m, k])` | identity-like matrix; 1s on the k-th diagonal |
| `np.identity(n)` | square identity |
| `np.diag(v[, k])` | 1-D `v` → matrix with `v` on the diagonal; 2-D `v` → its diagonal (a **view**) |
| `np.diagflat(v[, k])` | any-rank `v` flattened onto a diagonal |
| `np.vander(x[, N, increasing])` | Vandermonde matrix |
| `np.tri(n[, m, k])` | lower-triangular ones |

```csharp
np.eye(3);                     // 3×3 identity (float64)
np.eye(3, 5);                  // 3×5, 1s on the main diagonal
np.diag(np.array(new[] {1, 2, 3}));       // 3×3 with [1,2,3] on the diagonal
np.diag(np.array(new[,] {{1,2},{3,4}}));  // [1, 4]  (the diagonal, as a view)
np.vander(np.array(new[] {1, 2, 3, 4}), 3);
```

> `np.diag` on a 2-D input returns a **read-only view** of the diagonal (shared memory), matching NumPy — the docstring's "returns a copy" is wrong upstream too. See [Copies and views](copies-and-views.md).

### N-D constructors

`zeros`, `ones`, `empty`, and `full` take a shape and default to **float64**:

```csharp
np.zeros((2, 3));                      // 2×3 of 0.0  (float64)
np.ones((2, 3, 2));                    // 2×3×2 of 1.0
np.empty((100,), np.int32);            // uninitialized int32
np.full((2, 2), 7.5);                  // filled with 7.5
np.zeros((2, 3), np.int32);            // dtype override
```

`np.indices(shape)` returns one grid array per dimension (stacked), useful for evaluating functions on a regular grid:

```csharp
np.indices((3, 3));
// [[[0 0 0] [1 1 1] [2 2 2]],
//  [[0 1 2] [0 1 2] [0 1 2]]]
```

Coordinate grids also come from `np.meshgrid`, `np.mgrid`, and `np.ogrid` (open mesh). See the [API reference](../../api/index.md).

---

## 3. Replicating, joining, or mutating existing arrays

Assigning an array or slicing it does **not** copy — you get a view sharing memory. Copy explicitly with `.copy()` (or `np.copy`):

```csharp
var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
var b = a["0:2"];        // a VIEW of the first two elements
b[":"] = b + 1;          // writes through — a is now [2, 3, 3, 4, 5, 6]

var c = a["0:2"].copy(); // independent copy
c[":"] = 0;              // a is unaffected
```

> C# compound assignment (`b += 1`) **rebinds the variable** to a new array rather than mutating in place; write through a slice (`b[":"] = b + 1`) to modify shared memory. This is a genuine C#-vs-Python difference — see [Copies and views](copies-and-views.md#in-place-modification) and [Getting & Setting Values](../getting-and-setting-values.md#in-place-vs-reassignment).

Join existing arrays with `np.concatenate` / `np.stack` / `np.vstack` / `np.hstack` / `np.block` / `np.column_stack`, and replicate with `np.tile` / `np.repeat`:

```csharp
var A = np.ones((2, 2));
var B = np.eye(2, 2);
var C = np.zeros((2, 2));
var D = np.diag(np.array(new[] { -3, -4 }));
np.block(new object[] { new object[] { A, B }, new object[] { C, D } });  // 4×4
```

---

## 4. Reading arrays from disk

NumSharp reads and writes NumPy's own formats **byte-for-byte**:

```csharp
np.save("weights.npy", arr);                    // .npy — byte-identical to np.save
NDArray w = np.load_npy("weights.npy");         // typed load

np.savez("bundle.npz", a, b);                   // .npz archive
using var bundle = np.load_npz("bundle.npz");   // NpzFile (IDisposable)

var table = np.loadtxt("data.csv", delimiter: ",", skiprows: 1);  // text
```

`np.load` returns `object` (an `NDArray` for `.npy`, an `NpzFile` for `.npz`), matching NumPy's content-dependent return; prefer the typed `np.load_npy` / `np.load_npz` when you know the kind. The full I/O surface — versions, `fortran_order`, big-endian, `mmap_mode`, the dtype map, and what is *not* supported — is in [I/O with NumSharp](io.md).

---

## 5. Creating arrays from raw bytes

When you have a byte buffer in a known layout, reinterpret it directly:

```csharp
byte[] raw = File.ReadAllBytes("sensor.bin");
var readings = np.frombuffer(raw, np.float32);   // reinterpret bytes as float32

var parsed = np.fromstring("1 2 3 4", sep: " "); // parse numbers from text
var fromFile = np.fromfile("data.bin", np.int16);
```

`np.frombuffer` is also the universal bridge for handing any .NET or interop buffer to NumSharp — see [Any library via np.frombuffer](../interop/np-frombuffer.md).

---

## 6. Special library functions

`np.random` produces arrays from distributions, with **byte-identical seed/state parity** to NumPy 2.4.2 (MT19937 bit generator):

```csharp
np.random.seed(42);
var r = np.random.rand(2, 3);        // uniform [0,1)
var n = np.random.randn(2, 3);       // standard normal
var g = np.random.normal(0, 1, new Shape(1000));
```

A given seed reproduces the same sequence NumPy would produce. See the [random API](../../api/index.md).

---

## Common patterns

### Preallocate then fill

```csharp
var buf = np.empty((rows, cols), np.float64);   // uninitialized — fastest
buf[":"] = 0;                                    // or np.zeros if you need zeros
```

### Match an existing array's dtype and shape

```csharp
var like = np.zeros_like(template);   // same shape + dtype, zero-filled
var one  = np.ones_like(template);
```

### Build a coordinate grid

```csharp
var (xx, yy) = np.meshgrid(np.arange(3), np.arange(4));
```

---

## Troubleshooting

### "My integers came out int64 (or int32) unexpectedly"
`np.arange`/`np.zeros(..., no dtype)` integer results are **int64**; `np.array(int[])` is **int32**. Pass an explicit dtype when it matters.

### "Downcasting an array didn't raise like NumPy"
A C# `int[]` is a *strong* array and **wraps** on downcast (`np.array(new[]{300}, np.int8)` → `44`). Only *weak* scalar assignments range-check. See [Data types → Type promotion](../dtypes.md#type-promotion).

### "I changed a slice and the original changed too"
Slices are views. Use `.copy()` when you need independence — see [Copies and views](copies-and-views.md).

---

## API reference

| Category | Functions |
|----------|-----------|
| From sequences | `np.array`, `np.asarray`, `np.asanyarray`, `np.ascontiguousarray`, `np.asfortranarray`, `np.copy` |
| 1-D ranges | `np.arange`, `np.linspace`, `np.logspace`, `np.geomspace` |
| 2-D matrices | `np.eye`, `np.identity`, `np.diag`, `np.diagflat`, `np.vander`, `np.tri` |
| N-D fills | `np.zeros`, `np.ones`, `np.empty`, `np.full`, `np.zeros_like`, `np.ones_like`, `np.empty_like`, `np.full_like`, `np.indices` |
| Grids | `np.meshgrid`, `np.mgrid`, `np.ogrid` |
| Join / replicate | `np.concatenate`, `np.stack`, `np.vstack`, `np.hstack`, `np.dstack`, `np.column_stack`, `np.block`, `np.tile`, `np.repeat` |
| From disk / bytes | `np.load`, `np.load_npy`, `np.load_npz`, `np.loadtxt`, `np.frombuffer`, `np.fromstring`, `np.fromfile` |
| Random | `np.random.rand`, `np.random.randn`, `np.random.random`, `np.random.normal`, … |

---

## Related reading

- [Data types](../dtypes.md) — the 15 dtypes and how inference and promotion work.
- [Copies and views](copies-and-views.md) — the memory-sharing rules behind step 3.
- [I/O with NumSharp](io.md) — reading and writing arrays (step 4).
- [NumPy array creation guide](https://numpy.org/doc/stable/user/basics.creation.html) — the upstream article.
