# NumSharp quickstart

A fast overview of `NDArray` for people who already program in C#. It shows how 1-, 2-, and n-dimensional arrays are represented and manipulated — how to apply whole-array operations without `for` loops, and how `axis`/`shape` behave. It is the NumSharp counterpart of NumPy's [quickstart](https://numpy.org/doc/stable/user/quickstart.html); if you want the gentler on-ramp first, read [The absolute basics for beginners](absolute-basics.md).

> **Two syntax facts up front, because C# lacks two Python operators:**
> - Python slices `a[1:5:2]` are written as **strings** — `a["1:5:2"]`. Comma coordinates stay normal: `a[1, 3]`.
> - Python has `**` (power) and `@` (matmul); C# has neither on arrays. Use **`np.power(a, 2)`** and **`np.matmul(a, b)`** / **`a.dot(b)`**.

---

## The basics

NumSharp's main object is the homogeneous multidimensional array `NDArray` — a table of same-type elements indexed by non-negative integers. Dimensions are called **axes**. `[[1., 0., 0.], [0., 1., 2.]]` has 2 axes: the first of length 2, the second of length 3.

The important attributes of an `NDArray`:

| Attribute | Meaning |
|-----------|---------|
| `ndim` | the number of axes |
| `shape` | an `int[]` of the size along each axis (length == `ndim`) |
| `size` | total number of elements (product of `shape`) |
| `dtype` | a `DType` describing the element type (`np.int32`, `np.float64`, …); `dtype.name` is the NumPy name, `dtype.type` the CLR `Type` |
| `itemsize` | bytes per element (e.g. 8 for float64); == `dtype.itemsize` |
| `Storage` | the underlying data buffer — you rarely touch it; use indexing |

### An example

```csharp
var a = np.arange(15).reshape(3, 5);
a;
// [[ 0  1  2  3  4]
//  [ 5  6  7  8  9]
//  [10 11 12 13 14]]

a.shape;        // (3, 5)
a.ndim;         // 2
a.dtype.name;   // "int64"   (np.arange gives int64)
a.itemsize;     // 8
a.size;         // 15
a.GetType();    // NumSharp.NDArray

var b = np.array(new[] { 6, 7, 8 });
b;              // [6 7 8]
b.GetType();    // NumSharp.NDArray
```

---

## Array creation

Create an array from a .NET sequence; the dtype is deduced from the element type:

```csharp
var a = np.array(new[] { 2, 3, 4 });
a.dtype;                              // int32   (see note below)
var b = np.array(new[] { 1.2, 3.5, 5.1 });
b.dtype;                              // float64
```

> **⚠️ Divergence — integer default.** `np.array(new[]{2,3,4})` follows the .NET `int` type, so it's **int32**, where NumPy's `np.array([2,3,4])` is int64. `np.arange` *does* give int64. Pass a dtype when it matters. See [Array creation → int32-vs-int64](fundamentals/array-creation.md#the-int32-vs-int64-gotcha).

Nested sequences become 2-D (and deeper), and the dtype can be set explicitly:

```csharp
var b = np.array(new[,] { { 1.5, 2, 3 }, { 4, 5, 6 } });
// [[1.5 2.  3. ]
//  [4.  5.  6. ]]

var c = np.array(new[,] { { 1, 2 }, { 3, 4 } }, np.complex128);
// [[1.+0.j 2.+0.j]
//  [3.+0.j 4.+0.j]]
```

Fill functions take a shape (default dtype **float64**):

```csharp
np.zeros((3, 4));
np.ones((2, 3, 4), np.int16);   // dtype int16
np.empty((2, 3));               // uninitialized — fill it yourself
```

Sequences of numbers come from `arange` (like a stepped range) and `linspace` (a fixed count):

```csharp
np.arange(10, 30, 5);    // [10 15 20 25]
np.arange(0, 2, 0.3);    // [0. 0.3 0.6 0.9 1.2 1.5 1.8]   (float steps)
np.linspace(0, 2, 9);    // [0. 0.25 0.5 0.75 1. 1.25 1.5 1.75 2. ]

var x = np.linspace(0, 2 * np.pi, 100);   // np.pi is the constant π
var f = np.sin(x);                         // evaluate a function at many points
```

With float steps `arange` can't predict the element count exactly (floating-point precision), so prefer `linspace`. See [Array creation](fundamentals/array-creation.md).

---

## Printing arrays

Printing an `NDArray` (via `ToString()` / `Console.WriteLine`) lays it out like nested lists: the last axis left-to-right, the second-to-last top-to-bottom, higher axes separated by a blank line.

```csharp
np.arange(6);                 // [0 1 2 3 4 5]                       (1-D → row)

np.arange(12).reshape(4, 3);  // 2-D → matrix
// [[ 0  1  2]
//  [ 3  4  5]
//  [ 6  7  8]
//  [ 9 10 11]]

np.arange(24).reshape(2, 3, 4); // 3-D → list of matrices
```

NumSharp's array printing is a byte-exact port of NumPy's, so a large array is **truncated the same way** — the middle is skipped and only the corners shown:

```csharp
np.arange(10000);   // [   0    1    2 ... 9997 9998 9999]
```

Change the behaviour (e.g. print everything) with `np.set_printoptions` / `np.get_printoptions` (or the `np.printoptions` scope). See [Array printing](compliance.md).

---

## Basic operations

Arithmetic operators apply **elementwise**, producing a new array:

```csharp
var a = np.array(new[] { 20, 30, 40, 50 });
var b = np.arange(4);         // [0 1 2 3]
a - b;                        // [20 29 38 47]
np.power(b, 2);               // [0 1 4 9]     (C# has no ** operator)
10 * np.sin(a);               // [ 9.12945251 -9.88031624  7.4511316  -2.62374854]
a < 35;                       // [True True False False]
```

The `*` operator is **elementwise**, not matrix multiplication. Use `np.matmul` or `.dot` for the matrix product (C# has no `@` operator):

```csharp
var A = np.array(new[,] { { 1, 1 }, { 0, 1 } });
var B = np.array(new[,] { { 2, 0 }, { 3, 4 } });
A * B;            // elementwise → [[2 0] [0 4]]
np.matmul(A, B);  // matrix product → [[5 4] [3 4]]
A.dot(B);         // same → [[5 4] [3 4]]
```

### In-place operations behave differently in C#

In NumPy `a *= 3` and `b += a` mutate the array **in place**. In C#, compound-assignment operators **rebind the variable** — `a *= 3` compiles to `a = a * 3`, allocating a *new* array — so other references keep seeing the old data:

```csharp
var a = np.ones((2, 3), np.int32);
var alias = a;
a *= 3;
a;        // [[3 3 3] [3 3 3]]
alias;    // [[1 1 1] [1 1 1]]   ← NOT updated (differs from NumPy!)
```

To mutate the buffer **in place** (so aliases and views observe it), assign through a slice or use the `@out:` parameter:

```csharp
a[":"] = a * 3;                 // in-place: alias sees it too
np.multiply(a, 3, @out: a);     // in-place via the out= parameter
```

The `@out:` path also reproduces NumPy's in-place **casting rule**: writing a float result back into an int array raises the same error NumPy's `a += b` raises:

```csharp
var ai = np.ones(3, np.int32);
var bf = np.linspace(0, 3, 3);          // float64
np.add(ai, bf, @out: ai);
// ArgumentException: Cannot cast ufunc 'add' output from dtype('float64')
//                    to dtype('int32') with casting rule 'same_kind'
```

See [Copies and views → in-place modification](fundamentals/copies-and-views.md#in-place-modification).

### Upcasting

Operating on mixed types promotes to the more general type (NEP 50):

```csharp
var a = np.ones(3, np.int32);
var b = np.linspace(0, np.pi, 3);       // float64
var c = a + b;
c.dtype.name;                            // "float64"
var d = np.exp(c * new Complex(0, 1));   // multiply by the imaginary unit → complex
d.dtype.name;                            // "complex128"
```

Many aggregations are `NDArray` methods:

```csharp
var a = np.array(new[] { 0.5, 0.4, 0.55, 0.03, 0.75, 0.53 });
a.sum();   // 2.76
a.min();   // 0.03
a.max();   // 0.75
```

By default these treat the array as a flat list. Pass `axis` to aggregate along one axis:

```csharp
var b = np.arange(12).reshape(3, 4);
b.sum(axis: 0);      // [12 15 18 21]   (each column)
b.min(axis: 1);      // [0 4 8]         (each row)
np.cumsum(b, axis: 1);
// [[ 0  1  3  6]
//  [ 4  9 15 22]
//  [ 8 17 27 38]]
```

More at [Universal functions](fundamentals/ufuncs.md).

---

## Universal functions

Familiar math functions (`sin`, `cos`, `exp`, `sqrt`, …) operate elementwise and return an array — these are the *universal functions* (ufuncs):

```csharp
var B = np.arange(3);
np.exp(B);         // [1. 2.71828183 7.3890561 ]
np.sqrt(B);        // [0. 1. 1.41421356]
var C = np.array(new[] { 2.0, -1.0, 4.0 });
np.add(B, C);      // [2. 0. 6.]
```

NumSharp exposes ufuncs as direct `np.*` functions (and operators), with `out=`/`where=`/`dtype=` — see [Universal functions](fundamentals/ufuncs.md). NumSharp also has `all`, `any`, `argmax`, `argsort`, `mean`, `median`, `clip`, `cross`, `dot`, `sort`, `std`, `sum`, `var`, `where`, `apply_along_axis`, `bincount`, `corrcoef`, `vectorize`, `frompyfunc`, and the rest of the aggregation family.

> **`np.vectorize` / `np.frompyfunc` wrap a C# delegate.** `var vf = np.vectorize((int a, int b) => a > b ? a - b : a + b); vf(arrA, 2)` reads like NumPy and broadcasts element-wise; a gufunc `signature:` applies the delegate per core sub-array (`np.vectorize(x => np.sum(x), "(n)->()")`). The element-wise path is a fused `np.evaluate` + `NDExpr.Call` pass, so it runs 3–30× faster than NumPy's Python-loop `vectorize`. For a hot inner loop you can still write `np.evaluate` / `np.nditer<T>` directly — see [Extending NumSharp](advanced/extending-numsharp.md).

---

## Indexing, slicing and iterating

**1-D** arrays index, slice, and iterate like C# sequences — with string slices:

```csharp
var a = np.power(np.arange(10), 3);   // [0 1 8 27 64 125 216 343 512 729]
a[2];           // 8 (0-D)
a["2:5"];       // [8 27 64]
a[":6:2"] = 1000;   // set every 2nd element up to index 6
a;              // [1000 1 1000 27 1000 125 216 343 512 729]
a["::-1"];      // reversed
```

**Multidimensional** arrays take one index per axis, comma-separated in one bracket. NumSharp has no `np.fromfunction`, so build a `10*i + j` grid by broadcasting:

```csharp
var b = np.arange(5)[Slice.All, np.newaxis] * 10 + np.arange(4);
// [[ 0  1  2  3]
//  [10 11 12 13]
//  [20 21 22 23]
//  [30 31 32 33]
//  [40 41 42 43]]

b[2, 3];        // 23
b["0:5, 1"];    // [1 11 21 31 41]   (column 1 of every row)
b[":, 1"];      // same
b["1:3, :"];    // rows 1 and 2
b[-1];          // last row — same as b[-1, :]
```

When fewer indices than axes are given, the rest are complete slices. **Dots** (`...`) fill the missing axes — write them inside the index string:

```csharp
var c = np.array(new[,,] { { { 0, 1, 2 }, { 10, 12, 13 } }, { { 100, 101, 102 }, { 110, 112, 113 } } });
c.shape;        // (2, 2, 3)
c["1, ..."];    // same as c[1, :, :] → [[100 101 102] [110 112 113]]
c["..., 2"];    // same as c[:, :, 2] → [[2 13] [102 113]]
```

**Iterating** is with respect to the first axis — `foreach` over an `NDArray` yields the sub-arrays along axis 0:

```csharp
foreach (var row in b)      // each row is an NDArray
    Console.WriteLine(row);
```

To visit every element regardless of shape, iterate `b.flat` (boxed, C-order), or — for fast, unboxed element work — use `np.nditer<T>` (see [Iterating & Enumerating](iterating-and-enumerating.md)):

```csharp
foreach (var element in b.flat)
    Console.WriteLine(element);
```

For per-element math, prefer the vectorized form (`np.power(a, 1.0 / 3)`) over a loop. More at [Indexing on NDArray](fundamentals/indexing.md).

---

## Shape manipulation

### Changing the shape of an array

```csharp
var a = np.floor(10 * np.random.rand(3, 4));
a.shape;     // (3, 4)

a.ravel();       // flattened (a view where possible), C-order
a.reshape(6, 2); // a new shape (returns a new array; a is unchanged)
a.T;             // transposed view
a.T.shape;       // (4, 3)
```

`reshape` returns a *new* array; `resize` modifies the array **in place**:

```csharp
a.resize(2, 6);   // a itself is now (2, 6)
```

A dimension given as `-1` is inferred:

```csharp
np.arange(12).reshape(3, -1).shape;   // (3, 4)
```

See [Copies and views](fundamentals/copies-and-views.md) for when `ravel`/`reshape` copy.

### Stacking together different arrays

```csharp
var a = np.floor(10 * np.random.rand(2, 2));
var b = np.floor(10 * np.random.rand(2, 2));
np.vstack(a, b);   // stack along axis 0
np.hstack(a, b);   // stack along axis 1
```

`np.column_stack` stacks 1-D arrays as columns of a 2-D array (equivalent to `hstack` only for 2-D inputs). A 1-D array becomes a column with `[:, np.newaxis]`:

```csharp
var x = np.array(new[] { 4.0, 2.0 });
var y = np.array(new[] { 3.0, 8.0 });
np.column_stack(x, y);   // [[4. 3.] [2. 8.]]
np.hstack(x, y);         // [4. 2. 3. 8.]  (different — 1-D concat)
x[Slice.All, np.newaxis]; // [[4.] [2.]]  (column view)
```

For building arrays by stacking along one axis with range literals, `np.r_` and `np.c_` are handy (slices are strings):

```csharp
np.r_["1:4", 0, 4];   // [1 2 3 0 4]
```

See [Array creation → grid/slice DSL](fundamentals/array-creation.md) and [Broadcasting](broadcasting.md).

### Splitting one array into several smaller ones

`np.hsplit` splits along the horizontal axis — by count, or after given columns. `np.vsplit` splits vertically; `np.array_split` takes the axis to split along.

```csharp
var a = np.floor(10 * np.random.rand(2, 12));
np.hsplit(a, 3);              // 3 equal parts
np.hsplit(a, new[] { 3, 4 }); // split after columns 3 and 4 → 3 parts
np.vsplit(np.arange(16).reshape(4, 4), 2);  // 2 parts along axis 0
```

---

## Copies and views

Three cases, often a source of confusion — the full treatment is in [Copies and views](fundamentals/copies-and-views.md).

**No copy at all.** A plain assignment is a second reference to the same object (a C# reference), not a new array:

```csharp
var a = np.arange(12).reshape(3, 4);
var b = a;
ReferenceEquals(a, b);   // true — two names for one NDArray
```

**View / shallow copy.** `view()` (and slicing) makes a new array object over the *same* data:

```csharp
var c = a.view();
ReferenceEquals(c, a);   // false
c.flags.owndata;         // false — c is a view
np.shares_memory(a, c);  // true

var s = a[":, 1:3"];     // slicing returns a view
s[":"] = 10;             // s[":"] = 10 writes through into a
```

**Deep copy.** `copy()` duplicates data and metadata:

```csharp
var d = a.copy();
np.shares_memory(a, d);  // false
```

Copy after slicing when the parent is a big intermediate you no longer need, so its memory can be released:

```csharp
var big = np.arange(100_000_000);
var head = big["0:100"].copy();   // detaches from big
// big can now be collected
```

### Functions and methods overview

A categorized sampler of what NumSharp provides (see the [API reference](../api/index.md) and [coverage dashboard](coverage-support-dashboard.md) for the full list):

| Category | Members |
|----------|---------|
| **Creation** | `arange`, `array`, `copy`, `empty`, `empty_like`, `eye`, `identity`, `linspace`, `logspace`, `mgrid`, `ogrid`, `ones`, `ones_like`, `r_`, `zeros`, `zeros_like` |
| **Conversions** | `ndarray.astype`, `atleast_1d/2d/3d`, `asmatrix` |
| **Manipulations** | `array_split`, `column_stack`, `concatenate`, `diagonal`, `dsplit`, `dstack`, `hsplit`, `hstack`, `ndarray.item`, `newaxis`, `ravel`, `repeat`, `reshape`, `resize`, `squeeze`, `swapaxes`, `take`, `transpose`, `vsplit`, `vstack` |
| **Questions** | `all`, `any`, `nonzero`, `where` |
| **Ordering** | `argmax`, `argmin`, `argsort`, `max`, `min`, `ptp`, `searchsorted`, `sort` |
| **Operations** | `choose`, `compress`, `cumprod`, `cumsum`, `inner`, `ndarray.fill`, `imag`, `prod`, `put`, `putmask`, `real`, `sum` |
| **Statistics** | `cov`, `mean`, `std`, `var` |
| **Linear algebra** | `cross`, `dot`, `outer`, `linalg.svd`, `vdot` |

---

## Broadcasting rules

Broadcasting lets ufuncs work on inputs of different shapes. Two rules:

1. If inputs differ in rank, `1`s are prepended to the smaller shapes until all ranks match.
2. A size-1 axis behaves as if it had the other array's size along that axis (the value is repeated).

After that, all sizes must match. Full detail at [Broadcasting](broadcasting.md).

---

## Advanced indexing and index tricks

Arrays can be indexed by **arrays of integers** and **arrays of booleans**, beyond integers and slices.

### Indexing with arrays of indices

```csharp
var a = np.power(np.arange(12), 2);          // the first 12 squares
var i = np.array(new[] { 1, 1, 3, 8, 5 });
a[i];                                         // [1 1 9 64 25]  (elements at positions i)

var j = np.array(new[,] { { 3, 4 }, { 9, 7 } });
a[j];                                         // result has j's shape → [[9 16] [81 49]]
```

For a multidimensional target, a single index array indexes the **first** axis — the palette→color-image trick:

```csharp
var palette = np.array(new[,] {
    { 0, 0, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 255, 255 } });
var image = np.array(new[,] { { 0, 1, 2, 0 }, { 0, 3, 4, 0 } });
palette[image].shape;                         // (2, 4, 3) — a color image
```

Give an index array per dimension (they must share a shape):

```csharp
var a = np.arange(12).reshape(3, 4);
var i = np.array(new[,] { { 0, 1 }, { 1, 2 } });
var j = np.array(new[,] { { 2, 1 }, { 3, 3 } });
a[i, j];        // [[2 5] [7 11]]
a[i, 2];        // [[2 6] [6 10]]  (index array + scalar)
a[Slice.All, j].shape;   // (3, 2, 2)
```

Finding the maxima of time series — `argmax(axis:)` then gather:

```csharp
var data = np.sin(np.arange(20).astype(np.float64)).reshape(5, 4);
var ind = data.argmax(axis: 0);               // [2 0 3 1]
var time = np.linspace(20, 145, 5);
var time_max = time[ind];                      // [82.5 20. 113.75 51.25]
var data_max = data[ind, np.arange(data.shape[1])];  // one max per column
```

Index arrays can be an assignment target; duplicate indices leave the **last** value:

```csharp
var a = np.arange(5);
a[new[] { 1, 3, 4 }] = 0;              // [0 0 2 0 0]

var b = np.arange(5);
b[new[] { 0, 0, 2 }] = np.array(new[] { 1, 2, 3 });   // [2 1 3 3 4]  (index 0 written twice, last wins)
```

> As with the [in-place divergence](#in-place-operations-behave-differently-in-c) above, `a[new[]{0,0,2}] += 1` does **not** increment index 0 twice — C# expands it to `a[...] = a[...] + 1`.

### Indexing with boolean arrays

A boolean array of the *same shape* picks the elements where it's true (returns a 1-D copy), and works as an assignment target:

```csharp
var a = np.arange(12).reshape(3, 4);
var b = a > 4;
a[b];        // [5 6 7 8 9 10 11]
a[b] = 0;    // every element > 4 becomes 0
```

You can also give a **1-D boolean per axis** (its length must match that axis):

```csharp
var a = np.arange(12).reshape(3, 4);
var b1 = np.array(new[] { false, true, true });          // selects rows
var b2 = np.array(new[] { true, false, true, false });   // selects columns
a[b1, Slice.All];   // rows 1,2 → [[4 5 6 7] [8 9 10 11]]
a[b1];              // same
a[Slice.All, b2];   // columns 0,2 → [[0 2] [4 6] [8 10]]
a[b1, b2];          // paired → [4 10]
```

More at [Indexing on NDArray](fundamentals/indexing.md).

### The `ix_` function

`np.ix_` combines vectors so you get a result for every n-tuple — e.g. all `a + b*c` over triplets, via broadcasting (no full-size intermediate):

```csharp
var a = np.array(new[] { 2, 3, 4, 5 });
var b = np.array(new[] { 8, 5, 4 });
var c = np.array(new[] { 5, 4, 6, 8, 3 });
var ix = np.ix_(a, b, c);              // NDArray[] { ax (4,1,1), bx (1,3,1), cx (1,1,5) }
var result = ix[0] + ix[1] * ix[2];    // shape (4, 3, 5)
result[3, 2, 4];                        // 17  == a[3] + b[2]*c[4]
```

### Indexing with strings

NumPy indexes structured arrays by field name (`x['age']`). NumSharp has no structured dtype — see [Structured arrays](fundamentals/structured-arrays.md).

---

## Tricks and tips

### "Automatic" reshaping

Omit one dimension (as `-1`) and it's inferred:

```csharp
var a = np.arange(30);
var b = a.reshape(2, -1, 3);   // -1 means "whatever is needed"
b.shape;                        // (2, 5, 3)
```

### Vector stacking

Build a 2-D array from equal-length row vectors with `vstack`/`hstack`:

```csharp
var x = np.arange(0, 10, 2);   // [0 2 4 6 8]
var y = np.arange(5);          // [0 1 2 3 4]
np.vstack(x, y);               // [[0 2 4 6 8] [0 1 2 3 4]]
np.hstack(x, y);               // [0 2 4 6 8 0 1 2 3 4]
```

### Histograms

`np.histogram` returns the histogram values **and** the bin edges (it computes, it does not plot — plotting is a separate library, see below):

```csharp
var v = np.random.normal(2, 0.5, 10000);            // 10000 normal deviates, mean 2, σ 0.5
var (counts, binEdges) = np.histogram(v, bins: 50, density: true);
counts.shape;     // (50,)
binEdges.shape;   // (51,)
```

`HistogramResult` deconstructs into `(hist, bin_edges)` and also converts to `NDArray` (the histogram) or `NDArray[]`.

### Plotting

NumSharp has no built-in plotting (neither does NumPy — Matplotlib is separate). Hand an `NDArray`'s values (`arr.ToArray<double>()`) to a .NET charting library — **ScottPlot**, **OxyPlot**, **Plotly.NET** — or drive Matplotlib over your arrays through the [pythonnet bridge](interop/pythonnet-numpy.md).

---

## Further reading

- [The absolute basics for beginners](absolute-basics.md) — the gentler introduction.
- [Fundamentals and usage](fundamentals/index.md) — array creation, indexing, dtypes, broadcasting, copies/views, ufuncs, I/O.
- [Advanced usage and interoperability](advanced/index.md) — extending NumSharp, native backends, internals, interop.
- [NumPy API Coverage & Support](coverage-support-dashboard.md) · [API reference](../api/index.md).
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html) — the upstream article this converts.
