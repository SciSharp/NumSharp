# NumSharp: the absolute basics for beginners

Welcome to the absolute beginner's guide to NumSharp — the .NET port of NumPy. NumSharp gives C# the same thing NumPy gives Python: a multidimensional array type (`NDArray`) and a large library of functions (`np.*`) that operate on it efficiently. If you know NumPy, this reads almost identically; if you don't, this page builds the mental model from scratch.

This is the on-ramp. Each topic here has a deeper companion in [Fundamentals and usage](fundamentals/index.md), linked as we go.

> **The one thing to know up front:** C# has no `a[1:5]` slice syntax, so NumSharp writes slices as **strings** — `a["1:5"]`. Whole-number coordinates stay normal: `a[1, 3]`. That's the biggest surface difference from NumPy; everything else is close.

<!-- Tests: NumSharp.Tests.Documentation.AbsoluteBasicsDocTests — every code example on this page is executed and asserted in test/NumSharp.Tests/Documentation/AbsoluteBasicsDocTests.cs. Section → method:
     Array fundamentals → Fundamentals_CreateIndexMutateSliceView
     Array attributes → Attributes_NdimShapeSizeDtype
     How to create a basic array → Create_ZerosOnesEmptyArangeLinspace
     Adding, removing, and sorting → SortAndConcatenate
     How do you know the shape and size (3-D) → ShapeSizeNdim_3D
     Can you reshape an array → Reshape
     Convert 1D to 2D (newaxis/expand_dims) → NewAxis_RowAndColumn
     Indexing and slicing (boolean/nonzero) → BooleanMasksAndNonzero
     Create from existing data (stack/split/view/copy) → Stack_Split_ViewCopy
     Basic array operations → BasicOps_And_SumAxis
     Broadcasting → Broadcasting_Scalar
     More useful array operations → Aggregations_WithAxis
     Creating matrices → Matrices_IndexAggregateBroadcast
     Generating random numbers → Random_SeededDeterministic
     Unique items and counts → Unique_ValuesIndexCounts_And_Axis
     Transposing → TransposeAndT
     Reversing an array (flip) → Flip_And_InPlaceColumn
     Reshaping and flattening → FlattenIsCopy_RavelIsView
     Working with mathematical formulas → MseFormula
     Save and load → SaveLoad_Npy_And_Text
     Importing and exporting a CSV → SaveTxt_WithFmtDelimiterHeader
     Plotting arrays → ToArray_ForPlotting
     (How to reference NumSharp, Why NumSharp, What is an array, Reading examples, Getting help: no executable code.) -->

---

## How to reference NumSharp

After adding the `NumSharp` NuGet package, bring it into scope with a `using` directive:

```csharp
using NumSharp;
```

That puts the static class `np` and the `NDArray` type in scope, so `np.array(...)`, `np.zeros(...)`, etc. read just like Python's `np.` prefix.

---

## Reading the example code

Examples show a C# expression followed by its result in a comment:

```csharp
var a = np.array(new[,] { { 1, 2, 3 }, { 4, 5, 6 } });
a.shape;    // (2, 3) → long[] { 2, 3 }
```

Run them in a `dotnet run` file-based app, LINQPad, a console project, or a C# notebook. The `// →`/`//` text is the output, not code.

---

## Why use NumSharp?

.NET arrays (`int[]`, `double[,]`) and `List<T>` are excellent general-purpose containers. But for **large quantities of same-type numeric data** processed with whole-array math, an `NDArray` is faster and far more convenient: one type holds any rank, operations vectorize (SIMD) with no explicit loops, and slicing shares memory instead of copying. NumSharp shines exactly where NumPy does — bulk homogeneous numeric work on the CPU.

---

## What is an "array"?

An array stores data in a grid. A **1-D** array is like a list; a **2-D** array like a table; a **3-D** array like a stack of tables; and so on to any number of dimensions — which is why the type is called `NDArray` (**N-D**imensional array).

NumSharp arrays have the same restrictions NumPy's do:

- **All elements share one data type** (the array is *homogeneous*).
- **The total size is fixed** once created.
- **The shape is rectangular**, not jagged — every row of a 2-D array has the same number of columns.

Meeting these conditions is what lets NumSharp make the array fast and memory-efficient.

---

## Array fundamentals

Create an array from a .NET sequence:

```csharp
var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
a;          // [1 2 3 4 5 6]
```

Access an element by its **0-based** index. For a 1-D array this returns a 0-D array (a single element is still an `NDArray`); cast it to get the value:

```csharp
a[0];        // a 0-D array printing 1
(int)a[0];   // 1  ← the scalar value
```

The array is mutable:

```csharp
a[0] = 10;
a;           // [10 2 3 4 5 6]
```

Slicing uses **string** notation (`start:stop:step`, stop exclusive):

```csharp
a["0:3"];    // [10 2 3]
```

A major difference from a .NET array: slicing an `NDArray` returns a **view** — an object referring to the *same* data. Mutating the view mutates the original:

```csharp
var b = a["3:"];
b;           // [4 5 6]
b[0] = 40;
a;           // [10 2 3 40 5 6]  ← changed through the view
```

See [Copies and views](fundamentals/copies-and-views.md) for when operations return views vs copies.

Higher-dimensional arrays come from nested .NET arrays:

```csharp
var a = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
a;
// [[ 1  2  3  4]
//  [ 5  6  7  8]
//  [ 9 10 11 12]]
```

A dimension is sometimes called an **axis**. Index along each axis with commas inside **one** set of brackets — the element `8` is in row `1`, column `3`:

```csharp
a[1, 3];        // a 0-D array printing 8
(int)a[1, 3];   // 8
```

> Think of the column index as coming *last* and the row index as *second-to-last*; this generalizes to any number of axes.

---

## Array attributes

*Covers `ndim`, `shape`, `size`, `dtype`.*

```csharp
var a = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });

a.ndim;    // 2      number of axes
a.shape;   // (3, 4) → long[] { 3, 4 }
a.size;    // 12     total number of elements (product of the shape)
a.dtype;   // int32
```

> **⚠️ Divergence — the default integer dtype.** In NumPy `np.array([1,2,3])` is **int64**; in NumSharp `np.array(new[]{1,2,3})` is **int32**, because it follows the .NET `int` type. (`np.arange` and most intrinsic integer functions *do* give int64.) Pass a dtype explicitly when it matters: `np.array(new[]{1,2,3}, np.int64)`. See [Array creation → the int32-vs-int64 gotcha](fundamentals/array-creation.md#the-int32-vs-int64-gotcha).

`a.dtype` is a `DType` descriptor (`a.dtype.type` is the CLR `Type`, `(Type)a.dtype` the explicit cast). More in [Data types](dtypes.md).

---

## How to create a basic array

*Covers `np.zeros`, `np.ones`, `np.empty`, `np.arange`, `np.linspace`.*

```csharp
np.zeros(2);          // [0. 0.]     (default dtype is float64)
np.ones(2);           // [1. 1.]
np.empty(2);          // uninitialized — content is whatever was in memory; fill it yourself
np.arange(4);         // [0 1 2 3]
np.arange(2, 9, 2);   // [2 4 6 8]   (start, stop-exclusive, step)
np.linspace(0, 10, 5); // [ 0. 2.5 5. 7.5 10. ]  (evenly spaced, endpoints included)
```

`empty` is faster than `zeros` because it skips initialization — just be sure to fill every element afterward.

Specify the dtype explicitly with the second argument (the default is float64):

```csharp
np.ones(2, np.int64);   // [1 1]  as int64
```

More at [Array creation](fundamentals/array-creation.md).

---

## Adding, removing, and sorting elements

*Covers `np.sort`, `np.concatenate`.*

```csharp
var arr = np.array(new[] { 2, 1, 5, 3, 7, 4, 6, 8 });
np.sort(arr);   // [1 2 3 4 5 6 7 8]  (returns a sorted copy)
```

NumSharp also has `argsort` (indirect sort), `lexsort` (stable multi-key), `searchsorted` (find in a sorted array), and `partition` (partial sort) — see [Sorting & Searching](../api/index.md).

Join arrays with `np.concatenate`. In C# the arrays are passed as an **array** (`new[] { … }`), not a Python tuple:

```csharp
var a = np.array(new[] { 1, 2, 3, 4 });
var b = np.array(new[] { 5, 6, 7, 8 });
np.concatenate(new[] { a, b });   // [1 2 3 4 5 6 7 8]

var x = np.array(new[,] { { 1, 2 }, { 3, 4 } });
var y = np.array(new[,] { { 5, 6 } });
np.concatenate(new[] { x, y }, axis: 0);
// [[1 2]
//  [3 4]
//  [5 6]]
```

To *remove* elements, index the ones you want to keep (see [Indexing](#indexing-and-slicing) below).

---

## How do you know the shape and size of an array?

*Covers `ndim`, `size`, `shape`.*

- `arr.ndim` — the number of axes.
- `arr.size` — the total number of elements (product of the shape).
- `arr.shape` — a `long[]` of the element count along each axis.

```csharp
var example = np.array(new[,,] {
    { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } },
    { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } },
    { { 0, 1, 2, 3 }, { 4, 5, 6, 7 } } });

example.ndim;    // 3
example.size;    // 24
example.shape;   // (3, 2, 4)
```

---

## Can you reshape an array?

*Covers `reshape`.*

**Yes.** `reshape` gives an array a new shape without changing its data — the new shape must have the same number of elements as the original.

```csharp
var a = np.arange(6);   // [0 1 2 3 4 5]
var b = a.reshape(3, 2);
// [[0 1]
//  [2 3]
//  [4 5]]
```

`np.reshape` also takes an explicit shape and an optional `order`:

```csharp
np.reshape(a, (1, 6));   // [[0 1 2 3 4 5]]
```

`order` is `C` (row-major, the default), `F` (column-major), or `A`. It controls how indices map to memory order — see [Under the hood → indexing order](advanced/under-the-hood.md#multidimensional-array-indexing-order).

---

## How to convert a 1D array into a 2D array (adding a new axis)

*Covers `np.newaxis`, `np.expand_dims`.*

`np.newaxis` inserts one length-1 axis; where you put it decides row vs column:

```csharp
var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
a.shape;                       // (6,)

a[np.newaxis].shape;           // (1, 6)   — a row vector (axis prepended)
a[Slice.All, np.newaxis].shape; // (6, 1)  — a column vector (Slice.All is Python's `:`)
```

`np.expand_dims` does the same by explicit axis position:

```csharp
np.expand_dims(a, axis: 1).shape;   // (6, 1)
np.expand_dims(a, axis: 0).shape;   // (1, 6)
```

More at [Indexing → ellipsis and newaxis](fundamentals/indexing.md#dimensional-tools-ellipsis-and-newaxis).

---

## Indexing and slicing

Index and slice much like a .NET collection, but with string slices and comma coordinates:

```csharp
var data = np.array(new[] { 1, 2, 3 });
data[1];        // 2 (0-D)
data["0:2"];    // [1 2]
data["1:"];     // [2 3]
data["-2:"];    // [2 3]
```

**Boolean masking** selects the elements that satisfy a condition (returns a 1-D copy):

```csharp
var a = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });

a[a < 5];              // [1 2 3 4]
a[a >= 5];             // [ 5  6  7  8  9 10 11 12]
a[a % 2 == 0];         // [ 2  4  6  8 10 12]
a[(a > 2) & (a < 11)]; // [ 3  4  5  6  7  8  9 10]
```

The `&` and `|` operators combine boolean arrays and can also produce a boolean result directly:

```csharp
(a > 5) | (a == 5);
// [[False False False False]
//  [ True  True  True  True]
//  [ True  True  True  True]]
```

`np.nonzero` returns the indices where a condition holds — one index array per axis:

```csharp
var idx = np.nonzero(a < 5);   // NDArray[] of length 2 (rows, cols)
// idx[0] = [0 0 0 0]   (row indices)
// idx[1] = [0 1 2 3]   (column indices)

a[idx[0], idx[1]];             // [1 2 3 4]  — gather those elements
```

More at [Indexing on NDArray](fundamentals/indexing.md).

---

## How to create an array from existing data

*Covers slicing, `np.vstack`, `np.hstack`, `np.hsplit`, views, and `copy`.*

Slice out a section:

```csharp
var a = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
a["3:8"];   // [4 5 6 7 8]   (index 3 through 7)
```

Stack arrays vertically or horizontally:

```csharp
var a1 = np.array(new[,] { { 1, 1 }, { 2, 2 } });
var a2 = np.array(new[,] { { 3, 3 }, { 4, 4 } });

np.vstack(a1, a2);
// [[1 1]
//  [2 2]
//  [3 3]
//  [4 4]]

np.hstack(a1, a2);
// [[1 1 3 3]
//  [2 2 4 4]]
```

Split with `np.hsplit` — by count, or at given columns (returns an `NDArray[]`):

```csharp
var x = np.arange(1, 25).reshape(2, 12);
np.hsplit(x, 3);                // 3 equal parts
np.hsplit(x, new[] { 3, 4 });   // split after columns 3 and 4 → 3 parts
```

**Views vs copies.** Slicing returns a *view* — a new array object over the same data — so writing through it changes the original:

```csharp
var a = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
var b1 = a[0];      // first row — a view
b1[0] = 99;
a;
// [[99  2  3  4]
//  [ 5  6  7  8]
//  [ 9 10 11 12]]   ← the original changed
```

For an independent **deep copy**, use `.copy()`:

```csharp
var b2 = a.copy();  // changes to b2 never touch a
```

More at [Copies and views](fundamentals/copies-and-views.md).

---

## Basic array operations

Arithmetic is elementwise:

```csharp
var data = np.array(new[] { 1, 2 });
var ones = np.ones(2, np.int32);

data + ones;   // [2 3]
data - ones;   // [0 1]
data * data;   // [1 4]
data / data;   // [1. 1.]   (/ is true division → float64, matching NumPy)
```

Aggregate with `sum` (works for any rank), as an instance method or `np.sum`:

```csharp
var a = np.array(new[] { 1, 2, 3, 4 });
a.sum();        // 10
```

For a 2-D array, choose the axis to aggregate over:

```csharp
var b = np.array(new[,] { { 1, 1 }, { 2, 2 } });
b.sum(axis: 0);  // [3 3]   (down the rows)
b.sum(axis: 1);  // [2 4]   (across the columns)
```

More at [Universal functions](fundamentals/ufuncs.md).

---

## Broadcasting

You can operate between an array and a single number, or between arrays of different-but-compatible shapes. Convert miles to kilometers:

```csharp
var data = np.array(new[] { 1.0, 2.0 });
data * 1.6;   // [1.6 3.2]
```

NumSharp *broadcasts* the scalar across every element. Dimensions are compatible when they're equal or one of them is 1; otherwise you get an `IncorrectShapeException`. Full rules at [Broadcasting](broadcasting.md).

---

## More useful array operations

*Covers `max`, `min`, `sum`, `mean`, `prod`, `std`, and the `axis` parameter.*

```csharp
var data = np.array(new[] { 1, 2, 3 });
data.max();    // 3
data.min();    // 1
data.sum();    // 6
data.mean();   // 2.0
data.prod();   // 6
data.std();    // 0.816496580927726
```

By default these aggregate the whole array. Pass `axis` to aggregate along one axis — e.g. the minimum of each column with `axis: 0`:

```csharp
var a = np.array(new[,] { { 1, 2 }, { 5, 3 }, { 4, 6 } });
a.max(axis: 0);   // [5 6]   (per column)
a.max(axis: 1);   // [2 5 6] (per row)
a.min(axis: 0);   // [1 2]
```

---

## Creating matrices

Pass a nested .NET array to make a 2-D array (a "matrix"):

```csharp
var data = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
data[0, 1];      // 2
data["1:3"];     // [[3 4] [5 6]]
data["0:2, 0"];  // [1 3]
```

Aggregate a matrix over all elements, or per column/row with `axis`:

```csharp
data.max();          // 6
data.sum();          // 21
data.max(axis: 0);   // per column
data.max(axis: 1);   // per row
```

Arithmetic works between same-size matrices, and broadcasts when one operand has a single row or column:

```csharp
var m    = np.array(new[,] { { 1, 2 }, { 3, 4 }, { 5, 6 } });
var row  = np.array(new[,] { { 1, 1 } });   // shape (1, 2)
m + row;
// [[2 3]
//  [4 5]
//  [6 7]]
```

Create initialized 2-D arrays by passing a shape tuple:

```csharp
np.ones((3, 2));
np.zeros((3, 2));
np.random.rand(3, 2);   // uniform random in [0, 1)
```

---

## Generating random numbers

NumSharp's `np.random` reproduces NumPy's sequences bit-for-bit for a given seed (MT19937 bit generator):

```csharp
np.random.seed(42);
np.random.rand(2, 3);            // uniform [0, 1)
np.random.randint(0, 5, (2, 4)); // random ints in [0, 5) — low inclusive, high exclusive
```

More in the [random API](../api/index.md).

---

## How to get unique items and counts

*Covers `np.unique`.*

```csharp
var a = np.array(new[] { 11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20 });
np.unique(a);   // [11 12 13 14 15 16 17 18 19 20]
```

`np.unique` returns a `UniqueResult` that converts to an `NDArray` (the values) and **deconstructs** when you ask for extra outputs. Request the first-occurrence indices with `return_index`, or the frequency counts with `return_counts`:

```csharp
var (values, indices) = np.unique(a, return_index: true);
// indices = [0 2 3 4 5 6 7 12 13 14]

var (vals, counts) = np.unique(a, return_counts: true);
// counts = [3 2 2 2 1 1 1 1 1 1]
```

It works on 2-D arrays too. Without `axis` the array is flattened; pass `axis: 0` for unique **rows**:

```csharp
var a2d = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 1, 2, 3, 4 } });
np.unique(a2d, axis: 0);
// [[ 1  2  3  4]
//  [ 5  6  7  8]
//  [ 9 10 11 12]]

var (rows, idx, cnt) = np.unique(a2d, axis: 0, return_index: true, return_counts: true);
// idx = [0 1 2],  cnt = [2 1 1]
```

More at [np.unique](../api/index.md).

---

## Transposing and reshaping a matrix

*Covers `reshape`, `transpose`, `T`.*

```csharp
var data = np.arange(6).reshape(2, 3);
data.reshape(3, 2);   // reshape to 3×2

var arr = np.arange(6).reshape(2, 3);
arr.transpose();
// [[0 3]
//  [1 4]
//  [2 5]]
arr.T;   // same as transpose()
```

`.T` and `.transpose()` return **views** (no data moves). More at [Copies and views](fundamentals/copies-and-views.md).

---

## How to reverse an array

*Covers `np.flip`.*

```csharp
var arr = np.array(new[] { 1, 2, 3, 4, 5, 6, 7, 8 });
np.flip(arr);   // [8 7 6 5 4 3 2 1]
```

For a 2-D array, `np.flip` with no axis reverses everything; `axis: 0` reverses rows, `axis: 1` reverses columns:

```csharp
var m = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });
np.flip(m);           // reverse both axes
np.flip(m, axis: 0);  // reverse row order
np.flip(m, axis: 1);  // reverse each row
```

You can flip a single row or column and write it back through a view:

```csharp
m[1] = np.flip(m[1]);              // reverse the second row, in place
m[":, 1"] = np.flip(m[":, 1"]);    // reverse the second column, in place
```

More at [np.flip](../api/index.md).

---

## Reshaping and flattening multidimensional arrays

*Covers `flatten`, `ravel`.*

Both flatten to 1-D. The difference: `flatten` returns a **copy**, while `ravel` returns a **view** where possible (so it's cheaper, but writes affect the parent):

```csharp
var x = np.array(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } });

var a1 = x.flatten();
a1[0] = 99;   // x is unchanged (flatten copied)

var a2 = x.ravel();
a2[0] = 98;   // x[0,0] becomes 98 (ravel is a view)
```

More at [Copies and views](fundamentals/copies-and-views.md).

---

## How to get help

NumPy uses `help()` and IPython's `?`/`??`. In C# the equivalents are richer and live in the IDE:

- **IntelliSense / hover / F12** — every `np.*` function and `NDArray` member carries XML doc comments that surface as tooltips, parameter hints, and go-to-definition in Visual Studio, VS Code, and Rider.
- **The API reference** — the generated [NumSharp API docs](../api/index.md) list every type, method, and overload.
- **This documentation** — the [Fundamentals](fundamentals/index.md) and [Advanced](advanced/index.md) guides.

There's no runtime `help()` call; the type system and IDE tooling replace it.

---

## Working with mathematical formulas

Implementing array math reads like the formula. Here's mean squared error, `MSE = (1/n) · Σ (prediction − label)²`:

```csharp
var predictions = np.array(new[] { 1.0, 1.0, 1.0 });
var labels      = np.array(new[] { 1.0, 2.0, 3.0 });

var error = np.mean(np.square(predictions - labels));   // 1.6666666666666665
// equivalently: (1.0 / labels.size) * np.sum(np.square(predictions - labels))
```

`predictions` and `labels` can hold one value or a million — they only need the same size. The subtraction, squaring, and reduction all vectorize.

---

## How to save and load NumSharp arrays

*Covers `np.save`, `np.savez`, `np.savetxt`, `np.load`, `np.loadtxt`.*

The **`.npy`** binary format stores the data, shape, and dtype so an array round-trips exactly — and NumSharp's output is **byte-for-byte identical to NumPy's `np.save`**, so files interchange with Python:

```csharp
var a = np.array(new[] { 1, 2, 3, 4, 5, 6 });
np.save("filename.npy", a);
var b = np.load_npy("filename.npy");   // [1 2 3 4 5 6]
```

Use `np.savez` for several arrays in one `.npz` archive (and `np.savez_compressed` for a compressed one). `np.load` returns `object` (an `NDArray` for `.npy`, an `NpzFile` for `.npz`); the typed `np.load_npy` / `np.load_npz` avoid the cast.

Save to plain text (CSV/TXT) with `np.savetxt`, and read it back with `np.loadtxt`:

```csharp
var csv = np.array(new[] { 1.0, 2, 3, 4, 5, 6, 7, 8 });
np.savetxt("new_file.csv", csv);
np.loadtxt("new_file.csv");   // [1. 2. 3. 4. 5. 6. 7. 8.]
```

`.npy`/`.npz` are smaller and faster; text files are easier to share. Full details — versions, `fortran_order`, big-endian, `mmap_mode` — at [I/O with NumSharp](fundamentals/io.md).

---

## Importing and exporting a CSV

For an all-numeric CSV, `np.loadtxt`/`np.savetxt` are all you need:

```csharp
var a = np.array(new[,] {
    { -2.58, 0.43, -1.24, 1.60 },
    { 0.99, 1.17, 0.94, -0.15 } });
np.savetxt("np.csv", a, fmt: "%.2f", delimiter: ",", header: "1,2,3,4");
// file:
// # 1,2,3,4
// -2.58,0.43,-1.24,1.60
// 0.99,1.17,0.94,-0.15
```

`np.loadtxt` reads columns with `usecols`, skips headers with `skiprows`, and takes a `delimiter` — see [I/O with NumSharp](fundamentals/io.md).

> **Mixed / labeled CSVs** (string *and* numeric columns, like a `music.csv` with an `Artist` name column) don't fit a single homogeneous `NDArray`, and NumSharp has no structured/string dtype. For those, use a .NET CSV/dataframe library, [ML.NET](interop/mlnet.md), or pandas through the [pythonnet bridge](interop/pandas.md), then hand the numeric columns to NumSharp. See [Structured arrays](fundamentals/structured-arrays.md) for why and the alternatives.

---

## Plotting arrays

NumSharp has **no built-in plotting** (NumPy relies on Matplotlib, a separate library, too). To visualize an `NDArray`, hand its values to a .NET charting library — **ScottPlot**, **OxyPlot**, or **Plotly.NET** all take `double[]` / `double[,]`, which you get from `arr.ToArray<double>()` or `arr.Unsafe.Span<double>()`:

```csharp
var y = np.array(new[] { 2.0, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22 });
double[] values = y.ToArray<double>();
// hand `values` to ScottPlot / OxyPlot / Plotly.NET
```

Or drive Matplotlib itself over your NumSharp arrays through the [pythonnet bridge](interop/pythonnet-numpy.md), which crosses the buffer to numpy zero-copy.

---

## Where to next

You now have the whole surface. For depth, read the [Fundamentals and usage](fundamentals/index.md) articles — each expands a section above:

- [Array creation](fundamentals/array-creation.md) · [Indexing on NDArray](fundamentals/indexing.md) · [Copies and views](fundamentals/copies-and-views.md)
- [Data types](dtypes.md) · [Broadcasting](broadcasting.md) · [Universal functions](fundamentals/ufuncs.md)
- [I/O with NumSharp](fundamentals/io.md) · [Getting & Setting Values](getting-and-setting-values.md) · [Iterating & Enumerating](iterating-and-enumerating.md)

And when you're extending or embedding NumSharp, the [Advanced usage](advanced/index.md) section covers custom kernels, native backends, the internals, and interoperability.

---

*This page is the NumSharp counterpart of NumPy's [absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html). Original NumPy image credits: Jay Alammar (jalammar.github.io).*
