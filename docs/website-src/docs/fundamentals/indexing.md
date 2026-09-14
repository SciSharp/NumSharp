# Indexing on NDArray

`NDArray` is indexed with the `x[obj]` syntax, exactly like NumPy's `ndarray`. There are three kinds of indexing, and they behave differently:

- **Basic indexing** — integers and slices → returns a **view** (shared memory).
- **Advanced indexing** — integer arrays and boolean masks → returns a **copy**.
- **Field access** — string field names → *not supported* in NumSharp (no structured dtypes; see below).

This page explains the concepts and the one syntactic difference from NumPy that matters most (Python slice literals become C# **strings**). For the full catalogue of accessors — every getter and setter, with the conversion and write-through rules — see [Getting & Setting Values](../getting-and-setting-values.md).

<!-- Tests: NumSharp.Tests.Documentation.FundamentalsIndexingDocTests — every code example on this page is executed and asserted in test/NumSharp.Tests/Documentation/FundamentalsIndexingDocTests.cs. Section → method(s):
     Single-element indexing → Basic_SingleElement_NegativeAndCoordinate
     Partial index (sub-array view) → Basic_PartialIndex_ReturnsSubArrayView
     Slicing and striding → Basic_SliceAndStride
     Ellipsis and newaxis → Basic_EllipsisAndNewaxis
     Integer-array indexing → Advanced_IntegerArray_NegativeAllowed; Advanced_PairedIndexArrays_IterateAsOne; Advanced_IxUnderscore_MakesAGrid
     Raw int[] sole index is fancy → Advanced_RawIntArraySoleIndex_IsFancy_SelectsRows
     Coordinate access via GetData → Advanced_CoordinateAccess_UsesGetData
     Boolean-array indexing → Advanced_BooleanMask; Advanced_BooleanMask_FewerDims_SelectsLeadingAxes
     Flat iteration (write-through) → Flat_Flatiter_WritesThroughAnyLayout
     Assignment → Assignment_SliceFillAndBroadcast; Assignment_FancyDuplicates_LastWriteWins -->

---

## The one big syntax difference: slices are strings

C# has no `a[1:5:2]` slice syntax, so NumSharp passes the slice expression as a **string**:

| NumPy (Python) | NumSharp (C#) |
|----------------|---------------|
| `x[2]` | `x[2]` |
| `x[1:5]` | `x["1:5"]` |
| `x[1:5:2]` | `x["1:5:2"]` |
| `x[::-1]` | `x["::-1"]` |
| `x[:, 0]` | `x[":, 0"]` |
| `x[..., -1]` | `x["..., -1"]` |
| `x[1:3, :2]` | `x["1:3, :2"]` |

Separate integer arguments stay coordinates: `x[1, 3]` is the element at row 1, column 3. A whole slice expression — including commas that separate axes — goes inside **one** string.

---

## Basic indexing

### Single-element indexing

0-based, negative indices count from the end, and you do not need a bracket per dimension:

```csharp
var x = np.arange(10);
x[2];          // element 2
x[-2];         // element 8

x = x.reshape(2, 5);   // now 2-D
x[1, 3];       // 8
x[1, -1];      // 9
```

Indexing a multidimensional array with **fewer indices than dimensions** returns a sub-array — a **view** into the rest:

```csharp
var row = x[0];    // the first row → (5,) view, shares memory with x
row[2];            // 2
```

So `x[0, 2]` equals `x[0][2]`, though the second form makes a temporary sub-array first. NumSharp uses **C-order** indexing: the last index varies fastest in memory.

### Slicing and striding

The slice grammar is `i:j:k` (start, stop, step), `stop` exclusive, negative values counting from the end, and `k < 0` stepping backward:

```csharp
var x = np.arange(10);
x["1:7:2"];    // [1 3 5]
x["-2:10"];    // [8 9]
x["-3:3:-1"];  // [7 6 5 4]
x["5:"];       // [5 6 7 8 9]
x["::-1"];     // reversed
```

**Every basic-slice result is a view of the original array** — no data is copied. If the number of index items is fewer than the rank, `:` is assumed for the trailing axes.

> **Care with large parents.** A small slice keeps the *entire* parent buffer alive (the view references it). If you extract a small piece of a large array and discard the large one, call `.copy()` so the parent can be collected.

### Dimensional tools: ellipsis and newaxis

`...` (ellipsis) expands to as many `:` as needed to index every remaining axis; there may be only one:

```csharp
var x = np.arange(6).reshape(2, 3, 1);
x["..., 0"];       // same as x[:, :, 0]
```

`np.newaxis` inserts a length-1 axis, useful for setting up broadcasts:

```csharp
var x = np.arange(5);
var outer = x[np.newaxis].T + x[np.newaxis];   // (5,1) + (1,5) → (5,5)
```

---

## Advanced indexing

Advanced indexing is triggered when the index is an **integer array**, a **boolean array**, or a tuple containing one. It **always returns a copy** (contrast with basic slicing, which returns a view).

### Integer-array indexing

Select arbitrary elements by index. Negative values are allowed; out-of-range raises `IndexError`:

```csharp
var x = np.arange(10, 1, -1);          // [10 9 8 7 6 5 4 3 2]
x[np.array(new[] { 3, 3, 1, 8 })];     // [7 7 9 2]
x[np.array(new[] { 3, 3, -3, 8 })];    // [7 7 4 2]
```

With one index array per dimension, the arrays are **broadcast together and iterated as one** — the result has the broadcast index shape:

```csharp
var y = np.arange(35).reshape(5, 7);
y[np.array(new[] {0, 2, 4}), np.array(new[] {0, 1, 2})];   // [0 15 30]  (y[0,0], y[2,1], y[4,2])
```

To select a **grid** (the outer combination) rather than the diagonal, broadcast the indices or use `np.ix_`:

```csharp
var rows = np.array(new[] { 0, 3 });
var cols = np.array(new[] { 0, 2 });
x[np.ix_(rows, cols)];      // the 2×2 corner block
// x[rows, cols] alone would select only the diagonal [x[0,0], x[3,2]]
```

### ⚠️ The NumSharp gotcha: a raw `int[]` sole index is *fancy*

A raw C# `int[]`/`long[]` used as the **whole** index is advanced (fancy) indexing — it selects *rows*, matching NumPy's `a[[0, 2]]`:

```csharp
var a = np.arange(12).reshape(3, 4);
a[new[] { 0, 2 }];         // rows 0 and 2 → shape (2, 4)   ← FANCY, a copy
a[0, 2];                   // the element at (0, 2)          ← coordinate
```

To access an element (or sub-array) **by a coordinate array**, use `a.GetData(int[])`, not `a[coordArray]`:

```csharp
a.GetData(new[] { 0, 2 }); // sub-array/element at coordinate (0, 2) → value 2
```

This is the single most common porting surprise. When in doubt: separate integers = coordinate; an array object = fancy.

### Boolean-array indexing

A boolean mask selects the elements where it is `True`, in C-order, returning a 1-D copy:

```csharp
var x = np.array(new[,] { { 1.0, 2.0 }, { double.NaN, 3.0 } });
x[!np.isnan(x)];           // [1. 2. 3.]  (drops NaN)

var v = np.array(new[] { 1.0, -1.0, -2.0, 3.0 });
v[v < 0] = 20;             // masked assignment writes through → [1, 20, 20, 3]
```

A mask with the same shape as `x` yields a 1-D result of the `True` elements. A mask with fewer dimensions selects along the leading axes (equivalent to `x[mask, ...]`):

```csharp
var x = np.arange(35).reshape(5, 7);
var b = x > 20;
x[b["..., 5"]];       // rows where column 5 exceeds 20
```

### Combining basic and advanced indexing

When advanced indices are mixed with slices, the advanced part is a copy and the result shape follows NumPy's rules (advanced dims first if separated by a slice, in place if adjacent). For the exact shape logic and worked examples, see the [NumPy indexing guide](https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing) — NumSharp matches it.

---

## Field access — not supported

NumPy lets you index a **structured array** by field name (`x['age']`). NumSharp does not implement structured/record dtypes, so there is no string-field indexing. See [Structured arrays](structured-arrays.md) for why, and the .NET-idiomatic alternatives (parallel `NDArray`s, struct arrays, `NDArray<T>`).

---

## Flat iteration

`a.flat` is a **raveled view** in C-order (copies for a non-contiguous layout), and `a.flatiter` is the **write-through** flat accessor for every layout:

```csharp
foreach (var v in a.flat) { /* boxed, C-order */ }

var t = a.reshape(2, 3).T;      // transposed (non-contiguous) view
t.flatiter[5] = 99;             // writes THROUGH in logical C-order
t.flatiter["::2"] = 0;          // slice assignment on the flat view
```

Use `a.flatiter` (not `a.flat`) whenever you **assign** by flat index into a non-contiguous view — `a.flat` would copy and silently lose the write. For iteration order and unboxed/fast iteration, see [Iterating & Enumerating](../iterating-and-enumerating.md). `np.ndindex` and `np.ndenumerate` give index-space and `(index, value)` walks.

---

## Assigning to indexed arrays

Assignment mirrors indexing: the right-hand side must broadcast to the shape the index selects. Basic slices, masks, and fancy indices all write through to the original data.

```csharp
var x = np.arange(10);
x["2:7"] = 1;                        // fill a slice with a scalar
x["2:7"] = np.arange(5);             // or a shaped RHS (broadcasts)
x[x > 5] = 0;                        // masked assignment writes through
x[np.array(new[] {0, 2})] = -1;      // fancy assignment
```

Two behaviors to keep in mind:

- **Cross-dtype assignment coerces.** Writing a float into an int array truncates; an out-of-range *weak* scalar raises `OverflowException`. Full rules: [Getting & Setting Values → coercion](../getting-and-setting-values.md#value-coercion-on-assignment-nep50).
- **C# `+=` is not in-place.** `x += 1` rebinds `x` to a new array; other references do not see the change. NumPy's "buffered `x[[1,1,3,1]] += 1` increments once" example does not translate directly — in C# write through a slice (`x[":"] = x + 1`) to mutate memory. See [Getting & Setting Values → in-place](../getting-and-setting-values.md#in-place-vs-reassignment).

Duplicate fancy indices follow **last-write-wins** on assignment (`x[np.array(new[]{1,1})] = ...` keeps the last), and the iteration order of advanced assignment is otherwise unspecified — do not rely on it when an element is written more than once.

---

## Views vs copies at a glance

| Index form | Returns | Writing through it… |
|------------|---------|---------------------|
| `x[0]`, `x["1:3"]`, `x["..., -1"]`, `x.T` | **view** | mutates the parent |
| `x[np.array(new[]{0,2})]` (fancy read) | **copy** | does nothing to the parent — but `x[idx] = v` (setter) does |
| `x[mask]` (boolean read) | **copy** | same asymmetry — `x[mask] = v` writes through |
| `np.broadcast_to(x, shape)` | **read-only view** | throws (`assignment destination is read-only`) |

This is the rule that governs everything — see [Copies and views](copies-and-views.md).

---

## Common patterns

### Filter by condition

```csharp
var data = np.array(new[] { 1.0, -1.0, 2.0, -2.0 });
var positive = data[data > 0];       // [1., 2.]  (copy)
data[data < 0] = 0;                  // clamp negatives in place
```

### Select rows by index list

```csharp
var m = np.arange(20).reshape(4, 5);
var picked = m[np.array(new[] { 0, 2, 3 })];   // rows 0, 2, 3 → (3, 5)
```

### Take a diagonal grid vs a mesh

```csharp
m[np.array(new[]{0,1}), np.array(new[]{2,3})];      // [m[0,2], m[1,3]] — paired
m[np.ix_(np.array(new[]{0,1}), np.array(new[]{2,3}))]; // 2×2 block — meshed
```

---

## Troubleshooting

### "`a[new[]{0,2}]` returned rows, not the element at (0,2)"
A raw array as the sole index is **fancy**. Use `a.GetData(new[] { 0, 2 })` for coordinate access, or separate integers `a[0, 2]`.

### "My slice write didn't stick" / "the original changed unexpectedly"
Basic slices are **views** (writes stick and affect the parent); fancy/boolean *reads* are **copies**. Copy with `.copy()` when you want independence. See [Copies and views](copies-and-views.md).

### "`x += 1` didn't update another reference to the same array"
C# compound assignment rebinds the variable. Use `x[":"] = x + 1` for in-place mutation.

### "Assigning a float into an int array threw / wrapped"
That's NEP 50 coercion — weak scalars range-check, strong arrays wrap. See [Getting & Setting Values](../getting-and-setting-values.md#value-coercion-on-assignment-nep50).

---

## API reference

| Form | Example | Kind | Result |
|------|---------|------|--------|
| Integer | `x[1, 3]` | basic | element → 0-d view |
| Integer (partial) | `x[0]` | basic | sub-array view |
| Slice string | `x["1:3, :2"]` | basic | view |
| Ellipsis / newaxis | `x["..., 0]"`, `x[np.newaxis]` | basic | view |
| Integer array | `x[np.array(new[]{0,2})]` | advanced | copy |
| Raw `int[]` sole index | `x[new[]{0,2}]` | advanced | copy (selects rows) |
| Coordinate array | `x.GetData(new[]{0,2})` | — | element/sub-array |
| Boolean mask | `x[mask]` | advanced | 1-D copy |
| `np.ix_(...)` | `x[np.ix_(r, c)]` | advanced | meshed block |
| Flat write-through | `x.flatiter[i] = v` | — | any layout |

---

## Related reading

- [Getting & Setting Values](../getting-and-setting-values.md) — every accessor, with conversion and write-through rules.
- [Copies and views](copies-and-views.md) — when indexing shares memory.
- [Broadcasting](../broadcasting.md) — the shape rules that govern assignment.
- [Iterating & Enumerating](../iterating-and-enumerating.md) — `flat`, `nditer`, `ndindex`, `ndenumerate`.
- [NumPy indexing guide](https://numpy.org/doc/stable/user/basics.indexing.html) — the upstream article.
