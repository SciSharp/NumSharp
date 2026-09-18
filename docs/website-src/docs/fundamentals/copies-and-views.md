# Copies and views

An `NDArray` is two parts: a **data buffer** (the elements, in unmanaged memory) and **metadata** (dtype, shape, strides, offset). Two arrays can share the same buffer while presenting it differently through their metadata — that shared-buffer array is a **view**. An array with its own duplicated buffer is a **copy**.

Knowing which operations return which is the single most important thing to internalize about NumSharp: a write through a view mutates the original; a write through a copy does not. This is identical to NumPy, and it is what makes slicing cheap.

<!-- Tests: NumSharp.Tests.Documentation.FundamentalsCopiesViewsDocTests — every code example on this page is executed and asserted in test/NumSharp.Tests/Documentation/FundamentalsCopiesViewsDocTests.cs. Section → method(s):
     View shares buffer / Copy independent → View_SharesBuffer_WriteThrough; Copy_IsIndependent
     Basic → view, advanced → copy → Indexing_BasicIsView_AdvancedIsCopy
     reshape/ravel/flatten/T → Reshape_Ravel_Flatten_ViewVsCopy
     Broadcast views are read-only → Broadcast_IsReadOnly
     How to tell (@base / shares_memory) → HowToTell_Base_And_SharesMemory
     In-place modification → InPlace_CompoundAssignIsNotInPlace_SliceAssignIs
     ascontiguousarray → AsContiguousArray_MaterializesWhenNeeded -->

---

## View

A **view** reuses the original data buffer and only changes metadata — a different `Shape` (strides, offset) or `dtype`. Because the buffer is shared, **writing through a view changes the original**:

```csharp
var x = np.arange(10);              // [0 1 2 3 4 5 6 7 8 9]
var y = x["1:3"];                   // a view
x["1:3"] = np.array([10, 11]);
// x → [0 10 11 3 4 5 6 7 8 9]
// y → [10 11]   ← the view sees the change
```

Views are how NumSharp avoids copying: a slice of a gigabyte array costs a few bytes of metadata.

## Copy

A **copy** duplicates both the buffer and the metadata. Changes to a copy never touch the original. Copying is slower and uses memory, but is sometimes exactly what you want:

```csharp
var y = x["1:3"].copy();            // independent buffer
y[0] = 999;                         // x is unaffected
```

Force a copy with `.copy()` (or `np.copy(arr)`).

---

## Indexing operations

The rule is mechanical:

- **Basic indexing** (integers, slices, ellipsis, newaxis) → **always a view**. The selected elements can be addressed with an offset and strides into the original, so no data moves.
- **Advanced indexing** (integer arrays, boolean masks) → **always a copy**. The selected elements are gathered into a fresh buffer.

```csharp
var x = np.arange(10);
var v = x["1:3"];                   // basic → view      (v.@base is x's storage)

var m = np.arange(9).reshape(3, 3);
var c = m[np.array([1, 2])]; // advanced → copy   (c.@base is null)
```

> **The assignment asymmetry.** *Reading* `x[mask]` or `x[idx]` returns a copy — writing into that copy does nothing to `x`. But *assigning* `x[mask] = v` / `x[idx] = v` scatters back into `x` through the setter. So `var picked = x[mask]; picked[0] = 9;` leaves `x` unchanged, while `x[mask] = 9` changes it. This matches NumPy exactly. See [Indexing → Views vs copies](indexing.md#views-vs-copies-at-a-glance).

---

## Other operations

`reshape` returns a **view where possible, a copy otherwise**. It can reshape by adjusting strides for a contiguous array, but a layout that cannot be expressed with strides (e.g. after a transpose) forces a copy:

```csharp
var x = np.ones((2, 3));
var y = x.T;                        // transpose → non-contiguous view
var z = y.reshape(6);               // must copy (strides can't express it)
```

`ravel` returns a contiguous flattened **view** where possible; `flatten` **always copies**. When you specifically want a view most of the time, `x.reshape(-1)` is preferable:

```csharp
x.ravel();          // view if contiguous, else copy
x.flatten();        // always a copy
x.reshape(-1);      // view where possible
```

`.T` / `transpose`, `swapaxes`, `flip`/`fliplr`/`flipud`, `rot90`, `broadcast_to`, and field-free `np.diag` on a 2-D input are all **views** (O(1) stride tricks). `astype` copies (it changes the dtype and therefore the buffer).

---

## Broadcast views are read-only

A broadcast view has a **stride of 0** on the stretched axis — one stored element is read for many logical positions. Writing to it would corrupt every position that aliases that element, so NumSharp makes broadcast views **non-writeable**:

```csharp
var small = np.array([1, 2, 3]);
var big = np.broadcast_to(small, (1000000, 3));   // read-only view, ~3 elements of memory
big.Shape.IsBroadcasted;    // true
big.Shape.IsWriteable;      // false
big[0, 0] = 9;              // throws: assignment destination is read-only
var writable = big.copy();  // materialize all 3M elements to write
```

See [Broadcasting → Memory behavior](../broadcasting.md#memory-behavior).

---

## How to tell a view from a copy

NumSharp mirrors NumPy's `ndarray.base` and adds explicit checks:

```csharp
var x = np.arange(9);
var v = x.reshape(3, 3);            // view
var c = x[np.array([0, 2])]; // copy

(v.@base is not null);              // true   ← the simple boolean check (pattern match)
(c.@base is not null);              // false
```

`arr.@base` is NumPy's `ndarray.base` — `null` when the array owns its data, otherwise an `NDArray` wrapping the base storage; views chain to the **ultimate owner**, not intermediate views:

```csharp
var a = np.arange(10);              // a.@base is null      (owns data)
var b = a["2:5"];                   // b.@base is not null  (view of a)
var d = a.copy();                   // d.@base is null      (copy owns data)
np.shares_memory(a, b);             // true — b really does share a's buffer
```

> **Use `arr.@base is not null` for a boolean test, written with the `is`/`is not` pattern** — not `arr.@base != null`, because `!=` on `NDArray` is the elementwise-comparison operator, not a null check. `@base` differs from NumPy in one way: it builds a fresh wrapper each call, so `ReferenceEquals(b.@base, a)` is `false` even though they share memory (confirm the sharing with `np.shares_memory`). (NumSharp also has an internal `Storage.IsView` that says the same thing, but `Storage` is not part of the public API.)

To ask whether two arrays could share memory, use `np.shares_memory` / `np.may_share_memory`:

```csharp
np.shares_memory(x, v);             // true  — exact overlap solver
np.may_share_memory(x, c);          // false — fast bounds check
```

The `Shape` flags expose the layout that decides all of this: `IsContiguous`, `IsSliced`, `IsBroadcasted`, `IsWriteable`, `IsFContiguous`. See [NDArray](../NDArray.md).

---

## In-place modification

Writing through a view mutates shared memory. But C#'s **compound-assignment operators are not in-place** — the compiler expands `x += 1` into `x = x + 1`, allocating a new array and rebinding the variable. Other references keep seeing the old data:

```csharp
var x = np.array([1, 2, 3]);
var alias = x;
x += 10;                 // x → NEW array [11, 12, 13]
// alias is still [1, 2, 3]   ← differs from NumPy, where += is in-place!
```

To mutate the buffer in place (so aliases and views observe the change), **assign through a slice** — the setter path:

```csharp
x[":"] = x + 1;          // rewrites x's buffer; alias sees it
data["1:3"] += 5;        // expands to data["1:3"] = data["1:3"] + 5, but the LHS
                         // is a view, so the result is written back into the parent
```

This is a deliberate consequence of C# operator semantics, and one of the few places ported NumPy code behaves differently. See [Getting & Setting Values → in-place vs reassignment](../getting-and-setting-values.md#in-place-vs-reassignment).

---

## Common patterns

### Detach a small slice from a large parent

```csharp
var window = huge["1000:1010"].copy();   // parent can now be collected
```

A view keeps the whole parent buffer alive. Copy the piece you need if the parent is large and about to go out of scope.

### Guarantee a writeable, contiguous array

```csharp
var w = np.ascontiguousarray(maybeView);  // copies only if needed
```

### Modify in place safely

```csharp
arr[":"] = arr * 2;      // in-place double; views/aliases observe it
```

---

## Troubleshooting

### "I modified a slice and the original array changed"
Basic slices are views — that is by design. Use `.copy()` for independence.

### "assignment destination is read-only"
You are writing to a broadcast view (stride 0). Copy first: `var w = b.copy();`. See [Broadcasting](../broadcasting.md#memory-behavior).

### "`x += 1` didn't update another reference"
C# compound assignment rebinds; it does not mutate. Use `x[":"] = x + 1`.

### "`arr.@base != null` gave a weird array instead of true/false"
`!=` on `NDArray` is elementwise. Use `arr.@base is not null` (the `is not` pattern) for a boolean.

---

## API reference

| Member | Returns | Meaning |
|--------|---------|---------|
| `arr.copy()` / `np.copy(arr)` | `NDArray` | force an independent copy |
| `arr.@base` | `NDArray?` | the owning array (null if it owns its data); chains to the ultimate owner |
| `arr.@base is not null` | `bool` | simple public view check (use the `is not` pattern, not `!=`) |
| `np.shares_memory(a, b, max_work)` | `bool` | exact overlap solver |
| `np.may_share_memory(a, b, max_work)` | `bool` | fast bounds-based check |
| `arr.Shape.IsContiguous` / `.IsSliced` / `.IsBroadcasted` / `.IsWriteable` / `.IsFContiguous` | `bool` | layout flags |

| Operation | View or copy |
|-----------|--------------|
| Basic index / slice, `.T`, `transpose`, `swapaxes`, `flip`, `rot90`, `broadcast_to`, `np.diag` (2-D input) | **view** |
| Advanced index (integer array, boolean mask) | **copy** |
| `reshape`, `ravel` | **view if possible**, else copy |
| `flatten`, `astype`, `.copy()` | **copy** |

---

## Related reading

- [Indexing on NDArray](indexing.md) — which index forms view and which copy.
- [Getting & Setting Values](../getting-and-setting-values.md) — the write-through rules in full.
- [Broadcasting](../broadcasting.md) — why broadcast views are read-only.
- [NDArray](../NDArray.md) — storage, shape, and the anatomy behind views.
- [NumPy copies-and-views guide](https://numpy.org/doc/stable/user/basics.copies.html) — the upstream article.
