# Getting & Setting Values

Reading and writing array data is where you spend most of your time with an `NDArray`. NumPy offers roughly a dozen ways to do it - indexing, slicing, masks, fancy indices, `fill`, `put`, `place`, `copyto`, `fill_diagonal`, the flat iterator - and NumSharp mirrors all of them, with the same semantics and the same view/copy rules. This page is the map: every way to *get* a value, every way to *set* one, which converts and which doesn't, and which ones write through to shared memory.

It builds on [NDArray](NDArray.md) (the anatomy and the basics), [Dtypes](dtypes.md) (element types and coercion), and [Broadcasting](broadcasting.md) (shape rules, which also govern assignment). If a term here is unfamiliar, start there.

---

## The rule that governs everything: views vs copies

Whether a *get* lets you *set* through it depends on one thing - did it return a **view** (shared memory) or a **copy** (independent memory)?

```csharp
var a = np.arange(10);
var v = a["2:5"];        // slice → view - shares memory with a
v[0] = 999;              // writes through: a[2] is now 999

var c = a["2:5"].copy(); // explicit copy - independent
c[0] = 0;                // a[2] stays 999
```

| Access | Returns | Write-through? |
|--------|---------|----------------|
| Plain slice - `a["1:3"]`, `a[0]`, `a["..., -1"]`, `a.T` | **view** | yes - mutates the parent |
| Fancy index (read) - `a[idx]` | **copy** | no - but `a[idx] = v` (the setter) *does* touch the parent |
| Boolean mask (read) - `a[mask]` | **copy** | no - but `a[mask] = v` (the setter) *does* touch the parent |
| Broadcasted view - `np.broadcast_to(a, shape)` | **read-only view** | writes throw (`assignment destination is read-only`) |

The asymmetry on fancy/boolean indexing is deliberate and matches NumPy: **reading** `a[mask]` gathers into a fresh array (writing into *that* array does nothing to `a`), but **assigning** `a[mask] = value` scatters back into `a` through the setter. See [NDArray → Views vs Copies](NDArray.md#views-vs-copies--most-important-rule).

---

## Getting values

### Whole sub-arrays - the indexer

Python's slice notation is passed as a string; separate integer arguments are coordinates.

```csharp
var a = np.arange(20).reshape(4, 5);

a[0];              // np.a[0]          first row - reduces a dim → (5,) view
a[-1];             // np.a[-1]         last row
a[1, 2];           // np.a[1, 2]       single element → 0-d NDArray
a["1:3"];          // np.a[1:3]        rows 1-2 → (2, 5) view
a["1:3, :2"];      // np.a[1:3, :2]    rows 1-2, first two cols → (2, 2) view
a["::2"];          // np.a[::2]        every other row
a["::-1"];         // np.a[::-1]       reversed first axis
a["..., -1"];      // np.a[..., -1]    ellipsis + last column
```

Boolean and fancy indexing return copies:

```csharp
var arr = np.array([10, 20, 30, 40, 50]);

var mask = arr > 20;              // NDArray<bool>
arr[mask];                        // arr[mask]  → [30, 40, 50] (copy)

var idx = np.array([0, 2, 4]);
arr[idx];                         // arr[idx]   → [10, 30, 50] (copy)
arr[new[] {0, 2, 4}];             // same - a raw int[] as the SOLE index is fancy
```

> **`a[new[]{0,2}]` is fancy, not coordinate.** A raw `int[]`/`long[]` used as the *whole* index selects rows 0 and 2 (shape `(2, …)`), matching NumPy's `a[[0, 2]]`. Separate integers `a[0, 2]` stay coordinate access. For element/sub-array access by a coordinate array, use `a.GetValue(0, 2)` (see below), not `a[coordArray]`.

### Single elements - five accessors

Reading one element has five entry points, chosen by how many indices you have and whether you want conversion:

```csharp
var a = np.arange(12).reshape(3, 4);   // int64 (NumPy 2.x default integer)

// 1. Indexer - returns a 0-d NDArray (a single element is still an array, like NumPy 2.x)
NDArray elem = a[1, 2];
long v = (long)elem;                    // cast the 0-d out → converts

// 2. .item<T>() - flat index, CONVERTS if T differs from the dtype (NumPy parity)
long   v2 = a.item<long>(6);            // flat index 6 → row 1, col 2
double d2 = a.item<double>(6);          // reads the int64 as a double (converts)
object bx = a.item(6);                  // untyped → boxed long

// 3. GetValue<T> - N-D coordinates, EXACT dtype (T must equal the dtype)
long v3 = a.GetValue<long>(1, 2);

// 4. GetAtIndex<T> - flat index, EXACT dtype, no Shape math (fastest)
long v4 = a.GetAtIndex<long>(6);

// 5. Typed named accessors - GetDouble / GetInt32 / GetSingle / GetBoolean / GetDecimal / GetComplex
//    EXACT dtype: they read the matching typed slice directly and do NOT convert.
var f = np.zeros(4);                    // float64
double x = f.GetDouble(0);              // OK - array IS double
// a.GetDouble(0);  // on an int64 array this REINTERPRETS the raw bytes as a double - a silent,
//                  // wrong value (no throw, no conversion). Use a.item<double>(0) to convert.
```

**Which to use:**

| Accessor | Indices | Converts? | Use when |
|----------|---------|-----------|----------|
| `a[i, j]` (indexer) | N-D | yes (cast the 0-d out) | NumPy-like ergonomics; you don't mind the 0-d detour |
| `a.item<T>(i)` / `a.item(i)` | flat | **yes** | porting NumPy code; reading across dtypes |
| `a.GetValue<T>(...)` | N-D | no (exact) | coordinate access in typed code |
| `a.GetAtIndex<T>(i)` | flat | no (exact) | hot loops - fastest, no Shape math |
| `a.GetDouble(...)` etc. | N-D | no - dtype **must** match | you already know the dtype and want a non-generic call (reinterprets raw bytes if it doesn't) |

> `.item()` with **no arguments** unwraps any size-1 array (0-d, 1-element 1-D, 1×1 2-D) to a scalar, and throws `IncorrectSizeException` otherwise - the NumPy 2.x replacement for the removed `np.asscalar()`.

### Reading flat, in order

```csharp
foreach (var x in a.flat) { }          // boxed elements, C-order (a.flat is a raveled view)
long s = (long)a.flatiter[5];          // flat C-order element (getter returns object → unbox; see below)

// Fast, unboxed - by reference:
double total = 0;
foreach (ref double e in np.nditer<double>(a)) total += e;
```

For high-throughput reads use `np.nditer<T>` (unboxed, by reference) or `np.nditer_chunks<T>` (a `Span<T>` per inner loop). See [NDArray → Fast element iteration](NDArray.md#fast-element-iteration--npnditert-unboxed) and [NDIter](NDIter.md).

---

## Setting values

### 1. Elements

```csharp
var a = np.arange(12).reshape(3, 4);   // int64

a[1, 2] = 99;                          // indexer - CONVERTS the value to the array dtype
a.SetValue(99L, 1, 2);                 // N-D coordinates - EXACT dtype (value must be long here)
a.SetAtIndex(99L, 6);                  // flat index - EXACT dtype
var f = np.zeros((3, 4));              // float64
f.SetDouble(3.5, 1, 2);                // typed setter - EXACT dtype (writes raw double bytes)
```

The indexer converts; `SetValue<T>` / `SetAtIndex<T>` / `SetDouble` (and the other typed setters) write the value's bytes directly and require the value to match the dtype. Mirror of the getter table above.

### 2. Slices - scalar fill and broadcast assignment

Assigning to a slice writes into the underlying array (slices are views). A scalar fills; a shaped right-hand side broadcasts:

```csharp
var a = np.arange(20).reshape(4, 5);

a["1:3"] = 0;                          // a[1:3] = 0        fill rows 1-2 with 0
a[0] = 7;                              // a[0] = 7          fill first row (scalar broadcasts)
a[0] = np.zeros(5);                    // a[0] = np.zeros(5)  assign a whole row
a["1:3, :"] = np.array([1,2,3,4,5]);  // row vector broadcasts down the two rows
a["::2"] = -1;                         // every other row
```

The right-hand side must broadcast to the target's shape (see [Broadcasting](broadcasting.md)); a mismatch throws `IncorrectShapeException`.

### 3. Boolean mask

```csharp
var arr = np.array([10, 20, 30, 40, 50]);
arr[arr > 20] = -1;                        // arr[arr > 20] = -1  → [10, 20, -1, -1, -1]

var mask = arr < 0;                        // [F, F, T, T, T]
arr[mask] = np.array([1, 2, 3]);     // one value per True position → [10, 20, 1, 2, 3]
```

The value count must match the number of `True` positions (or be a single scalar, which fills all of them). Unlike the *read* `arr[mask]` (a copy), the *assignment* `arr[mask] = …` writes through to `arr`.

### 4. Fancy indices

```csharp
var arr = np.array([10, 20, 30, 40, 50]);
arr[np.array([0, 2, 4])] = 0;                          // arr[[0,2,4]] = 0 → [0, 20, 0, 40, 0]
arr[np.array([0, 2, 4])] = np.array([1, 2, 3]);  // one value per index
```

Duplicate indices follow last-write-wins, matching NumPy.

### 5. `ndarray.fill` - set every element in place

```csharp
var a = np.zeros((3, 4));
a.fill(7);                             // a.fill(7)   - every element = 7, IN PLACE, any layout
```

`fill` works across every memory layout (contiguous, F-order, sliced, transposed, negative-stride) and coerces the scalar to the dtype with NumPy's [scalar-assignment rules](#value-coercion-on-assignment-nep50). It returns `void` and mutates the array.

### 6. `ndarray.put` / `np.put` - scatter by flat index

```csharp
var a = np.arange(5);
a.put(np.array([0, 2]), np.array([10, 20]));         // a.put([0,2], [10,20]) → [10,1,20,3,4]
np.put(a, np.array([0, 2]), np.array([10, 20]));
np.put(a, np.array([4]), np.array([99]), "clip");    // out-of-range handled by mode
```

Indices are flat (C-order). `mode` controls out-of-range handling - **`"raise"`** (default; negative indices normalize once, then out-of-range throws), **`"wrap"`**, or **`"clip"`** - matching NumPy. Values shorter than the index list are reused cyclically.

> **Pass the indices/values as `NDArray`.** Because a bare C# `int` converts implicitly to both `long` and `NDArray`, the fully-literal single-index call `np.put(a, 4, 99)` is *ambiguous* and won't compile - wrap them (`np.put(a, np.array([4]), np.array([99]))`), as above.

### 7. `np.place` - fill mask positions from a value list

```csharp
var a = np.arange(6);
np.place(a, a > 2, np.array([100, 200]));   // → [0, 1, 2, 100, 200, 100]
```

`np.place(arr, mask, vals)` writes `vals` into the True positions of `mask`, **cycling** through `vals` (not broadcasting) and truncating an over-long list - `a.flat[mask] = vals` semantics.

### 8. `np.copyto` - the general conditional copy

```csharp
np.copyto(dst, src);                              // np.copyto(dst, src)
np.copyto(dst, src, where: mask);                 // only where mask is True
np.copyto(dst, src, casting: "unsafe");           // relax the cast rule
```

`np.copyto(dst, src, casting = "same_kind", where = null)` copies `src` into `dst` (broadcasting `src` to `dst`'s shape), optionally gated by a boolean `where` mask. `where` must be bool and broadcasts to the output; masked-off positions keep their prior contents. `casting` follows NumPy's rule names (`"no"`, `"equiv"`, `"safe"`, `"same_kind"`, `"unsafe"`).

### 9. `np.fill_diagonal` - set the diagonal in place

```csharp
var m = np.zeros((4, 4));
np.fill_diagonal(m, 5);                            // np.fill_diagonal(m, 5)  - 5 on the diagonal
np.fill_diagonal(m, np.array([1, 2, 3, 4])); // one value per diagonal slot
```

Values **tile cyclically and truncate** (they do not broadcast); `wrap: true` continues the diagonal on tall matrices. Returns `void`, mutates in place, and writes correctly through transposed / sliced / F-order / negative-stride views.

### 10. `ndarray.flatiter` - write through the flat iterator, any layout

```csharp
var t = np.arange(6).reshape(2, 3).T;   // a transposed (non-contiguous) view
t.flatiter[5] = 99;                      // t.flat[5] = 99  - writes through in logical C-order
t.flatiter["::2"] = 0;                   // slice assignment
t.flatiter[new[] {0, 2}] = np.array([7, 8]); // fancy assignment
```

`arr.flatiter` is NumSharp's `flatiter` (the type of NumPy's `arr.flat`). It reads and writes **through to the base** in logical C-order for *every* layout - including transposed, sliced, strided, negative-stride, and broadcast views. Negative flat indices wrap; out-of-range raises `IndexError`.

> **Why `flatiter`, not `flat`?** NumSharp's `arr.flat` returns a *raveled `NDArray`*, which **copies** for a non-contiguous layout - so `arr.T.flat[i] = v` would be silently lost. `arr.flatiter` is the write-through accessor; prefer it whenever you assign by flat index into a view.

---

## Value coercion on assignment (NEP50)

When a scalar is written to an array (`fill`, `flatiter` set, scalar indexer assignment), NumSharp coerces it to the dtype with NumPy 2.x's weak/strong scalar rules:

- **A C# primitive is a *weak* scalar** (NumSharp's analog of a Python literal).
  - Into an **integer** dtype it is **range-checked**: an out-of-range value **raises** `OverflowException` (`"Python integer 300 out of bounds for int8"`) rather than wrapping. A float source truncates toward zero first, then the truncated value is checked; `NaN` raises `ValueError`, `±inf` raises `OverflowException`, a complex source raises `TypeError`.
  - Into a **float/complex** dtype it casts, saturating to `±inf` on overflow (`float32.fill(1e300)` → `inf`).
- **A 0-d `NDArray` is a *strong* scalar** - it **wraps** on cast, matching an `np.int64` scalar. Use `NDArray.Scalar<T>(v)` when you need the strong behavior.
- A **higher-rank array** assigned as a scalar is a sequence → `ValueError("setting an array element with a sequence.")`.

```csharp
var a = np.zeros(3, dtype: np.int8);
a.fill(300);                       // OverflowException - weak int out of range for int8
a.fill(NDArray.Scalar<int>(300)); // OK - strong scalar wraps → stores 44
```

---

## In-place vs. reassignment

C# compound operators (`+=`, `*=`, …) are **not in-place** on an `NDArray` - the compiler expands `a += 1` into `a = a + 1`, producing a new array and rebinding the variable. Other references to the original do not see the change:

```csharp
var x = np.array([1, 2, 3]);
var alias = x;
x += 10;                 // x → NEW array [11, 12, 13]
// alias                 // still [1, 2, 3] - unlike NumPy!
```

To modify the array's memory in place, assign through a slice - the setter path:

```csharp
x[":"] = x + 1;          // in-place: rewrites x's buffer; alias sees it too
a["1:3"] += 5;           // still expands to a["1:3"] = a["1:3"] + 5, but the LHS is a view,
                         // so the result is written back into the parent
```

See [NDArray → Compound assignment](NDArray.md#compound-assignment).

---

## NumPy → NumSharp cheat-sheet

| NumPy writer | Form | NumSharp equivalent |
|---|---|---|
| `arr[0,1] = 5` | `[]` element | `arr[0,1] = 5` (value → 0-d NDArray, converts); or `arr.SetValue(5L,0,1)` (exact) / `arr.SetDouble(5,0,1)` (exact, double array) |
| `arr[1:3] = 0` | `[]` slice | `arr["1:3"] = 0` |
| `arr[mask] = 0` | `[]` bool mask | `arr[mask] = 0` |
| `arr[[0,2]] = [1,2]` | `[]` fancy | `arr[np.array([0,2])] = np.array([1,2])` (or a raw `int[]` index) |
| `arr.fill(5)` | method | `arr.fill(5)` |
| `arr.put([k],[v])` / `np.put` | method / func | `arr.put(...)` / `np.put(...)` |
| `np.place(arr, mask, vals)` | func | `np.place(...)` |
| `np.copyto(dst, src, where=)` | func | `np.copyto(dst, src, where: mask)` |
| `np.fill_diagonal(arr, v)` | func | `np.fill_diagonal(...)` |
| `arr.flat[5] = v` | flat `[]` | `arr.flatiter[5] = v` (write-through, any layout) |
| `np.putmask(arr, mask, v)` | func | **not present** - use `np.place(arr, mask, v)` or `np.copyto(arr, v, where: mask)` |
| `arr.itemset(k, v)` | method | **legacy / removed in NumPy 2.0.** NumSharp keeps an `itemset`, but it diverges: it takes **coordinates** (`itemset(coords, value)`), not a flat index, and does not reliably cast the value to the dtype. Don't use it - write `arr[0,1] = v` or `arr.SetValue(v, 0, 1)` instead. |

Reading has the mirror set - the indexer (`a[i,j]`), `a.item<T>(i)`, `a.GetValue<T>(...)`, `a.GetAtIndex<T>(i)`, the typed named getters, and `a.flatiter[i]`.

---

## Troubleshooting

### "My array changed when I modified a slice!"
Slices are views. `a["1:3"]` shares memory with `a`. Force a copy: `a["1:3"].copy()`.

### "NumSharpException: assignment destination is read-only"
You're writing into a broadcasted view (a stride-0 dimension), which is read-only to prevent cross-row corruption. Copy first: `var w = b.copy(); w["..."] = value;`. See [Broadcasting → Memory Behavior](broadcasting.md#memory-behavior).

### "`GetDouble` returned a nonsense number (or `SetDouble` corrupted the array)"
The typed named accessors (`GetDouble`/`SetDouble`/`GetInt32`/…) require the array to actually be that dtype - they read/write the raw bytes directly and do **not** convert. On a mismatched dtype they silently **reinterpret** the bytes (e.g. `int64` value `6` read via `GetDouble` comes back as `2.96e-323`), and `SetDouble` into a narrower dtype writes 8 bytes at the wrong stride and **corrupts memory**. Use `a.item<double>(i)` to read as double, or the indexer / `astype` to convert.

### "`OverflowException: Python integer … out of bounds`"
A weak C# scalar didn't fit the target integer dtype. That's NumPy 2.x behavior - weak scalars range-check instead of wrapping. Pass a strong scalar to wrap: `a.fill(NDArray.Scalar<int>(v))`, or widen the dtype.

### "`a += 1` didn't update another reference"
C# compound assignment rebinds the variable; it doesn't mutate. Use `a[":"] = a + 1` for in-place. See [In-place vs. reassignment](#in-place-vs-reassignment).

### "`arr.itemset(5, v)` set the wrong element"
NumSharp's legacy `itemset` takes **coordinates** (`itemset(coords, value)`), not NumPy's flat index. Use `arr.flatiter[5] = v` for a flat write, or `arr[i, j] = v` for coordinates.

---

## API Reference

### Getters

| Member | Indices | Converts? | Returns |
|--------|---------|-----------|---------|
| `a[i, j]` / `a["1:3"]` / `a[mask]` / `a[idx]` | mixed | - | `NDArray` (0-d for one element) |
| `a.item(i)` / `a.item<T>(i)` | flat | yes | `object` / `T` |
| `a.item()` / `a.item<T>()` | - | yes | size-1 array → scalar |
| `a.GetValue<T>(...)` / `a.GetValue(...)` | N-D | no / boxed | `T` / `object` |
| `a.GetAtIndex<T>(i)` / `a.GetAtIndex(i)` | flat | no / boxed | `T` / `object` |
| `a.GetDouble/GetInt32/GetSingle/GetBoolean/GetDecimal/GetComplex(...)` | N-D | no - dtype **must** match | the named scalar type (reinterprets on mismatch) |
| `a.flatiter[i]` / `["1:3"]` / `[idx]` | flat | yes (set) | scalar / `NDArray` |
| `a.flat` | - | - | raveled `NDArray` view |

### Setters

| Member | Scope | Notes |
|--------|-------|-------|
| `a[i, j] = v` | element | converts to dtype (NEP50 coercion) |
| `a["1:3"] = v` / `a[0] = row` | slice / row | broadcasts RHS; writes through the view |
| `a[mask] = v` | masked | writes through to the parent |
| `a[idx] = v` | fancy | last-write-wins on duplicates |
| `a.SetValue<T>(v, ...)` / `a.SetValue(v, ...)` | element (N-D) | exact dtype |
| `a.SetAtIndex<T>(v, i)` | element (flat) | exact dtype |
| `a.SetDouble(v, ...)` | element (N-D) | exact dtype (double array) |
| `a.fill(v)` | whole array | in place, any layout, NEP50 coercion |
| `a.put(indices, values, mode)` / `np.put(...)` | flat scatter | `mode`: `"raise"`/`"wrap"`/`"clip"` |
| `np.place(arr, mask, vals)` | masked | cyclic fill of the True positions |
| `np.copyto(dst, src, casting, where)` | conditional copy | `where` bool mask; `casting` rule |
| `np.fill_diagonal(a, val, wrap)` | diagonal | in place, cyclic values |
| `a.flatiter[i] = v` | flat | write-through, any layout |

---

See also: [NDArray](NDArray.md), [Dtypes](dtypes.md), [Broadcasting](broadcasting.md), [NDIter](NDIter.md), [Exceptions](exceptions.md).
