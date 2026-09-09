# Iterating & Enumerating

There are many ways to walk an `NDArray` element by element, row by row, or operand by operand — from a plain `foreach` to NumPy's full `nditer`. NumSharp mirrors NumPy's whole iteration surface and adds two typed, unboxed extensions C# can express that Python can't. This page maps every technique: what it yields, in what order, whether it boxes, and whether writes go through.

Two rules run through all of it, and they trip people up:

1. **Order is not uniform.** Most walks follow **logical C-order** (last axis fastest, honoring the array's strides) — `foreach`, `.flat`, `.flatiter`, `np.ndindex`, `np.ndenumerate`, `np.broadcast`. But **`np.nditer` and the typed `np.nditer<T>` default to memory order (`'K'`)**, matching NumPy. On a transposed or reversed view those two orders differ. See [Iteration order](#iteration-order-logical-c-order-vs-memory-k-order).
2. **Prefer the unboxed forms — every value walk has one.** A boxed `object` or a per-element `NDArray` view dominates a hot loop, so the examples on this page use the **typed, by-`ref T`** walks throughout: `np.flat<T>` (C-order), `np.nditer<T>` (memory order), `np.nditer_chunks<T>` (a `Span<T>` per inner loop), and `np.ndenumerate<T>().AsRef()` (index + `ref T`). Index-only `np.ndindex` has a non-allocating `.AsSpans()`. The boxed spellings (`a.flat`, `a.flatiter`, `np.ndenumerate(a)`, the `np.nditer` object, `np.broadcast`) remain for NumPy parity and LINQ, but you should not reach for them in a loop that runs — or better, don't loop at all (see [Don't loop when you can vectorize](#dont-loop-when-you-can-vectorize)).

This page is the user-facing guide. For the iterator *engine* — coalescing, buffering, casting, the kernel tiers, `NDIterRef` — see [NDIter](NDIter.md).

---

## Which one should I use?

The recommended, non-boxed forms first:

| You want | Use | Yields | Order | Writes through? |
|----------|-----|--------|-------|-----------------|
| Every element, unboxed, by ref, C-order | `foreach (ref T x in np.flat<T>(a))` | `ref T` | C-order | yes (`writeable: true`) |
| Every element, unboxed, by ref, memory order (fastest) | `foreach (ref T x in np.nditer<T>(a))` | `ref T` | memory (`'K'`) | yes (`writeable: true`) |
| A `Span<T>` per inner loop (to vectorize) | `foreach (Span<T> c in np.nditer_chunks<T>(a))` | `Span<T>` | memory (`'K'`) | yes (`writeable: true`) |
| `(index, ref value)` pairs, unboxed | `foreach (var e in np.ndenumerate<T>(a).AsRef())` | `ReadOnlySpan<long>` + `ref T` | C-order | yes (`writeable: true`) |
| The index space of a shape, non-allocating | `foreach (var idx in np.ndindex(3, 2).AsSpans())` | `ReadOnlySpan<long>` | C-order | — (no data) |
| Rows / sub-arrays along axis 0 | `foreach (NDArray row in a)` (N-D) | `NDArray` view | C-order | yes (views) |

Boxed spellings, kept for NumPy parity and LINQ — **not for hot loops**:

| API | Yields | Non-boxed equivalent |
|-----|--------|----------------------|
| `a.flat` (raveled `NDArray`) / `a.flatiter` | boxed `object` per element | `np.flat<T>(a)` / `a.flatiter.AsTyped<T>()` |
| `np.ndenumerate(a)` | `(long[], object)` | `np.ndenumerate<T>(a).AsRef()` |
| `np.ndindex(3, 2)` | fresh `long[]` per step | `np.ndindex(3, 2).AsSpans()` |
| `np.nditer(a, flags: …)` (the full flag object) | `NDArray[]` (live views) | `np.nditer<T>` / `np.nditer_chunks<T>` for value walks; the object only when you need flags/multi-operand |
| `np.nested_iters(a, axes)` | `NDIterator[]` per level | — (advanced/parity; no typed form) |
| `np.broadcast(a, b, …)` | `object[]` / `.iters[i]` | `np.nditer<T>` over a `broadcast_to` view |

`NDArray` is `IEnumerable`, so LINQ works (`a.Cast<double>().Sum()`) — but it boxes; use the typed walks above when it matters.

---

## `foreach` — iterate rows (sub-arrays) along axis 0

`NDArray` implements `IEnumerable`, and — like NumPy — iterating an **N-D** array walks the **first axis**, handing you each sub-array as a **view** (not a boxed scalar — this is a non-boxed, write-through walk):

```csharp
var m = np.arange(6).reshape(2, 3);
foreach (NDArray row in m)       // each `row` is an (N-1)-D VIEW of m — writes mutate m
    row[":"] = row * 2;          // [0 2 4] then [6 8 10]
```

- **N-D array** → each step is an `(N-1)`-D **view** along axis 0 (writing into it mutates the parent).
- **0-D array** → `foreach` throws `TypeError("iteration over a 0-d array")`, exactly as NumPy raises.
- **Empty array** → zero iterations.

> A **1-D** `foreach` yields **boxed scalars** — don't use it. For a 1-D element walk use the unboxed `foreach (ref int x in np.nditer<int>(v))` (or `np.flat<int>(v)`), which hands out a `ref int` with no boxing.

---

## Flat element iteration

### `np.flat<T>(a)` / `a.flatiter.AsTyped<T>()` — unboxed, by reference, C-order

The way to walk flat elements: a **`ref T`** in logical C-order — no boxing, no per-element view — writing through for *every* layout (transposed, sliced, strided, negative-stride, broadcast):

```csharp
var a = np.arange(6).reshape(2, 3).T;      // transposed (non-contiguous) view

// read by reference, C-order (values 0 3 1 4 2 5)
long total = 0;
foreach (ref long x in np.flat<long>(a))
    total += x;

// write through, C-order — the write a.flat cannot do on a view
foreach (ref long x in np.flat<long>(a, writeable: true))
    x *= 2;

// equivalently, off an existing flatiter:
foreach (ref long x in a.flatiter.AsTyped<long>(writeable: true))
    x *= 2;
```

- **`T` must be the array's exact dtype** — a `ref` cannot convert, so a mismatch throws (`astype` first). This is why the typed form can't do the NumPy scalar coercion the boxed setters do.
- **`np.flat<T>(a)` is `np.nditer<T>(a, order: 'C')` with the C-order default baked in** — same `NDIterRef` engine, same `ref T`, but defaulting to logical C-order to match `a.flatiter` (the order-configurable `np.nditer<T>` defaults to memory `'K'` order instead — see [Iteration order](#iteration-order-logical-c-order-vs-memory-k-order)).
- **The by-reference spelling `a.flatiter<T>` is impossible in C#** — a generic method can't share a name with the `flatiter` property — so the instance form is `a.flatiter.AsTyped<T>()`, and the static form `np.flat<T>(a)` (the typed overload of `np.flat(a)`).
- **0-d yields its one element; an empty array iterates zero times.**
- **Re-enumeration restarts** (the value holds no cursor). A read-only broadcast view refuses `writeable: true` with NumPy's message.

### `a.flat` / `a.flatiter` — the boxed forms (NumPy parity; indexers, cursor)

`a.flat` is a raveled **`NDArray`** and `a.flatiter` is a `FlatIterator` (NumSharp's analog of NumPy's `flatiter`). **Iterating either boxes** — use `np.flat<T>` above for that. What they offer beyond the typed walk is the **indexer / slice / fancy write-through** surface and the NumPy `flatiter` cursor:

```csharp
var t = np.arange(6).reshape(2, 3).T;   // transposed (non-contiguous) view

t.flatiter[5] = 99;                      // write-through single element, C-order, any layout
t.flatiter["::2"] = 0;                   // slice assignment
t.flatiter[new[] {0, 2}] = np.array(new[] {7, 8}); // fancy assignment

long i    = t.flatiter.index;            // cursor: flat C-order position
long[] c  = t.flatiter.coords;           // cursor: multi-index
NDArray b = t.flatiter.Base;             // the array being iterated
NDArray f = t.flatiter.copy();           // a fresh 1-D C-order copy
```

- **`a.flat` copies when non-contiguous** (it reshapes), so `a.T.flat[i] = v` is silently lost — that's exactly the defect `a.flatiter` (and `np.flat<T>`) fixes.
- The `flatiter[i]`/slice/fancy **setters** do NumPy's NEP50 scalar coercion (a weak out-of-range value raises rather than wrapping — see [Getting & Setting Values → flatiter](getting-and-setting-values.md#10-ndarrayflatiter--write-through-the-flat-iterator-any-layout)); the typed `ref` walk can't coerce (a `ref` can't convert), so use these setters when you need the coercion and `np.flat<T>` when you need speed.
- A *captured* `flatiter` is its own iterator (NumPy's `iter(f) is f`): its cursor is shared with `index`/`coords`, so `next()`/a second pass **resume**. `a.flatiter` builds a fresh one per access.

---

## Typed iteration — `np.nditer<T>` / `np.nditer_chunks<T>` (the fast path)

A NumSharp extension with no NumPy counterpart: Python has no unboxed generics, so NumPy's `nditer` must hand back an array object per element. NumSharp's can hand back a **reference** or a **span** — no boxing, no per-element view.

```csharp
var a = np.arange(6).astype(np.float64);   // T must be the EXACT dtype

// element-wise, by reference — reads and writes go straight to the array
double total = 0;
foreach (ref double x in np.nditer<double>(a))
    total += x;

foreach (ref double x in np.nditer<double>(a, writeable: true))
    x *= 2;                                 // writes through

// chunk-wise — one Span per inner loop, ready to vectorize
foreach (Span<double> chunk in np.nditer_chunks<double>(a, writeable: true))
    System.Numerics.Tensors.TensorPrimitives.Multiply(chunk, 2.0, chunk);
```

A **contiguous array arrives as a single chunk** — and so do F-contiguous, transposed and reversed views, which the iterator coalesces. Both work on every layout and all 15 dtypes.

**Rules of the road** (all matching NumPy's `nditer` where a counterpart exists):

- **The dtype must match exactly.** A `ref` cannot convert, so `np.nditer<double>` over an `int32` array **throws** (it would otherwise reinterpret bytes). Cast first with `astype`.
- **Order is `'K'` (memory order), not logical C-order** — see [Iteration order](#iteration-order-logical-c-order-vs-memory-k-order). Pass `order: 'C'` for the order `np.ndenumerate` uses.
- **`writeable: true`** opens the operand `readwrite` (and rejects a read-only broadcast view with NumPy's message). It is a caller contract, not a C# read-only guarantee — assigning through the `ref`/`Span` always writes.
- **`nditer_chunks<T>` needs a unit-stride inner loop** (a `Span<T>` is contiguous by definition). A stepped view like `a[":, ::2"]` is rejected up front with `NotSupportedException`; use `np.nditer<T>` (any stride) or a `.copy()`.
- **An empty array iterates zero times** (the boxed `np.nditer` instead requires the `zerosize_ok` flag — a deliberate difference so you don't guard every `foreach` with `if (a.size > 0)`).
- **Re-enumeration restarts.** The value returned holds no state; each `foreach` builds a fresh iterator — deliberately unlike the class-based `np.nditer`, which resumes.
- **Disposal is automatic under `foreach`.** These are `ref struct` enumerators; if you drive `MoveNext()` by hand, dispose it yourself. Being `ref struct`s, they cannot escape to a field, lambda, or `async` frame.

These are the fastest per-element and per-chunk walks by a wide margin, because they remove the per-element `NDArray` view and boxing. For current numbers see the [NDIter benchmark report](NDIter.md#measured-behavior).

---

## .NET interop — `Span<T>` / `Memory<T>` / arrays / `IEnumerable<T>`

Two ways to hand an `NDArray` to plain .NET code — a **zero-copy aliasing** side (`nd.Unsafe`) and a **safe copying / enumerable** side (extensions). Both cover all 15 dtypes.

### `nd.Unsafe` — zero-copy BCL views (aliasing; you must keep the NDArray alive)

`nd.Unsafe` exposes the array's unmanaged buffer as built-in .NET types **without copying**. They are called *unsafe* for one specific reason: **none of them takes a reference on the buffer**, so if the `NDArray` is disposed or garbage-collected while a view is still in use, the view dangles and corrupts memory. **Keep the NDArray alive (side by side) for the whole lifetime of the view.**

```csharp
var a = np.arange(6).astype(np.float64);       // keep `a` in scope while using any view below

Span<double>          s   = a.Unsafe.Span<double>();          // aliases; writes go through
ReadOnlySpan<double>  ro  = a.Unsafe.ReadOnlySpan<double>();
Memory<double>        m   = a.Unsafe.Memory<double>();        // manager-backed; still does NOT root `a`
ReadOnlyMemory<double> rm = a.Unsafe.ReadOnlyMemory<double>();
Span<byte>            raw = a.Unsafe.Bytes();                 // the raw bytes (size * itemsize)
unsafe { double* p = a.Unsafe.Pointer<double>(); }           // raw pointer to logical element 0

s[2] = 99;                        // a[2] is now 99 — the Span IS the array's memory
TensorPrimitives.Multiply(s, 2.0, s);   // feed .NET vector APIs directly, no copy
```

- **Alias, not copy** — writing through `Span`/`Memory`/`Pointer` mutates the array.
- **C-contiguous, exact dtype, ≤ `int.MaxValue` elements** for `Span`/`ReadOnlySpan`/`Memory`/`ReadOnlyMemory`/`Bytes`. An **offset slice** (`a[2:5]`) qualifies — it is C-contiguous, and the view aliases exactly that logical window. A transposed / F-contiguous / strided / broadcast view has no single contiguous region, so these **throw** `InvalidOperationException` (copy first with `np.ascontiguousarray(a)` / `a.copy()`, or walk it with `np.nditer<T>`). A wrong `T` **throws** `ArgumentException`.
- **`Pointer<T>()` works for any layout** — it returns the address of logical element 0; for a non-contiguous array you walk the rest yourself using `nd.strides` (byte strides). It does not root the buffer either.
- **Non-throwing forms:** `bool TryGetSpan<T>(out Span<T>)` / `TryGetMemory<T>(out Memory<T>)` return `false` (instead of throwing) when the array is not C-contiguous, the dtype doesn't match, or it has more than `int.MaxValue` elements.
- **Over 2 GB:** `Span<T>`/`Memory<T>` are 32-bit-length; for a larger array use `nd.Unsafe.AsSpan<T>()` (a long-indexable `UnmanagedSpan<T>`) or `Pointer<T>()`.

### Safe bridges — copy / enumerate (result outlives the NDArray)

Unlike the aliasing views, these **materialize or copy** into caller-owned .NET storage, so the result is safe once you have it. Elements are unboxed.

```csharp
double[] all      = a.ToArray<double>();                  // C-order copy (any layout) — already on NDArray
double   mean     = a.AsEnumerable<double>().Average();   // LINQ / IEnumerable<T>, unboxed, any layout
var      evens    = a.AsEnumerable<int>().Where(x => x % 2 == 0).ToList();

double[] fromIter = np.flat<double>(a).ToArray();         // materialize any iterator (C-order here)
int      copied   = np.nditer<double>(a).CopyTo(dest);    // pour an iterator into your own Span<T>/array
```

- **`nd.AsEnumerable<T>()`** walks logical C-order for **every layout** (reads through strides), so it needs no contiguity and never copies the whole array — only the enumerator is a heap object (one allocation, no per-element boxing). Reach for it when you need LINQ or an `IEnumerable<T>`-shaped API; for the fastest walk prefer `foreach (ref T x in np.flat<T>(a))`.
- **`.ToArray()` / `.CopyTo(Span<T>)`** on the typed iterators (`np.flat<T>` → C-order, `np.nditer<T>` → memory order, `np.nditer_chunks<T>`) turn an in-progress walk into a `T[]` or fill a caller buffer. `CopyTo` returns the element count and throws if the destination is too short.
- The **boxed** iteration objects (`a.flatiter`, `np.ndenumerate(a)`, `np.ndindex(...)`) already implement `IEnumerable`, so LINQ (`.ToArray()`, `.Select(...)`) works on them directly — boxed, for parity/convenience.

---

## Index and `(index, value)` enumeration

### `np.ndindex(...)` — the index space of a shape

A pure C-order odometer over an index space — no operands, no data, just the coordinates. It walks an index *space*, not an array, so there are no element values and therefore **no `ref T` form** (unlike `np.flat<T>`); its non-boxed variant is `.AsSpans()`, which yields the multi-index as a **`ReadOnlySpan<long>` over a reused buffer** — an entire walk allocates nothing:

```csharp
foreach (var idx in np.ndindex(3, 2).AsSpans())   // idx is ReadOnlySpan<long>: [0,0] [0,1] [1,0] …
    Console.WriteLine($"{idx[0]},{idx[1]}");       // idx valid until the next step — copy to keep it

foreach (var idx in np.ndindex(a.shape).AsSpans()) { … }  // both spellings bind (ints, or a shape array)
```

A zero-length dimension yields nothing; `np.ndindex().AsSpans()` yields exactly one empty index (the 0-d space). A negative dimension throws `ArgumentException` at construction. `.AsSpans()` **restarts** each `foreach`.

> The bare `np.ndindex(3, 2)` (without `.AsSpans()`) yields a fresh `long[]` per step and is its own iterator (a second `foreach` resumes) — kept for NumPy parity and LINQ (`np.ndindex(...).ToArray()`); prefer `.AsSpans()` in a loop.

### `np.ndenumerate<T>(a).AsRef()` — `(index, ref value)` pairs, unboxed

Walks an array in **logical C-order**, handing you the coordinate and the value **by reference** — no boxing, no per-step allocation, write-through:

```csharp
var a = np.array(new[,] {{1, 2}, {3, 4}});
foreach (var e in np.ndenumerate<int>(a).AsRef(writeable: true))
{
    // e.Index is a ReadOnlySpan<long> (reused buffer); e.Value is a ref int
    e.Value += (int)e.Index[0];      // writes through to a
}
```

- Order is always C-order regardless of layout (an F-contiguous or reversed view reads through its strides); a 0-d array yields one pair with an empty index; an empty array yields nothing.
- `e.Index` is a span over a buffer **reused each step** — copy it (`e.Index.ToArray()`) to keep it past the current iteration, exactly as with a `Span<T>` chunk.
- `T` must be the exact dtype; `writeable: true` opens the array read-write (a read-only broadcast view is refused); re-enumeration restarts.

> The boxed `np.ndenumerate(a)` (→ `(long[], object)`) and the by-value `np.ndenumerate<T>(a)` (→ `(long[], T)`, T unboxed but copied, fresh `long[]` per step) remain for NumPy parity and LINQ (`.Select(...)`, `.ToArray()`); prefer `.AsRef()` in a loop.

---

## The full `np.nditer` object (advanced / NumPy parity — boxed)

`np.nditer(...)` returns an `NDIterator` — the managed face of NumSharp's iterator engine and a port of NumPy's `numpy.nditer`. It yields **`NDArray[]`** (a live view per operand), so it **boxes** and allocates a per-element view. Reach for it **only** when you need NumPy's flag surface — `multi_index`, multiple operands, output allocation — that the typed walks can't express.

> **For a plain value or element walk, do NOT use this — use `np.nditer<T>` / `np.nditer_chunks<T>` / `np.flat<T>`** (the typed section above). The single-operand `foreach (var vals in it) … vals[0] …` loop is the slowest walk on this page (an `NDArray` view per element); it exists for parity, not for use.

```csharp
var a = np.arange(6).reshape(2, 3);

// the reason to use the object: track the multi-index (a flag the typed walks don't carry)
using (var it = np.nditer(a, flags: new[] {"multi_index"}))
    while (!it.finished)
    {
        Console.WriteLine($"{string.Join(",", it.multi_index)} = {(long)it[0]}");  // it[0] is a boxed view
        it.iternext();
    }
```

> **You must dispose it.** The iterator owns unmanaged state; `using` (or `it.close()`) frees it and flushes any buffered/overlap write-backs — exactly like NumPy's `with np.nditer(...) as it:`. A finalizer is only a safety net.

**Key contract** (full details in [NDIter → Managed `np.nditer` contract](NDIter.md#managed-npnditer-contract)):

- **Enumeration yields `NDArray[]`**, always — `vals[0]` for one operand (NumPy yields a bare 0-d array there; C# has no such union). The values are **live alias views** onto the iterator's data pointer: they change on the next step and are invalid after disposal. **Copy anything you need to keep.**
- **It is its own iterator** (`iter(x) is x`): a second `foreach` **resumes**; call `it.reset()` to rewind.
- Order defaults to `'K'` (memory) — see below.

### External loop, buffering, and modifying in place

```csharp
var a = np.arange(6).reshape(2, 3);

// external_loop: the iterator hands you the largest contiguous chunk (a 1-D view) per step
// (boxed; for an unboxed chunk walk use np.nditer_chunks<T>, which hands out a Span<T>)
using (var it = np.nditer(a, flags: new[] {"external_loop"}))
    foreach (var vals in it) Console.WriteLine(vals[0]);   // one chunk: [0 1 2 3 4 5]

// readwrite: mutate the operand in place
var c = np.arange(3);
using (var it = np.nditer(c, op_flags: new[] {"readwrite"}))
    foreach (var vals in it)
        vals[0].SetAtIndex(Convert.ToInt64(vals[0].GetAtIndex(0)) * 2, 0);
// c is now [0 2 4]
```

### Multiple operands and allocated outputs

```csharp
// broadcast two operands together, like np.nditer([a, b])
using (var it = np.nditer(new[] {np.array(new[] {1, 2, 3}), np.array(new[,] {{10}, {20}})}))
    foreach (var vals in it)
        Console.Write($"{(long)vals[0]}/{(long)vals[1]} ");   // 1/10 2/10 3/10 1/20 2/20 3/20

// a null operand is an output slot the iterator ALLOCATES (NumPy's None)
using (var it = np.nditer(
    new[] {np.arange(6), null},
    op_flags: new[] {new[] {"readonly"}, new[] {"writeonly", "allocate"}},
    op_dtypes: new DType[] {np.int64, np.int64}))
{
    foreach (var vals in it)
        vals[1].SetAtIndex(Convert.ToInt64(vals[0].GetAtIndex(0)) * 2, 0);
    NDArray doubled = it.operands[1];   // [0 2 4 6 8 10]
}
```

The full flag vocabulary (`c_index`/`f_index`, `common_dtype`, `reduce_ok`, `delay_bufalloc`, per-op `contig`/`no_broadcast`/`writemasked`, …) and the property/method surface (`shape`, `itersize`, `index`, `value`, `itviews`, `copy()`, `remove_axis()`, `enable_external_loop()`, …) are documented in [NDIter](NDIter.md#managed-npnditer-contract). The engine behind it — coalescing, buffering, casting, masked writes, reductions — is the rest of that page.

---

## Nested loops — `np.nested_iters` (advanced / NumPy parity — boxed)

`np.nested_iters(op, axes)` returns one `NDIterator` per entry in `axes` (outermost first), all iterating the **same** buffer over disjoint axis groups. Advancing an outer level re-bases its inner levels, so plain nested `foreach`es walk the array in nested loops. It shares `np.nditer`'s boxed `NDArray[]`/`it[0]` surface (there is no typed form) — use it only when you genuinely need nested disjoint-axis loops with `multi_index`:

```csharp
var a = np.arange(12).reshape(2, 3, 2);
var levels = np.nested_iters(a, new[] { new[] {1}, new[] {0, 2} }, flags: new[] {"multi_index"});
using var outer = levels[0];
using var inner = levels[1];

foreach (var _ in outer)                     // walks axis 1
    foreach (var __ in inner)                // walks axes 0 and 2, re-based to the outer position
        Console.WriteLine($"{string.Join(",", outer.multi_index)} | " +
                          $"{string.Join(",", inner.multi_index)} = {(long)inner[0]}");
```

`axes` needs at least 2 entries and no axis may repeat across entries (verbatim NumPy `ValueError`s otherwise). **Dispose every returned level.** The `multi_index` path is bit-exact with NumPy; without `multi_index` the pure value-stream *order* can differ (a documented `op_axes` limitation — track `multi_index`, or prefer explicit `order: 'C'`/`'F'`, when order is part of your contract).

---

## Broadcast iteration — `np.broadcast` (NumPy parity — boxed)

`np.broadcast(a, b, …)` is NumPy's `numpy.broadcast`: it resolves the broadcast shape without materializing data and lets you iterate the operands together. Its per-operand streams and value-tuples **box**; for an unboxed walk, `broadcast_to` each operand to `bc.shape` and drive it with `np.nditer<T>` (order `'C'`). Use `np.broadcast` for the metadata (`shape`/`size`/`numiter`) and for parity.

```csharp
var a = np.array(new long[] {1, 2, 3});          // (3,)
var b = np.array(new long[,] {{10}, {20}});       // (2, 1)
var bc = np.broadcast(a, b);

bc.shape;      // (2, 3)
bc.numiter;    // 2   (operand count)
bc.size;       // 6

// per-operand flat streams, each stretched to the result shape (C-order)
bc.iters[0];   // yields 1, 2, 3, 1, 2, 3
bc.iters[1];   // yields 10, 10, 10, 20, 20, 20

// iterate the broadcast itself: one value-tuple per element, with a live cursor
foreach (object[] vals in bc)
    Console.Write($"{vals[0]}+{vals[1]} ");   // 1+10 2+10 3+10 1+20 2+20 3+20
bc.index;      // 6 (== size, exhausted)
bc.reset();    // rewind the cursor to iterate again
```

Like NumPy, the object is its own iterator (`iter(b) is b`) with a single `.index` cursor and a `.reset()`. NumSharp accepts **any number of operands** (NumPy caps at 64), and the `.iters` are re-enumerable. See also [Broadcasting → `np.broadcast`](broadcasting.md#npbroadcastarray1-array2-).

---

## Iteration order: logical C-order vs memory (`'K'`-order)

This is the single most common surprise. Take `arange(6).reshape(2,3)` and a reversed view `a[:, ::-1]`:

| Technique | Default order | `a[:, ::-1]` yields |
|-----------|---------------|---------------------|
| `foreach`, `.flat`, `.flatiter`, `np.flat<T>`, `np.ndenumerate` (+`.AsRef()`), `np.ndindex` (+`.AsSpans()`), `np.broadcast` | **logical C-order** | `2 1 0 5 4 3` |
| `np.nditer`, `np.nditer<T>`, `np.nditer_chunks<T>` | **memory order (`'K'`)** | `0 1 2 3 4 5` |

Memory order is what lets the iterator coalesce reversed / F-contiguous / transposed views into a single fast chunk, and it matches NumPy's `nditer` exactly. When you need the two to agree, pass **`order: 'C'`** to the `nditer` family:

```csharp
var a = np.arange(6).reshape(2, 3);   // arange is int64
var rev = a[":, ::-1"];

foreach (ref long x in np.nditer<long>(rev)) { … }             // 0 1 2 3 4 5 (memory)
foreach (ref long x in np.nditer<long>(rev, order: 'C')) { … } // 2 1 0 5 4 3 (logical)
```

Conversely, `np.ndenumerate` always gives you logical C-order — use it (or `order: 'C'`) when the coordinates must line up with the values.

---

## Writing through while iterating

| Technique | Mutating the yielded item touches the array? |
|-----------|----------------------------------------------|
| `foreach` over N-D | **yes** — each row is a view |
| `foreach` over 1-D | no — boxed scalar copies |
| `a.flat` | **no** — copies if non-contiguous (writes lost) |
| `a.flatiter` | **yes** — write-through, any layout |
| `np.flat<T>(a, writeable: true)` / `a.flatiter.AsTyped<T>(true)` | **yes** — by `ref T`, any layout |
| `np.nditer<T>(a, writeable: true)` | **yes** |
| `np.ndenumerate<T>(a).AsRef(writeable: true)` | **yes** — by `ref T` (`e.Value`), any layout |
| `np.nditer(a, op_flags: ["readwrite"])` | **yes** — mutate `vals[0]` via `SetAtIndex` |
| `np.ndenumerate` (boxed/by-value), `np.ndindex` (+`.AsSpans()`), `np.broadcast` | no — read-only walks |

A read-only **broadcast** view (stride-0) refuses write-enabled iteration: `np.nditer<T>(bc, writeable: true)` and `np.nditer(bc, op_flags: ["readwrite"])` throw the NumPy read-only message. Copy first if you must write.

---

## Don't loop when you can vectorize

The fastest loop is the one you don't write. Before reaching for any iterator, check whether a vectorized op, a reduction, or a fused expression already does it — these run SIMD kernels over the whole array with no per-element boxing:

```csharp
// instead of: foreach (ref double x in np.nditer<double>(a, writeable: true)) x = x * 2 + 1;
var r = a * 2 + 1;                    // vectorized
np.evaluate((NDExpr)a * 2 + 1, @out: a);   // fused: one pass, no temporaries (NumSharp extension)

// instead of accumulating in a foreach:
double s = (double)np.sum(a);         // reduction kernel (np.sum returns a 0-d NDArray → cast out)
```

When you *do* need to walk elements yourself, `np.nditer<T>` / `np.nditer_chunks<T>` are the fast choices; the boxed `np.nditer` `it[0]` per-element loop is the slowest (it allocates an `NDArray` view per element). See [NDIter → Fused kernels](NDIter.md#fused-kernels-in-production) and [`np.evaluate`](NDIter.md#general-expression-fusion--npevaluate).

---

## Cursor & re-enumeration semantics

Whether a second pass restarts or resumes depends on the technique — NumSharp matches NumPy's `iter(x) is x` rule:

| Technique | Second `foreach` |
|-----------|------------------|
| `np.nditer`, `np.nested_iters` levels | **resumes** (own iterator); `reset()` to rewind |
| `a.flatiter` captured in a variable (`var f = a.flatiter`) | **resumes** (shares the `index`/`coords` cursor); each fresh `a.flatiter` access is a new iterator |
| `np.ndindex`, `np.ndenumerate` (boxed/by-value), `np.broadcast` | **resumes** (own iterator); `broadcast.reset()` rewinds |
| `np.flat<T>`, `a.flatiter.AsTyped<T>()`, `np.nditer<T>`, `np.nditer_chunks<T>`, `np.ndenumerate<T>().AsRef()`, `np.ndindex().AsSpans()` | **restarts** (fresh iterator each `foreach`) |
| `foreach` over an `NDArray`, `a.flat` | **restarts** (fresh enumerator each `foreach`) |

---

## Troubleshooting

### "`TypeError: iteration over a 0-d array`"
`foreach` needs at least one axis. A 0-d array is a single value — read it with `(T)a` or `a.GetValue<T>()`.

### "My writes through `.flat` disappeared"
`a.flat` copies for a non-contiguous array. Use `a.flatiter` (write-through) for flat assignment into a view.

### "`np.nditer<double>` threw on my int array"
Typed iteration can't convert (`ref` can't convert). `astype` first: `np.nditer<double>(a.astype(np.float64))`.

### "The values from `np.nditer` are wrong after the loop / changed unexpectedly"
`it[i]` / `vals[i]` are **live views** onto the iterator; they move each step and die on dispose. Copy what you keep: `var kept = vals[0].copy();`.

### "`np.nditer_chunks<T>` threw `NotSupportedException`"
A `Span<T>` can't describe a stepped inner loop (`a[":, ::2"]`). Use `np.nditer<T>` (any stride) or iterate `a.copy()`.

### "My transposed view iterated in the 'wrong' order"
`np.nditer` defaults to memory order. Pass `order: 'C'` for logical C-order — see [Iteration order](#iteration-order-logical-c-order-vs-memory-k-order).

### "`np.nditer` leaked / an in-place write was lost"
Dispose it (`using` or `close()`). Buffered and overlap write-backs flush on dispose; skip it and the last window can be dropped.

---

## API reference

### Techniques

| API | Returns | Element | Order | Boxes? |
|-----|---------|---------|-------|--------|
| `foreach (var x in a)` | — | `NDArray` (N-D) / scalar (1-D) | C | yes |
| `a.flat` | `NDArray` (raveled view) | scalar | C | yes |
| `a.flatiter` / `np.flat(a)` | `FlatIterator` | scalar | C | yes |
| `np.flat<T>(a, writeable)` / `a.flatiter.AsTyped<T>(writeable)` | `FlatRefIter<T>` | `ref T` | C | **no** |
| `np.nditer<T>(a, writeable, order)` | `NDRefIter<T>` | `ref T` | K | **no** |
| `np.nditer_chunks<T>(a, writeable, order)` | `NDChunkIter<T>` | `Span<T>` | K | **no** |
| `np.ndindex(shape…)` | `NDIndex` | `long[]` | C | — |
| `np.ndindex(shape…).AsSpans()` | `NDIndexSpans` | `ReadOnlySpan<long>` (reused) | C | **no** |
| `np.ndenumerate(a)` / `<T>` | `NDEnumerate` / `<T>` | `(long[], object)` / `(long[], T)` | C | yes / no |
| `np.ndenumerate<T>(a).AsRef(writeable)` | `NDEnumerateRef<T>` | `ReadOnlySpan<long>` + `ref T` | C | **no** |
| `np.nditer(a, …)` | `NDIterator` | `NDArray[]` | K | yes |
| `np.nested_iters(a, axes, …)` | `NDIterator[]` | `NDArray[]` per level | per level | yes |
| `np.broadcast(a, b, …)` | `Broadcast` | `object[]` / `.iters[i]` | C | yes |

### `.NET` interop

| API | Returns | Copy or alias? | Notes |
|-----|---------|----------------|-------|
| `nd.Unsafe.Span<T>()` / `ReadOnlySpan<T>()` | `Span<T>` / `ReadOnlySpan<T>` | **alias** (doesn't root) | C-contiguous, exact dtype, ≤`int.MaxValue` |
| `nd.Unsafe.Memory<T>()` / `ReadOnlyMemory<T>()` | `Memory<T>` / `ReadOnlyMemory<T>` | **alias** (doesn't root) | manager-backed; same constraints |
| `nd.Unsafe.Bytes()` / `ReadOnlyBytes()` | `Span<byte>` / `ReadOnlySpan<byte>` | **alias** | raw `size*itemsize` bytes; C-contiguous |
| `nd.Unsafe.Pointer<T>()` | `T*` | **alias** | logical element 0; **any layout** |
| `nd.Unsafe.TryGetSpan<T>(out)` / `TryGetMemory<T>(out)` | `bool` | **alias** | non-throwing; `false` if not spannable |
| `nd.Unsafe.AsSpan<T>()` | `UnmanagedSpan<T>` | **alias** | long-indexable (>2 GB); NumSharp type |
| `nd.ToArray<T>()` | `T[]` | **copy** | C-order, any layout |
| `nd.AsEnumerable<T>()` | `IEnumerable<T>` | walk (no full copy) | LINQ, C-order, any layout, unboxed |
| `np.flat<T>(a).ToArray()` / `np.nditer<T>(a).ToArray()` / `np.nditer_chunks<T>(a).ToArray()` | `T[]` | **copy** | C / memory / memory order |
| `(iter).CopyTo(Span<T>)` | `int` | **copy** | fills caller buffer; returns count |

### `FlatIterator` (a.flatiter)

| Member | Description |
|--------|-------------|
| `this[long i]` get/set | Scalar at flat C-order index; negative wraps; write-through |
| `this[string]` / `[int[]]` / `[long[]]` / `[NDArray]` get/set | Slice / fancy access; write-through |
| `index` / `coords` | Cursor: flat position / multi-index |
| `Base` / `size` | The base array / element count |
| `next()` / `copy()` | Advance-and-return / fresh 1-D C-order copy |
| `AsTyped<T>(writeable)` | Typed, by-`ref T`, C-order view of the same walk (unboxed); `T` = exact dtype |

### `NDIterator` (np.nditer) — selected surface

| Member | Description |
|--------|-------------|
| `finished` / `iternext()` / `reset()` | Iteration protocol |
| `this[int i]` / `value` | Current operand view / all operand views |
| `multi_index` / `index` | Coordinate / flat index (needs `multi_index` / `c_index`\|`f_index`) |
| `shape` / `ndim` / `itersize` / `nop` / `operands` / `dtypes` / `itviews` | Introspection |
| `copy()` / `remove_axis(i)` / `remove_multi_index()` / `enable_external_loop()` | Reshaping the iteration |
| `close()` / `Dispose()` | **Required** — frees state, flushes write-backs |

### `Broadcast` (np.broadcast)

| Member | Description |
|--------|-------------|
| `shape` / `ndim` (`nd`) / `size` / `numiter` | Broadcast result metadata |
| `iters[i]` | Per-operand flat C-order stream (`NDFlatIterator`) |
| `index` / `reset()` | Live cursor / rewind |
| `foreach` → `object[]` | Per-operand value tuple per element |

---

See also: [NDIter](NDIter.md) (the iterator engine, flags, kernels), [Getting & Setting Values](getting-and-setting-values.md), [Broadcasting](broadcasting.md), [NDArray](NDArray.md).
