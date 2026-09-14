# Universal functions (ufunc) basics

A **universal function** (ufunc) operates on an `NDArray` element by element, with broadcasting, type casting, and a small set of standard options. `np.add`, `np.sqrt`, `np.exp`, `np.less`, and the arithmetic/comparison operators are all ufuncs: each takes a fixed number of array inputs and produces a fixed number of array outputs, looping in fast C#/SIMD code rather than a C# `for` loop.

NumSharp implements the ufunc *model* — elementwise semantics, broadcasting, NEP 50 casting, `out=`/`where=`/`dtype=` — but exposes each ufunc as a **direct `np.*` function** (and operator), not as a NumPy-style ufunc *object* with `.reduce`/`.accumulate`/`.at` methods. This page covers the model and the porting differences.

---

## Elementwise operation

The simplest ufunc is addition:

```csharp
np.array(new[] { 0, 2, 3, 4 }) + np.array(new[] { 1, 1, -1, 2 });   // [1 3 2 6]
np.add(np.array(new[] { 0, 2, 3, 4 }), np.array(new[] { 1, 1, -1, 2 }));  // same
```

Every arithmetic (`+ - * / %`, unary `-`), comparison (`== != < > <= >=`), and logical (`& | !`) operator is a ufunc, and each has a named function form:

| Category | Operators | Named ufuncs |
|----------|-----------|--------------|
| Arithmetic | `+ - * / %` | `add`, `subtract`, `multiply`, `divide`/`true_divide`, `mod`/`remainder`, `power`, `floor_divide` |
| Comparison | `== != < > <= >=` | `equal`, `not_equal`, `less`, `greater`, `less_equal`, `greater_equal` |
| Logical / bitwise | `& \| !` | `logical_and/or/not/xor`, `bitwise_and/or/xor`, `invert` |
| Unary math | — | `sqrt`, `exp`, `log`, `sin`, `cos`, `abs`, `negative`, `square`, `sign`, `floor`, `ceil`, … |

---

## Broadcasting

Ufuncs apply [broadcasting](../broadcasting.md) so inputs of different shapes still combine — a size-1 dimension is stepped with stride 0 (the same stored element feeds every position along that axis):

```csharp
var a = np.ones((3, 4));
var b = np.array(new[] { 1, 2, 3, 4 });   // (4,)
a + b;                                     // (3, 4) — b broadcast across rows
```

See [Broadcasting](../broadcasting.md) for the full shape rules.

---

## Type casting

When a ufunc's inputs have different dtypes, NumSharp picks a result dtype with NumPy 2.x's promotion rules (NEP 50): the inputs are cast to a common type, the loop runs there, and the result carries that dtype.

```csharp
var i = np.array(new[] { 1, 2, 3 });       // int32
var f = np.array(new[] { 1.5, 2.5, 3.5 }); // float64
(i + f).dtype;                             // float64  — int32 promoted
```

A key NEP 50 subtlety: **weak scalars** (C# primitive literals) adopt the array's dtype rather than upcasting it, while **strong** operands (arrays, 0-d `NDArray`) promote normally:

```csharp
var x = np.array(new[] { 1, 2, 3 }, np.int8);
(x + 1).dtype;                             // int8  — weak scalar 1 does not upcast
(x + np.array(new[] { 1 })).dtype;         // int16 — strong array promotes
```

Full promotion rules and the 15×15 table are in [Data types → Type promotion](../dtypes.md#type-promotion) and [NumPy Compliance](../compliance.md).

---

## `out=`, `where=`, and `dtype=`

The elementwise ufuncs take NumPy's three keyword-style options, exposed as one overload shaped like NumPy's signature — `f(x[, x2], NDArray out = null, NDArray where = null, DType dtype = null)`:

```csharp
var a = np.array(new[] { 1.0, 2.0, 3.0, 4.0 });
var dst = np.zeros(4);

np.sqrt(a, @out: dst);                          // write into dst, return it
np.add(a, 10, @out: dst);                       // dst = a + 10
np.add(a, 10, @out: dst, where: a > 2);         // only where the mask is true; other slots keep prior contents
np.add(0.1, 0.2, dtype: np.float32);            // run the loop at float32 precision
```

- **`out`** — write the result into an existing array (any compatible layout); it is returned. `out` may alias an input (overlap-safe). The result is cast to `out`'s dtype under NumPy's `same_kind` rule.
- **`where`** — a boolean mask that broadcasts to the output; positions where it is `false` keep their prior contents (so pair `where` with a meaningful `out`).
- **`dtype`** — selects the *loop* dtype, i.e. the precision the computation runs at (`np.add(0.1, 0.2, dtype: np.float32)` computes in float32 even into a float64 `out`). It also gates which loops exist — a float-only ufunc rejects an integer `dtype`.

`@out` is spelled with the `@` because `out` is a C# keyword. Pass `where`/`dtype` by name.

---

## Reductions

A reduction collapses an axis with a binary ufunc (sum, product, min, max, …). NumSharp exposes these as **direct functions** with `axis`, `dtype`, and `keepdims`:

```csharp
var x = np.arange(9).reshape(3, 3);
np.sum(x, axis: 1);                 // [3 12 21]  — reduce along axis 1
np.sum(x, axis: (0, 1));            // 36         — reduce all axes (tuple)
np.prod(x.astype(np.float64), axis: 0);
np.max(x, axis: 0, keepdims: true); // shape (1, 3)
```

### The reduce upcast rule

For **`sum`/`prod`/`cumsum`/`cumprod`** with no explicit `dtype`, an integer or boolean input **smaller than the default integer** is upcast to int64 to avoid overflow — matching NumPy:

```csharp
np.array(new[] { 1, 2, 3 }, np.int32);
np.sum(...).dtype;      // int64  — int32 sum widens
np.mean(...).dtype;     // float64
np.max(...).dtype;      // int32  — min/max/amax preserve the input dtype
```

`abs`, `sign`, `min`, `max`, and the comparisons **preserve** the dtype; only the accumulating reductions widen. This is NEP 50 alignment — see the [DirectILKernelGenerator NEP50 table](../il-generation.md) and [Data types](../dtypes.md).

Pass an explicit `dtype` to control it: `np.sum(x, dtype: np.float64)`.

---

## NumSharp does not expose ufunc-object methods

NumPy attaches `reduce`, `accumulate`, `reduceat`, `outer`, and `at` to the ufunc *object* (`np.add.reduce(x)`, `np.multiply.accumulate(x)`, `np.add.at(x, idx, v)`). NumSharp has no ufunc-object surface — use the **direct functions** instead:

| NumPy ufunc method | NumSharp equivalent |
|--------------------|---------------------|
| `np.add.reduce(x, axis)` | `np.sum(x, axis)` |
| `np.multiply.reduce(x)` | `np.prod(x)` |
| `np.maximum.reduce(x)` / `np.minimum.reduce(x)` | `np.max(x)` / `np.min(x)` |
| `np.add.accumulate(x)` | `np.cumsum(x)` |
| `np.multiply.accumulate(x)` | `np.cumprod(x)` |
| `np.add.at(x, idx, v)` (unbuffered scatter-add) | `np.add(x, v, @out: ...)` / `np.put`/`np.place` (note: no *accumulating* scatter) |
| `np.add.outer(a, b)` | `a[":, np.newaxis"] + b` (broadcast) |

`np.maximum`/`np.minimum` (the elementwise pairwise ufuncs) exist as direct functions and are distinct from `np.max`/`np.min` (the reductions), exactly as in NumPy.

---

## Fused expressions — `np.evaluate` (NumSharp extension)

Chaining ufuncs (`a * b + c`) normally allocates an intermediate per operation. NumSharp adds `np.evaluate`, which compiles an expression tree into **one** pass — every elementwise node runs inside a single inner loop, reading each operand once and allocating no intermediates:

```csharp
NDArray r = np.evaluate((NDExpr)a * b + 2);                 // fused a*b+2, one pass
NDArray s = np.evaluate(NDExpr.Sum((NDExpr)a * b));         // fused sum(a*b), no temp
```

This is the NumSharp analog of `numexpr.evaluate` — measured 3–6× faster than the equivalent NumPy chain on large arrays. Per-node dtypes follow the same NEP 50 rules as the individual ufuncs. See [NDIter](../NDIter.md) and the API reference.

---

## Common patterns

### In-place elementwise transform

```csharp
np.multiply(a, 2.0, @out: a);        // double a in place (out aliases input)
```

### Masked update

```csharp
np.add(a, 100, @out: a, where: a < 0);   // add 100 only to negative elements
```

### Reduce with overflow safety

```csharp
np.sum(np.array(new byte[] { 200, 200, 200 }));   // int64 result — no uint8 overflow
```

---

## Troubleshooting

### "`np.add.reduce` doesn't exist"
NumSharp has no ufunc-object methods. Use `np.sum` (see the [table above](#numsharp-does-not-expose-ufunc-object-methods)).

### "My int32 sum came back int64"
That's the reduce upcast rule (NEP 50) — accumulating reductions widen small integer inputs. `min`/`max`/`abs` preserve the dtype. Pass `dtype:` to override.

### "`out` didn't change / masked positions had garbage"
With `where`, the `false` positions keep whatever `out` held before — initialize `out` (e.g. from a copy) so masked-off slots are meaningful.

### "A weak scalar didn't upcast my array"
By design (NEP 50): `int8_array + 1` stays int8. Use a strong operand (`+ np.array(new[]{1})`) or an explicit dtype to promote.

---

## API reference

| Feature | Form |
|---------|------|
| Elementwise ufunc | `np.add(a, b)`, `np.sqrt(a)`, operators `+ - * /`, `== < >`, `& \| !` |
| `out=` | `np.sqrt(a, @out: dst)` — write into `dst`, returned; may alias an input |
| `where=` | `np.add(a, b, @out: dst, where: mask)` — masked; false slots keep prior contents |
| `dtype=` | `np.add(a, b, dtype: np.float32)` — selects the loop precision/loop |
| Reductions | `np.sum`, `np.prod`, `np.min`, `np.max`, `np.mean`, `np.std`, `np.var`, `np.cumsum`, `np.cumprod` (with `axis`, `dtype`, `keepdims`) |
| Pairwise | `np.maximum`, `np.minimum`, `np.fmax`, `np.fmin` |
| Fused | `np.evaluate((NDExpr)…)`, `NDExpr.Sum/Prod/Min/Max/Mean` |

---

## Related reading

- [Broadcasting](../broadcasting.md) — the shape rules ufuncs apply.
- [Data types](../dtypes.md) — casting and promotion (NEP 50).
- [Indexing on NDArray](indexing.md) — where `out`/`where` write.
- [IL Generation](../il-generation.md) — how the elementwise loops are compiled (SIMD) and the NEP 50 reduce table.
- [NumPy ufunc basics guide](https://numpy.org/doc/stable/user/basics.ufuncs.html) — the upstream article.
