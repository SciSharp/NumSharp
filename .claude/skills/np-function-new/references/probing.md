# Probing NumPy & NumSharp

Everything here exists so Phase 1 produces a **behavior matrix** (predict NumPy's exact output for
any input) and Phase 4 can **diff the two implementations on identical inputs**. Probes are cheap —
run many small ones rather than one clever one.

## The NumPy side — `python_run`

Always verify the pin first; a different NumPy silently invalidates every observation:

```bash
python_run <<'EOF'
import numpy as np
assert np.__version__ == "2.4.2", np.__version__
import inspect
print(inspect.signature(np.foo))     # exact parameter names, order, defaults
EOF
```

### Behavior-matrix sweep (dtype × params)

```bash
python_run <<'EOF'
import numpy as np
dtypes = [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
          np.int64, np.uint64, np.float16, np.float32, np.float64, np.complex128]
for dt in dtypes:
    a = np.array([3, 0, 7, 0], dtype=dt)          # adapt values to the op
    try:
        r = np.foo(a)                              # sweep params in an inner loop
        print(f"{np.dtype(dt).name:>10}: dtype={r.dtype} shape={r.shape} {r!r}")
    except Exception as e:
        print(f"{np.dtype(dt).name:>10}: {type(e).__name__}: {e}")
EOF
```

The `except` branch is not error handling — it is **data collection**. Error type + text verbatim
go into the matrix; NumSharp must raise the same words.

### Ufunc loop introspection — the dtype policy in one line

For any ufunc, read the loop table BEFORE sweeping dtypes:

```python
print(np.copysign.types)      # ['ee->e', 'ff->f', 'dd->d', 'gg->g'] → float-only:
                              # bool rides e (Half), all ints ride d (Double),
                              # complex absent → input-coercion TypeError
```

This is the result-dtype policy, the promotion table, and the rejection list at once — the dtype
sweep then just confirms it. (Wiring a new ufunc? → `new-ufunc.md`.)

### Two-input ops probe dtype PAIRS, not dtypes

For binary/two-array ops (isin's element × test_elements, copysign's x1 × x2), sweep the outer
product of dtypes — mixes are where promotion bugs live. Probed examples worth copying: isin with
int32 element in float64 test_elements (promotes, works); isin of int64 `-1` against uint64 `2**63`
(NEP50 promotes both to float64 → `False`, no error).

### Edge-case sweep (run for every op, no exceptions)

- Empty: `np.array([], dtype=...)`, `np.zeros((0,3))`, and an empty **view** `np.zeros((4,3))[2:2]`.
- 0-d: `np.array(5.0)` (owning) AND `a[0,0]` (0-d view) — they behave differently in some paths.
- 1-element 1-D (`np.array([5.0])`) — ambiguous against 0-d in sloppy code.
- Value specials: NaN, ±inf, `-0.0`, integer extremes, subnormals if float.
- Axis: negative, repeated (`(0,0)`), out-of-range — capture WHICH error fires FIRST (NumPy's
  validation order is observable and part of parity; e.g. rot90 reports "Axes must be different."
  before the range check).
- Scalars: python `2` / `2.5` (NEP50 weak) vs `np.int64(2)` (strong) — promotion differs.
- Bool input — many ufuncs remap or reject bool with a distinct no-loop error.
- Rank acceptance: feed 0-d, 1-D, 2-D to functions expecting 1-D — the rejections come from the
  conversion layer with their own texts (`object of too small depth for desired array` /
  `object too deep for desired array`), not from the function.
- **Validation order**: when two things are wrong at once (bad ndim + bad dtype, bad axis pair),
  WHICH error fires first is observable parity — probe pairwise-invalid combos and replicate the
  order (rot90's "Axes must be different." before the range check is the canonical example).
- NaN findability in search/set ops (`isin([nan],[nan])` → `False`) — probe, never assume
  (`design-recipes.md` → NaN recipe).

### View-vs-copy identity

```python
r = np.foo(a)
print(np.shares_memory(r, a), r.flags['WRITEABLE'], r.flags['C_CONTIGUOUS'])
r_ = np.foo(a); a[...] = <mutation>; print(r_)     # write-through proves view
```

### Layout sweep

Reuse the recipes in `test/oracle/layout_catalog.py` (the fuzz corpus builders) — or the quick
manual set: `a[::2]`, `a[::-1]`, `a.T`, `a[1:5]` (offset), `np.broadcast_to(a, ...)` (read-only),
`np.asfortranarray(a)`, `a[:, None]`. Pick every family in `variations.md` that can reach the op.

### Bit-exact value capture

When "looks equal" is not enough (float rounding, NaN payloads, signed zero):

```python
print(r.tobytes().hex())
```

and compare against C# bytes (below). This is the same standard the fuzz oracle enforces.

## The NumSharp side — `dotnet_run`

The internals template — `#:` paths **must be absolute**; the AssemblyName matches
`InternalsVisibleTo` so `Shape.strides/offset`, `Storage`, flags etc. are visible:

```bash
dotnet_run <<'EOF'
#:project K:/source/NumSharp/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using NumSharp;

var a = np.arange(12).reshape(3, 4);
var r = np.foo(a);
Console.WriteLine(r.ToString(true));               // repr form — dtype visible
Console.WriteLine($"shape=({string.Join(",", r.shape)}) writeable={r.Shape.IsWriteable}");
EOF
```

Bytes for bit-compare:

```csharp
unsafe { var bytes = new ReadOnlySpan<byte>(r.Storage.Address, checked((int)(r.size * r.dtypesize)));
         Console.WriteLine(Convert.ToHexString(bytes).ToLowerInvariant()); }
// contiguous result only — copy() a view first
```

## Side-by-side parity diff

Drive both from the same literal inputs and diff the transcripts:

1. Write ONE case list (values, dtype, params) — keep it in the script, not your head.
2. `python_run` prints `case_id → repr / dtype / error text / bytes`.
3. `dotnet_run` prints the same lines for NumSharp.
4. Any line differing = a Phase-1 misunderstanding or a bug. Resolve it before writing more code.

`ToString(true)` is a byte-exact port of NumPy 2.4.2's `repr` printing, so repr-level diffs are
meaningful — but for float-heavy ops still do one `tobytes().hex()` pass.

## Timing probes (pre-benchmark sanity only)

- **Never time Debug.** `dotnet run -c Release - < script.cs` — file-based `dotnet_run` compiles
  `#:project` NumSharp.Core with `DebuggableAttribute(DisableOptimizations)`, ~2× slow on
  hand-written loops. `#:property Optimize=true` does NOT fix Core, only the script.
- best-of-N (take the min), warmup excluded, correctness asserted before every timed section.
- These probes answer "is the design sane?"; the quotable numbers come from the **benchmark** skill.
