# Design Recipes — mapping a function's shape to the house integration

Pattern 1 (compose) vs Pattern 2 (engine + kernels) is too coarse on its own. Real functions land
in recurring shapes; match yours to a row before designing from scratch. (The table's examples were
probed against NumPy 2.4.2 during the skill's gap-finding experiment.)

| Function shape | Recipe | Example |
|---|---|---|
| Pure composition of existing np.* | Pattern 1 file in `<Area>/`; no engine touch; **run the dependency audit first** | `logspace` = linspace → power → astype |
| New elementwise ufunc | The full 12-touchpoint checklist in `new-ufunc.md` | `copysign`, `heaviside`, `hypot` |
| Composition + algorithm-selection param | Compose the default path from existing ops; a new kernel ONLY for the kind that needs one | `isin` (kind='sort' via argsort/searchsorted; kind='table' wants a table kernel) |
| Index-consuming op | Fancy-index composition first (NumPy itself does); dedicated gather/scatter kernel only if the benchmark demands | `take_along_axis` (NumPy builds an index tuple) |
| Validation-gated kernel, data-dependent output | Direct kernel + two-pass sizing + cast-machinery validation | `bincount`, `nonzero` |

## Recipe: dependency audit (before committing to composition)

NumPy's own source names your composition targets — read it and list them (logspace's body is six
lines: broadcast/ndmin dance → `linspace` → `power` → `.astype(dtype, copy=False)`). For each
target, verify the **NumSharp counterpart supports the modes NumPy uses** — not just that it
exists. Example finding: full logspace parity needs array start/stop + `axis=` on linspace, and
NumSharp's linspace is scalar-only. A missing mode is a scope decision made explicitly and recorded:

- **Extend the dependency first** (preferred — parity debt compounds; the extension is in-scope for
  the session because the function cannot ship without it), or
- **Ship documented partial parity** — the restriction goes in the CLAUDE.md paragraph AND an
  `[OpenBugs]`/issue so it is tracked, never silent.

## Recipe: index-consuming ops (take_along_axis, put_along_axis, choose, compress…)

Probed semantics that generalize across the family:

- **Any integer dtype** is accepted as indices (uint8 works); float AND bool are rejected with
  ``IndexError: `indices` must be an integer array`` — an IndexError, not TypeError, and bool is
  explicitly not an integer here.
- **Negative indices wrap**; out-of-bounds raises
  `IndexError: index 9 is out of bounds for axis 1 with size 3` (per-axis text).
- **ndim contract before value checks**: take_along_axis demands exact ndim match
  (``ValueError: `indices` and `arr` must have the same number of dimensions``) with per-dim
  broadcasting of the non-axis dims — arr's dims and indices' dims broadcast BOTH ways.
- **Validation order is observable parity** — probe pairwise-invalid inputs to learn which error
  fires first; replicate the order.
- **Index-typed OUTPUTS are Int64** — NumPy's `intp` on 64-bit platforms (NumPy 2.x is int64 even
  on Windows). House precedent: `Backends/Default/Indexing/Default.NonZero.cs` allocates Int64.
  Never Int32.
- **Results own their memory** (fancy-index family) — never views, always C-contiguous.

## Recipe: data-dependent output size

House pattern is **pre-size-then-fill** (optimizations.md §D; `np.nonzero` = IL popcount pass →
exact alloc → IL scan pass). For bincount: pass 1 = max-reduce (fold the ≥0 validation into the
same pass), alloc `max(max+1, minlength)`, pass 2 = scatter-add. Result dtype policy: counts →
Int64 (`intp`), weighted accumulation → Double (`float64`) — probed, not guessed.

## Recipe: validation through the cast machinery (verbatim errors for free)

NumPy's input gates are frequently NOT bespoke messages — they are the safe-casting layer speaking:

- `bincount([1.5])` → `TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
  according to the rule 'safe'`
- complex weights → same text with `complex128`/`float64`.
- ndim gates come from the C conversion layer: 2-D x →
  `ValueError: object too deep for desired array`; 0-d x →
  `ValueError: object of too small depth for desired array`.

Route NumSharp's validation through the equivalent machinery (`np.can_cast` semantics + shared
gate helpers) instead of hand-writing prose — the texts then match for free and stay matched.

## Recipe: algorithm-selection params (kind= / mode= / side=)

- Enumerate the valid values INCLUDING the None-auto default; probe the invalid-value text
  verbatim (`ValueError: Invalid kind: 'bogus'. Please use None, 'sort' or 'table'.`).
- Each kind carries its own constraint errors (`The 'table' method is only supported for boolean
  or integer arrays. Please select 'sort' or None for kind.`).
- **The auto-selection heuristic is behavior** — isin's None picks sort-vs-table from a memory
  heuristic in the source; replicate the formula (read `_arraysetops_impl.py`), don't invent one.
- C# shape: `string kind = null`, validated up front, NumPy's default spelled the same way.

## Recipe: dtype= that is NOT a ufunc loop-dtype

Ufunc `dtype=` selects the LOOP (compute precision — project CLAUDE.md ufunc section). Non-ufunc
functions usually **post-cast** instead: logspace computes `power(base, y)` at float precision and
then `.astype(dtype, copy=False)` — probed: `logspace(0, 3, 4, dtype=int32)` → exact
`[1, 10, 100, 1000]`. Read the source to see which semantics your function has; implementing the
wrong one diverges on every inexact case.

## Recipe: multiple return arrays

House precedents: `np.modf` returns `(NDArray, NDArray)`; `np.average` + `np.average_returned`
splits NumPy's `returned=True` arity-change into a second name. Prefer the tuple for fixed pairs
(`frexp`, `divmod`, `histogram`); the `_returned`-suffix split when a flag changes the return arity.

## Recipe: NaN in search/set/sort-adjacent ops

Equality-based membership can't find NaN: `isin([nan], [nan])` → `False` (NaN ≠ NaN through the
sort-equality path). Sort order puts NaN last. Never assume either way — probe, then pin the
result in a test, because both "obviously True" and "obviously False" intuitions are wrong for
some function in this family.
