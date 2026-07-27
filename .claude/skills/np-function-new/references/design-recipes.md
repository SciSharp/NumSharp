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
| NumPy OBJECT / protocol API (iterator, context manager) | Object-API recipe below — house class shape, own-iterator semantics, no oracle tier | `nditer`, `ndindex`, `ndenumerate`, `broadcast`, `printoptions` |
| Reduction (axis/keepdims/accumulator) | Reduction recipe below — accumulation ORDER is parity | `sum`, `mean`, `quantile` |
| Multi-array joiner | Sequence-input C# shape + mutual-exclusion + casting recipe below | `concatenate`, `stack` |
| RNG distribution | Seed-exact sequence parity recipe below — no oracle tier exists | `random.normal` |
| Tolerance/predicate | Scalar-return + tolerance recipe below | `allclose`, `isclose`, `array_equal` |

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

## Recipe: NumPy object / protocol APIs (nditer, ndindex, ndenumerate, broadcast, printoptions)

Some np.* names are not "array in → array out" — they are **objects with a protocol** (iteration,
context management). Parity shifts from "same bytes" to "same protocol semantics", and a handful of
C#-vs-Python object-model gaps have to be decided deliberately, because each one compiles fine while
being subtly wrong.

- **House shape: lowercase factory → PascalCase nested class**, following `np.broadcast` →
  `np.Broadcast` (`Creation/np.broadcast.cs`). C# forbids a nested type and a static method sharing
  a name, and the factory is the call site users type, so the FACTORY keeps NumPy's spelling:
  `np.ndindex(…)` → `np.NDIndex`, `np.nditer(…)` → `np.NDIterator`.
- **Own-iterator semantics.** NumPy's iteration objects satisfy `iter(x) is x` — one live cursor, so
  a second pass RESUMES instead of restarting. That is observable behavior, so replicate it *and*
  say so in the XML doc: it violates what every C# reader assumes about `IEnumerable`.
- **`GetEnumerator()` must NOT return `this` when the object owns unmanaged state.** `foreach`
  disposes the enumerator it obtains. `np.Broadcast` can hand out `this` safely because its
  `Dispose` is a no-op; an object holding native state cannot — returning `this` CLOSES it at the
  end of any `foreach`/LINQ pass, and every property read afterwards throws. Return a thin wrapper
  whose `Dispose` is a no-op and which drives the same live cursor: identical semantics, no
  self-destruction. This one is nasty because the common case still *looks* fine — the values come
  out correctly and only a later property read fails.
- **Advance timing is parity.** Python's `__next__` returns the value at the current position and
  THEN advances, so after consuming one element the cursor already reads 1 — observable through
  `copy()`, `iterindex`, or anything else exposing position. C# requires `MoveNext()` to advance
  before `Current` is read, so mirror Python by **publishing first, advancing after**. That is safe
  when the published value captures an ABSOLUTE pointer; where it doesn't (a buffer the next step
  refills) call the hazard out in the docs, exactly as NumPy does with its
  `[x.copy() for x in it]` idiom.
- **Python's arity-dependent returns need ONE C# shape.** `nditer` yields a bare 0-d array for one
  operand and a tuple for several; C# has no union type. Pick the uniform form (`NDArray[]`,
  matching `np.Broadcast`'s `object[]`) and document the single-operand `[0]`.
- **Views into live state stay live.** If the object hands out views onto a moving pointer they are
  valid only until the next step — the same contract as NumPy. Document it; do not silently
  deep-copy, which would break write-through.
- **No oracle tier, and say so.** These have no `(dtype, shape, bytes)` result for the corpus to
  bit-compare, so they are legitimately absent from `gen_oracle.py`/`OpRegistry.cs`. Record that in
  the CLAUDE.md paragraph so a later reader knows it was a decision, not an omission. Unit tests
  built from probed protocol transcripts are the gate.

## Recipe: giving an engine `ref struct` a long-lived managed owner

NumSharp's hottest engine types are `ref struct`s (`NDIterRef`) so they can hold raw pointers without
GC overhead. A public object API whose lifetime spans user code is a class — and a `ref struct`
cannot be a class field. The bridge is the already-heap-allocated state:

- **Detach** the state pointer to the managed owner, then **Borrow** a NON-owning copy of the ref
  struct for the duration of each call (`Backends/Iterators/NDIter.Detach.cs`).
- **Audit what else the ref struct was holding.** A release-the-pointer helper that leaves the
  companion state behind is a silent correctness bug: `NDIterRef.ReleaseState` nulls the state but
  strands `_writebackOriginals`, so a COPY_IF_OVERLAP temp would never be copied back. Detach must
  carry operands, write-backs, and any cached delegates — the last one for speed, since re-resolving
  it per step undoes the point of caching.
- The owner now holds unmanaged memory, so it needs `IDisposable` + a finalizer safety net, and its
  teardown must reproduce the ref struct's `Dispose` ORDER (flush buffers → resolve write-backs →
  free), not just its steps.

## Recipe: exposing an engine `ref struct` to USER code (typed/unboxed iteration)

The mirror image of the recipe above. When the goal is not a NumPy-shaped object but *speed* — let
callers touch the engine with no managed object per element — do NOT give it a managed owner at all.
Ship the `ref struct` itself, in two pieces (`APIs/np.nditer.Typed.cs`).

- **Split the ENUMERABLE from the ENUMERATOR, and the split is load-bearing.** The factory returns a
  `readonly struct` holding only the operands and options — **no unmanaged state** — whose
  `GetEnumerator()` builds a fresh engine handle. The `ref struct` enumerator owns that state and
  disposes it. Returning `this` from `GetEnumerator()` here is a **use-after-free**: `foreach`
  disposes what it obtains, so the first pass frees the state the second would walk. Note this is a
  DIFFERENT answer from the protocol-object recipe above, where the fix was a no-op-`Dispose`
  wrapper sharing one live cursor — that preserves NumPy's resume semantics, which a value-typed
  API should not try to have. Consequence to document: re-enumeration RESTARTS.
- **`foreach` disposes a `ref struct` enumerator through the pattern** — no `IDisposable`, no
  `using`, nothing for the caller to remember. That is what frees the state, and it is the main
  reason this shape beats handing out a disposable class. Hand-driven `MoveNext` loops must dispose
  themselves; say so.
- **Drive by CHUNK, not by element.** Turn `EXTERNAL_LOOP` on and let `MoveNext` walk the inner loop
  with pointer arithmetic, touching the iterator only at a chunk boundary — a contiguous array is
  then ONE engine call for the whole walk. Measured: the naive `Iternext()`-per-element form was 3×
  slower. Micro-tuning the cursor past that did not pay (a leaner one-cursor variant measured no
  better than the two-pointer form).
- **Never buffered.** A `ref`/`Span` into a buffer dangles the moment the next fill lands. Unbuffered
  pointers are absolute into the operand — which is also what makes it safe to advance the iterator
  *before* handing out the chunk you already captured.
- **An exact-dtype guard is mandatory, not a nicety.** A `ref T` cannot convert, so a mismatch
  silently REINTERPRETS the bytes (probed: `nditer<double>` over int32 yields `2.1e-314, …`).
  Throw, and name the fix in the message.
- **`Span<T>` implies unit stride.** A stepped view cannot be expressed. Detect it at
  `GetEnumerator` — for a single operand the inner stride is fixed for the whole iteration, so this
  is knowable up front — and fail there rather than mid-loop. Buffering does **not** rescue it:
  probed, NumPy's own buffered `external_loop` also yields non-contiguous chunks for `a[:, ::2]`.
- **Idiomatic-C# divergences are allowed here, unlike on a parity surface** — this is an extension,
  so an empty array iterates zero times instead of demanding NumPy's `zerosize_ok`. Pin each one
  with a test and say *why* in the XML doc; the bar is "a C# caller would be surprised otherwise",
  not "NumPy does it".
- **Route around engine defects rather than widening the change**, and record them. Absorbing them
  in one internal helper (`TypedIterHelpers.ReadInnerLoop`) kept the public surface honest while
  leaving `NDIterRef` alone: `GetInnerLoopSizePtr()` access-violates on a 0-d operand (reads
  `Shape[-1]`), and `GetInnerStrideArray()` returns ELEMENT strides where the kernel contract takes
  BYTE strides.

## Recipe: mirror NumPy's LAYERING, not only its behavior

NumPy splits work between a Python wrapper and a C core, and the split is informative. Work the
Python layer does is work NumSharp's `np.*` entry point should do; work the C layer does belongs in
the engine. Getting this backwards produces a correct-but-misplaced implementation that the next
person can't find.

Concretely: `np.nditer`'s flag-string parsing, argument conversion and error texts all live in
`nditer_pywrap.c` (the wrapper), not in the iterator itself — so they belong in `APIs/np.nditer.cs`,
and NumSharp's `NDIterRef` correctly has none of them.

The corollary is a common finding: **the wrapper often fills gaps the engine leaves.** NumPy infers
an allocated operand's dtype by promoting the other operands; NumSharp's engine instead demands an
explicit dtype and throws. That inference is wrapper work in NumPy, so implementing it in the
wrapper restores parity without touching the engine. When you find the engine "missing" a behavior,
check which NumPy layer supplies it before proposing an engine change.

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

## Recipe: reductions (axis / keepdims / dtype / out)

- One NumPy-shaped overload: `(a, axis, dtype, out, keepdims, where, initial)`. Axis needs BOTH
  `int?` (null = flatten) and `int[]` — `axis=()` is NOT `axis=None`: the empty tuple reduces
  NOTHING (identity pass over the array; probed). `np.sum`'s 10-positional-overload pile predates
  this convention — `quantile` is the modern in-house exemplar.
- `dtype=` is the ACCUMULATOR/loop dtype, not a post-cast — probed: `sum([1000,1000] as i8,
  dtype=f2)` computes in float16 (precision loss visible). NEP50 default accumulators (sum(i4)→i8
  etc.) are tabled in project CLAUDE.md.
- Empty reductions: identity value (sum→0, prod→1) or the verbatim no-identity error
  (`zero-size array to reduction operation minimum which has no identity`).
- Axis validation: `AxisError: axis 2 is out of bounds for array of dimension 2` and
  `ValueError: duplicate value in 'axis'`.
- **Float accumulation order is parity.** NumPy sums pairwise (block 128, unroll 8). A different
  tree (e.g. multi-accumulator SIMD) drifts 1 ULP once N crosses the blocking threshold — probed:
  f32 sum of 1000×0.1 is `0x42c80004` vs NumPy's `0x42c80002`, while the ≤24-element fuzz corpus
  stays green. Either replicate NumPy's exact tree or document the divergence in
  `MisalignedRegistry` (np.evaluate's f64 accumulation is the documented-divergence precedent).
  See `audit.md` for the envelope lesson.
- Plan the `nan*` sibling in the same design — they ride the Masking kernels
  (`.Masking.NaN.cs` / `.Reduction.NaN.cs`), not separate algorithms.

## Recipe: multi-array joiners (concatenate / stack / block family)

- C# sequence shape (house pattern, `Creation/np.concatenate.cs`): `NDArray[]` primary overload +
  tuple-arity conveniences (`(a,b)`, `(a,b,c)`, …) so call sites read like NumPy's tuples.
- `axis: int?` where `axis=None` means ravel-then-join (probed).
- `out=` and `dtype=` are mutually exclusive — verbatim:
  `concatenate() only takes `out` or `dtype` as an argument, but both were provided.`
- `casting=` routes through the cast machinery → the safe-cast rejection text comes out verbatim
  for free (probed identical to NumPy).
- Structural errors, all verbatim (probed): empty sequence (`need at least one array to
  concatenate`), 0-d inputs (`zero-dimensional arrays cannot be concatenated`), and the
  per-dimension mismatch text that names the dimension, indices, and sizes — reproduce its format
  exactly.
- Output layout is voted: F only when ALL inputs are F-contiguous and not all also C — the
  project CLAUDE.md concatenate note.

## Recipe: random distributions (np.random.*)

- Parity is **byte-identical draw sequences from a seed** — the draw ORDER (including rejection
  loops and the Marsaglia-polar gauss-cache carry) is the contract, not just the distribution.
  Probed: seed(42) `normal(5)` bytes exactly match NumPy's, and `normal(size=5)` equals five
  `normal()` calls because the cache persists across call granularity.
- No oracle tier covers RNG. Verification = fixed-seed expected-bytes/values unit tests +
  `get_state`/`set_state` round-trips (house precedent: the random tests / `OpenBugs.Random.cs`).
- Param validation texts are terse — `ValueError: scale < 0` — capture verbatim.
- NumPy broadcasts array-valued distribution params (`loc=[0,100]`, `scale=[[1],[2]]` — probed);
  scalar-only C# params are a recorded restriction until extended (dependency-audit rule).

## Recipe: tolerance predicates & scalar returns

- Return types follow NumPy's: `allclose` → C# `bool` (NumPy returns a python bool),
  `isclose` → `NDArray<bool>`. Don't invent an NDArray return for the scalar one.
- The formula is asymmetric — `|a-b| <= atol + rtol*|b|` — probed:
  `isclose(1.0, 1.1, rtol=.1)` is True, `isclose(1.1, 1.0, rtol=.1)` is False. Pin both.
- NumPy accepts ARRAY rtol/atol (they broadcast; probed `[False, True]`); `double`-only C# params
  are a recorded restriction.
- `equal_nan=` gates NaN self-equality; ±inf compare by exact equality.

## Note: exception-type mapping

Texts are matched verbatim; TYPES map by house convention — ValueError→`ArgumentException`,
AxisError→`AxisOutOfRangeException` (or the house `AxisError`), shape ValueError→
`IncorrectShapeException`, cast-flavored TypeError→`InvalidCastException`. Prefer the
**message-only** exception constructors: the paramName ctors append `(Parameter 'axis')` /
`Actual value was 2.` noise onto otherwise-verbatim texts (visible on sum's AxisError today).

## Recipe: NaN in search/set/sort-adjacent ops

Equality-based membership can't find NaN: `isin([nan], [nan])` → `False` (NaN ≠ NaN through the
sort-equality path). Sort order puts NaN last. Never assume either way — probe, then pin the
result in a test, because both "obviously True" and "obviously False" intuitions are wrong for
some function in this family.
