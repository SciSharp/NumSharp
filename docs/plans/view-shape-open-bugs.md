# View-Shape open bugs — fix plan and progress

> The view-Shape allocation fix (`de35a0f2`, 2026-10-03) found three bugs and left them open. Commit `2f92d91d`
> pinned them as `[OpenBugs]` tests. A probe on 2026-10-05 found more triggers of the same root causes, which no test
> covered. This plan fixes them one at a time. Each fix starts only after three proofs: NumPy's behavior, NumSharp's
> divergence, and that the divergence is a parity bug and not a documented `[Misaligned]` choice.
>
> Source of truth: `refs/numpy` (2.4.2):
> - `numpy/_core/src/multiarray/arrayobject.c`: `array_new` (`ndarray.__new__`) and `PyArray_CheckStrides`;
> - `numpy/_core/src/multiarray/ctors.c`: `PyArray_NewFromDescr_int`, which copies the dimensions into the array;
> - `numpy/random/_generator.pyx` and `numpy/random/mtrand.pyx`: `choice`.
>
> Probes: NumPy 2.4.2 (win-amd64) and NumSharp at `2821a708`, both on 2026-10-05.

## Progress

Legend: `[ ]` open, `[~]` in progress, `[x]` done. The four steps of a row are P (NumPy's behavior proved), D (NumSharp's
divergence proved), B (confirmed as a parity bug) and F (fixed and gated).

| # | Bug | Tests that pin it | P | D | B | F | Commit |
|---|---|---|---|---|---|---|---|
| 1 | The wrap constructors install the caller's layout | 4 pins + new | [x] | [x] | [x] | [x] | `ec28e5b1` |
| 2a | The `Shape` indexer has a setter | 3 pins + `AuditV2.T1_10` | [x] | [x] | [x] | [x] | `bd26afb8` |
| 2b | `NDArray.shape` returns the array's own dimension array | new | [x] | [x] | [x] | [x] | `b88bcb81` |
| 2c | `Shape.Dimensions` and `Shape.Strides` return the shape's own arrays | new | [x] | [x] | [x] | [x] | (the commit after `d3c863ef`) |
| 2d | The public `Shape` constructors keep the caller's array | new | [x] | [x] | [x] | [ ] | |
| 3 | `choice` without replacement returns a view | 3 pins + 1 control | [x] | [x] | [x] | [x] | `0f904216` |

Order of work: 1, 2a, 2b, 2c, 2d, 3 (highest severity first; see the 2026-10-05 scoring: bug 1 scored 72 to 85, bug 2
scored 75 to 80, bug 3 scored 8).

## The process for each bug

1. **NumPy's behavior.** Read the NumPy source and probe NumPy 2.4.2. Record the answers in this file.
2. **NumSharp's divergence.** Probe NumSharp with the same scenarios, and run the pinned tests. Record the answers.
3. **Parity verdict.** Confirm that no `[Misaligned]` test, `MisalignedRegistry` entry or `.claude/CLAUDE.md` decision
   documents the divergence as intended.
4. **Fix and gate.**
   - Fix the bug in NumPy's structure.
   - Remove `[OpenBugs]` from the pins and move them next to the tests of the code they cover.
   - Add tests for the triggers that no test covered.
   - Run the gates: the new tests, the whole `[OpenBugs]` set, the full suite on net10.0 and net8.0, and the oracle
     `FuzzMatrix` set when the fix touches an oracle-gated path.
   - Measure the hot paths that the fix touches (before and after).
   - Commit, then mark the row here.

---

## Bug 1 — the wrap constructors install the caller's layout

### NumPy

`ndarray.__new__(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)` (`arrayobject.c` `array_new`):

- `shape` is a sequence of DIMENSIONS. `PyArray_IntpConverter` copies it.
- With a buffer and no `strides`, `PyArray_NewFromDescr_int` computes the strides with `_array_fill_strides`: C order,
  or F order when `order='F'`. The array starts at `buffer + offset`.
- The buffer must hold at least `offset + itemsize * prod(shape)` bytes, else
  `TypeError: buffer is too small for requested array`. A larger buffer is allowed (the array uses a prefix).
- Explicit `strides` are checked by `PyArray_CheckStrides`, else
  `ValueError: strides is incompatible with shape of requested array and size of buffer`.

The natural C# port of `np.ndarray(x.shape, buffer=values)` is `new NDArray(values, x.Shape)`.

### Probes (the same scenarios on both sides)

| Scenario | NumPy 2.4.2 | NumSharp at `2821a708` |
|---|---|---|
| B1.1 shape of `arange(20)[::4]`, 5 values | `[1, 2, 3, 4, 5]`, strides `(1)` | strides `(4)`: **reads past the buffer** |
| B1.2 the same with `strides=x.strides` | `ValueError: strides is incompatible …` | (no strides parameter) |
| B1.3 shape of `arange(24).reshape(4,6)[:, ::2]`, 12 values | `[[0,1,2],[3,4,5],[6,7,8],[9,10,11]]`, strides `(3,1)` | strides `(6,2)`: **reads past the buffer** |
| B1.4 B1.1 with `order='F'` | `[1, 2, 3, 4, 5]`, strides `(1)` | strides `(4)`: **reads past the buffer** |
| B1.5 shape of `arange(6).reshape(2,3).T`, C | `[[0,1],[2,3],[4,5]]`, strides `(2,1)` | strides `(1,3)`: `[[0,3],[1,4],[2,5]]` (**wrong order**) |
| B1.6 B1.5 with `order='F'` | `[[0,3],[1,4],[2,5]]`, strides `(1,3)` | strides `(1,3)`: same as NumPy |
| B1.7 shape of `arange(5)[::-1]`, C | `[1, 2, 3, 4, 5]`, strides `(1)` | stride `-1`, offset 4: `[5, 4, 3, 2, 1]` (**reversed**) |
| B1.8 shape of `broadcast_to(arange(3.), (2,3))`, 6 values | `[[0,1,2],[3,4,5]]`, writeable | strides `(0,1)`: `[[0,1,2],[0,1,2]]`, **read-only** |
| B1.9 shape of `arange(20)[2:7]` (contiguous slice) | `[1, 2, 3, 4, 5]` | `[1, 2, 3, 4, 5]` (same) |
| B1.12 0-d shape over 1 value | `7.0` | `7` (same) |
| B1.13 shape of `zeros((0,3))[::2]`, 0 values | shape `(0,3)`, strides `(3,1)` | strides `(0,1)` (metadata differs, no read) |
| B1.15 the `IArraySlice` overload with B1.1's shape | as B1.1 | strides `(4)`: **reads past the buffer** |

### Parity verdict

A parity bug. No `[Misaligned]` test or registry entry covers it. `.claude/CLAUDE.md` lists the wrap constructors as an
open bug ("NOT covered: the WRAP constructors install the given layout over the CALLER's buffer as given"). The root
cause: `NDArray(Array, Shape, char)` and `NDArray(IArraySlice, Shape, char)` pass the caller's `Shape` to
`UnmanagedStorage.Allocate(IArraySlice, Shape)`, which installs its strides, offset and flags over the caller's buffer.
Only `order='F'` with `NDim > 1` rebuilt the layout.

### Fix design

- The two public wrap constructors (and the `NDArray<T>` twins that forward to them) read only the DIMENSIONS of their
  `shape` argument, as NumPy's `shape` parameter: C strides by default, F strides with `order='F'` for every `NDim`,
  offset 0, writeable. A shape that already is that layout is kept (no allocation).
- `UnmanagedStorage.Allocate(IArraySlice, Shape)` keeps trusting the layout it is given. It is the analog of NumPy's C
  API (`PyArray_NewFromDescr` with a data pointer and strides, where "the user is responsible for correctness"), and
  internal code builds views through it. Its documentation will say so.
- Audit before the change: every internal caller of the two wrap constructors that passes a non-C layout on purpose. A
  temporary runtime check in the worktree throws on such a layout while the full suite runs; a grep covers the callers
  that the suite does not reach.

### Not in this bug (related constructor divergences, proposed as a follow-up)

- B1.10: a buffer that is too small. NumPy raises `TypeError: buffer is too small for requested array`; NumSharp raises
  `ArgumentException: values.Length does not match shape.Size`. Same outcome (an error), different type and text.
- B1.11: a buffer that is larger than the shape. NumPy uses a prefix; NumSharp raises the same `ArgumentException`.
- NumSharp's own `np.ndarray(shape, dtype, buffer, order)` defaults to `order='F'` and float32 (NumPy: `order=None`,
  i.e. C, and float64), and ignores `order` without a buffer (NumPy: `np.ndarray((2,3), order='F')` is F-contiguous).

### Status

**Done.** What changed:

- `OrderResolver.ResolveBufferOrder(char)`: NumPy's `ndarray` order rule. `'F'`/`'f'` is F; `'C'`, `'A'`, `'K'` in
  either case are C; anything else is `ValueError: order must be one of 'C', 'F', 'A', or 'K' (got 'X')` (verbatim).
  The constructors ignored invalid codes and treated lowercase `'f'` as C before.
- `Shape.ForWrappedBuffer(char)` (`View/Shape.Allocation.cs`): the dimensions-only layout. It keeps a shape that
  already has exactly that layout (no allocation), sets `WRITEABLE` when only the flag differs, and otherwise builds a
  new C or F layout from a copy of the dimensions. Strides are compared exactly, because the C/F flags skip extent-1
  axes and are set on every empty shape.
- Both public wrap constructors (and the `NDArray<T>` twins, through them) call it after validating `order`.
- `np.ones`' core now installs its kept layout through `new UnmanagedStorage(slice, shape)`, as `np.full` does. The
  audit below showed it was the only internal caller that passed a non-C layout to a public wrap constructor on
  purpose (the KEEPORDER layout of `ones_like(a, order='K')`).
- `UnmanagedStorage.Allocate(IArraySlice, Shape, bool)` documents that it installs its layout as given, like NumPy's C
  API.

Audit of internal callers: a temporary check logged every non-C layout that reached the two constructors during the
full NumSharp.Tests and Oracle runs (2,084 lines). Apart from the 4 pins, every hit came from `np.ones` /
`np.ones_like` with a dense F or permuted layout. A grep of Core and the interop packages found no other caller that
passes a view layout.

Tests (`test/NumSharp.Tests/Creation/WrapConstructor.ViewShape.Tests.cs`, 20 tests, every value probed against NumPy
2.4.2): the 4 former pins, the untested triggers (reversed, transposed in C and F, broadcast, empty stepped), guards
(contiguous slice, 0-d, F at rank 3), the order codes and the invalid-order error, all 15 dtypes through both
overloads, the generic twins, in-place wrapping of a 1-D array, a multi-dimensional C# array, the unchanged size
check, and `np.ones` / `np.ones_like` keeping KEEPORDER layouts. Against the old constructors, 14 of the 20 fail; the 6
that pass are guards of behavior that was already right.

Gates: NumSharp.Tests net10.0 17,873 passed and net8.0 17,872 passed (CI filter, 0 failed); Oracle 226 passed; the
`[OpenBugs]` set fails 68 (was 72: exactly the 4 moved pins).

Cost (ns per call, best of 15, the baseline build `0cb7ed09` and the fix measured in two interleaved runs; NumPy
2.4.2 alongside):

| Call | Baseline | Fixed | NumPy |
|---|---|---|---|
| `new NDArray(double[1000], (1000,))` | 462 / 584 | 514 / 445 | 269 |
| `new NDArray(double[10000], (100,100))` | 194 / 320 | 186 / 203 | 272 |
| `new NDArray(double[10000], (100,100), 'F')` | 200 / 207 | 222 / 236 | 394 |
| `np.ones((1000,))` | 325 / 323 | 327 / 324 | 769 |
| `np.ones((100,100))` | 1,581 / 1,519 | 1,521 / 1,478 | 1,695 |
| `np.ones_like(F (100,100))` | 1,570 / 1,532 | 1,523 / 1,573 | 1,863 |

No change beyond run-to-run noise, except about +15 ns on the F wrap: the F layout is built from a copy of the
dimensions, so it does not share the caller's array while bug 2 is open. Drop the copy once 2d makes shapes immutable.
The 1-D wrap of a `double[1000]` stays slower than NumPy (0.5 to 0.6 NPY/NS) in both builds: pinning the managed array
(`GCHandle.Alloc`) is that floor, not the layout rule.

---

## Bug 2 — an array's dimensions and strides are reachable and writable

### NumPy

`PyArray_NewFromDescr_int` (`ctors.c`) allocates the array's own `dimensions` and `strides` and copies the caller's
values into them. `ndarray.shape` and `ndarray.strides` return new tuples, which cannot be changed. The only way to
change an array's shape is the `a.shape = …` setter, which validates the request like `reshape`.

| Scenario | NumPy 2.4.2 |
|---|---|
| B2.1 `s = [3,4]; a = np.zeros(s); s[0] = 100` | `a.shape == (3, 4)` |
| B2.2 the same with `s = np.array([3, 4])` | `(3, 4)` |
| B2.3 the same through `np.empty`, `np.ones`, `np.full`, `ndarray(s, buffer=…)`, `rng.random` | `(3, 4)` each |
| B2.4 `d = a.shape; d[0] = 100` | error: a tuple has no item assignment |
| B2.5 `a.strides[0] = 1` | error: a tuple has no item assignment |
| B2.6 `a.shape = s; s[0] = 100` | `a.shape == (6, 2)` |
| B2.7 `y = x + 1; x.shape = (12,)` | `y.shape == (3, 4)` |
| B2.8 a view keeps its own dimensions after `x.shape = (2, 6)` | `(1, 4)` |

### NumSharp at `2821a708`

| Probe | NumSharp | NumPy |
|---|---|---|
| P1 `d = x.shape; d[0] = 100` | `x.shape == (100,4)`, `x.size == 12` | `(3,4)` |
| P2 `long[] s = {3,4}; a = np.zeros(s); s[0] = 100` | `(100,4)` | `(3,4)` |
| P3 the same with `int[] s` | `(3,4)` (same) | `(3,4)` |
| P4 `x.Shape.Strides[0] = 1000` | strides `(1000,1)` | (no such route) |
| P5 `x.Shape.Dimensions[0] = 100` | `(100,4)` | (no such route) |
| P6 `a = new NDArray(new double[6], new Shape(dims)); dims[0] = 100` | `(100,3)` | `(2,3)` |
| P7 `a = arange(6).reshape(dims); dims[0] = 100` | `(2,3)` (same) | `(2,3)` |
| P8 `np.empty(dims)`, `np.ones(dims)`, `np.full(dims, 7.0)`, then `dims[0] = 100` | `(100,3)` each | `(2,3)` |
| P9 `a.shape = dims; dims[0] = 100` | `(3,2)` (same) | `(3,2)` |
| P10, P11 `np.ones(s)` / `np.empty(s)`, then `s[0] = 100` | `(100,3)` each | `(2,3)` |
| P12 `s = x.Shape; s[0] = 100` | `x.shape == (100,4)` | `(3,4)` |
| P13 `v = x[0:1]; s = v.Shape; s[1] = 50` | `v.shape == (1,50)` | `(1,3)` |
| P16 `y = x + 1; d = x.shape; d[1] = 9` | `x` AND `y` become `(3,9)` | `y` stays `(3,4)` |

The pinned scenarios (`np.zeros(s)`, `new NDArray(dtype, s)`, `rng.random(s)`, then `s[0] = 100`) belong to the same
family. A changed dimension does not change the cached size or the buffer, so a read of a "new" row reads past the
buffer (probed: `a.GetDouble(50, 0)` returned heap memory with no error).

### Parity verdict

A parity bug. No `[Misaligned]` test or registry entry covers it, and `.claude/CLAUDE.md` documents the opposite
intent: "Shape readonly struct — Immutable after construction (NumPy-aligned)". A master-era pin,
`AuditV2_LogicShapeStorage.T1_10_Shape_SetIndexer_Mutates_Readonly_Struct`, already reports the setter ("ideally, the
mutation should not be possible"). The root cause has two halves:

- Arrays and shapes SHARE dimension and stride arrays freely (an elementwise result shares its input's dimensions, an
  allocation keeps the caller's shape). Sharing is safe only if nothing can write into the arrays.
- Five public routes write into them: the `Shape` indexer setter (2a), the `NDArray.shape` getter (2b), the
  `Shape.Dimensions` and `Shape.Strides` getters (2c), and the public `Shape` constructors, which keep the caller's
  `long[]` (2d).

### Fix design

Keep the sharing (it costs nothing) and close the five routes, which is NumPy's contract: an array's dimensions are its
own, and what NumPy hands out is an immutable snapshot.

- **2a.** Remove the `Shape` indexer setter. `Shape` becomes immutable, as documented. Core has no caller of the setter
  (verified by removing it and building Core). Breaking change: user code that writes `shape[i] = v` no longer compiles.
- **2b.** `NDArray.shape` returns a copy (NumPy returns a new tuple). Core reads switch to the internal
  `Shape.dimensions` field, so no internal path allocates.
- **2c.** `Shape.Dimensions` and `Shape.Strides` return copies. Core reads switch to the internal fields.
- **2d.** The public `Shape` constructors that take a caller's array copy it (NumPy's `PyArray_IntpConverter`). A new
  `params ReadOnlySpan<long>` overload lets inline arguments (`new Shape(3, 4)`) allocate once, as today. Internal
  callers that pass an array they built themselves use an internal constructor that adopts it.
- The pinned tests write `s[0] = 100`, which no longer compiles after 2a. They become tests of the NumPy contract: the
  indexer is read-only (reflection), and the C# analog of NumPy's list (`long[] dims`) can be changed after
  `np.zeros(dims)` and similar calls without changing the array.

Performance gate: the number of allocations and the time of `np.zeros((3,4))`, `x + 1` (1K), `np.sum(x, axis)`,
`reshape`, a slice and `new NDArray(values, shape)`, before and after 2b and 2d. No regression is accepted on an
internal path.

### Status

**2a done.** The indexer `Shape.this[int]` has no setter (documented why). Removing it compiled every project that
references Core: Core, Bitmap, the OpenBLAS / pythonnet / MLNet / OnnxRuntime / ParquetNet packages, NumSharp.Tests,
Oracle, the five interop test suites and the C# benchmark; a grep of the examples found no writer. Only four tests wrote
through it:

- `AuditV2_LogicShapeStorage.T1_10` (a master-era `[OpenBugs]` pin of the same setter) is now the regression test
  `T1_10_Shape_Indexer_IsReadOnly`.
- The three bug-2 pins now hold NumPy's list as the caller's own `long[]` (`np.zeros(dims)`,
  `new NDArray(dtype, new Shape(dims))`, `rng.random(new Shape(dims))`, then `dims[0] = 100`) and stay `[OpenBugs]`
  until 2d.
- New `test/NumSharp.Tests/View/Shape.Immutability.Tests.cs`: no `Shape` indexer is writable; the read accessor keeps
  NumPy's tuple indexing (negative axes, an out-of-range axis throws).

Gates: NumSharp.Tests net10.0 17,876 passed and net8.0 17,875 passed (0 failed), Oracle 226 passed, `[OpenBugs]` set
fails 67 (was 68: `T1_10` left it). No runtime cost (a removed setter).

**2b done.** `NDArray.shape` returns a copy of the dimensions on every read (a 0-d array returns its empty array:
nothing to write into), documented why.

The copy alone made internal paths allocate. A first build changed only the getter, and these calls allocated more
than 2a (bytes per call, best of 15): `x + y` +32, `np.sum(m, 0)` and `np.mean(m, 1)` +144, `np.concatenate` and
`np.stack` +320, `np.dot` +160, `np.sort` +112, `np.where` and `np.cumsum` +32, with times up 7% to 28%. So every
internal read moved to the internal field `Shape.dimensions`:

- The list came from the compiler: `[Obsolete(error)]` on the getter for one audit build gave 384 read sites in 115 Core
  files (247 `x.shape[i]`, 4 `.Length`, 133 others; no `x.shape = …` setter site needed a change).
- Every alias of the result was checked for a later write through the alias (indexed assignment, `ref`/`out`,
  `Array.Reverse/Sort/Fill/Clear/Copy` as destination, `AsSpan`, followed through second-level aliases): none. The one
  site that stores it (`UnmanagedStorage.ReplaceData`'s `_shape = nd.shape`) converts it to an immutable `Shape`. The
  callees that receive it (`KeepdimsReshape`, `ComputeAxisGeometry`, `BuildNestedRec`, `SameShape`, `ShapeTuple`, the
  kernels that pin it with `fixed`) only read it. So the switch is exactly the behavior before 2b.
- 381 reads in 109 files were switched mechanically at the compiler's positions (`x.shape` to `x.Shape.dimensions`,
  a bare `shape` inside `NDArray` to `this.Shape.dimensions`).
- The change was written on `bd26afb8` and rebased onto `bb5b2fcf`, which added `ReductionArguments.cs` (the
  `np.all`/`np.any` reduce prologue). A second audit build on the rebased tree listed its 6 new reads; all are
  read-only (the broadcast walk writes only its own fresh `bshape`), so they were switched too: 387 reads in 110 files.
- Kept on the copying getter, on purpose: the three public wrappers whose job is to hand out NumPy's shape
  (`MaskedArray.shape`, `np.ma.shape(a)`, `UniqueResult.shape`), and the 19 reads in five random-sampling files that
  another session was editing at the time (`Generator.Choice*.cs`, `Generator.Multivariate.cs`,
  `np.random.choice.cs`, `np.random.multivariate_normal.cs`); those are correct, and cost one small allocation each.
  The second audit build listed exactly these 22 sites and no other.

After the switch every one of the 22 measured calls allocates exactly what it did at 2a, byte for byte, and the times
are within noise (`np.sort` rechecked three times: 7.33–7.45 µs at 2a, 7.37–7.61 µs now).

Tests (`View/Shape.Immutability.Tests.cs`, +6): a write into `x.shape` leaves `x` alone and each read is a new array;
it does not reach an array that shares the dimensions (`y = x + 1`, probe P16); a view's `shape` is a copy; 0-d is
empty; `MaskedArray.shape` and `np.unique(...).shape` return copies; the setter reads the caller's array once (guard).
Against the old getter the four copy tests fail; the guards pass.

Gates (on `bb5b2fcf` + 2b): NumSharp.Tests net10.0 17,885 passed and net8.0 17,884 passed (0 failed), Oracle 227
passed, the four interop suites that need no native binary pass (System.Numerics.Tensors 101 passed, OnnxRuntime 207
passed + 2 skipped, MLNet 206 passed + 1 skipped, ParquetNet 11 passed; 0 failed). The `[OpenBugs]` failing set is
the same as on `bb5b2fcf` without 2b: 69 tests by name. One more pin, the timing test
`T2_NonZero_Is_Much_Slower_Than_NumPy`, failed once in the baseline's full run and passes alone on both builds (noise).

**2c done.** `Shape.Dimensions` and `Shape.Strides` return a copy on every read (a 0-d shape returns its empty array and
an uninitialized `default(Shape)` returns `null`, as before), documented why. NumPy: the shape and the strides are
tuples (B2.4, B2.5), and its `strides` setter validates the request, so no NumPy route reaches an array's own arrays
(probes P4, P5).

The list of reads came from the compiler again, this time over the whole solution: `[Obsolete]` WARNINGS (CS0618, not
suppressed anywhere) on `Shape.Dimensions`, `Shape.Strides` and `NDArray.shape` for one build, so every project
(Core, the friend packages, the tests, the examples and the benchmark) compiled and reported its sites:

- Core: 13 `Dimensions` + 26 `Strides` reads in 12 files. An alias analysis over all 186 `Dimensions`/`Strides` sites
  of the solution found no write through an alias (indexed assignment, `ref`/`out`, an in-place `Array.*` call or
  `AsSpan`) and no escape to a caller (`NDIndexWalker._dims` is a private read-only field, read only). So the 39 Core
  reads were switched mechanically to the internal fields (`dimensions`, `strides`): exactly the behavior before 2c.
- The friend packages (MLNet 4 + 2, OnnxRuntime 3, pythonnet 3 + 2) keep the public getters. They run once per export,
  not per element, write nothing through the result, and read through the public surface on purpose (`.claude/CLAUDE.md`:
  a backend reads `a.Shape.Strides`); the cost is one small allocation per export.
- The same build listed `NDArray.shape` in Core at exactly the 22 sites 2b kept on purpose, so no target framework
  hides another internal read.
- The change was written on `b88bcb81` and rebased onto `d3c863ef` (bug 3's fix `0f904216`, which also switched the 19
  random-sampling `x.shape` reads, and an F-order broadcast fix). A second whole-solution audit build on the rebased
  tree found no `Dimensions`/`Strides` read left in Core, and `NDArray.shape` only at the three public wrappers 2b kept
  (plus two uses of its setter, `result.shape = finalShape`).

Cost: the calls whose Core paths had the 39 reads (`m.strides`, memoryview `Length` and indexer, an N-D `np.dot`,
a strided integer `np.reciprocal`, `np.cumsum`/`np.cumprod` over an axis and transposed, a strided `np.std`/`np.var`,
`np.array(double[][])`, `np.indices(Shape)`, `np.unique(..., return_inverse, axis=0)`, an `np.ndenumerate` walk)
allocate the same bytes as at `b88bcb81` (`np.cumprod` over 1,000 / 10,000 / 100,000 calls: 2,290,344 /
22,891,720 / 228,914,856 B in both builds) and their times are within the run-to-run noise (two interleaved rounds;
`np.std` moved between 9.0 and 13.3 µs on both builds).

Tests (`View/Shape.Immutability.Tests.cs`, +7): a write into `x.Shape.Dimensions` / `x.Shape.Strides` leaves `x` alone
and each read is a new array; it does not reach an array sharing them (`y = x + 1`); a view's (a reversed axis) and a
free `Shape` value's are copies; 0-d returns empty arrays and `default(Shape)` null (guards). Against the old getters
the five copy tests fail; the guards pass.

Gates (on `d3c863ef` + 2c, with the OpenBLAS binary staged, so the backend and host-pinned tests run instead of
skipping): NumSharp.Tests net10.0 18,168 passed and net8.0 18,167 passed (0 failed, 12 skipped), Oracle 255 passed
(1 skipped); the interop suites pass: System.Numerics.Tensors 101, OnnxRuntime 207 + 2 skipped, MLNet 206 + 1 skipped,
ParquetNet 11, pythonnet 736 of 736. The `[OpenBugs]` failing set is the same 66 tests by name as on `d3c863ef`
without 2c (bug 3's fix took the count from 69). On `b88bcb81` the change gave the same answers: 17,892 / 17,891
passed without the binary staged and the same 69 `[OpenBugs]` tests as 2b. The pythonnet suite needs the staged binary
(gitignored, `tools/fetch_openblas.py`): without it, 26 of its tests fail the same way with and without 2c ("the interop
suite runs with OpenBLAS as the default backend").

2d: not started.

---

## Bug 3 — `choice` without replacement returns a view

### NumPy

`Generator.choice` (`_generator.pyx`) builds a fresh index array (Floyd's algorithm into `np.empty(size)`, the tail
shuffle's `.copy()`, or the weighted `found` array) and ends with `idx.shape = shape`, an IN-PLACE shape assignment.
The result owns its data. `RandomState.choice` (`mtrand.pyx`) does the same on the weighted path. Its uniform path
without replacement is `self.permutation(pop_size)[:size]`, a slice, so that result is a view.

| Scenario | NumPy 2.4.2 | NumSharp at `2821a708` |
|---|---|---|
| B3.1 `default_rng(42).choice(1000, 6, replace=False)` | `[652 770 432 858 438 88]`, owndata True | same values, owndata **False** |
| B3.2 `default_rng(42).choice(5, 3, replace=False, p=p)` | `[3 2 4]`, owndata True | same values, owndata **False** |
| B3.3 `RandomState(42).choice(5, 3, replace=False, p=p)` | `[2 4 3]`, owndata True | same values, owndata **False** |
| B3.4 `RandomState(42).choice(10, 6, replace=False)` (control) | `[8 1 5 0 7 2]`, owndata False | same values, owndata False (same) |
| B3.5 `default_rng(42).choice(1000, (2,3), replace=False)` | owndata True | owndata **False** |
| B3.6 `default_rng(42).choice(20000, 1000, replace=False)` (tail shuffle) | owndata True | owndata **False** |
| B3.7 `default_rng(42).choice(1000, 6, replace=True)` | owndata True | owndata True (same) |
| B3.8 `default_rng(42).choice(arange(10)*10, 4, replace=False)` | owndata True | owndata True (same) |

### Parity verdict

A parity bug. No `[Misaligned]` test or registry entry covers it. The values match; the ownership flag and what follows
from it do not (`base` is not None, and a `resize` that changes the size refuses). The root cause:
`Generator.Choice.Sampler.cs` (`ChoiceNoReplaceUniform`, `ChoiceNoReplaceWeighted`) and `np.random.choice.cs`
(`LegacyChoiceNoReplaceWeighted`) end with `arr.reshape(shape)`, which returns a view.

### Fix design

Assign the shape in place, as NumPy's `idx.shape = shape` does, so the fresh index array keeps ownership. The legacy
uniform path without replacement stays a view (the control test).

### Status

**Fixed** on 2026-10-05 (`0f904216`), together with three more ownership and layout gaps of random
results that a probe found. (Earlier the same day this section read "held": the fix was the other session's uncommitted
change that bug 1's work saw in the main working tree.)

- **The fix.** `ChoiceNoReplaceUniform`, `ChoiceNoReplaceWeighted` (`Generator.Choice.Sampler.cs`) and
  `LegacyChoiceNoReplaceWeighted` (`np.random.choice.cs`) write their draws straight into a fresh array of the final
  shape (`shape.AsClean()`), and that array is the result: NumPy's `idx.shape = shape` on a fresh array. The `reshape`
  view is gone. The legacy uniform path stays NumPy's slice view (the control test).
- **The probe.** 780 cases on NumPy 2.4.2 and NumSharp: every `choice` path at the sizes None, `()`, `(k,)`, `(2,3)` and
  `(0,)`; every sampler of both APIs at five sizes; `permutation`; `multivariate_normal`. It compared values, dtype,
  shape, strides and flags. It found three more gaps, fixed in the same commit:
  - `multivariate_normal` (both APIs) returned a reshape view. NumPy ends with `x.shape = tuple(final_shape)`, an
    in-place assignment, so its result owns its data. Now `result.shape = finalShape`.
  - `Generator.permutation(x, axis > 0)` returned `np.take`'s C-ordered array, which owns its data. NumPy's
    `arr[tuple(slices)]` returns its MapIter buffer (the indexed axis first) transposed back: a view, F-contiguous for
    2-D input, with permuted strides beyond. The new `np.FancyIndexAlongAxis` builds that buffer with one tiled IL
    gather (`DirectILKernelGenerator.TakeLeading.cs`).
  - The transpose of an EMPTY array was a fresh array that owned its data, and it was writeable even when its source
    was read-only. NumPy's `PyArray_Transpose` always returns a view. `Default.Transpose` no longer has a size-0
    shortcut.
- **Rode along.** The 19 `x.shape` reads that `b88bcb81` left in the five random-sampling files for this change now
  read `x.Shape.dimensions` (the bug 2b rule: inside Core, never the copying getter).
- **Gates.** `RandomSampling/RandomResultOwnership.Test.cs` (17: the three pins without `[OpenBugs]`, the control, every
  `choice` path and size, `multivariate_normal` at four sizes on both APIs, the `permutation` layouts, a sweep of every
  sampler of both APIs at five sizes), `Indexing/np.take.FancyAxis.Test.cs` (8: every dtype, every unroll remainder,
  several tiles, empty operands, every source layout, the argument checks) and `Manipulation/np.transpose.Empty.Test.cs`
  (3). On the code before the fix, 17 of the 20 ownership and transpose tests fail; the 3 that pass guard paths that
  were already correct.
- **Perf** (NPY/NS, P-cores pinned, best of 15):
  - `choice` tail shuffle 3.9–5.0× (was 2.0–2.6×: the reshape copy and a managed array are gone);
  - Floyd 1.05–4.1× (was 1.02–3.5×; 1.05× at `choice(1_000_000, 10_000)` is the PCG64 draws, the same count on
    both sides); weighted 1.5–2.9× on both APIs;
  - `multivariate_normal` unchanged, 1.5–2.0×;
  - `permutation` along a non-leading axis 2.2–6.2× on 11 shapes and 1.19× for a 1-byte last axis. NumPy's strided
    byte loop is about 2 cycles an element there; the old `np.take` route ran 0.48× on that shape.
- **Not done.** NumSharp's general fancy index `a[:, idx]` still returns a C-ordered array that owns its data, where NumPy
  returns the same transposed MapIter view. That is a library-wide indexing gap; `np.FancyIndexAlongAxis` is a building
  block for it.
