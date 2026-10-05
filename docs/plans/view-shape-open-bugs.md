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
| 2b | `NDArray.shape` returns the array's own dimension array | new | [x] | [x] | [x] | [x] | (the commit after `bb5b2fcf`) |
| 2c | `Shape.Dimensions` and `Shape.Strides` return the shape's own arrays | new | [x] | [x] | [x] | [ ] | |
| 2d | The public `Shape` constructors keep the caller's array | new | [x] | [x] | [x] | [ ] | |
| 3 | `choice` without replacement returns a view | 3 pins + 1 control | [x] | [x] | [x] | [ ] | (another session, see bug 3) |

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
passed + 2 skipped, MLNet 206 passed + 1 skipped, ParquetNet 11 passed; 0 failed). The `[OpenBugs]` failing set is the same as on `bb5b2fcf` without 2b: 69 tests by name. One more
pin, the timing test `T2_NonZero_Is_Much_Slower_Than_NumPy`, failed once in the baseline's full run and passes alone on
both builds (noise).

2c and 2d: not started.

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

**Held.** On 2026-10-05, while bug 1 was being committed, an uncommitted change to
`src/NumSharp.Core/RandomSampling/Generator.Choice.Sampler.cs` appeared in the main working tree. It is another
session's work on this very bug (it writes the result straight into a fresh array of the final shape, "that owns its
data, as NumPy's result does"). To avoid a conflicting fix, bug 3 is left to that change; tick this row when it lands
with the three pins un-pinned and the control still a view.
