# Dtype logic out of the np.* layer

Living plan and state document. Branch `strings`, measured at `2e872477`. Started 2026-10-24.

**Task (verbatim):** "So if we were to agree that we need to move all logic out of np.* into deeper where dtype handling
resides or even inside tensorengine (anywhere inside) Where are the challenges or can we simply move all with no issues?
Write a plan covering how to and answer my questions"

**State:** analysis and plan. No code has moved. The decisions in [§5](#5-decisions-for-the-owner) are open. Progress
goes into the [§9 state log](#9-state-log), newest entry first.

**The answer.** It cannot simply be moved, but nothing blocks it. Most dtype lines can move as they are, as the same
helper called one level down, and two families already show the result: `np.sum` and the four sort functions have no
text line at all. A line-by-line move fails in thirteen ways, measured in [§3](#3-can-it-simply-move-the-thirteen-challenges).
The five that shape the plan:

1. NumPy checks arguments and dtypes in one interleaved order, so the unit of a move is a function's whole body, not its
   dtype lines (C1).
2. The engine's members take less than NumPy's signatures: of 29 reduction and scan members, only `ReduceAdd` takes
   `out=` (C2).
3. Many functions have no engine member to move into (C3).
4. The engine calls back up into np.* 383 times, so code moved down that keeps those calls is not deeper (C4).
5. The oracle has holes that a reordered check can pass through: the three copies of the quantile prologue already
   disagree and no oracle case noticed (C10).

The plan moves each function family's whole body behind its public method, one family per commit, behind two ratchet
tests and a NumSharp-against-NumSharp snapshot.

**Inside `TensorEngine` itself?** Yes for every operation NumPy implements in C: those become (or already are)
`TensorEngine` members, and the dtype switch sits in `DefaultEngine`'s override. For the functions NumPy writes in
Python, the plan recommends the engine layer but not the public `TensorEngine` contract (D1); both are possible.

**Background:** the 2026-10-22 analysis of the chararray boundary (231 `NDCharArrayBoundary.Sees(` on this branch) and
its follow-ups, which found that text dtype decisions sit in two places: 205 lines inside the engine and 535 lines in
the np.* layer, with no central dtype dispatch.

---

## 1. What "out of np.*" has to mean

Three layers:

| Layer | What it is | What it holds | What it never holds |
|---|---|---|---|
| **F**, the facade | the public `np.*` functions, the `NDArray` members and operators, the module classes (`np.linalg`, `np.fft`, ...) | the public signature and its XML docs; the conversions C# forces (implicit conversions at the call site, several C# overloads normalized into one call); the class-boundary statement (matrix, chararray); one call into O | a dtype test, numeric or text |
| **O**, the operation | a `TensorEngine` member (pluggable, reached through `a.TensorEngine`), or an internal operation class (`QuantileEngine`, `AxisSort`, `NDPolySeries`) | NumPy's function body, statement for statement: its argument checks and its dtype checks in NumPy's order, the conversions NumPy makes late, the switch to a family core | a call into F |
| **K**, the cores | the IL kernels, `NDIter`, `Backends/Strings` (S, U, StringDType), the cast implementations, the `DType` classes and their slots | the loops, per dtype family | NumPy's Python-level statements |

Rules: F calls O. O calls O or K, never F. A dtype test lives only in O or K. A conversion that exists only because of
C# (a C# `string` read as Python's `str`, the legacy implicit `string` to `NDArray` reading) may stay in F; the ratchet
lists each with its reason.

**This keeps NumPy's structure** (CLAUDE.md, core principle 2). NumPy's Python signature is F, NumPy's Python body is
O, NumPy's C (the ufunc loops, the ArrayMethods, the dtype slots) is K. The dtype statements NumPy itself writes in
Python do not disappear: `if a.dtype.kind == "c": raise TypeError(...)` in `quantile` and
`if issubclass(z.dtype.type, _nx.complexfloating)` in `angle` move, unchanged and in NumPy's order, from the public
method into O.

**Which O** (decision D1):

- A NumPy operation implemented in C (the ufuncs with their reduce and accumulate, sort / partition / searchsorted,
  take / put / where, copy and cast, nonzero, the matmul family) is a `TensorEngine` member. Its dtype switch lives in
  `DefaultEngine`'s override, as `ReduceAdd`'s StringDType route already does.
- A NumPy function implemented in Python (the quantile family, histogram, cov, gradient, trapezoid, average, angle, the
  set routines, numpy.polynomial, numpy.ma) is an internal operation class that composes `TensorEngine` members. It
  belongs to the engine layer (new ones under `Backends/Default/<Family>/`, see C3) but not to the public pluggable
  contract: a backend replaces the kernels the operation calls, not NumPy's Python.
- Values entering storage (creation, assignment, IO, fill) belong to the `DType` class: NumPy's `setitem` / `getitem`
  slots (`PyArray_Pack`), which `DTypeMeta` does not port yet. The DType oracle lists `setitem` as exempt because
  "element packing goes through NPTypeCode switches (SetAtIndex / the cast kernels)".

**What the tree already shows:**

- `np.sum` has no text line. `TensorEngine.ReduceAdd` takes the whole call, `out=` included, and the StringDType route
  sits in `DefaultEngine` (`Backends/Default/Math/Reduction/Default.Reduction.Add.cs:51-56`).
- `np.sort`, `np.argsort`, `np.partition` and `np.argpartition` have no text line. `AxisSort.Sort` switches StringDType,
  fixed-width text and numbers as its first statements (`Backends/Default/Sorting/AxisSort.cs:74-86`).
- `np.cumsum` is the opposite case (`APIs/np.cumsum.cs:18-45`). `ReduceCumAdd(arr, axis_, dtype)` has no `out=`, so
  `np.cumsum` keeps two text routes and the numeric `WriteScanToOut`, and `VStringReduces` runs twice per call: once in
  `np.cumsum`, once in `Default.Reduction.CumAdd.cs:30`.
- `np.median` routes StringDType to `NDVStringStatistics.Median` before `QuantileEngine.Compute`
  (`Statistics/np.median.cs:42-70`). NumPy's `_ureduce` prologue therefore exists three times (`QuantileEngine.Compute`,
  `QuantileText.Compute`, `NDVStringStatistics.Median`), and the copies disagree (C10).

---

## 2. Inventory at 2e872477

Scope of "the np.* layer" below: `src/NumSharp.Core` without `Backends/`, `Char/`, `APIs/np.strings*` and `APIs/np.char*`
(the text API itself, 81 lines, out of scope). Comment lines are not counted. Commands: [Appendix C](#appendix-c-measurement-commands).

| What | Count |
|---|---|
| Text dtype tests in the np.* layer (calls into `Backends/Strings`) | 535 lines in 135 files |
| ... of them in `np.*.cs` facade files | 398 lines |
| Numeric dtype tests in the np.* layer (a heuristic pattern, a lower bound) | 1,287 lines |
| ... in `np.*.cs` facade files | 440 lines in 100 of 369 files |
| ... in `NDArray.*.cs` member files outside `Backends/` | 165 lines |
| ... by directory | Utilities 318, Polynomial 134, Logic 103, Manipulation 98, Ma 75, Math 71, Statistics 65, Casting 64, Creation 63, Indexing 52, Selection 49, APIs 41, RandomSampling 29, LinearAlgebra 25, Sorting_Searching_Counting 22, IO 19, Matrixlib 18, DTypes 16, Operations 13, Fourier 12 |
| Text dtype tests inside the engine (`Backends/Default`) | 205 lines in 62 files |
| np.* layer lines that call the engine object (`x.TensorEngine.X(`) | 217 |
| np.* layer lines that call a static `DefaultEngine` member | 53 |
| np.* layer lines that call a static helper of `Backends/Default/Sorting` (`AxisSort`, `AxisPartition`, ...) | 64 |
| np.* calls made from inside the engine | 383 calls: `Backends/Default` 89 computations + 142 structural, `Backends/Strings` 48 + 96, `Backends/Kernels` 7 + 1 |
| `TensorEngine` | 151 public members (133 abstract, 18 virtual) and one internal virtual (`Evaluate`); `DefaultEngine` is the only subclass |
| `TensorEngine` reduction and scan members | 29, of which one (`ReduceAdd`) takes `out=` |
| `[NDScoped]` | on 152 facade `np.*.cs` files, 28 `Backends/` files |
| `NDMatrixBoundary.PlainFor(` regions | 226, of which 51 in `NDArray.*.cs` members |
| `<exception>` tags on public `np.*` members that describe a text error | 163 |
| Test lines that name the internals this plan moves first (`NDStringUfuncs`, `NDVString*`, `QuantileText`, `QuantileEngine`, `AxisSort`, the two boundaries) | 147 lines in 19 files |
| Oracle cases | main corpus 820,494; string oracle 266,059 in 64 sections |

The 535 text lines by kind. Each kind has a different deeper place, and the per-file lists are in
[Appendix A](#appendix-a-text-lines-by-kind).

| Kind | Lines / files | What the lines do | The deeper place |
|---|---|---|---|
| K1 the quantile family | 108 / 10 | route text to `QuantileText` / `NDVStringStatistics` before `QuantileEngine.Compute` | `QuantileEngine.Compute` (exists) |
| K2 reductions whose engine call takes less than NumPy's | 57 / 10 | the reduction family's text routes (`VStringReduces`, `ReduceIntoOutRoute`) in `np.cumsum`, `np.cumprod`, `np.nancumsum`, `np.nancumprod`, `np.trace`, `np.ptp`, `np.any`, `np.all`, `np.count_nonzero` and the matrix reductions | a widened reduction member |
| K3 compositions above the engine | 150 / 43 | error replays and text routes in functions that compose engine calls (histogram, cov, gradient, trapezoid, searchsorted, angle, convolve, the linalg guards, ...) | NumPy's own composition through engine calls where it gives NumPy's answer, else an operation class |
| K4 values entering storage | 153 / 52 | creation, IO, casting, assignment and fill branching on text | the `DType` class slots and one coercion path |
| K5 numpy.polynomial and numpy.ma | 61 / 14 | NumPy's Python statements of those packages | mostly O already (package internals); the legacy `np.poly*` facades move into them |
| K6 operators and members with a raw fast path | 6 / 6 | `!`, `negative`, `positive`, the failed-comparison fallback of `==` / `!=` | the engine member each calls |

---

## 3. Can it simply move? The thirteen challenges

| # | Challenge | Nature | Where it is handled |
|---|---|---|---|
| C1 | NumPy interleaves argument checks and dtype checks | a rule | a batch moves a whole body ([§6](#6-method-per-batch)) |
| C2 | the engine's members take less than NumPy's signatures | work | widen the members (D2, P1b) |
| C3 | many functions have no engine member | work | operation classes (D1) |
| C4 | the engine calls up into np.* 383 times | work | computational up-calls cut in each batch, structural ones in P6 |
| C5 | C# converts typed arguments before any of our code runs | inherent | `object` parameters converted in O at NumPy's position |
| C6 | error texts name the public function | work | O entries take the operation's name |
| C7 | scopes and the leak gates | work | the scope stays on F; the leak gates in every batch |
| C8 | the class boundary must run first, above O | inherent | F keeps the boundary statement; its own plan (P7) |
| C9 | numeric dtype logic is larger than text, and part of it is not facade logic | work | a classification before P5 |
| C10 | oracle holes and drifted copies of one prologue | risk | the snapshot gate (P0) and D4 |
| C11 | rewriting a composition changes kernels and cost | risk | benchmarks in every batch |
| C12 | docs and tests are tied to where the code lives | work | in every batch |
| C13 | branch logistics | inherent | D6, one family per commit |

### C1. NumPy interleaves argument checks and dtype checks

**Evidence.** `np.quantile` in NumPy 2.4.2 (`refs/numpy/numpy/lib/_function_base_impl.py:4475-4483`):

```python
a = np.asanyarray(a)
if a.dtype.kind == "c":
    raise TypeError("a must be an array of real numbers")    # a dtype check
weak_q = type(q) in (int, float)
q = np.asanyarray(q)                                         # a late conversion
if not _quantile_is_valid(q):
    raise ValueError("Quantiles must be in the range [0, 1]") # an argument check
if weights is not None: ...                                  # more argument checks, then _quantile_unchecked
```

NumSharp's `np.quantile` keeps that order: `RejectComplexQuantileData(a)` before `ValidateQuantile(q)`
(`Statistics/np.quantile.cs:62-63`). A move of only the dtype line into `QuantileEngine.Compute`, which runs after
`ValidateQuantile`, makes `np.quantile(complex_array, 1.5)` raise the range ValueError instead of NumPy's TypeError.

**Consequence.** O owns NumPy's whole prologue, argument checks included. F does nothing that NumPy's body does.

### C2. The engine's members take less than NumPy's signatures

**Evidence.** Of the 29 reduction and scan members of `TensorEngine`, one takes `out=`:

```csharp
public abstract NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null);
public abstract NDArray ReduceCumAdd(NDArray arr, int? axis_, DType dtype = null);
public abstract NDArray ReduceAMax(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null);
public abstract NDArray NanSum(NDArray a, int? axis = null, bool keepdims = false);
public abstract bool All(NDArray nd);
public abstract NDArray<bool> All(NDArray nd, int axis);
```

Wherever the member takes less, the np.* function keeps the rest of NumPy's call: `np.cumsum` keeps its text routes
and the numeric `WriteScanToOut`; `np.all` keeps the reduce prologue (`ReductionArguments`) for `out=` / `where=` /
`keepdims=`; `np.trace` sums through `TensorEngine.Sum` without `out=` (`Indexing/np.trace.cs:116`) and keeps three text
routes (`np.trace.cs:76-87`); `np.max` reaches its keyword engine `DefaultEngine.MaxMinNumPy` as a static member.

**Consequence.** Widen the members before their functions move (D2). `TensorEngine` is public and abstract, so a
widened member is a breaking change for a subclass outside the repo; none is known, and 0.71.0 is not released.

### C3. Many functions have no engine member to move into

**Evidence.** 53 np.* layer lines call a static `DefaultEngine` member and 64 call a static sorting helper: they reach
engine code without the engine object. The quantile family, histogram, cov, gradient, trapezoid, angle, ptp, trace and
the NaN scans have no `TensorEngine` member at all.

**Consequence.** For each family, D1 decides between a `TensorEngine` member and an internal operation class. New
operation classes go under `Backends/Default/<Family>/`; existing ones (`QuantileEngine` in `Statistics/`, the `NDPoly*`
classes in `Polynomial/Package/`) stay where they are, because moving files only adds conflicts.

### C4. The engine calls back up into np.*

**Evidence.** 383 np.* calls in engine code:

| | Computations | Structural / creation |
|---|---|---|
| `Backends/Default` | 89 (15 `evaluate`, 8 `multiply`, 8 `true_divide`, 5 `isnan`, 5 `subtract`, 4 `argmax`, 4 `argmin`, ...) | 142 (31 `zeros`, 28 `copyto`, 14 `getbufsize`, 12 `_FindCommonType`, 10 `ascontiguousarray`, 10 `broadcast_to`, ...) |
| `Backends/Strings` | 48 (8 `amax`, 5 `multiply`, 4 `where`, 4 `subtract`, 3 `isnan`, ...) | 96 (26 `copyto`, 11 `broadcast_to`, 9 `broadcast_shapes`, 5 `expand_dims`, ...) |
| `Backends/Kernels` | 7 (3 `fabs`, 2 `_FindCommonScalarType`, 2 `rint`) | 1 (`promote_types`) |

**Consequence.** Code moved down that keeps calling np.* runs F → O → F → O. Each up-call runs the facade again: its
class-boundary test, its conversions, its error texts naming another function. Each batch replaces the computational
up-calls of the code it moves with the engine member or the family core. The structural ones (allocation, `copyto`,
`broadcast_to`, `getbufsize`) are harmless for plain arrays and move in P6 to internal helpers.

### C5. C# converts typed arguments before any of our code runs

**Evidence.** An implicit conversion (a C# array, a C# scalar marked `WasPythonLiteral`, a `string`) runs at the call
site, before the method's first statement, so it precedes every check NumPy makes, whatever layer holds the check.
NumPy converts some arguments late: `quantile` converts `q` after the dtype check on `a` (C1). For a typed parameter
(`double[] q`) the conversion cannot fail, so the order cannot be seen. For an `object` parameter (array_like, converted
by `PolySequence` / `asanyarray`) the conversion can fail (a ragged list, an unsupported object), and where it happens
is a choice.

**Consequence.** F converts only what the C# signature forces. An `object` parameter that NumPy converts after a check
goes to O unconverted, and O converts it at NumPy's position.

### C6. Error texts name the public function

**Evidence.** `QuantileText.Compute(string op, ...)`, `VStringReduces("cumsum", ...)`, `ThrowIfReduce("median", ...)`;
`np.amax`, `np.max` and `ndarray.max` all raise texts that say `maximum`.

**Consequence.** An O entry takes the operation's name wherever NumPy's text names the function, as `QuantileText`
does. This is part of widening a member.

### C7. Scopes and the leak gates

**Evidence.** `[NDScoped]` sits on 152 facade `np.*.cs` files and 28 `Backends/` files. The leak gates
(`UndisposedIntermediateTests`, `BufferReleaseSweepTests`, the `LeakCatalogue.*` completeness audit over every declared
member) assume the current placement.

**Consequence.** The scope stays on the facade method: scopes nest per thread, so it covers the temporaries of the O
code it calls. An O member that other O code calls with no facade above it needs its own scope or must dispose by hand.
Every batch runs the leak gates.

### C8. The class boundary must run first, above O

**Evidence.** 355 `NDMatrixBoundary.Sees(` and 231 `NDCharArrayBoundary.Sees(` occurrences (one of the 231 is a
`cref` in a doc comment) decide a subclass's answer before the dtype; `np.median`'s chararray rule (`MedianOfChararray`)
comes before its text switch (`Statistics/np.median.cs:50-51`). The 226 `PlainFor` regions make nested public calls
compute plain; 51 of them are in `NDArray.*.cs` members that call `np.*`. In `np.quantile` the class decision even
feeds the text route: `bool chararray = NDCharArrayBoundary.Sees(a)` is passed to `QuantileOfText`.

**Consequence.** F keeps the boundary statement as its first statement, so this plan does not depend on the class
decision (alternative A, static only, or B, the subclass protocol with a woven entry). Where the class changes the
computation (the `chararray` flag above), F passes the flag into O as a parameter until the class plan decides. A
member that calls O directly instead of `np.*` under `PlainFor` drops its region.

### C9. Numeric dtype logic is larger than text, and part of it is not facade logic

**Evidence.** 440 numeric dtype-test lines in `np.*.cs` and 165 in `NDArray.*.cs`, against 398 text lines in `np.*.cs`.
Of the 1,287 numeric lines of the whole np.* layer, Utilities (318), Casting (64) and DTypes (16) are core code that
happens to live outside `Backends/`.

**Consequence.** Before P5, each remaining numeric line in a facade file is classified: NumPy's Python statement (moves
with the body into O), a NumSharp fast path (into the engine member or the kernel), a C#-only conversion (stays in F,
listed in the ratchet), core code (no move).

### C10. Oracle holes and drifted copies of one prologue

**Evidence.** `np.median` and `np.quantile` over `axis=(1, 2)` of a `(0, 3, 4)` array. NumPy raises
`cannot reshape array of size 0 into shape (0,newaxis)` for float64 and U3 alike. NumSharp returns a `(0,)` float64 for
median and quantile of float64, raises a TypeError from the closing mean's add.reduce for median of U3, and matches only
for quantile of U3 ([Appendix B1](#b1-the-quantile-prologues-disagree)). No oracle case caught it, and the three copies
of the prologue diverged without a red test. Merging copies, which every K1 / K2 move does, picks one copy's behavior.

**Consequence.** P0 adds a snapshot gate: NumSharp's own outcome for a generated call matrix, recorded on the parent
commit and replayed on the child; a move commit changes nothing. A divergence found on the way is fixed in its own
commit, proven first with a NumPy probe and oracle cases (D4).

### C11. Rewriting a composition changes kernels and cost

**Evidence.** `np.angle` of a real input casts `z` to float64, calls `arctan2(0.0, z64)` and casts to the result tier,
after a text pre-check (`Math/np.angle.cs:35-38`) that exists only because the float64 cast would parse text. NumPy's
own statement `arctan2(0, z)` with a weak 0 gives the same dtype and bytes for every real dtype and raises NumPy's
texts for U and StringDType through the engine's arctan2 guard ([Appendix B2](#b2-npangle-as-numpys-own-statement)), so
the pre-check can go. But it computes the float16 tier in a float16 loop where the current path computes in float64.

**Consequence.** Every batch that rewrites a composition benchmarks the functions it touches (NPY/NS, two pinned
P-cores, best of rounds). Where NumPy's own composition is slower, O keeps the faster path together with the error
replay that path needs; either way the facade holds neither.

### C12. Docs and tests are tied to where the code lives

**Evidence.** 163 `<exception>` tags on public `np.*` members describe text errors; 147 test lines in 19 files name the
internals that move first.

**Consequence.** The facade keeps its docs (the public API's contract, which must stay true); every new or widened O
member gets full docs; tests that name a moved internal are re-pointed in the same commit.

### C13. Branch logistics

**Evidence.** The text code exists only on `strings` (rebased onto `journey4` on 2026-10-12). The same facade files are
edited by other work: the one-signature pass changed every indexer and dunder on 2026-10-20 (`c32a300c`). `journey4` is
the open pull request (#631).

**Consequence.** Batches are small (one family per commit, so a conflict stays local), on `strings` for every family
with a text line (D6).

---

## 4. What can move as it is

- **K1** (108 lines): the dtype switch becomes `QuantileEngine.Compute`'s first statement. Same calls, same order, so
  no behavior change (P1a, step 1).
- **K6** (6 lines): each operator's pre-check goes into the engine member it calls.
- **K2** (57 lines), once the members are widened: the text routes go into the member, once, and `VStringReduces` stops
  running twice.

That is 171 of the 535 text lines. K3, K4 and K5 (364 lines) need a decision per function: NumPy's own composition, an
operation class, or the `DType` slots.

---

## 5. Decisions for the owner

| # | Decision | Recommendation |
|---|---|---|
| D1 | Where an O lives | By NumPy's own split: a C-level operation is a `TensorEngine` member; a Python-level function is an internal operation class that composes members (new ones under `Backends/Default/<Family>/`); values entering storage go to the `DType` slots. The alternative, every function as a public virtual `TensorEngine` member, puts all dtype handling literally inside the engine class but grows its public pluggable surface from 151 members to several hundred and asks a backend author to face NumPy's Python. |
| D2 | How to widen engine members | In place, before 0.71.0 ships: change the abstract signatures, implement them in `DefaultEngine`, list the change under Breaking Changes. The alternative, new virtual overloads whose default refuses, breaks nobody but leaves two forms of each operation. |
| D3 | The unit of a batch | A function family's whole body, text and numeric together. `np.cumsum` shows why: its text routes and its numeric `out=` handling are one prologue. |
| D4 | A divergence found during a move | Fixed in its own commit in the same batch, proven first (NumPy probe, oracle cases), never inside the move commit, so every move commit is behavior-preserving. |
| D5 | The ratchets | From P0 on: a new dtype test in a facade file, or a new np.* call in `Backends/`, fails the tests, other sessions' work included. The allowlists only go down. |
| D6 | The branch | `strings` for every family with a text line. Numeric-only families (P5) after `strings` merges into `journey4`, or on `strings` by agreement. |
| D7 | The class boundary | Its own plan (alternative A or B from the 2026-10-22 analysis). Here F keeps the boundary statement as its first statement; if B is chosen, the woven entry takes that statement too. |

---

## 6. Method per batch

1. **Inventory.** The family's dtype and text lines, from the ratchet allowlist.
2. **Read NumPy.** The function's source in `refs/numpy`: which statements are Python (they move to O as they are) and
   which are C (they belong to the engine member or the kernel).
3. **Target.** Per D1: a `TensorEngine` member (widened per D2), an operation class, or the `DType` slots.
4. **Snapshot.** Record NumSharp's outcomes for the family's call matrix on the parent commit (P0).
5. **Move.** The facade keeps its signature, its XML docs, the boundary statement and the conversions C# forces;
   everything else moves to the target in NumPy's statement order. The computational np.* calls of the moved code become
   engine or core calls (C4).
6. **Gates.** Build on net8.0 and net10.0 with no warnings; the snapshot identical; `NumSharp.Tests` on both frameworks
   (`TestCategory!=OpenBugs&TestCategory!=HighMemory`); the oracle (`FuzzMatrix`, the family's string-oracle sections);
   the leak gates; the benchmark cells of the touched functions (NPY/NS, no cell slower beyond noise).
7. **Ratchets.** Lower the allowlist counts in the same commit.
8. **Docs and tests.** The facade's docs still true; new members documented in full; tests re-pointed.
9. **Divergences.** Each one found: a NumPy probe, oracle cases, a fix in its own commit (D4).
10. **State log.** One entry in [§9](#9-state-log).

---

## 7. Phases

### P0. Guard rails

- [ ] **R1, facade dtype ratchet.** A source-walking test modeled on `NativeAllocationChokepointTests` (same project,
      same walk): the text pattern and the numeric pattern of [Appendix C](#appendix-c-measurement-commands) over the
      facade files (`np.*.cs` and `NDArray.*.cs` outside `Backends/`, `Matrixlib/NDMatrix*.cs`), with an allowlist of
      file → count. A new file or a higher count fails and prints the line. Allowed C#-only conversions carry a reason.
      The patterns are a heuristic (they miss a dtype comparison written another way); an analyzer rule beside the
      NDW rules of `NumSharp.Build` can replace them once the counts are small.
- [ ] **R2, up-call ratchet.** The same over `Backends/**`: np.* calls per file, computations and structural calls
      counted apart.
- [ ] **S, the snapshot tool** (`tools/facade-move/`, a `dotnet run` script). For a list of np.* names it reflects
      every public overload and calls it over a generated matrix: the 15 dtypes and S / U / StringDType × the layouts
      C, F, strided, reversed, broadcast, 0-d, empty × each optional argument's variants, invalid ones included (an
      axis out of range or repeated, an `out` of the wrong shape, dtype or writeability, a refused `dtype=`). It writes
      one outcome per call: dtype, shape, flags and a SHA-256 of the bytes, or the exception type and message. Run on
      the parent and on the child; the diff of a move commit is empty.
- [ ] **Pilot.** One family of each easy kind, to prove the method before it scales: `np.median` / `np.quantile`
      (K1 step 1), `np.cumsum` (K2, `ReduceCumAdd` widened with `out=`), `np.angle` (K3, NumPy's own statement).

### P1. The deeper place exists (171 text lines, plus the numeric lines of the same functions)

- [ ] **P1a, the quantile family** (K1, 108 lines in 10 files; 6 functions, 32 public overloads). Step 1: the switch
      (StringDType, fixed width, numbers) as `QuantileEngine.Compute`'s first statement, `Compute` gaining `string op`
      for the texts that name the function. No behavior change. Step 2, its own commit: one shared prologue in NumPy's
      order (`_ureduce`: the axes, the keepdims `out` target, the axis merge; then q's rank; then the method), then the
      family core; this fixes the divergence of Appendix B1. The facade keeps only the class statements and the
      normalization of its C# overloads (`q` as `double`, `double[]`, `long` or `NDArray`); q's checks move with the
      prologue (C1); the `chararray` flag crosses into O as a parameter (C8).
- [ ] **P1b, reductions with a narrower member** (K2, 57 lines in 10 files). Widen `ReduceCumAdd` and `ReduceCumMul`
      (`out=`), `All`, `Any` and `CountNonZero` (`keepdims=`, `out=`, `where=`), and the members `np.trace` and
      `np.ptp` reduce through; move the bodies of `np.cumsum`, `np.cumprod`, `np.nancumsum`, `np.nancumprod`, `np.trace`,
      `np.ptp`, `np.any`, `np.all`, `np.count_nonzero`; route the 14 lines of `NDMatrix.Reductions.cs` through the same
      members.
- [ ] **P1c, operators** (K6, 6 lines in 6 files).

### P2. Compositions above the engine (K3, 150 lines in 43 files)

For each function: first try NumPy's own composition through engine calls, with a probe, as Appendix B2 did for
`np.angle`; where it gives NumPy's answer at no cost, the error replay goes away. Otherwise the body moves into an
operation class and the replay stays there. Order by size: `np.histogram` 16, `np.histogramdd` 9, `np.trapezoid` 8,
`np.searchsorted` 8, `np.cov` 8, `GufuncGuard` 7, `np.gradient` 6, `np.linalg` 5, `np.array_equal` 5, `np.insert` 5,
`np.isin` (VString) 5, `np.average` 5, then the files of four lines or fewer.

### P3. Values entering storage (K4, 153 lines in 52 files)

- [ ] **Design first**, as its own section of this plan before any code: NumPy's `setitem` / `getitem` slots on
      `DTypeMeta` (`PyArray_Pack`), implemented by each class (the numeric builtins by the current `NPTypeCode`
      switches, the text classes by `Backends/Strings`), and one coercion path for array creation (the `array_coercion.c`
      port that today lives in `Polynomial/Package/NDPolySequence.cs`).
- [ ] Then creation (`np.array`, `np.asarray`, `np.full`, `np.ones`, `np.frombuffer`, ...), assignment (`np.copyto`,
      `np.place`, `np.put`, `np.putmask`, `fill`, the indexers), IO (`np.loadtxt`, `np.fromfile`, `np.tofile`, the text
      codecs).

### P4. numpy.polynomial and numpy.ma (K5, 61 lines in 14 files)

- [ ] The legacy facades `np.polyval`, `np.polyint`, `np.polyarithmetic`, `np.poly` and `poly1d` move their bodies into
      operation classes. The package internals (`NDPolySeries`, `NDPolyCalc`, `NDPolyVander`, `NDPolyEval`,
      `NDMaskedArray`) are O already: their dtype statements are NumPy's Python statements, and they stay.

### P5. Numeric-only functions

- [ ] Classify the numeric lines left in facade files after P1 to P4 (C9), then move per family by the method of [§6](#6-method-per-batch).

### P6. The engine stops calling np.*

- [ ] The computational up-calls left after P1 to P5 become engine or core calls; the structural ones become internal
      helpers (allocation, `copyto`, `broadcast_to`, the buffer size). R2 ends at zero computations; any structural
      call left is listed with its reason.

### P7. The class boundary

- [ ] Its own plan (D7).

### Later, outside this plan

A loop registry (NEP 43 `ArrayMethod` loops for ufuncs, registered per DType class, one dispatcher raising NumPy's
no-loop error) would replace the 205 per-member text guards inside `DefaultEngine`. `docs/plans/dtype-system.md`
already specifies it for ufuncs; today `ArrayMethod` serves casts only.

---

## 8. Out of scope

- `np.strings`, `np.char` and `Char/` (the text API itself, 81 lines): dtype-specific by definition. They follow the same
  F / O rule when their own pass comes.
- The engine's per-member text guards (205 lines in 62 files): already deep. See "Later".
- The class boundary (P7).
- Behavior changes, except a divergence found and proven during a move (D4).

---

## 9. State log

- **2026-10-24.** Analysis on `strings` at `2e872477`: the inventory of §2, the challenges of §3, the probes of
  Appendix B. No code moved. Correction of an earlier count given on 2026-10-22 ("15 of the 16 engine reduction members
  lack `out=`"): the engine has 29 reduction and scan members, of which one (`ReduceAdd`) takes `out=`. Waiting for
  D1 to D7.

---

## Appendix A. Text lines by kind

Lines that call into `Backends/Strings`, per file, at `2e872477`.

**K1, 108 lines in 10 files.** `Statistics/np.nanpercentile.cs` 24, `Statistics/np.nanquantile.cs` 24,
`Statistics/np.median.cs` 14, `Statistics/np.percentile.cs` 14, `Statistics/np.quantile.cs` 14,
`Statistics/np.nanmedian.cs` 8, `Statistics/QuantileText.cs` 4, `Statistics/QuantileWeighted.cs` 3,
`Statistics/QuantileEngine.cs` 2, `Statistics/np.quantile.Text.cs` 1.

**K2, 57 lines in 10 files.** `Matrixlib/NDMatrix.Reductions.cs` 14, `APIs/np.cumsum.cs` 7, `Indexing/np.trace.cs` 7,
`Math/np.nancumsum.cs` 7, `Statistics/np.ptp.cs` 7, `APIs/np.cumprod.cs` 5, `Math/np.nancumprod.cs` 5,
`Logic/np.all.cs` 2, `Logic/np.any.cs` 2, `APIs/np.count_nonzero.cs` 1.

**K3, 150 lines in 43 files.** `Statistics/np.histogram.cs` 16, `Statistics/np.histogramdd.cs` 9,
`Math/np.trapezoid.cs` 8, `Sorting_Searching_Counting/np.searchsorted.cs` 8, `Statistics/np.cov.cs` 8,
`LinearAlgebra/GufuncGuard.cs` 7, `Math/np.gradient.cs` 6, `LinearAlgebra/linalg/np.linalg.cs` 5,
`Logic/np.array_equal.cs` 5, `Manipulation/np.insert.cs` 5, `Manipulation/np.isin.VString.cs` 5,
`Statistics/np.average.cs` 5, `Logic/np.logical.cs` 4, `Manipulation/NDArray.unique.cs` 4,
`Math/NDArray.Convolve.cs` 4, `Math/NDArray.Correlate.cs` 4, `APIs/np.nditer.cs` 3, `Logic/np.array_equiv.cs` 3,
`Manipulation/np.isin.cs` 3, `Math/np.angle.cs` 3, `Math/np.emath.cs` 3, `Math/np.sinc.cs` 3,
`Sorting_Searching_Counting/np.digitize.cs` 3, `APIs/np.evaluate.cs` 2, `LinearAlgebra/linalg/np.linalg.arrayapi.cs` 2,
`LinearAlgebra/np.cross.cs` 2, `Manipulation/np.unique_values.cs` 2, `Sorting_Searching_Counting/np.lexsort.cs` 2,
`Sorting_Searching_Counting/np.nanargmax.cs` 2, `Fourier/np.fft.RawFft.cs` 1, `LinearAlgebra/linalg/np.linalg.inv.cs` 1,
`LinearAlgebra/linalg/np.linalg.norm.cs` 1, `LinearAlgebra/linalg/np.linalg.svd.cs` 1, `LinearAlgebra/np.dot.cs` 1,
`LinearAlgebra/np.einsum.Contract.cs` 1, `Manipulation/NDArray.unique.Text.cs` 1, `Math/np.deg2rad.cs` 1,
`Math/np.diff.cs` 1, `Math/np.ediff1d.cs` 1, `Math/np.i0.cs` 1, `Math/np.interp.cs` 1, `Math/np.rad2deg.cs` 1,
`Math/np.unwrap.cs` 1.

**K4, 153 lines in 52 files.** `Manipulation/np.copyto.Strings.cs` 12, `APIs/np.loadtxt.Sinks.cs` 11,
`Creation/np.frombuffer.cs` 7, `Creation/np.full.cs` 6, `Indexing/np.place.cs` 6, `Manipulation/np.copyto.VString.cs` 6,
`APIs/np.fromfile.cs` 5, `Creation/np.array.cs` 5, `APIs/np.tofile.cs` 4, `Creation/np.asanyarray.cs` 4,
`Creation/np.ones.cs` 4, `IO/TextReading/TextIOCodec.cs` 4, `Selection/NDArray.Indexing.VString.cs` 4,
`Casting/Implicit/NDArray.Implicit.Array.cs` 3, `Creation/NDArray.Scalar.cs` 3, `Creation/np.frombuffer.Strings.cs` 3,
`Creation/np.full_like.cs` 3, `Creation/np.ndarray.cs` 3, `Creation/np.vander.cs` 3,
`IO/TextReading/TextIOCodec.Incremental.cs` 3, `Indexing/np.putmask.cs` 3, `Manipulation/NDArray.resize.cs` 3,
`Manipulation/np.pad.cs` 3, `RandomSampling/BoundedIntegers.Broadcast.cs` 3,
`Casting/Implicit/NDArray.Implicit.ValueTypes.cs` 2, `Creation/np.asarray.cs` 2, `DTypes/StringDTypeMeta.cs` 2,
`IO/PyLiteral.cs` 2, `Indexing/np.choose.cs` 2, `Indexing/np.fill_diagonal.cs` 2, `Indexing/np.put.cs` 2,
`Indexing/np.put_along_axis.cs` 2, `Manipulation/NDArray.fill.cs` 2, `Manipulation/np.copyto.cs` 2,
`Manipulation/np.repeat.cs` 2, `Manipulation/np.trim_zeros.cs` 2, `RandomSampling/SeedSequence.cs` 2,
`Utilities/PyUnicode.cs` 2, `APIs/np.fromstring.cs` 1, `APIs/np.where.cs` 1, `Casting/NDArray.getfield.cs` 1,
`Casting/NDArray.tobytes.cs` 1, `Creation/np.array.string.cs` 1, `Creation/np.concatenate.Strings.cs` 1,
`IO/TextReading/LoadTxtSources.cs` 1, `IO/TextReading/TextIOCodec.Unicode.cs` 1, `Indexing/np.diag.cs` 1,
`Indexing/np.take.cs` 1, `Indexing/np.take_along_axis.cs` 1, `Manipulation/np.resize.cs` 1,
`Selection/NDArray.Indexing.Masking.cs` 1, `Selection/NDArray.Indexing.Strings.cs` 1.

**K5, 61 lines in 14 files.** `Polynomial/Package/NDPolySeries.cs` 15, `Ma/NDMaskedArray.cs` 13,
`Polynomial/Package/NDPolyCalc.cs` 8, `Polynomial/np.polyval.cs` 5, `Polynomial/Package/NDPolyVander.cs` 4,
`Polynomial/np.polyint.cs` 4, `Polynomial/Package/NDPolyEval.cs` 3, `Polynomial/np.polyarithmetic.cs` 3,
`Ma/NDMaskedArray.SetOps.cs` 1, `Polynomial/Package/NDPolyEval.FromRoots.cs` 1, `Polynomial/Package/NDPolyFit.cs` 1,
`Polynomial/Package/np.polynomial.polyutils.cs` 1, `Polynomial/np.poly.cs` 1, `Polynomial/poly1d.cs` 1.

**K6, 6 lines in 6 files.** `Math/NDArray.negative.cs` 1, `Math/NDArray.positive.cs` 1,
`Operations/Elementwise/NDArray.Equals.cs` 1, `Operations/Elementwise/NDArray.NOT.cs` 1,
`Operations/Elementwise/NDArray.NotEquals.cs` 1, `Operations/Elementwise/NDArray.Primitive.cs` 1.

---

## Appendix B. Probes

Scratch build of `2e872477` against NumPy 2.4.2, 2026-10-22.

### B1. The quantile prologues disagree

`zeros((0, 3, 4))` reduced over `axis=(1, 2)`; NumPy's `_ureduce` merges the two axes and fails the reshape.

| Call | NumPy 2.4.2 | NumSharp | Prologue that answered |
|---|---|---|---|
| `median(float64)` | ValueError: cannot reshape array of size 0 into shape (0,newaxis) | a `(0,)` float64 array | `QuantileEngine.Compute` |
| `median(U3)` | the same ValueError | a TypeError from the closing mean's add.reduce | `np.median`'s text route |
| `quantile(float64, 0.5)` | the same ValueError | a `(0,)` float64 array | `QuantileEngine.Compute` |
| `quantile(U3, 0.5)` | the same ValueError | the same ValueError | `QuantileText.Compute` |

### B2. np.angle as NumPy's own statement

`np.arctan2(0, z)` with a weak C# literal 0 (NumPy's `angle` body for a real `z`) against NumPy's `np.angle(z)`. Inputs
`[-2, 0, 3]` cast to each integer dtype, `[True, False, True]`, and `[-2.0, -0.0, nan]` with the positive NaN of Python.

| Input | Result |
|---|---|
| bool, int8, uint8, int16, int32, int64, float16, float32, float64 | dtype and bytes equal to NumPy's `np.angle`, byte for byte |
| U | NumPy's "not supported for the input types" text, raised by the engine's arctan2 guard |
| StringDType | NumPy's no-loop text naming `_PyLongDType` and `StringDType`, raised by the engine's arctan2 guard |

Trap met: .NET's `double.NaN` is the negative quiet NaN (`0xfff8...`) and Python's `np.nan` the positive one; a probe
that feeds `double.NaN` shows a NaN-sign divergence that is not there.

### B3. The boundary depth reaches user code (input to P7)

Inside `np.apply_along_axis(func1d, 1, matrix)`, the callback's `np.sum(other_matrix, 0)` is a plain `(2,)` array
(NumPy: a 1 × 2 matrix) and `np.ravel(chararray)` a plain array (NumPy: a chararray). Outside the callback, and with a
plain operand, NumSharp gives NumPy's classes. Cause: `NDMatrixBoundary.Wrapped` raises the per-thread depth around the
whole computation, the user's callback included.

### B4. The cost of a plain view instead of the depth (input to P7)

Two P-cores, fastest of 25 batches of 20,000 calls: a plain view of a chararray, created and disposed, 287 ns;
`np.ravel(chararray)` today 563 ns, by substitution (plain view, plain ravel, chararray view of the result) 708 ns.

---

## Appendix C. Measurement commands

Run from the worktree root. Comment lines are excluded by `grep -v -E "^\S+:[0-9]+:\s*//"`.

```bash
# The np.* layer: src/NumSharp.Core without Backends/, Char/, APIs/np.strings*, APIs/np.char*
LAYER='grep -v src/NumSharp.Core/Backends/ | grep -v -E "src/NumSharp.Core/Char/|src/NumSharp.Core/APIs/np\.(strings|char)"'
NOCOMMENT='grep -v -E "^\S+:[0-9]+:\s*//"'

# Text dtype tests: calls into Backends/Strings
R='NDStringUfuncs\.|NDVString[A-Za-z]*\.|NDString[A-Z][A-Za-z]*\.|Backends\.Strings\.'

# Numeric dtype tests (heuristic, a lower bound: it misses dtype comparisons written another way)
P='(typecode|GetTypeCode|\.typecode\(\))\s*(==|!=|is)\s*NPTypeCode\.|switch\s*\([a-zA-Z_.]*(typecode|GetTypeCode)\)|IsFloatTc\(|\.IsInteger\(\)|IsComplex|NPTypeCode\.(Complex|Decimal|Half|Boolean|Char)\b'

grep -rn --include=*.cs -E "$R" src/NumSharp.Core | eval "$LAYER" | eval "$NOCOMMENT" | wc -l           # 535
grep -rn --include=*.cs -E "$P" src/NumSharp.Core | eval "$LAYER" | eval "$NOCOMMENT" | wc -l           # 1,287
grep -rn --include='np.*.cs' -E "$P" src/NumSharp.Core | eval "$LAYER" | eval "$NOCOMMENT" | wc -l      # 440
grep -rn --include='np.*.cs' -E "$R" src/NumSharp.Core | eval "$LAYER" | eval "$NOCOMMENT" | wc -l      # 398
grep -rn --include='NDArray.*.cs' -E "$P" src/NumSharp.Core | grep -v src/NumSharp.Core/Backends/ | eval "$NOCOMMENT" | wc -l  # 165

# Engine reduction and scan members, and how many take out=
grep -h -E "public (abstract|virtual) [A-Za-z<>]+ (Reduce[A-Za-z]*|Sum|Mean|AMax|AMin|ArgMax|ArgMin|NanSum|NanProd|NanMax|NanMin|CountNonZero|All|Any)\(" \
  src/NumSharp.Core/Backends/TensorEngine*.cs | awk '{ if ($0 ~ /@out/) o++; n++ } END { print n, o }'   # 29 1
```

The np.* calls made from inside the engine were counted per call, comment lines skipped, by a script that splits them
into structural / creation calls (`zeros`, `empty`, `copyto`, `broadcast_to`, `getbufsize`, the dtype names, the
promotion helpers, ...) and computations (everything else).
