# Dtype logic out of the np.* layer

Living plan and state document. Branch `strings`, measured at `2e872477`. Started 2026-10-24.

**Task (verbatim):** "So if we were to agree that we need to move all logic out of np.* into deeper where dtype handling
resides or even inside tensorengine (anywhere inside) Where are the challenges or can we simply move all with no issues?
Write a plan covering how to and answer my questions"

**State:** P0 done (2026-10-24): the two ratchets, the snapshot tool and a three-family pilot (`np.cumsum` / `np.cumprod`,
`np.median`, `np.angle`), each move byte-identical on the snapshot. The owner answered the decisions of
[§5](#5-decisions-for-the-owner); the bug the pilot found got its fix commit (`7a386fdd`, D4). P1a done (2026-10-24): the
quantile family's bodies in `QuantileEngine` (`3bce5a9f`), NumPy's one `_ureduce` prologue in every route (`a726b671`,
Appendix B1 fixed), and the two bugs that work found, fixed per D4 (`1e8eca9e`, `8c2cdbef`). P1b done (2026-10-24) for
the seven functions it named; its NDMatrix.Reductions item turned out to need the reduction members' `out=` and is
now P1d. P1c done (2026-10-24): the six operator files hold no dtype line, two of the moves fixing a divergence each
(`a6b25a7e`, `f37f3f17`). P1d is next. Progress goes into the [§9 state log](#9-state-log), newest entry first.

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

The owner's answers (2026-10-24), and what they mean for the work:

| # | Answer | Applied as |
|---|---|---|
| D1 | "look into it and decide when time comes to impl" | Decided per family when it moves. The pilot: `np.cumsum` / `np.cumprod` widen a `TensorEngine` member (NumPy does them in C); `np.median` moves into the existing internal class `QuantileEngine`; `np.angle`, a NumPy Python function, becomes the internal static `DefaultEngine.Angle`. |
| D2 | yes | Widen in place: `ReduceCumAdd` / `ReduceCumMul` gained `out=` (`fe18431f`), listed under Breaking Changes. |
| D3 | "idk, our code is healthy, use it as reference" | The current behavior is the reference: every move is held to NumSharp's own snapshot before and after, and a batch is whatever unit keeps that true (a function's whole body, text and numeric together). |
| D4 | yes, small bugfix commits | A divergence found on the way gets its own commit (the pilot: `3f88d53f`). |
| D5 | yes, on `strings` only | R1 / R2 run on `strings`. |
| D6 | `strings` | Everything lands on `strings`. |
| D7 | a TODO after the last phase | The class boundary stays in the facades until P7. |

The recommendations as written before the answers:

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

- [x] **R1, facade dtype ratchet** (`4d73fcfc`). A source-walking test modeled on `NativeAllocationChokepointTests` (same project,
      same walk): the text pattern and the numeric pattern of [Appendix C](#appendix-c-measurement-commands) over the
      facade files (`np.*.cs` and `NDArray.*.cs` outside `Backends/`, `Matrixlib/NDMatrix*.cs`), with an allowlist of
      file → count. A new file or a higher count fails and prints the line. Allowed C#-only conversions carry a reason.
      The patterns are a heuristic (they miss a dtype comparison written another way); an analyzer rule beside the
      NDW rules of `NumSharp.Build` can replace them once the counts are small.
- [x] **R2, up-call ratchet** (`4d73fcfc`). The same over `Backends/**`: np.* calls per file, computations and structural calls
      counted apart.
- [x] **S, the snapshot tool** (`0b179445`) (`tools/facade-move/`, a `dotnet run` script). For a list of np.* names it reflects
      every public overload and calls it over a generated matrix: the 15 dtypes and S / U / StringDType × the layouts
      C, F, strided, reversed, broadcast, 0-d, empty × each optional argument's variants, invalid ones included (an
      axis out of range or repeated, an `out` of the wrong shape, dtype or writeability, a refused `dtype=`). It writes
      one outcome per call: dtype, shape, flags and a SHA-256 of the bytes, or the exception type and message. Run on
      the parent and on the child; the diff of a move commit is empty.
- [x] **Pilot.** One family of each easy kind, to prove the method before it scales: `np.median` (K1, `0e77d200`;
      `np.quantile` went to P1a, see the state log), `np.cumsum` / `np.cumprod` (K2, `ReduceCumAdd` / `ReduceCumMul`
      widened with `out=`, `fe18431f`), `np.angle` (K3, moved as it is, `21a6cff3`; NumPy's own statement is not a
      drop-in, see the state log).

### P1. The deeper place exists (171 text lines, plus the numeric lines of the same functions)

- [x] **P1a, the quantile family** (K1, 108 lines in 10 files; 6 functions, 32 public overloads). Step 1: the switch
      (StringDType, fixed width, numbers) as `QuantileEngine.Compute`'s first statement, `Compute` gaining `string op`
      for the texts that name the function. No behavior change. Step 2, its own commit: one shared prologue in NumPy's
      order (`_ureduce`: the axes, the keepdims `out` target, the axis merge; then q's rank; then the method), then the
      family core; this fixes the divergence of Appendix B1. The facade keeps only the class statements and the
      normalization of its C# overloads (`q` as `double`, `double[]`, `long` or `NDArray`); q's checks move with the
      prologue (C1); the `chararray` flag crosses into O as a parameter (C8). Done: `3bce5a9f` (step 1, which put the
      bodies in `QuantileEngine.Quantile` / `NanQuantile` / `NanMedian` rather than in `Compute`), `a726b671` (step 2),
      and the D4 fixes `1e8eca9e` and `8c2cdbef` ([§9](#9-state-log)).
- [x] **P1b, reductions with a narrower member** (K2, 57 lines in 10 files; `np.cumsum` / `np.cumprod` done in the
      pilot). Widen `All`, `Any` and `CountNonZero` (`keepdims=`, `out=`, `where=`), and the members `np.trace` and
      `np.ptp` reduce through; move the bodies of `np.nancumsum`, `np.nancumprod`, `np.trace`, `np.ptp`, `np.any`,
      `np.all`, `np.count_nonzero`; route the 14 lines of `NDMatrix.Reductions.cs` through the same members. Done for
      the seven functions ([§9](#9-state-log)); the NDMatrix item is P1d.
- [x] **P1c, operators** (K6, 6 lines in 6 files). Each operator now calls the member its NumPy twin is: `negative()`
      `TensorEngine.Negate`, `positive()` and the unary `+` `TensorEngine.Positive`, `!` np.logical_not, `==` / `!=`
      the internal `DefaultEngine.RichCompareEquality` (array_richcompare). Done ([§9](#9-state-log)).
- [ ] **P1d, the reduction members' `out=`** (found by P1b). `NDMatrix.Reductions.cs`'s 15 text and 6 numeric lines are
      not in its any / all / ptp members (those hold none) but in the `out=` paths of `matrix.sum` / `prod` / `mean` /
      `var` / `std`, which implement NumPy's reducelike rule themselves (`ReduceTextInto`, `LoopWith`,
      `StoreVariance`): the loop is `promote_types(out, a)` without the small-integer widening, text resolves on
      (out, a), the result is stored with an unsafe cast. No engine member carries that: `ReduceProduct`, `ReduceMean`,
      `ReduceVar` and `ReduceStd` take no `out=`, and `ReduceAdd`'s `out=` keeps the NEP 50 accumulator (no public
      path reaches it with an out today: np.sum has no `out=` keyword). So the step is: give those members NumPy's
      `out=` (D2), route the matrix members through them, and let np.sum / np.prod / np.mean / np.var / np.std expose
      `out=` (the api-parameter-parity items).

### P2. Compositions above the engine (K3, 150 lines in 43 files)

For each function: first move the body as it is into an operation class (a pure move, snapshot-identical). Then try
NumPy's own composition through engine calls, as its own commit, with the snapshot as the referee: where it gives the
same answers, the error replay goes away; where it does not, the snapshot names an engine divergence the composition
was working around, which gets its own fix first (D4). The pilot's `np.angle` is the second case (state log). Order by size: `np.histogram` 16, `np.histogramdd` 9, `np.trapezoid` 8,
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

- **2026-10-24, P1c (operators) done.** Commits on `strings`, developed in a scratch worktree and landed by
  fast-forward; every move is held to the snapshot of the six operators (2,672 calls: NDArray.negative,
  NDArray.positive, unary `+`, `!`, `==`, `!=`, the second operand varied):
  - `96df5d99` the snapshot tool reaches NDArray's operators (a target `NDArray.op_X`) and varies a second array
    operand (its copy, int32 and float64 ramps, `U5` and StringDType text, a 0-d, a broadcast row, a shape that does
    not broadcast), which reaches the failed-comparison rule of `==` / `!=`.
  - `a6b25a7e` (D4) `NDArray.negative()` is `TensorEngine.Negate`, np.negative's member: its own per-dtype loop threw
    for unsigned and char inputs, where NumPy's negative loops wrap (uint8 `[1, 2, 255]` is `[255, 254, 1]`, probed),
    and returned an F-ordered copy of a broadcast operand (NumPy: C-ordered). 64 snapshot lines change, all of them
    those two classes. R1 -1 text / -15 numeric.
  - `f37f3f17` (D4) the unary `+` and `positive()` are `TensorEngine.Positive`, np.positive's member: `+` of a bool
    array returned a bool copy where NumPy's positive ufunc has no bool loop (its TypeError, probed for (2,), (0, 3)
    and 0-d operands), and `+` of an F-ordered array returned a C-ordered copy (NumPy's ufunc output is F). 27 snapshot
    lines change, all the operator's. R1 -2 text / -1 numeric.
  - `e322eb1d` `!` is np.logical_not for every operand; its raw scalar loop for C-contiguous operands and its text
    guard go (pure move; the comparison kernels make it 3.8x-43x faster from 100K elements: float64 43x / 24x, bool
    10x / 11x-16x, int32 5x / 3.8x-4.1x at 100K / 1M; the commit's summary line says 5x-43x, which its own int32 1M
    cell contradicts). R1 -1 text.
  - `33c4807d` `==` / `!=`: NumPy's array_richcompare (the ufunc, or `_failed_comparison_workaround`'s constant where
    a string operand has no loop) moves into the internal `DefaultEngine.RichCompareEquality` (D1: an operator
    protocol around a ufunc, so no TensorEngine member); the operators keep C#'s null and uninitialized-array answers.
    R1 -4 text.
  - Left in `Operations/Elementwise`: `NDArray.Equals.cs`' 5 numeric lines (array_equal's `CanHoldNaN` switch, P5) and
    `NDArray.TextComparison.cs`' 1 text line (how a C# string becomes the comparand of `==` with a text array, a value
    entering storage: P3).
  - At `33c4807d`: NumSharp.Tests net10.0 20,462 passed / 11 skipped and net8.0 20,461 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed (Layering 6 / 6).
- **2026-10-24, P1b (reductions with a narrower member) done for its seven functions.** Commits on `strings`, developed in
  a scratch worktree and landed by fast-forward; every move is held to the snapshot (78,450 calls over np.all, np.any,
  NDArray.all, NDArray.any, np.count_nonzero, np.trace, NDArray.trace, np.ptp, np.nancumsum and np.nancumprod;
  23,136 more over np.trace / np.diagonal for the diagonal move), byte-identical each time:
  - `6da76c63` the snapshot tool varies `where=` (six masks built from the operand, two NumPy refuses) and the int
    `offset` / `axis1` / `axis2` of np.trace and np.diagonal.
  - `0349955e` np.all / np.any (D1: ufunc.reduce, C → engine members). `TensorEngine.All` / `Any(nd, int? axis,
    keepdims)` widened in place (keepdims, a null axis for every axis), the tuple members made abstract, and two new
    members carry NumPy's whole call (axis, out=, keepdims, where=): DefaultEngine's LogicalReduceNumPy, the facades'
    two bodies merged on `reduceAll`. R1 -4 text / -1 numeric; R2 pins the out= store (np.copyto).
  - `a7ab4d99` np.count_nonzero: the empty-array statements into `DefaultEngine.CountNonZero` (the engine already began
    with the facade's text stand-in). R1 -1 text.
  - `199feaaf` np.trace (D1: PyArray_Trace, C → `TensorEngine.Trace`) and np.diagonal's view (`TensorEngine.Diagonal`,
    PyArray_Diagonal), which the moved trace reads without calling up. R1 -7 text / -7 numeric; R2 pins the
    np.ascontiguousarray of the diagonal and the out= store.
  - `e23fa183` np.ptp (D1: `_methods._ptp`, Python → the internal `DefaultEngine.Ptp`, as np.angle's), its np.amax /
    np.amin calls now the engine members they reached (`ReduceAMax` / `ReduceAMin`). R1 -7 text; R2 pins the store.
  - `ea479759` np.nancumsum / np.nancumprod (D1: Python → internal `DefaultEngine.NanCumSum` / `NanCumProd`), their
    np.cumsum / np.cumprod / np.isnan calls now `ReduceCumAdd` / `ReduceCumMul` / `IsNan`. R1 -12 text / -7 numeric; R2
    pins np.where (no engine select exists yet).
  - `01b898c9` the facades the moves reduced to one call document every exception it raises.
  - **Not done here: NDMatrix.Reductions.** Its dtype lines are the `out=` paths of matrix.sum / prod / mean / var / std,
    which need the reduction members' `out=` with NumPy's reducelike loop rule (the plan assumed they sat in any / all
    / ptp): now step P1d. Found on the way: `ReduceAdd`'s `out=` keeps the NEP 50 accumulator where NumPy's add.reduce
    with an out and no dtype= runs `promote_types(out, a)` (NumPy: an int8 [100, 100, 56] summed into a bool out is
    False, the int8 sum wrapping to 0; the engine member stores True, probed). No public path passes an out to it
    today (np.sum has no `out=`), so it is P1d's to fix, not a D4 commit.
  - The engine up-calls the moves pinned (R2, with reasons): the out= stores through np.copyto (all, trace, ptp), the
    np.ascontiguousarray before trace's sum, np.where in the NaN scans. P6 replaces them.
  - Every commit: NumSharp.Tests net10.0 20,460 passed / 11 skipped and net8.0 20,459 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed (Layering 6 / 6).
- **2026-10-24, P1a (the quantile family) done.** Commits on `strings`, developed in a scratch worktree and landed by
  fast-forward:
  - `3bce5a9f` step 1, the pure move: the bodies of np.quantile, np.percentile, np.nanquantile, np.nanpercentile and
    np.nanmedian into `QuantileEngine.Quantile` / `NanQuantile` / `NanMedian` (`Statistics/QuantileEngine.Functions.cs`,
    q carried as `QuantileArgument`), statement for statement; the facades keep only the class rules. 340,944 snapshot
    calls identical; R1 -85 text / -3 numeric lines, six facade files out of the allowlist.
  - `a726b671` step 2, the shared prologue (Appendix B1): `QuantileEngine.UReduce` (`Statistics/QuantileEngine.UReduce.cs`)
    is NumPy's `_ureduce` up to the reduction function — the axes, keepdims' out index, the merge of several reduced
    axes (data and weights alike) — and every route calls it first: the numeric engine, the fixed-width text route, the
    three StringDType statistics, the weighted branch, the nan forms' `isnan` refusal, the two text-q answers, np.median's
    text-out route and its chararray path. The copies it replaced disagreed: numeric median / quantile merged with an
    explicit length after q's rank and the method (B1's data gave an empty result), and the text routes refused before
    keepdims' out index. A 298-call NumPy probe: 166 matching before, 287 after, the 121 changes all toward NumPy; the
    snapshot (340,944 calls) is byte-identical, so nothing outside the prologue moved. New oracle section `qprologue`
    (5,640 cases, strict; 1,217 fail on the parent) and 7 unit tests. R1 -1 text line (np.median.cs), R2 -1 structural
    call (NDVStringStatistics.Quantile.cs).
  - `1e8eca9e` the first D4 fix, found by the step-2 probe: np.nanquantile / np.nanpercentile with a q of two or more
    dimensions over merged axes. NumPy's `_nanquantile_ureduce_func` moves q's axes with
    `from_ax = [axis + i for i in range(q.ndim)]` on `_ureduce`'s UNnormalized `axis=-1`, so
    `np.nanquantile(arange(24.).reshape(2, 3, 4), q of shape (3, 2), axis=(1, 2))` is `(2, 2, 3)` where np.quantile gives
    `(3, 2, 2)`; NumSharp gave np.quantile's order, for numbers and StringDType alike, and an `out=` had to take it.
    `UReduced.NumPyAxis` carries the axis as NumPy hands it on, and `AlongAxisLayout` follows NumPy's statements
    (apply_along_axis' buffer, then the move). 686 new `qprologue` cases (part three; 145 fail on the parent); the
    snapshot changes on exactly the 16 lines that reach it; 2 unit tests.
  - `8c2cdbef` the second D4 fix, found while probing the first: the nan forms over data that is NaN everywhere. NumPy's
    `_nanquantile_1d` answers a slice with nothing left by `np.full(q.shape, np.nan, dtype=arr1d.dtype)[()]` before any
    check (a 3-D q and an unknown method are never looked at): Python's POSITIVE NaN, a view for a q array (a scalar for
    a 0-d q), stacked and moved along an axis, and the store `out[...] = result` broadcasts. NumSharp's early answer for
    a 3-D q or an unknown method filled .NET's negative `double.NaN` into an owning q-first buffer and required an out of
    exactly the result's shape; the kernel route returned an owner over the flattened data. 732 new `qprologue` cases
    (part three's all-NaN data and part four; 321 fail on the parent); 1 unit test; the snapshot identical.
  - **Left as they were, with the reason.** np.median / np.nanmedian keep their result-then-copy `out=` rule where the out
    passes keepdims' index: NumPy writes through `np.mean(part, axis, out=out)`, and NumSharp's np.mean has no `out=`
    (the engine's `Mean` member lacks it, C2; P1b widens `All`, `Any` and `CountNonZero` only). Numeric np.quantile over
    an empty reduced axis raises `IndexOutOfRangeException` with NumPy's IndexError text (the house type).
  - NumSharp.Tests net10.0 20,457 passed / 11 skipped and net8.0 20,456 / 11 at `a726b671`, 20,460 / 11 and 20,459 / 11
    at `8c2cdbef`; Oracle net10.0 and net8.0 411 / 2 skipped, 0 failed (the Layering ratchets included).
- **2026-10-24, the D4 fix after P0.** Commits on `strings`, landed by fast-forward:
  - `7a386fdd` the empty-operand early return the K3 finding named, in `np.arctan2` and the six float-tier binary
    ufuncs that shared it (logaddexp, logaddexp2, nextafter, copysign, hypot, heaviside): an empty operand now gets
    NumPy's loop dtype and broadcast shape, or NumPy's error for shapes that do not broadcast (they returned a clone of
    the operand). New helper `DefaultEngine.EmptyBinaryResult`. The relayout gap RL07 (the logaddexp half, carved on
    2026-10-07) is closed and its 40 cases are back in `relayout.jsonl`. The finding's other half, a 0-d operand with a
    C# literal promoting as if both were weak, is X24 and stays out of this plan.
  - `9972006f` found on the way: the relayout generator took the matrix-operand tiers as sources, so a regeneration
    re-drew thousands of variants from other source cases without adding a cell; it skips them now. A full
    regeneration of the relayout tiers on strings' corpus (-17 / +20 variants, from tiers redrawn after it) is left as a
    separate step.
  - NumSharp.Tests net10.0 20,449 passed / 11 skipped and net8.0 20,448 / 11; Oracle net10.0 410 / 2 skipped, 0 failed;
    the relayout gaps still diverging: 35 / 35 (RL01-RL03).
- **2026-10-24, P0 done.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward:
  - `4d73fcfc` the ratchets. Their scanner is literal-aware (comments and string contents blanked) and decision-focused,
    so its counts differ from the inventory's grep: R1 starts at 590 text lines (137 files) and 1,131 numeric lines (136
    files), because it also counts `NPTypeCode.String` tests (84 lines the inventory's pattern missed) and every case arm
    of a dtype switch; R2 at 132 computations and 264 structural calls in 83 files (it also counts module calls such as
    `np.strings.*`).
  - `3f88d53f` a bug the snapshot tool found on its first run, fixed per D4: `np.cumsum(uint64 (3, 4),
    dtype=complex128)` ended the process with an access violation. The flat scan's converting IL kernel combined a
    float16 / complex128 accumulator with a plain add / mul (invalid IL for float16, the raw bits read as 16-byte complex
    values past the input for complex128), and the axis path refused the conversion. Now NumPy's cast-first, byte-exact
    over a 144-case probe; the scan tier gained 1,440 `dtype=` cases (it had none, which is how the bug survived), and the
    surface gaps SU38 (this bug) and SU39 (fixed by `f44c9579`, whose bookkeeping had been left behind) left the carve.
  - `0b179445` the snapshot tool, with a pairwise phase added after the first runs: an `out=` sized for one `axis`, a
    `method` with an array `q`, are where a moved prologue can reorder checks. Two runs of one build are byte-identical.
  - `fe18431f` K2: `np.cumsum` / `np.cumprod` into widened `ReduceCumAdd` / `ReduceCumMul` (D2). 36,648 snapshot calls
    identical.
  - `0e77d200` K1: `np.median` into `QuantileEngine.Median`. 340,944 quantile-family calls identical. Scope change: the
    pilot named `np.median` / `np.quantile`, but in `np.quantile` (and percentile, nanquantile, nanpercentile) the
    chararray class decision sits between NumPy's argument checks and the weights branch, in 8 overloads with four forms of
    `q`: moving it needs P1a's shared prologue, so it went there. The class decision can run first without changing
    anything (it is a pure type test, and the Kept rule re-enters the function, whose checks then run), which is the
    shape P1a will use.
  - `21a6cff3` K3: `np.angle` into `DefaultEngine.Angle`, its `np.arctan2` / `np.multiply` calls into the engine members
    they forward to, `AngleRealTier` with it (the fused angle node reads it there). 324 calls identical.
  - **K3 finding.** NumPy's own statement `arctan2(0, z)` (a weak 0) was tried in place of the moved composition, as
    Appendix B2 suggested. The snapshot rejected it: 162 of 324 lines changed, because the engine's `arctan2` diverges
    where the composition worked around it. Probed through the public API: `np.arctan2(0, empty int8)` returns int8
    (NumPy float16; int32 and bool likewise keep their dtype; `DefaultEngine.ATan2` short-circuits an empty operand with
    `return y.Clone()`, the same pattern as the relayout gap RL07 of logaddexp / logaddexp2), and a 0-d operand with a C#
    literal promotes as if both were weak (`np.add(0-d float32, 1.5)` is float64, NumPy float32: the known weak-scalar
    item X24). Appendix B2's probe used 3-element arrays only, which is why it did not show. Lesson for P2: a composition
    rewrite is its own commit, after the pure move, refereed by the snapshot (§7 P2 says so now).
  - Every commit: NumSharp.Tests net10.0 20,445 passed / 11 skipped and net8.0 20,444 / 11; Oracle net10.0 410 / 2
    skipped (net8.0 too for `3f88d53f`), 0 failed.
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

**Fixed** by `a726b671` (P1a step 2): every route runs `QuantileEngine.UReduce`, and all four calls raise NumPy's
ValueError; the oracle section `qprologue` gates the matrix around it.

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
