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
(`a6b25a7e`, `f37f3f17`). P1d done (2026-10-24): `ReduceAdd`, `ReduceProduct`, `ReduceMean`, `ReduceVar` and `ReduceStd`
carry NumPy's `out=` (the reducelike loop rule, the reduce prologue, `_mean` / `_var`'s statement order),
numpy.matrix's reductions hold no text line, and six divergences found on the way got their fix commits (D4); the public
np.sum / np.prod / np.mean / np.var / np.std keep no `out=` keyword until the owner's one-signature decision. P2 is
next. Progress goes into the [§9 state log](#9-state-log), newest entry first.

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
classes in `Polynomial/Package/`) stay where they are, because moving files only adds conflicts. **Correction
(2026-10-25, P2):** an operation class whose statements call np.* functions (every P2 composition: NumPy's Python
calls NumPy's functions) sits beside its facade instead (`Statistics/HistogramEngine`, `Math/TrapezoidEngine`), because
ratchet R2 counts np.* calls against every file under `Backends/` and would grow; `Backends/Default/<Family>/` holds the
ones that compose engine members only.

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
- [x] **P1d, the reduction members' `out=`** (found by P1b). `NDMatrix.Reductions.cs`'s 15 text and 6 numeric lines are
      not in its any / all / ptp members (those hold none) but in the `out=` paths of `matrix.sum` / `prod` / `mean` /
      `var` / `std`, which implement NumPy's reducelike rule themselves (`ReduceTextInto`, `LoopWith`,
      `StoreVariance`): the loop is `promote_types(out, a)` without the small-integer widening, text resolves on
      (out, a), the result is stored with an unsafe cast. No engine member carries that: `ReduceProduct`, `ReduceMean`,
      `ReduceVar` and `ReduceStd` take no `out=`, and `ReduceAdd`'s `out=` keeps the NEP 50 accumulator (no public
      path reaches it with an out today: np.sum has no `out=` keyword). So the step is: give those members NumPy's
      `out=` (D2), route the matrix members through them, and let np.sum / np.prod / np.mean / np.var / np.std expose
      `out=` (the api-parameter-parity items). Done ([§9](#9-state-log)) but for the last part: the facades' `out=`
      keyword is a signature change, which waits on the owner's one-signature decision.

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

- **2026-10-27, P2's eleventh family: np.isin moved into `IsinEngine`, its three paths NumPy's.** Commits on `strings`,
  developed in a scratch worktree and landed by fast-forward. The referees: the snapshot (np.isin, 5,202 calls),
  `twin_isin.py` (the tool's operands, test sets and kinds rebuilt in NumPy: dtype, shape and SHA-256 per result, or the
  exception's type and text), `ab_isin.py` (parent against fix, each changed line classified by the twin), and NumPy
  probes per fix, written into the unit tests:
  - `ed738514` the snapshot tool reaches np.isin's test set and kind (two roles, given only to np.isin's parameters: np.sort,
    np.argsort and np.partition name a parameter kind too): test_elements as the operand's own values and sets chosen by
    NumPy's path decision (int32 / float64 ramps, U5 / StringDType / S5 text, a 0-d, an empty set, narrow and wide
    integer ranges, NaN / signed-zero / infinite floats, uint64 past 2**63, a complex set, a broadcast set, a stepped
    view), kind as NumPy's names and two it refuses ("x", and "Sort": the check is case-sensitive). 5,202 calls where the
    function was one SKIP line. Before any fix the twin agreed on 5,034 of the 5,069 comparable lines (133 decimal lines
    have no NumPy twin).
  - `36dcf448` the move (D1: NumPy writes isin in Python, `numpy/lib/_arraysetops_impl.py`): the body and the helpers into
    `Manipulation/IsinEngine.cs`, statement for statement, and `np.isin.VString.cs` into `IsinEngine.VString.cs`; the
    facade keeps its signature, documentation, class boundary and one call. 5,202 calls byte-identical; R1's two rows go
    (np.isin.cs 3 text and 38 numeric, np.isin.VString.cs 15 text and 1 numeric).
  - `51d51d58` (D4) a char array is NumPy's uint16: `kind='table'` takes it (NumSharp refused it as not an integer
    array), and a char set read by the StringDType route holds numbers, which no text equals
    (`np.isin(StringDType ['a', 'zz'], char ['a', 'b'])` is `[False, False]`; it was `[True, False]`). The char lead of the
    searchsorted entry. 5 lines, all fixed.
  - `af32cde3` (D4) `kind='table'` over a set whose range is `2**63 - 1` or more raises NumPy's `ValueError: Maximum
    allowed dimension exceeded` (its table allocation, an empty element too), after the RuntimeError of a range unsafe
    for the set's dtype. 21 lines, all fixed. A range NumPy can address but not allocate (its MemoryError, 8 EiB) is
    still answered: the documented divergence (`[Misaligned]`).
  - `d5672a4d` (D4) NumPy's sort path runs its statements where they are not the exact membership: a concatenation that
    merges values (a 64-bit integer past 2**53 in float64 or complex128: `np.isin(uint64 [2**64 - 1, 2**64 - 2, 3],
    np.arange(24.))` is `[False, True, True]`) and `assume_unique` (repeated values meet as neighbours; it was a
    documented divergence). The sort statements move out of the text route into `SortPathFlat`, shared. 9 lines, all
    fixed.
  - `276c9676` step 2 (C11): NumPy's composition through public calls (the table method's helper table, masks, compaction
    and fancy gather; the loop path's one comparison pass per set value; the sort statements) against the routes, in one
    process, rounds interleaved, on two pinned P-cores: x0.54-x5.21 NumPy, geomean x1.25, slower than the routes in 23 of 26 cells and within 3% in the three where the routes already are NumPy's statements (U5 text, assume_unique over a permutation, a 64-bit integer past 2**53 beside float64). The table method and the loop path now run as one
    pass each. `TableMembership`: elements of one byte (of two from 8,192 on) read a table over their dtype's whole domain,
    one load each; wider ones NumPy's bool table over the set's range (inverted under invert), one 64-bit range test and
    one load, uint64 included beside an unsigned operand. `LoopProbe`: four element vectors per step against every
    broadcast set value, up to 9 values per vector lane (a crossover sweep over 6 dtypes and 10 set sizes). 5,202 calls
    byte-identical. NPY/NS: geomean x4.21 over 26 cells (x2.14 before): the table method x1.87-x34.7 (100K int32 x0.49 -> x1.87, uint8 x0.48 -> x3.40, bool x4.93 -> x34.7, 1M int32 x1.24 -> x6.07), the loop path x1.74-x6.41 (100K float64 at 16 values x0.90 -> x2.23, 1M float32 at 64 values x3.23 -> x6.41), the sort path's unchanged routes x1.50-x11.6, and assume_unique over presorted floats x0.42 (below). 22 planted bugs, all killed (one survived the first two-byte test, whose elements missed the negative members).
  - Evidence at `276c9676`: the twin agrees with NumPy 2.4.2 on all 5,069 comparable lines (5,034 at `ed738514`).
  - **Found:**
    - The hash membership (`HashMembership`, unchanged by the family) runs 2.5 to 3.5 times slower on one instantiation
      or another, by where its code or data lands: over 100K int64 elements against 1,000 values 620-990 us with the
      default runtime settings and 264-280 us with dynamic PGO or tiered compilation off, and
      `[MethodImpl(AggressiveOptimization)]` moves the slowdown onto int32 (287 -> 933 us) and float32 at 1M elements
      (11.5 -> 15.4 ms). Not changed; the sort-path cells' before / after differences are this, not the step. Open.
    - An AVX2 gather of the range table's bytes measured 1.7x faster than the scalar probe over 100K int32 elements; not
      taken: Intel's GDS microcode mitigation slows gathers on the 6th to 11th generation cores (the take kernels
      rejected a gather for the same CPU dependence).
    - `assume_unique` over 100K presorted floats runs NumPy's statements at 0.42x NumPy: its argsort is a timsort, which
      finds the two sorted runs (270 us), where NumSharp's radix argsort takes 2 ms whatever the order; over a
      permutation of the same values NumPy's argsort takes 15.9 ms and the call runs at x5.17.
  - Left: the `kind='table'` MemoryError divergence (above); the hash membership's placement sensitivity (above).
  - At `276c9676`: NumSharp.Tests net10.0 20,629 passed / 11 skipped and net8.0 20,628 / 11; Oracle net10.0 and net8.0 411 / 2 skipped, 0 failed; the carved-gap replays find every carved case of their 82 gaps still diverging.
  - Next in P2: `np.average` (NDW012 at np.average.cs 148 / 163), then the files of four lines or fewer.
- **2026-10-27, P2's tenth family: np.insert moved into `InsertEngine`, its two branches NumPy's statements.** Commits on
  `strings`, developed in a scratch worktree and landed by fast-forward. The referees: the snapshot (np.insert's twelve
  overloads, 79,584 calls), `twin_insert.py` (the tool's operands, positions and values rebuilt in NumPy: dtype, shape,
  C / F / OWNDATA / WRITEABLE flags and SHA-256 per result, or the exception's type and text), `ab_insert.py` (parent
  against fix, each changed line classified by the twin), and NumPy probes per fix, written into the unit tests:
  - `22eea5c0` the snapshot tool reaches np.insert's positions and values (six roles, given only to np.insert /
    np.delete parameters): obj as a C# int / long, a slice, an int[] / long[] sequence, an index array or a boolean mask;
    values as an array or any object. 79,584 calls where each overload was one SKIP line; np.delete runs too (17,046
    calls; its 21 numeric lines are P5's). Before any fix the twin agreed on 64,559 of the 78,028 comparable lines (1,556
    decimal lines have no NumPy twin).
  - `b9774efb` the move (D1: NumPy writes insert in Python, `numpy/lib/_function_base_impl.py`): the bodies into
    `Manipulation/InsertEngine.cs`, statement for statement; the twelve overloads keep their signatures, documentation,
    class boundaries, null checks and one call. 79,584 calls byte-identical (np.delete's 17,046 too); R1's np.insert.cs
    row goes (7 text, 4 numeric).
  - `f5cc5a7f` (D4) an object value is converted where NumPy converts its values, after the axis and index checks (an
    out-of-range literal, a text that does not parse or a complex into a real array raised its conversion error first),
    and the empty index converts an array or collection value too. 848 lines: 630 fixed, 210 still differing (the next
    fixes), 8 decimal.
  - `a4552248` (D4) the single-index branch on NumPy's `indices.size == 1` for every positions argument (a one-position
    slice or mask took the multi-index branch, which refused two values for one position), and its IndexError prints obj
    (`index [10]`, a mask itself). 2,952 lines: 2,874 fixed, 32 still differing (the next fix), 46 decimal.
  - `f9669a8c` (D4, library) a view assignment (`a[1:3] = v`, `a[:, ::2] = v`, `a[0] = v`) checks its value with NumPy's
    `PyArray_AssignArray` rule before it writes: the value's leading extent-1 axes beyond the region's rank are stripped,
    the error prints the stripped shape, and a strided region raises NumPy's ValueError where np.broadcast_to's
    IncorrectShapeException came out (and refused a value with an extra leading axis, which NumPy assigns). The rule
    moves from np.copyto onto `Shape.AssignmentBroadcastStrides`, its stripped view onto
    `UnmanagedStorage.AliasWithoutLeadingAxes`; R2's UnmanagedStorage.Setters.cs row 3 -> 1. The numpy.ma surface gap
    MS32 had carved 9 cases of this (a non-C operand's setitem text): they return to the gate, 19 stay carved. No
    snapshot line changes (np.insert's routes make no view assignment).
  - `1a11edb5` (D4) the single-index branch assigns its values as that view assignment (`could not broadcast input array
    from shape (2,1) into shape (2,)`; a value with extra leading axes is inserted), and a one-element float index
    converts the values without broadcasting them (the slice's TypeError comes first). 1,261 lines: 1,253 fixed, 8
    decimal.
  - `e3c91f64` (D4) the multi-index branch is NumPy's statements on the positions: the negative shift, the stable
    argsort, the rank shift in the index array's dtype (an unsigned index array is the add's same_kind refusal, an axis
    longer than the dtype holds the Python int's OverflowError), old_mask's bounds over N + numnew with the negative wrap
    (an index below -N wraps, the error prints the shifted index), the values' fancy assignment (MapIter's text, an empty
    index included) and the mask's assignment of the array (a one-slab array broadcasts, the boolean assignment's text
    otherwise; two indices naming one slot keep the later value). The data still moves in one concatenate, in position
    order. 7,936 lines: 7,684 fixed, 38 still differing (a Python list into a StringDType array, below), 214 decimal.
  - Step 2 (C11): NumPy's composition built from the same public pieces (np.empty in the array's order and the
    single-index branch's three view assignments; the multi-index branch's two fancy assignments) against the
    concatenate routes, in one process, rounds interleaved, on two pinned P-cores: slower in 10 of 18 cells (x2.1-x2.35
    at 1K elements, x1.11-x1.64 at 100K, x2.3 and x15 for ten rows and ten columns into a (1000, 1000) array), within 3%
    in the 6 memory-bound cells, faster in 2 small ones (a (5, 6) array by 12%, ten positions into 1,000 elements by
    22%). The concatenate routes stay; no code change. NPY/NS of the routes kept: x1.09-x12.1, geomean x2.89 (the
    composition's: x0.79-x5.19, geomean x2.05).
  - Evidence at `e3c91f64`: the twin agrees with NumPy 2.4.2 on 77,000 of the 78,028 comparable lines (64,559 at
    `22eea5c0`). The 1,028 left: 882 exception texts of Python-list values (NumPy packs each element into the dtype with
    `PyArray_Pack`; NumSharp converts the list first: P3's array coercion) and 146 lines of the documented StringDType ->
    complex cast (NumPy 2.4.2 puts the real part in both parts).
  - **Found:**
    - A view assignment raised np.broadcast_to's text and refused a value with extra leading axes into a strided region:
      fixed library-wide by `f9669a8c`.
    - A Python-list value converts as a whole, library-wide, where NumPy packs each element (probed):
      `np.insert(int8 array, 1, [7, 1000])` stores -24 (NumPy: `OverflowError: Python integer 1000 out of bounds for
      int8`), `[7, None]` inserts only the 7 because `np.asanyarray([7, None])` drops the None (NumPy: an object array,
      and insert's `TypeError: int() argument must be ...`), `[7, 1j]` into float64 stores 0 for the 1j (NumPy:
      `TypeError: float() argument must be a string or a real number, not 'complex'`); `np.asarray(object[], np.int8)`
      raises NotSupportedException and `np.array(object[], np.int8)` ArgumentOutOfRangeException (NumPy: the
      OverflowError). Input to P3's single coercion path.
    - np.nansum / np.nanprod of a one-element view at a non-zero offset (a (1, 1) view at offset 3, `a["3::5"]`) read
      the buffer's first element, flat and along an axis: their size-1 route replaces NaNs with np.where, which reads a
      C-contiguous operand at an offset from the buffer's start (the array-object oracle's gaps VF30 / VA03, carved).
      Not fixed here.
    - One net8.0 Oracle run of `a4552248` aborted with an access violation in
      `DirectILKernelGenerator.NanReduceContiguousAxis<double>` (np.nansum along an axis, reached by
      MatrixOperand_LibraryNeverReachesAMatrixRule through the sum_identity tier). It happened once in about eight
      net8.0 Oracle runs of this family (the rerun passed), and not again alone under guard pages or in a race loop of
      7.25 million np.nansum / np.nanprod calls (matrix operands built as the corpus builds them over C, F, stepped and
      reversed views, every axis, beside a thread that allocates and collects; no crash, no wrong value). Unexplained,
      open.
  - Left: the Python-list values (P3); the StringDType -> complex cast (documented); ten positions into 1,000 elements at
    x1.16 NumPy, under the 1.5x bar (21 concatenate pieces; NumPy's composition gives x1.48).
  - At `e3c91f64`: NumSharp.Tests net10.0 20,621 passed / 11 skipped and net8.0 20,620 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed; the carved-gap replays find every carved case of their 82 gaps still diverging.
  - Next in P2: `np.isin` (VString; the char lead of the searchsorted entry), `np.average`, then the files of four lines
    or fewer.
- **2026-10-27, P2's ninth family: np.array_equal and np.array_equiv moved into `ArrayEqualEngine`, their statements
  NumPy's for every dtype.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The
  referees: the snapshot (np.array_equal, NDArray.array_equal and np.array_equiv, 3,426 calls), `twin_array_equal.py`
  (the tool's operands and second operands rebuilt in NumPy: the bool, or the exception's type and text), and NumPy
  probes per step, written into the unit tests:
  - `34d1531f` the snapshot tool reaches np.array_equiv's second operand `a2` (954 new lines). Before any fix the twin
    agreed on 3,224 of the 3,257 comparable lines (169 decimal lines have no NumPy twin); the 33 left were all
    NDArray.array_equal with `equal_nan=true` and an S / U operand.
  - `282b4167` the move (D1: NumPy writes both functions in Python): the facades' bodies and the member's numeric
    statements into `Logic/ArrayEqualEngine.cs`, statement for statement. np.array_equal's last statement called the
    member NDArray.array_equal, so its `CanHoldNaN` switch (§9's P1c entry had left it for P5) moved with the family.
    3,426 calls byte-identical; R1's three rows go (np.array_equal 6 text, np.array_equiv 3 text, NDArray.Equals.cs 5
    numeric).
  - `69d6b55d` (D4) the member runs np.array_equal's statements: with `equal_nan` a fixed-width text operand meets
    NumPy's isnan refusal (it compared the text), and a StringDType NaN-like NA is a NaN (an array holding one was
    unequal to its own copy). 33 lines change, all to NumPy's answers.
  - `953de977` step 2: one NumPy statement sequence for every dtype through the `==` operator and np.isnan, probed on
    both sides to give the replays' answers; the shapes answer only for a pair `==` has no loop for (C11: NumPy builds
    the all-False array there). `NDStringUfuncs.EqualHasStringLoop` is that rule for the operator's
    `FailedComparison` too, and `EqualityScan.TryAllEqual` declines text, which the old routes had kept away from its
    byte scan (S2 against S3 share a typecode: the scan would compare text against padding and read past a narrower
    buffer). 3,426 calls byte-identical; the StringDType NaN path 2.2x faster without its gather, no cell slower.
  - Evidence at `953de977`: the twin agrees with NumPy 2.4.2 on all 3,257 comparable lines (3,224 at `34d1531f`).
  - **Found:** a text operand reached the fused equality scan's byte loop only by the routing order; nothing reachable
    computed a wrong answer, but the step-2 composition needed the decline in the kernel itself.
  - Left: nothing in the snapshot; NumPy's `asarray` of a Python object (a list, None) is the C# conversion to NDArray,
    and a null is C#'s convention (two nulls equal, a null and an array not).
  - At `953de977`: NumSharp.Tests net10.0 20,612 passed / 11 skipped and net8.0 20,611 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed.
  - Next in P2: `np.insert`, then `np.isin` (VString; the char lead of the searchsorted entry), `np.average`, then the
    files of four lines or fewer.
- **2026-10-26 / 27, P2's eighth family: np.linalg moved into `LinalgEngine`, np.dot into `TensorEngine.Dot`, their
  statements NumPy's.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees:
  the snapshot (np.linalg's 35 overloads, 65,876 calls per mode: managed, and with the bundled OpenBLAS at one thread;
  70,040 once np.dot and NDArray.dot joined it), `twin_linalg.py` (the tool's operands and arguments rebuilt in NumPy with
  OPENBLAS_NUM_THREADS=1, so NumPy's own scipy-openblas build runs at the thread count the backend uses: dtype, shape,
  C / F / OWNDATA / WRITEABLE flags and SHA-256 per result, a NumPy scalar's own flags, or the exception's type and text),
  `ab_linalg.py` (parent against fix, each changed line classified by the twin), and NumPy probes per fix, written into
  the unit tests:
  - `289c0254` the snapshot tool reaches np.linalg: pools of square matrices in six layouts, a stack, positive definite,
    singular, rectangular and empty matrices; b, multi_dot's array lists, every ord / p, UPLO, qr's mode, matrix_power's
    n, ind, the axis lists, the nullable tolerances; `SNAPSHOT_OPENBLAS=1` installs the bundled backend at one thread.
    Before any fix the twin agreed on 53,714 of the 63,034 comparable lines of the backend snapshot.
  - `547e512c` the move (D1: NumPy writes numpy.linalg in Python around the `_umath_linalg` gufuncs): the bodies into
    `LinearAlgebra/linalg/LinalgEngine*.cs`, an operation class beside the facades, one partial per facade file, statement
    for statement; the gufuncs stay TensorEngine members. 65,876 calls byte-identical in both modes; R1's seven np.linalg
    rows go.
  - `9cf9d74b` (D4) det / slogdet of one matrix and lstsq's rank are NumPy's scalars: the gufunc's return step, in the
    TensorEngine members that are the gufuncs. 1,221 lines change: 1,220 fixed.
  - `42c5c08d` (D4) norm / vector_norm refuse a string order before the axis is read. 520 lines: 516 fixed, 4 decimal.
  - `b4a0ff84` (D4) the keepdims norms are NumPy's reshapes, writeable views (`LinalgEngine.ReshapeLikeNumPy`). 732 lines:
    672 fixed, 9 decimal, 51 left (a 0-d input's keepdims norm: VF01).
  - `e6decc72` (D4) the general vector order is NumPy's four statements (a float16 / float32 operand keeps its dtype; a
    0-d sum is raised with scalarmath's pow, `ScalarPower`). 548 lines, all fixed.
  - `61898534` (D4) cond's inverse-based orders: the inverse in float64 / complex128 ignoring errors
    (`InvIgnoringErrors`: a singular matrix's NaN inverse, so inf inside a stack too), both norms evaluated, the scalar.
    275 lines: 271 fixed.
  - `3f4fbf27` (D4) tensorsolve's statements (the slice of a.shape, `a.size != prod ** 2`, the in-place shape). 404 lines:
    374 fixed, 15 decimal, 15 char (refused by design).
  - `04dce29f` (D4) tensorinv of a 0-d operand raises `reshape()`'s TypeError after the inverse. 36 lines, all fixed.
  - `2b8a68c5` (D4) norm's |x|^2 is `(x.conj() * x).real`, rounded as simd_cmul rounds it. 11 lines: 7 fixed, 4 VF01.
  - `4753dc94` (D4) linalg.cross is the length check and then np.cross; the surface gap SU2 goes (1 case back). 625
    lines: 587 fixed, 38 decimal.
  - `f89777fc` the surface tier's SU3 and SU4 return to the gate: np.histogram's `98ea7bcf` had closed them, and the carve
    was never redrawn (15 cases back).
  - `013713eb` (D4) norm's fast path ravels in memory order and dots a complex operand's two parts (their sum is
    scalarmath, `PolyNumber.Binary`). No snapshot line changes (its values are small integers); a unit test and a live
    probe of long operands.
  - `dae524d8` (D4) the Frobenius norm over an axis pair is one reduction in NumPy's order (`SumAxisPairOfCOrderSquares`,
    `HalfRunSumsAddedInTurn`, `NDArray.AdoptAxisPermutation`): a probe of 17,560 results all NumPy's (4,026 of 8,780
    before). 2 snapshot lines, both fixed.
  - `319821e4` the np.dot move (D1: `PyArray_MatrixProduct2` is C): `TensorEngine.Dot(x, y, out)` widened in place (D2),
    the text answers and `new_array_for_sum`'s out= check join the product. 70,040 calls byte-identical; R1's np.dot.cs row
    goes.
  - `1e5f8697` (D4) a 0-d operand: the common dtype, multiply's `out=` outside cblas, a 0-d out returned as a scalar. 387
    lines: 333 fixed, 42 to NumPy's values in the text-scalar stand-in, 12 decimal.
  - `7a6ba438` (D4) multi_dot's statements; the out-flags gap VO01 goes (4 cases back). 1,282 lines: 1,203 fixed, 22
    stand-in, 57 decimal.
  - `ee9ce021` (D4) np.dot's N-D route calls the dtype's dot function per output element (complex imaginary parts,
    integer wrap, ddot's strided order); the matrix gap MX10 goes (3 cases back). A 936-call probe 338 -> 860
    byte-identical (the 76 left: float32's sdot, two unit strides past the BLAS vector blocks); 15 snapshot lines, all
    fixed; N-D int64 x0.026 -> x1.12 NumPy.
  - `8b8dc280` (D4) a 0-d text operand's read-only out= is refused before multiply's missing loop. 19 lines, all fixed.
  - `19240760` (D4) svd / pinv / matrix_rank with `hermitian=True` are NumPy's eigh statements. 170 lines, all fixed.
  - `68513a2c` (D4) qr's raw and economic modes return NumPy's objects (the K-order copy factored in place, raw's
    transposed view). 103 lines, all fixed.
  - `0ef1aaec` (D4) matrix_power(a, 0) is `empty_like(a)` filled with the identity (the operand's order), in a plain region
    (`NDMatrixBoundary.PlainFor`): the first run of this change failed the chararray boundary audit, its np.empty_like
    reaching the chararray rule from inside the library. 18 lines: 17 fixed, 1 decimal.
  - `4fbad833` (D4, found by that audit's lead) qr's factored copy in the same plain region: `68513a2c` had made raw /
    economic of a chararray holding numbers raise `Can only create a chararray from string data.` (NumPy returns plain
    float64 arrays). The string corpus has qr only in the default modes, so the audit could not see it. 0 snapshot lines.
  - `66fd68c2` (D4) solve's gufunc checks through `GufuncGuard.CheckCoreDimensions` (signatures `Solve` / `Solve1`):
    stacked axes that do not broadcast get NumPy's remapped-shapes text. 12 lines, all fixed.
  - `65ccc213` (D4, library; found by the twin's empty-matrix matrix_rank lines) a reduction's 0-d result over an empty
    axis (np.sum, np.prod, np.mean, np.count_nonzero, np.trapezoid), np.count_nonzero of a 0-d input and np.nansum /
    np.nanprod along a 1-D input's axis are NumPy's scalars: six exits skipped `MarkReductionScalar` (the rule shipped in
    0.70.0). An 87-call probe 33 -> 0 differing from NumPy; the snapshot tool over 13 reductions (67,290 calls) changes 173
    lines, each only a 0-d result's WRITEABLE flag; 156 np.linalg lines, all on the empty matrix: 154 fixed, 2 decimal.
  - Evidence at `65ccc213`: the twin agrees with NumPy 2.4.2 on 63,357 of the 67,052 comparable lines (2,988 decimal lines
    have no NumPy twin); 53,714 of 63,034 at `289c0254` (np.linalg alone, before any fix). The count is derived: the
    twin's last full run (`ee9ce021`'s snapshot, 62,882 agreeing) minus the 475 lines the later commits' A/Bs moved to
    NumPy's answers (none moved away); a direct run over the final snapshot was stopped by the system for low memory
    (its process had grown to 27 GB) and not repeated. The 3,695 left: 2,627 VF01 flag lines (a 0-d result of norm,
    vector_norm, matrix_norm, cond, np.dot, NDArray.dot or multi_dot is writeable where NumPy returns a scalar), 859 char
    lines, 148 object-dtype computations, 60 text-scalar stand-ins, 1 slogdet NaN-bits line.
  - **Found:** VF01 reaches np.linalg's compositions (norm's sqrt, cond's ratio, np.dot without out=): their 0-d results
    are writeable where NumPy returns scalars, the library gap of the array-object oracle, not this family's. NumSharp's
    eager unary ufunc writes C order for a permuted operand (np.conjugate, the matrix tier's MX05), which
    `SvdHermitian` routes around with `np.empty_like` and out=. NumPy's default argsort on an AVX2 host is x86-simd-sort's
    bitonic argsort, which keeps ties in order up to three elements and reorders them from four: hermitian SVDs of four
    or more rows with tied absolute eigenvalues can order their columns otherwise (NumSharp's argsort is stable). The
    chararray boundary audit (`CharArrayOperand_LibraryNeverReachesACharArrayRule`) checks only the paths the string
    corpus reaches: a facade without a chararray rule hands a chararray operand to the library as it is, so every
    library call of a public np.* function with that operand needs a plain region (`NDMatrixBoundary.PlainFor`). The
    snapshot tool calls a nullable `int?` axis at its default only (np.nansum's axis forms are not in it).
  - Left: the text-scalar stand-in (NumPy returns a Python str for a 0-d text out=); object-dtype computations (np.dot /
    tensordot of numbers with StringDType: NumPy computes Python objects, NumSharp refuses); char operands (refused in
    linalg by design); float32 N-D np.dot (scipy-openblas' strided sdot, which no managed order reproduces: bounded, the
    backend gives NumPy's bits); slogdet of a complex matrix holding NaN / inf (1 line, NaN bits); matrix_power(a, 0) of
    a chararray is a plain array where NumPy's empty_like keeps the chararray (the class boundary, P7); LinalgEngine's
    own np.* calls wait for P6.
  - At `65ccc213`: NumSharp.Tests net10.0 20,610 passed / 11 skipped and net8.0 20,609 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed.
  - Next in P2: `np.array_equal`, then `np.insert`, `np.isin` (VString), `np.average`, then the files of four lines or
    fewer.
- **2026-10-26, P2's seventh family: np.gradient moved into `GradientEngine`, its statements NumPy's, a memory-safety
  bug and six divergences fixed.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The
  referees: the snapshot (76,410 calls over np.gradient's five overloads, 84,216 after `01b8706b`), `twin_gradient.py` (the
  tool's operands, spacings, spacing lists, axis and edge_order rebuilt in NumPy; dtype, shape, C / F / OWNDATA / WRITEABLE
  flags and SHA-256 per component, or the exception's type and text), `ab_gradient.py` (parent against fix, each changed
  line classified by the twin), `cmp_bytes.py` (a `SNAPSHOT_BYTES=1` line's elements against NumPy's, bit for bit), and
  NumPy probes per fix, written into `Math/np.gradient.NumPyPort.Test.cs` as raw bits:
  - `f0918452` the snapshot tool reaches np.gradient's arguments: the one spacing (34 kinds: C# numbers, NaN / inf, bool,
    Half, float, decimal, char, str, None, 0-d arrays, coordinates of every dtype, evenly spaced, as C# arrays, a column,
    one too many, special values, a zero step, decreasing, strided), the per-axis spacings, the spacing list and edge_order
    (0 and -1, which NumPy's single check admits, and 3).
  - `382d0738` (D4, memory safety) every spacing value was read with `NDArray.GetDouble`, which indexes the float64 lane of
    the storage's typed-slice union whatever the dtype: float32 / float16 / bool coordinate differences were read as
    other bytes and past their buffer (results changed from run to run), a 0-d int64 step 2 became the denormal 1e-323,
    an empty axis' coordinates read an empty buffer where NumPy raises IndexError. 1,214 lines change: 976 fixed (236 of
    them the lines that changed from run to run), 33 decimal, 205 left for the next commits. 68,305 of the 74,671
    comparable lines agree (67,329 before).
  - `1089768b` the move (D1: NumPy writes gradient in Python): the body and its helpers move into
    `Math/GradientEngine.cs`, beside the facade, statement for statement; 76,410 calls byte-identical; R1's row for
    np.gradient.cs (6 text / 5 numeric lines) goes.
  - `c443f174` (D4) the components are NumPy's `np.empty_like(f, dtype=otype)` (order 'K'): an F-ordered, broadcast or
    transposed input gives F-ordered components, a permuted one its permutation's order (they were C). 1,198 lines
    change: 1,027 fixed, 106 decimal, 65 left. 69,332 agree.
  - `01b8706b` the tool's spacings gain Python lists (a `List<double>`, an `object[]` of C# ints, evenly spaced, a ragged
    list): 7,806 new lines, the 76,410 earlier ones byte-identical.
  - `b0f4f9a8` (D4) a Python list spacing is coordinates, as NumPy's `np.ndim` reads it: the port's np.ndim knew only
    NDArrays and C# arrays, so a `List<double>` reached the double conversion's InvalidCastException and a ragged list a
    KeyNotFoundException in dtype discovery (`PolySequence.NDim` / `ToArrayOrNonNumeric` now, np.array's rules). 2,944
    lines change: 2,742 fixed, 48 decimal, 154 left. 76,841 of the 82,402 comparable lines agree (74,099 before).
  - `10ab09a1` (D4) each scalar spacing keeps its NumPy kind through NumPy's statements (`PolyNumber`): every step was a
    C# double, a weak Python float, so a 0-d float64 step or a collapsed float64 coordinate step (`diffx[0]`, a NumPy
    scalar) computed a float32 input in float32 where NumPy computes in float64 (one ULP lower in every element of the
    unit test), a float16 / char step was refused (InvalidCastException), a str was parsed as a number, a None refused
    with ArgumentNullException, and the non-uniform edges were widened to float64 where NumPy keeps the coordinates'
    dtype. 4,448 lines change: 4,241 fixed, 73 decimal, 134 left. 81,082 agree.
  - `8af1310f` (D4) the edges are NumPy's integer-indexed statements: the length-1 slices made a 1-D input's edges ufuncs
    where NumPy's are scalarmath (a sum of two NaNs kept the second operand's payload, a complex product was the fused
    one), and an axis of 1 or 2 elements under edge_order 0 / -1 raised a broadcast error where NumPy raises IndexError.
    928 lines change: 896 fixed, 32 decimal. 81,978 agree.
  - `c1bdf5d9` (D4) a bool input stays bool with float64 components, as NumPy's otype branches give it: the port raised
    the boolean-subtract TypeError up front, so bool data with coordinates and second-order edges had no gradient, an
    empty axis skipped the shape check's ValueError, and a 0-d input without an axis raised. 364 lines change: 352 fixed,
    12 decimal. 82,330 agree.
  - `c0c0832d` (D4) a text input with coordinates is refused after the coefficients and with their dtype, NumPy's
    `a * f[s2]`: the text route named float64 for float32 / float16 / complex128 coordinates. 64 lines change, all fixed.
    82,394 agree. (Its commit message says the work began at 68,305; that is the count after `382d0738`, the count before
    it is 67,329.)
  - Evidence at `c0c0832d`: the twin agrees with NumPy 2.4.2 on 82,394 of the 82,402 comparable lines (1,814 decimal lines
    have no NumPy twin). The 8 others are complex128 operands holding (+nan, -nan): np.divide by `2+0j` gives (-nan, -nan)
    where NumPy keeps the positive NaN in the real part, the library residual pinned by `OpenBugs.NaNOperandOrder`'s
    `Complex128_AddMultiplyDivide_KeepNumPysNaNPerPart`. R1 / R2: np.gradient.cs has no row; GradientEngine.cs sits beside
    its facade (its statements call np.*), as TrapezoidEngine does.
  - **Found:** `NDArray.GetDouble` is not a conversion (it reads the float64 lane of any storage; its doc comment's
    NullReferenceException no longer happens since the typed-slice union), and `poly1d.Call(double)` reads np.polyval's
    decimal result with it (P4); NumSharp's `np.subtract(bool, bool, dtype=float64)` raises the boolean-subtract TypeError
    where NumPy computes; np.divide's complex NaN per part (above).
  - Left: a list spacing holding None or another object is NotSupportedException (NumPy computes an object array); the
    complex NaN residual; GradientEngine's own np.* calls wait for P6.
  - At `c0c0832d`: NumSharp.Tests net10.0 20,582 passed / 11 skipped and net8.0 20,581 / 11; Oracle net10.0 411 / 2
    skipped, 0 failed, and net8.0 411 / 2 skipped, 0 failed.
  - Next in P2: `np.linalg`, then the list's order.
- **2026-10-26, P2's sixth family: the product gufuncs np.matmul / np.vecdot / np.matvec / np.vecmat moved into
  `TensorEngine` members with `GufuncGuard`, their checks NumPy's, their results NumPy's objects.** Commits on `strings`,
  developed in a scratch worktree and landed by fast-forward. The referees: the snapshot (95,580 calls over the four
  functions: 5,220 results, 90,360 exceptions), `twin_gufunc.py` (the tool's operands, x2, axes, axis, keepdims, dtype,
  casting, order and out= rebuilt in NumPy; dtype, shape, C / F / OWNDATA / WRITEABLE flags and SHA-256 per line, a NumPy
  scalar result with its own flags), `ab_gufunc.py` (parent against fix, each changed line classified by the twin: fixed,
  regressed, still different, decimal), and NumPy probes per fix: a 34-case layout probe, and a seeded random differential
  of 20,000 calls over the four functions (transposed, stepped, reversed and broadcast operands of up to three loop axes,
  keepdims, dtype=, order=, axes= / axis= placing the core axes anywhere; shape, byte strides, flags, dtype and a value
  digest):
  - `6b143941` the snapshot tool reaches the gufuncs' arguments: x2 (with a new "transposed" second operand, so a (3, 4)
    operand meets the (4, 3) partner a matrix product accepts), axes= lists (each function's cores, an axis out of range,
    a repeated axis, an entry of the wrong length), matmul's nullable keepdims and the casting names.
  - `b7b1b4d4` the move (D1: NumPy writes the gufuncs in C, `ufunc_generic_fastcall` over the gufunc loops): new abstract
    `TensorEngine.Matmul(x1, x2, out, axes, axis, keepdims, dtype, casting, order)` and `Vecdot` / `Matvec` / `Vecmat(x1,
    x2, out, axes, axis, keepdims, dtype)`, implemented in `Backends/Default/LinearAlgebra/Default.Gufuncs.cs` statement for
    statement (each np.moveaxis became the engine's MoveAxis); `GufuncGuard` moved to `Backends/Default/LinearAlgebra/` with
    the order / casting checks np.einsum shares. 95,580 calls byte-identical. R1 has no row to change (GufuncGuard.cs is not
    an np.* file, and the four facades had no dtype test); R2 pinned the structural calls left (np.expand_dims, np.copyto,
    np.ascontiguousarray / np.asfortranarray). The twin then agrees with NumPy on 68,122 of the 94,638 comparable lines (942
    decimal lines have no NumPy dtype); each root cause of the rest is one of the commits below.
  - `84ee0bd2` (D4) NumPy's parse order: axis with axes, the dtype= class, a read-only out= (`output array is read-only`,
    before the product, where numeric operands reached np.copyto's text after it), matmul's casting and order names, then
    the loop refusal of text. 2,157 lines change: 2,142 to NumPy's outcome, 15 decimal. 70,264 agree.
  - `3f0e2f22` (D4) NumPy's `validate_casting`: each input's cast to the loop and the loop's cast to out=, before the gufunc
    checks. vecdot / matvec / vecmat cast a `dtype=` operand unchecked (`np.vecdot(complex, float, dtype=np.float64)` dropped
    the imaginary parts), np.matmul computed `casting='no'` over int32 and float64; a fixed-width text operand under a
    numeric `dtype=` and `casting='unsafe'` is converted as NumPy's loop converts it, and a StringDType operand's no-loop
    text names the requested class. 15,595 lines change: 15,457 fixed, 85 decimal, 53 left for the next commits. 85,721
    agree.
  - `ec9a384a` (D4) NumPy's gufunc checks as one port (`GufuncSignature` + `GufuncGuard.CheckCoreDimensions`): keepdims and
    axis support, `_validate_num_dims` (matmul's flexible n? / m? dropped from every operand, out= included),
    `_parse_axes_arg` / `_parse_axis_arg` and `_get_coredim_sizes`, with NumPy's texts. matmul left its ranks and core sizes
    to its kernel (an InvalidOperationException and an IncorrectShapeException of its own words), matvec / vecmat never
    checked an axes= entry for repeats. 7,395 lines change: 7,259 fixed, 136 decimal. 92,980 agree.
  - `65d3cde3` (D4) the loop axes broadcast as `npyiter_fill_axisdata` does, with the iterator's texts (`operands could not
    be broadcast together with remapped shapes [original->remapped]: ...`, `non-broadcastable output operand ...`, the
    reduction that is not enabled): matmul raised its kernel's IncorrectShapeException, vecdot the elementwise broadcast's
    text, a mismatched out= np.copyto's text after the product. 444 lines change: 432 fixed, 12 decimal. 93,412 agree.
  - `f8dd92bf` (D4) the results own their data: matvec / vecmat drop the product's promoted axis, vecdot's keepdims inserts
    the kept axis, and an axes= output entry remaps, each by relabelling the fresh product (`NDArray.AdoptDroppedAxis` /
    `AdoptInsertedAxis`, new; the kept axis takes one element's stride, NumPy's innermost core axis). The array-object
    oracle's VF27 (12 cases) returns to its tier. 1,123 lines change: 1,074 fixed, 40 decimal, 9 keep a value difference
    (their flags now NumPy's). 94,484 agree (93,410 at the parent, under the twin corrected to read a NumPy scalar's own
    flags).
  - `f624bb44` (D4) a 0-d result without out= is NumPy's scalar stand-in (an owner, read-only; `GufuncGuard.Deliver` marks
    it): matmul of two vectors returned a writeable 0-d view, vecdot of empty and float16 vectors writeable owners. VF01's
    matmul case returns to its tier. 46 lines change: 43 fixed, 3 decimal. 94,527 agree.
  - `7109bbc0` (D4) np.matmul's order= keeps a vector pair's 0-d result: np.ascontiguousarray / np.asfortranarray
    (ndmin=1) made it shape (1,).
  - `ae96f8af` (D4) the loop axes follow NumPy's iterator order (`npyiter_find_best_axis_ordering` over the inputs' loop
    parts, the core axes innermost in C order, `Shape.NpyIterOutputShape`): with two or more loop axes over F-ordered,
    permuted or broadcast inputs the results were C-ordered, and np.vecdot took its product's layout. The 34-case probe: 17
    layouts differed, 0 now; the random differential 15,994 of 16,000.
  - `ec61b110` (D4) order= lays out the iterator's axes before an axes= output entry remaps them, with exact strides (an
    extent-1 axis gets its F stride under order='F'). The random differential: 20,000 of 20,000. R2: Default.Gufuncs.cs's
    last structural calls go and its row is removed.
  - `15004724` (D4) float16 np.vecdot is NumPy's HALF_dot (float32 products, a float32 sum in order, one narrow) through the
    float16 matmul loop, which already matched NumPy; the products were rounded to float16 before the sum. 16 random sets:
    15 differed, 0 now. 22 snapshot lines fixed; 94,549 agree.
  - `06d3a13f` (D4) the float16 product kernels give a NaN result NumPy's NaN: in NumPy's HALF loops a product of two NaNs
    keeps the second factor's and the float32 sum its first NaN, where NumSharp's kernels left the operand order to RyuJIT
    (`Utilities/NDHalfDot.cs` encodes NumPy's order; an output is recomputed only when its float32 sum is NaN). It reaches
    np.dot, np.inner, np.tensordot, np.vdot, np.convolve and np.correlate too. 1 snapshot line fixed; 94,550 agree.
  - `b594d750` (D4) np.matmul and np.einsum read the order= letter in either case (NumPy's order converter), and
    casting=None is NumPy's TypeError `casting must be str, not NoneType`.
  - `546ea9e3` the facades' docs follow the fixes, and four signature constants without a reader go.
  - Evidence at `546ea9e3`: the twin agrees with NumPy 2.4.2 on 94,550 of the 94,638 comparable lines. The 88 others: 80
    exception texts that name NumSharp's char where NumPy's proxy is uint16 (inherent), and 8 float32 / float64 /
    complex128 values that NumPy computes through scipy-openblas, which the managed kernels sum in another order (5 over
    the special-value operand, matvec's through gemv and vecdot's through zdotc; 3 complex128 vecmat lines, through zgemm,
    because gemv cannot conjugate the vector). R2 at the end:
    GufuncGuard.cs 0 / 1 (Deliver's np.copyto into out=), TensorEngine.LinearAlgebra.cs 1 / 5 (np.reshape gone),
    Default.MatMul.cs 0 / 5 (np.squeeze gone).
  - **Found:** the managed float32 / float64 / complex128 vecdot (and np.dot of two vectors, np.vdot) do not follow NumPy's
    per-dtype dotfunc where it is reproducible without BLAS: scipy-openblas' ddot sums sequentially below 16 terms, sdot in
    double below 32, zdotc in four separate sums below 8, and its strided kernels keep their own accumulators. Random
    vectors differ in the last bit (float64 from 16 terms, float32 from 15 in vecdot, complex128 at every length).
    np.convolve models these regimes for contiguous operands (`NDArray.SlidingDot`); vecdot's rows can be strided. And
    np.einsum's default loop (NumPy's sum_of_products) keeps another NaN than the matmul route NumSharp composes, the
    documented divergence of its contraction.
  - Left: vecdot / matvec / vecmat take no casting= / order= (NumPy's ufunc keywords; a signature change, which waits on
    the owner's one-signature decision); the dotfunc regimes above; VF01 for the elementwise ufuncs, clip, round, conj,
    nan_to_num and dot / inner / kron (71 cases).
  - At `546ea9e3`: NumSharp.Tests net10.0 20,575 passed / 11 skipped and net8.0 20,574 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed (Layering 6 / 6).
  - Next in P2: `np.gradient`, then the list's order.
- **2026-10-26, P2's fifth family: np.cov moved into `CovEngine`, its text route NumPy's statements, two divergences
  fixed.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees: the snapshot
  (31,392 calls over np.cov / np.corrcoef), `twin_cov.py` (the tool's operands, y, weights, rowvar, bias, ddof and dtype
  rebuilt in NumPy; dtype, shape, C / F / OWNDATA / WRITEABLE flags and SHA-256 per line), `ab_verdict.py --twin
  twin_cov`, `ulp_cov.py` (a value mismatch measured element by element against the bytes a `SNAPSHOT_BYTES=1`
  snapshot carries), and a NumPy probe per fix:
  - `82a6e78a` the snapshot tool reaches np.cov / np.corrcoef's optional second set of variables y (base null), the
    nullable ddof and the frequency / analytic weights (built from the operand: one per observation for either
    orientation, and the forms NumPy refuses); `SNAPSHOT_BYTES=1` appends each array result's bytes. Its first run on
    the parent did not finish: the next fix.
  - `51e20fda` (D4, a process crash) a `StringDType()` `dtype=` request is a text request: np.cov tested only the
    fixed-width class (`IsStringRequest`), so a text operand's parse gave StringDType text again and cov recursed
    until the stack overflowed, and a numeric operand raised NumSharp's storage-lane NotSupportedException. Now
    NumPy's answers: the mean's divide no-loop text, add.reduce's "no identity" over no observations, the weighted
    mean's DTypePromotionError, the empty (0, 0) float64. All 1,380 dtype=T lines agree with the twin.
  - `8e8c3210` (D4) an integer or bool `dtype=` refuses NumPy's in-place `X -= avg[:, None]` (the float64 mean's
    difference cannot be cast back same_kind: NumPy's UFuncTypeError, the house ArgumentException, through
    `DefaultEngine.ValidateOutCast`, now internal); NumSharp cast it back unsafely and returned an integer covariance.
    2,100 snapshot lines change: 2,082 to NumPy's outcome, 18 decimal.
  - `d5783534` the move (D1: numpy.cov is Python): the body and every private member of np.cov.cs into
    `Statistics/CovEngine.cs` (beside the facade, C3), statement for statement; CovOfText's parsed-text call re-enters
    `CovEngine.Cov`. 31,392 calls byte-identical; R1 np.cov.cs 8 text / 7 numeric lines -> out of the allowlist.
  - `6127173b` step 2: the text route converts X and y, concatenates, checks the weights and lets np.average refuse,
    instead of naming the refusal from descriptor classes (a probe: NumSharp's np.average gives NumPy's outcome over
    the twelve text forms the route builds). 31,392 calls byte-identical. The concatenation's observation check stays
    in the route for NumPy's ValueError (NumSharp's np.concatenate raises the house IncorrectShapeException).
  - Evidence at `8e8c3210` (the move and step 2 change no line): the twin agrees with NumPy 2.4.2 on 26,176 of the
    31,036 comparable lines (356 decimal lines have no NumPy dtype); every other line is one of three known classes —
    the OWNDATA flag of the open gap VA04 (cov ends with ndarray.squeeze, which in NumSharp returns a new view where
    NumPy returns the array itself, VF04), 699 value lines each within 4 epsilon of the result's largest element
    (without the OpenBLAS backend the managed Gram / dot product sums in another order than NumPy's BLAS; 4 of them
    place a NaN differently because an infinite scale meets a 1e-18 imaginary residue NumPy's product leaves at 0), and
    8 lines where the twin names NumSharp's char as numpy's UInt16DType in a promotion text. The twin models one
    documented divergence: NumPy 2.4.2's StringDType -> complex cast writes the real part into both parts, NumSharp
    writes the value complex() parses (`NDTextParse.ParseVStringElement`).
  - Left: VF04 / VA04 (np.squeeze returning its operand itself changes what a caller's disposal reaches: its own
    decision).
  - At `6127173b`: NumSharp.Tests net10.0 20,551 passed / 11 skipped and net8.0 20,550 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed (Layering 6 / 6).
  - Next in P2: `GufuncGuard`, then the list's order.
- **2026-10-25, P2's fourth family: np.searchsorted moved into `TensorEngine.SearchSorted`, the divergences around the
  move fixed.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees: the
  snapshot (44,676 calls over np.searchsorted / NDArray.searchsorted), `twin_searchsorted.py` (the tool's haystacks, keys,
  sides and sorters rebuilt in NumPy; a scalar key's index, an array result's dtype, shape, C / F flags and SHA-256, or
  the exception's type and text; a C# int / double key as NumPy's Python int / float), `ab_verdict.py --twin
  twin_searchsorted`, and a NumPy probe per fix:
  - `24ad50ae` the snapshot tool reaches np.searchsorted's keys (15 arrays built from the operand, or a C# int / double
    key), its side (both names and the three forms NumPy refuses) and its sorter (the operand's argsort in four dtypes and
    the forms NumPy refuses, out-of-range entries included), over pools of 1-D haystacks (sorted, strided, reversed,
    unsorted, empty, one 2-D). Its first two runs were not identical (46 lines): the next fix.
  - `5d409ad1` (D4, memory safety) the search kernel checks every sorter entry its bisection reaches, as NumPy's
    argbinsearch does: an entry outside the haystack is NumPy's `ValueError: Sorter index out of range.`, where the kernel
    read the haystack at it. 832 snapshot lines change: 788 to NumPy's outcome, 44 decimal (no NumPy dtype).
  - `67293fac` (D4) keys and a sorter that are views at an offset (an np.split child: C-contiguous, Shape.offset > 0)
    are read from the view; the kernel calls passed the storage's start. The array-object oracle's VA01 (4 cases) and the
    relayout tiers' RL01 (15: np.digitize and np.isin's complex / float16 route reach it through np.searchsorted) are
    fixed: their cases return to their tiers, four `[OpenBugs]` pins become regression tests.
  - `c6561f85` (D4) a char array meets StringDType as NumPy's uint16 does: the object search reads a char element as a
    Python int (`VStringSearch.ReadObjects`), so `'<' not supported between instances of 'int' and 'str'` where the
    element was a one-character str. 10 snapshot lines change, all 10 to NumPy's outcome.
  - `01a71fc9` the move (D1: np.searchsorted is ndarray.searchsorted, i.e. `PyArray_SearchSorted` in C, so
    `TensorEngine` members). Three members, one per key form (a C# int, a C# double, an NDArray of keys): the scalar
    forms search from the stack in ~50 ns per call, the array form with a 0-d key takes ~740 ns. Every private member of
    np.searchsorted.cs into `Backends/Default/Sorting/Default.SearchSorted.cs`, statement for statement, except two
    statements written in place (same answers): `promote_types` over type codes (`SearchSortedCommonType`: no up-call,
    R2 unchanged) and np.insert's private `IsIntegerIndexDtype` (`IsSearchSorterIntegerDtype`). 44,676 calls
    byte-identical; R1 np.searchsorted.cs 10 text / 23 numeric lines -> out of the allowlist.
  - No step 2: the moved routes already run NumPy's statements (`TextSearch` / `VStringSearch` are NumPy's binary search
    over its descriptors), there is no replayed answer to remove.
  - Evidence at `01a71fc9`: the twin agrees with NumPy 2.4.2 on all 43,594 comparable snapshot lines (1,082 decimal
    lines have no NumPy dtype); the scalar route costs 41-61 ns per call after the move (45-59 before).
  - **Found for np.isin's family** (P2's list): np.isin's StringDType route reads a char test set's elements as strs
    too (`np.isin(StringDType ['a', 'zz'], char ['a', 'b'])` is `[True, False]`; NumPy's uint16 gives `[False, False]`,
    and `invert=True` `[True, True]`, probed). The same root cause as `c6561f85`, fixed with that family.
  - At `01a71fc9`: NumSharp.Tests net10.0 20,549 passed / 11 skipped and net8.0 20,548 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed (Layering 6 / 6).
  - Next in P2: `np.cov`, then the list's order.
- **2026-10-25, P2's third family: np.trapezoid moved, its text route NumPy's statements, its spacing NumPy's first
  operand.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees: the snapshot
  (7,962 calls), `twin_trapezoid.py` (the tool's operands, sample points, spacings and axes rebuilt in NumPy; dtype,
  shape, C / F flags and SHA-256 per line), `ab_verdict.py --twin twin_trapezoid`, and a random differential
  (`rand_trap_gen.py` / `rand_trap_check.cs`: integrands of 13 dtypes in random views — transposed, stepped, reversed,
  offset, broadcast —, points absent, 1-D or N-D in views of their own, broadcasting, of a higher rank, of a wrong length;
  every spacing and axis; raw bits, flags and error texts; 24,000 cases over three seeds):
  - `4bd797be` the snapshot tool reaches np.trapezoid's optional sample points x (null, then 17 forms built from the
    operand) and its spacing dx (a fraction, negative, -0.0, NaN, inf); an int axis now starts from its default.
  - `6fd84579` the move (D1: NumPy writes trapezoid in Python): the body and every private member of np.trapezoid.cs
    into `Math/TrapezoidEngine.cs`, statement for statement; the facade keeps the matrix boundary and the null check.
    7,962 calls byte-identical; R1 np.trapezoid.cs 9 text / 4 numeric lines -> out of the allowlist. Placement: beside
    the facade, not under `Backends/Default/` as C3 says, because the moved statements call np.* functions and R2 counts
    those against every file under `Backends/` (HistogramEngine sits in `Statistics/` for the same reason; C3 now says
    so).
  - `6b535945` (D4) the spacing is the multiply's FIRST operand for dx too (NumPy's `d * (...)` with `d = dx`): the order
    decides which NaN a product of two NaNs keeps and how a complex product places its `0 * NaN` terms. A NumPy probe of
    a NaN dx against an integrand's negative NaN (1-D, the flat route, a stepped view) pins it in a unit test.
  - `a101b54d` an `[OpenBugs]` pin the fix unmasked (group 15 of `OpenBugs.Reductions.cs`): np.sum along an outer axis
    keeps the accumulator's NaN where NumPy's `out = out + h[k]` keeps the new row's. Library-wide, not fixed here.
  - `b0ad5af9` step 2 (and D4): the text route runs NumPy's statements (np.diff and the reshape, the list assignment's
    axis check, `y[1:] + y[:-1]`, `d * (...)`, `/ 2.0`, `.sum(axis)`) instead of replaying the multiply's no-loop
    error. With an integer spacing NumPy goes further: a fixed-width concatenation meets the string multiply's "The 'out'
    kwarg is necessary" TypeError, a StringDType one is repeated and then refused by the division. 48 snapshot lines now
    NumPy's answer, 0 regressed.
  - Evidence at `b0ad5af9`: the twin agrees with NumPy 2.4.2 on 7,793 of the 7,809 comparable snapshot lines (153
    decimal lines have no NumPy dtype); the other 16 are complex128 over NaN and infinite values and differ only in NaN
    bits (checked NaN-blind, element by element). The random differential: 23,809 of 24,000 cases identical, the other
    191 NaN-bit-only; no value, shape, layout, flags or error difference. The NaN-bit classes are the library's (probed:
    complex division and complex sums keep another part's NaN than NumPy; np.sum along an outer axis, group 15).
  - Every commit: the snapshot as above; at `b0ad5af9` NumSharp.Tests net10.0 20,542 / net8.0 20,541 passed, 11
    skipped; Oracle net10.0 and net8.0 411 / 2 skipped, 0 failed.
  - Next in P2: `np.searchsorted`, then the list's order.
- **2026-10-25, P2's second family: np.histogramdd / np.histogram2d moved, the divergences around the move fixed; and
  np.histogram's step 2.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees
  as for the first family, plus `twin_histogramdd.py` (the tool's samples, sequences, second operands, per-dimension bins
  and ranges rebuilt in NumPy) and a probe pair per fix (a NumPy script and its NumSharp twin, printed alike):
  - `f37a44cc` np.histogram's step 2: its text answers come from NumPy's statements (the cumulative loop for text
    weights, np.diff of text edges, the range mask's comparison with C# doubles — NumSharp's weak Python floats —, np.amin
    of fixed-width text). 58,356 snapshot calls byte-identical. `NDArray.Scalar(x)` was tried first for the range mask:
    360 lines then named `Float64DType` (NumPy's text for an np.float64 operand, not a Python float).
  - `97836742` the snapshot tool reaches the sample as a sequence of columns, the per-dimension bins as an `int[]`,
    `NDArray[]` or `object[]`, and the per-dimension ranges, built from the operand for the dimension count the call sees
    (135,300 calls). `02b4e532` fixes the tool: its `object[]` variants came from a switch whose natural type was NDArray
    (NumSharp's implicit conversions make it the best common type of int, double, string, bool, double[] and NDArray),
    so every "Python scalar" entry was a 0-d array; 1,214 of the twin's mismatches were the tool's input.
  - `4832082e` (D4) histogram2d's first statement is NumPy's `len(x) != len(y)`: a 0-d operand is `TypeError len() of
    unsized object`, and the length is the first dimension, not the size.
  - `0ad171a5` (D4) histogramdd hands its weights to np.bincount as given (it raveled them), so a 0-d or deeper weights
    array meets bincount's depth check before its cast.
  - `9f71efec` (D4, memory safety) histogramdd's sample statement, `sample = np.atleast_2d(sample).T; N, D =
    sample.shape`: a sequence of arrays of different shapes is array coercion's inhomogeneous-shape ValueError (through
    `PolySequence.NDim`, the house port of `update_shape`) where the columns were read with the first one's length, past
    the end of a shorter one (the snapshot was not deterministic on those lines); 0-d entries are one dimension of n
    points, no entries one empty dimension, 2-D entries and 3-D samples "too many values to unpack (expected 2)", a 0-d
    sample one point; histogram2d's `[x, y]` goes through the same statement.
  - `b074641b` the move (D1): every private member of np.histogramdd.cs and np.histogram2d.cs into
    `Statistics/HistogramEngine.Dd.cs`, statement for statement; `Histogramdd` / `Histogram2d` get `[NDScopedCovered]`
    (their only callers are scoped). 135,300 calls byte-identical, np.histogram's 58,356 too; R1 np.histogramdd.cs 10 text
    / 5 numeric lines -> out of the allowlist.
  - `492c8552` (D4) np.bincount raises NumPy's three ValueErrors as ValueError (the first family's finding (2)).
  - `5765818d` (D4, memory safety) NumPy searches every dimension before it reads any rightmost edge: an empty edge array
    is `edges[i][-1]`'s IndexError after every search (a StringDType search refusal comes first), and the fused search
    kernel no longer runs on it — it read `e[numEdges - 1]`, the element before the buffer.
  - `c4b727e9` (D4) histogram2d's shared bins are `xedges = yedges = asarray(bins)`: a list goes through array coercion,
    once, the same array for both dimensions.
  - `bfb2b9f6` (D4) a complex column runs NumPy's statements in complex128 (np.histogram's complex outer edges and
    linspace; a range or an empty column keeps float64 edges, as histogramdd's linspace has no dtype).
  - `43ee8a6e` (D4) edges holding a NaN take NumPy's searchsorted (the fused kernel counts edges, which is the binary
    search's answer only over sorted ones).
  - `3a364832` step 2: a text column's outer edges are np.amin's refusal and text edges' widths np.diff's, instead of
    replayed errors; 135,300 calls byte-identical. Left a replay, with the reason: 1-D text weights' safe-cast TypeError
    (NumSharp's np.bincount does not convert text, and raises casts as InvalidCastException, the house type).
  - Evidence at `43ee8a6e`: the twin agrees with NumPy 2.4.2 on all 133,158 comparable snapshot lines (2,142 decimal
    lines have no NumPy dtype). The remaining divergence of the probes, kept: NumPy stacks None entries, or StringDType
    text beside a number, into an object array and computes with Python objects; NumSharp raises NotSupportedException.
  - Every commit: NumSharp.Tests net10.0 20,536-20,540 and net8.0 20,535-20,539 passed / 11 skipped (the wall-clock test
    `T2_NonZero_Is_Much_Slower_Than_NumPy` failed once per framework on a loaded host and passed on the rerun); Oracle
    net10.0 and net8.0 411 / 2 skipped, 0 failed, at `9f71efec` and at `3a364832`.
  - Next in P2: `np.trapezoid`, then the list's order.
- **2026-10-25, P2's first family: np.histogram / np.histogram_bin_edges moved, and the divergences behind the move
  fixed.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees: the snapshot
  (41,604 calls at the move, 58,356 once `38dda739` added the cumulative path's variants), a NumPy twin of its lines
  (`twin_histogram.py` in the session's scratchpad: the tool's operands, bins, ranges and weights rebuilt in NumPy, each
  result's SHA-256), an A/B verdict per fix (`ab_verdict.py`: fixed / still-differ / regressed against the parent's
  snapshot), and random differentials for what the snapshot's fixed operands cannot reach (`rand_hist_gen.py` and its
  successors: NumPy 2.4.2's raw bits or error text, Linux NumPy's answers for int32):
  - `adaaa190` the snapshot tool reaches the histogram family's arguments (bins as a count, an estimator, an array or a
    `double[]`; the range; complex, `U5` and StringDType weights; a struct-of-arrays result rendered field by field).
  - `ba939e91` the move (D1: NumPy writes these functions in Python, so an internal operation class): every private
    member of np.histogram.cs into `Statistics/HistogramEngine.cs`, statement for statement; the eight overloads open the
    matrix boundary's plain region and make one call. 41,604 calls byte-identical; R1 np.histogram.cs 14 text / 10
    numeric lines -> out of the allowlist. The twin at the move: 39,546 of 40,710 comparable lines agreed.
  - `38dda739` the snapshot tool: explicit edges with a NaN last edge, repeated and infinite edges, bool / int8 /
    float32 / float16 weights (NumPy's cumulative path keeps their dtype through an in-place add).
  - `f0a6e3ee` (D4) explicit edges holding a NaN count as NumPy's cumulative loop does (`HistogramCumulativeLoop`:
    argsort, searchsorted, `cum_n +=`, np.diff). **Correction of its subject:** binning each sample gave `[3, 5]` for
    `bins=[0, nan, 2]` over the twelve samples, not `[2, 0]` (the snapshot's hash a9928084ac1eb583 is int64 `[3, 5]`,
    found by a hash search after the commit).
  - `0ca6504e` (D4) weights on explicit edges add into their total in place (`cum_n += cw[bin_index]`): the weights'
    dtype, an int8 total wrapping, bool weights refused. **Correction of its subject:** the widened sum it replaced was
    int64 for bool and the signed integers and uint64 for the unsigned ones (np.cumsum's dtype), not "int64 for every
    integer", and float16 / float32 weights already kept their dtype (its listed float16 / float32 cases did not change).
  - `16f72407` (D4) an empty array of edges gives NumPy's empty histogram (it raised "negative dimensions are not
    allowed").
  - `2e153994` (D4) the estimators clamp a char sample's width to 1 (`np.issubdtype(a.dtype, np.integer)`, which holds
    for NumSharp's char).
  - `782fee4e` int32 data spanning the whole int32 range pin Linux NumPy's answer: NumSharp follows the LP64 model (C
    long is int64), and Windows NumPy's `_unsigned_subtract` leaves int32 (its C long there) signed and raises.
  - `98ea7bcf` (D4) complex data meet NumPy's equal-width statements in complex128 (complex edges, the complex index
    arithmetic, the estimators stopping where NumPy's complex values meet real-only code).
  - `6bcc8e92` (D4) zero estimator bins no longer reach the fused count kernels, which wrote one slot before a zero-length
    result (STATUS_HEAP_CORRUPTION, the process killed at a later case of the random differential).
  - `5f7a8311` (D4) the single edge of zero float16 / float32 bins is linspace's start.
  - `12240c9c` (D4) an estimator's bin count goes through `int(np.ceil(delta / width))` and np.linspace's allocation:
    NumPy's errors for a NaN or infinite quotient and for an edge count too large to allocate (MemoryError with NumPy's
    `_size_to_string`, "array is too big", np.arange's IndexError at 2^63, "Maximum allowed size exceeded").
  - `ce0418dd` (D4) the estimators compute in NumPy's dtypes (NEP 50 per statement: a float16 / float32 peak-to-peak
    value, rice's and scott's widths and stone's candidate widths in the data's dtype, stone's jhat from `p_k = hist / n`,
    Python's first minimum, and data with no spread widening by ±0.5 in the data's dtype).
  - `dde4d2ef` (D4) zero bins over outer edges with no width left (a float32 1e9 widens to itself) raise NumPy's
    IndexError (index INT64_MIN: 0 / 0 in the index statement) before the bincount refusal.
  - `94c8fe87` (D4) float16 / float32 data over a range divide by the range's float64 `norm_denom`, a strong np.float64
    (`DirectILKernelGenerator.Histogram.Range.cs`); a range narrower than the dtype's spacing then gives NumPy's
    IndexError. `5b80f8ea` takes those kernels' privatized accumulators from the buffer pool (the Oracle's native
    allocation chokepoint gate refused a raw allocation site).
  - Evidence at `5b80f8ea`: the twin agrees on 57,252 of 57,262 comparable snapshot lines (1,094 decimal lines have no
    NumPy dtype); the 10 left are the six int32 lines above and four char lines with text edges (NumSharp's char compares
    with text; the twin's uint16 proxy cannot). Random differentials: 30,000 random cases, 29,909 identical; 20,000
    small-sample estimator cases, 24,898 no-spread cases, 20,000 narrow-range cases and 400 large range cases, all
    identical.
  - **Findings, not P2's.** (1) np.argsort's tie order: NumPy's default kind is introsort (x86-simd-sort for 32 / 64-bit
    lanes on AVX2 / AVX-512), NumSharp's a stable radix sort, so equal keys come out in another order; the cumulative
    path sums its weights in that order (`cw = np.cumsum(sw)`), and the 91 random cases still differing are all
    explicit-edge weighted cases a last bit apart. (2) np.bincount raises NumPy's ValueError text as an
    ArgumentException (its own convention); the histogram's refusals inherit it. (3) stone's `p_k.dot(p_k)` is cblas
    ddot in NumPy; without the OpenBLAS backend the managed np.dot can differ in the last bit on 16 bins or more. (4)
    NumSharp's histogram takes an int bin count, so an edge count NumPy could allocate on a large machine raises
    NumPy's MemoryError text here. (5) float16 histograms without a range run the fused kernel's three software float16
    roundings per sample: 0.63x NumPy at 10M (the range kernel's one rounding runs 0.85x-0.97x; float32 over a range
    went from 5.4x / 2.4x to 4.1x / 1.8x NumPy at 100K / 10M, a float64 division per sample). (6) `b7a42363`, a lost GC
    anchor (lifetime gap LT30): the net8.0 Oracle run of these commits aborted once with an access violation in
    np.nansum's axis kernel, a race the next run did not repeat; a probe with tiered compilation off reproduced it
    (np.nansum(temporary, axis) wrong in 7 of 300 calls, np.nanprod's the same access violation), and `[NDAnchored]` on
    `DefaultEngine.NanSum` / `NanProd` / `NanMinMax` closed it (0 of 300 over the whole nan family), with lifetime-gate
    pins and IL-guard rows.
  - Left for this family: the text routes' replayed answers (`HistogramTextAnswer` decides some answers from sizes and
    dtypes alone) against NumPy's own composition through engine calls, the plan's step 2, its own commit with the
    snapshot as the referee; np.histogramdd / np.histogram2d next.
  - At `b7a42363`: NumSharp.Tests net10.0 20,527 passed / 11 skipped and net8.0 20,526 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed.
- **2026-10-24, P1d (the reduction members' `out=`) done.** Commits on `strings`, developed in a scratch worktree and
  landed by fast-forward. The moves are held to the snapshot of numpy.matrix's reductions (62,742 calls over
  NDMatrix.sum / prod / mean / var / std), byte-identical each time; each D4 fix changes only the lines it names, and
  where it changes answers a NumPy twin of the snapshot's lines (`twin_snapshot_lines.py` in the session's scratchpad:
  the tool's operands, outs and calls, the SHA-256 of the result's bytes) checks them:
  - `792c8ce3` the snapshot tool reaches numpy.matrix's members (a target `NDMatrix.f`: np.asmatrix of each pool
    operand, an NDMatrix-typed out made a matrix too, `ddof` varied). Its first run was not deterministic: 42
    NDMatrix.prod lines in a float16 / complex128 `dtype=` loop changed between two runs of one build, the next fix.
  - `d84021b8` (D4; matrix gap MX08) np.prod in a cast loop casts the input first (`ReduceProductInCastLoop`, as the
    add reduction does): the flat float16 / complex128 products read a narrower input as the loop dtype (past the end
    of an int8 buffer), and the axis kernels saturated a negative integer to 0 in an unsigned loop where NumPy wraps it.
    A 4,563-call NumPy probe: 485 cells differed before, 21 after (below, "Left"). MX08's 15 cases are back in
    `matrix.jsonl` (26,819 cases, 114 carved under 7 ids; MX03 had been fixed by `012ac869` without the ledger saying
    so, corrected in the docs commit).
  - `157544a2` (D2) `ReduceProduct` / `ReduceMean` / `ReduceVar` / `ReduceStd` take `out=` (breaking for a
    `TensorEngine` subclass), and numpy.matrix's `out=` paths of prod / mean / var / std move into them statement for
    statement (`Backends/Default/Math/Reduction/Default.Reduction.Out.cs`: `ReduceIntoOut`, `ReduceTextIntoOut`,
    `MeanIntoOut`, `VarianceIntoOut`, `StdIntoOut`), stored through NDIter.Copy where the matrix called np.copyto.
    Snapshot byte-identical. R1 NDMatrix.Reductions.cs 15 text / 6 numeric -> 10 / 3.
  - `6543726f` (D4) `ReduceAdd`'s `out=` is NumPy's add.reduce with an out operand: the loop `promote_types(out, a)`
    without NEP 50's widening (int8 [100, 100, 56] summed into a bool out stores False; the member stored True) and the
    reduce prologue's texts. No public np.* call passed an out to it, so no public answer moved; snapshot byte-identical.
  - `f1068d9b` matrix.sum's `out=` is that member: the matrix's `ReduceTextInto`, `LoopWith` and `NormalizedAxis` go,
    and its float16 mean reads `DefaultEngine.MeanIsFloat16Result`. Snapshot byte-identical. R1 NDMatrix.Reductions.cs
    -> 0 text / 2 numeric (`IsIntegerLike`, which NDMatrix.Methods and NDMatrix.Operators call too: their decisions).
  - `9826ea5b` (D4) mean / var / std with `out=` check their arguments in NumPy's statement order: `_mean` counts the
    axis first, `_var` takes the mean and subtracts before its last umr_sum parses the out (the matrix had checked the
    out first). 1,984 snapshot lines change, every one an out refusal that becomes an earlier statement's (text
    refusals 1,426, AxisError 176, StringDType's division without a loop 194, its reduction errors 132, the boolean
    subtract 36, decimal's division by a zero count 20); no value changes.
  - `87b39e6c` (D4) a char loop of the add / multiply reductions is the uint16 loop of the oracle's proxy
    (`ReduceInCharLoop`): np.sum / np.prod with `dtype=char` threw flat, np.prod along a non-innermost axis saturated
    through a double, np.mean's flat char loop did not wrap, and the `out=` rule's char loop threw (numpy.matrix's sum /
    prod of a char matrix into a bool out, since the class landed). A dtype=uint16 twin: 32 / 32 values identical (9
    differed). 5 snapshot lines change, those matrix calls; the np.* reductions' snapshot (55,884 calls) is unchanged.
  - `aa788516` (D4) the accumulations' `out=` loop rule (`AccumulateLoop`): np.cumsum / np.cumprod / np.nancumsum /
    np.nancumprod with `out=` and no `dtype=` accumulate in `promote_types(out, a)`. Their snapshot (41,616 calls): 105
    lines change, and the twin agrees on all 105 (on none of the parent's) and on every one of the 1,746 value lines of
    their float64 / int8 / bool `out=` matrix.
  - `ac2a75f1` (D4) np.trace's `out=` is the diagonal's add.reduce with that out (`ReduceAdd`): the loop rule and the
    prologue's texts, where it summed in NEP 50's accumulator and raised its own texts. Its snapshot (23,784 calls): 804
    lines change (792 texts, 12 values); the twin agrees on all 1,128 modelled `out=` lines, where the parent disagreed
    on 388. R2: Default.Trace.cs 2 -> 1 structural up-call (the out= store's np.copyto went).
  - Left: the public `out=` keyword of np.sum / np.prod / np.mean / np.var / np.std (a signature change, waiting on
    the owner's one-signature decision); the flat float32 np.prod of an F-ordered or reversed array, which multiplies in
    another order than NumPy's iterator (1 ULP over 12 values, or NaN where its order meets inf before a 0: the 21
    cells of `d84021b8`'s probe), for its own fix; a reduction into an `out=` of another dtype past `np.getbufsize()`
    elements, where NumPy carries the partial result through the out's dtype between buffer fills (documented on
    `ReduceIntoOut`; an accumulation has no such residual, probed).
  - At `ac2a75f1`: NumSharp.Tests net10.0 20,486 passed / 11 skipped and net8.0 20,485 / 11; Oracle net10.0 and net8.0
    411 / 2 skipped, 0 failed (Layering 6 / 6).
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
