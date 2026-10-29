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
np.sum / np.prod / np.mean / np.var / np.std keep no `out=` keyword until the owner's one-signature decision. P2 has
landed twenty-eight families (2026-10-29): of K3's text lines only np.evaluate's two are left, its engine member being
declared in `TensorEngine.cs`, and np.fft.RawFft's one (E3). P4's three D4 divergences are fixed (2026-10-29). On 2026-10-28 the owner deferred the rest of P2 while the engine layout was being settled,
then resumed it the same day beside PE and P3, with P4 (np.emath, the seventeenth family, came first). Later that day the owner kept the array object
out of the engine: `TensorEngine` holds the functions that compute values, while creation, conversion, views, joins and
selection or assignment by position stay on `NDArray`, `UnmanagedStorage` and `Shape` (D1). PE's E6, E4, E1, E5, E7
and E2 are done (2026-10-28 / 29): `TensorEngine.cs` is split into partial files named after NumPy's sources, eleven of
the twelve operation classes are `TensorEngine` members implemented by `DefaultEngine` (or, for np.insert / np.delete,
internal members of `NDArray`), np.diagonal's view left the engine for `NDArray` and `Shape`, and the sort family and
the max / min keywords are members; numpy.linalg and numpy.fft are the `TensorEngine.Linalg` and `TensorEngine.Fft`
sub-engines (`LinalgEngine` / `FftEngine`, defaults `DefaultLinalgEngine` / `DefaultFftEngine`; E3 on 2026-10-29), so
PE is done. P3's design is written ([§7](#p3-the-array-object-and-values-entering-storage-k4-153-lines-in-52-files));
batch 1, the `DType` element slots, is done (`95b7167e`, 2026-10-29), and the np.arange divergence its gate found is
fixed per D4: np.arange is NumPy's `PyArray_ArangeObj` through the slots (`8be47c75`). Batch 2, item selection, is done
(`f5547943`, `ac49b5b7`): the bodies of np.take, np.put, np.putmask, np.place, np.repeat, np.choose, np.compress and
np.resize are `UnmanagedStorage` members, ndarray.resize's is `UnmanagedStorage.Resize`, and each chooses its
reference-holding route by one `HasReferences` decision. Batch 3, assignment, is done (`794fa292`, `448bde50`,
`fd1fa7af`, `713fbaf3`): np.copyto's internals and ndarray.fill's body are `UnmanagedStorage.Assign*.cs` members,
np.concatenate's body is `NDArray.ConcatenateInto` in `NDArray.Multiarray.cs`, and R1 leaves the array object's homes
out; the text casts got a batch of their own, 9b. Batch 4, coercion, is done (`8a2d53fe`, `8889e868`,
`a45a11f6`, `9eb1fe39`, `c8bc8396`): the bodies of np.array, np.asarray and np.asanyarray are `NDArray.Multiarray.cs`
members, the coercion is `UnmanagedStorage.Coercion*.cs` (the text builder, and NumPy's coercion of a sequence moved
from numpy.polynomial), its stores and its discovery go through the DType classes, and the object-list divergence
it found is fixed per D4 (`3518c7cd`: np.array and np.asanyarray of a C# object list are NumPy's coercion of a
Python list). Batch 5, constructors, is done (`8bbd4e4c`, `e23cef2b`): the bodies of np.full, np.full_like
and np.ones are `NDArray.Numeric.cs` members, np.eye's is `NDArray.Eye` in `NDArray.TwoDimBase.cs`, np.ndarray's is
`UnmanagedStorage.ArrayNew` in `UnmanagedStorage.Ctors.cs` beside the storage-lane fills, the ones are the Python
int 1 converted by the class's `setitem`, and the divergences a replay of their snapshot through NumPy found are
fixed per D4 (`f048ad69`, `4d1a0d02`, `253b8f45`). Batch 6, buffers and IO, is done, in three parts. The first, np.frombuffer (`87263eb9`,
`ebff0b12`): its body is NumPy's `PyArray_FromBuffer` in `UnmanagedStorage.Buffer.cs` (one member for the buffers
np.ndarray takes, one for an address, one for a span), its REFCHK asks `HasReferences` and a non-native descriptor is
decoded by the class's `copyswapn`; the replay's divergences were fixed first (`32bfa4a7`, `271f24d0`), so the move
took one body instead of seven copies. The second (`f41dab51`, `2b775799`, `1c5a20be`, `0edc1edc`): the bodies of
np.fromfile, np.fromstring and ndarray.tofile are `IO/ArrayFileIO.cs` (NumPy's `PyArray_FromFile`,
`PyArray_FromString`, `PyArray_ToFile`), tobytes' and byteswap's are `UnmanagedStorage` members, their dtype questions
are class slots (`HasScanFunc`, `HasFromStr`, the `NPY_LIST_PICKLE` flag, and `copyswapn`, which ndarray.byteswap now
swaps through), and the three divergences the replay found were fixed first (`79fe2630`, `c4c072d1`, `ac8ed7a4`).
The third (`06320e8b`, `d1ce3bef`): np.loadtxt's body - `_read`'s checks, `read_rows`, the converters and
the sinks - is `IO/TextReading/LoadTxtReader{,.Conversions,.Sinks}.cs`, np.savetxt's is `IO/SaveTxtIO.cs`, a
converter's value is stored through the class's `setitem` (NumPy's `npy_to_generic`), and the two divergences the
replay found were fixed first (`222cfcb5`: savetxt's null arguments are None; `d92bd08f`: loadtxt's conversion error
names a non-native dtype). R1 has no batch 6 row left. Batch 9, the computing functions in P3's files, is done
(2026-10-29): np.linspace / np.logspace / np.geomspace are `TensorEngine` members in `TensorEngine.CoreFunctionBase.cs`,
np.vander in `TensorEngine.TwoDimBase.cs`, np.trim_zeros in `TensorEngine.FunctionBase.cs`, and the six divergences a
replay of their snapshot through NumPy found are fixed per D4. Batch 7, NumPy's Python functions of the array object,
is done (2026-10-29): np.where's body is `NDArray.Where` in
`NDArray.Multiarray.cs` (every engine
np.where call, the fixed-width text route's included, calls it), np.real / np.imag's are `NDArray.RealPart` / `ImagPart`
in the new `NDArray.TypeCheck.cs` with ndarray.imag's setter, ndarray.conjugate's is `NDArray.Conjugate`, np.extract's,
np.asarray_chkfinite's, np.select's and np.piecewise's statements and np.delete's helpers are `NDArray.FunctionBase.cs`
members, np.pad's are `NDArray.Pad` / `NDArray.PadCallable` in the new `NDArray.ArrayPad.cs`, np.ix_'s,
np.fill_diagonal's and np.r_ / np.c_'s indexer's are `NDArray.Ix`, `NDArray.FillDiagonal` and
`NDArray.AxisConcatenate` in the new `NDArray.IndexTricks.cs`, the along-axis trio's are `NDArray.TakeAlongAxis`,
`NDArray.PutAlongAxis` and `NDArray.ApplyAlongAxis` in the new `NDArray.ShapeBase.cs`, np.indices' are
`NDArray.Indices` / `NDArray.IndicesSparse` in `NDArray.Numeric.cs`, np.diag's and np.diagflat's `NDArray.Diag` /
`NDArray.DiagFlat` in `NDArray.TwoDimBase.cs`, np.mgrid's and np.ogrid's `NDArray.MGrid` / `NDArray.OGrid` in
`NDArray.IndexTricks.cs`, np.ravel_multi_index's, a function that computes, the engine member
`TensorEngine.RavelMultiIndex` (`TensorEngine.Multiarray.cs`, with NumPy's `astype_anyint`, which np.unravel_index's
member shares), and the divergences a replay of
their snapshot through NumPy found are fixed per D4 (np.where's result dtype and order, np.select's broadcast and cast
texts, np.piecewise's constants, np.delete's error axis, np.broadcast_shapes' pair, np.pad's check order, values and
computations, np.ix_'s result object, np.r_'s matrix copy and Python int entries, np.fill_diagonal's value conversion,
np.apply_along_axis's text scalar width, np.indices' allocation order, arange refusals and sparse views, the slice
fields' Python numbers, np.mgrid / np.ogrid's nd_grid statements, np.ravel_multi_index's null coordinate array);
R1 has no batch 7 row left. Batch 8, the indexers, is done (2026-10-29 / 30): the typed fetch and store an advanced
index ends in are `UnmanagedStorage.MapIterGet` / `MapIterSet` in the new `Selection/UnmanagedStorage.Mapping.cs`
(NumPy's mapping.c below the indexers), the StringDType route decisions ask `HasReferences`, the indexers' four up-calls
under `Backends/` are array-object members, and the thirteen divergences a replay of the indexers' snapshot and a
6,000-key getter / setter differential against NumPy found are fixed per D4 (prepare_index's entry classification and
error order, the store's value conversion and order, Python bools and 0-d integer arrays as NumPy reads them,
`PyArray_AssignArray`'s strip, list values, GetIndices / SetIndices, the caller's key left unchanged, integer slices as
integers, the advanced block's placement); R1's lines left in the indexer files are index-kind tests and the string
routes' representation. P5 is done
(2026-10-29): its six
families and, handed over by the coordinator, the lines first assigned to PE (np.argmax's out= pair, np.unravel_index,
np.packbits / np.unpackbits, np.bincount, and the text lines of np.kron and np.median) are engine members or engine
partials, and the two divergences found on the way are fixed per D4 (np.argmax / np.argmin check a text operand's
`out=` before they compare, `127c3c98`; np.nan_to_num refuses text and complex fills as NumPy's copyto does,
`d313e41b`). P6's structural part is under way (2026-10-29 / 30): the engine's np.copyto, allocation,
np.broadcast_to / np.broadcast_shapes, buffer-size and view / copy calls are the array object's members (R2's
structural calls 263 -> 47, the indexers' four with batch 8), and the two defects its snapshots found are fixed per D4
(a failed iteration's buffers are not written back, `dc807d59`; `np.array(IEnumerable<T>, size)` of a shorter
enumeration raises np.fromiter's error, `f48b5677`). Progress goes into the
[§9 state log](#9-state-log), newest entry first.

**The engine layout (the owner's decision of 2026-10-28, D1 in [§5](#5-decisions-for-the-owner)).** Every NumPy function
NumSharp moves that computes values becomes a `TensorEngine` member. Those NumPy writes in Python are declared in
partial files named after NumPy's source files (`TensorEngine.FunctionBase.cs`, `TensorEngine.NanFunctions.cs`, ...),
and `DefaultEngine` implements them. numpy.linalg and numpy.fft are the only new sub-engines (`TensorEngine.Linalg`,
`TensorEngine.Fft`, beside `Blas`). The functions of the array object (creation, conversion, views and shape, joins and
splits, selection and assignment by position) are not engine members. They live on `NDArray`, `UnmanagedStorage` and
`Shape` partials, and what one element of a dtype means lives in the `DType` slots. The twelve operation classes P1 / P2
created before the decision move in phase [PE](#pe-the-engine-layout), and
[Appendix D](#appendix-d-where-every-numpy-function-lives) gives every NumPy function's home.

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

**Inside `TensorEngine` itself?** Yes, for every function that computes values (D1, decided 2026-10-28):
- An operation NumPy implements in C is an abstract `TensorEngine` member, and its dtype switch sits in `DefaultEngine`'s
  override.
- A function NumPy writes in Python is an abstract `TensorEngine` member too. It is declared in the partial file named
  after its NumPy source file, and `DefaultEngine` implements it with NumPy's statements.
- numpy.linalg and numpy.fft live in two sub-engines, `TensorEngine.Linalg` and `TensorEngine.Fft`. A provider library can
  replace their kernels, as `Blas` lets OpenBLAS replace the BLAS / LAPACK routines.

The functions of the array object stay out of it (the owner, later on 2026-10-28). They create, convert, view, join,
split, select or assign elements by position and compute no value. In NumPy they are the array object's own C
(`src/multiarray`: `ctors.c`, `array_assign_array.c`, `shape.c`, `item_selection.c`, ...) or Python code over it, and no
provider replaces them. In NumSharp they live on `NDArray`, `UnmanagedStorage` and `Shape`, where nothing can override
them, and their per-dtype element semantics live in the `DType` slots.

Until that decision the plan recommended the engine layer, but not the public contract, for NumPy's Python functions.

**Background:** the 2026-10-22 analysis of the chararray boundary (231 `NDCharArrayBoundary.Sees(` on this branch) and
its follow-ups, which found that text dtype decisions sit in two places: 205 lines inside the engine and 535 lines in
the np.* layer, with no central dtype dispatch.

---

## 1. What "out of np.*" has to mean

Three layers:

| Layer | What it is | What it holds | What it never holds |
|---|---|---|---|
| **F**, the facade | the public `np.*` functions, the `NDArray` members and operators, the module classes (`np.linalg`, `np.fft`, ...) | the public signature and its XML docs; the conversions C# forces (implicit conversions at the call site, several C# overloads normalized into one call); the class-boundary statement (matrix, chararray); one call into O | a dtype test, numeric or text |
| **O**, the operation | for a function that computes values, a `TensorEngine` member: abstract, reached through `a.TensorEngine`, declared in the partial file of its NumPy source module and implemented by `DefaultEngine`; or a member of the `TensorEngine.Linalg` / `TensorEngine.Fft` sub-engine. For a function of the array object, an internal member of an `NDArray`, `UnmanagedStorage` or `Shape` partial, not virtual. The package internals of numpy.polynomial and numpy.ma (`NDPolySeries`, `NDMaskedArray`) keep their own classes. Until E2 lands, also `LinalgEngine`, the last of the twelve operation classes P1 / P2 created (phase PE's E1 moved the other eleven) | NumPy's function body, statement for statement: its argument checks and its dtype checks in NumPy's order, the conversions NumPy makes late, the switch to a family core | a call into F |
| **K**, the cores | the IL kernels, `NDIter`, `Backends/Strings` (S, U, StringDType), the cast implementations, the `DType` classes and their slots | the loops, per dtype family | NumPy's Python-level statements |

Rules: F calls O. O calls O or K, never F. A dtype test lives only in O or K. A conversion that exists only because of
C# (a C# `string` read as Python's `str`, the legacy implicit `string` to `NDArray` reading) may stay in F; the ratchet
lists each with its reason.

**This keeps NumPy's structure** (CLAUDE.md, core principle 2). NumPy's Python signature is F, NumPy's Python body is
O, NumPy's C (the ufunc loops, the ArrayMethods, the dtype slots) is K. The dtype statements NumPy itself writes in
Python do not disappear: `if a.dtype.kind == "c": raise TypeError(...)` in `quantile` and
`if issubclass(z.dtype.type, _nx.complexfloating)` in `angle` move, unchanged and in NumPy's order, from the public
method into O.

**Which O** (decision D1, as the owner decided it on 2026-10-28; the 2026-10-24 rule, which made NumPy's Python
functions internal operation classes, is in [§5](#5-decisions-for-the-owner)):

- **NumPy's C core that computes values** is an abstract `TensorEngine` member: `_multiarray_umath`'s ufuncs with their
  reduce and accumulate, the ndarray methods that compute (the reductions and scans, sort / partition / searchsorted,
  nonzero, clip, round, trace), and the C functions that compute (the matmul family, dot / inner / vdot, bincount,
  lexsort, packbits). Its dtype switch lives in `DefaultEngine`'s override, as `ReduceAdd`'s StringDType route already
  does.
- **A function NumPy writes in Python that computes values** (the quantile family, histogram, cov, gradient, trapezoid,
  average, angle, the set routines, np.emath, the legacy np.poly*) is also an abstract `TensorEngine` member.
  - It is declared in the partial file named after its NumPy source file: `numpy/lib/_function_base_impl.py` gives
    `TensorEngine.FunctionBase.cs`, `numpy/lib/_nanfunctions_impl.py` gives `TensorEngine.NanFunctions.cs`, and so on;
    [Appendix D](#appendix-d-where-every-numpy-function-lives) has the whole map.
  - `DefaultEngine` implements it with NumPy's statements, in NumPy's order.
  - No new operation classes.
- **A sub-engine** is a settable `TensorEngine` property holding an abstract class with a public default implementation.
  It reaches every array, because it sits on the cached engine. It exists only where an outside library can provide the
  capability as a unit, or where NumSharp ships a port of one:
  - `Blas`, the BLAS / LAPACK routines (exists);
  - `Linalg`, numpy.linalg: its Python statements over the `_umath_linalg` kernels;
  - `Fft`, numpy.fft: the `_raw_fft` statements over the pocketfft kernels.

  A provider package derives from the default class and overrides the kernels. The views and rolls of these two
  namespaces are not members: `linalg.diagonal` and `linalg.matrix_transpose` are facades over the array object's
  `diagonal` and `swapaxes`, `fft.fftshift` and `fft.ifftshift` facades over `roll`.
- **The array object** (decided later on 2026-10-28): a function that creates, converts, views, reshapes, joins, splits,
  or selects and assigns elements by position is an internal member of the array object, not a `TensorEngine` member.
  Nothing overrides it: views share their buffer and their strides must stay inside it, and no provider replaces such
  a function. It takes the partial file named after its NumPy source file, on the class its work belongs to, as the
  owner asked when `fcb36637` folded the `StridePerm` helper class into `View/Shape.StridePerm.cs`:
  - the functions over arrays on `NDArray` partials (`NDArray.Multiarray.cs`, `NDArray.ShapeBase.cs`,
    `NDArray.TwoDimBase.cs`, ...); the ndarray methods are `NDArray`'s own members;
  - the memory work under them on `UnmanagedStorage` partials: `UnmanagedStorage.Assign.cs` (`array_assign_array.c`,
    `array_assign_scalar.c`), `.ItemSelection.cs` (`item_selection.c`), `.Coercion.cs` (`array_coercion.c`), `.Ctors.cs`
    (`ctors.c`), `.Buffer.cs` (exists);
  - the layout math on `Shape` partials: `Shape.Broadcasting.cs`, `.Reshaping.cs`, `.Allocation.cs` (exist), `.Axes.cs`
    (new).

  [Appendix D](#appendix-d-where-every-numpy-function-lives) places each of the 111 functions. `TensorEngine` keeps
  the ten array-object members it shipped in 0.70.0 (`GetStorage`, `CreateNDArray` ×2, `Transpose`, `SwapAxes`,
  `MoveAxis`, `RollAxis`, `Cast`, `BooleanMask`, `BooleanMaskSet`) as they are, and gets no new one (E7).
- **Values entering storage** (creation, assignment, IO, fill) belong to the `DType` class: NumPy's `setitem` / `getitem`
  slots (`PyArray_Pack`), which `DTypeMeta` does not port yet. The DType oracle lists `setitem` as exempt because
  "element packing goes through NPTypeCode switches (SetAtIndex / the cast kernels)". The function bodies around them
  (np.array, np.copyto, np.put, ...) are the array object's.
- **Computation or array object.** A function computes values when its result depends on comparing or combining element
  values: arithmetic, comparisons, logic, reductions, scans, sorting and searching, products, transforms, statistics. It
  belongs to the array object when the caller's positions, shapes, masks, indices or fill values decide its result, or
  when it reads only metadata. The borderline cases, decided by the P2 session on 2026-10-28 at the owner's request:
  - `where(cond, x, y)`: the array object (the caller's mask selects). `where(cond)` alone is `nonzero`, and its facade
    calls the engine's `NonZero`.
  - `arange`: the array object, filled through the `DType` class's `fill` slot, as NumPy fills it (`ctors.c` calls
    `PyArray_ArrFuncs.fill`, no ufunc). `linspace`, `logspace` and `geomspace`, which NumPy computes with ufuncs, stay
    on the engine (`CoreFunctionBase`).
  - `pad`: the array object. Its statistic modes and `linear_ramp` call the engine's members, as NumPy's `pad` calls
    np.mean, np.median, np.max, np.min and np.linspace.
  - `trim_zeros` and `real_if_close`: the engine. Each returns a view or the input itself, but a test of the values
    decides which.
  - `tril`, `triu`, `select`, `extract`, `place`, `putmask`, `choose`, `compress`, `take`, `put`, `repeat`: the array
    object (positions or the caller's masks decide). `nonzero`, `argwhere`, `flatnonzero` and `count_nonzero` compute
    indices from values: the engine.
  - `piecewise`, `apply_along_axis`, `apply_over_axes` and `fromfunction`: the array object. They are drivers; the
    values come from the caller's function.
  - `astype` and the shipped `Cast` member: the array object, the values converted by the `DType` layer's casting
    implementations, as NumPy's casts are ArrayMethods of the dtypes.
  - `kron` and `vander`: the engine (they multiply and raise to powers).
  - `fftfreq` and `rfftfreq` compute: they stay on `Fft`.
  - `iterable`, `isscalar` and `isfortran` read the object or its flags: the array object. `iscomplexobj`, `isrealobj`
    and `datetime_as_string` read or format the descriptor: the dtype layer. `frompyfunc` makes a ufunc object: with
    the classes. `base_repr` and `binary_repr` format a Python int: with printing.
- **Packages with an object model of their own keep it:** numpy.random (its seam is NumPy's `BitGenerator`), the
  numpy.polynomial package and numpy.ma (their internals are O already), np.strings / np.char ([§8](#8-out-of-scope)).

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

**Consequence.** Every family gets one home: a `TensorEngine` member when it computes values, an internal member of the
array object (an `NDArray`, `UnmanagedStorage` or `Shape` partial) when it does not. D1, as decided on 2026-10-28, puts
NumPy's Python functions that compute on `TensorEngine` too, so no family needs an operation class any more.
- **Declarations** go to the `TensorEngine` partial file of the function's NumPy module, under `Backends/`. They hold no
  np.* call. The array object's members need none: they are not virtual.
- **Implementations** are `DefaultEngine`'s partials, or the array object's. They stay beside their facades while their
  statements still call np.* functions, because ratchet R2 counts np.* calls against every file under `Backends/`: the
  twelve classes' bodies hold 534 such calls. They move in P6: `DefaultEngine`'s under `Backends/Default/`, the array
  object's beside their classes (`Backends/` for `NDArray` and `UnmanagedStorage`, `View/` for `Shape`).

History. Until 2026-10-28 the consequence was an internal operation class for each Python-level family; twelve were
created in P1 / P2, and phase [PE](#pe-the-engine-layout) moves them into members. The 2026-10-25 correction placed such
a class beside its facade (`Statistics/HistogramEngine`, `Math/TrapezoidEngine`) for the same R2 reason.

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

The owner's answers (2026-10-24; D1 again twice on 2026-10-28), and what they mean for the work:

| # | Answer | Applied as |
|---|---|---|
| D1 | 2026-10-24: "look into it and decide when time comes to impl". 2026-10-28, after an evaluation of the operation classes: "1. Confirmed, 2. Ok, 3. Fft, 4. Yes", and "the 'other' engines should be part of TensorEngine". Later on 2026-10-28, after asking "Why do we move functions that are not math like np.copyto or np.frombuffer?" and seeing the split proposed: "Update our plan with suggestions updated. Borderline functions that need a decision should be decided by you" | **Until 2026-10-28**, decided per family: C-level families widened a `TensorEngine` member, and Python-level families became internal operation classes. Twelve were created: `QuantileEngine`, `HistogramEngine`, `AverageEngine`, `CovEngine`, `GradientEngine`, `TrapezoidEngine`, `ConvolveEngine`, `LinalgEngine`, `ArrayEqualEngine`, `InsertEngine`, `IsinEngine`, `UniqueEngine`. **Since 2026-10-28**, the rule is [§1](#1-what-out-of-np-has-to-mean)'s "Which O": (1) a sub-engine only where an outside library can provide the capability, so `Blas`, `Linalg` and `Fft`; (2) `Linalg` takes the ten `_umath_linalg` members off `TensorEngine` (all ten shipped in 0.70.0 as public virtual members, so this breaks a subclass or a direct caller; accepted); (3) the FFT kernels are overridable members of the `Fft` sub-engine, not an interface; (4) NumPy's Python functions sit in partial files named after NumPy's source files; (5) they are `TensorEngine` members, implemented by `DefaultEngine`. **Later on 2026-10-28** the rule was narrowed: (6) `TensorEngine` holds the functions that compute values; the functions of the array object (111 of the 404 the first layout put on the engine: creation, conversion, views and shape, joins and splits, selection and assignment by position) are internal members of `NDArray`, `UnmanagedStorage` and `Shape` partials, not overridable, with their per-dtype element semantics in the `DType` slots; the borderline cases were decided by the P2 session, as the owner asked ([§1](#1-what-out-of-np-has-to-mean)); the ten array-object members `TensorEngine` shipped in 0.70.0 stay, and `Diagonal` (unreleased) returns to the array object (E7). Phase [PE](#pe-the-engine-layout) applies it to the twelve classes. |
| D2 | yes | Widen in place: `ReduceCumAdd` / `ReduceCumMul` gained `out=` (`fe18431f`), listed under Breaking Changes. |
| D3 | "idk, our code is healthy, use it as reference" | The current behavior is the reference: every move is held to NumSharp's own snapshot before and after, and a batch is whatever unit keeps that true (a function's whole body, text and numeric together). |
| D4 | yes, small bugfix commits | A divergence found on the way gets its own commit (the pilot: `3f88d53f`). |
| D5 | yes, on `strings` only | R1 / R2 run on `strings`. |
| D6 | `strings` | Everything lands on `strings`. |
| D7 | a TODO after the last phase | The class boundary stays in the facades until P7. |
| D8 | open | The 61 NumPy Python functions whose facades hold no dtype logic (marked † in [Appendix D](#appendix-d-where-every-numpy-function-lives)) are outside this plan, which moves dtype logic. Under the narrowed D1, 47 of them belong to the array object (the stack and split families, `broadcast_to`, the flips, `tri*`, `meshgrid`, ...), 12 compute (the five windows, `corrcoef`, `outer`, `tensordot`, `polyder`, `polyfit`, `fix`, `einsum_path`), and 2 format a Python int (`base_repr`, `binary_repr`). Recommended: in a later pass, the 12 become `TensorEngine` members and the 47 move into their `NDArray` / `Shape` partials, so that every array function has one home; the 2 stay facades. The alternative leaves them all as facades. |
| D9 | open | Installing an overriding `TensorEngine` for every array. `BackendFactory` has no setter, and an array binds its engine when it is created, so a `DefaultEngine` subclass reaches only the arrays it is given. The sub-engine properties sit on the cached engine and reach every array. |

The recommendations as written before the answers:

| # | Decision | Recommendation |
|---|---|---|
| D1 | Where an O lives | By NumPy's own split: a C-level operation is a `TensorEngine` member; a Python-level function is an internal operation class that composes members (new ones under `Backends/Default/<Family>/`); values entering storage go to the `DType` slots. The alternative, every function as a public virtual `TensorEngine` member, puts all dtype handling literally inside the engine class but grows its public pluggable surface from 151 members to several hundred and asks a backend author to face NumPy's Python. *Superseded on 2026-10-28 by the owner's answer above: NumPy's Python functions are `TensorEngine` members after all. They are grouped by NumPy's source files, and the pluggable sub-engines are limited to what a library provides. Later the same day the engine was limited to the functions that compute values: the array object's functions stay on `NDArray`, `UnmanagedStorage` and `Shape`, and values entering storage go to the `DType` slots, as recommended here.* |
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
3. **Target.** Per D1, one of four places:
   - for a function that computes values, an abstract `TensorEngine` member, widened per D2 when it exists, declared in
     the partial file of the function's NumPy module ([Appendix D](#appendix-d-where-every-numpy-function-lives)) and
     implemented by `DefaultEngine`;
   - a member of `TensorEngine.Linalg` / `TensorEngine.Fft`;
   - for a function of the array object, an internal member of an `NDArray`, `UnmanagedStorage` or `Shape` partial, in
     the partial file named after the function's NumPy source;
   - the `DType` slots.

   A member has one NumPy-shaped signature: the facade normalizes its C# overloads before the call.
4. **Snapshot.** Record NumSharp's outcomes for the family's call matrix on the parent commit (P0).
5. **Move.** The facade keeps its signature, its XML docs, the boundary statement and the conversions C# forces;
   everything else moves to the target in NumPy's statement order. The facade's one call is `a.TensorEngine.X(...)`
   for a computation and the array object's member otherwise; a sub-engine is read into a local first
   (`var linalg = a.TensorEngine.Linalg;`, the `Blas` rule). The computational np.* calls of the moved code become
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

For each function: first move the body as it is into its D1 target, a pure move, snapshot-identical. Since 2026-10-28
the target is a `TensorEngine` member in the partial file of its NumPy module, implemented by `DefaultEngine`, or a
member of the `Linalg` / `Fft` sub-engine; a function of the array object goes to its `NDArray` / `UnmanagedStorage` /
`Shape` partial instead (every file left in P2 computes). Before that it was an operation class; phase
[PE](#pe-the-engine-layout) converts those. Then try
NumPy's own composition through engine calls, as its own commit, with the snapshot as the referee: where it gives the
same answers, the error replay goes away; where it does not, the snapshot names an engine divergence the composition
was working around, which gets its own fix first (D4). The pilot's `np.angle` is the second case (state log). Order by size: `np.histogram` 16, `np.histogramdd` 9, `np.trapezoid` 8,
`np.searchsorted` 8, `np.cov` 8, `GufuncGuard` 7, `np.gradient` 6, `np.linalg` 5, `np.array_equal` 5, `np.insert` 5,
`np.isin` (VString) 5, `np.average` 5, then the files of four lines or fewer.

The owner deferred the rest of P2 on 2026-10-28 and resumed it the same day (see the State). np.emath came first, as
the first family under the revised D1:
- Its members go to `TensorEngine.Scimath.cs`, after `numpy/lib/_scimath_impl.py`, and `DefaultEngine` implements them.
- Their names collide with the ufunc members (`sqrt`, `log`, `power`, ...), so they take the namespace as a prefix:
  `EmathSqrt`, `EmathLog`, `EmathPower`, ...

The rest of the list moved on 2026-10-29 (np.radians / np.degrees, np.sinc, np.i0, np.diff, np.ediff1d, np.digitize,
np.unwrap, np.cross, np.nanargmax / np.nanargmin, np.interp: the state log). Left: np.evaluate's two text lines, the
positional operands' refusal. They run before the keyword checks that `BuildEvaluateOptions` does in the facade, so
moving them alone would reorder the errors, and moving both means a `TensorEngine.Evaluate` that takes the keywords
(`dtype`, `casting`, `order`, `where`) and checks them after the operands, declared in `TensorEngine.cs`. The
snapshot tool cannot referee it either: np.evaluate's `NDExpr` argument is SKIP. Open for the owner (2026-10-29):
whether the engine member takes the raw keywords.

### PE. The engine layout

The owner decided it on 2026-10-28 (D1); the rule is [§1](#1-what-out-of-np-has-to-mean)'s "Which O". This phase
converts what P1 / P2 built before the rule, one family per commit. Each move is snapshot-identical and follows the
method of [§6](#6-method-per-batch). New families do not wait for it: they land under the revised D1 directly.

Where the operation classes and `DefaultEngine`'s Python-level statics go (E1):

| Partial file (NumPy source) | Takes over |
|---|---|
| `TensorEngine.FunctionBase.cs` (`lib/_function_base_impl.py`) | `QuantileEngine`'s quantile / percentile / median, `AverageEngine`, `CovEngine`, `GradientEngine`, `TrapezoidEngine`, `DefaultEngine.Angle` |
| `TensorEngine.NanFunctions.cs` (`lib/_nanfunctions_impl.py`) | `QuantileEngine`'s nanquantile / nanpercentile / nanmedian, `DefaultEngine.NanCumSum` / `NanCumProd` / `NanMeanNumPy` / `NanVarStdNumPy` |
| `TensorEngine.Histograms.cs` (`lib/_histograms_impl.py`) | `HistogramEngine`, histogram2d included (NumPy's `_twodim_base_impl` wrapper of histogramdd) |
| `TensorEngine.ArraySetOps.cs` (`lib/_arraysetops_impl.py`) | `IsinEngine`, `UniqueEngine` |
| `TensorEngine.Numeric.cs` (`_core/numeric.py`) | `ArrayEqualEngine`, `ConvolveEngine` |
| `TensorEngine.Methods.cs` (`_core/_methods.py`, beside `_mean` / `_var` / `_std`) | `DefaultEngine.Ptp` |
| `NDArray.FunctionBase.cs` (`lib/_function_base_impl.py`; the array object, E7) | `InsertEngine` with np.delete's body |

- [x] **E1, NumPy's Python layer on `TensorEngine`.** Done 2026-10-28 (§9), but `LinalgEngine`, which is E2's. The
      declarations in the table's `TensorEngine` partial files are abstract and documented in full; `DefaultEngine`
      implements them with today's bodies. `InsertEngine` moves to the array object instead: np.insert and np.delete
      join and split by position and compute no value (E7). Rules for the contract:
  - One NumPy-shaped signature per function: the facade keeps normalizing its C# overloads.
  - The plumbing parameters leave the signature. Quantile's `string op` becomes separate members (`Quantile` /
    `Percentile` in FunctionBase, `NanQuantile` / `NanPercentile` in NanFunctions), and `QuantileArgument` an
    `object q`. The `bool chararray` flag belongs to the class boundary, so it stays on an internal overload that the
    facade calls for a chararray operand (P7).
  - A public entry point owns its `[NDScoped]`. The `[NDScopedCovered]` entries (Cov, Histogramdd, Histogram2d,
    Trapezoid) rely on a scoped caller, which a public member cannot. The array object's members are internal, so
    Insert keeps its `[NDScopedCovered]` under its scoped facade.
  - NumPy's private helpers (`_ureduce`, `_commonType`, `_assert_stacked_square`, ...) stay `internal static`, on
    `DefaultEngine` or `DefaultLinalgEngine`. Their callers outside the facades re-point: the text routes
    (`Backends/Strings`, `QuantileText`), the quantile kernel (`DirectILKernelGenerator.Quantile`), `QuantileWeighted`,
    `NDPolyFit` and `NDMatrix.Operators` (LinalgEngine's helpers), `Default.SearchSorted` (InsertEngine's).
  - The `NDArray` members that call a class today (`unique`, `convolve`, `array_equal`) become facades over the member.
  - Placement follows C3.
- [x] **E2, `TensorEngine.Linalg`.** Done 2026-10-29 (`36a1f084`, §9): `LinalgEngine` (numpy.linalg's 29 computing
      functions, 31 members) with `DefaultLinalgEngine` over ten `protected virtual` kernels; the four Array-API
      forwarders' facades apply their main-namespace twin's class boundary.
  - The name moves: today's internal `LinalgEngine` (one of the twelve) becomes the public abstract contract holding
    numpy.linalg's functions. Its statements move to a public `DefaultLinalgEngine`, over `protected virtual`
    `_umath_linalg` kernels.
  - The kernels keep today's order: `Blas` first, then the managed LU for det / slogdet / inv / solve, or
    `OpenBlasMissingBackendException` for the other six. An empty operand still answers without LAPACK (`LinAlgEmpty`).
  - Ten members leave `TensorEngine`: Cholesky, Det, Slogdet, Eig, Eigh, Inv, Solve, Lstsq, Qr and Svd. All ten shipped
    in 0.70.0 as public virtual members, so the move goes under Breaking Changes.
  - The products (Dot, Matmul, Inner, Vdot, Vecdot, Matvec, Vecmat) stay on `TensorEngine`: they are NumPy's C core.
- [x] **E3, `TensorEngine.Fft`.** Done 2026-10-29 (`1b0ba640`, §9): `FftEngine` (numpy.fft's 16 computing functions)
      with `DefaultFftEngine` over three `protected virtual` kernels; `fftshift` / `ifftshift` stay facades over the
      array object.
  - A public abstract `FftEngine` holds numpy.fft's 16 functions that compute (Appendix D: the two shifts roll the array
    object). A public `DefaultFftEngine` holds the `_raw_fft` statements of `Fourier/np.fft.RawFft.cs`, over
    `protected virtual` c2c / r2c / c2r kernels, each a call into `PocketFFTDriver.Execute` with its kind fixed.
  - `np.fft` (`FourierModule`) stays the facade.
  - An MKL, FFTW or cuFFT package derives from `DefaultFftEngine` and overrides the kernels.
- [x] **E4, members that already exist** (done 2026-10-28, `df3a30c5`) move to the partial file of their NumPy Python
      file. No API changes:
  - IsClose, AllClose, Argwhere, FlatNonZero, CountNonZero and Correlate go to Numeric;
  - IsPosInf and IsNegInf go to UfuncLike;
  - NanSum, NanProd, NanMax and NanMin go to NanFunctions;
  - MoveAxis and RollAxis are functions of the array object: they join the other shipped array-object members in
    `TensorEngine.ArrayObject.cs` (E6, E7).
- [x] **E5, C-level statics become members.** Done 2026-10-28 (§9): the sort members (`2f52ffea`), `ReduceAMax` /
      `ReduceAMin` / `NanMax` / `NanMin` with `out=` / `initial=` / `where=` (`bf862520`); `MeanOverAxes` /
      `ReduceAddOverAxes` need no member (E1 made np.average's body a `DefaultEngine` partial, so they are the engine's
      own helpers).
  - np.sort / np.argsort / np.partition / np.argpartition / np.lexsort get `TensorEngine` members (Sort, ArgSort,
    Partition, ArgPartition, LexSort). `AxisSort`, `AxisPartition` and `RadixSort` stay their kernels.
  - `DefaultEngine.MaxMinNumPy` becomes `ReduceAMax` / `ReduceAMin` with `out=` / `initial=` / `where=` (D2).
  - `MeanOverAxes` / `ReduceAddOverAxes` become the tuple-axis forms of `ReduceMean` / `ReduceAdd`.
  - `RichCompareEquality` stays internal: it is the operator protocol around `Compare`.
- [x] **E6, the C core's files.** Done 2026-10-28 (`df3a30c5`). `TensorEngine.cs` splits into `TensorEngine.Umath.cs`
      (the ufuncs), `.Methods.cs` (the ndarray methods that compute), `.Multiarray.cs` (the C functions that compute)
      and `.ArrayObject.cs` (the ten array-object members shipped in 0.70.0, E7). This is file organization only.
- [x] **E7, the array object stays off `TensorEngine`.** Done 2026-10-28 (`7408f707`, `8f3407eb`, §9); the shape math of
      its last bullet is P6's.
  - No function of the array object becomes an engine member. Its functions move into their `NDArray`,
    `UnmanagedStorage` and `Shape` partials when P3 or D8's pass moves their family
    ([§1](#1-what-out-of-np-has-to-mean), [Appendix D](#appendix-d-where-every-numpy-function-lives)); `InsertEngine`
    goes to `NDArray.FunctionBase.cs` (E1).
  - `TensorEngine.Diagonal`, added by `199feaaf` and unreleased, leaves the engine before 0.71.0 ships: np.diagonal's
    view becomes an `NDArray` member over a `Shape` partial. `Trace`, added by the same commit, computes and stays.
  - The ten array-object members `TensorEngine` shipped in 0.70.0 stay as they are, and their callers keep calling
    them: `GetStorage`, `CreateNDArray` ×2 (no caller in src or tests), `Transpose`, `SwapAxes`, `MoveAxis`, `RollAxis`,
    `Cast`, `BooleanMask`, `BooleanMaskSet`. Removing them would break a subclass for a layering reason alone; see
    "Later".
  - The shape math `DefaultEngine` holds today (`Broadcast`, `ResolveReturnShape` and `AreBroadcastable` in
    `Default.Broadcasting.cs`; `normalize_axis_tuple` and `check_and_adjust_axis` in `Default.Transpose.cs`) moves to
    `Shape` partials (`Shape.Broadcasting.cs`, a new `Shape.Axes.cs`) in P6, with their callers.

The gates are §6's, plus three rules:
- A test per sub-engine sets a probe subclass on the cached engine's property (`Linalg`, `Fft`), calls the facade, checks
  that the probe ran, and restores the default. It is `[DoNotParallelize]`: it changes process-wide state.
- Never name a member group or property `Math`: inside the engine classes it would shadow `System.Math` (148 uses in
  35 files). Nor `Sorting`, the `NumSharp.Backends.Sorting` namespace (10 uses).
- A NumPy sub-namespace function whose name a core member already has takes the namespace as a prefix (`EmathSqrt`).

### P3. The array object and values entering storage (K4, 153 lines in 52 files)

- [x] **Design first** (2026-10-29, at `1fcb4663`): below, with the per-file table in
      [Appendix E](#appendix-e-p3s-lines-by-file).
- [ ] Then the batches of the design, in its order.

**P3's design (2026-10-29).** P3 owns the R1 lines of 80 facade files (P5's classification): 702 lines, 469 numeric and
233 text. K4 adds 20 text lines in nine files R1 does not scan (the text readers of `IO/TextReading/`,
`IO/PyLiteral.cs`, `DTypes/StringDTypeMeta.cs`, two `RandomSampling/` files, `Utilities/PyUnicode.cs`). A script sorts
each line by the mechanism NumPy uses for the same decision (`classify_p3.py`, over R1's own line dump; a guide, each
batch re-reads its lines):

| Mechanism | Lines (text) | NumPy | Where it goes |
|---|---|---|---|
| typed dispatch | 301 (0) | none: the descriptor carries the item size, the slots carry the per-type code | moves with its body into the array object's storage partial (B); keyed on `TypeCode`, the storage discriminator |
| the storage class asked by name | 89 (89) | the descriptor's fields and slots | the slot that answers the question (item size, `HasReferences`, `copyswapn`) |
| reference-holding moves | 69 (69) | `PyDataType_REFCHK` chooses the within-dtype transfer function | one `HasReferences` decision per array-object function, routing to the `StringDType` moves that exist |
| index kind | 64 (0) | `prepare_index`'s `PyArray_ISBOOL` and integer checks | stays (C): the role of an index operand, not the values of the array |
| dtype= requests | 43 (0) | `if dtype is not None and arr.dtype != dtype` | moves with its body (A) |
| per-dtype values | 37 (0) | a Python value stored (`1`, `0`) or read (`float(a)`) | `setitem` / `getitem` |
| stored Python values | 28 (26) | `setitem`, through `PyArray_Pack` | `setitem` |
| text casts | 23 (23) | the casting impls' `resolve_descriptors` and loops | the `CastingImpl` objects of `DTypes/` |
| weak literals | 11 (0) | `result_type` with Python scalars (NEP 50) | `DTypePromotion`'s weak path |
| codecs and Python-object helpers | 10 (10) | the IO code's own codec calls | stays with the IO code (not a dtype decision) |
| discovery | 7 (7) | `discover_descr_from_pyobject` | that slot (it exists), through the coercion path |
| truth | 3 (3) | ArrFuncs `nonzero` | `nonzero` |
| element reads | 2 (2) | `getitem` | `getitem` |
| ufunc refusals | 2 (2) | the ufunc's type resolution | np.vander becomes a member (batch 9) |
| not classified | 13 | | read in its batch |

**The slots.** On `DTypeMeta`, NumPy's names; internal until a user-defined DType needs them public.
- `setitem(DType descr, object value, byte* item, IArraySlice buffer)` and
  `getitem(DType descr, byte* item, IArraySlice buffer)`, NEP 42's slots 7 and 8 (`PyArrayDTypeMeta_SetItem` /
  `GetItem`). `buffer` is NumSharp's: a `StringDType` array keeps its string allocator on its buffer
  (`VStringArraySlice`) where NumPy keeps it on the descriptor; the other classes ignore it.
  - the numeric builtins by today's `PyArray_Pack` port, `ReductionInitial.PackStore` and `Box` (batch 1 calls them
    where they are; they move from `Backends/Default/Math/Reduction/` with `Pack`, batch 4);
  - `S` / `U` by `NDStringStorage.WriteElement` and the element reader of `UnmanagedStorage.Strings.cs`;
  - `StringDType` through its allocator (`SetVStringItem`, `NDVStringMove.StoreScalar`).
  `PyArray_Pack` itself (a known scalar, a 0-d array's cast, discovery, then `setitem`) becomes
  `UnmanagedStorage.Coercion.cs`'s `Pack`.
- `copyswapn`, the legacy ArrFuncs slot that `PyArray_FromBuffer`'s byte-swapped reads, `PyArray_FromFile` and the
  `.npy` reader still call: by item size for numbers (by half for complex), per 4-byte unit for `U`, nothing for `S`.
  `StringDType` has none: NumPy refuses a buffer for a dtype that holds references.
- `fill`, ArrFuncs (`PyArray_Arange` fills through it): the builtins' ramps of np.arange; null for the text classes,
  which is NumPy's refusal. `HasFill` is NumPy's `funcs->fill != NULL`, the test `PyArray_ArangeObj` makes before
  it computes anything.
- `nonzero`, ArrFuncs (`StringDType` defines it too): the truth `np.argwhere` and np.trim_zeros read, today
  `NDStringTruth`'s nonzero flavor.
- `HasReferences` (exists; `VStringDTypeMeta` answers true) with `get_clear_loop` / `get_fill_zero_loop` (exist, null
  for every class): the question NumPy's `PyDataType_REFCHK` asks. `StringDType`'s clear loop is the allocator's free.
- The casting impls (exist as `CastingImpl`; their `get_strided_loop` is unused): the text casts' refusals
  (`NDStringCasting.InvalidCast`, `NDStringCopy.ThrowIfCannotConvert`) and loops (`NDTextConversion.Convert`,
  `NDStringTruth`'s cast flavor) become `StringCastingImpls` / `VStringCastingImpls`' `resolve_descriptors` and loops,
  reached through `DTypeCasting`.

A slot replaces a test of the dtype's name, never a typed kernel. It runs once per call or once per Python value, as
NumPy's do; bulk moves keep their IL kernels.

**The homes** ([Appendix D](#appendix-d-where-every-numpy-function-lives)):
- `UnmanagedStorage.ItemSelection.cs` (`item_selection.c`, `compiled_base.c`'s `arr_place`),
  `UnmanagedStorage.Assign.cs` (`array_assign_array.c`, `array_assign_scalar.c`, `PyArray_CopyObject`),
  `UnmanagedStorage.Coercion.cs` (`array_coercion.c`), `UnmanagedStorage.Ctors.cs` (`ctors.c`),
  `UnmanagedStorage.Buffer.cs` (exists; `PyArray_FromBuffer`), and a new `UnmanagedStorage.Mapping.cs` for the typed
  fetch and store of `mapping.c` the indexers dispatch today.
- The `NDArray` partials named after NumPy's sources (`NDArray.Multiarray.cs`, `.FunctionBase.cs`, `.Numeric.cs`,
  `.ShapeBase.cs`, `.TwoDimBase.cs`, `.IndexTricks.cs`, `.ArrayPad.cs`, `.TypeCheck.cs`, ...): NumPy's statements of the
  functions that are Python in NumPy.
- A partial stays beside its facades while its statements call np.*, as `DefaultEngine`'s did (C3; R2 counts np.* calls
  under `Backends/`). P6 moves it under `Backends/Unmanaged/`.
- R1 learns the home names: Appendix D's `NDArray.<NumPy source>.cs` partials hold NumPy's statements (A), so the
  scanner leaves them out, as the names of the `Default.*.cs` partials beside the facades leave those out.
  `NDArray.FunctionBase.cs`'s hand pin (`32478d0c`) goes with that.

**The batches**, in order. Each is a pure move first, then the slot conversion as its own commit; snapshot-identical
(§6), the family's string-oracle sections green, R1 down, R2 not up. The snapshot tool learns each family's arguments
first, as it did for np.fft.
1. **Slots.** `setitem`, `getitem`, `copyswapn`, `fill` and `nonzero` declared on `DTypeMeta` and implemented by every
   class from today's code, no caller yet. Gate: a matrix over every class (a store read back, a double swap,
   np.arange's ramps, `nonzero` against np.nonzero) beside the paths they come from.
2. **Item selection** (10 files, 57 lines): np.take, np.put, np.putmask, np.place, np.repeat, np.choose, np.compress,
   np.resize and the ndarray members into `UnmanagedStorage.ItemSelection.cs`; the `StringDType` routes behind one
   `HasReferences` decision per function. Done (`f5547943`, `ac49b5b7`): ndarray.resize's body went to
   `UnmanagedStorage.Reshaping.cs` (NumPy's `PyArray_Resize`, `shape.c`), and np.take.FancyAxis.cs's index line goes
   with batch 8 (mapping.c's fancy indexing).
3. **Assignment** (6 files, 46 lines): np.copyto (three files), ndarray.fill, np.concatenate's text path and the store
   behind the indexers into `UnmanagedStorage.Assign.cs`; the text casts through the casting impls. Done (`794fa292`,
   `448bde50`, `fd1fa7af`, `713fbaf3`), with three changes to this text: np.concatenate's body went whole, its string
   route included, to its Appendix D home `NDArray.Multiarray.cs` (that route is `PyArray_ConcatenateArrays` over
   descriptors, the function's own statements), and R1 learned the home names with it; the store behind the indexers
   is already `UnmanagedStorage.SetData` (`UnmanagedStorage.Setters.cs`) and stays there; the text casts are batch 9b.
4. **Coercion** (6 files, 71 lines): np.array, np.asarray, np.asanyarray, the string-array overload, `NDArray.Scalar`
   and the implicit conversions from C# arrays into `UnmanagedStorage.Coercion.cs`, with `NDPolySequence.cs`'s port and
   `NDStringArrays`' text collection; discovery through `discover_descr_from_pyobject`, stores through `setitem`.
   Done (`e07f3016`, `8a2d53fe`, `8889e868`, `a45a11f6`, `9eb1fe39`, `c8bc8396`, `3518c7cd`), with three changes to
   this text: the bodies of np.array, np.asarray and np.asanyarray went to their Appendix D home `NDArray.Multiarray.cs`
   (`_array_fromobject_generic`), the coercion to three files (`UnmanagedStorage.Coercion.cs`, `.Coercion.Text.cs` for
   the text builder, `.Coercion.Sequence.cs` for `NDPolySequence.cs`'s port) and the export to a C# array to
   `UnmanagedStorage.Convert.cs`; the stores run as a new slot, `DTypeMeta.SetFreshItems` (NumPy's
   `PyArray_AssignFromCache` loop, `setitem` per value, `StringDType`'s arena reserved once); discovery is
   `discover_dtype_from_pyobject` and `find_scalar_descriptor` over the slots. The two walks (the text builder and the
   sequence walk) are not one yet: see the state log.
5. **Constructors** (6 files, 60 lines): np.empty, np.zeros, np.ones, np.full and their `_like` forms, np.ndarray,
   np.eye's ones into `UnmanagedStorage.Ctors.cs`, where np.arange already is (`8be47c75`: NumPy's
   `PyArray_ArangeObj` through `setitem` and `fill`, the D4 fix batch 1's gate called for).
   Done (`87bd99c1`, `8bbd4e4c`, `e23cef2b`, `f048ad69`, `4d1a0d02`, `253b8f45`), with two changes to this
   text: the bodies went to their Appendix D homes, np.full's, np.full_like's and np.ones' to `NDArray.Numeric.cs`,
   np.eye's to `NDArray.TwoDimBase.cs`, np.ndarray's (arrayobject.c's `array_new`) and the storage-lane fills to
   `UnmanagedStorage.Ctors.cs`; np.empty, np.zeros, np.empty_like, np.zeros_like and np.ones_like had nothing to move
   (one call into the allocating constructor or into np.ones), so R1 counted 42 lines in 5 files.
6. **Buffers and IO** (12 files, 175 lines): np.frombuffer into `UnmanagedStorage.Buffer.cs` (`copyswapn` for a
   non-native descriptor); np.fromfile, np.fromstring, ndarray.tofile, tobytes, getfield and byteswap; np.loadtxt's
   converters into `IO/TextReading/` (NumPy's `textreading/conversions.c`: a converter per type, the generic one through
   `setitem`).
   First part done, np.frombuffer (`0854f1ca`, `32bfa4a7`, `271f24d0`, `87263eb9`, `ebff0b12`), with one change to
   the method: the D4 fixes came before the move. The numeric overloads were seven copies of NumPy's checks with rules of
   their own, and the fix that gives every overload `PyArray_FromBuffer` (the string route's checks and np.ndarray's
   view) collapses them into one body: moving the seven copies first would have moved code the next commit deletes.
   The body moved to `UnmanagedStorage.Buffer.cs`, with the address and span routes as members of their own;
   `copyswapn` decodes a non-native descriptor, and got `byteswap`'s SIMD swap for contiguous runs
   (`Utilities/NDByteOrder.cs`). R1: 16 text and 93 numeric lines left (np.frombuffer.cs, np.frombuffer.Strings.cs).
   Second part done (`592ee2c4`, `79fe2630`, `c4c072d1`, `ac8ed7a4`, `f41dab51`, `2b775799`, `1c5a20be`, `0edc1edc`,
   `0d10b507`),
   the D4 fixes again before the moves: np.fromfile, np.fromstring and ndarray.tofile into `IO/ArrayFileIO.cs`, tobytes
   and byteswap into `UnmanagedStorage.Convert.cs` / the new `UnmanagedStorage.Byteswap.cs`, getfield converted where it
   is. The text readers' refusals ask `HasScanFunc` / `HasFromStr`, tofile's binary refusal the `NPY_LIST_PICKLE` flag,
   and ndarray.byteswap swaps through `copyswapn`, which gained a fused copy-and-swap pass and a strided kernel. R1: 16
   text and 7 numeric lines left.
   Third part done, so batch 6 is done (`0f092c78`, `222cfcb5`, `d92bd08f`, `06320e8b`, `d1ce3bef`), the D4 fixes before
   the move: np.loadtxt's body into `IO/TextReading/LoadTxtReader{,.Conversions,.Sinks}.cs` (`_read`'s checks,
   `read_rows`, the converters, the sinks) and np.savetxt's into `IO/SaveTxtIO.cs`, the facades keeping their overloads
   and the source / target handling. A converter's value, and a float16 field, are stored through the class's
   `setitem` (NumPy's `npy_to_generic` / `npy_to_generic_with_converter`); the sinks stay one per DType class, as
   NumPy's converters are one per type number. R1: 26 text and 16 numeric lines left (np.loadtxt.cs,
   np.loadtxt.Parsers.cs, np.loadtxt.Sinks.cs, np.savetxt.cs).
7. **NumPy's Python functions of the array object** (22 files, 152 lines) into their `NDArray` homes: np.where,
   np.select, np.piecewise, np.extract and np.delete's rest; np.pad (`ArrayPad`); np.r_, np.ix_ and np.fill_diagonal
   (`IndexTricks`); np.indices (`Numeric`); np.diag (`TwoDimBase`); np.take_along_axis and np.put_along_axis
   (`ShapeBase`); np.real and np.imag (`TypeCheck`); np.asarray_chkfinite; np.mgrid, np.ogrid, np.apply_along_axis and
   np.ravel_multi_index's lines. First part done (`953af9f3`, `d6f814d8`, `e8ee6dab`, `00aa4cff`, `ecf4ad9c`): np.where,
   np.real / np.imag with ndarray.imag's setter and ndarray.conjugate, np.extract and np.asarray_chkfinite (see the
   state log); np.extract's fast path asks `HasReferences`. Second part done (`3f06e8a4`, `c0ac4ef5`, `582ddced`,
   `a318249e`): np.select, np.piecewise and np.delete's rest, with the D4 fixes their replay and the first part's found
   (`4fa16cbf`, `a186234e`, `8f86bd03`, `d3d4a01a`, `3c68f7cd`). Third part done (`f4816d56`, `3023494e`): np.pad,
   with the D4 fixes its replay found (`33fd5151`, `839950c2`, `54ad9427`). Fourth part done (`5f4d6015`, `ac83cba8`,
   `b07d19d2`, `f66c85cd`): np.r_ / np.c_, np.ix_ and np.fill_diagonal (`IndexTricks`), with the D4 fixes their
   replay found (`70646c73`, `cf75f39f`, `d6b4926e`); np.fill_diagonal's StringDType route asks `HasReferences`.
   Fifth part done (`19dbc34b`, `70533e1b`, `b25f717d`): np.take_along_axis, np.put_along_axis and
   np.apply_along_axis (`ShapeBase`) with `NDVStringStatistics.Quantile.cs`' two pins, and the D4 fixes its replay
   and the full test run found (`10626746`, `62683028`). Sixth part done (`37ac64c2`, `1d375850`, `3df20fb6`,
   `aa7b3f77`): np.indices (`Numeric`) and np.diag / np.diagflat (`TwoDimBase`), with the D4 fix the replay found
   (`408644f1`); np.diag's StringDType route asks `HasReferences`. Seventh part, first half, done (`c89c5dc3`,
   `646a15a0`, `4a634a37`, `dc1ce4f6`): np.mgrid / np.ogrid (`IndexTricks`), with the D4 fixes the replay found
   (`e0d2f1ef`, `6f0f2e33`, before the move); their mesh dtype asks `result_type`. Second half done (`ac239976`,
   `61128970`, `b85c03c0`, `a60ec561`): np.ravel_multi_index (`TensorEngine.RavelMultiIndex`, `Multiarray`, with
   `astype_anyint`), with the D4 fix the replay found (`eb82d855`, before the move); `astype_anyint` asks the
   descriptor's kind. Batch 7 is done.
8. **The indexers** (8 files, 98 lines; `Selection/NDArray.Indexing.*.cs`, NumPy's `mapping.c`; already the array
   object): the `StringDType` routes behind `HasReferences`, the typed fetch / store switches into
   `UnmanagedStorage.Mapping.cs`. The index-kind tests stay. Done (`09d818fa`, `8a102c92`, `25ae1802`, `3aa50b5d`),
   with the thirteen D4 fixes the replay and a getter / setter differential found (`086b6d0c` .. `932cef87`, before
   the move): the typed fetch and store with their subspace copies are `UnmanagedStorage.MapIterGet` / `MapIterSet`
   (protected forwarders keep NDArray's old names), the five StringDType decisions ask `HasReferences`, and R1's eight
   files 98 -> 58 lines (the fixed-width text routes' unit view and the StringDType route bodies stay, see the state
   log).
9. **The computing functions in P3's files** (5 files, 30 lines) become `TensorEngine` members (Appendix D):
   np.linspace, np.logspace and np.geomspace (`TensorEngine.CoreFunctionBase.cs`), np.vander
   (`TensorEngine.TwoDimBase.cs`), np.trim_zeros (`TensorEngine.FunctionBase.cs`). Done (`ca9a0246`, `315b2f0e`,
   `7c93e25a`), and the six divergences a replay against NumPy found are fixed per D4 (`a7c23095`, `88afce1d`,
   `fb17feec` with `beb55575`, `5de82c92`, `681f9608`); no slot conversion (np.trim_zeros' text truth stays a bulk
   kernel, see the state log).
9b. **The text casts through the casting impls.** The text classes' `CastingImpl`s (`StringCastingImpls.cs`,
    `VStringCastingImpls.cs`; their `resolve_descriptors` exist) gain `get_strided_loop`: the converters NumSharp
    calls by name today (`NDTextConversion` for text and numbers, `NDStringTruth`'s cast flavor for text to bool, the
    unit copies of `NDStringCopy` between widths and between `S` and `U`) become their loops, and their refusals
    (`NDStringCasting.InvalidCast`, `NDStringCopy.ThrowIfCannotConvert`) the impls' answers. The assignment
    (`UnmanagedStorage.Assign*.cs`: NumPy's `raw_array_assign_array` and its masked form, chunked as NumPy's casts are),
    np.concatenate's string route and the later batches' casts then get their loop from `DTypeCasting`
    (`PyArray_GetDTypeTransferFunction`) instead of testing the `S` / `U` lane. It follows the moves because one set of
    loops serves batches 3, 4, 6 and 8.
10. **What stays** (5 files, 13 lines, and the index-kind tests of the other batches), listed by R1 after P3 with its
    reason: the C# conversions (`NDArray.Implicit.ValueTypes.cs`, `NDArray.ToString.cs`: a C# cast is not a NumPy
    statement; they call `getitem` and the coercion path), the DType layer's own files (`np.dtype.cs`, `np.dtypes.cs`)
    and the typed iterator's type check (`np.ndenumerate.cs`); outside R1, the IO codecs, the Python-object helpers,
    `DTypes/`' own slot code and np.random's text refusals.

### P4. numpy.polynomial and numpy.ma (K5, 61 lines in 14 files)

- [x] (2026-10-28, `39ae0ffb`; the D4 fixes the twin names landed 2026-10-29, `fb9b62a0` to `1b0bd94d`, see the state
      log) The legacy facades
      `np.polyval`, `np.polyint`, `np.polyarithmetic`, `np.poly` and `poly1d` move their bodies into
      `TensorEngine` members in `TensorEngine.Polynomial.cs`, after `lib/_polynomial_impl.py` (D1). The package internals
      (`NDPolySeries`, `NDPolyCalc`, `NDPolyVander`, `NDPolyEval`,
      `NDMaskedArray`) are O already: their dtype statements are NumPy's Python statements, and they stay.

### P5. Numeric-only functions

- [x] (2026-10-29) Classify the numeric lines left in facade files after P1 to P4 (C9): below.
- [x] (2026-10-29) Move P5's families, one per commit, by the method of [§6](#6-method-per-batch), and the
      coordinator's lines below, handed over the same day ([§9](#9-state-log)).

**The classification (2026-10-29, at `ab206a01`).** R1's own scanner (`LayeringScanner.CountFacadeDtypeTests`, copied
into a file-based app that prints every counted line) finds 717 numeric and 255 text lines in 121 facade files. C9's
classes:
- **A**, NumPy's statement: it moves with the function's body to its home.
- **B**, a NumSharp fast path: it moves into the engine member or the kernel.
- **C**, a C#-only conversion or a class-boundary statement: it stays in the facade and R1 keeps it listed.
- **D**, core code: the dtype layer or a class's own code. It does not move.

Where the numeric lines go:
- **P5, 23 lines in 9 files, six families.** Each facade holds only numeric lines, and each home is a partial this
  phase may edit:
  - np.nan_to_num, np.isreal / np.iscomplex and np.real_if_close, in a new `TensorEngine.TypeCheck.cs`
    (`lib/_type_check_impl.py`);
  - the set routines np.union1d / np.intersect1d / np.setxor1d / np.setdiff1d, in `TensorEngine.ArraySetOps.cs`;
  - np.sort_complex (`FunctionBase`) and np.roots (`Polynomial`);
  - np.conjugate's loop resolution (NumPy's conjugate has no bool loop) and np.isposinf / np.isneginf's complex
    refusal, into the members they already call (`DefaultEngine.Conjugate`, `IsPosInf`, `IsNegInf`).
- **The coordinator's engine files, 25 lines** (handed to P5 on 2026-10-29 and moved; the table's last column says
  so). np.argmax's out= pair, which np.nanargmax shares (`Methods`),
  np.bincount, np.packbits and np.unravel_index (`Multiarray`), and the text lines of np.kron and np.median.
- **E3, 4 lines:** np.fft.RawFft.
- **P3, 469 lines:** the array object, Creation/, IO, the indexers, the C# conversions and DTypes/. P3's own design
  classifies them line by line; the table gives the files.
- **Stay, 196 lines (C and D):**
  - the dtype routines np.can_cast, the np.isdtype family, np.find_common_type, np.common_type and np.min_scalar_type
    (111), whose home is the DType layer;
  - numpy.matrix's own class code (27);
  - the classes finfo, iinfo, memoryview, vectorize and flatiter (50), whose subject is the dtype;
  - the matrix rule's result dtype in the np.nanmean / np.nanstd / np.nanvar facades (3, P7's);
  - the typed iterator's and np.asscalar's C# type check;
  - np.random.multinomial's own statement, the polynomial package's port and the operators' text route.

The 255 text lines are P3's K4 files, np.evaluate's two (P2's) and np.fft.RawFft's one (E3). Per file:

| File | numeric | text | Class | Home | Moves in |
|---|---|---|---|---|---|
| `Logic/np.isneginf.cs` | 1 | 0 | A (the complex refusal) | `DefaultEngine.IsNegInf` (`TensorEngine.UfuncLike.cs`) | P5 |
| `Logic/np.isposinf.cs` | 1 | 0 | A (the complex refusal) | `DefaultEngine.IsPosInf` (`TensorEngine.UfuncLike.cs`) | P5 |
| `Logic/np.isreal_iscomplex.cs` | 5 | 1 | A (isreal, iscomplex); D (isrealobj, iscomplexobj) | `TensorEngine.TypeCheck.cs` `Isreal` / `Iscomplex`; the two *obj predicates are dtype routines and stay (R1 keeps their 3 lines) | P5 |
| `Logic/np.nan_to_num.cs` | 10 | 0 | A, B | `TensorEngine.TypeCheck.cs` `NanToNum` (new partial, `lib/_type_check_impl.py`) | P5 |
| `Manipulation/np.setops.cs` | 2 | 0 | B (the NaN rewrite after unique) | `TensorEngine.ArraySetOps.cs` `Union1d` / `Setxor1d` / `Setdiff1d` (np.intersect1d needs no NaN pass) | P5 |
| `Math/np.conjugate.cs` | 1 | 0 | A (the ufunc's loop resolution: no bool loop) | `DefaultEngine.Conjugate` (`TensorEngine.Umath.cs`) | P5 |
| `Math/np.real_if_close.cs` | 1 | 0 | A | `TensorEngine.TypeCheck.cs` `RealIfClose` | P5 |
| `Polynomial/np.roots.cs` | 1 | 0 | A | `TensorEngine.Polynomial.cs` `Roots` | P5 |
| `Sorting_Searching_Counting/np.sort_complex.cs` | 1 | 0 | A | `TensorEngine.FunctionBase.cs` `SortComplex` | P5 |
| `APIs/np.evaluate.cs` | 0 | 2 | text | P2's last lines: needs a keyword-taking `TensorEngine.Evaluate` (see P2) | P2 (open) |
| `Indexing/np.unravel_index.cs` | 2 | 0 | A, B | `TensorEngine.Multiarray.cs` `UnravelIndex`; `ThrowIfNotIntIndices` (NumPy's `astype_anyint`, 1 line) is shared with np.ravel_multi_index and moves with it | P5 (handed over) |
| `LinearAlgebra/np.kron.cs` | 0 | 1 | text | `TensorEngine.ShapeBase.cs` `Kron` (file under LinearAlgebra/) | P5 (handed over) |
| `Manipulation/np.packbits.cs` | 11 | 0 | A, B | `TensorEngine.Multiarray.cs` `Packbits` / `Unpackbits` | P5 (handed over) |
| `Sorting_Searching_Counting/np.bincount.cs` | 3 | 1 | A | `TensorEngine.Multiarray.cs` `Bincount` | P5 (handed over) |
| `Sorting_Searching_Counting/np.nanargmax.cs` | 9 | 0 | A (np.argmax's out= contract, shared with np.argmax) | `TensorEngine.Methods.cs` (`ReduceArgMax` / `ReduceArgMin` with out=) | P5 (handed over) |
| `Statistics/np.median.cs` | 0 | 2 | text (the chararray rule's answer) | `DefaultEngine.MedianOfChararray` in the quantile family's engine partial (`Statistics/Default.Quantile.Median.cs`) | P5 (handed over) |
| `Fourier/np.fft.RawFft.cs` | 4 | 1 | A | `TensorEngine.Fft` | E3 (coordinator) |
| `APIs/np.fromfile.cs` | 0 | 6 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.fromstring.cs` | 0 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.loadtxt.Parsers.cs` | 15 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.loadtxt.Sinks.cs` | 0 | 24 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.loadtxt.cs` | 0 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.savetxt.cs` | 1 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.tofile.cs` | 6 | 5 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.where.cs` | 8 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Casting/Implicit/NDArray.Implicit.Array.cs` | 48 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Casting/Implicit/NDArray.Implicit.ValueTypes.cs` | 3 | 4 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Casting/NDArray.ToString.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Casting/NDArray.byteswap.cs` | 1 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Casting/NDArray.getfield.cs` | 0 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Casting/NDArray.tobytes.cs` | 0 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/NDArray.Scalar.cs` | 0 | 5 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.arange.cs` | 18 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.array.cs` | 3 | 5 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.array.string.cs` | 0 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.asanyarray.cs` | 0 | 4 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.asarray.cs` | 0 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.asarray_chkfinite.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.concatenate.Strings.cs` | 0 | 4 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.concatenate.cs` | 2 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.dtype.cs` | 2 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.eye.cs` | 4 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.frombuffer.Strings.cs` | 0 | 4 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.frombuffer.cs` | 93 | 12 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.full.cs` | 3 | 8 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.full_like.cs` | 1 | 4 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.geomspace.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.linspace.cs` | 23 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.logspace.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.mgrid.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.ndarray.cs` | 1 | 10 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.ogrid.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.ones.cs` | 4 | 6 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.r_.cs` | 8 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Creation/np.vander.cs` | 0 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `DTypes/np.dtypes.cs` | 0 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.choose.cs` | 7 | 7 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.compress.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.diag.cs` | 0 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.extract.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.fill_diagonal.cs` | 0 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.indices.cs` | 3 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.ix_.cs` | 2 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.ndenumerate.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.piecewise.cs` | 7 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.place.cs` | 2 | 9 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.put.cs` | 0 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.put_along_axis.cs` | 1 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.putmask.cs` | 2 | 4 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.ravel_multi_index.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.select.cs` | 10 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.take.FancyAxis.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.take.cs` | 4 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Indexing/np.take_along_axis.cs` | 9 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/NDArray.FunctionBase.cs` | 8 | 7 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/NDArray.fill.cs` | 1 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/NDArray.resize.cs` | 0 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.apply_along_axis.cs` | 1 | 0 | B (a span write when the dtypes match) | the array object (`NDArray.ShapeBase`) | P3 (coordinator) |
| `Manipulation/np.copyto.Strings.cs` | 2 | 15 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.copyto.VString.cs` | 0 | 7 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.copyto.cs` | 5 | 7 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.delete.cs` | 20 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.pad.cs` | 33 | 12 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.repeat.cs` | 9 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.resize.cs` | 0 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Manipulation/np.trim_zeros.cs` | 0 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Math/NDArray.ComplexAccessors.cs` | 2 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Math/np.imag.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Math/np.real.cs` | 1 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.Masking.cs` | 0 | 3 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.PrepareIndex.cs` | 12 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.Selection.Getter.cs` | 41 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.Selection.Setter.cs` | 23 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.Selection.cs` | 4 | 0 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.Strings.cs` | 3 | 1 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.VString.cs` | 0 | 5 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `Selection/NDArray.Indexing.cs` | 1 | 2 | P3's design | the array object (Appendix D) / IO / the DType slots | P3 (coordinator) |
| `APIs/np.finfo.cs` | 11 | 1 | D (the dtype is the class's subject) | the class | stays |
| `APIs/np.flatiter.cs` | 1 | 1 | D | the class | stays |
| `APIs/np.iinfo.cs` | 21 | 1 | D (the dtype is the class's subject) | the class | stays |
| `APIs/np.memoryview.cs` | 16 | 2 | D (the buffer protocol's format codes) | the class (or the `DType` slots, P3) | stays |
| `APIs/np.nditer.Typed.cs` | 1 | 0 | C (the C# T against the dtype) | the typed iterator | stays |
| `APIs/np.vectorize.cs` | 2 | 0 | D | the class | stays |
| `Logic/np.can_cast.cs` | 80 | 1 | D (a dtype routine) | the DType layer (`DTypeCasting`; DTypes/ is P3's) | stays |
| `Logic/np.common_type.cs` | 6 | 1 | D (a dtype routine) | the DType layer | stays |
| `Logic/np.find_common_type.cs` | 10 | 2 | D (a dtype routine) | the DType layer | stays |
| `Logic/np.min_scalar_type.cs` | 0 | 1 | D (a dtype routine) | the DType layer | stays |
| `Logic/np.type_checks.cs` | 15 | 2 | D (dtype routines: issctype, isdtype, sctype2char, ...) | the DType layer | stays |
| `Manipulation/np.asscalar.cs` | 1 | 0 | C (the C# T against the dtype) | the facade | stays |
| `Matrixlib/NDMatrix.Construction.cs` | 8 | 0 | D | the class | stays |
| `Matrixlib/NDMatrix.Indexing.cs` | 3 | 0 | D | the class | stays |
| `Matrixlib/NDMatrix.Methods.cs` | 1 | 0 | D | the class | stays |
| `Matrixlib/NDMatrix.Operators.cs` | 12 | 0 | D (numpy.matrix's defmatrix.py statements) | the class | stays |
| `Matrixlib/NDMatrix.Reductions.cs` | 2 | 0 | D | the class | stays |
| `Matrixlib/NDMatrix.cs` | 1 | 0 | D | the class | stays |
| `Operations/Elementwise/NDArray.TextComparison.cs` | 0 | 1 | C (the operators' text route) | the operators (P1c) | stays |
| `Polynomial/Package/np.polynomial.polyutils.cs` | 1 | 1 | D (the package's port; the numeric hit is a PolyNumberKind test) | numpy.polynomial | stays |
| `RandomSampling/np.random.multinomial.cs` | 1 | 0 | A (np.random's own statement) | np.random (outside both) | stays |
| `Statistics/np.nanmean.cs` | 1 | 0 | C (the matrix rule's result dtype) | the facade's class boundary (P7) | stays |
| `Statistics/np.nanstd.cs` | 1 | 0 | C (the matrix rule's result dtype) | the facade's class boundary (P7) | stays |
| `Statistics/np.nanvar.cs` | 1 | 0 | C (the matrix rule's result dtype) | the facade's class boundary (P7) | stays |

### P6. The engine stops calling np.*

- [ ] The computational up-calls left after P1 to P5 become engine or core calls; the structural ones become calls into
      the array object's members (the allocation, `UnmanagedStorage.Assign.cs`'s assignment for `copyto`,
      `broadcast_to`, the buffer size). R2 ends at zero computations; any structural call left is listed with its
      reason. Then `DefaultEngine`'s partials of NumPy's Python layer move under `Backends/Default/`, the array object's
      beside their classes (C3), and the shape math `DefaultEngine` holds moves to `Shape` partials (E7).
  - [x] The computational part (2026-10-29, see the state log): R2's computations 132 -> 16, each converted call the
        engine member its facade dispatches to, every family snapshot-identical. The 16 left are pinned in
        `engine_upcall_allowlist.tsv` with their reasons: np.where (5) and np.apply_along_axis (2) are the array
        object's (§1, decided 2026-10-28) and leave when P3's batch 7 gives them their `NDArray` homes;
        `Backends/Strings/NDStringMethods.cs` (9) holds numpy.strings' own statements, the text API's pass (§8).
  - [ ] The structural calls: 263 at `c4cc72d9`, 82 after families 1 to 5 (2026-10-29, see the state log: the
        copies, the allocations, the broadcasts, the buffer size, the one-call views and copies), 78 after P3's batch 8
        took the indexers' four, 47 after family 6 (2026-10-30: np.ascontiguousarray, np.expand_dims, np.flip and
        np.ravel, whose statements became array-object members). Left: the coercion calls, the promotion helpers, the
        text API's statements, the write-once policy and the `flatiter` object.
  - [ ] The moves under `Backends/Default/`, beside their classes and into `Shape` partials.

### P7. The class boundary

- [ ] Its own plan (D7).

### Later, outside this plan

A loop registry (NEP 43 `ArrayMethod` loops for ufuncs, registered per DType class, one dispatcher raising NumPy's
no-loop error) would replace the 205 per-member text guards inside `DefaultEngine`. `docs/plans/dtype-system.md`
already specifies it for ufuncs; today `ArrayMethod` serves casts only.

D8's pass, if the owner takes it: of the 61 NumPy Python functions without dtype logic (marked † in
[Appendix D](#appendix-d-where-every-numpy-function-lives)), the 12 that compute become `TensorEngine` members and the
47 of the array object move into their `NDArray` / `Shape` partials, by the same method. D9 is the open question of
installing an overriding engine for every array.

The ten array-object members `TensorEngine` shipped in 0.70.0 (E7) stay until a decision of their own: removing them
breaks a subclass or a direct caller for a layering reason alone. `CreateNDArray`, which nothing calls, is the first
candidate.

---

## 8. Out of scope

- `np.strings`, `np.char` and `Char/` (the text API itself, 81 lines): dtype-specific by definition. They follow the same
  F / O rule when their own pass comes.
- The engine's per-member text guards (205 lines in 62 files): already deep. See "Later".
- numpy.random, the numpy.polynomial package and numpy.ma keep their own object models (D1). Their internals do not
  become `TensorEngine` members.
- The class boundary (P7).
- Removing the ten array-object members `TensorEngine` shipped in 0.70.0 (see "Later").
- Behavior changes, except a divergence found and proven during a move (D4).

---

## 9. State log

- **2026-10-30, P6: the structural calls, family 6, and the two defects the family snapshots found (D4).** Three
  commits on `strings` from the parallel session, developed in the scratch worktree and landed by fast-forward:
  `dc807d59` first (on `932cef87`), then family 6 and the second fix, rebased onto `b60cce17`. R2's structural calls
  78 -> 47 (batch 8 took 82 -> 78), 46 -> 30 files; its 9 computations are unchanged.
  - `dc807d59` (D4; the snapshot class "np.true_divide of a decimal operand with a zero divisor into a float64 `out=`"
    of the families 1 to 5 entry): `NDIterRef.Dispose` is NumPy's `NpyIter_Deallocate`. When one of the iterator's
    drivers threw, the held buffers are cleared (`NDIterBufferManager.ClearBuffers`, NumPy's `npyiter_clear_buffers`)
    and the COPY_IF_OVERLAP temporaries discarded; the fills completed before the error stay written, and a write-back
    that throws on the success path is a failure too. The flag, `NDIterState.Failed`, is set by the drivers' exception
    filter `NDIterRef.MarkFailed` (ForEach, ForEachRows, ExecuteGeneric, ExecuteReducing, BufferedReduce,
    ExecuteBinary, ExecuteUnary, ExecuteReduction, ExecuteComparison, ExecuteScan, ExecuteCopy and the 2-D block
    kernels). A loop a caller writes around the iterator (np.nditer's element walk) is not a failure: its buffers are
    written back, as NumPy's `with` block writes them back. Tests `Backends/Iterators/NDIterFailedIterationTests.cs`
    (4; three fail on the parent). The drivers' try / catch-when costs nothing measurable (six small calls, 350 to
    3,094 ns, within the runs' spread).
  - `4baf2671` family 6, 31 calls. np.ascontiguousarray, np.expand_dims, np.flip and np.ravel held statements of their
    own after their class boundaries, so the statements moved into the array object's homes (Appendix D) and the
    facades call them: `NDArray.AsContiguousArray(a, dtype)` (`Creation/NDArray.Multiarray.cs`, NumPy's
    `array_ascontiguousarray`; 14 calls), `NDArray.ExpandDims(a, axis)` (`Manipulation/NDArray.ShapeBase.cs`; 13),
    `NDArray.FlipAxes(m, axes)` (np.flip's view, moved into `Manipulation/NDArray.FunctionBase.cs`; 1) and
    `NDArray.Ravel(a, order)` (NumPy's `PyArray_Ravel`, moved from np.ravel, which np.ravel and ndarray.ravel call; its
    KEEPORDER test is `Shape.IsDenseInSomeAxisOrder`; 3). The snapshot of the 36 functions that reach the sites
    (572,583 lines) is the parent's but for np.divmod's `where=` results: 92 keys after the rebase, all np.divmod
    `where=` cases, whose masked-off elements of a fresh result are uninitialized (as in NumPy; 435 keys differ
    between two runs of the parent).
  - `f48b5677` (D4; the snapshot class "`np.array<T>(IEnumerable<T>, size)` of a shorter enumerable"): the overload is
    NumPy's `np.fromiter(data, dtype=T, count=size)`. Fewer items raise NumPy's `ValueError: iterator too short:
    Expected N but iterator had only M items.` (it returned `size` elements whose tail was heap contents), a longer
    enumeration is advanced exactly `size` times, an exception of the enumeration propagates, and a failed call frees
    its buffer at once. A negative size keeps its ValueError, the documented divergence (NumPy's `count=-1` reads every
    item, which is `np.array(IEnumerable<T>)` here). No caller in the repository: an `[Obsolete]` probe over the test
    projects, benchmarks, examples and tools found none. Tests `Creation/np.array.SizedEnumerable.Test.cs` (4; two fail
    on the parent).
  - The snapshot-noise list of the families 1 to 5 entry loses its two defects; the masked bool -> StringDType store,
    the 2**32 - 1 text repetition and np.divmod's `where=` results stay (pre-existing, the last as in NumPy).
  - Gates: NumSharp.Tests (20,904 on net10.0, 20,903 on net8.0) and the oracle gates IndexOracleTests,
    UndisposedIntermediateTests and Layering (42 on net10.0) pass at `f48b5677`
    (`TestCategory!=OpenBugs&TestCategory!=HighMemory`).
  - Next on P6's structural part: the coercion calls (np.array 4, np.asanyarray 3), the promotion helpers, the text
    API's pins, `np.PrefersWriteOnce` and `np.FlatIterator`.

- **2026-10-30, P3: batch 8 (the indexers) - batch 8 is done.** Eighteen commits on `strings`: the tool commit and the
  thirteen D4 fixes landed first by fast-forward (2026-10-29, rebased onto `780cca88`, tip `932cef87`), then the move,
  the conversion, the indexers' R2 up-calls and this docs commit (rebased onto `dc807d59`). The D4 fixes come before the
  move, so the move is pure over NumPy's statements. R1: the eight indexer files 98 -> 58 lines (14 -> 11 text, 84 -> 47
  numeric; 43 files, 36 text and 254 numeric lines over all facades). R2: 4 calls leave (three rows).
  - `09d818fa`: the snapshot tool reaches NDArray's indexers - the accessors `get_Item` / `set_Item`, the protocol
    member `__setitem__` and the fancy pair `GetIndices` / `SetIndices` - with index keys built from the operand's rank,
    the fancy pair's index lists and the stored values, through the item-selection wiring. 74,296 cases. Replayed
    through NumPy 2.4.2: 2,573 lines differed.
  - D4, before the move, each with NumPy-probed tests that fail on its parent (`test/NumSharp.Tests/Selection/*.NumPyPort.Test.cs`):
    - `086b6d0c` an index entry is classified as `prepare_index` walks it: a double / float16 / decimal is refused with
      NumPy's text (a double was rounded: `a[0.6, ":"]` returned row 1), an integer beside another entry on a 0-d
      array is "too many indices", a 0-d string array names a refused entry before the rank is counted;
    - `0978e4c2` a store converts its value with the array's descriptor after the index (`__setitem__` of a Python int
      past every C# integer packs it; a list or array into one element is the dtype's setitem's; None after the
      index's checks; a `ulong` index basic; a 0-d integer array per axis a full integer index);
    - `e4b4b8d5` a Python bool is NumPy's 0-d bool array: alone the 0-d mask, beside other entries the advanced block;
    - `798804a3` a 0-d integer array with no index array beside it is `HAS_SCALAR_ARRAY`: a view, copied with
      KEEPORDER by the getter, assigned through by the setter;
    - `ed09b132` a store's order: the index, the value's conversion, the index arrays' values;
    - `6683ede9` a view store is `PyArray_AssignArray`, its leading unit dimensions stripped;
    - `31608c83` a list into a selection is converted with the array's descriptor by NumPy's route (every item by its
      setitem; a view's discovery limited to its rank; a StringDType fancy store through MapIter's temporary);
    - `734b06e7` `GetIndices` / `SetIndices` are `a[tuple(indices)]`;
    - `87e85f4d` the StringDType fancy store's temporary takes its shape from the index (`MapIterResultDims`, no proxy
      gather), so the broadcast error wins over an out-of-bounds index as in NumPy.
    - After them the replay differed in 681 lines, every class one NumSharp keeps: 660 "flags only" (an advanced
      result's layout and ownership flags: NumPy returns MapIter's result as a view of its buffer - transposed for an
      index behind a slice -, NumSharp a C-ordered array that owns its data; the values are equal; the layout class is
      pinned in `OpenBugs.FancyIndex.cs`), 14 stores whose
      OverflowError texts follow the platform's C long (Windows NumPy against NumSharp's LP64 pin) and 7 stores through
      an F-ordered index array that repeats a position (NumPy documents the last write to a repeated position as
      unpredictable).
    - Then a differential of the getter and the setter against NumPy over 6,000 random advanced keys (index arrays of 0
      to 2 dims, an empty one, a column, slices, integers, the ellipsis, new axes, C# and 0-d bools, a mask, on 0-d to
      4-D and empty arrays; the store writes an arange of the getter's shape through the same key), replayed with the
      integers spelled as C# integers, `Slice.Index(n)`, `"n"` and longs: `8bdc78a0` the errors in NumPy's order
      (prepare_index's boolean shapes, the integers' bounds, the index arrays' broadcast with NumPy's text, a space after
      every shape, then their values; 742 -> 16 of 12,000 outcomes), `7dfc898c` the caller's key array is never written
      (the conversions left arrays in it that the `[NDScoped]` dispatch had disposed, their buffers released, and a
      reused key read them; found while auditing the key forms), `2efc4ad9` an integer slice and an integer's notation
      beside an advanced entry are NumPy's integer (1,693 -> 16 in that spelling), `932cef87` the advanced block goes in
      front whenever a slice, a new axis or the ellipsis separates the advanced entries, a zero-span ellipsis included,
      in the getter's routes and the setter's single-index-array route (16 -> 0). After them 0 of the 12,000 differ in
      each spelling, and the replay is the same 681 lines.
  - `8a102c92`, the move: the typed fetch and store the indexers end in are `UnmanagedStorage` members in the new
    `Selection/UnmanagedStorage.Mapping.cs` (NumPy's mapping.c below the indexers; beside them because their statements
    call np.copy, np.broadcast_arrays, np.broadcast_to, np.atleast_1d, np.zeros, np.ones and np.any): `MapIterGet` (the
    typed dispatcher) with `MapIterGet<T>`, `MapIterGetSubspace<T>` and `MapIterGetSubspaceStrided<T>`; `MapIterSet`
    with `MapIterSetFromText`, `MapIterSet<T>`, `TryScatterKernel<T>`, `MapIterSetSubspace<T>` and
    `MapIterSetSubspaceStrided<T>`; the offset backstop and its message. The bodies are the old ones, extracted by a
    script, renamed, made internal and calling `NDArray.NormalizeIndexArray` / `PrepareIndexGetters` /
    `PrepareAssignedValue` qualified; the two index-array helpers stay with the indexers (index-kind tests) and are
    protected internal. NDArray keeps the eight protected members under their old names as forwarders (`<inheritdoc
    cref>`): types deriving from NDArray could call them, and AuditV2's T1_15a invokes `SetIndicesNDNonLinear` by
    reflection. The 19 internal call sites call the storage members. Every moved member is documented in full (most had
    no docs, or a copy-pasted summary naming a parameter that does not exist). Snapshot byte-identical to the parent's;
    the rebind check found only the moved members binding to each other, `Shape` in its Color Color form and type
    arguments; no doc warning in the touched files with Core's NoWarn cleared.
  - `25ae1802`, the conversion: the five StringDType route decisions (the masked get and set, the advanced get and
    set's string entry, `__setitem__`'s MapIter temporary) ask `dtype.Meta.HasReferences`, NumPy's
    `PyDataType_REFCHK`, instead of `Storage.IsVString`. Snapshot byte-identical (StringDType, `U` and `S` operands in
    its pools).
  - `3aa50b5d`: the four up-calls P6's families left for this batch - np.broadcast_to in `Default.BooleanMask.cs` (2)
    and `UnmanagedStorage.Setters.cs` (1), np.asanyarray in `NDArray.Contains` (1) - are `NDArray.BroadcastTo` and
    `NDArray.AnyArrayFromObject`. Snapshot of the indexers and `Contains` / `__contains__` (75,251 cases)
    byte-identical.
  - What the batch leaves in R1's indexer files, by mechanism: 46 index-kind tests (an index array's bool or integer
    dtype, the index-op kinds) and `KeepLaneSharingClass`' descriptor-class test (a result allocated by its lane takes
    the source's lane-sharing class, which NumPy's allocation from the source's descriptor gives for free; it goes when
    the typed allocations take a descriptor); the fixed-width text routes' decisions (`typecode == NPTypeCode.String`:
    an `S` / `U` element has no typed kernel wider than 16 bytes, so the index runs on the array's unit view - a
    representation of the storage, not a NumPy decision) and their None store's class singleton; the StringDType route
    bodies' calls into the text backend (`NDVStringMove`). AssignThroughView's StringDType item store
    (`view.IsVString`) is the setitem slot's question and stays with batch 4's packing.
  - Errata of P6's family commits (the parallel session's; commit messages cannot be amended): `effad8e6` says 42
    functions where it converted 41, `ac6cbacb` 37 where 36.
  - P3's tool commits (this batch's `09d818fa` among them) join the 0.71.0 notes' snapshot-tool line when P3 ends.
  - Gates: NumSharp.Tests (20,896 on net10.0, 20,895 on net8.0) and the oracle gates IndexOracleTests,
    UndisposedIntermediateTests, BufferReleaseSweep and Layering (42 on net10.0) pass after each commit
    (`TestCategory!=OpenBugs&TestCategory!=HighMemory`; one wall-clock test, T2_NonZero_Is_Much_Slower_Than_NumPy,
    failed once under the parallel session's load and passed three reruns alone).
  - Next on P3: batch 9b (the text casts through the casting impls) and batch 10 (what stays).

- **2026-10-29, P6: the structural calls, families 1 to 5.** Five commits on `strings` from the parallel session
  (developed in the scratch worktree on `c4cc72d9`, landed by fast-forward). Each family is one commit: the engine's
  calls of one kind become the array object's member that holds the facade's statements after its argument handling,
  the snapshot of the functions that reach the sites is identical to the parent's, and R2 is regenerated. R2's
  structural calls 263 -> 82 (264 when the item was written), 81 -> 49 files; its 9 computations are unchanged
  (the pinned `NDStringMethods.cs` statements).
  - `effad8e6` np.copyto, 63 calls: `UnmanagedStorage.Copyto(dst, src, casting, where)`
    (`Manipulation/UnmanagedStorage.Assign.cs`) holds np.copyto's statements after its casting parse (the matrix
    region, the null check, the mask's conversion, `PyArray_AssignArray` or a None source), and np.copyto's array
    overload calls it; the engine passes the casting level it spelled (`NPY_UNSAFE_CASTING`, or `NPY_SAME_KIND_CASTING`
    where it relied on the default). NanVarStd's weak-zero store is `CopytoPythonScalar`. 802,542 lines.
  - `ac6cbacb` the allocations, 51 calls: np.zeros / np.empty are the allocation constructor
    `new NDArray(dtype, shape, fillZeros)` (their facades' own statement), np.ones / np.full / np.eye `NDArray.Ones` /
    `NDArray.Full` / `NDArray.Eye`, np.ones_like its plain operand's statements, np.arange(stop)
    `UnmanagedStorage.ArangeObj`. 332,730 lines.
  - `f727ebc1` np.broadcast_to / np.broadcast_shapes, 30 calls: `Shape.BroadcastTo` (np.broadcast_to's one-way check,
    `ValidateBroadcastTo`, moved from np, and its layout; `View/Shape.Broadcasting.cs`) for a shape source,
    `NDArray.BroadcastTo` (the read-only view, in the new `Creation/NDArray.StrideTricks.cs`) for an array source,
    `Shape.BroadcastShape` (np.broadcast_shapes' body) for the text routes' joint shapes; np.broadcast_to's nine
    overloads call the two members. 609,444 lines.
  - `4cb1a1cd` the buffer size, 16 calls: `NDIterBufferManager.CurrentBufferSize`, np.getbufsize's whole body and what
    NDIter's buffering already read; its doc no longer says the buffer size never changes a computed result. 260,049
    lines.
  - `7db8e5fc` the views and copies whose facade is one array-object call, 21 calls: np.transpose / np.moveaxis the
    shipped `TensorEngine.Transpose` / `MoveAxis`, np.reshape `x.reshape(shape)`, np.copy `x.copy('K')` (np.copy's
    default order, where the method's is C), np.take `UnmanagedStorage.TakeFrom` with mode raise, np.atleast_1d of a 0-d
    array its `(1,)` view. 582,474 lines.
  - Not converted yet: np.ascontiguousarray (14), np.expand_dims (13), np.flip (1) and np.ravel (3), whose statements
    need an array-object member first; the coercion calls (np.array, np.asanyarray), the promotion helpers
    (`_FindCommonType` and its relatives, np.result_type, np.promote_types, np.can_cast), the text API's statements
    (`np.strings.OutputDescriptor`, `np.dtypes.StringDType`), the write-once policy (`np.PrefersWriteOnce`) and the
    `flatiter` attribute's object (`np.FlatIterator`). Left for after batch 8: `Default.BooleanMask.cs`
    (np.broadcast_to, 2), `UnmanagedStorage.Setters.cs` (np.broadcast_to, 1) and `NDArray.Container.cs`
    (np.asanyarray, 1).
  - The class boundaries do not fire at a converted call: the engine runs inside its public function's boundary, as
    both boundaries' caller audits prove over the corpus.
  - Snapshot noise: two runs of the parent differ on some keys, and those keys are left out of the comparison
    (`snapcmp.py`: a key that varies between parent runs, or whose outcome a rerun of the change matches). Every class
    is pre-existing:
    - a masked bool -> StringDType store, whose staged cast reads the masked-off bytes of an uninitialized bool result
      (it raises "invalid value encountered in bool to string cast" or not, by memory content);
    - a text repetition of 2**32 - 1, which fails to allocate or not, by memory pressure;
    - np.divmod's `where=` results, whose masked-off elements of a fresh result are uninitialized, as in NumPy;
    - `np.array<T>(IEnumerable<T>, size)` of a shorter enumerable, whose tail is uninitialized memory (a defect: NumPy's
      np.fromiter raises "iterator too short");
    - np.true_divide of a decimal operand with a zero divisor into a float64 `out=` (a defect, reproduced by a
      4,000-call probe): `NDIterRef.Dispose` writes a buffered iteration's buffers back even when its inner loop threw,
      where NumPy's `NpyIter_Deallocate` clears them on error (`npyiter_clear_buffers`) and discards WRITEBACKIFCOPY.
      The out receives the half-filled decimal buffer (3,982 of 4,000 calls), and when a garbage decimal's scale byte
      passes 28 the write-back's `Decimal.VarR8FromDec` raises IndexOutOfRangeException in place of the
      DivideByZeroException (15 of 4,000).
  - Core's warnings: one NDW012 fewer (`Default.Reduction.VarStd.cs`, RowBlockDeviationSum's accumulator), a false
    negative of the analyzer: the accumulator is still left to the finalizer, but its reshaped view now reaches
    `TensorEngine.Transpose`, which NDW012 counts as egress where it knew np.transpose as a NumSharp op.
  - Gates: NumSharp.Tests (20,855 on net10.0, 20,854 on net8.0) and the oracle gates IndexOracleTests,
    UndisposedIntermediateTests and Layering (42 on net10.0) pass at `7db8e5fc`
    (`TestCategory!=OpenBugs&TestCategory!=HighMemory`).
- **2026-10-29, P3: batch 7, seventh part, second half (np.ravel_multi_index) - batch 7 is done.** Five commits on
  `strings` from the parallel session (developed in the scratch worktree on `c98a8245` and landed by fast-forward). The
  D4 fix comes before the move, so the move is pure over NumPy's statements. R1: 2 numeric lines leave with the move
  (np.ravel_multi_index.cs 0 / 1, its int64 in-place test; np.unravel_index.cs 0 / 1, `astype_anyint`): batch 7 has no
  row left (43 files, 39 text and 291 numeric lines). R2: unchanged (the engine file sits beside the facade).
  - `ac239976`: the snapshot tool reaches np.ravel_multi_index (its three overloads were SKIP). The array overloads'
    coordinate list carries the pool operand (role `RavelCoords`, 12 variants: the operand twice, with a C- and an
    F-ordered int64 ramp, a 0-d coordinate, a broadcast row, a vector that does not broadcast, a bool and a float array,
    too few and too many, none, a null entry); dims, mode and modes take roles of their own (`RavelDims`, `RavelMode`,
    `RavelModes`; the order is np.unravel_index's `UnravelOrder`), and the scalar overload's coordinate tuple is its
    operand (`RavelScalarCoords`, 11 labels). 29,106 cases, the same with and without `NUMSHARP_DEBUG_GUARD_PAGES=1`.
    Replayed through NumPy 2.4.2 on Windows and on Linux (28,354 cases, the char and decimal pools skipped): 315
    diverged, all one class.
  - `eb82d855` (D4, before the move): a null coordinate array is NumPy's None, which `astype_anyint` refuses:
    `TypeError: only int indices permitted`, in the list's order after the count, mode, order and dims checks, where
    NumSharp leaked a NullReferenceException. The 315; no replay case differs after it, on Windows or Linux.
  - `61128970`, the move: np.ravel_multi_index's array body is the abstract `TensorEngine.RavelMultiIndex` in
    `TensorEngine.Multiarray.cs`, implemented in `Indexing/Default.RavelMultiIndex.cs` beside the facade - the null
    checks, the order, the modes, the ravel strides and the count, then the old body, statement for statement - with
    `BroadcastCoordinates`, `HasDims`, `ExecuteRavelMultiIndex`, `ValidateRavelInputs` and `astype_anyint`
    (`ThrowIfNotIntIndices`), which `Default.UnravelIndex.cs` now calls as DefaultEngine's own. The member's mode is an
    `object`, NumPy's mode argument (a string for every axis, a sequence of strings, None), so the two public array
    overloads share it; `np.ConvertClipmodeSequence(object, int)` is NumPy's `PyArray_ConvertClipmodeSequence` for it
    (another object: `TypeError: clipmode not understood`). The scalar overload stays the facade's (`RavelStrides` and
    `ThrowIfZeroDimension` become internal). Snapshot of np.ravel_multi_index and np.unravel_index (38,178 cases)
    byte-identical, sorted; the rebind check found nothing.
  - `b85c03c0`, the conversion: `astype_anyint` asks the descriptor - kind `'b'`, `'i'` or `'u'`, NumPy's
    `PyArray_ISINTEGER || PyArray_ISBOOL` - where it listed Boolean, Char and the integer NPTypeCodes; the two in-place
    read tests keep the NPTypeCode (they ask about the storage lane the kernel reads). Snapshot byte-identical.
  - `a60ec561`: a unit test calls the engine member with NumPy's mode objects the public overloads cannot pass (a list
    of strings, a None entry, a float, an object, the wrong length, an empty dims), each with NumPy 2.4.2's probed
    answer.
  - Perf (two pinned P-cores; before the move / after the conversion / NumPy, us, best of four interleaved runs): small
    (2,) int64 coordinates 0.544 / 0.556 / 1.111, 1K int64 2.84 / 2.77 / 4.98, 100K int64 210.7 / 211.7 / 462.6, 100K
    int32 (the cast route) 284.7 / 285.7 / 501.5, 100K with modes (wrap, clip) 196.5 / 194.9 / 371.1, 0-d (NumPy's
    scalar) 0.482 / 0.493 / 1.081, the scalar overload 0.032 / 0.032 / 1.611: every cell within the runs' spread; NPY/NS
    1.8 to 50.
  - Core's warnings unchanged (712 in a `--no-incremental -v n` build; the two NDW012 of the moved body moved with it).
  - Gates: NumSharp.Tests (20,854 on net10.0, 20,853 on net8.0) and NumSharp.Tests.Oracle (414 on both) pass
    (`TestCategory!=OpenBugs&TestCategory!=HighMemory`, at `b85c03c0`), and so do the layering ratchets; after
    `a60ec561` its test class and the index-conversion tests (50) on both frameworks.
  - Next: batch 8 (the indexers, the coordinator's), 9b and 10.

- **2026-10-29, P3: batch 7, seventh part, first half (np.mgrid, np.ogrid).** Six commits on `strings` from the parallel
  session (developed in the scratch worktree on top of the sixth part, rebased onto `a0bf3516` and landed by
  fast-forward). The D4 fixes come before the move, so the move is pure over NumPy's statements. R1: 2 numeric lines
  leave (np.mgrid.cs 0 / 1 with the D4, np.ogrid.cs 0 / 1 with the move; neither file has a row now); R2: unchanged (the
  home is under `Indexing/`).
  - `c89c5dc3`: the snapshot tool reaches np.mgrid / np.ogrid: the indexer of each nd_grid object (np.MGridClass /
    np.OGridClass's get_Item) is called on that object and its key is the operand (role `GridKey`), built per call from
    77 labels (`GridKeyLabels`): single slices, meshes, Slice objects, the empty key, entries that are no slice, and the
    edges NumPy's Python arithmetic decides (zero steps over ints and floats, imaginary steps without a stop, the error
    order, -0.0, NaN / infinite / huge / underflowed sizes, Python ints past 2**53 and past int64). 154 cases, the same
    with and without `NUMSHARP_DEBUG_GUARD_PAGES=1`. Replayed through NumPy 2.4.2 on Windows and on Linux: 78 diverged.
  - D4, two commits before the move:
    - `e0d2f1ef`: a slice expression's fields are the Python numbers they spell (SliceSpec keeps a Python int exactly,
      `IntStart` / `IntStop` / `IntStep`, BigInteger; digits are an int, Infinity / NaN floats), and np.r_ / np.c_'s
      slice branch (np.mgrid / np.ogrid's single slice too) takes arange's int route over Python ints within int64: a
      zero step is `division by zero` (it was `float division by zero`), ints past 2**53 are exact
      (np.r_[2**53+1:2**53+3] had four elements of two). np.mgrid's mesh sizes divide Python ints exactly. 16 of the 78.
    - `6f0f2e33`: np.mgrid / np.ogrid are NumPy's `nd_grid.__getitem__` statements: the except branch for a key of one
      slice (`GridSingleSlice`: arange, or `arange(0, n) * step + start` for an imaginary step, a fresh array where
      linspace gave a view and forced the last point onto the stop), and the try branch for several (`GridAxisSize`:
      `int(abs(step))` / `math.ceil((stop - start) / step)` in Python's arithmetic, in key order; `GridMeshDtype`;
      indices or one arange per axis; `GridFillLine`: `i * step + start` in the mesh dtype, the step and start converted
      as NumPy's ufunc calls convert them). A zero step is Python's text, an imaginary step of one point needs no stop
      (with more, a single slice is Python's TypeError), a mesh's missing stop is raised where NumPy's AttributeError is
      (NumSharp's ValueError, now naming the slice), NaN / infinite / past-int64 sizes raise NumPy's texts where
      `(long)Math.Ceiling` was undefined, and an int64 mesh's start or step past int64 is NEP 50's conversion error (the
      LP64 text). The other 62.
  - `646a15a0`, the pure move: np.MGridClass's Build is `NDArray.MGrid`, np's NdGridLines `NDArray.OGrid`, with their
    helpers, in `Indexing/NDArray.IndexTricks.cs` (beside np.r_ / np.c_'s `AxisConcatenate`). The facades keep their
    result types, docs and the indexers' one call. Snapshot of np.mgrid, np.ogrid, np.r_ and np.c_ (7,426 cases)
    byte-identical; the rebind check found `Shape.Adopt` only.
  - `4a634a37`, the conversion: the mesh dtype is NumPy's `result_type(*num_list)` asked of `DTypePromotion.ResultType`
    over NEP 50's weak Python scalar classes (each distinct operand once), the single imaginary slice's
    `result_type(start, stop, step_float)` likewise, and the fills dispatch on the descriptor's kind. Snapshot
    byte-identical.
  - `dc1ce4f6`: a slice field's int literal is read in one scan (`ScanIntLiteral`), and the sizes are exact doubles with
    CPython's fast path for small ints; a `-0` field is the int 0 (+0.0 where it meets a float, np.r_[-0:3.0]).
  - After the D4 no replay case differs on Linux but the 7 missing-stop AttributeError cases (the documented
    divergence); Windows NumPy also says `int too big to convert` for the 2 int64-conversion cases.
  - Perf (two pinned P-cores; before the D4 / now / NumPy, us): every cell at parity or faster - mgrid[0:5, 0:3] 0.950 /
    0.931 / 12.79, mgrid[0:100, 0:100] 11.83 / 11.79 / 40.6, mgrid[0:1:100j, 0:1:100j] 20.14 / 19.51 / 33.5, ogrid[0:5,
    0:3] 2.31 / 1.94 / 10.69, ogrid[0:1:1000j, 0:1:1000j] 6.56 / 3.13 / 14.20, mgrid[0:1:100000j] 201.5 / 55.6 / 550.9,
    mgrid[0:5:0.5] 0.736 / 0.713 / 1.771; NPY/NS 1.7 to 13.7.
  - Core's doc warnings 3,828 -> 3,824.
  - Gates: NumSharp.Tests (20,850 on net10.0, 20,849 on net8.0) and NumSharp.Tests.Oracle (414 on both) pass
    (`TestCategory!=OpenBugs&TestCategory!=HighMemory`, run before the rebase onto `a0bf3516`); after it, the layering
    ratchets and the Creation tests on both frameworks, the snapshot of np.mgrid / np.ogrid / np.r_ / np.c_
    byte-identical.
  - Left for batch 7: np.ravel_multi_index (`TensorEngine.Multiarray.cs` with `astype_anyint`), the second half.

- **2026-10-29, P3: batch 6, third part (np.loadtxt, np.savetxt) - batch 6 is done.** Five commits on `strings`: the
  tool commit landed first (`0f092c78`), the others were developed in the scratch worktree on it, rebased onto
  `dbc282ed` (batch 7's sixth part) and landed by fast-forward. `06320e8b`'s message names the savetxt fix by its
  first hash, `d9917927` (now `222cfcb5`). R1: 26 text and 16 numeric lines leave (np.loadtxt.cs 2 / 0,
  np.loadtxt.Parsers.cs 0 / 15, np.loadtxt.Sinks.cs 23 / 0, np.savetxt.cs 1 / 1): batch 6 has no row left. R2:
  unchanged (the new homes are under `IO/`).
  - `0f092c78`: the snapshot tool reaches np.loadtxt and np.savetxt (each overload was one SKIP line). np.loadtxt's
    SOURCE is the operand: the pool's value sequence written as rows of text (a 0-d operand one field, a 1-D one a
    column, an N-D one its first axis' rows) in the C# form its first parameter takes - a temporary file, also
    gzip-compressed, a missing one, a byte stream, a TextReader, a list of lines - plain, comma-separated, with comments,
    with blank lines, with a ragged last row, quoted. np.savetxt writes X (the pool operand) into a sink whose bytes join
    the outcome: a temporary file, a MemoryStream, a StringWriter (new `Role.SinkWriter`). Every other parameter takes
    variants by function and name (`Role.TextIOArg`: comments, delimiter, converters from a menu the replay names alike,
    skiprows, usecols, ndmin, encoding, max_rows, quotechar; fmt, newline, header, footer), savetxt's fmt list is built
    from X's columns, and savetxt's phase B / C pools are the layouts it writes (a 0-d or 3-D X is refused first: with the
    general pools 41,228 of its 70,254 lines repeated that refusal). np.loadtxt 48,839 lines, np.savetxt 49,428: batch 6's
    snapshot is 298,015 lines. The other names' lines are byte-identical, the new ones identical over two runs and under
    `NUMSHARP_DEBUG_GUARD_PAGES=1`.
  - The replay of the new lines through NumPy 2.4.2 (`replay6c.py`): 98,267 lines, 1,576 C#-only (the char and decimal
    pools, C# null sources); 85,490 identical or in the documented non-native class (2,498 lines: NumPy's result compared
    as a native copy, with NumPy's own flags - a first run compared the copy's flags and flagged 1,298 squeezed results that
    were no difference), 11,201 differing, two faults fixed per D4 before the move:
    - `222cfcb5`: np.savetxt's string arguments are Python's None when null, refused where NumPy's statements meet the
      None - fmt at once (`invalid fmt: None`), delimiter where it is joined (AttributeError; a multi-`%` format never
      reads it), header / footer at `len()`, comments / newline in their concatenations (a real row's TypeError becomes
      NumPy's Mismatch, a complex row's stays plain), a footer's error after every row was written; a None spec in a list
      is `str.join`'s TypeError. NumSharp wrote the file, printed `invalid fmt: ` or threw NullReferenceException.
      11,001 lines.
    - `d92bd08f`: np.loadtxt's conversion error names a non-native dtype by its type string (`... to >i4 at row 0,
      column 2.`), NumPy's `%S` of the field's descriptor; NumSharp named the native lane. 200 lines.
    After them the replay has 0 differing lines: 96,691 identical (2,498 of them in the non-native class), 1,576 skipped.
  - `06320e8b`, the pure move: np.loadtxt's body - its "config" and "read_rows" regions (`_read`'s checks,
    `read_rows`) and the whole of np.loadtxt.Parsers.cs (the numeric field parsers, NumPy's `conversions.c` /
    `str_to_int.c`) and np.loadtxt.Sinks.cs (the sinks) - becomes `IO/TextReading/LoadTxtReader{,.Conversions,.Sinks}.cs`,
    one `internal static partial class LoadTxtReader`, so the three partials keep their cross-references; np.savetxt's
    "internals" region becomes `IO/SaveTxtIO.cs`. The facades keep their overloads, docs and source / target handling.
    The members the facades call became internal; np's `CI` was spelled out at its one use. Snapshot byte-identical.
  - `d1ce3bef`, the conversion: a converter's value is stored by the class's `setitem` (NumPy's
    `npy_to_generic_with_converter`: `PyArray_Pack`), called with the native twin of a non-native request because the
    sink writes the native lane; a float16 field outside the ASCII fast path is the float16 class's `setitem` (NumPy's
    `npy_to_generic`); the unsized-string test is the descriptor's kind and size (`_read`'s `dtype.kind in 'SUM'`);
    savetxt's complex test reads the kind (`np.iscomplexobj`), its Python-object rows ask `HasReferences`. Kept on
    purpose: the sinks chosen per DType class (NumPy's `get_from_ucs4_function` switches on the type number), the strict
    parsers per lane, the Mismatch text naming the lane. Snapshot byte-identical, and identical under guard pages.
  - Gates: Core's warnings unchanged (15,720); the text I/O tests (134) and the text I/O oracle tier pass; the D4 tests are
    NumPy-probed and fail on their parents (`IO/SaveTxtTests.cs` `NullStringArguments_AreNone_NumPysErrorsAtNumPysPoints`,
    `IO/LoadTxtTests.cs` `ConversionError_NamesTheRequestedDescriptor`). The four suites on the stack rebased onto
    `30fa03d0`: NumSharp.Tests 20,833 / 20,832 and the oracle suite 414 / 414 passed on net10.0 / net8.0 (11 and 2
    skipped), Release, `TestCategory!=OpenBugs&TestCategory!=HighMemory`; after the second rebase, onto `dbc282ed`, the
    build, the Layering ratchets (7) and NumSharp.Tests (20,839 / 20,838) passed again.
  - Next: batch 8 (the indexers), 9b and 10; batch 7's seventh part (np.mgrid / np.ogrid, np.ravel_multi_index) is the
    parallel session's. Two findings for the owner: np.median of complex128 data with a NaN returns the middle mean where NumPy returns the NaN
    element (62 cases of batch 7's np.pad replay); a C# int literal in an operator's no-loop TypeError is named int32 where
    NumPy names the Python int int64 (78 cases of batch 7's np.apply_along_axis replay; the known C#-literal typing).

- **2026-10-29, P3: batch 7, sixth part (np.indices, np.diag).** Five commits on `strings` from the parallel session
  (developed in the scratch worktree on top of the fifth part and landed by fast-forward). R1: 1 text and 3 numeric
  lines leave (np.diag.cs 1 / 0, np.indices.cs 0 / 3; neither file has a row now); R2: unchanged (the homes are under
  `Creation/`, not `Backends/`).
  - `37ac64c2`: the snapshot tool reaches np.indices / np.indices_sparse, which take no array: their dimensions are a
    shape operand built as a constructor's (an int[], a long[], a Shape) from a pool of their own (`IndicesShapeLabels`:
    the constructor shapes plus a (2, 2) grid, whose aranges a bool dtype allows, and a (200,) one, whose int8 arange
    wraps), every dtype over every shape; a sparse grid's overflowing shape makes no line (a 2**31-element arange on both
    sides). np.diag / np.diagflat's k, which was never varied, takes np.eye's variants, over a pool with a strided and a
    reversed vector. 2,772 cases (612 for np.indices, 2,160 for np.diag / np.diagflat).
  - `1d375850` and `3df20fb6`, the pure moves: np.indices' builders are `NDArray.Indices` / `NDArray.IndicesSparse` in
    `Creation/NDArray.Numeric.cs` (beside np.full / np.ones); np.diag's switch, np.diagflat's call and their diagonal
    scatter `DiagonalEmbed` are `NDArray.Diag` / `NDArray.DiagFlat` in `Creation/NDArray.TwoDimBase.cs` (beside np.eye).
    The facades keep their null checks, the int[] / Shape widening and np.diag / np.diagflat's numpy.matrix /
    numpy.char rules. Snapshot byte-identical under `NUMSHARP_DEBUG_GUARD_PAGES=1` and without it; the rebind check
    found the types `NDArray` / `Shape` / `TensorEngine` only.
  - `aa7b3f77`, the slot conversion: the diagonal scatter's StringDType route asks `DTypeMeta.HasReferences`. Snapshot
    byte-identical. np.indices had no class decision to convert (its typecode tests went with the D4).
  - D4, from the replay of the 2,772 cases through NumPy 2.4.2 on Windows and on Linux (2,532 replayed, char and decimal
    skipped): 369 of np.indices' 612 differed on both; np.diag / np.diagflat's 1,920 matched (on Linux 5 differ: the
    host-dependent uint32 / uint64 special operands the pool builds).
    - `408644f1`: np.indices is NumPy's statements. The dense grid is allocated first with the descriptor (a negative
      dimension is the allocation's ValueError `negative dimensions are not allowed`, an empty tuple of a text dtype an
      empty text array); one `arange(dim, dtype)` per axis refuses what NumPy's refuses (a text dtype at any length, bool
      past two elements: TypeError, even for an empty grid) and is assigned with broadcasting; an int64-lane grid keeps
      the IL kernel. The sparse grid is views of its aranges (`flags.owndata` False), a negative dimension an empty axis.
      369 cases.
  - After the D4 no replay case differs on Windows; on Linux the 5 host-dependent np.diag / np.diagflat operands.
  - Perf (two pinned P-cores): a `(1000, 1000)` float64 grid 618 us (x7.01 NumPy; it was 3,887 us, x1.11), int32 x6.06,
    float32 x6.03, a `(100, 100, 100)` float64 one x6.46; a `(30, 40)` grid of those dtypes x0.88-x0.92 (NumPy's fixed
    cost, two aranges and two broadcast copies); int64 keeps the kernel (x2.3 at `(30, 40)`, x4.2 at `(1000, 1000)`),
    where the arange route measures 616 us against the kernel's 1,036 us: a lever left for later.
  - Core's doc warnings 3,829 -> 3,828 (the D4).
  - Gates: NumSharp.Tests 20,837 / 20,836 passed on net10.0 / net8.0 (11 skipped) and the oracle suite 414 / 414 (2 skipped), Release, `TestCategory!=OpenBugs&TestCategory!=HighMemory`. Of the 6 new tests (`Indexing/np.indices.NumPyPort.Test.cs`, NumPy-probed on Windows and Linux), 5 fail on the parent of `408644f1`; the sixth pins the values the arange route assigns, which equal the old cast's.
  - Left for batch 7: the lines of np.mgrid, np.ogrid and np.ravel_multi_index.

- **2026-10-29, P3: batch 7, fifth part (np.take_along_axis, np.put_along_axis, np.apply_along_axis).** Five commits on
  `strings` from the parallel session (developed in the scratch worktree on `6f1da8dd`, rebased onto `0f092c78`,
  batch 6c's tool commit, and landed by fast-forward). R1: 3 text and 11 numeric lines leave (np.put_along_axis.cs 2 / 1,
  np.take_along_axis.cs 1 / 9, np.apply_along_axis.cs 0 / 1; the three files have no row now); R2:
  `NDVStringStatistics.Quantile.cs`' two np.apply_along_axis pins leave (2 computations -> 0, its pin's reason with
  them).
  - `19dbc34b`: the snapshot tool reaches np.apply_along_axis (both overloads were SKIP: a delegate first): its func1d
    from a fixed menu NumPy names alike (np.sum, the first element, a constant int64, a constant str, negate, np.cumsum, a
    float32 cast, the slice itself, v[:2], np.outer) and the *args overload's function with none, one or two args. With
    np.take_along_axis, np.put_along_axis and the nan-quantile functions whose text route drives np.apply_along_axis:
    254,172 cases.
  - `70533e1b`, the pure move: the new `Manipulation/NDArray.ShapeBase.cs` (NumPy's `lib/_shape_base_impl.py`) holds
    `NDArray.TakeAlongAxis`, `NDArray.PutAlongAxis` with its scatter core and `NDArray.ApplyAlongAxis` with their helpers;
    np's `PyShapeTuple` did not move (NDArray has its own, the same text for every shape the core passes). The facades keep
    their numpy.matrix / numpy.char rules and null checks; np.apply_along_axis's *args overload still calls the
    one-argument facade. `NDVStringStatistics.Quantile.cs` calls `NDArray.ApplyAlongAxis` (the two R2 pins). Snapshot
    byte-identical under `NUMSHARP_DEBUG_GUARD_PAGES=1`; the rebind check found PyShapeTuple (deliberate) and the types
    `NDArray` / `Shape` only.
  - `b25f717d`, the slot conversion: the three StringDType decisions (the cell gather, the cell scatter, the refusal of the
    raw span write) ask `DTypeMeta.HasReferences`. Snapshot byte-identical.
  - D4, from the replay of the trio's 13,896 cases through NumPy 2.4.2 on Windows and on Linux (13,152 replayed, char and
    decimal skipped; 714 / 722 differed):
    - `10626746`: np.apply_along_axis takes NumPy's `asanyarray` of its first result: a text scalar (an element `v[0]` of
      a fixed-width slice, `np.sum` of a StringDType slice) makes the buffer as wide as that first value (`<U1` for
      `'3'`; StringDType's Python str a U array), and later results are truncated to it (`TextScalarAsArray`). 45 cases.
    - `62683028`, found by the full test run (the strings oracle's `tsort` tier: 31 np.nanquantile cases over StringDType
      rows): the StringDType quantile route applies `asanyarray` to each row's result itself and handed the 0-d `U` array
      on unmarked, which `10626746` then re-read as an `np.str_`, dropping a trailing NUL (a first result `'ab\0'` gave a
      `<U2` buffer where NumPy's is `<U3`). The route marks its `asanyarray` results as explicit 0-d arrays
      (`MarkExplicitZeroDimArray`), which `ApplyAlongAxis` keeps; the snapshot is line for line the same (its text pools
      hold no value ending in NUL). Its message names `10626746` by the hash it had before the rebases,
      `cfd6c053`.
  - After the D4 669 replay cases differ on Windows and 677 on Linux, none in the moved code's own work: 568 null
    `indices` (the house's ArgumentNullException where NumPy fails on None's attributes), 78 no-loop texts of the test
    function's `v + sum(a)` over bytes (the `+` operator names a C# int literal int32, NumPy's Python int int64), 16 first
    out-of-bounds indices of a non-contiguous index array (the documented visiting order), 7 complex np.cumsum columns
    starting with -0.0 (np.cumsum's own pin), and on Linux 8 host-dependent special operands.
  - Core's doc warnings 3,830 -> 3,829 (the move).
  - Gates: NumSharp.Tests 20,828 / 20,827 passed on net10.0 / net8.0 (11 skipped) and the oracle suite 414 / 414 (2 skipped), Release, `TestCategory!=OpenBugs&TestCategory!=HighMemory` (on net8.0 the R1 ratchet re-run on the committed tree: the full run had read the next part's uncommitted moves). The first full run found the regression `62683028` fixes (31 strings-oracle cases). The 2 new tests (`Manipulation/np.apply_along_axis.Test.cs` `TextScalarResult_IsAsWideAsTheFirstValue`, `Backends/Strings/StringDTypeOrderTests.cs` `NanQuantile_AlongAxis_BufferIsAsWideAsTheFirstRowsStr`, NumPy-probed) fail on the parents of their commits.
  - Left for batch 7: np.indices (`Numeric`); np.diag (`TwoDimBase`); the lines of np.mgrid, np.ogrid and
    np.ravel_multi_index.
- **2026-10-29, P3: batch 6, second part (np.fromfile, np.fromstring, ndarray.tofile, tobytes, getfield, byteswap).**
  Ten commits on `strings`: the tool commit landed first (`592ee2c4`), the others were developed in the scratch worktree
  on it, rebased onto `6f1da8dd` (batch 7's fourth part) and landed by fast-forward. R1: 16 text and 7 numeric lines
  leave (np.fromfile.cs 6 / 0, np.fromstring.cs 2 / 0, np.tofile.cs 5 / 6, NDArray.byteswap.cs 1 / 1,
  NDArray.getfield.cs 1 / 0, NDArray.tobytes.cs 1 / 0; the six files have no row now); R2: unchanged (the new homes are
  `IO/`, `DTypes/` and `Backends/Unmanaged/`, and no np.* call moved into `Backends/`).
  - `592ee2c4`: the snapshot tool reaches np.fromstring, np.fromfile and ndarray.tofile, whose first parameter is no
    NDArray (each overload was one SKIP line). np.fromstring's TEXT and np.fromfile's FILE are the operand - the pool's
    value sequence as text (a str or UTF-8 bytes; joined by a comma or a space, with unmatched data after it, with a
    NUL), the operand's C-order bytes or that text in a stream (also one that cannot seek) or a temporary file, a
    missing file, null - and tofile writes into a sink whose bytes join the outcome (" WROTE hash/length"), as a read
    stream's final position does (" POS n"). The separator, the count, the dtype, np.fromfile's offset and tofile's
    format get roles. np.fromstring 11,588 lines, np.fromfile 41,682, NDArray.tofile 6,372: batch 6's snapshot is
    199,758 lines, the other names' lines byte for byte as before.
  - The replay of the new lines and of tobytes / getfield / byteswap through NumPy 2.4.2 (`replay6b.py`): 62,360 lines,
    14,319 of them C#-only or NumSharp-only (a stream that cannot seek 11,229, a C# null 1,364, char 854, decimal 854,
    and 18 byteswap lines of a StringDType array, where NumPy 2.4.2 crashes the process). 46,565 were identical or in a
    documented class and 1,476 differed, three faults fixed per D4 before the moves:
    - `79fe2630`: ndarray.tofile's `sep` and `format` are CPython's `s` arguments - a null one is None, refused with
      `TypeError("tofile() argument 2 must be str, not None")` (3 for the format, `sep` first) before the file is opened
      or the stream used; NumSharp wrote binary for a null `sep` and `str()` for a null `format`. 992 lines.
    - `c4c072d1`: ndarray.getfield and setfield read their dtype with `PyArray_DescrConverter`, so a null dtype is None,
      float64 (a float64 view, or `new type is larger than original type`); NumSharp raised ArgumentNullException. 468
      lines.
    - `ac8ed7a4`: ndarray.tobytes converts its order before it looks at the array (array_tobytes'
      `PyArray_OrderConverter`): an empty array with order 'Z' raises NumPy's ValueError where it returned no bytes. 16
      lines.
    After them 34 lines differed, all getfield of a `U5` array as a wider field at a byte stride NumSharp cannot lay out
    (it stores strides in whole elements: its documented NotSupportedException), now a class of the replay. The final
    replay: 48,041 lines identical to NumPy or in a documented class (a non-native dtype decoded into a native copy
    2,172; a non-native dtype read from TEXT 420, where NumPy's scanners store native bytes under the swapped
    descriptor; the integer text scan's final position 240, where Windows NumPy's MSVC `fscanf` consumes the character
    that ends a failed scan and Linux NumPy, whose glibc `fscanf` NumSharp ports, does not - probed under WSL; the
    unsized `S` with a count 216, NumPy's zero-width array; the getfield byte stride 34), np.fromstring's uninitialized
    counted tail compared through poisoned allocations as NumSharp's 0, and 0 differing.
  - `f41dab51`, the pure move: the bodies of np.fromfile, np.fromstring and ndarray.tofile - NumPy's `PyArray_FromFile`,
    `PyArray_FromString` (`ctors.c`) and `PyArray_ToFile` (`convert.c`) - move into the new `IO/ArrayFileIO.cs`
    (Appendix D: IO's bodies live in `IO/`), with CPython's `s` converter they share (`CStringArgument`, `Utf8Strict`).
    The facades keep `array_fromfile` / `array_fromstring` / `array_tofile`: the argument conversion, the removed binary
    mode of fromstring, the offset-with-a-separator TypeError, the file handling and its FileNotFoundException text.
  - `2b775799`, the conversion: NumPy's text-reader slots join `DTypeMeta` as the questions NumPy asks of them
    (`HasScanFunc`: `scanfunc != NULL`, `PyArray_FromFile`'s test; `HasFromStr`: `fromstr != NULL`,
    `PyArray_FromString`'s; true for the numeric classes, decimal and char included), REFCHK asks `HasReferences`, a
    non-native binary read is decoded in place by the class's `copyswapn` (it called byteswap and dropped the result:
    one NDW012 fewer), tofile's binary refusal is NumPy's `NPY_LIST_PICKLE` flag test (StringDType carries the new
    constant), tobytes routes by `HasReferences`, and getfield's descriptor route serves a parametric class (it tested for
    a fixed-width string). Not converted, on purpose: tofile's text mode renders each element with typed code keyed on
    the storage lane, the Python object NumPy's getitem returns (the plan's "typed dispatch moves with its body").
  - `1c5a20be`, the second pure move: the bodies of ndarray.tobytes and ndarray.byteswap - `PyArray_ToString`
    (`convert.c`) and `PyArray_Byteswap` (`methods.c`) - become `UnmanagedStorage.ToBytes` (in
    `UnmanagedStorage.Convert.cs`, beside `ToManagedArray`) and `UnmanagedStorage.Byteswap` (the new
    `UnmanagedStorage.Byteswap.cs`), the array object's own members.
  - `0edc1edc`, the second conversion: ndarray.byteswap swaps through the class's `copyswapn`, as `PyArray_Byteswap`
    calls `descr->f->copyswapn` - one call over a one-segment array, one per inner run of a single-operand EXTERNAL_LOOP
    iterator otherwise - so the unit switch (the item, a complex number's halves, a `U` element's code units) and the
    StringDType test leave the body. `copyswapn` gained a fused copy-and-swap pass for contiguous runs that do not
    overlap (`NDByteOrder.CopyReverseRun`, the kernel byteswap's copy called directly; an overlapping source keeps the
    memmove, then the swap), a strided kernel (`NDByteOrder.ReverseStrided`: the unit width switched on once, one
    `BinaryPrimitives.ReverseEndianness` per unit, where the slot reversed bytes one at a time - routed through the old
    slot, byteswap's strided in-place swap was ~10% slower), and StringDType's in-place identity (D14, which byteswap
    decided itself before). Measured, 1M items, pinned to two P-cores: the in-place swap of a stride-2 view float64
    3.24 -> 0.87 ms, int32 2.13 -> 0.76, float16 1.53 -> 0.76, complex128 5.9 -> 2.2; np.frombuffer's non-native decode
    `>f8` 0.90 -> 0.41, `>c16` 1.95 -> 0.74; the other forms unchanged.
  - `0d10b507`: timed against NumPy 2.4.2 before landing, that strided swap was still slower for the 2-, 4- and 8-byte
    dtypes (NPY/NS 0.46-0.90): `ReverseStrided` ran an inner loop over the units of every element. An element of one
    unit and a complex number's two halves now take a loop of one step per element, as NumPy's `_strided_byte_swap` is
    one loop per item size: float64 0.685 ms (NPY/NS 1.15), int32 0.382 (0.97), float16 0.347 (1.00) - the memory floor
    both share -, complex128 2.170 (1.92). Snapshot byte-identical.
  - `b63f1193`: `NumPyTextScan`'s remark said Windows NumPy matched glibc on every probed input; it now names the
    difference the replay found (the MSVC `fscanf` position above).
  - Gates: the snapshot was byte-identical through both moves and both conversions (the second one also under
    `NUMSHARP_DEBUG_GUARD_PAGES=1`); Core's warnings 15,732 -> 15,728 (the dropped byteswap result's NDW012). The tests
    of the D4 fixes are NumPy-probed and fail on their parents (`Casting/NDArray.tofile.Test.cs`
    `NullSepOrFormat_IsNone_RefusedBeforeAnythingIsWritten`, `Casting/NDArray.getfield.Test.cs`
    `GetField_NullType_IsNoneFloat64`, `Casting/NDArray.tobytes.Order.Test.cs` `InvalidOrder_EmptyArray_StillThrows`, and
    the object array's null checks in `ListArrayComputeTests`). `DTypes/DTypeSlotTests.cs` (17) asks the new slots of
    all 27 classes, pins the `NPY_LIST_PICKLE` flag, and holds `copyswapn` to an independent byte reversal over its
    three swap paths and an overlapping source in both directions (it compared with ndarray.byteswap, which now runs
    through it); planting `Disjoint => true` fails it. The four suites on the rebased stack up to `b63f1193`:
    NumSharp.Tests 20,829 / 20,828 passed on net10.0 / net8.0 (11 skipped) and the oracle suite 414 / 414 (2 skipped),
    Release, `TestCategory!=OpenBugs&TestCategory!=HighMemory`; after `0d10b507` the byteswap, slot, buffer, file and
    npy tests (263) passed again.
  - Next: batch 6's third part, np.loadtxt's converters into `IO/TextReading/` with np.savetxt's two lines (R1: 26 text
    and 16 numeric lines in np.loadtxt.Sinks.cs, np.loadtxt.Parsers.cs, np.loadtxt.cs and np.savetxt.cs); then batch 8
    (the indexers), 9b and 10; batch 7's rest is the parallel session's. The 0.71.0 notes' snapshot-tool line (Testing &
    Tooling) stops at `2c25668e` (np.interp): the eighteen tool commits since (np.fft, np.evaluate,
    NDArray.Contains, the `where=` masks, P5's and P3's families) are added to it in one pass when P3 is done.

- **2026-10-29, P3: batch 7, fourth part (np.r_ / np.c_, np.ix_, np.fill_diagonal).** Eight commits on `strings` from the
  parallel session (developed in the scratch worktree on `88426e34`, rebased onto `592ee2c4`, batch 6b's tool commit, and
  landed by fast-forward). R1: 4 text and 10 numeric lines leave (np.r_.cs 1 / 8, np.fill_diagonal.cs 3 / 0, np.ix_.cs
  0 / 2; the three files have no row now); R2: unchanged (the home is `Indexing/`, not under `Backends/`).
  - `5f4d6015`: the snapshot tool reaches np.ix_ (its `params object[]` had no role: every overload was SKIP) and
    np.r_ / np.c_, which are no functions: the targets "np.r_" / "np.c_" call the indexer of `np.AxisConcatenator` on
    the object. np.ix_'s sequences and the key are built from the pool operand (IxArgsLabels: the operand, its flattened
    values, masks, typed C# arrays, lists, empty ones of both kinds, refused ranks, None, none; ConcatKeyLabels: C#
    scalars of every kind beside the operand, arrays, text, slices, every directive form), on 1-D pools first. With
    np.fill_diagonal and np.diag: 10,944 cases.
  - `ac83cba8`, the pure move: the new `Indexing/NDArray.IndexTricks.cs` (NumPy's `lib/_index_tricks_impl.py`) holds
    np.ix_'s body (`NDArray.Ix`), np.fill_diagonal's (`NDArray.FillDiagonal`, with the diagonal scatter np.diag shares,
    `NDArray.TryDiagonalScatter`) and the statements of np.r_ / np.c_'s indexer (`NDArray.AxisConcatenate`, with the
    directives, the entry conversion and the dtype resolution; the nested types renamed `ConcatOperand` /
    `ConcatWeakKind`). `np.AxisConcatenator` keeps its fields, its indexer's numpy.matrix rule and the C# spelling of a
    slice (`SliceSpec`, `ParseSliceToken`), which np.ogrid / np.mgrid share. The snapshot is byte-identical under
    `NUMSHARP_DEBUG_GUARD_PAGES=1`; the rebind check found no moved name binding to another `NDArray` member.
  - `b07d19d2`, the slot conversion: np.fill_diagonal's StringDType route asks `DTypeMeta.HasReferences` (NumPy's
    `PyDataType_REFCHK`) instead of testing the storage for a StringDType slice. Snapshot byte-identical.
  - `f66c85cd`: the tool's np.fill_diagonal values gain Python lists (`object[]`: ints with one past int8, a str, a
    float beside a NaN, None, an empty list), which NumPy stores item by item with the dtype's setitem: 11,514 cases.
  - D4, from the replay of the snapshot through NumPy 2.4.2 on Windows and on Linux (WSL; 10,280 cases replayed, char
    and decimal skipped; 1,174 / 1,165 differed):
    - `70646c73`: np.ix_ of ONE sequence returns a new view, as NumPy's reshape to the same shape does (`PyArray_View`):
      it returned the operand itself, or the owning array a conversion made. An EMPTY sequence is cast to intp only when
      it is not an ndarray: a typed C# array or `Memory<T>` keeps its dtype (np.ix_(np.array([], np.int32)) is int32).
      590 cases.
    - `cf75f39f`: np.r_ / np.c_'s "r" / "c" directive is NumPy's `matrix(res)`, a C-ordered copy (an all-F concatenation
      gave an F-contiguous matrix), and a Python int entry is stored by the result dtype's setitem (`AsStoreValue`,
      `PyArray_Pack`) with the LP64 texts: past the C long "Python int too large to convert to C long", where NumSharp
      said "Python integer ... out of bounds for int8". 162 cases on Linux, 134 on Windows (Windows' 32-bit C long has
      other texts).
    - `d6b4926e`: np.fill_diagonal converts its value as `a.flat[:end:step] = val` does, `PyArray_FromAny(val, a.dtype,
      FORCECAST)` before the slice (the new `NDArray.FlatStoreValue`): an empty diagonal still raises, a str is parsed by
      the setitem and quoted as the str ('x', not np.str_('x')), a Python list is stored item by item ([300, 2] into
      int8 raises, it wrapped to 44), a Python int past int64 into an S / U array stores its str(). 170 cases.
  - After the D4s 243 replay cases differ on Linux and 280 on Windows, none in the moved code's own work: 224 are
    np.r_["0"] / np.c_["0"] (one bare string: NumPy's bmat branch raises NameError, NumSharp reads it as a directive, the
    documented divergence), 8 complex64 results (one complex width), on Linux 11 special-value operands whose float ->
    uint cast differs by host, on Windows 48 overflow texts of its 32-bit C long.
  - `e806779f`: np.fill_diagonal's facade documents the conversion and its exceptions; the facades' two unresolved crefs
    are fixed. Core's doc warnings 3,832 -> 3,830.
  - Gates: NumSharp.Tests 20,826 / 20,825 passed on net10.0 / net8.0 (11 skipped) and the oracle suite 414 / 414 (2 skipped), Release, `TestCategory!=OpenBugs&TestCategory!=HighMemory`. The 8 new tests (`Indexing/IndexTricks.NumPyPort.Test.cs`, NumPy-probed on Windows and Linux) all fail on the parents of their commits.
  - Left for batch 7: np.indices (`Numeric`); np.diag (`TwoDimBase`); np.take_along_axis, np.put_along_axis and
    np.apply_along_axis (`ShapeBase`, with the two np.apply_along_axis pins of `NDVStringStatistics.Quantile.cs`); the
    lines of np.mgrid, np.ogrid and np.ravel_multi_index.

- **2026-10-29, P3: batch 6, first part (np.frombuffer).** Five commits on `strings`: the tool commit landed first, the
  others were developed in the scratch worktree on `61063970`, rebased onto `ef80958a` (batch 7's third part, np.pad)
  and landed by fast-forward:
  - `0854f1ca`: the snapshot tool reaches np.frombuffer. Its BUFFER is the operand, in every C# form of the overload's
    first parameter: a byte[], the operand's own MemoryView, an ArraySegment / a Memory of bytes inside a larger array, a
    Memory over unmanaged memory, an address with its byte length, the generic overload's typed array. The count, the
    byte offset, the address overloads' byte length and the dtype (a descriptor, an NPTypeCode, a name) get roles, and
    np.frombuffer<TSource> is instantiated for every element type. That gives 137,388 lines; before, the 16 non-generic
    overloads were one SKIP line each and the two generic ones made no line. The other batch 6 names' lines are kept
    byte for byte, and the run is deterministic under `NUMSHARP_DEBUG_GUARD_PAGES=1`.
  - The replay of those lines through NumPy 2.4.2 (`replay6a.py`): of the 120,476 lines NumPy can express (16,910
    C#-only or decimal / char), 30,558 were identical and 89,918 differed. The numeric overloads were seven copies of the
    checks with rules of their own, so the D4 fixes came before the move (the change to the method, see batch 6's item):
    - `32bfa4a7`: every overload runs `PyArray_FromBuffer` - the string route's checks and np.ndarray's view, for every
      dtype. It removes ArgumentException where NumPy raises ValueError (32,349 lines), views reporting OWNDATA (30,487),
      a segment or array-backed memory read to the end of the array behind it (12,869), the buffer-size rule for every
      count (7,184), a count past the buffer capped (2,477), InvalidOperationException for BufferError (1,200) and a
      non-contiguous memoryview reaching the bytes (120), a non-native descriptor read as native by the address, Memory and
      typed-array overloads (1,096), and a memoryview reinterpreted through views of the source's elements (189). 89,918
      lines -> 1,308.
    - `271f24d0`: the dtype names are read by np.dtype (array_frombuffer's `PyArray_DescrConverter`): 'bogus' raises
      np.dtype's text with NumPy's words, the empty name is refused (it read float64), '!' is no byte-order prefix.
      1,308 lines -> 0.
    After them every line matches NumPy 2.4.2 or a documented class: 6,675 non-native results (decoded into a native
    copy, compared with NumPy's `astype(native)`), 2,314 of the generic overload's default dtype (its element type),
    1,308 refused names under np.dtype's house type (NotSupportedException with NumPy's TypeError text), 654 complex64
    names (one complex width).
  - `87263eb9`, the pure move: the body - `FromBufferObject`, `FromBufferPointer`, `FromBufferSpan`, `FromBufferCount`,
    `FromBufferDescr`, `ThrowIfReferenceDType` - moves into `UnmanagedStorage.Buffer.cs`. The facade keeps its overloads
    and the name conversion through np.dtype. The call np.ndarray(...) becomes its body, `ArrayNew`, so no np.* call
    reaches Backends/.
  - `ebff0b12`, the conversion: REFCHK asks `HasReferences`. The address and span routes decode a non-native descriptor
    with the class's `copyswapn`, its first production caller. The slot gets NumPy's `_basic_copyn` block copy and
    `byteswap`'s SIMD swap for contiguous runs, moved into the new `Utilities/NDByteOrder.cs`: through the per-item slot
    the decode of 1M float64 items took 4.3 ms against 0.9 for the old copy + byteswap, and now takes the same.
  - Gates: the snapshot was byte-identical through the move and the conversion. R1: 16 text and 93 numeric lines
    leave (np.frombuffer.cs, np.frombuffer.Strings.cs, gone); R2 is unchanged; doc warnings lose one NDW012. The 11 new
    tests are NumPy-probed: `Creation/np.frombuffer.PyArrayFromBuffer.Test.cs` (7, all fail on their parent) and
    `Creation/np.frombuffer.DtypeNames.Test.cs` (4, 3 fail on theirs). The four suites passed, and so did the 114 interop
    tests over np.frombuffer, live against NumPy 2.4.2. After the rebase: both test projects built, R1 / R2 and
    NumSharp.Tests (net10.0, net8.0) passed again.
  - Next: batch 6's rest (np.fromfile, np.fromstring, ndarray.tofile, tobytes, getfield, byteswap, np.loadtxt's
    converters), batch 8 (the indexers), 9b and 10; batch 7's rest is the parallel session's.

- **2026-10-29, P3: batch 7, third part (np.pad).** Five commits on `strings` from the parallel session (developed in the
  scratch worktree on `61063970` and landed by fast-forward). R1: `np.pad.cs`' 9 text and 33 numeric lines leave (the
  file has no row now); R2: unchanged (np.pad's home is not under `Backends/`).
  - `f4816d56`: the snapshot tool reaches np.pad (every overload was SKIP): pad_width in each C# form (an int, an
    `int[]`, an `int[,]` built from the operand's rank, a (before, after) tuple, a dict), the mode (every name NumPy has,
    one it refuses, two callables from a menu NumPy names alike, null), constant_values / end_values / stat_length /
    reflect_type and a callable's kwargs: 93,720 cases.
  - `3023494e`, the pure move: np.pad's statements are `NDArray.Pad` (a named mode, still `[NDScoped]`) and
    `NDArray.PadCallable` (a callable, unscoped so no scope reaches into the caller's function) in the new
    `Manipulation/NDArray.ArrayPad.cs` (NumPy's `lib/_arraypad_impl.py`), with every helper whole; the ten facades keep
    their numpy.matrix regions. The snapshot is byte-identical under `NUMSHARP_DEBUG_GUARD_PAGES=1`; the rebind check
    found no moved name binding to another `NDArray` member (`Array.Empty` needed `System.`: inside `NDArray`, `Array` is
    the instance property).
  - No slot conversion: np.pad's dtype lines choose a route inside the moved body (the text constants' route, the text
    refusals of linear_ramp and the stat modes, a statistic's rounding into an integer array); its text casts are batch
    9b's.
  - D4, from the replay of the snapshot through NumPy 2.4.2 (91,160 cases replayed, char and decimal skipped; 55,325
    differed):
    - `33fd5151`: the check order (the widths through np.asarray's integral test and `_as_pairs` over the TRUE rank, then
      the mode and its keywords) and NumPy's texts (`mode 'None' is not supported` and the keyword set
      `{'constant_values', 'stat_length'}` as ValueErrors, `np.broadcast_to`'s and `x.min()`'s texts); reflect_type (any
      value but 'odd' reflects as 'even'; given when neither null nor "even"); `_as_pairs` for the widths and stat_length
      (a (2, 1) table is one width per axis, the negative test before the broadcast, a real stat_length rounded half to
      even); the result object (a zero-width pad keeps `_pad_simple`'s F order, the callable runs for every vector, a 0-d
      input keeps its value).
    - `839950c2`: constant_values / end_values through NumPy's `_as_pairs`: one or two values are NumPy scalars, a larger
      table `tolist()`'s Python scalars (range-checked into every integer dtype; an `int[,]` raised InvalidCastException),
      an array of any shape is the (ndim, 2) table.
    - `54ad9427`: the statistics are np.mean / np.median of the chunk view (a uint64 'mean' wrapped its sum), and
      linear_ramp is linspace's arithmetic in its computation dtype with NumPy's whole-call zero-step guard (a bool input
      pads; it raised the bool subtract's TypeError).
  - After the D4s 2,489 replay cases differ from NumPy, none in np.pad's own computation: 2,016 keyword sets of two names
    in another order (NumPy's set order follows Python's randomized string hashes), 411 `mode='empty'` results
    (uninitialized memory on both sides), and 62 'median' pads of a complex128 chunk holding a NaN, which are np.median's:
    NumPy's `_median_nancheck` returns the NaN element (`np.median([1+2j, nan+1j, 3, 4+1j])` is `nan+1j`, NumSharp's
    `2+1j`). Reported, not this batch's.
  - Core's doc warnings 3,840 -> 3,834 (the moved file's six, fixed with the D4s).
  - Gates (on the landed tree): NumSharp.Tests 20,807 / 20,806 passed on net10.0 / net8.0 (11 skipped) and the oracle suite 414 / 414 (2 skipped), Release, `TestCategory!=OpenBugs&TestCategory!=HighMemory`.
  - Left for batch 7: np.r_, np.ix_ and np.fill_diagonal (`IndexTricks`); np.indices (`Numeric`); np.diag (`TwoDimBase`,
    with the diagonal scatter it shares with np.fill_diagonal); np.take_along_axis, np.put_along_axis and
    np.apply_along_axis (`ShapeBase`, with the two np.apply_along_axis pins of `NDVStringStatistics.Quantile.cs`); the
    lines of np.mgrid, np.ogrid and np.ravel_multi_index.
- **2026-10-29, P3: batch 7, second part (np.where's D4, np.select, np.piecewise, np.delete's rest).** Ten commits on
  `strings` from the parallel session (developed in the scratch worktree on `d8c7d00a`, rebased over batch 5's landing
  and `0854f1ca`, and landed by fast-forward). R1: 3 text and 37 numeric lines leave (np.select.cs 3 / 10,
  np.piecewise.cs 0 / 7, np.delete.cs 0 / 20); R2: 1 computation leaves (`NDStringArrays.cs`). A correction to the first
  part's entry below: its "no divergence from NumPy was found" was written before the replay of its snapshot through
  NumPy 2.4.2 ran; the replay found np.where's three, fixed here (`4fa16cbf`). The messages cite their commits by the
  hashes before the rebases: `e2815667` = `569b1850`, `2278724b` = `c0ac4ef5`, `33730826` = `582ddced`, `dec91a2c` =
  `a318249e`, `e929947b` = `a186234e`.
  - `4fa16cbf` (D4): np.where(condition, x, y) is NumPy's `PyArray_Where` in its result dtype and its order. The result
    dtype is `PyArray_ResultType` under NEP 50 (a 0-d array and a NumPy scalar are strong, two weak C# literals give the
    default dtype: np.where(c, 7, 7) is int64, np.where(c, np.array(2), int8 array) int64, np.where(c, np.float16(1.5),
    7) float16; NumSharp's legacy rule read every 0-d array as weak), the dtype is resolved before the broadcast
    (`StringDType` beside a number raises `DTypePromotionError`, not the shape error), and a broadcast failure is
    NumPy's iterator text (`operands could not be broadcast together with shapes (3,4) (7,) (3,4) `). Five tests that
    pinned the C# type of two literals take NumPy's answers; `Logic/np.where.NumPyOrder.Test.cs` (6) is new.
  - `3f06e8a4`: the fixed-width text route of np.where (`NDStringArrays.Where`, batch 4's file, released by the
    coordinator for this line) calls `NDArray.Where` instead of np.where - the last np.where call P6 pinned in the engine.
    R2: 1 computation leaves.
  - `569b1850`: the snapshot tool reaches np.select, np.piecewise (every overload was SKIP), np.delete's boolean mask,
    np.take_along_axis / np.put_along_axis's indices and values and np.fill_diagonal's val (through the item-selection
    wiring, gated by function: no other target's case changed).
  - `c0ac4ef5`, np.delete's rest: the seven helpers np.insert's and np.delete's statements share (`WithSourceOrder`,
    `PrepareAxisContext`, `ToInt64Scalar`, `ToInt64Vector`, `DuplicateArray`, `SliceAlongAxis`, `PythonSliceIndices`)
    join them in `NDArray.FunctionBase.cs`; np.diff / np.ediff1d / np.gradient / np.unwrap's engine partials call
    `NDArray.SliceAlongAxis`. R1: np.delete.cs 0 / 20 leave.
  - `582ddced`, `a318249e`, the pure moves: np.select's statements are `NDArray.Select` with its six helpers, and
    np.piecewise's are `NDArray.Piecewise` (one member per C# form of condlist) with `PiecewiseCore` and its ten
    helpers, appended to `NDArray.FunctionBase.cs`; the facades keep their numpy.matrix rules and null checks. R1:
    np.select.cs 3 / 10 and np.piecewise.cs 0 / 7 leave.
  - No slot conversion: the lines that left choose a route by dtype inside the moved bodies (np.select's weak-literal
    ranks and its text descriptor, np.piecewise's constant coercion), which the home keeps or the D4s below replaced.
  - D4, from the replay of the select / piecewise / delete snapshot (38,358 cases; 2,775 differed):
    - `a186234e`: `Shape.BroadcastShape` is NumPy's `_broadcast_shape` (np.broadcast's axis-by-axis scan, which names
      the pair that disagrees on the leftmost conflicting axis; NumPy's groups of 64 operands; "broadcast dimensions too
      large." past the 64-bit element count), and np.broadcast_shapes uses it: it scanned operand by operand, so with
      two conflicting axes it named another pair (np.broadcast_shapes((1,4), (1,5), (2,1), (3,1)) names args 2 and 3).
      np.broadcast and np.broadcast_arrays still raise the text without the pair (reported).
    - `8f86bd03`: np.select's three np.broadcast_arrays calls raise that text (718 cases), and a C# number literal
      reaches np.copyto as the array np.broadcast_arrays makes of the Python scalar (int64 / float64 / complex128, of the
      choice group's shape): an int beside a uint8 array is "Cannot cast array data from dtype('int64') ..." (it was
      "Cannot cast scalar from dtype('int32') ...", 45 cases), and a C# byte there is refused as NumPy's Python int is
      (it was copied in). `NDArray.Select` is `[NDScoped]`.
    - `d3d4a01a`: np.piecewise stores a constant as NumPy's `y[cond] = func` through the mask setter's own Python-scalar
      packing: the index is checked first (8 cases), a complex into an integer array is int()'s text (192), into bool /
      text its truth / str() (168), None is stored (536 cases of `[None, 0]`: NaN into float, int()'s TypeError into
      int; it was an ArgumentNullException), np.float16(300) into int8 is cast to 44, and the fused path declines a C#
      array constant (it stored the array's first element). `CoerceScalarConstant` and `CheckFloatFitsInteger` are gone.
    - `3c68f7cd`: np.delete's out-of-bounds error of several indices names the keep mask's axis 0 (NumPy's
      `keep[obj,] = False`; 184 cases), and an integer list of one element takes the single-value path, which names the
      deletion axis.
  - After the D4s 924 replay cases differ from NumPy, every one in a documented class: np.delete's result object of a
    mask deletion (NumPy's MapIter view, OWNDATA false and F-ordered or permuted behind a slice, against NumSharp's
    owning C-ordered copy: the fancy-index layout gap, 856), NumSharp's single complex width (np.select with a Python
    complex beside float16 / float32: NumPy's complex64, 50), the string ufunc's no-loop text naming a weak C# int int32
    where NumPy names int64 (np.add inside a piecewise callable, 12; reported), and the complex multiply's NaN choice in
    the callable `v * v` (6). None of them matched before; 1,851 cases are fixed.
  - Each move: the snapshot byte-identical under `NUMSHARP_DEBUG_GUARD_PAGES=1` (98,808 cases for np.delete and
    np.insert, 9,564 for np.select, 9,570 for np.piecewise); the rebind checks found no moved name binding to another
    `NDArray` member; Core's doc warnings 3,847 -> 3,842 on `d8c7d00a` (3,840 after the rebase; the D4s' docs, and the
    moves' two unresolved kernel crefs resolve again).
  - Gates (on the rebased tree): NumSharp.Tests 20,796 / 20,795 passed on net10.0 and net8.0, NumSharp.Tests.Oracle 414
    on net10.0, and on net8.0 413 with the R1 ratchet failing only because the next family's move sat uncommitted in the
    working tree during the run (the ratchets rerun on the committed sources pass, 7 / 7); run on the tree rebased over
    `95f69a70` (`0854f1ca`, landed since, changes only the snapshot tool, and the snapshot of these families is
    byte-identical through the merged tool).
  - Left for batch 7: np.pad (`ArrayPad`); np.r_, np.ix_ and np.fill_diagonal (`IndexTricks`); np.indices (`Numeric`);
    np.diag (`TwoDimBase`, with the diagonal scatter it shares with np.fill_diagonal); np.take_along_axis,
    np.put_along_axis and np.apply_along_axis (`ShapeBase`, with the two np.apply_along_axis pins of
    `NDVStringStatistics.Quantile.cs`); the lines of np.mgrid, np.ogrid and np.ravel_multi_index. The snapshot tool
    reaches np.diag, np.fill_diagonal and the along-axis pair; the others need their roles first.

- **2026-10-29, P3: batch 5 (constructors).** Six commits on `strings`, developed in the scratch worktrees and landed by
  fast-forward (the move rebased over batch 7's first part, which touched none of its files but the R1 allowlist):
  - `87bd99c1`: the snapshot tool reaches the constructors - np.empty / np.zeros / np.ones / np.full / np.eye /
    np.identity / np.ndarray take a SHAPE as their operand (every overload without an array parameter was SKIP), built
    per call in the C# form of the parameter; np.full / np.full_like's fill value, the `_like` forms' `shape=`, np.eye's M
    and k, the creation order and np.ndarray's buffer / offset / strides get roles, and the family's generic overloads
    are instantiated for every element type. 42,635 lines for the eleven names (the 7,055 lines before kept byte for
    byte), deterministic under `NUMSHARP_DEBUG_GUARD_PAGES=1`.
  - `8bbd4e4c`, the pure move: np.full's, np.full_like's and np.ones' bodies are `NDArray.Full` / `FullLike` / `Ones` in
    the new `Creation/NDArray.Numeric.cs` (numeric.py's `a = empty(...); copyto(a, fill_value, casting='unsafe')`), with
    their helpers and `WithLaneSharingDescr` (`NDArray.flatten.cs` names it unqualified now); np.eye's is `NDArray.Eye` in
    the new `Creation/NDArray.TwoDimBase.cs`; np.ndarray's is `UnmanagedStorage.ArrayNew` (arrayobject.c's `array_new`)
    in `UnmanagedStorage.Ctors.cs` with its helpers, beside the storage-lane fills `FullCore` / `FullFillElement` /
    `OnesCore` and np.arange's `PyArray_ArangeObj`. np.empty, np.zeros and the other `_like` forms were one call into the
    allocating constructor or into np.ones already: nothing of theirs moved. R1: 42 lines leave (29 text, 13 numeric:
    np.eye.cs, np.full.cs, np.full_like.cs, np.ndarray.cs, np.ones.cs).
  - `e23cef2b`, the conversion: the one of np.ones and np.eye is the Python int 1 converted once by the class's
    `setitem` (`getitem` boxes it for np.ones' lane fill; np.eye copies the converted element into the other diagonal
    slots, or stores each through `setitem` for a class that `HasReferences`), replacing two switches over the storage
    lanes; the `StringDType` routes of np.full / np.full_like / np.ones and np.ndarray's ask `HasReferences`;
    np.ndarray's unsized-string refusal asks the item size (`PyDataType_ISUNSIZED`). Snapshot identical. It carries one
    change, NumPy's: np.eye of a `StringDType(coerce=False)` raises where it stored '1' (the class's setitem refuses the
    Python int; np.ones of it still casts an int64 1).
  - The replay of the snapshot through NumPy 2.4.2 (`replay5.py`: every line's call rebuilt from its key in the boundary
    map): of the 38,682 lines NumPy can express (3,952 decimal and char lines skipped), 37,830 were identical and 852
    differed. Fixed per D4, each with NumPy-probed tests:
    - `f048ad69`: np.eye raises in np.zeros' order with its texts - the order's letter, the device, the C-or-F rule, then
      the dimensions ("negative dimensions are not allowed", without NumSharp's "(N=-1)" suffix). 45 lines, and
      np.identity's 1.
    - `4d1a0d02`: np.empty_like / np.zeros_like / np.ones_like / np.full_like check the order's letter before the device, as
      NumPy's empty_like parses its arguments in order. 96 lines.
    - `253b8f45`: np.full / np.full_like's fill value - without a dtype a 0-d array keeps its dtype and a `BigInteger` (a
      Python int) is discovered by value (both raised ArgumentNullException; 18 lines), an empty result converts no text
      (144), a `StringDType` result stores a Python int as its text (76), a fill array that does not broadcast into a
      fixed-width text result raises copyto's text (192). 430 lines, and 18 that raise the object-array refusal now
      instead of ArgumentNullException.
    After them 280 lines differ, each a class NumSharp documents: 126 OverflowError texts (NumPy's Windows build;
    NumSharp pins the LP64 text), 62 C# ints that keep int32 (np.asarray(7) is int32 in NumSharp), 47 np.ndarray
    refusals (26 strides that split an element, 13 negative offsets, 8 `StringDType` arrays over raw bytes), 36 object
    arrays (None or a Python int past 64 bits without a dtype), and 9 MemoryError texts of the allocator past 2**62
    bytes, whose text is .NET's where NumPy's names the size (left for the allocator, not the constructors).
  - Gates: the snapshot byte-identical through the move and the conversion; no moved name binding to another member;
    R1 tightened by the move, then unchanged; R2 unchanged; doc warnings unchanged (the five NDW012 moved with their
    lines); the 12 new tests (9 fail on the conversion); NumSharp.Tests and NumSharp.Tests.Oracle on net10.0 and net8.0
    after the move, the conversion and the fixes.
  - Next: batch 6 (buffers and IO), batch 8 (the indexers), 9b and 10; batch 7's rest is the parallel session's.

- **2026-10-29, P3: batch 7, first part (np.where, np.real / np.imag, ndarray.conjugate, np.extract,
  np.asarray_chkfinite).** Seven commits on `strings` from the parallel session (developed in the scratch worktree on
  `a5ba156b`, rebased over batch 4's landing and `87bd99c1`, landed by fast-forward). R1: 3 text and 14 numeric lines
  leave; R2: 4 computations leave.
  - `54905a89`: the snapshot tool reaches the first families: np.where's x / y given as C# values (its three object
    overloads were SKIP: a weak Python int, the other Python scalars, NumPy scalars, a strong 0-d array, a decimal, a C#
    array, null), np.extract's arr and ndarray.real / imag's setters' value (SKIP: required arrays the tool did not
    name). The batch's first snapshot (np.where, np.extract, np.real, np.imag, the four real / imag accessors,
    np.asarray_chkfinite, np.diag): 21,540 lines, deterministic with and without guard pages. The rebase kept both
    sides of `87bd99c1`'s constructor roles.
  - `953af9f3`, the pure move: np.where(condition, x, y)'s body is `NDArray.Where` (NumPy's `PyArray_Where`) with its ten
    kernel helpers, appended at the class end of `NDArray.Multiarray.cs`; the four facades keep their numpy.matrix
    regions and their operand conversions. R1: 2 text and 8 numeric lines leave.
  - `d6f814d8`: the engine's four np.where calls (`Default.Reduction.Nan.cs`, `Default.Reduction.NanScan.cs`,
    `NDVStringMove.cs` twice), which P6 had pinned for this home, call `NDArray.Where`. R2: 4 computations leave. Its
    message names the move by its hash before the rebase, `3b410248` (now `953af9f3`). The np.where of the fixed-width
    text route in `NDStringArrays.cs` (batch 4's file) stays pinned.
  - `e8ee6dab`: np.real / np.imag's statements (NumPy's `val.real` / `val.imag`, `getset.c`) are `NDArray.RealPart` /
    `ImagPart` in the new `Math/NDArray.TypeCheck.cs`, ndarray.imag's setter's are `NDArray.ImagSet` beside them, and
    ndarray.conjugate's are `NDArray.Conjugate` (`PyArray_Conjugate`, appended to `NDArray.Multiarray.cs`). np.real
    applies the numpy.matrix rule first now, as np.imag does: a real matrix passes through `Wrapped` and comes back as
    itself (probed, with the parts of a complex matrix and a real matrix's read-only imaginary zeros). R1: 1 text and 4
    numeric lines leave (`NDArray.ComplexAccessors.cs`, np.real, np.imag).
  - `00aa4cff`: np.extract's and np.asarray_chkfinite's statements are `NDArray.Extract` and `NDArray.AsarrayChkfinite`
    in `NDArray.FunctionBase.cs`, with `TryExtractFast` and `IsFloatFamily`. R1: 2 numeric lines leave.
  - `67ddb6c4`: OperandRootsTests' LT07 row names `NDArray.Where` (the suite's one failure after the move; the message
    cites the move as `c486593d`, its hash after the first rebase).
  - `ecf4ad9c`, the slot conversion: np.extract's fast path asks `DTypeMeta.HasReferences` instead of testing the
    storage for a `StringDType` slice (the message cites the move as `22d953a0`, now `00aa4cff`). The other lines that
    left choose a route by dtype inside the moved bodies (np.where's text route and its bool and promotion tests, the
    complex tests of the real / imag / conjugate statements, the AllFloat test), which the homes keep: no slot answers
    them.
  - Each move: the snapshot byte-identical under `NUMSHARP_DEBUG_GUARD_PAGES=1` (the 21,540 cases; 29,550 for np.nansum,
    np.nanprod, np.nancumsum, np.nancumprod and np.where around the engine switch; 1,908 for ndarray.conjugate / conj);
    the rebind checks found no moved name binding to another `NDArray` member; Core's doc warnings 3,848 -> 3,847. No
    divergence from NumPy was found in these families, so no D4 commit.
  - Gates (on the rebased tree): NumSharp.Tests 20,759 / 20,758 passed on net10.0 and net8.0; NumSharp.Tests.Oracle 414
    passed on both.
  - Left for batch 7: np.select, np.piecewise and np.delete's rest (`FunctionBase`); np.pad (`ArrayPad`); np.r_, np.ix_
    and np.fill_diagonal (`IndexTricks`); np.indices (`Numeric`); np.diag (`TwoDimBase`, with the diagonal scatter it
    shares with np.fill_diagonal); np.take_along_axis, np.put_along_axis and np.apply_along_axis (`ShapeBase`, with the
    two np.apply_along_axis pins of `NDVStringStatistics.Quantile.cs`); the lines of np.mgrid, np.ogrid and
    np.ravel_multi_index. Of these the snapshot tool reaches only np.diag so far.

- **2026-10-29, P3: batch 4 (coercion).** Seven commits on `strings`, developed in the scratch worktree and landed by
  fast-forward (the four after `8a2d53fe` were rebased over batch 9's landing, which touched no file of theirs but the
  allowlist and the docs):
  - `e07f3016`: the snapshot tool reaches the coercion family - np.array, np.asarray, np.asanyarray, `NDArray.Scalar`
    and the conversion operators take a C# value, which the tool builds from each pool operand in every form the
    parameter takes (typed, jagged and object arrays, nested object lists and grids, rows, a List / an iterator / a
    Memory / a tuple, a scalar, a str, a str array, a MemoryView, null); every overload whose first parameter is not an
    NDArray was SKIP. 56,886 lines, deterministic under `NUMSHARP_DEBUG_GUARD_PAGES=1` (an uninitialized tail of
    `np.array<T>(IEnumerable, size)` differed between runs without it).
  - `8a2d53fe`, the pure move, part 1: the bodies of np.array, np.asarray and np.asanyarray are `NDArray` members in
    `NDArray.Multiarray.cs` (NumPy's `_array_fromobject_generic`, by operand kind: `ArrayFromObject`, `ArrayFromManaged`,
    `AnyArrayFromObject`, `ArrayFromTyped`); `NDArray.Scalar`'s, np.array(str)'s and the C# array conversions are
    `UnmanagedStorage` members in the new `UnmanagedStorage.Coercion.cs` and `UnmanagedStorage.Convert.cs`. The facades
    keep the class boundaries, the device and `like=` handling and the C# null checks.
  - `8889e868`, part 2: `NDStringArrays`' text collection (the shape walk, the descriptor's discovery, the stores into a
    fixed-width, `StringDType` or numeric result, np.asarray's request test) moves into the new
    `UnmanagedStorage.Coercion.Text.cs`, its export to a C# array into `UnmanagedStorage.Convert.cs`.
  - `a45a11f6`, part 3: the port of NumPy's coercion of a Python sequence (`PolySequence`,
    `Polynomial/Package/NDPolySequence.cs`) is `UnmanagedStorage`'s, in `Creation/UnmanagedStorage.Coercion.Sequence.cs`
    (git mv): `FromSequence` / `FromSequenceOrNonNumeric` / `SequenceItems` / `SequenceNDim`; its 26 callers (numpy.char,
    numpy.matrix, numpy.polynomial, np.histogramdd, np.gradient) name the new members.
  - `9eb1fe39`, the conversion's first part, stores through the classes: the text builder's three store loops (a
    `StringDType` request's `VStringArraySlice.SetRange`, a numeric request's `PackStore` into a stack slot, the
    fixed-width `WriteElement`) are one sequence, NumPy's `PyArray_FromAny`: the descriptor (requested, or discovered for
    an unsized request), the allocation (zero-filled only for reference-holding cells, `NPY_NEEDS_INIT`), the values
    through the class. The run is a new NumSharp slot, `DTypeMeta.SetFreshItems` (NumPy's `PyArray_AssignFromCache` loop
    of `PyArray_Pack`: `setitem` per value by default; `StringDType` overrides it with `SetRange`, which reserves its arena
    once - the per-value first version failed `StringDTypeFillTests.Array_ReservesTheArenaOnce`). `NDArray.Scalar`'s
    text lane is that route too.
  - `c8bc8396`, the conversion's second part, discovery through the classes: the text builder's `Discover` is the dtype
    half of `PyArray_DiscoverDTypeAndShape` - `UnmanagedStorage.DiscoverDTypeFromPyObject` (`discover_dtype_from_pyobject`:
    the requested class when it claims the value, else the value's own class), `FindScalarDescriptor`
    (`find_scalar_descriptor`: the class's `discover_descr_from_pyobject`, cast to the requested class) and
    `DTypePromotion.PromoteTypes` per value (`handle_promotion`). `NumberTextWidth`'s hand-kept table of
    `REQUIRED_STR_LEN` is gone: the casting impls give the same widths for all 15 storage classes (probed).
    `_PyLongDType.discover_descr_from_pyobject` is NumPy's `discover_descriptor_from_pylong` (int64, uint64, object by
    value); the string classes size a C# char as one character.
  - `3518c7cd`, D4: np.array and np.asanyarray of a C# object list (an `object[]`, a jagged `object[][]`, an `object[,]`
    grid, a tuple, a `List<object>`) are NumPy's coercion of a Python list. np.array threw for every such list without
    text, np.asanyarray read it with C#'s types (an `object[]` of ints was int32) and cast a dtype request after
    discovery (300 into int8 wrapped to 44). `UnmanagedStorage.FromSequence(seq, requested)` is NumPy's walk with the
    descriptor set: the shape alone, then `setitem` per scalar item and an unsafe cast per array item. A probe of 368
    calls against NumPy: 181 identical before, 358 after (the rest: four object arrays refused as documented, four
    OverflowError texts where NumPy's Windows build differs from the LP64 text NumSharp pins, two artefacts of the
    probe). Breaking: np.asanyarray of C# ints in such a list is int64 (was int32), of C# floats float64 (was float32);
    a dtype request packs every item (OverflowError, TypeError where values wrapped or lost their imaginary part); an
    invalid order raises before the operand is converted. An empty object grid keeps its own dims (NumPy's lists cannot
    state one with no rows). The old converters `ConvertNonGenericEnumerable`, `ConvertObjectGrid` and `ConvertTuple`
    are gone.
  - Gates, every commit: the coercion snapshot identical to the parent's (the pure moves and the first conversion),
    or its changes classified line by line (the second conversion: 13 lines, the refusal text of a Python int past 64
    bits; D4: 10,551 lines in the list forms of np.array and np.asanyarray); a probe of 5,608 np.array calls over mixed
    object lists (text, bytes, char, None, Python and NumPy scalars, nested lists) under nine requests identical across
    the moves and conversions; R1 / R2 tightened; the doc warnings unchanged by the moves and conversions, one fewer with D4 (the
    deleted converters' NDW012); NumSharp.Tests and NumSharp.Tests.Oracle on net10.0 and net8.0.
  - R1 for the batch: 76 lines leave (text 25, numeric 51): `NDArray.Implicit.Array.cs` 51, `NDArray.Scalar.cs` 5,
    np.array.cs 8, np.asanyarray.cs 4, np.asarray.cs 2, np.array.string.cs 1, np.copyto.Strings.cs 1, np.pad.cs 3,
    np.loadtxt.Sinks.cs 1. The facades left hold no dtype test.
  - Left for later (recorded): the coercion still has two walks - the text builder (C# arrays, any value under a
    request; text discovery through the slots) and the sequence walk (lists, tuples, enumerables, ndarray items; its
    dtype an NPTypeCode, so an ndarray item of a text dtype is refused); one walk with descriptor promotion would serve
    both. np.array of a ragged jagged TYPED array truncates silently (`[[97], [99]]` from `{97}, {99, 100}`: the typed
    path's `Shape.ExtractShape`). `NDArray.Scalar(value, dtype)` keeps its documented C# conversion (an out-of-range
    value wraps where NumPy's `np.array(300, int8)` raises). An `IEnumerator` keeps NumSharp's own reading (NumPy does
    not iterate one).
  - Next: batch 5 (constructors).

- **2026-10-29, P3: batch 9 (the computing functions in P3's files).** Ten commits on `strings` from the parallel
  session (developed in a scratch worktree on `8a2d53fe`, landed by fast-forward):
  - `6a7f4bf9`: the snapshot tool reaches the batch. np.linspace / np.logspace / np.geomspace take a (start, stop) pair
    as their operand (all eight overloads were SKIP: no array parameter), with their own roles for num, base, axis and
    device; np.trim_zeros' trim spec and np.vander's N get roles; np.vander runs on the 1-D pools of np.convolve and
    np.trim_zeros on operands with zero borders. 11,788 calls, deterministic with and without guard pages.
  - `ca9a0246`, the pure move: np.linspace / np.logspace / np.geomspace (real and complex ends) are the abstract
    `TensorEngine.Linspace` / `Logspace` / `Geomspace` in the new `TensorEngine.CoreFunctionBase.cs`
    (`numpy/_core/function_base.py`), implemented in `Creation/Default.CoreFunctionBase.cs` with their eight helpers.
    The facades have no operand, so they call `np.EngineOf(null)`: the default engine (np.fft.fftfreq's precedent).
    R1: 25 numeric lines leave.
  - `315b2f0e`: np.vander is `TensorEngine.Vander` in the new `TensorEngine.TwoDimBase.cs`, implemented in
    `Creation/Default.TwoDimBase.cs`, `[NDScoped]` as the facade is. R1: 3 text lines leave. Its message counts 2,826
    np.vander calls; the snapshot holds 1,456 (`7c93e25a`'s message corrects it).
  - `7c93e25a`: np.trim_zeros is `TensorEngine.TrimZeros` in `TensorEngine.FunctionBase.cs`, implemented in
    `Manipulation/Default.TrimZeros.cs`. R1: 2 text lines leave.
  - Each move: the snapshot byte-identical to the parent's under `NUMSHARP_DEBUG_GUARD_PAGES=1`; no moved name binding
    to a `DefaultEngine` / `TensorEngine` member; R2 unchanged; doc warnings unchanged at 3,850.
  - No slot conversion. np.trim_zeros' text truth stays the bulk truth kernel (`NDStringTruth`'s nonzero flavor)
    inside `DefaultEngine.TrimZeros`: the `nonzero` slot reads one element, and "a slot replaces a test of the dtype's
    name, never a typed kernel". The batch's other lines (np.linspace's typed fill, the dtype requests, np.vander's
    ufunc refusals) were to become engine members, which they are.
  - A replay of the snapshot through NumPy 2.4.2 (each call's arguments to NumPy; dtype, shape, flags, C-order bytes or
    the error compared) and an edge probe of 160 calls the pools do not hold found six divergences, fixed per D4 with
    tests built from NumPy's output (`Creation/np.spacing.NumPyEdges.Test.cs`):
    - `a7c23095` np.linspace's negative count is NumPy's ValueError, checked first (NumSharp raised the allocation's
      text, or the device's): `TensorEngine.Linspace` gains `device=`, checked after the count. 140 snapshot calls.
    - `88afce1d` np.linspace / np.logspace: one sample is `0 * (stop - start) + start` (NaN for an infinite or NaN end,
      +0.0 for a start of -0.0), and a step that underflows to zero takes gh-5437's `(i / div) * delta + start`
      (`LinspaceValue`). The pools hold finite ends only; the edge probe found both.
    - `fb17feec` np.linspace's text dtypes are NumPy's `astype` of the float64 samples (NumSharp refused them). 172
      snapshot calls. `beb55575` releases the samples with a using declaration instead: `fb17feec`'s CastSpacingResult
      call added an NDW012.
    - `5de82c92` np.geomspace's text dtypes fail NumPy's statements right after the zero check (`StringDType`:
      `result_type`'s DTypePromotionError; `S` / `U`: `sign`'s no-loop error), before the count and the axis. 212 calls.
    - `681f9608` np.geomspace's `axis=-1` returns NumPy's moveaxis view (OWNDATA false unless the cast copies). 80
      calls.
  - Left, documented: the complex path of np.geomspace differs from NumPy in the last bits of 40 snapshot calls (its
    documented composition through NumSharp's complex power and log10, accurate but not byte-reproducible); the device
    error's type stays the house `ArgumentException` (NumPy's text). The replay shows no other difference for
    np.linspace (1,164 calls), np.logspace (1,442) or np.geomspace (1,528).
  - Gates: NumSharp.Tests 20,752 passed on net10.0 and net8.0 (on net10.0 one concurrency gun of the dictionary tests,
    `Gun_SwapBackRacing_BitTaggedIntKeys_KeyPathNeverTears`, did not fire under the machine's load and passed alone
    three times); NumSharp.Tests.Oracle 414 passed on both; Core's doc warnings 3,850 -> 3,848.
  - Next on this session's side: batch 7.

- **2026-10-29, P6: the computational part.** Ten commits on `strings` from the parallel session (developed in a
  scratch worktree, rebased onto `ce5d6ce3` and then onto P3's batch 3, `f90b131f`, landed by fast-forward). R2's
  computations 132 -> 16, its structural calls 264 unchanged. Each converted call is the engine member its facade
  dispatches to (a facade is its class boundaries plus `x1.TensorEngine.X(x1, x2, dtype, out, where)`), in the
  engine's argument order: `this` in an instance method, the operand's engine in a static one; np.evaluate becomes
  `TensorEngine.Evaluate` with default options (the facade's for a call without keywords); np.real / np.imag of a known
  complex array the lane aliases they build. No class rule fired at a converted call: the operands are scratch arrays
  of the route, or sit in a `PlainFor` region, or meet a call that refuses; the regions stay and their comments name
  what they guard.
  - `75e61cae` np.var / np.std / np.nanvar / np.nanstd / np.nanmean (`Default.Reduction.VarStd.cs`, `.NanVarStd.cs`,
    `.NanMean.cs`), 44 calls; SqrtLikeNumPy's decimal walk is an index loop instead of `np.nditer<decimal>`. Snapshot
    48,192 lines identical; one NDW012 fewer.
  - `01a86de4` np.sum's bool axis counts and its logical-OR loop, np.count_nonzero's sum, np.nansum / np.nanprod's NaN
    mask and `NDArray.Contains`, 5 calls; 27,804 lines.
  - `3a6c820b` np.isclose / np.allclose, np.power's fast paths, np.evaluate's delegated reductions (ptp, nanmin,
    nanmax, argmax, argmin) and its logical streams, 25 calls; 116,222 lines. `d56859d6` follows it: the flat argmax /
    argmin index it left to the finalizer (two new NDW012) is read and disposed by a helper (60,888 np.evaluate lines
    identical).
  - `5c17faec` numpy.linalg's empty-operand lstsq residuals and the conjugating products' `Conjugate` (a pointer loop
    over the fresh copy instead of `np.nditer<Complex>`), 8 calls; 45,810 lines.
  - `d19c237f` the text routes of the core functions (`Backends/Strings`), 34 calls. Read against §8 file by file: a
    helper that serves a core function's text route is engine work (np.where's and np.copyto's / astype's ASCII checks,
    the logical ufuncs over text, the bool-loop scans of text, `StringDType`'s mean / var / average / ptp and order
    statistics; np.flatnonzero computes indices from values, the engine's by §1); `NDStringMethods.cs` is numpy.strings'
    own Python statements (the computed-width wrappers' sizing), left for the text API's pass. 664,893 lines identical.
    The snapshot's text pools are ASCII digits and plain `StringDType`, so a probe of 3,395 calls covered the rest (S
    bytes at or above 0x80 through both of np.where's decode loops and np.copyto's masked verdicts, the unit above
    U+10FFFF, NaN-like NAs, 600-element lines, a numeric NA object as q), identical as well.
  - The snapshot tool, taught before each family: `6b141173` NDArray.Contains's value (`ContainsValue`), `30a449d8`
    np.evaluate's expression (23 forms that reach every root reduction it runs or delegates), `3aa56f41` zeroes the
    masked-off elements of a fresh elementwise result under where= before hashing (they are uninitialized, NumPy's
    ufunc rule: 590 np.power lines differed between two runs of one build). Some messages cite hashes from before the
    two rebases: `2090acde` (`6b141173`, cited by `01a86de4` and `30a449d8`), `d7b572b7` (`30a449d8`, cited by
    `3aa56f41` and `3a6c820b`), `ffeab6a4` (`3aa56f41`, cited by `3a6c820b`) and `e63755d2` (`3a6c820b`, cited by
    `d56859d6`).
    Across the first rebase the merged tool gives `ce5d6ce3`'s 75,416 lines of np.copyto / NDArray.fill /
    np.concatenate and the pre-rebase lines of np.evaluate and NDArray.Contains byte for byte; the second rebase
    touched no file of the stack but the plan.
  - Left, pinned in `engine_upcall_allowlist.tsv` with their reasons: np.where in `Default.Reduction.Nan.cs`,
    `.NanScan.cs` (NumPy's `_replace_nan` is `np.copyto(a, val, where=mask)`), `NDStringArrays.cs` and
    `NDVStringMove.cs` (2), and np.apply_along_axis in `NDVStringStatistics.Quantile.cs` (2): the array object's (§1,
    decided 2026-10-28), for P3's batch 7; `NDStringMethods.cs`' 9 (§8).
  - Gates, every code commit: the snapshot identical to the parent's; R2 tightened; Core's Release warnings
    (`--no-incremental`, `-p:WarningLevel=4`) 3,854 -> 3,853 (VarStd.cs's NDW012), none up; NumSharp.Tests and
    NumSharp.Tests.Oracle green on net10.0 and net8.0 (on the stack rebased onto `ce5d6ce3`, on its tip, and on the
    tip rebased onto `f90b131f`).

- **2026-10-29, P3: batch 3 (assignment).** Five commits on `strings`, landed by fast-forward:
  - `ce5d6ce3`: the snapshot tool reaches np.copyto (its three non-generic overloads were SKIP; the generic one was not
    enumerated: the tool now closes it over the 15 unmanaged types NumSharp stores) and NDArray.fill (Role.AssignArg).
  - `794fa292`, the pure move, part 1: np.copyto's internals (`CopytoAssignArray` = `PyArray_AssignArray`,
    `CopytoAssignRawScalar`, the weak-scalar half, the string and `StringDType` routes) into
    `Manipulation/UnmanagedStorage.Assign.cs`, `.Assign.Strings.cs` and `.Assign.VString.cs`; the four public
    overloads keep the argument parsing and the C# forms of the source. R1: the three files 36 -> 2 lines.
  - `448bde50`, part 2: ndarray.fill's body is `UnmanagedStorage.FillWithScalar` (`PyArray_FillWithScalar`) in
    `.Assign.Fill.cs`; its helpers, NDArray instance methods until then, take the array as `self`. R1: 4 -> 0.
  - `fd1fa7af`, part 3: np.concatenate's body is `NDArray.ConcatenateInto` (`PyArray_ConcatenateInto`) in the new
    `Creation/NDArray.Multiarray.cs`, with its copy paths and its string route (`np.concatenate.Strings.cs` is gone).
    R1 learns the home names (`LayeringScanner.ArrayObjectHomes`: Appendix D's eleven `NDArray.<NumPy source>.cs`
    partials, left out of the facade scan; a test holds the rule), and `NDArray.FunctionBase.cs`'s hand pin goes with
    it.
  - `713fbaf3`, the conversion's first part: the seven reference-holding decisions of the assignment ask
    `HasReferences` of the descriptors NumPy tests (`PyDataType_REFCHK`) instead of `Storage.IsVString`.
  - Gates for each: the snapshot of np.copyto, NDArray.fill and np.concatenate byte-identical to the parent's (75,416
    lines, `StringDType` / `U5` / `S5` operands in its pools); no unqualified name of moved code binding to a member of
    its new class; R2 unchanged; doc warnings 3,854 -> 3,851 (crefs that resolve in their new homes). NumSharp.Tests
    and NumSharp.Tests.Oracle green on net10.0 and net8.0 after the moves and after the conversion.
  - R1 for the batch's six files: 46 lines -> 2 (np.copyto.cs: the generic overload's char-into-text test;
    np.copyto.Strings.cs: the str overload's 0-d text array, both coercion, batch 4), plus `NDArray.FunctionBase.cs`'s
    15 lines with the home rule.
  - Deferred, a design decision of this batch: the text casts move into the casting impls' loops as batch 9b, after
    the moves (one set of loops serves batches 3, 4, 6 and 8); the values' packing into a text descriptor goes to
    `setitem` with batch 4.
  - Next: batch 4 (coercion).

- **2026-10-29, P3: batch 2 (item selection).** Three commits on `strings`, landed by fast-forward:
  - `f5547943`, the pure move. Into `Indexing/UnmanagedStorage.ItemSelection.cs` (new; NumPy's `item_selection.c`,
    `compiled_base.c`'s `arr_place` and `fromnumeric.py`'s resize), as internal static members with their helpers:
    `TakeFrom`, `PutTo`, `PutMaskCore`, `PlaceCore`, `Repeat` (both overloads' bodies), `ChooseCore`, `Compress` (the
    body) and `ResizeRepeating` (np.resize's body: the same tiling memory work as np.repeat, so its statements sit
    beside it although they are Python in NumPy). ndarray.resize's body went into
    `Backends/Unmanaged/UnmanagedStorage.Reshaping.cs` as `Resize` (NumPy's `PyArray_Resize`): it calls no np.*, so
    it is under `Backends/` already. `ItemSelection.cs` stays beside its facades because its statements call np.*
    (C3; P6 moves it down). The facades keep the class boundaries, `PlainFor`'s region, the C# null checks and their
    argument conversions; `OperandRootsTests` finds `PlaceCore` / `PutMaskCore` on `UnmanagedStorage`.
    Gates: the snapshot of the 14 targets byte-identical to the parent's (144,451 lines, the tool taught their
    arguments in `bfb19bf6`); a check that no unqualified name of the moved code binds to an `UnmanagedStorage`
    member; R1 56 -> 2 lines in the nine files; R2 and the doc warnings unchanged; NumSharp.Tests and
    NumSharp.Tests.Oracle green on net10.0 and net8.0.
  - `d7f5f62d`: NativeAllocationChokepointTests' pin for bincount's counting table names
    `Default.Bincount.cs`, where `81074517` moved it; the oracle suite failed on every run since. Found by this
    batch's gate run.
  - `ac49b5b7`, the conversion. The ten route decisions ask `HasReferences` of the descriptor NumPy tests
    (`PyDataType_REFCHK`: `self` for take, put, putmask, place, compress and resize, `ret` for repeat, the common
    descriptor for choose, `PyArray_ClearBuffer`'s for ndarray.resize) instead of the storage's `IsVString` (or, for
    np.choose, the class of its string descriptor). Snapshot byte-identical (its pools hold 14,260 `StringDType`
    cases); both suites green on both frameworks.
  - What the batch leaves, by mechanism: the values' conversion into the target's descriptor (`PutValuesLikeNumPy`'s
    `StringDType` store; np.putmask's and np.place's list overloads, R1's two remaining lines) goes with batch 4
    (`Pack` through `setitem`); the masks' text -> bool cast and `PlaceValuesLikeNumPy`'s text conversions with
    batch 3 (the casting impls); np.choose's text promotion with the weak literals (`DTypePromotion`); the index-kind
    tests stay.
  - Next: batch 3 (assignment).

- **2026-10-29, P5: done.** Commits on `strings`, developed in the scratch worktree and landed by fast-forward, with
  P2's referees (the snapshot under `DOTNET_TieredCompilation=0` and `NUMSHARP_DEBUG_GUARD_PAGES=1`, each family's
  unit tests on net10.0 and net8.0, its oracle tiers):
  - `28869e8b` the classification (P5's section). The snapshot tool: `8abc0910` varies np.nan_to_num's fills and
    np.real_if_close's tol; `23c76eb7` varies np.unravel_index's dimensions, order and flat index (both overloads were
    SKIP), np.packbits / np.unpackbits' bitorder and count and np.bincount's minlength, and runs np.bincount on 1-D
    operands of valid bins (`Magnitudes`): its old pool held only negative 1-D values, so 15 of its 1,224 lines reached
    the kernels, against 647 of 2,964 now.
  - The six families, each byte-identical on its snapshot, R1's row gone (np.isreal_iscomplex.cs's reduced to the
    three dtype-routine lines of isrealobj / iscomplexobj), R2 unchanged, each member abstract and documented in the
    partial of its NumPy file, `DefaultEngine`'s override beside the facade:
    - `ec16612f` np.nan_to_num: `NanToNum` in the new `TensorEngine.TypeCheck.cs`, `Logic/Default.TypeCheck.cs` (the
      fused pass and the copyto composition); 12,798 calls.
    - `b499bf98` np.isreal / np.iscomplex / np.real_if_close: `Isreal` / `Iscomplex` / `RealIfClose`; 1,152 calls.
    - `595e112c` np.union1d / np.setxor1d / np.setdiff1d: `Union1d` / `Setxor1d` / `Setdiff1d`
      (`TensorEngine.ArraySetOps.cs`), `Manipulation/Default.SetOps.cs` with `CanonicalizeSetOpNaN`; np.setops.cs is
      gone; 3,426 calls.
    - `f0fd683d` np.sort_complex: `SortComplex` (`TensorEngine.FunctionBase.cs`); 234 calls.
    - `a6dae9ea` np.roots: `Roots` (`TensorEngine.Polynomial.cs`), its body in `Polynomial/Default.Polynomial.cs`; the
      facade keeps NumPy's dispatcher refusal; 234 calls.
    - `e7d7869a` np.conjugate's loop resolution (NumPy's conjugate has no bool loop: int8 without `dtype=`) and
      `203b28ef` np.isposinf / np.isneginf's complex refusal, into the members the facades already called; 12,108 and
      1,908 calls.
  - The coordinator's lines, handed over the same day:
    - `d1675ec5` np.argmax / np.argmin's out= contract: `ReduceArgMax` / `ReduceArgMin` (`TensorEngine.Methods.cs`)
      take `out=`, as `ReduceAMax` took its keywords in E5; `DefaultEngine` computes the indices (`ArgMaxIndices` /
      `ArgMinIndices`, the old bodies) and writes them through `WriteArgOut` / `ValidateArgOut`
      (`Sorting_Searching_Counting/Default.ArgOut.cs`, beside the facades because it calls np.copyto). np.nanargmax.cs's
      row (0 / 9) is gone; 15,240 calls.
    - `28773dcc` np.unravel_index: `UnravelIndex` (`TensorEngine.Multiarray.cs`), `Indexing/Default.UnravelIndex.cs`;
      3,114 calls of the array overload. Its row goes from 0 / 2 to 0 / 1: the line left is `ThrowIfNotIntIndices`
      (NumPy's `astype_anyint`), which np.ravel_multi_index shares; it moves with that function (P3's batch 7 names its
      lines, and Appendix D puts both in `Multiarray`).
    - `a4d0d69a` np.packbits / np.unpackbits: `Packbits` / `Unpackbits`, `Manipulation/Default.PackBits.cs`. Their
      (n_outer, axisLen, nel) collapse came from np.repeat.cs's private `ComputeAxisGeometry` (P3's batch 2 file); it is
      the internal `Shape.SplitAtAxis` now (`View/Shape.AxisGeometry.cs`), and np.repeat can take it when it moves;
      2,646 + 6,054 calls.
    - `81074517` np.bincount: `Bincount`, `Sorting_Searching_Counting/Default.Bincount.cs`; 2,964 calls.
    - `82abb696` np.kron: `Kron` in the new `TensorEngine.ShapeBase.cs` (kron is `_shape_base_impl.py`'s only computing
      function), `LinearAlgebra/Default.Kron.cs`; 954 calls.
    - `73dfcd07` np.median's chararray answer: `DefaultEngine.MedianOfChararray` (internal) beside the quantile
      family's text statements in `Statistics/Default.Quantile.Median.cs`; the facades decide the class and throw what
      it returns. 7,374 calls (plain operands) identical; the string oracle's chararray cases pass.
  - Two D4 fixes, each pinned by unit tests that carry NumPy's probed texts:
    - `127c3c98` np.argmax / np.argmin check a text operand's out (its shape, then its cast) before they compare, as
      `_PyArray_ArgMinMaxCommon` does: a `StringDType` None-like NA raised its comparison error over a bad out
      (np.nanargmax already checked first; its pre-check is now `ArgTextChecks` in `Default.ArgOut.cs`). The snapshot
      is unchanged: its text pool holds no None-like NA.
    - `d313e41b` np.nan_to_num refuses a text or complex fill as NumPy's copyto refuses it (same_kind into a float
      array: `Cannot cast scalar from dtype('<U1') to dtype('float64') according to the rule 'same_kind'`), at the
      statement that meets the fill; it parsed the text ('1.5' filled 1.5, 'x' raised .NET's FormatException). A char
      fill is its code unit. 654 of 12,798 snapshot lines change, every one a text-fill case, each NumPy's answer (27 of
      them are an earlier row fill's broadcast error, NumPy's order).
  - R1 after P5: 106 files, 249 text and 651 numeric lines (121 / 255 / 717 at the classification; E3's and
    np.arange's lines left with them). R2 unchanged. Left of P5's classification: np.evaluate's two text lines (P2, an
    owner decision), np.unravel_index's `astype_anyint` line (with np.ravel_multi_index), P3's files, and the lines
    that stay (C, D).

- **2026-10-29, P3: batch 1 (the slots) and the np.arange fix it found.** Two commits on `strings`, landed by
  fast-forward:
  - `95b7167e`: `DTypeMeta.setitem` / `getitem` (NEP 42's slots 7 and 8) and the ArrFuncs `copyswapn` / `fill` /
    `nonzero`, internal, implemented by every class with storage from the code each question is answered by today:
    - the 19 numeric classes: `ReductionInitial.PackStore` / `Box` with the swap of a non-native descriptor, NumPy's
      `@NAME@_fill` operation for operation (C's promotion, `HALF_fill` in float32, complex by parts, `BOOL_fill`'s
      TypeError), `@fname@_nonzero` (a float16 by its bits, a swapped copy for a non-native float);
    - `S` / `U`: the element codec of every text store and read;
    - `StringDType`: its buffer's allocator, the cell checked to be one of the buffer's, `nonzero` NumPy's (an NA cell
      is nonzero only for a NaN-like NA object), `copyswapn` refused;
    - the classes without storage refuse.
    The signatures gained `IArraySlice buffer` (the `StringDType` allocator lives on its buffer in NumSharp).
    Gate: `DTypeSlotTests` (16). Every class is held beside the path the slot comes from: the flat iterator's store,
    the element read, `ndarray.byteswap`, `np.nonzero`, `StringArraySlice` / `VStringArraySlice`. It is also held to
    NumPy 2.4.2's bytes: the stores and their errors, non-native descriptors, `fill` against np.arange's ramps
    (integer wraps, the float bits, HALF_fill's tie at 0.2998), and the nonzero indices of the specials.
  - The gate's np.arange comparison failed for float16. NumSharp's np.arange computed every float ramp as
    `start + i * step` in double. NumPy stores `start` and `start + step` and fills `start + i * delta` in the
    dtype's own arithmetic. Over 2,000 random calls per dtype, 68 % of float64, 92 % of float32 and 91 % of
    float16 results differed. The oracle's 14 arange cases were all `arange(0, 7, 1)`.
  - `8be47c75` (D4): np.arange is NumPy's `PyArray_ArangeObj`, ported into
    `Backends/Unmanaged/UnmanagedStorage.Ctors.cs` (`ctors.c`, Appendix D's home):
    - the length from Python's arithmetic on the bounds (float and Python-int overloads, the underflow and
      divide-by-inf counts, the 2**63 case NumPy's x86 cast makes empty);
    - `start` and `start + step` stored through `setitem`, the rest through `fill`; `HasFill` (NumPy's
      `funcs->fill != NULL`) checked first.
    Divergences removed with it: the long overloads were exact only below 2**53; out-of-range stores wrapped instead
    of raising NumPy's OverflowError; a zero step (and any step below 1e-15) raised NumSharp's own ArgumentException
    instead of Python's ZeroDivisionError; text dtypes had no NumPy text. np.ma.arange's zero-step gap MS21 closed with
    it: the ma surface tier's two cells returned to the gate.
    Gate: the oracle tier `arange_full.jsonl` (475 cases, zero excuses; 272 fail on the parent), `np.arange.NumPyPort`
    (7), the battle tests' zero-step texts. R1 lost `np.arange.cs`'s 18 lines; R2 is unchanged.
    Cost: np.arange(10) 0.41 -> 0.49 us (the two slot stores; NumPy 0.54); the large ramps unchanged within the host's
    noise.
  - Next: batch 2 (item selection).

- **2026-10-29, P3: the design.** Written into P3's section and Appendix E, before any code (the plan's rule):
  - Scope at `1fcb4663`: R1 counts 702 lines (469 numeric, 233 text) in P3's 80 facade files; K4 adds 20 text lines in
    nine files R1 does not scan. `classify_p3.py` (scratchpad `p3/`) sorts each line of R1's own dump by the mechanism
    NumPy uses for the same decision: typed dispatch 301, the storage class asked by name 89, reference-holding moves
    69, index kind 64, dtype= requests 43, per-dtype values 37, stored Python values 28, text casts 23, weak literals
    11, codecs 10, discovery 7, truth 3, element reads 2, ufunc refusals 2, unclassified 13.
  - The slots: `setitem` / `getitem` (NEP 42's 7 and 8, from today's `PyArray_Pack` port and the text stores),
    `copyswapn`, `fill` and `nonzero` (the legacy ArrFuncs NumPy still calls), `HasReferences` with the clear and
    zero-fill loops made live, and the casting impls' loops for the text casts. A slot replaces a test of the dtype's
    name, never a typed kernel.
  - The homes: Appendix D's `UnmanagedStorage` partials (with a new `.Mapping.cs` for the indexers' typed fetch and
    store) and the `NDArray` partials named after NumPy's sources; a partial stays beside its facades while it calls
    np.* (C3); R1 learns the home names and leaves those partials out.
  - Ten batches: the slots first; then item selection, assignment, coercion, the constructors, buffers and IO, NumPy's
    Python functions of the array object, the indexers, the computing functions in P3's files (np.linspace /
    np.logspace / np.geomspace, np.vander, np.trim_zeros as `TensorEngine` members), and what stays (13 lines in five
    files, the index-kind tests, the IO codecs).
  - Next: batch 1 (the slots).

- **2026-10-29, PE: E3 done, so PE is.** `1b0ba640` on `strings`, landed by fast-forward:
  - numpy.fft's statements were the `np.fft` module object's own method bodies (`FourierModule`). They are now the
    public `DefaultFftEngine` (`Fourier/DefaultFftEngine*.cs`, beside the facades for ratchet R2; `np.fft.RawFft.cs`
    moved in by git mv), and the public abstract `FftEngine` (`Backends/FftEngine.cs`) is the contract: numpy.fft's 16
    functions that compute, NumPy's names, order and defaults. `fftshift` / `ifftshift` stay facades over np.roll
    (Appendix D); the E3 text's "18 functions" was corrected to Appendix D's 16.
  - `_raw_fft` ends in the kernel of its kind: `C2CKernel` (fft / ifft), `R2CKernel` (rfft_n_even / rfft_n_odd),
    `C2RKernel` (irfft), each one call into `PocketFFTDriver.Execute` with `isReal` / `isForward` fixed (the driver,
    public since 0.70.0, is unchanged). A kernel receives what NumPy's statements resolved: the length, the normalized
    axis, the factor in the operand's real dtype, the checked out, and the two flags NumPy's loop selection needs (a
    float32 / float16 call, a factor that is NumPy's integer 1). `TensorEngine.Fft` (beside `Linalg`) holds the assigned
    engine or the shared default; null restores it; fftfreq / rfftfreq read the default engine's.
  - Decided on the way: nested calls follow `DefaultLinalgEngine`'s rule (the np.fft facade where NumPy calls its module
    function: hfft's irfft, ihfft's rfft, rfft2's rfftn, irfft2's irfftn); the N-D transforms keep running the engine's
    own `_raw_fft` per axis, because only the statements carry a float32 operand's precision across NumSharp's
    complex128 intermediate. hfft's facade keeps its class boundary in front of the call (the member's np.conjugate
    would keep a matrix). The facades without a temporary of their own drop `[NDScoped]`; the members own it.
  - Gates: the fft snapshot (138,720 calls) byte-identical to the parent `aee2e626` (the tool commit that taught it
    np.fft); a 64-call null-operand, out= and numpy.matrix / chararray probe identical to the parent; a reflection diff
    showing `FourierModule` and `PocketFFTDriver` unchanged, `TensorEngine.Fft` and the two new types; the build's
    3,216 warning lines and 215 unresolved crefs unchanged; `FftSubEngineTests` (5: the default, a probe subclass
    reached by all 16 functions for arrays created before the assignment, the nested compositions, a kernel provider
    seeing the resolved arguments and returning its result, out= through the kernel); ratchet R1 lost the
    `Fourier/np.fft.RawFft.cs` row.
  - Next: P3 (its design section first).

- **2026-10-29, P2's nineteenth to twenty-eighth families and P4's D4 fixes.** Commits on `strings`, developed in a
  scratch worktree and landed by fast-forward. The referees: the snapshot (`DOTNET_TieredCompilation=0` under
  `NUMSHARP_DEBUG_GUARD_PAGES=1`), each family's unit tests on net10.0 and net8.0, its oracle tier where it has one, and
  for P4 `twin_poly.py` (each snapshot line of the seven legacy functions rebuilt in NumPy 2.4.2, recorded with
  `SNAPSHOT_BYTES=1`).
  - The moves, each byte-identical on its snapshot, R1's row gone (np.nanargmax.cs's reduced), R2 unchanged; each member
    declared abstract and documented in full in the partial of its NumPy file, `DefaultEngine`'s override beside the
    facade (it calls np.*), the facade keeping its class boundaries and calling `np.EngineOf(operand)`; no moved name is
    declared in another `DefaultEngine` or `TensorEngine` file (an unqualified np.* call inside a moved body is
    qualified `np.`):
    - `2a335bac` np.radians / np.degrees: `Radians` / `Degrees` (`TensorEngine.Umath.cs`, NumPy's ufuncs of those names,
      which deg2rad / rad2deg alias), `Backends/Default/Math/Default.RadiansDegrees.cs`; 24,216 calls.
    - `be34c571` np.sinc: `Sinc` (`TensorEngine.FunctionBase.cs`), `Math/Default.Sinc.cs`, the fused pass; 234 calls.
    - `0a09203b` np.i0: `I0`, `Math/Default.I0.cs`; the per-precision cephes kernels stay in np.i0.cs (np.kaiser shares the
      double one), their delegates internal; 234 calls.
    - `5b4f10a1` np.diff: `Diff(a, n, axis, prepend, append)`, `Math/Default.Diff.cs` with the stencil helpers (np.DiffAxis
      stays np's: Default.Trapezoid imports np's statics); 774 calls.
    - `b3d0fff2` np.ediff1d: `Ediff1d(ary, to_end, to_begin)` (`TensorEngine.ArraySetOps.cs`), `Math/Default.Ediff1d.cs`
      over np.diff's helpers; 234 calls.
    - `51d13591` np.digitize: `Digitize(x, bins, right)`, `Sorting_Searching_Counting/Default.Digitize.cs`; 1,920 calls.
    - `d221d740` np.unwrap: `Unwrap(p, discont, axis, object period)`. NumPy's `period` is one keyword whose Python type
      (float or int) picks the float or the integer path, so one member serves both facades (a boxed double or null, a
      boxed long or int); `Math/Default.Unwrap.cs`; 775 calls (the tool reaches the double overload; unwrap.jsonl and the
      unit tests hold the integer one).
    - `7c9bc5fc` np.cross: `Cross` (`TensorEngine.Numeric.cs`), `LinearAlgebra/Default.Cross.cs`; 2,646 calls;
      products.jsonl passes.
    - `dcb3623b` np.nanargmax / np.nanargmin: `NanArgMax` / `NanArgMin` (`TensorEngine.NanFunctions.cs`),
      `Sorting_Searching_Counting/Default.NanArgMax.cs`. The out= pair `NanArgWriteOut` / `ValidateNanArgOut` stays np's
      (np.argmax calls it), internal, so R1's row goes from 2 text / 15 numeric lines to 0 / 9 (its dtype switch). The
      `long` overloads convert the member's 0-d index, where they ran the flat ArgMax kernels: 6,612 calls over both
      overloads, identical; nanreduce.jsonl's 1,900 nanargmax / nanargmin cases pass.
    - `2c25668e` the snapshot tool reaches np.interp (both overloads were SKIP): xp and fp are fixed arrays (a ramp in
      several dtypes and layouts, repeated, decreasing and NaN points, complex and text values, the shapes NumPy
      refuses), the fills a double or a Complex, the period seven values; 30,720 calls. `135c91e7` np.interp:
      `Interp(x, xp, fp, object left, object right, period)`, NumPy's fills being Python floats or complexes (a complex
      fill reaches the real loop only to be refused), `Math/Default.Interp.cs`; 30,720 calls identical.
  - P4's D4 fixes. Before them the twin agreed on 5,590 lines, 683 differed and 1,516 raised the same type with another
    text; at no step did a line that agreed before differ:
    - `fb9b62a0` NumPy measures a coefficient array with `len()`, its first axis, where the members read `size`:
      np.polyval of an N-D p evaluates row by row, broadcast against x (it raised IndexError), np.polyint divides by
      `arange(len(p), 0, -1)` and measures k by `len()`, np.polyadd / np.polysub pad by the first axes' difference (an
      N-D operand then meets np.concatenate's rank error), np.polydiv loops `len(u) - len(v) + 1` times. +1,712 lines.
    - `264e35cf` (found beside the next) np.polydiv forms `w = u[0] + v[0]`, so rows that do not broadcast raise first,
      u's row first; in np.result_type's dtype, because NumSharp's + takes a 0-d operand as weak. +42.
    - `d9778991` np.polydiv's `q[k] = d` is a scalar setitem: a d with dimensions raises `setting an array element with a
      sequence.` (a complex slot `only 0-dimensional arrays can be converted to Python scalars`), probed over the float
      and complex destination and source dtypes; the complex path had read one element of the row. +252.
    - `21b5c845` np.poly / np.roots of a 0-d array or null raise `dispatcher for __array_function__ did not return an
      iterable`: their dispatchers return the argument itself, which NumPy iterates before the body runs. The facades
      raise it; `TensorEngine.Poly` stays poly's body and takes a 0-d input as one root, as `np.poly.__wrapped__` does.
      The [Misaligned] test that pinned the old answers became a test of NumPy's. +34 (np.roots: its 36 0-d lines).
    - `4935fc54` (found by the twin after the first fix) the pad's zeros take the operand's dtype descriptor, NumPy's
      `zeros(diff, a1.dtype)`: a text type code met NumSharp's fixed-width NotSupportedException. +84.
    - `1b0bd94d` np.polyval's N-D path computes each statement in NumPy's dtype, passed explicitly: with the operators a
      0-d x was weak (an int8 p with a 0-d int64 x gave int8). +30.
    - `01783d66` the members' docs. After: 7,744 lines agree, 12 differ, 32 have another text: the char proxy's class
      name (20), np.polymul's StringDType object product (10, NumPy computes Python objects) and a text of its legacy
      descriptor refusal (12), and two complex128 lines of NaN and inf whose imaginary NaN has the other sign (NumSharp's
      complex add / multiply NaN residual, OpenBugs.NaNOperandOrder).
  - Gates: the polynomial and roots unit tests (203) on both frameworks; FuzzCorpusTests.Poly, the matrix-operand,
    relayout and surface tiers and linalg_parity on both (their open-gap replays unchanged: 330 + 10 matrix-operand and
    20 relayout carved cases still diverge); the post-fix baseline (8,455 calls with np.roots) identical after the rebase.
  - Not done: np.evaluate's text lines (see P2). Found on the way, outside this work: np.argmax's out= contract still
    sits in its facade (P5), and the 1-D np.polyval path computes its first `0 * x` in result_type(p, x), which differs
    from NumPy for a real x holding inf with a complex p (NaN + NaNj against NaN + 0j); no snapshot line reaches it.

- **2026-10-29, PE: E2 done.** `36a1f084` on `strings`, landed by fast-forward:
  - The internal operation class `LinalgEngine` (`LinearAlgebra/linalg/LinalgEngine*.cs`) became the public
    `DefaultLinalgEngine` (git mv, beside the facades for ratchet R2), and `LinalgEngine` the public abstract contract
    (`Backends/LinalgEngine.cs`): numpy.linalg's 29 computing functions as 31 members (Appendix D's list; `Tensordot`
    and `VectorNorm` have a second C# form for NumPy's int-or-tuple argument). NumPy's names, order and defaults.
  - The ten `_umath_linalg` members left `TensorEngine` (shipped in 0.70.0 as public virtual: Breaking) for
    `DefaultLinalgEngine`'s `protected virtual` kernels `CholeskyKernel` ... `SvdKernel`, body for body (LinAlgEmpty,
    the operand engine's `Blas`, the managed LU or `OpenBlasMissingBackendException`, whose text still names the
    operand's engine). `TensorEngine.Linalg` (beside `Blas`) holds the assigned engine or the shared default; null
    restores the default.
  - Decided on the way: the five Array-API members Appendix D lists (`Matmul`, `Tensordot`, `Trace`, `Vecdot`,
    `MatrixNorm`) are NumPy's one-statement wrappers; their facades now apply the main-namespace function's class
    boundary before they call the member, because the default's call into `np.matmul` from a library frame at depth
    zero is what the matrix-boundary caller audit forbids. `Qr` takes the caller's mode and resolves it first (the
    facade resolves it too, for its matrix rule). Helpers that call a kernel or a virtual sibling are instance methods.
  - Gates: the linalg snapshot (66,113 calls) byte-identical to its baseline managed and with the bundled OpenBLAS; a
    78-call null-operand and matrix / chararray probe identical to the parent; a reflection diff showing exactly the ten
    removed members, the `Linalg` property and the two new types; `LinalgSubEngineTests` (5: the default, a probe
    subclass reaching all 31 members for arrays created before the assignment, nested calls, a kernel override serving
    the factorisations with NumPy's statements kept, the missing-backend text).
  - Next: E3 (`TensorEngine.Fft`); then P3.

- **2026-10-28, PE: E5 done.** Commits on `strings`, landed by fast-forward:
  - `04db655a` the snapshot tool varies the max / min family's `initial=` (np.max / np.amax / np.min / np.amin /
    np.nanmax / np.nanmin and NDArray's max / min): before it a `where=` mask always met NumPy's no-identity error, so the
    masked and seeded reductions never ran. 114,402 calls at the parent, a second run identical, the "small" group
    unchanged.
  - `bf862520` `ReduceAMax` / `ReduceAMin` (Methods) and `NanMax` / `NanMin` (NanFunctions) take `out=`, `initial=` and
    `where=`, appended after their existing parameters so every positional call binds as before (a third `out` would have
    taken a C# bool through the implicit bool -> NDArray conversion). np.nanmax / np.nanmin's pair joined though the plan
    named only ReduceAMax / ReduceAMin: `MaxMinNumPy` also serves fmax / fmin, which would otherwise have stayed past the
    engine. The facades make one call (`EngineOf`, so a null `a` keeps its exception); a call with any keyword takes
    `MaxMinNumPy` as its first statement; `dtype=` with a keyword is NotSupportedException (np.max has no `dtype=`). The
    IL guard's LT11 row names the widened signature. 114,402 calls byte-identical, a 50-call null / keyword-error probe
    too; the max / min tests (765) and the oracle's eight reduction tiers with the ratchets pass on both frameworks.
  - `MeanOverAxes` / `ReduceAddOverAxes` (decided, no commit): the plan made them members because the operation class
    AverageEngine reached them past the engine. E1 made np.average's body a `DefaultEngine` partial, so they are the
    engine's own helpers. A `ReduceAdd(int[] ...)` overload would also make every `ReduceAdd(a, null, ...)` call ambiguous
    (22 in src and tests, and any caller of 0.70.0's member), and its iterator has float / complex / decimal add loops
    only, so no public member was added.
  - Next: E2 (`TensorEngine.Linalg`), E3 (`TensorEngine.Fft`); then P3.

- **2026-10-28, PE: E6, E4, E1 and E7 done, E5 begun (the sort members).** Commits on `strings`, developed in a
  scratch worktree and landed by fast-forward, one family per commit. Every move is byte-identical on the snapshot,
  taken with `DOTNET_TieredCompilation=0` under `NUMSHARP_DEBUG_GUARD_PAGES=1`, against `df3a30c5` (E7's and E5's moves
  against their parents, whose tool knows their arguments):
  - `df3a30c5` E6 and E4: `TensorEngine.cs` keeps `Blas` and `Evaluate`; the other members sit in `TensorEngine.Umath.cs`
    (88), `.Methods.cs`, `.Multiarray.cs`, `.Numeric.cs` (IsClose, AllClose, Argwhere, FlatNonZero, CountNonZero ×2,
    Correlate, Correlate2), `.NanFunctions.cs` (NanSum, NanProd, NanMin, NanMax), `.UfuncLike.cs` (IsPosInf, IsNegInf)
    and `.ArrayObject.cs` (the ten array-object members of 0.70.0 and Diagonal). File organization only: the reflection
    dump of TensorEngine's 177 members is identical.
  - E1, the operation classes onto the partial files (each class `git mv`'d to `Default.<Name>*.cs` beside its facades,
    which R2 needs while it calls np.*; `public partial class DefaultEngine` in NumSharp.Backends; the entries public
    overrides; the helpers keep their statements, names and access):
    - `393ee639` np.angle -> `TensorEngine.Angle` (`TensorEngine.FunctionBase.cs`, new);
    - `8fc122ac` np.ptp -> `Ptp` ×2 (`.Methods.cs`);
    - `feb098a9` np.nancumsum / nancumprod / nanmean / nanvar / nanstd -> `.NanFunctions.cs`;
    - `68ed63bf` IsinEngine -> `Isin` (`TensorEngine.ArraySetOps.cs`, new);
    - `d2bc63f8` UniqueEngine -> `Unique`, `UniqueValues`, `UniqueCounts`, `UniqueInverse`, `UniqueAll`;
    - `adf600d7` ArrayEqualEngine, ConvolveEngine -> `ArrayEqual`, `ArrayEquiv`, `Convolve` (`.Numeric.cs`);
    - `d8545c5e` QuantileEngine -> `Quantile`, `Percentile`, `Median` (FunctionBase) and `NanQuantile`, `NanPercentile`,
      `NanMedian` (NanFunctions); q is NumPy's array_like (`object`), the axis NumPy's general form (`int[]`);
    - `ba16ae12` AverageEngine -> `Average` (a pair: the average and the sum of weights), `7d339c3f` CovEngine -> `Cov`,
      `a45832b2` GradientEngine -> `Gradient` (+ `np.EngineOf`), `4101af06` TrapezoidEngine -> `Trapezoid`;
    - `307aa982` HistogramEngine -> `HistogramBinEdges`, `Histogram`, `Histogramdd`, `Histogram2d`
      (`TensorEngine.Histograms.cs`, new);
    - `7408f707` InsertEngine and np.delete's statements -> the array object: `Manipulation/NDArray.FunctionBase.cs`,
      internal `NDArray.Insert` / `NDArray.Delete` overloads (E7's rule).
  - Snapshot calls per group: small 31,698; sets 18,930; numeric 25,890; quantile 480,552; stats 140,118 (and
    np.average_returned 16,548); hist 197,748; insert 118,969.
  - Name checks: DefaultEngine (and, for Insert, NDArray) is one namespace of member names, so each class's member names
    were checked against a reflection dump of DefaultEngine's, and its unqualified identifiers against them, before the
    move. Renamed where a name was taken: UniqueEngine Compute / Splitmix -> `UniqueCompute` / `UniqueSplitmix`;
    IsinEngine `PythonInt` -> `IsinPythonInt` (it hid the PythonInt class: CS0119); QuantileEngine Compute ->
    `QuantileCompute`, BooleanSubtractError -> `QuantileBooleanSubtractError` (np.var's helper: CS0111), its op-string
    bodies -> `QuantileFunction` / `NanQuantileFunction`; GradientEngine Add / Mul / Div -> `GradientAdd` /
    `GradientMul` / `GradientDiv` (DefaultEngine's ufunc Add); HistogramEngine IsFinite / Ptp -> `HistogramIsFinite` /
    `HistogramPtp`. In NDArray the moved code spells `System.Array` in full (NDArray has an `Array` property).
  - Static bodies where a static helper calls the function again: `CovFunction` (the parsed-text route),
    `HistogramFunction` (Stone's and the complex estimators), `HistogramddFunction` / `Histogram2dFunction`
    (histogram2d calls histogramdd); each override is `[NDScoped] X(...) => XFunction(...)`.
  - Scopes (E1's rule): the `[NDScopedCovered]` entries became `[NDScoped]` overrides (Cov, Histogramdd, Histogram2d,
    Trapezoid), and so did the unscoped bodies that relied on a scoped facade (Isin, the four Array-API unique members,
    Histogram, HistogramBinEdges). Bodies whose facades were unscoped stay unscoped (the quantile family,
    ArrayEqual, ArrayEquiv). Insert and Delete are internal and keep `[NDScopedCovered]` under their scoped facades.
  - Dispatch: each facade calls its operand's engine; where the body itself refuses a null operand (the quantile family,
    np.gradient, the histogram family) it goes through `np.EngineOf` (the operand's engine, or the default one), so the
    null still reaches the body's ArgumentNullException in NumPy's order.
  - np.median / np.nanmedian lost their one-axis bodies: an `int?` axis is passed as a one-element array, as NumPy's
    `_ureduce` turns an int into a 1-tuple; the quantile snapshot covers every overload and is identical.
  - Public API against `df3a30c5`: 41 new abstract `TensorEngine` members (the 31 above, `Einsum` and the nine `Emath`
    members of P2) and their 41 `DefaultEngine` overrides, and `Diagonal` gone (E7, below); nothing else changed on
    `TensorEngine`, `DefaultEngine`, `NDArray` or `np` (a reflection diff of the public and protected surface).
  - E7, `8f3407eb`: `TensorEngine.Diagonal` (`199feaaf`, unreleased) leaves the engine. np.diagonal's view is the
    internal `NDArray.Diagonal` (`Indexing/NDArray.diagonal.cs`, beside the ndarray method) over the new `Shape` partial
    `Shape.Diagonal` (`View/Shape.Diagonal.cs`: PyArray_Diagonal's statements up to the view, the exceptions keeping
    np.diagonal's argument names), and `DefaultEngine.Trace` reads its diagonal through it. 63,187 calls (np.diagonal,
    np.trace, their ndarray methods, np.linalg's Array-API forms and np.einsum) byte-identical to the parent `32478d0c`,
    the 26,004 outside np.einsum to `df3a30c5` too (the tool learned np.einsum's arguments in `34d1d3fa`).
  - Two things the full-suite checkpoint after E1 found, each with its own commit (D4):
    - `6b887534` np.ptp(null) raised a NullReferenceException: `8fc122ac` made both facades call `a.TensorEngine.Ptp`
      before `np.EngineOf` existed. They call `EngineOf(a)` now. A probe of every E1 facade with a null operand in each
      position (48 calls) matches `df3a30c5`: ptp was the only one whose ArgumentNullException was lost.
    - `32478d0c` ratchet R1 counted `Manipulation/NDArray.FunctionBase.cs` as a new facade file with 7 text and 8 numeric
      lines. R1 scans `NDArray.*.cs` outside `Backends/`, and the array object's homes are such files: np.insert's lines
      (in the operation class InsertEngine, which R1 never scanned, since `b9774efb`) and np.delete's boolean-mask test
      are pinned there by hand with the reason that P3 moves the array object's per-dtype semantics into the DType
      slots; np.delete.cs went from 21 to 20 numeric lines.
  - E5, the sort family (the rest of E5 is open):
    - `e185a4b3` the snapshot tool reaches the sort family: kind (the sort kinds and the select kind, names NumPy parses
      and refuses), stable, order, kth as a C# int / int[] / index array, np.lexsort's key list built from the operand.
      Every kth / keys overload was a SKIP line before. 79,710 calls at the parent, a second run identical, the "sets"
      group unchanged.
    - `a2942d35` (D4, a 0.70.0 behaviour) a null kth index array with a bad axis raised ArgumentNullException before the
      axis check; it now rides to the kth stage as the int[] form's null did, NumPy's order for kth=None (AxisError for
      `np.partition(a, None, axis=5)`, probed). The snapshot moved in exactly those 320 lines, each now its int[] twin's.
    - `2f52ffea` np.sort / np.argsort / np.partition / np.argpartition / np.lexsort and the ndarray methods sort /
      argsort / partition / argpartition: seven abstract members, `Sort`, `SortInPlace`, `ArgSort`, `Partition`,
      `PartitionInPlace`, `ArgPartition` in `TensorEngine.Methods.cs` and `LexSort` in `TensorEngine.Multiarray.cs`. The
      in-place pair carries ndarray.sort / ndarray.partition (PyArray_Sort / PyArray_Partition compute in place; np.sort
      / np.partition are copies around them), so every computing ndarray method reaches the engine. kth is one `object`
      (NumPy's array_like). DefaultEngine's overrides sit beside the facades (`Default.Sort.cs`, `Default.LexSort.cs`;
      they call np.asanyarray / np.argsort, R2) over the AxisSort / AxisPartition kernels, which stay. Average's private
      ArgSort became AverageArgSort. The 79,710 calls byte-identical, a 22-call null probe too; R1's np.lexsort.cs row
      went.
  - Found, open: the net10.0 oracle aborted once in three full runs at `8f3407eb` with an AccessViolationException in
    `DirectILKernelGenerator.NanReduceContiguousAxisDouble128` under `np.nansum` along an axis
    (`FuzzCorpusTests.SumIdentity`): the stack LT30 fixed by anchoring `DefaultEngine.NanSum`, which is still anchored
    (the IL guard passes). `SumIdentity` alone passes with and without guard pages, and net8.0 passed. With the operand
    rooted, the likelier cause is an early release of its block elsewhere; not reproduced yet.
  - Suites at `8f3407eb`, in a worktree without the OpenBLAS binaries staged: NumSharp.Tests net8.0 20,434 passed / 283
    skipped, net10.0 the same but for one wall-clock test the parallel session's load failed
    (`T2_NonZero_Is_Much_Slower_Than_NumPy`, 185 ms against its 32 ms bound; it passes alone); Oracle net8.0 379 / 35,
    net10.0 379 / 35 in two of three runs (the third is the access violation above). At `7408f707`, OpenBLAS staged:
    NumSharp.Tests 20,706 / 11 and Oracle 411 / 2, but for the two failures `6b887534` and `32478d0c` fixed. The E5
    commits: their targeted tests (532), the ratchets and the lifetime IL guard on both frameworks.
  - Next: the rest of E5 (`ReduceAMax` / `ReduceAMin` with `out=` / `initial=` / `where=`, the tuple-axis
    `ReduceMean` / `ReduceAdd`), E2 (`TensorEngine.Linalg`), E3 (`TensorEngine.Fft`); then P3. The parallel session
    took P2's last families, P4's fixes and P5.
- **2026-10-28, P4: the legacy polynomial functions are abstract `TensorEngine` members (`TensorEngine.Polynomial.cs`);
  their D4 fixes are open.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The
  referees: the snapshot (np.poly, np.polyval, np.polyint, np.polyadd, np.polysub, np.polymul, np.polydiv: 8,221 calls)
  with `DOTNET_TieredCompilation=0` under `NUMSHARP_DEBUG_GUARD_PAGES=1`, and `twin_poly.py` (each line rebuilt in
  NumPy, the outcome rendered the snapshot's way):
  - `0b55cdd4` the snapshot tool varies np.polyint's `m` (IntArg: 0, 1, 2, -1, 5) and `k` (None, a 0-d, one value, five,
    two, integers, text; the C# double of the convenience overload). The other functions were reached already
    (np.polydiv's `v` through np.searchsorted's Keys role, by its name). The twin agrees on 5,590 of 7,806 comparable
    lines and 17 are VF01 (414 decimal lines have no twin).
  - `39ae0ffb` the move: `TensorEngine.Poly`, `Polyval`, `Polyint`, `Polyadd`, `Polysub`, `Polymul` and `Polydiv`,
    abstract and documented in full; `DefaultEngine`'s overrides in `Polynomial/Default.Polynomial.cs` beside the
    facades, with `PolyAddSub`, `NaiveComplexScalarMultiply` and `PolyMinType`; the facades keep their boundary
    statements (np.polyval's matrix and chararray rules) and one call. 8,221 calls byte-identical; R1 drops the four
    rows (np.poly.cs 1 / 3, np.polyarithmetic.cs 3 / 1, np.polyint.cs 4 / 0, np.polyval.cs 5 / 0). `poly1d` is a class
    (Appendix D) and keeps its constructor's text line.
  - **Open, D4** (found by the twin, not fixed; each wants its own commit):
    - NumSharp measures a coefficient array by its SIZE where NumPy takes `len()` (its first axis): np.polyval iterates
      `p.size` elements (an N-D p raises IndexError where NumPy evaluates row by row and broadcasts), np.polyint divides
      by `arange(size, 0, -1)` (`(3,4) (12,)` in the broadcast error where NumPy says `(3,4) (3,)`), np.polyadd /
      np.polysub pad by the size difference (a (3,4) and a (4,3) pad nothing and subtract; NumPy pads one row and
      np.concatenate refuses the ranks), np.polydiv loops `size(u) - size(v) + 1` times (a (3,4) u with a 7-element v
      raises where NumPy returns q (1,) and r (3,4)). One root cause, about 1,900 twin lines.
    - np.polydiv stores `q[k:k+1] = atleast_1d(d)` where NumPy stores `q[k] = d`: a row d raises NumSharp's broadcast
      text where NumPy raises `setting an array element with a sequence.`.
    - np.poly(0-d): NumPy's `__array_function__` dispatcher returns the 0-d argument itself, which is not iterable
      (`TypeError: dispatcher for __array_function__ did not return an iterable`); NumSharp computes the one root.
  - Left otherwise: NumSharp's char class name in no-loop texts (the twin's uint16 proxy).
- **2026-10-28, P2's eighteenth family: np.einsum's contraction is the abstract `TensorEngine.Einsum`
  (`TensorEngine.EinsumFunc.cs`); three D4 fixes, the largest NumPy's iterator taking `out=` in.** Commits on `strings`,
  developed in a scratch worktree and landed by fast-forward. The referees: the snapshot (np.einsum's three overloads:
  37,183 calls) with `DOTNET_TieredCompilation=0` under `NUMSHARP_DEBUG_GUARD_PAGES=1`, and `twin_einsum.py` (each line
  rebuilt in NumPy, `numpy.einsum` called, the outcome rendered the snapshot's way; recorded with `SNAPSHOT_BYTES=1`, so
  a float result is compared by value when its bits differ):
  - `34d1d3fa` the snapshot tool reaches np.einsum's arguments: subscripts built from the operand's rank (views, sums, a
    transpose, the implicit output, a diagonal, ellipses, two- and three-operand contractions, an outer product, three
    refused forms), operand lists built from the operand (with a copy, an int32 ramp, U5 text, a 0-d, three of it, none)
    and optimize. Before any fix the twin agreed on 26,871 of 36,791 comparable lines (391 decimal lines have no NumPy
    twin).
  - `33bc9c03` the move: `TensorEngine.Einsum(subscripts, operands, out, dtype, order, casting, optimize)` abstract and
    documented in full in the new `Backends/TensorEngine.EinsumFunc.cs`; `LinearAlgebra/np.einsum.Contract.cs` renamed
    to `LinearAlgebra/Default.Einsum.cs` (`DefaultEngine`'s override, beside the facade: it calls np.*), the facade's
    argument checks moving in with it (a public entry point checks its own arguments); np.einsum keeps its overloads,
    the boundaries and one call on the first operand's engine (the default engine when there is none, as np.evaluate
    does). 37,183 calls byte-identical; R1 drops np.einsum.Contract.cs's row (1 text, 4 numeric lines); R2 unchanged.
  - `f8f34d6f` (D4) a StringDType `dtype=` request is text: the refusal asked only for the fixed-width classes, so the
    request reached the numeric path and NumSharp's own NotSupportedException. 162 lines change; 36 agree, the other 126
    are the next two commits' lines (with `out=` given, read-only or text) and NumPy's SystemError (the commit message
    counts 36 SystemError lines; 18 are, the other 18 are read-only `out=` lines).
  - `e0d7da1f` (D4) the optimize check is np.einsum_path's `path_type` resolution (`np.ResolveOptimize`, now internal),
    run right after the subscripts' null check, before the operand count and the order / casting converters (NumPy's
    einsum calls einsum_path first): a bare number raises `object of type 'int' has no len()`, a `(name, True)` memory
    limit and a two-element list are accepted (in np.einsum_path too: `IsNumber` takes a bool). 2,340 lines change;
    1,935 agree.
  - `7daa5e1c` repairs three `//` comment blocks and two doc tags a re-wrap script broke in the move (its message names
    the move by its pre-rebase hash `15aa02c5`).
  - `b4946bb2` (D4) `out=` joins the iterator, as in NumPy's `c_einsum` (one NpyIter over the operands and the out,
    `NPY_ITER_COMMON_DTYPE`, out READWRITE): a read-only out raises `operand array with iterator write flag set is
    read-only` first; the loop dtype is the common dtype of the operands and the out (int8 operands with an int64 out
    are summed in int64; they wrapped in int8); the out's read and write casts are checked after the operands', before
    anything is computed; the text refusal takes the out in. 2,505 lines change; 2,302 agree, the rest are NumPy's
    SystemError (198) and NumSharp's char class name (5).
  - After the fixes the twin reads: 30,969 agree, 3,309 VF01 (0-d results; a 0-d operand on the view path is a view, a
    text scalar renders with its own length), 69 float sums rounded in another order (NumSharp contracts through np.sum
    / np.matmul, NumPy's default einsum through its sum-of-products loop), 1,968 documented parser texts (with
    `optimize=` NumPy parses with its Python parser), 437 NumPy SystemErrors (its c_einsum loses the TypeError it set
    for a text `out=`), 33 NumPy views past an empty buffer (`get_single_op_view` skips its extent check when a label's
    first extent is 0; NumSharp refuses the diagonal), 1 NumPy StringDType out it cannot read back, 5 char class names.
  - Step 2 (C11): nothing to compose; the body already is NumPy's optimize= path (`bmm_einsum`) over the matrix
    products, so the move and the fixes change no kernel and there is nothing to benchmark.
  - Left: the out's extent check runs after the contraction (NumPy's broadcast error precedes the casts; a call with
    both a wrong extent and a refused cast reports the cast); an explicit path in `optimize=` is not validated against
    the operands (NumPy's einsum_path raises for an incomplete one); the documented parser difference.
- **2026-10-28, P2's seventeenth family: np.emath's nine functions are abstract `TensorEngine` members
  (`TensorEngine.Scimath.cs`, prefixed `Emath`), the first family under the revised D1.** Commits on `strings`,
  developed in a scratch worktree and landed by fast-forward. The owner resumed P2 and P4 beside PE and P3 ("Continue
  to PE+P3, ... parallely rest of P2+P4, ... Then P5"). The referees: the snapshot (the nine functions: 3,546 calls)
  with `DOTNET_TieredCompilation=0` under `NUMSHARP_DEBUG_GUARD_PAGES=1`, and `twin_emath.py` (each line rebuilt in
  NumPy, `numpy.emath` called, the outcome rendered the snapshot's way; a line that differs is re-run through NumPy's
  statements with NumSharp's library rules modelled one at a time):
  - `80cce839` the snapshot tool reaches np.emath: a target `np.emath.f` is every public instance member of
    `EmathModule`, called on the module object; power's exponent `p` is the second operand (logn's `x` already was).
    The twin agrees on 2,516 of the 3,349 comparable lines (197 decimal lines have no NumPy twin). The other 833 are
    library rules, none of them emath's own statements:
    - 317 VF01: a 0-d result is a writeable 0-d array where NumPy returns a read-only scalar.
    - 366 F1: NumPy's `_tocomplex` picks complex64 for single / byte / short / ubyte / ushort input; NumSharp has one
      complex type. The lines agree with `_tocomplex` forced to complex128.
    - 51 the branch's weak 0-d rule: logn's 0-d log result and power's 0-d operand are read as weak by the division /
      the power ufunc, where NumPy reads a 0-d array or scalar as strong. The lines agree with those operands passed as
      Python scalars.
    - 99 complex results: NumSharp's complex128 log / log10 / log2 / sqrt differ from NumPy's by one ULP on finite
      values (below), its complex arccos / arcsin / arctanh keep their documented envelope, and complex power is F5.
  - `73bbcc86` the move: `TensorEngine.EmathSqrt`, `EmathLog`, `EmathLog10`, `EmathLog2`, `EmathLogn(n, x)`,
    `EmathPower(x, p)`, `EmathArccos`, `EmathArcsin` and `EmathArctanh`, abstract and documented in full in the new
    `Backends/TensorEngine.Scimath.cs`; `DefaultEngine`'s overrides in `Math/Default.Scimath.cs` beside the facade (they
    call np.*), with the domain-fix helpers (`FixRealLtZero`, `FixRealAbsGt1`, `FixIntLtZero`, the text refusal) as its
    private statics. None of the moved names occurs elsewhere in NumSharp.Core, so the merge into `DefaultEngine`
    rebinds no call. np.emath keeps the signatures, the boundary statements and one call. 3,546 calls byte-identical;
    R1 drops np.emath.cs's row (3 text and 3 numeric lines); R2 unchanged.
  - No D4 commit: the twin found no divergence in emath's statements.
  - Step 2 (C11): nothing to change; each override is NumPy's statements over the np.* ufuncs, one per line, so there
    is no composition to try and nothing to benchmark.
  - **Found:** NumSharp's complex128 `np.log` / `np.sqrt` (and log10 / log2) differ from NumPy 2.4.2's by one ULP on
    ordinary finite values, contrary to the branch notes' "bit-for-bit" claim for the complex unary ufuncs:
    `np.log(3-3j).real` is 1.4451858789480825 against NumPy's 1.4451858789480823, `np.log(5+2j).real`
    1.683647914993237 against 1.6836479149932368, `np.sqrt(11-11j).real` 3.6439229674969837 against
    3.6439229674969833 (plain np.log gives the same bits as np.emath.log; a 1-element array and a 0-d array too).
    A ufunc matter for the complex unary work, outside this plan.
  - The suites ran on the move before its two rebases onto PE's commits: NumSharp.Tests net10.0 20,435 passed / 283
    skipped and net8.0 20,434 / 283, Oracle net10.0 and net8.0 379 / 35 skipped, 0 failed (the scratch worktree lacked
    the gitignored OpenBLAS runtimes, so the backend's tests were Inconclusive). After each rebase: the snapshot
    byte-identical, the emath tests (61) on both frameworks and the layering ratchets green.
  - Next in P2: np.einsum's contraction (`TensorEngine.EinsumFunc.cs`), then P4 (`TensorEngine.Polynomial.cs`), then
    the families whose partial files PE creates.
- **2026-10-28, D1 narrowed: the array object stays out of `TensorEngine`.** No code changed; this entry records the
  owner's decision and what it changes in the plan. The owner asked "Why do we move functions that are not math like
  np.copyto or np.frombuffer? ... is the storage not a set of in-memory classes?", read the split the P2 session
  proposed, and answered "Update our plan with suggestions updated. Borderline functions that need a decision should be
  decided by you".
  - **Found:**
    - Appendix D placed each function by how NumPy writes it (ufunc, C, fromnumeric wrapper, Python file), not by what
      it does. So of the 404 callables on `TensorEngine` or a sub-engine, 111 were functions of the array object
      (np.copyto, np.array, np.zeros, np.concatenate, np.reshape, np.take, np.stack, ...), 3 belonged to the dtype layer
      and 3 were no array function; 287 compute.
    - §1 and P3 already sent "values entering storage" to the `DType` slots, against Appendix D. np.frombuffer was
      outside the engine already (IO).
    - None of 18 array-object facades makes a `TensorEngine.` call: copyto, frombuffer, array, asarray, zeros, empty,
      full, ones, concatenate, take, put, putmask, place, where, repeat, choose, loadtxt, fromfile. They work on
      `UnmanagedStorage`, `Shape`, `NDIter` and the kernels.
    - In NumPy these functions are `src/multiarray`'s array object (64 files: `ctors.c`, `array_assign_array.c`,
      `array_coercion.c`, `shape.c`, `item_selection.c`, `mapping.c`, ...); np.concatenate copies each operand in with
      the same `PyArray_AssignArray` np.copyto calls (`multiarraymodule.c:512`, `:2006`). The computations are
      `src/umath` (42 files) and the Python numerics. NumPy extends per dtype (`DTypeMeta` slots, ArrayMethods) and per
      operand type (`__array_function__`, which np.copyto has: `_core/multiarray.py:1085`), and links BLAS / LAPACK at
      build time; nothing replaces its array object.
    - `TensorEngine` already holds ten array-object members, all shipped in 0.70.0: `GetStorage`, `CreateNDArray` ×2
      (no caller in src or tests), `Transpose`, `SwapAxes`, `MoveAxis`, `RollAxis`, `Cast`, `BooleanMask`,
      `BooleanMaskSet`. This plan added an eleventh, `Diagonal` (`199feaaf`, unreleased). `TensorEngine` creates storage
      (`GetStorage`, from NDArray's constructor) but holds no storage semantics.
  - **Decided** (the P2 session, as asked): the criterion and the borderline cases in §1's "Which O"; the ten shipped
    members stay as they are; `Diagonal` returns to the array object before 0.71.0 ships (E7); `InsertEngine` goes to
    the array object (E1).
  - **Applied to:** the State, the engine-layout paragraph, "Inside `TensorEngine` itself?", §1's table and "Which O",
    C3's consequence, D1 and D8 in §5 (and the old D1 recommendation's note), §6's steps 3 and 5, P2's target, PE (E1's
    table and rules, E4, E6, the new E7), P3, P6, "Later", §8, and Appendix D (the engine's table, the array object's
    table, the groups outside both).
  - **Counts after the split:** `TensorEngine` 287 callables (NumSharp has 272), the array object 111 (106), outside
    both 239 (208). D8's 61 split into 47 array object, 12 computations and 2 formatting helpers.
  - Next: unchanged. The owner picks the next work from P3 to P7; P3 is now the array object's phase.
- **2026-10-28, D1 decided again: the engine layout.** No code changed; this entry records the owner's decision and what
  it changes in the plan. The owner answered an evaluation of the twelve operation classes:
  - a sub-engine only where an outside library can provide the capability (`Blas`, `Linalg`, `Fft`);
  - `Linalg` takes the ten `_umath_linalg` members;
  - the FFT kernels are overridable members of `Fft`;
  - NumPy's Python functions sit in partial files named after NumPy's source files, as `TensorEngine` members that
    `DefaultEngine` implements.

  Applied to: §1's "Which O", the answer's "Inside `TensorEngine` itself?", C3's consequence, the D1 row of §5 (D8 and D9
  added, both open), §6's steps 3 and 5, P2's target, the new phase [PE](#pe-the-engine-layout), P4, P6, "Later", §8, and
  the new [Appendix D](#appendix-d-where-every-numpy-function-lives).
  - Measured: NumPy 2.4.2 has 637 public callables in numpy, numpy.linalg, numpy.fft, numpy.random and numpy.emath plus
    the ndarray methods; NumSharp has 586 of them by name. In the engine 404: NumPy's C core 183 (ufuncs 106, ndarray
    methods 45, C functions 32), its Python layer 172 in 18 partial files, the sub-engines 49 (Linalg 31, Fft 18).
    Outside it 233: classes and types, np.random, the ndarray methods themselves, printing and configuration, IO, the
    dtype routines.
  - Of the 172 Python-layer functions:
    - 44 already sit in an operation class, a `DefaultEngine` static or a `TensorEngine` member (PE moves them);
    - 65 wait in ratchet R1 (P2 to P5 move them, np.emath first);
    - 61 have facades without dtype logic (D8);
    - 2 are not in NumSharp (`fromfunction`, `row_stack`).
  - **Found:** all ten `_umath_linalg` members shipped in 0.70.0 as public virtual `TensorEngine` members; the evaluation
    the owner answered named four (Cholesky, Det, Inv, Solve). So E2 breaks all ten. `System.Math` is used 148 times in
    35 engine files, which rules out a member named `Math`.
  - Next: the owner picks the next work from P3 to P7; the rest of P2 stays deferred, as the owner told the P2 session
    on 2026-10-28. When P2 resumes, np.emath is the first family under the revised D1 (`TensorEngine.Scimath.cs`,
    members prefixed `Emath`).
- **2026-10-28, P2's sixteenth family: np.nditer's dtype code moved into the iterator engine and the DType layer
  (`NDIterRef.CurrentView`, `DTypePromotion.AdaptDescriptorToArray`); five D4 fixes on the way.** Commits on `strings`,
  developed in a scratch worktree and landed by fast-forward. D1: NumPy writes np.nditer in C, so its dtype code goes
  to K, not to a new operation class (none was created). The referees: the snapshot (np.nditer, both overloads: 82,986
  calls) with `DOTNET_TieredCompilation=0` under `NUMSHARP_DEBUG_GUARD_PAGES=1`, `twin_nditer.py` (each line rebuilt
  in NumPy, the iterator pulled to its end like Python's `for x in it`: per operand the views' dtype, the run-length
  list of their shapes, their C / F / OWNDATA / WRITEABLE flag sets and one SHA-256 over every step's bytes, or the
  exception's type and text; an allocated output's views compared without their bytes, which are uninitialized on both
  sides), `ab_nditer.py` (parent against fix, each changed line classified), and two setter probes for what the
  snapshot cannot reach (`it[i] = value`: 10 operand dtypes x 14 values x with and without `external_loop`, 280 cases;
  values that overlap the operand and edge values, 16 cases):
  - `40a4e96d` the snapshot tool reaches np.nditer's arguments (the operand list, every global and per-operand flag,
    op_dtypes with text and StringDType requests, op_axes, itershape, buffersize) and renders an iterator as one ITER
    summary. Before any fix the twin agreed on 52,638 of the 82,365 comparable lines (621 decimal lines have no NumPy
    twin): 29,099 lines differed only in the views' OWNDATA flag, 556 in the reference check's place, 72 in the
    argument converters' order (the first count also took 545 lines whose allocated output hashed uninitialized
    memory and 16 lines of NumSharp's `dtype(char)` text, both twin gaps, closed in the twin).
  - `ac37ef1e` (D4) the 0-d element views do not own their data (`flags.owndata` False, the iterator NumPy's base):
    `AliasView<T>` / `AliasStringView` wrapped them as owners. 29,409 lines changed, each only in the OWNDATA letter.
  - `01388c87` (D4) flags, order and casting are converted before the operand count (NumPy's argument parser runs
    them before `npyiter_prepare_ops`): 72 lines.
  - `70f18167` (D4) the reference check runs per operand where `npyiter_prepare_one_operand` runs it (after the
    write flag, the size and the descriptor adaptation, before the shape pass and the common dtype), and skips an
    output the iterator allocates (a StringDType output needs no `refs_ok`; it was refused):
    `NDIterRef.ThrowIfReferenceOperandUnsupported`, called by np.nditer's PrepareOperands and by the engine's own
    loop. 556 lines; the twin agrees on all 82,365.
  - `997dc839` (D4) the setter stores through the storage's own store, NumPy's `PyArray_CopyObject`: a 0-d array is
    cast (`np.array(300)` into int8 stores 44; it was packed as a Python int and raised), null is Python's None in a
    numeric operand (it raised `ArgumentNullException`). The hand-written store predated the C# literal mark of plan
    X27, which `UnmanagedStorage.SetData` reads. The setter probe: 262 of 280 cases agreed before, 280 after; the
    overlap probe 16 of 16 before and after. Snapshot unchanged (it does not reach the setter).
  - `5991bf98` the nditer_protocol oracle replay builds its Python int with C#'s implicit conversion (a marked
    literal): it used `NDArray.Scalar`, a 0-d array since X27, and passed only because the old setter packed every 0-d
    value; 3 of the tier's 2,987 cases went red at `997dc839` on both frameworks.
  - `ecab6bbe` the move: the views (`CurrentView` and its helpers: the 15-lane switch, the string views, the string
    descriptor and allocator of the memory a pointer is on) into `Backends/Iterators/NDIter.OperandViews.cs`, and
    `PyArray_AdaptDescriptorToArray` into `DTypePromotion`, now the one copy np.nditer, astype (`NDStringCasting`) and
    numpy.ma's astype share. There were two copies, np.nditer's private one and `NDStringCasting`'s, which numpy.ma's
    astype called (the commit message says each caller had its own). 82,986 calls byte-identical; R1 drops
    np.nditer.cs's row (10 text and 15 numeric lines); R2 unchanged.
  - `990eb14c` (D4) a write flag cleared on an iterator view cannot be set again (NumPy's view has the iterator for
    its base, which exports no buffer; NumSharp set it again on a written operand's view): `CurrentView` marks each
    view's storage `WriteProtected`, the flag `setflags` consults and nothing else reads. A sub-view of a 0-d view
    refuses too, where NumPy's (its base is the writeable 0-d view) may be set again: NumSharp flattens a view's base
    to the owning storage. Snapshot unchanged.
  - Step 2 (C11): nothing to compose (NumPy writes np.nditer in C). The family's commits against the base `40a4e96d`
    (`bench_nditer.cs`, three interleaved rounds of each build pinned to two P-cores, best of rounds, microseconds,
    measured at `ecab6bbe`): the per-step pulls within the base's spread (float64 over 1K elements 232.8 -> 247.2, the
    base's own rounds 232.8-251.6; 100K, a pair, U2 and StringDType +0.5 % to +3.9 %), `external_loop` and the
    construction unchanged (one operand 0.348 -> 0.339, NPY/NS x1.47 -> x1.51; three with an allocated output 1.155 ->
    1.207, x0.84 -> x0.80; StringDType 0.448 -> 0.451, x1.75 -> x1.73). The setter costs more: +18 % with a C# literal
    (1,185 -> 1,397 per 1,000 stores, NPY/NS x0.37 -> x0.31) and +30 % with a 0-d array (792 -> 1,033, x0.36 ->
    x0.28), the price of the shared store (`PackForStore`'s 0-d temporary, `GetData`'s view and a boxed fill per call)
    over the hand-written pack it replaced; the store's fixed cost belongs to P3 (values entering storage). The
    per-step cells stay at the per-view floor the branch notes record (x0.10-x0.32).
  - **Found:**
    - R1 counts patterns, not decisions: `Storage.IsVString` in the string-descriptor helper was never counted. The
      helper moved with the counted lines it serves.
    - The oracle adapter's comment said "NumPy's Python int" while its code built a NumPy scalar: a harness that
      matched the old, wrong setter can hide that setter's divergence. Replay helpers built before X27 may hold more of
      these (`NDArray.Scalar` standing for a Python literal).
    - The owner's 2026-10-27 position that NumPy-level operation classes should be overridable engines (memory
      `engines-overridable-evaluation`): this family needed no class, but the two families landed before it today
      (`UniqueEngine`, `ConvolveEngine`) have the shape the owner questioned; their place waits on that decision.
  - Left: np.nested_iters' own argument order ("Must provide at least one operand" is an ArgumentException there, and
    its converters were not checked); the sub-view residual of `990eb14c`; the setter's cost (P3).
  - At `990eb14c`: NumSharp.Tests net10.0 20,707 passed / 11 skipped and net8.0 20,706 / 11; Oracle net10.0 and
    net8.0 412 / 2 skipped, 0 failed.
  - Next in P2: np.emath, then np.sinc, np.digitize and the files of two lines or fewer (the K3 list's order).
- **2026-10-28, P2's fifteenth family: np.convolve / np.correlate moved to their layers (`TensorEngine.Correlate` /
  `Correlate2`, `ConvolveEngine`, the kernels as `SlidingDot`), their mode NumPy's converter, the swapped product
  reversed in place.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees:
  the snapshot (np.convolve, np.correlate, NDArray.convolve, NDArray.correlate: 44,928 calls) with `SNAPSHOT_BYTES=1`
  and `DOTNET_TieredCompilation=0` under `NUMSHARP_DEBUG_GUARD_PAGES=1`, `twin_sliding.py` (each line rebuilt in NumPy:
  dtype, shape, C / F / OWNDATA / WRITEABLE flags and SHA-256, or the exception's type and text; NaN bits a verdict of
  their own), `ab_lines.py` (parent against fix, the changed lines grouped by function, mode variant and outcome), and a
  NumPy probe of the mode converter (28 values through both functions, the order against empty, 0-d and 2-D operands):
  - `3fba426d` the snapshot tool reaches np.convolve / np.correlate's `v` (a kernel role: the operand's copy, int32 /
    float64 / short / one-element / one-element reversed / long / special-value / bool / text / 0-d / 2-D / empty /
    strided / reversed kernels) and their `mode` (the near misses `Full`, `v`, `same `, an unknown `x`, a quote, the
    empty text and null), on 1-D pools with a 24-element layout: 44,928 calls (5,976 before, every 2-D operand stopping
    at NumPy's depth check). Before any change the twin agreed on 36,744 of the 43,776 comparable lines (1,152 decimal
    lines have no NumPy twin).
  - `1f304ebc` the move (D1: NumPy writes convolve in Python and correlate in C): `multiarray.correlate` /
    `correlate2` become the abstract `TensorEngine.Correlate` / `Correlate2` (DefaultEngine's `CorrelateNumPy`:
    `PyArray_Correlate` / `PyArray_Correlate2` with `_pyarray_correlate`'s checks, in
    `Backends/Default/Math/Default.Correlate.cs`), np.convolve's Python statements the operation class
    `ConvolveEngine` (`Math/ConvolveEngine.cs`), and the sliding kernels that were NDArray members the class
    `SlidingDot` (`Backends/Default/Math/Sliding/SlidingDot{,.Long}.cs`). 44,928 calls byte-identical at
    `TieredCompilation=0`; R1 drops the family's three rows (8 text and 20 numeric lines); R2 unchanged.
  - `cf50c452` (D4) the mode is NumPy's `PyArray_CorrelatemodeConverter`: ValueError with NumPy's texts (it was
    ArgumentException, with .NET's ` (Parameter 'mode')` suffix), the bad argument in Python's repr (`(got "it's")`; it
    was `(got 'it's')`), a null mode, NumPy's None, TypeError `convolve/correlate mode not understood` (it was a
    NullReferenceException), a lone surrogate CPython's UnicodeEncodeError before the parser. 22,652 snapshot lines
    changed, every one a mode-error line; the twin's mismatches 7,032 -> 628.
  - `25b193e6` step 2 (C11): a swapped np.correlate product is reversed in its own buffer (NumPy's `_pyarray_revert`; it
    took a reversed view and a copy: 1,066 ns of a 1,641 ns call over eight elements) and a one-dimensional operand view
    is copied by one strided move (`CopyOneDimensional`: the general copy spent ~300 ns of setup beyond the allocation
    on np.convolve's three-element reversed kernel). The snapshot byte-identical; NPY/NS np.correlate(3, 8 values)
    x0.57 -> x1.87, np.convolve(8, 3) x1.08 -> x1.33, np.convolve(int32 1000, 10, 'same') x3.76 -> x4.34,
    np.convolve(float64 1000, 10) x1.96 -> x2.24. The move itself measured within +3.4 % of the base on every cell
    but the 8 x 3 correlate, +5.5 % (+26 ns, inside that cell's 0.47-0.56 us spread across eight base runs; the
    commit message's "+-3 %" undercounts it).
  - Evidence at `25b193e6`: the twin agrees on 43,148 of the 43,776 comparable lines. The 628 left: 414 lines whose
    result holds another NaN's bits than NumPy's (318 of them over the special-value kernel, the rest over the
    special-value pool: where NaNs and infinities meet in a sum of products, which NaN survives is the compiled loop's
    operand order, the class the branch notes record for float add / multiply), and 214 lines where NumPy computes a
    StringDType product as Python objects (an object array: NotSupportedException, documented).
  - **Found:**
    - At the default tiering two runs of ONE build differ in five complex lines of NDArray.correlate (NaN signs): the
      hand-written complex dot kernels give one NaN sign as tier-0 code and another optimized, and the optimized one is
      NumPy's. This family's A/Bs run with tiering off; the tier-0 class is open.
    - The twin hung for 27 minutes inside NumPy: its OBJECT fallback beside a StringDType operand computes
      `4294967295 * "x"` for a uint32 operand (gigabyte strings). The twin now tames the numeric operand
      (`np.ones_like`): the answer is an object array either way.
    - NDArray.copy of a short strided one-dimensional view spends ~300 ns of setup beyond the allocation (775 ns for
      three float64 values, a bare allocation 439): library-wide; the sliding kernels now bypass it.
    - A facade whose body moved keeps `[NDScoped]`: `OperandRootsTests` (gap LT22) wants the weaver's operand roots at
      NDArray.convolve / correlate's public entry, and the scope one call deeper does not satisfy it (both failed until
      the scopes came back).
    - NumPy also takes the integers 0 / 1 / 2 as the mode (`integer convolve/correlate mode must be 0, 1, or 2`
      otherwise); NumSharp's mode parameters are strings, so that branch has no counterpart (documented).
    - Harness traps: a background `cmd //c x.cmd` from Git Bash, and a nested `cmd /c y.cmd` inside a batch file, do not
      resolve a relative script name (the nested cmd starts in another directory), so the runners are named by absolute
      path (`%~dp0`); Python's `-I` ignores `PYTHONIOENCODING`, so a probe printing non-cp1252 text prints `ascii()`; a
      mutated file restored with its old timestamp (`shutil.copy2`) is older than the mutant's build output, so MSBuild
      skips the rebuild until the file is touched.
  - Left: the 414 NaN-bit lines and the tier-0 NaN signs; np.convolve over 8 x 3 values at x1.33 (the reversed view,
    then the kernel and the result at the NDArray allocation floor).
  - At `25b193e6`: NumSharp.Tests net10.0 20,703 passed / 11 skipped and net8.0 20,702 / 11; Oracle net10.0 and net8.0
    412 / 2 skipped, 0 failed.
  - Next in P2: np.nditer (`APIs/np.nditer.cs`), then np.emath, np.sinc, np.digitize and the files of two lines or
    fewer (the K3 list's order).
- **2026-10-28, P2's fourteenth family: np.unique moved into `UniqueEngine`, its flat routes NumPy's `_unique1d`
  statements; np.argsort's float16 / complex128 radix on the way.** Commits on `strings`, developed in a scratch
  worktree and landed by fast-forward. The referees: the snapshot (np.unique, NDArray.unique's two members,
  np.unique_values / unique_counts / unique_inverse / unique_all and the four set routines, which run np.unique's body
  over their concatenation: 13,728 calls) run under `NUMSHARP_DEBUG_GUARD_PAGES=1`, `twin_unique.py` (each line rebuilt
  in NumPy: dtype, shape, C / F / OWNDATA / WRITEABLE flags and SHA-256 of every array of a result struct, or the
  exception's type and text; NumPy's values-only hash order recognized as such), `ab_unique.py` (parent against fix,
  each changed line classified by the twin), and NumPy probes for what the snapshot cannot reach (its pool holds no
  offset view and no tie that two routes resolve differently), compared by raw bits and written into the unit tests:
  signed zeros (14 lines x 20 routes), NaN bits (6 inputs x 16 calls), complex NaN classes (3 lines x 16 flag
  combinations), complex NaN-part order (8 inputs x 6 routes):
  - `40806f6c` the snapshot tool renders the arrays of np.unique's result structs (`STRUCT Name(prop=...)` over their
    NDArray properties: 2,326 of the family's 6,570 lines were an opaque `VAL <type>`) and reaches the set routines'
    second operand `ar2`. 13,728 calls. Before any fix the twin agreed on 12,507 of the 13,144 comparable lines (584
    decimal lines have no NumPy twin).
  - `7dbd3ef5` the move (D1: NumPy writes unique in Python): np.unique's statements after its class boundaries, the
    instance members' routing and text checks, the Array-API bodies, and the sort + mask, hash and radix kernels that
    were NDArray instance members, into the internal operation class `UniqueEngine`
    (`Manipulation/UniqueEngine{,.Kwargs, .Hash,.Radix,.Text,.VString}.cs`). 13,728 calls byte-identical; R1 drops the
    family's six rows (7 text and 93 numeric lines).
  - `d8562466` (D4) with an axis, a C-contiguous view at an offset (an np.split / np.array_split / np.unstack child) is
    read from its own first element: the row path's slab helpers read the storage's base, so `np.unique(np.split(a,
    3)[1], axis=0)` returned the parent's first rows with their counts and indices (silent wrong values); the
    long-indexed complex fallback had the same read.
  - `5364e93e` (D4) the values with an axis are NumPy's view at axis 0 too (`reshape_uniq`'s `np.moveaxis`, so
    `flags.owndata` is False; the numeric row path returned the gathered owner). 312 lines fixed.
  - `86f0f944` (D4) the values-only and counts-only routes of float32 / float64 write NumPy's sort NaN, the positive
    quiet NaN, for two elements or more (NumPy's in-place `ar.sort()`); the `[OpenBugs]` pin of
    `OpenBugs.SortUniqueNaN.cs` passes and stays as a regression test.
  - `80c3b6a9` np.argsort of float16 and complex128 sorts by radix (they ran `Array.Sort` over an index array with a
    class comparer): a 16-bit key with both zeros folded and the NaNs appended in input order, and a stable partition by
    complex NaN class with per-class 64-bit keys. The same indices: 10,562 random lines against the replaced kernel,
    1,200 NumPy cases, 6 planted bugs killed. NPY/NS float16 x1.58-x15.6 (x0.40-x1.02), complex128 x0.91-x4.97
    (x0.27-x0.61). It lands before the next fix because that fix, on the comparison argsort, ran float16 2.2x and
    complex128 1.2x slower than the partition route it replaces.
  - `fa98ba8a` (D4) the index and inverse routes over float16 / float32 / float64 / complex128 (and float16 / complex128
    counts) are NumPy's argsort-and-gather statements over that stable order: a collapsed NaN run's value and index are
    the same NaN's, a run of zeros keeps the first zero in input order (it kept `-0.0`), complex NaN classes come in
    `CDOUBLE_LT`'s order. Probes: complex NaN classes 30 of 48 lines differed, then 0; signed zeros 98 of 280, then 42
    (each on the values-only / counts-only routes: NumPy's x86-simd-sort orders equal keys its own way); NaN bits 64 of
    96, then 2 (float16 values alone, closed by `6871394e`). NPY/NS at 100K: float16 `return_index` x1.16 -> x8.56,
    complex128 x0.72 -> x3.35.
  - `bed773f6` (D4) the long-indexed fallback (above `Array.MaxLength` elements) runs the same statements on the same
    routes (it kept the unstable partition and an introsort, and for complex the smallest position of a collapsed NaN
    run as its index); tested through reflection on small lines.
  - `b3403dd4` (D4) complex values with a NaN part come out in NumPy's sort order without equal_nan: np.unique takes the
    hash path for both settings and sorts (it partitioned them in partition order); np.unique_counts / np.unique_values
    sort them the same way; above 2^28 elements the argsort route gives the hash path's answer (under equal_nan the
    collapsed class is its first value in C order). Probe: 18 of 48 lines differed, then 2 (equal NaN-part values
    below).
  - `a267ee2a` step 2 (C11): the complex hash path sorts its distinct values by radix from 2,048 on, and a line of
    32,768 elements or more whose first sixteenth is over 90 % distinct goes to the argsort route (the hash costs ~15 ns
    per element plus ~110 ns per distinct value, the argsort route 50-70 ns: they break even near 0.3 n distinct). The
    insert loop is split at the checkpoint: a test inside it cost 10 us at 1K. np.unique_counts(complex) at 100K x0.63
    -> x1.82; a cardinality sweep at 100K / 1M x1.59-x4.74 (the hash alone fell to x1.07).
  - `6871394e` step 2: float16 values alone take the stable argsort route too, which is NumPy's answer where NumPy's
    float16 in-place sort keeps equal values in input order (the scalar `quicksort_<half_tag>` on hosts without
    AVX512_SPR / ICL: an insertion sort for lines of up to 16 elements), and NPY/NS x0.95 -> x5.49 at 100K. The NaN-bits
    probe matches on all 96 lines.
  - Evidence at `6871394e`: the twin agrees on 12,819 of the 13,144 comparable lines (12,507 at `40806f6c`). The 325
    left: 264 lines in NumPy's hash order (values alone over integers, bool, complex and text with `sorted=False`, and
    np.unique_values: the documented divergence, NumSharp sorts); 25 char lines whose `DTypePromotionError` against a
    StringDType `ar2` names NumSharp's own `CharDType` (the twin's uint16 proxy names `UInt16DType`); and 36 lines of
    NumPy's order between equal values over the snapshot's operands with both zeros, float32 / float64 values alone and
    counts alone (x86-simd-sort, 26 lines) and float16 lines of 24 and 48 elements (NumPy's introsort, 10 lines: values,
    counts and an inverse alike). The A/B of the last two commits, which were not snapshotted one by one, changed 2
    lines, both np.union1d over a float16 operand with both zeros: one is now NumPy's (24 elements; NumPy keeps the
    first zero), one moved into the introsort's tie order (48 elements; NumPy keeps `+0.0`, the stable order the first
    zero, `-0.0`; np.unique of the same 24 values already differed so at `40806f6c`).
  - **Found:**
    - NumPy's float16 in-place sort and quicksort argsort are its scalar introsort on a host without AVX512_SPR / ICL
      (`numpy/_core/meson.build` builds the 16-bit SIMD sort for those only), and complex128's on every host: stable up
      to 16 elements, its own tie order past them. Only `return_index` asks NumPy for mergesort, so an inverse without
      an index meets that order too. Probed for complex128 (`probe_czero_long.py`, 840 line and route pairs): over
      random lines of 17-100 elements with zeros of both signs, NumPy's counts and inverse routes keep another zero than
      the first in input order in 7-14 of 20 lines; its values alone (the hash) and `return_index` never do, nor any
      route up to 16 elements. NumSharp keeps the first zero on every complex route (the stable argsort and the hash
      alike).
    - NumPy's float32 / float64 in-place sort is x86-simd-sort's (AVX2 here): which zero represents a run of zeros is
      CPU-dependent, and the values-only routes keep the radix's `-0.0`.
    - The order of equal NaN-part complex values on NumPy's hash path (`nan+0j` and `nan-0j`) is MSVC's
      `std::unordered_set` iteration order (NumPy hashes `-0.0` as `+0.0`), not reproduced; NumSharp keeps input order.
    - The snapshot proves the move and `5364e93e`; `d8562466`, `86f0f944`, `fa98ba8a`, `bed773f6` and `b3403dd4` rest on
      the probes and unit tests (each fails on its parent).
    - A span sort with a struct comparer was slower than the delegate sort it would replace (98 us at 1,000 values, 21
      ms at 80,000).
    - The legacy protected `NDArray.unique<T>()` has no caller; its doc was malformed (CS1570), and its float16 NaN
      sorts first (`Half.CompareTo`): the doc says so now (probed).
    - The branch CLAUDE.md pointed at `docs/UNIQUE_DESIGN.md`, which is in `docs/stale-docs/` since 2026-09-04 (fixed);
      eight other design docs moved there are still named by their old paths.
  - Left: float32 / float64 values alone and counts alone, x0.51-x0.86 NumPy (its SIMD sort against NumSharp's radix);
    complex128 np.unique_counts at 1K, x0.91 (NumPy's sort at its fixed cost); the long-indexed routes are tested only
    through reflection on small lines; NDW012 at eight sites of `UniqueEngine.Text.cs`, moved as they were.
  - At `6871394e`: NumSharp.Tests net10.0 20,692 passed / 11 skipped and net8.0 20,691 / 11; Oracle net10.0 and net8.0
    412 / 2 skipped, 0 failed; the carved-gap replays find every carved case of their 82 gaps still diverging.
  - Next in P2: np.convolve / np.correlate (`Math/NDArray.Convolve.cs` 4, `Math/NDArray.Correlate.cs` 4 and 1).
- **2026-10-27, P2's thirteenth family: np.logical_and / or / xor / not moved into `TensorEngine.LogicalAnd` /
  `LogicalOr` / `LogicalXor` / `LogicalNot`; the ufunc machinery's NumPy orders and layouts fixed on the way; one pass
  for the common call.** Commits on `strings`, developed in a scratch worktree and landed by fast-forward. The referees:
  the snapshot (the four functions, 39,048 calls) run under `NUMSHARP_DEBUG_GUARD_PAGES=1` (every buffer comes fresh and
  zeroed, so the masked-off slots of a where= result are zeros on both sides: without it the pool handed out other
  garbage once the move dropped the facades' scopes, and 2,202 where= lines differed by those slots alone),
  `twin_logical.py` (each line rebuilt in NumPy: dtype, shape, C / F / OWNDATA / WRITEABLE flags and SHA-256, or the
  exception's type and text; NumPy's masked-off slots zeroed too), `ab_logical.py` (parent against fix, each changed line
  classified by the twin), and a ufunc matrix (`ufunc_matrix.py` / `.cs`: 55 ufuncs x 14 cases, 738 cells, each call
  bound by reflection to the ufunc's NumPy-shaped overload: a read-only out= / a non-bool where= / a refused dtype= in
  every pair, broadcast failures with out= and where=, a 0-d result's flags, a where= result's layout over F operands):
  - `780ba0a7` the move (D1: NumPy's logical ufuncs are C loops): four abstract TensorEngine members, implemented in
    `Backends/Default/Logic/Default.Logical.cs` statement for statement; the facades keep the numpy.matrix and
    numpy.char.chararray boundaries. 39,048 calls byte-identical under guard pages; R1 drops `Logic/np.logical.cs 4 3`.
    Before any fix the twin agreed on 34,459 of the 38,436 comparable lines (612 decimal lines have no NumPy twin).
  - `e39ad6a6` (D4) logical_not answers a text operand under dtype= / out= / where= as NumPy does: the parse errors
    first, then a fixed-width operand under a dtype other than bool finds no loop (logical_not's loops are bool-output
    only), under dtype=bool fails the input cast, and a StringDType operand's no-loop text names the requested class. 522
    lines fixed.
  - `c04e16ec` (D4) logical_not's broadcast error lists only its own operands: the comparison computing a numeric NOT
    listed its hidden 0-d zero ("(3,4) () (3,4,2) "); the operand is joined with out= / where= first, after NumPy's parse
    checks. 184 lines fixed. Its first form joined before the parse checks; the A/B showed 20 regressions (an int64 mask
    reported after the broadcast) before it landed.
  - `1006659f` (D4, library; plan api-parameter-parity X22) a ufunc's shape errors with out= / where= list every operand
    and come in NumPy's two phases (`npyiter_fill_axisdata`): the joint broadcast of the inputs, out= and where=, then
    out's no-stretch check against the joint shape (`ResolveUfuncIterationShape`, `BroadcastUfuncInputs`). The binary /
    comparison / power routes listed the inputs only, the out-join failure left where= out, and the stretch was checked
    before the mask joined. 162 matrix cells and 1,104 lines fixed.
  - `73e80d21` (D4, library) every ufunc parses out= and where= before it resolves its loop: `UfuncLoopCode` runs the
    read-only and mask checks after the dtype= class check for every operand (they ran for text only), so they come ahead
    of a no-loop dtype= and an input cast. 126 matrix cells and 777 lines fixed; no exception text of the family differs
    from NumPy's any more.
  - `00da3f93` (D4, library) a fresh where= result takes NumPy's iterator layout, the mask voting (`WhereResultShape`
    over `Shape.NpyIterOutputShape`): F operands under an F mask, a row or a column give an F result, a permuted operand's
    order is kept; the three out= / where= routes and np.power's two where= routes allocated C. 108 matrix cells and 28
    lines fixed (2 decimal).
  - `e41c90dc` step 2 (C11): NumPy's loops are one C pass each; the composition made three passes, two bool temporaries
    and a 0-d zero (logical_and / or / xor at 1,000 numeric elements x0.53-x0.91 NumPy, logical_not x1.03-x1.43). One-pass
    routes in front of it (`TryLogicalRun` /
    `TryLogicalNotRun`): per 4,096-element block the comparison engine's `x != 0` (`x == 0`) run writes the truth values
    on the stack and one `Vector<byte>` AND / OR / XOR combines them, for contiguous operands of one shape (or one of one
    value), no mask, and no out= or a bool out= of that layout sharing no memory. The same bytes, layout and result type
    by construction (a bool operand's truth included: the bitwise ufunc reads a nonzero byte as True); the snapshot
    byte-identical, 11 planted bugs killed. NPY/NS over 72 cells (float64 / int32 / bool x 1K / 100K / 10M, plain and
    out=): geomean x3.04 (x2.02), min x0.79 (x0.53), 21 cells under x1.5 (39); numeric 1K x1.81-x2.85 plain and x4.0-x10.9
    into an out=.
  - Evidence at `e41c90dc`: the twin agrees with NumPy 2.4.2 on 37,074 of the 38,436 comparable lines (34,459 at
    `780ba0a7`); the 1,362 left are each a 0-d result's WRITEABLE flag (VF01, below). The ufunc matrix: 684 of 738 cells
    NumPy's (288 at `c04e16ec`), the 54 left VF01's.
  - **Found:**
    - ldexp, divmod, modf and frexp handle out= themselves and keep their own shape texts and parse order: ldexp's input
      broadcast lists the inputs only, divmod composes floor_divide and remainder (a failure lists one output; out=(o,
      read_only) computes the quotient before the remainder's refusal), ldexp and frexp report an out= that does not fit
      as np.copyto's text, and all four check a non-bool mask before a read-only out (`probe_multiout.py`). Plan
      api-parameter-parity X39, open.
    - VF01 reaches every elementwise ufunc: a 0-d result is a writeable 0-d array where NumPy returns a read-only scalar
      (the matrix's 54 cells; recorded since the gufunc family).
    - NumPy's logical_or over float64 reads its operands faster at 10M (7.5-9.1 ms for 170 MB) than the blocked route
      (9.3-9.4 ms whatever the block, 512 to 8,192 elements measured): the two cells under x1.0 (x0.79 into an out=, x0.97
      plain).
    - A helper inserted in front of a member's signature lands between the member's doc comment and the signature (CS1572
      / CS1571 caught it in `00da3f93` before it landed): insert in front of the doc comment.
  - Left: X39; VF01 for the elementwise ufuncs; the bool-bool cells and logical_not of bool stay on the composition's one
    SIMD pass (x1.14-x1.44 at 1K / 100K, x1.0-x1.07 into an out= at 10M, memory-bound on both sides).
  - At `e41c90dc`: NumSharp.Tests net10.0 20,669 passed / 11 skipped and net8.0 20,668 / 11; Oracle net10.0 and net8.0
    412 / 2 skipped, 0 failed; the carved-gap replays find every carved case of their 82 gaps still diverging.
  - Next in P2: np.unique's text route (`Manipulation/NDArray.unique.cs` 4), then np.convolve / np.correlate
    (`Math/NDArray.Convolve.cs` 4, `Math/NDArray.Correlate.cs` 4).
- **2026-10-27, P2's twelfth family: np.average moved into `AverageEngine`, its statements NumPy's; the np.sum and
  np.evaluate residuals it met fixed on the way.** Commits on `strings`, developed in a scratch worktree and landed by
  fast-forward. The referees: the snapshot (np.average / np.average_returned, 33,096 calls), `twin_average.py` (the
  tool's operands, weights and axis tuples rebuilt in NumPy: dtype, shape, C / F / OWNDATA / WRITEABLE flags and SHA-256
  per result, or the exception's type and text), `ab_average.py` (parent against fix, each changed line classified by
  the twin), a random differential (`rand_avg_gen.py` / `rand_avg_check.cs`: 3,000 NumPy 2.4.2 cases over random
  layouts, axes, weights and dtypes, raw bytes and flags compared), and NumPy probes per fix, written into the unit
  tests:
  - `2051e724` the snapshot tool reaches np.average's weights and axis tuples (two roles given only to np.average /
    np.average_returned's parameters: weights of the operand's shape, along one axis or two in either order, a 0-d, zero
    sums, NaN, a broadcast view, other dtypes and text, the shapes NumPy refuses; axis tuples unsorted, empty, repeated,
    out of range). 33,096 calls (16,140 before). Before any fix the twin agreed on 23,129 of the 32,404 comparable lines
    (692 decimal lines have no NumPy twin).
  - `e3d51474` the move (D1: NumPy writes average in Python, `numpy/lib/_function_base_impl.py`): the body and its helpers
    into `Statistics/AverageEngine.cs`, statement for statement; the facades keep the numpy.matrix boundary and the null
    check. 33,096 calls byte-identical; R1 drops `Statistics/np.average.cs 5 44`.
  - `35df73cc` (D4, memory safety) NDIterRef's drivers run no inner loop over an empty iteration, as NumPy's callers test
    `NpyIter_GetIterSize` first: an empty iteration whose empty axis is not the innermost still reported the innermost
    extent as its inner count, so `np.average(np.zeros((11, 0, 14)), axis=2, weights=...)` read and wrote 14 elements past
    its empty buffers through the fused weighted-sum kernel (an access violation, or a corrupted heap that ended the
    process later with exit code 127). Found by the random differential, which died at a random later case. ForEach,
    ExecuteGeneric, ExecuteReducing, BufferedReduce, ForEachRows and the 2-D block routes; fenced-memory tests.
  - `79575624` (D4) `_weights_are_valid`'s own types: TypeError without an axis, ValueError along one (both were
    ArgumentExceptions), and an empty axis tuple is an axis given on a StringDType operand too. 2,934 lines fixed.
  - `9ea8bc83` (D4) the result dtype is NumPy's `result_type` over the descriptors, so text weights get NumPy's refusals
    (a sized string the weight sum's dtype= refuses, a DTypePromotionError beside StringDType) where the numeric table
    raised NotSupportedException. 1,308 lines fixed; 20 char lines name NumSharp's own `CharDType` (NumPy's algorithm
    for a user dtype; the twin's uint16 proxy names NumPy's classes).
  - `a2e9fe85` (D4) an axis tuple of two axes or more, and the empty tuple, is ONE add.reduce over the whole set:
    `DefaultEngine.ReduceAddOverAxes` / `MeanOverAxes` run the port of NumPy's reduction iterator (`NumPyReduceIterator`)
    with NumPy's add loops (`NumPyAddLoops`); the mean of means and the empty tuple's own operand are gone. 423 lines
    fixed.
  - `f25aa817` (D4) np.average_returned's count without weights is NumPy's `avg.dtype.type(a.size / avg.size)`: a scalar
    beside a scalar mean, a fresh array only beside a mean with dimensions. 492 lines fixed.
  - `da7adc53` (D4, library) np.sum and np.mean over an operand with a reversed or a broadcast axis run NumPy's reduction
    iterator: step 2's composition summed views on those layouts, where np.sum still had four residuals (plan
    api-parameter-parity finding 5: a reversed axis walked in memory order; finding 10: a broadcast operand reduced flat
    in the fold's own order; X12: a broadcast operand along an axis; X14: the broadcast flat mean ignoring dtype=,
    `[OpenBugs]` groups 10 to 13). ReduceAdd sends an inexact loop over such an operand to `ReduceAddThroughIterator`;
    the add loops gained vector forms (contiguous, broadcast value with NumPy's region NaN rule, reversed), a several-rows
    form, NumPy's pairwise sum of n copies of one value, HALF_add through the house SIMD kernel in place, and the
    iterator's buffer fill of a stride-0 source. The oracle's finding-5 excuses (mean / ma.mean / ma.anom) and the
    generators' carves go: mean_loops 6,590 cases (+40), ma_meanvar 26,544 (+165), the relayout tier regenerated (15,117:
    it had not been since `67293fac`, so the regeneration also met the source tiers' drift since). NPY/NS 2.03x geomean
    over 47 cells; the one cell under parity is float64 (1000, 1000) reversed rows along axis 0, 0.88x-0.94x, bound by
    memory on both sides. Random differential: 33 cases better, 0 worse.
  - `ff8a2fcf` (D4, np.evaluate) the streamed row sums keep the running sum's NaN in a short row (2 to 7 elements, 1 to 3
    complexes): the inline copy of the fold's base loop left `res += x[k]` to RyuJIT, which returned the later of two
    NaNs (112 of 21,320 two-NaN rows), so `np.evaluate(Sum(x, axis))` and np.sum disagreed. Found by step 2's identity
    check of its fast path (49 of 11,520 cases differed, NumPy's answer the composition's in all 24 it could load).
    complex128's NumPy rule (the real part keeps the running sum's NaN, the imaginary part the element's) is pinned, not
    fixed: `OpenBugsReductionTests.Sum_Complex128_ShortRun_KeepsNumPysNaNPerPart` (group 8).
  - `97a3cc51` step 2 (C11): with weights, NumPy's composition statement by statement (the weight sum, its zero check,
    `np.multiply(a, wgt, dtype=result_dtype).sum(axis)`, the division); the fused weighted-sum kernel's routes, a few ULP
    from NumPy's pairwise sums, no longer run (the kernel stays: public API since 0.70.0). The products' sum streams
    through np.evaluate where that is byte-identical (same dimensions, one memory order, float32 / float64 / complex128,
    one axis or every axis: 11,520 of 11,520 operand pairs identical). The oracle's W6-C excuse (every value divergence
    of average) goes; `[OpenBugs]` group 14 (a tuple axis) passes and moves to `np.average.TupleAxes.Test.cs`. 4,090 lines
    fixed. NPY/NS: geomean 3.54x over 35 cells, min 1.68x (float32 (100, 1000) along axis 0 with 1-D weights); the old
    fused routes measured 4.09x with other bits.
  - Evidence at `97a3cc51`: the twin agrees with NumPy 2.4.2 on 32,376 of the 32,404 comparable lines (23,129 at
    `2051e724`). The 28 left: 20 error texts naming `CharDType` (above) and 8 unweighted complex128 means over special
    values whose real-part NaN sign differs (the complex sum's NaN choice, group 8). The random differential: 2,998 of
    3,000 identical (1,482 at `35df73cc`, before the D4 fixes); the 2 left are a complex NaN sign and the flags of an
    empty-input mean (np.mean over an empty (0, 9, 9) along axis 0: NumPy's result is F-ordered, NumSharp's C; np.mean's
    empty-strides residual).
  - **Found:**
    - Group 15 stays open: the plain C-contiguous axis-0 sum runs the axis kernel, whose slab fold keeps the FIRST NaN of
      a column where NumPy's elementwise add keeps the last (the iterator route of `da7adc53` keeps NumPy's).
    - NumPy's complex128 pairwise sum chooses its NaN per part (above); the complex block's accumulators and combine are
      not probed yet, so group 8's complex pin covers the base loop only.
    - The NDW012 analyzer flags `var ret = ReduceAdd(...); return FinishMeanLikeNumPy(ret, ...)` (MeanOverAxes,
      MeanLikeNumPy) although the callee disposes or returns `ret`: false positives, unchanged.
    - The Fuzz README's relayout table was stale (15,108 variants, 35 gap cases under 3 ids); corrected with the
      regeneration.
  - Left: a decimal operand with NaN weights raises np.sum's OverflowException (NaN cast into decimal; no NumPy analog)
    where the old route returned a value; np.average's broadcast-weight cells form the product (np.evaluate would
    materialize a stride-0 operand anyway).
  - At `97a3cc51`: NumSharp.Tests net10.0 20,659 passed / 11 skipped and net8.0 20,658 / 11; Oracle net10.0 and net8.0
    412 passed / 2 skipped, 0 failed.
  - Next in P2: the files of four lines or fewer.
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

## Appendix D. Where every NumPy function lives

Every public callable of NumPy 2.4.2's `numpy`, `numpy.linalg`, `numpy.fft`, `numpy.random` and `numpy.emath`, and the
`ndarray` methods, 637 in all, placed under D1 as the owner decided it on 2026-10-28 (the engine layout, then the array
object kept out of the engine):
- "NumPy" counts the callables, and "NumSharp" the names NumSharp has on the matching surface (`np`, `np.linalg`,
  `np.fft`, `np.random`, `np.emath`, `NDArray`).
- Names in *italics* are missing from NumSharp.
- † marks the 61 functions of D8 (a facade without dtype logic).

Measured 2026-10-28: NumPy's side by introspection, NumSharp's names by reflection over a build of `97a3cc51`. No public
name changed between that commit and `67249651`.

Two questions place a callable:
- **What it does** chooses the class ([§1](#1-what-out-of-np-has-to-mean) has the criterion and the borderline cases).
  A function that computes values is a `TensorEngine` member, or a member of the `Linalg` / `Fft` sub-engine. A
  function of the array object is an internal member of an `NDArray` partial, over the memory work of `UnmanagedStorage`
  partials and the layout math of `Shape` partials.
- **Where NumPy writes it** names the partial file, in both classes:
  - a ufunc goes to `Umath`;
  - a function written in C (a builtin, or an `__array_function__` dispatcher over one) goes to `Multiarray`;
  - a function of `_core/fromnumeric.py` (NumPy's wrappers over the ndarray methods) goes to `Methods`;
  - any other function written in Python goes to the partial named after its source file;
  - numpy.linalg and numpy.fft go to their sub-engines, numpy.emath to `TensorEngine.Scimath.cs`.

Classes, np.random, the dtype routines, IO, printing and configuration stay outside both, however NumPy writes them.

```python
import inspect, numpy as np

def implemented_in(obj):
    if inspect.isclass(obj):
        return "class"
    if isinstance(obj, np.ufunc):
        return "ufunc"
    body = getattr(obj, "_implementation", obj)   # an __array_function__ dispatcher's Python body, if it has one
    if inspect.isfunction(body):
        return inspect.getsourcefile(body)        # .../numpy/lib/_function_base_impl.py, ...
    return "C"
```

### `TensorEngine`: the computations, 287 (NumSharp 272)

| Home | NumPy source | NumPy | NumSharp | Functions |
|---|---|---|---|---|
| `TensorEngine.Umath.cs` | the ufuncs | 106 | 97 | abs, absolute, *acos*, acosh, add, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, *asin*, asinh, *atan*, *atan2*, atanh, bitwise_and, bitwise_count, *bitwise_invert*, *bitwise_left_shift*, bitwise_not, bitwise_or, *bitwise_right_shift*, bitwise_xor, cbrt, ceil, conj, conjugate, copysign, cos, cosh, deg2rad, degrees, divide, divmod, equal, exp, exp2, expm1, fabs, float_power, floor, floor_divide, fmax, fmin, fmod, frexp, gcd, greater, greater_equal, heaviside, hypot, invert, isfinite, isinf, isnan, *isnat*, lcm, ldexp, left_shift, less, less_equal, log, log10, log1p, log2, logaddexp, logaddexp2, logical_and, logical_not, logical_or, logical_xor, matmul, matvec, maximum, minimum, mod, modf, multiply, negative, nextafter, not_equal, positive, *pow*, power, rad2deg, radians, reciprocal, remainder, right_shift, rint, sign, signbit, sin, sinh, spacing, sqrt, square, subtract, tan, tanh, true_divide, trunc, vecdot, vecmat |
| `TensorEngine.Methods.cs` | `_core/fromnumeric.py` over the ndarray methods, `_core/_methods.py` | 28 | 25 | all, amax, amin, any, argmax, argmin, argpartition, argsort, around, clip, cumprod, cumsum, *cumulative_prod*, *cumulative_sum*, max, mean, min, nonzero, partition, prod, ptp, *round*, searchsorted, sort, std, sum, trace, var |
| `TensorEngine.Multiarray.cs` | the C functions of `_multiarray_umath` | 12 | 9 | bincount, *busday_count*, *busday_offset*, dot, inner, *is_busday*, lexsort, packbits, ravel_multi_index, unpackbits, unravel_index, vdot |
| `TensorEngine.FunctionBase.cs` | `lib/_function_base_impl.py` | 22 | 22 | angle, average, bartlett†, blackman†, corrcoef†, cov, diff, digitize, gradient, hamming†, hanning†, i0, interp, kaiser†, median, percentile, quantile, sinc, sort_complex, trapezoid, trim_zeros, unwrap |
| `TensorEngine.Numeric.cs` | `_core/numeric.py` | 12 | 12 | allclose, argwhere, array_equal, array_equiv, convolve, correlate, count_nonzero, cross, flatnonzero, isclose, outer†, tensordot† |
| `TensorEngine.ShapeBase.cs` | `lib/_shape_base_impl.py` | 1 | 1 | kron |
| `TensorEngine.TwoDimBase.cs` | `lib/_twodim_base_impl.py` | 1 | 1 | vander |
| `TensorEngine.NanFunctions.cs` | `lib/_nanfunctions_impl.py` | 14 | 14 | nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean, nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd, nansum, nanvar |
| `TensorEngine.ArraySetOps.cs` | `lib/_arraysetops_impl.py` | 11 | 11 | ediff1d, intersect1d, isin, setdiff1d, setxor1d, union1d, unique, unique_all, unique_counts, unique_inverse, unique_values |
| `TensorEngine.Polynomial.cs` | `lib/_polynomial_impl.py` (legacy np.poly*) | 10 | 10 | poly, polyadd, polyder†, polydiv, polyfit†, polyint, polymul, polysub, polyval, roots |
| `TensorEngine.Scimath.cs` | `lib/_scimath_impl.py` (np.emath; members prefixed `Emath`) | 9 | 9 | arccos, arcsin, arctanh, log, log10, log2, logn, power, sqrt |
| `TensorEngine.TypeCheck.cs` | `lib/_type_check_impl.py` | 4 | 4 | iscomplex, isreal, nan_to_num, real_if_close |
| `TensorEngine.UfuncLike.cs` | `lib/_ufunclike_impl.py` | 3 | 3 | fix†, isneginf, isposinf |
| `TensorEngine.CoreFunctionBase.cs` | `_core/function_base.py` | 3 | 3 | geomspace, linspace, logspace |
| `TensorEngine.Histograms.cs` | `lib/_histograms_impl.py` (+ histogram2d) | 4 | 4 | histogram, histogram2d, histogram_bin_edges, histogramdd |
| `TensorEngine.EinsumFunc.cs` | `_core/einsumfunc.py` | 2 | 2 | einsum, einsum_path† |
| `TensorEngine.Linalg` | numpy.linalg | 29 | 29 | cholesky, cond, cross, det, eig, eigh, eigvals, eigvalsh, inv, lstsq, matmul, matrix_norm, matrix_power, matrix_rank, multi_dot, norm, outer, pinv, qr, slogdet, solve, svd, svdvals, tensordot, tensorinv, tensorsolve, trace, vecdot, vector_norm |
| `TensorEngine.Fft` | numpy.fft | 16 | 16 | fft, fft2, fftfreq, fftn, hfft, ifft, ifft2, ifftn, ihfft, irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftn |

`TensorEngine.ArrayObject.cs` keeps the ten array-object members shipped in 0.70.0 (E7); they are not counted here.

### The array object, 111 (NumSharp 106)

| Home | NumPy source | NumPy | NumSharp | Functions |
|---|---|---|---|---|
| `NDArray`'s members (the ndarray methods; `NDArray.Methods.cs` where a member is missing) | `_core/fromnumeric.py` over the ndarray methods | 17 | 15 | choose, compress, diagonal, matrix_transpose, *ndim*, permute_dims, put, ravel, repeat, reshape, resize, *shape*, size, squeeze, swapaxes, take, transpose |
| `NDArray.Multiarray.cs` | the C functions of `_multiarray_umath` | 18 | 17 | arange, array, asanyarray, asarray, ascontiguousarray, asfortranarray, concat, concatenate, copyto, empty, empty_like, *from_dlpack*, may_share_memory, nested_iters, putmask, shares_memory, where, zeros |
| `NDArray.FunctionBase.cs` | `lib/_function_base_impl.py` | 13 | 13 | append†, asarray_chkfinite, copy†, delete, extract, flip†, insert, iterable†, meshgrid†, piecewise, place, rot90†, select |
| `NDArray.Numeric.cs` | `_core/numeric.py` | 13 | 12 | *fromfunction*, full, full_like, identity†, indices, isfortran†, isscalar†, moveaxis, ones, ones_like†, roll†, rollaxis, zeros_like† |
| `NDArray.ShapeBase.cs` | `lib/_shape_base_impl.py` | 14 | 13 | apply_along_axis, apply_over_axes†, array_split†, column_stack†, dsplit†, dstack†, expand_dims†, hsplit†, put_along_axis, *row_stack*, split†, take_along_axis, tile†, vsplit† |
| `NDArray.TwoDimBase.cs` | `lib/_twodim_base_impl.py` | 13 | 13 | diag, diagflat†, eye, fliplr†, flipud†, mask_indices†, tri†, tril†, tril_indices†, tril_indices_from†, triu†, triu_indices†, triu_indices_from† |
| `NDArray.CoreShapeBase.cs` | `_core/shape_base.py` | 8 | 8 | atleast_1d†, atleast_2d†, atleast_3d†, block†, hstack†, stack†, unstack†, vstack† |
| `NDArray.TypeCheck.cs` | `lib/_type_check_impl.py` | 2 | 2 | imag, real |
| `NDArray.IndexTricks.cs` | `lib/_index_tricks_impl.py` | 4 | 4 | diag_indices†, diag_indices_from†, fill_diagonal, ix_ |
| `NDArray.StrideTricks.cs` | `lib/_stride_tricks_impl.py` | 3 | 3 | broadcast_arrays†, broadcast_shapes†, broadcast_to† |
| `NDArray.ArrayPad.cs` | `lib/_arraypad_impl.py` | 1 | 1 | pad |
| `NDArray.CoreAsArray.cs` | `_core/_asarray.py` | 1 | 1 | require† |
| facades over the array object | numpy.linalg | 2 | 2 | diagonal, matrix_transpose |
| facades over the array object | numpy.fft | 2 | 2 | fftshift, ifftshift |

The memory and layout work under these members:
- `UnmanagedStorage` partials (`Backends/Unmanaged/`):
  - `UnmanagedStorage.Assign.cs` (new): `array_assign_array.c` and `array_assign_scalar.c`, the assignment np.copyto
    runs; the store (`SetData`, NumPy's `PyArray_CopyObject`), np.concatenate, np.putmask and the fills call it.
  - `UnmanagedStorage.ItemSelection.cs` (new): `item_selection.c` and `compiled_base.c`'s `arr_place`: take, put,
    putmask, place, repeat, choose, compress. The IL kernels stay where they are.
  - `UnmanagedStorage.Coercion.cs` (new): `array_coercion.c`, moved from `Polynomial/Package/NDPolySequence.cs`.
  - `UnmanagedStorage.Ctors.cs` (new): `ctors.c`, the allocations behind empty, zeros, full, ones and the `_like` forms,
    and np.arange through the `DType` class's `fill` slot.
  - `UnmanagedStorage.Buffer.cs` (exists): np.frombuffer's body, NumPy's `PyArray_FromBuffer` (np.frombuffer itself is
    counted with IO below).
- `Shape` partials (`View/`): `Shape.Broadcasting.cs` (exists; takes `DefaultEngine.Broadcast`, `ResolveReturnShape` and
  `AreBroadcastable`), `Shape.Reshaping.cs` and `Shape.Allocation.cs` (exist), `Shape.Axes.cs` (new:
  `normalize_axis_tuple`, `check_and_adjust_axis`, the axis permutations of transpose, swapaxes, moveaxis and rollaxis).

### Outside both, 239 (NumSharp 208)

- classes and types (ndarray, dtype and the scalar types, nditer, broadcast, finfo / iinfo, vectorize, poly1d, errstate,
  `frompyfunc`, which makes a ufunc object, ...): 71 (NumSharp 61);
- np.random (Generator, RandomState, the bit generators: NumPy's own seam): 63 (NumSharp 61);
- the ndarray methods (`NDArray` members: those that compute call the `TensorEngine` members, the others are the array
  object's own): 53 (NumSharp 51);
- printing, formatting and error / buffer configuration (`base_repr`† and `binary_repr`† included): 15 (NumSharp 11);
- IO (np.frombuffer's body on `UnmanagedStorage.Buffer.cs`, the rest in `IO/`; their dtype lines go to the `DType`
  slots, P3): 12 (NumSharp 9);
- dtype routines (the DType layer; `iscomplexobj`, `isrealobj` and `datetime_as_string` included): 14 (NumSharp 12);
- other: 11 (NumSharp 3), numpy.matrix's asmatrix and bmat, printoptions, LinAlgError, the utilities and the test runners, of which NumSharp has the first three.

## Appendix E. P3's lines by file

R1's lines of P3's 80 facade files at `1fcb4663` (`LayeringScanner`'s own dump), each sorted by `classify_p3.py` into
the mechanism NumPy uses for the same decision (the table of P3's design names them), and the batch of the design that
moves the file. Then the nine files with K4 text lines that R1 does not scan.

| File | numeric | text | Lines decide | Batch |
|---|---|---|---|---|
| `Indexing/np.choose.cs` | 7 | 7 | class 4, refs 3, weak 2, index 1, other 1, request 1, typed 1, value 1 | 2 |
| `Indexing/np.compress.cs` | 1 | 0 | index 1 | 2 |
| `Indexing/np.place.cs` | 2 | 9 | cast 4, class 3, index 2, refs 2 | 2 |
| `Indexing/np.put.cs` | 0 | 2 | refs 2 | 2 |
| `Indexing/np.putmask.cs` | 2 | 4 | index 2, refs 2, cast 1, class 1 | 2 |
| `Indexing/np.take.FancyAxis.cs` | 1 | 0 | index 1 | 2 |
| `Indexing/np.take.cs` | 4 | 3 | class 2, other 2, index 1, refs 1, value 1 | 2 |
| `Manipulation/NDArray.resize.cs` | 0 | 3 | refs 3 | 2 |
| `Manipulation/np.repeat.cs` | 9 | 2 | typed 9, refs 2 | 2 |
| `Manipulation/np.resize.cs` | 0 | 1 | refs 1 | 2 |
| `Creation/np.concatenate.Strings.cs` | 0 | 4 | class 3, cast 1 | 3 |
| `Creation/np.concatenate.cs` | 2 | 0 | request 2 | 3 |
| `Manipulation/NDArray.fill.cs` | 1 | 3 | setitem 2, class 1, refs 1 | 3 |
| `Manipulation/np.copyto.Strings.cs` | 2 | 15 | cast 12, nonzero 2, index 1, other 1, setitem 1 | 3 |
| `Manipulation/np.copyto.VString.cs` | 0 | 7 | refs 6, cast 1 | 3 |
| `Manipulation/np.copyto.cs` | 5 | 7 | class 5, request 3, cast 2, other 2 | 3 |
| `Casting/Implicit/NDArray.Implicit.Array.cs` | 48 | 3 | typed 48, discover 1, getitem 1, setitem 1 | 4 |
| `Creation/NDArray.Scalar.cs` | 0 | 5 | refs 2, setitem 2, class 1 | 4 |
| `Creation/np.array.cs` | 3 | 5 | setitem 3, typed 2, class 1, discover 1, request 1 | 4 |
| `Creation/np.array.string.cs` | 0 | 1 | setitem 1 | 4 |
| `Creation/np.asanyarray.cs` | 0 | 4 | discover 2, setitem 2 | 4 |
| `Creation/np.asarray.cs` | 0 | 2 | class 1, discover 1 | 4 |
| `Creation/np.arange.cs` | 18 | 0 | typed 15, request 2, index 1 | 5 |
| `Creation/np.eye.cs` | 4 | 1 | typed 4, class 1 | 5 |
| `Creation/np.full.cs` | 3 | 8 | refs 4, setitem 3, class 2, request 2 | 5 |
| `Creation/np.full_like.cs` | 1 | 4 | refs 2, class 1, request 1, setitem 1 | 5 |
| `Creation/np.ndarray.cs` | 1 | 10 | class 6, refs 4, request 1 | 5 |
| `Creation/np.ones.cs` | 4 | 6 | refs 4, typed 4, class 1, setitem 1 | 5 |
| `APIs/np.fromfile.cs` | 0 | 6 | class 3, codec 2, refs 1 | 6 |
| `APIs/np.fromstring.cs` | 0 | 2 | class 1, refs 1 | 6 |
| `APIs/np.loadtxt.Parsers.cs` | 15 | 0 | typed 15 | 6 |
| `APIs/np.loadtxt.Sinks.cs` | 0 | 24 | refs 8, codec 5, setitem 5, class 3, discover 2, other 1 | 6 |
| `APIs/np.loadtxt.cs` | 0 | 2 | class 1, other 1 | 6 |
| `APIs/np.savetxt.cs` | 1 | 1 | class 1, value 1 | 6 |
| `APIs/np.tofile.cs` | 6 | 5 | typed 4, codec 3, value 2, class 1, getitem 1 | 6 |
| `Casting/NDArray.byteswap.cs` | 1 | 1 | class 1, value 1 | 6 |
| `Casting/NDArray.getfield.cs` | 0 | 1 | class 1 | 6 |
| `Casting/NDArray.tobytes.cs` | 0 | 1 | refs 1 | 6 |
| `Creation/np.frombuffer.Strings.cs` | 0 | 4 | class 3, refs 1 | 6 |
| `Creation/np.frombuffer.cs` | 93 | 12 | typed 85, class 12, request 8 | 6 |
| `APIs/np.where.cs` | 8 | 2 | request 4, index 3, class 2, other 1 | 7 |
| `Creation/np.asarray_chkfinite.cs` | 1 | 0 | value 1 | 7 |
| `Creation/np.mgrid.cs` | 1 | 0 | request 1 | 7 |
| `Creation/np.ogrid.cs` | 1 | 0 | request 1 | 7 |
| `Creation/np.r_.cs` | 8 | 1 | weak 5, request 2, class 1, index 1 | 7 |
| `Indexing/np.diag.cs` | 0 | 1 | refs 1 | 7 |
| `Indexing/np.extract.cs` | 1 | 0 | index 1 | 7 |
| `Indexing/np.fill_diagonal.cs` | 0 | 3 | refs 2, setitem 1 | 7 |
| `Indexing/np.indices.cs` | 3 | 0 | request 3 | 7 |
| `Indexing/np.ix_.cs` | 2 | 0 | index 2 | 7 |
| `Indexing/np.piecewise.cs` | 7 | 0 | value 6, index 1 | 7 |
| `Indexing/np.put_along_axis.cs` | 1 | 2 | refs 2, index 1 | 7 |
| `Indexing/np.ravel_multi_index.cs` | 1 | 0 | index 1 | 7 |
| `Indexing/np.select.cs` | 10 | 3 | weak 4, class 3, index 3, request 3 | 7 |
| `Indexing/np.take_along_axis.cs` | 9 | 1 | typed 8, index 1, refs 1 | 7 |
| `Manipulation/NDArray.FunctionBase.cs` | 8 | 7 | refs 5, typed 3, value 3, class 2, index 2 | 7 |
| `Manipulation/np.apply_along_axis.cs` | 1 | 0 | request 1 | 7 |
| `Manipulation/np.delete.cs` | 20 | 0 | typed 20 | 7 |
| `Manipulation/np.pad.cs` | 33 | 12 | typed 23, class 7, value 5, setitem 4, index 2, request 2, other 1, refs 1 | 7 |
| `Math/NDArray.ComplexAccessors.cs` | 2 | 1 | value 2, class 1 | 7 |
| `Math/np.imag.cs` | 1 | 0 | value 1 | 7 |
| `Math/np.real.cs` | 1 | 0 | value 1 | 7 |
| `Selection/NDArray.Indexing.Masking.cs` | 0 | 3 | class 2, refs 1 | 8 |
| `Selection/NDArray.Indexing.PrepareIndex.cs` | 12 | 0 | typed 8, index 2, other 2 | 8 |
| `Selection/NDArray.Indexing.Selection.Getter.cs` | 41 | 1 | index 22, typed 19, class 1 | 8 |
| `Selection/NDArray.Indexing.Selection.Setter.cs` | 23 | 2 | typed 16, index 7, class 2 | 8 |
| `Selection/NDArray.Indexing.Selection.cs` | 4 | 0 | value 3, other 1 | 8 |
| `Selection/NDArray.Indexing.Strings.cs` | 3 | 1 | index 3, setitem 1 | 8 |
| `Selection/NDArray.Indexing.VString.cs` | 0 | 5 | refs 4, class 1 | 8 |
| `Selection/NDArray.Indexing.cs` | 1 | 2 | class 2, index 1 | 8 |
| `Creation/np.geomspace.cs` | 1 | 0 | request 1 | 9 |
| `Creation/np.linspace.cs` | 23 | 0 | typed 16, value 6, request 1 | 9 |
| `Creation/np.logspace.cs` | 1 | 0 | request 1 | 9 |
| `Creation/np.vander.cs` | 0 | 3 | ufunc 2, refs 1 | 9 |
| `Manipulation/np.trim_zeros.cs` | 0 | 2 | class 1, nonzero 1 | 9 |
| `Casting/Implicit/NDArray.Implicit.ValueTypes.cs` | 3 | 4 | cast 2, class 2, value 2, index 1 | stays |
| `Casting/NDArray.ToString.cs` | 1 | 0 | value 1 | stays |
| `Creation/np.dtype.cs` | 2 | 0 | request 2 | stays |
| `DTypes/np.dtypes.cs` | 0 | 2 | class 2 | stays |
| `Indexing/np.ndenumerate.cs` | 1 | 0 | typed 1 | stays |

| File (not scanned by R1) | K4 text lines | Lines decide | Batch |
|---|---|---|---|
| `DTypes/StringDTypeMeta.cs` | 2 | discover 2 (the slot's own code) | stays |
| `IO/PyLiteral.cs` | 2 | codec 2 (Python's float repr) | stays |
| `IO/TextReading/LoadTxtSources.cs` | 1 | codec 1 (a Python type name) | stays |
| `IO/TextReading/TextIOCodec.Incremental.cs` | 3 | codec 3 | stays |
| `IO/TextReading/TextIOCodec.Unicode.cs` | 1 | codec 1 | stays |
| `IO/TextReading/TextIOCodec.cs` | 4 | codec 4 | stays |
| `RandomSampling/BoundedIntegers.Broadcast.cs` | 3 | ufunc 2, class 1 (np.random's own refusals) | stays |
| `RandomSampling/SeedSequence.cs` | 2 | class 2 (a Python type name) | stays |
| `Utilities/PyUnicode.cs` | 2 | codec 2 (the Unicode database) | stays |
