# API parameter parity: every NumPy parameter, present and working

Living plan and state document. Branch `journey4`. Started 2026-10-05.

**Task (verbatim):** "Take all the missing or non functioning paramters or api mismatch like out or where or others.
write them in a plan with progress marking and set on a task to fix one by one. Proving parity and that it is a bug of
parity and only then fix."

**State:** see [§7 State log](#7-state-log) (newest entry first) and the checklists in [§5](#5-items).

**Out of scope here:** the seven view-`Shape` bugs (`OpenBugs.ViewShape.cs`: the wrap constructors and the shared
dimension array). A parallel session owned them in its own plan, `docs/plans/view-shape-open-bugs.md`, which is done
(2026-10-05, last fix `5c9d7925`). Its `np.ndarray` follow-ups joined item D7 below.

---

## 1. Goal

A NumPy call ported to C# word for word binds, and it gives NumPy 2.4.2's result. For every `np.*` function and
`ndarray` method that NumSharp has:

1. **Present.** Each NumPy parameter exists with NumPy's name. A positional-or-keyword parameter sits at NumPy's
   position, so a positional port binds the same argument. A keyword-only parameter may sit anywhere (a ported call
   names it). A positional-only parameter needs only its slot.
2. **Working.** Each parameter changes the result the way NumPy's does: values, dtype, shape, the returned object
   (`out=` returns `out` itself), errors with NumPy's message.
3. **Proven first.** No item is fixed before a NumPy 2.4.2 probe and a NumSharp probe show the divergence. The probe
   output is recorded in the item.

## 2. Method

**Inventory** (2026-10-05, at `2821a708`). `dump_api.cs` reflects every public overload of `np` (383 names) and
`NDArray` (135 names). `param_gaps.py` compares each name NumPy also has with NumPy 2.4.2's `inspect.signature` under
the rules of §1. Result: 66 `np.*` functions and 14 `ndarray` methods with a gap (the `ndarray.clip` row is a false
positive of the `**kwargs` merge). The scripts are in `tools/api-parameter-parity/`; rerun them to measure progress.

"Working" cannot come from signatures. Its sources: the `[OpenBugs]` pins that test parameters (phase A), the probe
done for each item, and the audit of existing parameters (phase X).

**Cycle per item:**
1. Probe NumPy 2.4.2 and NumSharp with the same inputs. Record both outputs in the item. No divergence: close the item
   as "no bug".
2. Read NumPy's implementation in `refs/numpy`; port its structure.
3. Implement on the existing engines (IL kernels, `NDIter`); all 15 dtypes; NumPy's error texts and error order.
4. Tests from NumPy output. A pinned item loses its `[OpenBugs]` and moves next to the function's tests. An oracle
   excuse that named the bug (`MisalignedRegistry`) is removed.
5. Check the speed of the touched paths against NumPy (NPY/NS).
6. Commit; mark the item `[x]` with the commit.

## 3. Exclusions (not gaps)

| Parameter | Where | Reason |
|---|---|---|
| `like=` | creation functions | NumPy's `__array_function__` protocol; NumSharp has no array subclasses or protocol. |
| `subok=` | everywhere | NumSharp has no `ndarray` subclasses. |
| `casting=`, `order=`, `signature=` | ufuncs | NumSharp models none of them in its ufunc surface (`.claude/CLAUDE.md`, the CBLAS product family). |
| `order=` | `sort`, `argsort` | Field order of structured dtypes; NumSharp has no structured dtypes. |
| `einsum_call=` | `einsum_path` | NumPy-internal (documented in `.claude/CLAUDE.md`, `np.einsum_path`). |
| order of keyword-only parameters | everywhere | A ported call names them, so their position cannot change a binding. |

## 4. Priorities

- **Phase A**: a parameter defect that an `[OpenBugs]` test already pins (2026-10-03/04). Fix first.
- **Phase B**: the reductions' missing `out=` / `initial=` / `where=` / `dtype=` / `mean=` / `correction=`.
- **Phases C–F**: sorting, creation, manipulation, ufunc-shaped functions.
- **Phase G**: names and positions only (a named or positional port fails to compile).
- **Phase H**: rarely used parameters.
- **Phase S**: a signature gate so that a fixed item cannot regress.
- **Phase X**: audit of parameters that exist but may not work.

## 5. Items

Status: `[ ]` open, `[~]` in progress, `[x]` done (commit), `[-]` closed without a code change (reason).

### Phase A: pinned parameter defects

- [x] **A1 `np.all` / `np.any` / `ndarray.all` / `ndarray.any`: a non-bool `where=` mask** (and the rest of the reduce
  prologue). Done 2026-10-05 (`bb5b2fcf`).
  - Proof: NumPy 2.4.2 raises `TypeError: Cannot cast array data from dtype('X') to dtype('bool') according to the
    rule 'safe'` for every non-bool mask dtype ("Cannot cast scalar from" for a 0-d mask), before it checks the axis
    or any shape. NumSharp returned a value for all 11 mask dtypes, and with a bad shape or axis it reported that
    error instead. A 7,772-case matrix (axis forms x keepdims x out= dtypes / ranks / extents / read-only x where=
    dtypes / shapes) found the rest of the prologue missing: a mask larger than the input stretched the input
    silently, an out of the wrong rank or extent raised a generic text, a read-only out was not checked first.
  - Fix: `ReductionArguments` (`Backends/Default/Math/Reduction/ReductionArguments.cs`), one port of NumPy's reduce
    prologue (`PyUFunc_GenericReduction`, `PyUFunc_ReduceWrapper`, `npyiter_fill_axisdata`) for every reduction;
    `np.all` / `np.any` (both axis forms) call it when out= or where= is given. The shared ufunc mask check
    (`DefaultEngine.ValidateWhereMask`) now says "scalar" for a 0-d mask, as NumPy does for the ufuncs too.
  - Gate: the new `reduce_kwargs` oracle tier (S2), 4,752 cases, 0 excused; 432 of them diverge on the code before
    the fix. Unit tests in `Logic/np.all_any.tuple_out_where.Test.cs` (the pin moved there; two tests that asserted
    the old behaviour now assert NumPy's).
  - Also fixed: a bare `null` axis (`np.all(a, null, null, false, w)`, NumPy's positional `None`) was ambiguous
    between the `int?` and `int[]` overloads (CS0121); the `int[]` overloads now carry `OverloadResolutionPriority(-1)`.
    The other functions with the same ambiguity are item G4.
- [x] **A2 `np.var` / `np.std`: `ddof=` and `dtype=` do not work** — at a degenerate size (one element per output,
  an axis of length 1, `ddof >= n`, empty float16/float32 input) and, as the proof below found, at every size for
  `dtype=`. Pins: the 7 `VarStd_*` tests in `OpenBugs.Reductions.cs`; oracle excuse `MisalignedRegistry` V1. Done
  2026-10-05 (`41785501`).
  - Proof (2026-10-05, at `bb5b2fcf`): a 22,110-case NumPy 2.4.2 matrix (`var`/`std` × 13 dtypes × one element per
    output / empty / `ddof` at or past the count / `dtype=` of every kind at ordinary sizes × C/F/strided/reversed/
    transposed layouts × axis forms × keepdims). NumSharp: 12,212 exact, 678 within 64 ULP, 8,284 grossly different
    (wrong dtype, NaN against a number, a value where NumPy raises), and 936 cases WRITE PAST THE RESULT BUFFER:
    the empty-input path does `SetAtIndex(double.NaN, i)` into a float16/float32/int32/complex128 result (heap
    corruption, `0xC0000374`, for float16). NumPy's `_var` has no special path for a small size; it is
    `sum((x - mean)**2) / max(n - ddof, 0)` computed with ufuncs in the input's float dtype or in `dtype=` (an
    integer `dtype=` truncates the mean; `np.std(..., axis, dtype=int)` raises the `sqrt` output-cast
    UFuncTypeError; a complex input with `dtype=float64` keeps only the real part of the mean).
  - Design proof: a prototype that ports `_var` statement for statement over NumSharp's own ufuncs matched NumPy in
    22,102 of the 22,110 cases, error texts included. The 8 others: 2 are the `np.sum` reversed-view finding (§6,
    item 5), 6 were item X3 (fixed first).
  - Fix (`Default.Reduction.VarStd.cs`, `VarStdNumPy`; `ReduceVar` / `ReduceStd` delegate to it, and the old
    shortcuts, moment loops and axis-kernel wrappers are gone): NumPy's statements in order. The axis is normalized
    first (AxisError for a 0-d operand too); an integer / bool / char input sums in float64; the mean is `ReduceAdd`
    (NumPy's add.reduce schedule) divided in place like `true_divide(arrmean, intp n, out=arrmean, casting='unsafe')`
    (the float64 loop for every real dtype, Smith for complex, an unsafe cast for an integer `dtype=`); `x` is laid out
    as NpyIter lays out `arr - arrmean` (`DeviationLayout`, with the mean's vote taken from the layout NumPy allocates
    for a keepdims sum, `KeepdimsReductionLayout` — NumSharp's own keepdims result is C-ordered on a permuted input,
    and its vote put `x` in C order, one ULP off at the first probed case); the square is NumPy's statement for the
    case (`x*x`; `x*x` in complex for a real input with a complex `dtype=`; `real² + imag²` for a complex or bool
    input with a complex `x`); the sum is `ReduceAdd`, then `/ max(n - ddof, 0)`; std is `sqrt(ret, out=ret)` for an
    array (an integer loop raises NumPy's output-cast error) and `ret.dtype.type(sqrt(ret))` for a scalar.
    Engines: one fused `np.evaluate` pass for the deviations; a flat reduction over one memory run without a cast
    streams them (`np.evaluate`'s flat Sum, no temporary); float16 / decimal / no-dynamic-code hosts use the eager
    ufuncs (np.evaluate's half arithmetic costs ~8 ns per element, the SIMD half kernels 1.3); an integer or narrower
    float input is cast into the buffer first and evaluated in place (np.evaluate converts a mixed-width leaf one
    element at a time: 0.42 against 0.16 ns per element). Inputs of 8 MiB of squared deviations and more are summed
    in L2-sized chunks with NumPy's order kept: whole outputs along the outermost kept axis (`ChunkedDeviationSum`),
    row blocks with a running `np.add(acc, slab)` when every kept axis lies inside the reduced one
    (`RowBlockDeviationSum`), whole buffer fills of a flat cast loop added in the loop dtype
    (`FlatFillDeviationSum`); a cast loop chunks only where NumPy's fills cannot split an output
    (`CastLoopChunksExactly`). The 936 heap-corrupting empty cases are gone with the old empty path.
  - Proof of the fix: the 22,110-case matrix — 22,108 exact, the 2 ULP cases of finding 5; a 1,408-case large-size
    matrix (up to 1,000,003 elements, C/F/stepped/reversed, `dtype=` casts) — all exact; a 7,700-case layout matrix (3-D
    permutations with and without a reversed axis, broadcast, offset, stepped, 0-d, 5-D; 8 dtypes) — 7,210 exact, and
    each of the 490 others inherits a difference of `np.sum(a, axis, dtype, keepdims=True)` itself (a reversed axis,
    finding 5, or a broadcast operand's sum), checked case by case. All three again with the chunked routes forced:
    the same answers.
  - Gates: oracle tier `varstd.jsonl` (16,108 cases, zero excuses: every catalog layout × dtype × axis × keepdims ×
    ddof {0, 1}; ddof at and past the count; `dtype=` for every (input, loop) pair; special values; `np.setbufsize`
    fills; order-sensitive sizes; the char proxy) and `VarStdChunked` (the same tier with the chunked routes forced,
    the run counter asserted); `MisalignedRegistry` excuses no np/ndarray var/std case (V1, the dtype blanket, the
    unbounded summation blanket, the "axis SIMD min/max skips NaN" branch and the unary ULP branch all stopped
    covering them — finding 7; tightness test `VarStd_NoDivergenceIsExcused`); `reduce_start.jsonl` gained the 24
    var/std cells it carved (8,610); the decimal tier's var/std reference is exact decimal in NumSharp's summation
    order (finding 8); unit tests `Math/np.var.NumPyPort.Test.cs` (15: the 7 former pins, `dtype=` loops, the errors,
    the permuted layout, float16 / char / decimal, the chunked routes byte for byte).
  - Speed (NPY/NS, one pinned P-core, best of rounds; 116 cells: 7 dtypes × 1K / 100K / 10M × flat / axis 0 / axis 1,
    plus strided / F / ddof / `dtype=` forms): geomean 3.07×, minimum 1.33×, none below 1.0 (the old engine: from
    0.03× on F-ordered and integer flat inputs to 38× on contiguous float32 at 1K, with the wrong answers above).
    Flat float64 4.1× at 10M, complex128 4.1×, bool 14×; float64 axis 0 / 1 at 10M 3.0× / 3.8× (were 0.95× / 1.5×
    before the chunked routes). Under 1.5×: the 1K axis cells (1.33–1.43×: the building blocks' fixed cost — two
    `np.sum` calls of ~1.1 µs, two in-place divisions of ~0.6 µs and one `np.evaluate` of ~0.95 µs; the same calls
    made by hand without the boundary scope take 4.8 µs against NumPy's 6.9), and bool axis 0 at 10M (1.38×: the
    mean's sum is item X7). Measured on `4d06d591`; after the rebase onto `5c9d7925` (the new code adopts the shapes it
    builds) an interleaved A/B of the cells that moved in a full rerun (two runs per build, one pinned P-core) showed no
    difference beyond run-to-run noise (flat float64 at 100K 16.9 / 18.7 µs against 22.3 / 20.2 µs on the old base).
  - Behavior change outside NumPy: Decimal (no NumPy dtype) raises `DivideByZeroException` for an empty input or
    `ddof >= n` (decimal has no NaN or infinity; the old engine returned a float64 NaN for an empty input and a
    NEGATIVE decimal variance for `ddof > n`).
  - Left open: finding 5 and a broadcast operand's sum (inherited, B1); item X6 (cast-loop rows longer than the
    buffer, inherited, pinned); `ddof` as a float (finding 6, B4); `out=` / `where=` / `mean=` / `correction=` (B4);
    `np.ma.var` / `np.ma.std` (A6); the unused var/std kernels (X8).
- [x] **A3 `np.nanstd` / `np.nanvar`: `axis=` and `keepdims=` give the wrong shape** — and, as the proof below
  found, the values, the NaN rules and `dtype=` too. Pin: `NanStdVar_1DAxis_ReturnNumPysShape`; oracle excuses: the
  nan-family branches (W4-A shape, W4-B value, dtype). Done 2026-10-05 (`d1759ec8`).
  - Proof (2026-10-05, at `41785501`): a 45,832-case NumPy 2.4.2 matrix (`nanvar`/`nanstd` × float16/32/64,
    complex128, bool and five integer dtypes × NaN patterns — none, one, every third, all, a NaN row, ±inf, a complex
    NaN or inf part, `-0.0`, float16 overflow — × shapes `()` to `(2,3,2)` and empties × axis forms × keepdims × `ddof`
    0, 1 and past the non-NaN count × C/F/stepped/reversed/transposed × `dtype=`). NumSharp: 33,013 exact, 1,005 within
    64 ULP, 7,666 grossly different, and the 4,148 `dtype=` cases could not be called (no parameter). The gross
    classes: keepdims dropped for every integer input and for a flat or one-element float input; a 1-D axis gave
    `(1,)`; `ddof` at or past the non-NaN count gave 0 instead of NaN; a ±inf element gave 0 instead of NaN; float16
    overflow stayed finite; an axis on a 0-d integer input did not raise. The ULP class: float16/float32 summed in
    double, complex squared as `re² + im²`.
  - NumPy's `nanvar` (`numpy/lib/_nanfunctions_impl.py`): `_replace_nan` (a K-order copy, NaN -> 0) — an input that
    is not inexact has no mask and returns `np.var` —, the `dtype` check (TypeError "If a is inexact, then dtype must be
    inexact"), `cnt = np.sum(~mask, axis, dtype=intp, keepdims=True)`, `avg = np.sum(arr, axis, dtype, keepdims=True) /
    cnt` (in place, `casting='unsafe'`), `np.subtract(arr, avg, out=arr, casting='unsafe')` — the deviations are
    rounded into the INPUT's dtype —, zeros at the NaNs again, `np.multiply(arr, arr, out=arr)` (complex:
    `np.multiply(arr, arr.conj(), out=arr).real`), `var = np.sum(sqr, axis, dtype, keepdims)`, `dof = cnt - ddof`,
    `var / dof` in place, and `np.nan` wherever `dof <= 0`; `nanstd` takes `sqrt(var, out=var)` (a scalar:
    `var.dtype.type(np.sqrt(var))`).
  - Fix (`Default.Reduction.NanVarStd.cs`, `NanVarStdNumPy`; `np.nanvar` / `np.nanstd` are thin wrappers and gained
    `dtype=`, B7's first part): those statements, in order. A non-inexact input (and NumSharp's decimal, which holds no
    NaN) takes `np.var` / `np.std` (A2). The axis follows ufunc.reduce's rule, because nanvar's first axis consumer is
    `np.sum`: on a 0-d input axis 0 and -1 reduce flat, any other axis is AxisError. Both sums run over NumPy's K-order
    copy (`Shape.KeepOrder()`): a dense input (one memory run flat, C/F along an axis) without a cast is that copy
    already, so the mean's sum streams over the input with its NaNs zeroed and a flat variance streams the squared
    deviations too (three reads, no buffer); otherwise the copy is materialized for `ReduceAdd` and the squared
    deviations are written by one fused `np.evaluate` pass into a K-order buffer (`Cast(a - avg, T)`, then `Square` or
    `Real(d * Conj(d))` — both proved bit-exact against NumPy's `simd_cmul`, 0 of 1,036 values differ). float16 and
    hosts without dynamic code take NumPy's own eager statements. The divisions take the float64 loop (int64 counts)
    and Smith's complex loop; `dof <= 0` writes the positive quiet NaN.
  - Proof of the fix: the 45,832-case matrix — all exact, errors included; a 2,872-case large-size matrix (up to
    1,000,003 elements; sparse NaNs, all-NaN rows and columns; C/F/stepped/reversed/transposed incl. 3-D; `dtype=`
    casts; keepdims) — all exact; a 10,306-case view-chain matrix (every 3-D permutation with and without a reversed
    axis, steps, offsets, broadcasts, 0-d, 5-D; NaN-sparse and all-NaN-slice bases) — all exact and none inherited
    (finding 9).
  - Gates: oracle tier `nanvarstd.jsonl` (13,382 cases, zero excuses: every catalog layout refilled with finite values
    and NaNs × dtype × axis × keepdims × `ddof` {0, 1}; NaN patterns; `ddof` past uneven non-NaN counts; `dtype=` for
    every (input, loop) pair; special values; `np.setbufsize` fills; order-sensitive sizes; empty inputs; 0-d axes;
    the char proxy); `nanvar` / `nanstd` left the registry's nan-family blankets (`NanReduceOps`) and joined
    `ByteExactArithmeticUnaryOps` (tightness test `NanVarStd_NoDivergenceIsExcused`, which fails when they are put
    back); `reduce_start.jsonl` gained the 56 nanstd/nanvar cells it carved (8,666); unit tests
    `Math/np.nanvar.NumPyPort.Test.cs` (12: the moved pin, NaN for no degrees of freedom and its positive bits, inf,
    `dtype=` rounding, the complex square, the TypeError, the 0-d axes, integer inputs, float16, empty inputs, the
    permuted layout).
  - Speed (NPY/NS, one pinned P-core, best of rounds; 86 cells: float64 / float32 / float16 / complex128 / int32 ×
    {nanvar and nanstd flat, nanvar per axis} × 1K / 100K / 10M, plus F / strided / `dtype=` forms of the two float
    dtypes; NaNs every 7th element): geomean 3.71×, minimum 1.51× (int32 along an axis at 1K: `np.var`'s route, where
    NumPy's nanvar is np.var too). NumPy's nanvar makes about 11 passes over the data; flat float64 at 100K 9.1×, at
    10M 5.2×; float64 along axis 0 at 10M 2.75×; float16 along axis 1 at 1K 1.72× (the eager route).
  - Left open: B7's `out=` / `where=` / `mean=` / `correction=` and NumPy's order; `ddof` as a float (finding 6);
    np.nanmean's integer keepdims (A7, found here).
- [x] **A4 max/min family: `out=`, `initial=`, `where=` and NumPy's order** for `np.max`, `np.min`, `np.amax`,
  `np.amin`, `np.nanmax`, `np.nanmin`, `ndarray.max`, `ndarray.min`. NumPy: `(a, axis=None, out=None, keepdims,
  initial, where)`; NumSharp: `(a, axis, keepdims, dtype)`. Pins: the 4 `MaxMinFamily_*` tests. Done 2026-10-05
  (`1dcaf0d8`).
  - [x] A4a NumPy's parameter order + `out=`
  - [x] A4b `initial=`
  - [x] A4c `where=`
  - Proof (2026-10-05, at `a0d64cbd`): the parameters did not exist — `np.max(a, axis=0, out=o)`, `initial=` and
    `where=` do not compile, and the 4 pins (bound to NumPy's signature by reflection) failed. Building the oracle tier
    against plain `np.max` found three more divergences behind the keywords: complex128 along a reversed axis returned
    another NaN than NumPy's first (the IL kernels walk memory order), a broadcast float input lost NumPy's NaN payloads
    and ±0 ties (the old kernels match only the value), and the fresh result of an axis reduction over a permuted input
    was C-ordered where NumPy keeps the input's axis order (292 cases of the tier's strides facet; library-wide, see
    the fix). `np.max(bool, axis=1)` was 0.19× NumPy at 10M (no early exit).
  - NumPy (`numpy/_core/fromnumeric.py::_wrapreduction`, `umath/reduction.c`, `nditer_constr.c`): `np.max` IS
    `np.maximum.reduce` (`np.min` minimum, `np.nanmax` / `np.nanmin` of an ndarray `np.fmax.reduce` / `np.fmin.reduce`),
    so the keywords are `PyUFunc_ReduceWrapper`'s, in its order: the reduce prologue (A1's `ReductionArguments`); the
    loop dtype `promote_types(out, a)` (out acts as the first input); `PyArray_Pack(initial)` into the loop dtype (a
    Python int range-checked, a Python float truncated, NaN / inf refused for an integer loop, a NumPy scalar
    `int()`-converted by a SIGNED integer loop and cast by every other, a 0-d array cast, a str parsed, a sequence
    refused with NumPy's texts); no identity: a mask raises "... does not have an identity, so to use a where mask one
    has to specify 'initial'", an empty reduced axis "zero-size array to reduction operation ..."; the result starts as
    the initial value or as each output's first element (`PyArray_CopyInitialReduceValues`, an unsafe cast from the
    input), then ONE buffered NpyIter over `[out, a, where]` folds the rest in: inner-loop calls whose cuts decide the
    float bits (each `simd_reduce_c` call re-canonicalizes a NaN; a ±0 tie depends on the call's vector section),
    masked calls cut into runs of selected elements (`generic_masked_strided_loop`), and a cast out round-trips through
    its own dtype at every buffer fill unless the buffer is reused (`NPY_OP_ITFLAG_BUF_REUSABLE`: only inside the reduce
    double loop, with the previous fill's pointers). A fresh result is laid out in the iterator's axis order
    (`npyiter_new_temp_array`).
  - Fix: `np.max.Keywords.cs` / `NDArray.max.Keywords.cs` (NumPy's form at overload priority 1; NumSharp's positional
    `(a, axis, keepdims)` and its `dtype=` extension stay, documented there); `DefaultEngine.MaxMinNumPy`
    (`Default.Reduction.MinMax.Keywords.cs`): NumPy's statements in order, then one of four routes — a broadcast input
    whose answer cannot depend on how often an element recurs reduces one copy (`MaxMinBroadcastShortcut`: every loop
    but float32 / float64 always; those only when every NumPy call is elementwise); an integer input under a wider
    loop reduces in its own dtype and casts (`max` commutes with a monotone cast); a same-dtype call without `where=`
    runs the exact engines seeded with the initial value (`TryMaxMinSeeded`: np.max's speed); everything else runs
    `NumPyReduceIterator`, a port of NumPy's reduction iterator (axis sort with C winning ties, coalescing,
    `npyiter_find_buffering_setup`, fills, the reduce double loop, the first-visit skip, masked runs, the out buffer's
    reuse, SIMD cast kernels for buffer copies) over NumPy's loops: the SIMD rules for float32 / float64 / integers,
    `logical_or` / `logical_and` for bool, sequential `BINARY_LOOP`s for float16 / char / decimal, vectorized complex128.
    `ReductionInitial` is `PyArray_Pack`. Also fixed: plain `np.max` / `np.min` of complex128 and of a broadcast input
    go through the iterator (NumPy's bits), plain bool along its innermost axis too (early exit), np.nanmax / np.nanmin
    of a broadcast float input; `AllocateReductionResult` lays a fresh axis result out like NumPy for every reduction
    that uses it (`Shape.ReductionOutputShape`: np.sum / np.mean / np.var / the nan-family along an axis of an F-ordered
    or permuted input were C-ordered; `ReductionKeepOrderTests` now asserts NumPy's layout).
  - Performance work (each route proven bit-identical to NumPy's per-call structure, then measured): the masked calls of
    every loop in one piece where the loop is order-free (integers, bool, sequential) and, for the float SIMD loops,
    NumPy's per-run calls through per-run transfer functions (a run's own work leaves the out's dependency chain:
    `NRunContiguous` / `PRunContiguous`, closed forms of `simd_reduce_c` derived from the associativity of the lane
    ops); several input rows folded per call where the out does not move (`Loop.BinaryRows`); complex128 in two vector
    passes per 1024-element block (the fold of a block is one of its elements: the first NaN, else the first extreme)
    and a vectorized elementwise keep-blend; the out's per-core buffer copies as one strided conversion.
  - Proof of the fix: the tier below, every case exact. Randomized probes against the per-call reference: complex
    loops 64,000 cases, several-row folds 42,000, masked runs 192,000 — 0 mismatches. Planted bugs: 24 mutants over the
    iterator (buffer reuse, run cuts, first visit, buffer cap, C-wins rule), the initial conversion and the wide paths
    (complex lane order and NaN path, the run transfer functions' ties and canonicalization, the row pairing, the
    broadcast shortcut, the cast rewrite, the strided walk, the mask scan) — every one turns 1–558 tier cases red.
  - Gates: oracle tier `minmax_kwargs.jsonl` (35,847 cases, zero excuses: every catalog layout × axis × keepdims × the
    keyword forms for the four functions; mask forms; out= of every dtype pairing in C / strided / F / reversed layouts;
    the float bits under where= runs; `np.setbufsize` fills and a narrowing out past 8,192 elements; initial= of every
    kind into every loop dtype; NumPy's errors in order; an out overlapping the input; the aliases; the fresh result's
    layout; 3,000 seeded random combinations; section W sized for the wide paths; the char proxy), strict in
    `MisalignedRegistry`; coverage key `minmax_kwargs` (np.amax / np.amin lost their reviewed aliases: they are gated
    directly); unit tests `Selection/np.max.Keywords.Test.cs` (11: the 4 former pins, the LP64 overflow texts, NumPy
    scalars, the bool literals, the cast-out round trip, complex NaN order, an empty result).
  - Speed (NPY/NS, both sides pinned to two P-cores — finding 12; 237 cells: init= / where= / out= / cast out / nanmax /
    complex / broadcast / bool forms × float64 / float32 / int32 / int8 / float16 / bool / complex128 × flat and both
    axes × 1K / 100K / 10M, plus plain-np.max twins): geomean 3.98×, minimum 0.93×. where= 1.5–10,000× (NumPy cuts every
    call into runs; a bool OR stops at the first selected True); complex128 1.5–3.9× (were 0.97–1.37×); a broadcast
    integer input 10–560×; an integer input into a wider out 3.6–11.6×. Of the 34 cells under 1.5×, 27 ride plain
    np.max's engine and match their plain twins (10M inputs bound by L3 / DRAM bandwidth — a float64 10M stream is at
    one core's ~27.8 GB/s — and plain np.max's documented 100K slab gaps); 7 are keyword paths that are NumPy's own
    structure: a float64 → float32 out (a round trip at every fill; 1.0–1.3×), a broadcast float64 input reduced flat
    (each fill reduces the row copied into the buffer 8 times; 1.1–1.3×), bool along axis 0 at 10M (1.02×, L3-bound).
  - Left open: plain np.max's 10M / slab gaps (its engine, documented under "Math — Reductions"); the cast-out and
    broadcast-flat float routes above; a 0-d bool ARRAY holding true is read as Python's `where=True` (C# cannot tell
    `np.array(True)` from the literal; `[Misaligned]`, NumPy raises the no-identity error without initial=); X10
    (np.any / np.all keep C order: they do not use `AllocateReductionResult`).
- [x] **A5 `np.trapezoid`: an `axis=` outside `y`.** NumPy raises `IndexError: list assignment index out of range`;
  NumSharp raises `ArgumentOutOfRangeException`. Pin: `Trapezoid_AxisOutsideY_RaisesNumPysIndexError`. Done
  2026-10-05 (this commit).
  - Proof (2026-10-05, at `76b03855`): a 480-case NumPy 2.4.2 matrix (y of rank 0–3 × axis −6..5 × ten x forms: none,
    a 1-D x of the axis's length / of 5 / empty / of shape `(4,)`, a 0-d x, N-D x of shapes `(1,4)`, `(3,4)`,
    `(2,3,4)`, `(2,2,3,4)`), each outcome compared as type + text, or dtype + shape: 217 equal, 224 differ — every one
    NumPy's `IndexError: list assignment index out of range` against NumSharp's `ArgumentOutOfRangeException("axis N is
    out of bounds for array of dimension M")`. The x errors that come first (diff's AxisError for an axis outside x as
    well, its ValueError for a 0-d x) and every result were already equal.
  - NumPy's `trapezoid` (`numpy/lib/_function_base_impl.py`) never checks the axis. It assigns into Python lists of
    length `y.ndim` — `shape[axis] = d.shape[0]` for a 1-D x (after `d = diff(x)`), then `slice1[axis] =
    slice(1, None)` — so Python's list assignment raises, after an N-D x's `diff(x, axis=axis)`. NumSharp already ran
    the statements in that order.
  - Fix (`Math/np.trapezoid.cs`, `TrapezoidAxis`): raise `IndexError("list assignment index out of range")`, the
    house's leaked-Python-error rule (`np.tensordot`, `np.fft`). The code that computes a result is unchanged.
  - Proof of the fix: the 480-case matrix — all equal.
  - Gates: 106 `err_axis_outside_y` cases appended to the `multioutput` oracle tier (687 cases, floor 680; 78 of them the
    IndexError, 12 diff's AxisError and 16 its ValueError, which still come first; every y rank × both ends of the axis
    × none / 1-D / empty 1-D / N-D x that admits the axis / N-D x that does not / 0-d x, plus seven other dtypes); the
    generator no longer skips NumPy's IndexError; trapezoid excluded from the registry's K7 branch (finding 15); the pin
    moved to `Math/np.trapezoid.Test.cs` without `[OpenBugs]` and widened to 14 cases; `Trapezoid_0D_Throws` and
    `Trapezoid_XErrorsComeBeforeYAxisErrors` assert the type and text again. With the old error planted back, 78 tier
    cases turn red (66 before the K7 change).
- [ ] **A6 `np.ma.var` / `np.ma.std` with a mask: the masked path is not NumPy's.** `NDMaskedArray.var` computes its
  own formula (it never reaches `np.var`): 242 cases of the ma oracle tier differ — a slice NumPy masks (all masked,
  or `count - ddof <= 0`) carries NaN instead of the masked 0, and some slices are finite where NumPy has +inf. They
  were hidden by the registry's unbounded "summation precision" blanket until A2 removed var/std from it (finding 7);
  `MisalignedRegistry.IsMaVarStd` keeps the blanket for the ma ids only and names this item.
- [x] **A7 `np.nanmean`: an integer or bool input drops `keepdims` for `axis=None`.** NumPy keeps the reduced axes
  (shape `(1,)*ndim`); NumSharp returns 0-d. 117 cases of the nanreduce oracle tier (every integer dtype and bool ×
  13 layouts) sit under the registry's (W4-A) shape branch, which names only this bug since A3. Found by A3. Done
  2026-10-05 (`a7f11b3e`), with B6's `dtype=`.
  - Proof (2026-10-05, at `d1759ec8`): an 11,415-case NumPy 2.4.2 matrix (`nanmean` over float16 / float32 / float64 /
    complex128, bool and integer dtypes × NaN patterns as A3's × shapes `()` to `(2,3,2)` and empties × axis forms ×
    keepdims × C / F / stepped / reversed / transposed × `dtype=`): 9,429 exact (errors included), 207 within 64 ULP,
    538 grossly different, and the 1,241 `dtype=` cases could not be called. The gross classes: keepdims dropped for
    a flat integer / bool input (240); a complex mean with an infinite part (264: NumPy `(inf, nan)` through its complex
    division, NumSharp `(inf, -0.1)`); float16 overflow stayed finite (30); 4 axis errors. The ULP class: float16 /
    float32 summed in double.
  - NumPy's `nanmean` (`numpy/lib/_nanfunctions_impl.py`): `_replace_nan(a, 0)` — an input that is not inexact has no
    mask and returns `np.mean(arr, axis, dtype, out, keepdims)` —; the `dtype` / `out` checks (TypeError "If a is
    inexact, then dtype must be inexact"); `cnt = np.sum(~mask, axis, dtype=intp, keepdims=keepdims)`; `tot =
    np.sum(arr, axis, dtype, out, keepdims)`; `avg = _divide_by_count(tot, cnt, out)` (`divide(tot, cnt, out=tot,
    casting='unsafe')`; an all-NaN slice keeps the division's `0 / 0`, NaN).
  - Fix (`Default.Reduction.NanMean.cs`, `NanMeanNumPy`; `np.nanmean` is a thin wrapper and gained `dtype=`): those
    statements on A3's routes. A non-inexact input (and decimal) is `ReduceMean`, which is NumPy's `_mean` since X9 —
    X9 was found here: the probe's last ULP case was `np.mean(int32, axis=0, dtype=complex128)`. The axis follows
    ufunc.reduce's rule (np.sum is nanmean's first axis consumer). The count streams over the input; a dense input
    without a cast streams its NaN-zeroed sum (np.evaluate), anything else materializes NumPy's K-order copy for
    `ReduceAdd`; float16 and hosts without dynamic code take NumPy's eager statements. The division is A3's
    `DivideByCountsLikeNumPy`. The eager zero fill is `np.putmask` with a scalar (`ZeroWhereMask`, shared with A3's
    eager nanvar route): `np.copyto(where=)` cost 253 µs over a float16 `(250, 400)` and `np.putmask` 21 µs (X11).
  - Proof of the fix: the 11,415-case matrix — all exact, errors included; a 1,078-case large-size matrix (up to
    1,000,003 elements; sparse NaNs, all-NaN rows and columns; C / F / stepped / reversed / transposed; `dtype=` casts;
    keepdims) — all exact; a 3,451-case view-chain matrix — 3,450 exact, and one inherited through np.mean: a broadcast
    int32 row reduced flat with `dtype=float32` (finding 10). A3's probes rerun exact after the shared zero fill
    (45,832 + 2,872 + 10,306).
  - Gates: oracle tier `nanmean.jsonl` (5,017 cases, zero excuses: every catalog layout refilled with finite values and
    NaNs × dtype × axis × keepdims; NaN patterns incl. NaN / inf complex parts; `dtype=` for every input against nine
    loops; special values; `np.setbufsize` fills (5 X6 cells carved); order-sensitive sizes; empty and 0-d inputs; the
    char proxy); `nanmean` left the registry's nan-family set (`NanReduceOps`) and joined `ByteExactArithmeticUnaryOps`,
    the (W4-A) shape branch is removed (its last case was this bug; the nan-family value branch fell 161 → 114, the rest
    nansum / nanprod / nanmedian); tightness test `NanMean_NoDivergenceIsExcused`; unit tests
    `Math/np.nanmean.NumPyPort.Test.cs` (5).
  - Speed (NPY/NS, one pinned P-core, best of rounds; 72 cells: float64 / float32 / float16 / complex128 / int32 × flat
    and both axes × 1K / 100K / 10M, plus F / strided / `dtype=` forms of the two float dtypes; NaNs every 7th element):
    geomean 5.82×, minimum 1.70× (float64 → float32 `dtype=` at 10M: NumPy's cast loop on both sides). NumPy's nanmean
    makes about six passes; flat float64 at 100K 18×, float32 along an axis at 10M 9–11×, float16 along axis 1 at 100K
    6.3× (was 1.37× before the putmask fill). The A3 float16 cells improved with it (nanvar flat 100K 3.4× → 8.4×).
  - Left open: B6's `out=` / `where=` and NumPy's parameter order; the unused nanmean struct kernels (X8); finding 10
    reaching an integer input through np.mean.

### Phase S: signature gate

- [x] **S2 `reduce_kwargs` oracle tier** (`gen_oracle.py reduce_kwargs`, `OpRegistry.ReduceKwargs.cs`): each
  reduction through its NumPy-shaped overload with out= / where= / axis forms / keepdims and the reduce-prologue
  errors. Started with A1 (`np.all`, `np.any`); each later reduction item adds its function to `REDUCE_KWARGS_FNS`.
- [ ] **S1 Signature-parity gate** for `np.*` and `ndarray` methods (the model: `LinAlgSignatureParityTests` and the
  random-API G3 gate). NumPy's signatures in a committed JSON; one reasoned exemption per open item of this plan,
  each exemption self-retiring.

### Phase B: reductions

- [ ] B1 `np.sum`, `ndarray.sum`: `out=`, `initial=`, `where=`; NumPy's order `(a, axis, dtype, out, keepdims, …)`.
- [ ] B2 `np.prod`, `ndarray.prod`: `out=`, `initial=`, `where=`.
- [ ] B3 `np.mean`, `ndarray.mean`: `out=`, `where=`; and `dtype=` without an axis (no overload takes it: `np.mean(a,
  dtype=np.float32)` does not compile; the axis forms do). The `dtype=` loops themselves are NumPy's since X9 (the
  engine's np.mean takes every axis form).
- [ ] B4 `np.std`, `np.var`, `ndarray.std`, `ndarray.var`: `out=`, `where=`, `mean=`, `correction=`; NumPy's order
  `(a, axis, dtype, out, ddof, keepdims)`.
- [ ] B5 `np.nansum`, `np.nanprod`: `dtype=`, `out=`, `initial=`, `where=`.
- [ ] B6 `np.nanmean`: `dtype=`, `out=`, `where=`. (`dtype=` landed with A7.)
- [ ] B7 `np.nanstd`, `np.nanvar`: `out=`, `where=`, `mean=`, `correction=`; NumPy's order. (`dtype=` landed with A3.)
- [ ] B8 `np.argmax`, `np.argmin`, `ndarray.argmax`, `ndarray.argmin`: `out=`.
- [ ] B9 `np.cumsum`, `np.cumprod`: parameter names `a` and `dtype` (today `arr` and `typeCode`).
- [ ] B10 `np.average`: `returned=` (one result type for NumPy's array-or-tuple return, as `np.unique` does).
- [ ] B11 `np.percentile`, `np.quantile`, `np.nanpercentile`, `np.nanquantile`: `weights=`.

### Phase C: sorting

- [ ] C1 `np.sort`, `ndarray.sort`: `stable=`.
- [ ] C2 `np.argsort`, `ndarray.argsort`: `kind=`, `stable=`; first parameter named `a` (today `nd`).

### Phase D: creation

- [ ] D1 `np.empty`, `np.zeros`, `np.ones`, `np.full`: `order=` at NumPy's position `(shape, dtype, order)`.
- [ ] D2 `np.empty_like`, `np.zeros_like`, `np.ones_like`, `np.full_like`: `shape=`; `order=` at NumPy's position.
- [ ] D3 `np.linspace`: `retstep=` (array-or-tuple result type), `axis=`.
- [ ] D4 `np.indices`: `sparse=` (array-or-tuple result type; `np.indices_sparse` exists today).
- [ ] D5 `np.asanyarray`: `copy=`.
- [ ] D6 `np.array`: `ndmax=`; first parameter named `object`.
- [x] D7 `np.ndarray(shape, dtype, buffer, offset, strides, order)`: `offset=`, `strides=`. Unblocked: the view-`Shape`
  plan is done (2026-10-05). It adds three follow-ups of its bug 1 (`docs/plans/view-shape-open-bugs.md`, "Not in this
  bug"):
  - B1.10: a buffer that is too small must raise `TypeError: buffer is too small for requested array` (NumSharp:
    `ArgumentException: values.Length does not match shape.Size`).
  - B1.11: a buffer that is larger than the shape must give an array over its prefix (NumSharp raises the same
    `ArgumentException`).
  - NumSharp's `np.ndarray(shape, dtype, buffer, order)` defaults to `order='F'` and float32 (NumPy: `order=None`, which
    is C, and float64), and ignores `order` without a buffer (NumPy: `np.ndarray((2,3), order='F')` is F-contiguous).

  **Done by a parallel session** in `f17c3b2d` (its own plan, `docs/plans/np-ndarray-parity.md`, items N1–N13 — N6 is
  B1.10 / B1.11, N7 / N8 are `offset=` / `strides=`; its oracle tier `ndarray_ctor.jsonl` has zero excuses). Checked
  here after the fact with the same inputs as above: `np.ndarray((2,3))` is float64 and C-contiguous,
  `order: 'F'` without a buffer is F-contiguous, `offset: 8` and byte `strides: [8]` over a 48-byte buffer read
  `[185207048 252579084]` and `[50462976 185207048 319951120]` (NumPy's values), a 100-element int32 shape over it
  raises `TypeError: buffer is too small for requested array`, and a 2-element uint8 shape reads the prefix `[0 1]`.

### Phase E: manipulation

- [ ] E1 `np.stack`: `out=`, `dtype=`, `casting=`.
- [ ] E2 `np.hstack`, `np.vstack`: `dtype=`, `casting=`.
- [ ] E3 `np.reshape`: `order=`, `copy=`; `ndarray.reshape`: `copy=`.
- [ ] E4 `np.transpose`: parameter `axes` (today misspelled `premute`).

### Phase F: ufunc-shaped functions

- [ ] F1 `np.frexp`, `np.modf`: NumPy's `out=(o1, o2)` tuple form (today `out1`/`out2`, `outFrac`/`outIntegral`);
  `np.frexp` `dtype=`.
- [ ] F2 `np.ldexp`: `dtype=`.

### Phase G: names and positions

- [ ] G1 Parameter names: `np.around(a)`, `np.array_equal(a1, a2)`, `np.array_repr(arr)`,
  `np.broadcast_to(array, shape)`, `np.can_cast(from_)`, `np.iinfo(int_type)`,
  `np.iscomplex` / `np.isreal` / `np.iscomplexobj` / `np.isrealobj(x)`, `np.isscalar(element)`, `np.put(ind, v)`.
- [ ] G2 `ndarray.astype`: NumPy's order `(dtype, order, casting, subok, copy)` (today `(dtype, copy, order, casting)`).
- [ ] G3 `ndarray.dot`: `out=`.
- [ ] G4 A bare `null` axis (NumPy's positional `None`) is ambiguous (CS0121) where an `int?` and an `int[]` overload
  share the position: `np.average`, `np.average_returned`, `np.flip`, `np.gradient`, `np.median`, `np.nanmedian`,
  `np.percentile`, `np.quantile`, `np.nanpercentile`, `np.nanquantile`, `np.ptp`, `np.trim_zeros` (found by A1, which
  fixed `np.all` / `np.any`). Confirm each with a compile probe first: other parameters may already disambiguate.

### Phase H: rarely used

- [ ] H1 `np.array2string`: `formatter=`, `legacy=`; `np.set_printoptions`: `formatter=`, `legacy=`, `override_repr=`.
- [ ] H2 `np.vectorize`: `doc=`, `excluded=`, `cache=`.
- [ ] H3 `np.dtype`: `align=`, `copy=`.

### Phase X: parameters that exist but may not work

- [ ] X1 Audit: every existing `out=`, `where=`, `dtype=`, `keepdims=`, `axis=`, `ddof=`, `casting=` parameter against
  NumPy (returned object, values, dtype, shape, errors). Each finding becomes an item here.
- [ ] X2 `np.maximum`, `np.minimum`, `np.add` (binary ufuncs): an out-of-range C# integer literal wraps
  (`np.maximum(int8, 1000)` gives -24); NumPy raises `OverflowError` (NEP 50 packing). Found by the `np.clip` work.
- [x] **X3 binary ufuncs: two F-ordered operands where one broadcasts give WRONG values.** Found by A2 (the `_var`
  port subtracts a keepdims mean from an F-ordered input). Done 2026-10-05 (`d3c863ef`).
  - Proof: `np.subtract(F (2,3,2), F (2,1,2))` paired the elements wrongly (NumPy `[-100, -199, -98, …]`, NumSharp
    `[-100, -294, -98, …]`). A 2,814-case probe (every binary function × 9 dtypes × 7 shape pairs × both operand
    orders, against the same call on C copies): 824 wrong — `add` / `subtract` / `multiply` / `true_divide`,
    `maximum` / `minimum` / `fmax` / `fmin`, `bitwise_and` / `or` / `xor` and the C# operators, every 3-D and 4-D
    pair, every SIMD dtype under the small-broadcast cutoff (8,192 elements), float16 at any size. 2-D pairs,
    complex, bool and the scalar-only ops were right.
  - Cause: the direct binary route allocated the result in F order when every non-scalar operand was strictly F
    by its OWN flags — a `(2,1,2)` F array is — but its kernels write the result linearly in C-coordinate order
    over the coalesced dims, and the broadcast axis (stride 0) keeps the dims from coalescing into one run.
  - Fix (`DefaultEngine.BinaryOp.cs`): the direct route allocates F only when every non-scalar operand has the
    result's own dims (`HasNonScalarBroadcastOperand`); otherwise it computes C and the existing post-kernel branch
    relabels it F (NumPy's layout). The small-broadcast shortcut no longer sends an all-F set to the direct route:
    `NDIter` allocates it F with no copy (16³ float64: 1.4 µs, NumPy 2.7 µs; the direct route + copy measured 7.1).
  - Gate: the `pp_fbcast_*` pair layouts of the binary oracle tiers, 3,074 cases (388 diverge before the fix), and
    `Backends/Kernels/FOrderBroadcastBinaryTests.cs` (3 tests, NumPy-probed literals; all 3 fail before the fix).
- [x] **X5 `np.sum(a, axis, dtype=…)`: an axis that leaves one output, in a cast loop, is not NumPy's order.** Found
  by A2's large-size check (`np.var(x, axis=0, dtype=np.float32)` of a 1-D float64 array). Done 2026-10-05 (`4d06d591`).
  - Proof: `np.sum(x, axis=0, dtype=np.float32)` of a 1-D float64 array of 20,000 elements gave `0x4706ada8` where
    NumPy (and NumSharp's own `axis=None`) give `0x4706ada6`; the same for `(n,1)` along 0, `(1,n)` along 1 and
    `(1,n,1)` along 1, at 20,000 and 100,003 elements. The cast-free forms, float32 → float64 and int → float matched.
  - Cause: NumPy's iterator drops extent-1 axes, so an axis that leaves one output runs the FLAT schedule — a cast
    loop in fills of `np.getbufsize()`, `*out += pairwise_sum(fill)` each. NumSharp's axis cast route cast a whole
    copy and summed it in one pairwise call. (`.claude/CLAUDE.md` listed "an axis reduction in a cast loop whose
    reduced runs exceed the buffer is not modeled" as a residual; this is its one-output part.)
  - Fix (`Default.Reduction.Add.cs`, `TryExactOneOutputAxisSum`): a cast loop over such an axis runs the flat engine
    and returns the axis result's shape; so does a REVERSED axis without a cast (the axis kernel walks it in memory
    order, NumPy in logical order — finding 5; the new oracle section found 32 such cases). Cast-free forward calls
    keep the axis kernel (already exact, and 10–20 % faster there). NPY/NS for 1-D float64 → float32: 4.5× at 1K, 2.0× at 100K, 1.3× at 10M (the old route took 1.37 µs /
    20.6 µs / 6.8 ms; NumPy 2.0 µs / 37 µs / 5.9 ms).
  - Gate: section "o" of the `sum_identity` oracle tier (832 cases under `np.setbufsize(16/32)`; 248 diverge before
    the fix) and `np_sum_flat_schedule_tests.Sum_AxisLeavingOneOutput_RunsTheFlatSchedule` (17 NumPy-probed cases).
  - Open (the rest of that residual): an axis reduction with SEVERAL outputs whose reduced runs exceed the buffer in
    a cast loop (`(2, 20000)` along 1 with `dtype=float32`): item X6.
- [ ] X4 float16 `add` / `subtract` / `multiply` / `divide` with a broadcast, non-contiguous operand set run the
  direct route's scalar General loop: `(30,40,20) F * (30,1,20) F` takes 328 µs (NumPy 72 µs, 0.22×), and
  `(100,100,100)` 27 ms (NumPy 3.0 ms). The same before X3 (HEAD measured 320 µs while computing wrong values).
  Speed only; the values are NumPy's.
- [ ] **X6 `np.sum(a, axis, dtype=…)`: a pinned row (the reduced axis innermost) longer than `np.getbufsize()` in a
  cast loop.** NumPy's buffered fills split the row (`out += pairwise(part)` per fill); NumSharp's cast-loop axis
  route sums it in one call. `np.sum(x (8,25), axis=1, dtype=np.float32)` under `np.setbufsize(16)` differs at 3 of 8
  sums; `np.var` / `np.std` with that `dtype=` inherit it. Pin:
  `OpenBugsReductionTests.Sum_CastLoopAxis_RowLongerThanTheBuffer_SplitsAcrossFills`; the varstd tier carves the
  cells (section B). Found by A2's oracle tier; the several-output part of X5's residual.
- [ ] **X7 `np.sum(bool / int, axis=0, dtype=float64)` (a SLAB cast loop) is slower than NumPy.** At `(2500, 4000)`:
  bool 27.3 ms (NumPy 5.0, 0.18×), int32 9.1 ms (NumPy 4.3); `astype(bool → float64)` takes 11.4 ms (no SIMD kernel).
  Speed only. `np.var` / `np.std` of a bool / int input along axis 0 inherit it (bool at 10M: 1.38× NumPy, the only
  10M var/std cell under 1.5×).
  - Part done with X9 (2026-10-05), the float64 loop np.mean's default needed: an integer source of 32 bits or less
    whose largest possible sum `L × max|x|` is within 2^53 (every partial sum exact, so every order gives NumPy's
    sum), or an int64 / uint64 source along a non-innermost axis (every route of the widening kernel is sequential per
    output, NumPy's slab order), runs the axis widening kernel instead of the cast copy
    (`IsExactIntegerWideningAxisSum`); a bool source along an axis counts its nonzero elements with np.evaluate's
    CountNonzero axis stream (C / F operands, NumPy's K-order result; `TryCountBoolAxisSum`), in the int64 loop too.
    np.mean at (2500, 4000) along axis 0: bool 67.6 ms → 0.18 ms (NumPy 5.1, 28×), int32 3.16×, int64 1.70×.
  - Left open: the float32 / float16 cast loops along a non-innermost axis (the blocked cast copy gathers column
    blocks): np.mean(int32, axis=0, dtype=float32) 0.31× and np.mean(float16, axis=0) 0.80× NumPy at 10M; an int64
    innermost axis (the pairwise order needs the cast copy) 1.20×. The fix is NumPy's own schedule: cast a block of
    rows along the reduced axis and add it into the running result.
- [ ] **X8 The var/std kernels A2 no longer reaches.** `DirectILKernelGenerator.VarSimdHelper` / `StdSimdHelper`
  (`Masking.VarStd.cs`), the `ReductionOp.Var` / `ReductionOp.Std` axis kernels (`Reduction.Axis.VarStd.cs`) and
  `NDAxisIter`'s `VarAxisDoubleKernel` / `StdAxisDoubleKernel` computed var/std in their own order; nothing calls them
  since the port. A3 left the same for nanvar / nanstd: `NDNanReductionKernels`' four `NanSquaredDeviation*Kernel`
  structs, `DirectILKernelGenerator.NanVarSimdHelperFloat` / `NanVarSimdHelperDouble` (`Masking.NaN.cs`) and the
  `ReductionOp.NanVar` / `ReductionOp.NanStd` IL kernels (no caller outside the generator). A7 left np.nanmean's:
  `NDNanReductionKernels`' `NanMeanAccumulator`, `NanMeanComplexAccumulator` and the four `NanMean*Kernel` structs.
  Decide: remove them, or keep them for a future non-NumPy fast mode.
- [x] **X9 `np.mean`: every `dtype=` loop along an axis, an integer loop flat, a flat decimal mean and a broadcast
  uint64 mean are not NumPy's.** Found by A7: np.nanmean of an integer input is np.mean, and its one ULP case was
  `np.mean(int32, axis=0, dtype=complex128)`. Done 2026-10-05 (`f8b32354`).
  - Proof (2026-10-05, at `d1759ec8`): a 9,660-case NumPy 2.4.2 matrix (`np.mean` over int8 / int32 / int64 / uint64 /
    bool / float16 / float32 / float64 / complex128 × no dtype and nine loops × shapes `(1,)` to `(2049,)` with
    length-1 axes and empties × axis forms × keepdims × C / transposed / stepped / reversed): 6,724 exact, 2,936 off.
    The classes: an axis cast loop read each element as a double and summed in one sequential pass (int32 → float16
    never overflowed where NumPy gives inf / NaN; float64 → float32 never rounded per element); a complex sum was
    divided part by part (`14 / 3` where NumPy's Smith division gives `14 * (1/3)`: 1 ULP, and an infinite part did
    not turn the other into NaN); float16 summed in double; an integer or bool loop on an empty input threw
    `InvalidCastException` along an axis and gave a float64 NaN flat (NumPy: INT_MIN, True); a flat integer loop did
    not wrap; a reversed 1-D view walked memory order. Holding np.mean strict in the oracle found two more under the
    reduce tier's blankets: a flat decimal mean returned float64, and the flat fallback summed a broadcast uint64 in a
    wrapping uint64 accumulator (101.8 for NumPy's 3.69e18).
  - NumPy's `_mean` (`numpy/_core/_methods.py`): `rcount = _count_reduce_items(arr, axis)` (an `np.intp`, a STRONG
    int64; the axis normalized against the rank, so a 0-d input refuses every axis); the loop `dtype=`, else float64 for
    bool / integers and float32 for float16 (`is_float16_result`); `ret = umr_sum(arr, axis, dtype, keepdims)`; then
    `true_divide(ret, rcount, out=ret, casting='unsafe')` (and `arr.dtype.type(ret)` for a float16 result), or
    `ret.dtype.type(ret / rcount)` for a scalar.
  - Fix (`Default.Reduction.Mean.cs`, `MeanLikeNumPy`): those statements, with np.sum's `ReduceAdd` as `umr_sum`, for
    every float / complex loop along an axis (a length-1 axis keeps its exact identity add), every integer / bool loop
    on any shape, a flat decimal mean, and a flat operand the exact engine declines (a broadcast one) whenever the old
    fold is not NumPy's value (an int64 / uint64 input, a sum past 2^53, a loop other than float64). The division is
    the engine's dense-block division where it is NumPy's quotient (float64, complex, float32 while `n <= 2^24`), else
    A2's `DivideByCountLikeNumPy` (the float64 loop, NumPy's unsafe cast back; it now disposes its own temporaries).
    A float16 result rounds twice for an array and once for a scalar, as NumPy's two branches do. X7's integer / bool
    part rode along, because np.sum's cast loop became np.mean's speed limit.
  - Proof of the fix: the 9,660-case matrix 9,615 exact; the other 45 are inherited, 3 per float loop over a reversed
    view (finding 5) and 3 per bool loop over a transposed 3-D input (layout only, X10). The A7 probe's 11,415 nanmean
    cases are exact, its integer cases included.
  - Gates: oracle tier `mean_loops.jsonl` (6,550 cases, zero excuses: every catalog layout × dtype × axis × keepdims,
    `dtype=` for every input against nine loops, empty and 0-d inputs, special values, order-sensitive sizes,
    `np.setbufsize` fills; it carves 16 finding-5 cells, 5 X6 cells and 24 broadcast-flat cells, each naming its pin);
    np.mean held strict in every tier (`ByteExactArithmeticUnaryOps`; out of the reduction dtype, axis-NaN and
    summation branches; `ma.mean` keeps them, `IsMaMean`) except a narrow branch for finding 5's layout (2 reduce-tier
    cases); tightness tests `Mean_NoDivergenceIsExcused`, `Mean_ReversedViewAxis_IsTheOnlyExcusedLayout`,
    `MaMean_KeepsTheReductionBlankets_OnlyForMaIds`; unit tests `Math/np.mean.NumPyPort.Test.cs` (9); `[OpenBugs]`
    pins `Sum_ReversedViewAxis_WalksTheLogicalOrder` (finding 5) and `Sum_BroadcastOperand_Flat_IsNumPysCallSchedule`
    (finding 10).
  - Speed (NPY/NS, one pinned P-core, best of rounds; 54 cells: int32 / int64 / bool / float64 / float32 / float16 /
    complex128 along axis 0 and 1 × 1K / 100K / 10M, plus F, flat and `dtype=` forms): geomean 2.62× (the old, wrong
    routes 1.83×). The 12 cells under 1.5×: the same-type slab means at 100K and 10M (float64 / float32 / complex128
    along axis 0, 1.0–1.35×, the unchanged route; at 10M a DRAM ceiling — float64 axis 0 streams 80 MB at 26 GB/s,
    NumPy at 23, one core's limit ~27.8); X7's open float loops (int32 → float32 0.31×, float16 0.80× at 10M); an int64
    innermost axis 1.20× and a flat int64 → int32 loop 1.43× at 10M; float64 → float32 along axis 1 1.34× at 10M.
    complex128 / float64 axis 0 at 100K swing between 0.8× and 1.2× on both builds (allocator regime, interleaved A/B).
  - Left open: B3 (`out=`, `where=`, the flat `dtype=` overload); X7's float loops; X10; finding 5; finding 10.
- [ ] **X10 `np.any` / `np.all` along an axis of an F-ordered or permuted input return C order; NumPy keeps K order.**
  Probe (2026-10-05): `np.any(b.reshape(3, 5, 6).T, axis)` is F-contiguous in NumPy for every axis and C in NumSharp;
  a `(1, 0, 2)`-permuted float64 input along axis 2 gives NumPy strides `(1, 5)` (elements), NumSharp `(3, 1)`. The
  values are right, and the oracle compares C-order bytes, so only a flags or strides facet sees it. np.sum's bool
  loop (`dtype=bool`) and np.mean's bool loop are np.any and inherit it. Found by X9's probe.
- [ ] **X11 `np.copyto(dst, scalar, where=mask)` into a float16 array is slow.** 2.5 ns per element over a `(250, 400)`
  float16 with every 7th element masked: 253 µs, where `np.putmask(dst, mask, scalar)` — the same assignment for a
  same-shape mask — takes 21 µs (float32 / float64 masked copies are fast). Speed only. nanvar / nanmean's eager float16
  route now zeroes with putmask (A7, `ZeroWhereMask`); other callers of `np.copyto(where=)` into float16 keep the gap.
  Found by A7's benchmark.

## 6. Findings

1. **One reduce prologue serves every reduction** (A1). NumPy checks a reduction's arguments in one fixed order
   before it reads an element: a read-only out (while parsing `out`, so before everything else), the mask dtype, the
   axis, a repeated axis, out's rank, then the reduction iterator's two passes over `[out, input, where]`. The input is
   `NO_BROADCAST` (a mask may not make it stretch), the mask may not have more axes than the input, a kept reduced
   axis of out must be 1 long, and an out axis of length 1 cannot broadcast (the iterator has no `REDUCE_OK`).
   `ReductionArguments` ports it once; items A4 and B1–B7 reuse it.
2. **A 0-d mask is a "scalar" in NumPy's text** ("Cannot cast scalar from dtype('int64') …"), in the ufuncs too. The
   shared check said "array data" for every mask (fixed with A1). NumPy accepts a NumPy SCALAR mask (`np.int64(1)`)
   by value; NumSharp has no separate scalar type, so a 0-d non-bool mask always raises, like NumPy's 0-d array.
3. **A positional `None` axis does not compile** where an `int?` and an `int[]` overload coexist (G4).
4. **A function built from ufuncs inherits their bugs; probe the building blocks first** (A2). Composing `_var`
   from NumSharp's ufuncs found X3 in `np.subtract` within the first replay.
5. **`np.sum` along an axis of a view with a reversed axis is not NumPy's order** (found by A2's building-block
   check, 452 cases over float16/float32/float64/complex128 × C/F/stepped/reversed/transposed × flat/axis). Every
   layout is bit-exact except an axis reduction over a reversed view (`a[::-1]` reduced along an axis: 5 of 7
   per dtype differ along the reversed axis, 1–3 along the other). A `_var` port inherits it only there (the mean's
   sum; the deviation sum runs over a fresh array). Belongs with B1. Pinned since X9 by
   `OpenBugsReductionTests.Sum_ReversedViewAxis_WalksTheLogicalOrder` (`np.sum(b[::-1], axis=0)` of cancelling float32
   rows: NumPy `[0, 1, 8]`, NumSharp `[1, 0, 8]`); np.mean, held strict since X9, keeps a narrow registry branch for
   this layout in the older tiers, and the mean_loops tier carves it.
6. **`ddof` is an `int?` in NumSharp; NumPy accepts a float** (`np.var(x, ddof=1.5)` divides by `n - 1.5`). Belongs
   with B4 / B7 (`correction=` is NumPy's float alias of `ddof`).
7. **Removing an excuse is not enough; every branch that can catch the op must be checked** (A2). With V1 gone,
   np.var / np.std divergences were still excused by four generic branches: the dtype blanket for every reduction
   but sum / prod, the UNBOUNDED "summation/two-pass precision" blanket, the "axis SIMD min/max skips NaN" branch
   (any reduction with an axis whose diffs are NaN) and the unary "~ULP" branch (any one-operand op within 2 ULP).
   The fix is a tightness test that asserts `null` for the op across every kind of divergence
   (`VarStd_NoDivergenceIsExcused`). The same blanket hid `np.ma.var`'s real gaps (A6).
8. **Decimal sums are sequential in memory order** (A2). `np.sum(decimal)` adds an F-ordered array column by column,
   and a 28-digit decimal sum can round differently in another order, so the decimal tier's C-order reference
   differed from the port in the last digit on `f_contiguous_2d`. Its reference now sums the way NumSharp does: a
   dense operand in address order, anything else in logical order; per output along the axis. Its axis cases used to
   replicate the old kernel's double intermediate (15 digits).
9. **A function that copies its input first does not inherit np.sum's view residuals** (A3). NumPy's nanvar sums a
   K-order copy (`np.array(a, copy=True)`: dense, positive strides), never the view, so the np.sum differences that
   np.var inherits on a view with a reversed axis or a broadcast operand (finding 5) cannot reach it: the 10,306-case
   view-chain matrix is exact without exceptions, where np.var's had 490 inherited cases. A port that streams over
   the view instead must prove the view's walk equals the copy's (dense inputs only, here).
10. **The flat sum of a broadcast operand is not NumPy's call schedule** (X9). NumPy's iterator cannot coalesce a
   stride-0 axis with its neighbour, so it calls the add loop once per broadcast row (`out += pairwise(row)`);
   NumSharp's exact flat engine declines a broadcast operand and the elementwise fold sums in its own order: 1 ULP on
   12 of the mean_loops tier's broadcast cells (np.sum shares it). Pinned by
   `OpenBugsReductionTests.Sum_BroadcastOperand_Flat_IsNumPysCallSchedule`. Belongs with B1. An integer operand in a
   float32 / float16 loop reaches it too, through np.sum's cast copy (one pairwise call over the materialized copy where
   NumPy fills a buffer per row): A7's view-chain probe found it for `nanmean(int32 broadcast row, dtype=float32)`.
11. **Hold the ported function strict across every tier, not only its own** (X9, as finding 7 for A2). The reduce tier's
   mean cases sat under the unbounded summation branch, which had hidden a gross divergence (a broadcast uint64 mean
   of 101.8 for NumPy's 3.69e18) and a dtype one (decimal); the new tier alone would not have reached either.
12. **Pin a .NET benchmark to TWO CPUs, never one** (A4). A process pinned to a single CPU (`start /affinity 4`)
   starves .NET's background tier-1 compiler even at normal priority: a hot path stays at tier 0 forever. np.max with a
   float32 out over a float64 `(100, 1000)` measured ~100 µs per call pinned to one CPU, 13 µs with tiered compilation
   off, and 12–13 µs pinned to two P-cores (mask `0x14`) once it tiers up after ~0.4 s; a JIT summary showed 2 tier-1
   compilations in the whole one-CPU process. A full benchmark mixes the two regimes (a cell tiers up only when the
   main thread happens to block), so earlier one-CPU numbers are suspect for paths that tier up late.
13. **A repeated element changes nothing an order-free or idempotent loop can see** (A4). Integer / bool / sequential
   max and min give the same answer over one copy of a broadcast axis as over every repeat, so the reduction can cut
   it (10–560× for broadcast integer inputs). The float SIMD loops cannot in their reduce form — every `simd_reduce_c`
   call re-canonicalizes a NaN, so the number of calls shows in the bits — but can in their elementwise form, where each
   op is idempotent bit for bit.
14. **Compose a NumPy call's runs off the dependency chain** (A4). A masked reduction is one inner-loop call per run of
   selected elements, and each call is a ~40-cycle chain through the out's value; with runs of a few elements the run
   count, not the element count, sets the time (NumPy ~9 ns per run). The lane ops are associative, so each call's
   result is a closed form of the out's value and numbers taken from the run alone — only a compare-and-select stays on
   the chain (float64 where= 0.94× → 2.2× NumPy).
15. **Plant the old behavior back to prove a new error gate** (A5). The new tier passed on the fixed build, but with the
   old error planted back only 66 of its 78 IndexError cases turned red: the registry's K7 branch ("NEP50 weak-scalar
   (error path)") excuses any value or error-text divergence of an op with two operands, one of them 0-d, and so covered
   every 0-d y with an x. A trapezoid 0-d operand is never a weak scalar, so the branch now skips trapezoid, as it skips
   clip_pyint / clip_ufunc. K7 is keyed on the operand shape, not the op, so any tier's 0-d-operand case can fall into
   it; on the fixed build it excuses 6 cases in the whole oracle suite, all in `errors_full` and all its documented
   class (`bitwise_and` / `bitwise_or` / `bitwise_xor` of an int64 / uint64 pair with a 0-d operand), checked by
   tagging its reason with the case id for one run.

## 7. State log

- **2026-10-05**: A5 done (np.trapezoid raises NumPy's `IndexError: list assignment index out of range` for an axis
  outside y; 106 `err_axis_outside_y` cases in the `multioutput` tier; trapezoid out of the registry's K7 branch); finding
  15 recorded. Commit: this one.
- **2026-10-05**: A4 done (np.max / np.min / np.amax / np.amin / np.nanmax / np.nanmin / ndarray.max / ndarray.min take
  NumPy's `out=` / `initial=` / `where=` in NumPy's order, through a port of NumPy's reduction iterator; the
  `minmax_kwargs` oracle tier; plain np.max of complex128, broadcast and bool inputs and the fresh axis-result layout of
  every reduction fixed on the way); findings 12–14 recorded. Commit `1dcaf0d8`.
- **2026-10-05**: D7 marked done: the np.ndarray parity session fixed it in `f17c3b2d` (plan `np-ndarray-parity.md`);
  its behaviors checked here against NumPy's values. Next: A4.
- **2026-10-05**: A7 done (np.nanmean is NumPy's nanmean; `dtype=` added, B6's first part; the `nanmean` oracle tier;
  nanmean left the registry's nan-family blankets and the (W4-A) shape branch is gone; the eager float16 zero fill of
  nanmean and nanvar / nanstd is `np.putmask`); X11 recorded, X8 and finding 10 extended. Commit `a7f11b3e`.
- **2026-10-05**: X9 done (np.mean is NumPy's `_mean`: every `dtype=` loop, the flat integer loops, decimal, a broadcast
  uint64; the `mean_loops` oracle tier; np.mean held strict in every tier), with X7's integer / bool part; X10 and
  findings 10–11 recorded, finding 5 pinned. Found by A7, which is in progress. Commit `f8b32354`.
- **2026-10-05**: A3 done (np.nanvar / np.nanstd are NumPy's nanvar / nanstd; `dtype=` added, B7's first part; the
  `nanvarstd` oracle tier; nanvar / nanstd left the registry's nan-family blankets); A7 and finding 9 recorded; the
  generator's unknown-mode list completed (13 modes were missing). Commit `d1759ec8`.
- **2026-10-05**: A2 done (np.var / np.std are NumPy's `_var` / `_std`; the `varstd` oracle tier; the chunked
  deviation sums); A6, X6, X7, X8 and findings 7–8 recorded. Rebased onto `5c9d7925` (the view-`Shape` plan's last fix:
  the new code adopts the shapes it builds with `Shape.Adopt`); D7 unblocked and given that plan's `np.ndarray`
  follow-ups. Commit `41785501`.
- **2026-10-05**: X5 done (`np.sum` along an axis that leaves one output, in a cast loop), found by A2's large-size
  check; B3 widened (`np.mean` has no flat `dtype=` overload). Commit `4d06d591`.
- **2026-10-05**: X3 done (binary ufuncs with an F-ordered broadcast operand), found while proving A2; X4 and
  findings 4–6 recorded; A2 in progress (proof and design proof above). Commit `d3c863ef`.
- **2026-10-05**: A1 done (np.all / np.any reduce prologue, the `reduce_kwargs` tier S2, the null-axis priority);
  G4 added from its finding. Commit `bb5b2fcf`.
- **2026-10-05**: plan written from the inventory at `2821a708` (66 `np.*` + 14 `ndarray` names with a gap) and the
  `[OpenBugs]` pins of 2026-10-03/04.
