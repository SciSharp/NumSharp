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
  2026-10-05 (`f18bc09e`).
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
- [x] **A6 `np.ma.var` / `np.ma.std` with a mask: the masked path is not NumPy's.** `NDMaskedArray.var` computes its
  own formula (it never reaches `np.var`): 242 cases of the ma oracle tier differ — a slice NumPy masks (all masked,
  or `count - ddof <= 0`) carries NaN instead of the masked 0, and some slices are finite where NumPy has +inf. They
  were hidden by the registry's unbounded "summation precision" blanket until A2 removed var/std from it (finding 7);
  `MisalignedRegistry.IsMaVarStd` keeps the blanket for the ma ids only and names this item. Widened to `np.ma.mean`
  (var's mean) and `np.ma.anom` (`self - mean`). Done 2026-10-06 (`afcc343b`).
  - Proof (2026-10-05, at `f18bc09e`): with the ma ids' blankets removed, 254 of the 28,782 `ma_reduce` cases differed
    (masked slots holding NaN for NumPy's masked 0, unmasked infinities NumPy masks, other dtypes). A Python model of
    numpy.ma's statements (below) equalled NumPy 2.4.2 on 548,794 cases, so the port follows the statements, not the
    observed outputs.
  - NumPy's statements (`numpy/ma/core.py`). `mean` without a mask is `super().mean(axis, dtype)[()]` — ndarray's
    `_mean` run ON the masked array, so its `true_divide(ret, rcount, out=ret)` passes `MaskedArray.__array_wrap__`,
    whose divide domain (`|ret| * tiny >= rcount`) writes 1 into a quotient and masks it; a masked 0-d result is the
    `masked` singleton. With a mask (an all-False one too) it is `dsum * 1. / cnt`: `self.sum(dtype=f8 | f4 | None)`
    (the sum of `filled(0)`, a K-order copy, or the data itself under an all-False mask), times a STRONG float64 0-d
    through ma.multiply, divided through ma.true_divide (non-finite and domain quotients masked, the dividend put back),
    a 0-d sum plain scalar arithmetic, a 0-d count of zero `masked`, a float16 result `np.float16(...)` (mask kept).
    `var` without a mask is `_var` on the masked array (`mean=` passed on, a Python number WEAK there): both divides
    meet the same domain, the deviations are squared IN PLACE (`multiply(x, x, out=x)`, complex
    `x.view((f8, (2,)))`), a 0-d operand whose mean is masked raises "output array is read-only", a complex deviation
    carrying a mask raises "Changing the dtype to a subarray type…". With a mask: `cnt = count - ddof`;
    `danom = self - mean` through ma.subtract (`getdata` makes a Python number a STRONG array); `danom *= danom` in
    place (complex: `abs(danom) ** 2` through ma.power, a non-finite square masked); `divide(danom.sum(axis), cnt)`;
    an array result's mask REPLACED by `mask_or(mask.all(axis), cnt <= 0)` (shrunk to nomask when all-False), a masked
    0-d one `masked`, an unmasked 0-d one a scalar even for `cnt < 0`. `std` is `sqrt(var(...))` through ma.sqrt and
    IGNORES `mean=`. `anom` is `self - mean`, the mean expanded only for a nonzero axis (`if not axis`).
  - Fix (`src/NumSharp.Core/Ma/NDMaskedArray.cs`): those statements — `NomaskMeanLikeNumPy` / `MaskedMeanLikeNumPy`,
    `NomaskVarLikeNumPy` / `MaskedVarLikeNumPy`, `std`, `anom`, over `ArrayWrapDivideDomainFill`, `ItemOfMaskedResult`,
    `MaMultiplyByStrongOneLikeNumPy`, `MaDivideLikeNumPy`, `MaSubtractLikeNumPy`, `SquareInPlaceLikeNumPy`,
    `AbsSquaredLikeNumPy`, `FilledForReduction` (NumPy's K-order `filled`) and `DefaultEngine.DivideByCountLikeNumPy`
    (now internal). Every intermediate a later sum walks is allocated in NumPy's ufunc layout (`UfuncResultBuffer`,
    `Shape.NpyIterOutputShape`) and squared in place (finding 16); np.power's result is computed by the plain call and
    copied, because its out= route is not NumPy's (X13); a `mean=` that does not broadcast raises the subtract ufunc's
    text.
  - Proof of the fix: the four functions over 2,147,745 probe calls (13 dtypes; shapes `()` to `(40,33)` and 130
    elements; C / F / transposed / stepped / reversed / broadcast and every 3-D axis permutation; nomask, all-False,
    random, all-masked, one-unmasked-per-row and masked-first-row masks; every axis, keepdims, ddof 0–2, four `dtype=` loops,
    five `mean=` forms; special values): every result equal (kind, dtype, shape, data at EVERY slot, mask or nomask)
    but NaN payloads (9,062) and np.sum's own order residuals the operand view reaches (4,663: finding 5 on reversed
    views, findings 10 / X12 on broadcast operands).
  - Gates: oracle tier `ma_meanvar.jsonl` (26,379 cases, zero excuses: every catalog layout × the 13 dtypes ×
    nomask / all-False / some; the mask classes with keepdims and ddof 0 / 1 / 2 / 5; `dtype=` in eight loops; the
    `mean=` forms; special values incl. overflowing squares; sizes past the pairwise block; empty and 0-d inputs; 294
    NumPy errors). Beyond `filled(0)` and `getmaskarray`, each case pins the raw data at every slot and the result's
    form (the singleton, a scalar, nomask or a mask array). The generator carves 165 cells whose mean sums the operand
    view on a residual layout (38 finding 5, 127 broadcast, each only where the sum is not order-free); replaying the
    uncarved file, 64 differ, all of them carved. `ma_reduce`: mean / var / std / anom are out of every blanket
    (`IsMaVarStd` / `IsMaMean` deleted; the harness's own excuses skip them too) but a narrow finding-5 branch (3 cases);
    tightness tests `MaMeanVarStd_NoDivergenceIsExcused` and `Mean_ReversedViewAxis_IsTheOnlyExcusedLayout`; unit tests
    `Ma/np.ma.meanvar.NumPyPort.Test.cs` (9). Planted bugs: 14 / 14 killed by the unit tests and, separately, by the
    tier (two needed new unit cases, one a tier fix: the "huge" values now overflow the squares).
  - Speed (NPY/NS, two pinned P-cores, best of rounds; 310 cells: float64 / float32 / float16 / complex128 / int32 ×
    nomask / random mask (+ all-False for float64) × mean / var / std / anom × flat and axis forms × 1K / 100K / 1M):
    geomean 2.83×, minimum 0.78×; 1K 4.51×, 100K 2.16×, 1M 2.23×. The 49 cells under 1.5× are shared primitives:
    `np.copyto(where=)` under a random mask, numpy.ma's `filled` (X11: 266–627 µs of a 100K masked mean; NumPy's own
    copy costs 300–395 µs, so these cells sit at 0.93–1.5×), float16 subtraction with a column-broadcast operand (X15:
    float16 nomask anom along axis 1, 0.86×), and the memory-bound nomask means of 1M elements along axis 0 (1.00×
    float64, 0.93× complex128). Re-measure after X11 and X15.
  - Re-measured after X11 (2026-10-06, the same 310 cells, NumPy's numbers reused): geomean 4.60× (was 2.83×), minimum
    0.86×; random-mask cells 5.40× (were 1.89×, minimum 2.27×). Six cells stay under 1.5×, none of them the masked
    copy: float16 nomask anom along axis 1 at 100K (0.86×, X15) and the nomask means of 1M elements (1.03–1.35×,
    memory-bound: the same-type slab mean of X9).
  - Left open: X11–X15 (below); `out=` of ma.mean / var / std (NumPy's MaskError text included); the public
    `NDMaskedArray.filled()` returns a C-order copy where NumPy returns the data itself or a K-order copy (the
    reductions use `FilledForReduction`); `Shape.ResolveReturnShape` reads a `(1,)` operand as a scalar (the port uses
    `np.broadcast_shapes`).
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
- [x] X2 `np.maximum`, `np.minimum`, `np.add` (binary ufuncs): an out-of-range C# integer literal wraps
  (`np.maximum(int8, 1000)` gives -24); NumPy raises `OverflowError` (NEP 50 packing). Found by the `np.clip` work.
  Done 2026-10-07 with X21 and X24 (`d0a6f4c8`).
  - Fix (`DefaultEngine.BinaryOp.cs`, `DefaultEngine.CompareOp.WeakScalar.cs`): `ExecuteBinaryOp` checks a 0-d
    Python-int operand against the FINAL loop dtype before any kernel runs (`CheckWeakPyIntOperand`, NumPy's
    `npy_update_operand_for_scalar` + `PyArray_Pack`: "Python integer 1000 out of bounds for int8", the LP64 texts
    past int64). A bool array meets the int in NumPy's int64 loop. `dtype=` moves the check to that loop. A float16
    / float32 loop re-packs an int past 2^53 through the correctly rounded double, two roundings as in NumPy
    (`RepackWeakIntForFloatLoop`: `float32 + (2**60 + 2**36 + 1)` is 2^60). np.clip, np.arctan2 and the float-tier
    binary ufuncs share the re-pack; the shifts share the check.
  - Gates: the `weak_scalar` tier's arithmetic sections (A, O), `Logic/np.comparison.WeakScalar.Test.cs`
    (`BinaryUfuncWeakIntPackTests`).
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
- [x] X4 float16 `add` / `subtract` / `multiply` / `divide` with a broadcast, non-contiguous operand set run the
  direct route's scalar General loop: `(30,40,20) F * (30,1,20) F` takes 328 µs (NumPy 72 µs, 0.22×), and
  `(100,100,100)` 27 ms (NumPy 3.0 ms). The same before X3 (HEAD measured 320 µs while computing wrong values).
  Speed only; the values are NumPy's.
  Done with X15 (2026-10-06, `1185f39c`): the set is walked with its axes reversed into an F result, and every inner
  run takes the SIMD widen-compute-narrow kernels: `(30,40,20) F * (30,1,20) F` 38.6 µs (NumPy 76.9 µs, 1.99×; was
  328 µs), `(100,100,100) F * (100,1,100) F` 0.91 ms (NumPy 3.30 ms, 3.62×; was 13.8 ms).
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
- [x] **X11 `np.copyto(dst, src, where=mask)` runs a branch per element (and a generic loop for float16 / complex).**
  Done with X16 (2026-10-06, `2dffb8d9`): the masked copy is a new bit-copy kernel per element width; the speed is in
  X16.
  2.5 ns per element over a `(250, 400)` float16 with every 7th element masked: 253 µs, where
  `np.putmask(dst, mask, scalar)` — the same assignment for a same-shape mask — takes 21 µs. Speed only. nanvar /
  nanmean's eager float16 route now zeroes with putmask (A7, `ZeroWhereMask`). Found by A7's benchmark.
  - A6 (2026-10-06) widened it: with a RANDOM mask (35 % set; a regular mask hides it, the branch predicts) every
    dtype is slow — `np.copyto(dst, 0, where=m)` over 100K elements is 266 µs for float64 / float32 / int32 and
    579–601 µs for complex128 / float16 (NumPy 300–395 µs), and `np.putmask` is 205–225 µs (NumPy 265–290 µs): both
    branch per element. Three causes in `DirectILKernelGenerator.Cast.Masked.cs`: `ResolveStrategy` returns
    `(MemoryCopy, simdBits 0, vstep 0)` for a same-dtype pair, so the kernel's SIMD `ConditionalSelect` blend for
    same-dtype rows is dead code; a 0-d (stride-0) source never takes the SIMD path; and float16 / char / bool / complex /
    decimal have no masked kernel at all (np.copyto's generic per-element `ConvertValue` loop). numpy.ma's `filled` is
    this copy, so it is most of every masked ma reduction (A6's cells under 1.5×). The lever: a branch-free blend for
    same-dtype copies of every element width (1 / 2 / 4 / 8 / 16 bytes; a bit copy needs no dtype), a broadcast-vector
    path for a stride-0 source, a 1-element source cast once to the destination dtype, and a branch-free scalar select
    for the strided rows.
- [ ] **X12 `np.sum` / `np.mean` along an axis of a broadcast operand are not NumPy's order.** NumPy's reduction
  iterator adds an operand with a stride-0 axis in its own call schedule: a `(k,1)` column broadcast to `(k,m)` and
  reduced over axis 0 is m sequential sums (the stride-0 axis is the inner loop), a float16 sum along the stride-0 axis
  rounds once per call (HALF's float32 accumulator), a cast loop casts in buffered fills. NumSharp's axis kernels add in
  another order: float32 `(9,1)→(9,3)` axis 0 `0x40466668` vs NumPy `0x40466664`; float16 `(3,1)→(3,7)` axis 1 rounds
  per element; int32 `(1,9)→(4,9)` axis 1 in float32 `0xce2eb2a8` vs `0xce2eb2aa`. A probe over rows, columns, a middle
  stride-0 axis and a 0-d broadcast in every loop found 510 of 5,382 cases differing (the flat ones are finding 10).
  Pinned by `OpenBugsReductionTests.Sum_BroadcastOperand_Axis_IsNumPysOrder`; carved from `ma_meanvar.jsonl`
  ("broadcast"). Found by A6. Belongs with B1.
- [x] **X13 `np.power(x, scalar, out=o)` misses NumPy's scalar-exponent fast paths.** NumPy's DOUBLE_power /
  FLOAT_power take `1/x`, 1, `sqrt(x)`, `x`, `x*x` for a stride-0 exponent of -1, 0, 0.5, 1, 2 on every route; NumSharp's
  plain call does too, its ufunc out= route calls the CRT `pow`: one ULP off on 116 of 200,000 float64 and 22 of
  200,000 float32 inputs for 2.0 (`757.7617475757002 ** 2` is `...03c1` in NumPy, `...03c2` through out=). Pinned by
  `OpenBugsUfuncOutTests.Power_Out_ScalarExponent_TakesNumPysFastPath`. Found by A6 (its `abs(danom) ** 2` uses the
  plain call and copies). Done 2026-10-06 (`4116b608`), with X19 and X20.
  - Proof (NumPy 2.4.2 vs `6f00ae14`):
    - A 34-call twin probe. The out= and where= routes called pow for a 0-d exponent (`-0.0 ** 0.5` gave +0.0, NumPy
      -0.0). `(1,) ** (1,)` and `0-d ** (1,)` took the fast path where NumPy's trivial loop calls pow, and
      `0-d ** (1,)` returned a 0-d result (X19). float16 took the fast path or the double bridge (X20).
    - A special-value probe: 31 values × 31 exponents × float16 / float32 / float64, through full and 0-d exponents.
      44 cells differed.
    - The new `power_fast` tier: 1,785 of its first 7,773 cases differ on the parent.
  - Cause: NumPy's float power loops test the exponent's STEP in each inner-loop call (`steps[1] == 0`), so the bits
    depend on which calls the ufunc machinery makes. Three stages decide:
    - `check_for_trivial_loop`: a cast or unaligned 0-d input, or a 1-D input of at most bufsize elements, is copied;
      any other such input, and an out of another dtype or unaligned, refuses the trivial loop.
    - `try_trivial_single_output_loop`: a 0-d input has step 0, a 1-D operand its own stride, an N-D operand the item
      size.
    - NpyIter: axis order, flips, COPY_IF_OVERLAP, coalescing and `npyiter_find_buffering_setup`'s cost model. A
      column exponent reaches the loop unbuffered (step 0) only for rows longer than bufsize / 2.

    NumSharp's plain call took the fast path for every one-element exponent; its ufunc out= / where= routes never did;
    float16 took it too.
  - Fix:
    - The step decision is ported once (`DefaultEngine.UfuncInnerSteps.cs`, `NumPyUfuncInputSteps`). It is generalized
      from np.clip's const-loop decision, which now calls it. np.power's float32 / float64 loops route on it
      (`Default.Power.StrideZero.cs`, `PowerFloatLoop`).
    - A single-valued exponent with a fast value runs the matching SIMD unary kernel (square / sqrt / reciprocal /
      positive) or a fill with 1 into the result or the out. With where=, the values are computed first and copied
      with the mask (np.copyto): 2.5–9× faster than one kernel call per mask run. A 0-d plain call skips the shape
      prologue.
    - A varying exponent that NumPy's calls see with step 0 (wide columns) runs one NDIter pass with NumPy's per-call
      branch on each element (`EmitPowerStrideZeroBody`). When both inputs are the loop dtype, the kernel also has a
      vector body (`PowerStrideZeroVector128/256/512`, generic over the two float loops): the broadcast loops hand it a
      uniform exponent vector, so one compare selects one vector operation.
    - Everything else is pow (`ExecuteBinaryOp`).
  - Gates: the `power_fast` oracle tier (7,902 cases, zero excuses); unit `Math/np.power.StrideZero.Test.cs` (11 tests,
    NumPy-probed bits); `Backends/Kernels/OneElementOperandRankTests.cs` (X19). Planted bugs: 13 — the trivial-loop
    copy bound, the cast-out refusal, the buffering stop, the cost comparison, the buffered broadcast's step 0, the
    trivial 1-D step, the float32 exponent comparison, the 0-d shortcut, where='s mask, the vector square, the scalar
    reciprocal, float16's tail and its broadcast exponent. The tier and the unit tests kill all 13. Two survived the
    first pass and showed missing cases (finding 27); they are the tier's section I now.
  - Speed (NPY/NS, two pinned P-cores, best of rounds):
    - 90 fast-path cells (float64 / float32 × 1K / 100K / 1M × -1, 0, 0.5, 1, 2 × plain / out= / where=): geomean
      6.0–6.2× over two runs. 100K and 1M: 2.0–20×.
    - The 1K plain float64 cells: 1.3–2.9× from run to run. They sit at the NDArray allocation floor: `np.empty(1000)`
      alone costs 0.20 µs disposed and 0.42 µs left to the finalizer, against NumPy's 1.0–1.9 µs per call. With
      disposed results: `x ** 0` 3.0×, `x ** -1` 1.9× (division-bound on both sides), `x ** 2` 6.2×. Power's own cost
      over `np.ones` / `np.reciprocal` / `np.square` is 0.01–0.04 µs.
    - The 24 pow cells (`x ** 3`, full-shape exponents): 0.92–1.15×. Both sides call the CRT pow per element.
    - Varying column exponents: (2, 5000) 1.67×, (16, 8192) 4.4×, (100, 10000) 3.5×. The old route called pow there:
      0.16–0.29×; the scalar body alone gave 0.85× at (2, 5000).
  - Left open, found here: X21, X22, X23.
- [ ] **X14 The engine's flat mean of a broadcast float16 / complex128 operand ignores `dtype=`.**
  `TensorEngine.Mean(b, null, float32)` of a broadcast float16 `(4,5)` is the float16 `0x3c33` where NumPy's
  `np.mean(b, dtype=np.float32)` is the float32 `0x3f866666`; a broadcast complex128 with `dtype=float64` stays complex
  (NumPy: the real parts' mean). `ReduceMean`'s flat route sends a broadcast float / complex operand, which
  `TryExactFlatMean` declines, to the elementwise fold, which keeps the input dtype. `np.mean` has no flat `dtype=`
  overload yet (B3), so callers meet it through the engine. Pinned by
  `OpenBugsReductionTests.Mean_BroadcastOperand_Flat_HonorsDtype`. Found by A6: the mean_loops tier carves every flat
  broadcast float cell as finding 10, an order residual, which also hid this dtype bug (finding 18).
- [x] **X15 A float16 binary ufunc with a column-broadcast operand is slow.** `np.subtract(float16 (250,400),
  float16 (250,1))` takes 1,141 µs (11 ns per element) where NumPy takes 377 µs and the same-shape float16 subtract takes
  72 µs. Speed only. numpy.ma's anom of float16 along axis 1 meets it (0.86× NumPy). Found by A6's benchmark.
  Done 2026-10-06 (`1185f39c`), with X4.
  - Proof (NumPy 2.4.2 vs `72696ec1`, two pinned P-cores, 100K float16): add / subtract / multiply / divide with a
    column, a row, an outer-product pair, a strided or a reversed operand 0.30–0.50× (~10 ns an element, every layout
    the same), against 4.1× for same-shape operands and 5.6× for a 0-d one. The values were NumPy's.
  - Cause: (Half, Half) → Half arithmetic declines the iterator (its scalar body bridges through double) and the direct
    route served every broadcast / strided layout with `HalfArithStrided`: a per-element coordinate decomposition (a
    div and a mod per axis per element) around the scalar widen-compute-narrow. The out= route went to the iterator,
    whose float16 body is scalar (~5 ns an element): numpy.ma's anom subtracts a column mean into an out= buffer, so
    fixing the plain call alone left it at 0.86×.
  - Fix (`DirectILKernelGenerator.Binary.Arith.Half.cs`, `DefaultEngine.BinaryOp.cs`): `HalfArithStrided` walks the
    outer axes with an incremental odometer and hands each inner run to `HalfArithRun`, which picks the SIMD kernel
    for the run's strides — both contiguous, one broadcast (operand roles kept), both broadcast (one value), or a
    strided operand gathered into a 256-element stack block first. A broadcast all-F set is walked with its axes
    reversed into an F result (X4). The out= trivial route takes broadcast and strided float16 inputs into a C or F
    out (`TryHalfArithBroadcastInto`; an input that is the out runs in place, any other overlap goes to the
    iterator's copy).
  - Tests: `Math/HalfArithKernelTests.cs` (+3: every layout against the contiguous kernel bit for bit, random bit
    patterns with NaN payloads, runs across the gather block; the reversed F walk; out= against the iterator route in
    C and F outs, in place, a column view and the transpose of the out); `TrivialBinaryOutTests` now expects the
    trivial route for float16 F inputs into a C out. Planted bugs: 13 (roles swapped, gather stride, odometer rewind,
    the reversed walk, the gather block, the double broadcast, the strided-left / broadcast-right pair, the right
    offset, the out= alignment, alias and overlap checks, the F out's order); the unit tests kill 12, the oracle 7;
    the 13th (the F result copied to F again) is equivalent: one extra copy.
  - Speed (NPY/NS, two pinned P-cores): 24 plain cells (column, row, left column, divide, strided, reversed, F set,
    outer product × 1K / 100K / 1M) geomean 3.91×, minimum 1.55× (1K row, run-to-run; 3.3× after a warm pass);
    100K: 3.7–7.2×. 15 out= cells (column, row, strided, F into an F out, in place × 1K / 100K / 1M) 3.58–7.41×.
    numpy.ma's 310 cells 4.60× → 4.80×, minimum 0.86× → 1.07× (the anom cell 0.86× → 5.97×; the five cells under 1.5×
    are the memory-bound 1M nomask means).
- [ ] **X18 A fresh elementwise result is laid out C (F only for an all-F set) and walked in C order; NumPy lays it out
  with NpyIter's K order and walks memory order.** Found by X15's benchmark (2026-10-06). `np.add(x.transpose(1,0,2),
  1.0)` over a float32 `(3,4,5)` gives NumPy strides `(20, 80, 4)` (the permutation kept) and NumSharp `(60, 20, 4)`
  (C); values identical. An F operand with a `(1, n)` row, which is C- and F-contiguous at once, is computed in C order
  with strided reads and then copied to F: `np.subtract(float32 (400,250) F, (1,250))` 66 µs, NumPy 14 µs (0.22×;
  float64 runs into NumPy's slow allocation and shows 2×). The rule `ShouldProduceFContigOutput` approximates NumPy's
  for two operands; NumPy's is `Shape.NpyIterOutputShape` (finding 16) plus a K-order walk. Layout and speed; every
  elementwise ufunc.
- [x] **X16 `np.copyto` is not NumPy's `array_copyto`: wrong values for an overlapping copy, NumPy's checks missing or
  in another order, weak scalars wrapped.** Found by X11's probe (2026-10-06). Done 2026-10-06 (`2dffb8d9`).
  - Proof (NumPy 2.4.2 vs NumSharp at `04e0e703`; `copyto_probe`, 86 calls; the house exception types mapped to
    NumPy's): 54 differ.
    - Silent wrong values. A masked copy never checked overlap: `np.copyto(a[1:], a[:-1], where=m)` gave
      `[0, 0, 2, 2, 2, 5, 5, 5, 8, 8]`, NumPy `[0, 0, 2, 2, 3, 5, 5, 6, 8, 8]`. An overlap probe (1,500 random views of
      one buffer, 1-D and 2-D, stepped / reversed / transposed, masked or not) differed in 251 cases: 112 masked 1-D, 135
      masked 2-D, and 4 unmasked 1-D, where NDIter.Copy copied the source first and NumPy walks its raw order, which
      reads its own writes when the strides differ (`a[3:19:2] = a[6:14]`: NumPy `[…, 12, 10, 14, 12, 16, 10, …]`,
      NumSharp `[…, 12, 11, 14, 12, 16, 13, …]`; slice assignment shares the path).
    - Accepted where NumPy refuses: a mask that does not broadcast (`(1,1)` into `(3,)` — NumPy never strips a mask's
      axes), a source that does not broadcast under a false, 0-d false or `(1,)` false mask (NumPy checks the source
      first), a `(1,)` mask into a 0-d destination.
    - Refused where NumPy accepts: a source with leading unit axes beyond the destination's rank (`(1,3)` and
      `(1,1,3)` into `(3,)`, `(1,)` and `(1,1)` into `()`), a redundant self-copy of a read-only array
      (`np.copyto(b, b)`), a C# float into float32 under `'safe'` and `'no'`, a C# int into int64 under `'equiv'` (both
      legal for NumPy's weak Python scalars), a C# int into float32 under `'safe'`.
    - Wrapped where NumPy raises: `np.copyto(int8, 1000)` gave `-24` (NumPy "Python integer 1000 out of bounds for
      int8", masked or not); `'equiv'` accepted an int into int32 (NumPy "cannot cast Python int to int32 under the
      casting rule 'equiv'"); `-1` into uint8 reported a cast error instead of NumPy's OverflowError.
    - Other errors and orders: "where must be a boolean array, got dtype('int64')" instead of NumPy's cast text; the
      iterator's "operands could not be broadcast together with remapped shapes …" instead of "could not broadcast input
      array / where mask from shape … into shape …"; the casting-name errors as ArgumentException / ArgumentNullException;
      "Cannot cast array data" where NumPy says "Cannot cast scalar"; a read-only destination and a bad cast reported
      before a bad mask, a bad source shape reported as a cast error.
  - Fix (`Manipulation/np.copyto.cs`): a statement-for-statement port of `array_copyto` → `PyArray_AssignArray` /
    `PyArray_AssignRawScalar`.
    - Arguments: the casting converter, the mask's bool check, a C# number as NumPy's Python scalar (`copyto<T>`:
      `npy_mark_tmp_array_if_pyscalar`, `npy_find_descr_for_scalar` — the common DType of the abstract Python class and
      the destination's — `npy_update_operand_for_scalar`'s `'equiv'` TypeError, then `PyArray_Pack`'s range checks
      through `ReductionInitial.TryPack`), a `BigInteger` overload for a Python int of any size. bool / Half / char /
      decimal are strong.
    - The checks in NumPy's order (above), each with NumPy's text.
    - The masked copy: `RawArrayIter.PrepareThree` (`PyArray_PrepareThreeRawArrayIter`) and the 1-D backward rule, then
      a new kernel (`DirectILKernelGenerator.MaskedAssign.cs`, X11): a bit copy of the destination's element width, one
      kernel per width (1 / 2 / 4 / 8 / 16 bytes) for all 15 dtypes; an IL walker over the outer axes; a run stage with a
      SIMD blend for contiguous runs (`ConditionalSelect`, AVX2 mask widening, a hoisted vector for a broadcast source),
      a branch-free select off running pointers for strided runs and tails, memmove / fill for a constant-true run. With
      more than one prepared axis the operands cannot overlap, so the walk puts innermost the axis where most operands are
      contiguous (NumPy's order wins ties; the bits are the same either way).
    - A source of another dtype, or an overlapping source NumPy copies first, is staged in the destination's memory
      order: one element per distinct source element, on the stack up to 16 KB, else a pooled buffer, through the cast
      kernels NDIter.Copy uses.
    - NDIter.Copy's overlap rule (np.copyto's unmasked copy, slice assignment, every assignment through it) is NumPy's:
      copy first for a multi-axis destination or an opposite-direction 1-D one; otherwise NumPy's raw 1-D order
      (`AssignRaw1D`).
    - nanmean / nanvar's eager zero fill (`ZeroWhereMask`, A3 / A7) is NumPy's own `np.copyto(arr, 0, where=mask,
      casting='unsafe')` again: 3.7 µs over a float16 `(250, 400)` with every 7th element masked, where the
      `np.putmask` route A7 took takes 22.7 µs (NumPy's copyto 124 µs).
  - Gates: the `copyto_full` oracle tier (13,462 cases, zero excuses: every check against every later one, the texts,
    13 dtypes × 25 layouts × 14 mask forms, long runs, 16 cast pairs, 1,100 overlaps, C# literal sources under every
    casting rule, 0-d / empty); unit `Manipulation/np.copyto.NumPyPort.Test.cs` (21 tests; 12 of the first 18 fail
    on `04e0e703`, the last three were added for mutants only the tier caught); the overlap probe 1,500 / 1,500 and the call probe 82 / 86 (the four left: the house exception types
    aside, NumPy's `where=1` scalar mask (finding 2), a Python list of ints with no C# spelling, the LP64 text for a
    Python int past int64, and the probe's own complex print format). Planted bugs: 21 edits (the redundant self-copy, the 2-D
    overlap temporary, the source's axis stripping, the 0-d mask, the lane mask, the 1-D backward rule, the staging's
    coalescing and stride rewrite, the `'equiv'` check, the weak scalar's dtype, NDIter's temporary rule and raw order,
    the 1-D flip, the SIMD blend, the 16-byte pair mask, the unrolled select, a scalar source's mask, the overlap
    staging, the 0-d source route, the strided staging copy, a broadcast run's memmove): the tier kills 20, the unit
    tests all 21 (the tier cannot build the 21st's input, a source view whose axes overlap: `np.ndarray` with strides
    `(8, 8)`).
  - Divergences left (documented): the exception types are the house's; a 0-d non-bool mask always raises; NumSharp
    pins the LP64 Python-int texts; one complex width.
  - Speed: NPY/NS, two pinned P-cores, best of rounds; 192 cells: bool / int8 / uint16 / float16 / int32 / float32 /
    float64 / int64 / complex128 × 1K / 100K / 1M × a 0-d source, an array source, a row source, a row mask, a strided
    destination, an F destination, a float32 → float64 cast (each under a 35 % random mask), and no mask. Geomean
    9.46×. The 165 masked cells: geomean 12.4×, minimum 1.51× (the row-mask complex128 1M cell streams 32 MB, DRAM /
    L3-bound on both sides, 1.21–1.51× across runs); by form: 0-d source 38×, array 23×, row source 23×, row mask
    10.6×, strided destination 4.0×, F destination 4.6×, cast 5.5× (were 2.5–6 ns per element: X11). The unmasked copy
    is memcpy on both sides: 1K 4.3–7×, 100K / 1M 0.73–1.18× (finding 23). Two levers measured on the way: staging the
    cast source on the stack instead of in an NDArray took the 1K masked float32 → float64 copy from 0.78 to 0.32 µs
    (NumPy 1.15 µs; finding 22), and running pointers in the strided select loop took a 1K strided-destination copy
    from 0.77 to 0.48 µs (NumPy 1.15 µs; finding 24). numpy.ma's masked reductions, whose `filled` is this copy, went
    from 2.83× to 4.60× (A6).
- [ ] **X17 The old masked cast kernels have no caller left.** `DirectILKernelGenerator.Cast.Masked.cs`
  (`TryGetMaskedCastKernel`, the per-pair masked cast with the dead same-dtype blend X11 found) served only np.copyto's
  masked path, which X16 replaced. Delete it, or route a future masked cast through it after a parity check.
- [x] **X19 A one-element operand with more dimensions than the other loses its leading axes.** Found by X13's probe
  (2026-10-06): `np.power(0-d, (1,))` returned a 0-d result, NumPy's is `(1,)`. Done 2026-10-06 (`4116b608`).
  - Proof: a 165-call probe (24 binary ufuncs, comparisons, logical ops, shifts and C# operators × `(7,) op (1,1)` →
    `(1,7)`, `(1,1) op (7,)`, `(3,7) op (1,1,1)` → `(1,3,7)`, `0-d op (1,)` → `(1,)`, `(1,) op (1,1)` → `(1,1)`): 100
    wrong shapes. The values were right.
  - Cause: the scalar-broadcast shortcuts of the binary route (`TryTrivialContiguousBinaryOp`), the comparison route
    (`TryTrivialContiguousComparisonOp`) and the shift route (`TrySimdScalarShift`) gave the result the array operand's
    shape. np.power's old fast path did the same.
  - Fix: the shortcuts decline when the one-element operand has more dimensions; the broadcasting routes compute
    NumPy's shape.
  - Gate: `Backends/Kernels/OneElementOperandRankTests.cs` (2 tests: 165 shapes and the values; 100 shapes fail before
    the fix).
- [x] **X20 np.power on float16 and np.float_power are not NumPy's loops.** Found by X13's probe. Done 2026-10-06
  (`4116b608`).
  - Proof: NumPy's HALF_power is `PyUFunc_ee_e_As_ff_f` with powf and has NO fast path. NumSharp's float16 power took
    np.power's fast path (`-0.0 ** 0.5` gave -0.0, NumPy +0.0; `-inf ** 0.5` NaN, NumPy +inf; `sNaN ** 1` kept the NaN
    signalling) and otherwise bridged through double. Over every float16 base × 17 exponents, 7,161 of 1,114,112
    results differed: `sNaN ** 0` gave 1, not NaN (the BCL widen quiets the NaN first), integer powers of a negative
    NaN lost its sign, and one result rounded once from double instead of from powf's float. np.float_power
    (`PyUFunc_dd_d` with pow) took np.power's fast path: `float_power(757.7617475757002, 2.0)` gave `x*x` (`...03c1`),
    NumPy pow (`...03c2`).
  - Fix: float16 power is NumPy's pipeline — the exact widen (`HalfToFloatScalarExact`: a signalling NaN stays
    signalling), MathF.Pow (the CRT powf NumPy calls) and npy_float_to_half's narrow. The emitted scalar paths call
    `PowerHalfNumPy`; whole arrays run the widen-compute-narrow kernels' new `HalfPower` (a vector widen, one block
    loop of MathF.Pow, a vector narrow; every layout through the X15 routes). np.float_power runs the power kernel and
    never the fast path.
  - Gates: the `power_fast` tier (float16 in sections A and F, float_power in G); `Float16_IsPowfWithoutAFastPath`,
    `FloatPower_NeverTakesTheFastPath`; a probe over every float16 base × 17 exponents through five routes (full,
    0-d, reversed, odd slice, 0-d base): 0 differences.
  - Speed: float16 power 1.52–1.56× at 1K / 100K / 1M. The old fast path was 2.6–5.8× for 2.0 and 0.5 but gave
    other bits; the double bridge was 0.54× for 3.0. powf costs ~5 ns an element on both sides and NumPy's scalar half
    conversions ~3.5 ns more, so the ceiling is ~1.7×. A per-8-lane form (extract each lane, powf, rebuild the vector)
    reached only 1.32–1.41×. float_power: 0.96–1.06× (pow on both sides; the old 14–28× came from computing `x*x`,
    which is not NumPy's value).
- [x] **X21 `np.power(unsigned, -1)` raises the wrong error.** NumPy packs the weak Python int into the unsigned loop
  first: `np.power(np.uint8([2]), -1)` raises `OverflowError: Python integer -1 out of bounds for uint8` (uint32 and
  uint64 the same). NumSharp reaches the integer loop's check and raises `ArgumentException: Integers to negative
  integer powers are not allowed.` It is X2's NEP 50 packing class on np.power: `np.add(np.uint8([2]), -1)` gives `[1]`
  in NumSharp and the same OverflowError in NumPy. The `power_fast` generator leaves these cases out. Found by X13.
  Done 2026-10-07 with X2 (`d0a6f4c8`): `Default.Power.cs` runs the read-only out check, the mask check and X2's pack
  check before the negative-power error, in NumPy's order. Gate: `BinaryUfuncWeakIntPackTests`.
- [ ] **X22 A binary ufunc's out= broadcast error omits the out's shape.** `np.power(ones(3), ones(4), out=empty(3))`:
  NumPy "operands could not be broadcast together with shapes (3,) (4,) (3,) " (every operand, the out included),
  NumSharp "… with shapes (3,) (4,) " (`IncorrectShapeException`). The 0-d exponent form and the "non-broadcastable
  output operand" text match. Text only. Found by X13; the `power_fast` generator leaves the case out.
- [ ] **X23 np.evaluate's Power node misses NumPy's scalar-exponent fast path, and np.kaiser inherits it.**
  `np.evaluate(NDExpr.Power(x, 2.0))` computes pow (`757.7617475757002 ** 2` = `...03c2`, `-0.0 ** 0.5` = +0.0) where
  NumPy's unfused `x ** 2.0` takes DOUBLE_power's fast path (`x*x` `...03c1`, sqrt -0.0). np.kaiser evaluates NumPy's
  `((n - alpha) / alpha) ** 2.0` through that node: 146 of 4,796 windows (M from 2 to 1,200 × beta 0, 5, 8.6, 14)
  differ in at least one element, by 1–2 ULP (`np.kaiser(272, 5.0)[13]`: NumPy `0x3fb79bb596a301c3`, NumSharp
  `...c5`; the windows tier has no such M). The fix: a Power node whose exponent is a constant or a hoisted 0-d
  parameter emits NumPy's stride-0 branch (`PowerStrideZero<T>` and its vector forms) for the float32 / float64 loops.
  Found by X13. Next item.
- [x] **X24 A comparison with a Python float compared at float64 against a float32 / float16 array: silent wrong
  values.** Done 2026-10-07 with X2, X21 and X25 (`d0a6f4c8`).
  - Proof: `np.greater(np.array(new[] { 0.1f, 0.5f }), 0.1)` gave `[True, True]`; NumPy `[False, True]`. Under NEP 50
    the weak `0.1` is packed into float32 first, and `0.1f > 0.1f` is False. All six comparisons and the C# operators
    were affected: a literal that is not exact in the array's dtype (`0.1`, `1/3`), an int past 2^24 for float32 or
    past 2048 for float16, and a complex literal (complex64 parts).
  - Cause: the comparison dispatch promoted a 0-d operand strongly (`_FindCommonScalarType`). The arithmetic routes
    already used the array-scalar table.
  - Fix (`DefaultEngine.CompareOp.WeakScalar.cs`, `ExecuteComparison`): a 0-d Python-scalar operand against an array
    is packed into the array's precision before both routes run. The pack is `PyArray_Pack`: the correctly rounded
    double first, then the float16 / float32 narrow, overflow to inf; complex parts are rounded to float32.
    - A 0-d against a 0-d stays strong. A 0-d NDArray against an array is read as a literal (np.clip's rule,
      `[Misaligned]`).
    - An integer array against an in-range Python int of another width is re-packed into the array's dtype. This is
      NumPy's `special_integer_comparisons` in-range branch; the values do not change.
    - numpy.ma stays strong: its comparison wrappers promote both operands (`PromoteMaScalar`, `strongOperand`), and
      its domains see `getdata` data.
    - np.evaluate compares a complex constant against a float16 / float32 array with the constant's parts rounded
      to float32.
    - An empty comparison returns its empty result without a kernel. The out= route compares at
      `GetComparisonType`, so int64 / uint64 compare exactly.
  - Gates: `weak_scalar.jsonl` (15,050 cases, zero excuses; 8 of 8 planted bugs killed; its section W kills a wrong
    lane order in the packed mask store); `Logic/np.comparison.WeakScalar.Test.cs`, `Ma/np.ma.StrongScalar.Test.cs`.
    The numpy.ma harness now refuses the old "NEP50 weak-scalar" dtype excuse.
  - Speed (NPY/NS, two pinned P-cores), with X25's kernels: `f32 > 0.1` 0.82× / 1.17× / 1.73× at 1K / 100K / 10M (was
    0.49× / 0.09× / 0.56×). `f16 > 0.1` 15× (the same-dtype float16 kernel). `u8 > 128` 3.9× at 10M. At 100K NumPy's
    loop is the same packed form, so ~1.2× is its ceiling. The 1K cells of array op scalar (0.56–0.83×) are X26.
- [x] **X25 The SIMD comparison and predicate kernels read one element past their inputs and wrote one bool past
  their result.** Found by X24's kernel work. Done 2026-10-07 (`d0a6f4c8`).
  - Proof: `np.greater(a[:7], b[:7], out=o[:7])` (float32) overwrote slot 7 of o's parent buffer. The 4×-unrolled
    and one-vector loops ran while `i <= n - step + 1`, so they ran one body too many when `n % step == step - 1`
    (float32: n = 7, 15, 31, 39, 63, 127, …; every SIMD dtype). The same-dtype array-against-array kernel
    (`SimdFull`) and the contiguous float / signbit predicate kernel (isnan, isinf, isfinite, isposinf, isneginf,
    signbit) had it since they were written. A fresh result's extra byte landed past its allocation, because the
    pool's buckets have exact sizes. A NUMSHARP_DEBUG_GUARD_PAGES sweep of 58,320 elementwise / comparison / cast /
    flat-reduction cases faults at `equal Boolean n=31` on the old build.
  - Fix: the loops subtract the whole step (`i <= n - step`).
  - Gates: `Backends/Kernels/ComparisonBroadcastKernelTests.cs`. It checks sentinel slots past the result for the
    three comparison kernels and the predicate kernel at every tail length, and an out= view's parent buffer; all
    three tests fail with the old bound. The guard-page sweeps (72,630 cases, strided / reversed / broadcast inputs
    and axis reductions included) pass.
  - Also in the kernels:
    - The scalar-broadcast comparison kernels (`array op scalar`, `scalar op array`) were scalar loops (0.72 ns an
      element; NumPy 0.06). They now compare a whole vector per step, with the scalar broadcast once.
    - Four masks of 2-, 4- and 8-byte lanes are packed into result bytes, as NumPy's `npyv_pack_b8_*` do. 1-byte lanes
      store `mask & 1`.
    - The single-mask store reads the lane bits as 64 bits. The old code stored them in a 32-bit local, which cannot
      hold a Vector512 mask's 64 lanes (int8 / uint8 / bool on an AVX-512 host; not testable on this host).
    - Measured over 100K elements against the per-vector store: float32 7.05 → 5.15 µs, float64 14.7 → 9.95, int16
      6.8 → 3.8, int8 ~28 → 2.8.
- [ ] **X26 A C# literal operand costs ~0.8 µs a call.** `x > 0.5f` at 1K float32 takes 1.46 µs; with a prebuilt 0-d
  operand it takes 0.50 µs, as fast as array op array. The operator never disposes the 0-d NDArray of the implicit
  conversion, so every call leaves a finalizable object: allocation, finalization, and the pool's paced collections.
  The cost is library-wide: `f32 + 0.1` at 1K is 0.62× NumPy for the same reason. Levers: a pooled per-thread 0-d
  operand for the operators, or a scalar overload set that disposes its temporary. Found by X24.
- [x] **X27 Storing a C# literal wrapped where NumPy raises, at every store entry point: silent wrong values.** The
  neighbour class of X24, found by its search. Done 2026-10-08 (commit 51abf1ad).
  - Proof (a 2,662-case probe: 11 dtypes × 11 literals × 22 store entry points against Linux NumPy 2.4.2, the LP64
    texts NumSharp pins): 873 differences. `a[0] = 1000` on int8 stored -24 (NumPy: `OverflowError: Python integer 1000
    out of bounds for int8`); `a[mask] = double.NaN` on int32 stored INT_MIN (`ValueError: cannot convert float NaN to
    integer`); `a[0] = new Complex(3, 4)` on float64 stored 3 (`TypeError: float() argument must be ...`);
    `np.full(3, -1, dtype=np.uint8)` filled 255; `float32.fill(2**60 + 2**36 + 1)` rounded once (NumPy twice, through
    the double: 0x5d800000, not 0x5d800001); `uint8.fill((Half)300)` raised where NumPy casts to 44; `np.put(a, [0],
    [])` raised ArgumentException where NumPy does nothing. After the fix: 179, all documented — np.copyto's house
    exception type and complex64 text, the C-undefined float64 → uint64 cast (MSVC pinned), `np.append`'s C# int32,
    np.where / np.select's complex64, and `np.array(object[], dtype)` (X28).
  - Cause: an `NDArray` parameter received a literal as a strong 0-d array and cast it; the `object`-typed stores ran
    their own partial weak-scalar checks (`CheckWeakScalarFitsInteger` in fill and the flat iterator: no LP64 texts,
    one rounding, no NumPy-scalar rule) or plain `Converts.ChangeType`.
  - Fix: the implicit conversion operators mark the 0-d array (`NDArray.WasPythonLiteral` = NumPy's
    `NPY_ARRAY_WAS_PYTHON_LITERAL` for an integer / float / double / Complex; `WasNumPyScalar` for Half / char), and
    the store paths pack a marked value with `ReductionInitial.TryPack` (the complete `PyArray_Pack` port) through
    `NDArray.PackForStore`; object-typed stores use `TryPack` / `AsStoreValue` directly. A NumPy scalar follows NumPy's
    route split (probed): int()-converted with the range check into a SIGNED integer on the `PyArray_Pack` routes
    (integer index, slice views, fill, `flat[i]`, `np.pad`), CAST on the `PyArray_FromAny` routes (fancy, mask, put,
    putmask, place, put_along_axis, insert, full, asarray, flat slices). np.full / np.full_like are NumPy's
    `copyto(casting='unsafe')` (a Python int range-checked, a float cast). np.put follows `PyArray_PutTo`'s order (an
    empty values array is a no-op). np.pad stores NumPy scalars (`ReductionInitial.PackNumPyScalar`). fill_diagonal packs
    before its early returns. Only the stores read the marks: the ufuncs keep reading every 0-d operand as weak.
  - Also fixed: a basic-index setter with an EMPTY selection (`a["4:"] = v`) stranded one pooled buffer per call (the
    empty view's fresh zero-length buffer was never released; found by the zero-leak gate's new store cases).
  - Gates: section S of `weak_scalar.jsonl` (15,301 cases, zero excuses; 5 of 5 planted bugs killed, 54–2,030 cases
    each); `Indexing/np.store.PythonLiteral.Test.cs` (14). The index oracle's setter-dtype replay now builds its
    np.int64 / np.float64 values as strong `NDArray.Scalar`s (the implicit conversion would mark them Python literals).
- [ ] **X28 `np.array(object[], dtype)` is not supported.** A C# `object[]` is NumPy's Python list in the boundary map,
  and `np.array([1000, 0], dtype=np.int8)` packs each element (OverflowError); NumSharp's `np.array(Array, DType)`
  throws `ArgumentOutOfRangeException` for any `object[]` (and `NotSupportedException` without a dtype), while
  `np.asanyarray(object[])` works. The coercion exists as numpy.polynomial's `PolySequence` (a port of
  `array_coercion.c`). Found by X27.
- [ ] **X29 A 0-d view from integer indexing stores like a 0-d array, not like NumPy's NumPy scalar.** `a[i] = b[j]`:
  NumPy's `b[j]` is a NumPy scalar, which a SIGNED integer destination range-checks on the `PyArray_Pack` routes
  (`int8[i] = int64_array[j]` with 1000 raises); NumSharp's `b[j]` is an unmarked 0-d view, which is cast (-24). Every
  in-range value agrees. Lever: mark the getter's full-integer-index result `WasNumPyScalar`. Found by X27.
- [x] **X30 numpy.ma's arraysetops were rewritten compositions, not NumPy's statements: in1d / isin flags never masked,
  ma.unique without `return_index` / `return_inverse`.** Done 2026-10-08 (commit fa7108b9; the rundown's bug 2).
  - Proof: removing the `[known gap] ma.isin/in1d` excuse turned all 8 corpus cases red (mask `[FFFFFF]` against
    NumPy's `[FFTTFF]`); a 120,000-call differential against a stable copy of NumPy's statements then showed
    `unique`, `intersect1d`, `setxor1d`, `union1d` and `setdiff1d` diverging too (NaN beside masked slots, masked keys
    tied with equal values, setxor1d's filled comparison). NumPy: `ma.in1d([3, 5], ma.array([3, 9], mask=[0, 1]))` is
    `[True, --]`; NumSharp gave `[True, False]` with no mask.
  - Cause: each ran the plain set operation over the compressed values and appended one masked entry by a presence
    rule. NumPy's answer comes from `_unique1d`, `ma.concatenate`, the masked sort / argsort (masked slots keyed by the
    fill value) and the masked `==` / `!=` (two masked slots equal; every comparison that reads a masked slot masked).
  - Fix: `Ma/NDMaskedArray.SetOps.cs` ports `numpy/ma/extras.py` statement by statement (unique with NumPy's
    `return_index` / `return_inverse` and a `MaskedArrayModule.UniqueResult` tuple; `ma.concatenate` gains
    `axis=None`). Neighbours found and fixed on the way: `np.argsort`'s float32/float64 radix put every `-0.0` before
    `+0.0` (NumPy's stable order keeps them in input order; `np.lexsort` inherited it), and `np.unique` of complex128
    without flags kept a NaN from an unstable partition (NumPy's hash path keeps each class's first element).
  - Perf: geomean 3.3× NumPy (float64 3.6–8.7×, int64 1.5–2.6×) after two exact levers: a masked integer argsort sorts
    the unmasked values alone when none equals the fill key, and in1d's `order.argsort()` of a permutation is its
    inverse permutation (one scatter).
  - Gates: `ma_setops.jsonl` (7,030 cases, zero excuses, 8 planted bugs killed), `Ma/np.ma.setops.NumPyPort.Test.cs`,
    `Sorting/np.argsort.SignedZero.Test.cs`, `Manipulation/np.unique.ComplexHash.Test.cs`.
- [x] **X31 numpy.ma's put with a hard mask was a rewrite, not NumPy's statements: an N-D hard-mask put wrote where NumPy
  raises IndexError; np.put / np.take checked in another order and np.put read overlapping operands mid-write.** Done
  2026-10-08 (commit 23a51194; the rundown's bug 3).
  - Proof: NumPy's `MaskedArray.put` (`numpy/ma/core.py`) looks the indices up with `self._mask[indices]` BEFORE it
    writes. On an N-D mask that lookup indexes axis 0, and its N-D result then boolean-indexes the 1-D index array, so
    almost every N-D hard-mask put raises (`index 4 is out of bounds for axis 0 with size 2`, `too many indices for
    array: array is 1-dimensional, but 2 were indexed`) and writes nothing. NumSharp applied the mask to the flat
    positions and wrote. The 1-D branch differed too: NumPy zero-pads short values (`values.resize(indices.shape)`,
    never a cycle), refuses masked values (`MaskedArray.resize` always raises), and refuses a caller's ndarray of another
    size that it does not own or that another reference holds (`ndarray.resize`'s refcheck).
  - Fix: `MaskedArrayModule.put` runs NumPy's statements over the house primitives (the mask lookup, `values.resize`
    as `ResizeValuesForHardPut`, the two boolean selections, np.put of the data, np.put of the mask, the shrink to
    nomask). One documented divergence: a caller's same-size values are relabeled through an alias view, not reshaped
    in place (NumSharp's `data` / `mask` return internal arrays, so an in-place reshape of a caller's array could write
    into a masked array's storage).
  - Neighbours, found by the search and fixed:
    - `np.put` is `PyArray_PutTo` in its order: mode, writeability, the index cast ('safe'), `cannot replace elements of
      an empty array`, the values packed, an empty values array a no-op, then the target copied when its values or
      indices may share its memory (written back after). Raise-mode errors are NumPy's IndexError (`index 9 is out of
      bounds for axis 0 with size 4`), with NumPy's partial writes. Before: a values view of the target read slots the
      put had already written (silent wrong values), ArgumentOutOfRangeException where NumPy raises IndexError.
    - `np.take` is `PyArray_TakeFrom`: a 0-d source's axis checked against one dimension (AxisError), an `out` checked
      for shape, then cast, then writeability, `cannot do a non-empty take from an empty axes.`, and the bounds of an
      empty slab checked.
    - One clip-mode converter, NumPy's `PyArray_ClipmodeConverter` (`Indexing/np.clipmode.cs`: the first letter, then
      the exact lowercase name, NumPy's two texts), for put / take / choose / ravel_multi_index, and the sequence form's
      length text.
    - The selection kernels read their operands linearly, while `astype`'s default order 'K' keeps an F-order operand
      F-ordered: put's values and indices, take's indices, place's values and mask, putmask's mask and unravel_index's
      indices were read in memory order (silent wrong values: `np.take(src, int16 F-order [[3, 2], [1, 0]])` read
      `[3, 1, 2, 0]`). Each cast asks for order 'C' now.
    - The indexers raise NumPy's `prepare_index` IndexErrors: a boolean index that does not fit names its axis
      (`boolean index did not match indexed array along axis 1; size of axis is 3 but size of corresponding boolean axis
      is 4`, where the old IndexOutOfRangeException named axis 0 and the mask's first length), an index array that is
      neither integer nor bool is `arrays used as indices must be of integer (or boolean) type` whatever its size (an
      empty float array was an empty fancy index), checked before the too-many check, and an empty bool array in a tuple
      stays a mask (NumPy force-casts an empty index to intp only when it is a Python list).
    - The boolean-mask getter allocated its result zero-filled, from OS-zeroed pages, so every call page-faulted it
      (165 of the 184 µs of a 40,000-element int64 selection, 0.58× NumPy). Both gathers write every element, so the
      result skips the fill (`NDArray.ForOverwrite`): 31.6 µs, 3.3× NumPy. ma.put's hard branch runs two of them.
  - Perf (NPY/NS, two pinned P-cores, a warm-up pass first; int64 / float64 × 1K / 100K / 1M): ma.put soft 3.1–5.9×,
    hard 1.57–4.7× (1.10–1.51× before the boolean-mask fix), a plain target 2.4–4.2×, np.put 2.5–4.9×, np.take 3.1–3.4×.
  - Gates: `ma_put.jsonl` (3,660 cases, 1,900 NumPy errors with strict error types, zero excuses); the aliasing tier's
    `put_selfvals` / `put_selfidx` (243 cases: values or indices a view of the target); `Ma/np.ma.put.NumPyPort.Test.cs`
    (10), `Indexing/np.put.take.NumPyPort.Test.cs` (6), `Selection/IndexErrors.NumPyPort.Test.cs` (5). 14 planted bugs,
    each killed: the six ma.put ones by the tier (34–479 cases each), np.put's missing overlap copy by the aliasing tier
    (50 cases), the order, clip-mode, take and index-error ones by the unit tests (one survivor, an empty float index
    accepted, needed two more unit cases: the dtype error comes before a 0-d target's too-many error).
- [ ] **X32 Error texts the X31 search found and did not fix.**
  - `OrderResolver`'s order-argument texts (used library-wide) are not NumPy's `order must be one of 'C', 'F', 'A', or
    'K' (got 'Q')`.
  - `np.ravel_multi_index` / `np.unravel_index` validate `order` with one ArgumentException text; NumPy raises the text
    above while it parses the arguments, and `only 'C' or 'F' order is permitted` after the mode conversion.
  - `np.insert` with values that do not broadcast raises IncorrectShapeException with an NDIter remapped-shapes text;
    NumPy raises `ValueError: could not broadcast input array from shape (2,2) into shape (2,)`.
  - A multi-index's invalid-item text quotes `':'` and appends `(got ...)`; NumPy's is `only integers, slices (`:`),
    ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices`.
  - An `object[]` (Python list) store value is cast through `np.asanyarray`, not packed element by element (X28's
    coercion gap, seen from the store side).
- [x] **X33 numpy.ma's numeric fill value was the caller's raw C# value: never converted to the data dtype, never
  materialized, never refused.** Done 2026-10-08 (commit de07ca81; the rundown's bug 4; the `ma.apply_along_axis` and
  `default_fill_value(obj)` neighbours in the commit after it).
  - Proof: NumPy's `_check_fill_value` stores `np.array(np.asarray(fill_value, dtype=ndtype))`, a 0-d array of the
    data's dtype, and the getter returns `_fill_value[()]`. NumSharp kept the C# value as given: `astype(int64)` kept
    7.5 (NumPy: `np.int64(7)`), `fill_value=300` on int8 was accepted (NumPy: `TypeError: Cannot convert fill_value 300
    to dtype int8`), and every read computed the default again, so a later `fill_value = 5` stored an int8 where NumPy
    writes 5 INTO the int64 default it stored on the first read (`np.int64(5)`). The text dtypes already had the
    store / convert / materialize model (26f7e32c).
  - Fix (`Ma/NDMaskedArray.cs`): `_fill_value` is an NDArray (null = None) for every dtype.
    - `CheckFillValue` is `_check_fill_value`: a str for a non-text dtype is refused by its type; anything else goes
      through `NDArray.AsStoreValue` (`PyArray_FromAny`): a C# number is a Python scalar, packed with the range checks,
      its OverflowError / ValueError rewrapped as NumPy's TypeError with the value in Python's `%s`
      (`PolyIndexArgument.Format`: `1e+20`, `nan`); a complex into a real dtype raises Python's own `int()` / `float()`
      TypeError; a `Half` / `char` / `decimal` (NumPy scalars) or an array is cast (`np.float16(300)` into int8 is 44).
    - The getter stores `DefaultFill` on its first read: the DEFAULT's own dtype (int64 999999 for signed integers,
      uint64 for unsigned, NumPy's `np.uint(...)`; float64 1e20 for every float; complex128; bool True).
    - The setter writes INTO a stored array (`_fill_value[()] = target`, unsafe), with NumPy's three texts for an array
      with dimensions into a stored 0-d fill.
    - `UpdateFrom` (`_update_from`) shares the stored object; `FinalizeFrom` (`__array_finalize__`) stores a converted
      copy, so `astype` / `view` / `byteswap` / `argpartition` convert a STORED fill and leave an unread one unread.
    - `filled()` casts the stored fill under `same_kind` (`np.copyto`), as NumPy does.
    - The repr prints the fill by the STORED dtype (`fill_value=np.int64(999999)` beside int8 data, `fill_value=1` when
      the dtypes agree).
    - The masked singleton's getter and setter raise `AttributeError: attributes of masked are not writeable`.
  - Neighbours, found by the search and fixed:
    - `byteswap`, `argpartition`, `ma.cumsum` and `ma.cumprod` gave the result the SOURCE's mask, so a mask write through
      the result reached the source; NumPy copies it (`__array_finalize__`'s `_mask.astype(...)`, `__setmask__`'s
      `make_mask_none` + `.flat =`). `byteswap` / `argpartition` also came back soft.
    - `view(dtype)` with a mask and another itemsize raised NotSupportedException; NumPy's dtype setter raises
      `ValueError: cannot reshape array of size 4 into shape (16,)`, which NumSharp raises as its house
      IncorrectShapeException. A view finalizes at the SOURCE dtype (NumPy sets the dtype afterwards), resets a stored
      fill when a dtype is given, and ignores `fill_value=` when nothing is stored.
    - `x + 1` on an int8 / uint8 / int16 / uint16 / bool masked array stayed narrow; numpy.ma's `getdata` makes a Python
      int a strong int64. The C# literal reaches `operator +(NDMaskedArray, NDArray)` (an NDArray is the better conversion
      target than `object`) as a marked 0-d int32 array, which `BareScalarStrongDtype` now reads as the Python int.
    - The masked print showed float16 / float32 elements with their own short repr (`1.732`); NumPy's
      `_insert_masked_print` converts with `astype(object)`, which makes Python floats (`1.732421875`).
    - The masked indexer parsed a lone slice string twice (once per array), ~0.8 µs each: `x["1:"]` took 2.84 µs, 1.33–
      1.40× NumPy. It parses once now (1.77 µs). `GetView` rewrites its slice ARRAY (a newaxis entry becomes
      `Slice.All`), so the mask indexes a copy of the parsed array (finding 45).
    - `ma.apply_along_axis` skipped NumPy's closing statement `result.fill_value = ma.default_fill_value(result)`: the
      result stores the default converted to ITS dtype, so an int8 / uint8 / int16 / uint16 result raises `Cannot convert
      fill_value 999999 to dtype uint8` (after every `func1d` call) and every other result reads the default at its own
      dtype (`np.int32(999999)`, `np.float16(inf)`, the repr's plain `fill_value=999999`). `ResultDefaultFill` stores it
      through `StoreFill`, so it stays in the scope with the new data and mask (finding 43). journey4's surface tier
      carved this as gap MS6 (`b08d0759`).
    - `ma.default_fill_value(obj)` took only a dtype; NumPy's `_get_dtype_of` takes an array (its dtype), a masked array, a
      scalar or a list (`np.asanyarray(obj).dtype`). A `default_fill_value(object)` overload does that (a `byte[]` is the
      bytes scalar, a `List` / `object[]` a Python list), and None answers NumPy's `'?'` (`default_filler['O']`) in both
      overloads (a `null` binds the `DType` one). A literal `0` binds the object overload (the two user-defined
      conversions `0 → NPTypeCode → DType` and `0 → NPY_TYPES → DType` are ambiguous, so the `DType` overload is not
      applicable), as NumPy's `default_fill_value(0)` is 999999.
    - The other fill-value gaps journey4's surface tier carved (MS7 `masked_object` sets no fill, MS8 its range error,
      MS9 its no-match shrink, MS12 `set_fill_value` stores the raw value, MS13 `common_fill_value` compares raw values,
      MS14 `maximum_fill_value(bool)`) already match NumPy on this branch (this fix and 77b68d89); un-carve them and drop
      their `OpenBugs.MaSurface.cs` pins when the branch is merged into journey4.
  - Divergence (documented): `filled()` whose stored fill cannot reach the data under `same_kind` (a complex fill shared
    with `ma.absolute`'s float result) is NumPy's object-array fallback; NumSharp raises NotSupportedException.
  - Not changed: the masked SETTER still parses a lone slice string twice (not measured).
  - Leak gates: a masked result's fill is now an array the result owns, so the corpus sweep's `CollectDisposables`
    collects it, and `Keep` / `LeakCatalogue.KeepNow` keep the operands' and fixtures' current fills.
  - Perf (NPY/NS, two pinned P-cores, a warm-up pass first; int8 / int64 / float64 × 1K / 100K / 1M; 93 cells): geomean
    5.60×, min 1.79× (`fill_value = 3` at 100K). Construct with a fill 8.3–13.2×, construct and read 9.8–19.5×,
    `astype` 2.8–5.8×, `filled()` 3.4–14.4×, `view` 4.8–7.8×, slice 2.1–3.3×, `common_fill_value` 7.4–10.7×, the setter
    1.8–2.6×, `x + 1` 2.5–5.2×, the repr 8.5–14.3×.
  - Gates: `ma_fill.jsonl` (3,867 call-history scripts: construct → read / set / derive → observe; zero excuses, strict
    types); `Ma/np.ma.fill_value.NumPyPort.Test.cs` (10); `NDMaskedArrayTests.Indexer_Get_SliceString_ParsedOnce_KeepsMaskAligned`.
    15 planted bugs, each killed: 11 by the tier (3–556 cases each), three (`filled()` casting unsafe, `cumsum` and
    `byteswap` sharing the source's mask) by the unit tests only, the slice-array copy by the indexer test. Two (the
    unsigned default stored as int64, C#'s exponent format in the error text) were at first killed by the tier alone;
    the unit tests now catch them too (finding 42).
- [x] **X34 np.linalg's factorisations threw on an empty operand without a backend; NumPy answers it without
  arithmetic.** Done 2026-10-08 (this commit; the rundown's bug 5).
  - Proof: with no backend, `qr` / `svd` / `svdvals` / `eig` / `eigvals` / `eigh` / `eigvalsh` / `cholesky` / `lstsq` of
    an empty matrix or a zero-size batch (`(0, 3)`, `(3, 0)`, `(2, 0, 0)`, `(0, 3, 3)`) raised
    `OpenBlasMissingBackendException`, and so did the composites over them (`pinv` / `matrix_rank` / `cond` / the SVD
    norms of a zero-size batch). NumPy's gufunc loops zero times over an empty batch, and LAPACK returns at once for an
    empty matrix, after which NumPy's C wrappers write the identities LAPACK left out: `np.linalg.svd(np.zeros((0, 3)))`
    has `Vh = eye(3)`, `np.linalg.qr(np.zeros((3, 0)), 'complete')` has `Q = eye(3)`.
  - Fix: `Backends/Default/LinearAlgebra/LinAlgEmpty.cs` gives NumPy's empty result for the six seams (Cholesky, Eig,
    Eigh, Lstsq, Qr, Svd), and each `TensorEngine` virtual takes it BEFORE it asks a backend, so the answer is the same
    with or without OpenBLAS. Dtypes are the backend's: the operand's `result_t`, the real type for S and eigh's
    eigenvalues, complex128 for `geev` (the realness collapse then makes an empty real operand's result real: NumPy's
    `all(w.imag == 0)` is vacuously true). `lstsq` of an M×0 matrix: x `(0, K)`, rank int32 0, no singular values, and
    the residuals NumPy's SEQUENTIAL `res += el*el` over every row of b (a `cumsum` down the columns, squares in double;
    `lstsq(zeros((3, 0)), [1, 2, 3])` is `[14.]`). The composites inherit it through their own NumPy statements:
    `matrix_rank` of an empty matrix raises `S.max()`'s ValueError, the 2-norm is 0 (`amax(initial=0)`), -2 raises
    `amin`'s, `cond` refuses an empty matrix (`_is_empty_2d`), a zero-size batch gives empty results.
  - Neighbours, found by the tier and fixed:
    - NDIter refused a written zero-size operand: its reduction check tested `stride == 0`, NumPy's
      `npyiter_fill_axisdata` tests the operand's SHAPE (its dimension is 1 or missing under a longer axis). A zero-size
      array's strides are 0 without a broadcast, so `np.nditer<double>(np.zeros((3, 0)), writeable: true)` threw (the
      empty complex Frobenius norm runs it). `NDIterRef.IsBroadcastAxis` is the shape test; a writeable array with an
      explicit 0 stride now iterates too, as in NumPy. The refusal itself is NumPy's `ValueError` numbering the axis from
      the inside (`(1, 3, 4)` against `(2, 3, 4)` is "dimension 2"; it said 0, as an `ArgumentException`).
    - An empty reduction without an identity raised the base `ArgumentException` on the engine routes (`np.max` /
      `np.min` / the nan forms / `np.ptp`, `np.argmax` / `np.argmin`, np.evaluate's `Min` / `Max` / `ArgMax` /
      `ArgMin`); NumPy raises `ValueError`, and the keyword route already did. `ValueError` derives from
      `ArgumentException`, so a handler for that still catches it; 20 unit tests pinned the exact old type.
  - Not changed: `lstsq` of a non-empty `a` with an empty `b` still needs the backend (`gelsd` computes `a`'s singular
    values).
  - Gates: `linalg_empty.jsonl` (5,075 cases, 1,465 NumPy errors; zero excuses and strict error types: `CheckError`
    accepts only the house exception of NumPy's name for valueclass `linalg_empty`);
    `LinearAlgebra/LinAlgEmpty.NumPyPort.Test.cs` (6); the dtype-spread ledger's eight LAPACK entries (cond, eig, eigh,
    eigvals, norm, pinv, svd, svdvals) retired, the tier spreads them over 7–8 dtypes. 16 planted bugs, every one
    killed (the tier turns 2–107 cases red for the 11 it can see; the unit tests catch the other five — the NDIter
    axis number and type, the empty max / argmax / argmin types): the identities of a complete QR and a full SVD,
    eigh's and the SVD's real dtypes, the residual order (only after the single-column cases, finding 48), the rank's
    int32, raw's transposed `h`, the solution's N rows, the complex abs2, the stacked identity, the shape test, and the
    four error types.
  - Perf (NPY/NS, two pinned P-cores, best of 9, 20 cells: qr / svd / svdvals / eig / eigvals / eigh / eigvalsh /
    cholesky / lstsq / pinv / matrix_rank / cond / norm over empty matrices and zero-size batches, results disposed):
    geomean 5.92×, min 1.62× (`lstsq` of a 3×0 matrix with values in `b`: the residual sum's five allocations, ~6 µs);
    the full SVD of `(0, 512)` (a 512×512 identity) 3.76×, of `(100, 64, 0)` (100 identities of order 64) 6.80×.
- [x] **X35 np.nditer refused string arrays; the iterator carries NumPy's descriptors.** Done 2026-10-08 (this commit;
  the rundown's bug 6).
  - Proof: `np.nditer(np.array(['ab', 'c']))` raised NotSupportedException ("the iterator sizes elements by numeric
    storage lane"); NumPy yields 0-d `<U2` views, converts between widths and between `S` and `U` under buffering or a
    copy route, and writes a `readwrite` view through.
  - Fix: NDIter's construction takes the per-operand descriptors (`NDIterRef.AdvancedNew(..., opDescrs)`,
    `NDIter.Descriptors.cs`): NumPy's `npyiter_check_casting` read and write-back checks, the copy routes of
    `npyiter_allocate_arrays` (a read-only 0-d cast operand, `copy`, `updateifcopy` written back at close, the
    "required copying or buffering" TypeError), allocated operands laid out in the settled iteration order, string runs
    in the buffer manager and the string copy kernels. The wrapper (`np.nditer.cs`) is nditer_pywrap's: the
    started / finished flags, every getter's and setter's checks and texts, `it[i] = v` packing a Python int with
    NumPy's range check, the views' write flags, the delayed-allocation state; `np.nested_iters` splits the flags and
    dtypes per level (`NpyIter_NestedIters`).
  - Neighbours, found by the tiers and fixed:
    - the flat index rides its own strides (NumPy's index "operand"; `NDIterState.IndexStrides`): it was recomputed from
      the coordinates, so after `remove_multi_index` coalesced the axes an `f_index` counted in C order; an
      index-tracking iterator now coalesces when the index allows, as NumPy's does, and GotoIndex reads the strides;
    - `shape` without a multi-index lists the axes innermost first (NumPy's axisdata order);
    - `remove_multi_index` only coalesces: it re-sorted the axes by stride, so a forced C / F order walked memory
      order, and the ascending layout reversed any walk that did not collapse to one axis;
    - the descriptor path orders K-order axes with NumPy's `npyiter_find_best_axis_ordering` (a zero stride is
      ambiguous, C order wins a conflict), with no broadcast fallback to C order;
    - an allocated reduction output walked the reduced axis outermost (it abstained from the contiguity test, and
      `CoalesceAxes` cannot leave an iterator partly merged);
    - the op_axes reduction checks are `npyiter_check_reduce_ok_and_set_flags` (innermost axis first, NumPy's texts and
      ValueError, the operand axis named); an explicit reduction axis needs no `reduce_ok`; an explicit reduction along
      an axis longer than 1 is NumPy's ValueError;
    - a construction that failed after allocating an output leaked it to the finalizer (`_allocatedSlots`);
    - a stale `[OpenBugs]` (the buffered refill, T1.2) checked by reflection for a method name; it now iterates.
  - Not changed: a BUFFERED reduction through np.nditer computes wrong values for every dtype (the legacy double loop;
    `OpenBugs.NditerBufferedReduce.cs`, 4 pins with NumPy's values) — the fix is a port of NumPy's buffered-reduce
    design (`npyiter_find_buffering_setup`'s core / outer split and the reduce double loop). A buffered iterator's
    `index` is uninitialized memory in NumPy (left out of the tiers). Conversions between text and numbers raise
    phase 2's `StringComputationNotSupportedException`. A null `it[i] = null` is ArgumentNullException (NumPy's
    `int()` TypeError).
  - Gates: `nditer_construct.jsonl` (9,574 cases) and `nditer_protocol.jsonl` (1,219: its `coalesce` and `reduce`
    sections hold the neighbours), zero excuses and strict error types; the string oracle's `striter` section (5,756);
    `APIs/np.nditer.NumPyParity.Test.cs` (24); `Backends/Iterators/NDIterReductionAxisEncodingTests.cs` (+1). Nine
    planted bugs (the index flip, the index in coalescing, the re-sort, the ordering, the error axis, the partial
    coalesce, the shape order, the index carry, the allocation release), every one killed by the unit tests.
  - Perf (NPY/NS, two pinned P-cores, best of 21): construction 1.67× (one operand), 1.90× (`c_index`), 1.77× (an
    op_axes reduction); a `c_index` walk of 10,000 elements 13.9×, an `f_index` walk after `remove_multi_index`
    13.3×; an external-loop walk of 1M doubles (one chunk) 1.33×, of a 10,000-string array 1.25×. Under parity: a
    construction that allocates an output, 0.96× (`np.zeros(1000)` is about half of its 1.1 µs; the up-front
    allocation is reused when it already has the iteration layout, which saved ~0.6 µs), and a per-element walk of
    10,000 strings, 0.14× (the per-element NDArray view floor every `it[0]` loop pays, ~0.5 µs a view).

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
   Its axis twin is X12 (A6), and numpy.ma's mean / var / anom reach both wherever `filled(0)` is the view itself
   (no mask, or an all-False one).
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
16. **A port's eager intermediates must take NumPy's layout** (A6). NumPy lays a ufunc's result out with NpyIter (a
   permuted input's axis order is kept), and a later sum walks that memory order. NumSharp's eager ufuncs write C order
   for an input that is neither C- nor F-contiguous, so a port composed of them adds in another order (two ULP on a
   transposed 3-D int32 variance). numpy.ma's port allocates each intermediate with `Shape.NpyIterOutputShape` and
   writes into it in place, as NumPy's `out=x` does.
17. **An out= route is not always the plain call** (A6). np.power's out= route skips NumPy's scalar-exponent fast
   paths (X13), so writing power's result into a layout buffer with out= changed 73 probe cases; computing it with the
   plain call and copying keeps NumPy's bits.
18. **A carve can hide a bug of another class** (A6). The mean_loops tier carves every flat broadcast float cell as an
   order residual (finding 10); the cells' uncarved replay showed a dtype bug under them (X14). Replay a tier uncarved
   and classify every carved divergence before committing the carve (the `ma_meanvar` generator takes `--uncarved`).
19. **Benchmark masked operations with a random mask** (A6). X11's first measurement (every 7th element masked) made
   the branchy masked copy look fast for float32 / float64; with a 35 % random mask every same-dtype masked copy costs
   2.5–6 ns per element, in NumPy too.
20. **Check what a probe layout is** (A6). The A6 probe's "T" layout (`np.ascontiguousarray(x.swapaxes(0, -1))
   .swapaxes(0, -1)`) reverses every axis of a 2-D or 3-D input, which is F-contiguous, the same as "F": it covered no
   other permutation. The in-place-square layout bug (finding 16) showed only in a second probe over the explicit axis
   permutations.
21. **NumPy copies an overlapping source first only for a multi-axis or an opposite-direction 1-D destination** (X16).
   Every other overlapping 1-D assignment walks NumPy's raw order: a negative destination stride flipped (both operands
   from their last element), then backward when the source starts below the destination inside it. With equal strides
   that order equals copying first; with unequal strides it reads its own writes (`a[3:19:2] = a[6:14]`). NDIter.Copy
   (every assignment: np.copyto, slice assignment) copied first for every non-unit-stride 1-D overlap, and np.copyto's
   masked path never checked overlap at all.
22. **A small internal temporary must not be an NDArray** (X16). A 1K masked float32 → float64 copy spent 0.54 of its
   0.78 µs building the staged source as an NDArray and running NDIter.Copy (allocation, ARC, the iterator's state)
   around a 0.2 µs cast. A stack buffer (≤ 16 KB, `[SkipLocalsInit]`) or a pooled one, filled by the same cast kernels,
   takes it to 0.32 µs.
23. **An unmasked copy is memcpy, in NumPy too** (X16). `np.copyto(dst, src)` of one dtype runs at
   `Buffer.MemoryCopy`'s speed (probed: 13.1 µs for 1 MB, 14.0 µs for 800 KB, both equal to the bare call); NumPy's
   memmove is as fast. At 100K and 1M the cells sit at 0.73–1.18× run to run, bandwidth-bound on both sides; only the
   1K cells, where NumPy's call overhead counts, are faster (4.3–7×).
24. **Address a strided loop with running pointers** (X16). The select loop's `dst + i * stride` (three multiplies per
   element) cost a third of its time: 0.77 → 0.48 µs for 1,000 strided elements with pointers bumped by the stride and
   the loop unrolled four times (each element's store still before the next element's loads, so an overlapping run
   keeps NumPy's order).
25. **Measure the route the consumer takes** (X15). Fixing float16's broadcast kernels on the plain call left
   numpy.ma's anom at 0.86×: it subtracts into an out= buffer, whose route (the iterator's scalar float16 body) the
   plain-call benchmark never ran. A speed item names its consumers' call forms (plain, out=, where=, in place).
26. **A loop that tests its own steps makes the call schedule part of the answer** (X13). NumPy's float power loops
   branch on `steps[1] == 0`, so whether `x ** 2` is `x*x` or pow depends on the calls the trivial loop and NpyIter
   make: a cast out, the buffer size, an extent-1 axis or a column's row length change the bits. Porting the loop body
   is not enough. The decision is ported once (`NumPyUfuncInputSteps`) and shared by the two loops that read their
   steps (np.clip's const loop and power).
27. **A mutation survivor is a missing case until it is proved equivalent** (X13). Two of 13 mutants of the step
   decision survived the 7,773-case tier: no case had one-element operands with a cast out, and none had rows of
   exactly bufsize with every input cast. NumPy's answer flips at both (probed), so they became section I.
28. **A restored source must look newer than the mutant's build** (X13). The mutation script restored files with
   `shutil.copy2`, which keeps the old modification time, so MSBuild's incremental check kept the last mutant compiled
   in: the next "real" build failed 3 tier cases. Restore with a fresh modification time (copy, then `os.utime`) and
   rebuild before trusting a result.
29. **A float NaN constant is not a test input** (X13). RyuJIT folds float constants through double, so
   `BitConverter.UInt32BitsToSingle(0x7F800001)` written as a constant can reach the code as a QUIET NaN, and
   `MathF.Pow(constant sNaN, 0)` can fold to 1: a signalling-NaN test passed or failed with the JIT tier. Build such
   inputs from integer arrays (`np.array(bits).view(np.float32)`) or read them from an array at run time.
30. **C's float32 `1.0 / x` is a double division, and its double rounding is innocuous** (X13). FLOAT_power's
   reciprocal divides in double and rounds to float on the store; 53 ≥ 2·24 + 2 bits, so a float32 division gives the
   same bits. Verified over all 2^32 float32 patterns, scalar and vector, NaN payloads included (a planted difference
   every 1,000,003 patterns was counted exactly).
31. **A test can pass only because an earlier test compiled a kernel** (X13).
   `InnerLoopKernelKeyTests.Production_Route_Registers_Kernel_Under_Both_Keys` read the inner-loop caches after an
   int16 add into a contiguous out. Since 2026-10-04 that call takes the trivial out= loop and never compiles the
   kernel, so the test passed only when it ran after another one. It now uses a strided out.
32. **A workaround can outlive its bug** (A6 / X13). numpy.ma's port computes `abs(danom) ** 2` with the plain call
   and copies the result into its layout buffer, because the out= route skipped the fast path (finding 17). X13 fixed
   that route, so the copy can become a plain `out=` (not changed here; A6's tiers hold the values either way).
33. **A value gate never sees a write past the result** (X25). The oracle and the unit tests compare the elements of a
   result, and a kernel's extra byte lies outside every array they compare. The comparison kernels' overrun passed
   every gate since it was written. It showed up only when a planted lane-order bug failed at a length the unrolled
   loop should never have reached. Use sentinel slots past an `out=` view or past a directly called kernel's buffer,
   and the opt-in NUMSHARP_DEBUG_GUARD_PAGES pool: every buffer then ends against an inaccessible page, so a read
   or a write past it faults at once. Print each case before it runs; after a fault, the last line names it.
34. **NumPy stores a NumPy scalar two ways, by route** (X27). `PyArray_Pack` lets a signed integer DType treat every
   NumPy scalar as its own (`signed_integers_is_known_scalar_types`) and int()-converts it with the range check, so
   `a[0] = np.float16(300)` on int8 raises; `PyArray_FromAny` converts a NumPy scalar like a 0-d array (a cast), so
   `a[[0]] = np.float16(300)` stores 44. Which one a store uses follows its C code, not its spelling: an integer index,
   a slice view (`PyArray_CopyObject`), `fill` and `flat[i]` pack; fancy, mask, put, putmask, place, insert, full,
   asarray and flat slices convert. A Python scalar is packed on both.
35. **A store test that reads only the destination misses the temporaries** (X27). The zero-leak gate found that an
   empty-selection assignment stranded a buffer (the empty view's own) and that the oracle replay's mask leaked through
   a reshape VIEW that kept a counted reference on the disposed flat array — neither shows in a value. Dispose a
   reshape view as well as its source.
36. **NumPy's quicksort tie order is host-dependent, so an oracle for sort-based code must classify ties** (X30). NumPy
   2.4.2's `quicksort` argsort is unstable from 4 elements for 32/64-bit dtypes (x86-simd-sort AVX2; AVX-512 hosts run
   other networks) and from 17 for the others (introsort). It decides a numpy.ma set operation only when an unstable
   sort meets equal keys that are not interchangeable; replaying NumPy's statements with stable sorts and keeping a
   case only when the answers agree (and asserting every dropped case was such a tie) separates the inherent cases from
   the port's bugs, where a blanket excuse would hide both.
37. **A warm-up of 250 ms does not reach tier-1 for the first cells of a benchmark** (X30). The first int64 cells of the
   set-operation benchmark ran 1.3–2× slow (an identical `np.argsort` of the same permutation cost 19.1 µs in the first
   block and 8.6 µs in the second); `DOTNET_TC_CallCountingDelayMs=0` plus a warm-up pass over every cell before
   measuring removed it.
38. **`astype`'s default order 'K' under a kernel that reads by flat position is a silent-wrong-values class** (X31).
   A cast copy of an F-order operand stays F-ordered, so a kernel that reads the cast operand linearly reads it in memory
   order. Five selection functions had it (put, take, place, putmask, unravel_index); every oracle case gave them C-order
   operands. Ask for order 'C' wherever the consumer is linear.
39. **An oracle whose operands are independent buffers cannot see aliasing** (X31). The `ma_put` tier killed every
   ma.put mutant and missed np.put's missing overlap copy, because two corpus operands never share a base. A case that
   needs one has to rebuild the view inside the replay (the aliasing tier's `put_selfvals` / `put_selfidx`).
40. **The leak sweep replays a case twice over the same operands, so a mutating case must not consume its own output**
   (X31). `put_selfidx`'s second run read the values its first run wrote (100..105) as indices and raised; the case now
   puts into a copy of its target.
41. **A zero-filled result that a kernel overwrites page-faults on every call** (X31). `new NDArray(dtype, shape)` takes
   OS-zeroed pages, whose first touch costs ~0.5 µs per 4 KB; the pool's recycled buffer (`fillZeros: false`,
   `NDArray.ForOverwrite`) costs nothing to touch. The boolean-mask getter lost 165 µs of 184 to it. np.repeat allocates
   its four results the same way (`Manipulation/np.repeat.cs`, lines 133, 168, 224 and 275) — not measured, not changed.
42. **AwesomeAssertions' object `Be` compares boxed numbers across types, and `WithMessage` is a case-insensitive
   wildcard pattern** (X33). `x.fill_value.Should().Be(999999L)` passed with the getter returning `999999UL`, and
   `WithMessage("Cannot convert fill_value 1e+20 …")` passed C#'s `1E+20`: two planted bugs the oracle tier caught and the
   unit tests did not. A test of a NumPy scalar's TYPE needs an explicit `GetType()` check; a test of NumPy's text needs
   whole-string equality (`.Which.Message.Should().Be(text)`).
43. **A stored attribute's lifetime follows WHO stores it** (X33). A fill built with a NEW masked array stays in the
   ambient scope with its data and mask and leaves with the result (`Yield`); detaching every new fill made an
   intermediate's fill (`ma.trace`'s `astype` copy) a finalizer leak. A fill stored into an EXISTING array (the getter's
   default, the setter's replacement) must be detached, or a scope disposes it under a longer-lived array.
44. **A state that depends on call history needs scripts, not cases** (X33). Whether `astype` converts a fill, and to
   what, depends on whether something READ the fill first (an unread float64 array casts to int64 with the fill 999999,
   a read one with `long.MinValue`). The `ma_fill` tier records scripts (construct → read / set / derive → observe) and
   replays them step for step; a single-call case cannot express the order.
45. **`UnmanagedStorage.GetView` rewrites its slice ARRAY argument** (X33): a newaxis entry becomes `Slice.All` once the
   axis is inserted. Reusing one parsed `Slice[]` for two arrays gives the second the first's shape without the new axes
   — the same element count, so only a SHAPE assertion shows it (or, with two newaxis entries around an integer, a
   too-many-indices error). Give each consumer its own copy of the array; the `Slice` objects are not written.
46. **A zero stride is not a broadcast** (X34). NDIter refused a written operand wherever a stride was 0; NumPy refuses
   it only where the operand's DIMENSION is 1 or missing. A zero-size array has zero strides on the axes before its
   zero-length axis (NumPy zeroes them all), so the typed iterator refused `np.zeros((3, 0))` read-write — and only an
   empty complex Frobenius norm ever handed it one.
47. **An oracle that maps a NumPy error to a class of .NET types cannot see a type regression** (X34). The ordinary
   tiers accept `ArgumentException` for NumPy's `ValueError`, so the empty `np.max` / `np.argmax` family raising the base
   type passed every tier for months; the first tier that holds the exact type (`linalg_empty`, valueclass-strict
   `CheckError`) found it on its first run.
48. **Only a long SINGLE column tells a pairwise sum from a sequential one** (X34). NumPy's pairwise sum runs one
   accumulator for fewer than 8 elements, and an axis-0 sum with two or more outputs is the sequential slab schedule
   either way; only one output over more than 8 rows takes the flat pairwise schedule. The first tier had a 20-row `b`
   of TWO columns, so a planted pairwise residual (`np.sum(squares, 0)` for the cumsum) passed the tier AND the unit
   tests. The 1-D and `(20, 1)` forms of the same column (one large square, nineteen ones: sequential 1e16, pairwise
   1e16 + 16) catch it. The same run hid the gap a second way: the unit filter's baseline already had one failing test
   (an `[OpenBugs]` pin, `Var_Float32_NumPyReturnsFloat64`, whose premise was false: NumPy's var of float32 is float32;
   removed), so "1 failed" read as a kill. Count failures against the unmutated tree, by name.
49. **A tracked flat index must ride strides, not be recomputed from coordinates** (X35). The coordinates name the
   operands' axes only until something coalesces them; NumPy keeps the index as one more "operand" whose strides are
   permuted, flipped and merged with the others, so `remove_multi_index` (or an index-tracking iterator's own
   coalescing) cannot break it. NumSharp recomputed it through `Perm`, which went on reading C coordinates of a merged
   axis — the f_index after `remove_multi_index` counted `0 1 2 …`. No tier saw it: the protocol scripts read `shape`
   only after the walk had finished (past the end) and left index iterators out of `remove`.
50. **An operand that abstains from the order still shapes the coalescing** (X35). Zeroing an allocated operand's
   strides keeps it out of the axis sort (NumPy has no array for it yet), but it also kept it out of the contiguity test
   that picks the coalescing pass — and a reduction output, laid out later, has stride 0 on its reduced axes, so the
   full-collapse pass left the iterator partly merged in its ascending layout and walked the reduced axis outermost.
   Decide the pass from the operands' REAL strides, or use the pass that merges whatever it can.
51. **A short warm-up measures tier 0** (X35). The first np.nditer benchmark (2,000 calls per round, 300 ms of warm-up)
   read 0.78 µs for a construction that takes 0.31 µs once tiered up (20,000 calls per round, 400–500 ms of warm-up);
   it made a 1.7× win read as a 0.66× loss. Size a construction benchmark's rounds in milliseconds, not calls.

## 7. State log

- **2026-10-08**: X35 done (np.nditer over strings: the iterator carries descriptors — cast checks, copy routes,
  string buffers, nditer_pywrap's state contract, `np.nested_iters`' per-level split; the neighbours: the flat index
  as strides, the shape order, `remove_multi_index`, NumPy's K-order sort on the descriptor path, the allocated
  reduction output's order, the op_axes reduction checks, the failed construction's leak). The `nditer_construct` and
  `nditer_protocol` tiers and the `striter` string section, zero excuses. The buffered reduction is pinned
  (`OpenBugs.NditerBufferedReduce.cs`). Findings 49–51 recorded. Commit: this commit.

- **2026-10-08**: X34 done (np.linalg's factorisations of an empty matrix or a zero-size batch are NumPy's answer without
  a backend: `LinAlgEmpty`, taken before the backend; NDIter's write-operand reduction check is NumPy's shape test, with
  NumPy's ValueError and axis number; the empty reductions raise NumPy's ValueError). The portable `linalg_empty` tier
  (5,075 cases, zero excuses, strict error types). Findings 46–48 recorded. Commit: this commit.

- **2026-10-08**: X33 done (numpy.ma's fill value as NumPy stores it, for every dtype: `_check_fill_value`'s conversion
  and texts, the default materialized at its own dtype on the first read, the setter writing into a stored array,
  `_update_from` sharing and `__array_finalize__` converting; `byteswap` / `argpartition` / `cumsum` / `cumprod` own
  their masks; `view`'s reshape error; `x + 1`'s strong int64; the float16 masked print; the masked slice parsing its
  string once). The `ma_fill` tier (3,867 call-history scripts, zero excuses, strict types); 15 planted bugs killed.
  Findings 42–45 recorded. Commit: de07ca81. The commit after it adds two neighbours: `ma.apply_along_axis` stores the
  default fill at its result's dtype (journey4's gap MS6) and `ma.default_fill_value` takes an array, a scalar or a list
  (`_get_dtype_of`; None is `'?'`); 2 more planted bugs killed by the unit tests.

- **2026-10-08**: X31 done (numpy.ma's put as NumPy's statements; np.put as `PyArray_PutTo`, np.take as
  `PyArray_TakeFrom`; one clip-mode converter; the selection kernels' order-'C' casts; the indexers' `prepare_index`
  IndexErrors; the boolean-mask getter without a zero fill). The `ma_put` tier (3,660 cases, zero excuses, strict error
  types) and 243 aliasing-tier put cases; 14 planted bugs killed. X32 and findings 38–41 recorded. Commit: 23a51194.

- **2026-10-08**: X30 done (numpy.ma's arraysetops as NumPy's statements; `np.argsort`'s ±0 order; `np.unique`'s
  complex hash representative). The `ma_setops` tier (7,030 cases, zero excuses; 8 planted bugs killed); the
  `ma_sortsetops` tier lost its isin / in1d and complex unique excuses. Findings 36–37 recorded. Commit: fa7108b9.

- **2026-10-08**: X27 done (storing a C# literal is NumPy's `PyArray_Pack` at every store entry point: the literal
  marks, the NumPy-scalar route split, np.full's unsafe copyto, np.put's order and empty-values no-op, np.pad's NumPy
  scalars; the empty-selection setter leak). Section S of the `weak_scalar` tier (15,301 store cases, zero excuses; 5
  planted bugs killed). X28, X29 and findings 34–35 recorded. Commit: 51abf1ad.

- **2026-10-07**: X24 done with X2, X21 and X25. Every comparison packs a Python scalar into the array's precision
  (NEP 50); every binary ufunc range-checks a weak Python int against its loop and re-packs one past 2^53 into a
  float loop; numpy.ma stays strong. The `weak_scalar` oracle tier (15,050 cases, zero excuses). The SIMD comparison
  and predicate kernels no longer read or write one element past their arrays (a 72,630-case guard-page sweep found
  no other kernel that does). The scalar-broadcast comparison kernels are vectorized, and the masks are stored packed. X26 and
  finding 33 recorded. Commit: `d0a6f4c8`.
- **2026-10-06**: X13 done with X19 and X20. np.power's float loops take NumPy's stride-0 fast path exactly when
  NumPy's calls see a zero exponent step: the trivial-loop / NpyIter decision is ported once and shared with np.clip.
  float16 power is HALF_power's powf; float_power never takes the fast path; a one-element operand with more
  dimensions keeps its leading axes in every binary ufunc, comparison and shift. The `power_fast` oracle tier (7,902
  cases, zero excuses; 13 planted bugs killed); `OpenBugs.UfuncOut.cs` removed (its pin is a unit test now). X21–X23
  and findings 26–32 recorded; X23 (np.kaiser's 1–2 ULP) is next. Commit `4116b608`.
- **2026-10-06**: X15 done with X4 (float16 add / subtract / multiply / divide on every broadcast and strided layout
  and through out=: the SIMD widen-compute-narrow kernels per inner run, a reversed walk for broadcast F sets; 13
  planted bugs); numpy.ma re-measured (4.80×); X18 and finding 25 recorded. Commit `1185f39c`.
- **2026-10-06**: X16 done with X11 (np.copyto is NumPy's `array_copyto`: the checks in NumPy's order, weak C#
  scalars, the overlap rule shared with NDIter.Copy and slice assignment, a new masked bit-copy kernel; the
  `copyto_full` oracle tier, zero excuses; 21 planted bugs killed); A6 re-measured (4.60×); X17 and findings 21–24
  recorded. Commit `2dffb8d9`.
- **2026-10-06**: A6 done (np.ma.mean / var / std / anom are numpy.ma's statements; the `ma_meanvar` oracle tier with
  raw data and result forms, zero excuses; `ma_reduce` holds the four strict but a finding-5 branch; 14 planted bugs
  killed); X11 widened (random masks, the dead same-dtype SIMD blend); X12–X15 and findings 16–20 recorded. Commit
  `afcc343b`.
- **2026-10-05**: A5 done (np.trapezoid raises NumPy's `IndexError: list assignment index out of range` for an axis
  outside y; 106 `err_axis_outside_y` cases in the `multioutput` tier; trapezoid out of the registry's K7 branch); finding
  15 recorded. Commit `f18bc09e`.
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
