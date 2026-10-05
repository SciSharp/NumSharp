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
  prologue). Done 2026-10-05.
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
  2026-10-05.
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
- [ ] **A3 `np.nanstd` / `np.nanvar`: `axis=` and `keepdims=` give the wrong shape.** A 1-D float32/float64 array
  along axis 0 gives `(1,)` (NumPy 0-d); one element with `keepdims` drops the axis. Pin:
  `NanStdVar_1DAxis_ReturnNumPysShape`.
- [ ] **A4 max/min family: `out=`, `initial=`, `where=` and NumPy's order** for `np.max`, `np.min`, `np.amax`,
  `np.amin`, `np.nanmax`, `np.nanmin`, `ndarray.max`, `ndarray.min`. NumPy: `(a, axis=None, out=None, keepdims,
  initial, where)`; NumSharp: `(a, axis, keepdims, dtype)`. Pins: the 4 `MaxMinFamily_*` tests.
  - [ ] A4a NumPy's parameter order + `out=`
  - [ ] A4b `initial=`
  - [ ] A4c `where=`
- [ ] **A5 `np.trapezoid`: an `axis=` outside `y`.** NumPy raises `IndexError: list assignment index out of range`;
  NumSharp raises `ArgumentOutOfRangeException`. Pin: `Trapezoid_AxisOutsideY_RaisesNumPysIndexError`.
- [ ] **A6 `np.ma.var` / `np.ma.std` with a mask: the masked path is not NumPy's.** `NDMaskedArray.var` computes its
  own formula (it never reaches `np.var`): 242 cases of the ma oracle tier differ — a slice NumPy masks (all masked,
  or `count - ddof <= 0`) carries NaN instead of the masked 0, and some slices are finite where NumPy has +inf. They
  were hidden by the registry's unbounded "summation precision" blanket until A2 removed var/std from it (finding 7);
  `MisalignedRegistry.IsMaVarStd` keeps the blanket for the ma ids only and names this item.

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
  dtype=np.float32)` does not compile; the axis forms do).
- [ ] B4 `np.std`, `np.var`, `ndarray.std`, `ndarray.var`: `out=`, `where=`, `mean=`, `correction=`; NumPy's order
  `(a, axis, dtype, out, ddof, keepdims)`.
- [ ] B5 `np.nansum`, `np.nanprod`: `dtype=`, `out=`, `initial=`, `where=`.
- [ ] B6 `np.nanmean`: `dtype=`, `out=`, `where=`.
- [ ] B7 `np.nanstd`, `np.nanvar`: `dtype=`, `out=`, `where=`, `mean=`, `correction=`; NumPy's order.
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
- [ ] D7 `np.ndarray(shape, dtype, buffer, offset, strides, order)`: `offset=`, `strides=`. Unblocked: the view-`Shape`
  plan is done (2026-10-05). It adds three follow-ups of its bug 1 (`docs/plans/view-shape-open-bugs.md`, "Not in this
  bug"):
  - B1.10: a buffer that is too small must raise `TypeError: buffer is too small for requested array` (NumSharp:
    `ArgumentException: values.Length does not match shape.Size`).
  - B1.11: a buffer that is larger than the shape must give an array over its prefix (NumSharp raises the same
    `ArgumentException`).
  - NumSharp's `np.ndarray(shape, dtype, buffer, order)` defaults to `order='F'` and float32 (NumPy: `order=None`, which
    is C, and float64), and ignores `order` without a buffer (NumPy: `np.ndarray((2,3), order='F')` is F-contiguous).

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
  port subtracts a keepdims mean from an F-ordered input). Done 2026-10-05.
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
  by A2's large-size check (`np.var(x, axis=0, dtype=np.float32)` of a 1-D float64 array). Done 2026-10-05.
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
- [ ] **X8 The var/std kernels A2 no longer reaches.** `DirectILKernelGenerator.VarSimdHelper` / `StdSimdHelper`
  (`Masking.VarStd.cs`), the `ReductionOp.Var` / `ReductionOp.Std` axis kernels (`Reduction.Axis.VarStd.cs`) and
  `NDAxisIter`'s `VarAxisDoubleKernel` / `StdAxisDoubleKernel` computed var/std in their own order; nothing calls them
  since the port. Decide: remove them, or keep them for a future non-NumPy fast mode.

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
   sum; the deviation sum runs over a fresh array). Belongs with B1.
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

## 7. State log

- **2026-10-05**: A2 done (np.var / np.std are NumPy's `_var` / `_std`; the `varstd` oracle tier; the chunked
  deviation sums); A6, X6, X7, X8 and findings 7–8 recorded. Rebased onto `5c9d7925` (the view-`Shape` plan's last fix:
  the new code adopts the shapes it builds with `Shape.Adopt`); D7 unblocked and given that plan's `np.ndarray`
  follow-ups. Commit: the A2 commit on `journey4`.
- **2026-10-05**: X5 done (`np.sum` along an axis that leaves one output, in a cast loop), found by A2's large-size
  check; B3 widened (`np.mean` has no flat `dtype=` overload). Commit: the X5 commit on `journey4`.
- **2026-10-05**: X3 done (binary ufuncs with an F-ordered broadcast operand), found while proving A2; X4 and
  findings 4–6 recorded; A2 in progress (proof and design proof above). Commit `d3c863ef`.
- **2026-10-05**: A1 done (np.all / np.any reduce prologue, the `reduce_kwargs` tier S2, the null-axis priority);
  G4 added from its finding. Commit `bb5b2fcf`.
- **2026-10-05**: plan written from the inventory at `2821a708` (66 `np.*` + 14 `ndarray` names with a gap) and the
  `[OpenBugs]` pins of 2026-10-03/04.
