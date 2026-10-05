# API parameter parity: every NumPy parameter, present and working

Living plan and state document. Branch `journey4`. Started 2026-10-05.

**Task (verbatim):** "Take all the missing or non functioning paramters or api mismatch like out or where or others.
write them in a plan with progress marking and set on a task to fix one by one. Proving parity and that it is a bug of
parity and only then fix."

**State:** see [§7 State log](#7-state-log) (newest entry first) and the checklists in [§5](#5-items).

**Out of scope here:** the seven view-`Shape` bugs (`OpenBugs.ViewShape.cs`: the wrap constructors and the shared
dimension array). A parallel session owns them in its own plan. Item D7 below touches the same constructors and waits
for that work.

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
- [~] **A2 `np.var` / `np.std`: `ddof=` and `dtype=` do not work** — at a degenerate size (one element per output,
  an axis of length 1, `ddof >= n`, empty float16/float32 input) and, as the proof below found, at every size for
  `dtype=`. Pins: the 7 `VarStd_*` tests in `OpenBugs.Reductions.cs`; oracle excuse `MisalignedRegistry` V1.
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
- [ ] B3 `np.mean`, `ndarray.mean`: `out=`, `where=`.
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
- [ ] D7 `np.ndarray(shape, dtype, buffer, offset, strides, order)`: `offset=`, `strides=`. Waits for the view-`Shape`
  session (the same constructors).

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
- [ ] X4 float16 `add` / `subtract` / `multiply` / `divide` with a broadcast, non-contiguous operand set run the
  direct route's scalar General loop: `(30,40,20) F * (30,1,20) F` takes 328 µs (NumPy 72 µs, 0.22×), and
  `(100,100,100)` 27 ms (NumPy 3.0 ms). The same before X3 (HEAD measured 320 µs while computing wrong values).
  Speed only; the values are NumPy's.

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

## 7. State log

- **2026-10-05**: X3 done (binary ufuncs with an F-ordered broadcast operand), found while proving A2; X4 and
  findings 4–6 recorded; A2 in progress (proof and design proof above). Commit: the X3 commit on `journey4` (see
  `git log -- docs/plans/api-parameter-parity.md`).
- **2026-10-05**: A1 done (np.all / np.any reduce prologue, the `reduce_kwargs` tier S2, the null-axis priority);
  G4 added from its finding. Commit `bb5b2fcf`.
- **2026-10-05**: plan written from the inventory at `2821a708` (66 `np.*` + 14 `ndarray` names with a gap) and the
  `[OpenBugs]` pins of 2026-10-03/04.
