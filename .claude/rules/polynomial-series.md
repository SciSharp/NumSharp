---
paths:
  - "src/NumSharp.Core/Polynomial/**"
  - "src/NumSharp.Core/Backends/Kernels/**/*Poly*.cs"
  - "test/NumSharp.Tests/Polynomial/**"
  - "test/NumSharp.Tests.Oracle/Fuzz/OpRegistry.Poly*.cs"
  - "test/NumSharp.Tests.Oracle/Fuzz/corpus/poly*.jsonl"
  - "benchmark/polynomial/**"
---

# The polynomial package (1/2): evaluation, the additive family and polyutils, calculus

The `numpy.polynomial` PACKAGE (not the legacy `np.polyval` family): facade `np.polynomial.{polynomial, chebyshev,
legendre, laguerre, hermite, hermite_e}`, instance-property modules (a nested static class cannot share `np.polynomial`'s
name, CS0542). Series algebra, Vandermonde matrices and roots: `polynomial-algebra.md`.

## Evaluation: `{p}val`, `{p}val2d`, `{p}val3d`, `{p}grid2d`, `{p}grid3d`, `{p}valnd`

(`polyval`/`chebval`/`legval`/`lagval`/`hermval`/`hermeval` and their twins; `polyvalfromroots` is not implemented.)
- Driver `Polynomial/Package/NDPolyEval.cs`, NumPy's Python layer: `np.array(c, ndmin=1)`, int/bool series → float64, a
  Python-scalar x stays weak, the `tensor` reshape, the `_valnd`/`_gridnd` two-pass compositions, NumPy's errors.
- Kernels `Backends/Kernels/ILKernelGenerator.Polynomial{,.Typing,.Emitter,.Lanes,.ConstPool,.Eval}.cs`: per-chunk IL over
  `NDIterRef.ForEach`, emitted by one typed emitter from per-basis STEP TABLES (NumPy's source-order expression trees,
  never re-associated, each node typed by NEP 50), so there is no per-basis or per-dtype C#.
- Bit-exact with NumPy 2.4.2 on every dtype pair and layout, the result LAYOUT included: the tier `polyeval.jsonl` (zero
  excused; section L is a "strides" facet over every x × series layout; section J a Python-sequence x (tuples, nested
  lists, empty, ragged, NumPy-scalar items, coerced by `PolySequence`) and a Python int x past int64 (a C# `BigInteger`,
  weak: CPython int arithmetic on the x-only terms, inf / OverflowError past float64); section K an array_like c and
  array_like ordinates). `MisalignedRegistry`'s generic unary/complex ULP branches are carved out for these ops.
- Every array_like parameter is `object`: c of `{p}val*`/`{p}grid*`/`{p}valnd` (NumPy's `np.array(c, ndmin=1)`, converted
  BEFORE x, so a ragged c's error wins) and the ordinates of `{p}val2d`/`{p}val3d`/`{p}valnd` (`_valnd`'s
  `np.asanyarray`: a Python int is a STRONG int64 0-d array, so `polyval2d(2, 3, float32_c)` is float64 where
  `polygrid2d(2, 3, float32_c)`, whose scalars stay weak, is float32; every ordinate converts before the shape check, and
  the shape check runs before c).
- Traps:
  - NumPy runs the first 1–2 Clenshaw steps in the COEFFICIENT dtype (`c0 = c[-2]` is a float32 scalar until it meets
    x): pre-casting a float32 series to the float64 loop changes every point. The emitter PEELS those steps to a dtype
    fixpoint (kernel class `min(nc, P+3)`), then loops.
  - A Python-scalar x is WEAK (`chebval(2.0, float32_c)` is float32), and its x-only subtrees (`2*x`, `x*0`,
    `(2*nd-1) - x`) are CPython arithmetic: BigInteger ints, and a Python int → double is CORRECTLY ROUNDED (.NET's
    `(double)BigInteger` truncates). A 0-d NDArray x is STRONG. C# `char`/`Half`/`decimal` have no Python literal and are
    strong 0-d arrays.
  - Three complex multiplies: an ARRAY op is NumPy's fused `simd_cmul`; a 0-d result is scalar math (the naive product),
    except an op touching the raw 0-d x array, which NumPy runs as a ufunc (so chebval with a 0-d complex x differs from
    both the array-x and the Python-scalar results); an N-D series at a per-point x whose result has ONE element, with
    `c[k]` and x of different ndim, gets a third form (the trivial ufunc loop refuses mixed ndims, NpyIter's one-element
    iteration gives every stride 0, so `CDOUBLE_multiply` runs its MSVC-contracted `loop_scalar`: im =
    `fma(ai, br, ar*bi)` where simd_cmul has `fma(ar, bi, ai*br)`), which is `PolyUnitBroadcast` +
    `PolyComplexProduct.LoopScalar`: values carry NumPy's ndim, and the peel runs to the joint dtype+ndim fixpoint. The
    emitter reproduces all three per op (the third form may also hold for `np.multiply` in general; unverified).
  - Vector lanes for EVERY dtype pair (`...Lanes.cs`): 256-bit on AVX2 (float64 4 lanes, float32/float16 8, complex128 2)
    and every other per-point dtype at the same lane count: int32 containers re-wrapped to int8/uint8/int16/uint16/char
    width after each op (NumPy's `2*x` wraps in int8), packed 2-lane containers, float16 as 8 float32 lanes with 2/4/8
    live, complex `[re,im]×2` with a vfmaddsub `simd_cmul` and Smith division (a shared divisor `/nd` prepared once per
    block). A lane-table miss compiles scalar chains (the same bits) and bumps `ILKernelGenerator.PolyVectorFallbacks`
    (zero over the dtype matrix, unit-tested).
  - One DynamicMethod per STAGE, or the JIT stops inlining: once a method's inline budget is spent (its local-variable
    limit / time budget; `AggressiveInlining` does not override it) it inlines nothing more, so a monolithic kernel left
    its lane helpers, even `Vector256.Load`, as CALLS. The kernel is a stride dispatcher → part (vector/scalar ×
    broadcast/per-point series) → U-chain stage → remainder stage, each its own DynamicMethod. Diagnose with
    `DOTNET_JitDisasm="NDPolyEval_*"` + `DOTNET_JitStdOutFile` and count `call`s.
  - A float16 series in a float64/complex loop gang-loads: one 16-byte load plus an exact widen serves 8/W adjacent
    chains (each takes its lanes with one `vpermpd`).
  - N-D series are never buffered (the kernel reads `c[k]` at `c0 + k*kstride`, outside the operand the iterator knows);
    the 1-D form buffers a strided x (the coefficients live in auxdata).
  - The result LAYOUT is NumPy's last ufunc output's. The oracle compares bytes in C order, so a result in the wrong order
    passes every value check; only the "strides" facet sees it. NumPy's rule: NpyIter lays out each ufunc output from its
    operands (`Shape.NpyIterOutputShape`, a port of `npyiter_find_best_axis_ordering` + `npyiter_new_temp_array`); an
    operand abstains on a stride-0 or extent-1 axis, and between disagreeing operands C order wins (unlike np.copy's
    'K' rule, `KeepOrder`, which sorts a broadcast axis innermost). chebval copies its series with order 'K'; the other
    five read it as given; an int series converts by `astype` ('K'), except in polyval, where it is `c + 0.0` (NpyIter).
    `NDPolyEval.SeriesLayout` and `ResultShape` hold this (closed forms with C fast paths, and the step table replayed
    over layouts for an N-D series at tensor=False). The section-L facet ZEROES the strides of extent-1 axes and of empty
    results (NumPy assigns those per loop path, and NumPy 2.x reports all-zero strides for any zero-size array): never
    compare them raw.
  - Benchmarks: a file-based `dotnet run` REUSES its cached build, NumSharp.Core included, when the script did not change
    (`--no-cache` after every Core edit); one warm-up pass only QUEUES tier-1 (warm, sleep for the background JIT, warm
    again); NumPy's loops are cheapest when L1/L2-resident, so a 100K-only dtype sweep hides the worst cells.

## The additive family and `np.polynomial.polyutils`

`{p}add`, `{p}sub`, `{p}trim`, `{p}line` for the six bases, the 24 module constants `{p}domain/zero/one/x`, and
`polyutils.as_series`, `trimseq`, `trimcoef`, `getdomain`, `mapparms`, `mapdomain` (`format_float` is not here). Engine:
`Polynomial/Package/NDPolyNumber.cs` (`PolyNumber`: one operand of NumPy's Python-level scalar code) + `NDPolySeries.cs`
(the functions) + the facade `Polynomial/Package/np.polynomial.polyutils.cs` (`[ModuleName("np.polynomial.polyutils")]`);
every element loop is an IL kernel in `Backends/Kernels/Direct/DirectILKernelGenerator.PolySeries.cs` (the trim scan, the
in-place combine, the tolerance scan, casts, and one-element scalarmath kernels behind flat slot arrays). Bit-exact with
NumPy 2.4.2: `polyseries.jsonl` (zero excused: long series, block boundaries, getdomain window crossings, complex64
results value-compared, tuple / nested-sequence cases (section O), `trimseq` over Python sequences (P), the deferred
object/str refusal (Q), `mapdomain` result layouts with a "strides" facet (R)), `Polynomial/PolynomialSeriesTests.cs`,
`PolynomialResultLayoutTests.cs`.
- The C# boundary is the house NEP 50 map: bool/integers/float/double/`Complex`/`BigInteger` are Python scalars,
  `Half`/`char`/`decimal` NumPy scalars, an `NDArray` (0-d too) or a typed C# array an ndarray, `object[]`/`IList` a
  Python list, a `ValueTuple` a Python tuple; that KIND decides whose arithmetic runs. NumPy never converts a domain, it
  indexes it, so there are THREE arithmetics: Python∘Python is CPython (exact ints, `long_true_divide`, 3.12's
  `_Py_c_quot`, ZeroDivisionError with CPython's three texts), NumPy scalars are scalarmath (NEP 50, the naive complex
  product, inf/nan instead of raising), and anything touching an ndarray is a ufunc (`pycomplex op np.float64` is still
  CPython: complex's methods accept a float subclass).
- `_add`/`_sub` update the LONGER operand in place (c2 on a tie; `_sub` with `len(c1) <= len(c2)` negates c2 then adds c1,
  visible in a NaN's sign) and trim. `trimseq` returns the input ITSELF or the VIEW `seq[:k]` (writes reach the input),
  and for a Python sequence the same KIND (`trimseq(object[])` is the list itself or a new list of the kept items, an
  `NDArray[]` staying one; a tuple itself or a new ValueTuple; any other enumerable itself or an `object[]`; zero-ness is
  Python's `item != 0`, an array item by its truth value with NumPy's ValueError past one element; None or a number is
  `len()`'s TypeError). `trimcoef` checks `tol < 0` before converting. `{p}line` is `np.array` DISCOVERY
  (`polyline(1, 2)` is int64). `mapdomain` converts x only when it is not an int/float/complex/`np.generic` (a bool x
  becomes a 0-d array). The constants are ONE shared, writeable, scope-detached instance each (NumPy's module attributes:
  a write persists), holding an extra ARC reference so a caller's `Dispose()` cannot free them; their corpus cases vary
  the facet (value, identity, writeable, owndata).
- Fast paths, each proven against the exact general lane:
  - generic tuple overloads `mapparms<T0..T3>((T0,T1), (T2,T3))` and `mapdomain<T0..T3>(double | Complex | NDArray x, …)`:
    a C# tuple literal binds them (identity beats boxing), and elements are read without boxing
    (`PolyNumber.FromValue<T>`). One per x kind is required: with only the `NDArray` one,
    `mapdomain(complexX, (-1, 1), (0, 2))` is CS0121-ambiguous against `mapdomain(Complex, object, object)`;
  - the CPython machine-number lane (`NDPolySeries.PyNum`): Python int (within `long`) / float / complex operands of
    tuples and lists run CPython's arithmetic on machine numbers; an int intermediate leaving `long` (checked subtract,
    `Math.BigMul`) bails to the exact `BigInteger` lane;
  - a fused scalarmath kernel (`GetPolyMapParmsKernel`) for two 1-D ndarray domains of one non-bool dtype (the
    `ABCPolyBase` case): the six operations in one call, intermediates wrapped/rounded in the dtype through locals of its
    CLR type (a `stloc` to an int8 local truncates exactly like the per-op narrowing store), float64 true division for
    integer dtypes via `EmitConvertTo`;
  - per-thread reusable 0-d parameters for `mapdomain`'s one fused `np.evaluate` pass.

  The registry replays every `mapparms`/`mapdomain` case through the object overload, the generic overload (tuples
  rebuilt element-typed by reflection) and `NDPolySeries.MapParmsGeneral` (the exact reference), requiring all three to
  agree to the byte; `MachineNumberLane_AgreesWithTheExactGeneralLane_BitForBit` adds seeded comparisons.
- Long series (from `PolyHouseKernelThreshold`, 64 elements) take SIMD routes, each byte-identical to the scalar route
  (corpus sections K long series, L complex64 array loops, M block boundaries, N getdomain across its windows):
  - `{p}add`/`{p}sub`: `CombineViaHouseKernels` materializes the in-place target block by block (1,024 elements: memcpy /
    SIMD cast / SIMD strided copy), negates it where NumPy negates it, then updates it with the same-dtype `SimdFull`
    kernel, when both operands are vector-readable (`VectorReadable`: contiguous, or strided with a result of at most 4
    bytes or a sub-word source, and a vector cast); float16 always.
  - `as_series`' conversion copy (`CopyInto`): memcpy / the contiguous cast / the SIMD strided copy; a converting strided
    source is copied, then cast through an L1 block.
  - `getdomain`: one blocked pass for every layout of a float64/float32/integer x (`TryDomainBlocked`): NumPy's
    C-contiguous copy taken 8 KB at a time (read in place from a unit-stride x, packed into L1 scratch by `CopyInto`
    otherwise), both the min and the max folded while the window is in L1 by the engine's exact schedule
    (`NumPyMinMaxReduce.FoldGroups` per full window, `Finish` for the last); bit-identical to one `simd_reduce_c` over the
    whole copy, because windows are whole 8-vector groups counted after element 0. An integer x is order-free, so a
    reversed one is folded as the forward run it covers (`PolySeriesView.Reversed`); a float x keeps its order (the
    ±0/NaN bits depend on it). uint64 packs converted to float64 (AVX2 has no unsigned 64-bit compare). float16/char
    decline to the copy-then-reduce route.
  - `mapdomain`: `MapDomainAffine` for a float64/float32/complex128 loop shared by both operations over 1-D (any stride)
    or C-contiguous points: converted 8 KB at a time into L1 scratch, then `PolyAffineDouble`/`PolyAffineSingle`
    (multiply THEN add, two roundings, never an FMA) or `PolyComplex128Affine` (`simd_cmul` + add). The fused
    `np.evaluate` pass keeps what it runs at vector speed (`FusedPassIsFaster`: a contiguous x on a vector widen edge or
    of bool; a strided x already of the loop dtype or on an edge from 8,192 points). complex64 loops run on the float32
    kernels (`MapDomainComplex64`). Hook `NDPolySeries.DisableAffineMapDomain`.
  - These rely on the library-wide cast kernels (`kernels.md`): AVX2 int → float, float16 → float64 exactly as NumPy's
    `ToDoubleBits` (a signalling NaN stays signalling, which `cvtps2pd` and `(double)Half` would quiet), the 4/8-byte
    strided copy, and `EmitConvertTo`'s packed float32 ↔ float64.
- Traps:
  - CPython's NaN operand priority is per operator, and a C# operator does not pin it: when both operands are NaN, x86
    returns the FIRST source's, and RyuJIT swaps commutative `a + b`/`a * b` for register allocation. CPython 3.12 (MSVC):
    `float_add` returns the RIGHT operand's NaN, `float_sub`/`float_div` the left's, `float_mul` the left's once the call
    site is specialized (`BINARY_OP_MULTIPLY_FLOAT`) but the right's on its first, generic execution (the steady state is
    modelled), and every operation inside `_Py_c_sum/_diff/_prod/_quot` the left's. `_Py_c_quot`'s NaN branch is
    `Py_NAN` = the POSITIVE `0x7ff8…` (.NET's `double.NaN` is `0xfff8…`). All Python arithmetic goes through
    `PyScalar.{FloatAdd,FloatSub,FloatMul,FloatDiv,ComplexSum,ComplexDiff,ComplexProd}` (explicit NaN tests).
  - A C# switch expression takes its arms' natural type: `Kind switch { Int => long, Float => double, _ => Complex }` is
    typed `Complex`, so an uncast arm boxes every real result as a Complex. Cast each arm to `object`.
  - Python's evaluation order is observable only through errors: mapparms reads each element once, in Python's order
    (old's subtraction before new is indexed), so an error in `old` wins over a short `new`.
  - `stackalloc` scratch is zero-filled unless the method carries `[SkipLocalsInit]` (`CopyInto` runs once per block).
  - A strided source reads more than its block: a stride-2 float64 block of 16 KB drags 32 KB of source lines in, which
    with the scratch overflows a 48 KB L1, so blocks are 8 KB.
  - 100K-point timings move 1.5–2× between runs with the allocator's page state: A/B two routes in one process,
    interleaved (`DisableAffineMapDomain`), and re-measure NumPy back to back.
  - Measure a cast before calling it vectorized. The scalar `cvtss2sd`/`cvtsd2ss` behind `conv.r8`/`conv.r4` merge into
    their destination register, a false dependency that makes a scalar float32 → float64 loop latency-bound.
  - NumPy's scalar-broadcast complex multiply is `simd_cmul` at every length for a trivially iterable call, even one
    element (`s * np.array([x])`); `loop_scalar` needs NpyIter's stride-0 single-element iteration.
  - `GetUInt32(i)` on a 2-D array reads the ROW coordinate `i`, not flat index `i`: flatten first.
  - C# picks an `NDArray` overload over an `object` one for ANY C# array argument (`NDArray` is the better conversion
    target, even through a user-defined conversion), so an `object[]` (a Python list) would go through NumSharp's
    implicit array conversion, which cannot interpret `object` elements. A list gets its own `object[]` overload
    (`mapdomain`, `trimseq`).
  - Sign-mixed zeros alone cannot tell NumPy's min/max schedule from a sequential fold (with every element a tie, the
    highest lane's last zero wins, which holds the array's last element too). A discriminating case shares the extreme
    between lanes out of order (a -0.0 early in the highest lane, a +0.0 late in lane 0 among smaller values), and its
    length leaves no scalar tail (a tail decides ties by itself).
  - Repeating extremes hide a dropped element: plant a UNIQUE min and max at every position around each window edge
    (`GetDomain_BlockedPass_FoldsEveryElement`).
  - An A/B that alternates routes batch by batch lets one route's allocations evict the other's working set: time each
    route as a whole warm batch and confirm against a standalone run.
  - `mapdomain` over an N-D non-C x: transpose, compute C, relabel; never write NumPy's layout directly. NumSharp's NDIter
    walks a pair of operands sharing a PERMUTED layout in logical order, gathering from both (it reverses axes only for
    all-F operands). `MapDomainWith` runs every route over `np.transpose(x, layout.DenseAxisOrder())` (C-contiguous for a
    dense permuted x: the affine SIMD route), then `NDArray.AdoptLayout` (`Backends/NDArray.AdoptLayout.cs`, shared with
    np.clip and np.trapezoid) relabels the C result's storage with NumPy's layout (`UnmanagedStorage.SetShapeUnsafe`: no
    element moves, OWNDATA kept; it checks the contract, a fresh C-contiguous offset-0 result, and copies otherwise).
  - RyuJIT rewrites `(-a) + b` into `b - a`, which changes a NaN's sign: the NegateAdd combine kernel stores the negated
    value and reloads it before the add (`EmitPolyCombine`; `Sub_EqualLengths_ConvertedFloat16Subtrahend_StillNegatesFirst`).
    The oracle tokenizes NaN, so only a byte-level unit test sees it.
  - Two NaNs meeting in `_add` / `_sub` keep NumPy's operand's NaN: NumPy's update is an in-place ufunc over two arrays,
    and its win-amd64 add keeps the SECOND operand's NaN (the other series, or c1 after `_sub`'s negation), subtract the
    first. `EmitPolyCombine` emits the add through `EmitScalarOperation(…, NaNOperandOrder.NumPyUfunc)`
    (`AddSub_NaNPair_KeepsNumPysOperand`); the series algebra runs `_add`/`_sub` in every step.

## Calculus: `{p}der` and `{p}int`

For the six bases, with every parameter (`m`, `k`, `lbnd`, `scl`, `axis`). Driver `Polynomial/Package/NDPolyCalc.cs`:
NumPy's Python prologue in NumPy's statement order, which is also its error order. Kernel
`Backends/Kernels/Direct/DirectILKernelGenerator.PolyCalculus.cs`: the six recurrences are DATA (`PolyCalcRoutines`:
head steps, a j loop, tail steps; each statement is NumPy's source expression token for token), and ONE whole-array
kernel is emitted per (basis, der/int, dtype, source dtype, 1-D, fused scale). Bit-exact with NumPy 2.4.2:
`polycalc.jsonl` (zero excused; argument kinds (M), zero-size / 5-D / extreme-int series (N), widened scale (O)),
`Polynomial/PolynomialCalculusTests.cs` (NumPy's `TestIntegral`/`TestDerivative` ported, bitwise where NumPy's statement
sequence makes them exact; byte dumps per dtype family; kernel structure), `Polynomial/PolynomialArgumentKindsTests.cs`.
- How it is built:
  - One buffer, in place, for any order: the recurrences never need a value after they overwrite it, provided the result
    is written one row over (a derivative writes `der[q]` to row q+1, the result being rows [m, n); an integral writes
    `tmp[q]` to row q with `c[q]` at row q+1, so m spare rows sit above the loaded series).
  - Columns are independent (a position in `c.shape[1:]` never meets another), so the kernel takes them in
    cache-resident blocks (~512 KB) through EVERY order of a derivative, with the evaluation family's lane kinds over the
    columns, then a scalar tail.
  - The load stage fuses the conversion and the scale: int/bool series → float64 and `c *= scl` in the same first pass;
    it reads the caller's layout in place when every row's columns sit at one stride (every 1-D and 2-D series,
    C-mergeable N-D), otherwise NDIter copies first.
  - NumPy's three arithmetics, per statement: `c *= scl` is always an array op (`simd_cmul`, operands (c, scl)); a 1-D
    series' recurrence statements are SCALARMATH (the naive complex product, the `ScalarMath` key); an N-D series' are
    ufuncs.
  - The integral's correction `tmp[0] += k[i] - {p}val(lbnd, tmp)` runs through the evaluation family's `NDPolyEval.Val`
    plus `PolyNumber`: scalarmath and setitem's cast for a 1-D series, an in-place ufunc for an N-D one.
  - The weak Python ints of the recurrence (`2*j`, `j+1`, …) are converted per (block, j) through a double and then the
    cast, so float16 constants past 2048 round to even and past 65504 become inf, as in NumPy.
- Result objects are NumPy's (the corpus records C/F/OWNDATA of each result as a `"facet": "flags"` case): the growing
  orders are `np.moveaxis` views of fresh C-order buffers (OWNDATA false); `m == 0` is the K-order copy (OWNDATA true; an
  F series stays F); der with `m >= n` is `c[:1]*0`, laid out by NpyIter's KEEPORDER vote over `c[:1]`
  (`Shape.MultiSortedStridePerm`), then moved back, except by `hermeder`, which skips the moveaxis (a NumPy quirk); an
  integral whose every order took the n == 1 zero branch is a view of the moved copy.
- Errors in NumPy's order and texts: a complex scl into a real series is caught at the first `c *= scl`, before an empty
  series' IndexError; an array constant for a 1-D series is setitem's ValueError into a REAL series but `complex()`'s
  TypeError (`only 0-dimensional arrays can be converted to Python scalars`) into a COMPLEX one; for an N-D series, a
  complex correction is the in-place add's UFuncTypeError, which names complex64 when the series is float16/float32 and
  the lbnd or k is a Python complex (NumPy's complex64 loop).
- The 1-D complex64 value: a 1-D float16/float32 series with a Python-complex lbnd makes NumPy compute `{p}val` in
  complex64 SCALARMATH; the evaluation kernel works in complex128, so `NDPolyCalc.PyVal1D` interprets the weak-x step
  trees (`PolySteps.RewriteForWeakX`) with `PolyNumber`, which emulates complex64 scalars exactly.
- Widened scale: a strong scalar that PROMOTES the series (float64 against float32, float32/float64 against float16)
  makes NumPy's `c *= scl` run the WIDER multiply loop and cast back. `PolyCalcKey` carries the loop dtype (`ScaleLoop`,
  `HasWidenedScale`), and the load stage calls `PolyLaneOps.{F32ScaleF64,HalfScaleF64,HalfScaleF32}` (vector) and their
  `*Scalar` twins (tails, 1-D series). The float64 → float16 narrow must round ONCE (`npy_double_to_half`): the vector
  path rounds the float64 product to float32 with ROUND-TO-ODD (`RoundToOddF32`: truncate + a sticky bit) and then
  narrows to float16 to nearest even, exact because 24 ≥ 11 + 2 bits (a plain f64 → f32 → f16 chain double-rounds at
  float16 ties). A NaN coefficient keeps its payload, quieted. An ARRAY scl (derivatives only) and a promotion the kernel
  has no widened loop for (decimal) run `ScaleInPlace` = `np.multiply(c, scl, out: c, dtype: promote(series, scl))` per
  order; the explicit `dtype:` is load-bearing, because the ufunc computes a float32 array times a 0-d float64 with a
  float32 `out` in float32. A complex scale into a real series raises NumPy's UFuncTypeError there.
- Argument kinds: every array_like of the package (`{p}der`/`{p}int`'s c, k, lbnd and scl, `{p}val*`/`{p}grid*`'s c and
  x, the ordinates, `mapdomain`'s x, `as_series`' items, `{p}add`/`{p}sub`'s operands, `trimseq`'s items) goes through
  `Polynomial/Package/NDPolySequence.cs` (`PolySequence`), a port of `array_coercion.c`
  (`PyArray_DiscoverDTypeAndShape_Recursive` + `update_shape`) and its fill:
  - The C# map (header of `NDPolyNumber.cs`): `ITuple` (ValueTuple, System.Tuple) is a Python tuple; `object[]` and every
    other C# array whose elements are not a NumSharp dtype (jagged `double[][]`, `NDArray[]`, `BigInteger[]`) and any
    other `IList`/`IEnumerable` are lists, a multi-dimensional `object[,]` a list of rows; a typed dtype array (`double[]`,
    `float[,]`) and a `Memory<T>` of a dtype are NDARRAYS (strong dtype); `bool`/integer primitives/`BigInteger`/`float`/
    `double`/`Complex` are Python scalars, `Half`/`char`/`decimal` NumPy scalars, a string a str. A list of ONE scalar kind
    converts in one pass at any depth: a typed collection (`List<double>`, `HashSet<int>`, LINQ, …; the discovered dtype
    per its C# element) or a flat `object[]` of Python floats / ints (`TryTypedCollection` / `TryFlatLeaf`) becomes one
    array leaf with the walk's two shape updates.
  - Shape: the first leaf reached fixes the dims; a later leaf at another depth, or a sequence of another length, is
    ragged: NumPy's `setting an array element with a sequence. The requested array has an inhomogeneous shape after N
    dimensions. The detected shape was (…) + inhomogeneous part.` (ValueError). An EMPTY sequence ends the dims without a
    dtype vote (`[]` is float64 `(0,)`, `[np.zeros(0, int8), []]` int8 `(2, 0)`); an array leaf that cannot be assigned
    raises the broadcast ValueError (`could not broadcast input array from shape (0,3) into shape (0,)`).
  - Dtype: each leaf's DEFAULT descriptor (a Python int → int64, uint64 up to 2^64−1; a NumPy scalar / array keeps its
    own), promoted STRONGLY (array coercion is not NEP 50: `[np.float32(1), 1.0]` is float64). A str / None leaf, a Python
    int past uint64 and any unknown object are refused (`NotSupportedException`) AFTER the walk, so a ragged input still
    reports NumPy's ValueError; that holds where NumPy computes with the array at once (evaluation, calculus).
    `polyutils.as_series` (the additive family, the whole series algebra) first checks every argument's size and dims and
    then its common type, so there `ToArrayOrNonNumeric` hands back NumPy's str / object array as a SHAPE plus a deferred
    refusal (`PolySeriesView.NonNumericArray`).
  - Calculus-specific order: `np.ndim` runs on lbnd and scl before anything is converted (a ragged sequence raises the
    inhomogeneous text there; a str is 0-d); k's items are converted only when their order uses them, AFTER that order's
    `{p}val(lbnd, tmp)` (Python's left-to-right `k[i] - {p}val(…)`), so a bad lbnd wins over a bad later constant; a str
    lbnd is refused only where NumPy evaluates at it (never for m == 0, the one-coefficient zero branch, or before a later
    argument's own error); a `BigInteger` past float64 raises CPython's `int too large to convert to float`
    (`OverflowException`), one within float64 but past float32/float16 is inf.
  - A conversion is the call's intermediate, released however the call ends: `{p}val` disposes an x it converted,
    `_valnd`/`_gridnd` dispose a partial result when a later pass fails, `mapdomain(object x)` its converted points; the
    additive and calculus facades are `[NDScoped]`, the evaluation facades dispose by hand. A pool-balance probe
    (`SizeBucketedBufferPool` takes − returns around a call whose result is disposed) must build its operands OUTSIDE the
    measured lambda, or it counts its own arrays.
- Traps:
  - float16 N-D rounds every op back onto the f16 grid (narrow + widen), so it is far slower than float32; a float32
    working buffer narrowed once at the end would remove that.
  - The oracle's flags facet REPLACES the result, so the registry (`OpRegistry.PolySeries.cs` `CalcFacet`) must dispose
    the result it drops (the leak gate `UndisposedIntermediateTests` reads one escaped buffer per flags case otherwise).
  - A quoted `--filter` passed from Git Bash through `cmd //c` reaches dotnet with LITERAL quotes (`\"X\"`): a
    `TestCategory!=…` filter then matches everything and a `ClassName~…` filter nothing. Read the filter from a file in the
    `.cmd` script (`set /p FILTER=<filter.txt`).
  - `describe()` serializes an operand's BASE in C order, so a layout is a C-contiguous base plus a view: an F series is
    the `.T` of a C buffer holding `base.T`.
  - Random scales never hit a float16 double-rounding hazard (a product must land within ~2^-24 relative of a float16 tie);
    build them: `_pc_f16_hazards` scans every finite float16 against scales like `1 + 2^-11 ± 2^-40`.
  - Two C# traps when building nested Python sequences: a lone `object[]` passed to a `params object[]` parameter IS the
    params array, not its one item (wrap it: `new object[] { inner }`); and a `?:` or switch expression whose arms are
    `object[]` and `NDArray` is typed `NDArray` through the implicit `Array → NDArray` conversion. Cast every arm to
    `object`.
  - `start /b /wait X.cmd` from Git Bash runs X under `cmd /K`, which can wait on console input forever after X
    finishes: wrap it as `start /b /wait /affinity … cmd /c X.cmd`.
