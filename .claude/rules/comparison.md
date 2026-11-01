---
paths:
  - "src/NumSharp.Core/Logic/**"
  - "src/NumSharp.Core/Backends/Kernels/ComparisonRun.cs"
  - "src/NumSharp.Core/Backends/Kernels/Direct/*{Comparison,Predicate,TransposeTile,StridedMove}*.cs"
  - "src/NumSharp.Core/Backends/Default/Math/DefaultEngine.CompareOp*.cs"
  - "src/NumSharp.Core/Backends/Default/Logic/**"
  - "src/NumSharp.Core/Backends/Iterators/NDIter.Execution.ComparisonRun.cs"
  - "src/NumSharp.Core/Manipulation/np.isin.cs"
  - "test/NumSharp.Tests/Logic/**"
  - "test/NumSharp.Tests/Backends/Kernels/{Comparison,TransposeTile}*.cs"
---

# Comparison and logic

## Predicates and membership

- `np.signbit(x, out=None, where=True, dtype=None)` is the IEEE sign-bit predicate (a full ufunc; `isposinf`/`isneginf`
  are `isinf(x) & ~signbit(x)` / `isinf(x) & signbit(x)`); the result is always bool. It is not `x < 0`: on floats
  `-0.0` → True and a NEGATIVE NaN → True, `+0.0`/`+inf`/a positive NaN → False. Per dtype: Half/Single/Double test the
  sign bit; signed integers use `x < 0`; unsigned integers, bool and char are always False; Decimal is strictly negative
  (`Math.Sign < 0`, so `-0.0m` → False); Complex has no loop (NumPy's `ufunc 'signbit' not supported for the input
  types…` TypeError). It rides the fused predicate kernel of `isnan`/`isinf`: Single/Double and Int32/Int64 vectorize
  through a per-lane `ExtractMostSignificantBits` plus a PDEP bool store; narrow, unsigned, Half and Decimal take the
  scalar route. `dtype=` is validate-only (bool loops only). Gates: `Logic/np.signbit.Test.cs`, the `logic` tier.
- `np.isin(element, test_elements, assume_unique=false, invert=false, kind=null)` runs NumPy's `_isin`
  (`numpy/lib/_arraysetops_impl.py`) after the facade's class boundary, as `TensorEngine.Isin`
  (`Backends/Default/Manipulation/Default.Isin.cs`; `Default.Isin.VString.cs` holds the `StringDType` route). NumPy
  answers on one of three paths, and NumSharp takes the same one (`NumPyPath`):
  - the table method (an integer or bool pair, a char array being NumPy's uint16; `kind` None or `'table'`; the set's
    range safe for its dtype and at most `6·(len(ar1)+len(ar2))`, or `kind='table'`) runs as one pass
    (`TableMembership`): elements of one byte, and of two bytes from 8,192 on, read a table over their dtype's whole
    domain (256 or 65,536 flags indexed by the element's bits); wider elements read NumPy's bool table over the set's
    range (inverted under `invert`), each element as its 64 bits, one `(ulong)(v - min) <= range` test and one load,
    exact modulo 2**64 (a uint64 operand beside a signed one is declined);
  - the loop path (`len(ar2) < 10·len(ar1)**0.145`) runs as one pass (`LoopProbe`, hosts with 256-bit vectors): four
    element vectors per step against every broadcast set value, at most 9 values per vector lane; float16 / complex /
    decimal and the int64 / uint64 pair decline;
  - the sort path runs NumPy's statements (`SortPathFlat`: unique, one concatenate, a stable argsort, the
    adjacent-equality flags) where they are not the exact membership: a 64-bit integer past 2**53 concatenated in
    float64 / complex128 (two values merge and the first is flagged: `np.isin(uint64 [2**64-1, 2**64-2, 3],
    np.arange(24.))` is `[False, True, True]`), and `assume_unique` (an element is flagged when an equal value follows
    it or lies in the set).

  Every other case, and each pair a kernel declines, is the exact membership in `result_type(ar1, ar2)`
  (`ComputeMembership`: a counting table for a narrow integer range, else an open-addressing hash with a splitmix64
  finalizer for the integer and float32 / float64 dtypes, else sort + `searchsorted`; uint64 beside a signed integer
  through the uint64 domain with a sign filter), which is NumPy's answer there. NaN is never a member; -0.0 equals 0.0.
  A non-contiguous `element` is read in logical C order, and the answer is a view of a fresh 1-D array reshaped to
  `element`'s dimensions, as NumPy's `reshape` returns it. Text runs NumPy's loop and sort statements over the text
  kernels (`strings-compute.md`; a number beside text compares unequal on the loop path but EQUAL on the sort path,
  NumPy's own disagreement). Errors in NumPy's order and texts: `Invalid kind: '…'. Please use None, 'sort' or 'table'.`
  (ValueError, case-sensitive), `kind='table'` on a non-integer array (ValueError), a range past the set dtype's maximum
  (NumPy's RuntimeError text, as `InvalidOperationException`), a range of `2**63 - 1` or more (`ValueError: Maximum
  allowed dimension exceeded`, NumPy's table allocation; an empty element too). One deliberate divergence
  (`[Misaligned]`): a `kind='table'` range NumPy can address but not allocate (its 8 EiB `MemoryError`) is answered, the
  membership needing no table. Floor: `assume_unique` over presorted floats (NumPy's timsort finds the two sorted runs).
  An AVX2 gather of the range table is faster but not taken: Intel's GDS microcode mitigation slows gathers on 6th–11th
  generation cores. Gates: `Manipulation/np.isin.Test.cs`, `np.isin.NumPyPort.Test.cs`, the `groupa` tier.
- `np.iterable(y)` is NumPy's `try: iter(y); return True; except TypeError: return False`, an O(1) type check
  (`NDArray.iterable`, `Manipulation/NDArray.FunctionBase.cs`; the facade `Logic/np.is.cs`): `null` → false; an `NDArray`
  → `ndim != 0` (a 0-d array is the only non-iterable array, as `NDArray.GetEnumerator()` throws for it); `string` →
  true; any `IEnumerable` → true; scalar value types → false. The NDArray test precedes the `IEnumerable` test (NDArray
  implements it). `[Misaligned]`: a bare `IEnumerator` cursor (one that is not itself `IEnumerable`; LINQ's iterators
  are) and `ValueTuple`/`Tuple`/`ITuple` are iterable in Python but not `foreach`-able in C#, so they return false.
  Unit-tested (`Logic/np.iterable.Test.cs`), like `isscalar`, and both run over every C# value kind in the facade-move
  snapshot tool's object operand.
- `np.isscalar(obj)` (`NDArray.isscalar`, `Backends/NDArray.Numeric.cs`) is True for a C# primitive, a `decimal`, a
  `Complex`, a `bool`, a `string` and a 0-d one-element NDArray (NumSharp has no scalar objects: a 0-d array is what a
  NumPy scalar becomes, though NumPy's own 0-d array answers False). Pinned divergences (NL11,
  `OpenBugs.NpLayer.cs`): a `Half` (NumPy's float16 scalar) and a `BigInteger` (a Python int) answer False, and a `Type`
  object of a primitive (`typeof(int)`) True, where NumPy's `isscalar(int)` is False.

## The comparison ufuncs

`np.equal` … `np.greater_equal` and `isnan`/`isfinite`/`isinf` expose one NumPy-shaped overload each,
`f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)`, returning `NDArray` (NumPy's
`np.less(a, b, out=f64)` returns the f64 out itself; True → 1 at any numeric out dtype). A plain call returns an
`NDArray<bool>` instance, so the C# operators (`==`, `<`, …) keep the `NDArray<bool>` static type through
`AsGeneric<bool>()`. `dtype=` is validate-only: bool loops only (`dtype: Boolean` is a no-op; anything else raises `No
loop matching the specified signature and casting was found for ufunc <name>`). Comparisons compare at
`result_type(lhs, rhs)` inside the kernel (`greater(i8 2^53+1, f8 2^53)` → False, `equal` → True). Engine members follow
the order `(inputs, typeCode, out, where)`.

A Python scalar is WEAK in every comparison (NEP 50). `ExecuteComparison` (`DefaultEngine.CompareOp.WeakScalar.cs`)
converts a 0-d Python-scalar operand (a C# bool / integer / float / double / `Complex`; `Half` / `char` / `decimal` are
strong NumPy scalars) against an array into the array's precision before either route runs: `PyArray_Pack`, i.e. the
correctly rounded double, then the float16 / float32 narrow (an int past 2^53 rounds twice, overflow is inf), complex
parts to float32 (complex64). So `np.greater(f32 [0.1, 0.5], 0.1)` is `[False, True]` and `== 0.1` `[True, False]`, as
NumPy's. A 0-d against a 0-d stays strong; a 0-d NDArray against an array is read as the literal (`[Misaligned]`,
np.clip's rule). An integer array against an in-range Python int of another width is re-packed into the array's dtype
(NumPy's `special_integer_comparisons` in-range branch: values unchanged, the same-dtype kernel runs). numpy.ma stays
STRONG (`PromoteMaScalar(strongOperand)`; its domains see `getdata` data). Otherwise the comparison type is the promotion
of the two dtypes (`ComparisonKernelKey.GetComparisonType` → `np._FindCommonScalarType`; int64 against uint64 compares
exactly). The binary ufuncs' weak Python int is checked against the loop dtype the same way (`ufuncs.md`). Gates: the
`weak_scalar` tier (zero excuses), `Logic/np.comparison.WeakScalar.Test.cs`, `Ma/np.ma.StrongScalar.Test.cs`.

The logical ufuncs `np.logical_and` / `np.logical_or` / `np.logical_xor` / `np.logical_not` are `TensorEngine.LogicalAnd`
/ `LogicalOr` / `LogicalXor` / `LogicalNot` (`Backends/Default/Logic/Default.Logical.cs`); the facades keep the
numpy.matrix and numpy.char.chararray boundaries. NumPy's loops are bool-output only (a non-bool `dtype=` has no loop,
`dtype=bool` is a no-op); a text operand reads its truth (non-empty is True) through NumPy's bool cast, and logical_not has
no string loop (a fixed-width operand under `dtype=bool` fails the input cast, under another `dtype=` finds no loop). The
composition reduces a numeric operand to `x != 0` (logical_not: `x == 0`) and combines through the bitwise ufunc, which
combines a bool operand's TRUTH (bytes 2 and 1 AND to 1). One-pass routes serve the common call (`TryLogicalRun` /
`TryLogicalNotRun`): contiguous operands of one shape (both C or both F) or one of one value, no `where=`, no out or a
bool out of that layout sharing no memory; per 4,096-element block the comparison engine's run writes the truth values
on the stack and one `Vector<byte>` operation combines them: the composition's bytes, layout and result type by
construction (gate `Logic/np.logical.OnePassRoute.Test.cs`, hooks `DefaultEngine.DisableLogicalRun` / `LogicalRuns`).
Floors: the bool-bool cells and logical_not of bool (the composition's one SIMD pass) and memory-bound 10M float64 cells.

## The comparison engine

Every operand layout and dtype pair runs the vector kernels, for the operators and the `np.*` forms, with and without
`out=`.
- What NumPy does (`loops_comparison.dispatch.c.src`, `PyUFunc_SimpleBinaryComparisonTypeResolver`): its loops are
  vectorized for three operand forms only (two contiguous inputs, or one input a single value, each with a contiguous
  bool output). A stepped or reversed input, an input of another dtype (cast through a buffer first) and a stepped output
  run its scalar loop; so do complex128 and float16. int64 against uint64 has exact loops.
- `ComparisonRun` (`Backends/Kernels/ComparisonRun.cs`), one cached instance per (operator, left dtype, right dtype),
  takes one inner-loop chunk (three pointers, three byte strides, a count) and runs it through the three contiguous
  vector kernels (`SimdFull`, `SimdScalarRight`, `SimdScalarLeft`) whatever the strides and dtypes:
  - direct: an input that already is a contiguous run of the kernel's dtype, or one value, is read in place; a long run
    that starts off a vector boundary gets a short first kernel call (`AlignmentPeel`: the input stream for lanes of two
    bytes or more, the output for one-byte lanes; from 4 KB);
  - staged: any other input is copied and converted 8 KB at a time into a stack block the kernel reads while it is in
    L1 (the SIMD copy removes a step of two or a reversal, the unrolled strided move
    `Direct/DirectILKernelGenerator.StridedMove.cs` any other stride, the SIMD cast converts);
  - flipped: inputs that run backwards in the kernel's dtype are read forwards from their last element and the bool
    results stored reversed;
  - scattered: a bool output that is not contiguous receives each block through a strided byte store;
  - adopted: an array of another dtype against ONE value is compared in the array's own dtype (below);
  - a chunk under 64 elements runs the per-element kernel.

  Two more forms belong to a whole iteration of rows (`RunRows`): tiled (an operand whose successive rows are adjacent
  elements, such as an F-ordered matrix beside a C-ordered one, or a matrix against its transpose) and repeated (a
  matrix against one short row, several rows per kernel call).
- Three callers:
  - `TryTrivialContiguousComparisonOp` → `Run`: two contiguous arrays of one shape, or an array and one value, of any two
    dtypes; also two one-element operands of different rank.
  - `TryRowsComparisonOp` (`DefaultEngine.CompareOp.Rows.cs`) → `RunRows`: an iteration of R runs of N elements, without
    an iterator, read from the strides with NumPy's coalescing rule (`stride[d] == stride[d+1] * extent[d+1]`): a matrix
    against a row or a column, an outer product, a stepped or reversed view. It declines to the iterator route for a
    result NumPy lays out in F order, an iteration of three or more levels, and many rows under 16 elements whose inputs
    are contiguous or one value (the iterator's 2-D block kernel loops such rows inside one call), unless they are a
    matrix against one row (`ComparisonRun.RepeatsRow`).
  - `NDIterRef.TryExecuteComparisonRun` → one `Chunk` call per inner-loop chunk: every other iteration, and every `out=`
    call (`ExecuteComparisonUfuncInto`; an out of another dtype is filled through the iterator's Boolean buffer). The
    `out=` route then tries the whole-array kernel (`NDIterRef.TryExecuteComparison`) when the inputs share a dtype and
    the out needs no cast. `where=` keeps its masked route.
- Scalar adoption is exact, never a cast (`ComparisonRun.ScalarAdoption`). The weak-scalar step above runs BEFORE it, so
  an operand NEP 50 treats as weak arrives in the array's dtype already, and adoption serves the strong 0-d operands
  (`char`, `Half`, `decimal`) and the out-of-range Python ints (`ComparisonRunTests` holds its element reference to that
  rule: `Expected` / `PackWeak` for a Python scalar against a float16 / float32 array, `RaisesOverflow` for a bool array
  against a Python int at or past 2^63). NumPy's comparison is
  `convert(a) op convert(v)` in the promoted dtype, which has the same answer as `a op a'` for one value `a'` of the
  array's dtype, or the same answer for every element:
  - integer array, integer value: `a' = v` inside the array dtype's range; outside it, the chunk is decided;
  - integer array of at most 32 bits, float value `t`: `a > t` is `a > floor(t)`, `a >= t` is `a >= ceil(t)`, `a < t` is
    `a < ceil(t)`, `a <= t` is `a <= floor(t)`; `==` can hold only for an integral `t` in range;
  - float16 / float32 array, wider value `t`: `>` and `<=` use the largest array-dtype value not above `t`, `>=` and `<`
    the smallest not below it (one step on the stored bit pattern from the rounded value); `==` can hold only when `t`
    is an array-dtype value;
  - a NaN value decides the chunk (False; True for `!=`).

  Not adopted (they keep the staged conversion): a 64-bit integer array against a float, complex, decimal or char value.
- Tiles (`ComparisonRun.RunTiles`): an F-ordered matrix read in C order has rows whose elements are one COLUMN apart,
  so a row touches N cache lines (N pages once a column is 4 KB long). The elements of one column ARE adjacent, so a tile
  of B rows reads B contiguous elements of each column and deals them out to B rows of a stack block.
  - The kernel (`Direct/DirectILKernelGenerator.TransposeTile.cs`, C# with AVX2 intrinsics): groups of 2 × 4 elements of
    8 bytes, 4 × 8 of 4 bytes, 8 × 16 of 2 bytes, 16 × 32 of one byte. Each 256-bit register is built from two columns
    (the second through `vinserti128` with a memory operand, which uses no shuffle port), so the in-lane transposition
    gives rows of twice the columns and needs no lane-crossing shuffle. Integer unpacks for every width (current Intel
    cores run them on two ports, the floating-point ones on one). A tile that is not a whole number of groups ends with a
    group that overlaps the one before it, so there is no scalar tail; below one group, for 16-byte elements and without
    AVX2, the tile is moved by the strided move kernel along its longer side.
  - The tile: at most 8 KB of the kernel dtype (16 KB is slower for 1- and 2-byte elements); a whole row wide when 8 rows
    of it fit, else the row is cut evenly; then as many rows high as fit. From a row SPAN of 4 MB (elements a row × the
    column stride) the fewest rows are 16 (4- and 8-byte elements) or 32: each column piece is then a cache miss and
    further up a TLB miss. The last tile in each direction is moved back to end on the operand's edge and computes some
    results twice (`RunRows` never has an output that is an input).
  - The operand beside it: contiguous rows are read in place (or converted into the other block), one value a row is
    read in place, a second tiled operand gets its own block. One kernel call a tile when every operand's rows of the tile
    follow each other and the output's do; else one call a row.
  - Not tiled: 16-byte elements with rows over 64 elements (a 16-byte element is one move whatever the route),
    conversions to complex128 (the cast is slower than the per-element kernel), two F operands (the result is F-ordered:
    the iterator route), `out=` calls (they arrive as chunks).
- A matrix against one row (`ComparisonRun.RunRepeatedRow`): when the matrix's rows follow each other, there are 16 rows
  or more and a row is at most 2,048 bytes, the row is repeated in an 8 KB stack block (filled by doubling) and a block of
  rows is compared per call, instead of one call (and one scalar tail) per row. Above 2,048 bytes a row, or with fewer
  than 16 rows, the block's copy costs more than the calls it saves.
- Kernels (`Direct/`): the comparison vector kernels (`EmitComparisonBroadcastSimdLoop` for array-against-scalar, with the
  scalar broadcast once; four masks stored packed on AVX2 hosts, `EmitMasksToBools`, NumPy's `npyv_pack_b8_*`); complex128
  (`DirectILKernelGenerator.Comparison.Complex.cs`: four elements a step, two loads split into real and imaginary lanes
  with `vunpcklpd` / `vunpckhpd`, NumPy's `CLT` rule (a NaN in an imaginary part makes the ordered operators False), mask
  bits to bytes by `pdep` or integer arithmetic without BMI2; `>` is `<` with the operands exchanged, `!=` the complement
  of `==`); int64 against uint64 in either order (`.Comparison.MixedSign.cs`: NumPy's exact loops as machine compares, a
  negative lane decided by its sign, the others compared unsigned, eight elements a step); the strided move
  (`.StridedMove.cs`: four typed moves a pass at any two byte strides); the tile transposition (`.TransposeTile.cs`). The
  comparison `SimdFull` kernel and the contiguous predicate kernel (isnan / isinf / isfinite / isposinf / isneginf /
  signbit) never run a body past `n` (the sentinel tests in `ComparisonBroadcastKernelTests.cs`). bool has no shuffle
  kernel of its own, so bool results and inputs are copied as uint8.
- `np.less(int64, uint64, out=o)` compares exactly on every route.
- Gates: `Backends/Kernels/ComparisonRunTests.cs` (every dtype pair × operator × 1-D layout against an element reference
  written from NumPy's rules; 2-D and 3-D layouts; the four `out=` forms (a stepped out's skipped slots are not written);
  long runs off a vector boundary; scalar adoption over every uint8 / int8 value, every float16 bit pattern and the
  float32 values around each threshold; int64 against uint64 on every route; an out that is an input; `Run` and
  `RunRows` on raw memory with sentinel margins and against inaccessible pages (`FencedMemory`); the run against the
  routes the callers fall back to when `ComparisonRun.Disabled` is set; the tile and repeated-row routes against the row-by-row loop and
  against C-ordered copies, with the counters `ComparisonRun.TileRuns` / `RepeatedRowRuns` and the switches
  `TilesDisabled` / `RepeatedRowDisabled`), `Backends/Kernels/TransposeTileTests.cs` (tile shapes × widths × column
  strides × both row directions, with sentinel margins and against inaccessible pages),
  `Backends/Kernels/ComparisonBroadcastKernelTests.cs`, and the unit suite and oracle replay again under
  `NUMSHARP_DEBUG_GUARD_PAGES=1`.
- Performance floors (the cells under parity with NumPy): two stepped float64/int64 operands larger than L2 (NumPy's
  scalar loop overlaps its work with the memory latency); a C# literal operand on a small call (its 0-d temporary is a
  large share of the call; a held 0-d operand is faster); `np.less(a, b, out)` of two float64 arrays near L2 size (the
  transfer bound). Contiguous operands at L2 size run at the transfer bound on both sides. An empty broadcast result
  such as `(0,3) < (3,)` returns before any kernel (the `result.size == 0` guard of `DefaultEngine.CompareOp.cs`: the
  kernels assume at least one element).
