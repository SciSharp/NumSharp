# NumSharp Project Instructions

NumSharp is a .NET port of Python's NumPy library targeting **1-to-1 API and behavioral compatibility with NumPy 2.x**.

## NumPy Reference Source

A full clone of the NumPy repository is available at `refs/numpy/`. Use this as the authoritative reference for API behavior, edge cases, and implementation details when implementing or verifying NumSharp functions.

## Core Principles

1. **Match NumPy Exactly**: Run actual Python/NumPy code first, observe behavior, replicate in C#
2. **Match NumPy Implementation Patterns**: Don't just match behavior - match NumPy's implementation structure. If NumPy has a clean approach and NumSharp has spaghetti code, refactor to match NumPy's design
3. **Edge Cases Matter**: NaN handling, empty arrays, type promotion, broadcasting, negative axis
4. **Breaking Changes OK**: Breaking changes are acceptable to match NumPy
5. **Test From NumPy Output**: Tests should be based on running actual NumPy code

**When fixing bugs:** Don't just patch symptoms. Check `refs/numpy/` for how NumPy implements the same functionality, then refactor NumSharp to match NumPy's structure.

## Definition of Done (DOD) - Operations

Every np.* function and DefaultEngine operation MUST satisfy these criteria:

### Memory Layout Support
- **Contiguous arrays**: Works correctly with C-contiguous memory (SIMD fast path)
- **Non-contiguous arrays**: Works correctly with sliced/strided/transposed views
- **Broadcast arrays**: Works correctly with stride=0 dimensions (read-only)
- **Sliced views**: Correctly handles Shape.offset for base address calculation

### Dtype Support
All 15 NumSharp types must be handled (or explicitly documented as unsupported):
Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Half, Single, Double, Decimal, Complex

### NumPy API Parity
- Function signature matches NumPy (parameter names, order, defaults)
- Type promotion matches NumPy 2.x (NEP50)
- Edge cases match NumPy (empty arrays, scalars, NaN handling, broadcasting)
- Return dtype matches NumPy exactly

### Testing
- Unit tests based on actual NumPy output
- Edge case tests (empty, scalar, broadcast, strided)
- Dtype coverage tests

## Supported Types (15)

| NPTypeCode | C# Type | NPTypeCode | C# Type |
|------------|---------|------------|---------|
| Boolean | bool | UInt64 | ulong |
| Byte | byte | Char | char |
| SByte | sbyte | Half | System.Half |
| Int16 | short | Single | float |
| UInt16 | ushort | Double | double |
| Int32 | int | Decimal | decimal |
| UInt32 | uint | Complex | System.Numerics.Complex |
| Int64 | long |  |  |

All operations must handle all 15 types via type switch pattern.

**Fixed-width strings (`NPTypeCode.String`, Stage D phase 1 — branch `strings`, 2026-10-07).** NumPy's `str_` (`U`,
one native UCS-4 code point per character) and `bytes_` (`S`, one byte) live in ONE storage lane whose descriptor
(`UnmanagedStorage.Descr`) carries class and width (`<U5` itemsize 20, `|S5` 5) — so `NPTypeCode.String` alone names no
dtype and every string allocation takes a DType. Phase 1 = storage, element access (`item` → `string` / `byte[]`,
`NDArray.AsStringArray`, indexer writes cut to the width), C# conversion (`np.array(string[])`, `np.array(object[] of
byte[])` — a `byte[][]` is a NUMERIC jagged array), dtype integration (`StringDTypeMeta`, promotion, casting rules),
movement (every copy runs the numeric kernels through UNIT VIEWS `D + (chars,)`, `NDStringCopy`), string↔string
`astype` (ASCII `S ↔ U` with Python's Unicode error texts), `.npy` (byte for byte, mmap too; **breaking:** a `<U1` file
reads as a `<U1` string array, `Char` still WRITES `<U1`) and printing (CPython's `repr`, `PyRepr`) — all byte-identical
to NumPy 2.4.2. Computing on text is phase 2 (`docs/plans/strings-phase2.md`, landing subphase by subphase): the
comparisons and truth values (P2.1) and concatenation / repetition (P2.3: `np.add` / `+` over two strings of one class,
`np.multiply(text, int, out=)`, `np.strings.add` / `np.strings.multiply`) compute byte-identical to NumPy through the IL
text kernels (`Backends/Kernels/ILKernelGenerator.Text{,.Emit,.Concat}.cs`: element widths are emit-time constants, every
element is fixed-size chunk loads / stores, so no per-element branch follows its length — NumPy's own repeat loop runs a
variable-size memcpy per copy and loses to that 2–4×) driven by `Backends/Strings/NDString{Comparison,Truth,Arithmetic}.cs`.
**Text ↔ number conversion is phase 1 (P1.1, 2026-10-12, `docs/plans/strings-phase1-completion.md`)** — NumPy's legacy
casts, byte for byte: a number becomes its scalar's `str()` (`Backends/Strings/NDTextFormat.cs`: the float positional /
scientific windows per type, float16 / float32 shortest digits, complex forms; an unsized `'U'` / `'S'` takes
`REQUIRED_STR_LEN`, 21 for int64, 32 for the floats, 64 for complex; NumSharp's `char` converts as uint16, `decimal` as
float64), a text is parsed by CPython 3.12's `int()` / `float()` / `complex()` (`NDTextParse.cs`, with the pinned Unicode
tables of `Utilities/PyUnicode.cs`: decimal digits of every script, `str.isspace()`, the 4300-digit limit, strict UTF-8
for `S → complex`). `NDTextConversion.cs` walks NumPy's raw assignment order and commits blocks of 128 (a failing
element leaves the earlier blocks written, as NumPy does), converts only the selected runs under a mask, and quotes the
NumPy scalar in an array cast's error (`np.str_('x')`) but the Python object in a store of a C# string (`'x'`: a C#
string is a Python `str`, `NDArray.WasPythonStr`; **breaking:** `NDArray x = "[1, 2, 3]"` is a 0-d `<U9`, no longer
parsed numbers). Overflow texts are the LP64 ones (`Python int too large to convert to C long`). Every route converts:
`astype`, setitem / `fill` / `full` / `ones` / `pad` / `copyto` / `np.array`, numeric ufuncs into a text `out=` (with
`where=` the out's text is parsed into the loop dtype first, as NumPy's READWRITE buffer does), `np.add` under a string
`dtype=` over numbers, numeric-`dtype=` reductions and `np.cov` over text (parsed first). 2.0–7.9× NumPy at 1K / 100K / 1M.
**The NEP 55 `StringDType` descriptor exists (P1.2, 2026-10-12); its arrays come with P1.3.** `VStringDTypeMeta` is
the class (`numpy.dtypes.StringDType`, kind / char `'T'`, num 2056, a 16-byte cell aligned to 8, parametric, NO singleton:
every descriptor is fresh). It shares the `NPTypeCode.String` lane with `S` / `U` (decision D4: the descriptor's class
tells the layouts apart — test `descr.Meta is VStringDTypeMeta`, never the typecode; `FromTypeCode(String)` and
`FromScalarType(string)` stay `StrDType`). `VStringParameters` holds `na_object` / `coerce` and NumPy's derived fields;
`np.dtypes.StringDType(bool coerce = true)` and `np.dtypes.StringDType(object na_object, bool coerce = true)` build it
(a bool sentinel goes BY NAME: a positional bool binds `coerce`), and so does `np.dtype("T")`. Equality is NumPy's
`na_eq_cmp` through `Backends/Strings/NDPyObject.cs` (CPython's `str()` / `repr()` / `==` over the C# boundary map, with
NumPy's scalar rules when an `np.float16` (`Half`) or a decimal takes part: `np.float16(2048) == 2049` is True); the hash
is coarse for numbers so it can follow that non-transitive equality. The cast table is `casts.cpp`'s
(`VStringCastingImpls.cs`), promotion admits only `U` (the common instance takes the present sentinel and the FIRST
descriptor's coerce), `isdtype` refuses `T`, a `T` instance as a ufunc `dtype=` is NumPy's new-style refusal.
`Backends/Strings/NDUtf8.cs` is the strict UTF-8 codec with CPython 3.12's texts (a lone surrogate in a string sentinel is
`UnicodeEncodeError` already). Gate: oracle section `tdtype` (1,188 strict) + `DTypes/StringDTypeDescriptorTests.cs` +
`Backends/Strings/StringDTypeProtocolTests.cs`. Test trap: a lone surrogate cannot travel in `[DataRow]` or
`[DynamicData]` (UTF-8 metadata, XML-serialized rows); keep such strings in the test body.
**NEP 55 `StringDType` ARRAYS exist (P1.3–P1.6, P1.9 — 2026-10-12; plan `docs/plans/strings-phase1-completion.md`).**
Storage is NumPy's byte for byte (`Backends/Strings/VStringAllocator.cs` = `static_string.c`: the 16-byte cell, short
strings in the cell, medium / long ones in a per-BUFFER arena, heap strings after a longer rewrite; the cells' memory block
owns the allocator, so views keep it alive). Rules a new route must keep:
- **Never move a `T` cell as raw bytes** (an arena offset or a heap pointer means something only to its own allocator).
  `DTypeSize` / `dtypesize` THROW for a `T` storage so a raw route fails loudly; byte arithmetic reads `ItemBytes` (16).
  Gathers run the numeric route on the CELL view and re-home each cell (`NDVStringMove.Gather` / `Rehome` /
  `RehomeMulti`); scatters write int64 marks and duplicate the marked values (`ScatterMarks`); copies go through
  `NDVStringCopy.Copy` (NumPy's overlap rules, the destination-stride walk that decides arena offsets).
- **A store's value is NumPy's `PyArray_FromAny(values, dtype)`** (`NDVStringMove.StoreValue`): a Python object (a C#
  literal — marked `WasPythonLiteral` / `WasPythonStr` / `WasPythonBool` —, null for `None`, a boxed value, the elements of
  a C# array or list) by `stringdtype_setitem` (the NA object, its `str()`, refused under `coerce=False`), an ndarray by the
  cast. An index of integers alone given a Python object is NumPy's `array_assign_item`: setitem straight into the element
  (`UnmanagedStorage.SetVStringItem`), with setitem's MemoryError texts; a slice assignment and `fill` run the copy loop
  ("... in string to string cast"). Probed by laying crafted cells over a buffer (`np.ndarray((2,), T, buffer=…)`).
- **Out-of-memory paths are tested by fault injection**: `VStringAllocator.FailAllocationAfter(n)` makes the (n+1)-th arena
  or heap allocation of THIS thread fail (`DisarmFaultInjection` cancels); production pays one volatile read. Each
  operation must raise NumPy's text for its loop and dispose its half-built result (`StringDTypeFaultTests`: the live
  allocator count returns to its baseline).
- The oracle carries `T` values as values, never cells (`"t"` + `"values"`, `{"na": true}`), and compares `tobytes()`
  cells with the heap pointers masked; sections `tstore` / `tcreate` / `tmove` / `tcast` (8,324 strict).
- NumPy 2.4.2 bugs on `T` that NumSharp does not reproduce (pinned by unit tests, kept out of the corpus): `np.place`
  segfaults, `np.choose` / `np.put` / `np.putmask` copy cells raw, `bytes → T` skips UTF-8 validation, `T → complex` puts the
  real part into the imaginary part, a short cell copied onto itself (`a[...] = a[:1]`) becomes `''`.
- Still refused loudly (P1.7 / phase 2): np.nditer buffering or casting a `T` operand (allocating one works), `.npy`,
  `loadtxt` / `savetxt`, numpy.ma with `T`, and computation (comparisons, `np.sum`, sorting, `np.strings`).
- **Speed rules** (NPY/NS geomean 5.35×, minimum 1.80×, over copy / arena copy / stepped copy / take / mask / concatenate /
  where / repeat / `np.array(list)` / `U → T` / `T → U8` at 1K / 100K / 1M): allocator memory is reported through the
  pool's coarse accounting (`SizeBucketedBufferPool.AccountForeignAllocation` / `AccountForeignRelease`), never one
  `GC.AddMemoryPressure` per growth (that drove constant gen-2 collections). A bulk copy into an allocator with no arena
  text reserves the arena's exact total first (`TryReserve`; the arena's one reallocation site is `TryGrowArena`, pinned by
  the native-allocation chokepoint gate): only the capacity differs from NumPy's 125 % steps, never an offset. The
  pre-pass counts a text only where `heap_or_arena_allocate` puts it in the arena: a destination cell whose flags are 0
  once the free has cleared MISSING (a fresh cell, or an NA set on one); a cell that held a short string takes its new
  text on the HEAP, so counting it reserves arena space nothing uses. A fast path
  must give NumPy's cell for every input it takes: a short NON-EMPTY source into a cell that holds no arena or heap text
  is the source cell (two 8-byte moves; an empty source is `pack_empty`, which keeps the destination's state — a heap or
  arena destination must go through `free`), and `np.array(list, T)` writes a short ASCII text straight into its fresh cell.
Traps: a `byte[][]` without a dtype is a uint8 matrix (pass an `S` dtype for bytes objects); C# binds a literal to an
`NDArray` overload before an `object` one (`np.insert(t, 1, "x")` takes the literal's NDArray route — box it to reach the
object route); `nd.CopyTo(byte*)` binds the generic `CopyTo<T>(T*)` (cast to `void*`). Coverage of a change is measured
on the lines it ADDED (a cobertura run of the unit + oracle suites joined with `git diff -U0 <base>`): phase 1's is
99.4 % (3,600 / 3,623; the rest cannot run: host limits and guards behind excluded conditions). Mutation pass: 27 planted
bugs, 26 killed, 1 equivalent. A crashed test host counts as a kill (a corrupt cell can end it without a "Failed" line).
Where NumPy REFUSES (no string loop) NumSharp raises NumPy's type + text; where NumPy COMPUTES something not ported yet
(sort/search, the other `np.strings` functions) it throws
`StringComputationNotSupportedException` — both AFTER NumPy's earlier checks in NumPy's order (`dtype=` class check,
read-only `out=`, non-bool `where=`, the axis under one of three rules, the quantile q/axis/method/empty-axis order,
a computing loop's output cast and broadcast; `Backends/Strings/NDStringUfuncs{,.Precedence}.cs`). Every engine ufunc
member resolves its `dtype=` through `UfuncLoopCode(dtype, "<numpy name>", x1, x2, out, where)` so an unsized string
request (`'U'`, `'S'`) gets NumPy's per-ufunc answer. Gate: the string oracle (`test/oracle/gen_strings_oracle.py` →
`NumSharp.Tests.Oracle/Strings`, 43,825 cases in 23 sections, `strconv` = 4,487 conversion cases, `tdtype` = 1,188 descriptor cases; a phase-2 case passes as
"deferred" only while NumSharp throws `StringComputationNotSupportedException`, so phase 2 is gated by the same corpus,
and a section may defer no more than its ceiling (`StringOracleTests.DeferralCeilings`, 537 in all: sorting, searching,
set routines, histograms, quantiles), so a computation answered today cannot slide back; `NUMSHARP_STRING_ORACLE_DEFERRED=1`
lists every deferred case with its refusal) + `Backends/Strings/StringCallOrderTests.cs` + `StringShapeDecidedTests.cs`
+ `StringComparisonTruthTests.cs` + `StringArithmeticTests.cs` + `TextNumberConversionTests.cs`. **Validation pass (2026-10-07)** — the deferral rule is now exact: defer ONLY where NumPy
reads the text (compares, concatenates, parses, formats into a result); where dtypes, sizes or shapes decide NumPy's
answer, give that answer. So (1) an empty array, lines of one element, or a pair the comparison operators cannot compare
are answered from the shape — truth tests, counts, the nonzero family, arg-reductions, sort / partition, `np.unique` and
its Array-API forms, `isin`, the set routines, `array_equal` / `array_equiv`, `lexsort`, `sort_complex`, `trim_zeros`
(`NDStringUfuncs.ShapeDecided.cs`: the numeric path runs on a bool STAND-IN of the operand's shape, values restored from
the input) and binary ufuncs over an empty broadcast (`BinaryGuard`); (2) NumPy's checks before a conversion run in its
order (interp / digitize / repeat depth then 'safe' cast, conversions of nothing, place / putmask: an ndarray value meets
the 'safe' rule while a C# array is a Python list converted directly — new `Array` overloads, insert / delete obj,
searchsorted, choose, concatenate's out texts); (3) the rest of the surface refuses with NumPy's text: the linalg
gufuncs' input casts, the products' order (0-d → alignment → out → itemsize), convolve / correlate depth, SeedSequence
and random integer bounds over text arrays (CPython `int()` ported in `PythonInt`), the histogram family, quantile `q`,
reductions / scans with `out=`, nanargmax's out, polynomials, emath, unravel / ravel (`only int indices permitted`),
`common_type` / `min_scalar_type` / `can_cast` of arrays, `ediff1d`, text `where=` masks, `complex(U0)`. Numeric fixes it
found, all NumPy 2.4.2: `norm` of an empty operand (inf / 1 / 2 reduce with `initial=0`), `matrix_power(a, 1) is a`,
`np.cross` broadcasts its leading axes before the dtype (np.broadcast's text), `histogramdd`'s H is a VIEW of the padded
counts and the caller's edges come back as the same object, histogramdd's per-dimension scalar bin checks, `tensorinv`
with `ind > ndim`, argsort of a 0-d, 1-D `unique(axis=0)` = the flat path, `concatenate(axis=None)` of 0-d operands,
insert of a one-element sequence, delete of an N-D / int8 obj, digitize / trim_zeros / insert / delete error classes,
`min_scalar_type(NDArray)` (returned null). Oracle sections `strshape` / `strvalid` / `strsurface` (1,674 / 173 / 2,768)
pin all of it. **numpy.ma over text** (the same day; section `strma`, 2,762 cases over every layout; unit
`Backends/Strings/StringMaskedArrayTests.cs`, `Ma/NDMaskedArrayNumPyStatementsTests.cs`) runs numpy.ma's statements in
numpy.ma's order. Where the ufunc behind a member has no string loop, the member refuses with NumPy's text: the masked
reductions, products, extrema, `clip`, `allclose`, `median`, `trace` and `masked_values` (`add.reduce`'s resolved-dtypes
TypeError, the comparison no-loop texts that name `StrDType` / `Float64DType` classes, `isinf`'s safe-casting text, the
str / `_PyFloatDType` DTypePromotionError, `data type must provide an itemsize`). Where the member reads the text (a
number formatted into a filled slot, truth values, a sort with masked text), it defers. The text fill value is
MATERIALIZED at its own width (`'N/A'`, `<U3` / `|S3`) when it is first read. A later assignment writes INTO that stored
array (`'hello'` → `'hel'`), and only `__array_finalize__` converts it to a new dtype (`<U1` → `'N'`). The pass found
numpy.ma bugs for every dtype, fixed to numpy.ma's statements:
- fill / hard-mask inheritance: the `_arraymethod` views inherit, `take` does not, the `*_like` functions copy the mask,
  `_fromnxfunction` results always carry a mask array, `concatenate` shrinks an all-False mask;
- `ma.array` is soft by default, `masked_array` inherits the hardness, and a caller's bool mask is ADOPTED
  (`getmask(x) is m`);
- `getdata`, `filled`, `compressed` and `compress_nd` return views; `shrink_mask` works in place;
- `argsort`'s default axis flattens from 2-D; `alltrue` / `sometrue` are logical reductions along axis 0; `count` without
  a mask is a writeable full array; `flatnotmasked_edges` follows NumPy's three cases;
- `trace` casts the diagonal first when there is a mask; `minimum` / `maximum` take two operands
  (`where(less(a, b), a, b)`: a NaN does not propagate); `allclose` runs statement for statement; `average` raises the
  0-d `AttributeError`; `setdiff1d` keeps the first operand's dtype; `unique` keeps the first masked element's data.

It found np.* bugs too: `np.stack` of 0-d operands is `(n,)`, `np.convolve` checks emptiness before the length swap,
`np.vander` over text, `np.clip`'s no-loop text names Python-scalar bounds `int64` / `float64`, and `np.unique` of a
contiguous view with a non-zero `Shape.offset` read the elements before the view.

A follow-up the same day ported numpy.ma's attribute and operator semantics for every result kind
(`Ma/NDMaskedArrayInheritanceTests.cs`; probed member by member against NumPy, 259 cases, all NumPy's but 4
representational ones):
- the masked ufuncs take the first masked operand's fill value (RAW) and hard mask; `ma.power` takes its base's only and
  fills invalid slots with it; the shifts fill masked slots with 0 and inherit nothing;
- `-x`, `~x`, `%`, `&`, `|` and `^` run ndarray's ufunc and `__array_wrap__` (no fill-back, `%`'s domain filled with 1,
  a weak C# number); the ordering operators run `_comparison` (no fill-back, a shrunk COPY of the mask, the fill value
  cast to bool, `getdata` making a C# number strong);
- `copy`, `astype`, `view`, `diagonal`, `clip`, `round`, `sort`, `anom` and `diff` inherit; `real`, `imag` and `mT` reset;
- masks are shared or copied as NumPy shares or copies them (a binary ufunc built its mask on the operand's own mask
  object, so masking the result masked the operand);
- `ma.sort` of a hard-masked array writes back through the hard `__setitem__`; a float masked slot sorts by NaN;
- `ma.dot` always carries a mask array (an empty contraction masks every slot); `ma.diff` broadcasts a 0-d prepend;
  `masked_where(copy=False)` masks the input in place.

**numpy.ma's arraysetops are NumPy's statements** (2026-10-08; `Ma/NDMaskedArray.SetOps.cs`; gates the zero-excuse
oracle tier `ma_setops.jsonl` (7,030 cases: `gen_ma_oracle.py setops`) + `Ma/np.ma.setops.NumPyPort.Test.cs`):
`unique` (now with NumPy's `return_index` / `return_inverse`, a `MaskedArrayModule.UniqueResult` tuple), `in1d`, `isin`,
`intersect1d`, `setxor1d`, `union1d`, `setdiff1d` replay `numpy/ma/extras.py` over `_unique1d`, `ma.concatenate` (now
with `axis=None`), the masked sort / argsort (masked slots keyed by NaN for a floating dtype, the dtype maximum
otherwise) and the masked `==` / `!=` (two masked slots equal, a masked slot unequal to every value, every comparison
that reads a masked slot masked). So a flag whose sorted neighbour is masked is masked (`ma.in1d([3, 5], ma.array([3,
9], mask=[0, 1]))` is [True, --]), NaNs collapse only behind an UNMASKED last NaN, a masked bool slot sorts among the
unmasked Trues, a hard mask ORs into `ma.unique`'s in-place sort, and setxor1d compares FILLED values (999999 equals a
masked int slot). The old compositions (the plain set operation over the unmasked values plus one masked entry by a
presence rule) never masked an in1d flag. **One inherent divergence:** NumPy's quicksort is unstable (x86-simd-sort
networks for 32/64-bit dtypes, introsort past 16 elements), NumSharp's sort is stable, so the two differ only where an
unstable sort meets equal keys that are not interchangeable (a masked key tied with an equal unmasked value, ±0, NaN
payloads); the generator replays a stable copy of NumPy's statements (`test/oracle/ma_setops_reference.py`) and drops
exactly those cases. Rode along: `np.argsort`'s float32/float64 radix keyed `-0.0` below `+0.0` (NumPy's stable sort
keeps them in input order; `np.lexsort` inherited it), and `np.unique` of complex128 without flags kept a NaN from an
unstable partition where NumPy's hash path keeps each class's FIRST element. Perf (NPY/NS, 1K / 100K / 1M, int64 and
float64 with 10% masks, two pinned P-cores): geomean 3.3×, float64 3.6–8.7×, int64 1.5–2.6× (the 1M int64 cells
1.5–1.7×; NumSharp's int64 masked argsort sorts the unmasked values alone when no unmasked value equals the fill key,
and in1d's `order.argsort()` of a permutation is its inverse permutation, one scatter).

**`ma.put` is NumPy's `MaskedArray.put` / `ma.put`, statement for statement** (2026-10-08; `Ma/NDMaskedArray.cs`; gates
the zero-excuse, type-strict oracle tier `ma_put.jsonl` (3,660 cases: `gen_ma_oracle.py put`) +
`Ma/np.ma.put.NumPyPort.Test.cs`). Under a HARD mask (a mask array present, even an all-False one) NumPy runs `mask =
self._mask[indices]; values = narray(values, subok=True); values.resize(indices.shape); indices = indices[~mask]; values
= values[~mask]` before `self._data.put`, and so does NumSharp, with every consequence: the mask lookup indexes the
FIRST axis of an N-D mask, so every hard-masked N-D put raises IndexError before it writes (`too many indices for array:
array is 1-dimensional, but 2 were indexed`, or the lookup's own out-of-bounds text); short values are ZERO-PADDED, not
cycled (`put([0, 2, 3], 5)` writes 5, 0, 0; a C# number or array is a fresh `np.array`, which resizes); masked values
(the masked constant included) raise `A masked array does not own its data and therefore cannot be resized.`; an
`NDArray` values of another size raises `ndarray.resize`'s errors (the caller holds it, so NumPy's reference check
refuses; single-segment and OWNDATA first — the house `IncorrectShapeException`); the lookup ignores `mode`, the write
does not (under `clip` a negative index writes, and unmasks, index 0); a literal is converted before the cast, so it
wraps where the soft branch raises the OverflowError. The soft branch writes a masked values' DATA (the masked
constant's 0.0, cast — it used to leave those slots' data), a plain target goes through `ndarray.put`, and the mask is
written in place and shrunk to nomask when all-False. **One divergence (`[Misaligned]`):** for an `NDArray` values of
the indices' size but another shape NumPy reshapes the caller's array in place; NumSharp relabels an alias, because
`x.data` / `x.mask` hand out a masked array's own arrays (NumPy hands out fresh views). Neighbours fixed with it (see
"Selection" and "Indexing"): `np.put` / `np.take` as `PyArray_PutTo` / `PyArray_TakeFrom`, the shared clip-mode
converter, the indexer's `prepare_index` IndexErrors, and the C-order casts of five selection functions.

**The fill value is stored as NumPy stores it, for every dtype** (2026-10-08, plan `docs/plans/api-parameter-parity.md`
X33, the rundown's bug 4; `Ma/NDMaskedArray.cs`; gates the zero-excuse, type-strict oracle tier `ma_fill.jsonl` (3,867
call-history scripts: `gen_ma_oracle.py fill`) + `Ma/np.ma.fill_value.NumPyPort.Test.cs`). `_fill_value` is an NDArray
(null = NumPy's None) — the text path's model, which the numeric dtypes now share:
- **Conversion** (`MaskedArrayModule.CheckFillValue`, `_check_fill_value`): a given fill becomes a 0-d array of the DATA's
  dtype through `NDArray.AsStoreValue` (`PyArray_FromAny`): a C# number is a Python scalar, packed with the range checks
  (`5.7` on int8 is 5; `300` raises `TypeError: Cannot convert fill_value 300 to dtype int8`, the OverflowError /
  ValueError rewrapped, the value in Python's `%s`); a complex into a real dtype raises Python's own `int()` / `float()`
  TypeError; a `Half` / `char` / `decimal` (NumPy scalars) or any array is cast (`np.float16(300)` on int8 is 44).
- **The default is materialized on first read** (`DefaultFill`, `_check_fill_value(None)`) at the DEFAULT's own dtype:
  int64 999999 for every signed integer dtype, uint64 for unsigned (char too), float64 1e20 for every float (decimal
  too), complex128, bool. So `fill_value` of int8 data is a `long`, the repr prints `fill_value=np.int64(999999)`, and
  `filled()` casts it under `same_kind` (63 in int8). The repr reads the getter, so printing materializes too.
- **The setter writes INTO a stored array** (`_fill_value[()] = target`, unsafe): `fill_value = 5` on a fresh int8 array
  reads back `(sbyte)5`, after a read of the default `5L`; `fill_value = null` after `fill_value = 5` writes 999999 into
  the int8 fill (63). An array with dimensions into a stored 0-d fill is NumPy's `ValueError: setting an array element
  with a sequence.` (complex: `only 0-dimensional arrays can be converted to Python scalars`).
- **Inheritance**: `_update_from` (the `_arraymethod` family, views, the masked ufuncs, the operators) SHARES the stored
  object, so once a fill was set or read a set on either array reaches both (`x.copy()` included); `__array_finalize__`
  (constructors, `astype`, `view`, `byteswap`, `argpartition`, `masked_where`, the `*_like` functions) stores a converted
  COPY — `astype(int64)` of a float64 array whose default 1e20 was READ gives `long.MinValue` (MSVC's out-of-range cast),
  of an unread one 999999. The masked singleton refuses both getter and setter (`AttributeError: attributes of masked are
  not writeable`).
- **Lifetime**: a NEW array's fill (StoreFill) is tracked by the ambient scope like its data and leaves with it
  (`Yield`); a fill stored into an EXISTING array (the getter's default, the setter) is detached (`Own`).
- **Divergence**: `filled()` whose stored fill cannot reach the data under `same_kind` (a fill of another kind shared with a
  derived array: `ma.absolute(complex_with_fill).filled()`) is NumPy's OBJECT-array fallback; NumSharp raises
  `NotSupportedException` (no object dtype).
Neighbours fixed with it: `byteswap` / `argpartition` / `ma.cumsum` / `ma.cumprod` gave the result the SOURCE's mask
(a mask write through the result reached the source; NumPy copies it) and `byteswap` / `argpartition` came back soft;
`view(dtype)` of an array with a mask and another itemsize raises NumPy's text `cannot reshape array of size 4 into shape
(16,)` (the house IncorrectShapeException, NumPy's ValueError; was NotSupportedException); the indexer parses a lone slice
string once for the data and the mask (`x["1:"]` 2.84 → 1.77 µs; GetView rewrites its slice ARRAY, so the mask indexes a
copy of it); `ma.apply_along_axis` stores the default fill at its result's dtype (NumPy's closing
`result.fill_value = ma.default_fill_value(result)`: an int8 / uint8 / int16 / uint16 result raises `Cannot convert
fill_value 999999 to dtype uint8`, an int32 one reads `np.int32(999999)`); `ma.default_fill_value(obj)` takes an array, a
scalar or a list as well as a dtype (NumPy's `_get_dtype_of`; None is `'?'`); `x + 1` on an int8 / uint8 / int16 / uint16 / bool masked array is numpy.ma's
strong int64 (the literal reaches `operator +(NDMaskedArray, NDArray)` as a marked int32 0-d array, which
`BareScalarStrongDtype` now reads as the Python int); the masked print renders float16 / float32 elements as the Python
floats `astype(object)` makes of them (`1.732421875`, not `1.732`).

Open: a 0-d result NumPy returns as a plain
scalar stays a 0-d masked array. (`argwhere` / `flatnonzero` / `unravel_index` return NumPy's views since 2026-10-08:
see "Sorting & Searching". An empty matrix's QR / SVD without the OpenBLAS backend is NumPy's answer since
2026-10-08: see "The BLAS/LAPACK API surface". `np.nditer` iterates strings since 2026-10-08: see "Iteration".)
**Trap:** NumPy's ufunc dispatch cache makes some answers depend on call HISTORY (`np.prod(int8, dtype='U')` names no
loop in a fresh process but finds the string repeat loop after `np.multiply(u, int8, dtype='U')`); the corpus records
the fresh-process answer. **Movement perf** (NPY/NS, two pinned P-cores, 100K / 1M `<U8`): copy 5.0 / 3.7×, strided /
reversed / transposed copies 2.5–4.4×, `astype` width changes 2.7–4.2× and `S ↔ U` 44–86×, concatenate 3.3–3.6×, take /
fancy / mask 1.7–3.1×, `np.where` 3.7–5.8×, `np.array(string[])` 2.2×. The routes: a same-layout pair is one block move
(`NDStringCopy.TryCopyDenseBlock`), a non-overlapping pair over ≤ 2 effective axes runs `GetElementCopyKernel` (one
constant-size cpblk/initblk per element — the unit view's narrow rows cost 4–8 ns an element), `np.where` selects
whole elements (`GetWhereBytesKernel`), a lone index array / prefix mask gathers whole rows on the fancy-index kernel
route (an explicit trailing `:` on the unit view forced the general index path, 28× slower), a contiguous fill doubles
block moves, and every fully-overwritten `U` result skips NumPy's NEEDS_INIT zero fill (`NDArray.ForOverwrite`).
Floors: 1K calls ~1.0–1.3× (the per-call NDArray floor), `copyto` / set-slice / fill ~1.0–1.4× (both sides move the same
bytes at memory speed), `NDArray.AsStringArray` at 1M 0.7× (.NET's GC of 1M live strings). Design + status: `docs/plans/dtype-system.md` §5 "Stage D phase 1"; user docs
`docs/website-src/docs/fundamentals/strings-and-bytes.md`.

**Perf notes:**
- SByte / Byte / Int*/UInt* / Single / Double — full SIMD via the mixed-type kernel's `SimdFull` execution path (V128/V256/V512 detected at startup).
- Half — **no vector f16 arithmetic or F16C in the BCL** (`Vector<Half>` throws), yet most of the f16 ufunc surface BEATS NumPy 2–60× bit-exactly via three techniques picked by "does the op need real float math?": bit-level AVX2 on raw ushort lanes (sign/negate/abs, min/max/ptp/nanmin/nanmax on every layout flat and per axis — NumPy's sequential HALF loops bit for bit, first NaN / first ±0 in NumPy's visiting order, see "Math — Reductions" —, maximum/minimum/fmax/fmin, the six comparisons, clip, floor/ceil/trunc/rint — all on the proven sign-magnitude order map `key = bits ^ (0x8000 | asr(bits,15))`, NaN keys strictly outside the finite band), SIMD widen-compute-narrow for add/sub/mul/div (Giesen f16→f32 → one f32 vector op → RTNE narrow — bit-exact incl. the MSVC operand-order NaN-payload host pin, 2.2–3.6×; on every layout since 2026-10-06: row / column broadcasts, strided and reversed views gathered in 256-element blocks, broadcast F sets walked in F order, and out= — 4–7× NumPy, where a per-element coordinate decomposition ran 0.22–0.5×, plan api-parameter-parity X4 / X15), and scalar-F16C widen-unary (`EmitUnaryHalfViaFloat`: `(Half)MathF.X((float)h)`, NumPy's npy_half model) for 16 transcendentals — conversion-bound ≈ parity, the physical ceiling. `np.power` is NumPy's HALF_power since 2026-10-06 (plan api-parameter-parity X20): the exact widen (a signalling NaN stays signalling), the CRT powf, npy_float_to_half, NO scalar-exponent fast path — `HalfPower` runs a vector widen and narrow around one MathF.Pow loop per 256-element block, 1.52–1.56× NumPy (powf is ~5 ns an element on both sides; the old double bridge differed on 7,161 of 1,114,112 base × exponent pairs). Full design, semantics contract (NaN-guarded npy_half comparators, first-NaN/first-zero folds, sNaN quieting), dispatch map and traps (portable Vector256 ushort ops don't lower — 30×; RyuJIT re-swaps commutative intrinsic operands — NaN priority must be an explicit blend; AVX2 lacks 16-bit variable shifts): **`docs/FLOAT16_DESIGN.md`**.
- Complex — scalar path via `System.Numerics.Complex` operators / `Complex.Pow`. ~2× slower than NumPy. The
  complex128 **unary** ufuncs (`Utilities/NDComplexMath.cs`: sqrt/log/log2/log10/log1p/exp/exp2/expm1/square/
  reciprocal/sin/cos/tan/sinh/cosh/tanh/arcsin/arccos/arctan/arcsinh/arccosh/arctanh/conjugate/negative/positive)
  match NumPy 2.4.2 **bit-for-bit including the NaN SIGN** — win-amd64 runs MSVC UCRT's complex functions, whose
  "produce a NaN" slots emit the positive `NPY_NAN` (`0x7ff8…`), while `.NET double.NaN` is `0xfff8…`; on x86 both
  engines share deterministic SSE, so the only divergences were NumSharp's `double.NaN`/`inf-inf`/`Complex.Exp`
  seeds. Fixed per-path (positive-NaN constant where NumPy canonicalises; `hypot`/`cexp`/`ctanh` propagate the
  input NaN's sign; `csinh/ccosh` canonicalise so `csin/ccos` follow the transform negate; `square` = true
  `vfmaddsub` with `re*im` in the addend slot; `reciprocal` = the exact `CDOUBLE_reciprocal` loop's `-1/d`
  division, not a `-scl` negate). **Gated**: the oracle compares these ops' complex NaN **raw-byte**
  (`ComplexNanContractOps` + `BitDiff.Compare(nanBitExact)` + `DiffHasSignFlip` hard-fail), verified over the full
  ±0/±inf/±NaN grid (both NaN signs). The flat float `sum` / `mean` now run NumPy's own pairwise order (see
  "Math — Reductions"), but a sum of SEVERAL NaNs still returns another NaN's bits: NumPy's MSVC-compiled
  `pairwise_sum` keeps the accumulator's NaN in its leaf loop and the second operand's NaN in its tree combine
  (8 NaNs with payloads 1–8 → payload 8; NumSharp → 1). Value is NaN either way — the oracle tokenizes float NaN —
  and the case is an `[OpenBugs]` pin with NumPy's bits. Complex/float16 `sign`/`maximum` preserve the input NaN
  bits, matching NumPy.
- Decimal — scalar path via `DecimalMath.Pow`. Highest precision, slowest.

## Architecture

```
NDArray           Main class (like numpy.ndarray)
├── Storage       UnmanagedStorage (raw pointers, not managed arrays)
├── Shape         Dimensions, strides, offset calculation
└── TensorEngine  Computation backend (DefaultEngine = pure C#)

np                Static API class (like `import numpy as np`)
├── np.random     NumPyRandom (1-to-1 seed/state with NumPy)
└── np.*          Functions in Creation/, Math/, Statistics/, Logic/, etc.
```

### NumSharp.Core is 100 % managed C# — optional packages are the only native path

`NumSharp.Core` has **no native dependency and no P/Invoke**: every kernel is its own managed code,
and that is the default a user gets. A native backend arrives ONLY as a separate NuGet package, and
Core's whole knowledge of it is ONE abstraction plus ONE settable property on the engine:

```csharp
public interface IBlasBackend                                    // Backends/IBlasBackend.cs
{
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;   // optional (DIM)

    // + 15 more DIMs in IBlasBackend.LinearAlgebra.cs, all defaulting to false:
    //   products      TryInner TryVdot TryVecdot TryMatvec TryVecmat
    //   factorisations TryCholesky TryDet TrySlogdet TryEig TryEigh TryInv
    //                  TryLstsq TryQr TrySolve TrySvd
}

public abstract partial class TensorEngine
{
    public IBlasBackend Blas { get; set; }   // null (the default) = NumSharp's own managed kernels
}
```

The package implements the interface and assigns the property from a `[ModuleInitializer]`, so
**referencing the package is the whole opt-in**; assigning null takes it back out. Every member is
`Try`-shaped: a backend answers only for the operands it implements and returns false for the rest,
so an optional package can change *which* implementation computes a product, never *whether* one can
be computed. The engine is **not** subclassed and not replaced — deliberately: an `NDArray` binds
its engine at CONSTRUCTION and `np.dot(a,b)` dispatches on `a.TensorEngine`, so swapping the engine
would only reach arrays created afterwards, while a property on the cached singleton takes effect
for arrays that already exist (probed: 38/38 construction/operation/IO paths reach the one
`EngineCache<DefaultEngine>` instance).

Three rules the review pinned, all with a reproduction (`docs/stale-docs/GEMM_PARITY.md` §9): read `Blas` into
a **local** before calling it (test-then-call on a settable property was ~2 % NREs under concurrent
`Disable()`); a failed `OpenBlasEngine.Enable` is a **no-op** (it used to unload the working library first,
leaving `Enabled` reporting true while products silently reverted to managed kernels); and a backend
needs **no** `InternalsVisibleTo` — read operands as `(T*)a.GetData().Address + a.Shape.Offset` with
`a.Shape.Strides` (all public; `a.GetData<T>().Address` densifies non-contiguous views and must not
be paired with strides). `TryMatMulBatched` is a default interface method taking a whole stacked
product, so per-product setup is hoisted the way NumPy hoists it out of its gufunc's outer loop —
bit-identical to the per-element route (25/25 A/B), 0.80× → **6.47×** on 2000 stacked 8×8 f32.

| Package | Implements | Serves | Why |
|---------|-----------|--------|-----|
| `NumSharp.Interop.OpenBLAS` (`src/NumSharp.Interop.OpenBLAS/`) | `OpenBlasBackend : IBlasBackend` | `np.dot`, `np.matmul`, `np.inner`, `np.vdot`, `np.vecdot`, `np.matvec`, `np.vecmat` for float32/float64/**complex128** | matrix + vector products through OpenBLAS — faster on large matrices, and **byte-identical to NumPy**. **Bundles the binaries NumPy itself pins** (scipy-openblas 0.3.31.22.0, byte-identical to numpy 2.4.2's) as per-RID runtime assets, so parity needs no Python installed. See below and `docs/stale-docs/GEMM_PARITY.md`. |
| `NumSharp.Interop.OnnxRuntime` (`src/NumSharp.Interop.OnnxRuntime/`; **UNRELEASED**, see below) | nothing on the engine — a CONVERSION library (`NDArrayOnnxInterop` verbs + `InferenceSessionExtensions` + `Postprocess`), like the pythonnet bridge | `nd.AsOrtValue(mode)` (an `OrtValue` over the NumSharp buffer; ARC-rooted handle `OrtTensor`, whose `IsView` says view or copy), `nd.ToOrtValue()` (ORT-allocated copy), `AsDenseTensor<T>(mode)`/`ToDenseTensor<T>` (legacy tier, always row-major), `ortValue.ToNDArray()` (owning copy) / `ortValue.AsNDArray(ownsValue, mode)` (zero-copy lease over ORT's buffer, CPU only; a counted NumSharp alias for a value the package made), the readers `ToNDArrays`/`ToMap`/`ToMaps`/`GetMapKeys`/`ReadStringTensor` (each numeric array a view that OWNS ORT's copy of its element), `session.Run(NDArray…)` (metadata-driven names incl. overridable initializers and `optional(tensor)` inputs, NumPy-`casting` dtype coercion checked on every call, declared-shape validation, copy / owning-view outputs), `session.RunInto` (pre-allocated outputs; an output that overlaps an input is computed from a copy under Auto, refused under View), `session.RunIntoAsync`/`RunAsync` + `IsRunAsyncSupported` (ORT's native async, ORT ≥ 1.30.0, a session with ≥ 2 intra-op threads; `RunAsync` allocates only a numeric output of a fixed declared shape), `Postprocess.{Softmax,LogSoftmax,Sigmoid,Argmax,TopK}` (one source file compiled into both bridges, `src/NumSharp.Interop.Shared/`) | ONNX Runtime does not compute NumSharp ops; it consumes tensors NumSharp produces and produces tensors NumSharp reads — so no `TensorEngine` seam, no `[ModuleInitializer]`, no native asset. Depends on `Microsoft.ML.OnnxRuntime.Managed [1.16.0, 2.0.0)` ONLY (the native flavours CPU/Gpu/DirectML are mutually exclusive — the consumer picks). **`OnnxCodecMode { Auto, View, Copy }`** (the members of the pythonnet codec's `NumpyCodecMode`) on every entry that could share memory: Auto = a view when ORT can read the buffer in place (C-contiguous + a bit-exact dtype), else a copy the handle owns; View throws instead of copying; Copy always copies. 12 dtypes zero-copy incl. Half↔Float16 bit-reinterpret; Char→UInt16; Decimal→Double temp the handle owns (Auto/Copy; `Run` needs `casting: "same_kind"`); **text**: fixed-width `U`/`S` ↔ ORT string tensors, ALWAYS a copy (a trailing NUL is lost until NumSharp has `StringDType`: the boxed TODO at the top of `NDArrayOnnxInterop.Text.cs`; text ↔ number casts do not exist yet); Complex/BFloat16 refused; sparse tensors, non-tensor `Run` outputs and non-`NDArray` inputs are deferred. Imports of NumSharp memory (a value the package made, a fed input passed through) are counted aliases that inherit read-only; only an OWNING view adds GC memory pressure. **Five ORT facts the 2026-10-10 rework measured and pinned** (plan + record `docs/plans/onnxruntime-rework.md`): (1) a model OUTPUT that is a FED input (no node produces it) is never written into a pre-allocated buffer (it keeps its contents, or gets the initializer's DEFAULT) and the value ORT returns for it lies in the FED memory — `InferenceSessionExtensions.FedTensor` (by name, ORT's own rule) makes `RunInto` copy it and the zero-copy form alias it; (2) ORT completes `RunAsync` on its OWN intra-op thread, where `session.Dispose()` never returns, and awaits without `ConfigureAwait(false)` (a blocked UI thread waits forever) — the bridge's task uses `RunContinuationsAsynchronously` and hides the caller's context / scheduler from ORT; before 1.30.0 the managed `RunAsync` does not pin its argument arrays (#32015), so the forms refuse older ORT; (3) ORT does not guard a RELEASED `OrtValue` (a native call on one is an access violation that ends the process) — the handle records its disposed value in a weak table and every import verb throws `ObjectDisposedException`; (4) `NodeMetadata` reports a declared scalar and an unknown rank identically (`dims=[]`), so `RunAsync` does not allocate such an output and `Run`'s shape check leaves it to ORT; (5) the 1.16.0 managed assembly carries AssemblyVersion 0.0.0.0, so ONLY the NuGet range sets the floor (no `FileLoadException`). Gate: `test/NumSharp.Tests.Interop.OnnxRuntime` (311 at ORT 1.31.0; at the 1.16.0 floor the 20 tests of the async behaviour are Inconclusive and the test of the refusal runs; 38 committed `.onnx` models from `test/oracle/gen_onnx_models.py`, which takes model names so a new model does not rewrite the committed ones — incl. the 2026-10-02 technique suites: every-dtype × 47-layout export matrix (a cell per mode), property-based, metamorphic, ORT-consumer (six feed routes through ORT's own Identity), bit-pattern fuzz, concurrency stress, ARC/fault injection, boundary (imports of a tensor ≥ 2 GB that ORT owns are REFUSED up front: ORT exposes tensor memory only as an `int`-length Span) and API approval; the 2026-10-03 lifetime/GC suite `LifetimeGcTests` (every finalization order of 8 scenarios as a staged REAL collection, held Value/Tensor/Source survival, in-flight copies; finalizers ABANDON, never free eagerly; an OrtValue keeps its handle via a ConditionalWeakTable, a tensor's memory manager keeps it by back-reference; analysis `docs/plans/interop-lifetime-gc-analysis.md`); and the 2026-10-10 rework suites (`SessionRunModeTests`, `ImportAliasTests`, `TextTests`, `PassThroughOutputTests`, `SessionRunAsyncTests`, `ValueOwnershipTests`, `MemoryManagerTests`); coverage 94.3 % lines / 93.6 % branches, six mutation samples (102 of 106 planted bugs killed; the 4 others are equivalent or not observable from outside). The `interop-test` CI job's ONNX Runtime steps run floor 1.16.0 AND current 1.31.0 (the current leg is the only one that runs the async tests: never set it below 1.30.0), no Python (its own `onnxruntime-interop-test` job until 2026-09-23). **Mutation runs restore a file from memory only:** copy untracked files out of the tree before one.) Docs `docs/website-src/docs/interop/onnxruntime.md`; plan + business review `docs/plans/onnxruntime*.md`. |
| `NumSharp.Interop.MLNet` (`src/NumSharp.Interop.MLNet/`; **UNRELEASED**, see below) | nothing on the engine — a CONVERSION library (`NDArrayMLNetInterop` verbs + `NDArrayDataView` + `Postprocess`), like the ONNX/pythonnet bridges | `nd.AsDataView(colName)` (**vector column** — a lazy `IDataView` reading the NumSharp buffer through its strides, ANY layout, ARC-pinned; the `NDArrayDataView` is `IDisposable`) / `nd.AsDataView(string[])` (**scalar column per feature**) / `nd.ToDataView(…)` (snapshot), `view.ToNDArray("col")` (materialize a column — scalar→(R,), fixed-size vector→(R,C)), **`vbuffer.AsNDArray<T>()` (ZERO-COPY view over a dense VBuffer's private `T[]` via a compiled-expression accessor `VBufferAccessor<T>` — pinned, write-through, leased; sparse/empty→copy)**, `nd.ToVBuffer<T>()` / `vbuffer.ToNDArray()` (COPY), `Postprocess.{Softmax,LogSoftmax,Sigmoid,Argmax,TopK}`. Feed a model: `transformer.Transform(nd.AsDataView("Features")).ToNDArray("Score")` — IDataView is the whole bridge, no per-transformer runner. | ML.NET does not compute NumSharp ops; `IDataView` is its universal currency, so feeding a pipeline and reading its output are both conversions — no `TensorEngine` seam, no `[ModuleInitializer]`, no native asset, and no `ITransformer` dependency. Depends on `Microsoft.ML.DataView [2.0.0, 6.0.0)` ONLY (the standalone contract package; there is NO standalone `Microsoft.ML.Data` package — `ITransformer`/`MLContext` are in the full `Microsoft.ML` the consumer brings, which NuGet unifies upward). 11 dtypes direct + Char→UInt16 one-way; VBuffer zero-copy is ONE-directional: **import** `vbuffer.AsNDArray<T>()` IS zero-copy (reflected compiled-expr access to the private `T[] _values` — VBuffer is managed-`T[]`-backed at every version 2.0.0/3.0.0/4.0.2, NO `ReadOnlyMemory<T>`/non-public ctor, so **export** `NDArray→VBuffer` can NEVER be zero-copy — a managed array can't alias NumSharp's unmanaged buffer → `ToVBuffer` copies; the bulk export path `AsDataView` is already zero-copy anyway); Half→Single (lossless) / Decimal→Double converted by the view verbs (owned temp), Complex refused; ML.NET has NO half/decimal/complex column type. Rank 1–2 only. Gate: `test/NumSharp.Tests.Interop.MLNet` (207 — incl. the 2026-10-02 technique suites (layout matrix, property-based, metamorphic, cursor-state differential vs ML.NET's own cursors, real-pipeline consumer, bit-pattern fuzz, concurrency, ARC/fault injection, boundary, API approval) and the 2026-10-03 lifetime/GC suite `LifetimeGcTests` (every finalization order as a staged real collection; a getter keeps its cursor reachable until its row is committed, proven against a no-KeepAlive control); mirrors the ONNX interop suite's shape modulo what ML.NET's model lacks: dtype map + every-column-type/CLR-type edges, export/import round-trips, the zero-copy `VBuffer.AsNDArray` (write-through/pooled-slice/sparse-fallback/lease-lifetime/derived-slice/compiled-accessor-available), ALL layouts×both modes, output-array flags, empties/0-d/degenerate shapes, special/extreme values bit-exact through EVERY path incl. a real pipeline (NaN uncanonicalized/±inf/-0 sign-bit/subnormal/int-min-max/ulong>2^53/char BMP), sparse/empty/boundary VBuffers, the cursor contract (Position/Batch/IdGetter/IsColumnActive/inactive+wrong-T throws/concurrent cursors), variable-length/text→throw, key→UInt32, lazy null-rowcount, error/null taxonomy, lifetime (source-disposed-safe, resize-refcheck-refuses-while-exported, many-views-one-buffer, disposed-source-cannot-export, finalizer net), THREAD-PARALLEL exports/cursors settling to baseline, doc-snippet examples, and REAL `Microsoft.ML` pipelines — NormalizeMinMax/Concatenate/Sdca/SdcaMaximumEntropy-multiclass-Score-vector/OneHot/MapValueToKey + POCO-loader cross-check; the `interop-test` CI job's ML.NET steps run floor 2.0.1 AND current 4.0.2, no Python (its own `mlnet-interop-test` job until 2026-09-23)). No zero-copy import lease / device-memory / multi-input-Run tests — ML.NET's model has no analog. Docs `docs/website-src/docs/interop/mlnet.md`. |
| `NumSharp.Interop.System.Numerics.Tensors` (`src/NumSharp.Interop.System.Numerics.Tensors/`; released from 0.71.0) | nothing on the engine — a CONVERSION library (`NDArrayTensorsInterop` verbs + `TensorSpanHandle<T>`), like the pythonnet bridge | `nd.AsTensorSpan<T>()` → `TensorSpanHandle<T>` (zero-copy `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` over the NumSharp buffer for ANY non-negative-stride layout incl. strided/transposed/broadcast, ARC-pinned; `.Span`/`.ReadOnlySpan` are rebuilt per access because a ref struct cannot be stored), `nd.ToTensor<T>()` (dense copy), `tensor.ToNDArray()` (owning copy), `tensor.AsNDArray()` (zero-copy view over the tensor's pinned buffer), `span.ToNDArray()` (copy) | `System.Numerics.Tensors` does not compute NumSharp ops (routing kernels through `TensorPrimitives` would abandon bit-parity with NumPy), so no `TensorEngine` seam, no `[ModuleInitializer]`, no native asset. Depends on `System.Numerics.Tensors [10.0.0, 11.0.0)` ONLY: pinned to 10.x because `Tensor<T>`/`TensorSpan<T>` are `[Experimental("SYSLIB5001")]`, and the package re-marks its surface `[Experimental("NUMSHARP_TENSORS")]` so a consumer opts in once. ALL 15 dtypes cross as their own CLR type (the BCL containers are unconstrained generics: Half directly, Decimal/Complex/Char too). A negative-stride view is refused (the BCL forbids negative strides; `ToTensor` or `np.ascontiguousarray` first). Namespace `NumSharp.Interop.Tensors`, not the assembly name: a `System.*` namespace would shadow the types it uses. Gate: `test/NumSharp.Tests.Interop.System.Numerics.Tensors`, both TFMs in the `interop-test` CI job (since 2026-10-02). Docs `docs/website-src/docs/interop/system-numerics-tensors.md`. |

**Unreleased packages (hidden since 2026-10-02).** Three interop packages are merged code but are NOT
released: `NumSharp.Interop.OnnxRuntime` and `NumSharp.Interop.MLNet` (the rows above) and
`NumSharp.Interop.ParquetNet` (Parquet columns decoded straight into `NDArray` buffers). The owner's
decision: they build, test and merge like everything else, but no NuGet package is created and no release
or public documentation mentions them until the owner releases them. `NumSharp.Interop.System.Numerics.Tensors`
was hidden with them on 2026-10-02 and released again the same day by the owner: it ships from 0.71.0
(the row below). "Hidden" is exactly three things:
1. **No package.** Each csproj sets `<IsPackable>false</IsPackable>`, which makes `dotnet pack` a silent
   no-op: exit 0, no `.nupkg`, and not even a compile (probed with a project holding a compile error).
   CI still names all three in the signing check and in `build-nuget`'s Build and Pack steps, and
   `create-release` lists exactly the `.nupkg` files that were packed (ids read from each nuspec), so
   packing, signing, publishing and the release text all follow that one property.
2. **No docs.** `docs/website-src/docfx.json` excludes `docs/interop/{onnxruntime,mlnet}.md`, because DocFX
   publishes EVERY `.md` under `docs/website-src`, toc entry or not. `toc.yml`, `interop/index.md`,
   `advanced/index.md`, `absolute-basics.md` and the System.Numerics.Tensors page do not name or link them.
   ParquetNet has no page yet (its `PackageProjectUrl` points at a page to write when it ships).
3. **Still gated.** The `interop-test` job builds and tests all three (ONNX Runtime and ML.NET at floor and
   current, ParquetNet on both TFMs). Nothing else compiles them in CI.

Keep them out of release notes (`docs/releases/RELEASE_<version>.md`; when that file is missing, GitHub
generates the notes from merged PR titles, so keep PR titles free of them too), the website, the root
README, and the shipped XML docs and README of every RELEASED package (a released package's IntelliSense
must not point at a bridge nobody can install). To release one, as System.Numerics.Tensors was: set its
`IsPackable` back to `true` (or delete the property), drop its page from the docfx.json exclude, and restore
its `toc.yml` entry and its `interop/index.md` row and "See also" link from git history. CI needs no edit.
Master got the same treatment for the three that had reached it (OnnxRuntime, MLNet,
System.Numerics.Tensors), which also took their pages off the public site; the System.Numerics.Tensors page
returns when journey4 merges.

**`NumSharp.Interop.OpenBLAS`** exists because no portable algorithm can match NumPy's float matrix
products: for f32/f64 mat@mat NumPy **always** calls cblas (since gh-23588 it copies non-blasable
operands into a temp rather than take its own portable loop), and scipy-openblas' `sgemm` uses an
arch-specific **multi-accumulator** scheme matching neither a sequential mul+add chain nor a
sequential FMA chain — NumSharp's managed GEMM differed on **94.5 %** of a `(128,784)@(784,128)` f32
product (max ~976 K ULP) and **45 %** at K=10, which compounds until 50 Adam steps spread it to 73 %
of a weight matrix. The backend is a route-for-route port of **both** NumPy dispatchers, because
they are different C code and **disagree** when an operand is not blasable (a stride-2 matrix @ vector:
`np.dot` takes gemv-on-a-copy, `np.matmul` the portable loop — 278/300 elements differ): `Dot` →
`cblas_matrixproduct` + the N-D `dotfunc` tail (which is NOT gemm), `MultiplyMatrix` →
`@TYPE@_matmul`'s five routes incl. the `a @ a.T` **syrk** shortcut and NumPy's copy/transpose
rules — which is why `IBlasBackend` has both `TryDot` and `TryMatMul2D` rather than one
matrix-product method. Strides are ported in ELEMENTS (NumPy's are bytes — same logic with
`itemsize == 1`). Scope is **Single/Double/Complex128** — the three dtypes NumPy itself routes
through cblas (`matmul.c.src`'s `#USEBLAS = 1,1,0,0,1,1,…` covers FLOAT/DOUBLE/CFLOAT/CDOUBLE).
complex128 rides `zgemm`/`zgemv`/`zsyrk`/`zdotu` (the UNCONJUGATED dot behind dot/matmul/inner/matvec)
plus `zdotc` (the CONJUGATING dot behind vdot/vecdot/vecmat) — with alpha/beta passed BY POINTER
(`{1,0}`/`{0,0}`) and the dots via the `_sub` out-pointer form, exactly as NumPy calls them —
because complex float accumulation is **NOT** associative and so genuinely needs the BLAS path
(served only when the loaded library exports the complex products; a bare real-only CBLAS declines
back to the managed complex kernel). Every OTHER dtype falls through to the managed kernel: integer
and bool products are bit-exact by construction — modular addition **is** associative, so summation
order cannot change the result.

**The package BUNDLES the binary** (2026-07-26). It ships the prebuilt OpenBLAS artifacts NumPy
itself pins — `scipy-openblas64==0.3.31.22.0` from numpy's `requirements/ci_requirements.txt`,
**verified byte-identical** to numpy 2.4.2's copy (both sha256 `74a40872…`; NumPy's mangled filename
IS that hash) — as `runtimes/<rid>/native/` assets for 8 RIDs (62 MB packed). This is not a reversal
of the old "ships no native binaries" rule but its resolution: the objection was to bundling *an*
OpenBLAS, and this is *the* one. **win-arm64 is scipy-openblas32**, because NumPy's own build selects
LP64 there — it exports `scipy_cblas_sgemm`, a symbol scheme `Bind` lacked (added; binding a
win-arm64 numpy's BLAS would have failed outright). Binaries are gitignored;
`tools/openblas-manifest.json` is the checked-in pin and `tools/fetch_openblas.py` verifies **two**
hashes per RID. Discovery order is fixed (`docs/stale-docs/OPENBLAS_DELIVERY_DESIGN.md` §3; runtime side
implemented, gate `OpenBlasDeliveryTests`): explicit path/`NUMSHARP_OPENBLAS_LIBRARY` (binding) →
override path(s) (`NUMSHARP_OPENBLAS_SEARCH_PATH` env, then a build-recorded `OpenBlasPath` marker;
non-binding) → build-staged **version override** (`openblas.source.json` marker, `mode:"version"` —
**hard-required**: a miss throws `OpenBlasRequiredOverrideException`, never falls through; sha-verified
when pinned) → explicit OpenBLAS root (`OPENBLAS_HOME`/`OPENBLAS_ROOT`, promoted ABOVE the bundle
2026-08-14 — a deliberate opt-in that trades parity-by-default for the named build) → **bundled**
(the parity default; §10.1 resolved: it stays ABOVE *ambient* machine tooling) →
machine-wide OpenBLAS (apt/brew/MacPorts/conda/vcpkg/source install dirs; honours the ambient
`VCPKG_ROOT`/`CONDA_PREFIX`; **not** parity) → bare names (64-bit **and**
32-bit spellings — both scipy-openblas64/ILP64 and scipy-openblas32/LP64 bind, and 32 is the only
build for 32-bit x86) → PATH sweep (last resort, non-standard names); `NUMSHARP_OPENBLAS_USE_BUNDLED=0` drops
the bundled entry. `numpy.libs` is NEVER scanned (tier deleted — we never grab OpenBLAS out of a
numpy installation; the bundle already IS that binary at the pin). **Build-time override** (design
§5–§7, shipped in `buildTransitive/{props,targets}` + an inline `RoslynCodeTaskFactory` task —
netstandard2.0 surface only, own `MiniJson`, since the factory can't resolve simple-name References):
`<OpenBlasVersion>` on ANY `PackageReference` (or `NUMSHARP_OPENBLAS_VERSION` — env wins) downloads
that scipy-openblas from PyPI at build, double-hash-verifies, caches the EXTRACTED lib per-user
(`%LOCALAPPDATA%/NumSharp/openblas`, wheel deleted, offline-capable once primed), stages it over the
bundle and writes the marker; failure = BUILD ERROR (hard requirement). `<OpenBlasPath>` = soft
read-in-place dir (also the download-to dir when combined with a version);
`<OpenBlasDelivery>package</OpenBlasDelivery>` bakes ALL 8 RIDs + markers into a dependent's own
nupkg. Two MSBuild traps encoded there: a target's `Condition` evaluates BEFORE `DependsOnTargets`
run (the has-override gate must sit on the TASK, not the target), and `dotnet pack` defaults to
Release on .NET 8+ while this repo's incremental build can declare stale Release outputs up-to-date
— always `-t:Rebuild` before packing. The cache is SELF-HEALING: every hit re-verifies the entry's
content hash (the dir name IS the claim) and a truncated/tampered entry is discarded and
re-downloaded; writes are temp+rename so killed/racing builds can't strand garbage under the hash
name. Gate: `tools/verify_build_override.sh` (15-step scripted nupkg-flow run incl. tamper,
mirror-needs-sha, invalid-knob, delivery=none, two-ref-conflict and poisoned-cache steps; network
once, then cache) + `OpenBlasDeliveryTests` (20 — priority pins and hostile markers).

**The vendored Fortran runtime, and the macOS loader step (2026-09-05).** The Linux/macOS wheels do
not statically link the GCC runtime: `libgfortran`/`libquadmath`/`libgcc_s` ride beside the main
under content-hashed names that the main's own `DT_NEEDED`/`LC_LOAD_DYLIB` entries name (a system
libgfortran can never stand in), so `fetch_openblas.py` co-stages them and the manifest pins each
(`vendored` per RID; a partial staging fails `--check` and is re-staged). Linux: beside the main in
`native/`, found by its `$ORIGIN` RUNPATH. **macOS is the trap:** `delocate` parks the deps in a
`.dylibs/` SIBLING of `lib/` and rewrites the dylib's load commands to `@loader_path/../.dylibs/<name>`,
but NuGet delivers native assets ONLY from under `runtimes/<rid>/native/` (nested paths included; a
sibling folder is silently dropped from every consumer). Mirroring the wheel's sibling layout produced
a package that looked complete and loaded on NO Mac — and CI was green because every suite consumes
the package by ProjectReference, where a Content item copied the sibling. The deps are now staged and
packed NESTED (`runtimes/osx-*/native/.dylibs/`, the layout a ProjectReference build gets too, so CI
exercises the consumer path), and `MacOsVendoredRuntime` (`src/NumSharp.Interop.OpenBLAS/`) makes the
dylib loadable after a plain `dlopen` fails: it parses the Mach-O load commands (`MachOImage`, header
only), plans the missing `@loader_path/` deps and their closure (sources: `<dir>/.dylibs/<name>`, then
`<dir>/<name>` — a RID-specific publish FLATTENS the sub-folder away, probed), and materializes them
**in place** for the `runtimes/<rid>/native` layout (relative dir symlink `runtimes/<rid>/.dylibs →
native/.dylibs`, else per-file copy; never outside `runtimes/`) or, when that folder is read-only or
the layout is flattened (`../.dylibs` would be the PARENT of the app), as a **per-user cache copy** of
main + deps in the exact relative layout (`NUMSHARP_OPENBLAS_CACHE_DIR`, else
`~/.cache/NumSharp/openblas/dyld/<sha256>/native/` + `.dylibs/`; content-hash keyed, every file
byte-compared to its source on every hit, temp+rename). `OpenBlasEngine.LibraryPath` keeps naming the
file discovery chose (the parity pin hashes THAT; the copy is verified identical); the new
`OpenBlasEngine.LoadedImagePath` is what dyld mapped, and `Info` says `(mapped from …)`. Two dead
ends, both verified: pre-loading the deps by absolute path (dyld matches by path/install name, and
delocate's deps carry the placeholder `/DLC/scipy_openblas64/.dylibs/…`) and relinking with
`install_name_tool` (macOS tooling + re-signing on a Linux staging runner; changes the pinned bytes).
The build-time version override task co-extracts the vendored deps into its cache entry (`.entry.json`
sidecar, all files hash-verified on every hit; a pre-sidecar main-only entry is discarded) and stages
them with the main, so an override loads on Linux/macOS too. Gates: `OpenBlasMacOsVendoredRuntimeTests`
(21, synthetic Mach-O images + the real staged dylibs, every OS) and the "Package consumer:" steps of
the `interop-test` CI job (a separate `package-consumer-smoke` job until 2026-09-23) running
`tools/verify_package_consumer.sh` + `tools/verify_build_override.sh` on all three OSes — the real nupkg
restored into a scratch consumer BY PACKAGEREFERENCE, loaded in the portable, read-only and flattened
layouts (the path no ProjectReference suite can exercise). They run LAST in that job, after every
test suite in it, because both scripts `-t:Rebuild` Core/OpenBLAS in Release in the repo tree and evict
this version from `~/.nuget/packages`; and that job's `env:` must never set
`NUMSHARP_OPENBLAS_CACHE_DIR` (verify_build_override.sh asserts the default per-user cache root on
Linux/macOS and does not unset it).

**Four load-bearing details:** the result bits depend on the BLAS **thread count** (1/2/4/24 threads
give four different answers); they ALSO depend on the **DYNAMIC_ARCH kernel** the CPU dispatches, so
bundling pins the build but *not* the answer — `OpenBlasEngine.Enable(coreType: OpenBlasEngine.CoreType)` pins that
too (probed: Haswell/Nehalem/Sandybridge/Katmai all differ; it is opt-in because the default must
match the local NumPy, which also dispatches by CPU; it must be set BEFORE load, and forcing an ISA
above the CPU **kills the process with SIGILL**, so `Enable` refuses what it can recognise). A
**named library is binding** (`OpenBlasEngine.Enable(path)` / `NUMSHARP_OPENBLAS_LIBRARY` is never silently
substituted, not even by the bundled copy). And the BLAS path is *faster* than the managed GEMM
anyway (1.7–13×; 2048³ f64 293 ms vs 3860 ms).

**Trap:** `Environment.SetEnvironmentVariable` does NOT reach a native `getenv` — .NET keeps its own
table, so setting `OPENBLAS_CORETYPE` that way reads back correctly and changes nothing. Go through
the CRT (`ucrtbase!_putenv_s` / `libc!setenv`); Windows' `SetEnvironmentVariableW` does not work
either.

API: `OpenBlasEngine.Enable(library, threads, coreType)` / `OpenBlasEngine.Disable()` / `OpenBlasEngine.Enabled` / `OpenBlasEngine.Info` /
`OpenBlasEngine.LibraryPath` / `OpenBlasEngine.LoadedImagePath` / `OpenBlasEngine.CoreName` / `OpenBlasEngine.IsBundledLibrary`
(decided by CONTENT — the loaded file must hash to a manifest pin embedded in the assembly, `BundledAssetPin`, so a
RID-specific/single-file publish that flattens the bundle into the app root still reports true — AND marker-aware: a
build-staged override folder reports false even with identical bytes; both evaluated live) / `OpenBlasEngine.CoreType`;
`NUMSHARP_OPENBLAS_CACHE_DIR` moves the per-user cache (macOS relocated dylibs + the build-override cache); `NUMSHARP_OPENBLAS_SEARCH_PATH` adds
highest-priority (non-binding) discovery locations tried before bundled;
`NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0` opts out of the module-load install (`OpenBlasEngine.BundleAutoinstall`;
the pre-rename `NUMSHARP_BLAS_BUNDLE_AUTOINSTALL` and `NUMSHARP_BLAS_AUTOINSTALL` spellings are RETIRED
and ignored — removed before any
release shipped them, pinned by `RetiredAutoinstallAlias_NoLongerSuppresses` — and a
required-override miss at module load is reported to stderr, backend left uninstalled — never
substituted). Gate: the **host-pinned**
`matmul_parity` corpus tier (589 cases — float32/float64/**complex128** `dot`·`matmul` plus the five
product gufuncs `inner`/`vdot`/`vecdot`/`matvec`/`vecmat`, all bit-exact with the backend; the complex
tier alone is 195 cases, the float ones stay byte-for-byte what they were)
which goes `Inconclusive`, never red, on a host that cannot load the pinned library. The pin matches
the library by **content hash**, not filename — bundling forced that and it is strictly stronger
(a name is mangled by delvewheel on one side and plain on the other, so name-matching skipped a
genuinely bit-identical host and would have accepted a differently-built same-named library).
Packaging contract: `MatmulParityBackendTests` (30 tests).

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Unmanaged memory | Benchmarked fastest; optimized for performance |
| Buffers of one page or more start on a cache line | `SizeBucketedBufferPool.AlignedMinBytes` (4096) / `BufferAlignment` (64), since 2026-10-10. The C allocator aligns to 16 bytes, so half of all buffers started in the middle of a 32-byte vector and every second vector load of a SIMD kernel crossed a cache line: two-array kernels lost 12–25 % at L2 sizes by the buffer's address. A kernel can align one stream with a prefix; only the allocator aligns all of them. See "The pool aligns its buffers" under Comparison & Logic. |
| Order-aware layout | Row-major (C-order) remains the default. `Shape` also tracks F-contiguity, and APIs with an `order` parameter resolve NumPy `C`/`F`/`A`/`K` modes through `OrderResolver`. |
| TensorEngine abstract | Pluggable computation backend behind one contract (`DefaultEngine` = pure C#) |
| View semantics | Slicing returns views (shared memory), not copies |
| Shape readonly struct | Immutable after construction (NumPy-aligned). Contains `ArrayFlags` for cached O(1) property access |
| Broadcast write protection | Broadcast views are read-only (`IsWriteable = false`), matching NumPy behavior |
| ILKernelGenerator + DirectILKernelGenerator | Runtime IL emission with SIMD V128/V256/V512 (~36K lines). Two classes split by kernel-driving contract: `ILKernelGenerator` (per-chunk kernels driven by NDIter — `np.where`, flat/pairwise reductions) and `DirectILKernelGenerator` (whole-array kernels, 63 partials in `Direct/`). |

## ILKernelGenerator + DirectILKernelGenerator

Runtime IL generation via `System.Reflection.Emit.DynamicMethod` for high-performance kernels. **Two physically distinct classes**, each encoding its kernel-driving contract in the type name:

### `DirectILKernelGenerator` — whole-array kernels (63 partials)

**Location:** `src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.*.cs`

**Contract:** A single kernel call processes the **whole array**. The kernel itself walks dimensions/strides; the iterator (if any) is only a setup helper. Signature shape varies per kernel family but typically:

```csharp
delegate void DirectKernel(
    void* input, void* output,
    long* strides, long* shape, int ndim,
    long iterSize);
```

Carries the bulk of NumSharp's elementwise, reduction, scan, cast, and selection kernels.

**Partial files in `Direct/` (63):**
| Category | Files |
|----------|-------|
| Core | `DirectILKernelGenerator.cs` (type mapping, SIMD detection), `.Scalar.cs` |
| Binary | `.Binary.cs`, `.MixedType.cs`, `.Shift.cs` |
| Unary | `.Unary.cs`, `.Unary.Math.cs`, `.Unary.Decimal.cs`, `.Unary.Vector.cs`, `.Unary.Predicate.cs`, `.Unary.Strided.cs` |
| Comparison | `.Comparison.cs` |
| Reduction (flat) | `.Reduction.cs`, `.Reduction.Arg.cs`, `.Reduction.Boolean.cs`, `.Reduction.NaN.cs` |
| Reduction (axis) | `.Reduction.Axis.cs`, `.Reduction.Axis.Arg.cs`, `.Reduction.Axis.Boolean.cs`, `.Reduction.Axis.Simd.cs`, `.Reduction.Axis.NaN.cs`, `.Reduction.Axis.VarStd.cs`, `.Reduction.Axis.Widening.cs` |
| Scan | `.Scan.cs` (CumSum, CumProd) |
| Masking | `.Masking.Boolean.cs`, `.Masking.NaN.cs`, `.Masking.VarStd.cs` |
| Cast & Copy | `.Cast.cs`, `.Cast.Masked.cs`, `.Cast.Scalar.cs`, `.Cast.Half.cs`, `.Cast.ToHalf.cs`, `.Cast.FloatNarrow.cs`, `.Cast.FloatWideInt.cs`, `.Cast.FloatToUInt.cs`, `.Cast.IntNarrow.cs`, `.Cast.ShortNarrow.cs`, `.Cast.SubwordCopy.cs`, `.Cast.SubwordNarrow.cs`, `.Cast.SubwordWiden.cs`, `.Cast.ToBool.cs`, `.Cast.Complex.cs`, `.Copy.cs` |
| Selection | `.Where.cs`, `.Where.Scalar.cs`, `.Place.cs`, `.Put.cs`, `.Take.cs`, `.TakeLeading.cs`, `.NonZero.cs`, `.Argwhere.cs`, `.Indices.cs`, `.Filter.cs`, `.Search.cs` |
| Linear algebra | `.MatMul.cs`, `.Trace.cs` |
| Other | `.Clip.cs`, `.Modf.cs`, `.Repeat.cs`, `.Quantile.cs`, `.WeightedSum.cs`, `.RavelMultiIndex.cs`, `.UnravelIndex.cs`, `.InnerLoop.cs` |

### `ILKernelGenerator` — NDIter-driven per-chunk kernels

**Location:** `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator*.cs` (root, not in `Direct/`)

**Contract:** Kernel processes **one inner-loop chunk** per call. The iterator (`NDIterRef`) drives the loop and advances pointers between calls. Matches NumPy's `PyUFuncGenericFunction` model:

```csharp
unsafe delegate void NDInnerLoopFunc(
    void** dataptrs,   // [nop] current operand pointers
    long*  strides,    // [nop] per-operand byte stride for inner loop
    long   count,      // number of elements this chunk
    void*  auxdata);   // op-specific extras (e.g. axis index)
```

Emits the per-chunk kernels run as an NDIter inner loop — `np.where` and the flat/pairwise reductions (partials `ILKernelGenerator.{Where,Reduction,Reduction.Pairwise}.cs`). Driven via `NDIterRef.Execute(kernelKey)`.

### Shared infrastructure

Both classes use these helpers at `src/NumSharp.Core/Backends/Kernels/` (root, outside `Direct/`):
- `VectorMethodCache.cs`, `ScalarMethodCache.cs` — reflection caches (Vector{128,256,512}, Math/MathF)
- `KernelOp.cs` — `BinaryOp`, `UnaryOp`, `ReductionOp`, `ComparisonOp`, `ExecutionPath` enums
- `BinaryKernel.cs`, `CopyKernel.cs`, `ReductionKernel.cs`, `ScalarKernel.cs` — kernel key structs and delegate types
- `StrideDetector.cs` — layout classification (Contiguous / Strided / Broadcast)
- `SimdMatMul.*.cs` — matmul SIMD primitives

**Execution Paths (DirectILKernelGenerator):**
1. **SimdFull** - Both operands contiguous, SIMD-capable dtype → Vector loop + scalar tail
2. **ScalarFull** - Both contiguous, non-SIMD dtype (Decimal) → Scalar loop
3. **General** - Strided/broadcast → Coordinate-based iteration

**The direct binary route writes its result in C-coordinate order over the coalesced dims** (`ExecuteKernelCoalesced` passes no result strides), so an F-allocated result is correct there only when every dim coalesces into one run. Two strictly F operands where one BROADCASTS (`(2,3,2)` F with `(2,1,2)` F) do not coalesce, and until 2026-10-05 that route allocated F for them anyway: scrambled values for add / subtract / multiply / divide, maximum / minimum / fmax / fmin and the bitwise ops (every SIMD dtype under the 8,192-element small-broadcast cutoff, float16 at any size). Now `HasNonScalarBroadcastOperand` stops that route from allocating F for a broadcast set directly, and the small-broadcast shortcut sends all-F sets to `NDIter`, which allocates F (plan `docs/plans/api-parameter-parity.md` X3; gate: the binary tiers' `pp_fbcast_*` layouts + `Backends/Kernels/FOrderBroadcastBinaryTests.cs`). Since 2026-10-06 (X4) a broadcast all-F set on that route (float16 add / subtract / multiply / divide, at any size) is walked with its axes REVERSED: the kernel's C-coordinate order over the reversed dims is the F memory order, so the linear write fills an F result directly (no C result copied to F, no 1,200-element strides along the inner axis).

**NEP50 Dtype Alignment (NumPy 2.x):**
| Operation | Returns |
|-----------|---------|
| `sum(int32)` | `int64` |
| `prod(int32)` | `int64` |
| `cumsum(int32)` | `int64` |
| `abs(int32)` | `int32` (preserves) |
| `sign(int32)` | `int32` (preserves) |
| `power(int32, float)` | `float64` |

**ILKernel Coverage:**
| Category | Operations |
|----------|------------|
| Binary | Add, Sub, Mul, Div, Power, FloorDivide, BitwiseAnd/Or/Xor |
| Shift | LeftShift, RightShift (SIMD for scalar, scalar loop for array) |
| Unary | Negate, Abs, Sign, Sqrt, Cbrt, Square, Reciprocal, Floor, Ceil, Truncate, Trig, Exp, Log, BitwiseNot |
| Reduction | Sum, Prod, Min, Max, Mean, ArgMax, ArgMin, All, Any, Std, Var |
| Scan | CumSum, CumProd (element-wise SIMD + axis support) |
| Comparison | Equal, NotEqual, Less, Greater, LessEqual, GreaterEqual |
| Clip/Modf | Clip, Modf (SIMD helpers) |
| Axis reductions | Sum, Prod, Min, Max, Mean, Std, Var (iterator path) |
| Cast (SIMD, contig+strided) | float→int32 (cvtt); float/c128→u32, →u64 (AVX2 mod-2³² reduce + hi/lo split, bit-exact incl. NumPy modular wrap — but note the NaN/±inf/out-of-range inputs are a **C-undefined conversion**: NumSharp reproduces the MSVC NumPy answer, glibc/gcc's differs, and so do one build's vectorized vs scalar loops, so that cell is a NumSharp regression pin rather than portable NumPy parity; see `Fuzz/README.md` → "Host-dependent values"; strided →u64 negcol `[:, ::-1]` via contiguous load + VPERMQ reverse, `::2` via 2-load deinterleave (b loaded at `2i+3` so no over-read), c128 negcol via UnpackLow+VPERMQ-0x72 deinterleave-reverse — reals for →u64, reals **and** imags for →bool); float→{i8,u8,i16,u16,char} (cvtt+Narrow); int→narrower-int (Narrow); Half↔ via Giesen bit-fiddle: f16→f32 widen (NaN-payload exact), f16→i64, f16→{i32,u32,u64,narrow}; X→Half (Giesen narrow, sNaN-preserving) incl. i64/u64→f16 (AVX2: clamp |v|≥65520→±70000 sentinel → low-32 PermuteVar8x32 pack → cvtdq2ps → Giesen); c128→{int,bool} (real deinterleave); *→bool (!=0; f16→bool strided via SubwordNarrow-style deinterleave (`::2`)/reverse (`::-1`) + NumPy half-truthiness `(bits & 0x7fff)!=0` so ±0→False, NaN/inf→True). Strided rows gather (4/8-byte) or stage gather→buffer→convert. Sub-word (1B/2B) strided/negcol (non-gatherable) use SIMD lane shuffles (`.Cast.SubwordCopy.cs`/`.Cast.SubwordNarrow.cs`): same-type & same-size-bit-reinterpret copies via VPACKUS deinterleave (`[:, ::2]`) / VPSHUFB reverse (`[:, ::-1]`); 2B-int→{1B,bool} via deinterleave/reverse + narrow; 1B-int→2B via deinterleave/reverse + sign/zero `Vector256.Widen` (`.Cast.SubwordWiden.cs`). The inner SIMD loop is inlined in the odometer. The sub-word deinterleave starts its SECOND load one element early and shifts the wanted half down instead of masking, so the load ends on the group's last element (2026-10-10: seven loops read one element past a stepped view, see "A strided copy or cast never reads outside its view" below). |

**A strided copy or cast never reads outside its view (fixed 2026-10-10).** The two-load deinterleave of the one- and
two-byte strided kernels (`a[::2]`: `.Cast.SubwordCopy.cs` ×2, `.Cast.SubwordNarrow.cs` ×2, `.Cast.SubwordWiden.cs` ×2,
the float16-to-bool loop of `.Cast.ToBool.cs`; all from 2026-06-20) loaded N elements at `s[2i]` and N at `s[2i+N]` for
a group of N results. That covers `s[2i+2N-1]`, ONE element past the last one the group needs.
- **The consequence.** Values were right, so no value test and no oracle case saw it. When the view's last element is
  its buffer's last element (`a[1::2]` of an even-length buffer, `a[::2]` of an odd-length one) the extra element is
  outside the buffer, and a buffer that ends where mapped memory ends made the copy an access violation:
  `np.zeros(new Shape(40 * 4096), NPTypeCode.Byte)["1::2"].copy()` ended the process (a zeroed array of 128 KB or more
  is its own `VirtualAlloc` region; NumPy returns 81,920 zeros). The same for int16 / uint16 / char / float16 copies,
  same-width casts (uint8 → int8), 2-byte → 1-byte / bool narrowing, 1-byte → 2-byte widening and float16 → bool.
- **The fix.** The second load starts at `s[2i+N-1]`. Its lanes hold the wanted elements in their HIGH half, so a
  logical right shift by one element width replaces the mask: the same instruction count, and the load ends exactly
  on `s[2i+2N-2]`. Measured in one process on (1000, 1000) → (1000, 500): 1-byte 26.6 µs before and 26.8 after,
  2-byte 87.7 and 87.8. The 4- and 8-byte kernels (`.Cast.WordCopy.cs`) already bounded their loop for this.
- **Gate.** `Backends/Kernels/StridedCastFenceTests.cs`: every source array is a view over `FencedMemory` (an
  inaccessible page directly below and above the buffer), for all 15 × 15 dtype pairs, 1-D and 2-D layouts, and
  lengths around each kernel's vector group; one test repeats the `np.zeros` call above. Each of the seven loops,
  reverted alone, ends the test host with its own kernel in the stack. See "Memory-safety gates" under Build & Test.

## Shape Architecture (NumPy-Aligned)

Shape is a `readonly struct` with cached `ArrayFlags` computed at construction:

```csharp
public readonly partial struct Shape
{
    internal readonly int _flags;         // Cached ArrayFlags bitmask
    internal readonly int _hashCode;      // Precomputed hash code
    internal readonly long size;          // Total element count
    internal readonly long[] dimensions;  // Dimension sizes
    internal readonly long[] strides;     // Stride values (0 = broadcast dimension)
    internal readonly long bufferSize;    // Size of underlying buffer
    internal readonly long offset;        // Base offset into storage
}
```

**ArrayFlags enum** (matches NumPy's `ndarraytypes.h`):
| Flag | Value | Meaning |
|------|-------|---------|
| `C_CONTIGUOUS` | 0x0001 | Data is row-major contiguous |
| `F_CONTIGUOUS` | 0x0002 | Data is column-major contiguous |
| `OWNDATA` | 0x0004 | Array owns its data buffer |
| `ALIGNED` | 0x0100 | Shape bit (cleared by `setflags(align=False)`); `flags.aligned` also needs `UnmanagedStorage.IsDataAligned` (false over a caller's buffer at a misaligning byte offset) |
| `WRITEABLE` | 0x0400 | False for broadcast views |
| `BROADCASTED` | 0x1000 | Has stride=0 with dim > 1 |

**Key Shape properties:**
- `IsContiguous` — O(1) check via `C_CONTIGUOUS` flag
- `IsFContiguous` — O(1) check via `F_CONTIGUOUS` flag
- `IsBroadcasted` — O(1) check via `BROADCASTED` flag
- `IsWriteable` — False for broadcast views (prevents corruption)
- `IsSliced` — True if offset != 0, different size, or non-contiguous
- `IsSimpleSlice` — IsSliced && !IsBroadcasted (fast offset path)

**Shapes and arrays SHARE dimension and stride arrays** (an allocation keeps its caller's shape, an elementwise result
keeps its input's), which is safe only because nothing public writes into them: `Shape` has no setter, `a.shape`,
`Shape.Dimensions` and `Shape.Strides` return COPIES (NumPy's tuples), and every public constructor, conversion and
method that takes a caller's `long[]` copies it (NumPy's `PyArray_IntpConverter`). Inside Core, read the internal fields
`x.Shape.dimensions` / `x.Shape.strides`, never `x.shape`, `x.Shape.Dimensions` or `x.Shape.Strides` (each allocates on
every read), and build a shape over an array Core created with the internal `Shape.Adopt(arr)` (also
`Adopt(arr, order)`, `Adopt(dims, strides)`, and `x.reshape(Shape.Adopt(arr))`), never `new Shape(arr)` or `(Shape)arr`,
which copy. Inline values (`new Shape(3, 4)`, `a.reshape(2, 3)`, `a.resize(2, 3)`) bind `params ReadOnlySpan<long>`
overloads and allocate once. Plan `docs/plans/view-shape-open-bugs.md`, bug 2.

## Critical: View Semantics

**Slicing returns views, not copies!** Memory is shared:
```csharp
var view = original["2:5"];  // Shares memory with original
view[0] = 999;               // Modifies original[2]!
var copy = original["2:5"].copy();  // Explicit copy
```

**A view keeps its owner's buffer alive, and so does a byte-reinterpreting alias (fixed 2026-10-01).** Every NDArray
holds one counted reference on its storage's slice (`TryAddRef` at construction, `Release` on `Dispose`, `Abandon`
from the finalizer), and an ordinary view shares the owner's slice. A view that REINTERPRETS the bytes as another
element type needs a slice of its own: a complex array's real/imaginary lane (`np.real`/`np.imag`/`.real`/`.imag`),
`view(dtype)`, `getfield`. That slice used to be a plain non-owning wrap (`ArraySlice.Wrap`, an immortal
`Disposer.Null`), so disposing the owner freed the buffer under a live alias. The alias then read whatever the pool
handed out next. `np.linalg.eigvals`' real result is `w.real` of a complex buffer its `[NDScoped]` wrapper disposes, so
a second eigvals call (or `{p}roots`) overwrote the first's values single-threaded, and under threads read other
threads' eigenvalues. Now `ArraySlice.WrapShared` gives the alias an `AllocationType.Forward` disposer, which forwards
`TryAddRef`/`Release`/`Abandon`/`IsReleased`/`IsUniquelyReferenced` to the owner's slice (an alias of an alias reaches
the owning block). Observable consequences, both NumPy's: the owner's refcount counts each live alias, so
`ndarray.resize(refcheck: true)` refuses while one lives; and the buffer returns to the pool when the LAST of owner
and aliases is released. Gate: `Backends/Unmanaged/ArcLifecycleTests.cs` (`ReinterpretingAlias_*`). **Corollary: an
INTERNAL alias must now be disposed** — or created inside an `[NDScoped]` member — because an undisposed one holds the
owner's buffer out of the pool until the finalizer runs. The leak gates caught the four unscoped members that left one
behind: the `real`/`imag` SETTERS and `setfield` (now `[NDScoped]`), and `view<T>()`/`getfield<T>()`. Those two used
to wrap the alias in an untyped NDArray and then `AsGeneric<T>()` it, which built a SECOND NDArray; they now construct
the typed array directly.

**Allocating from a view's Shape is safe (fixed 2026-10-03).** A `Shape` carries a view's strides, offset and buffer
size next to its dimensions, so the natural port `np.empty(x.Shape)` of NumPy's `np.empty(x.shape)` used to hand the
allocator the VIEW's addressing over a buffer of only `size` elements: `np.zeros(arange(20)["::4"].Shape)` read heap
memory, `np.polyval(p, a[::4])` (which allocates `np.zeros(x.Shape)`) crashed the CLR, and
`default_rng(42).integers(0, 100, x.Shape)` returned heap pointers. Every allocating entry point
(`UnmanagedStorage.Allocate(Shape, …)` behind `new NDArray(dtype, shape)` / `np.empty` / `np.zeros`, plus the `np.ones` and
`np.full` cores) now routes the shape through `Shape.ForFreshAllocation()` (`View/Shape.Allocation.cs`): a layout that
already describes a fresh buffer — offset 0, buffer size == element count, dense C/F or a KEEPORDER permutation — is
KEPT (the `copy(order='K')`, `NpyIterOutputShape` and `ConcatOutputShape` callers allocate exactly such shapes), and any
view layout (a step, an offset, a negative stride, a broadcast, a contiguous slice reporting its parent's buffer) is
re-laid out in C order, NumPy's answer for a dims-only request; the result is always writeable. A shape used ONLY for its
dimensions — NumPy's `size` tuple, every `np.random` sampler — goes through the stricter `Shape.AsClean()`, which also
re-lays out a dense F/permuted view (a sampler fills in memory order, so `x.T.Shape` would otherwise transpose the
draws) and returns its input unchanged when it already is the C layout. The WRAP constructors
`new NDArray(Array | IArraySlice, Shape, order)` (and the `NDArray<T>` twins) are NumPy's `np.ndarray(shape, buffer=…,
order=…)` and read the shape's DIMENSIONS only (`Shape.ForWrappedBuffer`, fixed 2026-10-05): C strides, or F for
`order='F'` at any rank, offset 0, writeable; `'A'`/`'K'` mean C and an invalid code is NumPy's `ValueError`
(`OrderResolver.ResolveBufferOrder`); and NumPy's size rule (2026-10-05): a smaller buffer is `TypeError: buffer is too
small for requested array`, a larger one is read from its start (a prefix sharing the caller's memory), the lone `-1`
takes every element. A view's Shape used to be installed as given (a stepped view read past the
buffer, a reversed one read backwards, a transposed one in F order, a broadcast one repeated row 0 read-only). Internal
code that must install a specific layout over a fresh buffer (`np.ones`'s KEEPORDER result) uses
`new UnmanagedStorage(IArraySlice, Shape)`, which trusts its layout like NumPy's C API. `Shape` is immutable: its
indexer has no setter (removed 2026-10-05; it wrote into a dimension array that shapes and arrays share), `a.shape`,
`Shape.Dimensions` and `Shape.Strides` return copies, and a caller's `long[]` is copied wherever it enters (2026-10-05:
`long[] s = {3, 4}; var a = np.zeros(s); s[0] = 100;` used to make `a.shape` `(100, 4)` over 12 elements; plan
`docs/plans/view-shape-open-bugs.md`, bug 2). Gates: `View/Shape.ForFreshAllocation.Tests.cs`,
`View/Shape.Immutability.Tests.cs`, `View/Shape.CallerArrays.Tests.cs`,
`Creation/FreshAllocation.ViewShape.Tests.cs`, `Creation/WrapConstructor.ViewShape.Tests.cs`,
`RandomSampling/RandomSizeViewShape.Test.cs`, and the oracle replay
`FuzzCorpusTests.RandomApiSizeLayouts` (every random overload taking a `Shape`, its size handed over as a view's layout
of the same dimensions, against NumPy).

**`NDArray<T>.reshape(Shape)` reads the DIMENSIONS of its argument (fixed 2026-10-10).** NumPy's reshape takes a shape
tuple, and the untyped `NDArray.reshape(Shape)` (`ReshapeCore`) always used the dimensions only. The typed overload
`NDArray<T>.reshape(Shape)` / `reshape(ref Shape)` relabelled its storage with the Shape AS GIVEN
(`UnmanagedStorage.Reshape` → `Shape.Reshape(Shape)`, which keeps the argument's strides):
- **The consequence.** A Shape taken from a view, a transposed or a broadcast array carried that array's strides over
  THIS array's buffer. `np.zeros(40, bool).AsGeneric<bool>().reshape(np.arange(80).reshape(5, 2, 8)[":, :, :4"].Shape)`
  had strides (16, 8, 1) over 40 elements: it read up to element 75, which is other heap memory (wrong values, and an
  access violation when the buffer ends where mapped memory ends). A comparison returns `NDArray<bool>`, so the natural
  port of `(a % 3 == 0).reshape(x.shape)` hit it. An existing test built its `where=` mask that way
  (`Indexing_KernelRoute_Tests.Where_NarrowRows_3D_LeadingAxisWalkedPerBlock`); its assertion read the same wrong mask
  on both sides, so it passed.
- **The fix.** The typed overload passes `newShape.AsClean()` on: a plain dimension list is kept as it is
  (`a.reshape(2, 3)`, `new Shape(2, 3)`: no extra allocation), any other layout is re-laid out in C order.
- **Not changed:** `reshape_unsafe(Shape)` (typed and untyped) still installs the Shape it is given; it is the
  unchecked relabel that broadcasting uses. The typed overloads also still COPY a non-contiguous source where the
  untyped route returns NumPy's strided view (`_attempt_nocopy_reshape`): a view / copy difference, not an addressing
  error.
- **Gate.** `Manipulation/TypedReshapeViewShapeTests.cs` (the Shapes of a stepped view, a column view, a reversed view,
  a transposed array and a broadcast array; write-through; NumPy's values).

## Slicing Syntax

```csharp
nd[":"]           // All elements
nd["1:5"]         // Elements 1-4 (stop exclusive)
nd["::2"]         // Every 2nd element
nd["-1"]          // Last element (reduces dimension)
nd["::-1"]        // Reversed
nd[":, 0"]        // All rows, first column
nd["..., -1"]     // Ellipsis fills dimensions
```

---

## Supported np.* APIs

Tested against NumPy 2.x.

### Array Creation
`arange`, `array`, `asanyarray`, `asarray`, `asarray_chkfinite`, `ascontiguousarray`, `asfortranarray`, `asmatrix`, `bmat`, `copy`, `empty`, `empty_like`, `eye`, `frombuffer`, `full`, `full_like`, `geomspace`, `identity`, `linspace`, `logspace`, `matrix`, `meshgrid`, `mgrid`, `ndarray`, `ogrid`, `ones`, `ones_like`, `require`, `tri`, `zeros`, `zeros_like`

**`np.ndarray(shape, dtype=float64, buffer=None, offset=0, strides=None, order='C')`** is NumPy's low-level
constructor (`ndarray.__new__`, `array_new` in `arrayobject.c`; plan `docs/plans/np-ndarray-parity.md`; probed against
2.4.2). Without a buffer it is `np.empty`'s allocation (uninitialized; C or F strides; an EMPTY result has all-zero
strides, as NumPy allocates it; explicit byte `strides` give an owning array under that layout, a stride of 0 stays
writeable). With a buffer it lays the array over the buffer's BYTES as `dtype` without copying: `offset` is a byte
offset, `strides` are BYTE strides, the shape `(-1,)` takes every whole element, a larger buffer is read from its
start. The checks run in NumPy's order (buffer, order, strides length, `PyArray_CheckStrides` with C's wrapping
arithmetic, `-1`/too-small, dimensions) with NumPy's texts. **Buffers** (`Backends/Unmanaged/NDBuffer.cs`, the port of
`PyArray_BufferConverter`): a C- or F-contiguous `NDArray` (else `ValueError: ndarray is not contiguous`) or its `.data`
memoryview (else `BufferError: memoryview: underlying buffer is not contiguous`) is BORROWED (the result counts its
references on the buffer array's block, its base collapses to the owner); a C# array of any rank whose elements hold no
references, `ArraySegment<byte>`/`Memory<byte>` (writable) and `ReadOnlyMemory<byte>` (read-only for good, NumPy's
`bytes`) are PINNED until the last array over them is released or collected (a failed call frees the pin at once);
`Half`/`char`/`decimal` export a read-only copy of their bytes (NumPy scalars); anything else is `TypeError: a
bytes-like object is required, not 'list'` with the Python type name. The result is ONE storage
(`UnmanagedStorage.OverBuffer`) over the buffer array's own slice when dtype and element grid match, else over a
forwarding alias of the occupied span (`ArraySlice.WrapShared(NPTypeCode, …)`). **Safer than NumPy:** after NumPy's
checks pass, an exact byte-range check refuses the four inputs NumPy lays outside the buffer (negative offset, `-1` with
a negative stride, an offset near 2^63, strides over an empty buffer). Other divergences (`[Misaligned]`): a stride that
splits an element is `NotSupportedException` (a stride that addresses no element is stored truncated), a non-native
dtype over a buffer decodes into a new array. The old signature's fourth positional `order` would bind the new `long
offset` (a `char` converts): an `[Obsolete(error: true)]` overload makes `np.ndarray(s, d, b, 'F')` a compile error.
**`flags.aligned` is computed from memory** (`UnmanagedStorage.IsDataAligned`: NumSharp's whole-element strides make the
storage's base address decide), so an odd-offset buffer and every view of it report unaligned, `setflags(align=True)`
raises NumPy's ValueError there, and `np.require(…, 'A')` copies. Perf (NPY/NS): over an `NDArray` 1.8–2.8×, over a
`byte[]` 1.0–1.4× (the GC pin + the base object); without a buffer it is `np.empty`'s allocator floor (0.1–0.9×, shared).
Gates: the `ndarray_ctor.jsonl` oracle tier (5,919 cases: values + a "layout" facet of flags and byte strides, zero
excuses) and `Creation/np.ndarray.Test.cs` (39). See `Creation/np.ndarray.cs`.

The `as*` conversion family mirrors NumPy: `asarray_chkfinite(a, dtype=None, order='K')` = `asarray` then raise `ValueError("array must not contain infs or NaNs")` if a **float-family** dtype (Half/Single/Double/Complex — NumPy's `typecodes['AllFloat']`; Decimal/int/bool skip the check) holds any inf/NaN, via a **fused single-pass NaN-poison SIMD reduction** (`Backends/Kernels/FiniteScan.cs`: `acc += v - v` — +0 for finite, absorbing-NaN for non-finite; AVX2 gather + reversed-contiguous fast path for strided/negative-stride views; ~2–27× NumPy contiguous, ≥1× strided). `require(a, dtype=None, requirements=None)` parses C/F/A/W/O/E flags (+aliases; single-string requirements iterate by char like NumPy, so `"F_CONTIGUOUS"` as one string raises), resolves an order and copies only if a remaining ALIGNED/WRITEABLE/OWNDATA flag is unsatisfied (ALIGNED fails after `setflags(align=False)` or over a caller's buffer at a misaligning byte offset; WRITEABLE for broadcast views; OWNDATA for views). `asmatrix(data, dtype=None)` is `matrix(data, dtype, copy=False)` and returns an **`NDMatrix`** (see `np.matrix` below): a new VIEW of an array (never the operand itself), 0-D→(1,1), 1-D→(1,N), 2-D unchanged, >2-D drops length-1 axes and must land on 2-D else `ValueError("shape too large to be a matrix.")`; also parses matrix literals (`"1 2; 3 4"`). See `Creation/np.{asarray_chkfinite,require,asmatrix}.cs`.

**`np.logspace` / `np.geomspace` — the log-scale linspace family** (NumPy 2.4.2 `numpy/_core/function_base.py`;
all probed against 2.4.2; gates `Creation/np.logspace.geomspace.Test.cs` (20) + the `creation` oracle tier —
28 logspace + 13 geomspace cases, 343/343 bit-exact). **Scalar `start`/`stop`/`base` (double), like NumSharp's
existing scalar `linspace`** — array-like `start`/`stop`/`base` is OUT of scope (the whole family is scalar-only;
that's a separate, larger feature). Full parameter parity otherwise: `num=50`, `endpoint=true`, `base=10.0`
(logspace only), `dtype=null`, `axis=0` (int + long `num` overloads). `axis` is a **no-op for scalar inputs** (the
output is always 1-D) but is still validated exactly like NumPy's trailing `moveaxis(y, 0, axis)`: only 0 and -1
are in range, else `AxisError("destination: axis {a} is out of bounds for array of dimension 1")` (verbatim,
reporting the original axis). `num<0` → `ValueError("Number of samples, {num}, must be non-negative.")`.
- **`logspace(start, stop, num, endpoint, base, dtype, axis)`** = `power(base, linspace(start, stop, num,
  endpoint))` then `.astype(dtype)`. The computation is ALWAYS float64 (scalar-double inputs) and only THEN cast —
  so an integer dtype **TRUNCATES toward zero** (unlike `linspace`, which FLOORS), and even `dtype=complex128` is
  float64-compute-then-cast (real + 0j), NOT a complex-domain computation. **BIT-EXACT vs NumPy** at every dtype:
  `Math.Pow`==`npy_pow` and the `start + i*step` interior commutes with NumPy's `arange*step + start` (win-amd64).
  A negative `base` with fractional exponents yields NaN (real power of a negative base), matching NumPy. Fused
  single-pass fill loop — no intermediate `linspace` array (`LogspaceCore`).
- **`geomspace(start, stop, num, endpoint, dtype, axis)`** — a geometric progression, endpoints given directly.
  Ported op-for-op from NumPy: `out_sign = sign(start)`, rotate `start`/`stop` onto the positive real axis,
  `logspace(log10(start), log10(stop), num, endpoint, base=10)`, overwrite the endpoints, `result *= out_sign`.
  The REAL path is **BIT-EXACT** (out_sign is ±1, so the rotate + final multiply are exact and the endpoints land
  the original `start`/`stop` byte-for-byte); decreasing, all-negative, and **mixed-sign** inputs (`geomspace(-1,1)
  → [-1, nan, nan, 1]`, log10 of the rotated-negative endpoint poisoning the interior) all fall out automatically.
  Zero endpoints → `ValueError("Geometric sequence cannot include zero")`, checked FIRST (before num, before axis —
  NumPy's order). A `Complex` start/stop **overload** and a real-input **`dtype=complex128`** both compute in the
  complex128 domain (the interior genuinely differs — `geomspace(1,8,4,dtype=complex)[1]==1.9999999999999998`, not
  the real path's exact 2.0 — so the domain is load-bearing); these compose NumSharp's NumPy-tuned complex power
  (`npy_cpow` port) + complex log10, so they are **ACCURATE within the documented ≤3-ULP complex-unary envelope but
  NOT byte-reproducible** (allclose, unit-test-pinned + EXCLUDED from the byte corpus — the `np.sinc` complex
  policy; verified worst relative error 1.0e-15 over the spiral/circle/dtype cases).
- **Perf (NPY/NS, Release, best-of-15, warm): ~parity at 1K (NDArray-construction floor), 1.2–1.3× at 100K–10M**
  (the single-pass fusion avoids NumPy's intermediate `linspace` passes). This is the **physical ceiling**: both
  are `Math.Pow`-bound (a bare `Math.Pow` loop of 1M measures 8.17 ms vs logspace's 8.29 ms — 98 % pure pow), and
  no bit-exact SIMD `pow` exists, so the 1.5× target is unreachable without abandoning parity — the same class as
  `arcsinh`/`float_power`-general/`gcd`. See `Creation/np.{logspace,geomspace}.cs`.

**`np.matrix(data, dtype=None, copy=True)` → `NDMatrix` — NumPy's `numpy.matrix`** (2026-10-09; port of
`numpy/matrixlib/defmatrix.py`; `src/NumSharp.Core/Matrixlib/`; probed against 2.4.2). The CLR type is `NDMatrix`
(the house `ND*` prefix), `public partial class NDMatrix : NDArray`; the NumPy names reach it: `np.matrix(...)`,
`np.asmatrix(...)` and `np.bmat(...)` return it, and `coverage/object_surfaces.py` maps `numpy.matrix` to
`NumSharp.NDMatrix` (80 of its 83 members available; `ctypes` / `dump` / `dumps` are missing as on ndarray).
- **Members and operators follow the STATIC type.** NumPy finds the subclass at run time; NumSharp binds the members
  and operators declared on `NDMatrix` at compile time. The same object through an `NDArray` reference is a plain
  2-D array to the operators, the indexers and the `NDArray` members: `a * a` is elementwise, `a[0]` drops an axis,
  `a.sum(0)` has N elements. Reason: every kernel and helper of the library is written against `NDArray`, and a
  run-time dispatch would change the meaning of that code for a caller who passed a matrix in.
  `NDMatrix.Clone()` HIDES `NDArray.Clone()` and does not override it for the same reason.
- **The `np.*` functions follow the OBJECT: the matrix boundary** (`Matrixlib/NDMatrixBoundary.cs`, 2026-10-10). A
  public function that receives an `NDMatrix` object through its `NDArray` parameter returns the class and the shape
  NumPy 2.4.2 returns for a `numpy.matrix` operand; the result is an `NDMatrix` typed `NDArray` (cast it for the
  matrix members, no copy).
  - **Rules** (one statement per public overload, 363 of them): `if (NDMatrixBoundary.Sees(a)) return
    NDMatrixBoundary.<Rule>((a, ...), static s => f(s.a, ...));`. `Wrapped` is `__array_wrap__` / subclass allocation
    (ufuncs, `np.transpose`, `np.concatenate`, `np.dot`, `np.linalg.inv`: a 1-d result becomes a 1 x N row, more than
    two long axes raise `shape too large to be a matrix.`, so `np.stack((m, m))` raises); `Reduced` is the matrix
    METHOD NumPy calls (`np.sum(m, 0)` is a 1 x N matrix, a scalar without an axis, and `keepdims: true` raises
    `TypeError: matrix.sum() got an unexpected keyword argument 'keepdims'`); `Plain` is `asarray` inside
    (`np.cross`, `np.einsum`, the whole `np.fft`, `np.linalg.norm`); `WrappedAll` / `WrappedPair` / `WrappedTriple` /
    `WrappedFirstTwoOfFour` are the functions with several results (`np.split`, `np.divmod`, `np.linalg.qr`: Q and R;
    `eig` / `eigh`: plain eigenvalues, matrix eigenvectors; `svd`: U and Vh; `lstsq`: x and the residuals); `WrappedIf`
    is `np.block`, whose class follows NumPy's ROUTE (repeated concatenation returns a matrix, the single-allocation
    route past 2*512*512 elements a new ndarray). Per-function facts NumPy's source decides: `np.linalg.solve` and
    `lstsq` take their class from `b`, `np.clip` from the array OR a bound, `np.iscomplex` is a matrix for a complex
    matrix and a plain array otherwise, `np.r_` / `np.c_` / `np.block` cannot give a matrix entry a third axis
    (`np.block([[[m]]])` raises "axis -3 is out of bounds for array of dimension 2").
  - **The boundary is the OUTERMOST public call.** A per-thread depth counter is raised by every rule around a
    second call of the function, and `Sees` is false while it is raised: nested public calls, `NDArray` members that
    are implemented by `np.*` functions and `NDMatrix` members compute with plain-array semantics. The state is
    packed in a tuple and handed to a `static` lambda (a capturing lambda would allocate at method entry on every
    call, plain arrays included).
  - **A function without a rule opens a region** (`using var plain = NDMatrixBoundary.PlainFor(a, ...)`, 203
    statements): its result is the plain array NumPy's `asarray`-based functions return, and the safe answer where
    NumPy's is not implemented. **A container stores the plain view** (`NDMatrixBoundary.PlainOf`): numpy.ma (the
    masked array's data and mask, `AsData`, `MaskOperand`) and the samplers' converted parameters
    (`RandomParam.Convert`), because a region cannot cover calls made after it closed.
  - **The contract is measured.** No method of the library may reach the matrix rule of a public function at depth
    zero. `FuzzCorpusTests.MatrixOperand_LibraryNeverReachesAMatrixRule` replays every corpus case that has a 2-D
    operand (311,725 cases: the ordinary tiers, the numpy.ma tiers, the index tiers) with those operands typed as
    matrices while `NDMatrixBoundary.AuditInternalCallers` is on, and requires the set of library callers to stay
    empty. It found 57 call paths when it was written (members, engine methods that call back up into `np.*`,
    functions without a rule); each is closed by a region or a rule. A new finding is closed the same way.
  - **Limits** (`[Misaligned]`, pinned in `Matrixlib/NDMatrixBoundary.Tests.cs`): a function whose C# return type is a
    typed array cannot return a matrix object (`np.isclose`, gap MO01; the typed `np.all` / `np.any` overloads,
    `np.nonzero` and `np.flatnonzero` are plain in NumPy too); `np.gradient` and `np.meshgrid` return plain arrays in
    their result structs (gaps MO06 / MO07, `OpenBugs.MatrixOperand.cs`); numpy.ma has no `masked_matrix` (NumPy's
    `ma.array(m).sum(axis=0)` is 1 x N); the `numpy.polynomial` package computes plain; a result is always a matrix
    VIEW of the plain result (no `flags.owndata` parity for the functions NumPy allocates as the subclass). NumPy
    ARTIFACTS, where matrix semantics leak into NumPy's own Python code, are not reproduced: gaps MO02 to MO05 of the
    tier, `np.linalg.vector_norm(m)` (an array of `|x|` in NumPy), `np.average(m, axis=1, weights=ndarray)` (an N x N
    matrix in NumPy), `np.apply_along_axis` when `func1d` depends on the slice's rank (NumPy hands it a 1 x N matrix).
  - **Perf** (two pinned P-cores, best of 15, a loaded host: +-3 % between runs). Plain arrays pay a type test per
    call: 44 cells (22 functions at 4 and 1,024 elements) read new / old geomean 1.02, inside the noise but for
    `np.asarray` (+0.7 ns) and `np.diagonal` (+10 ns, two regions). A matrix operand pays the rule, about 270 to 350
    ns a call (the scope that reclaims the hidden plain result, the second call, the matrix view), the same as the
    `NDMatrix` operators. Against NumPy's own matrix calls (32 cells): geomean 3.0x, min 1.1x (`m.sum(0)` at 2 x 2
    and the managed `np.dot` at 32 x 32), the `np.*` cells 2.2x to 12x.
- **A 0-d `NDMatrix` is a scalar, not a matrix.** The 15 C# scalar types convert implicitly to a 0-d `NDMatrix`, so
  `m * 2`, `m + 1.5` and `m[mask] = 0` compile and bind the `(NDMatrix, NDMatrix)` operators. `*` with a 0-d
  operand scales. Members whose NumPy result is always a scalar return a 0-d `NDArray` (`m[0, 0]`, `m.sum()`);
  the general indexers return a 0-d `NDMatrix` for a full integer index.
- **An explicit 0-d ARRAY is not a scalar to `*` (2026-10-10).** NumPy's `matrix.__mul__` tests
  `isinstance(other, ndarray)` before `isscalar(other)`, so `m * np.array(2)` is `np.dot(m, asmatrix(np.array(2)))`,
  the product with a 1 × 1 matrix: `ValueError: shapes (2,3) and (1,1) not aligned` unless `m` has one column. There
  is no mathematics in that error, only the order of two type tests. NumSharp has one 0-d type for both roles, so
  `np.array<T>(T)` and `np.asarray<T>(T)` set a per-object mark, `NDArray.IsExplicitZeroDimArray`
  (`Backends/NDArray.ZeroDimArray.cs`; `np.array(nd)` / `np.asarray(nd)` of a marked array carry it over), and
  `NDMatrix.Product` reads it: marked on the right → the 1 × 1 product; marked on the left → `np.dot(a0, m)`, the
  scalar route (a new owner, `0 + s * x`). **Unmarked means scalar**: every computed 0-d value scales as before
  (`m * m.sum()`, `m * np.dot(v, v)`, `m * np.linalg.norm(a)`, `m * a[0, 0]`, `m * NDArray.Scalar(2.0)`). A flags
  rule cannot do this: 13 of 39 probed scalar results (`np.dot(v, v)`, `np.linalg.det`, `np.linalg.norm`, `z + z`)
  come back writeable and owning, exactly like `np.array(2)`. **Rule for library code:** build a computed scalar
  with `NDArray.Scalar(x)`, never with `np.array(x)` (eight sites in `np.all` / `np.any` / the logical reductions /
  `np.histogram` were moved). Still divergent (`[Misaligned]`): a 0-d array built another way (`np.zeros(())`,
  `x.reshape(())`, `np.squeeze(x)`) scales, where NumPy raises; NumSharp's own kernels allocate their 0-d results
  with those functions, so marking there would make `m * np.linalg.norm(a)` raise.
- **Result forms** (`NDMatrix.cs`): `ViewOf(a)` is `a.view(matrix)` (never an owner); `From(r)` is a result NumPy
  allocates as a matrix (copy, astype, dot, take: the matrix takes the result's storage over and owns when it did);
  `FromWrapped(r)` is a ufunc result `__array_wrap__` returns as a view (operators, reductions, cumsum, clip: never
  an owner); `FromIndexing` is `matrix.__getitem__`'s rule. All apply the finalize rule `Shape.AsMatrix()`
  (`View/Shape.Matrix.cs`): 0-d → `(1,1)`, 1-d → `(1,N)`, more than two axes longer than 1 →
  `ValueError("shape too large to be a matrix.")`.
- **Indexing** (`NDMatrix.Indexing.cs`): a 1-d result is a COLUMN when `len(index) > 1 and isscalar(index[1])`
  (entry #1 by position: `m[..., 0]` is a column, a 0-d array is not a scalar), else a row; other ranks are
  returned as they are (`m[newaxis]` is a 3-d matrix object, as in NumPy). An index array or mask gives a matrix
  that VIEWS the gathered array (NumPy's `wrap_out_array`); a 0-d integer array index gives an owning copy
  (`HAS_SCALAR_ARRAY`). Index errors carry NumPy's texts (`ValidateBasic`).
- **Assignment**: matrix has no `__setitem__`, but ndarray's assignment gets the view of a subclass through its own
  `__getitem__` for an index of integers and slices. So the value is broadcast into the row or COLUMN:
  `m[:, 0] = [7, 8, 9]` raises `could not broadcast input array from shape (3,) into shape (3,1)`. One element
  stores through the dtype's setitem (an array value is refused; bool takes a one-element array's truth value;
  complex128 raises TypeError). The typed indexers take an `NDMatrix` or a number; `__setitem__(index, object)`
  takes any value.
- **`*` and `**`** (`NDMatrix.Operators.cs`): `m * x` is `np.dot(m, asmatrix(x))` (a 1-d `x` becomes a ROW), `x * m`
  is `np.dot(x, m)`; each binary operator has five forms (matrix/matrix, matrix/array, array/matrix,
  matrix/object, object/matrix). Scaling is `np.dot`'s scalar route: the dtype is `promote_types` of both
  operands (not the weak-scalar ufunc rule); for float32 / float64 / complex128 the result is a new C-contiguous
  owner and each element is `0 + scalar * x`, so a product of -0.0 is +0.0 (`?axpy` into zeros), except a
  ONE-element result (the direct product, -0.0 kept); every other dtype is the multiply ufunc, wrapped. The float
  scaling has THREE routes with one result (`Scale`): a C-contiguous float32 / float64 operand of the loop dtype
  (`m * 2.0`, the common case) runs ONE direct pass over raw memory (`ScaleContiguous`, the polynomial package's
  `PolyAffineDouble` / `PolyAffineSingle`: multiply then add, never an FMA); another operand of the loop dtype
  (`m.T`, a stepped view, a complex matrix) runs one fused `np.evaluate` pass from the measured crossover on
  (`FusedScaleMinElements` 4,096 for float views, `FusedScaleMinElementsComplex` 512); the rest runs the multiply
  ufunc, then adds the zero in place with the same raw kernel. All three give the bytes of
  `np.add(np.multiply(x, s, dtype: loop), 0)`, except the payload of a NaN that meets a NaN scalar (NumPy's comes
  from the BLAS `?axpy` kernel, which no route models). complex128 has no direct pass: that kernel multiplies in
  the other operand order, which can round the last bit differently. A plain
  0-d `NDArray` on the LEFT stands for a NumPy scalar and is the elementwise ufunc (`m.sum() * m`). C# has no
  `**`: `__pow__(int | object)`, `__ipow__`, `__rpow__`; `__imul__`. Reflected comparisons evaluate with the
  matrix as the first ufunc operand (the order of NumPy's broadcast error text).
- **Reductions** (`NDMatrix.Reductions.cs`): `sum / prod / mean / var / std / any / all / max / min` keep two axes
  (`keepdims=True` then `_collapse`); `argmax / argmin / ptp` use `_align` (axis 0 → row, 1 → column, a valid
  negative axis → `ValueError("unsupported axis")`). `(int axis, ...)` returns `NDMatrix`, `(int? axis = null, ...)`
  returns `NDArray` (the 0-d scalar without an axis). A matrix reduction is always NumPy's ARRAY form: a float16
  mean rounds twice, and `m.std(dtype: np.int64)` raises the sqrt cast error where `np.std` of the plain array
  truncates. With `out=` the loop is `promote_types(out.dtype, operand dtype)` without the small-integer widening
  (`reducelike_promote_and_resolve`), stored with an unsafe cast: an int8 sum into a bool out is the truth value
  of the int8 sum. `var` / `std` hand `out` to their LAST sum only (`DefaultEngine.VarStdNumPy(outDtype:,
  sumOnly:)`).
- **Constructor** (`NDMatrix.Construction.cs`): the four branches of `matrix.__new__` in NumPy's order. A matrix
  literal (`"1 2; 3 4"`, `"[1.5, 2; 3 4j]"`) is `_convert_from_string` plus a port of `ast.literal_eval` for one
  token (four number bases, `_`, floats, imaginary numbers, signs, parentheses, `True` / `False`; `SyntaxError`
  and `malformed node` errors). A sequence goes through `PolySequence`; with a `dtype` each element is stored by
  `PyArray_Pack` (`ReductionInitial.TryPack`): a Python int out of range raises OverflowError, a complex number
  into a real dtype raises TypeError.
- **Library fixes this needed** (they change `NDArray` too): `NDArray.PrepareAssignedValue`
  (`Selection/NDArray.Indexing.AssignedValue.cs`) is NumPy's MapIter value rule for every index-array assignment:
  a value with more axes than the selection loses its leading axes (a `(1, N)` matrix row fits an `(N,)`
  selection; it threw before) and a value that does not fit raises `shape mismatch: value array of shape ...
  could not be broadcast to indexing result of shape ...` before anything is written. A boolean mask of the
  array's own shape follows `array_assign_boolean_subscript` (a value with two or more axes is TypeError, a wrong
  length is `NumPy boolean array indexing assignment cannot assign 3 input values to the 6 output values where
  the mask is true`). `np.dot(1-D, 2-D)` and `np.matmul` with a 1-D operand return an OWNER (the inserted axis is
  dropped in place); `np.dot(N-D, 1-D)` reports a mismatch in `np.dot`'s words; `NDArray.Equals(object)` no
  longer leaks its comparison array; `np.asmatrix` returns a new view where it returned the operand (gap VF15
  retired).
- **Divergences**: the static typing above; a C# literal keeps its C# type (`matrix(int8) * 2` is int32 where
  NumPy's Python int gives int64); no tuple axes; the address in `ast` error texts is zero; complex128 scaling by
  a complex scalar is the ufunc product, which can differ from OpenBLAS `zaxpy` in the last bit.
- **Gates**: oracle tier `matrix.jsonl` (`test/oracle/gen_matrix_oracle.py`, 26,789 cases over 87 op keys, ZERO
  excuses, valueclass `matrix`): values, the result CLASS (`expected.cls`: matrix / ndarray / scalar), the memory
  facet (flags, strides, view / copy / identity against each operand) and NumPy's error type and text, over 14
  source layouts and the dtypes; operands rebuild as `NDMatrix` (`"matrix": true`). The tier is replayed three
  times, once per scaling route: `Matrix` (production: the counter proves the direct pass ran),
  `Matrix_FusedScaling` (direct pass off, fused pass forced) and `Matrix_TwoPassScaling` (both off), through the
  hooks `NDMatrix.DisableDirectScale` / `FusedScaleMinElementsTestOverride`. 144 cases are carved under 9 gap
  ids into `gaps/matrix_gaps.jsonl` (172 under MX01–MX11 until the strings rebase of 2026-10-12 fixed MX06 and
  MX11): every one is an `NDArray` member or `np.*` function below the class (pins in `OpenBugs.Matrix.cs`). Unit tests `Matrixlib/NDMatrix.Tests.cs` (42, NumPy's `test_defmatrix.py`
  ported plus the design pins; `Scaling_EveryRoute_GivesTheUfuncBits` holds the three routes to the ufunc
  reference at 70 × 70, past every threshold). The leak completeness gate audits every declared member
  (`LeakCatalogue.Matrix.cs`).
- **Perf** (NPY/NS, two pinned P-cores, 97 cells: 33 operations × 4 / 64 / 512 square; every C# cell warmed in
  four rounds): geomean 2.26×; construction 2.6–18×, `m[i]` / `m[:, j]` 5×, `m + m` 1.8–5×, `m * 2.0`
  2.6× / 1.9× / 25×, `m.sum()` 2.9–6.2×, `m * m` at 4×4 6.6×, `m.I` at 4×4 8.6×, literals 8–13×. The 31 cells
  under 1.5× are costs of the array library, not of the class: two index arrays (`m[rows, cols]` 0.33–0.69×, its
  assignment 0.18–0.27×: the fancy delegate route); `m > scalar` at 64×64 and 512×512 (1.2× and 1.0× since
  2026-10-10, were 0.52× and 0.25×: the next paragraph); products at 64×64
  without the OpenBLAS backend (`m * m` 0.58×, `m ** 3` 0.50×, `m.I` 1.12×); axis reductions (`m.sum(0)`
  1.1–1.4×, `m.mean(0)` and `m.argmax(1)` at 512 1.0–1.1×, `m.cumsum(0)` 1.2–1.3×); `m[mask]` (0.84× at 512);
  and members at the NDArray allocation floor (`m.copy()` 1.1–1.3×, `m[i, j]` and `m.A` 1.4–1.5×).
  **A C# literal operand costs 0.2–1.2 µs more than a held 0-d operand** (`m * 2.0` at 256 elements: 1.5 µs
  against 0.58 with `NDMatrix two = 2.0` held): the conversion's 0-d temporary is left to the finalizer, the same
  cost `a * 2.0` has on a plain array (plan X26, open). Measured over one second of calls (the mean, not
  best-of-rounds, which reads this cost 0.2–0.9 µs low; two runs): `(a > 0.75)` at 4×4 takes 1.7 µs against 0.47
  with a held operand, and a dropped `NDArray s = 0.75` alone 1.13 µs against 0.32–0.34 when it is disposed (a
  dropped object with an empty finalizer: 0.18–0.25 µs). The cost is the garbage collector, not a lock: with the
  dropped scalar the collections pause the thread for 0.45 s of each second (26 collections of ~17 ms, each one
  generation 1, because every finalizable temporary is promoted with what it references), against 0.08–0.11 s
  with a held or disposed operand (~1.4 ms a collection); `Monitor.LockContentionCount` moves by 0–11. The
  scalar pool runs dry meanwhile (~25,000 extra slots a second), which is a symptom. A scope around the call
  removes the pauses but not the scalar: `m > 0.75` at 64×64 takes 1.86–2.27 µs in an `NDScope`, 2.30–2.43
  without, 1.38–1.45 with a held operand.
  **`m > scalar` runs the vectorized array-against-scalar comparison kernels (2026-10-10).** The two kernel files
  of branch `strings`' commit `d0a6f4c8` were brought to journey4 unchanged
  (`Direct/DirectILKernelGenerator.Comparison.cs`, `.Unary.Predicate.cs`, and their test
  `Backends/Kernels/ComparisonBroadcastKernelTests.cs`: the same bytes on both branches, so the merge is clean).
  - **What they do.** `array op scalar` / `scalar op array` (`SimdScalarRight` / `Left`) compare a vector per step
    with the scalar broadcast once, when the array already has the comparison dtype
    (`EmitComparisonBroadcastSimdLoop`; they were scalar loops: 0.7 ns an element for float64, 0.36 for the
    integers, 1.4 for bool). Four masks are stored packed on AVX2 hosts (`EmitMasksToBools`, NumPy's
    `npyv_pack_b8_*`).
  - **The loop bound they also fix (silent memory corruption on journey4 until now).** The comparison `SimdFull`
    kernel and the contiguous predicate kernel (isnan / isinf / isfinite / isposinf / isneginf / signbit) ran one
    body too many when `n % step == step - 1`: one element read past the inputs and one bool written past the
    result (`a == b` over 31 bool elements, `np.isnan` over 7 float32; an `out=` view's next parent slot). The
    three sentinel tests of the test file fail on the old kernels.
  - **What stayed on `strings` until the rebase (2026-10-12).** The NEP 50 weak-scalar pack of the same commit
    (plan X24 / X2 / X21, a breaking change, with its `weak_scalar` tier) arrived when `strings` was rebased onto
    journey4: a Python-scalar 0-d operand is packed into a float16 / float32 array's dtype, and an in-range Python
    int into an integer array's dtype, before the comparison (`DefaultEngine.CompareOp.WeakScalar.cs`), so those
    literals now reach the array's own vector loop (not re-measured). Measured on journey4 BEFORE the rebase, when a
    literal that promotes the comparison stayed in the scalar loop (100K elements): `float32 > 50.5` 72 µs against
    7.9 for `50.5f`, `uint8 > 50` 39 µs against 3.9 for `(byte)50`, `int16 > 50` 37 against 4.7 for `(short)50`,
    `int32 > 50L` 39. A literal that does not promote took the vector loop already: an `int`, `float` or `double`
    against float64, an `int` against int32 / int64.
  - **Perf (NPY/NS; before).** `m > 0.75` 2.0× / 1.2× / 1.0× at 4 / 64 / 512 square (1.39× / 0.52× / 0.25×); a
    float64 array against a held 0-d operand 2.0× / 1.4× / 1.0× (1.9× / 0.40× / 0.22×); every SIMD dtype against
    a 0-d operand of its own dtype 1.9–2.4× at 1K, 0.86–1.9× at 100K, 1.4–4.0× at 10M (0.03–1.6× before). A
    re-run of the 97 cells reads geomean 2.35× with 29 cells under 1.5×.
  - **Two floors hold the 64 and 512 cells under 1.5×.** (1) At 64×64 the literal's 0-d temporary is 0.9 µs of
    the 2.1 µs call (`m > held` is 2.1×): plan X26. (2) The kernel already runs at the cost of READING its input:
    called directly it takes 0.36 µs for 4,096 float64 elements and 18.5 µs for 196,608, and a loop that only
    loads and compares takes 0.37 and 17.7. A 512×512 float64 matrix is 2 MB, the size of this CPU's L2: the
    read-only loop takes 30 µs there, the kernel 39, NumPy 40–48. So 1.5× is not reachable at that size.
  - **Finding (acted on the same day, see "The pool aligns its buffers" under Comparison & Logic): buffer alignment
    was a 20–35 % lottery for every SIMD kernel.** A buffer whose address
    is 16 mod 32 makes every second 32-byte load cross a cache line. Kernel-only at 100K elements: float32 4.8 µs
    aligned and 6.4 µs not, float64 9.3 and 12.5, int32 6.0 and 8.2; for 1- and 2-byte lanes the RESULT's
    alignment costs as much (int16 at 262,144: 9.0 and 12.8 µs). Pooled buffers are 16-byte aligned, so each
    array wins or loses by its address. Through the public operator, the same 100K float64 values take 10.1 µs
    in a view that starts on a 32-byte boundary and 13.3 µs in a view 16 bytes later (float32: 5.8 and 7.3). A
    64-byte aligned allocation would remove it library-wide: the pool does that now for buffers of one page or more.
  **Benchmark trap:** a harness that warms a cell once read 4.3 µs for a call that takes 1.4 at tier 1, and that
  number first put the fused threshold 4× too high. Warm every route twice with a pause, and time two routes in
  one process with interleaved rounds.
  See `Matrixlib/NDMatrix{,.Construction,.Indexing,.Operators,.Reductions,.Methods}.cs`, `Matrixlib/np.matrix.cs`.

### Shape Manipulation
`append`, `array_split`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `block`, `c_`, `column_stack`, `concat`, `concatenate`, `delete`, `dsplit`, `dstack`, `expand_dims`, `flatten`, `flip`, `fliplr`, `flipud`, `hsplit`, `hstack`, `insert`, `intersect1d`, `matrix_transpose`, `moveaxis`, `pad`, `permute_dims`, `r_`, `ravel`, `repeat`, `reshape`, `resize`, `roll`, `rollaxis`, `rot90`, `setdiff1d`, `setxor1d`, `split`, `squeeze`, `stack`, `swapaxes`, `tile`, `transpose`, `trim_zeros`, `union1d`, `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`, `unstack`, `vsplit`, `vstack`

**Set routines** `intersect1d`, `union1d`, `setxor1d`, `setdiff1d` (NumPy `_arraysetops_impl.py`, sorted/unique set
operations over flattened inputs; all probed against 2.4.2; gate `Manipulation/np.setops.Test.cs` (28) + the
`groupa` fuzz tier) are pure compositions of the existing `unique`/`sort`/`argsort`/`concatenate` machinery, ported
line-for-line from NumPy so their promotion, NaN handling and edge cases match by construction. `union1d(ar1, ar2)`
= `unique(concatenate(ravel(ar1), ravel(ar2)))` — unique sorted union in the promoted dtype. `intersect1d` returns
the sorted common values through a **3-overload split** that reproduces NumPy's bare-array-vs-tuple return
polymorphism: `intersect1d(ar1, ar2)` and `intersect1d(ar1, ar2, assume_unique)` are the bare-return forms
(→`NDArray`), while `intersect1d(ar1, ar2, assume_unique=false, return_indices=false)`→`NDArray[]{values, comm1,
comm2}` carries BOTH keywords optional so `np.intersect1d(a, b, return_indices: true)` (assume_unique omitted) ports
verbatim — the bare overloads have no `return_indices` parameter so they can't capture it, and C# betterness resolves
`intersect1d(a,b[,au])` to the bare `NDArray` form (the multi overload's extra optional loses), so the split is
unambiguous (empirically verified). The tuple path uses a **stable** argsort so the first occurrence in each input
wins (NumSharp's argsort is stable, verified). `setxor1d` returns
values in exactly one input; `setdiff1d(ar1, ar2, assume_unique=false)` returns `ar1`'s values absent from `ar2`
(built on `isin(..., invert=true)`), preserving `ar1`'s dtype and — under `assume_unique` — its duplicates. Empty
inputs collapse to NumPy's float64 empty (from the empty concatenate). NaN never matches itself, so a shared NaN is
excluded from `intersect1d` but each NaN survives `setxor1d`/`union1d`/`setdiff1d` as distinct (matching NumPy). A
surviving **float32/float64** NaN is **canonicalized to NumPy's positive quiet NaN** (`0x7ff8000000000000` /
`0x7fc00000`): NumPy's SIMD float sort (which `unique`/`sort` drive) rewrites EVERY NaN — negative-signed,
signalling, or payload-bearing — to that single value, while **float16 and complex128 NaNs are left exactly as-is**
(probed 2.4.2). NumSharp's `sort` writes that NaN too (since 2026-10-05: `AxisSort.SortLineF32/F64`; it wrote .NET's
negative NaN before, which also flipped `{p}fromroots`' NaN sign), but `unique` preserves payloads, so
`union1d`/`setxor1d`/`setdiff1d` run a guarded NaN-rewrite pass on a float32/float64 result — `np.where(np.isnan(r),
canonical, r)`, entered only when `np.any(np.isnan(r))` (≈0.3 ms on a 490 K result, skipped when no NaN survives) —
to stay bit-identical (`CanonicalizeSetOpNaN` in `np.setops.cs`; verified across neg/payload/multi-NaN × f16/f32/f64
+ complex against NumPy 2.4.2). `isin` (bool result) and `intersect1d` (NaN never shared) need no pass.
**Perf (NPY/NS, Release, best-of-21, warm):** **2.2–13.5×** — NumSharp's `unique`/`sort` outrun NumPy's at every
measured size (the wins widen at 1M, where NumPy's set-ops cost 60–400 ms); the NaN pass is O(result) and does not
move these numbers. See `Manipulation/np.{setops,union1d,intersect1d,setxor1d,setdiff1d}.cs`.

**`np.static unique` returns `np.UniqueResult`** (the shared core the set routines reuse) — NumPy's bare-array-OR-tuple
return expressed as ONE type, because C# cannot pick a return type from a runtime flag. **ONE** static overload
`unique(ar, bool return_index=false, bool return_inverse=false, bool return_counts=false, int? axis=null, bool
equal_nan=true, bool sorted=true)`→`UniqueResult` makes **every** NumPy call shape port verbatim — including the two
that used to need a C# idiom: `np.unique(ar, return_counts: true)` and `np.unique(ar, return_inverse: true)`
(**sole keyword, no `return_index`**), plus `np.unique(ar)`, `np.unique(ar, axis: 0)`, `np.unique(ar, equal_nan:
false)`. `UniqueResult` is a `readonly struct` (nested `np.UniqueResult`) that stands in for both NumPy shapes: it
converts **implicitly to `NDArray`** (the bare form — yields `values`, so `NDArray u = np.unique(ar)` and every
argument-passing/assignment site is unchanged) and **implicitly to `NDArray[]`** (the tuple form — the present
outputs in NumPy field order `[values, index?, inverse?, counts?]`); it indexes (`[k]` = k-th present output, matching
the old `NDArray[]` return), `Deconstruct`s (`var (values, counts) = np.unique(ar, return_counts: true)`), and exposes
named `values`/`indices`/`inverse_indices`/`counts` (**case-identical with NumPy's namedtuple fields** —
NumSharp's whole NumPy surface is lowercase, e.g. `nd.shape`/`nd.size`/`nd.dtype`; the last three are `null`
when not requested). **BREAKING (accepted):**
`np.unique(ar)` now returns `UniqueResult`, not a bare `NDArray` — so `np.unique(ar)[k]` selects the k-th OUTPUT (not
the k-th unique value → use `.values[k]`). To keep the ~230 existing call-sites working unchanged, the struct
**forwards the common `NDArray` surface to `values`** (`size`/`shape`/`Shape`/`ndim`/`dtype`/`typecode`/`array_equal`/
`GetAtIndex`/`GetAtIndex<T>`/`GetDouble`/`GetSingle`/`GetBoolean`/`GetInt32`) — the value the caller means in the
no-flags case — and the test `Should()` extension gained a `UniqueResult` overload (`FluentExtension.cs`). The
**instance** `NDArray.unique(int? axis=null, …)`→`NDArray` and `NDArray.unique(bool return_index, …)`→`NDArray[]`
are UNCHANGED (all internal consumers — `NDArray.unique.Hash.cs`, `np.unique_values.cs` — stay on them); only the
static `np.unique` was lifted to `UniqueResult`, which wraps the instance `NDArray[]`. The other three sibling
result structs (`UniqueAllResult`/`UniqueCountsResult`/`UniqueInverseResult`) are always tuples, so they carry no
`values`-forwarding surface — the asymmetry is deliberate (only `np.unique` has a bare-array case). Gate:
`Manipulation/np.unique.AxisEdgeCases.Test.cs` (`UniqueResult_*` + `BareReturn_*` pins); the whole test suite compiles
and passes against the new return type (differential-verified: `np.unique(x, return_counts=True)` bit-identical to
NumPy 2.4.2). See `Manipulation/np.unique.cs`.

**Array-API unique family** `unique_values`, `unique_counts`, `unique_inverse`, `unique_all` (NumPy 2.x
`_arraysetops_impl.py`; probed against 2.4.2; gate `Manipulation/np.unique_values.Test.cs` (19) + a 102-case
in-process differential across all 13 NumPy dtypes × layouts × NaN/empty/0-d — 102/102 bit-exact). All four are thin
wrappers over the existing `unique` machinery with **`equal_nan=False`** (each NaN is a DISTINCT value — the whole
point of these variants vs `np.unique`'s `equal_nan=True` default) and `axis=None` (flattened). NumPy returns
namedtuples; C# stands in with named-field `readonly struct`s whose fields are **case-identical with NumPy's**
(`UniqueAllResult{values,indices,inverse_indices,counts}`, `UniqueCountsResult{values,counts}`,
`UniqueInverseResult{values,inverse_indices}` — lowercase/snake_case, matching NumPy 2.4.2's `_fields`), each with implicit `NDArray[]`
conversion + `Deconstruct` + indexer (the `np.meshgrid` house shape). `unique_values` returns a bare `NDArray`.
`values` preserves the input dtype; `indices`/`inverse_indices`/`counts` are int64 (NumPy intp). `inverse_indices` is
reshaped to the ORIGINAL input shape (NumPy 2.0: reconstruct via `np.take(values, inverse_indices)`) — the sort path
already yields that for ndim ≥ 1, and `ReshapeInverseToInput` restores the `()` scalar shape a **0-d** input needs
(the flat path returns `(1,)` there — a latent `np.unique` 0-d bug the wrappers route around). **ONE deliberate
divergence, `unique_values` only (`[Misaligned]`):** NumPy's `_unique_hash` (`unique.cpp`) dedups **integer and
complex** dtypes through a `std::unordered_set` under `sorted=False`, returning them in platform/compiler-specific
hash-bucket order (neither sorted nor first-occurrence — e.g. `[0,8,16,24,1,9,17]` → `[24,16,8,0,17,9,1]` on
win-amd64, not reproducible in C#); NumSharp returns them SORTED (identical AS A SET, deterministic, portable, and
consistent with the `values` field of the other three). **Plain float32/64 are NOT in NumPy's hash map**, so they
fall to NumPy's own sort path and are already sorted — NumSharp's float `unique_values` is thus bit-exact with NumPy,
not divergent. **Hash fast path** (`NDArray.unique.Hash.cs`): the 10 integer-family dtypes (bool/byte/sbyte/int16/
uint16/int32/uint32/int64/uint64/char) take a purpose-built open-addressing table (byte-size-keyed generic hash,
load-factor-0.5 growth, **splitmix64 finalizer**) then a sort of the small unique set — mirroring NumPy's own
algorithm SELECTION (NumPy hashes int + complex, both in its `_unique_hash` map, and sorts floats). **Complex** joins
the hash path (NumPy's scalar lexicographic complex sort is slow, so it is a clean win, no all-unique regression);
**float/half/double/decimal keep the sort path**. Float hashing was measured end-to-end and REJECTED: it helps
duplicated data but regresses ~2× on all-unique input (insert N + scalar-sort N), and a cheap cardinality sample
provably cannot distinguish "100K-unique-in-1M" (hash helps) from all-unique (hash hurts). The **splitmix64
finalizer is load-bearing** — a single Fibonacci multiply then a low-bits table index clustered integer-valued
double bit patterns (`1.0/2.0/4.0…`, discriminating bits high, low bits zero) into one bucket at ~500 probes/element;
splitmix avalanches every bit down and is unchanged for integers. **Perf (NPY/NS, Release, best-of-rounds, warm):**
`unique_all` **2.3–7.6×** at every measured dtype/size/cardinality (the flagship "give me everything" call, where
NumPy pays for a mergesort argsort + cumsum + nonzero + diff); integer `unique_values` **1.3–5.4×**, integer
`unique_counts` **1.0–1.5×** at ≤1% cardinality; **complex `unique_values`/`unique_counts` 3.0–6.7×**; `unique_inverse`
**0.7–1.4×**. The remaining sub-1.5× cells — high-cardinality `unique_counts`, `unique_inverse`, and ALL float
`unique_values`/`unique_counts` (0.1–0.3×) — are bounded by NumPy's SIMD `vqsort` (`x86-simd-sort`), which NumSharp's
scalar sort core lacks; closing them is a shared-sort-core change, not a property of these wrappers. Full design +
the measured "why the axis comparator stays scalar / why floats stay sorted / why splitmix" decisions:
**`docs/UNIQUE_DESIGN.md`**. See `Manipulation/np.unique_values.cs` + `Manipulation/NDArray.unique.Hash.cs`.

**`np.unique` full parameter parity** — `unique(ar, return_index=false, return_inverse=false,
return_counts=false, axis=null, equal_nan=true, sorted=true)`, all six probed against 2.4.2 (gates
`Manipulation/np.unique.AxisEdgeCases.Test.cs` (9) + `NDArray.unique.Test.cs`, backed by a 450-case
in-process axis differential — 13 dtypes × axis{None,0,1,2,-1} × every return-flag combo × layouts
{C,F,transposed,reversed,sliced} × NaN/-0.0/complex/empty, 448 bit-exact + 2 excused f16). The **`sorted`**
parameter (NumPy 2.3) is accepted for API parity but is a **no-op — NumSharp always returns sorted output**:
NumPy's `sorted=False` hash-iteration order (integer/complex, values-only) is platform-specific and not
reproducible in C# (same divergence class as `unique_values`; identical as a set). An out-of-range axis raises
the house **`AxisError`** ("axis {a} is out of bounds for array of dimension {n}", NumPy-verbatim), reporting
the ORIGINAL axis. **The axis path compares each sub-array as a NumPy structured scalar** (`_arraysetops_impl`
consolidates the axis into a `('f0',…)` void dtype), which has two consequences ported exactly and easy to get
wrong: **`equal_nan` has NO EFFECT on the axis path** (the void dtype's kind isn't in `_unique1d`'s `"cfmM"`
guard, so its NaN-collapsing branch is skipped) — every NaN sub-array is DISTINCT (`nan != nan`), and a
**signed-zero sub-array collapses** (`-0.0 == +0.0`, the surviving zero's sign following input order). The slab
equality therefore uses IEEE `==` (not `.Equals`, which had `nan.Equals(nan)==true` — collapsing NaN rows — and
`(-0.0).Equals(0.0)==false` — splitting signed zeros, both wrong); the slab ORDER uses IEEE `<`/`>` (not
`CompareTo`, so `-0.0`/`+0.0` compare equal) with a stable tie-break by original index (reproducing NumPy's
mergesort argsort, so identical NaN slabs keep input order). **One deliberate divergence (`[Misaligned]`):** for
**float16 + axis + NaN**, NumPy 2.4.2's structured/void sort mis-orders NaN to the FRONT — inconsistent with its
OWN 1-D float16 sort and its float32/64 structured sorts (all NaN-last); NumSharp keeps the consistent, correct
NaN-to-end ordering across every float width (the unique-row set and counts are identical, only NaN position
differs). See `Manipulation/NDArray.unique.{cs,Kwargs.cs}`.

`flip`/`fliplr`/`flipud` return **O(ndim) views** (stride negation + base-offset shift via `Storage.Alias`,
the Transpose pattern — no slice resolution, no data movement), bit-identical to the `m[..., ::-1, ...]`
slice path for every layout incl. F-contiguous/transposed/stepped/broadcast (flip of a read-only broadcast
stays read-only). `flip(m, axis: null)` flips ALL axes; `flip(m, int[])` is the tuple form (empty array
= flip nothing; the axes are normalized NumPy-style: full `AxisError` pass — reporting the original
negative axis — THEN `ValueError("repeated axis")`, so `(0,0,5)` raises AxisError). 0-d: `flip(m)` returns
the scalar (NumPy's `m[()]`), `flip(m, 0)` raises AxisError. `fliplr` requires ndim≥2 / `flipud` ndim≥1
(`ValueError("Input must be >= 2-d." / ">= 1-d.")` verbatim). See `Manipulation/np.flip.cs`.

`rot90(m, k=1, axes=(0,1))` rotates in the plane of two axes by `90°·k` (counterclockwise from the first axis toward the second). Like NumPy it is a **pure view composition** of axis-flips + a transpose — no data moves, so the result shares memory with `m` (read-only when the source is, e.g. a broadcast view). `k` is taken mod 4 (Python-style, so negatives wrap); `k∈{0}` returns a full view, `k∈{2}` flips both axes, `k∈{1,3}` flip-then-transpose. Validation mirrors NumPy verbatim (`ValueError`, NumPy's own type — not normalize_axis_index's AxisError): `len(axes)!=2` → "len(axes) must be 2.", `axes[0]==axes[1] || |axes[0]-axes[1]|==ndim` → "Axes must be different." (this fires *before* the range check, so e.g. `axes=(0,2)` on 2-D reports "different", not out-of-range), else out-of-range. See `Manipulation/np.rot90.cs`.

`np.trim_zeros(filt, trim="fb", axis=null)` crops leading/trailing all-zero hyperplanes (NumPy 2.2.0+ N-dimensional bounding box; probed against 2.4.2). `trim` is case-insensitive ('f'=front, 'b'=back, "fb"/"bf"=both) and validated (`ArgumentException "unexpected character(s) in \`trim\`: '…'"`). `axis=null` trims the smallest bounding box over all axes; an explicit axis/axis-list (int overload + `int[]` overload) trims only those (NumPy's `normalize_axis_tuple(axis, ndim, argname="axis")`: every axis range-checked first → `AxisError "axis: axis 2 is out of bounds for array of dimension 2"`, then `ValueError "repeated axis in \`axis\` argument"`); an **empty** axis list or a **0-d** input returns the input unmodified. Always returns a **view** (dtype & ndim preserved); all-zero/empty inputs collapse the trimmed axes to length 0 (trim spec ignored, matching NumPy). Implementation is NOT NumPy's `argwhere.min/max` (which materializes every non-zero coordinate) — the bounding box is separable, so it takes the first/last non-zero index from `np.nonzero` (1-D) or `np.nonzero(np.any(filt, other_axes))` (N-D), touching only the trimmed axes. Measured **3.4×–361× faster than NumPy 2.4.2** (min on tiny 1-D, up to 361× on 3-D single-axis). Differential-fuzzed 400/400 bit-exact across 1–3D × {int/uint/float/bool} × sparsity × trim × axis. See `Manipulation/np.trim_zeros.cs`.

`permute_dims` and `matrix_transpose` are the NumPy 2.x / Array-API transpose aliases (both pure O(1) views). `np.permute_dims(a, axes=null)` is a straight alias of `transpose` (delegates to `TensorEngine.Transpose`). `np.matrix_transpose(x)` swaps the two innermost axes (`(..., M, N)→(..., N, M)`, i.e. `swapaxes(x, -1, -2)`) and requires `ndim ≥ 2`, else `ArgumentException "Input array must be at least 2-dimensional, but it is {ndim}"`; the companion property `ndarray.mT` does the same but raises the NumPy-verbatim `"matrix transpose with ndim < 2 is undefined"` (distinct message, both probed against 2.4.2). Read-only/broadcast inputs stay non-writeable through the alias. See `Manipulation/np.permute_dims.cs`, `Manipulation/np.matrix_transpose.cs`, `Backends/NDArray.cs` (`mT`).

`Transpose` axis-handling is NumPy-exact (probed against 2.4.2, shared by `transpose`/`permute_dims`/`swapaxes`/`moveaxis`/`rollaxis`): only `axes=null` reverses — an explicit **length-0** `axes` (`new int[0]`, NumPy's `()`/`[]`) is a length-0 permutation that matches **only a 0-d array** and raises `ValueError "axes don't match array"` for any `ndim ≥ 1`; wrong-length and repeated/negative-duplicate axes raise `ValueError` ("axes don't match array" / "repeated axis in transpose"); out-of-range axes raise `AxisError` (all NumPy's types since 2026-10-08, plan X36). `np.moveaxis` names its arguments (`AxisError "source: axis 5 …"` / `"destination: …"`, `ValueError "repeated axis in \`source\` argument"`), `np.swapaxes` too (`"axis1: …"` / `"axis2: …"`), and `np.rollaxis(a, axis, start)` raises NumPy's message-only `AxisError "'start' arg requires -2 <= start < 3, but 4 was passed in"` (start reported after its negative wrap). Every transpose is a view, an EMPTY array's too (`PyArray_Transpose` has no size-0 case; until 2026-10-05 NumSharp allocated a fresh owning array there, writeable even for a read-only broadcast — gate `Manipulation/np.transpose.Empty.Test.cs`). See `Backends/Default/ArrayManipulation/Default.Transpose.cs`.

**Stacking/joining details** (all probed against NumPy 2.4.2; tests in `Creation/np.{block,column_stack,concat,unstack}.Test.cs`):
- `concat` — NumPy 2.0 Array-API alias of `concatenate`; mirrors every overload (array form with `axis`/`out`/`dtype`/`casting` + tuple-arity forms). `Creation/np.concat.cs`.
- `unstack(x, axis=0)` (NumPy 2.1) — returns `NDArray[]` of **views** (shared memory; write-through), each `x.shape` minus the axis entry. Built by dropping the axis from dims/strides and advancing offset — no moveaxis call, no iterator. 1-D input → 0-d views; 0-d input → `ValueError("Input array must be at least 1-d.")` verbatim. `Creation/np.unstack.cs`.
- `column_stack(tup)` — sub-2-D inputs become columns via a single-alias `(N,1)` view (NumPy's `array(a, ndmin=2).T`), then `concatenate(axis=1)`; ndim≥2 pass through untouched (3-D+ works, like NumPy). `Creation/np.column_stack.cs`.
- `block(arrays)` — full port of `numpy/_core/shape_base.py`: depth-check walk (verbatim ValueError texts for depth mismatch / empty lists; `TypeError` for `ITuple`s), `_atleast_nd` leading-1 view padding, and BOTH algorithms behind NumPy's exact `list_ndim * final_size > 2*512*512` threshold — repeated-concatenate (intermediates disposed eagerly) vs single-allocation `_block_slicing` (per-leaf `NDIter.Copy`, `order='F' if allF and not allC`). C# nesting map: `object[]`/`NDArray[]`/1-D primitive arrays/`IList` = Python list; `NDArray`/scalars/rank≥2 rectangular arrays = leaves; `np.block(x)` depth-0 returns a **copy**. C#-scalar dtype note: leaves keep their C# type (`10` → int32), unlike Python ints (int64). `Creation/np.block.cs`.
- `concatenate` layout vote fixed to NumPy's ambiguity resolution: **F output only when all inputs are F-contiguous AND not all are also C-contiguous** — both-C&F inputs (e.g. `(N,1)` columns) now produce C order, matching `PyArray_CreateMultiSortedStridePerm`. Also added the uniform-tiny-slab interleave fast path (constant-size loads/stores when every per-outer slab is the same 1/2/4/8/16 bytes — the `(N,1)` column pattern; ~10-20× over per-slab `Buffer.MemoryCopy`).
- `UnmanagedStorage.GetData<T>()` now narrows to the shape's `[offset, offset+size)` window before returning/casting (zero-copy, still writes through) — previously a **contiguous view at a non-zero offset** (np.split / np.unstack children) returned the WHOLE backing buffer from `Data<T>()`.

`resize` ships as both `np.resize(a, new_shape)` (function — fills the enlarged output with **repeated copies** of `a` in C-order via an exact-sized doubling byte-tile; empty source / zero new-size → `zeros`; always C-contiguous; any input layout is raveled first) and `ndarray.resize(new_shape, refcheck=true)` (**in-place** — grows with **zeros**, shrinks by truncation, operates on the raw contiguous buffer so an F-contiguous resize relabels memory column-major). The method mirrors NumPy's guards verbatim (`IncorrectShapeException`): single-segment only, and when the byte size changes it must own its data (not a view) and — under `refcheck` — not be shared (`IArraySlice.IsUniquelyReferenced`, backed by the ARC block refcount); `refcheck:false` bypasses. Same-size resize is a pure in-place reshape (no ownership/reference check). See `Manipulation/np.resize.cs`, `Manipulation/NDArray.resize.cs`.

### Allocation & reshape guards (error parity)

The `size × itemsize` overflow class and the reshape rejection family, both ported from NumPy's own
C rather than hand-rolled. Probed against 2.4.2; gates: `Backends/AllocationGuardTests.cs` (13),
`Manipulation/ReshapeErrorParityTests.cs` (19), `Manipulation/ManipulationErrorParityTests.cs` (11).

- **`AllocationGuard`** (`Exceptions/AllocationGuard.cs`) is the dimension-check block of
  `PyArray_NewFromDescr_int`. **A size computed by multiplication cannot be trusted until its
  overflow is checked, and the failure is silent by construction** — a wrapped byte count looks
  like a small allocation, not an error. `np.zeros((2^31, 2^31))` of float64 used to *succeed*:
  element count 2^62 is representable but 2^62×8 = 2^65 wraps to **exactly 0**, so the allocator
  was asked for zero bytes and obliged — `[0,0]` read 0 and the last element read out of bounds;
  `np.ones` on the same shape wrote out of bounds. Two behaviours here are load-bearing and easy
  to get wrong: the running product starts at **itemSize** (the quantity checked is BYTES — which
  is why `np.zeros(2^60)` f8 is rejected while the same shape of int8 is merely `OutOfMemoryException`
  ≡ NumPy's `MemoryError`), and **a zero dimension does NOT short-circuit the scan** (NumPy keeps
  multiplying "as if" it were 1, so `np.zeros((0, 2^62))` f8 still reports "array is too big").
  Dims are scanned left to right so the error follows NumPy's order: `(-1, 2^62)` is
  `negative dimensions are not allowed`, `(2^62, -1)` is `array is too big; ...`.
  There are **two distinct wrap points** and a guard that sees only one misses the other:
  `(2^31, 2^31)` wraps the BYTE count, `(2^62, 2^62)` wraps the ELEMENT COUNT itself to 0 — which
  is why the guard keys off DIMENSIONS, never `shape.size`. `np.full` / `np.full_like` / `np.ones`
  allocate from `shape.size` directly instead of via `UnmanagedStorage.Allocate`, so they carry the
  call explicitly; `UnmanagedMemoryBlock<T>`'s two allocating ctors hold a byte-count backstop for
  any path that reaches an allocator without passing a `Shape`.
- **`ndarray.resize` is NOT the same loop** — `PyArray_Resize_int` differs from the creation guard
  on all three axes, all observable: it **breaks** at the first zero dim (so `a.resize((0,-1))` is
  legal and yields shape `(0,-1)`, while `np.zeros((0,-1))` raises), its text omits the "are"
  (`negative dimensions not allowed`), and an overflowing product is a **MemoryError**, not a
  ValueError. It therefore allocates by element COUNT and relabels — as NumPy does, mallocing
  `newnbytes` and only then writing the dimensions — instead of routing the requested dims through
  the creation guard.
- **`Shape.Reshape`** is now `_fix_unknown_dimension` + `convert_shape_to_string`. **ANY negative
  dim is the unknown one**, not just `-1` (probed: `np.zeros(6).reshape(-3)` is `(6,)`,
  `reshape(3,-5)` is `(3,2)`) — matching only `-1` let a SECOND negative ride through as a literal
  dimension, so `reshape(-1,-2)` silently produced the shape `(-3,-2)`, a negative-extent array.
  The zero product is tested **before** the division (`s_known == 0 || size % s_known != 0`), which
  is why `np.zeros((0,3)).reshape(-1,0)` no longer raises `DivideByZeroException`. Every rejection
  except the multiple-unknown one carries NumPy's single verbatim text,
  `cannot reshape array of size {n} into shape {s}`, where `{s}` follows `convert_shape_to_string`'s
  three quirks: LEADING unknowns are dropped (`reshape(-1,0)` → `(0)`), later ones read `newaxis`
  (`reshape(0,-1)` → `(0,newaxis)`), and a one-element shape closes with `,)` (`reshape(0)` → `(0,)`)
  — so `(0)` and `(0,)` both occur and mean different things. Type stays the long-standing
  `IncorrectShapeException` (house convention); the multiple-unknown case is `ValueError`
  ("can only specify one unknown dimension"), previously a bare `ArgumentException`.
- **`np.expand_dims(a, int)`** validates the axis against the OUTPUT ndim (`a.ndim + 1`) and reports
  the axis **as given**. `Shape.ExpandDimension` guards only the negative side and does it against
  the *normalized* value, so a positive overshoot fell through to `Arrays.Insert` and surfaced as
  `ArgumentOutOfRangeException ... (Parameter 'index')` — a parameter `expand_dims` does not have —
  a negative overshoot reported the normalized axis, and a **0-d input skipped the check entirely**
  (`expand_dims(scalar, 5)` silently returned shape `(1)`). The guard lives at the `np.*` boundary,
  not in `Shape.ExpandDimension`, because that method has internal callers (keepdims reductions,
  fancy-index shaping, `asmatrix`) not bound by NumPy's axis contract.

**Matched, do not re-prove:** the manipulation family's *values* are clean across the degenerate
inputs (`split`/`array_split`/`hsplit`/`vsplit` × {0, negative, empty}, `repeat`/`tile`/`roll` ×
{0, negative, empty axis}, `squeeze`, `moveaxis`, `swapaxes`). **Known remaining divergences**, all
typed-and-texted on both sides (text only, no raw framework leak): `np.split(a, 0)` raises
"number sections must be larger than 0." where NumPy leaks `ZeroDivisionError: integer modulo by
zero` (NumSharp is stricter, deliberately); `repeat(-1)` and `tile(-1)` texts differ in wording;
`np.broadcast_to(a, (2^62, 6))` builds the view where NumPy refuses with `iterator is too large`. (The axis
errors of this family — `split(..., axis=oob)`'s leaked `IndexError`, `squeeze`, the old parenthesized
`AxisOutOfRangeException` texts — are NumPy's since 2026-10-08; see "Axis arguments" below.)

**Axis arguments are NumPy's `normalize_axis_index` / `normalize_axis_tuple`, ported once** (2026-10-08, the
open-bugs rundown's bug 7 and the neighbours its sweep found; plan `docs/plans/api-parameter-parity.md` X36; gates
the type-strict, zero-excuse `axis_args.jsonl` oracle tier (15,444 cases) + the evaluate tier's C10 block (330) +
`Manipulation/AxisArgumentParity.Test.cs`).
`DefaultEngine.check_and_adjust_axis(ndims, axis, msgPrefix)` is NumPy's `check_and_adjust_axis_msg`: the range
test runs on the axis AS GIVEN, then the wrap, and the error is the house `AxisError` (NumPy's type, an
`ArgumentOutOfRangeException`) with NumPy's text, `"{msgPrefix}: axis 5 is out of bounds for array of dimension 2"`
(`new AxisError(axis, ndim, msgPrefix)`; NumPy names the argument where it normalizes several: moveaxis'
`source` / `destination`, swapaxes' and diagonal's `axis1` / `axis2`, cross' `axisa` / `axisb` / `axisc`,
average's and trim_zeros' `axis`, unstack's `source`). `DefaultEngine.normalize_axis_tuple(axis, ndim, argname,
allow_duplicate)` is NumPy's function: a NEW array (it used to rewrite the caller's in place), every axis
range-checked before the repeat check, a repeat NumPy's `ValueError "repeated axis"` / `"repeated axis in
\`{argname}\` argument"` (it used to REMOVE the repeat with `Distinct()`). Rules every caller follows, each a fixed
divergence:
- **Never wrap then test.** `while (axis < 0) axis += ndim` accepted `-ndim - 1` (wrapping twice to the last axis:
  `np.argmax(a2d, -3)` returned the last axis' answer) and NEVER TERMINATED for a negative axis of a 0-d input
  (`np.count_nonzero(0-d, axis=-2)` hung the oracle until its watchdog). Gone from argmax / argmin / cumprod /
  nancumprod / nansum / nanprod / the nan axis kernels / count_nonzero / squeeze.
- **Check before every shortcut.** `ReductionArguments.CheckReduceAxis` (PyUFunc_GenericReduction's integer branch:
  a 0-d input takes 0 / -1 and nothing else) runs first in prod / max / min / nanmax / nanmin — their 0-d and
  one-element shortcuts answered for any axis. Likewise cumsum / cumprod / argmax / argmin / compress.
- **0-d inputs:** the ufunc reductions, `count_nonzero` and `squeeze` accept axis 0 / -1 of a 0-d input (no axis
  selected) and report every other axis against rank 0; the `PyArray_CheckAxis` functions (argmax / argmin / argsort /
  cumsum / cumprod / take / compress / repeat / packbits / unpackbits) ravel a 0-d input first and report rank 1;
  `np.sort`, `np.rollaxis`, the tuple forms and `np.linalg.norm` / `vector_norm` (a tuple reduce) refuse every axis.
- **Python's IndexError where NumPy reads `a.shape[axis]` first:** `np.split` / `np.array_split` (before the
  sections check) and `np.linalg.cross` raise `IndexError "tuple index out of range"`; `np.fft.irfftn(a, axes=())`
  raises it on the caller's tuple ("list index out of range" only for the defaulted axes of a 0-d input).
- **NumPy's ValueError / TypeError around the axis:** a non-unit squeeze axis ("cannot select an axis to squeeze out
  which has size not equal to one"), two equal diagonal / trace axes, rot90's axes, split's sections, stack's shape
  mismatch, the stacking family's empty list ("need at least one array to stack" / "…to concatenate"; atleast_1d/2d/3d
  of no arrays is the empty tuple), diff / unwrap / ma.diff of a 0-d input, the quantile family's q range; the
  boolean subtract / negative refusals are NumPy's `TypeError` (they were `NotSupportedException`), and the quantile
  family's complex input is `TypeError "a must be an array of real numbers"`, raised first (it computed a value).
- **Values the rules decide:** `np.linalg.vector_norm(x, axis=tuple)` is NumPy's tuple branch (axes merged, the
  VECTOR norm; a 2-tuple reached norm's matrix branch, so the default order computed the spectral norm), and the
  quantile family's EMPTY tuple reduces no axis (it reduced every axis). `np.rollaxis` with nothing to move and
  `np.expand_dims(a, ())` return a fresh view, never the input.
- **Axes NumPy uses as Python LIST indices or values raise Python's errors:** np.pad's dict `pad_width` (NumPy 2.3+)
  is NumPy's dict branch — each width matched first (anything but an int or a 2-tuple of ints is `typing.assert_never`'s
  `AssertionError`, a new house type; a typed C# array is an ndarray and refused), then `seq[axis] = …` (a key outside
  `[-ndim, ndim)` is `IndexError "list assignment index out of range"`, every key of a 0-d array is one), then
  `np.asarray(seq)`'s `TypeError` (a 0-d array's empty dict, every row given bools, a width past int64), then the negative
  width's `ValueError` after every key; `np.linalg.tensorsolve`'s axes go through `allaxes.remove(k)`, so a negative axis
  or one past the rank is `ValueError "list.remove(x): x not in list"`.
- **np.evaluate's reductions follow their NumPy function's axis rule:** the ufunc kinds (and count_nonzero / nansum /
  nanprod / nanmean) take axis 0 / -1 of a 0-d child as no axis; Mean / Var / Std refuse it (`_count_reduce_items`);
  Average names its argument ("axis: …").
- `AxisOutOfRangeException` is `[Obsolete]` and derives from `AxisError`: nothing in NumSharp raises it any more.
- **Rode along:** default-order `vector_norm` / `norm` of a float32 / float64 operand streams `Sum(x * x, axis)` through
  np.evaluate (add.reduce's schedule, same bits, no temporary; 4.3-9.2x NumPy over tuples), and `np.ptp` over a tuple
  naming every axis takes the flat reductions. Still open: np.average over a tuple axis reduces one axis at a time (a mean
  of means), pinned as `OpenBugs.Reductions.cs` group 14.
- **One house convention kept in the strict tier:** NumPy's broadcast `ValueError` ("shape mismatch: objects cannot be
  broadcast…", np.cross' operands) stays NumSharp's `IncorrectShapeException`, library-wide.

### Broadcasting
`are_broadcastable`, `broadcast`, `broadcast_arrays`, `broadcast_to`

### Math — Arithmetic
`abs`, `absolute`, `acosh`, `add`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctan2`, `arctanh`, `asinh`, `atanh`, `cbrt`, `ceil`, `clip`, `convolve`, `correlate`, `cos`, `cosh`, `deg2rad`, `degrees`, `divide`, `divmod`, `exp`, `exp2`, `expm1`, `fabs`, `fix`, `float_power`, `floor`, `floor_divide`, `fmod`, `gcd`, `lcm`, `log`, `log10`, `log1p`, `log2`, `mod`, `modf`, `multiply`, `negative`, `positive`, `power`, `rad2deg`, `radians`, `reciprocal`, `remainder`, `rint`, `sign`, `sin`, `sinc`, `sinh`, `spacing`, `sqrt`, `square`, `subtract`, `tan`, `tanh`, `true_divide`, `trunc`

**`np.fix`** — round toward zero, dtype-preserving (NumPy `numpy/lib/_ufunclike_impl.py`, gated by the
`unary`-tier `trunc` corpus via the `fix→trunc` alias + `Math/np.fix.Test.cs`). In NumPy 2.4.2 `fix(x, out=None)`
is a VERBATIM delegation to `trunc(x, out=out)`, so `np.fix` is `np.trunc` by another name and inherits it
exactly — dtype preservation (int/bool identity, floats truncated), NaN/±inf pass-through, the sign of `-0.0`,
and the error surface (a complex input and an incompatible `out` dtype both leak the underlying `trunc` ufunc,
which is what NumPy's `fix` leaks too since it also delegates). Unlike a true ufunc it has NO `where`/`dtype`
parameter — the signature is `fix(x, out=null)`, `out` positional as in NumPy. One inherited `trunc` divergence:
`fix(complex)` raises `NotSupportedException` where NumPy raises `TypeError` (pre-existing in `trunc`, shared).
See `Math/np.fix.cs`.

**`np.fabs`** — the FLOAT-ONLY absolute value (NumPy `fabs` ufunc, `TD(flts, f='fabs', astype={'e':'f'})` —
float loops `efdg` only; probed against 2.4.2; gates `Math/np.fabs.Test.cs` (21) + the `unary`/`errors_full`
corpus tiers). It is the float sibling of `abs`/`absolute` and differs in exactly two ways, both dtype-level:
it PROMOTES bool/int to float (NEP50 tier: bool/int8/uint8→float16, int16/uint16/char→float32,
int32/uint32/int64/uint64→float64; float16/float32/float64/decimal preserved) — so `fabs(int)` is always a
float where `abs(int)` preserves int — and it has NO complex loop (`abs` maps complex→magnitude; `fabs`
REJECTS it). The **operation** is identical to `abs` on the float loops — clear the IEEE sign bit — so `fabs` rides
`UnaryOp.Fabs`, a distinct op that **aliases `UnaryOp.Abs` at every kernel emit site** (the 6 non-complex ones:
scalar `EmitAbsCall`, `Vector.Abs` name-map, the SIMD gate, the f16 selector, decimal, f16-scalar), inheriting
every layout, `out=`/`where=`/`dtype=`, and the f16/decimal paths for free. It is a separate enum value ONLY so
`UfuncName(UnaryOp.Fabs) == "fabs"` — the op drives both kernel selection (~10 sites) and the ufunc name in errors,
and a dedicated enum keeps fabs's specialness in the per-op dispatch layer rather than adding a name-override
parameter to the shared `ExecuteUnaryOp` executor (`Default.Fabs` → `ExecuteUnaryOp(nd, UnaryOp.Fabs, floatType,
out, where)`). The
promoting path casts int→float BEFORE the abs, so `fabs(int.MinValue)` is the exact float magnitude, never the
wrapped integer abs. **BIT-EXACT with NumPy 2.4.2** across all 12 NumPy-representable non-complex dtypes ×
26 layouts, including `-0.0→+0.0`, `±inf→+inf`, and a negative/payload-bearing NaN whose sign bit is cleared
while the payload is preserved (`0xfff8…abcdef → 0x7ff8…abcdef`). **Complex is refused with NumPy's exact three
error paths** (probed): no `dtype=` → `TypeError` "ufunc 'fabs' not supported for the input types…"; `dtype=`
float → "Cannot cast ufunc 'fabs' input from complex128 to <float> with casting rule 'same_kind'"; `dtype=`
complex/int → "No loop matching the specified signature and casting was found for ufunc fabs". The `out=` cast
error correctly names `"fabs"` (not `"absolute"`) because `UnaryOp.Fabs` is its own op with its own `UfuncName`
entry — no name-override parameter on the shared executor. **Perf (NPY/NS, Release,
best-of-15):** faster than NumPy on every measured cell — NumPy's `fabs` is a SCALAR CRT loop (no SIMD dispatch,
unlike `absolute`), while NumSharp rides `Vector.Abs`: float16 **4.5–24.8×**, float64 **1.75–13.9×**, int32
**1.6–10.8×**, `out=` paths 2.5–14× (the one sub-1.5× cell, fresh-alloc float32@100K at 1.19×, is the shared
allocation floor — its `out=` variant is 12.2×). See `Math/np.fabs.cs`, `Backends/Default/Math/Default.Fabs.cs`.

**`np.clip`** — a port of `numpy/_core/_methods.py::_clip` (the code behind `np.clip` and `ndarray.clip`), of
`fromnumeric.clip`'s keyword rule, and of the ufunc call it routes to (2026-10-03; probed against 2.4.2; gates the
`clip_pyint` fuzz tier (4,170 cases) + the `clip_ufunc` tier (2,302 cases) + `Math/np.clip.{PythonIntBounds,Ufunc}.Test.cs`
+ `Backends/Kernels/BroadcastPatternStageTests.cs`). Signature `clip(a, a_min=null, a_max=null, out=null, dtype=null,
min=null, max=null, where=null)`; `ndarray.clip(min, max, out, dtype, where)`. A 0-d integer bound (any C# integer
literal, or a char) is NumPy's Python int:
- **Keyword rule** (`fromnumeric.clip`, raised before any ufunc error, a read-only out included). `min=`/`max=` alone
  are the bounds (NumPy 2.x aliases). Both positionals plus a keyword → ValueError "Passing `min` or `max` keyword
  argument when `a_min` and `a_max` are provided is forbidden."; one positional plus a keyword → TypeError
  "clip() missing 1 required positional argument: 'a_min'" (or 'a_max'). C# has no "omitted" distinct from null, so a
  null positional counts as omitted only when a keyword is given (`[Misaligned]`: `np.clip(a, 1)` is legal here,
  NumPy's TypeError needs the argument absent).
- **Drop rule.** On an integer array, a bound at or past `iinfo(a.dtype)` (`min <= iinfo.min`, `max >= iinfo.max`)
  cannot clip anything and is DROPPED: `np.clip(uint8, 0, 256)` returns the array unchanged, and
  `np.clip(uint8, -1, 100)` only clips from above. Before this, the 0-d bound adopted the array's dtype and
  `astype` WRAPPED it (256→0 gave all zeros, int8 ±1000 gave all -24).
- **Routing.** No bound left → `positive` (a typed copy, with positive's own rules: `np.clip(bool_array)` raises
  its no-loop TypeError). One bound → `minimum`/`maximum`. Two → the clip kernel. The routed name is what every
  cast error quotes. `where=` and `dtype=` ride along (NumPy forwards `**kwargs`).
- **Packing.** A kept bound that does not fit the LOOP dtype raises OverflowError "Python integer 1000 out of bounds
  for int8" before any element is read, so an empty array raises too (`NDExprTypeRules.CheckPyIntPacksIntoLoop`,
  NumPy's `@TYPE@_safe_pyint_setitem`).
- **Range vs loop.** The drop rule uses the INPUT's range, while `dtype=` moves the packing:
  `np.clip(int8, 1000, 2000, dtype: np.int16)` is legal (all 1000).
- **Platform texts.** Past ±2³¹ NumPy's text follows the platform C `long`; NumSharp pins the LP64 (Linux/macOS)
  texts. A value past long.MaxValue reports "Python int too large to convert to C long", except into uint32/uint64.
- **Casts.** `dtype=` validates each input same_kind, in operand order (a Python int reaches any integer or inexact
  loop but not bool, named int64), so `np.clip(f64, min: 3.0, dtype: int32)` RAISES as in NumPy. `out=` accepts
  any same_kind-reachable dtype (computed at the loop dtype, then cast in through the iterator's buffers). A given
  `out` is returned itself.
- **Broadcasting** (NpyIter's joint broadcast). The array, both bounds, `out` and `where` broadcast together: a bound
  may be LARGER than the array (`np.clip(arange(3), zeros((2,3)), 1)` is (2,3)), and an out larger than every input
  receives the broadcast; out itself never stretches. NumPy's two texts, verbatim: "operands could not be broadcast
  together with shapes (3,) (4,) () (3,) " (every operand, 0-d as `()`, trailing space) and "non-broadcastable output
  operand with shape (3,) doesn't match the broadcast shape (2,3)".
- **where=** must be bool (the house `ArgumentException` "Cannot cast array data from dtype('int64') to dtype('bool')
  according to the rule 'safe'" where NumPy raises TypeError — same text). Masked-off positions of a given out keep
  their contents; of a fresh result they are uninitialized, as in NumPy.
- **Overlap.** An out that overlaps an input other than as its exact elementwise alias (`np.clip(x[:-1], 1, 4,
  out=x[1:])`) is computed as if from a copy (NDIter COPY_IF_OVERLAP); the in-place `np.clip(x, lo, hi, out=x)` is
  not copied. With where= the out is READWRITE (NumPy's masked ufunc rule), so masked-off positions come back intact.
  **Library-wide fix found here:** the ufunc out=/where= routes (`DefaultEngine.UfuncOut.cs`) and np.evaluate's
  masked route flagged a passed-in masked out WRITEONLY, so an overlapping out + where wrote heap garbage into the
  masked-off positions (`np.add(x[:-1], 1, out=x[1:], where=m)`); now READWRITE.
- **Error order** (NumPy's): keyword rule → read-only out → where= dtype → packing → dtype= input casts → out cast →
  broadcast / out shape.
- **Promotion.** 0-d bounds use the ufunc array-scalar table (`np._FindCommonArrayScalarType`), so a Char array with a
  float bound promotes to float64 instead of truncating the bound into char.
- **The float loop.** NumPy's clip loop runs its CONST fast path (clip.cpp `_npy_clip_const_minmax_`:
  `x < lo ? lo : x; x > hi ? hi : x`, a tie keeps the element, so `np.clip(-0.0, 0, 1)` is -0.0; a NaN bound fills
  the CALL with that NaN's own bits) whenever both bound strides of an inner-loop call are 0, else the general
  `MIN(MAX())` (a tie takes the bound). Which one runs depends on NumPy's trivial-loop test and NpyIter's axis order,
  coalescing and buffering — a (2,1) bound against a (2,3) array runs const when NpyIter does not buffer it; a
  one-element call off the trivial loop runs const (`Default.ClipNDArray.ConstLoop.cs` asks the shared port of that
  decision, `DefaultEngine.UfuncInnerSteps.cs`, which np.power's float loops use too since 2026-10-06; validated
  over 21,000 random layout / dtype / out / where / overlap cases against live NumPy, the -0.0 tie revealing the loop).
  On x86 the const compare-selects ARE `MAXPD(lo, x)` / `MINPD(hi, x)` (second operand on a tie or a NaN), one
  instruction per bound.
- **Result layout.** A fresh result takes NumPy's ufunc output layout (`Shape.NpyIterOutputShape`: the operands' memory
  order, the mask votes, C wins ties); a dense F or permuted input is clipped as a C run and relabelled
  (`NDArray.AdoptLayout`), no second pass.

Two execution routes (`Backends/Default/Math/Default.ClipNDArray{,.Iter,.ConstLoop}.cs`): the FLAT route (no where=,
every operand one dense run after the layout permutation, no harmful overlap) runs the whole-array kernel
(`DirectILKernelGenerator.Clip.cs`: output-alignment peel, then aligned stores; past 16 MiB out-of-place, non-temporal
stores + sfence); everything else is NumPy's own structure — ONE NDIter over (a, lo, hi, out[, where]) driving a
Tier-3B clip kernel (`DirectILKernelGenerator.Clip.InnerLoop.cs`). Three-input kernels with a broadcast input reached
SIMD only through the **N-ary broadcast-pattern stages** added with this work
(`DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs`, shared by every Tier-3B kernel): the 1-D shell, the 2-D
block kernel and the where= 2-D block kernel dispatch on the inputs' C/S pattern once per call (the all-C loop
inline, each other pattern a DynamicMethod STAGE compiled on its first call, so the JIT's per-method inline budget is
not split 2^N ways), and the gather route broadcasts a stride-0 input next to strided ones instead of gathering it
(binary ufuncs too). Fixed costs trimmed on the way: the scalar bounds of the flat route are read in place or
converted into stack slots (no 0-d astype per call), the kernel bodies and op-flag arrays are cached per key, the
dispatcher's operand array and np.clip's own empty `[NDScoped]` scope are gone (-224 B / call), and NDIter's
broadcast of differing operand shapes is one right-aligned pass instead of a List / Shape[] / ResolveReturnShape
round trip (every ufunc call with a 0-d or row operand).

Remaining divergences:
- A strong 0-d NDArray bound (NumPy's `np.int64(5)` / `np.array(5)`) cannot be told from a literal, so it is treated
  as a Python int; NumPy promotes with it (an int8 clip by `np.int64(1000)` is int64 there).
- The keyword rule's null-as-omitted reading (`np.clip(a, 1)`, `np.clip(a, 2, None, min=3)`) — `[Misaligned]`.
- where= with an overlapping out of ANOTHER dtype: NumPy round-trips the masked-off values through the loop dtype,
  NumSharp keeps them exactly (`[Misaligned]`).
- The where= mask's error TYPE (house ArgumentException vs NumPy TypeError; text identical).
- `np.broadcast_to` / `np.broadcast_arrays` keep an extent-1 axis's original stride where NumPy writes 0; only a
  one-element bound built by broadcast_to can change clip's const-loop decision through it (library-wide).
- `NDExpr.Clip` (np.evaluate) has no drop rule, throws on two null bounds and on an out-of-range literal, and runs the
  general loop on ties (`MIN(MAX())`) — the fused-expression node, not np.clip.

**Perf (NPY/NS, clock locked, one P-core, best-of-rounds; float64 / float32 / int32 at 1K / 100K / 10M):**
- broadcast bounds (`np.clip(m, row, 0.75)`, column bounds, row bound into out=): **4.7–27× at 100K, 1.5–8.6× at 10M**
  (were 0.2–0.7×: a ternary kernel with a stride-0 input ran a scalar loop);
- where= over 2-D rows with a row bound: **19–41× at 100K, 10–21× at 10M** (were 1.6–2.1×);
- strided input + 0-d bounds: 1.0–3.0× at 100K (the gather route no longer gathers the bounds);
- 1K: **1.4–14×** on every shape but float32 strided (1.1×: its two 0-d float64 bounds are cast to 0-d float32 arrays,
  the NDArray construction floor); the iterator route's fixed cost roughly halved;
- 10M contiguous into out=: 1.35–1.6× (non-temporal stores; were ~1.0×).

Cells still under 1.5×, all shared floors rather than clip costs: contiguous clip at 100K (~0.8–1.4×, L2-bandwidth
bound on both sides — NumPy's loop streams as fast; misaligned outputs no longer cost extra) and fresh 10M results
(page-fault bound, ~1.0–1.1×); 1-D where= (~1.0×: NumPy and NumSharp both run one inner-loop call per mask run —
admitting 1-D into the masked block kernel is the lever, it changes every masked ufunc); a buffered out= cast or
dtype= cast at 10M (~1.1×, two passes on both sides) and the int32→float64 dtype= result at 10M (page faults of the
80 MB result); an out overlapping the input at 100K (the COPY_IF_OVERLAP temporary on both sides).

See `Math/np.clip.cs`, `Backends/Default/Math/Default.ClipNDArray{,.Iter,.ConstLoop}.cs`,
`Backends/Kernels/Direct/DirectILKernelGenerator.{Clip,Clip.InnerLoop,InnerLoop.BroadcastPatterns}.cs`.

**The divmod family** — `remainder`/`mod`, `fmod`, `divmod` (all probed against 2.4.2; gates the
`divmod_power`/`multioutput` fuzz tiers + `Math` unit tests). `np.remainder` is an exact ALIAS of the existing
`np.mod` (in NumPy `remainder` IS the `mod` ufunc — floored remainder, result takes the DIVISOR's sign). **`np.fmod`**
is a NEW `BinaryOp.Fmod` — C-library remainder (truncated division, result takes the DIVIDEND's sign): `fmod(-7,3) ==
-1` where `mod(-7,3) == 2`. It rides the exact `Mod`/`FloorDivide` seam (`ExecuteBinaryOp` → scalar-only per-element
kernel `EmitFmodOperation` → `NDDivision.Fmod*`), so it inherits all layouts, NEP50 promotion (integer stays integer,
bool→int8), out=/where=/dtype=, and every dtype for free; integer `fmod` is C# `%` with the ÷0→0 and int/long MIN%-1
guards, float `fmod` is C# `a % b` (bit-identical to C `fmod`), Complex refused (NumPy TypeError). **`np.divmod`** is the
two-output ufunc returning the tuple `(floor_divide(x1,x2), remainder(x1,x2))` — VERIFIED byte-for-byte across every
edge (÷0→(±inf,nan)/(0,0)int, signed MIN/-1→(MIN,0), ±inf/nan/-0.0). Its `out=(q,r)`/`where=`/`dtype=` path composes
the already-validated `FloorDivide`+`Mod`; the common path runs a **fused single-pass IL kernel**
(`DirectILKernelGenerator.DivMod.cs`, a 4×-unrolled scalar two-in/two-out loop calling `NDDivision.Divmod*(a,b,out
mod)->floordiv` over contiguous, promoted operands — the `np.modf` materialize pattern). The integer `Divmod*` helpers
use the **one-idiv form** `q=n/d; r=n-q*d` (not a second `n%d`), halving the integer kernel cost. **Perf (NPY/NS,
Release, best-of-11):** `divmod` 100K **2.1–2.3×**, 10M f64/i32 **1.7–1.8×** (10M i64 1.26×, memory-bandwidth-bound);
`fmod` 100K **1.6–1.9×**, 10M **1.34–1.45×** (bandwidth-bound). **W1-A fix (rode along):** `EmitHalfOperation` now
computes float16 `mod`/`floor_divide`/`fmod` in **float32** (NumPy's HALF loop `astype 'e'->'f'`) via the
`NDDivision.*Single` helpers, so float16 ÷0 yields ±inf (was the NaN `[known bug]` W1-A) — float16 division is now
bit-exact with NumPy, and the W1-A `MisalignedRegistry` excuse is removed. See `Math/np.{divmod,math}.cs`,
`Backends/Default/Math/Default.{Fmod,DivMod}.cs`, `Utilities/NDDivision.{Fmod,Divmod}.cs`.

**`np.float_power`** — power at a MINIMUM precision of float64 (NumPy's `dd->d`/`DD->D` loops ONLY, probed 2.4.2;
gates the `divmod_power`/`out_where`/`specials`/`power_fast` fuzz tiers + `Math/np.float_power.Test.cs`). It is NOT a
new kernel: `Default.FloatPower` resolves the float loop (complex128 if either operand is complex, else float64) and
runs the power KERNEL on that loop (`ExecuteBinaryOp(BinaryOp.Power, …)`, same `Math.Pow` / `ComplexPowNumPy`) — never
np.power's scalar-exponent fast path: NumPy's float_power loop (`PyUFunc_dd_d` with pow) has none, so
`float_power(757.7617475757002, 2.0)` is pow's `...03c2`, not np.power's `x*x` `...03c1` (plan api-parameter-parity
X20; until 2026-10-06 it delegated to `Power` and took the fast path). Three behaviours make it DIFFER from `power`:
no fast path, and two loop-SELECTION rules handled in the wrapper: every real input (bool/int/float16/float32/decimal/char) promotes to float64 and a complex
operand to complex128 (so the result is always inexact float), and — because there is no integer loop — a negative
integer exponent is LEGAL (`float_power(2,-1)=0.5`, where `power` raises "Integers to negative integer powers").
`dtype=` may select ONLY float64/complex128 (else the verbatim "No loop matching…for ufunc float_power"); `out=`/
`where=` and every layout ride `Power`'s existing machinery. **Error taxonomy + ORDER are NumPy's** (read-only out →
non-bool where → dtype-no-loop → complex-input-can't-cast-to-float64 → out-cast → shape), raised in the wrapper with
the `float_power` ufunc name reusing the shared validators. Complex `float_power` inherits the documented complex-power
divergence (F5: `Complex.Pow`/`npy_cpow` host `cpow` ~ULP for non-integer/large-integer exponents, plus inf/NaN
edges) — bit-exact on the integer-exponent branch, F5-excused otherwise (same scope as `power`). **Perf (NPY/NS,
Release):** pow on every element on both sides — `x ** 2.0` 0.96–1.06×, int→float64 **1.05–1.22×**, the general
fractional-exponent path **~parity (0.86–0.99×)**: the `Math.Pow` physical ceiling (NumPy is equally
scalar-`pow`-bound, no SIMD `pow` exists — the same ceiling as `arcsinh`/`arccosh`). The old 4.5–51× for a fast
exponent came from computing `x*x` / `sqrt`, which is not NumPy's value. See `Math/np.float_power.cs`,
`Backends/Default/Math/Default.FloatPower.cs`, `Backends/TensorEngine.cs` (`FloatPower`).

**`np.power`'s float loops — NumPy's zero-exponent-step fast path, exactly where NumPy takes it** (2026-10-06, plan
`docs/plans/api-parameter-parity.md` X13 / X19 / X20; gates the `power_fast` tier + `Math/np.power.StrideZero.Test.cs`).
NumPy 2.4.2's DOUBLE_power / FLOAT_power test the exponent's STEP in each inner-loop call: with `steps[1] == 0` they read
the exponent once and write `1.0/x`, `1.0`, `sqrt(x)`, `x` (raw bits, a signalling NaN kept) or `x*x` for -1, 0, 0.5, 1,
2 (compared in the loop dtype: -0.0 takes the 0 branch, NaN none); any other exponent or a nonzero step is pow. The two
answers differ (`757.7617475757002 ** 2`: `x*x` `...03c1`, pow `...03c2`; `-0.0 ** 0.5`: sqrt -0.0, pow +0.0;
`-inf ** 0.5`: sqrt NaN, pow +inf), so WHICH CALLS the ufunc makes is part of the answer:
- **The decision is ported once** (`Backends/Default/Math/DefaultEngine.UfuncInnerSteps.cs`, `NumPyUfuncInputSteps`,
  shared with np.clip's const loop): `check_for_trivial_loop` (a cast or unaligned 0-d / ≤ bufsize 1-D input is copied;
  any other, or an out of another dtype or unaligned, refuses the trivial loop), `try_trivial_single_output_loop` (a 0-d
  input has step 0, a 1-D operand its own stride, an N-D operand the item size), then NpyIter (fill_axisdata, the axis
  sort, flips, COPY_IF_OVERLAP, coalescing, `npyiter_find_buffering_setup`'s cost model with `np.getbufsize()`; a
  read-only 0-d cast input is copied eagerly; a buffered single-stride broadcast keeps step 0). The decision is uniform
  over a call. Consequences a port must keep: a 0-d exponent is the fast path on every route (plain, out=, where=);
  `(1,) ** (1,)` is pow (the trivial loop's 1-D step) but `(5,) ** (1,)` and a one-element call with a cast out are
  fast (NpyIter, step 0); a column exponent `(R,1)` against `(R,n)` is fast only when NpyIter leaves it unbuffered
  (n > bufsize / 2 at the default; all inputs cast: rows of exactly bufsize stop the core and stay fast).
- **Execution** (`Default.Power.StrideZero.cs`, `PowerFloatLoop`): a single-valued exponent with a fast value runs the
  SIMD unary kernel (square / sqrt / reciprocal / positive) or a fill with 1, out= included, where= as compute-then-masked
  np.copyto; a varying exponent NumPy's calls see with step 0 runs one NDIter pass with NumPy's per-call branch per
  element and, when both inputs are the loop dtype, a vector body (`PowerStrideZeroVector128/256/512`, generic over
  float/double: the broadcast loops hand it a uniform exponent vector, one compare selects one vector op); everything
  else is pow. float16 (HALF_power) and np.float_power have NO fast path.
- **Perf (NPY/NS, two pinned P-cores):** 90 fast-path cells (float64 / float32 × 1K / 100K / 1M × five exponents ×
  plain / out= / where=) geomean ~6×, 100K / 1M 2.0–20×; 1K plain float64 cells 1.3–2.9× run to run, at the NDArray
  allocation floor (`np.empty(1000)` costs 0.20–0.42 µs of NumPy's 1.0–1.9 µs call; power's own cost is 0.01–0.04 µs);
  pow cells (`x ** 3`, full exponents) 0.92–1.15× (CRT pow on both sides); varying columns (2, 5000) 1.67×,
  (16, 8192) 4.4×, (100, 10000) 3.5× (were 0.16–0.29×).
- **Also fixed (X19):** a one-element operand with MORE dimensions than the other kept the array operand's shape in the
  binary, comparison and shift scalar-broadcast shortcuts (`(7,) op (1,1)` gave `(7,)`, NumPy `(1,7)`; 100 of 165 probe
  calls); the shortcuts now decline (gate `Backends/Kernels/OneElementOperandRankTests.cs`).
- **Open:** np.evaluate's Power node has no fast path, so np.kaiser differs from NumPy by 1–2 ULP in 146 of 4,796
  windows (X23); `np.power(unsigned, -1)` raises ValueError where NumPy raises OverflowError (X21); the out= broadcast
  text omits the out's shape (X22).

**`np.gcd` / `np.lcm` — the number-theoretic pair** (NumPy `npy_gcd@c@`/`npy_lcm@c@` in
`npymath/npy_math_internal.h.src` + the `@TYPE@_gcd`/`@TYPE@_lcm` integer loops; gates `Math/np.gcd.Test.cs`
(16) + the differential-fuzz `gcd` tier, 306 cases bit-exact). **INTEGER-ONLY** — bool/half/single/double/
decimal/complex, and the **uint64+signed → float64** NEP50 pair, name no loop and raise NumPy's verbatim
`ufunc '{name}' did not contain a loop with signature matching types (…, …) -> None` (as `TypeError`; the
`dtype=`-path variant closes with `-> {dtype}DType`). Uniform NEP50 promotion (both operands + output share
ONE integer dtype); a bool paired with an integer promotes to that integer loop (valid). New `BinaryOp.Gcd`/
`BinaryOp.Lcm` ride the exact `Fmod` seam — `Default.{Gcd,Lcm}` → `ExecuteBinaryOp` → the scalar per-element
kernel `EmitGcdLcmOperation` → `Utilities/NDGcdLcm.{Gcd,Lcm}*` (one helper per integer dtype incl. Char) — so
they inherit every layout, `out=`/`where=`/`dtype=`, and weak-scalar binding for free. The helpers port NumPy
exactly: unsigned Euclidean core (`while(a!=0){c=a;a=b%a;b=c;}`), signed = the core on operand MAGNITUDES then
reinterpret back, lcm = `gcd==0?0:|a|/gcd*|b|`. Two NumPy-exact wrapping behaviours pinned: a signed result can
be **negative** when the magnitude wraps (`gcd(int8 -128,-128)==-128`, `lcm(int32.Min,1)==int32.Min`) and the
lcm product **wraps** the dtype on overflow (`lcm(int16 21000,14000)==-23536`). **NO SIMD** (data-dependent
Euclidean loop) — like NumPy, which explicitly does not vectorize these. **Perf (NPY/NS, Release, best-of, 100K/
10M): ~parity (0.99–1.06×)** — the physical ceiling (measured: binary/Stein's GCD is SLOWER than division-based
Euclid on modern x86, there is no vector integer division or vector TZCNT, and the IL kernel already matches a
hand-inlined C# loop), the same class of ceiling as `arcsinh`/`float_power`-general. See `Math/np.gcd.cs`,
`Backends/Default/Math/Default.GcdLcm.cs`, `Utilities/NDGcdLcm.cs`.

**`correlate` / `convolve` — the sliding multiply-accumulate family** (NumPy `_pyarray_correlate` +
`small_correlate`; `Math/NDArray.SlidingDot.cs`). `np.correlate(a, v, mode='valid')` is cross-correlation
`c_k = Σ_n a_{n+k}·conj(v_n)`; `np.convolve(a, v, mode='full')` is `correlate(a, v[::-1])` (no conjugation).
Both share ONE 3-region engine — left ramp (scalar), middle (**outer-vectorized over output positions**,
`Vector<T>` separate-mul-add for the 10 SIMD dtypes; char via ushort), right ramp — plus a length-gated
`DotSimd` inner reduction (len≥32) for the ramps/few-output tail of large kernels. correlate is
non-commutative: when `len(a)<len(v)` the operands swap and the output is time-reversed (`_pyarray_revert`),
and a complex `v` is conjugated. Mode strings are case-SENSITIVE (NumPy's two near-miss messages reproduced).
**Bit-parity (managed, the default):** outer-vectorization keeps each output's `Σ_t a[i+t]·k[t]` in t-order, so
it is BYTE-IDENTICAL to the scalar sequential sum — hence to NumPy for the small-kernel regime (verified 1170
cases × 13 dtypes + 24 `groupa` fuzz cases). Integer/bool/decimal/Half are exact at EVERY size (integer SIMD
wrap and the reordered inner reduction are both modular-exact). LONG float32/float64 kernels — **and complex128
at any length** — carry a bounded-ULP divergence, because NumPy reduces exactly those positions through cblas
`?dot` (`?dotu` for complex) while Core has no cblas (the earlier "complex exact at every size" claim was wrong —
complex128 diverged on 690/699 outputs for a length-200 kernel).

**NumPy's dotfunc, per dtype, in its scalar regime (2026-09-30, plan U2).** scipy-openblas' `?dot` is a plain scalar
sum below its vector block, and each regime is now reproduced exactly, so every position shorter than the block is
byte-exact without the backend:
- ddot: a sequential double sum below 16 terms (`DotSimd`);
- sdot: float32 products rounded, then summed in a DOUBLE below 32 terms (`SdotManaged`; probed 0 of 9,300 differ, a
  float32 sum differed on up to 72%);
- zdotu: four separate double sums `Σar·br − Σai·bi`, `Σar·bi + Σai·br` below 8 terms (`ZdotuManaged`; 0 of 2,100
  differ, the naive per-term product on up to 90%) — then its RESULT, built with C99 complex arithmetic: scipy-openblas'
  `openblas_make_complex_double(re, im)` is `re + im*_Complex_I`, whose real part is `re + im*0`, so an infinite or NaN
  imaginary part turns the real part into NaN (`ZdotuResult`, also on the blocked kernel's stores; probed against the
  DLL NumPy loads: `np.convolve([1+0j], [inf+0j])` is `nan+nanj`, the plain sums give `inf+nanj`). Only when BOTH
  operands reach CDOUBLE_dot with a positive stride, though: `PyArray_Correlate` passes a ONE-element operand through
  with its own stride (a size-1 array is C-contiguous whatever its stride), and np.convolve's kernel is `v[::-1]`, so a
  fresh one-element kernel arrives with stride -16, which `blas_stride` refuses — CDOUBLE_dot's plain loop runs, no
  such NaN (`CdoubleDotPlain`; the caller decides with `DotOperandBlasable`: np.convolve reverses its kernel's stride,
  np.correlate's `conj(v)` is always fresh, a cast is always fresh). The backend is skipped there too, as NumPy skips
  cblas.

Only positions at or past the vector block stay bounded-ULP. **Long products without the backend**
(`Math/NDArray.SlidingDot.Long.cs`) run blocked kernels over exactly those positions:
- B consecutive outputs share one loop over the kernel: a broadcast, one load per vector, FMAs into eight chains;
- head/tail steps read tiny zero-padded edge windows, which requires a FINITE kernel (inf/NaN declines to the
  per-position kernels);
- float16 (HALF_dot: a sequential float32 sum at every length) gets an exact blocked kernel, byte-exact at every
  position.

`SlidingCorrelateInto` is the pointer-level core, which numpy.polynomial's series arena calls for its `np.convolve`
statements. Measured (paired, np.convolve 1000×1000): float64 **1.83×** (was ~0.9×), complex128 **1.43×** (was
0.25×). The complex ceiling: both sides do one 256-bit FMA per complex MAC, so only NumPy's per-position overhead is
left to win.

**Byte-parity with `NumSharp.Interop.OpenBLAS` (2026-08-20):** the backend closes that gap — this was the ONE
product-adjacent family that did NOT consult the BLAS seam. NumPy's `_pyarray_correlate` reduces every ramp
position, and the middle whenever `small_correlate` declines (a real kernel longer than 11, or ANY complex
kernel), with its per-dtype `dotfunc` (double-accumulated chunked cblas `?dot`/`?dotu`). When `TensorEngine.Blas`
also implements the new `ISlidingDotBackend` seam (`Backends/ISlidingDotBackend.cs`), `SlidingCorrelate` routes
those EXACT positions through the SAME scipy-openblas `sdot`/`ddot`/`zdotu` NumPy calls (served by
`OpenBlasEngine.SlidingDot`, reusing the product family's byte-exact `@name@_dot`), so
`np.correlate`/`np.convolve` become BYTE-IDENTICAL to NumPy for float32/float64/complex128 — same lever, same
three parity conditions (build, thread count, DYNAMIC_ARCH kernel) as the products. The `small_correlate`-eligible
regime (real float, n2 ≤ 11) STAYS on the managed kernel — already byte-exact there — so there is no mixed
managed/native path within a call, and a backend that declines the dtype (real-only CBLAS handed complex) leaves
the managed kernel in place. It is opt-in and per-position (one backend `?dot` per output, mirroring NumPy's own
per-position `dotfunc` call), so it trades the managed SIMD speed for parity — the whole reason the backend
exists. Live-verified against NumPy 2.4.2: managed diverges on 533–690/699 outputs (length-200 kernel), backend
byte-exact on every dtype × mode × region. Gate: `SlidingDotLiveParityTests` (11, live-numpy adjudicated); the
managed path stays the `groupa` fuzz default.

**Perf (managed default; NPY/NS, best-of-9, Release):** geomean **~4×** — common small-kernel/long-signal cases
**2.3–29×**, mid/large kernels **1.75–6×**; the sole at-parity corner is the large symmetric-kernel `full` case
(`2000×2000` ≈ **1.00×**, ramp-dominated pure dot bounded by OpenBLAS's tuned `ddot`). See `Math/np.correlate.cs`,
`Math/NDArray.{Correlate,Convolve,SlidingDot}.cs`, `Backends/ISlidingDotBackend.cs`,
`NumSharp.Interop.OpenBLAS/OpenBlasEngine.SlidingDot.cs`.

**Inverse hyperbolic** `arcsinh`/`arccosh`/`arctanh` (+ NumPy 2.0 Array-API aliases `asinh`/`acosh`/`atanh`, same ufunc) follow the `arcsin`/`sinh` engine seam (`ASinh`/`ACosh`/`ATanh` → `ExecuteUnaryOp` → `UnaryOp.{Asinh,Acosh,Atanh}` IL kernels) with the `f(x, out=, where=, dtype=)` ufunc surface + positional-dtype convenience overloads. **Real float32/float64 are BYTE-IDENTICAL to NumPy 2.4.2** — `Math.Asinh/Acosh/Atanh` and the `MathF` twins call the same MSVC `ucrtbase` CRT as `npy_asinh/acosh/atanh` (verified 0-diff over 4521 adversarial inputs at both widths, specials/±inf/NaN/subnormals/±0 included), same class as the platform-libm `expm1`/`log1p` cells. **Perf (NPY/NS, Release, best-of-11 warm, `out=`, 10M on byte-identical inputs):** `arcsinh` **0.99×**, `arccosh` **1.00×**, `arctanh` **0.97×** — parity, and this is the correct ceiling rather than a shortfall. These three are the ONE arc-family NumPy has **no active SIMD kernel** for on win-amd64 (its SVML `asinh`/`acosh`/`atanh` are AVX-512/Linux-gated, never compiled into the 2.4.2 wheel), so — unlike `exp`/`log`/`sin`/`cos`/`tanh`, which NumSharp ports bit-exactly and BEATS — there is nothing to port and byte-parity REQUIRES the same scalar `ucrtbase` CRT on both sides. The kernel is a **non-unrolled scalar loop** (`EmitUnaryScalarLoop`; these three fall through every `CanUseUnarySimd` branch, as they must — there is no `Vector<double>.Asinh`); its only edge over a naive managed loop is a raw-pointer direct `call` with no bounds checks, and that alone is enough to BEAT `System.Numerics.Tensors.TensorPrimitives.Asinh`, .NET's own SIMD transcendental library (one-process warm best-of-15, f64 10M, same array: NumSharp IL kernel **97.3 ms** < TensorPrimitives 101.1 ms < naive `Math.Asinh` loop 105.5 ms). Two speedup levers were POC'd and REJECTED: (1) 4×/8× **unrolling** the CRT-call loop does not help — the `ucrtbase` call latency dominates and the extra body is a wash-to-slightly-slower for asinh/atanh (measured); (2) **SIMD buys nothing byte-exactly** — TensorPrimitives is itself **0-ULP** vs NumPy over 10M f64 AND f32 inputs *precisely because it does NOT vectorize these three* (no correctly-rounded vector asinh/acosh/atanh exists, so it falls to the scalar CRT), which is the direct proof that a vector kernel would necessarily DIVERGE. The only faster route is a divergent SIMD polynomial (SLEEF/Cephes, ~1–4 ULP off), which breaks byte-parity and is therefore rejected. So there is no NumSharp-side overhead left to reclaim. Integer/bool tier to float per `ResolveUnaryFloatReturnType` (bool/i8/u8→f16, i16/u16/char→f32, i32+→f64); **float16 is BYTE-EXACT** via native `Half.Asinh/Acosh/Atanh` (which compute in float32 = NumPy's `astype 'e'->'f'`, `(Half)asinhf((float)h)` — verified 0 finite-diffs over all 65536 f16 values, and faster than the double bridge the rest of the arc-trig f16 tier still uses); Decimal via the decimal→double bridge (valid-domain 15-sig-fig; out-of-domain NaN/inf throws `OverflowException` on `(decimal)NaN` — a pre-existing bridge limitation shared by ALL decimal transcendentals: `arcsin(2m)`/`sqrt(-1m)`/`log(0m)` behave identically, no NumPy decimal analog). **Complex128** is derived from the byte-exact `Asin`/`Acos`/`Atan`(=`Catanh`) ports through NumPy's own msun involution `I·conj(·)` — a pure component-swap (`(z.Im, z.Re)`, zero arithmetic): `asinh(z)=swap(asin(swap z))`, `atanh(z)=catanh(z)` (already ported, drives `atan`), `acosh(z)=cacosh_formula(acos(z))` — so they inherit the whole complex-unary family's documented **≤3 ULP** envelope (the three identities are bit-exact inside NumPy itself, verified 0-diff over 20,036 inputs). `arccosh` inherits `arccos`'s one sub-DBL_MIN-imaginary pathological edge (`[Misaligned]` branch 7). Gates: `Math/InverseHyperbolicTests.cs` (16) + `NpApiOverloadTests_UnaryMath` regions + the `unary_extra`/`specials` fuzz tiers (all 14 dtypes × layouts). See `Math/np.{arcsinh,arccosh,arctanh}.cs`, `Utilities/NDComplexMath.cs`.

**`np.sinc`** — the normalized sinc `sin(pi*x)/(pi*x)`, a port of NumPy 2.4.2 `numpy/lib/_function_base_impl.py::sinc`
(probed against 2.4.2; gate `Math/np.sinc.Test.cs` (15) + the differential-fuzz `sinc` tier). It is a plain FUNCTION,
not a ufunc — the signature is `sinc(x)` with NO `out=`/`where=`/`dtype=`. NumPy's five-line composition is
`y = pi*x; eps = finfo(y.dtype).eps if y.dtype.kind=='f' else 1e-20; y = where(y!=0, y, eps); sin(y)/y` — where the
zero-replacement fills the removable singularity at every EXACT zero of `pi*x` (x==±0, plus any x so tiny that `pi*x`
underflows) with its limit value 1. **Two behaviours are load-bearing:** (1) DTYPE follows `pi*x`, a NEP 50 weak-float
promotion — bool / EVERY integer width / Char all → **float64** (NOT np.sin's i8→f16 / i16→f32 tiers), while
float16/float32/float64 are **preserved** and complex → complex128; (2) the zero test is on `pi*x`, not `x`
(the `!= 0` condition is verbatim, so sub-underflow denormals are handled exactly like NumPy). **Implementation is
ONE fused `np.evaluate` pass** — the pi multiply, the zero→eps select, `sin`, and the divide are folded into a single
inner-loop kernel over the shared `t=pi*x` subtree, so the operand is read once and the result written once (NumPy
materializes ~3 intermediates). **BIT-EXACT with NumPy 2.4.2 for every REAL dtype** — validated EXHAUSTIVELY on all 65,536 float16 inputs and over
1,000,000 adversarial float32 AND float64 inputs (subnormals / any-bit-pattern / near-integer-multiples / specials),
plus exhaustive int8/uint8/int16/uint16 — 0 diffs, because float64 arithmetic + divide are IEEE-exact and NumSharp's
`sin` is a bit-exact port of NumPy's float32 kernel / shares the scalar `ucrtbase` `Math.Sin` at float64.
**COMPLEX128's per-component BITS are NOT reproducible** and it is EXCLUDED from the byte corpus (pinned by an
`allclose` unit test): sinc composes `sin ∘ divide`, and NumSharp's complex sin differs from NumPy's UCRT `csin`
within its ≤3-ULP-relative envelope, so the individual re/im bits differ. The complex VALUE is nonetheless
ACCURATE — validated **≤~2.5 ULP RELATIVE** (max 5.65e-16 relative error) with **np.allclose(rtol=1e-5,atol=1e-8)
passing on ALL 200,000 random samples**; a raw per-component ULP only looks large (thousands) where one component
is tiny relative to `|z|`, which is a byte-reproducibility fact, not an accuracy one.
Decimal (no NumPy analog) rides the same fused expression through the decimal→double bridge. Because sinc is a
`sin`-composition, its float64 cells are host-libm-sensitive at the near-zero-crossings (`sin(pi*k) ≈ 0`), so the
`sinc` fuzz tier is **HOST-PINNED** to win-amd64 (`RunHostLibmCorpus` — `Inconclusive` off-Windows), like the sibling
`unary` tier. **Perf (NPY/NS, Release, best-of, warm):** **≈1.5–2.5× at 100K and 10M** (f64/int 2.2–2.5×, f32
~1.5–2.0× — the single-pass fusion halves NumPy's memory traffic); at 1K sinc sits at the per-op NDIter-setup floor
(~parity), sin being the dominant cost on both sides there. See `Math/np.sinc.cs`.

**NaN operand order of float add / multiply — a win-amd64 pin (2026-10-03).** When BOTH operands of a float32 /
float64 add or multiply are NaN, the result is one operand's NaN, and the COMPILED LOOP picks which: x86
`addpd`/`mulpd` return their FIRST source operand quieted, and the operations are commutative, so the compiler
chooses the order. numpy 2.4.2 win-amd64 (MSVC 19.44), probed on every path:
- add / multiply keep the SECOND operand's NaN: contiguous, strided, reversed, `out=`, `where=`, transposed,
  buffered 2-D broadcasts, 0-d × 0-d;
- except the AVX2 scalar-broadcast SIMD loop (one input stride 0, the other input and the out contiguous, length > 2
  vectors). Its blocks of 2 vectors (8 f64 / 16 f32 lanes) keep the BROADCAST operand's NaN, and its scalar tail the
  ARRAY operand's. The split restarts per inner-loop call: per `where=` run, per `np.getbufsize()` chunk when an
  operand is cast (a cast 0-d or ≤ bufsize 1-D input is copied whole: one call);
- subtract / divide keep the FIRST operand's NaN; float16 the second everywhere; scalar math (np.float64) the first.
- The reductions and scans run their own compiled orders, and NumSharp does NOT follow them yet. `sum` / `prod` use a
  pairwise order (over 9 NaNs with payloads 1–9, `sum` returns payload 8 and NumSharp payload 1): the flat `sum`
  now runs the same pairwise ORDER, but MSVC's leaf loop keeps the accumulator's NaN and its tree combine the second
  operand's, which the emitted folds do not encode (an `[OpenBugs]` pin). `cumsum` / `cumprod` (= `add.accumulate` /
  `multiply.accumulate`) keep the NEW element's NaN at each step (payloads 1, 2, 3, …). NumSharp's scan kernels keep
  the accumulator's (1, 1, 1, …): an open bug (`DirectILKernelGenerator.Scan.cs`, pinned below).

NumSharp encodes the order EXPLICITLY (`Backends/Kernels/Direct/DirectILKernelGenerator.NaNOperandOrder.cs`), because
RyuJIT may swap a commutative operation's operands, so the IL operand order proves nothing:
- vector: `(va & (vb == vb)) op vb` — a NaN in `vb` zeroes `va`, so the hardware can only return `vb`;
- scalar: `r = a op b; if (r != r && b != b) r = b + b`;
- broadcast: a COLD region path (`EmitBroadcastNaNRegion`), taken only when the broadcast value is NaN. The hot loops
  keep the plain instruction (the order is unobservable then). The NDIter, `out=` and `where=` routes pass the rule
  (`BroadcastNaNRule`: split per call, buffer chunk) as kernel auxdata, computed by
  `DefaultEngine.NumPyBroadcastNaNRule`.

Wired into the MixedType kernels (SimdFull / ScalarFull / SimdScalarLeft / SimdScalarRight / SimdChunk / General), the
Tier-3B inner-loop shell plus its strided broadcast dispatch, the 0-d × 0-d scalar delegate and NDExpr's
`BinaryNode`. A single np.evaluate node with a 0-d operand delegates to the direct kernel.

**Linux NumPy (GCC build) orders the operands DIFFERENTLY on the same CPU** (probed through WSL): its contiguous SIMD
body keeps the first operand's NaN and its tail the second's; its broadcast loop keeps the broadcast operand's NaN
everywhere; its strided / `out=` / `where=` paths keep the first. The order is a compiler artifact, pinned to
win-amd64 like the float16 and complex NaN-sign contracts.

Gates:
- oracle `nan_order.jsonl` (2,796 cases): NaN payloads compare raw through `params.nan_bits` on x86 hosts with
  X86_V3. On the fix's parent (2f92d91d) 936 of the first 2,678 cases fail. Section G (118) holds the paths a
  2026-10-03 gap probe validated as matching: float16 scans, complex128 subtract, the buffered N-D / cast / `where=`
  / evaluate / `dtype=` paths;
- `Math/NaNOperandOrderTests.cs` (13 tests). 12 fail on the pre-fix build. It includes the fuzz finding
  `max(lxor(isnan(b), a*0.0), (a+m)*(a+b))`, whose result is now SHA-256-identical to NumPy's.

Residuals — `[OpenBugs]` with NumPy's raw bits in `test/NumSharp.Tests/OpenBugs.NaNOperandOrder.cs` (7 tests, one per
class; the probe found 272 differing cases). Most are places where NumPy makes MORE inner-loop calls than NumSharp's
kernels, so the region rule restarts where NumSharp keeps the second operand's NaN:
- `cumsum` / `cumprod` of float32 / float64 / complex128 (above; complex128 per part);
- complex128 add (array layouts), multiply and divide — MSVC's complex loops choose the NaN per part and, for add,
  per unrolled vector pair. NumSharp's complex `cumprod` also multiplies element 0 by the identity instead of copying it;
- N-D broadcasts NumPy runs one row per call: an outer product with two rows such as `(2,1)+(1,64)`, and
  `(m,n)+(m,1)` with n > 4096;
- a 0-d against an N-D view NumPy buffers (a row slice `x[:, :k]`, reversed rows or columns, a 3-D slice): the
  region rule per buffer fill of whole rows;
- a reversed cast operand, from 9 elements on;
- `where=` together with a cast;
- np.evaluate trees deeper than one node with a NaN 0-d or size-1 broadcast input.

Cost: 2 extra µops per add / multiply vector. No measurable change at ≥ 4K elements, ≤ ~5–10 % at 1K (inside the
host's ±10–15 % run-to-run noise). A NaN 0-d broadcast got faster (0.60–0.80× the old time) through the fill path.

**ufunc `out=` / `where=` parameters** are supported on the elementwise core (NumPy semantics, probed against 2.4.2): binary `add`/`subtract`/`multiply`/`divide`/`true_divide`/`mod`/`power`/`floor_divide`/`arctan2`/`bitwise_and`/`bitwise_or`/`bitwise_xor`, unary `sqrt`/`exp`/`log`/`sin`/`cos`/`tan`/`abs`/`absolute`/`negative`/`square`/`log2`/`log10`/`log1p`/`exp2`/`expm1`/`cbrt`/`sign`/`floor`/`ceil`/`trunc`/`reciprocal`/`sinh`/`cosh`/`tanh`/`arcsin`/`arccos`/`arctan`/`deg2rad`(`radians`)/`rad2deg`(`degrees`)/`invert`(`bitwise_not`)/`rint`. `round_`/`around` take `out=` ONLY (np.round is a function, not a ufunc — no where/dtype; decimals≠0 cast errors name ufunc 'multiply' per NumPy's composition). `rint` is the TRUE ufunc form of round-half-to-even: unlike `round_`/`around` (which preserve integer dtype) it is float-tier (bool/i8/u8→f16, i16/u16→f32, i32+→f64, floats/complex preserved) and reuses `UnaryOp.Round`'s kernel (complex rounds real+imag; `dtype=<int>`→no-loop). floor/ceil/trunc have IDENTITY loops on every bool/int dtype (dtype preserved; np.round's int path is an identity copy); the loop dtype comes from the input tier (`sinh(i1, out=f8)` stores float16-precision values); reciprocal int 1/0 → signed MinValue (NumPy 2.4.2); sign/positive reject bool with the verbatim no-loop UFuncTypeError; bitwise/invert raise the no-loop TypeError for float inputs (probed order: bad where → no-loop → out-cast → shape). `out` joins the broadcast but is never stretched, requires a same_kind cast from the loop dtype (resolved from inputs), returns the same instance, and may alias an input (overlap-safe via COPY_IF_OVERLAP). `where` must be bool, broadcasts and joins the output shape; masked-off `out` slots keep prior contents. Engine plumbing: `Backends/Default/Math/DefaultEngine.UfuncOut.cs`. Like NumPy's ufunc (`try_trivial_single_output_loop` runs for a provided out too), a binary call whose operands are all trivially iterable — one dtype, out C- or F-contiguous, array inputs of out's dims in its order or one size-1 input, out disjoint from each input or the input itself — skips the iterator and calls the DirectIL kernel the fresh result uses (`TryTrivialBinaryOpInto`, `DefaultEngine.BinaryOp.cs`; 2026-10-04): float16 add/subtract/multiply/divide into an out ran the iterator's scalar body at 3.9 ns/element and now take the SIMD widen-compute-narrow kernels (0.55); other dtypes save the iterator's construction. For those four float16 ops the route also takes BROADCAST and STRIDED inputs into a C or F out (`TryHalfArithBroadcastInto`, 2026-10-06, plan `docs/plans/api-parameter-parity.md` X15: the strided widen-compute-narrow kernel walks the out's memory order; an input that is the out itself runs in place, any other overlap stays with the iterator's copy): 3.6–7.4× NumPy, where the iterator's scalar float16 body was ~5 ns an element. Gate: `Backends/Kernels/TrivialBinaryOutTests.cs` (every dtype through both routes, byte-compared, with the declines and the overlap rule) + `Math/HalfArithKernelTests.cs`.

**Each of those ufuncs exposes ONE NumPy-shaped overload** — `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)` mirroring NumPy's `f(x1[, x2], /, out=None, *, where=True, dtype=None)`: `out` is the second/third positional slot exactly like NumPy, `where`/`dtype` are reachable by name without `out`. `dtype=` selects the LOOP (NumPy loop-signature semantics, probed): computation runs at that precision even with `out=` (`sqrt([2.], out=f64, dtype=f32)` stores the f32-rounded value; `add(0.1, 0.2, out=f64, dtype=f32)` stores `0.30000001…`), `power(2, -1, dtype: f64) = 0.5` (the negative-int-exponent ValueError applies only to integer loops), `power(10, 11, dtype: f64) = 1e11` exactly (no compute-then-cast), `add(True, True, dtype: i32) = 2` (the bool→logical-OR remap keys off the FINAL loop dtype), `negative(bool, dtype: f64)` is legal, and inputs must reach the loop via same_kind casts (verbatim `Cannot cast ufunc '<name>' input [N] from ...` errors; binary errors are indexed, unary are not). Loop-existence gates raise `No loop matching ... ufunc <name>`: float-only ufuncs (sqrt/exp/log/trig + `divide`/`true_divide`) reject int/bool dtype; bitwise rejects float/complex/decimal dtype. `positive` is a full ufunc (identity loops at every dtype EXCEPT bool: plain `positive(bool)` and `dtype: bool` raise the verbatim `did not contain a loop with signature matching types <class 'numpy.dtypes.XDType'> -> ...` texts; `positive(bool, dtype: f64)` works). `round_`/`around` follow NumPy's non-ufunc shape `round(a, int decimals = 0, NDArray out = null)` (2nd positional is decimals, NOT out). Positional-dtype overloads (`np.sqrt(x, NPTypeCode.Single)`, `(x, Type)`) also exist for source compat as non-NumPy call forms (NumPy's 2nd positional is `out`). Tests: `Math/UfuncDtypeOverloadTests.cs`.

**Five float32 unary loops are BIT-EXACT with NumPy**, not merely close — `exp`, `log`, `sin`, `cos`, `tanh` —
because `Utilities/NDFloatMath.cs` ports the kernels NumPy 2.4.2 actually runs rather than calling the platform
libm: `simd_exp_FLOAT` and `simd_log_FLOAT` (`numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`, the
`SIMD_AVX2_FMA3` instantiation — on Windows the AVX-512 variant is `!defined(_MSC_VER)`-gated and never compiled)
and `simd_sincos_f32` (`loops_trigonometric.dispatch.cpp`, `NPY_SIMD_FMA3`). `rad2deg`/`deg2rad` join them by
forming their constant at the OPERAND's precision, as NumPy's `RAD2DEG`/`DEG2RAD` macros do. None of these kernels
is correctly rounded — NumPy documents 2.52 ULP for exp, 3.83 for log, 0.647 for the sine polynomial — so
`MathF.Exp`/`Log`/`Sin`/`Cos` (~correctly rounded) differed from NumPy on **35% / 4.7% / 6.9% / 7.8%** of float32
inputs, and the one-ULP-low `rad2deg` factor on **79%**; the blanket `unary ~ULP` fuzz excuse was absorbing all of
it silently. Each is now verified **exhaustively: all 2³² float32 bit patterns**, through both the SIMD kernel and
the scalar entry point (chunked-checksum sweep). The corpus tier `numpy_f32_kernels.jsonl` (140 cases) pins the
discriminating inputs in CI and `MisalignedRegistry`'s unary-ULP branch EXCLUDES these ops at a float32 result, so
a 1-ULP regression turns the gate red.

**`tanh` is the only one of the five that is also a FLOAT64 port**, and the only one that needed to be: NumPy ships
its own table-driven kernel at BOTH widths (`simd_tanh_f32`/`simd_tanh_f64` in
`numpy/_core/src/umath/loops_hyperbolic.dispatch.cpp.src`, converted by NumPy from Intel SVML), where exp/log/sin/cos
fall through to the platform's scalar `npy_*` at f8 and already agreed. So `MathF.Tanh`/`Math.Tanh` differed from
NumPy on **9.7% / 8.1%** of inputs (≤2 ULP / 1 ULP). tanh is odd, so the kernel works on `|x|` and ORs the sign back:
`[0, SATURATION_THRESHOLD)` splits into **32 subintervals at f32 / 16 at f64**, the exponent field of `|x|` indexes a
table of per-subinterval minimax coefficients plus a recentring offset `b`, and the answer is a degree-**6 / 16**
Horner evaluation of `P(|x| - b)`. The top subintervals hold `1 + 0·y + 0·y² …`, which is how the saturated range
returns exactly ±1 through the same arithmetic. **Unlike exp/log/sincos the FMA here is not a host pin** — that kernel
spells its Horner steps as `hn::MulAdd`, an *explicit* fused multiply-add, so `FusedMultiplyAdd` is a literal
transcription rather than a bet on MSVC's contraction. Verified over **all 2³² float32 inputs** (SIMD + scalar, and a
negative control confirms the checksums detect the pre-port kernel in 34 of 512 chunks) and, at float64, over
**4.83 billion** values — 2³² covering every sign × exponent × 2²⁰ mantissa prefix, plus 5.4×10⁸ full-width
splitmix64 patterns. Gate: `numpy_f32_kernels.jsonl` + the new **`numpy_f64_kernels.jsonl`** (24 cases), with tanh
carved out of the unary-ULP excuse at BOTH widths (`NumPyPortedFloat32Kernels` / `NumPyPortedFloat64Kernels`).

**One deliberate departure from NumPy's own AVX2 shape, and it is what makes tanh fast:** the coefficient fetch is a
**transpose, not a gather**. A subinterval's 8 f32 coefficients are 8 *consecutive* floats — exactly one `Vector256` —
so the whole table read is 8 contiguous row loads plus an 8×8 transpose, where a gather-per-coefficient issues 8
`vgatherdps` (64 element loads for the same 8 rows). Both were implemented and measured, bit-identical against the
same table: transpose **0.071 ms** vs gather 0.124 ms at 100K and **7.41** vs 12.87 ms at 10M — 1.7×, the difference
between losing to NumPy and beating it. float64 stays **scalar on purpose** (its lookup would cost 18 gathers per 4
lanes, and the scalar kernel already outruns NumPy); `Vector512` is composed from two verified `Vector256` halves
rather than an AVX-512 gather that cannot be tested on this host.

**Perf (NPY/NS, Release, best-of-21 warm, `out=`, 100K / 10M):** `tanh` **1.78× / 1.82×** at float32 and
**1.90× / 1.92×** at float64 — up from 0.18×/0.19× and 0.70×/0.68×, i.e. **9.9×/9.7×** and **2.7×/2.8×** faster than
the `MathF.Tanh`/`Math.Tanh` calls it replaces.

Specials (probed, and mostly consequences of NumPy blanking NaN lanes before its range compares): any NaN — quiet,
signalling, either sign, any payload — returns the canonical `0x7fc00000` (note .NET's own `float.NaN` is
`0xffc00000`); `exp` saturates at `x ≥ 88.72283935546875` → `+inf` and `x ≤ -103.97208404541015625` → `+0`, with
the band between producing subnormals through `fma_scalef_ps`'s exponent split; `log` returns `-inf` at ±0 and a
**negative** NaN for a negative argument (but a positive one for a NaN argument); `sin`/`cos` hand arguments past
their Cody-Waite limits (**different per function** — 117435.992 and 71476.0625) to the platform libm exactly as
NumPy does, so those inputs are libm-dependent by construction. NumPy also raises FP status flags
(`RuntimeWarning`) that NumSharp does not model — pre-existing, not introduced here.

**Host pin:** each kernel's quadrant is computed with a FUSED multiply-add because MSVC 19.44 (the numpy==2.4.2
win-amd64 wheel's compiler) contracts NumPy's separate multiply and add. It is observable at `x = 0xc26d0e6c`,
where `x·log2 e` lands on the exact tie `-85.5` and the two spellings round `k` to -85 vs -86 — a 1-ULP
difference. Same pin class as the MSVC-pinned cast kernels (`Fuzz/README.md` → "Host-dependent values").

**Perf (NPY/NS, Release, best-of-21 warm, `out=`, 100K / 10M):** `exp` **1.11× / 1.11×**, `log` **1.28× / 1.27×**,
`sin` **1.07× / 1.04×**, `cos` **1.05× / 1.06×**, `rad2deg` **11.7× / 2.42×** — NumSharp is now FASTER than NumPy
on every one, having been 0.33×/0.41×/0.20×/0.21× before. The ports are 1.4–5.2× faster than the `MathF.*` calls
they replace because they vectorize: `Utilities/NDFloatMath.Simd.cs` carries `Vector{128,256,512}<float>`
overloads (each algorithm is purely elementwise, so lane `i` is bit-identical to the scalar call — which is why
NumPy generates its own kernels from one template for `__m256` and `__m512`), gated by
`DirectILKernelGenerator.NumPyFloatKernelSimdAvailable` on hardware FMA exactly as NumPy gates its own; without
FMA the scalar entry point runs, same bits, one lane at a time. Two measured details worth keeping: the
rarely-taken subnormal branches are kept as BRANCHES rather than folded into branchless selects (folding exp's
denormal split cost ~0.17 ns/element and log's ~0.15 — a fifth of each kernel), and the vector overloads carry
`AggressiveOptimization` because without it tier-0 JIT made the first ~40 calls 27× slower than steady state.

**`exp2` (float32) is a fast SIMD port but — unlike the five above — NOT bit-exact, by necessity.** The numpy
2.4.2 win-amd64 wheel has **no vector exp2**: its SVML `simd_exp2` (`loops_umath_fp.dispatch`) is gated on
`NPY_HAVE_AVX512_SKX && NPY_CAN_LINK_SVML` (Linux/AVX-512 only), so Windows runs the SCALAR fallback `npy_exp2f` =
the MSVC CRT `exp2f`, one element at a time (measured 3× slower than NumPy's own SIMD `exp`). There is therefore
no reproducible NumPy float kernel to match, and a pure-float polynomial cannot reach the CRT's near-correctly-
rounded result — float rounding of the Horner chain alone leaves ~1 ULP on ~4.5% of inputs regardless of degree.
So `NDFloatMath.Exp2` computes `2^x = 2^n · 2^r` (`n = rint(x)`, `r ∈ [-0.5, 0.5]`) with **`2^r` evaluated in
DOUBLE** — a degree-8 minimax, 2^-37.7 relative error, ~13 bits below float32 — narrowed ONCE to float and then
scaled by the integer power `2^n` straight in the exponent field (exact, so it adds no second rounding for any
normal result). Result: **≤ 1 ULP vs NumPy on every input, ~0.2% differ**, all producing subnormal outputs (where
the narrow + `ScalefDenormal`-style denormal split double-rounds) — the SAME accuracy class as the old
`(float)Math.Pow(2, (double)x)` path it replaces (which was 0.04% at 1 ULP but ~5× slower than NumPy's scalar
loop). The hybrid is what makes it fast: a **full-double** scale via `Vector256.ConvertToInt64` was measured at
**343 µs / 100K (SLOWER than NumPy)** because that convert is AVX2-emulated, so the design keeps the `2^n` scale
in cheap float integer ops and widens to double ONLY for the poly. Vectorized in `NDFloatMath.Simd.cs`
(`Vector{128,256,512}<float>` — the floats widen to two `Vector{}<double>` halves via `WidenLower`/`WidenUpper`,
the poly runs per half, `Narrow` packs `2^r` back), gated by the shared `NumPyFloatKernelSimdAvailable` on hardware
FMA, lane-identical to the scalar entry point (verified 0-diff over ~7M inputs). It stays under the ≤ 2-ULP
`unary ~ULP` fuzz excuse — deliberately NOT added to the bit-exact `NumPyPortedFloat32Kernels` carve-out, since it
is not bit-exact. Specials match NumPy (probed): NaN → `0x7fc00000`, `x ≥ 128` → `+inf`, `x ≤ -150` → `+0`,
`±0` → `1`, integer `x` exact (`exp2(10) = 1024`). **Perf (NPY/NS, Release, best-of, warm, 1K / 100K / 10M):**
**1.82× / 2.40× / 3.12×** — up from ~0.2× (the `Math.Pow` bridge), i.e. ~12× faster at 100K. Other dtypes
(double/int/Half via `Math.Pow`, complex via `NDComplexMath.Exp2`) are unchanged. See
`Utilities/NDFloatMath.{cs,Simd.cs}` → `Exp2`; gate: the `unary_extra`/`specials` fuzz tiers +
`Math/Exp2KernelTests.cs`.

**Still the platform libm at float32/float64** (measured, documented, NOT ported): `expm1`/`log1p` (31.1% / 30.7%
and 33.6% / 15.1% of inputs differ), which NumSharp additionally computes as
`Exp(x)-1` / `Log(1+x)` — catastrophic for small |x| (`expm1(1e-8)` returns **0** where NumPy returns 1e-8) and a
signed-zero bug (`expm1(-0.0)` → `+0.0`). **Neither is portable** — NumPy calls the CRT for both,
so bit-parity is not reachable the way it was for exp/log/sin/cos/tanh; the expm1/log1p *accuracy* bug is a separate,
genuine defect worth fixing on its own terms (note .NET's `float.ExpM1`/`double.ExpM1`/`LogP1` are themselves just
`Exp(x)-1`/`Log(1+x)` and do NOT help). float16 loops (17 differing values out of all 65,536) and float64
`exp`/`log`/`sin`/`cos` already agree with NumPy bit-for-bit on this platform and need no port.
### Math — Reductions
`all`, `amax`, `amin`, `any`, `argmax`, `argmin`, `average`, `average_returned`, `count_nonzero`, `cov`, `cumprod`, `cumsum`, `diff`, `ediff1d`, `max`, `mean`, `median`, `min`, `percentile`, `prod`, `ptp`, `quantile`, `std`, `sum`, `var`

**The reduce prologue (`ReductionArguments`, plan `docs/plans/api-parameter-parity.md`).** Every `ufunc.reduce`-based
reduction checks its arguments in NumPy's order before it reads an element: a read-only `out=` first ("output array is
read-only", raised while NumPy parses `out`), then a non-bool `where=` mask ("Cannot cast array data|scalar from
dtype(…) to dtype('bool') according to the rule 'safe'"), the axis, a repeated axis, `out=`'s rank, then the reduction
iterator's shape checks over `[out, input, where]` — the input is never stretched by the mask, the mask may not have
more axes than the input, a kept reduced axis of `out=` must be 1 long, an `out=` axis of length 1 cannot broadcast —
with NumPy's texts (`PyUFunc_GenericReduction`, `PyUFunc_ReduceWrapper`, `npyiter_fill_axisdata`, ported once in
`Backends/Default/Math/Reduction/ReductionArguments.cs`). `np.all`/`np.any` run it when `out=` or `where=` is given;
the max/min and sum families reuse it as they gain those keywords. Gate: the `reduce_kwargs` oracle tier (each
reduction through its NumPy-shaped overload; zero excuses) + `Logic/np.all_any.tuple_out_where.Test.cs`. An EMPTY
reduction without an identity raises NumPy's `ValueError` on every route since 2026-10-08 (`zero-size array to
reduction operation maximum which has no identity` from `np.max` / `np.min` / `np.nanmax` / `np.nanmin` / `np.ptp` and
np.evaluate's `Min` / `Max`; `attempt to get argmax of an empty sequence` from `np.argmax` / `np.argmin`); the keyword
route already did, the engine routes raised the base `ArgumentException` (`ValueError` derives from it, so a handler for
that still catches it).

**`np.any` / `np.all` / `np.count_nonzero` along an axis are laid out like NumPy's reduction** (2026-10-09, plan
api-parameter-parity X10, found by the string oracle's `strcomp` section). NumPy allocates the result with
`npyiter_new_temp_array` over the reduce iterator, so an F input gives an F result and a permuted one keeps its memory
order; NumSharp allocated C. The logical reductions now allocate with `Shape.ReductionOutputShape`
(`DefaultEngine.LogicalReductionShape`), `NDAxisIter.ReduceBool` writes each output through the result's own strides,
keepdims re-inserts the axis with `ExpandDimension` (a C `Reshape` copied an F result), and the tuple-axis chain drops its
kept axes with their strides. `np.count_nonzero(a, axis)` is NumPy's own `a.astype(bool, copy=False).sum(axis,
dtype=intp)` (the per-output scalar loop it replaced wrote C order and ran 15–100× slower at 100K–1M), and np.sum's bool
count route (`TryCountBoolAxisSum`) no longer returns np.evaluate's reshaped keepdims VIEW (OWNDATA false; NumPy owns).
A bool C/F operand of ≥ 16K elements reduced to rank ≥ 1 runs np.evaluate's Any / All axis stream (`TryStreamLogicalAxis`;
long contiguous rows ≥ 4096 stay on the engine's per-row scan): 72 cells (C/F × axis × any/all × all-true / sparse /
half-true × 1K / 100K / 1M) geomean 4.2× NumPy, where the engine's strided per-output walk was 0.03–0.09× on F operands
along axis 1. Residuals: the "read every element" F-slab cells (an `all` of all-true data, an `any` of sparse data,
along axis 1) at 0.88–0.92× at 1M — the 1 MB operand streams from L2 at ~38 GB/s on both sides — and 1.12–1.38× at
100K; `all` along a long contiguous axis at 1M 1.25× (the same floor). Gate: `Logic/np.all_any.ResultLayout.Test.cs`
(probed strides for bool / int32 / U inputs, F, permuted, stepped, tuple axes) + the string oracle's `strcomp` section.

**`np.sum` / `np.mean` start from the add identity, and their FLAT float reductions are NumPy's add.reduce, bit for
bit** (`Backends/Default/Math/Reduction/Default.Reduction.Add{,.Exact,.Identity}.cs`, 2026-10-04):
- **The identity.** NumPy's `PyUFunc_ReduceWrapper` assigns the loop's identity (`+0.0` for every inexact loop) and
  only then runs the add loop — for ONE element and a length-1 axis too — so a sum of only `-0.0` is `+0.0` and a
  signalling NaN comes back quiet. `SumOfOneElement` / `SumOverLengthOneAxis` add the lone element(s) to a 0-d
  identity with the binary add kernel; they used to copy through `HandleScalarReduction` / `HandleTrivialAxisReduction`,
  which `amin`/`amax`/`nanmin`/`nanmax` keep (no identity). Both run BEFORE the cast-loop branch (the identity add casts
  the element itself). Inherited by `np.mean` (count division; complex divides by `n+0j` with NumPy's complex loop),
  `np.average`, `np.nansum`/`np.nanprod` (one element: the fused NaN-skipping kernels, which start from the identity),
  `np.nanmean`, `np.trapezoid`, `np.einsum`'s sums and `np.evaluate`'s Sum / Mean / NanSum / NanMean.
- **`dtype=` loops.** A loop that is neither the input nor its NEP 50 accumulator casts every element first
  (`ReduceAddInCastLoop`: float→int truncates, complex→real keeps the real part, `dtype=bool` is logical OR). An AXIS
  cast loop takes the axis kernel's own float32→float64 widening only where that is NumPy's sequential order (the
  reduced axis not the innermost, `IsSequentialWideningAxisSum`), else casts in blocks of `CastLoopBlockElements` (128K)
  along the outermost kept axis; a FLAT integer / bool cast loop (order-free: modular addition, OR) casts and sums in
  blocks along the outermost axis in memory (`np.sum(f64, dtype=int32)` at 10M: 6.8 → 4.5 ms, NumPy 7.0 ms).
- **The flat engine** (`NumPyAddReduce`; float16 / float32 / float64 / complex128 loops, the input already the loop
  dtype or cast into it). NpyIter's call schedule: axes sorted by |stride| and coalesced signed
  (`NumPyMinMaxReduce.FlatIteratorAxes`), then `npyiter_find_buffering_setup`'s cost model (`FlatBufferingDim`; a cast
  operand starts at cost 2): one run is one call in place (any stride; a reversed run in logical order); several runs
  whose inner run exceeds half the buffer are one call each; otherwise the operand is buffered in fills of
  `core·⌊bufsize/core⌋` that never cross an outer block (a cast run: fills of bufsize). Every call is
  `*out += pairwise_sum(call)` (real: 8 accumulators per 128-element leaf, split `n/2` down to a multiple of 8; complex
  `CDOUBLE_pairwise_sum`, 4 accumulators per part; float16's HALF_add narrows the running float32 sum per call). Fills
  follow `np.getbufsize()`. A fill is streamed: ≤1024-element sub-ranges of its pairwise recursion, each folded IN PLACE
  when it lies inside one run and the fold reads that stride with SIMD loads (the real part of a complex128 array is a
  stride-2 float64 run), else copied/cast into an L1 block — the same bits, no 64 KB buffer. `np.mean` is `_mean`:
  int/bool → float64 loop, float16 → float32 loop with a float16 result (keepdims rounds twice, float64 → float32 →
  float16), the quotient by the count in float64; complex via Smith's division.
- **The folds** (`ILKernelGenerator.Reduction.Pairwise{,.Runs}.cs`, `.Half.cs`): run kinds contiguous, reversed (load
  + lane reversal) and stride 2 (two overlapping loads + blend / permute, never past the last element), a two-leaf
  interleave for nodes whose halves are both leaves, complex two per register at any stride, float16 runs and an AVX2
  gather; `PairwiseFoldIsVectorized` says which strides fold with SIMD.
- **Perf** (NPY/NS, one pinned P-core, best-of-15, 272 cells, every result SHA-256-checked against NumPy): geomean
  **3.73×**, no cell under 1.0 — one element 4.28×, length-1 axis 4.51× (float16 sums 5.5–6.5×), `dtype=` loops 3.53×,
  complex mean 2.38×, float16 flat sum 8.83×, flat sum over 5 layouts × 4 float dtypes 3.36×, flat mean 3.52×. The 21
  cells under 1.5× are memory-bound on both sides: 15 at 10M (DRAM — one core reads at most 27.8 GB/s on this host;
  `np.sum` of float64 runs at 85 % of that, NumPy at 55 %, so 1.55× is the ceiling), 5 complex128 stride-2 / reversed
  / transposed cells at 100K–1M (L2/L3 — `np.sum(c128[::2])` at 100K takes 41–47 µs across runs, as fast as a bare
  loop of the same 128-bit loads at 47 µs; NumPy 52 µs), and one 1.4 µs call at 1K (1.47×, run-to-run noise).
- **Residuals:** a broadcast operand declines to the old fold (value-exact, not order-exact); a sum of SEVERAL NaNs keeps
  another NaN's payload (MSVC's `pairwise_sum` operand order, an `[OpenBugs]` pin); the multi-element nan-reductions
  keep their own folds (the W4-B excuse); an axis reduction in a cast loop whose reduced runs exceed the buffer is not
  modeled (pinned by `OpenBugsReductionTests.Sum_CastLoopAxis_RowLongerThanTheBuffer_SplitsAcrossFills`, plan X6;
  np.var / np.std with `dtype=` inherit it), except when the axis leaves ONE output (a 1-D operand's only axis, or siblings all of extent 1): NumPy runs
  that as the flat schedule, and so does NumSharp since 2026-10-05 (`TryExactOneOutputAxisSum`, plan
  `docs/plans/api-parameter-parity.md` X5; the axis route summed the whole cast copy in one call), and a one-output
  REVERSED axis takes the same route without a cast (the axis kernel walks it in memory order, NumPy in logical
  order; with several outputs that is still open).
- **Gates:** `sum_identity.jsonl` (23,953 cases, zero excuses; section "s" = 704 order-sensitive flat schedule cases
  under `np.setbufsize` 16 / 32, section "o" = 832 one-output axis schedules), `reduce_start.jsonl` (zero excuses), unit `Math/np.sum.{Identity,FlatSchedule,
  CastLoopBlocks}.Test.cs` and `Backends/Kernels/{PairwiseFoldRunKind,CopyRunSubword}Tests.cs`. The gist ports' 150 live
  metric scalars are byte-exact since (134 before; their tests are 0-ULP now).

**`np.var` / `np.std` are NumPy's `_var` / `_std`, statement for statement**
(`Backends/Default/Math/Reduction/Default.Reduction.VarStd.cs`, `VarStdNumPy`; 2026-10-05, plan
`docs/plans/api-parameter-parity.md` A2). The old moment kernels computed a two-pass variance in their own order and
answered one element per output without reading it: close values, other bits. Now every step is NumPy's and reuses the
engine's exact pieces:
- **The statements.** `rcount` after the axis is normalized (an axis on a 0-d array is NumPy's AxisError). The loop is
  `dtype=`, else float64 for an integer / bool / Char input, else the input's dtype. `arrmean = umr_sum(keepdims=True)`
  through `ReduceAdd` (np.sum's exact schedules), divided by the count in place as NumPy does
  (`DivideByCountLikeNumPy`: float16 / float32 in float64, complex by Smith's division, an integer loop truncates).
  `x = arr - arrmean` in NpyIter's layout (`DeviationLayout`: `Shape.NpyIterOutputShape` over `arr` and the mean's
  K-order keepdims layout; NumSharp's own keepdims result is C for a permuted input and would out-vote `arr`). The
  square in NumPy's three forms (`x*x`; `real² + imag²` of a complex `x` from a complex or bool input;
  `(x*conj(x)).real`). `ret = ReduceAdd(x², axis, dtype) / max(rcount - ddof, 0)`: NaN or `+inf`, never negative. std
  is `sqrt(out=ret)`: an integer loop raises NumPy's `Cannot cast ufunc 'sqrt' output from dtype('float64') to
  dtype('int32') with casting rule 'same_kind'` for an array result and truncates a scalar one.
- **No x² buffer where NumPy's order allows it.** A flat reduction over one memory run with no cast streams
  `Sum(Square(arr - mean))` through `np.evaluate` (the streamed pairwise sum, bit-identical). Otherwise one `np.evaluate`
  pass fills the squared deviations (`SquaredDeviationExpr`), or the eager ufuncs do for float16 and decimal (NDExpr's
  float16 costs ~8 ns an element). Above `VarStdChunkMinBytes` (8 MiB of x²) the deviations are computed and summed in
  chunks, each route byte-identical to the one-buffer route:
  - output chunks along the outermost kept axis (`ChunkedDeviationSum`);
  - row blocks with a running `np.add(acc, slab, out: acc)` per slab when the reduced axis is outermost
    (`RowBlockDeviationSum`, NumPy's SLAB order);
  - fills of `np.getbufsize()` folded by `NumPyAddReduce.TryReduceFlat` for a flat cast loop (`FlatFillDeviationSum`).

  `DefaultEngine.VarStdChunkBytesTestOverride` forces them; the oracle's `VarStdChunked` replays the whole tier
  through them.
- **Decimal:** exact decimal arithmetic in np.sum's order (sequential, memory order) and `DecimalMath.Sqrt`; a zero
  divisor raises `DivideByZeroException` (decimal has no NaN).
- **Perf** (NPY/NS, one pinned P-core, best of rounds, 116 cells: float64 / float32 / int32 / int64 / complex128 /
  float16 / bool × var and std flat, var per axis × 1K / 100K / 10M, plus strided, F, `ddof=1` and `dtype=` cells for
  the two float dtypes; every result checked): geomean 3.07×, min 1.33×. The 8 cells under 1.5×: the 1K axis cells
  (1.33–1.43×, the fixed cost of the building blocks: the same calls by hand take 4.8 µs, NumPy's `_var` 6.9 µs), bool
  along axis 0 at 10M (1.38×, np.sum's bool cast-loop axis route, plan X7), float32 along axis 0 at 100K (1.50×).
- **Residuals:** np.sum's cast-loop rows longer than the buffer (above, plan X6) reach var / std with `dtype=`;
  `ma.var` / `ma.std` are numpy.ma's own statements since 2026-10-06 (plan A6, the paragraph after np.mean's); the old
  kernels (`DirectILKernelGenerator.Reduction.Axis.VarStd.cs`, `.Masking.VarStd.cs`) have no caller left in var / std
  (plan X8).
- **Gates:** `varstd.jsonl` (16,108 cases, zero excuses: layouts × dtypes × axis × keepdims × ddof, `ddof >= n`,
  `dtype=` × 10 loops, special values, `np.setbufsize` fills, order-sensitive sizes) + `VarStdChunked`,
  `decimal_varstd.jsonl`, unit `Math/np.var.NumPyPort.Test.cs` (15).

**`np.mean` is NumPy's `_mean`, statement for statement** (`Backends/Default/Math/Reduction/Default.Reduction.Mean.cs`,
`MeanLikeNumPy`; 2026-10-05, plan `docs/plans/api-parameter-parity.md` X9). Along an axis the old kernel read a cast
loop's elements as doubles in one sequential pass, divided a complex sum part by part and summed float16 in double; an
integer / bool loop threw on an empty input; the flat fallback summed a broadcast uint64 in a wrapping accumulator.
- **The statements.** `rcount` (an `np.intp`, a STRONG int64; np.mean normalizes the axis itself, so a 0-d input
  refuses every axis); the loop (`dtype=`, else float64 for bool / integers, float32 for float16 with a float16
  result); `ret = ReduceAdd(arr, axis, keepdims, loop)` (np.sum's exact schedules: a cast loop casts each element, an
  integer loop wraps, the bool loop is OR); the division in place: the engine's dense-block division where it is
  NumPy's quotient (float64, complex by Smith's division, float32 while `n <= 2^24`), else A2's
  `DivideByCountLikeNumPy` (the float64 loop, an unsafe cast back: an empty slice's `0 / 0` is INT_MIN, or True). A
  float16 result rounds twice for an array (float64 → float32 sum → float16) and once for a scalar, NumPy's two
  branches.
- **Routes.** Every float / complex loop along an axis but a length-1 one (its identity add is exact) and every integer
  / bool loop take `MeanLikeNumPy`; the flat float loops keep `TryExactFlatMean` (already `_mean`). A flat operand that
  engine declines (a broadcast one) keeps the elementwise fold only where it is NumPy's value: the float64 loop over a
  narrow integer / bool / char input whose sum `size × max|x|` stays within 2^53 (an int64 / uint64 input, a larger
  sum, another loop and decimal take `MeanLikeNumPy`; the fold also costs no memory on a huge broadcast).
- **np.sum's integer / bool axis routes** (plan X7's done part, which this needed): an integer → float64 axis sum runs the
  widening kernel when the sum is NumPy's in every order (32 bits or less within the 2^53 bound) or the reduced axis
  is not the innermost (int64 / uint64: sequential per output, NumPy's slab order) — `IsExactIntegerWideningAxisSum`;
  a bool axis sum (int64 or float64 loop) counts nonzero elements with np.evaluate's CountNonzero axis stream for C / F
  operands (`TryCountBoolAxisSum`). np.mean(bool (2500,4000), axis=0) 67.6 ms → 0.18 ms (NumPy 5.1).
- **Perf** (NPY/NS, one pinned P-core, 54 cells): geomean 2.62×. Under 1.5×: the same-type slab means at 100K / 10M
  (1.0–1.35×, DRAM-bound at 10M), the float32 / float16 cast loops along axis 0 at 10M (int32 → float32 0.31×, float16
  0.80×: plan X7's open part, the blocked cast copy gathers columns), an int64 innermost axis 1.20×.
- **Residuals:** an axis reduction over a view with a reversed axis walks memory order (np.sum's finding 5); the flat
  sum of a broadcast float operand is not NumPy's per-row call schedule (finding 10); np.sum's X6 cast-loop rows. (The
  bool loop's result layout over a permuted input — np.any's, plan X10 — is NumPy's since 2026-10-09: the logical
  reductions allocate in the input's memory order.) The oracle holds np.mean strict in every tier
  (`ByteExactArithmeticUnaryOps`, out of the reduction blankets; `ma.mean` too since A6) but for a narrow finding-5
  branch (`IsReversedAxisReduction`).
- **Gates:** `mean_loops.jsonl` (6,550 cases, zero excuses), unit `Math/np.mean.NumPyPort.Test.cs` (9), the
  tightness tests `Mean_NoDivergenceIsExcused` / `Mean_ReversedViewAxis_IsTheOnlyExcusedLayout`.

**`np.ma.mean` / `np.ma.var` / `np.ma.std` / `np.ma.anom` are numpy.ma's statements** (`src/NumSharp.Core/Ma/NDMaskedArray.cs`;
2026-10-06, plan `docs/plans/api-parameter-parity.md` A6). They used to be a masked reduction of their own: a slice NumPy
masks carried NaN, an unmasked inf stayed where NumPy masks it into 0.0, and the variance kept the divide's mask.
- **Without a mask**: ndarray's `_mean` / `_var` run ON the masked array, so their `true_divide(out=...)` meets
  `MaskedArray.__array_wrap__`'s divide domain (`ArrayWrapDivideDomainFill`: a quotient with `|q| * tiny >= count` becomes
  1.0, masked; a masked 0-d result is `masked`). `mean=` passes to `_var` as given: a C# number is WEAK there.
- **With a mask** (an all-False one too): the mean is `dsum * 1. / cnt` — the sum of numpy.ma's K-order `filled(0)`
  (`FilledForReduction`; the data itself under an all-False mask), times a STRONG float64 one, divided through
  ma.true_divide (non-finite / domain quotients masked, the dividend put back); a 0-d sum is scalar arithmetic, a 0-d
  count of zero `masked`, a float16 result `np.float16(...)`. The variance subtracts through ma.subtract (a C# number
  STRONG via `getdata`), squares in place (complex: `abs(danom) ** 2` through ma.power, a non-finite square masked),
  divides by `count - ddof`, and REPLACES the mask with `mask_or(mask.all(axis), cnt <= 0)` (shrunk to nomask); a flat
  `count - ddof < 0` is a plain (negative) scalar. std is ma.sqrt of the variance and ignores `mean=`. anom is `self - mean`,
  the mean expanded only for a nonzero axis.
- **Layout**: every intermediate a later sum walks is allocated in NumPy's ufunc layout (`UfuncResultBuffer`,
  `Shape.NpyIterOutputShape`) and written in place: NumSharp's eager ufuncs write C order for a permuted input, which
  adds in another order (plan finding 16). np.power's result is computed by the plain call and copied (its out= route is
  not NumPy's, X13).
- **Gates:** `ma_meanvar.jsonl` (26,379 cases, ZERO excuses, valueclass `mameanvar`; raw data at every slot and the
  result's form — singleton / scalar / nomask / mask array — besides `filled(0)` and `getmaskarray`; 294 NumPy errors;
  165 cells carved where the mean sums the view on a layout np.sum still orders differently, finding 5 / 10 / X12);
  `ma_reduce` holds the four strict but for a finding-5 branch; unit `Ma/np.ma.meanvar.NumPyPort.Test.cs` (9). 14 planted
  bugs, each killed by the unit tests and by the tier.
- **Perf** (310 cells): geomean 4.80× since X11's masked copy and X15's float16 broadcast route (2026-10-06; 2.83×
  before), min 1.07×. The five cells under 1.5× are the memory-bound 1M nomask means (1.07–1.39×).

**`np.max`/`np.min`/`np.ptp` (flat AND axis) run NumPy's own reduction schedules, bit for bit**
(`Backends/Default/Math/Reduction/Default.Reduction.MinMax.Exact.cs`, `NumPyMinMaxReduce`; hooked first in
`Default.Reduction.AMax/AMin`, the old IL kernels only on a decline). For floats the ORDER is observable — which zero
sign survives a ±0 tie, and whether a NaN comes back canonical (`0x7ff8…`/`0x7fc00000`, after a lane reduce) or with
its payload (met only by scalar ops) — so value-exact is not enough. **Axis:** NumPy copies each output's first reduced
element, then walks a K-order `NpyIter` with `DONT_NEGATE_STRIDES`; the innermost axis is the smallest-|stride| extent>1
axis (tie → the LATER axis). Reduced axis innermost → ROW mode (a contiguous row = `simd_reduce_c` after the copied
element, a strided row = the scalar 8-accumulator unroll); otherwise SLAB mode (first slab copied, then `o = N(o, x_k)`
per reduced index — the sequential fold; contiguous slab runs fold EIGHT indices per pass through `CombineRun8`, evaluated
as a balanced tree `N(o, N(N(N(x0,x1),N(x2,x3)),N(N(x4,x5),N(x6,x7))))` — bit-identical because `N` is associative, see
below — so the output run is read and written once per eight slabs; the left chain it equals was measured SLOWER than
per-slab passes on L2-resident floats, eight dependent `vmaxp`+`vcmpordp`+`vblendvp` steps).
**Flat:** the 0-d output never votes, so axes coalesce SIGNED into runs; one
run = one call; several runs → `npyiter_find_buffering_setup`'s cost model (ported, `size >= maximum_size` stop
included): the innermost run longer than half the buffer → per-run calls, else the input is BUFFERED — copied in
iteration order into fills of `coresize·⌊bufsize/coresize⌋` (never across the outer block's end), each fill ONE
contiguous `simd_reduce_c` call, so fill boundaries decide ties and re-canonicalize a running NaN; the fill follows
`np.getbufsize()` (`np.setbufsize` changes NumPy's answer too). `N(a,b) = isnan(a) ? a : (a > b ? a : b)` is
associative, so the strided unroll is "the latest of x[0], lane 0…7, tail wins a tie" (probed 1,056/1,056) — which is
why its eight accumulators run as VECTOR lanes (a stride of −1 is one contiguous load + a lane reverse). A NaN-free
fast fold (plain `vmaxp` on groups proven NaN-free by `vcmpunordp`, breaking to NumPy's blend at the first NaN group)
keeps the exact contiguous path fast. Declines: broadcast operands and complex128 (→ the keyword path's reduction
iterator below, NumPy's bits; since 2026-10-05), Decimal/Bool/Char (old kernels, value-exact; bool along its innermost
axis → the keyword path's early-exit `logical_or` loop, 0.19× → ~2× NumPy at 10M), `dtype=` casts, rank > 64 (float16
never reaches it — its own exact route, below). Hooks `NDExpr.DisableExactMinMax`/`ExactMinMaxRuns` (shared with np.evaluate).
NPY/NS @100K: flat 1.14–2.30 on every layout (F/transposed/permuted 1.61–1.85, were 0.61–0.63), axis 1.28–5.2 (were
0.28–1.0 on non-C layouts); the slab fusion put the SLAB cells at 1.38–3.08 @100K (were 1.10–2.31) and the DRAM-bound
f64-family ones at 1.75–2.10 @4M (were 1.06–1.45). Gates: `Backends/Kernels/MinMaxExactScheduleTests.cs` (27,
NumPy-probed literals incl. the every-lane-pair tie test and a ±0 tie / NaN pair on every node of the slab tree;
22/22 schedule + 24/24 slab-fusion targeted mutants killed behind a green-baseline gate) + three NumPy replay oracles
(600 + 5,764 + 3,136 cases, 0 misses on every exact route). Known gaps: L3-bound 4M cells of 1- and 4-byte lanes trail
NumPy's identical schedule (flat/row 0.70–0.87, f32/i32 slab 0.87–0.95; pre-existing, NumSharp's 16-byte buffer
alignment suspected); the UNFUSED `np.max(x * 1, axis)` over a permuted input differs because NumSharp's eager
elementwise writes C order where NumPy's ufunc keeps the K-order layout.

**`np.max`/`np.min`/`np.amax`/`np.amin`/`np.nanmax`/`np.nanmin`/`ndarray.max`/`ndarray.min` take NumPy's `out=`/`initial=`/
`where=` in NumPy's order** (2026-10-05, plan `docs/plans/api-parameter-parity.md` A4; `Sorting_Searching_Counting/np.max.Keywords.cs`,
`Statistics/NDArray.max.Keywords.cs`, engine `DefaultEngine.MaxMinNumPy` in `Default.Reduction.MinMax.Keywords.cs`). Each is
a ufunc reduction (`np.maximum.reduce`; np.nanmax of an ndarray is `np.fmax.reduce`), so the keywords are
`PyUFunc_ReduceWrapper`'s, in its order: A1's reduce prologue (`ReductionArguments`); the loop dtype `promote_types(out, a)`;
`initial=` packed into it by a port of `PyArray_Pack` (`ReductionInitial.cs`: a C# integer literal is a Python int —
range-checked with the LP64 texts, never wrapped —, a `double` a Python float truncated into an integer loop, `Half`/`char`
NumPy scalars `int()`-converted by a SIGNED integer loop and CAST by every other, a 0-d array cast, a str parsed, a
sequence refused); without it a mask raises "... does not have an identity, so to use a where mask one has to specify
'initial'" and an empty axis "zero-size array to reduction operation ..."; the result starts as the initial value or each
output's first element (an unsafe cast). Overloads: NumPy's form at priority 1, NumSharp's positional `(a, axis, keepdims)`
(also priority 1: `np.max(a, 1, true)` must not bind `true` as a 0-d out) and its `dtype=` extension at 0; a keyword-less
call keeps the engine route above. The C# literal `true` for `where=` is NumPy's `where=True` (no mask) — so is a 0-d bool
ARRAY holding true, where NumPy treats `np.array(True)` as a mask needing `initial=` (`[Misaligned]`). Four routes:
(1) a broadcast input whose answer cannot depend on how often an element recurs reduces ONE copy (`MaxMinBroadcastShortcut`:
every loop but float32/float64; those only when every NumPy call is elementwise, since each reduce-form `simd_reduce_c` call
re-canonicalizes a NaN; a non-reduced stride-0 axis from 16,384 elements: 10–560× NumPy); (2) an integer input under a
wider loop reduces in its own dtype then casts (`max` commutes with a monotone cast; 3.6–11.6×); (3) same dtype, no `where=` →
the exact engines above SEEDED with the initial (np.max's own speed); (4) everything else → `NumPyReduceIterator`
(`Backends/Default/Math/Reduction/NumPyReduceIterator.cs`), a port of NumPy's reduction NpyIter: the axis sort (C wins a
tie), coalescing, `npyiter_find_buffering_setup`, fills, the reduce double loop, the first-visit skip, masked runs
(`generic_masked_strided_loop`), and a cast out's round trip through its own dtype at every fill unless the buffer is
REUSED (`NPY_OP_ITFLAG_BUF_REUSABLE`: only from the reduce double loop's own iternext, the out's pointer unchanged — this is
why `np.max(int16 [300, 100…], out=int8)` is 100 past 8,192 elements: the first fill leaves `int8(300) = 44`). Its loops
(`NumPyMinMaxLoops`): the SIMD rules for float32/float64/integers, `logical_or`/`logical_and` for bool (early exit; NumPy's
`min(a|b, 1)` elementwise), the sequential BINARY_LOOPs for float16/char/decimal, and complex128 VECTORIZED — each rule is an
associative SELECTION, so a 1024-element block folds in two vector passes (the extreme VALUE and whether a NaN occurs, then
the first element equal to it / the first NaN), 1.5–3.9× NumPy's scalar `CDOUBLE_maximum` (was 0.97–1.37×). Loop forms
beyond NumPy's two: a WHOLE masked call (order-free loops fold the selected elements in one pass; the float SIMD loops make
NumPy's per-run calls but compose each run through a closed form of `simd_reduce_c` — `NRunContiguous`/`PRunContiguous`, from
the lane ops' associativity — so only a compare-and-select stays on the out's dependency chain: float64 `where=` 0.94× →
2.2×) and SEVERAL ROWS per call where the out does not move (`Loop.BinaryRows`; rows folded in pairs by associativity). A fresh
axis result of every reduction that allocates through `AllocateReductionResult` (sum / mean / var / std / the nan family /
min / max) is now laid out like NumPy's (`Shape.ReductionOutputShape`, `npyiter_new_temp_array`: the input's axis order, not
C order). Gates: oracle tier `minmax_kwargs.jsonl` (35,847 cases, zero excuses, 24/24 planted bugs killed; README "The max /
min family with NumPy's keywords"), `Selection/np.max.Keywords.Test.cs` (11). NPY/NS over 237 cells: geomean 3.98×; the cells
under 1.5× ride the engine above (10M L3/DRAM-bound, its 100K slab gaps) or follow NumPy's own structure (a float64 →
float32 out round-tripping at every fill 1.0–1.3×, a broadcast float64 input reduced flat 1.1–1.3×).

**float16 `np.max`/`np.min`/`np.nanmax`/`np.nanmin` (flat AND axis, every layout incl. broadcast) are NumPy's sequential
HALF loops, bit for bit** (`Backends/Default/Math/Reduction/Default.Reduction.MinMax.Half.cs`, `DefaultEngine.HalfMinMaxReduce`,
taken before any other route; 2026-10-03). NumPy has no SIMD float16 compare loop: `HALF_maximum` is
`(ge(acc,x) || isnan(acc)) ? acc : x` (fmax: `|| isnan(x)`), a SELECTION, so the answer is "the FIRST NaN verbatim (payload,
sign, a signalling NaN unquieted), else the first extreme element" (fmax: NaN skipped, an all-NaN slice → its first element),
a ±0 tie keeping the earlier zero — "first" in NumPy's visiting order: each axis in increasing logical index
(`DONT_NEGATE_STRIDES` — a reversed run is visited from its HIGH address down), a flat reduction in K order
(`NumPySequentialReduce.FlatVisitOrder`). The selection is associative, so the walks run SIMD and stitch run winners in
order: FLAT drops broadcast axes, coalesces runs, sends ±1-stride runs ≥ 32 to the bit-level cores (`HalfRunWinnerBits`,
reversed runs rescan from the top) and folds the rest by a branch-light PRIORITY KEY (`HalfPriorityKey`: order key with ±0
merged, inverted for min, NaN → 0xFFFF or 0 — "take iff strictly greater" IS the fold step for all four rules); AXIS picks
ROW (a SIMD core per ±1-stride row, a branch-free masked fold for rows ≤ 16 — the improve-branch is a coin flip there —
the out-of-line branchy fold otherwise) or SLAB (the elementwise pair-select over 4096-lane blocks, a stepped inner run
gathered first). It replaced a last-NaN-wins axis kernel, a canonical-`Half.NaN` flat NDIter fallback and a reversed-order
flat walk (values right, bits wrong — the oracle tokenizes NaN and its pools hold one NaN), and nanmax/nanmin's scalar fold
(0.4–0.9× NumPy). Rode along: np.evaluate's flat `Max`/`Min` over a float16 child (`TryExactFlatMinMaxHalf`: leaf in place,
computed child in NumPy's layout — was last-NaN-wins), the `dtype=float16` flat path over another dtype (read float32
memory as float16 — now cast first, like the axis path), and the old `(Half,Half)` axis kernel key (an adapter onto the
walk). NPY/NS over 276 cells (4 ops × 23 layouts × 1K/100K/1M, P-core pinned): geomean 26× (1K 5.6×, 100K 41×, 1M 79×; was
1.6× with 169 cells under 1.5×) — NumPy's HALF loops cost ~5 ns/element; slowest cells: the 1K fixed-cost floor (rows of 3:
2.3×) and rows of 3 along axis 1 at 100K (4.2×). `np.ptp` (NumPy's `subtract(maximum.reduce, minimum.reduce)`) inherits the
walks: its NaN is the same first NaN, quieted by HALF_subtract's float32 arithmetic (3,622 of a 6,034-case NumPy probe had
other bits before, 0 now). Gates: `Math/HalfMinMaxLayoutTests.cs` (16, NumPy-probed bit literals — 15 fail on the old code;
the unit tests alone kill all 14 planted bugs of a mutation run, the oracle tiers 11) + the raw-byte `half_minmax` oracle
tier (5,900 cases, 1,789 of them fail on the old code) + the float16 SELECTION CONTRACT
(`FuzzCorpusTests.Kinds.HalfSelectionContractOps`: these four ops on a float16 operand are compared by raw bytes in EVERY
tier and any difference is a hard failure, no registry excuse) + the float16 ptp contract (the same, on x86 hosts only:
`inf - inf` makes the CPU's default NaN, negative on x86 and positive on arm64).

`cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None)` is a **pure
composition** over `average`/`dot`/`concatenate`/`atleast_2d`/`conjugate`/`squeeze` (no new kernel), a line-for-line
port of NumPy's `numpy/lib/_function_base_impl.cov`. `m`/`y` are 1-D or 2-D (>2-D → `ValueError`); `rowvar=False`
transposes the variable/observation relationship; `ddof` overrides `bias` (defaults: `bias→N`, else `N-1`). Result
dtype is `result_type(m[, y], float64)` (int/bool→float64, complex128 stays complex) unless `dtype` is given — but
NOT force-cast at the end, so `dtype=float32` **with** any weights still yields float64 (the float64 weights promote
`avg`→`dot`), matching NumPy exactly. `fweights` (integer frequency) / `aweights` (relative) combine to `w=fw*aw`;
the `fact = w_sum - ddof*sum(w*aw)/w_sum` normalization, the `fact<=0 → +inf` (DoF≤0) degenerate path, and the empty
`(0,0)`-float64 / single-observation `nan` / scalar `nan` edges all match NumPy. Error taxonomy is verbatim:
`ValueError` (`m has more than 2 dimensions`, `{f,a}weights cannot be negative`), `TypeError` (`fweights must be
integer`), and the new house `RuntimeError` (`cannot handle multidimensional {f,a}weights`, `incompatible numbers of
samples and {f,a}weights`); weights summing to zero raise `DivideByZeroException` (NumPy `ZeroDivisionError`). **Two
NumPy fidelity details ported deliberately:** `X -= avg[:, None]` is done in the operand's dtype (so `dtype=float32`
without weights keeps float32), and for **complex** input NumPy's `fact` inherits complex128 so its `.../w_sum`
division is Smith's-method `numer*(1/w_sum)` — reproduced as a reciprocal-multiply because it rounds 1 ULP off real
division and decides finite-vs-inf when `fact ~ 0`. Differential-verified across 2556 cases (5 dtypes × shapes ×
rowvar/bias/ddof × weight combos × y × layouts): bit-exact for small observation counts, otherwise within managed-GEMM
tolerance (cov is **GEMM-bound** — as fast as `np.dot`, i.e. slower than NumPy's BLAS by default but byte-identical
and BLAS-fast once `NumSharp.Interop.OpenBLAS` is referenced — which holds ONLY because `TryGramCov`, the ≤16-variable
managed Gram (syrk) fast path, is **gated to `TensorEngine.Blas is null`**: NumPy computes cov through BLAS, so with a
backend installed the product must take `np.dot`; the 0.70.0 release QA caught the Gram path still winning with the
backend present, 1–8 ULP off NumPy on 29/36 elements). Gate: `Statistics/np.cov.BattleTests.cs` (34). See
`Statistics/np.cov.cs`.

### Math — NaN-Aware
`nanmax`, `nanmean`, `nanmedian`, `nanmin`, `nanpercentile`, `nanprod`, `nanquantile`, `nanstd`, `nansum`, `nanvar`

**`np.nanmax`/`np.nanmin` (and np.evaluate's `NanMax`/`NanMin`) are NumPy's `np.fmax.reduce`/`np.fmin.reduce`, bit
for bit** — the ±0 sign of a tie and WHICH NaN an all-NaN slice returns (payload intact, never canonical) match NumPy
2.4.2 on every non-broadcast float layout. fmax's loop has TWO per-element ops (the "P rules", header of
`Default.Reduction.MinMax.Exact.cs`): the VECTOR op `npyv_maxp` (a tie keeps the later operand, two NaNs keep the
FIRST) inside each inner-loop call's vector section, the MSVC CRT `fmax` (tie → bitwise AND, i.e. `+0`; fmin → OR,
`-0`; two NaNs → the LAST) for the call's scalar tail and strided rows; `npyv_reduce_maxp` returns lane 0 RAW when
every lane is NaN. So the answer depends on where NumPy's calls CUT the reduction — `SlabCallTiling.Build` ports that
cut (`npyiter_find_buffering_setup`, `np.getbufsize()` honoured; a float32 in-place negative-stride input runs every
call scalar), the flat/row paths reuse `NumPyMinMaxReduce` with `TLane.PropagatesNaN == false`. **float16/complex128**
run NumPy's sequential BINARY_LOOPs (`HALF_fmax`/`CDOUBLE_fmax`) in NpyIter's visiting order: NaN skipped (complex: a NaN
in either part), the EARLIER of a tie kept, complex lexicographic, an all-NaN slice → its first NaN verbatim — float16
through the bit-level SIMD walks it shares with np.max/np.min (`Default.Reduction.MinMax.Half.cs`, "Math — Reductions"),
complex128 through the scalar fold of `Default.Reduction.Nan.Sequential.cs`. Integers/bool/char/decimal are `np.amax`/`np.amin`. **BREAKING
(NumPy-aligned):** an empty reduction RAISES `zero-size array to reduction operation fmax/fmin which has no identity`
(was a 0-d NaN); complex used to go to `np.amax` and PROPAGATE a NaN; float16 non-C layouts misread strides. Gates:
`Backends/Kernels/NanMinMaxExactScheduleTests.cs` (31, NumPy-probed literals; 36/36 schedule mutants + 21/24 fold
mutants killed, the 3 fold survivors equivalent by design) + three NumPy replay oracles (6,000+ f32/f64 cases ×
engine/leaf/computed, 2,080 f16/c128 incl. broadcasts — 0 misses) + the `nanreduce`/`specials` fuzz tiers, where
`nanmax`/`nanmin` are no longer in `MisalignedRegistry.NanReduceOps` (now ENFORCED). **The P rules' vector section is
ONE order-free fold** (`NumPyMinMaxReduce.FoldVectorsP`, `DefaultEngine.Evaluate.MinMax.cs`): `maxp` is an associative
selection ("the LAST largest non-NaN, else the FIRST NaN"), so NumPy's group tree + single-vector loop equal, per lane,
a BACKWARD scan with the plain `vmaxp(data, suffix)` from `∓inf` — one instruction per vector, the suffix never NaN, a
NaN data lane skipped, a tie keeping the later element — read through the 32-byte-ALIGNED vectors covering NumPy's
`x[1:]` section (NumPy's own loads split a cache line every other vector; the tree paid 1.4× the old value-only
kernel's time for it) with one `vpermd` rotating positions back to NumPy's lanes and a vectorized rescan of the rare
NaN-accumulator lane the fill cannot decide. Only for stretches of ≥ `FoldVectorsPMinElements` = 768 elements: the
fold's 1.5–5 ns fixed cost is paid per ROW (rows ≤ 512 elements ran 0.49–0.97× the tree), so shorter stretches keep the
literal tree untouched; and `Finish` dispatches to two SEPARATE methods (`FinishTree`, the old body verbatim /
`FinishFoldP`) because tier-1 PGO compiles one body from its early profile — a shared body laid the tree's loops out
cold whenever long reductions ran first (short rows 0.69–0.84×). NPY/NS @100K: flat C f64/f32 2.89/3.33 (were
1.93/2.40, the exact route now 0.96–1.01× the old kernel in-process — it was 0.74/0.67×, the "1.10×" once claimed did
not reproduce), F 2.98, rows of 1000 3.54 / f32 4.31, axis slab 2.07–2.55, evaluate `nanmax(a*b)` 2.16 (the 9.3 once
quoted was measured while NumPy's temporary page-faulted — the allocator-regime trap); rows below the threshold
0.96–1.05× unchanged. Known gaps: complex128 axis & non-C cells are a scalar fold (float16's were 0.41–0.69× NumPy until
2026-10-03, now the SIMD walks at ~25× geomean); flat contiguous @4M is DRAM-bound (0.89–1.06× the old kernel in-process, NPY/NS f64 ~1.5–1.6 / f32 ~1.0); a
broadcast float32/float64 input runs the keyword path's reduction iterator (NumPy's bits; it kept the old, value-exact
kernels until 2026-10-05, plan api-parameter-parity A4). Design + measurements + gates:
`docs/plans/ndexpr-evaluate.md` → "The P rules' order-free aligned fold".

**`np.nanvar` / `np.nanstd` are NumPy's `nanvar` / `nanstd`, statement for statement**
(`Backends/Default/Math/Reduction/Default.Reduction.NanVarStd.cs`, `NanVarStdNumPy`; `np.nanvar` / `np.nanstd` are thin
wrappers and gained `dtype=`; 2026-10-05, plan `docs/plans/api-parameter-parity.md` A3). The old implementation was a
moment loop in double with its own shortcuts; NumPy's (`numpy/lib/_nanfunctions_impl.py`) is ufunc statements:
- **The statements.** An input that cannot hold NaN (integer, bool, char and NumSharp's decimal) is `np.var` / `np.std`
  (A2). For float16 / float32 / float64 / complex128: a non-inexact `dtype=` raises "If a is inexact, then dtype must be
  inexact" BEFORE the axis is read. Then the axis follows ufunc.reduce's rule, because nanvar's first axis consumer is
  `np.sum`: on a 0-d input axis 0 / -1 reduce flat (np.var raises there). Then: the non-NaN count (int64, keepdims), the
  mean of the NaN-zeroed K-order copy (`Shape.KeepOrder()`) divided by it (the float64 loop, Smith's for complex), the
  deviations rounded into the INPUT's dtype (`subtract(out=arr, casting='unsafe')`: with `dtype=float64` a float32 input
  still squares float32 deviations), zeroed at the NaNs, squared (complex: the real part of NumPy's `simd_cmul`
  `x * conj(x)`, one rounding of `re*re + round(im*im)` — NOT np.var's `re² + im²`), summed, divided by `cnt - ddof`, and
  `np.nan` (the POSITIVE quiet NaN) wherever `cnt - ddof <= 0`. So np.var's `+inf` at `ddof == n` is NaN here, and an
  inf element is NaN (`inf - inf`).
- **Engines.** Both sums run over NumPy's K-order copy. A dense input without a cast (one memory run flat, C/F along an
  axis) already IS that copy: the mean's sum and the count stream over the input, and a flat variance streams the
  squared deviations too — three reads, no buffer. Otherwise the copy is materialized for `ReduceAdd`, and one fused
  `np.evaluate` pass writes the squared deviations into a K-order buffer. float16 and hosts without dynamic code run
  NumPy's eager statements (the SIMD half kernels are its HALF loops). Because the sums never walk the view,
  np.sum's view residuals (a reversed axis, a broadcast operand's sum) do not reach nanvar.
- **Perf** (NPY/NS, one pinned P-core, 86 cells: float64 / float32 / float16 / complex128 / int32 × flat / axis 0 / axis
  1 × 1K / 100K / 10M, plus F / strided / `dtype=` forms, NaN every 7th element): geomean 3.71×, min 1.51× (int32 along
  an axis at 1K, which is np.var's route). NumPy's nanvar makes about 11 passes; flat float64 is 9.1× at 100K and 5.2×
  at 10M, float16 along axis 1 at 1K 1.72×.
- **Residuals:** np.sum's cast-loop rows longer than the buffer (X6) reach `dtype=`; B7's `out=` / `where=` / `mean=` /
  `correction=` and NumPy's parameter order are open.
- **Gates:** `nanvarstd.jsonl` (13,382 cases, zero excuses), the `nanreduce` / `specials` / `reduce_start` tiers with
  nanvar / nanstd out of the nan-family blankets (`NanReduceOps`) and in `ByteExactArithmeticUnaryOps`, unit
  `Math/np.nanvar.NumPyPort.Test.cs` (12).

**`np.nanmean` is NumPy's `nanmean`, statement for statement** (`Backends/Default/Math/Reduction/Default.Reduction.NanMean.cs`,
`NanMeanNumPy`; `np.nanmean` is a thin wrapper with `dtype=`; 2026-10-05, plan `docs/plans/api-parameter-parity.md` A7).
An input that is not inexact (and decimal) is `np.mean` with the same keywords — NumPy's `_mean` since X9, which this
item found. An inexact input: the dtype TypeError, ufunc.reduce's axis rule (np.sum reads the axis first: a 0-d input
takes 0 / -1 as flat), the non-NaN count (`keepdims=keepdims`), the sum of the NaN-zeroed K-order copy in its loop, and
the in-place division (`DivideByCountsLikeNumPy`: the float64 loop, Smith's complex division; an all-NaN slice keeps
`0 / 0`). Routes as nanvar's: the count streams; a dense input without a cast streams its zeroed sum; otherwise the
K-order copy is materialized for `ReduceAdd`; float16 and hosts without dynamic code run NumPy's eager statements, whose
zero fill (shared with nanvar, `ZeroWhereMask`) is NumPy's own `np.copyto(arr, 0, where=mask)` — 3.7 µs into a
float16 `(250, 400)` since X11's masked kernel (A7 had routed it through `np.putmask`, 21 µs, while that copy took
253 µs). Perf (72 cells): geomean 5.82×, min 1.70×. Gates: `nanmean.jsonl` (5,017
cases, zero excuses; nanmean left `NanReduceOps` and joined `ByteExactArithmeticUnaryOps`), unit
`Math/np.nanmean.NumPyPort.Test.cs` (5).

### Bitwise
`bitwise_and`, `bitwise_or`, `bitwise_xor` (ufunc `out=`/`where=`/`dtype=` supported; float/complex/decimal INPUTS raise NumPy's coercion TypeError while a float/complex/decimal `dtype=` raises the no-loop text — distinct messages, both probed; probed order: bad `where` → no-loop → out-cast → shape), `invert`, `left_shift`, `right_shift`

### Comparison & Logic
`all`, `allclose`, `any`, `array_equal`, `equal`, `fmax`, `fmin`, `greater`, `greater_equal`, `isclose`, `iscomplex`, `iscomplexobj`, `isfinite`, `isin`, `isinf`, `isnan`, `isreal`, `isrealobj`, `isscalar`, `iterable`, `less`, `less_equal`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `maximum`, `minimum`, `not_equal`, `signbit`

**`np.signbit(x, out=None, where=True, dtype=None)`** — the IEEE sign-bit predicate (a full ufunc, unlike
`isposinf`/`isneginf`; result always **bool**), the primitive those two are defined on (`isinf(x) & ~signbit(x)` /
`isinf(x) & signbit(x)`). It is NOT `x < 0`: it reads the raw sign bit, so on floats `-0.0` → True and a
**negative** NaN → True while `+0.0`/`+inf`/positive NaN → False (probed 2.4.2). Per dtype: **Half/Single/Double**
test the IEEE sign bit; **signed integers** use `x < 0` (the two's-complement MSB IS the sign bit); **unsigned
integers / bool / char** are always False; **Decimal** (no NumPy analog) is strictly-negative (`Math.Sign < 0`, so
`-0.0m` → False, documented); **Complex** has no loop and raises NumPy's verbatim `ufunc 'signbit' not supported
for the input types…` `TypeError`. Rides the SAME fused predicate kernel as `isnan`/`isinf` — Single/Double **and**
signed Int32/Int64 vectorize via a per-lane `ExtractMostSignificantBits` + PDEP bool store (the mask IS the loaded
vector; no compare, no constant), narrow/unsigned/Half/Decimal take the scalar route (which already beats NumPy's
own scalar signbit loops). `dtype=` is validate-only (bool loops only). **Perf (NPY/NS, plain call):** faster than
NumPy on every dtype — i8/u8 **7–11×**, i32 **3.3–6.9×**, i64 **1.8–3.3×**, f32 **1.3–2.4×**; f64 **~1.4×**
(memory-bandwidth-bound reading the full 8-byte lanes, the shared DRAM ceiling). Gate: `Logic/np.signbit.Test.cs`
(19) + 338 cases in the differential-fuzz `logic` tier (all 13 dtypes × 26 layouts, bit-exact vs NumPy 2.4.2).
See `Logic/np.is.cs` (`signbit`), `Backends/Default/Logic/Default.SignBit.cs`,
`Backends/Kernels/Direct/DirectILKernelGenerator.Unary.Predicate.cs` (`EmitSignBitCall` + the extended SIMD gate).

`np.isin(element, test_elements, assume_unique=false, invert=false, kind=null)` is the element-wise membership
test — a bool array of `element`'s shape, True where `element[i]` is a value in the (flattened) `test_elements`
set (NumPy `_arraysetops_impl.py`). Comparison happens in `result_type(element, test)`, matching NumPy's
brute/sort/table methods (which all agree on the RESULT — `kind` only trades speed for memory). The engine picks
its own method by the test set's value range, always returning the correct membership: an integer/bool pair with a
range ≤ `6·(|element|+|test|)` takes a **counting-table gather** (build a bool lookup over `[min,max]`, then one
`take(mode='clip')` per element with a range mask — no widening of the element array, staying in its own dtype),
otherwise a **sort + `searchsorted` + equality probe** (which also covers float/complex/decimal/char — complex
sorts lexicographically like NumPy). NaN is never a member (NaN ≠ NaN falls out of the equality probe). The one
exact-integer trap is handled explicitly: **uint64 paired with a signed integer** promotes to float64 under NEP50,
which would call two distinct integers equal past 2^53, so that pair is routed through the uint64 domain with a
sign filter (`isin(uint64[2^63+1], int64[2^63-1])` is correctly `False`, matching NumPy's table method — pinned).
`kind` is validated verbatim: `Invalid kind: '…'. Please use None, 'sort' or 'table'.` (ValueError), and for
`kind='table'` a non-integer input raises `The 'table' method is only supported for boolean or integer arrays…`
(ValueError) and a signed range exceeding the dtype max raises NumPy's RuntimeError text (as
`InvalidOperationException`). Non-contiguous `element` (transposed/strided/negative-stride/broadcast) is read in
logical C-order and the bool result is reshaped to a C-contiguous copy of `element`'s dimensions. **Perf (NPY/NS,
Release, best-of-21, warm):** integer isin **1.9–3.0×**, float64 **2.4–3.4×**, large-range int **3.3×** — faster
than NumPy at every measured cell. **Two deliberate divergences** (`[Misaligned]`): with `assume_unique=true` on
NON-unique inputs NumPy documents "incorrect results" and its output depends on the internal method it picks;
NumSharp always returns the correct membership. And `kind='table'` with a large-but-range-safe integer range
computes via `searchsorted` (no table allocation), so it does not raise `MemoryError` where NumPy's literal table
allocation would. See `Manipulation/np.isin.cs`; gate: `Manipulation/np.isin.Test.cs` (37) + the `isin` cases in the
`groupa` differential-fuzz tier (22, across int32/float64/uint8/complex128 × invert/assume_unique/kind/layouts).

`np.iterable(y)` is NumPy's pure iterability predicate — `try: iter(y); return True; except TypeError: return False` (`numpy/lib/_function_base_impl.py`). It is NOT an iteration op: it never touches element data, so it uses **no kernel/NDIter/loop** (O(1) rank/type check in `Logic/np.is.cs`, mirroring `isscalar`). C# mapping, each matching NumPy 2.4.2's `iter()` outcome: `null`→false; **`NDArray`→`ndim != 0`** (the one surprise NumPy documents — a 0-d array is the *only* non-iterable array: `iter()` on it raises `TypeError`, exactly as `NDArray.GetEnumerator()` does, while every rank≥1 array incl. empties is iterable); `string`→true (Python strings are iterable); any `IEnumerable` (C# arrays/lists/dicts/sets)→true; every scalar value type (int/double/bool/char/Half/decimal/Complex/…)→false. The `NDArray` case precedes the `IEnumerable` catch because `NDArray` implements `IEnumerable`. Being a bool predicate over an arbitrary object rather than a value-producing kernel over array data, it is unit-test-only (`Logic/np.iterable.Test.cs`, 33 cases) and deliberately absent from the differential-fuzz corpus — same rationale as `isscalar`/`nditer`/`ndindex`. **Deliberate C# divergences** (adversarially probed, pinned `[Misaligned]`): NumSharp maps Python `iter()`-ability onto C# `foreach`-ability (`IEnumerable`), so four inputs iterable in Python but NOT `foreach`-able in C# return false where NumPy returns true — a bare `IEnumerator`/`IEnumerator<T>` cursor (no `GetEnumerator`) and `ValueTuple`/`Tuple`/`ITuple` (no `IEnumerable`). Real ported code passes the collection (array/`List`/`NDArray`), which is foreach-able and matches. Construction traps are clean: NumSharp's `np.array(5)`, `NDArray x = 5`, and `a[0,0]` all build genuine 0-d arrays (→false), and a bare scalar literal boxes to its own type rather than firing an `int→NDArray` implicit conversion into the `object` overload.

The six comparisons and `isnan`/`isfinite`/`isinf` expose **ONE NumPy-shaped overload each** — `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)` (no bare/out split). It returns plain `NDArray` — NumPy's `np.less(a, b, out=f64)` returns the f64 out itself; `True→1` at any numeric out dtype since bool casts same_kind to all of them. A plain call still returns an `NDArray<bool>` *instance* (TensorEngine contract), so the typed wrapper is one zero-alloc cast away and the C# comparison operators (`==`, `<`, …) keep the `NDArray<bool>` static type via `AsGeneric<bool>()`. `dtype=` is validate-only (probed 2.4.2): bool loops only — `dtype: Boolean` is a no-op, anything else raises `No loop matching the specified signature and casting was found for ufunc <name>`. Comparisons compare at `result_type(lhs, rhs)` inside the kernel (probed: `greater(i8 2^53+1, f8 2^53)` → False, `equal` → True). Engine members follow the house order `(inputs, typeCode, out, where)`.

**A Python scalar is weak in every comparison (NEP 50), and the array-vs-scalar kernels are SIMD** (2026-10-07, plan `docs/plans/api-parameter-parity.md` X24 / X25 / X2 / X21; gates the `weak_scalar` tier (15,050 cases, zero excuses) + `Logic/np.comparison.WeakScalar.Test.cs` + `Ma/np.ma.StrongScalar.Test.cs` + `Backends/Kernels/ComparisonBroadcastKernelTests.cs`):
- **The pack.** `ExecuteComparison` (`DefaultEngine.CompareOp.WeakScalar.cs`) converts a 0-d Python-scalar operand (a C# bool / integer / float / double / `Complex`; `Half` / `char` / `decimal` are strong NumPy scalars) against an array into the array's precision before either route runs: `PyArray_Pack`, i.e. the correctly rounded double, then the float16 / float32 narrow (an int past 2^53 rounds twice, overflow is inf), complex parts to float32 (complex64). So `np.greater(f32 [0.1, 0.5], 0.1)` is `[False, True]` (it compared at float64 and gave `[True, True]`). A 0-d against a 0-d stays strong; a 0-d NDArray against an array is read as the literal (`[Misaligned]`, np.clip's rule). An integer array against an in-range Python int of another width is re-packed into the array's dtype (NumPy's `special_integer_comparisons` in-range branch; values unchanged, the same-dtype kernel runs). numpy.ma stays STRONG (`PromoteMaScalar(strongOperand)`; its domains see `getdata` data).
- **The binary ufuncs' weak int.** `ExecuteBinaryOp` range-checks a 0-d Python int against the final loop dtype before any kernel runs (`CheckWeakPyIntOperand`: `np.maximum(int8, 1000)` is NumPy's `OverflowError: Python integer 1000 out of bounds for int8`, it wrapped to -24; a bool array meets the int in the int64 loop; np.power checks it before its negative-power error) and re-packs an int past 2^53 into a float16 / float32 loop (`RepackWeakIntForFloatLoop`).
- **The kernels.** `array op scalar` / `scalar op array` (`SimdScalarRight` / `Left`) compare a vector per step with the scalar broadcast once, when the array is already in the comparison dtype (`EmitComparisonBroadcastSimdLoop`; they were scalar loops, 0.72 ns an element). The 4×-unrolled bodies store their four masks PACKED into result bytes on AVX2 hosts (`EmitMasksToBools`, NumPy's `npyv_pack_b8_b16 / _b32 / _b64`: saturating packs + a permute + `& 1`); 1-byte lanes store `mask & 1`; the single-mask store reads the lane bits as 64 bits (a Vector512 mask has 64 lanes).
- **The loop bound.** A SIMD loop body over `step` elements runs while `i <= n - step`. The comparison `SimdFull` kernel and the contiguous predicate kernel (isnan / isinf / isfinite / isposinf / isneginf / signbit) ran while `i <= n - step + 1` since they were written: one element read past the inputs and one bool written past the result whenever `n % step == step - 1` (an out= view's next parent slot; past a fresh result's exact-size allocation). Gate for this class: sentinel slots past a directly called kernel's buffer, and `NUMSHARP_DEBUG_GUARD_PAGES=1` (every pooled buffer ends against an inaccessible page; a 72,630-case sweep faults on the old build at `equal Boolean n=31`).
- **Perf (NPY/NS):** `f32 > 0.1` 0.82× / 1.17× / 1.73× at 1K / 100K / 10M (was 0.49× / 0.09× / 0.56×); `f16 > 0.1` 15×; `u8 > 128` 3.9× at 10M. At 100K NumPy's loop is the same packed form. The 1K array-op-scalar cells (0.56–0.83×) are the C# literal's fixed cost: its 0-d NDArray is never disposed and its finalization costs ~0.5 µs a call (`x > s` with a prebuilt 0-d operand is 0.50 µs, as fast as array op array); open as plan X26, library-wide.

**The comparison engine: every operand layout and dtype pair runs the vector kernels (2026-10-10).** `==`, `!=`, `<`,
`<=`, `>`, `>=` as operators and as `np.equal` … `np.greater_equal`, with and without `out=`.
- **What NumPy does** (`loops_comparison.dispatch.c.src`, `PyUFunc_SimpleBinaryComparisonTypeResolver`). Its loops
  are vectorized for three operand forms only: two contiguous inputs, or one input a single value, each with a
  contiguous bool output. A stepped or reversed input, an input of another dtype (NumPy casts it through a buffer
  first) and a stepped output run its scalar loop at 0.6–1 ns an element; so do complex128 and float16. int64
  against uint64 has exact loops.
- **`ComparisonRun`** (`Backends/Kernels/ComparisonRun.cs`): one cached instance per (operator, left dtype, right
  dtype). It takes one inner-loop chunk (three pointers, three byte strides, a count) and runs it through the three
  contiguous vector kernels (`SimdFull`, `SimdScalarRight`, `SimdScalarLeft`), whatever the strides and dtypes:
  - *direct*: an input that already is a contiguous run of the dtype the kernel compares in, or one value, is read
    in place. A long run that starts off a vector boundary gets a short first kernel call (`AlignmentPeel`: the
    input stream for lanes of two bytes or more, the output for one-byte lanes; from 4 KB);
  - *staged*: any other input is copied and converted 8 KB at a time into a stack block, and the kernel reads the
    block while it is in L1. The house SIMD copy removes a step of two or a reversal, the unrolled strided move
    (`Direct/DirectILKernelGenerator.StridedMove.cs`, IL, 1 / 2 / 4 / 8 / 16 bytes) any other stride, the house SIMD
    cast converts;
  - *flipped*: inputs that run backwards in the kernel's dtype are read forwards from their last element and the
    bool results are stored reversed (one byte an element moved, not 8 or 16);
  - *scattered*: a bool output that is not contiguous receives each block through a strided byte store;
  - *adopted*: an array of another dtype against ONE value is compared in the array's own dtype (below);
  - a chunk under 64 elements runs the per-element kernel the iterator routes always used.

  Two more forms belong to a whole iteration of rows (`RunRows`), not to one chunk (both below):
  - *tiled*: an operand whose successive ROWS are adjacent elements (an F-ordered matrix beside a C-ordered one, a
    matrix against its transpose) is copied a tile of several rows at a time;
  - *repeated*: a matrix against ONE short row compares several rows per kernel call.
- **Three callers.**
  - `TryTrivialContiguousComparisonOp` → `Run`: two contiguous arrays of one shape, or an array and one value, of any
    two dtypes; also two one-element operands of different rank (they used to build an iterator for one element).
  - `TryRowsComparisonOp` (`DefaultEngine.CompareOp.Rows.cs`) → `RunRows`: an iteration of R runs of N elements,
    without an iterator. It reads the structure from the strides with NumPy's coalescing rule
    (`stride[d] == stride[d+1] * extent[d+1]`): a matrix against a row or a column, an outer product, a stepped or
    reversed view. It declines (the iterator route serves): a result NumPy lays out in F order, an iteration of three
    or more levels, and many rows under 16 elements whose inputs are contiguous or one value (the iterator's 2-D block
    kernel loops such rows inside one call), unless they are a matrix against one row (`ComparisonRun.RepeatsRow`).
    An iterator costs 428 ns to build and dispose and 23 ns for each chunk it hands out (40 float64 values); a
    kernel call per run costs 7.8 ns.
  - `NDIterRef.TryExecuteComparisonRun` → one `Chunk` call per inner-loop chunk: every other iteration, and every
    `out=` call (`ExecuteComparisonUfuncInto`; an out of another dtype is filled through the iterator's Boolean
    buffer). `where=` keeps its masked route (1.5–11× NumPy already).
- **Scalar adoption is exact, never a cast** (`ComparisonRun.ScalarAdoption`). The comparison NumPy defines is
  `convert(a) op convert(v)` in the promoted dtype. It has the same answer as `a op a'` for ONE value `a'` of the
  array's dtype, or the same answer for every element:
  - integer array, integer value: `a' = v` inside the array dtype's range; outside it the chunk is decided;
  - integer array of at most 32 bits, float value `t`: `a > t` is `a > floor(t)`, `a >= t` is `a >= ceil(t)`,
    `a < t` is `a < ceil(t)`, `a <= t` is `a <= floor(t)`; `==` can hold only for an integral `t` in range;
  - float16 / float32 array, wider value `t`: `>` and `<=` use the largest array-dtype value not above `t`, `>=` and
    `<` the smallest not below it (one step on the stored bit pattern from the rounded value); `==` can hold only
    when `t` is an array-dtype value. So `float32_array > 0.1` still compares against the double 0.1;
  - a NaN value decides the chunk (False; True for `!=`).

  Not adopted (they keep the staged conversion): a 64-bit integer array against a float, complex, decimal, char.
- **Tiles: an operand whose rows are adjacent elements** (`ComparisonRun.RunTiles`; the commit after the pool's).
  An F-ordered matrix read in C order has rows whose elements are one COLUMN apart: a row of N elements touches N
  cache lines, and N pages once a column is 4 KB long. The rows route staged such a row with one strided move a
  row: 0.3–0.6 ns an element for a (250, 400) matrix, 0.65–1.3 for (1000, 1000), 3.8–8.6 for (3162, 3162); NumPy's
  scalar loop pays the same. The elements of one column ARE adjacent, so a tile of B rows reads B contiguous
  elements of each column and deals them out to B rows of a stack block.
  - **The kernel** (`Direct/DirectILKernelGenerator.TransposeTile.cs`): AVX2 groups of 2 × 4 elements of 8 bytes,
    4 × 8 of 4 bytes, 8 × 16 of 2 bytes, 16 × 32 of one byte. Each 256-bit register is built from two columns (the
    second one through `vinserti128` with a memory operand, which uses no shuffle port), so the in-lane
    transposition gives rows of twice the columns and needs no lane-crossing shuffle. Integer unpacks for every
    width: current Intel cores run them on two ports, the floating-point ones on one (the 2-byte group runs at 0.13
    cycles an element where its 24 shuffles on one port would take 0.19; the 4-byte kernel went 0.13 → 0.08 ns an
    element with them, the 8-byte one 0.18 → 0.13 with two row groups a loop step).
    In L1: 0.13 / 0.08 / 0.05 / 0.03 ns an element for 8 / 4 / 2 / 1 bytes (a strided move: 0.2). A tile that is
    not a whole number of groups ends with a group that OVERLAPS the one before it, so there is no scalar tail.
    Below one group, for 16-byte elements and without AVX2 the tile is moved by the strided move kernel along
    its longer side.
  - **The tile**: at most 8 KB of the kernel dtype (16 KB is slower for 1- and 2-byte elements: (400, 250) uint8
    8.9 against 13.6 µs, (250, 400) int16 11.7 against 14.4). A whole row wide when 8 rows of it fit, else the row
    is cut evenly (8 rows against 4: (250, 400) float64 35.3 against 40.5 µs, (1000, 1000) 641 against 791); then as
    many rows high as fit. From a row SPAN of 4 MB (elements a row × the column stride) the fewest rows are 16
    (4- and 8-byte elements) or 32: each column piece is then a cache miss and further up a TLB miss, which more
    rows a piece pay for better than longer row pieces do ((3162, 3162) float32: 12.0 → 8.7 ms, int16 9.4 → 4.7).
    The last tile of each direction is moved back to end on the operand's edge and computes some results twice;
    `RunRows` never has an output that is an input.
  - **The operand beside it**: contiguous rows are read in place (or converted into the other block), one value a
    row is read in place, a second tiled operand gets its own block. One kernel call a tile when every operand's
    rows of the tile follow each other and the output's do; else one call a row.
  - **Not tiled**: 16-byte elements with rows over 64 elements (a 16-byte element is one move whatever the route:
    0.93× at 400 complex128 elements a row; rows of 3 to 40 gain 1.4–2.9× from the saved calls) and conversions to
    complex128 (the house cast is slower than the per-element kernel: 0.70×); two F operands (the result is
    F-ordered: the iterator route); `out=` calls (they arrive as chunks).
- **A matrix against one row** (`ComparisonRun.RunRepeatedRow`). A kernel call costs about 8 ns and a row that ends
  in a partial vector pays a scalar tail, which is most of a short row. When the matrix's rows follow each other,
  there are 16 rows or more and a row is at most 2,048 bytes, the row is repeated in an 8 KB stack block (filled by
  doubling) and a block of rows is compared per call. Rows under 16 elements used to go to the iterator's 2-D block
  kernel (1.5 ns a row plus 1 ns an element); as a matrix against one row they take this route
  (`RepeatsRow`, also asked by `TryRowsComparisonOp`). Above 2,048 bytes a row the block's copy costs more than
  the calls it saves (1.05× the time at 3,200 bytes), and with fewer than 16 rows too (1.1× at 7 rows).
- **Kernels added** (`Direct/`):
  - complex128 (`DirectILKernelGenerator.Comparison.Complex.cs`): four elements a step. Two loads are split into
    real and imaginary lanes (`vunpcklpd` / `vunpckhpd`), compared by NumPy's `CLT` rule (a NaN in an imaginary part
    makes the ordered operators False), and the mask bits become bytes (`pdep`, or integer arithmetic without BMI2).
    `>` is `<` with the operands exchanged, `!=` the complement of `==`. 0.66 ns an element for `<`, against
    NumPy's 0.8–1.45;
  - int64 against uint64, either order (`.Comparison.MixedSign.cs`): NumPy's exact loops as machine compares. A
    negative lane is decided by its sign, the others compare unsigned; eight elements a step. The emitted loop
    compared in `System.Decimal`;
  - the strided move (`.StridedMove.cs`): four typed moves a pass at any two byte strides;
  - the tile transposition (`.TransposeTile.cs`, C# with AVX2 intrinsics): see "Tiles" above.
- **Fixed on the way.**
  - `np.less(int64, uint64, out=o)` compared in float64 (inexact past 2^53); every route is exact now.
  - Two one-element operands of different rank (`(1,) < 0-d`) took 1.03 µs through an iterator; 0.44 µs now
    (NumPy 0.57).
  - Two library defects that the tests of this work exposed, each with its own commit: "A strided copy or cast never
    reads outside its view" and "`NDArray<T>.reshape(Shape)` reads the DIMENSIONS of its argument".
- **Gates.**
  - `Backends/Kernels/ComparisonRunTests.cs` (19 tests): every dtype pair × operator × 1-D layout against an element
    reference written from NumPy's rules; two- and three-dimensional layouts; the four `out=` forms (a stepped out's
    skipped slots are not written); long runs off a vector boundary; scalar adoption over every uint8 / int8 value,
    every float16 bit pattern and the float32 values around each threshold; int64 against uint64 on every route; an
    out that is an input; `Run` and `RunRows` on raw memory with sentinel margins, and against inaccessible pages
    (`FencedMemory`); the complex, mixed-sign and strided-move kernels; the run against the routes it replaced
    (`ComparisonRun.Disabled`).
  - A mutation pass of 35 planted defects (kernels, adoption, staging, stores, the rows route, dispatch): 33 fail
    one test or more. The two survivors give the same results (a range test at the boundary, where the restated
    comparison is all-False as well; a branch of the bit-pattern step its caller never reaches).
  - A scratch sweep against an independent reference (15 × 15 dtype pairs × 6 operators × layouts × `out=` forms ×
    2-D and 3-D broadcasts): 373,543,380 checks, 0 failures, with AVX2, with AVX2 off and with hardware intrinsics off.
  - The unit suite, the oracle replay, and both again under `NUMSHARP_DEBUG_GUARD_PAGES=1`.
  - **The tile and repeated-row routes** (5 more tests in `ComparisonRunTests.cs`, 24 in all, and
    `Backends/Kernels/TransposeTileTests.cs`, 3 tests): an F-ordered operand on either side, with reversed rows or
    columns, two F slices, a column against an F operand, and a matrix against one row, held to the element
    reference over 19 dtype pairs; the routes against the row-by-row loop and against C-ordered copies up to
    4.4 million elements (the taller tiles), with counters that prove which route ran
    (`ComparisonRun.TileRuns` / `RepeatedRowRuns`; switches `TilesDisabled` / `RepeatedRowDisabled`); `RunRows` on
    raw memory with a tiled operand on either side and every operand against inaccessible pages; the kernel over
    285 tile shapes × 5 widths × 3 column strides × both row directions, with sentinel margins and against
    inaccessible pages. A mutation pass of 32 planted defects: 31 fail (26 by an assertion, 5 by ending the test
    host: they read or write outside a buffer); the survivor gives the same results (a condition that the
    output's stride already implies). The sweep, with a section for these forms over all 225 dtype pairs:
    453,687,480 checks, 0 failures in the same three configurations.
  - **Trap of the mutation script:** a test host that ends in the middle of a run still prints a summary line for
    the tests that had finished ("Passed! - Failed: 0, Passed: 1, Total: 1"). Five out-of-bounds mutants first read
    as survivors that way. Compare the summary's total with a clean run's, or look for "aborted".
- **Perf** (NPY/NS, two pinned P-cores, tier-up forced with `DOTNET_TC_CallCountingDelayMs=0`, per-cell minimum over
  three NumSharp runs and two NumPy runs; 1,279 cells: A contiguous 13 dtypes × 6 operators × array / value forms ×
  1K / 100K / 10M, B 20 layouts, C 12 dtype pairs, D `out=` and `where=` forms, E C# literal operands, F tiny and
  edge operands). In brackets: the build before this work, measured the same way.
  - All cells: geomean **2.08×** (1.15×); under 1.5×: 451 (904); under 1.0×: 23 (445); under 0.5×: 0 (88); the
    lowest cell 0.86× (0.03×). The engine alone, before the pool aligned its buffers: 2.05×, 463, 43, 0.81×.
  - By section: A 1.85× (1.62×; no cell under 1.05×), B 2.16× (0.75×), C 3.09× (1.20×), D 2.25× (1.75×),
    E 1.47× (0.73×), F 1.43× (1.37×).
  - float64 at 100K elements, NumPy µs | before → after: a matrix against a row 37.1 | 75.8 → 14.6 (2.55×), against a
    column 38.1 | 107.6 → 11.2 (3.41×), an outer product 58.3 | 107.9 → 10.6 (5.49×), reversed columns
    133.2 | 113.1 → 25.4 (5.25×), a reversed 1-D pair 63.4 | 107.6 → 18.5 (3.42×), a stepped array against a value
    63.9 | 109.5 → 30.9 (2.06×), a fully broadcast operand 11.6 | 107.7 → 10.5 (1.10×).
  - Other dtypes at 100K: int32 against a column 20.6 | 108.5 → 8.0 (2.58×), uint8 reversed columns
    85.0 | 112.1 → 12.5 (6.79×), complex128 `<` 143.1 | 256.0 → 90.7 (1.58×) and against a value
    158.1 | 273.6 → 73.6 (2.15×).
  - Two dtypes at 100K: a float32 array against a float64 value 58.1 | 70.0 → 5.95 (9.77×), an int32 array against a
    float64 array 63.8 | 70.4 → 24.0 (2.65×), a uint8 array against an int32 value 45.5 | 35.3 → 3.7 (12.2×);
    int64 against uint64 62.1 | 387.7 → 28.0 (2.22×; at 1K 1.21 | 4.36 → 0.71, 1.72×).
  - `out=` at 100K: a stepped bool out 63.7 | 104.7 → 38.1 (1.67×), a float64 out 107.1 | 95.2 → 56.9 (1.88×).
  - C# literals at 100K: `float32 > 50.5` 6.8 | 70.3 → 7.1 (0.95×), `uint8 > 50` 4.5 | 45.3 → 4.7 (0.96×),
    `int32 > 50.5` 59.4 | 70.4 → 8.1 (7.3×), `float16 > 50.5` 261.8 | 275.2 → 15.7 (16.7×), `uint64 > 50`
    34.2 | 89.1 → 19.6 (1.74×).
  - The numpy.matrix benchmark (97 cells) reads geomean 2.30×; `m > 0.75` 1.99× / 1.29× / 1.20× at 4 / 64 / 512.
  - **With the tile and repeated-row routes** (the same protocol and cells): geomean **2.12×**; under 1.5×: 425;
    under 1.0×: 22; the lowest cell 0.87×; section B 2.27× (2.16× without them). NumPy µs | routes off → on, one
    executable (`CMP_OLD_ROUTES=1` switches them off):
    - C against F at 1K: float64 2.28 | 1.65 → 0.84 (1.38× → 2.72×), bool 1.87 | 2.11 → 0.67 (0.89× → 2.80×);
      at 100K: float64 78.2 | 63.5 → 35.3 (1.23× → 2.21×), float32 51.4 | 50.3 → 21.2 (1.02× → 2.42×), int32
      54.0 | 52.7 → 23.8 (2.27×), int64 83.1 | 67.1 → 39.6 (2.10×), uint8 46.5 | 32.9 → 9.3 (1.41× → 5.01×).
    - A matrix against a row at 100K: uint8 8.97 | 7.23 → 3.84 (1.24× → 2.33×), bool 8.95 | 7.09 → 3.71 (2.41×),
      float32 20.3 | 8.46 → 6.82 (2.97×); float64 (3,200 bytes a row) keeps its call a row (2.81×).
    - New cells (section G, 54 cells: 3.61× in the geometric mean, none under 1.5×; 1.17× and 48 under 1.5× with
      the routes off). C against F at 1M: float64 1281 | 1290 → 680 (0.99× → 1.88×), uint8 616 | 656 → 151
      (4.08×). A matrix against its transpose, 316 × 316: float64 74.2 | 63.7 → 36.6 (2.03×), uint8
      46.8 | 35.1 → 8.5 (5.50×); 1000 × 1000: float64 1104 | 1111 → 548 (2.02×). A (25000, 4) matrix against a
      row: float64 140 | 85.1 → 12.8 (1.65× → 10.99×), uint8 122 | 64.5 → 3.95 (30.8×); (250, 4): 3.6–4.8×.
    - The other cells did not move: geomean 0.981 of the earlier build's time over the 1,279 cells, 4 cells read
      10 % slower, and those read 0.97–1.02 in interleaved runs of the two builds.
- **The pool aligns its buffers (2026-10-10, the commit after the engine).** The engine left the contiguous
  two-array cells at 100K elements between 0.81× and 1.3×, and the reason was the buffer's ADDRESS. Kernel only, ns
  an element (input offset / output offset in bytes past a 64-byte boundary): float64 two arrays 0.141 at 0 / 0 and
  0.250 at 16 / 0; float32 0.074 and 0.126; uint8 against a value 0.026 at 0 / 0 and 0.036 at 16 / 16. A prefix
  (`AlignmentPeel`) aligns ONE stream; pooled buffers were 16-byte aligned, so each array won or lost by its address.
  NumPy's are too: `ctypes.data % 64` is uniform over 0 / 16 / 32 / 48 up to 800 KB.
  - **What changed** (`Backends/Unmanaged/Pooling/SizeBucketedBufferPool.cs`): a buffer of one page or more
    (`AlignedMinBytes` = 4096) starts on a 64-byte boundary (`BufferAlignment`), whether it is fresh, zeroed, reused
    from a bucket or outside the pooled window. The pool allocates a block 80 bytes larger, hands out the first
    boundary a header past its start, and keeps the block's own pointer and the requested size in that header.
    A header, not `NativeMemory.AlignedAlloc`: the zeroed path must stay `calloc` (lazily zeroed pages for a large
    `np.zeros`), and with the header both paths end in the same `NativeMemory.Free` on every operating system.
    Every Native block comes from the pool's `Take` / `TakeZeroed` (two sites in `UnmanagedMemoryBlock`), so the
    change is contained in the pool.
  - **Rule it adds:** `Return(ptr, bytes)` must get the size the buffer was taken with. The size selects the bucket
    AND tells whether the buffer has a header (a `Debug.Assert` compares it with the header's). Never free a pool
    pointer directly.
  - **A cache line, not a page.** A page boundary was not tried: it costs up to 4 KB a buffer, and buffers that all
    start at one page offset put every stream of a many-operand kernel (`np.evaluate`, `np.where`) in the same
    cache set at each step. A cache line is enough for the kernels measured: the distance between an output's and
    an input's offset inside a page does not change their speed (`bool == value` over 100,000 elements takes
    2.55–2.70 µs at twelve distances from 0 to 4032 bytes, `uint8 < uint8` 3.23–3.40).
  - **Measured** (the comparison benchmark, per-cell minimum of three runs of each build; the 10M cells again in
    interleaved runs): at 100K elements geomean 0.967 of the unaligned time, 82 of 614 cells 10 % or more faster,
    4 slower by 10 % or more (two one-byte offset views, `bool == True` with its literal's temporary, complex128 `>`
    against a value); at 1K 0.997; at 10M 0.986 (no cell moves 10 %). Two arrays at 100K, µs: float64 20.0 → 15.1
    (0.98× → 1.30× NumPy), float32 10.6 → 8.5 (1.01× → 1.25×), int32 11.4 → 9.4 (1.19× → 1.43×), int16 6.4 → 5.7
    (1.14× → 1.29×). An array against a value moved by 0–4 % (the prefix aligned its one stream already) and
    one-byte dtypes by 1–6 %.
  - Gates: `Backends/Unmanaged/PoolBufferAlignmentTests.cs` (4 tests); the unit suite in a Debug build, where the
    `Debug.Assert` on the returned size is live (18,478 passed, no assertion).
- **Floors** (with the aligned pool and the two row routes every cell under 1.0× is between 0.87× and 1.00×;
  22 cells of the 1,279):
  - *Two stepped float64 / int64 operands at 100K* (8 cells, 0.89–0.98×): 3.2 MB of source, above this CPU's 2 MB
    L2. NumPy's scalar loop overlaps its work with the memory latency.
  - *A C# literal operand at 100K* (9 cells, 0.88–1.00×): the literal's 0-d temporary costs about 1 µs of a 4.6 µs
    call (plan X26). The same comparison with a held 0-d operand is 1.07× or more (`uint8`: 3.6 µs, NumPy 5.6).
  - *Three 1K cells at parity*: two stepped float64 / int64 operands (0.96–0.98×). Probed: the two staging copies
    are 0.42–0.48 µs of the 0.61 µs the run takes (0.2 ns an element for each stream; a direct two-load
    deinterleave written for the probe measured 0.18), and the call's fixed part is 0.65 µs where contiguous
    operands take 0.43. No cheap lever was found.
  - *`np.less(a, b, out)` of two float64 arrays at 100K* (0.97×): both libraries take 14.8–15.6 µs, the transfer
    bound for 1.7 MB of operands and result in a 2 MB L2.
  - *The empty result `(0,3) < (3,)`* (0.87×): the empty-result guard of branch `strings` arrived with its rebase onto
    journey4 (2026-10-12; not re-measured).
  - *Between 1.0× and 1.5×* (403 cells): contiguous operands at 100K, in C order, F order, as a transposed pair or
    at an offset (both libraries run a vector loop at the L2 transfer bound: 1.05–1.5×), contiguous at 1K and tiny
    operands (the NDArray allocation floor: about 0.35 µs of a 0.45–0.55 µs call; 1.2–1.5×), two stepped operands
    of 4 bytes or more at 1K (1.3–1.5×), a one-byte matrix against a column at 100K (1.3×: 250 calls with a
    value each), literal operands at 1K (1.25–1.7×: plan X26).
  - *No longer floors:* C against F (2.1–5.0×, were 0.89–1.54×) and a one-byte matrix against a row (2.3–2.9×,
    were 1.2–2.0×).
- **Measurements worth keeping.** Staging, ns an element: uint8 at a step of two or reversed 0.04, any other stride
  0.2 through the strided move (the house generic strided copy takes 0.7 for a one- or two-byte dtype); float64 at a
  step of two 0.2. A reversed byte store of 1,024 results: 33 ns through the uint8 shuffle copy, 418 ns one byte at
  a time (bool has no shuffle kernel of its own, so results and bool inputs are copied as uint8). This host's clock
  held at about 3.4 GHz during these runs (a two-load vector loop takes 0.59 ns an iteration): do not estimate
  cycles at the nominal 5.5 GHz. During the tile work it read 2.8 GHz (a chain of dependent 64-bit multiplies, 3
  cycles each, took 1.06 ns a step): measure the clock before a port or latency argument. The host was also not
  quiet then (about 17 % background load): two routes were timed in one process, in turn, five passes, the minimum
  of each (`ComparisonRun.TilesDisabled` / `RepeatedRowDisabled`), and effects under 1.3× were not acted on.
- **Merged with `strings` (2026-10-12, `strings` rebased onto journey4).** The engine's edits stayed out of the hunks
  `strings` changed in `DefaultEngine.CompareOp.cs` (the top of `ExecuteComparisonOp`, the general fallback, the six
  overrides, the result allocation of the iterator route) and repeated its comparison-type lines of
  `DefaultEngine.UfuncOut.cs` byte for byte, so the rebase met no conflict there. The weak-scalar step
  (`ExecuteComparison` → `AdoptWeakComparisonScalar`) runs BEFORE `ExecuteComparisonOp` and the `out=` / `where=`
  route: an operand that NEP 50 treats as weak arrives in the array's dtype already, and the run's scalar adoption
  serves the strong 0-d operands (`char`, `Half`, `decimal`) and the out-of-range Python ints. `ComparisonRunTests`
  holds its element reference to that rule (`Expected` / `PackWeak` for a Python scalar against a float16 / float32
  array, `RaisesOverflow` for a bool array against a Python int at or past 2^63).

### Type Promotion & Dtype
`can_cast`, `common_type`, `datetime_data`, `dtype`, `dtypes` (the `np.dtypes` class module), `find_common_type`, `finfo`, `iinfo`, `isdtype`, `issubdtype`, `min_scalar_type`, `mintypecode`, `promote_types`, `result_type`

**The dtype system is NumPy 2.x's two-level model (NEP 40/41/42/43/50/56), Stage A of `docs/plans/dtype-system.md`**
(`src/NumSharp.Core/DTypes/`, gates `test/NumSharp.Tests/DTypes/*` + the `dtype_text` fuzz tier; every rule probed
against 2.4.2). A **`DTypeMeta`** is the DType CLASS (`type(np.dtype('f8'))` = `np.dtypes.Float64DType`): one live
object per class with the NEP 42 slots as virtuals — `CommonDType` (null = `NotImplemented`), `CommonInstance`,
`DefaultDescr`, `EnsureCanonical`, `DiscoverDescrFromObject`, `IsKnownScalarType`, the NEP 55 reservations
`HasReferences`/`GetClearLoop`/`GetFillZeroLoop`, and the per-class-pair **`CastingImpl`** table (NEP 43
`ArrayMethod`: `ResolveDescriptors` + `GetStridedLoop`, the latter unused until Stage C; `Casting` is the impl's
minimal safety). A **`DType`** is the INSTANCE (the descriptor, `numpy.dtype`): `Meta` + `byteorder` + the parametric
payload (`DatetimeMetadata`), with NumPy's whole surface — `num`/`kind`/`char`/`itemsize`/`alignment`/`name`/`str`/
`descr`/`isbuiltin`/`isnative`/`hasobject`/`flags`/`newbyteorder`, `ToString()` = `str(dtype)` and `ToString(true)` =
`repr(dtype)`. **`NPTypeCode` stays the storage/kernel discriminator**: every storage class exposes it as
`Meta.TypeCode`, kernels keep dispatching on it, and a class without storage (the datetime pair in Stage A) has
`TypeCode == Empty` — its descriptors parse, print, promote and cast-check, but converting one to `NPTypeCode`/`Type`
raises `NotSupportedException` naming Stage C rather than degrading to the "infer" state. Classes:
`LegacyBuiltinDTypeMeta` × 16 (the 15 storage types + the vestigial `String` slot; NumPy type numbers in the LP64
convention — `Int32 = 5 'i'`, `Int64 = 7 'l'`, `Half = 23`; Decimal 256 / Char 257 are the user-defined range, so their
descriptors report `isbuiltin == 2`, repr `dtype(decimal)`), `DatetimeDTypeMeta` × 2 (`DateTime64DType` 21 `'M'`,
`TimeDelta64DType` 22 `'m'`, PARAMETRIC), `PyScalarDTypeMeta` × 3 (`_PyLongDType`/`_PyFloatDType`/`_PyComplexDType`,
ABSTRACT — the NEP 50 weak literals). `DTypeRegistry` owns them (`FromTypeCode`/`FromScalarType`/`FromTypeNum`/
`FromName`, `Register` for future classes, casting-impl registration); `np.dtypes` is the NumPy-named facade with NumPy's
aliases (`ByteDType`, `IntDType`, platform `LongDType`, `LongLongDType`, `LongDoubleDType`→Float64, …).

**Engines.** `DTypePromotion` is `common_dtype.c`/`convert_datatype.c` verbatim: `CommonDType` (two-sided protocol, then
`DTypePromotionError` — a `TypeError` — with NumPy's text), `PromoteDTypeSequence` (`reduce_dtypes_to_most_knowledgeable`,
order-independent; the OLD fold dropped every other pair, so `result_type(i1, i1, f8, i1)` used to be `int8`),
`PromoteTypes` (identical-native fast path → common CLASS → default descr, or for a parametric class cast both operands
to it and take `CommonInstance`), `CastDescrToDType`, and `ResultType` = NEP 50: every `NDArray` (0-d too), `DType`,
`NPTypeCode`, `Type`, dtype string, `bool`/`char`/`Half`/`decimal` is STRONG; C# integer/`float`/`double`/`Complex`
literals are WEAK (`result_type(int8_dtype, 300)` → `int8`, `+ 1.5` → `float64`, lone literal → int64/float64/complex128).
For the storage builtins `CommonDType` IS the frozen `np._nptypemap_arr_arr` table (deferring to the larger type number,
as NumPy does), so every two-operand answer the `dtype_text` tier gates is unchanged; the three internal engines
(`np._FindCommonType*`, `NDExprTypeRules.PromoteStrong`, `NDIterCasting.PromoteTypes`) are deliberately untouched.
`DTypeCasting` is `PyArray_CanCastTypeTo`/`CheckCastSafety`/`GetCastInfo`/`EquivTypes`/`MinCastSafety` over the impl
table: the builtin impls encode the old `np.can_cast` rules exactly (same class → `equiv`, `promote(from,to)==to` →
`safe`, kind-order `b<u<i<f<c` → `same_kind`, else `unsafe`; Char↔UInt16 are distinct classes, never `equiv`), and the
table adds what an enum could not express — byte order (`can_cast('>i4','i4','equiv')` True, `'no'` False) and the
datetime unit rules (`time_to_time`: exact 10³ᵏ folds are no-casts, generic→concrete `safe`, concrete→generic `unsafe`,
timedelta years/months barrier `unsafe`, finer-and-divides `safe`, else `same_kind`; numeric→`m8` `safe` for bool/ints
except 64-bit unsigned (`same_kind`), floats `unsafe`; anything↔`M8` `unsafe`). Casting strings are case-sensitive with
NumPy's `casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '…')` `ValueError`.

**`np.dtype(string)` is `_convert_from_str` ported**: comma-string/sub-array → unsupported; byte-order char consumed
(`'|'` reads native; a NON-native prefix is KEPT — `np.dtype(">i4")` has `byteorder '>'`, `isnative false`, `str ">i4"`,
`repr dtype('>i4')`, `isbuiltin 0`); datetime typestrs (`M8`, `m8`, `M`, `m`, `datetime64[10ns]`, `timedelta64[s/2]` →
`m8[500ms]`, the divisor→multiple table incl. NumPy's `M8[W/11]` → `M8[0Y]` quirk) via `DatetimeMetaData.Parse` with the
verbatim `Invalid datetime metadata string "[5]" at position 2` / `Invalid datetime unit in metadata string "[5x]"` /
`divisor (7) is not a multiple of a lower-unit …` texts (a zero divisor raises where CPython crashes); one-char codes;
`kind+size` parsed with `strtol` semantics (`i04`, `i 4`, `i+4` are int32; `b1` bool; `i3`/`f16`/`?1` invalid); then the
name table. Invalid strings keep `NotSupportedException` but adopt NumPy's `data type 'X' not understood` /
`Alias 'bool8' was removed in NumPy 2.0. Use a name without a digit at the end.` wording. Builtins are singletons
(`np.dtype("i8")` is the same object every call, `isbuiltin 1`); datetime and non-native descriptors are fresh. `DType`
equality is STRUCTURAL (class + byte order + unit: `M8[ns] != M8[s]`, `M8[1s] == M8[s]`, `>i4 != i4`; hash =
class+byteorder so `hash(M8[ns]) == hash(M8[s])` as in NumPy); `Equals(object/string/Type/NPTypeCode)` COERCES like
`dtype.__eq__` (`d.Equals("garbage")` is false, never an error). `<`/`<=`/`>`/`>=` are NumPy's safe-cast ordering.

**Traps pinned by tests:** (1) NEVER add `==(DType, string)` operators — a `null` literal binds them over
`==(DType, DType)` and every `descr == null` inside the engine silently answers false (it NRE'd the whole promotion
engine once); the coercing string comparison lives on `Equals(string)`. (2) C# scalar literals bind NumSharp's
scalar→`NDArray` conversion before `params object[]`, so `np.result_type(int8_array, 300)` reaches the STRONG
`(NDArray, NDArray)` overload; the weak rule is reachable through the dtype: `np.result_type(a.dtype, 300)`. A dtype
STRING likewise binds string→`NDArray` (a character array) and is ambiguous against `params DType[]`, which is why
`result_type(params string[])`, `can_cast(string, DType, …)`, `issubdtype(string, string)`, `isdtype(string, …)` exist.
(3) `promote_types('m8[Y]', 'M8[D]')` is `M8[D]` — `PromoteTypes` casts the timedelta operand to the DATETIME class first
(keeping its unit) and only then runs the GCD, so the strict years/months barrier bites on `m8 × m8` only. (4)
NumPy's `dtype.__eq__` (`PyArray_EquivTypes`) is ASYMMETRIC for the metric-prefix folds (`M8[1000ms] == M8[s]` but not the
reverse); `DType.Equals` stays structural and the NumPy relation is `DTypeCasting.EquivTypes` (`[Misaligned]`). (5)
`result_type(f4, 1j)` is `complex128` (NumPy `complex64`) — the single-complex-width collapse (`[Misaligned]`). (6) The
legacy `DType._kind_list_map` (`'?'` for bool, `'S'` for char) is what `np._FindCommonType` orders kinds through and must
not move; the descriptor's own `kind` is `'b'`/`'u'`. Stage C (datetime storage, NaT, ISO-8601, arithmetic loops
registered as `ArrayMethod`s — `docs/plans/datetime64.md`) is next; Stage D phase 1 (fixed-width `S` / `U` storage) landed
on the `strings` branch (see "Supported Types" above), phase 2 (computing on text) and NEP 55 `StringDType` follow.

**Stage B landed (2026-09-09): `DType` is the ONE dtype spelling, end to end.** **`NDArray.dtype` returns `DType`**
(`Storage.Descr` — NumPy's `PyArrayObject.descr`: the class singleton for a builtin lane, so `a.dtype == b.dtype` /
`a.dtype == np.float64` are cheap structural compares and `ReferenceEquals(a.dtype, np.float64)` holds; a parametric
descriptor is carried explicitly through `Alias`/`Clone`/`ReplaceData` and reset when the lane changes). **The `np.float64`
family (`np.bool_`…`np.clongdouble`, `np.@decimal`, `np.intp`/`np.@long`/`np.@uint`, the throwing `complex64`/`csingle`/
`chars`) are `DType` descriptors**, not `Type`s — NumSharp has no scalar-type objects (a C# `double` IS the scalar), so the
descriptor is the one currency and `np.float64.itemsize`/`.name`/`.kind` read like NumPy. **Every `dtype`/`typeCode`
parameter across np.*/NDArray takes ONE `DType`** (Creation, IO, `frombuffer` entry, `indices`, `trace`, `clip`, `cumsum`/
`cumprod`/`prod`/`sum`, `nancumsum`/`nancumprod`, `amax`/`amin`/`std`/`var`, `cov`/`corrcoef`, `getfield`/`setfield`,
`finfo`/`iinfo`, Generator/`randint`/`uniform`, `concatenate`/`concat`, `matmul`/`vecdot`/`matvec`/`vecmat`/`einsum`
(`GufuncGuard.ToLoop`), `astype`/`view`, the `NDArray(DType, …)` ctors, `isdtype`, `find_common_type(DType[]…)`); the
`Type`/`NPTypeCode` twins are GONE — a `Type`, `NPTypeCode`, `NPTypeCode?` or dtype string converts implicitly, so
`np.zeros(3, typeof(int))`, `np.zeros(3, NPTypeCode.Int32)`, `np.zeros(3, "i4")`, `np.zeros(3, np.int32)` all bind the one
overload. **Deliberately kept as `Type`** (the concrete element type of a storage lane, not a user dtype request):
`TensorEngine.GetStorage(Type)`/`Cast(…, Type, …)`, `UnmanagedStorage.Allocate/Cast/AliasAs(Type)`, `NDArray.ReplaceData`,
`UnmanagedStorage.DType` (the CLR type — `Descr` is the descriptor), plus the engine-internal `NPTypeCode` currency
(`DefaultEngine.GetStorage(NPTypeCode)`, `finfo(NPTypeCode)` ctor, `NpyFormat.DtypeInfo`); the promotion family that
RETURNS `NPTypeCode` (`result_type`/`promote_types`/`can_cast` typed pairs, `find_common_type`, `_FindCommonType*`) is
unchanged — the `DType` overloads return `DType`. Four things Stage B had to get right, each pinned by a test:
(1) **`DType → Type` is an EXPLICIT conversion** (`(Type)a.dtype` or NumPy's `a.dtype.type`; `Type → DType` stays implicit).
`System.Type` declares its own `==`, so with conversions in BOTH directions `a.dtype == typeof(double)` was CS0034-ambiguous
between `Type.==` and `DType.==`, and adding `==(DType, Type)` overloads instead makes `descr == null` ambiguous (neither
target is "better" when both directions convert). One-way implicit is what makes `a.dtype == typeof(double)`,
`typeof(double) == a.dtype`, `a.dtype == "f8"`, `a.dtype == NPTypeCode.Double` all resolve to the structural, coercing
`DType.==`, and — the reason ~310 MSTest sites needed NO rewrite — `Assert.AreEqual(typeof(double), a.dtype)` infers
`T = DType` (with two-way conversion the inference fails, CS0411). Cost: a `Type` parameter fed with `.dtype` needs
`.dtype.type` / `(Type)` (Convert.ChangeType, kernel-factory lookups, `Type.Name` in messages → `dtype.type.Name`;
`DTypeDescriptorTests.TypeConversion_IsExplicit_SoTypeEqualityResolvesToDType`). (2) **Static-initialization cycle:** `np`'s
field initializers now run `DType.From` → `DTypeRegistry`'s cctor, which registers the 15×15 `BuiltinCastingImpl`s whose
minimal `Casting` read `np._nptypemap_arr_arr` — a table `static np()` fills AFTER the field initializers, so the registry saw
it as null → `TypeInitializationException` on first use. `ArrayMethod.Casting` is now `virtual` and `BuiltinCastingImpl`
computes it on FIRST READ (a per-impl constant, so laziness is unobservable); both orders (np-first, registry-first) are
probed. The `static np()` tables themselves are built as `Dictionary<(DType, DType), DType>` (the aliases ARE descriptors) and
the frozen `Type`/`NPTypeCode`-keyed maps derived from them. (3) **`isdtype` folded to the NumPy-exact `DType` overloads** — the
`NPTypeCode`/`Type` twins accepted `issubdtype`'s vocabulary (`"floating"`, `"integer"`), which NumPy's `isdtype` rejects with
`kind argument is a string, but 'floating' is not a known kind name.` (ValueError); the `NDArray` convenience overloads route
through `arr.dtype`. (4) **Test-side `Should()`:** `test/NumSharp.Tests/Utilities/DTypeAssertions.cs` adds
`Should(this DType)` → `DTypeAssertions` (`Be(DType)` — a `Type`/`NPTypeCode`/string converts in — `Be<T>()`, `NotBe`,
`BeOneOf`), declared in the `NumSharp.Tests` NAMESPACE so enclosing-namespace lookup binds every `x.dtype.Should()` (all
tests live in `NumSharp.Tests.*`) ahead of AwesomeAssertions' `Should(this object)` with no per-file `using` — the ~1,000
`.dtype.Should().Be(typeof(X))`/`.Be(np.X)`/`.Be<X>()` sites compile unchanged. **BREAKING (accepted):** `Type t = a.dtype`
and `Type t = np.float64` need a cast; `a.dtype.Name` → `a.dtype.name` (NumPy) or `a.dtype.type.Name` (C#); `np.X` where a
`Type` is required by signature (`Marshal.SizeOf(np.float64)`) → `np.float64.type`; reflection over `np.array(Array, Type, …)`
→ `typeof(DType)`; `np.isdtype(x, "floating")` raises. Gates: full suite net10.0 14,832 / net8.0 14,824 green, FuzzMatrix
30/30 + Oracle 176/176 (dtype_text tier unchanged), Interop 638/638.

### Selection
`choose`, `compress`, `extract`, `index_exp`, `indices`, `ix_`, `place`, `put`, `put_along_axis`, `putmask`, `ravel_multi_index`, `s_`, `select`, `take`, `take_along_axis`, `unravel_index`, `where`

`np.take_along_axis(arr, indices, axis=-1)` (NumPy `numpy/lib/_shape_base_impl.py`) is the per-slice
gather: it matches 1-D index and data slices oriented along `axis` and looks each output element up
in `arr` — the inverse operation to `argsort`/`argmax(keepdims=True)` (`take_along_axis(a, a.argsort(axis),
axis)` ≡ `sort(a, axis)`). NumPy implements it as advanced indexing (`arr[_make_along_axis_idx(...)]`);
NumSharp reproduces the identical semantics as ONE whole-array strided-odometer IL gather kernel
(`DirectILKernelGenerator.TakeAlongAxis.cs`, an outer odometer wrapped around a tight inner loop over
the innermost axis so the multi-level carry runs per-slice, not per-element), dtype-agnostic via a
byte-width-keyed (1/2/4/8/16) element copy. `indices` must be an INTEGER array (NumPy's
`issubdtype(dtype, integer)` — bool/float/complex raise `IndexError "\`indices\` must be an integer
array"`) whose ndim equals `arr`'s (else `ValueError "\`indices\` and \`arr\` must have the same number
of dimensions"`); the axis dimension of `indices` supplies `J` (need NOT equal `M = arr.shape[axis]`)
while every OTHER dimension broadcasts BOTH `arr` and `indices` (a stride-0 broadcast source view is
read fine; a non-axis conflict raises NumPy's fancy-index `IndexError "shape mismatch: indexing arrays
could not be broadcast together with shapes …"` listing each `arange`-grid shape). `axis=None` flattens
`arr` in C (logical) order and requires 1-D `indices` (else `ValueError "when axis=None, \`indices\`
must have a single dimension."`); a negative index normalizes ONCE (`idx += M`, advanced-indexing
semantics) and anything still out of range raises `IndexError "index {orig} is out of bounds for axis
{axis} with size {M}"` reporting the ORIGINAL value. The result is a fresh writeable C-contiguous copy
of `arr`'s dtype (any source layout — C/F/strided/reversed/sliced/transposed/broadcast — is read through
its own strides, no materialisation). **≥1.5× faster than NumPy on every measured variation** (geomean
~3.5–3.7× NPY/NS; the tightest cell is a random 16M axis-0 gather at ~1.55×, latency-bound). Differentially
fuzzed **24,000 value + 18,000 adversarial** cases bit-exact vs NumPy 2.4.2 (ranks 1–6, all axes, wrap/OOB,
all element widths, non-contiguous SOURCES × non-contiguous INDICES: F/reversed/strided/swapped/broadcast/
offset) — values, shapes, dtypes and error taxonomy 100% bit-exact.

**One deliberate divergence (documented, `[Misaligned]`):** when a *non-contiguous* `indices` array holds
*multiple* out-of-range values, NumSharp reports the first in the RESULT's C-order while NumPy reports
whichever its fancy-index `NpyIter` (`PyArray_MapIterCheckIndices`) visits first — index-memory/axis order
with negative-stride flipping, an internal artifact that is neither pure min-address (fails on 1-D negative
stride) nor pure logical C-order (fails on F-layout). Only the offending index VALUE differs; the error
TYPE, axis and size always match, and it never arises from `argsort`/`argmax` output (C-contiguous → exact).

**No SIMD gather (measured):** a hardware `VPGATHERQD/QQ` gather was benchmarked against the scalar loop
for the hot 4/8-byte contiguous case at **only ~1.16×** on this host (AVX2; no AVX-512), before the SIMD
bounds-validation + scalar OOB-fallback a raise-mode gather still needs — not worth a per-width gather
kernel or its CPU-dependence. The wins are structural: the outer-odometer/inner-loop split (per-slice
carry) and a branch-light unsigned-bounds resolve.

Gates: `Indexing/TakeAlongAxisTests.cs` (37) + 44 `take_along_axis` cases in the `groupa` differential-fuzz
tier. See `Indexing/np.take_along_axis.cs`.

`np.put_along_axis(arr, indices, values, axis)` (NumPy `numpy/lib/_shape_base_impl.py`) is the SETTER twin
of `take_along_axis`: for every position in the (broadcast) iteration space it reads one index and writes one
value into `arr` along `axis`. **In-place** — mutates `arr`, returns `void`. NumPy is `arr[_make_along_axis_idx(
...)] = values`, an advanced ASSIGNMENT; NumSharp reproduces it as the exact mirror of the take gather — a
whole-array strided odometer (`DirectILKernelGenerator.PutAlongAxis.cs`), dtype-agnostic via a byte-width-keyed
element copy. `axis` is **required** (no default, as in NumPy). It shares take's iteration-shape /
`arrStrides` / non-axis-broadcast machinery (so `J != M`, negative-wrap, index-broadcast, and the identical
fancy-index `IndexError` all fall out), and adds three setter-only behaviours, all probed against 2.4.2: (1)
**`values` is BROADCAST — not cycled** — to the indexing result shape (right-aligned, extra LEADING size-1 dims
stripped, more lenient than `broadcast_to`); a mismatch raises NumPy's verbatim `shape mismatch: value array of
shape … could not be broadcast to indexing result of shape …`. It is cast to `arr`'s dtype (assignment cast:
floats truncate toward zero, over/underflow wraps — the put/place/putmask sibling convention). Where several
positions collapse onto one element (a size-1 `arr` dim, or duplicate indices), the **last write in C-order
wins**. (2) **ATOMICITY** — every index is validated against the axis bound BEFORE the first store (NumPy's
`PyArray_MapIterCheckIndices`), so an out-of-bounds index leaves `arr` completely untouched; this is TWO IL
kernels — a dtype-agnostic `PutAlongAxisValidate` pass then a bounds-check-free `PutAlongAxisScatter` — rather
than one, which also keeps the scatter's hot loop branch-free on the bound. (3) **`axis=None`** treats `arr` as
`np.array(arr.flat)`, whose view-vs-copy split is load-bearing: a **C-contiguous** `arr` is written back through
the flat view (aliased storage), while a **non-contiguous** `arr` raises read-only (NumPy's flat copy is
read-only there — the dtype check still fires first). A **COPY_IF_OVERLAP** guard snapshots `values` when it may
alias `arr` (`put_along_axis(a, reversing_idx, a)` reverses via a copy). The validation ORDER mirrors NumPy
exactly: axis → dtype → ndim → writeable → non-axis broadcast → value broadcast → per-index bounds. **Perf
(NPY/NS, Release, best-of-15):** the argsort/argmax-along-axis reconstruction (put_along_axis's purpose) is
**1.68–2.18×** (NumPy builds `_make_along_axis_idx`'s arange grids + a MapIter; NumSharp is a direct odometer);
the degenerate `axis=None` flat 1-D case is **~1.03–1.11×**, the memory-bandwidth ceiling the whole scatter
family (`put`/`place`/`putmask`) hits. **One inherited divergence** (`[Misaligned]`, the same benign class as
`take_along_axis`): for a NON-contiguous `indices` with MULTIPLE out-of-bounds values, the reported offending
index VALUE follows the validation odometer's C-order while NumPy's follows its MapIter (memory/axis) order — the
error TYPE, axis and size always match, and argsort/argmax output (contiguous) is exact, so it never arises in
practice (validated: 10,200 randomized cases + 142 metamorphic layout invariants, 0 unexplained). Gates:
`Indexing/PutAlongAxisTests.cs` (41) + 48 `put_along_axis` cases in the `groupa` differential-fuzz tier. See
`Indexing/np.put_along_axis.cs`.

`np.select(condlist, choicelist, default=0)` (NumPy `numpy/lib/_function_base_impl.py`) draws each
output element from the choice whose condition is true, FIRST matching condition winning; positions
where every condition is false take `default`. C# shapes: `condlist` is `NDArray[]` (boolean arrays),
`choicelist` is `object[]` (an `NDArray[]` binds via array covariance; scalar choices need
`new object[]{…}`), `default` is `object` (`null` ≙ NumPy's `0`). **NEP50 weak scalars, select's own
rule:** only `int`/`float`/`double`/`Complex` literals are weak (adopt the other operands' dtype) —
`bool`, `char`, `Half`, `decimal`, arrays and every `NDArray` are strong (NumPy's
`type(x) in (int,float,complex) else asarray(x)`), so a weak-int default that overflows the resolved
dtype WRAPS (`int8` choice + `1000` default → int8, 1000→−24) rather than raising; the result dtype
is `result_type(*choices, default)` folded via `NDExprTypeRules.PromoteStrong`. Conditions and choices
broadcast in two SEPARATE groups; a non-bool condition is `TypeError("invalid entry {i} in condlist:
should be boolean ndarray")`, length mismatch / empty condlist are `ValueError` (verbatim). Two code
paths, both NumPy-exact in VALUE, split by layout: the **contiguous / no-cast / full-size-array-choice**
case (the one NumPy leaves memory-bound) runs a **fused single-pass IL kernel**
(`DirectILKernelGenerator.Select.cs`) — a reverse `Vector.ConditionalSelect` chain (seed = default;
overlay choices n−1→0 so the first true condition wins), 4×-unrolled with four independent
accumulators, the scalar default hoisted once via `CreateBroadcast`, reusing `np.where`'s exact
`EmitInlineMaskCreation` bool→lane expansion — so each condition and choice is read once and the result
is written once. Every OTHER shape (broadcast / strided / cast / **scalar-choice**) declines the fused
gate (`TrySelectFused`) and takes NumPy's OWN structure — allocate the default-filled result, then
`np.copyto(where=cond)` each choice in REVERSE — where each masked write rides NumSharp's SIMD
masked-cast kernel. The composition alone was already 1.2–8.5× NumPy everywhere EXCEPT the 8-byte
(int64/float64) all-array-choice large cases, which sat at 1.22–1.54× because its `(n+1)` masked
copytos each re-read AND re-write the whole 80 MB result; the fused kernel cuts that redundant result
traffic and lifts those cells to **1.6–2.4×** (min 1.60× at int64 n=1 10M with an ARRAY default — the
irreducible 4-array-traffic memory-bound floor). **Perf (NPY/NS, Release, best-of-rounds):** array
choices **1K 3.3–5.0× · 100K 8–18× · 10M 1.6–5.8×**; scalar choices (composition) **3.3–3.8×** — no
measured cell below 1.5×. See `Indexing/np.select.cs`; gates: `Indexing/np.select.Test.cs` (21 — incl.
the fused SIMD-body/unroll/remainder/offset-slice pins) + the `select` op in the `groupa`
differential-fuzz tier (11 cases, dtype × precedence × broadcast × transposed).

**`np.take` / `np.put` negative indices under `mode='raise'`** now match NumPy's
`check_and_adjust_index`: a negative index is normalized once (`idx += n`) before the bounds test, so
`take(a, [-1])` / `put(a, -1, v)` address the last element and an index still out of range after the
shift raises (the original value survives for the diagnostic). The IL kernels
(`DirectILKernelGenerator.{Take,Put}.cs`) previously rejected EVERY negative index in raise mode — a
real bug, pinned now by `SelectionTests` (`Take/Put_Raise_NegativeIndices_Normalize`) and negative
`take`/`put` + `wrap`/`clip` cases in the `groupa` fuzz tier.

**`take`/`put`/`place` kernel performance (three levers, all in `DirectILKernelGenerator.{Take,Put,Place}.cs`).**
The gather/scatter/masked-copy kernels moved off the **per-element `cpblk`-of-a-runtime-tiny-size**
pathology (a memcpy call where a `mov` belongs — measured ~2× slower; it made `take` *lose* to NumPy at
0.68×). The per-element copy is now specialized to the dtype width via `CopyKindFor`/`EmitElementCopy` —
a typed `Ldind`/`Stind` MOV (two 8-byte MOVs for 16-byte Complex/Decimal), keyed into the kernel cache;
`cpblk` survives only for multi-element `take` axis slabs. Two more levers on top: (1) **software
prefetch** in the `take` gather (`Sse.Prefetch0` of the slab `PrefetchDistance=32` indices ahead, gated
at emit-time on SSE **and** at the call site on the gathered footprint exceeding
`IndexPrefetchThresholdBytes = 2 MiB` — below that it is pure overhead, ~1.6× slower at 100K, so a
separate `(copyKind, prefetch)` kernel is cached and the small path stays lean); NumPy does not prefetch
in take, so this is a clean win at scale. (2) **A wrapping values cursor** in `put`/`place` replacing the
per-element **`i % valuesCount` integer division** (NumPy's iterator model) — the dominant cost for
cyclic values, and the reason `place` was stuck at ~1.0× (it idiv'd once per True). **Perf (NPY/NS,
Release, best-of-rounds):** `take` flat **100K 1.0–1.3× / 10M 1.7–2.5×** (prefetch), axis-slab
**4-byte 2.4–2.6× / 8-byte ~1.1×** (memcpy-bandwidth-bound); `put` **100K 2.0–3.1× / 10M 1.2–1.4×**
(random scatter, memory-latency-bound); `place` **100K 2.0–4.5× / 10M 4-byte 2.5× / 8-byte ~1.3×**
(bandwidth-bound). The sub-1.5× cells are all memory-latency / bandwidth / allocation-overhead bound —
the same walls NumPy hits — not algorithmic.

**Index dtype validation (correctness fix).** `take`/`put` used to `astype(int64)` any index array,
**silently truncating a float index** where NumPy raises. `CastIndicesToInt64` now rejects a
non-castable index dtype with NumPy's verbatim `TypeError("Cannot cast array data from dtype('…') to
dtype('int64') according to the rule '…'")` — and reproduces NumPy's SUBTLE rule split: `take` converts
under **`same_kind`** (so a `uint64` index is allowed) while `put` uses **`safe`** (which rejects
`uint64` too); each leaks its own rule name in the message. Integer/bool indices pass; float/complex are
refused. Pinned by `SelectionTests` (`Take_FloatIndices_Throws_SameKind`, `Put_FloatIndices_Throws_Safe`,
`Take_UInt64Indices_Allowed_But_Put_Rejects`) + the 16-byte-fidelity and cyclic-wrap tests.

**`np.put` / `np.take` are NumPy's `PyArray_PutTo` / `PyArray_TakeFrom`, statement for statement** (2026-10-08, found as
neighbours of numpy.ma's put; plan `docs/plans/api-parameter-parity.md` X31; `Indexing/np.{put,take,clipmode}.cs`; gates
`Indexing/np.put.take.NumPyPort.Test.cs`, the `ma_put` tier and the aliasing tier's `put_selfvals` / `put_selfidx` (243
cases: the values or indices a view of the target); probed against 2.4.2):
- **The mode** is `ClipModeConverter` (NumPy's `PyArray_ClipmodeConverter` + `clipmode_parser`), shared by `put`, `take`,
  `choose` and `ravel_multi_index`, and parsed FIRST: the first character selects, the exact lowercase spelling is
  required (`"Raise"` is `ValueError: Use one of 'clip', 'raise', or 'wrap' for clip mode`), an empty or unknown string
  is `ValueError: clipmode must be one of 'clip', 'raise', or 'wrap' (got 'bogus')`, `null` is NumPy's None (raise).
  `ravel_multi_index`'s `modes` is `ConvertClipmodeSequence` (`list of clipmodes has wrong length (1 instead of 2)`),
  converted before the coordinates are counted. NumSharp's old parser raised ArgumentException with its own text and
  read `""` as raise.
- **put's order:** mode → writeability → the index cast (safe; an EMPTY float index array is the TypeError too) →
  IndexError `cannot replace elements of an empty array` → the value packing (a literal is range-checked even when
  nothing is written) → the no-op of empty values → the kernel (out of bounds: IndexError, the indices before it written
  into a contiguous target, as NumPy's loop writes).
- **put's overlap copy:** NumPy writes into a copy of the target when it may share memory with the converted values or
  indices (`arrays_overlap`); NumSharp wrote in place — `np.put(a, [1, 2, 3], a[0:3])` gave `[0, 0, 0, 0, 4]` (NumPy
  `[0, 0, 1, 2, 4]`) and `np.put(a, a, v)` raised an IndexError on an index it had just written.
- **take's order:** AxisError (a 0-d source is raveled first, so only axis 0 / -1) → the index cast → the `out` checks
  (ValueError `output array does not match result of ndarray.take`, the safe cast, `WRITEBACKIFCOPY base is read-only`) →
  IndexError `cannot do a non-empty take from an empty axes.` only when the RESULT is non-empty → the gather, whose raise
  loop checks every index of every outer position even when the slabs are empty (`np.take(np.zeros((3, 0, 0)), [0],
  axis=1)` raises, an empty outer block checks nothing). Out of bounds is NumPy's IndexError (was
  ArgumentOutOfRangeException "(Parameter 'idx64')").
- **C-order casts:** put's values, the take / put index cast (`CastIndicesToInt64`), place's values and mask, putmask's
  mask and unravel_index's indices were converted with `astype`'s default `'K'` and read linearly by the kernels, so an
  F-order operand of another dtype was read in MEMORY order (`np.put` of an F-order int32 array into int64 wrote `[10,
  12, 11, 13]`, NumPy `[10, 11, 12, 13]`; `np.putmask` / `np.place` with an F-order int8 mask selected the wrong slots);
  every such cast is now `order: 'C'` — except unravel_index's, which since 2026-10-08 is laid out in NumPy's ITERATION
  order together with its result (see "Sorting & Searching").

**`np.putmask(a, mask, values)` — the position-cursor sibling of `np.place`** (NumPy `PyArray_PutMask` /
`npy_fastputmask`; `Indexing/np.putmask.cs`, kernel `DirectILKernelGenerator.PutMask.cs`). Writes
`a.flat[i] = values.flat[i % values.size]` wherever `mask` is True, walking both in C-order. It shares
`place`'s whole structure — the writeable-first check, the same-`size` (not shape) mask contract, the
non-bool-mask→`!=0` cast (NaN/inf→True), the `ascontiguousarray`+`copyto` writeback for non-contiguous
targets, and the `arrays_overlap`→ENSURECOPY guard (`NDMemOverlap.SolveMayShareMemory(maxWork:0)`, NumPy's
`NPY_MAY_SHARE_BOUNDS` — a fresh copy of `a` when `values`/`mask` may alias it, so `putmask(a, m, a[2:8])`
reads the ORIGINAL `a` for every cyclic value instead of an already-overwritten slot; probed against 2.4.2,
this was a real bug the contiguous fast path had before the guard) — and differs in exactly TWO probed ways
(2.4.2): the values cursor advances by **POSITION**
(every element, `j` in lockstep with `i`, wrapping at `nv`) rather than per-True, and an **empty
`values` is a silent no-op** rather than `place`'s `ValueError`. The IL kernel mirrors Place's typed-MOV
scatter but with the cursor advance OUTSIDE the mask-True branch, plus a cursor-free `nv==1` scalar-
broadcast fast path (NumPy's own `if (nv==1)` branch). Errors are NumPy-verbatim: read-only →
`"putmask: output array is read-only"` (reported first, before mask-size/nv), wrong size →
`"putmask: mask and data must be the same size"`. Like `place`/`put`, `values` is unconditionally
`astype`-cast to `a`'s dtype (the NumPy ndarray-`safe`-cast vs python-scalar-coercion error split is the
documented sibling simplification — the oracle never feeds cross-dtype values). All 15 dtypes; scalars
convert implicitly (`np.putmask(a, cond, 44)`). **Perf (NPY/NS, best-of-9, Release):** geomean **1.80×**,
faster than NumPy on every cell; **2.0–3.07×** at 1K/100K and **1.59–1.77×** at 10M for 1/4-byte dtypes;
the 8/16-byte 10M cells (int64/float64/complex128) compress to **1.08–1.36×** — the memory-bandwidth wall
the whole scatter family hits (both sides RMW scattered wide-element writes). Gates:
`Indexing/SelectionTests.cs` (`PutMask_*`, 30 — incl. the position-vs-place contrast, empty-values no-op,
transposed/negstride writeback, operand-overlap (values/mask aliasing `a`, full self-alias `putmask(a,m,a)`),
complex/float16 mask→bool (both parts / half-truthiness), mask-with-more-dims (size not shape), NaN mask,
char/decimal/complex) + the `putmask` fuzz tier
(`putmask.jsonl`, 264 — 11 dtypes × 3 value modes {scalar nv==1 / cycle nv==3 / long nv==size} × 8
layouts incl. the non-contiguous writeback path, all bit-exact vs NumPy 2.4.2).

**Subshaped fancy-SET into a NON-contiguous destination (correctness fix — silent corruption).** A
fancy assignment consuming FEWER axes than `ndim` (`arr[[0,2]] = v` on a 2-D+ array — each selected
index writes a whole trailing SUB-array) used to route unconditionally through `SetIndicesND<T>`, whose
block-`MemoryCopy` copies one CONTIGUOUS `subShape` per selected offset. That assumption holds ONLY for a
C-contiguous destination; into a **transposed / F-contig / row- or col-strided / negative-stride** view
the sub-arrays are strided in the buffer, so the copy landed each row at the wrong physical offsets —
silently corrupting the view in Release (`f = arr.reshape(3,4).T; f[[0,2]] = 99` set `f[2,0..2]`'s buffer
neighbours instead of `f`'s column-strided row 2). The dedicated `SetIndicesNDNonLinear<T>` — long a
`throw new NotImplementedException` stub behind a commented-out `//TODO:` dispatch — is now implemented
as the scatter mirror of the getter's `FetchIndicesNDNonLinear`: it walks each sub-array as an odometer
over the trailing `subShape` axes, advancing the destination offset **incrementally** through the view's
strides (section D "Incremental coord advance") from the C-contiguous (already broadcast to `retShape`)
value buffer, and the setter dispatches to it whenever `!source.Shape.IsContiguous`. The C-contiguous
block-copy fast path is untouched. Bit-exact with NumPy 2.4.2 across all 15 dtypes × {F-contig,
transposed 3-D, row/col-strided, negative-stride} × {scalar, sub-row broadcast, full grid} values
(264-case differential + Char/Decimal self-consistency). The scatter is memory-latency bound (a strided
write is a cache miss per element; NumPy is ~1.3× faster on the dense case and up to ~3× on the
cache-friendly col-strided one — 1.5× is not reachable on a memory-bound scatter, and the previously
"fast" path was returning WRONG data). Gate: the index oracle's new `order='K'` setter cases
(`gen_index_oracle` copies view bases with `order='K'` so the SET runs into a genuinely non-contiguous
destination — 11 appended `set:fancy*` cases over `AT`/`BT`/strided layouts, 7 of which diverge on the
pre-fix code) + `AuditV2` `T1_15a`/`T1_15b`. See `Selection/NDArray.Indexing.Selection.Setter.cs`.

### Grid / slice-expression DSL (`r_`, `c_`, `ix_`, `s_`, `index_exp`)

NumPy's `_index_tricks_impl.py` index-expression family. All probed against 2.4.2; tests in
`Creation/np.r_.Test.cs` + `Indexing/np.ix_.Test.cs`; oracle jobs in the `manip` tier
(`gen_oracle.gen_index_tricks` ↔ `OpRegistry` cases `r_`/`c_`/`ix_`, 2,262 cases).

**How a Python slice literal is spelled in C#.** C# has no `a[1:5:2]` syntax, so a slice is written
as a **string**: `np.r_[0:5]` → `np.r_["0:5"]`. Strings do double duty here exactly as they do in
NumPy, and are told apart by a colon — **a string containing `':'` is a slice expression, a string
without one is a NumPy special directive** (`"r"`, `"c"`, `"-1"`, `"0,2"`, `"1,2,0"`). The two
grammars are disjoint upstream too (a directive never contains a colon, because Python's own syntax
supplies the slices), so every NumPy expression transcribes verbatim. One string may hold several
comma-separated slices (`np.r_["1:3, 5:8"]` ≙ `np.r_[1:3, 5:8]`), and a `Slice` / `Slice[]` object
works as an entry too, so `np.s_[…]` composes into `np.r_[…]`.

- **`np.r_[…]` / `np.c_[…]`** — static properties returning `RClass`/`CClass`, both `AxisConcatenator`
  subclasses with a `this[params object[] key]` indexer (the port of NumPy's `__getitem__`). `r_`
  concatenates along axis 0; `c_` is `r_["-1,2,0", …]`, i.e. entries are upgraded to ≥2-D with the 1s
  POST-pended (a 1-D entry becomes a **column**) and stacked on the last axis. Slice entries become
  `arange(start, stop, step)`, or — for an **imaginary step** (`"−1:1:6j"`) — `linspace(start, stop, N)`
  with the stop **inclusive**. Integer-ness follows the LITERAL, not the value: `"0:5"` is int64 while
  `"0.0:5"` is float64. A **missing stop re-reads start as stop** (NumPy's `arange(start, None, step)`
  IS `arange(0, start, step)`), so `np.r_["2:"]` is `[0, 1]` and `np.r_["5::2"]` is `[0, 2, 4]`.
  Directives are reproduced whole: axis, `axis,ndmin`, `axis,ndmin,trans1d` (with NumPy's two DIFFERENT
  upgrade paths — the array branch's `defaxes` permutation vs the slice branch's `swapaxes(-1, trans1d)`),
  a silently-ignored 4th field, and `"r"`/`"c"` matrix coercion (the result object is an `NDMatrix`, typed
  `NDArray` by the indexer; cast it to use the matrix members). Verbatim errors: `special directives must be the first entry.`,
  `unknown special directive` / `unknown special directive '0,q'` (the comma form quotes, the bare form
  does not). **`ndmin` is deliberately UNCAPPED** — NumPy validates it against `NPY_MAXDIMS` and raises
  `ndmin must be <= ndmax (64)`, but NumSharp has no 64-dimension ceiling anywhere, so importing one
  here would make the DSL refuse ranks the rest of the library accepts; `np.r_["0,100000", x]` builds
  the 100000-dim array. It has to be CHEAP rather than merely bounded, because `ndmin` comes from a
  user-typed directive: `np.AtLeastNdView` prepends every axis in ONE `Storage.Alias` over
  `Shape.PrependDimensions` (O(ndim), 1.4 ms at `ndmin=100000`) where the per-axis `expand_dims` loop
  it replaced cloned dims+strides each step — quadratic, 27.6 s at 100000 and unbounded at 2³¹-1. An
  `ndmin` too large to allocate a shape for (`"0,2147483647"`) surfaces as `OutOfMemoryException` in
  ~1 ms, the same answer any other oversized NumSharp request gives. A non-positive `ndmin` stays a
  no-op, as upstream. The closed form is exact, not approximate: a prepended axis takes stride
  `dimensions[0] * strides[0]` (0 for a 0-d source), so strides, cached flags and the writeable bit
  are bit-identical to the loop — verified across 16 layouts × 6 ranks, broadcast views included.
- **`np.ix_(params object[])` → `NDArray[]`** — open mesh. The dtype is **PRESERVED**, not forced to
  `intp`: `ix_` does no integer validation, so a float or int8 sequence rides through and only fails at
  the later indexing call; the one exception is NumPy's own, casting an **empty non-ndarray** input to
  int64 so it does not default to float64. A bool sequence is a mask (`nonzero`). Results are **views**
  that write through, built with `expand_dims` rather than `reshape` so a stride-0/read-only operand
  keeps its strides AND its non-writeable flag (reshape materializes those and would hand back a
  writeable copy, which NumPy does not do). `ValueError("Cross index must be 1 dimensional")` — which
  a **null** sequence also raises, reproducing the outcome NumPy reaches via `asarray(None)`'s 0-d
  object array rather than leaking a bare `ArgumentNullException` from the conversion helper.
- **`np.s_[…]` / `np.index_exp[…]`** — capture an index expression instead of applying it. The
  **indexer surface mirrors `NDArray`'s**, because NumPy's `s_` passes ANY index object straight
  through (`np.s_[[1,2]]` is the list, `np.s_[..., mask]` the tuple) — so the contract is
  **`arr[X]` ≡ `arr[np.s_[X]]` for every X**, verified across the whole vocabulary. Three overloads
  cover what `NDArray`'s three public indexers do, each returning the array type that indexer
  consumes: `this[string]` and `this[params Slice[]]` → **`Slice[]`** for BASIC expressions (slices,
  slice strings, integers — `int` and `string` convert implicitly to `Slice`, so `np.s_[0]` and
  `np.s_["1:3", "::2"]` stay here), and `this[params object[]]` → **`object[]`** the moment an entry
  cannot be a `Slice` — an `NDArray` fancy index or boolean mask, `int[]`/`long[]`/`bool[]`, an
  `IList` sequence, or a basic/advanced mix like `np.s_[Slice.All, np.array(new[]{0,2})]`. Both feed
  `arr[…]` directly, including an `NDArray<T>` (its own indexers do not hide the inherited
  `this[params object[]]`). **C# divergence:** `s_` and `index_exp` are identical here — NumPy's pair
  differs only in `maketuple` (`np.s_[2::2]` is a bare slice, `np.index_exp[2::2]` a 1-tuple), but a
  lone `Slice` and a one-element `Slice[]` are literally the same call in C#. Both are kept so NumPy
  code ports verbatim; `maketuple` is exposed but has no observable effect.

**NEP50 weak scalars.** NumPy distinguishes a Python literal (`5`, weak — adopts the other operand's
dtype) from a NumPy scalar (`np.int64(5)`, strong). C# has no such split, so the mapping is by type:
`bool`, the eight integer primitives, `float`, `double` and `Complex` are **weak** (the C# literal is
the Python literal); `char`, `Half` and `decimal` — which have no Python literal — plus every `NDArray`
(0-d included) are **strong**. `NDArray.Scalar(1L)` is therefore the escape hatch: `np.r_[i8, 1L]` is
int8, `np.r_[i8, NDArray.Scalar(1L)]` is int64. A weak integer that does not fit the adopted dtype
raises `OverflowException("Python integer 1000 out of bounds for int8")` rather than wrapping, reusing
`NDExprTypeRules.CheckIntLiteralFits`; a weak float saturates to ±inf instead (`np.r_[f4, 1e300]`).
An all-literal key whose integer does not fit int64 lifts the default to **uint64** (`np.r_[2**63]`
and `np.r_[2**64-1]` are both uint64 upstream; in C# only a `ulong` can carry the value).
NumSharp has a single complex width, so NumPy's `result_type(float32, 1j) → complex64` collapses onto
`Complex`.

**Edge sweep (post-ship).** Probed outside the corpus envelope along the magnitude / structural /
resource / null / acceptance / adversarial / call-form dimensions. Three fixes came out of it — the
O(1) `ndmin` expansion, the uint64 weak default and `ix_(null)`, all described above. Matched and worth not
re-proving: `-0.0` keeps its sign bit, NaN/±inf/subnormals round-trip, the weak-integer boundaries are
exact to ±1 (`i8+127` fine, `+128` raises), `9e18:9e18+2` collapses to empty as it does upstream,
huge slice expressions raise an honest `OutOfMemoryException` rather than wrapping a byte count, and
NumPy's error ORDER holds (a bad directive is reported before the shapes behind it). `np.r_`/`np.c_`
are **stateless singletons** — a directive is copied to locals per call, so it neither persists to the
next call nor leaks across threads (checked under `Parallel.For`). Deliberate divergences: a `null`
ENTRY throws (NumPy builds an object array, a dtype NumSharp lacks — same call as `fill_diagonal`),
`np.ix_("abc")` is accepted as a char sequence (NumSharp's house string→char-array conversion; NumPy
rejects it as 0-d), and a sub-1e-15 slice step reports `np.arange`'s inherited "step can't be 0".

**Deliberate divergence — no `bmat` branch.** NumPy routes a single bare string key into
`matrixlib.bmat`, which resolves the words in it against the CALLER'S Python frame
(`sys._getframe().f_back`) as variable names. C# cannot do that, and the branch raises `NameError` for
every string literal anyway (`np.r_['1 2; 3 4']` → "name '1' is not defined"), so a lone string gets
the same reading as any other entry — which is what makes `np.r_["0:5"]` work at all.

**Perf (NPY/NS, best-of-21, Release, 100K / 10M).** Three cost classes, reported apart because a
geomean across them would describe nothing: **O(N) concatenation** (`r_[a,b]`, `r_[a,0,0,b]`, `c_[a,b]`)
**1.61× / 1.19×**, **O(N) generation** (the arange and imaginary-step linspace branches) **2.63× / 1.16×**,
and the **O(1) view** `ix_` **1.29× / 3.80×**. No cell below 1.0. The 10M ratios compress toward parity
because both sides are memory-bandwidth bound there — `r_`'s own overhead above `np.concatenate` measures
0.94 ms on a 27 ms call (3.6%), and two extra weak-scalar entries add ~0.06 ms.

See `Creation/np.{r_,c_}.cs`, `Indexing/np.{ix_,s_}.cs`.

**`np.ogrid` — the open-mesh grid.** Port of NumPy 2.x `numpy.ogrid` (the `sparse=True` instance of
`nd_grid`). It is NOT an `AxisConcatenator` — it is its own class — but it borrows the SAME slice
grammar via `AxisConcatenator.ParseSliceToken`/`SliceSpec`, so slices are spelled as strings
(`np.ogrid["0:3", "0:5"]` ≙ `np.ogrid[0:3, 0:5]`; one string may carry comma-separated slices;
`Slice` objects work too) with NO directives. A **single** slice returns a bare 1-D array (arange, or
linspace for an imaginary `Nj` step — stop inclusive); **multiple** slices return an open mesh — N
arrays each shape 1 in every axis but its own, built by `arange`→(astype)→`reshape`-to-`(1,…,size,…,1)`.
The whole mesh shares ONE dtype (NumPy's single `result_type` over every slice bound): int64 iff every
field of every slice is an integer literal, else float64 — so `np.ogrid["0:2", "0.0:2"]` yields TWO
float64 arrays, and an imaginary step floats the whole mesh. A single slice may omit its stop
(`np.ogrid["5:"]` = `arange(0, 5)`); a **multi** slice may NOT (NumPy sizes each axis from `stop-start`
and leaks an `AttributeError` — NumSharp raises a clear `ValueError`), and an imaginary step always
needs a stop. **Return type** `OGridResult` stands in for NumPy's array-or-tuple polymorphism (C#
cannot return both from one indexer): it converts implicitly to a bare `NDArray` for a single slice,
to `NDArray[]` for a mesh, and `Deconstruct`s (`var (y, x) = np.ogrid["0:3", "0:5"];`) — collapsing
NumPy's bare-array-vs-1-tuple trailing-comma distinction. Imaginary/float grids are **bit-exact** with
NumPy (same `linspace`/`arange` code path as `r_`). Values/shapes/dtypes verified against NumPy 2.4.2;
**≥1.5× faster on every measured variation** (geomean ~4.5×; NPY/NS). Gate:
`Creation/np.ogrid.Test.cs` (18). See `Creation/np.ogrid.cs`.

**`np.mgrid` — the dense-mesh twin of `ogrid`.** Port of NumPy 2.x `numpy.mgrid` (the `sparse=False`
`nd_grid`). Shares `ogrid`'s parsing and grammar via the extracted `np.ParseGridSpecs` / `GridMeshDtype`
helpers (same string-slice spelling, same shared-dtype rule, same missing-stop/imaginary `ValueError`s
with `mgrid` wording). A **single** slice returns a bare 1-D array (identical to `ogrid` single);
**multiple** slices return ONE dense stacked array of shape `(N, size_0, …, size_{N-1})` — `result[k]`
is the k-th coordinate broadcast across the grid. Built NumPy's way — `np.indices(sizes, typ)` then
each **non-trivial** axis (start≠0 / step≠1 / imaginary) is rescaled by overwriting its layer with the
1-D line broadcast across the grid (`np.copyto`); a pure `0:n` axis needs no post-pass, so the common
`np.mgrid[0:a, 0:b]` is a single fused `indices` fill. This is why `mgrid` gets its OWN build rather than
reusing `ogrid`'s lines: **NumPy sizes each axis by `ceil((stop-start)/step)`, which may be negative** —
`mgrid[5:0, 0:3]` raises `"negative dimensions are not allowed"` (via `indices`) where `ogrid[5:0, 0:3]`
clamps to an empty axis, and a zero step is a divide-by-zero — a genuine NumPy divergence between the two
that an arange-line-based `mgrid` would have silently papered over. **Return type** `MGridResult` is
NumPy's single array: implicit to `NDArray` (the stack), plus `Deconstruct`
(`var (x, y) = np.mgrid["0:5", "0:3"];`) and indexing into the per-axis grids. **This replaced the legacy
non-NumPy `np.mgrid(NDArray, NDArray)` method** (and its `NDArray.mgrid(rhs)` twin) — a deliberate
breaking change, since a property named `mgrid` cannot coexist with a method of the same name. 13-case
differential matrix bit-exact vs NumPy 2.4.2; **≥1.5× faster on every measured variation** (geomean
~6×; NPY/NS). Gate: `Creation/np.mgrid.Test.cs` (15). See `Creation/np.mgrid.cs`, `Creation/np.ogrid.cs`
(shared helpers).

**`np.meshgrid` — coordinate matrices from coordinate vectors (completed to full NumPy).** Port of NumPy
2.x `numpy.meshgrid` (`numpy/lib/_function_base_impl.py`). Replaced the old **2-input `Kwargs` stub** (which
was 2-D only and whose `copy` flag was a no-op) with the full variadic API: `meshgrid(x1, x2[, x3], string
indexing="xy", bool sparse=false, bool copy=true)` plus a `meshgrid(NDArray[] xi, …)` core for 4+ inputs.
Each input is flattened to its own open-mesh axis; **`"xy"`** (Cartesian, default) swaps the first two axes
vs **`"ij"`** (matrix) — 2-D outputs are `(N, M)` under xy and `(M, N)` under ij; **`sparse`** keeps the
open-mesh `(1,…,Ni,…,1)` shapes; **`copy`** materializes contiguous grids (false → non-contiguous broadcast
views, NumPy-verbatim). Each grid **PRESERVES its input's dtype** (not promoted). Bad `indexing` →
`ValueError "Valid values for `indexing` are 'xy' and 'ij'."`. **Return type** `MeshgridResult` — NumPy's
tuple: implicit to `NDArray[]`, `Deconstruct` (`var (xx, yy) = np.meshgrid(x, y);`), indexer. The old
`Kwargs` class and the `(NDArray, NDArray)` return are gone; in-repo callers migrated (the `NewDtypes`
sweep, the fuzz `OpRegistry` meshgrid case). Differential-verified bit-exact vs NumPy 2.4.2 (shape + dtype
+ values) across xy/ij × sparse × 1/2/3/4-D × dtype-preservation × copy-view. **Perf (NPY/NS):** dense grids
at scale are **4.7×–5.2×** (1000², 3-D 50³) and `copy=false` views **2.3×** — NumSharp's bulk broadcast-copy
beats NumPy's there; SMALL dense/sparse grids are **0.5×–1.2×**, bound by NumSharp's per-`NDArray`
construction/`.copy()` floor (~1.6 µs), the same ceiling documented for `fill_diagonal` — not the meshgrid
algorithm (it cannot use `mgrid`'s fused `indices` kernel because its values are arbitrary input vectors, not
coordinates). Gate: `Creation/np.meshgrid.Test.cs` (12). See `Creation/np.meshgrid.cs`.

**`np.ix_` — audited, no change.** The open-mesh index builder (`Indexing/np.ix_.cs`, documented above) was
re-verified against NumPy 2.4.2 across 1–3 operands incl. bool masks, empty (→intp), float-preserve, single,
and the 2-D `ValueError` — all matching, and it is already the differential-fuzz `ix_` tier (green). The only
"difference" is C# `int` literals being int32 where Python's are int64, which is language-wide, not an `ix_`
behaviour.

### Iteration
`nditer`, `ndenumerate`, `ndindex`, `nested_iters`, `flatiter` (plus the pre-existing `broadcast`), and
the NumSharp-extension typed forms `nditer<T>` / `nditer_chunks<T>`

**`np.nditer` carries NumPy's descriptors** (2026-10-08, the open-bugs rundown's bug 6; plan
`docs/plans/api-parameter-parity.md` X35). Construction is NumPy's (`NDIterRef.AdvancedNew(..., opDescrs)`,
`Backends/Iterators/NDIter.Descriptors.cs`): `npyiter_check_casting`'s read and write-back checks, the copy routes of
`npyiter_allocate_arrays` (a read-only 0-d cast operand, `copy`, `updateifcopy` written back at `close()`), the
"required copying or buffering" TypeError, allocated outputs laid out in the settled order (reused when the up-front
allocation already has it), and fixed-width strings: 0-d `S` / `U` views, buffered / copy-route conversions between
widths and between `S` and `U` (ASCII), and between text and numbers (NumPy's legacy casts, since P1.1). The
wrapper (`APIs/np.nditer.cs`) is `nditer_pywrap.c`: the started / finished flags (advance-on-entry pulls), every
getter's and setter's checks and texts ("Iterator is past the end", "Iterator is invalid", the delayed-allocation
text), `it[i] = v` packing a Python int with NumPy's range check, NumPy's write flags on the views; `np.nested_iters`
splits flags and dtypes per level (`NpyIter_NestedIters`: a buffered cast lands in the innermost level). Three engine
rules a port must keep: **the flat index rides its own strides** (`NDIterState.IndexStrides`, NumPy's index
"operand" — permuted, flipped and merged with the axes, so an index-tracking iterator coalesces when the index allows
and `remove_multi_index` cannot break it; it used to be recomputed from coordinates, which counted an `f_index` in C
order after a coalesce); **`shape` without a multi-index lists the iterator's axes innermost first** (NumPy's axisdata
order); **`remove_multi_index` only coalesces** (a forced C / F order keeps walking that order). On the descriptor
path the K-order sort is NumPy's `npyiter_find_best_axis_ordering` (`NDIterCoalescing.FindBestAxisOrderingNumPy`: a
zero stride is ambiguous, C order wins a conflict — no broadcast fallback to C); the internal callers keep the
min-stride sort. The op_axes reduction checks are `npyiter_check_reduce_ok_and_set_flags` (innermost axis first; an
explicit reduction axis needs no `reduce_ok`). **The construction checks are NumPy's, in NumPy's order** (2026-10-09,
plan X38; they were house `IncorrectShapeException`s, accepted inputs NumPy refuses, or NumPy's texts in another order):
the wrapper runs `npyiter_init`'s `itershape` rule ("'op_axes' and 'itershape' must have the same number of entries equal
to the iterator ndim" — it was accepted), `npyiter_check_op_axes` BEFORE the flag checks (a duplicate axis "contained
duplicate value 0" — it was accepted and iterated a wrong grid; an axis past NumPy's 64-dimension limit that the operand
lacks is "contained invalid values"), then `npyiter_fill_axisdata`'s shape pass (`FillAxisDataShapePass`): the plain
"operands could not be broadcast together with shapes (2,) (3,) ", the remapped "… [original->remapped]: (2,3)->(2,3)
(4,3) " and "… (2,3)  and requested shape (4,3)" texts, "Iterator input op_axes[0][0] (==5) is not a valid axis of
op[0], which has 2 dimensions " (NumPy prints the entry's index REVERSED; reproduced), "input operand has more dimensions
than allowed by the axis remapping", an `itershape` entry of 1 the operands stretch — and a NEGATIVE `itershape` entry is
deduced from the operands (it was refused; the engine takes the deduced shape). The stride pass's refusals are ONE loop,
`NDIterRef.ThrowStridePassRefusals` (the wrapper calls it before the common dtype, the engine for its internal callers):
iteration axes INNERMOST first, operands in order — `no_broadcast` (NumPy's `operand_different_than_broadcast` ValueError,
"non-broadcastable [output] operand with shape (3,) …", with `op_axes` "… [remapped to (3,1)] …", the remapped shape
printed reversed as NumPy prints it; it was an `InvalidOperationException` naming `(3)`), an explicit reduction axis of
extent other than 1, and the reduction rule (a ZERO-length iteration axis stretches an operand of extent 1 too). NumSharp
checked `no_broadcast`, the reduction length and the reduction rule in four separate passes, so with two refusals it
named the other operand. The wrapper hands the shape it computed to the engine (the internal
`NDIterRef.AdvancedNew(..., shapeValidated: true)` overload), which then skips its own broadcast computation and
`ValidateIterShape` — running both cost +30–170 ns a construction. **Open:** a BUFFERED reduction computes wrong values
for every dtype (the legacy double loop; `OpenBugs.NditerBufferedReduce.cs`) — a port of NumPy's buffered-reduce design
is the fix; and
`grow_inner` does not grow an uncast operand past the buffer size (NumPy walks `arange(10)` with a buffer of 3 as one
chunk of 10, NumSharp as 3, 3, 3, 1; values are right, only the chunking differs; `OpenBugs.NditerGrowInner.cs`) — the
fix is `npyiter_find_buffering_setup`'s core-size search in the general iterator's window sizing.
Gates: `nditer_construct.jsonl` (9,574), `nditer_protocol.jsonl` (1,219), the string oracle's `striter` section,
`APIs/np.nditer.NumPyParity.Test.cs`, and the unit suites `APIs/np.nditer.{Construction,Protocol,Strings}.Test.cs`
(2026-10-09; 51 / 94 / 59 tests from probed NumPy 2.4.2 output — the flag, argument, shape-pass, stride-pass and casting
checks, allocation, op_axes, itershape, seeks, ranges, setters, copies, buffering, axis removal and all 15 dtypes;
strings in `U` and `S` over every order and layout, width and class casts, the codec errors, the write-back routes,
allocation, op_axes / itershape and `nested_iters`).
Perf (NPY/NS): construction 1.78× (one operand, 321 ns), 2.5× (`c_index`), 3.2× (an `op_axes` reduction: 611 ns, 972
before the handover), 2.2× (`itershape`), index walks ~13×; under the bar: three broadcast operands 1.05× (850–860 ns,
~1.0× before X38 too — the engine's general multi-operand setup), a construction that allocates an output 0.96× (the
`np.zeros` floor), a per-element string walk 0.14× (the per-element view floor below).

**`np.nested_iters(op, axes, flags=null, op_flags=null, op_dtypes=null, order='K', casting="safe",
buffersize=0)` → `NDIterator[]`** — the port of NumPy's `NpyIter_NestedIters` (`nditer_pywrap.c`).
Returns one `np.NDIterator` per entry in `axes` (outermost first), all iterating the SAME operand buffer
over disjoint axis subsets; advancing an outer iterator re-bases every inner one to its new position, so
`foreach (var _ in i) foreach (var _ in j) …` walks nested loops. Built on the EXISTING primitives —
each level is an ordinary `NDIterator` with `op_axes = [axes[level]]` sharing iter0's operands, linked
parent→child, and re-based via `NDIterRef.ResetBasePointers` (which already existed). The nested levels
switch from the default publish-then-advance `MoveNext` to **NumPy's advance-on-entry protocol**
(`npyiter_next`'s `started` flag) so the child re-base and the in-body `multi_index`/`value` stay aligned
(guarded by `_nestedMode`, so plain `nditer` is untouched — 662 nditer tests green). Validation is
verbatim NumPy (`ValueError`): `axes` ≥ 2 entries ("axes must have at least 2 entries for nested
iteration"), no axis reused ("An axis is used more than once"), negative axis ("An axis is out of
bounds"); an axis past the operand's ndim is rejected at construction as `IncorrectShapeException` (house
convention) where NumPy raises `ValueError`. **The `multi_index` path — the documented primary use — is
bit-exact** with NumPy 2.4.2 across single/multiple operands, any axis order, any depth (coords AND values
pair identically). **The bare value stream (no `multi_index`) now matches NumPy's memory order too:** the
op_axes contiguity check (`NDIter.CheckAllOperandsContiguous`) judges the ITERATOR's remapped layout, not
the full operand, so a non-contiguous axis SUBSET — e.g. inner axes `[0,2]` of a C-contiguous `(2,3,2)`,
strides `(6,1)` — no longer coalesces into F-order (`0,6,1,7,…` → NumPy's `0,1,6,7,…`); verified bit-exact
over all 2- and 3-level axes partitions × value-stream/`multi_index` (a plain non-contiguous nditer was
always correct — the bug was specific to op_axes subsets). Value-read perf is bound by the same per-element
`it[0]` NDArray-view cost documented below (~4.6× on a 1M walk); the iteration ENGINE is fast. See
`APIs/np.nested_iters.cs` + the nested support in `APIs/np.nditer.cs` (the contiguity fix is in
`Backends/Iterators/NDIter.cs`); gate: `APIs/np.nested_iters.Test.cs` (7).

**`NDArray.flatiter` / `np.flat(a)` → `np.FlatIterator`** — a WRITE-THROUGH, C-order flat iterator, the
analog of NumPy's `flatiter` (the type of `a.flat`). NumSharp's `NDArray.flat` returns a raveled `NDArray`
and is deeply embedded (~15 core call sites), so its return type is IMMOVABLE and NumPy's `a.flat → flatiter`
surface cannot be reclaimed; the iterator lives on the NEW `flatiter` accessor instead. It fixes a genuine
`flat` defect: `flat` COPIES for a non-contiguous layout (via `reshape`), so `a.T.flat[i] = v` is silently
lost — `FlatIterator` reads AND writes THROUGH to the base in logical C-order for every layout (transposed,
sliced, strided, negative-stride, broadcast), because every element maps a flat index through the base's
strides (`Shape.TransformOffset`, via `GetAtIndex`/`SetAtIndex`) with **no per-element allocation**. Surface
(probed against 2.4.2): `this[long]` scalar get/set (negative wraps; OOB → `IndexError` "index N is out of
bounds for size M"); fancy (`int[]`/`long[]`/`NDArray`) and slice-string (`"1:4"`, `"::2"`) get/set; the
`index`/`coords` cursor (past-end `coords` leaves the slowest axis UNwrapped, matching NumPy: index==size on
`(2,3)` → `(2,0)`); `Base`, `size`, `copy()` (a fresh 1-D C-order array); and C-order iteration that shares
the cursor (NumPy's `iter(f) is f`, so a second pass RESUMES). Cross-dtype scalar assignment casts with NumPy
semantics: to an integer dtype an in-range value truncates toward zero, but an OUT-OF-RANGE **weak** scalar (a
C# primitive — NumSharp's NEP50 analog of a Python int/float) RAISES `OverflowException` ("Python integer 300
out of bounds for int8") rather than wrapping, matching NumPy's `a.flat[i]=v` bounds check (floats truncate
first, then the truncated value is range-checked; NaN/inf raise; a bool target never does). A **strong** scalar
— an `NDArray`, i.e. the fancy/slice setters or `a.astype` — still wraps, as NumPy wraps an `np.int64` scalar.
To a float/complex/decimal dtype it uses `Convert` (exact). **Perf (NPY/NS, Release, 100K, non-contiguous):**
same-dtype set **1.8×**, cross-dtype→float set **2.4×**, get **6.3×** — all FASTER than NumPy (the
allocation-free `GetAtIndex`/`SetAtIndex` hot path); cross-dtype→integer set is correctness-first (`astype`,
slower, rarer). `x.flat` is UNCHANGED (still the raveled `NDArray`).

**Typed, by-`ref T` flat iteration — `np.flat<T>(a, writeable=false)` / `a.flatiter.AsTyped<T>(writeable=false)`
→ `np.FlatRefIter<T>`.** The unboxed counterpart of the boxed `FlatIterator`: yields `ref T` straight into the
array's memory in the SAME logical C-order, write-through for every layout — the flat analog of `np.nditer<T>`.
It IS exactly `np.nditer<T>(a, order: 'C')` with a C-order default baked in (the order-configurable
`np.nditer<T>` defaults to memory `'K'`), so `FlatRefIter<T>` is a `readonly struct` whose `GetEnumerator()`
REUSES `NDRefIter<T>.Enumerator` pinned to `NPY_CORDER` — zero duplicated iteration logic, same `NDIterRef`
engine, verified bit-identical to the boxed `flatiter`'s C-order across contiguous/transposed/reversed/strided/
broadcast (probed). `T` must be the EXACT dtype (a `ref` can't convert → `ArgumentException`, so unlike the boxed
setters it does NO scalar coercion); `writeable:true` opens readwrite and refuses a read-only broadcast with
NumPy's message; 0-d yields its one element, empty iterates zero times, re-enumeration RESTARTS (holds no cursor,
unlike the boxed `flatiter` which resumes). **The instance spelling is `a.flatiter.AsTyped<T>()`, NOT
`a.flatiter<T>` — a generic method cannot share a name with the `flatiter` property (CS0102).** See
`APIs/np.flatiter.cs` (`flat<T>` + `FlatRefIter<T>` + `FlatIterator.AsTyped<T>`), reusing `NDRefIter<T>.Enumerator`
from `APIs/np.nditer.Typed.cs`; gate: `APIs/np.flatiter.Test.cs` (23 — `np_flatiter_tests` 13 boxed +
`np_flat_typed_tests` 10 typed).

The three NumPy iteration objects, all following the `np.broadcast` house shape — a lowercase
factory returning a PascalCase nested class — and all **their own iterator** (NumPy's `iter(x) is x`),
so a second enumeration RESUMES rather than restarting. Indices are `long` (NumPy's `intp`) and every
step yields a FRESH index array, never a recycled buffer.

- **`np.ndindex(...)` → `np.NDIndex`** — C-order odometer over a shape's index space, last axis
  fastest. `params long[]` takes both spellings NumPy allows (`np.ndindex(3, 2)` and
  `np.ndindex(arr.shape)`); a plain `int[]` overload exists because array covariance won't widen it
  (it is deliberately NOT `params`, or `np.ndindex()` would be ambiguous). `ndindex()` and an empty
  shape both yield ONE empty index (Python's `product()` with no iterables); any zero-length
  dimension yields nothing; a negative dimension raises `ArgumentException "negative dimensions are
  not allowed"` at CONSTRUCTION, before a single index is produced. Deliberately **not** NDIter-backed
  — NumPy itself moved `ndindex` off `nditer` onto `itertools.product`, and there are no operands to
  walk, only a counter. **`np.ndindex(...).AsSpans()` → `np.NDIndexSpans`** is the allocation-free form:
  a `ref struct` enumerator yielding the multi-index as a `ReadOnlySpan<long>` over a REUSED buffer
  (valid until the next step), so a whole index-space walk allocates nothing. `ndindex` walks an index
  SPACE not an array, so there is NO `ref T` form (no element values); `.AsSpans()` is its non-boxed
  equivalent, and it RESTARTS each foreach (holds no cursor). `Indexing/np.ndindex.cs`.
- **`np.ndenumerate(arr)` → `np.NDEnumerate`** — yields `(index, value)` for every element. Port of
  NumPy's `asarray(arr).flat` + `flatiter.coords`, so the order is always LOGICAL C-order whatever the
  layout: F-contiguous, transposed, reversed, sliced and broadcast views all read through their own
  strides (`Shape.TransformOffset`), a 0-d array yields exactly one pair with an EMPTY index, and an
  empty array yields nothing. NumSharp's `NDArray.flat` is a raveled `NDArray` rather than a `flatiter`
  object (no `coords` cursor), so the coordinates come from an odometer advanced in lockstep with the
  flat position — which is what `flatiter.coords` is. `np.ndenumerate<T>(arr)` is a NumSharp extension
  yielding unboxed `T` (by VALUE, LINQ-capable). **`np.ndenumerate<T>(arr).AsRef(writeable=false)` →
  `np.NDEnumerateRef<T>`** is the allocation-free, by-`ref T` form: each step yields an `Entry` ref
  struct carrying `Index` (a `ReadOnlySpan<long>` over a REUSED buffer) and `Value` (a `ref T`, so
  write-through). It COMPOSES the C-order value ref from `NDRefIter<T>.Enumerator` (the same engine
  `np.flat<T>` drives) with a lockstep C-order coords odometer — since a C-order walk visits logical
  positions 0..size-1, the k-th value and k-th unravelled coord always line up. `T` = exact dtype;
  `writeable:true` refuses a read-only broadcast; restarts each foreach. `Indexing/np.ndenumerate.cs`.
- **`np.nditer(...)` → `np.NDIterator`** — the public managed face of `NDIterRef`, i.e. the port of
  NumPy's `nditer_pywrap.c` (argument conversion, flag-string parsing, property surface, iteration
  protocol) over the C iterator NumSharp already had. Full signature parity
  (`op, flags, op_flags, op_dtypes, order='K', casting="safe", op_axes, itershape, buffersize`) with
  all 13 global and 15 per-op flag strings (incl. NumPy's `grow_inner`/`growinner` alias) and verbatim
  error texts — `Unexpected iterator global flag "…"`, `Unexpected per-op iterator flag "…"`,
  `order must be one of 'C', 'F', 'A', or 'K' (got 'Q')`, `casting must be one of …`,
  `Iterator flag EXTERNAL_LOOP cannot be used if an index or multi-index is being tracked`,
  `Iteration of zero-sized operands is not enabled`, `Must provide at least one operand`. Property
  surface: `nop`, `ndim`, `shape`, `itersize`, `operands`, `dtypes`, `value`, `it[i]`, `index`,
  `multi_index`, `iterindex`, `iterrange`, `itviews`, `finished`, `has_index`, `has_multi_index`,
  `has_delayed_bufalloc`, `iterationneedsapi`; methods `iternext`, `reset`, `close`, `copy`,
  `remove_axis`, `remove_multi_index`, `enable_external_loop`, `debug_print`. `APIs/np.nditer.cs`.
  **A written operand's reduction refusal is NumPy's SHAPE test** (2026-10-08, plan `api-parameter-parity.md` X34):
  `npyiter_fill_axisdata` marks a reduction only where the operand's own dimension is 1 or missing under a longer
  iteration axis (`NDIterRef.IsBroadcastAxis`), not wherever a stride is 0 — a zero-size array has zero strides without
  a broadcast, and `np.nditer<double>(np.zeros((3, 0)), writeable: true)` threw before (the empty complex Frobenius
  norm hit it). The refusal is NumPy's `ValueError` with NumPy's axis number, counted from the innermost axis
  (`(1, 3, 4)` against `(2, 3, 4)` is "dimension 2").

**Three traps `np.nditer` had to solve — do not re-break:**
- **`NDIterRef` is a `ref struct` and cannot be a class field.** The bridge is
  `NDIterRef.Detach`/`Borrow` (`Backends/Iterators/NDIter.Detach.cs`), which hands the heap
  `NDIterState*` — plus operands AND pending COPY_IF_OVERLAP write-backs — to the managed owner and
  re-borrows a non-owning `NDIterRef` per call. The pre-existing `ReleaseState`/`FreeState` pair is
  NOT enough on its own: it nulls the state but strands `_writebackOriginals` on the dying ref struct,
  so a copied-for-overlap temp would never be written back. `np.nditer` therefore MUST be disposed
  (`close()`/`using`), exactly like NumPy's `with np.nditer(...) as it:`.
- **`GetEnumerator()` must NOT return `this`.** `foreach` disposes the enumerator it obtains, and this
  class's `Dispose` frees unmanaged state — so returning `this` (the `np.Broadcast` pattern, safe only
  because Broadcast owns no unmanaged memory) CLOSES the iterator at the end of any `foreach`/LINQ
  pass and every later property read throws. A thin wrapper with a no-op `Dispose` shares the same
  live cursor, so the observable semantics are unchanged.
- **FIXED with the rebase of strings onto journey4 (2026-10-12): `MoveNext` is NumPy's `npyiter_next` now — it
  advances on entry (the `started` flag) and then publishes, so gap SU42 is closed and its pins in
  `OpenBugs.Surface.cs` are regression tests.** What the gap was, as found on 2026-10-08:
  **`MoveNext` published THEN advanced — NumPy does NOT.** NumPy's
  `__next__` (`npyiter_next`, `nditer_pywrap.c`) advances ON ENTRY once its `started` flag is set and
  only then returns `value`: after one `next()` the cursor still reads the element it returned
  (`iterindex` 0, probed against 2.4.2), and a `copy()` taken there yields from the SECOND element
  (the copy advances on its first `next()`). NumSharp's `MoveNext` publishes the element and then
  advances, so the `copy()` output agrees, but after k steps `iterindex` / `value` / `multi_index`
  read one element AHEAD, and in a BUFFERED readwrite loop (`foreach (var x in it) x[...] = ...`) the
  advance flushes the buffer before the loop body writes the last element of each buffer (or each
  external-loop chunk) — that write never reaches the operand. An unbuffered loop is unaffected (the
  published view points into the operand). Pinned by the surface tier (`surface_gaps.jsonl` SU42: the
  `protocol:` and write-back cases) and `OpenBugs.Surface.cs`. NumPy's separate hazard stays: under
  buffered `external_loop` a chunk view HELD past the next step aliases a refilled buffer (hence its
  `[x.copy() for x in it]` idiom).

`np.nditer` also fills a gap `NDIterRef` leaves: an ALLOCATE operand with no `op_dtypes` entry throws
in the engine, while NumPy INFERS the dtype by promoting the operands that have one — so
`np.nditer([a, null])` allocates `int64` for an `int64` input and `np.nditer([int_a, float_b, null])`
allocates `float64`. That inference lives in the wrapper, where NumPy puts it.

**Perf (NPY/NS, higher = NumSharp faster; best-of-9, Release, 100K float64):** `ndindex` **1.5×**,
`ndenumerate` **3.3×**, `nditer` with `multi_index` **8.0×**, `external_loop` ~parity — but the
per-element `it[0]` loop is **0.16–0.18×** (56–76 ms across runs; the variance is GC, it allocates
100K `NDArray`s). The iteration ENGINE is not the problem: a bare `iternext()` walk of 100K elements
takes 0.61 ms against NumPy's 2.60 ms for the same bare walk (**4.3×**). The whole deficit is the
per-element `NDArray` view (~0.6 µs each) — `new NDArray(storage, shape)` re-aliases the storage, so
the hot 0-d path takes the direct slice ctor instead (measured 2× cheaper) and the strided
`external_loop` path (once per CHUNK, not per element) keeps the storage route.

**NDIter's fixed cost, and where it went (2026-09-03).** A heap `NDIterState` is ONE native block:
the header plus a 1536-byte inline arena that `AllocateDimArrays` carves the dimension/operand arrays
from (`NDIterRef.AllocateStateBlock` / `ReleaseStateBlock`, `NDIter.State.cs`), recycled through a
bounded `[ThreadStatic]` 4-slot cache with `ResetForRecycle` re-zeroing the header and only the carved
bytes — every array the iterator hands out must start zeroed (`BaseOffsets` is `+=`-accumulated by
`FlipNegativeStrides`). The three calloc/free pairs it replaced were **78 ns of a 161 ns** three-operand
construction; construction now measures **62 ns (1-op) / 93 ns (3-op EXTERNAL_LOOP) / 113 ns (ufunc
flags)**, against `np.nditer`'s 258 / 698 / 923 ns. Arena overflow (≈ ndim × nop beyond ~8 × 8) falls
back to the separate blocks; stack states (`CreateCopyState`) have no arena. The production ufunc routes
identify their Tier-3B kernel by the packed `InnerLoopKernelKey` (a front cache over the string-keyed
`_innerLoopCache`; `ToCacheKey()` reproduces the legacy `npy_binop_{op}_{l}_{r}_{res}` string byte-exactly
on a miss) instead of interpolating that string per call (45 ns + 96 B of garbage, measured — more than the
iterator itself costs after the block cache). The unbuffered EXTERNAL_LOOP drivers
(`ForEach`/`ExecuteGenericMulti`/`ExecuteReducing`) call `ExternalLoopNext` directly, inlined with the
state's array pointers hoisted into locals, instead of through the cached advance delegate (per-chunk
driver overhead 5.6 → 4.7 ns; the remaining cost is the kernel delegate call plus the odometer — a narrow
inner loop is a KERNEL-selection matter, not iterator overhead: NumPy's `np.positive` on 4-wide rows runs
5.9 ns/row through its buffered transfer + one vectorized loop). The comparison `out=` route runs the
whole-array SIMD comparison kernel (`NDIterRef.TryExecuteComparison`) when the inputs share a dtype and
the bool out is contiguous in iteration order — it compiled a scalar-only body for every layout before
(`np.less(f64, f64, out=)` at 100K: 43.4 µs → 9.3 µs, NumPy 11.4). The three `*UfuncInto` routes skip
`Broadcast` + `ResolveUfuncIterationShape` when every operand has identical dims (identity transforms,
~55 ns + two Shape allocations), and `ResolveWritebacks` disposes the COPY_IF_OVERLAP temp eagerly and
reverts the operand slot to the original (it used to leak one pooled buffer per in-place ufunc call to
the finalizer). **Net at n=1 (ns, before → after | NumPy 2.4.2):** `add(out=)` 411 → 168 | 289,
`less(out=)` 366 → 140 | 300, `sqrt(out=)` 338 → 183 | 258, `negative(out=)` 329 → 153 | 267, in-place
`add(a, b, out=a)` 282 → 154 | 570; managed garbage per call 552 → 200 B. Gates:
`NDIterStateBlockTests`, `InnerLoopKernelKeyTests`, `NDIterComparisonOutRouteTests` + the `out_where`
fuzz tier. **Benchmark trap:** the nditer sheet's C# process must run with
`DOTNET_TC_CallCountingDelayMs=0` (the orchestrator now sets it) — tiered compilation's 100 ms
quiet-window rule kept the first rows of every section at tier-0 (`lessbool@1` 484 vs 124 ns for the
identical call), and the 2026-08-29 sheet's 1M/10M elementwise cells were host-contaminated on BOTH sides
(fresh: add@10M 0.85×, sqrt@10M 0.97×, not 0.33×/0.23×). **The discovery document** — cost model, the
measuring traps with their evidence, the reproducible probes (`benchmark/nditer/probes/`), the full
before/after tables and the ranked next levers (fancy-index → take/put kernels 2.5–3.5×, a 2-D kernel
contract for narrow rows, SIMD run scan for `where=`, the fresh-NDArray floor) — is
**`docs/NDITER_PERF_DISCOVERY.md`**.

**Narrow strided rows — the 2-D block kernel (2026-09-03, LANDED, then extended the same day).** The
per-chunk route drives a `(rows, w)` view with a CONTIGUOUS inner axis one row at a time under
EXTERNAL_LOOP, paying the odometer advance AND the per-chunk kernel's own SIMD-viability prologue on EVERY
row — so for a narrow `w` it was ~0.82× NumPy (`np.positive`), while a hand-written 2-D loop (prologue
once, inner SIMD run, per-row pointer bump) is ~1.7–2.1× NumPy — the whole gap is per-row overhead, not
memory traffic (buffering the strided rows into a contiguous window was measured SLOWER — the inner runs
are already SIMD-able in place). `ND2DElementwiseKernel`
(`Backends/Kernels/Direct/DirectILKernelGenerator.InnerLoop2D.cs`) loops the outer axis itself with
per-operand outer byte strides, called ONCE per 2-D block; it reuses the SAME scalar/vector emit bodies as
the per-chunk kernel, so results are byte-identical (elementwise ops carry no cross-element state).
Dispatched by `NDIterRef.{Is2DElementwiseShape,TryExecute2DElementwise}` (wired into the three packed-key
`ExecuteElementWise*` entry points, so it serves BOTH `out=` and allocating unary/binary routes), gated
on: unbuffered, no `where=` mask, EXTERNAL_LOOP, NDim ≥ 2, output inner axis element-contiguous, every
input inner axis contiguous OR broadcast (stride 1 or 0). **The review of the first report
(`docs/NDITER_2D_BLOCK_KERNEL.md` §11) found it had measured only w ≥ 4, f64, 2M and ops with a vector
body**, and that three of its own shapes LOST to NumPy: sub-vector rows (`sqrt(m[:, :3])` 0.41× — scalar
tail per row where NumPy finishes with a masked vector), inner-broadcast (`add(A, col)` 1.15× — the gate
demanded a contiguous inner for every operand), and ops without a vector body (`exp` on 4-wide rows
0.84×). It also found a GENERAL ITERATOR DEFECT: NDIter coalesced axes only when every operand was
contiguous and nothing was broadcast (its `CoalesceAxes` assumes the ascending pre-coalesce layout), so a
C-contiguous `(250000, 4)` array times a 0-d scalar iterated 2-D, one kernel call per 4-element row
(1.7 ms → 251 µs fixed) — NumPy's `npyiter_coalesce_axes` runs unconditionally. **All closed:**
`NDIterCoalescing.CoalesceAxesIterationOrder` merges adjacent axes every operand walks contiguously
(`stride[d] == stride[d+1]*shape[d+1]`, or a size-1 stride-0 axis) in iteration order on the
non-all-contiguous branch (subsumes `RemoveUnitAxes`; `a[:, ::2]` is ONE strided chunk exactly as NumPy
hands it out — the oracle's K1 excuse shrank to the `'A'`-order transposed-3-D case, and a 3-D
`x[:, :, :w]` now reaches the kernel as `(rows, w)` with no special flatten); the kernel contract carries
`innerByteStrides` and dispatches ONCE per call on the inner-mode pattern — unary {C, S}, binary
{CC, SC, CS} — each pattern a full 2-D loop with the S operand loaded and `Vector.Create`'d once per row;
32/64-bit lanes on a 256-bit AVX2 host finish each row's `innerCount % lanes` remainder (or the WHOLE row
when `innerCount < lanes`) with ONE `MaskLoad`→body→`MaskStore` (mask hoisted per call; integer
masked-off lanes filled with 1 so a software vector divide cannot throw; other hosts and 1/2-byte lanes
keep the scalar tail); a once-per-call row-shape dispatch picks generic / one-vector rows (`innerCount ==
lanes`: one full vector per row, no inner loop — the generic shape paid three compares per row for the
same work) / masked rows; a null vector body or non-SIMD dtype set gets a scalar-only 2-D block
(constant-stride addressing when every inner stride is its own element size, runtime strides otherwise);
and leading axes that do not fold into the block (`x[::2, :, :w]`) are walked by a per-BLOCK odometer in
the dispatcher. Rode along in the first generation: `positive` had no vector body — an identity vector
body was added (`EmitUnaryVectorOperation` Positive branch + `CanUseUnarySimd(Positive)`), so contiguous
positive is a SIMD copy (1.59× NumPy). **Measured (NPY/NS, pinned to one P-core, min of two rounds, old →
new):** `sqrt` w=2/3 **0.53/0.41 → 1.58/1.84** (100K) and 0.60/0.52 → 1.78/2.24 (2M); `add(A, col)` c=4
**1.15 → 6.5**, `multiply(view, 2.0)` c=4 **0.48 → 1.97**, `add(view, col)` c=4 **1.63 → 6.0**,
`view * 2.0` 1.10 → 4.8; `exp` w=4 0.84 → 1.22 (narrow now equals contiguous); `x[::2, :, :4]` positive
**0.70 → 2.12**, add 1.11 → 2.45; the w ≥ 4 cells the first report measured are unchanged or up
(2M positive w=4 1.91 → 2.33). Gates: a 16,831-check self-consistency probe (every mode × widths 1–100 ×
10 dtypes × reversed/strided-out/aliased/mixed-dtype/3-D/4-D/bool/Complex, bytes vs contiguous copies)
**0 mismatches**; FuzzMatrix **98/98** (K1 narrowed); main suite **14446/0**; `np.nditer` chunk lengths
and value streams identical to NumPy on the K1 layouts × C/F/A/K except the pre-existing `'A'`-order
case. **Measurement traps** (discovery doc §3 trap 11): on this hybrid host an UNPINNED benchmark thread
reads 2–3× slower uniformly and swings run to run (the first-generation absolute numbers did not
reproduce) — `narrow_probe.cs`/`numpy_twins.py` honour `NS_PROBE_AFFINITY`; 100K strided-stream cells move
±20 % with the same binary from placement alone. Still open: copy-class ops at w=2/3 gain only 1.1–1.3×
from the masked path (a half-vector store would add ~1.3× — a per-width specialization, not done);
buffered-cast routes are untouched; a strided-inner row that does not coalesce stays
on the per-chunk gather path.

**Three or more inputs with a broadcast input — the pattern stages (2026-10-03, np.clip's work).** The unary
{C, S} / binary {CC, SC, CS} dispatch above stopped at two inputs: a ternary kernel (np.clip's (x, lo, hi)) with
any stride-0 input ran the scalar 2-D block, the 1-D shell's all-contig requirement sent it to a gather with
zero-offset indices, and the where= block kernel to its scalar block. All three now dispatch on the inputs' C/S
pattern ONCE per call (`DirectILKernelGenerator.InnerLoop.BroadcastPatterns.cs`; up to 4 inputs, 15 patterns):
the all-C loop stays INLINE, every other pattern is a STAGE — a DynamicMethod with the kernel's own signature,
reached by one direct call and JIT-compiled on its FIRST call (probed: a DynamicMethod another one calls compiles
when the call first executes), because 2^N copies of a 4×-unrolled SIMD loop in one method would spend its inline
budget (the polynomial kernels' 3.3× trap). The gather route does the same for two or more inputs: a stride-0
input next to strided ones is read once and broadcast instead of gathered (`np.clip(x[::2], lo, hi)`,
`np.add(x[::2], 5.0, out=o)`; the caller's vector body is unchanged, so the add/multiply NaN-operand order holds).
Gates: `Backends/Kernels/BroadcastPatternStageTests.cs` (every pattern × 1-D / 2-D / where= / gather against the
materialized all-C route; a swapped pattern map fails 3 of its tests) + the `clip_ufunc` tier; 70,560 probe checks.
Who reaches the stages: np.clip's iterator route, the binary ufuncs' gather route (two inputs), and a caller's own
3–4-input kernel through `NDIterRef.ExecuteElementWise` / `ExecuteElementWiseTernary` / `ExecuteExpression`. NOT
np.evaluate: its fused kernels come from their own generator (`CompileFusedInnerLoop` / `EmitFusedSimdLoop`), so a
broadcast input of an np.evaluate tree does not run these stages (a JIT trace of `np.evaluate(a * col + c)` compiles
no stage; a 144,259-check probe found the fused results byte-identical to materialized inputs either way).

**Tier 1 of the discovery document's next-wins list, remeasured and closed (2026-09-03, second
pass; `docs/NDITER_PERF_DISCOVERY.md` §6.6, probe `benchmark/nditer/probes/fancy_where_probe.cs` +
`numpy_twins.py fancy_where`).** Three levers landed, one verified: (1) **the fancy-index operator
runs the take/put kernels** — `a[idx]`, `a[idx] = v`, `m[ridx]`, `m[ridx] = row` (ONE integer index
array over a C-contiguous source of any rank; the old route kept an offset delegate per element, a
materialised offset array and a second pass) go through `Selection/FancyIndexKernelRoute.cs`: a lean
FLAT gather/scatter (`DirectILKernelGenerator.GatherFlat.cs` — compile-time element width, running
cursors, NumPy's `check_and_adjust_index` as one add + one UNSIGNED compare) for primitive-width
slabs, the general take/put kernels (`cpblk`) for whole-sub-array slabs, and int32 indices read IN
PLACE (`idx32` kernel variants — `np.array(new int[]{…})` is int32, and `np.take`/`np.put` gained the
same in-place read). **NumSharp indexes with int64 everywhere; the int32 load width is kept on
measurement only** (discovery doc §6.6, `fancy_where_probe.cs` section D): against the int64-only
alternative — widen the index array to an int64 temp, run the int64 kernel — the in-place read is
1.7–2.5× at 1K (the temp's allocation floor), 1.1–1.8× at 100K (its convert + write), 1.2× at 10M
(its 12 bytes/index of extra traffic), while an int32 index costs the same as a native int64 one
(0.97–1.13×); all index ARITHMETIC stays int64 (the int32 is sign-extended on load — only the load
width and cursor step differ). Drop the variants if that ever stops being true. The setter pre-validates every index with a
`Vector<T>` bounds scan BEFORE its first store, exactly as NumPy's `mapiter_trivial_set` does, so a
bad index leaves the array untouched; the verbatim IndexError, the two value-broadcast ValueError
texts, last-write-wins on duplicates, view offsets on source AND index array (a contiguous slice
re-seats `Storage.Address` and keeps `Shape.offset` 0, a strided one keeps the base address and a
non-zero offset — `Storage.Address + Shape.offset × itemsize` is right for both) are pinned by
`Indexing.KernelRoute.Tests`. Pinned NPY/NS at
100K f64: `a[idx32]` 0.81 → **4.97**, `a[idx64]` 0.22 → **1.26** (NumPy's trivial path beaten),
`a[idx64] = v` 0.37 → **1.13**, `m[ridx] = row` 2.1 → **6.3** (10M: 0.33 → 1.27), `np.take(a, idx32)`
3.6 → **9.5**. (2) **The `where=` masked driver scans mask runs with SIMD** (`NDIter.Execution.cs`
`InvokeInner`): one movemask per 32-byte block, the runs inside walked by trailing-zero counts (a
per-boundary vector probe was tried first and regressed run-length-1 masks 0.78× — the per-block bit
walk keeps both regimes): all-false 27.7 → 1.5 µs, half 34 → 8.2, 64-runs 43 → 16.9, run-length-1
unchanged; every regime 1.1–3.9× NumPy. (3) **`where=` admitted into the 2-D block kernel**
(`DirectILKernelGenerator.InnerLoop2D.Masked.cs`, `NDIterRef.Is2DMaskedElementwiseShape`/
`TryExecute2DMasked`, wired into BOTH `ExecuteElementWise` overloads — the ufunc routes arrive with a
PACKED key, which the first wiring missed): the mask rides as the trailing operand; a byte-per-element
mask becomes each vector group's lane mask (pmovzx + compare) feeding the tails' masked
load/body/store, a `(rows,1)` mask gates rows, ops without a vector body get a masked scalar block;
the tail word is assembled from exactly `tail` bytes so the mask array is never over-read. Narrow-row
masked cells 0.4–0.9× → **1.3–10.5×** NumPy (w=4 `sqrt` 2492 → 505 µs), gated by an 8,358-check
self-consistency probe (0 mismatches) + `out_where.jsonl`. (4) **The coalescer's reach on copy/cast/
reduce verified** — pinned A/B of the `benchmark/layout` + `benchmark/cast` benches, pre-coalescer
commit vs `ccadeef4`: reduce 0.997×, elementwise 1.068× (strided 1.11×), cast 0.992× — no regression,
per-cell swings are the allocator-regime noise. **Two measurement traps found on the way** (discovery
doc §3, traps 12–13): `/` on an integer NDArray is TRUE division, so `(ar / 64 % 2) == 0` is a sparse
mask, not 64-runs — the first where= numbers compared different masks on the two sides (build masks
with `np.floor_divide`); and never A/B the live tree — a chain of `dotnet run` benches compiles the
tree as it is when each run starts, so editing during the chain hands later runs a half-edited tree
(both sides belong in detached worktrees). **The next session on this line starts at
`docs/NDITER_PERF_CONTINUATION.md`** — state, protocol, the measured remaining gaps (the fancy-index
shapes still on the delegate route: two index arrays 0.09×, slice + index array 0.05×, F-order
source rows 0.18×, strided index view 0.21×; the cast-`out` masked narrow rows 0.33×; few-row
leading-axis reductions 0.52×; the Tier 2 claims that did NOT reproduce pinned) and a ranked plan
with a design per lever (`benchmark/nditer/probes/neighbours_probe.cs` + `numpy_twins.py neighbours`
reproduce every number in it).

**Do NOT try to fix `it[0]` by re-seating a cached view** — measured, and it is a silent-wrong-answer
trap. `UnmanagedStorage` keeps **three** synchronized address caches: the public `byte* Address`, the
`IArraySlice InternalArray`, and a per-dtype `ArraySlice<T> _arrayXxx` field. Re-seating `Address`
alone makes `GetAtIndex`/`ToString`/casts correct while `np.sum(it[0])` and `it[0] * 2` silently
return **0**; re-seating `Address` + `InternalArray` fixes the reductions and still leaves the
arithmetic operators and comparisons stale. The re-seat is fast (0.68 ms boxed, ~16× NumPy, vs 59 ms
today) but is only safe if every cache is updated behind one `UnmanagedStorage` method — and any
future cache added there re-breaks it silently. That is the core-wide change, and this is why.
The NumSharp-idiomatic fast paths — the typed iterators below, `np.evaluate`, vectorized ops — avoid
the per-element view entirely instead.

**Gates:** unit tests from probed NumPy 2.4.2 output — `Indexing/np.ndindex.Test.cs` (22),
`Indexing/np.ndenumerate.Test.cs` (26), `APIs/np.nditer.Test.cs` (47), `APIs/np.nditer.NumPyParity.Test.cs` (24) and
`APIs/np.nditer.{Construction,Protocol,Strings}.Test.cs` (51 / 94 / 59, 2026-10-09) — and the oracle tiers that
materialize the protocols into comparable results: `iter.jsonl` (value streams, multi_index, c_index / f_index and
external-loop chunk lengths), `nditer_construct.jsonl` and `nditer_protocol.jsonl` (state traces as text). The unit
suites hold what a trace does not reach: the setters, copies, closed-state errors, buffered write-back and the
`no_broadcast` / `grow_inner` rules.

### Typed iteration — `np.nditer<T>` / `np.nditer_chunks<T>` (NumSharp extension)

The unboxed counterparts of `np.nditer`, over the same `NDIterRef` engine. NumPy has no equivalent
(Python has no unboxed generics), so there is no oracle tier — the gate is
`APIs/np.nditer.Typed.Test.cs` (23 tests), which pins them against the boxed `np.nditer` and
`ndenumerate` across every layout and all 15 dtypes.

```csharp
foreach (ref double x in np.nditer<double>(a, writeable: true)) x *= 2;              // element-wise
foreach (Span<double> c in np.nditer_chunks<double>(a, writeable: true))             // chunk-wise
    TensorPrimitives.Multiply(c, 2.0, c);
```

**Perf (NPY/NS, 100K float64, vs NumPy's `for x in np.nditer(a)` = 6.681 ms):** `np.nditer<T>`
**40×** (0.167 ms), `nditer_chunks<T>` **41×** (0.162 ms), chunks + `Vector<T>` **249×** (0.027 ms).
The raw-pointer floor is 0.047 ms, so the element walk costs ~1.6 ns/element over it — that residue is
`MoveNext`, and a leaner one-cursor variant measured no better. Layout cost is flat (0.16–0.19 ms for
contiguous / F / reversed / offset / broadcast).

**Five things this design had to get right — do not re-break:**
- **`GetEnumerator()` must NOT return `this`.** Unlike `np.Broadcast`, these own unmanaged state, so
  the first `foreach` frees what the second would walk — a use-after-free, caught by a test. The
  factory returns a **`readonly struct` enumerable holding no unmanaged state**, whose
  `GetEnumerator()` builds a FRESH `NDIterRef` (0.4 µs). So re-enumeration restarts — deliberately
  unlike the class-based `np.nditer`, which is its own iterator (NumPy's `iter(x) is x`) and resumes.
- **`foreach` disposes a `ref struct` enumerator through the same pattern**, no `IDisposable` needed;
  that is what frees the state. Hand-driven `MoveNext` loops must dispose themselves.
- **The walk is chunk-driven, not element-driven.** `EXTERNAL_LOOP` is always on and `MoveNext` walks
  the inner loop with pointer arithmetic, touching the iterator only at a chunk boundary — a
  contiguous array is ONE chunk, so per-element iterator cost is ~zero. The naive
  `Iternext()`-per-element form measured 0.47 ms, 3× worse.
- **Never buffered.** A `ref`/`Span` into a buffer dangles the moment the next fill lands; unbuffered
  pointers are absolute into the operand, which is also why the enumerator can advance the iterator
  *before* handing out the captured chunk.
- **Order is `'K'` (memory order), matching `np.nditer` exactly** — probed: `a[:, ::-1]` of
  `arange(6).reshape(2,3)` yields `0 1 2 3 4 5` in BOTH libraries, and `2 1 0 5 4 3` under `order='C'`.
  It is also what makes reversed/F/transposed views coalesce to one chunk. Pass `order: 'C'` for the
  logical order `ndenumerate` uses.

Two deliberate divergences, both pinned by tests: a **dtype mismatch throws** (a `ref` cannot convert,
so reinterpreting the bytes would hand out garbage — `np.nditer<double>` on an int32 array is an
error, not a cast), and an **empty array iterates zero times** where the boxed form demands NumPy's
`zerosize_ok` flag (forcing `if (a.size > 0)` around every `foreach` is not how C# collections
behave). `nditer_chunks<T>` additionally rejects a **non-unit-stride** inner loop up front —
`a[":, ::2"]` — because a `Span<T>` is contiguous by definition; the single-operand inner stride is
fixed for the whole iteration, so this is detected at `GetEnumerator` and never mid-loop.

**Four `NDIterRef` defects found while building this (all routed around, none fixed):**
`GetInnerLoopSizePtr()` **access-violates on a 0-d operand** (reads `Shape[-1]`; the internal
`ResolveInnerLoopCount` already guards it, the public accessor does not); `GetInnerStrideArray()`
returns **element** strides while the kernel contract (`ForEach`/`ExecuteGeneric`) takes **byte**
strides from the private `GetInnerLoopByteStrides()`, with no public byte-stride accessor;
`ResetToIterIndexRange` + `EXTERNAL_LOOP` **does not clamp the inner count to the range**, so it reads
past the array end — which blocks parallel range-splitting (NumPy requires `NPY_ITER_RANGED` at
construction for exactly this reason; splitting by array SLICE works today and measured **3.2×** on 8
workers); and NumPy's unbuffered `external_loop` coalesces `[:, ::2]` into **1** chunk where NumSharp
emits **500**. The first two are absorbed by `TypedIterHelpers.ReadInnerLoop`.

### .NET interop — `nd.Unsafe` BCL views + safe bridges (NumSharp extension)

Two ways to hand an `NDArray` to plain .NET code, both covering all 15 dtypes (generic `<T>` + a `typecode
== InfoOf<T>.NPTypeCode` guard). Gate: `APIs/NDArrayUnsafeInterop.Test.cs` (18, incl. a 15-dtype alias
sweep), green net8.0/net10.0.

**`nd.Unsafe` — zero-copy BCL views that ALIAS the unmanaged buffer and do NOT root it** (`Backends/
NDArray.Unsafe.Bcl.cs`, a second partial of the existing `NDArray._Unsafe` struct made `partial`; backed by
`Backends/Unmanaged/UnmanagedMemoryManager.cs`). The whole point — and the danger — is that none takes an
ARC reference on the memory holder: if the NDArray is disposed/GC'd while a view is live it dangles → keep
the NDArray side by side. Surface: `Span<T>()`/`ReadOnlySpan<T>()`, `Memory<T>()`/`ReadOnlyMemory<T>()`
(manager-backed — the `Memory` roots the tiny manager, never the NDArray), `Bytes()`/`ReadOnlyBytes()` (raw
`size*itemsize`, dtype-agnostic), `Pointer<T>()` (logical element 0, ANY layout), `TryGetSpan<T>`/
`TryGetMemory<T>` (non-throwing). **The method names ARE the return types** (`Span<T> Span<T>()`) — legal in
C#, and unavoidable since the pre-existing `AsSpan<T>()` returns NumSharp's `UnmanagedSpan<T>` (long-indexable,
the >2 GB path), not the BCL type. Span/Memory/Bytes require **C-contiguity** (an offset slice `a[2:5]`
qualifies — it is C-contiguous with `Storage.Address` re-seated; F/transposed/strided/broadcast **throw**
`InvalidOperationException`, TryGet* return false), the **exact dtype** (wrong `T` → `ArgumentException`), and
**≤ int.MaxValue** elements (Span/Memory are 32-bit-length; larger → `AsSpan<T>()`/`Pointer<T>()`). The
universal logical-start pointer is `(T*)Storage.Address + Shape.offset` (the documented rule: contiguous slice
re-seats Address & offset 0, strided keeps base + non-zero offset). Writes through Span/Memory/Pointer mutate
the array (verified). `Pointer<T>()` is raw — for a non-contiguous array it's just element 0 and the caller
walks via `nd.strides`.

**Safe bridges — COPY / enumerate into caller-owned storage (result outlives the NDArray)** (`APIs/
NDArrayNetInterop.cs`, extension methods). `nd.AsEnumerable<T>()` → `IEnumerable<T>` for LINQ: logical C-order
for EVERY layout (reads through strides via `GetAtIndex<T>`), unboxed elements, only the enumerator is a heap
object (dtype check fires eagerly, before deferred enumeration). `nd.ToArray<T>()` (pre-existing) is the
C-order copy. The typed iterators (`np.flat<T>`→C-order, `np.nditer<T>`→memory-order, `np.nditer_chunks<T>`)
get `.ToArray()` (materialize → `T[]`) and `.CopyTo(Span<T>)` (fill a caller buffer, returns count, throws if
short) — extensions on the `readonly struct` iterators (their Enumerators are the ref structs; the iterators
themselves are plain structs, so extension-by-value is fine). The boxed iteration objects
(`a.flatiter`/`np.ndenumerate`/`np.ndindex`) already implement `IEnumerable`, so LINQ works on them directly.
`AsEnumerable` is an extension (not a `np.*` static), so the oracle surface guard is unaffected. Docs:
`docs/website-src/docs/iterating-and-enumerating.md` → ".NET interop".

### Fused Expressions (NumSharp extension)
`evaluate` — `np.evaluate(expr[, operands][, out])` compiles an `NDExpr` tree to ONE NDIter pass: every elementwise node runs inside a single inner-loop kernel, so chained expressions allocate no intermediates and read each operand once (NumPy-ecosystem equivalent: `numexpr.evaluate`; measured 3.2–6.1× faster than NumPy 2.4.2 on 4M chains, 1.2–4× over NumSharp's own unfused chains — gate: the `benchmark/fusion` subsystem of `benchmark/run_benchmark.py`).

```csharp
NDArray r = np.evaluate((NDExpr)a * b + 2);                          // fused a*b+2
NDArray d = np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b)); // a,b dedup → 3 operand streams
NDArray s = np.evaluate(NDExpr.Sum((NDExpr)a * b), @out: x);        // one-pass sum(a*b), no temp
```

Semantics (all probed against NumPy 2.4.2, pinned in `NDEvaluateTests.cs`):
- **Dtypes follow NumPy result_type PER NODE** (`NDExpr.Typing.cs`): NEP50 strong-strong incl. the int/float tier crossing (`i4+f4→f8`); weak python-scalar literals (`i4+2→i4`, `f2+2.5→f2`, `bool+2→i64`, out-of-range → OverflowError); `true_divide` ints→f64; `arctan2` tier floats; `power`/`remainder`/`floor_divide` bool→i8; unary math tiers (`bool/i8→f16`, `i16→f32`, `i32+→f64`); comparisons→bool. `(i4*i4)+f8` wraps the multiply in int32 before promoting — bit-compatible with the unfused NumPy sequence.
- **Reductions are root-only**: `NDExpr.Sum/Prod/Min/Max/Mean(expr)` run a one-pass accumulating kernel over the inputs (NumPy reduce dtypes; f16/f32 sums accumulate in f64 and cast back (more precise than NumPy's pairwise); min/max NaN-propagate; empty: sum=0, prod=1, mean=NaN, min/max raise). **Plan P2 (`docs/plans/ndexpr-evaluate.md`) added the rest of the reduction surface, all bit-exact vs NumPy 2.4.2:** flat float32/float64/complex `Sum`/`Mean` + float `Prod` (M1) and their **axis** forms over a C-contiguous child (M2) DIVERT off the 4-accumulator fold — the host materializes the child through the elementwise engine, then reduces it with the SAME pairwise (`np.add.reduce`) / sequential (`np.multiply.reduce`) schedule NumPy runs (M3, `DefaultEngine.Evaluate.Stream.cs`: over contiguous inputs in one shared order the host now STREAMS that same schedule — the pairwise recursion driven host-side, each ≤8 KB leaf evaluated by the fused child kernel into L1 scratch and folded by the SAME cached fold — so no intermediate is allocated, bit-identical by construction and ~2× faster at 1M–4M; strided/broadcast/mixed-order inputs still materialize); `Any`/`All` (new bool ReduceKinds, logical OR/AND fold, `False`/`True` identities so an empty input is `False`/`True` not a throw) + `CountNonzero`/`NanSum`/`NanProd` (factory rewrites `Sum(x!=0)` / `Sum(Where(IsNaN(x),0,x))` / `Prod(Where(IsNaN(x),1,x))`) (M4a); and the ORDER-INDEPENDENT range / NaN-aware min-max kinds `Ptp`/`NanMin`/`NanMax` (M4b) — three new ReduceKinds the host DELEGATES (`EvaluateDelegatingReduce`: materialize the child once, then `np.ptp`/`np.nanmin`/`np.nanmax` over it), VALUE-exact by construction because a min/max/their difference is the same value in any order, so they carry NO ULP excuse and inherit NumPy's type-awareness (int child → plain min/max; `Ptp` preserves dtype so integer overflow WRAPS; `NanMin`/`NanMax` skip NaN, all-NaN→NaN; `Ptp` propagates NaN) — and `NanMin`/`NanMax` are BIT-exact too since 2026-09-23 (the ±0 sign and the all-NaN payload are schedule-dependent; `TryExactNanMinMaxEval` runs NumPy's fmax/fmin schedules over the buffer NumPy reduces — see "Math — NaN-Aware"), while `Ptp`'s bits over a permuted computed child stay unpinned; and the int64 INDEX kinds `ArgMax`/`ArgMin` (M4c-index) — two more ReduceKinds on the SAME `EvaluateDelegatingReduce` seam (materialize the child, then `np.argmax`/`np.argmin`, wrapping the flat scalar-`long` into a 0-d int64), returning **int64** (an index, so `ResolveReduceResultType`→`Int64` regardless of the child dtype), also bit-exact with NO ULP excuse: their index is not value-order-independent but depends on the C-order FIRST-tie / FIRST-NaN rule, which matches because the materialized child is exactly NumPy's own fresh C-contiguous buffer (verified over C/F/transposed/negstride/strided × ties × NaN); empty raises (no identity); and the SUMMATION kinds `NanMean`/`Var`/`Std` (M4c-summation) — three new ReduceKinds host-COMPUTED (`EvaluateStatReduce`, NOT delegated: the engine's own `np.nanmean`/`np.var` drift through its multi-accumulator flat sum) over the materialized child, reproducing NumPy's `nanmean` / `_var` op for op over the SAME pairwise (`PairwiseSumInto`) / axis (`ExactAxisSum`) sum M1/M2 use: `NanMean = pairwise_sum(NaN→0)/count_of_non_NaN`, `Var = pairwise_sum((x−mean)²)/max(N−ddof,0)` (`mean = pairwise_sum(x)/N`), `Std = sqrt(Var)`. `NanMean` follows Mean's dtype (complex128→complex128); `Var`/`Std` yield a **REAL float64** for a complex128 child (`\|x−mean\|²` computed as `re²+im²`, NOT an FMA — verified bit-exact vs `np.var(complex)` over 20K adversarial cases). `ddof` rides on `ReduceNode` (part of the program identity — hash + `StructureEquals` + string key). FLAT folds the child buffer in MEMORY order (bit-exact for a C- OR F-contiguous child); AXIS rides the engine's C-contiguous axis add.reduce. BIT-EXACT for **13 dtypes** with NO ULP excuse; **Half + Decimal REJECTED** (`NotSupportedException` — NumPy accumulates a float16 variance IN float16, needing a float16 pairwise kernel that dtype lacks, the "Half not diverted" gap M1/M2 share). and the weighted `Average` = `Σ(values·weights)/Σ(weights)` (M4c-average) — the FIRST reduction over TWO operand trees, so its OWN root node `WeightedAverageNode` (not a `ReduceNode`), host-computed (`EvaluateWeightedAverage`) over BOTH materialized children (`AvgValuesProgram`/`AvgWeightsProgram`) each cast to the ONE result dtype, reduced with the SAME pairwise sum (`ExactSumArray`) — so the FUSED `Average` is **BIT-EXACT with NumPy where the library `np.average`** (which sums through the drifting multi-accumulator engine fold) **is only allclose at large N**; result dtype keyed on the VALUES tree (int/bool → `result_type(values, weights, float64)`, else `result_type(values, weights)` — always float/complex, so an int·int product is float64 not a wrapping int64); zero total weight (empty input included) → `DivideByZeroException("Weights sum to zero, can't be normalized")` before the divide (NaN weight → NaN, not a raise); `NDExpr.Average(values, weights[, axis, keepdims])`. **`axis=None` + `keepdims`** (M5) is implemented for EVERY reduction kind: each flat factory has a `keepdims`-taking overload (a REQUIRED `bool`, so `Sum(x)` stays the 0-d form and `Sum(x, keepdims: true)` is the `(1,)*ndim` form), and the four host paths reshape their 0-d scalar result to `(1,)*childNdim` (a free reshape of the one element, so bit-exact with no new excuse; a 0-d child stays 0-d; direct `out=` validated for the keepdims shape). **The reduction KINDS + axis-forms surface is now complete** (M3 streaming landed for the Sum/Mean/Prod diverts; **tuple `axis` is DEFERRED** — every NumSharp reduction is single-axis (`int?`), a library-wide gap, not an evaluate one). Still folded / open: Half at any axis, complex `Prod` (npy_cmul FMA gap #12) (both E1-excused) — the strict-F-contiguous axis corner is CLOSED: `TryStreamAxisReduce` streams ALL-F operands as the C reduction of axis nd-1-k over the reversed dims (NumPy's memory-order schedule for the F child it materializes; 80/80 .npy-oracle cases bit-exact, E1 no longer exempts it), and the weighted `Average` STREAMS both sums (`TryStreamWeightedAverage` over `AvgNumeratorProgram`/`AvgDenominatorProgram`: no temporaries, 3.6–3.9× at 100K–4M, and NumPy-exact for F inputs where the old C-copy route was not); Half/Decimal RESULT dtypes are rejected across the M4c summation + average kinds (`NotSupportedException` — no float16/decimal pairwise kernel). Kinds: `Backends/Iterators/NDExpr.Evaluate.cs` (`NDExprReduceKind`, `WeightedAverageNode`, factories + the M5 `keepdims` overloads, typing); host: `Backends/Default/Math/DefaultEngine.Evaluate.cs` (`EvaluateReduce`/`EvaluateAxisReduce`/`EvaluateDelegatingReduce`/`EvaluateStatReduce`/`EvaluateWeightedAverage` + the shared `KeepdimsFlat`/`FlatReduceShape`/`ValidateFlatReduceOut`).
- Repeated NDArray references deduplicate to one iterator operand; `out=` follows the ufunc rules above; `ExecuteExpression` (Tier 3C) throws without `EXTERNAL_LOOP` (the ~40× per-element foot-gun) — `np.evaluate` configures the iterator itself.
- **Trivially iterable calls skip NDIter** (`DefaultEngine.Evaluate.Trivial.cs`, a port of NumPy's `try_trivial_single_output_loop`): identical dims on every non-0-d operand, each 1-D (any stride) or contiguous in ONE shared C/F order, no cast/`dtype=`/`where=`, and a provided `out` overlapping an input only EXACTLY → the fused kernel runs once, the same call NDIter makes after coalescing (byte-identical by construction). Everything else declines (never throws) to the unchanged NDIter pass. n = 8: prebuilt 527 → 255 ns, `out=` 332 → 75 ns (0 B); ~1.0× from 100K up. Hooks `NDExpr.DisableTrivialLoop`/`TrivialLoopRuns`; gate `NDEvaluateTrivialLoopTests`.
- **Flat `Any`/`All`/`CountNonzero` stream, not fold** (`DefaultEngine.TryStreamBoolFold`): the bool child is evaluated block by block into L1 scratch and folded with vectorized span scans (`IndexOfAnyExcept`/`IndexOf`/`Count`), Any/All stopping at the first deciding block — exact by construction (OR/AND/count are order-free). The factories' `x != 0` over a Boolean `x` is typed by NEP50 as an int64 compare (scalar kernel), so `NDExprProgram.NonzeroBoolOperandProgram` streams `x` itself through its SIMD kernel; a bare bool leaf is scanned IN PLACE (`Address + offset`, no copy). NPY/NS @100K: `any(z)` 4.2, `count_nonzero(mask)` 1.43, `count_nonzero(a>b)` ~0.9 (read-bandwidth ceiling), early-exit `any`/`all` orders of magnitude. **The AXIS forms stream too** (`DefaultEngine.TryStreamAxisBoolFold`, all-C or all-F operands — an F walk is the C walk of the reversed dims with an F-contiguous result): a contiguous reduced axis folds whole ROWS with the same span scans; otherwise each output's slab is one contiguous run, combined into an accumulator row a vector at a time (`nz = ~(v==0) & 1`; Any/All stop once every output of the current outer index is decided; the count runs in BYTE counters flushed into the int64 result every 255 slabs — the result row is CLEARED first, the allocation is uninitialized). Declines to the seeded fold for other kinds, empty axis/output, strided/broadcast/mixed order. NPY/NS: `any(a>b, axis=0)` 0.37 → **12.4** @100K, `count_nonzero(a>b, axis=0)` 4.8 / 2.1 @100K/4M, bool-leaf axis counts 10–53, F operands 3.5–5.2 @4M (the fold took 28 ms there) — no cell below NumPy.
- **Flat `Min`/`Max` are NumPy's own reduction schedule, bit for bit** (`DefaultEngine.Evaluate.MinMax.cs`, `NumPyMinMaxReduce`): a lane-exact port of `simd_reduce_c_{max,min}` (AVX2 build: `splat(x[0])`, 8-vector `maxn` groups, canonical +NaN horizontal step, SSE scalar tail; second operand wins a tie), so the SIGN of a ±0 result and a NaN's payload match NumPy — the old 4-accumulator fold matched only the value. Shaped as `Vector256<T>` on every host (the lanes ARE the contract), float lanes NumPy's `vmaxp`+`vcmpordp`+`vblendvp` under `Avx.IsSupported`. Routes: dense bare leaf in place (memory order), streamed computed child, materialized non-streamable float child (in the layout NumPy's own ufunc would allocate); a NON-dense bare leaf runs NumPy's ITERATOR schedule — the same flat walker as the engine's `np.max` (next paragraph under "Math — Reductions"); only a broadcast leaf keeps the fold. The AXIS forms (`Max(x, axis)`) are exact too: a bare leaf reduces in place through `ReduceAxis`, a computed child over all-C/all-F operands streams (`DefaultEngine.Evaluate.MinMaxAxis.cs`), anything else is materialized in NumPy's layout first. NPY/NS `max(a*b)` f64 @100K 0.87 → 1.55, i32 4.0; axis `max(a*b, 0/1)` 1.97 / 2.50, `min(i*i, 0)` 5.2. Corpus block C9 (signed zeros kept) gates it. **Trap it exposed:** `MisalignedRegistry`'s generic "(5) unary ~ULP" excuse (keyed on ONE operand) used to excuse any single-operand evaluate tree within 2 ULP — ±0 flips included — and hid 18 real divergences; it now skips `evaluate`. Fixed with it: flat reductions over a shared dense axis PERMUTATION (a transposed block) stream in MEMORY order (`CanStreamChild(allowPermuted)`/`IsSharedDensePermutation` — NumPy's K-order coalescing), and a flat `Mean` of an integer/bool child is NumPy's buffered-cast float64 sum (a pairwise sum per `np.getbufsize()` chunk; `np.setbufsize` changes the last bits in NumPy too).
- **`ArgMax`/`ArgMin` stream, over a fast engine axis argmax** (`DefaultEngine.Evaluate.ArgStream.cs`, `TryStreamArgReduce`, tried first in `EvaluateDelegatingReduce`): a BARE leaf goes straight to the engine argmax (every layout, no copy); a COMPUTED child over all-C operands (`CanStreamChild(allowF: false)` — only there is memory index k the logical C index NumPy's argmax visits) is produced block by block into L1 scratch — ROWS (reduced axis innermost / flat): the flat SIMD row kernel per row, a row longer than a block chunk by chunk with the chunk winners combined in increasing order by the same STRICT rule (a later tie never replaces; stop once decided); SLABS (outer axis): the engine's `SeedSlabChunk`/`FoldSlabRows` fed with the produced rows. F/strided/broadcast/mixed children, Half/Decimal/Complex, empty children and a bad axis decline to the materialize route (NumPy's errors kept). Exact by construction (the first-occurrence / first-NaN rule applied in logical C order). NPY/NS `argmax(a*b)` f64 flat/ax0/ax1 @4M 2.99 / 7.51 / 3.10, i32 ax0 9.04, i64 ax0 @100K 13.8 — no cell below NumPy (the old route had 16/48 under parity, down to 0.20×). Hooks `NDExpr.DisableStreamingReduce`/`StreamingReductions`; gates `NDEvaluateStreamingTests` `ArgReduce_*` (4) + `AxisArgFastPathTests` (8); 33/33 mutants red.
- **Compiled once per STRUCTURE, not per call** (`NDExprProgram` + `NDExprProgramCache`, `Backends/Iterators/NDExpr.{Program,Structure,Params}.cs`; plan `docs/plans/ndexpr-evaluate.md` §0.1): binding, the NEP50 typing pass, the vector plan and the kernel lookup run ONCE per (tree structure, input dtype signature, 0-d mask) and are cached twice — on the root instance (a hoisted / `Compile()`d tree: zero walks) and in a process-wide structural cache keyed by a 64-bit order-sensitive structural hash and VERIFIED node by node against the cached bound tree before use (a hash-only hit would run the wrong kernel; one entry per hash, 4096-entry cap, wholesale clear when full — the kernels stay in the kernel cache). So the natural spelling `np.evaluate((NDExpr)a * b + c)` REBUILT inside a loop costs **~0.46–0.51 µs / 0.86 KB per call at n=8** (old NDExpr 1.0 µs / 3.1 KB; the Phase-3-only build 1.58 µs; a prebuilt tree 0.41 µs / 0.59 KB; unfused `a*b+c` 0.35–0.37 µs / 0.94 KB; NumPy 0.52 µs), `out=` 0.36 µs / 0.46 KB, `Sum(a*b)` 0.39, `Where(a>b,a,b)` 0.45 — and fused beats the unfused chain on **13 of 15** probe rows at 1K with the rebuilt spelling (was 5/15; the two losses, `maximum(a,b)` 0.59× and `abs(a)` 0.63×, are single-op trees where the direct engine kernel's fixed cost is lower than an NDIter pass — plan P6.3). The library consumers (`np.sinc`, the windows, `trapezoid`, `gradient`) spell their trees inline and get this for free (the sinc tree at 1K: 4.9 → 3.4 µs, 1.7× the unfused chain). **A 0-d input is a hoisted PARAMETER, not a stride-0 operand** (`NDExpr.Params.cs`): its single element is copied into the kernel's aux block once per call and loaded into a local by the kernel prologue (the scalar, or its broadcast vector / lane mask in the SIMD body), so `a*b+k` with a 0-d `k` costs **0.39 µs prebuilt / 0.48 rebuilt** (was 0.68 / 1.25–1.30), ZERO per element (a stride-0 iterator operand cost 8–10 % per element on BOTH the SIMD and scalar paths — the fused shell's per-load broadcast branch and the strided fallback's stride multiply), and ONE kernel per structure: `np.hanning(M)` over 20 distinct M compiles 0 kernels (a literal `M-1` compiled 20). Typing is unchanged (a 0-d array is the STRONG scalar it always was — `int32 * 0-d int64 → int64`, where a C# literal is weak); the value is read BEFORE the pass (a parameter aliasing `out` keeps its original value, as NumPy's COPY_IF_OVERLAP gives it); every input 0-d → nothing is hoisted (0-d result, iterator path); the mask is part of the program identity and re-validated per call (`ndarray.resize` of a 0-d array re-resolves; a `Compile()`d handle refuses with a clear message); a dtype-pinned positional `Compile(types)` handle knows no shapes and streams everything. **Prefer a 0-d array over a literal for any scalar that varies at runtime** — a literal is baked into the IL and compiles one kernel per value (~0.2–1 ms JIT each, one kernel-cache entry each; the structural cache caps its own growth at 4096 and clears). **`expr.Compile()` / `expr.Compile(params NPTypeCode[] inputTypes)` → `CompiledExpression`** is the explicit handle (numexpr's `NumExpr(expr)`): eager JIT, `Evaluate(out)` for embedded arrays / `Evaluate(operands, out)` for positional trees, `IsPositional`/`IsReduction`/`InputTypes`/`ResultType`; a positional handle PINS its dtype signature (`TypeError` naming both signatures on a mismatch — never a silent recompile). Not the Tier-3C `Compile(inputTypes, outputType, cacheKey)`, which emits a raw kernel computed at ONE dtype. The inputs' broadcast is one N-ary single-allocation pass (`BroadcastInputDims`) whose mismatch lists EVERY operand shape like NumPy (`operands could not be broadcast together with shapes (2,3) (2,3) (4,) `, `IncorrectShapeException`); `out=` joins through `ResolveUfuncIterationShape` unchanged. **`Call` nodes:** the delegate/target SLOT is part of the kernel identity (two closures over one lambda body used to share a kernel, so the second read the first's captured state — fixed), and `DelegateSlots` dedups registration by delegate identity (target reference + method) / target reference, so a delegate held in a field and rebuilt into the tree per call keeps one slot and one kernel; a closure ALLOCATED per call is a new identity (new slot, new kernel JIT) — hold delegates in fields. Gates: `NDEvaluateProgramTests` (21), `NDEvaluateParamTests` (12), the `evaluate.jsonl` tier (its `pp_scalar_*` / `scalar_0d` layouts drive the parameter path bit-exact), `windows.jsonl`.

### Sorting & Searching
`argmax`, `argmin`, `argpartition`, `argsort`, `argwhere`, `flatnonzero`, `lexsort`, `nanargmax`, `nanargmin`, `nonzero`, `partition`, `searchsorted`, `sort`, `sort_complex`

**The index searches and index transforms return NumPy's objects** (2026-10-08, the open-bugs rundown's bug 8; plan
`docs/plans/api-parameter-parity.md` X37; gates the layout-parity oracle's argwhere / flatnonzero / `unravel` / `ravelmi`
families, the string oracle with no layout excuse for them, `Indexing/IndexSearchLayout.Test.cs`; probed against 2.4.2):
- **Views of one buffer** (they were new owning C arrays). `np.flatnonzero` is the column of nonzero's `(k, 1)` buffer;
  `np.argwhere` the TRANSPOSE of the `(ndim, k)` stack `np.asarray(np.nonzero(a))` makes — F-contiguous, byte strides
  `(8, 8k)` (a 0-d input: `argwhere(atleast_1d(a))[:, :0]`); `np.unravel_index` the columns of ONE interleaved
  `indices.shape + (len(dims),)` buffer (element stride `len(dims)`), laid out along the index array's iteration order —
  `NpyIter_CreateCompatibleStrides` (KEEPORDER, `DONT_NEGATE_STRIDES`) is `Shape.NpyIterOutputShape` over the index
  array: an F index array gives F-strided coordinates, a broadcast or reversed one C. All report OWNDATA false; a 0-d
  index array gives NumPy's scalars (fresh read-only 0-d arrays). Code that reads their raw buffers must use the strides.
- **unravel_index's statements:** the order letter (both cases), the dims overflow, the index dtype's TypeError, then
  `only 'C' or 'F' order is permitted`; an out-of-range index is NumPy's ValueError (was ArgumentOutOfRangeException),
  the FIRST one in the iteration order (the kernel runs over the indices laid out in that order); `shape=()` is legal.
  Positive dimensions take a Granlund–Montgomery reciprocal IL kernel (multiply-high + shifts: RyuJIT emits TWO hardware
  divides for `val % m` and `val / m`), 1.9–3.5× NumPy; a zero or negative dimension keeps the divmod kernel.
- **ravel_multi_index** is `arr_ravel_multi_index`: the result is its iterator's ALLOCATE output (KEEPORDER over the
  coordinate arrays: an F pair gives F, a mixed pair C; it was always C), the checks run in NumPy's order (order letter,
  modes, ravel strides, count, dtypes, broadcast with the iterator's text), `dims=()` gives 0, a zero dimension raises
  `cannot unravel if shape has zero entries (is empty).` in every mode, the errors are ValueError.
- **The order converter** is NumPy's `PyArray_OrderConverter` once (`OrderResolver.Convert`; `Resolve` raises its texts):
  `order must be one of 'C', 'F', 'A', or 'K' (got 'Q')` and a creation function's `only 'C' or 'F' order is permitted`
  are ValueErrors (they were ArgumentExceptions with drifted texts), and every letter is accepted in lowercase (np.nditer
  / np.nditer<T> refused 'c' / 'f'). `memoryview.tobytes` keeps CPython's rule (upper case only, None is C).
- `np.diag_indices` / `np.tril_indices` / `np.triu_indices` (and `_from`) return OWNING arrays, as NumPy's fresh ones are
  (they were aliases of their own buffers).
- Merge note: journey4's array-object oracle carves these as VF18 and VF35; after a merge their cases match NumPy and
  belong in the gate.

**Axis `argmax`/`argmin` fold from the STRIDES** (`Backends/Default/Math/Reduction/Default.Reduction.ArgAxis.Fast.cs`,
`TryExecuteAxisArgFast`, hooked into `ExecuteAxisArgReduction` — one hook serves both): the old per-output IL kernel
recomputed each output's base offset with a div/mod chain and walked the axis scalar-strided (axis 0 of a C block =
one element per cache line per step; a contiguous axis never used the flat SIMD row kernels). Now the non-axis dims
coalesce and the walk is picked from the strides: **Zero** (broadcast axis — all 0, nothing read), **Rows** (axis
contiguous — the flat SIMD row kernel per output, inline under 32), **Slab** (a non-axis run at least as tight — ≤1024
lanes folded together row by row, running best in L1 scratch, a `Vector256` compare per row for a ±1 run (−1 flipped),
an OVERLAPPING final vector instead of a scalar tail, the index written only for improved lanes), **AxisWalk** (the
axis is the tightest non-contiguous stride). NumPy's rule in LOGICAL axis order: first occurrence (−0 == +0), the
first NaN wins argmax AND argmin, integers strict, bool first True/False (0 when none; bool slabs stop once every lane
decided). Bool/ints/Char(as u16)/f32/f64; Half/Decimal/Complex and rank > 64 decline (hooks
`DefaultEngine.DisableFastAxisArg`/`FastAxisArgRuns`, thread-static). **The 64-bit flat row kernel is NumPy's
single-pass tournament** (`ArgMaxInt64Tournament`/`ArgMinInt64Tournament` in `DirectILKernelGenerator.Reduction.Arg.cs`,
port of `simd_argmax_{s64,u64}`): the generic two-pass ran `Vector256.Max<long>/<ulong>`, which has NO AVX2
instruction — 2.0–2.6× at 1K–100K, 1.0–2.2× at 4M; a 32-bit tournament measured SLOWER in cache (0.30–0.85×), so
8/16/32-bit lanes keep the two-pass; routing gated on `Vector256.IsHardwareAccelerated`. NPY/NS engine `np.argmax(a,
axis)`: ax0 f64 4.4 / 8.6–8.8 @100K/4M, f32/i32 ax0 @4M 16.4, i64 10.5–11.8, bool 95–106, F ax1 13.5–20, strided
2.1–3.5 (1.0–137× the legacy kernel); Rows over a memory-bound 4M buffer stays ~parity (the same per-row kernel NumPy
runs). Pre-existing, not fixed: `ReduceArgMax` wraps an out-of-range NEGATIVE axis (`while (axis < 0) axis += ndim`)
instead of raising NumPy's AxisError. Gate `Backends/Kernels/AxisArgFastPathTests.cs` (8).

**The issue-#623 six** — `partition`/`argpartition`/`lexsort`/`nanargmax`/`nanargmin`/`sort_complex` (all probed
against 2.4.2; gates `Sorting/np.{partition,lexsort,nanargmax,sort_complex}.Test.cs` (108) + ~2,170 differential
cases in the `sort`/`nanreduce` fuzz tiers) — are compositions/wrappers over the kernels that already shipped:
introselect (`Utilities/QuickSelect.cs`, the median/percentile primitive), the stable radix argsort
(`AxisSort`), and the `ArgMax`/`ArgMin` engine reductions. No third sort/select core was added.
- `partition(a, kth, axis=-1, kind='introselect', order=null)` + in-place `ndarray.partition` — the kth element(s)
  land in final sorted position, sides unordered. `Backends/Default/Sorting/AxisPartition.cs` drives one introselect
  line kernel per 1-D lane through **`AxisSort.DriveAllButAxis`** (the same NDIter IterAllButAxis loop the sorts
  use). Validation is NumPy's probed ORDER with verbatim texts: kind (`select kind must be 'introselect' (got '…')`)
  → order (`Cannot specify order when the array has no fields.`) → axis → writeable (`partition array is read-only`)
  → kth (`kth(=N) out of bounds (M)` reporting the POST-wrap value; bounds SKIPPED for size-0 arrays; the kth list
  is sorted so partitions compose, and multi-kth segments confine each introselect). Floats NaN-compact (NaN sorts
  last, ORIGINAL bit patterns preserved in the tail — unlike np.sort's canonicalizing radix); Complex uses NumPy's
  CDOUBLE_LT comparator, whose NaN order is positional ((1,NaN) BEFORE (NaN,1)) and deliberately NOT compacted.
  C# reads an empty `int[]` kth as NumPy's `np.array([], intp)` — a valid no-op copy (Python's bare `[]` is float64
  → TypeError; a typed array carries no such ambiguity). Result is a fresh C-contiguous copy (house np.sort
  convention; NumPy's `copy(order='K')` keeps F-order for F-inputs — values identical). **The NDArray-kth overload**
  (NumPy's array form, on np.partition/np.argpartition AND the ndarray methods) is what makes NumPy's kth-dtype
  rejections REACHABLE — bool kth `Booleans unacceptable as partition index`, non-integer (incl. Decimal)
  `Partition index must be integer` (TypeError), >1-D `object too deep for desired array` — with NumPy's probed
  STAGING: kind → order → too-deep (BEFORE axis: `partition(a, [[1]], axis=9)` reports too-deep) → axis →
  bool/integer dtype → wrap/bounds. Integer kth values cast to intp MODULARLY like NumPy (uint64 2^63 wraps
  negative and its bounds error reports the wrapped value; 2^64−1 wraps to a LEGAL −1); Char rides the integer
  route (house call); 0-d and any layout accepted.
- `argpartition` — same driver returning int64 indices; the source is only read (broadcast views legal). Built on
  the **index-tracking QuickSelect overloads** (`PartitionAtMany(T*, long*, …)` — values and their original indices
  swap in lockstep, the `RadixSort.ArgSortU32/U64` pattern). A 0-d input ravels to `(1,)` FIRST — NumPy's arg-side
  `PyArray_CheckAxis` quirk (axis/kth errors then report dimension/size 1, result `[0]`) while `np.partition(0-d)`
  raises AxisError; float NaN indices tail in encounter order (argsort's exact policy). **The fuzz tier pins the
  DERIVED kth-values** (`take(partition(a,ks),ks)` / the take_along_axis gather at ks): the arrangement BETWEEN kth
  anchors is introselect-implementation-specific on BOTH sides, so whole-output bytes are not contractual — the
  two-sided ≤/≥ invariant is unit-test-pinned instead.
- `lexsort(keys, axis=-1)` — indirect stable sort, LAST key primary. Pure composition reproducing `PyArray_LexSort`
  pass for pass: successive **stable** argsorts first→last key, each re-sorting the running permutation via
  `take_along_axis` — bit-identical indices to NumPy's mergesort passes because NumSharp's radix argsort is stable,
  ties included. Verbatim errors: `need sequence of keys with len > 0 in lexsort` (TypeError), `all keys need to be
  the same shape` (ValueError); 0-d keys let axis 0/-1 slip (NumPy's back-compat quirk) and size ≤ 1 early-returns
  a 0-filled int64 array. The single-NDArray overload reads the first axis as the key list (a 1-D input degenerates
  to N scalar keys → the 0-d `0`, NumPy's probed quirk).
- `nanargmax`/`nanargmin(a, axis=null, out=null, keepdims=false)` — NumPy's `_replace_nan(∓inf)` + all-NaN-slice guard
  (`ValueError("All-NaN slice encountered")`) then `argmax`/`argmin`; int/bool/char/decimal pass through untouched
  (NumPy's `mask=None` branch). The flat form returns `long`; the `int?`-axis overload returns the int64 NDArray and
  handles `keepdims` with `axis=null` (shape `(1,)*ndim`). **`out=` follows NumPy's probed argmax contract:** any
  bool/int dtype safe-castable to intp is accepted (bool through uint32 — uint64/floats raise the verbatim
  `Cannot cast array data from dtype('…') to dtype('int64') according to the rule 'safe'` TypeError), the shape must
  EQUAL the keepdims-adjusted result (`output array does not match result of np.argmax.` — nanargmin leaks
  "argmin"), the int64 indices cast UNSAFELY into it (an int8 out wraps index 200 → −56, a bool out reads
  index ≠ 0 — probed), the SAME instance returns, a strided out writes through its base, and the all-NaN guard
  fires BEFORE out is touched. Positional slot 3 is `out` (NumPy's order) — pass `keepdims:` by name; the implicit
  bool→NDArray conversion would otherwise bind a positional bool to `out`. Axis validation runs UP FRONT for every
  dtype path — the engine's argmax silently wraps out-of-range negative axes (`np.argmax(a2d, -3)` → `[1 1]`, a
  pre-existing argmax bug this family refuses to inherit) — with NumPy's rule that a 0-d array accepts axis 0/-1.
  The documented NaN+Inf untrusted-tie caveats fall out of the composition identically (probed cases pinned).
  **Auditing this family's dtype cells exposed and fixed three ENGINE argmax/argmin bugs** (all pre-existing, all
  now oracle-gated): flat `argmax(Decimal)` rode the IL arg kernel whose Bgt/Blt cannot compare a 16-byte struct
  and SILENTLY mis-answered (`argmax([3,9,1,5])` → 0) — Decimal now takes the same `ExecuteReducing` fallback
  family as Half/Complex (`DecimalArg{Max,Min}Kernel`); flat `argmax(Char)` threw NotSupportedException while the
  axis path supported char all along — it now rides the zero-copy uint16 reinterpret its amax already uses; and
  the strided/scalar IL step compared floats with `Bgt_Un`/`Blt_Un`, whose UNORDERED branch also fires when the
  ACCUM holds NaN, so every later finite value stole the index (`argmax` of a transposed `[nan,2,3,nan]` walked to
  3 instead of NumPy's first-NaN-wins 0; contiguous inputs were immune via the C# NaN helpers) — fixed to ordered
  `Bgt`/`Blt`. The reduce fuzz tier now generates the flat `axis=None` argmax/argmin cells whose absence hid all
  three (the old skip excluded flat argmax entirely; keepdims=False flat cells replay via
  `NDArray.Scalar(np.argmax(a))`, and the decimal oracle gained flat argmax cases).
- `sort_complex(a)` — copy + sort along the LAST axis **in the input's own dtype** (an int64 past 2^53 orders by
  its exact value), then up-cast. NumSharp's single `Complex` absorbs NumPy's complex64 cells (int8/16, uint8/16 —
  values identical, width documented; those four dtypes are excluded from the fuzz tier and unit-test-pinned).
  0-d leaks `ndarray.sort()`'s own `axis -1 is out of bounds for array of dimension 0`, exactly as NumPy leaks it.
  See `Sorting_Searching_Counting/np.{partition,argpartition,lexsort,nanargmax,sort_complex}.cs`.

### Linear Algebra
`diag`, `diagflat`, `diag_indices`, `diag_indices_from`, `diagonal`, `dot`, `einsum` (contracts via the matrix products / OpenBLAS), `einsum_path` (greedy/optimal contraction planner, byte-exact info string), `fill_diagonal`, `inner`, `mask_indices`, `matmul`, `matvec`, `outer`, `tensordot`, `trace`, `tril`, `tril_indices`, `tril_indices_from`, `triu`, `triu_indices`, `triu_indices_from`, `vdot`, `vecdot`, `vecmat`

`np.linalg.*`: `cholesky`, `cond`, `cross`, `det`, `diagonal`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `inv`, `lstsq`, `matmul`, `matrix_norm`, `matrix_power`, `matrix_rank`, `matrix_transpose`, `multi_dot`, `norm`, `outer`, `pinv`, `qr`, `slogdet`, `solve`, `svd`, `svdvals`, `tensordot`, `tensorinv`, `tensorsolve`, `trace`, `vecdot`, `vector_norm` (+ `LinAlgError`)

### The BLAS/LAPACK API surface — fully implemented via OpenBLAS (NSE only for factorisations with no backend)

Everything NumPy computes through the OpenBLAS binary the interop package bundles now has its public
API, its NumPy-exact validation, its engine seam AND — with `NumSharp.Interop.OpenBLAS` referenced —
its numerics. Whether the numerics exist WITHOUT a backend splits by what a portable managed algorithm
costs — a property of the problem, not a decision: the products and the LU family have managed kernels
(a GEMM and an unblocked LU are each a few hundred lines), the harder factorisations do not:

| | Members | Behaviour (installed backend vs none) |
|---|---|---|
| **Products** (CBLAS) | `inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`, `linalg.multi_dot`, `matrix_power` (n ≥ 0), the `linalg` Array-API forms, `linalg.norm` except matrix ord ∈ {2, -2, 'nuc'} | **They compute.** Managed kernels; the `IBlasBackend` invariant holds — a backend changes WHICH implementation runs, never WHETHER an answer exists. **With `NumSharp.Interop.OpenBLAS` referenced**, `dot`/`matmul`/`inner`/`vdot`/`vecdot`/`matvec`/`vecmat` route through cblas for **float32/float64/complex128**, byte-identical to NumPy (complex via `zgemm`/`zgemv`/`zsyrk`/`zdotu`/`zdotc`; `tensordot`/`multi_dot`/`matrix_power` inherit it by composing over `dot`/`matmul`) — the 5 product gufuncs override `IBlasBackend.Try{Inner,Vdot,Vecdot,Matvec,Vecmat}`. |
| **LU family** (LAPACK LU) | `det`, `slogdet`, `solve`, `inv`, and `tensorinv`/`tensorsolve`/`matrix_power` (n < 0) that compose on them | **They compute — managed fallback (`Backends/Default/LinearAlgebra/ManagedLu.cs`).** Without a backend, a from-scratch right-looking UNBLOCKED LU (`getf2`) + triangular substitution (`getrs`) runs: `allclose` to NumPy (bit-exact for tiny matrices, a few ULP apart as they grow — Core ships no BLOCKED LU to reproduce LAPACK's exact `gemm` accumulation, the same reason the products need the bundled binary for byte-parity), det via the `sign·exp(Σ log|Uᵢᵢ|)` fold (so `det([[5.]])` is `4.999999999999999`, not `5.0`), a singular operand (exact-0 pivot) → `LinAlgError("Singular matrix")` for solve/inv and `(0,-inf)`/det 0 for det/slogdet. The generic `<T,TOps>` core monomorphises for double/Complex (Single upcasts to double, NumPy's lite rule). The double hot loops are hand-written `Vector256<double>`+FMA kernels the same shape as `SimdMatMul`'s micro-kernel — the O(m³) rank-1 elimination is a REGISTER-BLOCKED GER (4 trailing rows/pass, pivot-row vector reused, 4 independent `Fma.MultiplyAddNegated` chains), the substitution's row updates 4×-unrolled FMA axpys — and past `BlockedThreshold`=256 the double factorisation takes a BLOCKED path (`FactorDoubleBlocked`: panel getf2 + laswp + TRSM + Schur GEMM) whose Schur update rides the cache-tiled `SimdMatMul.MatMulDouble`; complex stays scalar. **Faster than NumPy for small matrices** (n≤16: 2.2–4.7× det/inv/solve — dispatch-overhead bound, so SIMD on the tiny loops is a wash), ~parity at n≈64, and **below NumPy for large** (n≥256 ≈ 0.4–0.6×) — capped by the same managed-GEMM-vs-OpenBLAS ceiling as the products (SimdMatMul's managed dgemm is slower than the OpenBLAS dgemm NumPy's getrf calls), which the blocked path narrows but cannot close. **With `NumSharp.Interop.OpenBLAS` referenced** they route through LAPACK `gesv`/`getrf` instead, byte-identical to NumPy. **One documented `[Misaligned]` divergence:** a matrix whose zero pivot only surfaces after several eliminations (textbook `[[1,2,3],[4,5,6],[7,8,9]]`) lands on ~1e-16 in the unblocked kernel where blocked LAPACK cancels to exact 0 — so det is ~0 (allclose) but inv/solve return the large-but-finite inverse instead of raising. |
| **Other factorisations** (LAPACK) | `cholesky`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `pinv`, `lstsq`, `qr`, `svd`, `svdvals`, `matrix_rank`, `cond`, `norm` matrix ord ∈ {2, -2, 'nuc'} | **No backend: `NotSupportedException` from `TensorEngine`** — Core ships no managed QR/SVD/eigensolver, so there is nothing to fall back to; the message names the NumPy API, the LAPACK routine and the `IBlasBackend.Try*` member. **Except an EMPTY operand** (an empty matrix or a zero-size batch, since 2026-10-08, plan `api-parameter-parity.md` X34): NumPy's answer needs no arithmetic — shapes, zeros, the identity a full SVD (`U`/`Vh`) or a complete QR (`Q` of an M×0 matrix) carries, `lstsq`'s sequential sum of squares of `b` for an M×0 matrix — so every factorisation gives it in Core, BEFORE the engine asks a backend (`Backends/Default/LinearAlgebra/LinAlgEmpty.cs`; gate the portable `linalg_empty.jsonl` tier, 5,075 cases, zero excuses, strict error types). `lstsq` with a non-empty `a` and an empty `b` still needs the backend (gelsd computes `a`'s singular values). **With `NumSharp.Interop.OpenBLAS` referenced**, these compute through LAPACK for **float32/float64/complex128** (`potrf`/`geev`/`syevd`/`heevd`/`gesdd`/`geqrf`/`orgqr`/`gelsd`), byte-identical to NumPy — float32 upcast to double and cast back exactly as NumPy's `_commonType` forces (complex via the `z` routines; a float32 operand's COMPLEX `eig`/`eigvals` result is NumPy's complex64 — every component rounded to float32 — carried in complex128, #569); the factorisation `Try*` are overridden in `OpenBlasBackend`. |

**The seam is `TensorEngine.Blas` — one property, not two.** `IBlasBackend.LinearAlgebra.cs` adds 15
`Try*` members, ALL of them **default interface implementations returning false**, which is what let
the file exist without touching `OpenBlasBackend` at first — it now **overrides all 15** (5 products +
10 factorisations), while the interface defaults stay false so a third-party or real-only CBLAS backend
still compiles. (Default interface members are
interface-dispatched and invisible on the implementing class — pinned by a test that types the
variable as `IBlasBackend` precisely because it must.) LAPACK belongs on this property because it
lives inside the very binary already bundled: scipy-openblas ships `gesv`/`getrf`/`potrf`/`geev`/
`syevd`/`heevd`/`gesdd`/`geqrf`/`orgqr`/`gelsd` next to the BLAS symbols. `TensorEngine`'s members
are `virtual`, and each reads `Blas` into a **local** before calling it (GEMM_PARITY §9 — a
test-then-call on a settable property was ~2 % NREs under a concurrent `Disable()`).

**Validation runs before the throw**, so the error contract is gated the same whether or not a backend
is installed — `_assert_stacked_2d`/`_assert_stacked_square`/`_commonType` are ported verbatim
(`LinAlgError("{n}-dimensional array given. Array must be at least two-dimensional")`,
`"Last 2 dimensions of the array must be square"`, `TypeError("array type float16 is unsupported in
linalg")` — and `Decimal`/`Char`, which have no NumPy dtype, take float16's exit rather than
inventing a new error).

**Six NumPy behaviours worth not re-deriving** (all probed against 2.4.2; the whole 132-case matrix
was replayed through both libraries — 121 bit-exact, 11 expected NSE, 0 unexplained differences):

- **`np.inner` reports `b`'s shape ALREADY TRANSPOSED.** It is `matrixproduct(a, swapaxes(b,-1,-2))`,
  so `np.inner(ones((2,3)), ones((3,2)))` raises "shapes (2,3) and **(2,3)** not aligned".
- **`np.vdot`'s length mismatch is a RESHAPE error, and its own length check is dead code.**
  `array_vdot` flattens both operands through ONE reused `PyArray_Dims` buffer holding `-1`;
  `_fix_unknown_dimension` writes the resolved length back into it, so the second flatten asks for
  `(a.size,)`. Hence `vdot(ones(3), ones(5))` → "cannot reshape array of size 5 into shape (3,)", and
  `"vectors have different lengths"` in the C source is unreachable.
- **`vecdot` reduces in the LOOP dtype, not NEP50's accumulator** — its loops are `'ii->i'`, so an
  int32 pair stays int32 where `np.sum` would give int64. `vecdot`/`vecmat`/`vdot` conjugate their
  FIRST operand; `inner`/`matvec` do not.
- **`axes=` names the core axes per operand, and is the general form of `axis=`.** One entry per
  input AND output — `np.vecdot(m, v, axes: [[-1],[-1],[]])`. The output entry may be omitted ONLY
  where the output has no core axes, which in this family is `vecdot` alone; supplying both `axes`
  and `axis` raises `TypeError("cannot specify both 'axis' and 'axes'")`. For `matvec`/`vecmat` the
  output entry says where the result's core axis LANDS, so
  `np.matvec(a, b, axes: [[-2,-1],[-1],[0]])` transposes the answer.
- **`axis=`/`keepdims=` compute only on `vecdot` — but exist on all three.** `matvec`/`vecmat` have
  two DISTINCT core dims and NumPy raises `TypeError` for both, so they are offered rather than
  omitted: a ported call then fails the same way with the same message instead of failing to
  compile. The keepdims text names input 1's core rank, which differs per signature (1 for
  `matvec`, 2 for `vecmat`).
- **`keepdims=True` COMPOSES with `axes=` on `vecdot`.** Under keepdims the output gains the reduced
  core dimension (kept as size 1), so a PROVIDED output entry must be length 1 (`[[1],[1],[1]]`) and
  says WHERE the kept axis lands in the output — but it may still be OMITTED (`[[1],[1]]`), in which
  case the kept axis goes to the END. An explicit empty output entry (`[[1],[1],[]]`) then
  CONTRADICTS the kept dimension and is the AxisError "operand 2 has 1 core dimensions, but 0
  dimensions are specified"; the mirror-image `[[1],[1],[1]]` WITHOUT keepdims is "has 0 … but 1
  specified". The output axis is validated against the output rank (`axis N is out of bounds for
  array of dimension M`). The whole interaction routes through one `NormalizeAxes` override for the
  effective output core rank — the earlier code rejected every `axes`+`keepdims` call outright.
- **`np.linalg.matrix_rank` below rank 2 is a PREDICATE, not a count.** NumPy short-circuits with
  `int(not all(A == 0))`, so `matrix_rank([1,2,3])` is **1**, not 3, and an empty operand is 0
  (`all([])` is true). Easy to get wrong, because a count returns a plausible number for every
  non-degenerate input.
- **`tensordot` has THREE distinct rejections, reproduced in NumPy's order.** A repeated RAW axis
  value is `ValueError("duplicate axes are not allowed in tensordot")`, checked first, ahead of any
  shape or range test (so `([5,5],[0,1])` is a duplicate, not an index error). An out-of-range
  contraction axis is `IndexError("tuple index out of range")` — NumPy indexes the shape TUPLE with
  the raw axis (`as_[axes_a[k]]`), so `tensordot(a2d, b2d, 3)` (the int count exceeds ndim) and
  `([5,0],[0,1])` are index errors, NOT shape mismatches. Mismatched extents and unequal list lengths
  are `ValueError("shape-mismatch for sum")`; because the extent comparison short-circuits the loop,
  a mismatch found earlier HIDES a later out-of-range axis (`([0,1],[0,5])` settles as the shape
  mismatch). A negative count contracts NOTHING (`range(-axes,0)` is empty for `axes ≤ 0`), so
  `axes=-1` ≡ `axes=0` ≡ the outer product.
- **`np.linalg.multi_dot` requires every non-endpoint operand to be 2-D.** NumPy promotes a 1-D
  first/last operand (row/column) and THEN runs `_assert_2d` over all of them, so a 0-D, a 3-D+, or
  a 1-D operand in a MIDDLE position raises `LinAlgError("{ndim}-dimensional array given. Array must
  be two-dimensional")`. The guard is load-bearing, not cosmetic: without it the Cormen cost model
  reads a phantom `dimensions[1]` off a non-matrix and the chain either leaks a raw bounds error or —
  worse — returns a silently wrong-shaped product (a 3-D middle used to give `(2,4,2)` where NumPy
  refuses). `out=` is threaded into the FINAL dot, so it takes the TWO-dimensional product shape even
  for a vector endpoint (`multi_dot([v, B, C], out)` wants `(1,k)` and returns a raveled `(k,)` view,
  not `(k,)` in and out). Exactly two operands is a plain `np.dot` and skips the 2-D check entirely.
- **The `np.linalg` Array-API forms are NOT aliases.** `linalg.trace`/`linalg.diagonal` reduce the
  **last** two axes where `np.trace`/`np.diagonal` take the **first** (a `(2,3,3)` stack gives `(2,)`
  vs `(3,)`); `linalg.outer` demands 1-D where `np.outer` flattens anything; `linalg.cross` demands
  3-vectors. Each is implemented against its own contract.

**Two fixes rode along.** `NDArray.matrix_power(-1)` used to `throw new Exception("matrix_power just
work with int >= 0")` — never NumPy's rule (`a**-n` is `inv(a)**n`); it now takes that route, plus
binary exponentiation, a `LinAlgError` for non-square input, and an identity in the operand's own
dtype rather than always float64. And `np.dot`/`np.matmul`'s "not aligned" messages rendered shapes
as `(2, 3)` where NumPy writes `(2,3)` — now `Shape.ToPythonTuple()` (`ToString()` is left alone; it
renders for humans everywhere else).

**Adjacent bug, fixed 2026-10-07 (strings validation):** `np.stack` ran its operands through
`atleast_1d` first, so stacking three 0-d arrays gave `(3,1)` where NumPy gives `(3,)`; it now expands
each operand as given (gate `Creation/np.stack.Test.cs`). `linalg.cross` still builds its result with
`expand_dims`+`concatenate`, which gives the same answer.

### `np.einsum` — subscript language + contraction, the contraction routed through the matrix products

The parser is a port of `PyArray_EinsteinSum`'s parsing block plus `parse_operand_subscripts` and
`parse_output_subscripts` (`numpy/_core/src/multiarray/einsum.cpp`) — `EinsumSubscripts.cs`. **Every
expression NumPy accepts parses AND CONTRACTS, and every expression NumPy rejects is rejected with
NumPy's own text.** Both spellings work: the subscripts string and NumPy's sublist form
`np.einsum(a, [0,1], b, [1,2], [0,2])`.

**The contraction goes through OpenBLAS "like other functions", by construction.** It is a port of
NumPy's OWN `optimize=` path — `bmm_einsum` + `_parse_eq_to_batch_matmul` in `einsumfunc.py` — which
reduces every pairwise contraction to a `matmul` (plus a single-operand einsum that takes diagonals,
sums, and transposes the terms first), or to a broadcast `multiply` when nothing is contracted. So
the products land on `TensorEngine.Matmul`, which already dispatches on `Blas`: with
`NumSharp.Interop.OpenBLAS` referenced the float32/float64/complex128 contractions are **byte-identical
to NumPy's `einsum(..., optimize=True)`** — the route that also goes through tensordot/BLAS. NumPy's
DEFAULT `einsum` (`optimize=False`) is its own C `sum_of_products` loop and is NOT BLAS-identical even
inside NumPy (`np.einsum('ij,jk->ik', a, b)` ≠ `a @ b` bitwise there; probed 2.4.2, 0.70.0 release QA),
so a float comparison against a plain NumPy `einsum` call is a ≤3-ULP comparison, not a byte one.
Without the backend they fall to the managed GEMM (exactly as `np.matmul` itself does). There
is deliberately **no `TryEinsum` seam** — einsum reaches a backend the indirect way `np.tensordot` and
`np.linalg.multi_dot` do, through the product. The whole composition is a pure `np`-layer function —
`np.EinsumContract` (`LinearAlgebra/np.einsum.Contract.cs`), sitting beside `np.tensordot` and
`np.linalg.multi_dot` rather than on the engine (it uses no engine state): N operands fold
left-to-right, each step through `PairwiseContract`.

**Value parity, with a probed split.** Integer and boolean contractions are byte-exact everywhere
(modular / logical reduction is order-independent); every product-shaped case (matmul, batched
matmul, matvec, `ij,kj->ik`, tensor contractions, inner/outer/kron) reduces to `np.matmul`/
`np.multiply` bit-for-bit — verified: `einsum(...)` equals the direct product IN-PROCESS across
f64/c128 — so it inherits their parity. A pure FLOAT summation done OUTSIDE a product (`ij->i`, a
diagonal-then-sum, or the accumulation across three or more float operands) can differ in the last
ULP because its order follows `np.sum` and the left-to-right fold rather than NumPy's einsum
iterator; those stay `allclose`. All 15 dtypes contract (Half/Decimal/Char via the scalar matmul
paths, complex UNCONJUGATED — einsum never conjugates, unlike `vdot`). einsum preserves the operand
dtype like NumPy (`einsum('ij->i', int32)` is int32, NOT `np.sum`'s widened int64); two-operand
promotion is `result_type`. Differential-verified against NumPy 2.4.2 across ~490 cases: 307
value/dtype/edge (all dtypes × matmul/transpose/diagonal/trace/reductions/outer/inner/batched/
ellipsis/multi-diagonal/broadcast-contraction/3-operand/scalar) + a 186-case MEMORY-LAYOUT matrix
(F/transposed/strided/negative-stride/offset/broadcast on either operand × matmul/reduce/diagonal/
batched/ellipsis/permuted-output + empties, ALL bit-exact on integer-valued data).

**The single-operand VIEW path is NumPy's, keyword quirks included** (all probed on 2.4.2). One
operand + no `out=` + nothing summed → a VIEW of the operand: `einsum('ii->i', a)` is **writeable**
(`…[:] = 1` sets `a`'s diagonal — `np.diagonal`'s own contract stays read-only; einsum restores the
flag by re-aliasing from the OPERAND's storage, because `Storage.Alias()` inherits read-onlyness
from the storage it aliases — aliasing from the diagonal view's storage silently re-clears the bit),
a broadcast operand's view stays read-only, and `einsum('ij->ij', a)` is a DISTINCT view object,
never `a` itself (returning `a` would let `result.resize()` mutate the operand where NumPy's view
refuses). On this path NumPy **ignores `order=` entirely** (`order='C'`/`'F'` still return the same
non-contiguous diagonal view) and **discards a `dtype=` request** (`einsum('ii->i', int32,
dtype=int64)` is the int32 view) — both reproduced.

**`casting=` gates EVERY cast, with NumPy's iterator texts verbatim.** Each operand must reach the
loop dtype under the rule — `Iterator operand 0 dtype could not be cast from dtype('int64') to
dtype('float64') according to the rule 'no'` — which is what makes `casting='no'` reject mixed
dtypes; and the result must reach `out=`'s dtype — `Iterator requested dtype could not be cast from
dtype('float64') to dtype('int32'), the operand 1 dtype, according to the rule 'safe'` (out is
operand `nop` in NumPy's numbering). A same-rank `out=` EXTENT mismatch is a second NpyIter-leak
divergence (like the label-extent one): NumPy leaks the "remapped shapes" iterator text, NumSharp
words it about the contraction (`einsum() output parameter has shape (…) but the contraction
produces (…)`).

**`order=` resolves the computed result's layout** — and `'A'` AND `'K'` both come back F-contiguous
when EVERY input is F-contiguous (probed: an all-F matmul, hadamard or 3-op chain is F even at the
default `'K'`), C otherwise, where `'K'` leaves the already-C computed result untouched. The view
path above is exempt.

**The grammar half of the parse is cached** by `(subscripts, operand ranks)` — labels depend on
nothing else — in a 4096-entry `ConcurrentDictionary` (NumPy caches its own equation parse the same
way, `lru_cache(2**12)`); the extent/diagonal validation always runs per call against the shared
IMMUTABLE grammar instance (thread-hammered: 20K concurrent mixed valid/invalid calls, 0 errors).

**Perf (NPY/NS vs NumPy's DEFAULT einsum — `optimize=False`, the C iterator; Release, best-of-7).**
With `NumSharp.Interop.OpenBLAS` referenced: matmul 1024² f64 **23×**, 256² **5.6×**, 3-op chain
`ij,jk,kl->il` 64² **146×** (NumPy's default 3-op einsum is the naive 4-label iterator; even its
`optimize=True` path is ~0.9× vs ours), batched 128×16² **2.0×**, matvec 512² **1.9×**, hadamard 1M
**3–4×**, complex128 64² **2.9×**, outer ~**1.1×**. Without a backend the product cells fall to the
managed GEMM exactly as `np.matmul` does (1024² still **2.9×**, 256² **2.3×**, but matvec **0.21×**
— the managed product kernels' own documented floor: `np.dot(m,v)`/`np.matvec` measure the same
187µs themselves). Sub-1× cells, all inherited or fixed-cost: sub-5µs calls (4×4 matmul ~0.5×,
trace/diagonal/transpose views 0.5–0.9× — the ~1–2µs composition floor vs c_einsum's single C call;
vs NumPy's own `optimize=True` these same cells are **1.9–5.9×** in NumSharp's favor), big `ij->i`
~0.8× (the shared axis-sum's parity — einsum adds nothing on top), and int32 64² ~0.5× (the managed
int GEMM vs c_einsum's int SSE loop — our einsum equals our `np.matmul(i32)` there, and NumPy's own
`np.matmul(i32)` is SLOWER than its einsum). einsum-specific overhead is ~2µs/call, under 1 % by
64².

**NumPy has TWO einsum parsers** — the C one behind the default `optimize=False` and a Python one
(`einsumfunc.py`) behind the `optimize` path — and they word the same rejection differently.
NumSharp reproduces the C one whatever `optimize` says, since that is what a default call hits.

**The label encoding is NumPy's**, one `sbyte` per operand dimension: positive = the label's ASCII
code at first occurrence, **negative = the offset back to that first occurrence** (a repeated label,
i.e. a diagonal), `0` = a broadcast dimension from an ellipsis. So `"abbcbc"` over 6 dims is
`[97,98,-1,99,-3,-2]`.

**Six things that are easy to get wrong, all probed:**
- **The two operand-count messages read BACKWARDS.** NumPy walks operand by operand checking the
  delimiter after each chunk, so ONE operand against TWO terms raises *"more operands provided to
  einstein sum function than specified in the subscripts string"*. Reproduced verbatim anyway.
- **An impossible diagonal has TWO messages.** `nop == 1 && out == null &&` nothing summed takes
  NumPy's return-a-view attempt, which says *"dimensions in **single operand** for collapsing index
  …"*; anything else says *"dimensions in **operand {i}** for …"*. So `einsum("ii->i", a)` and
  `einsum("ii", a)` fail differently on the same array.
- **Every diagonal is validated BEFORE any cross-operand extent.** NumPy collapses all operands'
  diagonals first, so `einsum("ij,jj->ij", …)` reports operand 1's diagonal even though operand 0
  already fixed a conflicting `j`. Doing it in one pass reverses the two errors.
- **The implicit output is ASCII-ordered**, so an upper-case label sorts BEFORE a lower-case one:
  `einsum("zA", a)` transposes a `(2,3)` and `einsum("Az", a)` does not.
- **The sublist alphabet is `'ABC…Zabc…z'` — UPPER case first** (NumPy 2.4.2's `einsum_symbols`).
  Index order therefore equals ASCII order, which is what makes an inferred output come back in the
  caller's numbering. The intuitive guess (`a-z` then `A-Z`) is SILENTLY wrong, not an error: it only
  shows on an expression mixing the halves, where `einsum(a, [0,26])` on `(2,3)` returns `(3,2)`.
- **Spaces are legal anywhere** and skipped; a `>` without its `-` is scanned as a label and rejected
  as an invalid subscript, while a lone `-` gets the missing-arrow message.

**One deliberate divergence** (`[Misaligned]`): when a label's extents disagree between operands,
NumPy's C path leaks `NpyIter`'s *"operands could not be broadcast together with remapped shapes
[original->remapped]: (2,3)->(2,newaxis,3) (4,2)->(2,4)"* — iterator axis bookkeeping rather than
anything about the contraction. NumSharp raises the wording NumPy's OWN `einsumfunc.py` parser uses
for the identical condition: `Size of label 'j' for operand 1 (3) does not match previous terms (4).`
(whose two sizes also read swapped against the sentence — the first is the size already recorded).

**Pass the keywords BY NAME** — `np.einsum("ij->i", ops, @out: dst)`. They are keyword-only in NumPy,
and naming them is also what keeps them unambiguous here: NumSharp converts scalars to `NDArray`
implicitly, so a fully positional 7-argument call matches both the `params` overload and the full one
and the compiler rejects it.

**`np.einsum_path` — the contraction planner — is IMPLEMENTED** (`LinearAlgebra/{np.einsum_path,
EinsumPathPlanner,EinsumPath}.cs`), a route-for-route port of NumPy 2.4.2's `einsumfunc.py`: its
`_parse_einsum_input` parser, the `_greedy_path`/`_optimal_path` searches with the `_flop_count` cost
model, and the `einsum_path` info-string builder. `np.einsum_path(subscripts, operands, optimize)`
returns `(EinsumPath path, string repr)` — NumPy's `(['einsum_path', (1,2), (0,1)], "…")` tuple, where
`EinsumPath` is a `readonly struct` (Deconstruct + indexer + `Steps`/`Count` + `ToList()` for NumPy's
raw list + NumPy-format `ToString`). **The path and every numeric metric are byte-identical to NumPy**
(differential-fuzzed 1,525 random contraction networks: path + full printed string). `optimize=` takes
NumPy's whole `path_type` vocabulary — `false`/`null` (no-opt), `true`/`"greedy"`/`"optimal"`, a
precomputed `EinsumPath` (explicit path), and a `("greedy"|"optimal", memory)` tuple — in NumPy's exact
decision order, with the verbatim error taxonomy (`ValueError` for the parse/shape rejections;
`KeyError` — a new `Exceptions/KeyError.cs` — for an unknown path name, reproducing NumPy's buggy
`('Path name %s not found', '…')` leak; `RuntimeError` for an incomplete explicit path; `TypeError`
"Did not understand the path: …" / "object of type '…' has no len()" for a bad path object). The
returned `EinsumPath` ROUND-TRIPS into `np.einsum(…, optimize: path)` (also the raw list and the memory
tuple) — accepted and validated, though einsum STILL contracts pairwise left-to-right (documented
above), so the path changes nothing numerically, only which plan is displayed. **The cost model is
`BigInteger`, not `long`** — element counts and FLOP counts are computed with arbitrary precision
because NumPy uses Python bignum (`_compute_size_by_dict`/`_flop_count`). Individual DIMENSION extents
stay `long` (a single array dimension always fits int64, as it does in NumPy's `intp` shapes); only the
PRODUCTS need bignum. A contraction whose sizes overflow `long` — e.g. every dim 100000 in
`ea,fb,abcd,gc,hd->efgh`, whose naive FLOP count is 5×10⁴⁰ and largest intermediate 10²⁰ — used to wrap
to a NEGATIVE count, print garbage, and derail the greedy/optimal cost comparisons (giving up to the
single naive contraction); it now matches NumPy 2.4.2 byte-for-byte. **Two things done in integer
arithmetic on purpose:** the `.3e` FLOP/size formatting rounds HALF-TO-EVEN like Python (.NET's
`"0.000e+00"` rounds half-AWAY — 13825 → 1.383e+04 vs NumPy's 1.382e+04), so `Sci()` formats the
`BigInteger` directly; and the ellipsis PLACEHOLDER letters are the ONE inherent divergence — NumPy draws
them from a hash-randomized set (`I`/`f`/`P` across three processes), so NumSharp draws them
deterministically (the path and every number still match, only the `...`-expanded display letters
differ, which NumPy itself does not pin). The contraction-step POSITIONS in `EinsumPath` stay `int[][]`
(operand indices, capped tiny — NumPy returns the same small ints). Deliberately NOT ported:
`np.einsum_path`'s `einsum_call=True` hidden option (NumPy-internal; NumSharp's einsum has its own
left-to-right driver). Gate: `LinearAlgebra/EinsumPathTests.cs` (26 — incl. past-int32/past-int64/
past-int64-intermediate size pins) + `EinsumSubscriptParityTests.cs` (subscript grammar) +
`EinsumContractionTests.cs` (the values, against NumPy 2.4.2).

**Signature parity is its own gate.** `LinAlgSignatureParityTests` reflects over the whole surface
and asserts each function has an overload whose parameter NAMES appear in NumPy's ORDER, plus the
18 defaults NumPy states concretely — because renaming `UPLO` to `uplo` or swapping `keepdims` and
`ord` compiles, passes every value test, and silently breaks every named-argument call ported from
Python. It also asserts no `np.linalg` member is missing. Two defaults C# cannot spell (`object`
parameters may only default to null) are carried by a **null sentinel** and pinned behaviourally:
`matrix_norm`'s `ord='fro'` and `vector_norm`'s `ord=2`. Four NumPy ufunc keywords are deliberately
ABSENT from the gufuncs — `casting`, `order`, `subok`, `signature` — because NumSharp models none of
them anywhere in its ufunc surface (`signature` is what `dtype` already does; `subok` concerns
ndarray subclasses the library does not have), and accepting-then-ignoring would be worse.

Gate: `LinearAlgebra/{BlasProductApiTests, LinAlgErrorParityTests, LinAlgEngineSeamTests,
LinAlgSignatureParityTests, EinsumSubscriptParityTests}.cs` (92 tests), backed by a 451-case NumPy
differential matrix replayed through both libraries (dtype sweep × 10, memory-layout sweep × 8, the
`axes=` paths, einsum's whole subscript grammar, every rejection): 435 exact, 5 expected NSE, 11
documented divergences with verbatim messages, **0 unexplained**. `LinAlgEngineSeamTests` now pins the
**no-backend contract** (Core alone ships no LAPACK, so every factorisation still raises
`NotSupportedException` there); the numerics have all landed in `NumSharp.Interop.OpenBLAS` and are
gated by that package's live-parity tests **AND the host-pinned `linalg_parity` differential-fuzz tier**
(`test/NumSharp.Tests.Oracle/Fuzz/corpus/linalg_parity.jsonl`, 366 cases — the eigen/SVD/QR/Cholesky family
`cholesky`/`eig`/`eigvals`/`eigh`/`eigvalsh`/`svd`/`svdvals`/`pinv`/`matrix_rank`/`cond{None,2,-2}`/`lstsq`/`qr`
+ `norm{2,-2,nuc}`, and the **LU family** `solve`/`inv`/`det`/`slogdet`/`tensorinv`/`tensorsolve` +
`matrix_power(n<0)` (added 2026-08-21 — the OpenBLAS members that had ZERO committed oracle coverage),
byte-exact vs NumPy 2.4.2 on NumPy's own pinned scipy-openblas at threads=1, matched by CONTENT sha256 and
**Inconclusive off the pinned host** — the `matmul_parity` model. The LU family is `getrf`/`gesv`, so
deterministic pivoting makes EVERY output byte-reproducible with nothing excluded (`slogdet` is a tuple,
`det` 0-D/1-D, complex `slogdet` sign is unit-modulus complex, singular→`(0,-inf)`); only the eigen/SVD side
excludes complex-Hermitian `eigh` eigenvectors, float32-`eig`-with-complex-eigenvalues and non-SVD
`cond`/`norm` orders as not byte-reproducible — see `Fuzz/README.md` → "LAPACK factorisation family"). The 92
API/error-contract tests above remain unit tests — they assert signatures and verbatim messages, not values.

**Two known type-only divergences, messages verbatim on both sides:** the reshape/alignment errors
raise `IncorrectShapeException` where NumPy raises `ValueError` (long-standing house convention,
see the reshape section), and `AxisError` derives from `ArgumentOutOfRangeException` with
`paramName: "axis"`, so .NET appends `(Parameter 'axis')` to `.Message` — which is why the whole
library asserts axis texts with wildcards.

**The diagonal / triangular family** (NumPy's `_twodim_base_impl.py` + `_index_tricks_impl.py`; all probed against 2.4.2, tests in `Indexing/np.{diag,tri,diag_indices}.Test.cs`, oracle tiers in `manip`/`matmul`):

- `diag(v, k=0)` — **the two branches differ in kind**: a 1-D input *constructs* a fresh writeable C-contiguous `(n, n)` matrix (`n = v.size + |k|`) with `v` on the k-th diagonal; a 2-D input *extracts* via `diagonal(v, k)` and so returns a **read-only view sharing storage** (probed: `shares_memory` True, `writeable` False — the docstring's "a copy" is wrong). Anything else raises `ValueError("Input must be 1- or 2-d.")`. The construction path allocates the zero matrix and scatters `v` straight down the stride-`(n+1)` diagonal through the IL strided-store kernel (`DirectILKernelGenerator.DiagWrite` — a byte-width-keyed direct MOV loop, all 15 dtypes via itemsize ∈ {1,2,4,8,16}), NumPy's `res.flat[i::n+1] = v` with no element loop and without routing a strided destination through `NDIter.Copy`; it falls back to the old aliased-view + `SetData` route when dynamic codegen is unavailable (AOT). At 3162² this is **1.5× faster than the `SetData` route** and beats NumPy (~1.2×); at n≤316 it wins on small-n but is **allocator-bound at 100K** — a fresh lazily-committed `n×n` buffer faults ~n scattered pages on the sparse diagonal write, where NumPy reuses a warm L3-cached block (NumPy's own `diag` is 129 µs cache-defeated vs NumSharp's 78 µs fresh, so NumSharp's *fresh* alloc already beats NumPy's *fresh*; NumPy's benchmark edge is block caching NumSharp's allocator does not do).
- `diagflat(v, k=0)` — as `diag`'s 1-D branch but accepts **any** rank: the input is raveled in **logical C order** first (so an F-contiguous or strided view is read row-major, not memory-order), 0-d → `(1,1)`.
- `tri(N, M=None, k=0, dtype=float)` — ones at and below the k-th diagonal. NumPy computes `greater_equal.outer(arange(N), arange(-k, M-k)).astype(dtype)` (two full N×M passes + a bool temp); NumSharp blits a prefix of one pre-built ones row into each row, dtype-agnostically. Negative `N`/`M` clamp to an empty axis (NumPy reaches them through `arange`).
- `tril(m, k=0)` / `triu(m, k=0)` — always a **fresh writeable C-contiguous copy**, dtype preserved, applied to the last two axes. Two NumPy spellings leak through and are reproduced: a **1-D input yields a 2-D result** (`m.shape[-2:]` of `(n,)` is `(n,)`, so the mask is `tri(n)` → `(n,n)` and the row broadcasts down it — `tril([1,2,3])` → `[[1,0,0],[1,2,0],[1,2,3]]`), and a **0-d input raises** NumPy's leaked `tri()` missing-argument error, verbatim: `tri() missing 1 required positional argument: 'N'`. Because the retained columns of each row form one contiguous run, the mask is skipped entirely: the result is filled row-by-row (blit the kept run, zero the rest). Sources whose last axis is not unit-stride fall back to the literal `where(tri(...), m, 0)` composition.
- `fill_diagonal(a, val, wrap=False)` — **in place, returns void**. Values **tile cyclically and truncate**, they do not broadcast (`a.flat[:end:step] = val` semantics — probed: `[1,2,3,4]` into a 6×6 gives `[1,2,3,4,1,2]`, an over-long list is cut, and an **empty** one is a silent no-op). `wrap` only affects tall matrices. Errors verbatim: `array must be at least 2-d`, `All dimensions of input must be of equal length` (ndim > 2, non-hyper-cubic), `underlying array is read-only`. Implemented with **no element loop** — the targets resolve to real strides and are written by the IL strided-store kernel (`DirectILKernelGenerator.DiagWrite`), one stride-`Σstrides` run (or, when wrapping, `ceil(count/cols)` equally-strided blocks); a stride-0 source broadcasts the scalar, a contiguous/tiled source supplies the sequence, and both feed the same kernel with no NDIter setup. This replaces the old aliased-view + `SetData` route (which sent the strided destination through `NDIter.Copy`) and falls back to it when dynamic codegen is unavailable (AOT). The kernel is why it writes correctly through transposed, sliced, F-order and negative-stride views (verified for `wrap` × non-contiguous too, the one combination the oracle tier does not reach). `val` is `object` and accepts everything NumPy's array_like does — scalars, `NDArray`, C# arrays (incl. rectangular 2-D, which ravel), and `IEnumerable` collections. Only genuine scalars take the `NDArray.Scalar` fast path (it converts via `IConvertible`); arrays, collections, `Half` and `Complex` route through `asanyarray`. A **null** `val` throws `ArgumentNullException` — a deliberate divergence from NumPy, where `None` reaches the dtype converter and yields nan for a float array (TypeError for an integral one); in C# a null is a caller bug and silently writing nan would hide it.
- `diag_indices(n, ndim=2)` / `diag_indices_from(arr)` / `tril_indices(n, k=0, m=None)` / `triu_indices` / `tril_indices_from(arr, k=0)` / `triu_indices_from` / `mask_indices(n, mask_func, k=0)` — return `NDArray<long>[]` (the house tuple convention; `intp` is int64). `diag_indices` returns **the same array instance** in every slot, matching NumPy's `(idx,) * ndim` aliasing (a write through one is visible through all); `ndim <= 0` gives an empty array. `mask_indices`' arity follows the mask function's rank, so passing `diag` yields **one** index array. The `tril/triu_indices` generators skip NumPy's N×M bool mask + two broadcast coordinate grids + boolean gathers entirely: the triangle is separable, so the row/column runs are emitted directly in `O(output)` — **4.4× NumPy at 10M**.

**Perf (NPY/NS, best-of-rounds, Release):** geomean **1.83× at N=10M** (no cell below 1.0; min 1.13×) and **1.40× at N=100K**. `tril_indices`/`triu_indices` lead at 4.40×/4.55× (10M). `fill_diagonal` now **wins at every tier** through the `DiagWrite` IL kernel — **4.8× at 100K, 1.46× at 10M** (the 10M cell was 0.40× on the old `SetData` route, which paid `NDIter.Copy`'s per-strided-write overhead: 37.9 µs → 10.9 µs); it touches only `min(rows, cols)` elements, so a direct byte-width MOV loop is all the work there is. `diag`/`diagflat` **construct** win at small/large n but sit at **0.12–0.33× at 100K**, allocator-bound on the fresh-sparse-diagonal page faults documented above — a property of NumSharp's non-caching allocator, not the diag kernel (NumSharp's fresh alloc already matches NumPy's fresh; only NumPy's warm block cache pulls ahead in the tight benchmark loop). `diag_indices`/`diag_indices_from` are `arange`+wrap and win **2–3.6× / 9–14×** (a reported 0.54× on `diag_indices` was a contaminated snapshot — it is comfortably >1× on an idle host). The triangular fills pick their allocation strategy by size: below **64 MiB** the buffer comes back dirty so they allocate uninitialised and write every byte exactly once, above it `np.zeros` rides free OS zero-pages and only the retained run is touched (measured cliff — 64 MB: 6.0 ms vs 11.0 ms write-once; 72 MB: 18.9 ms vs 13.5 ms the other way; see `np.tri.cs` → `WriteOnceMaxBytes`).

### Window functions
`bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`

The five NumPy window generators (`numpy/lib/_function_base_impl.py`; all probed against 2.4.2). Each
takes a scalar point count `M` (`kaiser` also a shape parameter `beta`) and returns a **1-D float64**
taper — NumPy forces at-least-float64 via `np.array([0.0, M])`. **`M` is `double`, matching NumPy's own
stub type `_FloatLike_co`** (NOT int): an int caller (`np.hanning(5)`) binds via the implicit int→double
conversion and gets the identical result, while a **non-integer `M` reproduces NumPy's exact float-M
behavior** — the length is `len(arange(1-M, M, 2))` (a fractional count, so `np.hanning(5.7)` is 6
samples) and kaiser can carry a trailing NaN once `|(n-α)/α| > 1`. Output dtype is always float64.
Shared edge cases for the **cosine windows** (bartlett/blackman/hamming/hanning): **`M < 1` → empty
`(0,)`**, **`M == 1` → `ones(1,)`**. **kaiser has NO `M < 1` guard** (its one structural difference,
ported verbatim): `arange(0, M)` is empty for `M ≤ 0` (→ empty) but a **single element for
`0 < M < 1`** (→ a computed one-sample window, e.g. `np.kaiser(0.5, 5)` = `[0.0367…]`), while `M == 1`
is special-cased only to dodge the `alpha == 0` division. Formulae, verbatim from NumPy with `n = arange(1-M, M, 2)`
(kaiser uses `arange(0, M)`): bartlett `where(n≤0, 1+n/(M-1), 1-n/(M-1))`; hanning `0.5+0.5·cos(πn/(M-1))`;
hamming `0.54+0.46·cos(πn/(M-1))`; blackman `0.42+0.5·cos(πn/(M-1))+0.08·cos(2πn/(M-1))`; kaiser
`i0(β·√(1-((n-α)/α)²))/i0(β)` with `α=(M-1)/2` and `i0` the cephes Chebyshev modified-Bessel routine
NumPy's `np.i0` uses (internal `np.BesselI0` scalar helper, the `_i0_1`/`_i0_2`/`_chbevl` port; **not**
exposed as a public `np.i0` — kaiser is its only consumer).

**BIT-IDENTICAL to NumPy 2.4.2** (verified across 152 `(M, beta)` cases + the differential-fuzz `windows`
tier): the elementwise transform is built as an **`np.evaluate` fused expression in NumPy's exact
operation order** (so `(π·n)/(M-1)` rounds the same two steps NumPy's ufunc chain does), and float64
`add`/`sub`/`mul`/`div` are IEEE-exact while float64 `cos` is the same scalar `Math.Cos` the win-amd64
`ucrtbase` CRT and NumPy both call (kaiser's `i0` likewise rides `Math.Exp`/`Math.Sqrt`). Because those
transcendentals are host-libm, the oracle `windows` tier is **host-pinned** (`RunHostLibmCorpus` — strict
on win-amd64, `Inconclusive` off-Windows, like the FFT tier); `bartlett` is pure IEEE arithmetic and is
portable. **Fusion is the fast path, not just the clean one:** NumPy materializes ~6 intermediate arrays
per window; NumSharp materializes only `n` and fuses the rest into ONE pass, so **NPY/NS ≈ 1.8×–9× at
100K/10M** (bartlett's `where`-plus-two-divisions chain reaches 9× at 100K). At 1K the window sits at
NumSharp's per-op NDIter-setup floor (window generation is a compute-once, amortized operation — real use
generates one window and reuses it across many FFT frames). See `Math/np.windows.cs`; gates
`Math/np.windows.Test.cs` (22) + the `windows` fuzz tier (322 cases, incl. float-M).

### Fourier / FFT (`np.fft.*`)
`fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` — plus the companion complex accessors `conjugate`/`conj`, `real`, `imag`, `angle`.

The whole `numpy.fft` module (18 functions) is a **pure-managed C# port of pocketfft** — NumPy's own
vendored FFT engine — in `src/NumSharp.Core/Fourier/` (no native dependency, no P/Invoke). The
double/complex128 path is **bit-for-bit identical to NumPy 2.4.2**, not merely close, because the
port is a faithful **scalar** transcription and NumPy's win-amd64 wheel is itself scalar
(`POCKETFFT_NO_VECTORS` under MSVC) with twiddles from the same CRT (`Math.Cos`/`Sin` == MSVC
`ucrtbase`); **no explicit FMA is needed** (unlike the GEMM port), which is why `fft.jsonl` is a
PORTABLE fuzz tier rather than a host-pinned one. Full design + parity ledger:
**`docs/stale-docs/FFT_PARITY.md`**. **arm64 NUMPY is the exception, not NumSharp (proven
2026-09-23):** NumPy's arm64 wheels compile pocketfft with the default FP contraction on an FMA
baseline, so its `a*b + c*d` twiddle products (`sincos_2pibyn::operator[]`) and butterfly multiply-adds
(`MULPM`, `radf5`) round once (clang fuses the LEFT product: `fmuladd(a, b, c*d)`). Live arm64 NumPy's
`rfft` sits an ULP or so off NumSharp's in 73–98 % of float64 lanes (macos-latest, n=1024/1000/1021). A C# replica of
`rfftp`, fed macOS's libm twiddles, reproduces the SHA-256 of BOTH macOS results: NumSharp's when
evaluated literally, NumPy's when fused, at n=1024 and n=1000 (hashes in
`InteropTestBase.PocketFftFusedArithmetic`). The live interop gate keeps these cells strict on x64 and
reports them Inconclusive WITH the measurement on arm64 (`InteropTestBase.AssertExactUnlessNumPyFuses`,
`SpectrumLiveParityTests`, the Gist FFT/HPS cells). **Twiddle trap:** twiddles come from the PLATFORM
libm on both sides (`Math.Cos`/`Math.Sin` ↔ `std::cos`/`std::sin`), and libms disagree in the last
bit: at n=1024 Apple's `sin(72·π/4096)` and `sin(216·π/4096)` are 1 ULP above correctly rounded, and
ucrtbase misses correct rounding on 45 cos + 21 sin of the 1,025 candidate angles. So NumSharp's own
FFT bits are NOT host-independent for large n, and the offline `fft.jsonl` tier (win-amd64 bytes) is
portable only because ucrtbase, glibc and Apple agree on its sizes' twiddles (green on all three OSes).
A larger corpus size can turn host-pinned.

The facade is the `np.random` house shape — a lowercase property `np.fft` returning `FourierModule`,
so `np.fft.fft(x)` ports Python verbatim. Only **three 1-D kernels** do real work (`c2c`/`r2c`/`c2r`
in `PocketFFTDriver.Execute`); every N-D/2-D form composes 1-D transforms over `_raw_fftnd`, and
`hfft`/`ihfft` are compositions of `irfft`/`rfft` + `np.conjugate` + the norm swap (as NumPy defines
them). The **strided all-but-axis driver** (`PocketFFTDriver.cs`, port of `_pocketfft_umath.cpp`)
does `copy_input` (gather `min(nin,n)` strided elements, zero-pad) → `plan.Exec(buf, fct, dir)` →
`copy_output` per lane, reading each operand's own strides/offset so **every layout works** (C / F /
strided / transposed / reversed / broadcast-read / sliced-offset), and applies the **FFTPACK
half-complex packing** (`R0,R1,I1,…`, I0=0, even-`n` Nyquist term) exactly where NumPy's
`rfft_impl`/`irfft_loop` do. Engine files: `PocketFFT.Twiddle.cs` (`sincos_2pibyn` + `cmplx` +
prime/cost/good_size util), `.Complex.cs` (`cfftp` mixed-radix, codelets radix 2/3/4/5/7/8/11 +
generic), `.Real.cs` (`rfftp` radf/radb 2/3/4/5 + generic), `.Bluestein.cs` (`fftblue` chirp-z for
large-prime lengths), `.Plan.cs` (dispatch: mixed-radix when `largest_prime_factor(n)² ≤ n` else
cost-compare vs Bluestein, + a bounded 64-entry plan cache that is **bit-neutral** — NumPy's build
has `POCKETFFT_CACHE_SIZE==0`, so caching only saves twiddle setup).

**Layer-3 semantics are ported verbatim** (`np.fft.RawFft.cs`, port of `_pocketfft.py`; all probed
against 2.4.2): the norm factor `fct` (forward `None/backward→1`, `ortho→1/√n`, `forward→1/n`; the
inverse swaps direction FIRST, so `ifft` divides by `n`); `n<1` → `ValueError("Invalid number of FFT
data points ({n}) specified.")`; `n` truncates/zero-pads; `rfft` output `n//2+1`; `irfft`/`hfft`
default `n=2*(m-1)`; the N-D `s`/`axes` cook (`s`-without-`axes` → last `len(s)` axes, `s[i]==-1`
sentinel, `len(s)!=len(axes)` → `ValueError("Shape and axes have different lengths.")`). **Error
taxonomy matches NumPy's line-order exactly** — the bad-`norm` message has NO space after the first
comma on a FORWARD transform but a space on the inverse/`hfft`/`ihfft` (`_swap_direction`'s KeyError
path); an out-of-range axis is `IndexError("tuple index out of range")` when `n` is omitted (the
`a.shape[axis]` subscript fires first) but `AxisError` when `n` is given; `rfft`/`ihfft` of a COMPLEX
input is `TypeError("ufunc 'rfft_n_even'/'rfft_n_odd' not supported …")` (ufunc chosen by input
parity, fired only after n/norm/axis/out validate); a bad `out` is `ValueError("output array has
wrong shape.")`. `out=` accepts a **same_kind** cast from the loop dtype (irfft's float64 into a
complex128/float32 out; N-D wrappers accept `out` for signature parity but don't thread it, as NumPy
passes `out=None`). Helpers: `fftfreq`/`rfftfreq` are float64 generators with NumPy's probed check
order (divide-by-zero first, then `device`, then — `fftfreq` only — negative-length; a non-integer
`n` → `ValueError("n should be an integer")`); `fftshift`/`ifftshift` are dtype-preserving cyclic
rolls by `±(dim//2)` (0-d → no-op copy, `[Misaligned]`).

**ONE deliberate divergence (`[Misaligned]` F1), now genuinely dtype-only:** NumSharp has a single
complex type (complex128) and **no complex64**, so a **float32/float16** input returns complex128/float64
where NumPy 2.x returns complex64/float32/float16. The **VALUES are BIT-IDENTICAL to NumPy 2.4.2** (902/902,
every transform × size × norm × layout) — NumSharp reproduces NumPy's exact per-loop precision, which is
NOT a blanket "promote to double". NumPy's fft ufuncs carry a single (`Ff->F`/`ff->F`) and a double
(`Dd->D`/`dd->D`) loop and pick between them by operand shape: a **real float32/float16 input** to
fft/ifft/irfft promotes real→complex128 → the **double** loop → double compute rounded to the
complex64/float32/float16 output (`PocketFFTDriver.RoundInPlace`); **rfft of a float32 real input** and
**fft/ifft/irfft of a complex64-precision N-D intermediate** hit a **single** loop — but only when the
norm factor is a float (ortho/forward), since backward/None's int `fct=1` resolves to the double loop.
NumSharp mirrors this with a genuine single-precision pocketfft engine
(`Fourier/PocketFFT.{Complex,Real,Bluestein,Plan}.Single.cs`, `CmplxF`, `TwiddleF.At`) gated in
`PocketFFTDriver.Execute` on `floatPrec && !effNormUnity && (complex operand || rfft-float32)`; the norm
factor is computed in the input's real_dtype (`RawFft.FftFct`). float64/complex128/int/bool inputs are
**contractual (bit-exact)**; the helpers never diverge. The residual dtype cell needs complex64 (issue
#569), not FFT work. **The harness now VALUE-verifies these cells** (up-casts NumPy's complex64/float32/
float16 and bit-compares — `FuzzCorpusTests.Kinds.FftFloatValuesMatch`), so F1 is a checked dtype-only
excuse, not a skipped comparison. **Bonus:** this port uncovered and fixed a pre-existing bug in the
double `Cfftp`/`CfftpF` `pass7`/`pass11` **ido>1** codelets (they applied `special_mul` to `ca`/`cb`
instead of `ca±cb` — the PM — so `np.fft.fft(float64, n)` was silently wrong for any radix-7/11 factor at
ido>1: n=49/98/121/143/259/…, latent because the corpus n-sweep was only {4,12,13}). Gate:
`Fuzz/corpus/fft.jsonl` (2,000 cases, floor 1,700; n-sweep now includes the perfect squares
9/25/49/64/121/169 to gate every mixed-radix codelet's ido>1 branch pass3/5/7/8/11/passg — the bug
class above; `OpRegistry` all 16 transforms + 4 helpers, `gen_oracle.gen_fft`, `MisalignedRegistry` F1) +
**330** `Fourier`-namespace unit tests. See `Fourier/np.fft.{cs,Standard,Real,Hermitian,Helper,RawFft}.cs`
+ `Fourier/PocketFFT*.cs` (+ `*.Single.cs`); issues **#114** / **#569**.

### Polynomial package (`np.polynomial.*`) — evaluation family (U3)
`polyval`/`chebval`/`legval`/`lagval`/`hermval`/`hermeval` and their `*val2d`, `*val3d`, `*grid2d`, `*grid3d`, `*valnd` twins (36 names; `polyvalfromroots` open)

The `numpy.polynomial` PACKAGE (not the legacy `np.polyval` family) — facade `np.polynomial.{polynomial,
chebyshev,legendre,laguerre,hermite,hermite_e}` (instance-property modules, since a nested static class
cannot share `np.polynomial`'s name, CS0542), plan `docs/plans/numpy-polynomial.md` (U3 delivered, §10 engine).
Driver `Polynomial/Package/NDPolyEval.cs` (NumPy's Python layer: `np.array(c, ndmin=1)`, int/bool series →
float64, Python-scalar x stays WEAK, `tensor` reshape, `_valnd`/`_gridnd` two-pass compositions, verbatim
errors). Kernels `Backends/Kernels/ILKernelGenerator.Polynomial{,.Typing,.Emitter,.Lanes,.ConstPool,.Eval}.cs`:
Tier-3A per-chunk IL over `NDIterRef.ForEach`, emitted by ONE typed emitter from per-basis STEP TABLES
(NumPy's source-order expression trees, never re-associated, each node typed by NEP 50) — so no per-basis and
no per-dtype C#. **Bit-exact with NumPy 2.4.2** on every dtype pair and layout, result LAYOUT included (oracle tier
`polyeval.jsonl`, 23,866 cases, 0 excused — 4,650 of them from the 2026-10-01 review, section L: a "strides" facet over
every x × series layout, see the traps; 2,610 from the 2026-09-29 wholeness pass: section J, a Python-sequence x (tuples,
nested lists, empty, ragged, NumPy-scalar items — coerced by `PolySequence`, see U4) and a Python int x past int64 (a C#
`BigInteger`, weak: CPython int arithmetic on the x-only terms, inf / OverflowError past float64); section K, an
array_like c and array_like ordinates (below); `MisalignedRegistry`'s generic unary/complex ULP branches are carved out
for these ops so a drift fails). **Every array_like parameter is `object`** (the wholeness pass): c of `{p}val*`/
`{p}grid*`/`{p}valnd` (NumPy's `np.array(c, ndmin=1)`, converted BEFORE x, so a ragged c's error wins) and the
ordinates of `{p}val2d`/`{p}val3d`/`{p}valnd` (`_valnd`'s `np.asanyarray`: a Python int is a STRONG int64 0-d array,
so `polyval2d(2, 3, float32_c)` is float64 where `polygrid2d(2, 3, float32_c)`, whose scalars stay weak, is float32;
every ordinate converts before the shape check, the shape check runs before c). They were `NDArray`, so an
`object[]` compiled through the implicit array conversion and failed at run time, and a tuple did not compile. **Perf (NPY/NS, `benchmark/polynomial/`, 1,320 cells, every one SHA-256-checked):
geomean 7.5×, min 1.62×** — float64 `chebval` d10 16.9×/11.2×/36.5× at 1K/100K/10M, `chebval2d` 41.5×,
complex 23.7×@100K, float16 3.3×@10M, `lagval` (divider-bound) 2.7–14×, Python-scalar x 4.8–5.7×.

**Traps this family hit — do not re-break:**
- **NumPy runs the first 1–2 Clenshaw steps in the COEFFICIENT dtype** (`c0 = c[-2]` is a float32 scalar
  until it meets x): pre-casting a float32 series to the float64 loop changes 100 % of the points. The
  emitter PEELS those steps to a dtype fixpoint (kernel class `min(nc, P+3)`), then loops.
- **Python-scalar x is WEAK** (`chebval(2.0, float32_c)` is float32) and its x-only subtrees (`2*x`,
  `x*0`, `(2*nd-1) - x`) are CPython arithmetic — BigInteger ints, and a Python int → double is CORRECTLY
  ROUNDED (.NET's `(double)BigInteger` TRUNCATES: 42 oracle cells of `2**64-1` caught it). A 0-d NDArray x
  is STRONG. C# `char`/`Half`/`decimal` have no Python literal → strong 0-d arrays (the `np.r_` rule).
- **THREE complex multiplies:**
  - An ARRAY op is NumPy's fused `simd_cmul`.
  - A 0-d result is scalar math (the naive product), except an op touching the raw 0-d x array, which NumPy
    runs as a ufunc. chebval with a 0-d complex x differs from BOTH the array-x and the Python-scalar
    results.
  - An N-D series at a per-point x whose result has ONE element, with `c[k]` and x of different ndim, gets a
    third form. The trivial ufunc loop refuses mixed ndims, and NpyIter's one-element iteration gives every
    stride 0 (`nditer_constr.c` `if (bshape == 1) strides[iop] = 0`). So `CDOUBLE_multiply` runs its
    MSVC-contracted `loop_scalar`: im = `fma(ai, br, ar*bi)`, where simd_cmul has `fma(ar, bi, ai*br)`.
    This is `PolyUnitBroadcast` + `PolyComplexProduct.LoopScalar`: values carry NumPy's ndim, and the peel
    runs to the joint dtype+ndim fixpoint.

  The emitter reproduces all three, per op. The third form is probably library-wide for `np.multiply` too,
  unverified: `docs/plans/numpy-polynomial-review.md` step 10.
- **Vector lanes for EVERY dtype pair** (`...Lanes.cs`): 256-bit on AVX2 (float64 4 lanes, float32/float16
  8, complex128 2) and every other per-point dtype at the same lane count — int32 containers re-wrapped to
  int8/uint8/int16/uint16/char width after each op (NumPy's `2*x` WRAPS in int8), packed 2-lane containers,
  float16 as 8 float32 lanes with 2/4/8 live, complex `[re,im]×2` with vfmaddsub `simd_cmul` and Smith
  division (a shared divisor `/nd` prepared once per block). A lane-table miss compiles scalar chains (same
  bits) and bumps `ILKernelGenerator.PolyVectorFallbacks` — zero over the dtype matrix, unit-tested. Before
  the table, 227 of 936 small-size dtype cells were below 1.5× (min 0.25×).
- **One DynamicMethod per STAGE, or the JIT stops inlining:** once a method's inline budget is spent (its
  local-variable limit / time budget — AggressiveInlining does not override it) it inlines nothing more, so
  a monolithic kernel left its lane helpers — even `Vector256.Load` — as CALLS (a 23 KB
  complex kernel with 340 calls, 3.3× slower). The kernel is a stride dispatcher → part (vector/scalar ×
  broadcast/per-point series) → U-chain stage → remainder stage, each its own DynamicMethod (0–2 calls
  each). **Diagnose with `DOTNET_JitDisasm="NDPolyEval_*"` + `DOTNET_JitStdOutFile` and count `call`.**
- **float16 series in a float64/complex loop gang-load:** one 16-byte load + exact widen serves 8/W
  adjacent chains (each takes its lanes by one `vpermpd`); per-chain widening held those cells at 0.93–1.47×.
- **N-D series are NEVER buffered** (the kernel reads `c[k]` at `c0 + k*kstride`, outside the operand the
  iterator knows); the 1-D form buffers a strided x (coefficients live in auxdata).
- **The result LAYOUT is NumPy's last ufunc output's — and nothing pinned it until 2026-10-01.**
  - **Why it went unnoticed.** The oracle compares bytes in C order, so a result in the wrong order passes every
    value check. The N-D-series path had allocated C whatever its operands were.
  - **NumPy's rule.** NpyIter lays out each ufunc output from its operands (`Shape.NpyIterOutputShape`, a
    line-for-line port of `npyiter_find_best_axis_ordering` + `npyiter_new_temp_array`). An operand abstains on a
    stride-0 or extent-1 axis, and between disagreeing operands C order wins. That differs from np.copy's 'K' rule
    (`KeepOrder`), which sorts a broadcast axis innermost.
  - **The basis copies matter.** chebval copies its series with order 'K'; the other five read it as given. An
    int series converts by `astype` ('K'), except in polyval, where it is `c + 0.0` (NpyIter).
  - **Where it lives.** `NDPolyEval.SeriesLayout` and `ResultShape`: closed forms with C fast paths, and the step
    table replayed over layouts for an N-D series at tensor=False.
  - **The facet trap.** The section-L facet ZEROES the strides of extent-1 axes and of empty results. NumPy assigns
    those per loop path, and NumPy 2.x reports all-zero strides for any zero-size array. Never compare them raw.
- **Benchmark traps:** a file-based `dotnet run` REUSES its cached build — NumSharp.Core included — when the
  script did not change (`--no-cache` after every Core edit; a 936-cell run reproduced the old kernels to the
  microsecond); one warm-up pass only QUEUES tier-1 (warm, sleep for the background JIT, warm again); and
  NumPy's loops are cheapest L1/L2-resident, so a 100K-only dtype sweep hides the worst cells.

### Polynomial package — additive family + polyutils (U1)
`{p}add`, `{p}sub`, `{p}trim`, `{p}line` for the six bases, the 24 module constants `{p}domain/zero/one/x`, and
`np.polynomial.polyutils`: `as_series`, `trimseq`, `trimcoef`, `getdomain`, `mapparms`, `mapdomain` (54 names;
`format_float` belongs to the printing unit)

Plan `docs/plans/numpy-polynomial.md` (U1 delivered). Engine `Polynomial/Package/NDPolyNumber.cs` (`PolyNumber`: one
operand of NumPy's Python-level scalar code) + `NDPolySeries.cs` (the functions) + facade
`Polynomial/Package/np.polynomial.polyutils.cs` (`[ModuleName("np.polynomial.polyutils")]`, reachable as
`np.polynomial.polyutils`); every element loop is an IL kernel in `Backends/Kernels/Direct/DirectILKernelGenerator.PolySeries.cs`
(trim scan, in-place combine, tolerance scan, cast, and one-element scalarmath kernels behind flat slot arrays).
**Bit-exact with NumPy 2.4.2** — oracle tier `polyseries.jsonl` (19,478 cases, 0 excused: incl. 1,458 long-series, 45
block-boundary and 72 getdomain window-crossing cases, 752 complex64 results value-compared, 184 tuple / nested-sequence
cases — section O, coerced by `PolySequence`, see U4 — 26 `trimseq` cases over Python sequences, section P, the deferred
object / str refusal, section Q, and 780 mapdomain result layouts, section R: values + a "strides" facet over every
layout) + `Polynomial/PolynomialSeriesTests.cs` (39 tests) + `PolynomialResultLayoutTests.cs`. The
C# boundary is the house NEP 50 map: bool/integers/float/double/`Complex`/`BigInteger` are Python scalars,
`Half`/`char`/`decimal` NumPy scalars, an `NDArray` (0-d too) or typed C# array an ndarray, `object[]`/`IList` a Python
list, a `ValueTuple` a Python tuple — and that KIND decides whose arithmetic runs. NumPy never converts a domain, it
indexes it, so there are THREE arithmetics: Python∘Python is CPython (exact ints, `long_true_divide`, 3.12
`_Py_c_quot`, ZeroDivisionError with CPython's three texts), NumPy scalars are scalarmath (NEP 50, the NAIVE complex
product, inf/nan instead of raising), and anything touching an ndarray is a ufunc (`pycomplex op np.float64` is still
CPython — complex's methods accept a float subclass). `_add`/`_sub` update the LONGER operand in place (c2 on a tie;
`_sub` with `len(c1) <= len(c2)` negates c2 then adds c1, visible in a NaN's sign) and trim; `trimseq` returns the
input ITSELF or the VIEW `seq[:k]` (so writes reach the input) — and of a Python sequence the same KIND (the wholeness
pass: `trimseq(object[])` is the list itself or a new list of the kept items, an `NDArray[]` staying one; a tuple itself or
a new ValueTuple; any other enumerable itself or an `object[]`; zero-ness is Python's `item != 0`, an array item by its
truth value with NumPy's ValueError past one element; None or a number is `len()`'s TypeError); `trimcoef` checks
`tol < 0` before converting;
`{p}line` is `np.array` DISCOVERY (`polyline(1, 2)` is int64); `mapdomain` converts x only when it is not an
int/float/complex/`np.generic` (a bool x becomes a 0-d array). The constants are ONE shared, writeable, scope-detached
instance each (NumPy's module attributes — a write persists), holding an extra ARC reference so a caller's `Dispose()`
cannot free them; their corpus cases vary the FACET (value, identity, writeable, owndata).

**Fast paths (all proven against the exact general lane, never different):**
- **Generic tuple overloads** `mapparms<T0..T3>((T0,T1), (T2,T3))` and `mapdomain<T0..T3>(double | Complex | NDArray x, …)`:
  a C# tuple literal binds them (identity beats boxing), elements read without boxing (`PolyNumber.FromValue<T>`).
  One per x kind is REQUIRED — with only the `NDArray` one, `mapdomain(complexX, (-1, 1), (0, 2))` is CS0121-ambiguous
  against `mapdomain(Complex, object, object)`.
- **The CPython machine-number lane** (`NDPolySeries.PyNum`): Python int (within `long`) / float / complex operands of
  tuples AND lists run CPython's arithmetic on machine numbers; an int intermediate leaving `long` (checked subtract,
  `Math.BigMul`) BAILS to the exact `BigInteger` lane, so the lane can only be faster.
- **A fused scalarmath kernel** (`GetPolyMapParmsKernel`) for two 1-D ndarray domains of one non-bool dtype — the
  `ABCPolyBase` case: the six operations in one call, intermediates wrapped/rounded in the dtype through locals of its
  CLR type (`stloc` to an `int8` local truncates exactly like the per-op narrowing store), float64 true division for
  integer dtypes via `EmitConvertTo`.
- **Per-thread reusable 0-d parameters** for `mapdomain`'s one fused `np.evaluate` pass (a hoisted 0-d input is read
  once, before the pass).

The registry replays every `mapparms`/`mapdomain` case through the object overload, the generic overload (tuples
rebuilt element-typed by reflection) AND `NDPolySeries.MapParmsGeneral` (the exact reference), requiring all three to
agree to the byte; `MachineNumberLane_AgreesWithTheExactGeneralLane_BitForBit` adds 22,400 seeded comparisons.

**Long series take house SIMD routes (the 2026-09-27/28 parity audit).** The first corpus stopped at a dozen
coefficients, so every cell ran the scalar IL kernels; from `PolyHouseKernelThreshold` (64 elements) on the same
operations take vector routes, each byte-identical to the scalar one and pinned by corpus sections K (long series),
L (complex64 array loops), M (block boundaries) and N (getdomain across its windows):
- **`{p}add`/`{p}sub`** — `CombineViaHouseKernels`: the in-place target materialized block by block (1,024 elements:
  memcpy / SIMD cast / SIMD strided copy), negated where NumPy negates it, then updated by the same-dtype `SimdFull`
  kernel, when both operands are "vector-readable" (`VectorReadable`: contiguous, or strided with a result of at most 4
  bytes or a sub-word source, and a vector cast); float16 always.
- **`as_series`' conversion copy** (`CopyInto`) — memcpy / the house contiguous cast / the SIMD strided copy, a
  converting strided source copied then cast through an L1 block.
- **`getdomain`** — ONE blocked pass for every layout of a float64 / float32 / integer x (`TryDomainBlocked`): NumPy's
  C-contiguous copy is taken 8 KB at a time — read in place from a unit-stride x, packed into an L1 scratch by
  `CopyInto` otherwise — and both the min and the max are folded while the window is in L1 by the engine's exact
  schedule (`NumPyMinMaxReduce.FoldGroups` per full window, `Finish` for the last), which is bit-identical to one
  `simd_reduce_c` over the whole copy because windows are whole 8-vector groups counted after element 0 (the
  `StreamMinMax` argument). An integer x is order-free, so a reversed one is folded as the forward run it covers
  (`PolySeriesView.Reversed`); a float x keeps its order (the ±0 / NaN bits of min/max depend on it). uint64 packs
  converted to float64 (AVX2 has no unsigned 64-bit compare: the emulated folds cost 20 µs per 100K, the splice cast +
  float64 folds ~14). float16 / char decline to the copy-then-reduce route. Replaced: allocating the copy, then two
  engine reductions (100K stride-2 float64 / int64 / uint64: ~37 / ~45 / ~50 → ~23 / ~30 / ~31 µs; 16 points ~0.9 →
  ~0.3 µs; getdomain's 90 cells now 2.13–104× NumPy, were 1.26–75×).
- **`mapdomain`** — `MapDomainAffine` for a float64 / float32 / complex128 loop shared by both operations over 1-D (any
  stride) or C-contiguous points: converted 8 KB at a time into an L1 scratch, then `PolyAffineDouble`/
  `PolyAffineSingle` (multiply THEN add, two roundings — never an FMA) or `PolyComplex128Affine` (`simd_cmul` + add).
  The fused `np.evaluate` pass keeps what it runs at vector speed (`FusedPassIsFaster`: a contiguous x on a vector
  widen edge or of bool; a strided x already of the loop dtype or on an edge from 8,192 points, where that pass packs
  it). Measured in ONE process (`NDPolySeries.DisableAffineMapDomain`), fused/affine geomean 1.34 at 16 points,
  1.53–1.62 at 1,000, 1.36–1.46 at 100K, min 0.94. complex64 loops run on the float32 kernels
  (`MapDomainComplex64`).
- **New house kernels, bit-exact and library-wide through `TryGetCastKernel`:** AVX2 int → float
  (`Cast.IntToFloat.cs`: {int8, uint8, int16, uint16, char, int32, uint32} → {float32, float64}; int64 / uint64 →
  float64 by the exponent-bias splice, one round-to-nearest-even rounding — 2.2–3.3× the scalar loop, 134M random
  values identical); float16 → float64 exactly as NumPy's `ToDoubleBits` (a signalling NaN stays signalling, which
  `cvtps2pd` and `(double)Half` would quiet); the 4/8-byte same-type strided copy (`Cast.WordCopy.cs`: SIMD reverse,
  deinterleave, gather); and `EmitConvertTo`'s float32 ↔ float64 through the packed `cvtps2pd`/`cvtpd2ps`.

**Perf (NPY/NS, P-cores pinned, best-of-7, NumPy re-measured back to back; 867 cells = dtypes × 16 / 1,000 / 100K
(+1M) × contiguous / stride-2 / reversed; geomean 8.47×):** add/sub 1.70–24.3× (geo 6.7), trim 5.5–515× (geo 24.6),
trimseq 3.8–490× (geo 18.2), getdomain 2.13–104× (geo 8.3), mapdomain 1.96–33× (geo 4.9), as_series 1.11–10.3×
(geo 3.9), mapparms 1.18–8.7× (geo 4.1), `{p}line` 0.80–1.03×. The 26 cells under 1.5× are floors, measured:
**`{p}line` (21) is the NDArray allocation floor** (~230 ns to create and dispose a 2-element array; NumPy's whole
`np.array([off, scl])` is 260–430 ns); **`as_series` of two 100K arrays (4, 1.11–1.36×) is memcpy** — both sides copy
the same 2 × 800 KB, and two plain `Buffer.MemoryCopy`s into already-allocated buffers take 45.5 µs of the call's
44.7 (float64); **`mapparms(float64 array, (0, 2))` (1.18×, 0.35 vs 0.42 µs) is the exact general lane's dispatch** —
four element reads (~60 ns) and six `PolyNumber.Binary` operations (~25 ns each, a >100-byte struct through every
step) for a mix of NumPy scalars and Python ints; a typed program per argument-kind signature is the known lever.

**Traps this family hit — do not re-break:**
- **CPython's NaN operand priority is per-operator, and a C# operator does not pin it.** When both operands are NaN
  x86 returns the FIRST source's, and RyuJIT swaps commutative `a + b`/`a * b` for register allocation — two call sites
  of one expression disagreed. CPython 3.12 (MSVC, probed): `float_add` returns the RIGHT operand's NaN,
  `float_sub`/`float_div` the left's, `float_mul` the left's once the call site is specialized
  (`BINARY_OP_MULTIPLY_FLOAT`) but the right's on its first, generic execution (interpreter state — the steady state
  is modelled), and every operation inside `_Py_c_sum/_diff/_prod/_quot` the left's. `_Py_c_quot`'s NaN branch is
  `Py_NAN` = the POSITIVE `0x7ff8…` (.NET's `double.NaN` is `0xfff8…`). All Python arithmetic goes through
  `PyScalar.{FloatAdd,FloatSub,FloatMul,FloatDiv,ComplexSum,ComplexDiff,ComplexProd}` (explicit NaN tests), which the
  seeded property test found by comparing the two lanes.
- **A C# switch expression takes its arms' natural type:** `Kind switch { Int => long, Float => double, _ => Complex }`
  is typed `Complex` (both convert to it), so an uncast arm boxed every real result as a Complex — cast each arm to
  `object`.
- **Python's evaluation order is observable only through errors:** mapparms reads each element ONCE, but in Python's
  order (old's subtraction before new is indexed), so an error in `old` still wins over a short `new`.
- **`stackalloc` scratch is ZERO-FILLED unless the method carries `[SkipLocalsInit]`.** `CopyInto` runs once per block
  inside the combine and mapdomain loops, and zeroing its 8 KB scratch every call cost more than converting the block
  (mapdomain at 16 points: 1.03 → 1.34× the fused pass once removed).
- **A strided source reads more than its block:** a stride-2 float64 block of 16 KB drags 32 KB of source lines in,
  which with the scratch overflows a 48 KB L1 — 16 KB blocks ran 0.72× the fused pass where 8 KB kept parity.
- **100K-point timings move 1.5–2× between runs with the allocator's page state**, and a whole run can drift ~10%. A/B
  two routes in ONE process, interleaved (`DisableAffineMapDomain`), and re-measure NumPy back to back before quoting.
- **Measure a cast before calling it vectorized:** the generic emitter's int → float strategies ran at SCALAR speed
  (0.19–0.37 ns an element on an L1 block; only int16/uint16 → float32 vectorized), and the scalar `cvtss2sd`/
  `cvtsd2ss` behind `conv.r8`/`conv.r4` MERGE into their destination register — a false dependency that made a scalar
  float32 → float64 loop latency-bound (128 vs 37 µs).
- **NumPy's scalar-broadcast complex multiply is `simd_cmul` at every length for a trivially iterable call** — even ONE
  element (`s * np.array([x])`, probed 300/300); `loop_scalar` needs NpyIter's stride-0 single-element iteration.
- **Test trap: `GetUInt32(i)` on a 2-D array reads the ROW coordinate `i`, not flat index `i`** — flatten first.
- **C# picks an `NDArray` overload over an `object` one for ANY C# array argument** — `NDArray` is the "better conversion
  target" (it converts to `object`, never back), even though the conversion into it is user-defined. So with both
  `mapdomain(NDArray x, …)` and `mapdomain(object x, …)` an `object[]` (a Python list) went through NumSharp's implicit
  array conversion, which cannot interpret `object` elements, and threw at run time. A list gets its own `object[]`
  overload (`mapdomain`, `trimseq`), which beats both by identity; a typed `double[]` still binds the NDArray one.
- **Sign-mixed zeros ALONE cannot tell NumPy's min/max schedule from a sequential fold:** with every element a tie,
  each lane keeps its own last zero and the horizontal cascade lets the HIGHEST lane win, which holds the array's last
  element — the same answer. A discriminating case needs the extreme shared by lanes out of order: a -0.0 early in the
  highest lane and a +0.0 late in lane 0 among smaller values (NumPy returns the early -0.0). And a scalar tail
  decides ties by itself (the last element wins), so the length must leave none.
- **Repeating extremes hide a dropped element:** a window that skipped the last 8 elements of every block passed the
  route-agreement test on patterned data (extremes every 13 elements) and was only caught by planting a UNIQUE min and
  max at every position around each window edge (`GetDomain_BlockedPass_FoldsEveryElement`).
- **An A/B that alternates routes batch by batch lets one route's allocations evict the other's working set:**
  interleaved, the blocked getdomain measured 39–43 µs at 100K stride-2 int64/uint64; timed alone, 30. Time each
  route as a whole warm batch, and confirm against a standalone run.
- **mapdomain over an N-D non-C x: transpose, compute C, RELABEL — never write NumPy's layout directly.** NumPy keeps x's
  memory order, but NumSharp's NDIter walks a pair of operands sharing a PERMUTED layout in logical order, gathering
  from both. It only reverses axes for all-F operands. `np.add(xp, 1.0, out=)` costs 41 µs into xp's own (3-D
  transposed, 32³) layout against 17.5 µs into C — library-wide. So `MapDomainWith` runs every route over
  `np.transpose(x, layout.DenseAxisOrder())`, which is C-contiguous for a dense permuted x and takes the affine SIMD
  route. Then `NDArray.AdoptLayout` (`Backends/NDArray.AdoptLayout.cs`, shared with np.clip and np.trapezoid) relabels
  the C result's storage with NumPy's layout (`UnmanagedStorage.SetShapeUnsafe`): no element moves, OWNDATA is kept. It
  checks the contract (a fresh C-contiguous, offset-0 result) and copies when it does not hold. That shape had been
  0.62× NumPy (the 867-cell matrix measured 1-D layouts only) and is now 2.13×.
- **RyuJIT rewrites `(-a) + b` into `b - a`, and that changes a NaN's sign.** The NegateAdd combine kernel (`_sub`'s
  "negate c2, then add c1") got it when the subtrahend's conversion was an inlined float16 → float32/float64 helper.
  `b - a` gives a NaN subtrahend the subtraction's sign rule instead of the negated sign. Fixed by storing the
  negated value and reloading it before the add (`EmitPolyCombine`), and pinned by
  `Sub_EqualLengths_ConvertedFloat16Subtrahend_StillNegatesFirst`. The oracle tokenizes NaN, so only a byte-level
  unit test sees it.
- **Two NaNs meeting in `_add` / `_sub` keep NumPy's operand's NaN (re-review 2026-10-05).** NumPy's update is an
  in-place ufunc over two arrays, and its win-amd64 add keeps the SECOND operand's NaN (the other series, or c1 after
  `_sub`'s negation); subtract keeps the first. The short-series combine kernel (series under 64 elements) kept the
  target's NaN on every add, while longer series (the house kernels, 3169579a) already matched. `EmitPolyCombine` now
  emits the add through `EmitScalarOperation(…, NaNOperandOrder.NumPyUfunc)`. Pinned by
  `AddSub_NaNPair_KeepsNumPysOperand`; it also reaches the U2 algebra, which runs `_add` / `_sub` in every step.
  Measured cost: none (fixed/unfixed 0.93–1.05 over 11 cells).

### Polynomial package — calculus family (U4)
`{p}der` and `{p}int` for the six bases (12 names), with every parameter — `m`, `k`, `lbnd`, `scl`, `axis`

Plan `docs/plans/numpy-polynomial.md` (U4 delivered). Driver `Polynomial/Package/NDPolyCalc.cs` = NumPy's Python
prologue in NumPy's statement order, which is also its error order. Kernel
`Backends/Kernels/Direct/DirectILKernelGenerator.PolyCalculus.cs`: the six recurrences are DATA (`PolyCalcRoutines`
— head steps, a j loop, tail steps; each step's statements are NumPy's source expressions token for token) and ONE
whole-array kernel is emitted per (basis, der/int, dtype, source dtype, 1-D, fused scale). **Bit-exact with NumPy
2.4.2** — oracle tier `polycalc.jsonl` (27,526 cases, 0 excused; 21,646 at delivery, +5,880 from the 2026-09-29
wholeness pass: argument kinds (M), zero-size / 5-D / extreme-int series (N), widened scale (O)), 10/10 planted bugs
killed at delivery plus 4/4 widened-scale ones (44–73 red cases each). The smallest kill is 8 cases: the
block-boundary mutant, which only the >4096-column cells can see. Unit tests: `Polynomial/PolynomialCalculusTests.cs`
(20 — NumPy's `TestIntegral`/`TestDerivative` ported, made bitwise where NumPy's statement sequence makes them exact;
byte dumps per dtype family; kernel structure) + `Polynomial/PolynomialArgumentKindsTests.cs` (21 — the C# boundary
kinds with NumPy-probed bytes and texts).
**Perf (NPY/NS, `benchmark/polynomial/polycalc_*`, 720 cells, every one SHA-256-checked): min 1.71×, geomean 7.62×**:
- 1-D series: geomean 25× (NumPy runs a Python loop over the coefficients);
- N-D float64: 1.8–13×, geomean 4.7×;
- dtypes: geomean 7.3×;
- small N-D: geomean 11.7×;
- argument forms (lists, widened scale, array scl): 1.71–13×, geomean 5.7× — the floor is the float16 widened-scale N-D
  cells, the open float16 lever below.

How it is built:
- **One buffer, in place, for any order.** The recurrences never need a value after they overwrite it, provided
  the result is written one row over. A derivative writes `der[q]` to row q+1 (the result is rows [m, n)). An
  integral writes `tmp[q]` to row q with `c[q]` at row q+1, so m spare rows sit above the loaded series.
- **Columns are independent.** A position in `c.shape[1:]` never meets another, so the kernel takes them in
  cache-resident blocks (~512 KB) through EVERY order of a derivative. It uses the house lane kinds of U3
  (`.Lanes.cs`) over the columns, then a scalar tail.
- **The load stage fuses the conversion and the scale.** It converts int/bool series to float64 and applies
  `c *= scl` in the same first pass. It reads the caller's layout in place when every row's columns sit at one
  stride (every 1-D and 2-D series, C-mergeable N-D); otherwise NDIter copies first.
- **NumPy's three arithmetics, per statement.** `c *= scl` is always an array op: `simd_cmul`, operands (c, scl).
  A 1-D series' recurrence statements are SCALARMATH (the naive complex product, the `ScalarMath` key); an N-D
  series' are ufuncs.
- **The integral's correction reuses U3 and U1.** `tmp[0] += k[i] - {p}val(lbnd, tmp)` runs through U3's
  `NDPolyEval.Val` plus U1's `PolyNumber`: scalarmath and setitem's cast for a 1-D series, an in-place ufunc for
  an N-D one.
- **Weak Python ints** of the recurrence (`2*j`, `j+1`, …) are converted per (block, j) through a double and then
  the house cast. So float16 constants past 2048 round to even, and past 65504 become inf, as in NumPy.

**Result objects are NumPy's** (the corpus records C/F/OWNDATA of each result as a `"facet": "flags"` case):
- the growing orders: `np.moveaxis` views of fresh C-order buffers (OWNDATA false);
- `m == 0`: the K-order copy (OWNDATA true; an F series stays F);
- der with `m >= n`: `c[:1]*0`, laid out by NpyIter's KEEPORDER vote over `c[:1]` (`Shape.MultiSortedStridePerm`),
  then moved back — except by **hermeder, which skips the moveaxis** (a NumPy quirk, reproduced);
- an integral whose every order took the n == 1 zero branch: a view of the moved copy.

**Errors, in NumPy's order and texts:**
- a complex scl into a real series is caught at the first `c *= scl`, so it is reported before an empty series'
  IndexError;
- an array constant for a 1-D series is setitem's ValueError into a REAL series, but `complex()`'s TypeError
  (`only 0-dimensional arrays can be converted to Python scalars`) into a COMPLEX one — probed for every shape;
- for an N-D series, a complex correction is the in-place add's UFuncTypeError, which names **complex64** when the
  series is float16/float32 and the lbnd or k is a Python complex (NumPy's complex64 loop).

**The 1-D complex64 value.** A 1-D float16/float32 series with a Python-complex lbnd makes NumPy compute `{p}val` in
complex64 SCALARMATH. U3's evaluation kernel works in complex128, so `NDPolyCalc.PyVal1D` interprets U3's weak-x step
trees (`PolySteps.RewriteForWeakX`) with `PolyNumber` instead; `PolyNumber` emulates complex64 scalars exactly.

**Argument kinds — `np.array`'s coercion, ported (the 2026-09-29 wholeness pass).** Every array_like the package
converts — `{p}der`/`{p}int`'s c, k, lbnd and scl, `{p}val*`/`{p}grid*`'s c and x, `{p}val2d`/`{p}val3d`/`{p}valnd`'s
ordinates, `mapdomain`'s x, `as_series`' items, `{p}add`/`{p}sub`'s operands, `trimseq`'s items — goes through `Polynomial/Package/NDPolySequence.cs` (`PolySequence`), a statement-for-statement port of
`array_coercion.c` (`PyArray_DiscoverDTypeAndShape_Recursive` + `update_shape`) and its fill:
- **The C# map** (header of `NDPolyNumber.cs`): `ITuple` (ValueTuple, System.Tuple) is a Python tuple; `object[]` and
  every other C# array whose elements are not a NumSharp dtype (jagged `double[][]`, `NDArray[]`, `BigInteger[]`) and any
  other `IList`/`IEnumerable` are lists, a multi-dimensional `object[,]` a list of rows; a typed dtype array
  (`double[]`, `float[,]`) and a `Memory<T>` of a dtype are NDARRAYS (strong dtype); `bool`/integer primitives/
  `BigInteger`/`float`/`double`/`Complex` are Python scalars, `Half`/`char`/`decimal` NumPy scalars, a string a str.
  A list of ONE scalar kind converts in one pass, at any depth: a typed collection (`List<double>`, `HashSet<int>`,
  LINQ …, discovered dtype per its C# element) or a flat `object[]` of Python floats / ints (U5 wholeness pass,
  `TryTypedCollection` / `TryFlatLeaf`). It takes the walk's two shape updates and becomes one array leaf; the item walk
  cost ~40 ns an item.
- **Shape:** the first leaf reached fixes the dims; a later leaf at another depth, or a sequence of another length,
  is ragged — NumPy's verbatim `setting an array element with a sequence. The requested array has an inhomogeneous
  shape after N dimensions. The detected shape was (…) + inhomogeneous part.` (ValueError), N and the shape being
  what the walk still agreed on. An EMPTY sequence ends the dims without a dtype vote (`[]` is float64 `(0,)`,
  `[np.zeros(0, int8), []]` int8 `(2, 0)`); an array leaf that cannot be assigned raises the broadcast ValueError
  (`could not broadcast input array from shape (0,3) into shape (0,)`).
- **Dtype:** each leaf's DEFAULT descriptor (Python int → int64, uint64 up to 2^64−1; a NumPy scalar / array keeps
  its own), promoted STRONGLY — array coercion is not NEP 50 (`[np.float32(1), 1.0]` is float64). A str / None leaf,
  a Python int past uint64 and any unknown object are refused (`NotSupportedException`) AFTER the walk, so a ragged
  input still reports NumPy's ValueError. That holds where NumPy computes with the array at once (U3's evaluation,
  U4's calculus). `polyutils.as_series` — U1's additive family, U2's whole series algebra — first checks every
  argument's size and dims and then its common type, so there `ToArrayOrNonNumeric` hands back NumPy's str / object
  array as a SHAPE plus a deferred refusal (`PolySeriesView.NonNumericArray`, the U2 wholeness pass below).
- **Calculus-specific order, all probed:** `np.ndim` runs on lbnd and scl before anything is converted (a ragged
  sequence raises the inhomogeneous text there; a str is 0-d); k's items are converted only when their order uses
  them, AFTER that order's `{p}val(lbnd, tmp)` (Python's left-to-right `k[i] - {p}val(…)`), so a bad lbnd wins over a
  bad later constant; a str lbnd is refused only where NumPy evaluates at it (never for m == 0, the one-coefficient
  zero branch, or before a later argument's own error); a `BigInteger` past float64 raises CPython's
  `int too large to convert to float` (`OverflowException`), one within float64 but past float32 / float16 is inf.
- **A conversion is the call's intermediate, released however the call ends.** `{p}val` disposes an x it converted
  (the leak gate caught 432 sequence-x cases leaking one buffer each), `_valnd`/`_gridnd` dispose a partial result when
  a later pass fails (a ragged second ordinate), `mapdomain(object x)` its converted points; the U1/U4 facades are
  `[NDScoped]`, the U3 ones dispose by hand. A pool-balance probe (`SizeBucketedBufferPool` takes − returns around a
  call whose result is disposed) found the non-corpus ones — build its operands OUTSIDE the measured lambda, or the
  probe counts its own arrays.

**Widened scale.** A strong scalar that PROMOTES the series — float64 against float32, float32/float64 against
float16 — makes NumPy's `c *= scl` run the WIDER multiply loop and cast back. The kernel fuses that too: `PolyCalcKey`
carries the loop dtype (`ScaleLoop`, `HasWidenedScale`), and the load stage calls `PolyLaneOps.{F32ScaleF64,
HalfScaleF64,HalfScaleF32}` (vector) and their `*Scalar` twins (tails, 1-D series). The float64 → float16 narrow must
round ONCE (`npy_double_to_half`): the vector path rounds the float64 product to float32 with ROUND-TO-ODD
(`RoundToOddF32`: truncate + sticky bit) and then narrows to float16 to-nearest-even, which is exact because 24 ≥ 11 + 2
bits; a plain f64 → f32 → f16 chain double-rounds at float16 ties (section O's hazard cases, built by
`_pc_f16_hazards`). A NaN coefficient keeps its own payload, quieted (the first operand's NaN wins in NumPy's loop). An
ARRAY scl (derivatives only) and a promotion the kernel has no widened loop for (decimal) run `ScaleInPlace` =
`np.multiply(c, scl, out: c, dtype: promote(series, scl))` per order — the explicit `dtype:` is load-bearing, because
the house ufunc computes a float32 array times a 0-d float64 with `out` float32 in FLOAT32 (44 corpus cells red
without it); a complex scale into a real series raises NumPy's UFuncTypeError there.

Traps:
- **float16 N-D is ~30× slower than float32** (still 2.1–2.6× NumPy). The U3 float16 lane kind rounds every op back
  onto the f16 grid (narrow + widen), then the store narrows again. A cheaper in-float round buys only ~1.25×; the
  real lever, NOT done, is a float32 working buffer narrowed once at the end.
- **The oracle's flags facet REPLACES the result**, so the registry (`OpRegistry.PolySeries.cs` `CalcFacet`) must
  dispose the result it drops. Otherwise the leak gate (`UndisposedIntermediateTests`) reads one escaped buffer per
  flags case.
- **A quoted `--filter` passed from Git Bash through `cmd //c` reaches dotnet with LITERAL quotes** (`\"X\"`). A
  `TestCategory!=…` filter then matches EVERYTHING and a `ClassName~…` filter NOTHING. Read the filter from a file
  in the `.cmd` script (`set /p FILTER=<filter.txt`).
- **describe() serializes an operand's BASE in C order.** A layout must therefore be a C-contiguous base plus a view:
  an F series is the `.T` of a C buffer holding `base.T`. An F base broke 1,311 cases in the first oracle run.
- **Random scales never hit a float16 double-rounding hazard.** A product must land within ~2^-24 (relative) of a
  float16 tie for f64 → f32 → f16 to differ from NumPy's single rounding, which a random scale almost never does —
  the first hazard search found none. Build them: `_pc_f16_hazards` scans every finite float16 against scales like
  `1 + 2^-11 ± 2^-40`, which put the powers of two just past a tie.
- **Two C# traps when building nested Python sequences in tests and benchmarks:** a lone `object[]` passed to a
  `params object[]` parameter IS the params array, not its one item (wrap it: `new object[] { inner }`); and a `?:` or
  switch expression whose arms are `object[]` and `NDArray` is typed `NDArray` through NumSharp's implicit
  `Array → NDArray` conversion, so the list silently becomes an ndarray (or fails, `Cannot interpret System.Object as a
  data type`) — cast every arm to `object`.
- **`start /b /wait X.cmd` from Git Bash runs X under `cmd /K`**, which can wait on console input forever after X
  finishes; wrap it: `start /b /wait /affinity … cmd /c X.cmd`.

### Polynomial package — series algebra (U2)
`{p}mulx`, `{p}mul`, `{p}div`, `{p}pow`, `{p}fromroots` for the six bases and `X2poly`/`poly2X` for the five non-power
bases (40 names)

Plan `docs/plans/numpy-polynomial.md` (U2 delivered). NumPy runs these as Python loops over SHORT series, every
statement one of: `as_series`, `_add`/`_sub`, a `{p}mulx`, an ARRAY op with a Python int or NumPy scalar
(`c[-i] * xs`, `(c1 * (nd - 1)) / nd`, `c1[i:j] -= c2 * c1[j]`), `np.convolve`, `trimseq`, or scalarmath on one
element. The engine (`Polynomial/Package/NDPolyAlgebra.cs` + the per-basis `NDPolyAlgebra.Bases.cs`) replays that
Python statement for statement, in NumPy's order and dtypes, over raw series (`PolySer`: pointer, length, dtype, and
whether NumPy returns it as a view) in a per-thread bump arena (`PolyArena`). Each statement runs through the kernel
NumPy's statement runs through in NumSharp:
- array ops: the house ufunc kernels (`GetPolyHouseBinaryKernel`, flat slot fronts over `GetMixedTypeKernel`:
  simd_cmul, CDOUBLE_divide's Smith, the HALF loop);
- as_series / `_add` / `_sub`: U1's substrate (`TrimLength`, `CopyInto`, `CombineInto`);
- `{p}mulx`: an integral-layout calculus kernel (`PolyCalcKey.Mulx`); chebmulx of ≥ 2 terms the fused one-pass kernel
  (below);
- `np.convolve`: `NDArray.SlidingCorrelateInto`, the engine `np.convolve` itself runs, the byte-parity OpenBLAS route
  included;
- scalarmath: U1's one-element kernels (the naive complex product, NumPy's scalar division).

So the only C# is NumPy's own Python: its loops are over SERIES, never elements. **Bit-exact with NumPy 2.4.2**
except the convolution bases' BLAS-bound products without the backend (below). Oracle: `polyalgebra.jsonl` (28,144
portable cases, 0 excused; 26,163 at delivery) + the host-pinned `polyalgebra_parity.jsonl` (139 BLAS-bound products,
byte-exact with `NumSharp.Interop.OpenBLAS` at threads=1, Inconclusive off the pinned host). Unit tests
`Polynomial/PolynomialAlgebraTests.cs` (20: NumPy's `TestArithmetic`/`TestMisc` ported, dtype-quirk byte dumps,
decimal/char, the arena, the fused chebmulx kernel, the non-finite complex product, the ±0 divergence) and
`PolynomialAlgebraArgumentKindsTests.cs` (5: the wholeness pass below).
**Perf (NPY/NS, `benchmark/polynomial/polyalg_*`, 452 cells, every one checked against NumPy): min 1.57×, geomean
37.8×** (404 cells, min 1.45×, geomean 34.9× at delivery). By section (geomean): mulx 25×, mul float64 31×, other
dtypes 29×, div 67×, pow 28×, fromroots 35×, conversions 100×, Python-list arguments 39×, pow / maxpower argument
kinds 39× (min 5.1×: the object overloads' int() / comparison cost ~20 ns). The lowest cell is the complex128
1000×1000 product at 1.45–1.57× (run to run), a physical ceiling (below).

Dtypes follow NumPy's Python statement by statement, not a promotion rule:
- a recurrence-basis product (leg/lag/herm/herme) whose SHORTER factor has one term binds the Python int `c1 = 0`,
  which as_series makes float64 `[0.]`, so a float16/float32 product turns float64;
- `_div`'s remainder is computed against the int unit series `[0]*i + [1]`, so a float32 division's remainder is
  float64 while its quotient is float32 (polydiv, which skips `_div`, keeps float32 in both — its remainder a VIEW);
- poly2X starts from `res = 0` (float64); fromroots' lines are `np.array([-r, 1])` (strong array coercion: float16/32
  roots give float64); `{p}pow(c, 0)` is `[1]` in c's dtype.

**The arena.** A series costs a pointer bump, not an NDArray (~200 ns each, which was the whole cost of a 1-to-1
`NDArray` port). Every internal function returns its result through `Keep`, which moves it down to the function's
entry mark, so memory is bounded by the live series. Blocks come from `SizeBucketedBufferPool`, and growth blocks
(powers of two) go back to it when the outermost call exits. Fresh OS memory was ~150 µs of first-touch page faults
per 800 KB per call, more than a long product's convolution itself. Only the final result becomes an NDArray, with
NumPy's ownership: a trimseq slice is a VIEW (`flags.owndata` false).

**Products and np.convolve's BLAS switch.** NumPy's power- and Chebyshev-basis products are `np.convolve`, whose
dotfunc switches to OpenBLAS's VECTOR kernels once a dot is long enough: ddot at 16 terms, sdot at 32 (below that,
float32 products are summed in a DOUBLE), zdotu at 8 (below that, four separate double sums). float16's HALF_dot is a
sequential float32 sum at every length. Without the backend, NumSharp reproduces every scalar regime byte for byte
(`DotSimd`, `SdotManaged`, `ZdotuManaged`) and sums the vector-regime positions in its own order, bounded-ULP. With
`NumSharp.Interop.OpenBLAS` those positions run the same `?dot` NumPy calls, byte-exact. That split is why the corpus
has two tiers: the generator records every `np.convolve` a case makes (`_PAConvRecorder`) and routes a case with a
BLAS-bound product to the host-pinned file. `BlasBackendDelta` replays the affected ordinary cases backend-on:
6,522 affected, 6,480 identical outcomes, 42 flips byte-checked against NumPy at delivery; 7,064 / 7,001 / 63 with the
wholeness pass's cases (convolve / correlate included).

**The long managed products** (`Math/NDArray.SlidingDot.Long.cs`) are blocked kernels:
- a block of B consecutive output positions shares ONE loop over the kernel;
- each step is one broadcast plus one load per vector, fused multiply-adds into eight independent chains;
- the core reads the data directly; the ≤ B − 1 head/tail steps read tiny zero-padded edge windows.

A padding lane adds 0·k[s], which is 0 only for a FINITE kernel, so a kernel holding inf/NaN declines to the
per-position kernels. The positions shorter than the vector block keep their exact scalar-regime sums. float16 gets an
exact blocked kernel (sequential per lane, separate multiply and add), so every float16 position is byte-exact.
Measured (1000×1000): polymul float64 2.26×, float32 2.94×, float16 82× (NumPy's HALF_dot is scalar);
complex128 1.45×.

**THE CEILING — complex128.** A complex product is four real multiply-adds, one 256-bit FMA per complex MAC on both
sides (zdotu's AVX2 microkernel does exactly that). The blocked kernel runs at ~90% of FMA peak, so it wins only
NumPy's per-position call overhead: 1.45–1.5× at 1000×1000.
- Gauss's three-multiply trick was rejected: it loses componentwise accuracy on real × complex imaginary parts.
- FFT convolution was rejected: its error is normwise, not per coefficient.

**The fused chebmulx kernel** (`DirectILKernelGenerator.PolyAlgebra.cs`, `GetPolyChebMulxKernel`). chebmulx is
NumPy's one mulx that is not a loop but three array statements (`tmp = c[1:]/2; prd[2:] = tmp; prd[0:-2] += tmp`),
so NumPy streams it. The calculus kernel (vector over COLUMNS, one division per row) ran 0.43× on a 10000-term float16
series, and the statements through house kernels ran 1.05× complex, 1.48× float64. Their combined effect,
`prd[j] = c[j-1]/2 + c[j+1]/2`, is one IL pass:
- the vector stage uses U3's lane kinds and halves by a MULTIPLY by 0.5 (x/2 ≡ x·0.5 for every binary float: the
  exact half, rounded once);
- float16 halves in float32, then `PolyLaneOps.HalfHalveOnGrid` rounds the half to the f16 grid (the magic-number
  trick: `0.75 + |y|` has float32 ulp 2^-24, the f16 subnormal step). Then a PLAIN float32 add, rounded once by the
  store (HALF_add);
- complex uses Smith's branch with the divisor 2+0j prepared once (`CDivPrep`/`CDivBy`: rat = +0, scl = 0.5);
- head, remainder, tail and decimal run the literal scalar ops;
- the kernel reads a contiguous same-dtype series in place (no as_series copy).

Now 5.5–6.5× NumPy at 10000 terms. Gates: `ChebMulx_FusedKernel_VectorStage_MatchesScalarEmission`,
`ChebMulx_Float16_EveryBitPattern_IsTheHalfLoop`, and corpus section L (special values through the vector stage,
every float16 pattern below 2^-12). Planted-bug check: dropping the grid rounding turns 2 cases red, a plain complex
multiply 5.

**One documented divergence** (`[Misaligned]`, unit-pinned, kept out of the corpus): fromroots sorts its roots, and
where +0.0 and −0.0 meet, NumPy's x86-simd-sort keeps an order that depends on the CPU. NumSharp's sort puts −0.0
first, which decides a zero coefficient's sign in the recurrence bases. (A second one — a complex product of
non-finite values without the backend — was zdotu's C99 result construction, now modelled: see the wholeness pass.)
A NaN root is not a divergence: the sort turns it into the positive quiet NaN and `-r` negates it, as in NumPy (np.sort
wrote .NET's negative NaN until 2026-10-05, which flipped every NaN coefficient's sign; plan §12, F5).

**The wholeness pass (2026-09-30).** A probe generator (every C# argument kind of the boundary map × every U2
function and module, pow / maxpower kinds, error-order pairs, non-finite complex products, np.convolve /
np.correlate: 6,675 NumPy-generated cases) replayed through the facades by reflection found three gaps, all closed:
- **pow / maxpower took only `int` / `double` / `int?`.** Every facade gained `{p}pow(object c, object pow)` (the
  default limit: 16, None for polypow) and `{p}pow(object c, object pow, object maxpower)` (null is None), ported in
  `Polynomial/Package/NDPolyPowerArgument.cs` from NumPy's three statements: `power = int(pow)` is CPython's int() —
  bool 0/1, np.float16 (Half) / decimal truncate, a str parses (`"3"`, `" 3 "`, `"3_0"`) and then fails
  `power != pow`, NaN / inf / complex / None / list / tuple raise CPython's texts, an ndarray of one or more dims
  `only 0-dimensional arrays can be converted to Python scalars` whatever its size, a 0-d one converts its element;
  `power > maxpower` is exact for Python numbers (NaN never exceeded, -inf always), decimal, char, BigInteger and
  integer ndarrays, but NEP 50 for np.float16 and float / complex / bool ndarrays: the weak int goes to a double first
  (`int too large to convert to float` at 2**1024), then to the dtype (2049 rounds to float16 2048 — not exceeded;
  float32 double-rounds), complex compares by NumPy's CGT, bool in int64 (`int too big to convert` past int64); an
  ndarray's result must then have exactly one element (NumPy's two truth-value texts), its conversion error first.
  C# binds `pow` to the int / double overloads first (a `long` / `ulong` converts to double), so only kinds they cannot
  take reach the object ones. The corpus replays them (`OpRegistry.PolyAlgebra.cs` `PolyPow` binds as C# source would:
  a spec-encoded limit or a non-int / non-double power the object overloads).
- **A None / str series was refused on conversion**, before NumPy's own errors. `polyutils.as_series` converts every
  argument, checks sizes and dims, and only THEN meets the common type: a str array (a list holding a str) fails it
  (`Coefficient arrays have no common type`), an object array (None, a non-numeric object, an oversized int) makes it
  compute with Python objects. So `PolySequence.ToArrayOrNonNumeric` now reports such a sequence as NumPy's array
  SHAPE (`PolyNonNumericArray`: dims, object-or-str, the refusal), `AsCoefficientArray` makes it a
  `PolySeriesView.NonNumericArray` (None and an unconvertible scalar are one-element object arrays), and the refusal
  is raised by `NDPolySeries.CommonType` — or later, where NumPy still has checks to run: div's `c2[-1] == 0` (on the
  object copies: `polydiv([None], [False])` is ZeroDivisionError) and pow's power checks. U3 / U4 compute at once and
  keep refusing on conversion (`v.Refusal`), so their error order is unchanged.
- **np.convolve's complex dot** (see the dotfunc paragraph under correlate/convolve): zdotu's C99 result and
  CDOUBLE_dot's plain loop for a one-element operand. That made complex products with infinities / NaNs below the
  vector block managed-exact: 43 host-tier cases moved to the portable tier (`_PAConvRecorder.blas_bound` no longer
  counts non-finite values), and np.convolve / np.correlate gained the same fix (the `groupa` tier's new complex
  section, one-element operands of every stride included).
What remains is inherent: an OBJECT array's computation (NumPy returns object arrays of Python ints / None, or raises
its object arithmetic's TypeError) is `NotSupportedException`, and a vector-regime product can overflow to inf in one
summation order and not the other (3 of 1,510 probe cases).
Corpus: section M of `gen_polyalgebra` (pow / maxpower kinds, the deferred refusal filtered to outcomes NumPy reaches
before any object arithmetic — `_pa_object_land` — and the non-finite complex products), section Q of
`gen_polyseries` (the deferred refusal in U1), the `groupa` complex convolve / correlate block.

Traps:
- **A float16 lane kind's generic `Bin` rounds every op to the f16 grid** (narrow + widen + NaN blends). Three of
  them per output made the first fused float16 pass 5× slower (3.7 ns/element). Round only where NumPy's float16
  array holds the value (the halves), and let the store's narrow be the add's rounding.
- **Emit halving as a multiply, not `Bin(Divide)`**: vdivpd bounds a streaming pass at one division per element. The
  equivalence is exact only for a power-of-two divisor.
- **A corpus of random operands never reaches a vector stage's rare branches.** The subnormal float16 halves appear
  in no random series of the older corpus: the grid-rounding mutant turned only section L's 2 cases red, so without
  them it was invisible. Build the edge set explicitly.
- **Slow SIMD with no calls is instruction count, not inlining.** `DOTNET_JitDisasm="NDPolyChebMulx_*"` +
  `DOTNET_JitStdOutFile` showed the slow float16 stage fully inlined (0 calls) — the cost was the grid round trips.
- **A one-element array's stride is observable.** NumPy's dot passes a size-1 operand through with whatever stride it
  has (it is C-contiguous regardless), and cblas refuses a non-positive one — so `polymul(c, [x])` and
  `polymul(c, [x, y])` reach different complex dot code. Materializing operands (`MaterializeForSliding`) loses
  that stride: decide `DotOperandBlasable` from the caller's array first.
- **A corpus case NumPy only reaches through object arithmetic cannot be gated.** Filter the generator's
  candidates by NumPy's outcome (`_pa_object_land`: an object-array result, or a TypeError other than int()'s /
  the limit comparison's / len()'s) rather than by hand — the filter keeps exactly the errors NumPy raises
  before it computes.

### Polynomial package — Vandermonde family (U5)
`{p}vander`, `{p}vander2d` and `{p}vander3d` for the six bases (18 names)

Plan `docs/plans/numpy-polynomial.md` (U5 delivered + 2026-09-30 wholeness pass). **Bit-exact with NumPy 2.4.2**:
oracle tier `polyvander.jsonl` (31,180 cases, 0 excused; 16,612 at delivery) and unit tests
`Polynomial/PolynomialVanderTests.cs` (16) + `PolynomialVanderArgumentKindsTests.cs` (8). **Perf (NPY/NS,
`benchmark/polynomial/polyvander_*`, 522 cells, every one SHA-256-checked): min 1.53×, geomean 10.22×.**

- **Driver** (`Polynomial/Package/NDPolyVander.cs`): NumPy's statements in NumPy's order, which is also its error
  order.
  - 1-D: `_as_int(deg)` → `deg must be non-negative` → `np.array(x, ndmin=1)` → `np.empty`'s dimension conversion
    (`Maximum allowed dimension exceeded` at ≥ 2^63 − 1 rows, AllocationGuard's `array is too big` below it,
    MemoryError past the allocator) → the recurrence → `np.moveaxis`.
  - 2-D / 3-D run `_vander_nd`'s order: `len(deg)` (`object of type 'int' has no len()`, `len() of unsized object`) →
    `Expected N dimensions of degrees, got K` → `np.asarray(points) + 0.0` (strong promotion; a ragged stack is the
    inhomogeneous ValueError) → per dimension its checks and allocation, the product's from the second dimension on →
    an empty stack's reshape error.
  - The result is always a VIEW (OWNDATA false): F-contiguous for 1-D points, C and F for a scalar / degree 0 / empty,
    neither for N-D points.
- **The degree text is `format(deg, '')`, not `str()`** (`PolyIndexArgument` in `NDPolyIndexArgument.cs`): a 0-d
  array formats through its scalar (`np.float16(1000)` → `1000.0`), an ndim ≥ 1 array through `array_str`, a list /
  tuple through its items' REPR (`[np.float16(1e+03), array([1.5]), (2,)]`). `operator.index` takes bool, every
  integer width, `BigInteger`, char and a 0-d integer array — not a 0-d bool.
- **Kernel** (`Backends/Kernels/Direct/DirectILKernelGenerator.PolyVander.cs`):
  - the six recurrences are DATA (`PolyVanderRoutines`) in U4's step language, run forward, NumPy's source
    expressions token for token (`legvander`'s `v[i-1]*x*(2i-1)` is `(v[i-1]*x)*(2i-1)`);
  - one whole-array kernel per (basis, dtype, 1/2/3-D, source dtypes, streamed), as four DynamicMethod stages:
    - LOAD, fusing the conversion with NumPy's `+ 0.0` (-0.0 → +0.0, a sNaN quieted with its payload kept);
    - RECURRENCE over a block of points;
    - PRODUCT, writing each output row `(a*(dy+1)+b)[*(dz+1)+c]` straight into the final layout (3-D through a
      scratch row holding NumPy's rounded `V_x[a]*V_y[b]` temporary);
    - ROOT, walking the blocks;
  - blocks are sized so every per-dimension matrix stays in L1/L2 (NumPy materializes three to five full
    temporaries per degree);
  - a 2-D / 3-D stack of same-shape NDArrays is read IN PLACE per coordinate (each loaded from its own dtype: the
    two-step `stack + 0.0` rounds at most once, the same value).
- **Streamed products for results ≥ 32 MiB** (`NDPolyVander.NonTemporalMinBytes`, `PolyVanderKey.NonTemporal`):
  - each product row runs a scalar head up to the store alignment, then aligned non-temporal vector stores
    (`vmovntdq`; 16-byte for float16), then the scalar tail; the stage ends with `sfence`; the bytes are identical;
  - a 96.8 MB result takes ~12 ms instead of 15–18 ms, and a reused 63.9 MB buffer 1.6 instead of 3.7 ms;
  - test hooks `NDPolyVander.NonTemporalOverride` (force on / off) and `BlockOverride` (tiny blocks).
- **The floor** is the six `(100000, 121)` 2-D cells, at 1.53–1.67×. A fresh result above the pool's 64 MiB cap costs
  ~0.44 µs of demand-zero fault per 4 KB page on both sides (~10.4 ms of the ~14 ms). The float16 cells (1.59–2.7×)
  are U4's open float16 lever.
- **Wholeness pass (2026-09-30).** A probe replayed every C# argument kind of the boundary map against NumPy through
  the facades: 15,366 cases (x / y / z / deg × every kind, cross-kind stacks, error-order pairs, the object stack).
  Three gaps, all closed:
  - **Scalar points that stack into an OBJECT array are computed.** A Python int past uint64 (`BigInteger` 2**70) among
    vander2d/3d scalar points makes NumPy's stack object, and its per-element `+ 0.0` then hands every dimension its
    OWN number: `float(int)`, correctly rounded, with CPython's OverflowError past the float range; a float / complex;
    NEP 50's dtype for an np.float16 or a 0-d array (`NDPolyVander.ObjectScalarPoints`, via `PolyNumber.Binary`). One
    dtype → the kernel path (`StackScalars`). Mixed dtypes → `MixedDtypeVander`, each dimension's {p}vander in its own
    dtype, multiplied through np.multiply's promotion. A None / str element stays refused. Oracle section J (14,568).
  - **A `char` degree container reads `numpy.uint16`** in NumPy's len() text (`NDPolySeries.PythonTypeName`; it said 'Char').
  - **`List<T>` coercion was ~40 ns an item** (a 1000-point `List<double>` x ran 0.70–0.97× NumPy). The shared coercion
    (`NDPolySequence`, every polynomial unit) now reads a typed collection — any non-array `IEnumerable<T>` of a dtype:
    `List<T>`, `HashSet<T>`, `Queue<T>`, LINQ — from its values in one pass (`TryTypedCollection` / `CollectionArray`),
    with NumPy's discovered dtype: a `ulong` mix of both magnitudes is float64, `List<float>` float64, an empty one float64
    with no dtype vote. A NESTED flat list (a typed collection, or an `object[]` of Python floats / ints) becomes one array
    leaf in the walk (`TryFlatLeaf` / `FlatLeaf`): the same two shape updates, the same values. 1000-point `List<double>`
    x is now 13.6–18.3× NumPy; nested 1000-point 2-D lists 7.7–11.7×.

  Everything else matched: typed arrays, `Memory<T>`, `object[,]`, jagged and `NDArray[]` points; `short` / `byte` /
  `char` degrees binding the int overload; 5-D/6-D transposed / reversed points; result writeability. Concurrency is
  clean: 576 kernel first-compiles under contention and 6,400 contended calls matched the single-threaded bytes.

Traps:
- **NumPy's float multiply keeps the SECOND operand's NaN** when both are NaN (MSVC `FLOAT/DOUBLE_multiply`, every
  loop length). x86 `mulpd` keeps the first, and RyuJIT may swap a commutative multiply's operands. The outer product
  is the one place two independently sourced NaNs meet. `PolyVanderOps.Mul*` blends explicitly, but only when the
  product holds a NaN lane: a NaN-free product had no NaN operand. A random corpus never pairs two NaNs; it took
  special values in BOTH coordinates.
- **A valid huge degree over an object x**: NumPy computes `x + 0.0` with Python objects BEFORE the dimension error,
  where NumSharp refuses the object stack. Only degree errors that come first are corpus cases.
- **NumPy hangs on a huge degree over EMPTY points whose byte count does not overflow** (its Python loop runs 2^50
  times over nothing). NumSharp returns at once; never put one in a generator.
- **A lone `object[]` argument to a `params object[]` parameter IS the params array** (`L(L(1.0, 2.0))` is `L(1.0, 2.0)`):
  a probe that spells nested Python lists that way flattens them. Spell nesting out with `new object[] { … }`.
- **`np.multiply` of two same-shape float64 arrays keeps the FIRST operand's NaN where NumPy keeps the second's** (the
  broadcast shapes happened to agree). This is library-wide, and the oracle compares NaNs as tokens, so it is invisible
  there. The vander kernel imposes NumPy's priority itself; only the mixed-dtype object-stack path goes through
  np.multiply.
- **A page-fault-bound benchmark cell measures the host.** Fresh >64 MiB results swing with the OS's zeroed-page list
  (the same streamed kernel measured 11.8–16.2 ms across best-of-9 runs), so the biggest cells run best-of-9 on
  both sides. A/B such a kernel change in ONE process,
  interleaving the configurations round by round.

### Polynomial package — companion matrices and roots (U7)
`{p}companion` and `{p}roots` for the six bases (12 names)

Plan `docs/plans/numpy-polynomial.md` (U7 delivered + 2026-10-01 wholeness pass). **Bit-exact with NumPy 2.4.2**:
oracle tier `polyroots.jsonl` (8,195 portable cases: every companion, plus the roots that never reach LAPACK; 6,681 at
delivery) and the host-pinned `polyroots_parity.jsonl` (1,354 roots that run geev; 1,242 at delivery — byte-exact with
`NumSharp.Interop.OpenBLAS` at threads=1, Inconclusive off the pinned host), 0 excused. Planted bugs: four at delivery
turned 26 / 695 / 18 / 116 cases red, the wholeness pass's ten turned 8–210 red each. Unit tests
`Polynomial/PolynomialRootsTests.cs` (18). **Perf (NPY/NS, `benchmark/polynomial/polyroots_*`, 229 cells, every one
checked): geomean 4.65×** — companion geomean 7.7× (2.2–20× up to degree 50, 4–9× at degree 500–1000), degree-1–10
roots 1.5–5×. The cells under 1.5× are floors both sides share: LAPACK `geev` for roots of degree ≥ 50 (~1.0×) and
~90 % of a degree-10 complex root (zgeev 20.7 of ~22 µs: ceiling 1.43×), zeroing the 320 KB degree-200 companion
(alloc+zero 4.1 µs vs NumPy's `np.zeros` 3.9 µs; 1.27–1.63× run to run), and a 67 MB companion's page faults (~1.0×).

- **Engine** (`Polynomial/Package/NDPolyAlgebra.Roots.cs`, on U2's arena) — NumPy's statements in NumPy's order:
  - as_series, the length checks, and for two terms the linear root's 1×1 matrix (scalarmath: `-c0/c1`,
    lag `1 + c0/c1`, herm `-.5*c0/c1`);
  - `np.zeros((n, n))`, then the diagonals through `mat.reshape(-1)[k::n+1]` (`GetDiagWriteKernel`; a stride-0
    source broadcasts a scalar), then ONE in-place update of the last column.

  Every element operation runs through the house kernel NumPy's statement runs through: the binary ufunc loops, the
  conversions, np.sqrt's unary loop, np.multiply.accumulate's scan (herm/herme's `scl`), and np.arange through a new
  int64 ramp IL kernel (`DirectILKernelGenerator.PolyRoots.cs`). Roots = `np.linalg.eigvals` of the companion — ROTATED
  for every basis but the power series (`np.flip` over both axes, NumPy's `[::-1, ::-1]` view without a slice string)
  — sorted in place. Real roots are a VIEW of eigvals' complex result (OWNDATA false, stride 16 bytes), as NumPy's.
- **The dtype rule decides the bits.** The helper vectors (`scl`, `top`, `mid`) are FLOAT64, so wherever one meets the
  series the loop is `result_type(c.dtype, float64)`. A float16 / float32 series' cheb / leg / herm / herme last
  column is therefore computed in float64 and rounded ONCE into the matrix; poly / lag stay in the series dtype.
  This was verified over 9,600 random / special series with an explicit model of the statements before the code was
  written.
- **The zero matrix is a write-once allocation.** `new NDArray(fillZeros: true)` (calloc) handed out fresh OS pages,
  and the diagonal writes faulted each one: ~0.9 µs a page, 75 µs at degree 200. It now follows np.tri's policy
  (`np.PrefersWriteOnce`): up to 64 MiB a pooled buffer plus `np.ZeroBytes`; above that, OS-zeroed pages.
- **Library-wide fixes that rode along:**
  - **`np.linalg.eig`/`eigvals` of a float32 operand with complex eigenvalues** returned double-precision values.
    NumPy returns complex64 (geev in double, then `astype(complex64)`). `CollapseEig` now rounds each component to
    float32, keeping complex128 (#569): one cast and copy through a float64 view of a contiguous result, the per-lane
    form otherwise. Before the fix 113 of 3,000 float32 probe roots differed, and the live
    `EigLiveParityTests` matrix (1±1j) was float32-exact, so it could not see the gap; it now has ±i√2 too.
  - **`AssertFinite`** (eig/eigvals) uses the fused `FiniteScan.IsAllFinite` kernel instead of `np.all(np.isfinite)`.
    It is the same predicate; the bool temp cost 0.45 of eigvals' 0.9 µs wrapper on a 10×10.
- **Object series.** A Python int past uint64, None or another object among the terms makes as_series go on in the
  OBJECT dtype. The items keep their kinds (Python numbers, NumPy scalars, 0-d arrays):
  - **The length check** runs on the TRIMMED object array. `ObjectTrimLength` scans Python's `item != 0` from the end.
    A 0-d array item compares elementwise, so a trailing 0-d zero is trimmed too. `[None, 0]` and `[2**70, 0]` raise
    NumPy's ValueError.
  - **Two terms** are the linear root computed on the ITEMS (`ObjectLinearRoot`). The arithmetic is CPython for
    Python numbers, scalarmath for NumPy scalars and ufuncs for 0-d arrays, and the result is a numeric array of that
    number's dtype.
  - **Three or more**: NumPy builds an object companion matrix. `ObjectCompanionArithmetic` replays the element
    operations of its last-column statement that can raise, and raises NumPy's first error. That error is always a
    Python int too large for a float:
    - `integer division result too large for a float` for Python int / Python int;
    - `int too large to convert to float` for any other operand.

    The divisor is cast to the object dtype first, so a 0-d integer divisor is a Python int there. herm/herme
    multiply by their float64 helper first. When nothing raises, `{p}companion` refuses (an object matrix), and
    `{p}roots` raises eigvals' `ufunc 'isfinite' not supported…` TypeError before LAPACK, so no backend is needed.
  - **Remaining divergences** (NotSupportedException):
    - a one-term object series' roots (NumPy's `np.array([], dtype=object)`, `[Misaligned]`);
    - the object companion matrix itself;
    - CPython's TypeErrors from None/str arithmetic (the U2 policy).
- **Wholeness pass (2026-10-01).** A 3,096-case probe replayed against NumPy, with 2,982 exact, 0 wrong, and 114 in the
  object-dtype classes above. It covered:
  - every C# argument kind × both functions × six bases;
  - long series up to 1,100 terms, Hermite's `scl` underflow and float16 rounding past 2048;
  - edge numeric series: non-finite, extreme, subnormal, int64/uint64 extremes.

  It found the object-series gaps above, plus one LIBRARY-WIDE use-after-free. Byte-reinterpreting aliases did not keep
  their owner's buffer alive (see "Critical: View Semantics"), so eigvals' real result, and with it every real
  `{p}roots` result, could be overwritten by the next call. Concurrency was checked at 16 threads: 366 first calls and
  6,400 steady-state calls against the single-threaded bytes, 0 differences after the fix (NumPy: 0).

  Perf of the new object value path: the shared coercion (`NDPolySequence`, `NDPolySeries.AsCoefficientArray`)
  classified a Python int past uint64 by THROWING and catching NotSupportedException, ~1.4 µs a leaf. Two-term object
  roots ran at 1.13–1.66× NumPy. `PolyNumber.TryDiscoveredDtype` + `ObjectArrayRefusal` (the refusal built, never
  thrown) bring them to 2.9–4.6×, which speeds every unit's object-series path. New corpus sections:
  - K: two-term object series and edge numeric series, 1,142 portable + 112 host cases;
  - L: object series of three or more terms, 372 cases.
- **Errors in NumPy's order.** as_series' texts come first, then `Series must have maximum degree of at least 1.`.
  Roots of degree ≥ 2 then meet eigvals' checks in eigvals' order: finiteness (LinAlgError) before dtype (the float16
  / decimal TypeError). All of these fire before LAPACK, so they need no backend; with no backend, a root of
  degree ≥ 2 raises MissingBackendException.

Traps:
- **Decide the host split by recording, not predicting.** The generator replaces `np.linalg.eigvals` per case: the
  polynomial modules look it up at call time. geev ran exactly when eigvals got a finite float32 / float64 /
  complex128 matrix.
- **NumPy's SIMD sort orders value-tied ±0 elements by CPU.** The generator skips such results (none appeared in
  7,923 cases).
- **Zeroing is a floor, not overhead.** A hot 312 KB buffer clears at ~86 GB/s whatever the method: `Span.Clear`,
  `InitBlock` and an AVX2 loop all take 3.6 µs, and non-temporal stores 5.9 µs. A fused row-by-row build was slower.
- **Never put a `?:` with an `object[]` arm and an `NDArray` arm in a benchmark.** It is typed NDArray through the
  implicit Array→NDArray conversion. The runner crashed on the list cells until both arms were cast to `object`.
- **cmd's `%1` splits on commas.** `bench.cmd C/poly/,T/poly/` passes only `C/poly/`.

### Random (`np.random.*`)
`bernoulli`, `beta`, `binomial`, `chisquare`, `choice`, `dirichlet`, `exponential`, `f`, `gamma`, `geometric`, `get_bit_generator`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `logseries`, `multinomial`, `multivariate_normal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `permutation`, `poisson`, `power`, `rand`, `randint`, `randn`, `random_sample`, `rayleigh`, `seed`, `set_bit_generator`, `shuffle`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`, `uniform`, `vonmises`, `wald`, `weibull`, `zipf`

**Integer types are NumPy's (the 2026-09-27 type-parity audit; gate `RandomSampling/RandomTypeParity.Test.cs`).** The
legacy integers are C `long` in NumPy, and NumSharp models ONE width for all of them — the LP64 (Linux/macOS) int64,
`NumPyRandom.LegacyLong`: `randint`'s default dtype, `random_integers`, `permutation(n)`, `choice`'s indices,
`multinomial`'s counts and the discrete samplers, with `long` counts in. Windows NumPy returns the same VALUES as int32
wherever it accepts the input (it rejects bounds past `2**31`), which is why the Windows-authored `random_parity` /
`generator_parity` corpora record every legacy int WIDENED to int64 (`gen_oracle.py` `_RND_INT64_CAST`, the
`random_integers` branch). Where NumPy's own messages differ by platform (`Python int too large to convert to C long`
vs `int too big to convert`, `value too large to convert to uint32_t` vs `…C unsigned long`) the Linux text is pinned.
`SeedSequence` carries NumPy's member types (`spawn_key` `BigInteger[]`, `pool_size` `long`, `n_children_spawned`
`uint`, `pool`/`generate_state(long n_words, …)` NDArrays) and NumPy's string rule (`"0x…"` hex, a leading digit
DECIMAL — no octal: `"012"` is 12). `Seed` is a `uint`. After `set_bit_generator` swaps the singleton to a non-MT19937
engine, `np.random.seed(x)` re-seeds it as NumPy's MODULE function does (`engine.state = type(engine)(x).state`, cached
Gaussian kept) — only for `np.random` itself; another `RandomState` over such an engine refuses (the method rule).

**NumPy's call spellings port verbatim (the 2026-09-27 wholeness pass; gate `RandomSampling/RandomApiWholeness.Test.cs`,
plan `docs/plans/random-oracle-coverage.md` §10 items 15–20).** The random-API oracle binds overloads by reflection, so
call shapes are pinned at COMPILE time instead — 1,793 NumPy spellings were compiled and compared with NumPy 2.4.2 to
find these: (1) the nine all-default legacy samplers (`normal`, `uniform`, `exponential`, `poisson`, `gumbel`, `laplace`,
`logistic`, `lognormal`, `rayleigh`) take `size:` alone or any keyword subset — the size overload carries NumPy's
defaults and the one-draw overload's parameters are REQUIRED, so the two never compete; (2) numpy.random's classes through
the module — `np.random.PCG64(42)`, `np.random.SeedSequence(...)`, `np.random.Generator(bit_generator)`: 50 `NumPyRandom`
factories mirroring every constructor overload of the five engines, `SeedSequence` and `Generator`
(`np.random.classes.cs`; no `BitGenerator`, which NumPy refuses to instantiate) — **TRAP: inside `NumPyRandom` the
factories SHADOW the class names in expressions and crefs; spell `global::NumSharp.X.Member` / `cref="NumSharp.X"`**;
(3) a bare `null` seed binds (`[OverloadResolutionPriority(1)]` on the overload that takes it, polyfilled for net8.0 in
`Assembly/OverloadResolutionPriorityAttribute.cs` and honored across assemblies) and a typed null seed ARRAY is NumPy's
`None` too — fresh entropy; it used to seed `SeedSequence([])`, the same stream every run, invisible to the oracle's
entropy masks; (4) `size: default` is NumPy's `size=None` (`Shape` is a struct: `size: null` cannot compile) and binds
the `Shape` overload — the `int[]`/`long[]`/`long` size shims carry priority -1, without which
`multivariate_normal(mean, cov, size: default)` bound the `long` shim (size 0, an empty result); (5) `integers` and
`randint` take NumPy's ARRAY bounds (`BoundedIntegers.Broadcast.cs`, the `_rand_<dtype>` broadcast path: `can_cast`-
skipped bound checks in NumPy's order and words, buffered words carried across positions, the size-smaller-than-
broadcast and non-C-layout-float-bounds quirks) and `BigInteger` (Python-int) bounds —
`g.integers(0, BigInteger.Pow(2, 64), dtype: np.uint64)` is the full-range idiom. **Perf of the array path (NPY/NS,
P-cores `0xFFFF` pinned, best-of-21, 1K/100K/1M):** float bounds 5.4–6.2×, column-broadcast 2.7–2.9×, int64 1.0–2.1×
(1K is fixed cost), int32 1.25–1.9×, uint8 1.1–1.55× — faster than NumPy everywhere, but the narrow widths sit UNDER
the 1.5× bar. Cause, measured: ONE per-position loop serves every width, and PGO inlines each width's path plus the
devirtualized engine step into it (248-byte frame, `buf`/`bcnt` spilled) — pulling the loop into its own method or
force-inlining the sub-word helpers moved cost BETWEEN widths (uint8 up, int64/int32 down), never removed it.
Per-width chunk loops (the scalar `Fill` → `FillUInt8/16/32/64/Bool` shape, NumPy's own per-dtype
`_rand_<dtype>_broadcast`) are the known fix, deferred because the np-function rule forbids per-dtype functions. The
uint8 Lemire threshold `(255 - rng) % (rng + 1)` is a 256-entry table (`BoundedIntegers.LemireThreshold8`): the
division ran on 78% of draws at rng 199 — the pre-existing scalar uint8 fill went 1.00× → 1.21×, same stream.

**The KIND of array a random function returns is NumPy's (2026-10-05; gate `RandomSampling/RandomResultOwnership.Test.cs`,
plan `docs/plans/view-shape-open-bugs.md` bug 3).** Every sampler of both APIs returns a fresh writeable C-ordered array
that owns its data — `choice` without replacement and `multivariate_normal` included: NumPy ends them with an in-place
`idx.shape = shape` / `x.shape = final_shape`, so they allocate the final shape (`shape.AsClean()`) or assign
`result.shape`, never `reshape` (a view). NumPy's own two exceptions are kept: the legacy `RandomState.choice(int, size,
replace=False)` without `p` is a slice of `permutation(pop_size)` (a view), and `Generator.permutation(x, axis > 0)` is
`arr[tuple(slices)]`, NumPy's fancy-index layout — a VIEW of a fresh buffer whose first axis is the permuted one
(MapIter's buffer transposed back: F-contiguous for 2-D, permuted strides beyond). `np.FancyIndexAlongAxis`
(`Indexing/np.take.FancyAxis.cs`) builds it with ONE tiled IL gather (`DirectILKernelGenerator.TakeLeading.cs`: tiles of
~1 MB of source rows, each index writing one contiguous run, typed MOVs 4× unrolled or `cpblk`) — 2.2–6.2× NumPy over
11 shapes, but 1.19× for a 1-byte last axis (NumPy's strided byte loop is ~2 cycles/element). For a non-C source of
three or more dims NumPy also keeps the SOURCE's memory order for the other axes, which `np.FancyIndexAlongAxis` (it
copies the source to C first) does not yet. NumSharp's general fancy index does NOT follow NumPy's layout rules yet
(`a[:, idx]` is C-ordered and owning) — a library-wide indexing gap; both are pinned in `OpenBugs.FancyIndex.cs`.

### File I/O
`fromfile`, `fromstring`, `load`, `load_npy`, `load_npz`, `loadtxt`, `save`, `savetxt`, `savez`, `savez_compressed`, `tofile`

The `.npy`/`.npz` stack is a **port of NumPy 2.4.2's `numpy/lib/_format_impl.py`** (NEP-01) in
`IO/{NpyFormat,NpzFile,PyLiteral}.cs`, and the writer is **byte-for-byte identical to NumPy's own
`np.save`** — not merely readable by it. Format versions **1.0 / 2.0 / 3.0** (2- vs 4-byte header
length; latin-1 vs UTF-8), **64-byte `ARRAY_ALIGN`** data alignment (so the data is mmap-ready),
**`fortran_order`** both ways (write via the transpose; read via reshape-reversed + transpose), and
**big-endian** files byte-swapped to native on read (per-component: `>c16` swaps in 8s, `>U1` in 4s).
Headers parse through a real Python-literal parser (`PyLiteral`, standing in for `ast.literal_eval`)
rather than `IndexOf`/regex, so key order, whitespace, quoting, trailing commas and the Python 2 `3L`
suffix all work; `max_header_size` (default 10000) bounds hostile input and `allow_pickle` lifts it.

`np.load` returns `object` — `NDArray` for `.npy`, `NpzFile` for `.npz` — mirroring NumPy's
content-dependent return; **`np.load_npy` / `np.load_npz` are the typed forms** and need no cast.
`NpzFile` is lazy + cached, takes `"w"` or `"w.npy"` as keys, exposes `.Files` stripped, supports
`npz.f.weights` dot access (NumPy's `BagObj`) and **must be disposed**.

| Dtype | Maps to | Notes |
|-------|---------|-------|
| Boolean/SByte/Byte | `\|b1` / `\|i1` / `\|u1` | single-byte types take the `\|` prefix, never `<` |
| Int16…UInt64, Half, Single, Double | `<i2`…`<u8`, `<f2`, `<f4`, `<f8` | direct |
| Complex | `<c16` | `<c8` widens to `Complex` on read |
| Char | `<U1` | 2-byte UTF-16 ↔ 4-byte UCS-4; non-BMP rejected (needs a surrogate pair) |
| Decimal | — | `NotSupportedException`: no NumPy dtype (cast to Double) |

Object arrays, structured/subarray dtypes, `datetime64`/`timedelta64`, `\|S`/`<U`n>1/`\|V`, `<f16`
and `<c32` are parsed then rejected with a precise message — including the zero-width `<U0`/`\|S0`/`\|V0`,
which are *valid* NumPy dtypes and so report "unsupported", never "invalid descriptor".

**`mmap_mode` is implemented** (`NpyFormat.OpenMemmap`, port of NumPy's `open_memmap` + `numpy.memmap`):
`np.load(path, mmap_mode=…)` returns an `NDArray` backed by a `MemoryMappedFile` view — zero-copy,
the mapping released with the array (and its views) on dispose/GC. Modes match what NumPy actually
does *through `np.load`*: **`r`/`readonly`** → read-only (non-writeable) view, **`r+`** → read-write
(flushes to disk), **`c`** → copy-on-write (not flushed). The other four documented spellings validate
but fail downstream in NumPy too, reproduced verbatim: **`w+`** → `TypeError "object of type 'NoneType'
has no len()"`, **`write`/`copyonwrite`/`readwrite`** → `ValueError "invalid mode: '<mode>b'"` (they
contain the letter *w*, so `open_memmap` misroutes them into its create branch — NumSharp does **not**
truncate the file as NumPy's `w+` does). Validation happens **only on the `.npy` branch**: an `.npz`
ignores `mmap_mode` entirely (even an invalid value → `NpzFile`), matching NumPy. mmap needs a real
**path** — a stream/`byte[]` npy raises NumPy's `ValueError "…Memmap cannot use existing file handles"`.
NumSharp-specific rejections (`NotSupportedException`): big-endian and `<U1`/`<c8` files, which NumSharp
byte-swaps / widens on read and so cannot present as a zero-copy view. Design: `docs/MMAP_MODE_HANDOVER.md`;
tests: `IO/NpyMemmapTests.cs` (16).

**Known intentional divergences** (differential-verified, everything else agrees): the 5 unsupported
dtypes above are the *only* files NumPy loads that NumSharp refuses. NumPy reports `shape: (True,)`
and overflowing dims via a later TypeError/OverflowError while NumSharp rejects them up front — both
refuse, the text differs. Big-endian files byte-swap to native (NumPy keeps a byte-swapped dtype),
and `Char` round-trips U+0000 as a NUL where NumPy's `<U` reports `''` (the bytes are identical —
`<U` treats NUL as string padding).

**Three traps this port has already fallen into — do not re-break:**
- **`(1)` is not a tuple.** Only a comma makes one in Python, so `'shape': (1)` is the *int* 1 and
  NumPy rejects it. `PyLiteral.ParseTuple` tracks `sawComma` for exactly this.
- **A header-length field is attacker-controlled.** `ReadBytes` sizes its buffer from what the stream
  actually holds (seekable ⇒ exact) and grows as bytes arrive — never from `count`. A 28-byte file
  can claim a 4 GB header, and `new byte[claim]` turns that into a gigabyte spike; NumPy shrugs it
  off in ~2 KB because `fp.read(n)` allocates only what it returns. Pinned by
  `HostileHeaderLength_DoesNotAllocateTheClaim` (measures ~1.2 KB).
- **`NpzFile`'s name map needs NumPy's TWO passes**, not one interleaved loop: `dict(zip(stripped,
  full))` then `.update(zip(full, full))`. For an archive holding both `a` and `a.npy` — which both
  strip to key `a` — pass 2 re-points `a` at the entry literally *named* `a`; one loop resolves it to
  whichever came last. Within a pass, later entries win (a zip may repeat a name, and Python's dict
  keeps the last; .NET's `GetEntry` would return the first). Pinned by the `npz_*_names` cases.
- **Never buffer an npz member to sniff its magic.** `MemoryStream`/`byte[]` cap at 2 GB, so buffering
  made NumSharp *write* 5 GiB archives it then failed to *read* ("Stream was too long") while NumPy
  read them fine — and it doubled the cost of every ordinary load. `NpyFormat.ReadArray` is strictly
  forward-only, so `StartsWithNpyMagic` sniffs on one `entry.Open()` and the reader streams from a
  fresh one. Pinned by `Npz_StreamsMembers_RatherThanBufferingThem`.

**Size limits (measured, not assumed):** `.npy` and `.npz` both round-trip **> 4 GiB** — verified at
5 GiB against every 32-bit boundary (`int.MaxValue`, `uint.MaxValue`) with NumPy reading the result,
and the 5 GiB header is byte-identical to NumPy's. Array data lives in unmanaged memory and both IO
paths stream in 256 KB chunks, so no managed buffer ever scales with the array. Zip64 comes from
.NET's `ZipArchive` on demand (NumPy forces `force_zip64=True` for the same reason, gh-10776). The
only 2 GB ceilings left are the inherently `byte[]`-returning conveniences — `np.save(arr)`,
`np.savez(...)` → `byte[]`, and `NpzFile.GetRawBytes` — which is a `byte[]` limit, not a format one;
use the path/stream overloads above 2 GB. Tests: `NpyLargeFileTests` (`[HighMemory][LongIndexing]`,
excluded from CI; needs ~8 GB RAM + ~6 GB disk).

**Gates:** `test/oracle/gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip` (286 committed cases of real
NumPy output) replayed by `NpyOracleTests` under the **`NpyOracle`** category — read / byte-exact
write / header-only write / verbatim error / round-trip / npz / live-view write / hostile-allocation.
Reverse interop (NumPy reading NumSharp, the direction byte-equality can't cover for `.npz`) is the
manual gate `python test/oracle/verify_npy_interop.py`.

**`np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ',
encoding=None)`** is a port of NumPy 2.4.2's `_npyio_impl.py::savetxt` (`APIs/np.savetxt.cs`), **byte-for-byte
identical to NumPy's own output**. A 1-D array writes one value per line (NumPy's `atleast_2d(X).T`, `ncol=1`);
a 2-D array writes a row per line; 0-D / ≥3-D raise `ValueError("Expected 1D or 2D array, got {n}D array
instead")`. Each row is `format % tuple(row)` where `format` is the single spec joined `ncol` times by
`delimiter`, a caller-supplied multi-`%` string (delimiter then ignored), or the list/tuple of specs joined —
resolved with NumPy's verbatim errors (`ValueError "fmt has wrong number of % formats:  …"`, the new house
`AttributeError "fmt has wrong shape.  …"`). The `%`-engine is the **existing `PrintfFormatter`** shared with
`ndarray.tofile` (Python `%`-operator: flags `-+ 0#`, width, `.precision`, `d i u x X o e E f F g G s c %%`,
2-digit-min exponent, `nan`/`inf`, `-0.0`); the multi-column `format % tuple(row)` path added
`PrintfFormatter.FormatRow`/`FormatRowInto` (sequential-arg, appends straight into the batch buffer). Numeric
specs **widen to double** (matching NumPy's `float()`/`int()` coercion, so `int64` past 2^53 and `float32`
render at the double value) while `%s` uses each dtype's shortest scalar repr. On an INTEGER conversion
(`d i u x X o`) a `.precision` is Python's **minimum-digit** count, left-zero-filled (`%.5d`→`00042`,
`%.0d % 0`→`0`), the `0` flag composes with it (`%08.3d % 7`→`00000007`, Python-not-C), and with `#` the
zero-fill lands BETWEEN the `0x`/`0o` prefix and the digits (`%#08x`→`0x0000ff`). `%c` renders a true integer's
code point in `[0, 0x10FFFF]` and **raises** `OverflowError("%c arg not in range(0x110000)")` outside it; `%c`
on a **bool** and `%x`/`%X`/`%o` on a bool are the `TypeError`→dtype/format mismatch (a bool is not an int for
those conversions — probed), while `%d`/`%i`/`%u` accept a bool (→1/0). **Complex** wraps a single spec
as `' (%s+%sj)'` per column and applies the `+-`→`-` fix-up, exactly as NumPy. Overloads target a **filename**
(`.gz` → gzip; Python text-mode `\n`→`os.linesep` translation, so a file matches NumPy on the same platform —
CRLF on Windows), an open **`Stream`**, or a **`TextWriter`** (both verbatim `\n`, NumPy's file-handle path).
Reads any layout (C/F/strided/transposed/negative-stride/broadcast/sliced) in logical C-order via
`GetAtIndex`; all 15 dtypes (Char/Decimal have no NumPy analog but render via the same coercion). **Perf
(NPY/NS):** file-to-file **~2.1×**, `%d`/`%g` **3–25×** (NumPy's `savetxt` is a pure-Python per-row loop).
**Two deliberate divergences** (`[Misaligned]`): `%d`/`%i`/`%u` on **inf** raises `System.OverflowException`
where NumPy raises `OverflowError` (message identical; NumSharp has no `OverflowError` type — nan still raises
`ValueError` verbatim), and the AttributeError renders a C# `string[]` as a Python list repr. Gate:
`IO/SaveTxtTests.cs` (probed against NumPy 2.4.2). See `APIs/np.savetxt.cs`,
`Backends/Printing/PrintfFormatter.cs`.

**`np.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None,
unpack=False, ndmin=0, encoding=None, max_rows=None, quotechar=None)`** is the reader complement — a port of NumPy
2.4.2's `_npyio_impl.py::loadtxt` → `_read` plus its C text reader (`textreading/{tokenize.cpp,conversions.c,
str_to_int.c,rows.c}`), in `APIs/np.loadtxt.cs`. Round-trips `np.savetxt` (byte-exact values). The C reader always
yields a 2-D `(rows, cols)` array; **`ndmin` then squeezes size-1 axes** (so a single row/col → 1-D, a lone value →
0-d) or expands. The **tokenizer** is a faithful port: `delimiter=None` splits on **runs of whitespace** (leading/
trailing stripped, blank/comment-only lines → 0 fields → skipped) while an explicit single-char delimiter (incl. `' '`)
keeps empty fields; `quotechar` protects delimiters/comments and doubles as an escape; a single-char `comment` strips
to line-end (multi-char comment stripped per line, and rejected with `quotechar`). Whitespace is NumPy's
`Py_UNICODE_ISSPACE`, which is `char.IsWhiteSpace` PLUS the four C0 information separators **U+001C..U+001F** (the
only BMP difference, exhaustively verified) — so `"1\x1c2"` is two fields and a leading/trailing separator is
stripped by the field parsers (`IsLoadtxtWhitespace`/`TrimLoadtxtWs`, used by both tokenizer and numeric parsers).
The **per-dtype parsers** match NumPy exactly: **bool via int64** (`"0"`/`"1"`, `-1`→True, `"True"`→error), integers
**range-check** (`int8 "200"`→ error) and **unsigned reject negatives** (`uint8 "-1"`→error), **float** uses
`PyOS_string_to_double` semantics (case-insensitive `inf`/`infinity`/`nan`, rejects hex/`_`/trailing-junk/empty,
NaN→`+qNaN 0x7FF8…`), **complex** is `to_complex_int` with the imaginary unit fixed to **lowercase `j` only**
(`a`/`aj`/`a+bj`/`(a+bj)`/`1+-2j`; `"1+2J"`→error, matching NumPy's default `imaginary_unit='j'`). All 15 dtypes
(Char/Decimal via the same coercion). `converters`
(a `Func<string,object>` for all columns, or an `IDictionary<int,Func<string,object>>` per column), `usecols`
(int[], negatives from the end), `unpack` (transpose), `skiprows`/`max_rows` (blank/comment lines don't count toward
`max_rows`) and `encoding` are supported. Inputs: **filename** (`.gz` transparently decompressed; universal
newlines), open **`Stream`**, **`TextReader`**, or **`IEnumerable<string>`** (lines, embedded newlines split). Errors
are NumPy-verbatim: `ValueError("the number of columns changed from {a} to {b} at row {n}; …")`, `ValueError("could
not convert string '{tok}' to {dtype} at row {r}, column {c}.")` (**row 0-based, column 1-based**), `ValueError(
"invalid column index {k} at row {n} with {c} columns")`, `ValueError("Illegal value of ndmin keyword: …")`,
`TypeError` for a multi-char/newline delimiter. Fields are parsed straight from the line's **char span** (no per-field
string allocation on the common path). **Perf (NPY/NS):** integer/bool/discrete dtypes **1.5–1.9×**; **float-heavy
input ~1.0–1.1× (parity)** — bounded by the correctly-rounded `double.Parse`, the analog of NumPy's `strtod` (the
same floor on both sides, as with savetxt's `%.18e`). **Deliberate divergence:** the empty-input `UserWarning` is not
emitted (NumSharp has no warnings module; the array is still the correct empty result). Gate: `IO/LoadTxtTests.cs`
(probed against NumPy 2.4.2). Reuses the private `BytesToArray` helper from `np.fromfile.cs` (whose own parsers
truncate rather than range-check, so loadtxt carries its own strict ones). See `APIs/np.loadtxt.cs`.

**`np.fromstring(string, dtype=float, count=-1, sep)`** (`APIs/np.fromstring.cs`) parses a 1-D array from the
numbers in a text string — the SAME item reader `np.fromfile` uses for a text file, so it shares fromfile's
`SplitTokens`/`SelectTokens`/`TokensToArray` (NumPy shares its `fromstr` the same way): a whitespace-only `sep`
splits on any whitespace run, a non-whitespace `sep`'s spaces are a wildcard, one trailing separator is tolerated,
and a bad/missing item raises the verbatim `ValueError("string or file could not be read to its end due to
unmatched data")`. `count` caps the items (all 15 dtypes; default float64). The **binary mode was REMOVED in
NumPy 1.22**, so an empty/`null` `sep` raises `ValueError("The binary mode of fromstring is removed, use frombuffer
instead")` rather than reinterpreting the string's bytes (probed 2.4.2). Because it shares `fromfile`'s parser, it
inherits that parser's **deliberate lenient-superset traits** (differentially mapped vs NumPy 2.4.2): the imaginary
unit is lowercase `j` ONLY (`"1+2J"` rejected — fixed to match NumPy, was accepted), the parenthesised `"(1+2j)"`
form is accepted (a superset that round-trips `tofile`; NumPy rejects it), and integer over/underflow **wraps
deterministically** where NumPy's C parser is platform-specific — NumPy RAISES on a narrow-unsigned negative
(`uint8 "-1"`, though `uint64 "-1"` wraps) and CLAMPS overflow via `strtol` (Windows 32-bit `long`), both portably
avoided here (`loadtxt` carries strict range-checking parsers for full parity instead). Gate: `IO/FromStringTests.cs`.
See `APIs/np.fromstring.cs`. **Not implemented — `genfromtxt`/`fromregex`:** both require NumPy's structured/record
dtypes (and `genfromtxt` its masked-array + missing-value machinery), which NumSharp has no analog for; their
non-structured subset would only re-expose `loadtxt`.

### Printing / Formatting
`array2string`, `array_repr`, `array_str`, `format_float_positional`, `format_float_scientific`, `get_printoptions`, `printoptions` (IDisposable context), `set_printoptions`

`NDArray.ToString()` is a **byte-exact port of NumPy 2.4.2's array printing** (`numpy/_core/arrayprint.py` + `dragon4.c`): `ToString()` / `ToString(false)` → `np.array_str` (`[0 1 2]`, the `str()` form), `ToString(true)` → `np.array_repr` (`array([0, 1, 2], dtype=…)`, the `repr()` form). Covers decimal-point float alignment, the maxprec/unique/fixed floatmodes, exp-format cutoffs (per-dtype, native-precision ratio), nan/inf fields, complex, summarization at `threshold` (with `…` and edgeitems), line wrapping at `linewidth`, the 0-d `str`-vs-`repr` asymmetry (`5.0` vs `5.`), and repr dtype/shape suffixes. Float digit generation leans on .NET's shortest-round-trip `ToString("R")` (== Dragon4 unique) but routes **all rounding** through `ToString("F"|"E"+precision)` (rounds the true binary value, IEEE half-to-even) — never the shortest string (the latter diverges ~50 % on adversarial ties). NumSharp's `Char` dtype uses string rendering (no NumPy equivalent). Validated against NumPy 2.4.2 across ~18 000 fuzz cases.

### Other
`around`, `asscalar`, `copyto`, `round_`, `size`, `multithreading` (NumSharp extension — `np.multithreading(enabled, max_threads)` opt-in threaded kernels)

**`np.copyto(dst, src, casting='same_kind', where=True)` is NumPy's `array_copyto` → `PyArray_AssignArray` /
`PyArray_AssignRawScalar`, statement for statement** (`Manipulation/np.copyto.cs`; 2026-10-06, plan
`docs/plans/api-parameter-parity.md` X16 + X11; probed against 2.4.2):
- **Arguments.** The casting name first (`ValueError` "casting must be one of …", `TypeError` for null), then the mask's
  dtype (NumPy's "Cannot cast array data|scalar from dtype('int64') to dtype('bool') according to the rule 'safe'" under
  the house `ArgumentException`). A C# integer / `float` / `double` / `Complex` binds `copyto<T>` and is NumPy's WEAK
  Python scalar (`npy_find_descr_for_scalar`: the common DType of `_PyLongDType` / `_PyFloatDType` / `_PyComplexDType`
  and the destination's; stored with `PyArray_Pack`'s range checks — `np.copyto(int8, 1000)` raises "Python integer
  1000 out of bounds for int8", never wraps; `'equiv'` refuses any dtype but the scalar's own default). A `BigInteger`
  is a Python int of any size. `bool` / `Half` / `char` / `decimal` and every `NDArray` (0-d too) are STRONG: a 0-d
  int64 into int8 wraps, as `np.array(1000)` does.
- **Checks in NumPy's order:** 0-d source → AssignRawScalar ("Cannot cast scalar from …"); a redundant self-copy (same
  data, dtype, shape, strides) returns BEFORE the writeable check (`np.copyto(b, b)` on a read-only view is legal); read-only
  → "assignment destination is read-only"; the cast rule (`InvalidCastException`, NumPy's text); an overlapping source is
  copied first unless dst is 1-D with both runs in the same direction; the source broadcast with its leading extent-1
  axes beyond dst's rank STRIPPED ("could not broadcast input array from shape (2,3) into shape (3,)"); a 0-d mask
  (after the source check: false copies nothing); the mask broadcast, never stripped ("could not broadcast where mask
  from shape (1,3) into shape (3,)").
- **The masked copy** is a BIT copy of dst's element width (`DirectILKernelGenerator.MaskedAssign.cs`: one kernel per
  width 1/2/4/8/16 serves all 15 dtypes) over NumPy's prepared iteration (`Backends/Iterators/RawArrayIter.cs` =
  `PyArray_PrepareThreeRawArrayIter`: axes by dst's |stride|, negative dst strides flipped, coalescing) plus the 1-D
  backward rule. An IL walker drives a run stage: a SIMD blend for contiguous runs (`ConditionalSelect`, AVX2 `vpmovzx`
  mask widening, a hoisted vector for a broadcast source), a branch-free select off running pointers for strided runs
  and tails, memmove / fill for a constant-true run. With more than one prepared axis the operands cannot overlap, so the
  innermost axis is the one where most operands are contiguous (NumPy's order wins ties; `np.CopytoNumPyOrderOnly` is the
  A/B hook). A source of another dtype, or an overlapping one NumPy copies first, is staged in dst's memory order — one
  element per distinct source element, on the stack up to 16 KB, else a pooled buffer — through NDIter.Copy's cast
  kernels.
- **The unmasked copy** is `NDIter.Copy`, whose overlap rule is now NumPy's for every caller (slice assignment too):
  copy first for a multi-axis destination or an opposite-direction 1-D one; otherwise NumPy's raw 1-D order
  (`DirectILKernelGenerator.AssignRaw1D`), which reads its own writes when the strides differ (`a[3:19:2] = a[6:14]`).
  It used to copy first for every non-unit-stride 1-D overlap.
- **Divergences** (house conventions): the exception types (TypeError → `ArgumentException` for the mask,
  `InvalidCastException` for casts; OverflowError → `OverflowException`); a 0-d non-bool mask always raises (NumPy takes
  `where=1`); LP64 texts for a Python int past int64; one complex width.
- **Gates:** `copyto_full.jsonl` (13,462 cases, zero excuses), unit `Manipulation/np.copyto.NumPyPort.Test.cs` (21);
  21 planted bugs, all killed by the unit tests, 20 by the tier.
- **Perf** (NPY/NS, two pinned P-cores, 192 cells: 9 dtypes × 1K/100K/1M × scalar / array / row-source / row-mask /
  strided dst / F dst / cast / no mask): see the plan item X16. The masked forms beat NumPy 1.5–60× but for two
  complex128 1M cells that stream 32+ MB (DRAM / L3-bound on both sides, 1.2–1.5×); the unmasked copy is memcpy on both
  sides (0.97–1.22× at 100K / 1M).

### Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`, unary `-`
- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&`, `|`, `!`

### Indexing
- Integer and slice indexing (`nd[0]`, `nd[1:3]`, `nd[::-1]`)
- Boolean masking (`nd[mask]`) — read-only
- Fancy indexing (`nd[indices]`)
- **Raw `int[]`/`long[]` as the SOLE index is FANCY** (NumPy parity): `nd[new int[]{0,2}]`
  selects rows 0 and 2 (shape `(2,…)`), NOT the element at coordinate `(0,2)`. `nd[0,2]` (separate
  scalar ints) stays coordinate (`HAS_INTEGER`). **Coordinate (element/sub-array) access moved to the
  shim `nd.GetData(int[]/long[])`** (`Backends/NDArray.cs`) — e.g. `nd.GetData(0, 2)`. Internal
  coordinate callers must use `GetData`, not `nd[coordArray]`.
  The typed `NDArray<T>.this[int[]]` is unaffected — it returns a scalar `T` (inherently coordinate).
- **Index errors are NumPy's `prepare_index` IndexErrors** (2026-10-08, `Selection/IndexErrors.NumPyPort.Test.cs`): a
  boolean mask with more dimensions than the array is `too many indices for array: array is 1-dimensional, but 2 were
  indexed` (checked first); otherwise the first mask axis of non-zero length that differs is `boolean index did not
  match indexed array along axis 1; size of axis is 3 but size of corresponding boolean axis is 4` (a zero-length mask
  axis matches any length); an index array that is neither integer nor bool — empty and 0-d included — is `arrays used
  as indices must be of integer (or boolean) type`, before the too-many check. The single-mask indexer used to raise an
  IndexOutOfRangeException naming axis 0 and the mask's first length for every mismatch, the integer-index normalizer an
  IndexOutOfRangeException, and an EMPTY float index array became an empty fancy index (a tuple's empty 2-D mask
  consumed one axis instead of two).
- **The boolean-mask getter's result is not zero-filled** (2026-10-08, X31): both of its gathers write every element (one
  per True, counted from the same mask bytes), so `DefaultEngine.BooleanMask` allocates with `NDArray.ForOverwrite`. Its
  zero-filled allocation came from OS-zeroed pages that page-faulted on every call: a 40,000-element int64 selection took
  184 µs (0.58× NumPy), now 31.6 µs (3.3×).

**Storing a C# literal is NumPy's `PyArray_Pack` of a Python scalar, at every store entry point** (2026-10-08, plan
`docs/plans/api-parameter-parity.md` X27; gates section S of the `weak_scalar` tier (15,301 store cases, zero excuses)
+ `Indexing/np.store.PythonLiteral.Test.cs`; probed against NumPy 2.4.2):
- **The marks.** An `NDArray` parameter receives the literal `1000` and an explicit `np.array(1000)` as the same 0-d
  array, and NumPy stores them differently (`a[0] = 1000` on int8 raises `OverflowError: Python integer 1000 out of
  bounds for int8`; `a[0] = np.array(1000)` casts to -24). So the implicit conversion operators MARK the 0-d array
  (`Backends/NDArray.PythonLiteral.cs`): an integer primitive / `float` / `double` / `Complex` is
  `WasPythonLiteral` (NumPy's own `NPY_ARRAY_WAS_PYTHON_LITERAL` temporary flag), a `Half` / `char` is
  `WasNumPyScalar` (`np.float16` / `np.uint16`); `bool` and `decimal` are unmarked. The mark belongs to that one object
  (a view, copy, reshape or result is unmarked); `NDArray.Scalar(x)` / `np.array(x)` build an unmarked, strong 0-d.
  ONLY the store paths read the marks; the ufuncs and comparisons keep reading every 0-d operand as weak (np.clip's
  rule), because library code builds its scalar operands with `NDArray.Scalar`.
- **The pack.** `NDArray.PackForStore(value, dtype, pyArrayPackRoute)` packs a marked value with
  `ReductionInitial.TryPack` (the complete `PyArray_Pack` port: a Python int range-checked with the LP64 texts, a float
  truncated then range-checked, NaN / inf / complex refused with Python's texts, an int past 2^53 rounded TWICE into
  float16 / float32, a bool dtype takes the truth). A NumPy scalar follows NumPy's route split, probed: the
  `PyArray_Pack` routes (an integer index, a slice / ellipsis view, `fill`, `flat[i]`, `np.pad`) int()-convert it with
  the range check into a SIGNED integer (`a[0] = (Half)300` on int8 raises), the `PyArray_FromAny` routes (fancy, mask,
  `np.put`, `putmask`, `place`, `put_along_axis`, `insert`, `full`, `asarray`, flat slices) CAST it (44).
- **The entry points.** The indexer setters (`UnmanagedStorage.SetData` after an integer index's bounds check; the
  fancy core after the too-many-indices check but BEFORE the index bounds, as NumPy; the mask setter after the mask's
  shape check; an empty selection packs too), `fill` (`CoerceFillValue` is now one `TryPack`), `np.full` / `full_like`
  (NumPy's `copyto(casting='unsafe')`: a Python int range-checked, a float / complex cast unsafely —
  `np.TryCopytoUnsafeScalarElement`), `np.put` (NumPy's `PyArray_PutTo` order: the empty-target IndexError, then the
  values' pack even with no indices, then an empty values array is a NO-OP — it raised ArgumentException), `putmask` /
  `place` / `put_along_axis` / `insert` / `fill_diagonal` (packs before its early returns, so an empty array still
  refuses) / the flat iterator (`flat[i]` reports any pack failure as NumPy's `ValueError: Error setting single item of
  array.`) / `np.asarray(literal, dtype)`; `np.pad` stores each constant as a NumPy scalar
  (`ReductionInitial.PackNumPyScalar`: `_as_pairs` runs `np.array(constant_values)`, so int8 refuses 1000 where uint8
  pads with 232).
- **Kept as casts:** `SetValue` / `itemset` (NumSharp's low-level typed setters; `itemset` is gone from NumPy 2.0),
  an unmarked 0-d array everywhere, and a 0-d view from integer indexing (`a[i] = b[j]`: NumPy's `b[j]` is a NumPy
  scalar, which a SIGNED integer destination would range-check — `[Misaligned]`). Text into a numeric store is parsed
  like NumPy's setitem of a Python `str` (since strings P1.1: `float64.fill("5.5")` is 5.5, `int32[0] = "hello"` raises
  `invalid literal for int() with base 10: 'hello'`).
- **Found on the way:** a basic-index setter with an EMPTY selection (`a["4:"] = v`) stranded one pooled buffer per
  call (the empty view's fresh zero-length buffer was never released; `AssignThroughView` wraps and disposes the view).
  Open: `np.array(object[], dtype)` (a Python list with a dtype) is not supported at all (`ArgumentOutOfRangeException`;
  `np.asanyarray(object[])` works).

---

## Core Source Files

| Component | Location |
|-----------|----------|
| NDArray | `Backends/NDArray.cs` |
| UnmanagedStorage | `Backends/Unmanaged/UnmanagedStorage.cs` |
| Shape | `View/Shape.cs` |
| Slice | `View/Slice.cs` |
| TensorEngine | `Backends/TensorEngine.cs` |
| DefaultEngine | `Backends/Default/DefaultEngine.*.cs` |
| np API | `APIs/np.cs` |
| Diagonal / triangular family | `Creation/np.tri.cs`, `Indexing/np.{diag,tril,diag_indices,tril_indices,fill_diagonal}.cs` |
| unique family | `Manipulation/NDArray.unique.cs` + `NDArray.unique.Kwargs.cs` (sort+mask core + axis path), `Manipulation/np.unique.cs` (`np.unique`), `Manipulation/np.unique_values.cs` (Array-API `unique_values`/`unique_counts`/`unique_inverse`/`unique_all` + result structs), `Manipulation/NDArray.unique.Hash.cs` (int + complex hash fast path, splitmix64). Design + measured perf decisions: `docs/UNIQUE_DESIGN.md` |
| Selection family | `Indexing/np.{take,take_along_axis,put,put_along_axis,place,putmask,select}.cs`; IL kernels `Backends/Kernels/Direct/DirectILKernelGenerator.{Take,TakeAlongAxis,Put,PutAlongAxis,Place,PutMask,Select}.cs` (`Select` = fused single-pass reverse-`ConditionalSelect` chain; `TakeAlongAxis` = whole-array strided-odometer gather, byte-width-keyed; `PutAlongAxis` = the scatter mirror — a dtype-agnostic validate pass + a bounds-free scatter pass for NumPy's all-or-nothing assignment; `PutMask` = Place's typed-MOV scatter with a by-position cursor + `nv==1` scalar fast path) |
| numpy.matrix (`NDMatrix`) | `Matrixlib/NDMatrix.cs` (class, result forms, scalar conversions, T / H / I / A / A1), `.Construction.cs` (`matrix.__new__`, the literal parser), `.Indexing.cs`, `.Operators.cs`, `.Reductions.cs`, `.Methods.cs`, `Matrixlib/np.matrix.cs`, `Creation/np.{asmatrix,bmat}.cs`, `View/Shape.Matrix.cs` (the finalize rule), `Selection/NDArray.Indexing.AssignedValue.cs` (NumPy's value rule for index-array assignment) |
| The matrix boundary (`np.*` with a matrix operand) | `Matrixlib/NDMatrixBoundary.cs` (`Sees`, the rules `Wrapped` / `Reduced` / `Plain` / `WrappedAll` / `WrappedPair` / `WrappedTriple` / `WrappedIf`, the regions `PlainFor` / `Inside`, `PlainOf` for containers, the caller audit); one statement in each public `np.*` overload; oracle `matrix_operand.jsonl` via `OpRegistry.MatrixOperand.cs` and `FuzzCorpusTests.MatrixOperand.cs` |
| Comparison engine (`==` … `>=`) | `Backends/Default/Math/DefaultEngine.CompareOp.cs` (dispatch, the trivial route), `DefaultEngine.CompareOp.Rows.cs` (the two-level route without an iterator), `DefaultEngine.UfuncOut.cs` (`out=` / `where=`), `Backends/Kernels/ComparisonRun.cs` (one chunk of any layout and dtype pair through the vector kernels; scalar adoption; the tile and repeated-row routes of a rows iteration), `Backends/Iterators/NDIter.Execution.ComparisonRun.cs` (the iterator hook), `Backends/Kernels/Direct/DirectILKernelGenerator.Comparison.cs` (the IL vector kernels), `.Comparison.Half.cs`, `.Comparison.Complex.cs`, `.Comparison.MixedSign.cs`, `.StridedMove.cs`, `.TransposeTile.cs` (a tile of an F-ordered operand into contiguous rows). Tests `Backends/Kernels/ComparisonRunTests.cs`, `TransposeTileTests.cs`, `ComparisonBroadcastKernelTests.cs`; fenced memory `test/NumSharp.Tests/Utilities/FencedMemory.cs` |
| BLAS/LAPACK seam | `Backends/IBlasBackend.cs` + `IBlasBackend.LinearAlgebra.cs` (15 default `Try*`), `Backends/TensorEngine.LinearAlgebra.cs` (virtuals + `LinAlgHelper`); managed LU fallback (`det`/`slogdet`/`solve`/`inv`) in `Backends/Default/LinearAlgebra/ManagedLu.cs`; `Backends/ISlidingDotBackend.cs` (optional level-1 `?dot` seam for `correlate`/`convolve`, `OpenBlasEngine.SlidingDot`) |
| ONNX Runtime interop (package) | `src/NumSharp.Interop.OnnxRuntime/NDArrayOnnxInterop.{cs,Export.cs,Import.cs,NonTensor.cs,Text.cs}` (dtype maps, `IsCpuAccessible`, `AsOrtValue`/`ToOrtValue`/`AsDenseTensor`/`ToDenseTensor`, `ToNDArray`/`AsNDArray`, the non-tensor readers, the text crossing), `OnnxCodecMode.cs` (Auto / View / Copy), `OrtTensor.cs` (the `OrtTensor`/`OrtTensor<T>` handles, the value → handle and disposed-value tables, `ImportLease`), `UnmanagedMemoryManager.cs` (`Memory<T>` over the unmanaged buffer + `PinToken`), `InferenceSessionExtensions.cs` (`session.Run(NDArray…)`, `RunInto`, the overlap and fed-input rules) + `InferenceSessionExtensions.Async.cs` (`RunIntoAsync`/`RunAsync`, `PendingRun`), `src/NumSharp.Interop.Shared/Postprocess.cs` (compiled into this bridge and the ML.NET one); ORT C# source for reference at `refs/onnxruntime/csharp/` (sparse submodule, at `v1.31.0`) + the official samples at `refs/onnxruntime-inference-examples/c_sharp/` |
| ML.NET interop (package) | `src/NumSharp.Interop.MLNet/NDArrayMLNetInterop.{cs,Export.cs,Import.cs}` (dtype maps, `AsDataView`/`ToDataView`/`ToVBuffer`, `ToNDArray`, zero-copy `VBuffer.AsNDArray<T>` + `WrapExternal`/`Pin`/`Live{Exports,Imports}`), `NDArrayDataView.cs` (the NDArray-backed `IDataView` + strided lazy cursor + ARC-pin lifetime), `VBufferAccessor.cs` (compiled-expression getter for the private `VBuffer<T>._values`), `ImportLease.cs` (the last-view-releases GCHandle pin lease), `Postprocess.cs` (shared with ONNX). Depends on `Microsoft.ML.DataView` only |
| CBLAS product family | `LinearAlgebra/np.{inner,vdot,vecdot,matvec,vecmat,tensordot}.cs`, `LinearAlgebra/GufuncGuard.cs` |
| einsum | `LinearAlgebra/np.einsum.cs`, `LinearAlgebra/np.einsum.Contract.cs` (the contraction composition — pure `np`-layer, not on the engine), `LinearAlgebra/EinsumSubscripts.cs` (port of `einsum.cpp`'s parser) |
| `np.linalg` module | `LinearAlgebra/linalg/np.linalg.cs` (class + `_assert_*`/`_commonType` ports) and `np.linalg.{solve,inv,det,eig,svd,qr,cholesky,lstsq,norm,multi_dot,matrix_power,arrayapi}.cs`; `Exceptions/LinAlgError.cs` |
| Window functions | `Math/np.windows.cs` (bartlett/blackman/hamming/hanning/kaiser + the internal cephes `BesselI0`; fused via `np.evaluate`) |
| Fourier / FFT (`np.fft.*`) | `Fourier/np.fft.cs` (`FourierModule` facade), `Fourier/np.fft.{Standard,Real,Hermitian,Helper}.cs` (the 18 funcs), `Fourier/np.fft.RawFft.cs` (layer-3 port: `_raw_fft`/`_raw_fftnd`/`_cook_nd_args`/`_swap_direction`), `Fourier/PocketFFTDriver.cs` (strided 1-D driver + FFTPACK packing), `Fourier/PocketFFT.{Twiddle,Complex,Real,Bluestein,Plan}.cs` (managed pocketfft engine); companion accessors `Math/np.{conjugate,real,imag,angle}.cs`. Design + parity ledger: `docs/FFT_PARITY.md` |
| numpy.polynomial evaluation (`np.polynomial.*`) | `Polynomial/Package/np.polynomial{,.polynomial,.chebyshev,.legendre,.laguerre,.hermite,.hermite_e}.cs` (facades), `Polynomial/Package/NDPolyEval.cs` (NumPy's Python layer), `Backends/Kernels/ILKernelGenerator.Polynomial.cs` (step tables + `PyScalar`), `.Typing.cs` (NEP 50 per node, peeling), `.Emitter.cs` (typed emitter), `.Lanes.cs` (vector lane kinds for every dtype pair), `.ConstPool.cs` (weak-value regions), `.Eval.cs` (dispatcher/part/stage kernels). Plan + measurements: `docs/plans/numpy-polynomial.md` (U3); benchmark `benchmark/polynomial/` |
| numpy.polynomial additive family + polyutils (U1) | `Polynomial/Package/np.polynomial.polyutils.cs` (`PolyUtilsModule` facade, incl. the generic tuple overloads), `Polynomial/Package/NDPolyNumber.cs` (`PolyNumber`: Python / NumPy-scalar / ndarray operand + NumPy's operator dispatch), `Polynomial/Package/NDPolySeries.cs` (as_series/trimseq/trimcoef/getdomain/mapparms/mapdomain/{p}add/sub/line/constants + the `PyNum` machine-number lane), `Backends/Kernels/Direct/DirectILKernelGenerator.PolySeries.cs` (trim/combine/tolerance/cast/scalarmath IL kernels + the fused mapparms kernel), CPython arithmetic in `ILKernelGenerator.Polynomial.cs` (`PyScalar`: `IntTrueDivide`, `ComplexQuotient`, NaN-priority `Float*`/`Complex*` helpers). Oracle `polyseries.jsonl` via `OpRegistry.PolySeries.cs` |
| numpy.polynomial calculus family (U4) | `Polynomial/Package/NDPolyCalc.cs` (`{p}der`/`{p}int` driver: NumPy's prologue, one-buffer orchestration, the integral's lbnd correction, NumPy's result layouts, `PyVal1D`), `Backends/Kernels/Direct/DirectILKernelGenerator.PolyCalculus.cs` (`PolyCalcRoutines` recurrence tables + the per-(basis, direction, dtypes) whole-array kernel: load/convert/scale stage, recurrence stage, column blocks). Oracle `polycalc.jsonl` via `OpRegistry.PolySeries.cs`; benchmark `benchmark/polynomial/polycalc_*` |
| numpy.polynomial series algebra (U2) | `Polynomial/Package/NDPolyAlgebra.cs` (engine: `PolySer`, the pooled per-thread `PolyArena`, the entry points, the shared statements — as_series, `{p}mulx`, `np.convolve`, `_div`/`_pow`/`_fromroots`), `NDPolyAlgebra.Bases.cs` (the recurrence products, polydiv / chebdiv, the ten conversions), `NDPolyPowerArgument.cs` (`int(pow)` / `power != pow` / `power > maxpower` for the object-typed `{p}pow` overloads), `Backends/Kernels/Direct/DirectILKernelGenerator.PolyAlgebra.cs` (house-kernel / mulx slot fronts + the fused chebmulx IL kernel), `Math/NDArray.SlidingDot{,.Long}.cs` (`SlidingCorrelateInto`, NumPy's per-dtype dotfunc models, the blocked long products). Oracle `polyalgebra.jsonl` + host-pinned `polyalgebra_parity.jsonl` via `OpRegistry.PolyAlgebra.cs`; benchmark `benchmark/polynomial/polyalg_*` |
| numpy.polynomial Vandermonde family (U5) | `Polynomial/Package/NDPolyVander.cs` (`{p}vander`/`{p}vander2d`/`{p}vander3d` driver: NumPy's statement and error order, the in-place point sources, block sizing, the streamed-product switch, the object stack of scalars — `ObjectScalarPoints` / `MixedDtypeVander`), `NDPolyIndexArgument.cs` (`operator.index` + NumPy's `format(deg, '')` error text), `Backends/Kernels/Direct/DirectILKernelGenerator.PolyVander.cs` (`PolyVanderRoutines` recurrence tables, `PolyVanderOps` NaN-priority multiply + non-temporal stores, the load / recurrence / product / root stages). Oracle `polyvander.jsonl` via `OpRegistry.PolyVander.cs` + `OpRegistry.PolySeries.cs`; benchmark `benchmark/polynomial/polyvander_*` |
| numpy.polynomial companion / roots (U7) | `Polynomial/Package/NDPolyAlgebra.Roots.cs` (`{p}companion`/`{p}roots`: NumPy's statements over U2's arena — as_series, the length checks, the linear root's scalarmath, the write-once zero matrix, the diagonals through `GetDiagWriteKernel`, the last-column update through the house ufunc / cast / sqrt / cumprod kernels, the rotated `np.linalg.eigvals` + in-place sort; object series: `ObjectTrimLength`, `ObjectLinearRoot`, `ObjectCompanionArithmetic`), `Backends/Unmanaged/UnmanagedMemoryBlock`1.cs` (`AllocationType.Forward`: a reinterpreting alias's ARC forwards to its owner — `ArraySlice.WrapShared`), `Backends/Kernels/Direct/DirectILKernelGenerator.PolyRoots.cs` (the int64 ramp IL kernel behind np.arange), `LinearAlgebra/linalg/np.linalg.eig.cs` (`RoundComponentsToSingle`: a float32 operand's complex result carries NumPy's complex64 values). Oracle `polyroots.jsonl` + host-pinned `polyroots_parity.jsonl` via `OpRegistry.PolySeries.cs`; benchmark `benchmark/polynomial/polyroots_*` |
| Grid / slice-expression DSL | `Creation/np.r_.cs` (`AxisConcatenator` + `RClass`), `Creation/np.c_.cs`, `Creation/np.ogrid.cs` (`OGridClass` + `OGridResult` + shared `nd_grid` helpers), `Creation/np.mgrid.cs` (`MGridClass` + `MGridResult`), `Creation/np.meshgrid.cs` (`MeshgridResult`), `Indexing/np.{ix_,s_}.cs` |
| Array printing (NumPy parity) | `Backends/Printing/{PrintOptions,Dragon4,ElementFormatters,ArrayFormatter}.cs`, `APIs/np.array2string.cs`, `Casting/NDArray.ToString.cs` |
| Iterators | `Backends/Iterators/NDIter.cs`, `NDIter.Detach.cs` (ref-struct → managed-owner bridge) |
| Iteration APIs (nditer/ndindex/ndenumerate) | `APIs/np.nditer.cs`, `Indexing/np.{ndindex,ndenumerate}.cs` |
| Typed iteration (nditer&lt;T&gt;/nditer_chunks&lt;T&gt;) | `APIs/np.nditer.Typed.cs` (ref-struct enumerators + `TypedIterHelpers`) |
| Expression DSL (np.evaluate) | `Backends/Iterators/NDExpr.cs` (nodes + emission), `NDExpr.Typing.cs` (per-node NumPy result_type pass), `NDExpr.Evaluate.cs` (array leaves, binding, reductions, operators), `Backends/Default/Math/DefaultEngine.Evaluate.cs` (host) |
| ILKernelGenerator | `Backends/Kernels/ILKernelGenerator*.cs` (per-chunk, NDIter-driven) |
| DirectILKernelGenerator | `Backends/Kernels/Direct/DirectILKernelGenerator.*.cs` (whole-array, 63 partials) |
| Kernel shared infra | `Backends/Kernels/{VectorMethodCache,ScalarMethodCache,KernelOp,StrideDetector,SimdMatMul.*}.cs` |
| .npy/.npz format (NEP-01) | `IO/NpyFormat.cs` (port of NumPy's `_format_impl.py`), `IO/NpzFile.cs` (lazy archive), `IO/PyLiteral.cs` (header parser ≙ `ast.literal_eval`), `APIs/np.{save,load}.cs` |
| float32 NumPy-exact math (exp/log/sin/cos) | `Utilities/NDFloatMath.cs`, `Utilities/NDFloatMath.Simd.cs` |
| Type info | `Utilities/InfoOf.cs` |
| Generic NDArray | `Generics/NDArray\`1.cs` |

---

## Implementation Patterns

### Pattern 1: Compose existing np functions
```csharp
public static NDArray std(NDArray a, int? axis = null, ...)
{
    var mean_val = np.mean(a, axis, keepdims: true);
    return np.sqrt(np.mean(np.power(a - mean_val, 2), axis));
}
```

### Pattern 2: Delegate to TensorEngine
```csharp
public static NDArray sum(NDArray a, int? axis = null, ...)
{
    return a.TensorEngine.Sum(a, axis, typeCode, keepdims);
}
```
Use Pattern 2 when low-level optimization is needed.

## Type Switch Pattern

```csharp
switch (nd.typecode)
{
    case NPTypeCode.Boolean: return Process<bool>(nd);
    case NPTypeCode.Byte: return Process<byte>(nd);
    case NPTypeCode.SByte: return Process<sbyte>(nd);
    case NPTypeCode.Int16: return Process<short>(nd);
    case NPTypeCode.UInt16: return Process<ushort>(nd);
    case NPTypeCode.Int32: return Process<int>(nd);
    case NPTypeCode.UInt32: return Process<uint>(nd);
    case NPTypeCode.Int64: return Process<long>(nd);
    case NPTypeCode.UInt64: return Process<ulong>(nd);
    case NPTypeCode.Char: return Process<char>(nd);
    case NPTypeCode.Half: return Process<Half>(nd);
    case NPTypeCode.Single: return Process<float>(nd);
    case NPTypeCode.Double: return Process<double>(nd);
    case NPTypeCode.Decimal: return Process<decimal>(nd);
    case NPTypeCode.Complex: return Process<Complex>(nd);
    default: throw new NotSupportedException();
}
```

## GitHub Issues

Create issues on `SciSharp/NumSharp` via `gh issue create`. `GH_TOKEN` is available via the `env-tokens` skill.

### Feature / Enhancement

- **Overview**: 1-2 sentence summary of what and why
- **Problem**: What's broken or missing, why it matters
- **Proposal**: What to change, with a task checklist (`- [ ]`)
- **Evidence**: Data, benchmarks, or references supporting the proposal
- **Scope / Non-goals**: What this issue does NOT cover (prevent scope creep)
- **Benchmark / Performance** (if applicable): Before/after numbers, methodology, what to measure
- **Breaking changes** table (if any): Change | Impact | Migration
- **Related issues**: Link dependencies

### Bug Report

- **Overview**: 1-2 sentence summary of the bug and its impact
- **Reproduction**: Minimal code to trigger the bug
- **Expected**: Correct behavior (include NumPy output as source of truth)
- **Actual**: What NumSharp does instead (error message, wrong output, crash)
- **Workaround** (if any): How users can avoid the bug today
- **Root cause** (if known): File, line, why it happens
- **Related issues**: Link duplicates or upstream causes

## Performance Convention

All NumSharp-vs-NumPy benchmark ratios are reported as **NPY/NS**:

> **ratio = NumPy_ms / NumSharp_ms** — **`>1` = NumSharp FASTER, `<1` = NumSharp slower, `=1` = parity.**

**Higher is better.** Equivalently `speedup = NumPy_time / NumSharp_time`. A cell of
`0.5` means NumSharp takes 2× NumPy's time; `2.0` means NumSharp is 2× faster. Use this
direction **everywhere** — matrices, geomeans, commit messages, and the per-subsystem
`*_sheet.py` outputs (`nditer`/`layout`/`operand`/`cast`/`fusion`) — so one glance answers "are we faster?".

- Icons used in the matrices: ✅ `≥1.0` · 🟡 `≥0.5` · 🟠 `≥0.2` · 🔴 `<0.2`.
- Timing scripts MUST run `dotnet run -c Release - < script.cs` (Debug taints hand-written kernels ~2×; see `benchmark/CLAUDE.md`).
- Pin a C# timing process to TWO CPUs (`start /affinity 14`: two P-cores on this host), never one. A process pinned to a
  single CPU starves .NET's background tier-1 compiler even at normal priority, so a hot path can stay at tier 0 for
  good (np.max into a cast out measured ~100 µs per call on one CPU, 12–13 µs on two once tiered up after ~0.4 s); a run
  mixes the regimes, because a cell tiers up only when the main thread happens to block (plan api-parameter-parity
  finding 12). Pin the NumPy twin the same way.
- best-of-rounds (take the min), warmup excluded, correctness checked before every timed row.
- The canonical harness is `benchmark/run_benchmark.py` — the op/dtype/N matrix plus five appended subsystems: `benchmark/nditer/` (iterator aspects), `benchmark/layout/` (reduction/copy/elementwise × memory layout × dtype), `benchmark/operand/` (1-D/scalar/mixed-operand/broadcast layouts), `benchmark/cast/` (astype src→dst matrix), `benchmark/fusion/` (np.evaluate). The `run-benchmarks.ps1` "Status Icons" table reports the *inverse* (NS/NPY, lower = better) — prefer this NPY/NS convention.
- The op/dtype/N matrix is **BenchmarkDotNet** (`benchmark/NumSharp.Benchmark.CSharp`) timed against a **warm NumPy** process (`benchmark/NumSharp.Benchmark.Python`) across **1K / 100K / 10M × all 15 dtypes** (~615 ops/size), joined on `(op, dtype, N)`. Two methodology guards beyond `-c Release`: the **InProcessEmit** toolchain (sibling `.claude/worktrees/` checkouts hold same-named projects the out-of-process toolchain refuses to build) and a **25 ms-capped 50-iteration** job (so µs–ms array ops skip BenchmarkDotNet's nanosecond invocation ramp — without it the full matrix takes days). The NDIter subsystem reports a section that crashes all retries (the known intermittent `AccessViolation`) as **NA/IGNORED**, never a failure.
- **What we commit & reference is `benchmark/history/`, not the gitignored `benchmark/results/<ts>/` raw scratch.** Every run writes a committable snapshot `benchmark/history/<date>_<sha>/` (MANIFEST + report.{md,json,csv} + numpy-results.json + all subsystem `*_results.{md,tsv}` + cards — the json/csv are gitignored at the benchmark root, so this is their only tracked home) and repoints `benchmark/history/latest` (a git symlink). Built by `benchmark/scripts/snapshot_history.py` (called by `run_benchmark.py`; `--no-history` opts out). Reference the stable `benchmark/history/latest/benchmark-report.md`. The publish ritual (run → review → commit `benchmark/history/`) is what `.github/workflows/benchmark.yml` performs post-release. See `benchmark/CLAUDE.md` → "History snapshots & the publish ritual".
- **Reports & UI surfaces** (canonical → human-facing; they drift, so know which is which). The canonical text report is `benchmark/benchmark-report.md` (the merged op-matrix + the five appended subsystem sections, NPY/NS with a `%NumPy🕐` column and `merge-results.py`'s credibility bands — `▫ negligible` for sub-µs/>20× rows excluded from geomeans, `⚪` for unjoined C# rows). `benchmark/README.md` is a **static orientation guide, NOT the report**. The human-facing **UI** is the **hand-built** DocFX page `docs/website-src/docs/benchmarks-dashboard.md` (the website's "Benchmarks" hub — HTML/CSS/JS with dtype heatmaps, numbers embedded inline), which links the raw reports at `benchmark/history/latest/*`; it is edited by hand, seeded from `benchmark/benchmark-dashboard.md` (the gitignored ASCII sheet from `benchmark/scripts/render_dashboard.py`, itself NOT wired into `run_benchmark.py`/CI). Full map: `benchmark/CLAUDE.md` → "Reports & UI surfaces".

## Build & Test

```bash
# Build (silent, errors only)
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0
```

### Strong naming (identity, NOT authenticity)

Every assembly in the repo is strong-named with **`Open.snk` at the repo root**, configured once in
**`Directory.Build.props`** — `SignAssembly` defaults to true and any project opts out with
`<SignAssembly>false</SignAssembly>`. Public key token: **`cc7b13ffcd2ddd51`**.

**The key is Microsoft's PUBLISHED open-source key** — byte-identical to
`dotnet/arcade`'s `src/Microsoft.DotNet.Arcade.Sdk/tools/snk/Open.snk`, the same identity carried by
`netstandard`, `System.Memory` and `System.Buffers`, and the same key **TensorFlow.NET** signs with
(which is why we keep it: a NumSharp-owned key would break the cross-repo
`InternalsVisibleTo("TensorFlowNET.UnitTest")`). Its private half is public by design, so this is
assembly **identity** — version binding, telling this `NumSharp` from another — and **not** a
security property. `InternalsVisibleTo` is therefore not an access control. Real authenticity is
Authenticode + NuGet author signing, needs a code-signing certificate, and does not exist here yet.

**Consequences to know:**
- Every `InternalsVisibleTo` MUST carry `PublicKey=` — keyless is `CS1726`, a hard error. There are
  7: five in `NumSharp.Core/Assembly/Properties.cs`, two in `NumSharp.Interop.OpenBLAS/AssemblyInfo.cs`.
  The literal is held in `NumSharpFriendKey` / `BlasFriendKey` consts so it is written once per assembly.
- `dotnet run` scripts outside the repo root need signing properties to see internals — see
  "Scripting with `dotnet run`".
- Gate: `test/NumSharp.Tests/Assembly/StrongNameTests.cs` (6 tests) asserts the token, the full
  public key, that friend access actually resolves, and that no friend declaration is keyless.

**How it was broken before (2019 → 2026-07):** `SignAssembly` sat in a `Publish|AnyCPU`-only
PropertyGroup while CI builds and packs `Release`, so **every published NumSharp shipped unsigned**
(`PublicKeyToken=null`) — and the `Publish` configuration did not even compile, because all seven
friend declarations were keyless. A `#if !SIGNING` guard was meant to cover that, but `SIGNING` was
defined only in the *test* csproj, a different assembly, where it did nothing. Nothing asserted the
identity, so none of it surfaced. Hence the runtime gate rather than trusting the build.

### Running Tests

Tests use **MSTest v3** framework with source-generated test discovery.

```bash
# Run from test directory
cd test/NumSharp.Tests

# All tests (includes OpenBugs - expected failures)
dotnet test --no-build

# Exclude OpenBugs (CI-style - only real failures)
dotnet test --no-build --filter "TestCategory!=OpenBugs"

# Run ONLY OpenBugs tests
dotnet test --no-build --filter "TestCategory=OpenBugs"

# Exclude multiple categories
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"

# Run specific test class
dotnet test --no-build --filter "ClassName~BinaryOpTests"

# Run specific test method
dotnet test --no-build --filter "Name~Add_Int32_SameType"
```

### Output Formatting

```bash
# Results only (no messages, no stack traces)
dotnet test --no-build 2>&1 | grep -E "^(failed|skipped|Test run|  total:|  failed:|  succeeded:|  skipped:|  duration:)"

# Results with messages (no stack traces)
dotnet test --no-build 2>&1 | grep -v "^    at " | grep -v "^     at " | grep -v "^    ---" | grep -v "^  from K:" | sed 's/AssertFailedException: //'

# Verbose output (shows passed tests too)
dotnet test --no-build -v normal
```

### Keeping the suite fast (2026-09-24: NumSharp.Tests 262 s → ~24 s, Oracle 37 s → 12 s per framework, local)

MSTest runs the ~16K tests **serially**, so a handful of slow tests set the wall time. The rules came out
of finding them (the diagnosis, per-test numbers and mutation checks are in commit messages / the files):

- **Never force a FULL GC per measurement.** A forced full collection marks the whole live heap, and in a
  full run that heap is mostly MSTest's per-test bookkeeping (~120 MiB → ~78 ms per
  `Collect/WaitForPendingFinalizers/Collect`, vs <10 ms in isolation) — so a GC-drain loop that is instant
  alone costs a minute in the suite, and grows with every test added. Tests that read process-global
  counters (pool take/return, ARC refcounts after a finalizer) use `test/NumSharp.Tests/Utilities/GcQuiescence.cs`
  (linked into the Oracle): `CollectYoung()` for per-window hygiene, `OpenWindow()`/`Undisturbed()` to prove no
  collection — including a background GC, whose `CollectionCount` moves at its START but whose finalizers run
  at its END — touched the window, `CollectFull()` once per test at most. Lifetime tests asserting what a
  collection does to objects they just built use `CollectSince(epoch)` ("still reachable": young is exact while
  no collection ran since the epoch, full otherwise) and `WaitCollected(weakRef)` ("now collectable": young
  first, full as the fallback); a class whose counter windows need a clean backlog drains FULL once in
  `[ClassInitialize]` and young per test; temp dirs holding memmaps go through `MappedFileCleanup.DeleteDirectory`.
- **Concurrent-collection storms have two budgets** (`test/NumSharp.Tests/Collections/StressBudget.cs`): each
  call site spells `StressBudget.Pick(full: <authored>, ci: <per-push>)`. Per-push runs the CI budget;
  `NUMSHARP_TEST_STRESS=full` restores the authored workload exactly, and the nightly `collections-stress` job
  in `fuzz-soak.yml` runs it. Size a new storm the same way rather than letting it dominate every push. A
  WALL-CLOCK storm (a time-boxed race hunt) gets a short per-push window only after calibration: re-introduce
  the defect it pins and confirm every per-push run still fails, pinned to 4 CPUs and unpinned (the numbers
  live on `OrderedDictionaryContractTests` and the swap-back gun). Size a per-push LIVE SET under the
  large-object threshold: every interior removal copies the whole generation, and past 4,096 entries the
  compact dictionary's index (the power of two above 1.5× capacity, in `long`s) is ≥ 128 KB, so each copy is a
  gen-2 collection (PhasedStorms/PartitionedAllOps/KeyOwnedOps/IndexPathReaders ran 190–294 gen-2 per run). A
  budget shrink is calibrated like a window: a mutation the storm catches, at the old AND new budget, pinned
  and unpinned (the per-storm evidence is in each budget comment). Two traps from doing it: a MUTATED run is
  faster than a clean one (a reader that sees the violation throws and leaves, so its writers finish sooner —
  time only clean same-build A/Bs), and shrinking the op count can silently drop a cadence-gated operation
  (FullSurfaceChaos' mid-storm `Clear` — budget the cadence masks with the ops).
- **Hot product paths use `[GeneratedRegex]`, never the static `Regex.Match/Split/Replace`.** On .NET 8 the
  static helpers share a 15-entry cache whose eviction thrashes once it is full of stale entries — every
  string slice (`nd["1:3"]`) re-parsed its pattern (9–18 µs instead of 0.6–2.8 µs per `ParseSlices`;
  microgpt 2.5 → 0.57 s on net8.0). Idle polls in stress tests call `Thread.Yield()`, not `Thread.SpinWait` —
  on .NET 8 a `SpinWait` loop is an FCALL the GC cannot suspend, and blocking collections stalled 24–43 s.
- **A strided view is not a scalar-path probe.** The unary kernels gather strided inputs into the vector
  kernel, so "contiguous vs `::2`" compares the vector kernel with itself (`Exp2_Scalar_Equals_Simd` was
  vacuous until 63a94fa2); compare against the scalar entry point (`NDFloatMath.X(float)`) directly, and
  confirm with a planted scalar-only and vector-only slip that each assertion can fail.
- **Oracle coverage gates read `CorpusSurvey`, replays stream `FuzzCorpus.Open`.** Never re-parse the whole
  ~170 MB corpus into `FuzzCorpus.Case` objects for a coverage question (op keys, layouts, dtypes, params,
  outcome kinds): `CorpusSurvey.Files` is a header-only scan built once per process (extend `SurveyCase` + its
  fidelity test if a gate needs a new field). A replay that visits each case once enumerates `FuzzCorpus.Open(file)`
  (count up front, one parsed case alive at a time) — a materialized whole-tier `List<Case>` gets promoted
  through the GC generations, which cost more than the parse itself.
- **Test hosts run with `TieredPGO=false` and opt out of EcoQoS.** A test run is short and JIT-bound, so
  PGO's instrumented tier never pays back (Oracle 15.1 → 12.2 s, Analyzer 19.0 → 17.5 s pinned; NumSharp.Tests
  neutral) — set in the three test csprojs, overriding the repo-wide default. `TestHostQualityOfService` (a
  `[ModuleInitializer]`, Windows-only) disables execution-speed power throttling, because Windows parks a
  windowless `testhost` on a hybrid CPU's E-cores (i9-13900K: NumSharp.Tests 47 s unpinned vs 27 s on P-cores).
  **Measurement trap:** A/B single-threaded suites pinned to the P-cores (affinity `0xFFFF` on the dev box) —
  unpinned runs mix two ~1.8×-apart populations and swamp any knob.
- **A test that can never return must FAIL, not hang** (2026-10-01). NumPy's rejection samplers never accept a
  draw at some values their checks reject (`logseries` at `p = 1`; `zipf` at `a <= 1` or NaN), so a regressed
  check HANGS a test instead of failing it. A local `dotnet test` has no timeout; CI's test steps have
  `timeout-minutes`. An in-progress build whose array check read `p <= 1` left an oracle testhost spinning for 5
  days (84 CPU-hours) on `rnd/logseries/bcast:viol:p>=1/857`, and a process dump was needed to name the case.
  The guards now in place:
  - Oracle corpus replays: the per-case `CaseWatchdog` (`Fuzz/README.md` → Gate semantics). It ends the host
    with `FailFast`, naming the case, after 2 min (`NUMSHARP_ORACLE_CASE_TIMEOUT_SECONDS`; 0 disables).
  - Unit tests that drive those bounds: `[Timeout]` or an explicit deadline thread.

  Give any new test that feeds a sampler its rejected boundary the same guard. Diagnose a hung testhost with
  `dotnet-dump collect -p <pid>`, then `analyze` with `clrthreads` (the Cooperative thread is the spinner),
  `setthread <n>` + `clrstack`, and `dso` → `dumpobj` on the `FuzzCorpus+Case` for its `Id`. A zombie
  testhost also LOCKS its build output (`bin/Debug`): build Release until it is gone.

### Memory-safety gates: fenced memory and guard pages (2026-10-10)

A kernel that READS past its operand gives right values (the extra bytes are discarded), and a sentinel margin shows
writes only. So a value test, an oracle case and a sentinel test all pass over such a read. It is a crash waiting for
the one buffer that ends where mapped memory ends. Two tools make it fail a test run:

- **`FencedMemory`** (`test/NumSharp.Tests/Utilities/FencedMemory.cs`; Windows, Inconclusive elsewhere). Native memory
  with an inaccessible page directly below its first byte and directly above its last one (the region is reserved
  whole, only the data pages are committed).
  - `fence.Place(bytes, atEnd)` gives the address of a block against the low or the high page.
  - `np.frombuffer((IntPtr)p, bytes, dtype)` makes that block an array, so the public API runs over it.
  - A violation is not a failed assertion: the test host ends ("Test host process crashed") with the stack of the
    kernel that left its operand.
- **`NUMSHARP_DEBUG_GUARD_PAGES=1`** (the pool's diagnostic mode). Every pooled buffer ends at an inaccessible page;
  there is no page below it. Run a test class, the whole unit suite or the oracle replay with it after you write
  pointer code:
  `NUMSHARP_DEBUG_GUARD_PAGES=1 dotnet test --no-build -c Release --framework net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"`.
  The tests that count pool takes and returns fail in this mode, because it bypasses the pool: four in
  `NumSharp.Tests` (`NoOperationDefersItsBufferRelease`, `EightThreads_MixedOps_ZeroErrors_ZeroStrandSlope`,
  `ScopedSurface_ZeroStrandsPerCall`, `TupleCarrier_Polydiv_ZeroStrands`) and five in the oracle project (the
  `…LeaveNoUndisposedIntermediates` gates and `OperatorNot_ReclaimsUntypedMakeGenericWrapper`). A CRASH is the
  finding. The unit suite takes 1 m 44 s this way and the oracle replay 3 m 37 s.

Rules these tools gave:
- **A stride kernel is visible only on a view that ENDS at its buffer's last element.** `a[::2]` of an even-length
  buffer leaves one element behind the view, which absorbs a one-element over-read. Test `a[1::2]`, `a[2::3]` and the
  reversed forms that START at the last element as well.
- **A vector load of N elements reads N elements.** A deinterleave of two loads covers one element more than the
  group needs (see "A strided copy or cast never reads outside its view"); a gather with a full mask does not.
- **A test whose two sides share a wrong operand proves nothing.** `Where_NarrowRows_3D_LeadingAxisWalkedPerBlock`
  compared `np.add(..., where: mask)` with `np.where(mask, ...)` over a mask that addressed memory outside its buffer
  (the typed reshape above). Both sides read the same wrong mask, and the test passed from 2026-09-03 on.

What the tools found on their first day (all fixed, each with its own gate): the sub-word deinterleave
(`StridedCastFenceTests`), `NDArray<T>.reshape(Shape)` (`TypedReshapeViewShapeTests`), and no read or write outside an
operand in the comparison engine (`ComparisonRunTests`: `Run_BetweenInaccessiblePages_…`,
`RunRows_BetweenInaccessiblePages_…`). After the two fixes the whole unit suite and the whole oracle replay (every
corpus case of every operation and layout) run under guard pages without a crash.

## Test Categories

Tests use typed category attributes defined in `TestCategory.cs`. Adding new bug reproductions or platform-specific tests only requires the right attribute — no CI workflow changes.

| Category | Attribute | Purpose | CI Behavior |
|----------|-----------|---------|-------------|
| `OpenBugs` | `[OpenBugs]` | Known-failing bug reproductions. Remove when fixed. | **EXCLUDED** via filter |
| `Misaligned` | `[Misaligned]` | Documents NumSharp vs NumPy behavioral differences. | Runs (tests pass) |
| `WindowsOnly` | `[WindowsOnly]` | Requires GDI+/System.Drawing.Common | Excluded on non-Windows |
| `LongIndexing` | `[LongIndexing]` | Arrays with size > int.MaxValue (>2B elements) | Runs (excluded only if also HighMemory) |
| `HighMemory` | `[HighMemory]` | Requires 8GB+ RAM | **EXCLUDED** via filter |
| `LargeMemoryTest` | `[LargeMemoryTest]` | Memory-heavy non-bugs (combines OpenBugs+HighMemory) | **EXCLUDED** via filter |
| `FuzzMatrix` | `[TestCategory("FuzzMatrix")]` | NumPy differential gate — replays the committed oracle corpus bit-exact (see Differential-Fuzz Pipeline). | Runs (the gate) |

### How CI Excludes Categories

The CI pipeline (`.github/workflows/build-and-release.yml`) uses MSTest's `--filter` to exclude categories:

```yaml
- name: Test (net10.0)
  run: |
    dotnet test test/NumSharp.Tests/NumSharp.Tests.csproj \
      --configuration Release --no-build --framework net10.0 \
      --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"
```

This filter excludes all tests with `[OpenBugs]` or `[HighMemory]` attributes from CI runs. Tests pass locally when the bug is fixed — then remove the `[OpenBugs]` attribute.

### Usage

```csharp
// Class-level (all tests in class)
[TestClass]
[OpenBugs]
public class BroadcastBugTests { ... }

// Method-level
[TestMethod]
[OpenBugs]
public void BroadcastWriteCorruptsData() { ... }

// Documenting behavioral differences (NOT excluded from CI)
[TestMethod]
[Misaligned]
public void BroadcastSlice_MaterializesInNumSharp() { ... }
```

### Local Filtering

```bash
# Exclude OpenBugs (same as CI)
dotnet test --filter "TestCategory!=OpenBugs"

# Run ONLY OpenBugs tests (to verify fixes)
dotnet test --filter "TestCategory=OpenBugs"

# Run ONLY Misaligned tests
dotnet test --filter "TestCategory=Misaligned"

# Combine multiple exclusions
dotnet test --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=WindowsOnly"
```

**OpenBugs files**: `OpenBugs.MatrixOperand.cs` (the matrix boundary's gaps outside its oracle tier: `np.gradient` and `np.meshgrid` return plain arrays in their result structs where NumPy returns matrices, MO06 / MO07), `OpenBugs.cs` (general), `OpenBugs.Bitmap.cs` (bitmap), `OpenBugs.ApiAudit.cs` (API audit), `OpenBugs.ILKernelBattle.cs` (IL kernel), `OpenBugs.Random.cs` (random), `OpenBugs.BroadcastReduce.cs` (broadcast reduce), `OpenBugs.Reductions.cs` (how a reduction STARTS: flat `cumsum` loses a leading `-0.0`, the managed size-1 `dot`/`inner` operands take the opposite of NumPy's scalar branch, complex one-element `prod` skips the `1+0j` identity, `median` returns the middle element instead of its `np.mean`, complex `cumprod` multiplies element 0 by `1+0j`, the default `einsum`'s zero-filled accumulator for bare products; the N-D size-1 flat `cumprod` returning 0-d, which NaN a flat sum of several NaNs keeps, an axis sum in a cast loop whose pinned rows are longer than `np.getbufsize()` (group 9: NumPy splits them across fills), an axis sum over a view with a reversed axis walking memory order (group 10, plan finding 5), the flat sum of a broadcast operand not following NumPy's per-row call schedule (group 11, finding 10), an axis sum of a broadcast operand (group 12, plan X12) the engine's flat mean of a broadcast float16 / complex128 operand ignoring `dtype=` (group 13, X14) and `np.average` over a tuple axis reducing one axis at a time — a mean of means, weighted a sum of sums, where NumPy runs ONE add.reduce over the axis set (group 14, X36; the fix is the library-wide tuple-axis add.reduce); raw-bit asserts, because `0.0 == -0.0`. The add.reduce identity class — a sum/mean/nansum/nanmean/average/trapezoid/einsum-sum/np.evaluate-sum of only `-0.0` keeping `-0.0`, one-element nan-reductions returning `(1,)`, the complex 1-D `nansum` throw, the float16 nan-axis wrong-element reads, `np.mean`'s float64 empty result — is fixed and its tests moved to `Math/np.sum.Identity.Test.cs`), `OpenBugs.FancyIndex.cs` (fancy indexing against NumPy's three `mapping.c` routes: an N-D index array that is not C-contiguous, beside a slice / ellipsis / integer, is read in MEMORY order — wrong values from the getter, wrong positions from the setter; and the result LAYOUT — NumPy's transposed MapIter view for an index behind a slice, the index arrays' order for a lone index or arrays on every axis, the source's order for the other axes of a non-C source, incl. `Generator.permutation` of a non-C 3-D source), `OpenBugs.EmptyStrides.cs` (a zero-size array NumPy allocates has all-zero strides, a slice keeps its parent's strides on an extent-0 axis; NumSharp does neither), `OpenBugs.PolynomialNaN.cs` (the `numpy.polynomial` U3/U4 step-table integral kernels keep the wrong NaN per statement — `lagint` float16 overflow / `hermeint` float16 NaN-in-`k` / `lagint` float64 `scl=+inf`; raw-bit asserts vs NumPy 2.4.2, win-amd64), `OpenBugs.SortUniqueNaN.cs` (`np.unique`/`unique_values` keep a float32/float64 NaN's payload and sign where NumPy's sort-first unique returns the canonical positive quiet NaN; `np.sort` ignores `kind`, canonicalizing where NumPy's `stable`/`mergesort` keep every NaN's bits), `OpenBugs.Matrix.cs` (the array library below numpy.matrix, found by the matrix tier: np.dot N-D × M-D computes complex128 in float64, the flat np.prod in a cast loop, a fancy assignment into an offset view throws, and four result-layout gaps — cumprod, imag of a real F array, round of a broadcast operand, conjugate of a transposed strided view), `OpenBugs.ErrorParity.cs` (FIXED regression pins since 2026-10-08, plan X36: the error-MESSAGE gaps a 2026-10-06 sweep found — the stacking family's empty operand list leaking .NET's "Value cannot be an empty collection." (now NumPy's `ValueError` "need at least one array to concatenate" / "…to stack"), `moveaxis`/`rollaxis`/`swapaxes`/`squeeze`'s legacy `axis (5) … dimension (2)` texts (now NumPy's `AxisError` with its `source:`/`destination:`/`axis2:` prefix), and `moveaxis`'s length mismatch as a bare `Exception` with a stray backtick (now NumPy's `ValueError`)). `OpenBugs.UfuncOut.cs` is gone: its pin (`np.power(x, 2.0, out=o)` called `pow` where NumPy takes `x*x`) was fixed with plan X13 (2026-10-06) and moved to `Math/np.power.StrideZero.Test.cs`. `OpenBugs.IndexSearchLayout.cs` is gone too: `argwhere` / `flatnonzero` / `unravel_index` return NumPy's views since 2026-10-08 (plan X37), and its tests moved to `Indexing/IndexSearchLayout.Test.cs`. Group 5 (`np.var`/`np.std` at a degenerate size) is fixed: np.var / np.std are NumPy's `_var` since 2026-10-05, its tests moved to `Math/np.var.NumPyPort.Test.cs`, and the registry's **V1** branch is gone. `OpenBugs.ReductionKeywords.cs` (the max / min family without NumPy's `out=`/`initial=`/`where=`) is fixed the same day (plan A4): its tests moved to `Selection/np.max.Keywords.Test.cs`. Group 4 (`np.nanstd`/`np.nanvar` returning `(1,)` for a 1-D axis) is fixed the same day: they are NumPy's `nanvar`, and its test moved to `Math/np.nanvar.NumPyPort.Test.cs`. The LIFETIME OpenBugs (lost GC anchors) live in the Oracle project, not here, because the bug exists only in optimized code and they need the lifetime child process (see "Differential-Fuzz Pipeline"): none is open since 2026-10-09 (the 28 found were fixed by the weaver's operand roots and pinned in the gate; `LifetimeOracleTests.Gaps.cs` and its per-gap tests are gone). A new open gap gets a `Lifetime_LT<nn>_*` test beside the aggregate `Lifetime_KnownGaps` (and `LifetimeOracleTests.Scaled.cs`'s `LifetimeScaled_KnownGaps`).

**Auditing the whole category (2026-10-05).** Run `TestCategory=OpenBugs&TestCategory!=HighMemory` (`[LargeMemoryTest]` adds the
OpenBugs category to memory-heavy NON-bug tests) in Release AND Debug on both TFMs; a test that passes in all four has its
tag removed, its comments rewritten from "BUG" to "FIXED (commit)" and a `<summary>` of what it now guards. That pass found
77 of 152 passing and removed 75 tags (`OpenBugs.ILKernelBattle.cs`, `OpenBugs.BroadcastReduce.cs` and `OpenBugs.Char.cs`
now hold only fixed regression pins; the Char combos are still carved from `gen_oracle.char_tier`). Kept on purpose:
`T2_All_Contiguous_Is_Much_Slower_Than_NumPy` (a wall-clock limit that fails in a Debug build) and
`BufferReleaseSweepTests.KnownDeferredReleases_StillDeferring` (the tag is the empty backlog's switch). **Discovery trap:**
MSTest's `TestClassAttribute` is NOT inherited, so a class with `[TestMethod]`s and no `[TestClass]` of its own is never
discovered — no filter runs it, tagged or not, and nothing reports it. The TUnit → MSTest migration (ac020336) left seven
such classes (66 tests); the five with `[OpenBugs]` tests got `[TestClass]` in that pass, and the two left are
`[LargeMemoryTest]`-only (`DirectILKernelGenerator_LargeArray_BattleTest`, `Unmanaged/AllocationTests`). Find them by
reflection over the built `NumSharp.Tests.dll` (types declaring `[TestMethod]` methods without the attribute), not by grep.

## Differential-Fuzz Pipeline (NumPy oracle)

Proves every NDIter-backed op is **bit-identical** to NumPy 2.4.2 across the input space. NumPy is the oracle: Python generates a **committed, bytes-exact corpus**; the C# harness replays the operand bytes and bit-compares — **no Python at test time or in CI**.

```
test/oracle/                          corpus generators (NumPy 2.4.2 — run by hand / nightly soak)
  layout_catalog.py                   memory-layout builders (the 40 variations: 26 single + 9 pair + 5 where)
  gen_oracle.py                       deterministic op matrices (363 corpus op keys across value/error/kind/artifact tiers);
                                      per-mode dtype axes widened to ALL_DTYPES; Char woven into 20 tier files
                                      via the uint16 proxy (char_tier, relabelled uint16->char)
  gen_decimal_oracle.cs               INDEPENDENT C# oracle for Decimal (no NumPy analog): naive scalar
                                      System.Decimal math -> decimal_{unary,binary,reduce,scan,power,
                                      varstd,matmul,astype,stat,where,sort,manip}.jsonl
  gen_index_oracle.py                 getter/setter index oracle (token index over 15 base recipes)
  fuzz_random.py                      seeded random fuzzer (imports the other two)
  gen_npy_oracle.py                   .npy/.npz format oracle: REAL np.save/savez output + a manifest ->
                                      NumSharp.Tests/IO/corpus/npy_oracle.zip (286 cases)
  interop_write.cs                    the reverse direction: NumSharp writes, verify_npy_interop.py reads
  verify_npy_interop.py               manual gate — real NumPy reads NumSharp's output (needs Python + SDK)
test/NumSharp.Tests.Oracle/Fuzz/          C# replay harness (no Python)
  FuzzCorpus.cs                       rebuilds EXACT NDArray views from (dtype,shape,strides,offset,bytes)
  OpRegistry.cs                       op-name → NumSharp call (pairs 1:1 with gen_oracle.py)
  BitDiff.cs / Shrinker.cs            bit-exact compare (NaN tokenized; Decimal by canonical VALUE so 1.0m≡1.00m) / shrink a failure to 1 element
  MisalignedRegistry.cs               the documented, excused divergences (intended differences and excused cases)
  FuzzCorpusTests.cs                  one [FuzzMatrix] test per op-corpus file (checks dtype + shape + bytes + error parity)
  IndexOracleTests.cs                 index get/set differential gate (curated + dtype + seeded-random tiers)
  MetamorphicTests.cs                 oracle-free invariants (round-trips / involutions / identities — no NumPy)
  HarnessSelfTests.cs                 proves the harness has teeth (BitDiff detects value/NaN/-0 diffs; non-vacuous)
  corpus/*.jsonl                      committed corpus (116,339 rows / 64 files; ordinary op 103,208 +
                                      index 12,426 + Decimal 703 + two host pins; 5,506 Char proxy rows),
                                      copied to test output via the csproj glob
test/NumSharp.Tests/IO/            .npy/.npz format gate (no Python)
  NpyOracleCorpus.cs                  opens npy_oracle.zip, rebuilds arrays from the manifest
  NpyOracleTests.cs                   one [NpyOracle] test per claim: read / byte-exact write / header-only
                                      write / verbatim error / round-trip / npz / live-view write /
                                      hostile-allocation / corpus non-vacuity
  NpyLargeFileTests.cs                the 2 GB and 4 GB walls; the 5 GiB pair is [HighMemory] (not in CI)
  AllocationTests.cs                  MinAllocated: best-of-N, since GetTotalAllocatedBytes is process-wide
  corpus/npy_oracle.zip               committed real-NumPy output (286 cases), copied via the csproj glob
  NumpyCompatibilityTests.cs          hand-made NumPy fixtures (test_compat/) — v2.0/v3.0/fortran/scalar/
  NpyFormatVersionTests.cs            empty/bool/multi/compressed. They resolve test_compat RELATIVE to the
                                      working dir and must fail loudly when it is absent: they used to
                                      `return` silently, and nothing copied the fixtures, so on a clean
                                      checkout all 13 passed while asserting NOTHING. The csproj now copies
                                      test_compat\* — do not drop that rule.
```

- **Generators live in `test/oracle/`** and write the corpus into `test/NumSharp.Tests.Oracle/Fuzz/corpus/` (path resolved relative to `test/oracle/`). CI replays the committed corpus, never the generators.
- **Result kinds & error messages.** `expected.kind` widens the corpus past "one array": `array`
  (default — dtype+shape+bytes), `scalar` (a C# scalar wrapped 0-d), `dtype` (compared by NumPy
  dtype NAME — gates the promotion table directly), `text` (printing, verbatim) and `tuple` (N
  slots, **arity asserted**). `error: {type, text}` records NumPy's exception verbatim, so error
  cases assert the actual MESSAGE, not merely that something was thrown (the legacy
  `expects_throw`-only cases keep the weaker assertion). Three tiers use them: **`iter.jsonl`**
  (4,611 — materialized `ndindex`/`ndenumerate`/`nditer`/`nested_iters`/`broadcast` TRACES: the value stream,
  multi_index, c_index/f_index and external_loop chunk lengths across order C/F/A/K × 22 layouts;
  traversal ORDER has no other gate, and every other tier depends on it), **`dtype_text.jsonl`**
  (2,618 — promotion/dtype predicates, printing, scalar predicates and `nonzero` at any rank),
  **`multioutput.jsonl`** (64 — full tuple arity + every slot for split/unstack/broadcast/meshgrid/
  modf/average/unique), **`creation.jsonl`** (302), **`conversion.jsonl`** (1,078), and
  **`errors_full.jsonl`**
  (688 raising cells over 22 distinct NumPy messages — the cells every value generator SKIPS) and
  **`out_where.jsonl`** (3,727 — ufunc `out=`/`where=` over 7 out layouts × 9 mask layouts × 28
  ufuncs; each case records BOTH the returned view AND the entire base buffer behind `out`, so
  masked-off slots are proven to keep their prior contents and a kernel writing outside a
  strided/offset/negstride window is caught — previously reached only through `maximum_out`/
  `minimum_out`/`clip_out`, 11 contiguous unmasked cases each).
  Comparators: `FuzzCorpusTests.Kinds.cs`; callables: `OpRegistry.Kinds.cs`; ledger: `Fuzz/README.md`
  → Table 0.
- **`FuzzMatrix` layers**: `FuzzCorpusTests` (ordinary managed kernels + host-pinned backend tiers +
  the deduplicated backend delta), `IndexOracleTests`, `MetamorphicTests`, harness self-tests, and
  `OracleSurfaceCoverageTests`. The public-surface guard inventories `np` 321 / `np.linalg` 31 /
  `np.fft` 18 / `np.random` 48 and rejects an unclassified new API. A failing value case auto-shrinks.
- **Dtype coverage**: per-mode dtype axes widened toward `ALL_DTYPES`. **Char** (no NumPy dtype) contributes
  5,506 uint16-proxy rows across 20 files. **Decimal** (no NumPy analog) rides 703 cases from the
  independent C# scalar oracle. Real product bugs are carved+pinned or fixed, never registry-excused.
- **Regenerate** (deterministic; needs `numpy==2.4.2`): `python test/oracle/gen_oracle.py <mode>`
  (modes include `conversion creation multioutput iter dtype_text out_where errors_full smoke astype_full
  binary divmod_power comparison unary reduce where place matmul rounding bitwise unary_extra nanreduce
  scan stat logic modf manip sort tail params aliasing copyto copyto_full power_fast weak_scalar errors groupa numpy_f32
  matmul_parity linalg_parity poly einsum specials precision products random_parity fft`; `precision` additionally
  needs `mpmath`) + the index/Decimal/NPY generators, then `dotnet build` to copy corpus assets.
- **NEP 50 weak Python scalars** (`weak_scalar.jsonl`, 30,351 cases, 2026-10-07/08, `gen_oracle.py weak_scalar`): the six
  comparisons and the binary ufuncs with one operand the C# literal NumPy's Python int / float / complex / bool maps to
  (plan api-parameter-parity X24 / X25 / X2 / X21) — every dtype against the scalars where the precision decides, both
  sides, np.* and both operator overloads, out= / where=, the iterator layouts, ints past 2^53 into float16 / float32
  loops (two roundings), weak ints into integer loops (NumPy's OverflowError), long arrays through the
  scalar-broadcast kernels' packed vector loops (section W), empty broadcasts, and section S (15,301, plan X27): the
  literal STORED at every store entry point (the seven indexer routes, fill, full, full_like, put, putmask, place,
  put_along_axis, insert, fill_diagonal, pad, the flat iterator, asarray, copyto), Python scalars and the NumPy scalars
  Half / char (whose route decides int() vs a cast). Every other tier passes arrays, which NumPy treats as strong. Held
  to ZERO excuses; 5 of 5 planted store bugs killed. Generator drops: a where= comparison of an integer array with an
  out-of-range Python int (NumPy 2.4.2 segfaults there), a bool array against an int past int64, a store error naming
  the platform C long ("too large to convert", "too big to convert": the unit tests pin the LP64 texts), np.copyto's
  complex64 cast text.
- **np.power's float loops** (`power_fast.jsonl`, 7,902 cases, 2026-10-06, `gen_oracle.py power_fast`): np.power as
  NumPy's DOUBLE_power / FLOAT_power / HALF_power and np.float_power (plan api-parameter-parity X13 / X19 / X20) — every
  float base layout × 0-d / one-element / (1,1) / broadcast / Python-scalar exponents, integer / bool / complex bases,
  the trivial loop's cast copies and NpyIter's buffering threshold at `np.setbufsize(256)` and the default, varying
  exponents (columns, middle axes, rows, strided, extra leading axes), out= / where= layouts, dtype=, float_power, two
  boundaries a mutation pass found (one-element operands with a cast out; rows of exactly bufsize with every input cast),
  empties and errors. Bases hold values on which the fast path and pow differ. Held to ZERO excuses (`MisalignedRegistry`
  returns null for the op); 1,785 of its first 7,773 cases differ on the parent `6f00ae14`. Generator drops (other
  items): a 0-d float32 / float16 base with a Python-scalar exponent (the NEP 50 0-d divergence), complex bases with
  non-integer exponents (F5), unsigned bases with -1 (X21), an out= that does not broadcast (X22).
- **Truthful vs precise** (`precision.jsonl`, 72 truth-bearing cases): each case carries `expected.truth` — the correctly-rounded mathematical reference (exact `Fraction` / 200-bit mpmath, generator-side only) — over precision-ADVERSARIAL inputs (wide-magnitude/cancellation sums at N≤2049, large-mean variance, near-1 products, expm1/log1p small-|x|) the ordinary 8–36-element pools cannot express. Policy (the vision is byte-identical NumPy parity): **bit-exact to NumPy passes without truth ever being read — precise never fails**; truth only adjudicates divergences. Not-less-truthful than NumPy → excused "prefer-precise" parity debt (being MORE accurate than NumPy is still a divergence to close by porting NumPy's algorithm, never a win); less truthful beyond 4×/+8 ULP slack → precision LOSS, red unless a bounded known-bug branch covers it (`MisalignedRegistry` P1–P3; the unbounded summation blanket is gated on truth-absence so losses can't hide in it). Findings on arrival, excused bounded ≤256 ULP (P3): f32 var/std accumulation 55/26 ULP vs truth (NumPy 3/2), negative-stride reduce path 11–32 ULP (NumPy exact).
- **Reduction start** (`reduce_start.jsonl`, 8,666 cases; 7,942 at delivery, 2026-10-04): every reduction, scan and product over all-`-0.0` operands (size 1, length-1 axes, C/transposed/reversed/stepped, f64/f32/f16/c128), so the SIGN of a zero result shows whether NumSharp starts the way NumPy does (`add.reduce` from the `+0.0` identity, accumulate copying element 0, complex `prod` times `1+0j`, `cblas_matrixproduct`'s size-1 scalar branch, the default `einsum`'s zero-filled output). Held to ZERO excuses (`MisalignedRegistry` returns null for its `negzero` valueclass). The cells NumSharp gets wrong are the generator's carve (`_rs_carved`, one rule per root cause, each naming its `[OpenBugs]` pin in `OpenBugs.Reductions.cs`); drop a rule and regenerate when its path is fixed — the identity-seeded add reduction dropped the sum / mean / nan-family / trapezoid / np.evaluate-sum rules the same day (1,506 → 862 carved cells, exactly the cells that still differ), and np.var / np.std as NumPy's `_var` dropped the one-element std / var dtype rule on 2026-10-05 (838 carved cells), np.nanvar / np.nanstd as NumPy's `nanvar` the nanstd / nanvar shape rule (782). Finding: the nanreduce tier's `NanReduceOps` known-bug branches excuse 1,483 of 10,416 cases, and their value branch hid float16 `nansum`/`nanprod` wrong-element reads.
- **var / std** (`varstd.jsonl`, 16,108 cases, 2026-10-05, `gen_oracle.py varstd`): np.var / np.std as NumPy's `_var` / `_std` over every layout × dtype × axis × keepdims × ddof, `ddof` past the count, `dtype=` × 10 loops, special values, `np.setbufsize` 16 / 32 fills (`params.bufsize`) and order-sensitive sizes. Held to ZERO excuses (`MisalignedRegistry` returns null for its `varstd` valueclass, and var / std are in `ByteExactArithmeticUnaryOps`, so no blanket takes them in any tier; `ma.var` / `ma.std` too since A6). `VarStdChunked` replays it through the chunked deviation sums. Carve: 12 cast-loop cells whose pinned rows are longer than the buffer (np.sum's X6 residual, `OpenBugsReductionTests.Sum_CastLoopAxis_RowLongerThanTheBuffer_SplitsAcrossFills`).
- **nanvar / nanstd** (`nanvarstd.jsonl`, 13,382 cases, 2026-10-05, `gen_oracle.py nanvarstd`): np.nanvar / np.nanstd as NumPy's `nanvar` / `nanstd` over every catalog layout (refilled with finite values and NaNs) × dtype × axis × keepdims × ddof, NaN patterns (one, all, an all-NaN slice per axis, inf, a NaN complex part), `ddof` past uneven non-NaN counts, `dtype=` × 7 loops, special values, `np.setbufsize` fills, order-sensitive sizes, empty inputs and 0-d axes. Held to ZERO excuses (valueclass `nanvarstd`; nanvar / nanstd left the registry's nan-family branches, `NanReduceOps`, and joined `ByteExactArithmeticUnaryOps`). Carve: the same 12 cast-loop cells as `varstd` (np.sum's X6 residual).
- **mean** (`mean_loops.jsonl`, 6,550 cases, 2026-10-05, `gen_oracle.py mean_loops`): np.mean as NumPy's `_mean` over every catalog layout × dtype × axis × keepdims, `dtype=` for every input against nine loops (float16 / float32 / float64 / complex128 / int8 / uint8 / int32 / int64 / bool), empty and 0-d inputs, special values (complex parts, uint64 past 2**63, broadcast), order-sensitive sizes (float offsets; int64 near 2**62) and `np.setbufsize` fills. Held to ZERO excuses (valueclass `meanloops`; mean is in `ByteExactArithmeticUnaryOps` and out of the reduction blankets in every tier, `ma.mean` too since A6; one narrow branch takes np.sum's finding-5 layout in the older tiers). Carves, each pinned: finding 5 (16 cells), X6 (5), the broadcast flat schedule (24).
- **nanmean** (`nanmean.jsonl`, 5,017 cases, 2026-10-05, `gen_oracle.py nanmean`): np.nanmean as NumPy's `nanmean` over every catalog layout (refilled with finite values and NaNs) × dtype × axis × keepdims, NaN patterns (one, all, an all-NaN slice per axis, NaN / inf complex parts), `dtype=` for every input against nine loops, special values, `np.setbufsize` fills, order-sensitive sizes, empty and 0-d inputs. Held to ZERO excuses (valueclass `nanmean`; nanmean left `NanReduceOps` and joined `ByteExactArithmeticUnaryOps`; the nan-family shape branch is removed). Carve: 5 cast-loop cells (np.sum's X6 residual).
- **ma mean / var / std / anom** (`ma_meanvar.jsonl`, 26,379 cases, 2026-10-06, `gen_ma_oracle.py meanvar`): numpy.ma's statements (plan A6) over every catalog layout × the 13 dtypes × nomask / all-False / some masks, the mask classes (all masked, one unmasked element per row, a masked first row) with keepdims and ddof 0 / 1 / 2 / 5, `dtype=` in eight loops, the `mean=` forms (Python float / int / complex, np.float32 / np.float64, plain / masked / 0-d arrays, a non-broadcastable one), special values (non-finite, overflowing squares, -0.0, integer extremes), sizes past the pairwise block, empty and 0-d inputs, and 294 NumPy errors (`expects_throw`, type + text — the first ma tier with error cases). Each case also records `raw` (the data at EVERY slot, masked ones included) and `form` (`masked` singleton / `scalar` / `nomask` / `mask`), which `FuzzCorpusTests.Ma` compares; held to ZERO excuses (valueclass `mameanvar`: the registry returns null and the ma harness's own excuses skip it). Carve (`_mv_carved`): 165 cells whose mean sums the operand view (no mask or an all-False one) on a layout np.sum still orders differently — finding 5 (38) and broadcast operands (127, finding 10 / X12) — each only where the sum is not order-free; `python gen_ma_oracle.py meanvar --uncarved <path>` writes the whole matrix for the check that every carved divergence is one of those (64 of the 165 differ).
- **np.copyto** (`copyto_full.jsonl`, 13,462 cases, 2026-10-06, `gen_oracle.py copyto_full`): np.copyto as NumPy's `array_copyto` (plan api-parameter-parity X16 / X11) over 13 dtypes × destination and source layouts × 14 mask forms (dense, random, row, column, strided, reversed, 0-d, the C# literals), long runs at every element width, 16 cast pairs, 1,100 overlapping views of one buffer (masked and unmasked), C# literal sources as NumPy's Python scalars (weak ints / floats / complexes, the strong bool, every casting rule), 0-d and empty destinations, and 43 error jobs that gate NumPy's check ORDER. Every case records the destination's whole base buffer. Held to ZERO excuses (`MisalignedRegistry` returns null for the op). Generator drops: a Python int at or past 2**31 that NumPy rejects with its platform C `long` text (LP64 pinned), and complex64 texts.
- **The array-object oracle** (`test/oracle/gen_viewflags_oracle.py`, 2026-10-07; README "The array-object oracle"): four tiers gate the OBJECT an API returns — values plus a `mem` facet (C / F / OWNDATA / WRITEABLE / ALIGNED flags, byte strides, view-of / IS each operand, tuple slot sharing) — `view_flags.jsonl` (`viewflags`, 194 APIs × 21 source layouts), `out_flags.jsonl` (`outflags`, 149 out-capable APIs × 6 out layouts: returns the out ITSELF, writes only its slots), `shape_args.jsonl` (`shapearg`: the same dims as `Shape` / `long[]` / `int[]` / `int` / a view's Shape), `array_args.jsonl` (existing op keys, 55 SECONDARY operand positions × 9 layouts). Operands rebuild as owners (`own: true`) or read-only (`writeable: false`). Zero excuses; every known divergence is CARVED by `_gap_of` into `corpus/gaps/<tier>_gaps.jsonl` (34 gap ids, 474 cells since the rebase of strings onto journey4 on 2026-10-12, whose fixes closed VF03, VF05, VF11, VF18, VF23-VF25, VF28, VF33, VF35, VF39, VF40 and narrowed VF01, VF08, VF12, VF26, VF32 — 46 ids and 705 cells before, since VF15 (np.asmatrix returning its operand) was fixed with NDMatrix on 2026-10-09; exact: each diverges, no kept case does), replayed by the `*_KnownGaps` `[OpenBugs]` tests, and the ones with consequences reproduce in `NumSharp.Tests/OpenBugs.ViewFlags.cs`. **Worst class found: C-contiguous views at a non-zero offset (`np.split` / `np.unstack` / `np.array_split` children) are read from the parent's start** by `np.searchsorted`, `np.where`, `np.ldexp`, flat `np.nansum` / `np.nanprod`, `np.repeat`, `np.unique` and the set routines (silent wrong values; `np.repeat`, `np.unique` and the set routines read the view right since the strings rebase); `a[:, idx]` on one throws. Coverage join: `oracle_map.json` `viewflags` / `outflags` / `shapearg` param keys (`params.fn`). The leak sweep labels these keys' families `op:fn`, and the leaks they exposed sit in `KnownEscapes` with ceilings. Portability: libm-backed out= rows take zeros, FFT lengths stay in {2, 3, 4, 6}.
- **The relayout tiers** (`test/oracle/gen_relayout_oracle.py`, 2026-10-07; README "The relayout tiers"; run it AFTER every other generator, its sources are their files): every (op key, operand position, dtype class) cell of the corpus is surveyed for 19 ndarray KINDS (C / F / permuted / stepped, each also at an extra offset; reversed; stepped-and-reversed; broadcast and one-element broadcast, also at an offset; 0-d owner; 0-d view at an offset; one-element 1-D; empty; empty view inside a real buffer; + read-only), and each missing cell is filled by re-emitting a source case that a plain NumPy call reproduces byte for byte, one operand moved into the kind, NumPy's answer recomputed (derived-value kinds only for the elementwise / reduction families; an emptied operand fills a value cell AND a "could not be broadcast" cell). `relayout.jsonl` (15,068; 14,964 until the strings rebase of 2026-10-12 returned RL04-RL06's 104) + `relayout_host.jsonl` (1,728, via `RunHostLibmCorpus`). **Two validation holes it closed:** a variant inherits its source tier's excuses, and unbounded ones hid layout misreads — `RelayoutLayoutInvariance` holds, with NO registry, that NumSharp's outcome equals its own outcome on a C-ordered copy (`FuzzCorpus.ReconstructCanonical`, built by index arithmetic) wherever NumPy's does not change (60 `layout_sensitive` variants skipped); and the NEP50 0-d branches excuse ANY dtype/value difference of a case with a 0-d operand — the relayout tiers are exempt (`MisalignedRegistry.IsRelayoutCase`), the generator dropping the ufunc / np.where variants whose answer depends on weak 0-d promotion. Found (carved exactly by emulating each bug into `gaps/relayout_gaps.jsonl`, 179 cells, 7 ids, repros in `NumSharp.Tests/OpenBugs.Relayout.cs`; 75 cells under RL01-RL03 and RL07 since the strings rebase of 2026-10-12 fixed RL04-RL06, whose cases moved back into the gate without a regeneration): the C-contiguous-offset misread class reaches np.digitize, complex/float16 np.isin, float16/complex np.nancumsum/nancumprod, np.ldexp's mantissa, np.unique_values, both set-routine operands and 0-d `ndarray.repeat` (RL01–RL05); the NDArray comparison operators with an EMPTY left operand skip broadcasting and leak a buffer (RL06), np.logaddexp/logaddexp2 return a clone of an empty operand (no broadcast error, no float promotion; RL07). F / permuted / stepped / broadcast views at an offset and empty views showed NO misread anywhere. **Harness change:** an empty operand descriptor with a buffer now rebuilds as an empty VIEW over it.
- **The surface tier** (`test/oracle/gen_surface_oracle.py`, 2026-10-08; README "The surface tier"): contracts for the APIs the coverage inventory listed as available but unproven — np.linalg cross / diagonal / outer / trace (the Array-API forms), the histogram family, array_equiv, shares_memory / may_share_memory, broadcast_shapes, apply_along_axis / apply_over_axes, bmat, the text formatters, the ndarray members (tolist, tofile, setfield, base, dtype, shape, device, to_device, data), the finfo / iinfo / dtype / ndarray.flags / poly1d attributes, the np.broadcast / flatiter / np.nditer members after k steps, the print and buffer options, dtype.type, frompyfunc and the inventory's aliases (numpy.cumulative_sum / cumulative_prod -> np.cumsum / np.cumprod, numpy.shape). Inputs the operand corpus cannot express ride params: a FUNCTION is a name from a fixed menu both sides map to one computation, two VIEWS of one buffer are step recipes applied to one owner, global options are set and restored inside the case. Op keys are the API names (`coverage/oracle_map.json` prefix rules `linalg` / `finfo` / `iinfo` / `dtype` / `poly1d` / `broadcast` / `flatiter` / `nditer`), so after it no in-scope available row lacks a contract. `surface.jsonl` (15,866 over 139 op keys, valueclass `surface`, zero excuses) + `gaps/surface_gaps.jsonl` (602 cells under 28 ids, carved exactly by modelling each bug; 986 under 42 until the rebase of strings onto journey4 on 2026-10-12 fixed SU1, SU25-SU34, SU40-SU42 and SU4's 'auto' / 'fd'; 42 repros in `NumSharp.Tests/OpenBugs.Surface.cs`, 16 of them untagged regression pins since). The second pass (same day, from the inventory's remaining rows) added poly1d.r / poly1d.roots (np.roots' strip-and-linear paths over 9 coefficient sets x 5 dtypes x 4 kinds, no LAPACK), np.nditer's iteration protocol (members after k Python `next()` calls) and close()'s write-back contract (updateifcopy / buffered / external-loop / plain operands x 6 kinds), which found SU41 (an updateifcopy operand of another dtype is refused without buffering; NumPy iterates a copy close() writes back) and SU42 (the protocol is one element ahead and a buffered readwrite loop loses the last write of each buffer). Worst findings: np.cumsum / np.cumprod with dtype=complex128 over integers return the input's raw bytes, np.cumsum(dtype=bool) stores counts, np.finfo's float32 / float64 minexp / resolution / tiny are wrong, np.nditer's K order over an F operand plus a broadcast one walks C (it sorts axes by the smallest stride; NumPy lets a stride-0 operand abstain) and remove_multi_index re-sorts, the float formatters print `Infinity.` and keep carried zeros, apply_along_axis crashes on a stride-0 view at Shape.offset k, np.nditer.remove_axis crashes under order='F'. Leaks recorded in `KnownEscapes`: np.MemoryView.tobytes of a non-contiguous array (np.array_equiv with an empty operand was the other, gone with the strings rebase). The relayout generator skips this tier. TRAP (found by the second pass): `NDArray.SetAtIndex<T>(T, long)` checks T against the dtype only with a `Debug.Assert` — in Release a double written into a float32 array lands as 8 bytes over the NEXT two elements (past the buffer's end for the last one); a test adapter must write each dtype with its own type (`OpRegistry.Surface.Objects.cs` `SetElement`; pinned in `OpenBugs.Surface.cs`). TRAP: a corpus regenerated with `--uncarved` and copied into `bin/.../Fuzz/corpus` stays there after the carved regeneration (copy-if-newer keeps the newer bin file), so the gate replays the uncarved file and fails — copy the carved corpus and gap files into both `bin/Release/{net10.0,net8.0}` (or rebuild) after an uncarved replay. What the inventory still lists as available but unproven after this pass, by design: numpy.ma.polyfit (needs a LAPACK backend), numpy.nditer.debug_print (a C-level stdout dump) and the six `BitGenerator` / engine `.lock` rows (a `threading.Lock`; the random-API oracle exempts it).
- **The numpy.ma surface tier** (`test/oracle/gen_ma_surface_oracle.py`, 2026-10-08; README "The numpy.ma surface tier"): contracts for the 51 numpy.ma functions still listed as available but unproven — the creation wrappers, column_stack / dstack / hsplit / diagflat / resize, the fill-value helpers, getmask / is_mask / isarray / isMA / isMaskedArray / ndim / shape / size / nonzero / ids, harden_mask / soften_mask, masked_object, putmask, convolve / correlate / cov / corrcoef, the edge finders, compress_nd / compress_rowcols, ndenumerate, apply_along_axis / apply_over_axes (all but polyfit, which needs a LAPACK backend). Masked operands in every relayout data kind x eight mask patterns (nomask, an explicit all-False mask, mixed, all, first, last, alternating, runs). The ma harness gained `tuple` and `text` result kinds (a Python object as its normalized repr: a NumPy scalar as `.item()`, a None slice step as 1) and `RunMaGapCorpus`; masked results record the FORM and, where NumPy defines it, the RAW data at masked slots. `ma_surface.jsonl` (27,157, valueclass `masurface`, zero excuses: MisalignedRegistry and the ma harness's own excuses skip it) + `gaps/ma_surface_gaps.jsonl` (6,987 cells under 23 ids, carved exactly; 10,123 under 37 ids MS1-MS37 until the rebase of strings onto journey4 on 2026-10-12 fixed MS3, MS6-MS14, MS16, MS25, MS28, MS35 and MS2's column_stack / dstack / diagflat; 31 repros in `NumSharp.Tests/OpenBugs.MaSurface.cs`, 13 of them untagged regression pins since). Worst findings: putmask and a hard-masked setitem cast unsafely (NumPy's np.copyto raises same_kind TypeError), set_fill_value / masked_object never convert or set the fill value and common_fill_value compares raw C# values, zeros_like / ones_like / empty_like drop the mask, cov / corrcoef with a 1-D x and y under rowvar=False decide rowvar after the concatenation, corrcoef of one variable throws, ids ignores a view's offset, an all-False result mask is dropped to nomask (convolve / correlate / cov / apply_along_axis). TRAP (found here): a carve on the result FORM hides value / mask divergences behind it (the form is compared first) — `NUMSHARP_MA_SKIP_FORM=1` replays without the form check; and the ma generator's old putmask block never emitted a case (its values operand was not a view of its base, so `describe` raised into a skip).
- **The matrix tier** (`test/oracle/gen_matrix_oracle.py`, 2026-10-09; README "The matrix tier"): numpy.matrix as `NDMatrix` — the constructor (arrays of every rank, matrices, literals, sequences), T / H / A / A1 / I, the row / column / scalar rule of indexing and the assignment rule that follows from it, `*` as the matrix product with scaling in np.dot's dtype, `**`, the elementwise operators in their five C# forms, the reductions that keep two axes (dtype=, out=, ddof), the methods ndarray hands down, repr / str and iteration, over 14 source layouts and the dtypes. A result is held to its values, its CLASS (`expected.cls`), its memory facet and NumPy's error type and text; operands rebuild as `NDMatrix` (`"matrix": true` in the descriptor). `matrix.jsonl` (26,789 cases over 87 op keys, valueclass `matrix`, zero excuses) + `gaps/matrix_gaps.jsonl` (144 cells under 9 ids, carved exactly; 172 under MX01–MX11 until the rebase of strings onto journey4 on 2026-10-12 fixed MX06 and MX11; every gap is an `NDArray` member or `np.*` function below the class; 7 repros in `NumSharp.Tests/OpenBugs.Matrix.cs`). Worst findings: np.dot of an N-D operand with an M-D one computes complex128 in float64 (the imaginary parts are lost), the flat np.prod in a cast loop returns NaN or throws, a two-entry fancy assignment into a C-contiguous view at an offset throws. The tier is replayed once per float scaling route: `Matrix` (production), `Matrix_FusedScaling` (`NDMatrix.DisableDirectScale` + `FusedScaleMinElementsTestOverride = 2`), `Matrix_TwoPassScaling` (both single-pass routes off). TRAP (found here): a carve rule written from a failure summary over-carves error cases (NumSharp raises NumPy's text before it reaches the bug) — check the rule set against an uncarved replay's failure dump in both directions (`check_carve.py` pattern: failing but not carved, carved but passing).
- **The matrix-operand tier** (`test/oracle/gen_matrix_operand_oracle.py`, 2026-10-10; README "The matrix-operand tier"; run it AFTER every other generator, its sources are their files): the public `np.*` functions with a `numpy.matrix` OPERAND. Existing corpus cases that a plain NumPy call reproduces byte for byte are re-emitted with one two-dimensional operand typed as a matrix (`"matrix": true`, layout `matrix@<position>`), and NumPy's answer is recomputed: its class (`expected.cls`), dtype, shape, values, or its error type and text. 4,645 NumPy answers over 260 functions (680 of them errors); budget 2 cases per (op, position, dtype class, param keys, axis / keepdims) and 8 per (op, position, axis / keepdims). `matrix_operand.jsonl` (3,885) + `matrix_operand_host.jsonl` (420, via `RunHostLibmCorpus`) + `gaps/matrix_operand{,_host}_gaps.jsonl` (340 cells under MO01 to MO05, carved exactly: every carved case diverges, no kept case does; `MatrixOperand*_KnownGaps`). The CLASS is compared without a registry call, and `MisalignedRegistry` returns null for a shape, threw, error-text or expected-error divergence of a case whose layout starts with `matrix@` (the source tier's value excuses stay; the NEP50 0-d blankets are off). MO01: `np.isclose` (typed return). MO02: `np.count_nonzero` with an axis (NumPy hands `keepdims` to `matrix.sum`). MO03: the median / percentile / quantile family (NumPy's keepdims and ravel paths meet matrix methods). MO04: `np.roll`, `np.ediff1d`, `np.vdot`, `np.union1d`, a refused `np.append` (`ravel()` of a matrix stays 1 x N). MO05: `np.polyval`, `np.i0`, narrow-float `np.sinc`, `np.trapezoid` with a matrix x (`*` is the matrix product inside NumPy). The registry calls the function form for operator keys (`OpRegistry.MatrixOperand.cs`). TRAPS: NumPy 2.4.2's `np.stack` / `np.dstack` with a matrix operand raise and CORRUPT NumPy's heap, so the generator makes every matrix call in a worker process it retires after such an error; `CheckError` reports "NumPy raises but NumSharp produced a result" as a VALUE divergence, so a source tier's value excuse hid error-parity gaps until expected-error cases were made strict for this tier; an uncarved truth table copied into `bin/.../Fuzz/corpus` stays there after the carved regeneration (copy the carved files back before running the gate). The caller audit `MatrixOperand_LibraryNeverReachesAMatrixRule` (above, "the matrix boundary") is the tier's second half. Functions outside the truth table took their rule from NumPy's source and two direct probes; they are pinned by `Matrixlib/NDMatrixBoundary.Tests.cs` (24) and, for the LAPACK functions, `NumSharp.Tests.Interop/MatrixOperandLapackTests.cs` (5).
- **Products & random streams**: `products.jsonl` (408) gates the CBLAS family, tensor products,
  cross/cov/corrcoef, and Array-API vector/matrix norms. `random_parity.jsonl` (38 portable) +
  `random_parity_host.jsonl` (108 host-libm) pin 35 stream methods/distributions; only managed
  `multivariate_normal` (no LAPACK backend) stays carved and `[OpenBugs]`-pinned (the other seven samplers were
  uncarved 2026-09-25).
- **Random-API oracle** (`test/oracle/gen_random_oracle.py`, plan `docs/plans/random-oracle-coverage.md`, README
  "The random-API oracle"): every public member of the random world (493 in `test/oracle/random_surface.json`,
  G1-pinned to reflection) replayed by EXACT C# signature (`params.sig`, reflection-invoked; an omitted optional is
  the declared default, NumPy is called without it) on every engine x the 10 fixed seeds, recording NumPy's result
  AND the receiver's full post-call state. Tiers `random_api` (portable 9,820) / `_host` (win-amd64 libm 15,795) /
  `_mvn` (NumPy's own scipy-openblas 730) / `_lp64` (Linux NumPy via WSL 234). Reflection means the replay cannot
  see CALL SHAPES — whether NumPy's spelling compiles in C# and binds the overload it proved; those are pinned
  compile-time by `RandomSampling/RandomApiWholeness.Test.cs` (the wholeness pass, see "Random"). Gates `RandomApiCoverageTests`
  (corpus-only): G2 overloads, G3 parameters (omitted + non-default, two values, null + non-null, `params` 0 and 2+,
  every accepted enumerated value + one rejection, NumPy's parameter NAMES and order vs
  `test/oracle/random_numpy_signatures.json`), G4 engines, G5 seeds, G6 state; exemptions reasoned and self-retiring.
  Nightly `fuzz-soak.yml` jobs `random-api-lp64` + `random-api-soak` replay the families under 10 fresh seeds
  (`RandomApiSoak`, `NUMSHARP_RANDOM_SOAK_DIR`). After a surface change: `NUMSHARP_WRITE_RANDOM_SURFACE=1` on the G1
  test, then regenerate.
- **Managed vs OpenBLAS without duplicate noise**: ordinary tiers always run Core managed. The
  host-pinned `matmul_parity`/`linalg_parity` tiers run backend-on. `BlasBackendDelta` replays only
  the 1,775 affected ordinary cases: 1,747 identical outcomes are deduplicated; the 28 flips are
  byte-compared with NumPy on the pinned host. Outcome equality includes threw/result, dtype, shape,
  and bytes; a backend change on a non-CBLAS dtype is always a hard failure.
- **LAPACK factorisations**: `linalg_parity.jsonl` (366) is the FIRST *corpus* value gate for the eigen/SVD/QR/Cholesky family `cholesky`/`eig`/`eigvals`/`eigh`/`eigvalsh`/`svd`/`svdvals`/`pinv`/`matrix_rank`/`cond{None,2,-2}`/`lstsq`/`qr` + `norm{2,-2,nuc}` AND the **LU family** `solve`/`inv`/`det`/`slogdet`/`tensorinv`/`tensorsolve` + `matrix_power(n<0)` (added 2026-08-21 — the OpenBLAS members that had ZERO committed oracle coverage) — **HOST-PINNED like `matmul_parity`** (`linalg_parity.host.jsonl`, same `MatmulParityPin`), because Core ships no managed LU/QR/SVD/eigensolver: they compute ONLY through `NumSharp.Interop.OpenBLAS`, byte-exact on NumPy's own pinned scipy-openblas at **threads=1**, Inconclusive off the pinned host. tuple results (svd/eig/eigh/qr/lstsq/**slogdet**) + array siblings; float32 byte-exact too (NumPy's "lite" double-compute-round). The LU family is `getrf`/`gesv` (deterministic pivoting), so EVERY output is byte-reproducible — nothing excluded (`det` 0-D/1-D, complex `slogdet` sign unit-modulus, singular→`(0,-inf)`); only the eigen/SVD side excludes complex-Hermitian `eigh` eigenvectors (heevd phase, cross-process — covered by `eigvalsh`), float32-`eig`-complex-eigenvalues (complex64 vs complex128), non-SVD `cond`/`norm` orders (1-ULP reduction-order). 366/366 bit-exact on the pinned host.
- **Polynomial / einsum / cross / cov**: `poly.jsonl` (`gen_oracle.py poly`) gates the PORTABLE polynomial family (`poly` 1-D, `polyval`, `vander`, `polyder`, `polyint`, `polyadd`/`polysub`/`polymul`, `polydiv` tuple, `poly1d`) — pure arithmetic/convolution/Horner, bit-exact everywhere. `einsum.jsonl` gates `einsum` (integer + small-exact-float contractions + the view path; NONZERO operands, since a signed zero diverges via NumPy's `+0.0`-seeded `sop` accumulator) + `einsum_path` (the info STRING, `text` kind, shape-derived, non-ellipsis). The `products` tier gained **`cross`** (the lone product-family gap; multiply-subtract, bit-exact f64/f32/c128/i64) and **`cov`/`corrcoef`** (normalized dot, byte-exact SMALL/unweighted; weighted `fweights`/`aweights` is 1-ULP off → battle-tests). The backend polynomial ops (`roots`/`polyfit`/`poly`-of-a-matrix, via `eigvals`/`lstsq`) ride the host-pinned `linalg_parity` tier. The pure single-operand ops are carved out of the "unary ~ULP" blanket excuse so a ≤2-ULP drift fails.
- **Ownership (the deterministic over-disposal)** (`Fuzz/Ownership/` + `UndisposedIntermediateTests.{Ownership,ForeignMemory}.cs`, 2026-10-09): no call may dispose, release, under-count, swap or pool memory it does not own — its caller's operands / masks / index and value arrays / `out=` targets, the leak fixtures, the process-wide shared arrays (`np.ma.nomask`, the masked constant, the 24 numpy.polynomial module constants). The leak sweep's pool balance CANNOT see it: `NDArray.Dispose` is idempotent and every sweep family runs an unmeasured warm call first, so an op that disposes its operand returns the buffer there and the measured window balances (`PoolBalance_CannotSeeADisposedOperand`). `OwnershipLedger` snapshots each array (its block's ARC kind / counted references / freed flag through `ArcProbe`'s compiled accessors on the private `Disposer`) and checks it after every region of every case in all four leak runners (corpus 669,232 cases, backend 547,010, catalogue 438, property reads 614), plus a `PoolScan` of the 64 most recent buffers of the caller memory's bucket sizes (a full-pool scan made the sweep 25× slower). `Corpus_ForeignOperands_StayTheCallers` replays one case per (op, layout, dtype) with every operand rebuilt over six `ForeignKind`s — pooled, a pinned caller `T[]`, `np.ndarray(buffer: byte[])`, `np.frombuffer`, a non-owning raw pointer, caller memory with a release action — 446,268 runs in ~6 s, counting the caller's release action. Found: no library over-disposal; two HARNESS ones fixed (the sweep disposed polynomial constants returned as results; the masked axis_arg replay disposed its own operand). `ReferenceRetained` (a reference to the caller's memory outlived the call: `array_str` leaves 8 views, char `min`/`max` a `uint16` alias) is printed, not judged. Planted bugs: 4 of 4 caught, the leak gate blind to 3. Ledger: `test/NumSharp.Tests.Oracle/Fuzz/README.md` "Ownership gate".
- **Lifetime (lost GC anchors, the GC-driven over-disposal)** (`Fuzz/Lifetime/`, 2026-10-09, `test/oracle/gen_lifetime_oracle.py`): a method that reads an operand's raw pointer and runs its kernel with no later use of the operand lets optimized code drop the operand at the pointer read; when the caller passed a temporary, a collection during the kernel (another thread, or the pool's GC pacing) finalizes its memory block and the buffer is pooled or freed under the kernel: silently wrong values, spurious `IndexError`s, fatal access violations. No other tier can see it (they hold their operands). Cases carry operand RECIPES (2-8 MB; NumPy's answer as dtype/shape/SHA-256) in their own directory (`Fuzz/Lifetime/corpus/`, outside the generic `Fuzz/corpus/` sweeps); the replay runs the Oracle assembly as a CHILD process with `DOTNET_TieredCompilation=0` (tier-0 keeps every local alive and hides the bug; a crash ends only the child, which is resumed after the crashed case; entry point `LifetimeChildProgram.Main`, so the csproj sets `GenerateProgramFile=false`), builds each `temp` operand inside the call expression, and a stress thread collects, drains finalizers and poisons the buffers the finalizers return. A control (the bug pattern by hand) must be caught or the run is inconclusive. **Release only**: a Debug build of NumSharp.Core or the Oracle assembly keeps every local live too (its JIT optimizer is disabled whatever the tiering), so the tests are inconclusive there at once — run them with `dotnet test -c Release` (CI does). Gate `LifetimeOracleTests.Lifetime` (`[FuzzMatrix]`, 350 calls, zero divergences, ~20 s per framework); 108 of its cases PIN (`pin`, every one-operand-at-a-time variant kept) the 28 gaps this tier found and the weaver's operand roots FIXED on 2026-10-09 (LT01–LT28: take/put/place/putmask/along_axis, where(c,x,y), copyto where= and every `TensorEngine.Cast` via `NDIter.CopyAs` (LT09: `NDIter.Copy(NDArray, NDArray)` passes only the storages on), max initial=+where= and float16, flat cumsum/cumprod/nancumsum, searchsorted/digitize/bincount, integer reciprocal, the managed dot/inner/matvec/vecmat/2-D dot and LU, convolve/correlate, polyder, polyvander, bool `!`, flat `all`, ToMuliDimArray, the unique family's hash path); `Lifetime_CorpusContract` keeps every fixed gap pinned, and a red pinned case is reported as `FIXED gaps regressed: LTnn`. Open gaps (none) are carved into `lifetime_gaps.jsonl` (`GAPS` in the generator; `FIXED` holds the 28). **Scaled tier** (`Fuzz/Lifetime/LifetimeScaled.cs`): every op key of the ordinary and masked-array corpus (up to 2 cases of distinct layouts each; a masked case's operands are `np.ma.masked_array(scaled data, scaled mask)`, its result compared by mask + unmasked data) with its operands enlarged into ~2 MiB temporaries (a new leading axis K, or K copies along axis 0; K grows ×4 while the result stays linear and the call under 2 ms) and compared with the same call over held operands — no NumPy answer needed. Gate `LifetimeScaled_EveryOpKey` (`[FuzzMatrix]`, 4 stressed calls per case, 1,355 cases / 719 op keys incl. the `ma.*` ones, ~1,242 scaled; clean at 96 per case on 2026-10-09); `KnownScaledGaps` (EMPTY since 2026-10-09: its 21 op keys went back into the gate) maps the op keys that lose an operand to the OPEN lifetime gap whose site they reach (with the path and an optional per-case predicate; `LifetimeScaled_KnownGapsContract` keeps each entry tied to a still-open gap; `[OpenBugs]` `LifetimeScaled_KnownGaps` replays them) — the tier keeps no gap ids of its own, a new site gets an LT id with dedicated lifetime cases. `NUMSHARP_LIFETIME_SCALED_REPS` sets the depth (nightly `lifetime-scaled` job in fuzz-soak.yml: 32). TRAPS: JIT liveness is per call site over every later path (`astype` keeps its receiver because a later branch reads `this.Shape`; the same cast through `TensorEngine.Cast` loses it; `np.unique` keeps its receiver, `np.unique_counts` does not); a pass is not proof (short windows go uncaught for 96 runs); the child watchdog is per case (`CaseTimeout`). Detection is a race, never a false positive: a red gate found a real lost anchor — carve it under a new gap id (LT29 next; in the scaled gate: map it to the open gap whose site it reaches, or give the site a new gap). **The fix is the weaver's OPERAND ROOTS, never a hand-placed `GC.KeepAlive`** (a scope roots only what it TRACKS — arrays built while it is open — never the caller's operands): every method that opens an NDScope (`[NDScoped]`, or a hand-written `NDScope.Open()`) copies its NDArray-reaching parameters (and `this` of a class) into `object` shadow locals at entry and passes them to `NDScope.Anchor` (GC.KeepAlive, no code) before every `ret` (`ScopeWeaver.InjectParameterAnchors`); an unscoped entry point that hands an operand's address to a kernel carries `[NDAnchored]` (roots only, no scope, not inherited; NDW019 refuses a body-less / async / iterator / tail-call / also-scoped / nothing-to-root method; the analyzer's NDW013 and the MSBuild fallback scan cover it). A new site: scope or anchor its entry point. Deterministic guards (no race): `test/NumSharp.Tests/Lifetime/OperandRootsTests.cs` reads the woven IL (every scope-opening method with an NDArray parameter calls `NDScope.Anchor`, every `[NDAnchored]` method, each fixed site's entry point) and `WeaverOperandRootTests` (Build.Analyzer) pins the transform; planted check: roots switched off → the gate failed 90 cases naming 25 of the 28 gaps, the IL guard all 37 checks. Ledger + recipes + traps: `test/NumSharp.Tests.Oracle/Fuzz/README.md` "Lifetime gate" and "Scaled lifetime tier".
- **Run the gate**: `dotnet test --filter "TestCategory=FuzzMatrix"`. Each case is bit-exact (pass), a documented difference in `MisalignedRegistry` (excused, never silent), or a failure (red). Full divergence ledger: `test/NumSharp.Tests.Oracle/Fuzz/README.md`.
- **Per-API parity in the coverage artifact**: `coverage/generate_coverage.py` joins every committed contract of this corpus, plus the `.npy`/flags/layout oracles, onto its API row (`coverage/oracle_evidence.py` + the reviewed `coverage/oracle_map.json`). Each row then reads *available* AND *oracle-verified* (direct contracts, an identity alias — the same NumPy object mapped to the same NumSharp member — or a reviewed delegation alias). The dashboard and summary.md show it; at 2026-09-25 the headline is 486/560 oracle-verified vs 535 available. **The join is strict.** A NEW op key must resolve: bare keys go `np.*` first, dotted keys by namespace, anything else needs a `fuzz.keys` entry. A NEW corpus file must be named by a replay suite. Stale/redundant map entries fail too. Otherwise the coverage generator (docs CI) fails, naming the key. Gate: `coverage/test_oracle_evidence.py` (incl. a cross-check that every C# `EquivalentAliases`/`NdarrayAliases`/`MaAliases` entry has its reviewed alias).

### The `.npy`/`.npz` format oracle (same philosophy, separate corpus)

Same shape as above — NumPy is the oracle, the corpus is committed, no Python at test time — but the
claim is stronger: NumSharp's writer must be **byte-identical** to `np.save`, not merely readable.

- **Regenerate**: `python test/oracle/gen_npy_oracle.py` (deterministic; needs `numpy==2.4.2`), then `dotnet build`.
- **Run the gate**: `dotnet test --filter "TestCategory=NpyOracle"` — 286 cases across every dtype ×
  {0-d, empty, 1-D, 2-D, 3-D, unit, empty-2d/3d} × {C, F, strided, reversed, offset, broadcast,
  transposed} × versions {1.0, 2.0, 3.0} × {little, big, native} endian, plus value fidelity (NaN
  payloads incl. sNaN, subnormals, signed zero, integer extremes, BMP seams) and 42
  malformed / unsupported / hostile files whose messages must match NumPy verbatim.
- **`kind: "header"` cases exist because two writer branches are unreachable from a real array.** The
  growth padding is normally INVISIBLE — shrink the body by 5 chars and the alignment padding grows by
  5, leaving the file identical — so a wrong growth axis (`shape[0]` vs `shape[-1]`) passes every
  ordinary test; it only shows on shapes that tip the header across a 64-byte bucket, and those need
  10¹⁷ elements to allocate. The v1.0→v2.0 auto-selection boundary likewise sits at 21,817 dimensions.
  Both are driven through the header dict, exactly as NumPy's `_write_array_header` was.
- **Reverse interop** (manual; needs Python + the SDK): `python test/oracle/verify_npy_interop.py` has
  NumSharp write 28 files and real NumPy read them. This is the only way to prove `.npz` output,
  whose ZIP framing legitimately differs from Python's `zipfile` and so cannot be byte-compared.

## CI Pipeline

`.github/workflows/build-and-release.yml` — test on 3 OSes (Windows/Ubuntu/macOS), build NuGet on tag push, create GitHub Release, publish to nuget.org. The `FuzzMatrix` gate runs here (replays the committed corpora; no Python). **The packages embed NO README** (since after 0.70.0; the "No embedded package README" guard in `Directory.Build.props` fails `dotnet pack` if a project sets `PackageReadmeFile`): an embedded readme would lock each version's nuget.org README tab, so after a release publishes the owner pastes the GitHub release markdown into every package version's README by hand (nuget.org → Manage → Readme) — the step is in the `changelog` skill.

`.github/workflows/fuzz-soak.yml` — nightly soak: sweeps one fixed seed (a deterministic canary whose corpus `source_sha256` should repeat night to night) plus nine fresh random seeds through `test/oracle/fuzz_random.py` (200K cases each, ~1.8M fresh cases/night), replays them, and uploads any failing corpus; copy a shrunk repro into `Fuzz/corpus/regressions/` to pin it on every CI thereafter.

`.github/workflows/benchmark.yml` — decoupled post-release perf run (triggers on a published Release or manual dispatch — a slow/failed benchmark must never gate a release). Runs the whole `benchmark/run_benchmark.py` harness (op/dtype/N matrix + the five subsystems), then commits the refreshed `benchmark-report.md` + subsystem `*_results.*` + cards + a `benchmark/history/<date>_<sha>/` snapshot (+ the `latest` symlink) to master with `[skip ci]` and redeploys the docs. It does **not** refresh `benchmark/README.md` or the hand-built `docs/website-src/docs/benchmarks-dashboard.md` (the UI hub) — both are maintained by hand. (Its "render docs" step still emits `docs/website-src/docs/benchmark-matrix.md`/`benchmark-iterator.md` + references a `benchmarks.md`, none of which the current site toc links — legacy drift; the live hub is `benchmarks-dashboard.md` + `history/latest/*`.)

`.github/workflows/example-pythonnet.yml` — the pipeline of `examples/NumSharpPythonnetGhActionsCicd`: an MSTest project (18 tests) that embeds CPython through pythonnet and asserts NumPy, pandas, SciPy, scikit-learn, Matplotlib and PyTorch over NumSharp arrays through `NumSharp.Interop.pythonnet`, with a `LiveExports`/`LiveImports` leak check after each test. It runs on the 3 OSes when the example or the workflow changes (a `paths` filter; also `workflow_dispatch`): `setup-python` 3.12 with a pip cache keyed on the example's `requirements.txt`, `pip install` into the job's interpreter, then `dotnet test` with `NUMSHARP_PYTHONNET_PYTHON` = `steps.python.outputs.python-path` (binding; unset under `CI=true` FAILS, never skips; unset on a dev machine is Inconclusive). Its console twin `examples/NumSharpPythonnetLocalExample` finds a local Python (`NUMSHARP_PYTHONNET_PYTHON`, then `.venv` beside the project, then `PATH`) and is written like the twelve tutorials (silent, outcome comments; the tests are its proof). Both csprojs are the REGULAR form, by the owner's rule (2026-10-10, for every copy-ready example): exactly what `dotnet new` + `dotnet add package NumSharp.Interop.pythonnet` write — one plain `PackageReference` with its version, no custom MSBuild property, no `ProjectReference` switch, no `SignAssembly` / `IsPackable` line (in the repo `Directory.Build.props` strong-names them; 0 warnings). So they build against the PUBLISHED package, never this repo's source: they may use only API that 0.70.0 has (`typecode`, never `dtype`; `NDArrayPythonInterop.RegisterCodec()`, never `Py.RegisterNumSharpCodec()`), a source change cannot break or fix them, and the workflow tests the released package (the interop suite of the main workflow covers the source). Facts both hosts encode (probed 2026-10-09, Windows + WSL): `PythonEngine.ProgramName` is set for EVERY interpreter, venv or not — without it a venv's packages are absent, on Windows `sys.executable` is the .NET host, and on Linux the engine takes its paths from the first `python3` on `PATH` (a segfault when that is another Python); pythonnet 3.0.5's `Runtime.Shutdown` runs 87 full `PyGC_Collect` passes (7.3 s with the six libraries imported — the Local example calls `gc.collect(); gc.freeze()` first, 0.2 s; the test fixture keeps the plain shutdown); `torch.__version__` is a `str` subclass that `As<string>()` refuses (`ToString()`); `torch.optim` lazily imports ~800 modules (`torch._dynamo`, `sympy`), so the examples update weights by hand; the .NET app host (`<app>.exe`, `testhost.exe`) is not long-path aware on Windows, so a module file past 260 characters fails to import as `No module named …` (`python.exe` and `dotnet <app>.dll` are). Not yet run on GitHub's macOS runner when this was written.

`examples/NumSharpOpenBlasDefaultExample` and `examples/NumSharpOpenBlasAdvancedExample` (2026-10-10) — two console apps over the PUBLISHED `NumSharp.Interop.OpenBLAS` 0.70.0, in the same regular csproj form; no workflow runs them. **Default**: the one package reference and no configuration — `OpenBlasEngine.Enable()` first, then the products, the LAPACK tour (it returns the largest residual of the identities it shows) and a matmul timing with `Disable()` / `Enable()` as the switch. **Advanced**: `OpenBlasVersion="0.3.31.188.0"` as an ATTRIBUTE of the `PackageReference` (the OpenBLAS of NumPy 2.4.4; the child-element form of the docs is equivalent), a `Program.cs` that names no package type and sets `NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL=0` in-process, `Enable(threads: 1, coreType: OpenBlasEngine.CoreType)` in a `NoInlining` method, and the SHA-256 of one float32 product for each thread count. Facts they encode, all measured against the published 0.70.0 (Windows 11 + WSL Ubuntu 24.04, one i9-13900K):
- **A package reference alone does NOT install the backend (open product gap).** The `[ModuleInitializer]` runs when the runtime first prepares a method that names a type of the package, and Core never loads the assembly. A program that references the package and calls only `np.*` runs `dot` / `matmul` on the managed kernels and gets `OpenBlasMissingBackendException` — whose text says to reference the package that it already references — from `cholesky`, `qr`, `svd`, `svdvals`, `eig`, `eigvals`, `eigh`, `eigvalsh`, `lstsq`, `pinv`, `matrix_rank`, `cond` and `norm` with `ord` 2 / `'nuc'` (`solve` / `inv` / `det` / `slogdet` take the managed LU). The quick start of the docs works because its method names `OpenBlasEngine.Info`; the consumer of `verify_package_consumer.sh` names the type too, so no gate sees the gap. "Referencing the package is the whole opt-in" (docs, package README, the exception text) is therefore true only for code that names the package before it computes.
- **The initializer runs at the ENTRY of the first method that names the package**, before that method's first statement, in every JIT mode probed (tiered on / off, PGO off, QuickJit off, ReadyToRun off, JitStress). So an in-process opt-out must be set by code that names no package type and inlines no method that does.
- **Kernel pin (`coreType`)**: applied when the call is the first load; a no-op when the loaded kernels are the same; `InvalidOperationException` when the library is loaded with others; `PlatformNotSupportedException` when the CPU cannot run them; an UNKNOWN name raises nothing, because OpenBLAS ignores it (read `CoreName`). `OPENBLAS_CORETYPE` / `OPENBLAS_NUM_THREADS` in the process environment work with no code.
- **Build-time pin on glibc Linux (open product gap).** The fetch task matches the wheel by `"-" + <anchor of the manifest wheel's platform tag> + "."` (`manylinux2014_x86_64`), and scipy-openblas from 0.3.33.0.0 publishes `manylinux_2_27_x86_64.manylinux_2_28_x86_64`: the build fails with `expected exactly one scipy-openblas64 0.3.33.112.0 wheel for platform tag 'manylinux2014_x86_64', found 0`. So the OpenBLAS of NumPy 2.5 cannot be pinned on linux-x64 / linux-arm64 (it builds and runs on Windows, as does 0.3.34.237.0); 0.3.29.0.0 fails the same way (its compound tag lists `manylinux_2_17` first). Stageable on all 8 RIDs: 0.3.29.265.1 to 0.3.31.188.0.
- **Parity, measured.** The bundled DLL is the file inside numpy 2.4.2's wheel (sha256 `74a40872…0377`) and the pinned 0.3.31.188.0 DLL the one inside numpy 2.4.4's (`63c857e7…1080`). The float32 `(512,784) @ (784,512)` product of `np.random.seed(7)` inputs has NumPy's hash at 1, 2 and 24 threads for both pairs, and the 1- and 2-thread hashes are equal on Windows and Linux. NumPy's pins (`requirements/ci_requirements.txt` at each tag): 2.2.6 / 2.3.0 → 0.3.29.0.0; 2.3.2–2.3.4 → 0.3.30.0.1; 2.3.5 / 2.4.0 / 2.4.1 → 0.3.30.0.7; 2.4.2 → 0.3.31.22.0; 2.4.3 → 0.3.31.126.1; 2.4.4 → 0.3.31.188.0; 2.5.0 / 2.5.1 → 0.3.33.112.0.
- **Output size.** A portable build output carries the libraries of all 8 RIDs (201 MB); `dotnet publish -r <rid> --self-contained false` gives 23 MB (Windows) / 29 MB (Linux), with the override and its marker flattened beside the app. The released NumSharp package ships the NDW012 analyzer, so example code without an `NDScope` warns.

## Scripting with `dotnet run` (.NET 10 file-based apps)

### Accessing Internal Members

NumSharp has many key types/fields/methods marked `internal` (Shape.dimensions, Shape.strides, NDArray.Storage, np._FindCommonType, etc.). To access them from a `dotnet run` script, override the assembly name to match an existing `InternalsVisibleTo` entry — **and sign the script**:

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
```

**How it works:** NumSharp declares `[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript, PublicKey=…")]` in `src/NumSharp.Core/Assembly/Properties.cs`. The `#:property AssemblyName=NumSharp.DotNetRunScript` directive overrides the script's assembly name (which normally derives from the filename) to match, granting full access to all `internal` and `protected internal` members.

**Why the two signing properties.** NumSharp is strong-named (repo-root `Directory.Build.props`), so its friend declarations name a `PublicKey` — and a keyed `InternalsVisibleTo` matches ONLY an assembly signed with that key. An unsigned script with the right *name* is not a match, and the failure is a compile error on the first internal member, not a warning.

A script living **under the repo root** needs neither property: `Directory.Build.props` is an implicit file for file-based apps, so it inherits signing automatically (this is why `test/oracle/*.cs` just works). A script run from **anywhere else** — the scratchpad, a temp dir — inherits nothing and must pass both properties itself. Use an absolute path for the key.

### Accessing Unsafe code
NumSharp uses unsafe in many places, hence include `#:property AllowUnsafeBlocks=true` in scripts.

### Script Template (copy-paste ready)

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
```

(The last two are required only for scripts OUTSIDE the repo root — see above.)

### Key Internal Members Available

| Member | What it exposes |
|--------|----------------|
| `shape.dimensions` | Raw long[] of dimension sizes |
| `shape.strides` | Raw long[] of stride values |
| `shape.size` | Internal field: total element count |
| `shape.offset` | Base offset into storage (NumPy-aligned) |
| `shape.bufferSize` | Size of underlying buffer |
| `shape._flags` | Cached ArrayFlags bitmask |
| `shape.IsWriteable` | False for broadcast views (NumPy behavior) |
| `shape.IsBroadcasted` | Has any stride=0 with dimension > 1 |
| `shape.IsSimpleSlice` | IsSliced && !IsBroadcasted |
| `shape.OriginalSize` | Product of non-broadcast dimensions |
| `arr.Storage` | Underlying `UnmanagedStorage` |
| `arr.GetTypeCode` | `NPTypeCode` of the array |
| `arr.Array` | `IArraySlice` — raw data access |
| `np._FindCommonType(...)` | Type promotion logic |
| `np.powerOrder` | Type promotion ordering |
| `NPTypeCode.GetGroup()` | Type category (int/uint/float/etc.) |
| `NPTypeCode.GetPriority()` | Type priority for promotion |
| `NPTypeCode.AsNumpyDtypeName()` | NumPy dtype name (e.g. "int32") |
| `Shape.NewScalar()` | Create scalar shapes |

### Common Public NDArray Properties

| Property | Description |
|----------|-------------|
| `nd.shape` | Dimensions as `int[]` |
| `nd.ndim` | Number of dimensions |
| `nd.size` | Total element count |
| `nd.dtype` | The dtype descriptor (`DType`, NumPy's `ndarray.dtype`); `nd.dtype.type` is the CLR element `Type`, `(Type)nd.dtype` the explicit cast |
| `nd.typecode` | Element type as `NPTypeCode` |
| `nd.T` | Transpose (swaps axes) |
| `nd.flat` | 1D iterator over elements |

## Adding New Features

1. Read NumPy docs for the function
2. **Run actual Python code** to observe exact behavior and fuzzy all possible inputs to define a behavior matrix.
3. Check existing similar implementations
4. Implement behavior matching exactly that of numpy.
5. Write tests based on observed NumPy output
6. Handle all 15 dtypes

---

## Q&A - Design & Architecture

**Q: Why unmanaged memory instead of Span<T>/Memory<T>?**
A: Benchmarking showed unmanaged memory was fastest. NDArray is self-managed memory allocation optimized for performance.

**Q: Why are there TWO classes — `ILKernelGenerator` AND `DirectILKernelGenerator`?**
A: They encode two different kernel-driving contracts. `DirectILKernelGenerator` (63 partials in `Direct/`) emits whole-array kernels: one call processes the entire array; the kernel walks dimensions/strides itself. `ILKernelGenerator` (root) emits per-chunk kernels matching NumPy's `PyUFuncGenericFunction` contract: the iterator (`NDIterRef`) drives the loop and the kernel only processes one chunk. The split makes the contract explicit in the type name. A kernel lives in whichever class matches how it is driven — `ILKernelGenerator` for kernels run as an NDIter inner loop (`np.where`, flat/pairwise reductions), `DirectILKernelGenerator` for kernels that own their full-array traversal.

**Q: When should I write kernels in `ILKernelGenerator` vs `DirectILKernelGenerator`?**
A: Match the driving contract. A kernel that runs as the inner loop of an NDIter pass (the iterator advances the operand pointers, the kernel handles one chunk) belongs in `ILKernelGenerator`. A kernel that takes the whole array plus its shape/strides and walks it itself belongs in `DirectILKernelGenerator`.

**Q: Why is TensorEngine abstracted?**
A: So alternative backends (GPU/CUDA, SIMD intrinsics, MKL/BLAS) can plug in behind one contract; `DefaultEngine` (pure C#) is the engine.

**Q: How closely does the API match NumPy?**
A: Goal is as close as possible - all edge cases included (NaN handling, multi-type operations, broadcasting). Target is NumPy 2.x.

**Q: Does np.random match NumPy's random state/seed behavior?**
A: Yes, 1-to-1 matching. The bit generator is **MT19937** (NumPy's Mersenne Twister, `RandomSampling/MT19937.cs`) with Marsaglia-polar Gaussian + cached-Gaussian carry and NumPy's state-tuple `get_state`/`set_state` format, so a given seed produces byte-identical sequences to NumPy 2.4.2.

**Q: What are the primary use cases?**
A: Anything that can use the capabilities - porting Python ML code, standalone .NET scientific computing, integration with TensorFlow.NET/ML.NET.

**Q: Which subsystem is the most intricate?**
A: Slicing/broadcasting — offset/stride calculations with contiguity detection. The `readonly struct Shape` with cached `ArrayFlags` centralizes it behind O(1) flag checks.

**Q: How is NumPy compatibility validated?**
A: Written by hand based on NumPy docs and original tests. Testing philosophy: run actual NumPy code, observe output, replicate 1-to-1 in C#.

**Q: What's the pattern for adding new np.* functions?**
A: Sometimes uses other np functions (no DefaultEngine needed). Sometimes requires DefaultEngine for optimization. Tests should be based on actually running NumPy code and imitating the outcome.

**Q: Are breaking changes acceptable?**
A: Yes - breaking changes are accepted to align with NumPy 2.x behavior.

---

## Q&A - Core Components

**Q: What are the three pillars of NumSharp?**
A: `NDArray` (user-facing API), `UnmanagedStorage` (raw memory management), and `Shape` (dimensions, strides, coordinate translation). They work together: NDArray wraps Storage which uses Shape for offset calculations.

**Q: What is Shape responsible for?**
A: Shape is a `readonly struct` containing dimensions, strides, offset, bufferSize, and cached `ArrayFlags`. Key properties: `IsScalar`, `IsContiguous`, `IsSliced`, `IsBroadcasted`, `IsWriteable`, `IsSimpleSlice`. Methods: `GetOffset(coords)`, `GetCoordinates(offset)`. NumPy-aligned: broadcast views are read-only (`IsWriteable = false`).

**Q: How does slicing work internally?**
A: The `Slice` class parses Python notation (e.g., "1:5:2") into `Start`, `Stop`, `Step`. It converts to `SliceDef` (absolute indices) for computation. `SliceDef.Merge()` handles recursive slicing (slice of a slice).

**Q: What are the special Slice instances?**
A: `Slice.All` (`:` - all elements), `Slice.Ellipsis` (`...` - fill dimensions), `Slice.NewAxis` (insert dimension), `Slice.Index(n)` (single element, reduces dimensionality).

**Q: What is NDIter?**
A: The NumPy-aligned multi-operand iterator. It handles C/F/A/K order, broadcasting, external loops, buffering, casting, masks, reductions, and synchronized traversal for copy and elementwise kernels. Copy and multi-operand execution go through `NDIter.Copy` and the multi-operand iterator.

**Q: How does broadcasting work?**
A: Shapes align from the right. Dimensions must be equal OR one must be 1. Dimension of 1 "stretches" to match. Implemented via `DefaultEngine.Broadcast()` which resolves compatible shapes.

**Q: What is InfoOf<T>?**
A: Static type information cache to avoid runtime reflection. Provides `InfoOf<T>.Size` (bytes), `InfoOf<T>.NPTypeCode`, `InfoOf<T>.Zero`, `InfoOf<T>.MaxValue/MinValue`.

**Q: What is NDArray<T>?**
A: Generic typed wrapper providing type-safe access. Returns `T` from indexer instead of NDArray. Has typed `Address` pointer (`T*`) and `Array` property (`ArraySlice<T>`).

**Q: When does DefaultEngine use parallelization?**
A: Parallelization is minimal. Most operations use SIMD vectorization instead for performance.

---

## Q&A - Operations & Operators

**Q: How do arithmetic operators work?**
A: All operators (`+`, `-`, `*`, `/`, `%`, unary `-`) are defined in `NDArray.Primitive.cs`. They delegate to `TensorEngine.Add()`, `Subtract()`, etc. Scalar operands are wrapped via `NDArray.Scalar()`.

**Q: How do comparison operators work?**
A: Element-wise comparisons (`==`, `!=`, `>`, `<`, etc.) return `NDArray<bool>`. Defined in `NDArray.Equals.cs`, `NDArray.Greater.cs`, etc. Support broadcasting.

**Q: What indexing modes are supported?**
A: Integer indices, string slices (`"1:3, :"`), Slice objects, boolean masks, fancy indexing (NDArray<int> indices), and mixed combinations. All in `Selection/NDArray.Indexing*.cs`.

**Q: How is linear algebra implemented?**
A: Core ops (`dot`, `matmul`, `outer`) in `LinearAlgebra/`; `trace`/`diagonal` in `Indexing/`. The rest of what NumPy sends to OpenBLAS — the product family (`inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`) and the whole `np.linalg` module — sits in `LinearAlgebra/` and `LinearAlgebra/linalg/`, dispatching through `TensorEngine.Blas`. The products compute with managed kernels; the LU family (`det`/`slogdet`/`solve`/`inv`) has a managed fallback (`Backends/Default/LinearAlgebra/ManagedLu.cs`); the remaining LAPACK factorisations (`cholesky`/`qr`/`svd`/eigen/`lstsq`) raise `NotSupportedException` until a backend is installed. See "The BLAS/LAPACK API surface" above.

**Q: Which `np.linalg` factorisations work without a backend, and which throw?**
A: The **LU family** — `det`, `slogdet`, `solve`, `inv` (and `tensorinv`/`tensorsolve`/`matrix_power(n<0)` that compose on them) — has a **managed fallback** (`Backends/Default/LinearAlgebra/ManagedLu.cs`): a from-scratch unblocked LU + substitution that computes them `allclose` to NumPy without any native dependency (see "The BLAS/LAPACK API surface"). Everything else — `cholesky`, `qr`, `svd`/`svdvals`/`pinv`/`matrix_rank`/`cond`/`lstsq`, the eigensolvers, and the SVD-defined matrix norms — throws `NotSupportedException` until a backend is installed, except over an EMPTY operand (an empty matrix or a zero-size batch), whose NumPy answer needs no arithmetic and comes from Core (`LinAlgEmpty`, since 2026-10-08). The dividing line is what a portable managed algorithm costs: an LU with partial pivoting is a hundred lines, but a portable `gesdd`, `geev` or `gelsd` is a library in its own right, and NumSharp.Core carries none. For the throwing family the API, its NumPy-exact validation and its verbatim error messages all exist and are tested; only the numerics wait, and the exception names the NumPy API, the LAPACK routine and the `IBlasBackend.Try*` member to implement, so it reads as an install instruction.

**Q: Is the managed LU family bit-exact with NumPy?**
A: No — `allclose`, not bit-exact, by construction. NumPy computes this family through LAPACK's BLOCKED `getrf`/`gesv`, whose Schur-complement update runs through an arch-specific `gemm`; that accumulation order is exactly what NumSharp cannot reproduce in managed code (the same reason the matrix PRODUCTS need the bundled binary for byte-parity). `ManagedLu` is the classic right-looking UNBLOCKED factorisation, so it agrees with NumPy to floating-point tolerance — bit-for-bit for tiny matrices where neither side blocks, a few ULP apart as they grow — and it is FASTER than NumPy for small matrices (lower dispatch overhead than Python+LAPACK) while losing to native blocked BLAS3 for large ones. When a backend is installed it wins the seam, so the managed path only runs in the no-backend configuration. The one boundary case where the two genuinely disagree — an exactly-singular matrix that cancels to a true zero pivot in blocked LAPACK but not unblocked (textbook `[[1,2,3],[4,5,6],[7,8,9]]`) — is a documented `[Misaligned]` divergence.

---

## Q&A - Development

**Q: What's in the test suite?**
A: MSTest v3 framework in `test/NumSharp.Tests/`. Many tests adapted from NumPy's own test suite, plus the differential-fuzz corpora. Broad coverage across operations, dtypes, and edge cases. Uses source-generated test discovery (no special flags needed).

**Q: What .NET version is targeted?**
A: Library multi-targets `net8.0` and `net10.0`. Tests also multi-target both frameworks.

**Q: What are the main dependencies?**
A: No external runtime dependencies. `System.Memory` and `System.Runtime.CompilerServices.Unsafe` are built into the .NET 8+ runtime.

**Q: What projects use NumSharp?**
A: TensorFlow.NET, ML.NET integrations, Gym.NET, Pandas.NET, and various scientific computing projects.

**Q: Can I save/load NumPy files?**
A: Yes, and byte-exactly. `np.save` writes `.npy`, `np.savez`/`np.savez_compressed` write `.npz`, and `np.load` reads both (returning `object`; `np.load_npy`/`np.load_npz` are the typed forms). `IO/NpyFormat.cs` is a port of NumPy 2.4.2's `_format_impl.py`, so a saved file is byte-for-byte what NumPy's own `np.save` would produce — versions 1.0/2.0/3.0, 64-byte alignment, `fortran_order`, and big-endian reads included. See the File I/O section above for the dtype map and the gates.

**Q: Why does `np.load` return `object` instead of `NDArray`?**
A: Because NumPy's does: `np.load` yields an `ndarray` for a `.npy` and an `NpzFile` for a `.npz`, dispatching on the file's magic bytes rather than its extension, and C# has no union type to express that. `np.load_npy` and `np.load_npz` are the typed escapes — prefer them when the kind is known; each also reports a directed error if handed the other kind.

**Q: What random distributions are supported?**
A: An extensive NumPy-matching set (40+ distributions and samplers — see the Supported np.* APIs → Random list), all on the `NumPyRandom` class in `RandomSampling/`, with 1-to-1 seed/state parity to NumPy.

---

## Detailed Documentation

@ARCHITECTURE.md for comprehensive technical details and @CONTRIBUTING.md for development workflow.
