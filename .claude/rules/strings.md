---
paths:
  - "src/NumSharp.Core/Backends/Strings/{NDStringStorage,NDStringArrays,NDStringCopy,NDStringCodec,NDStringCasting,NDTextConversion,NDTextFormat,NDTextParse,NDUtf8,NDPyObject,VStringAllocator,NDVStringMove,NDVStringCopy,NDVStringCasting,NDStringUfuncs,NDStringUfuncs.Precedence,NDStringUfuncs.ShapeDecided,NDReduceLikeResolution}.cs"
  - "src/NumSharp.Core/DTypes/{StringDTypeMeta,VStringDTypeMeta,VStringParameters,StringCastingImpls,VStringCastingImpls}.cs"
  - "src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Coercion.Text.cs"
  - "src/NumSharp.Core/Utilities/{Converts.NumPyText,PyUnicode}.cs"
  - "src/NumSharp.Core/Backends/Printing/FloatDigits.cs"
  - "test/NumSharp.Tests/Backends/Strings/**"
  - "test/NumSharp.Tests.Oracle/Strings/**"
  - "test/oracle/{gen_strings_oracle,soak_strings_oracle}.py"
---

# The string dtypes: storage, conversion, movement

NumPy's three string classes, byte-identical to NumPy 2.4.2: `bytes_` (`S`, one byte a character), `str_` (`U`, one
native UCS-4 code point a character) and NEP 55's `StringDType` (`T`, UTF-8 text in 16-byte cells). Computing on text:
`strings-compute.md` (comparisons, arithmetic, reductions, ordering) and `strings-methods.md` (the named `np.strings`
functions); numpy.char and chararray: `char-chararray.md`; numpy.ma over text: `masked-arrays.md`; the classes in the
dtype system: `dtypes.md`; user docs: `docs/website-src/docs/fundamentals/strings-and-bytes.md`.

## The rule

- Where NumPy computes on text, NumSharp computes, byte for byte; where NumPy refuses (no string loop), NumSharp raises
  NumPy's type and text. Both happen after NumPy's earlier checks, in NumPy's order: the `dtype=` class check, a
  read-only `out=`, a non-bool `where=`, the axis under one of three rules, the quantile q / axis / method / empty-axis
  order, a computing loop's output cast and broadcast (`Backends/Strings/NDStringUfuncs{,.Precedence}.cs`).
  `StringComputationNotSupportedException` is only `np.evaluate`'s refusal of a text operand.
- Where dtypes, sizes or shapes decide NumPy's answer, NumSharp gives it without reading the text: an empty array, lines
  of one element, a pair the comparison operators cannot compare (truth tests, counts, the nonzero family,
  arg-reductions, sort / partition, `np.unique` and its Array-API forms, `isin`, the set routines, `array_equal` /
  `array_equiv`, `lexsort`, `sort_complex`, `trim_zeros`: `NDStringUfuncs.ShapeDecided.cs` runs the numeric path on a
  bool stand-in of the operand's shape and restores values from the input) and binary ufuncs over an empty broadcast
  (`BinaryGuard`). NumPy's checks before a conversion run in its order (interp / digitize / repeat depth, then the
  'safe' cast; place / putmask: an ndarray value meets the 'safe' rule while a C# array is a Python list converted
  directly).
- Every engine ufunc member resolves its `dtype=` through `UfuncLoopCode(dtype, "<numpy name>", x1, x2, out, where)`, so
  an unsized string request (`'U'`, `'S'`) gets NumPy's per-ufunc answer. `add` / `multiply` reductions that touch text
  (a text input, `out=` or `dtype=`) resolve their loop with a port of NumPy's `reducelike_promote_and_resolve` and its
  legacy type resolvers (`Backends/Strings/NDReduceLikeResolution.cs`), giving a FRESH NumPy process's answer: NumPy's
  dispatch cache makes a later call's answer depend on earlier ones (`np.prod(int8, dtype='U')` names no loop in a fresh
  process but finds the string repeat loop after `np.multiply(u, int8, dtype='U')`); the oracle records the fresh answer.
- `numpy.polynomial` over text follows NumPy's statements up to the first one that computes with the strings themselves,
  then raises `NotSupportedException` (Python's string arithmetic is not modelled).

## Model

- One storage lane, `NPTypeCode.String`, for all three classes. The descriptor (`UnmanagedStorage.Descr`) carries the
  class and the width (`<U5` itemsize 20, `|S5` 5, a `T` cell 16), so every string allocation takes a DType, and the lane
  alone names no dtype (`dtypes.md`). Test `descr.Meta is VStringDTypeMeta` to tell `T` from `S` / `U`, never the code.
- `StringDType`: `VStringDTypeMeta` (`numpy.dtypes.StringDType`, kind / char `'T'`, num 2056, a 16-byte cell aligned to
  8, parametric, NO singleton: every descriptor is fresh). `VStringParameters` holds `na_object` / `coerce` and NumPy's
  derived fields. Built by `np.dtypes.StringDType(bool coerce = true)` and `np.dtypes.StringDType(object na_object, bool
  coerce = true)` (a bool sentinel goes BY NAME: a positional bool binds `coerce`) and by `np.dtype("T")`.
- `T` equality is NumPy's `na_eq_cmp` through `Backends/Strings/NDPyObject.cs` (CPython's `str()` / `repr()` / `==` over
  the C# boundary map, with NumPy's scalar rules when an `np.float16` (`Half`) or a decimal takes part:
  `np.float16(2048) == 2049` is True); the hash is coarse for numbers so it can follow that non-transitive equality. The
  cast table is `casts.cpp`'s (`VStringCastingImpls.cs`); promotion admits only `U` beside `T` (the common instance takes
  the present sentinel and the FIRST descriptor's `coerce`); `isdtype` refuses `T`; a `T` instance as a ufunc `dtype=`
  is NumPy's new-style refusal.
- `Backends/Strings/NDUtf8.cs` is the strict UTF-8 codec with CPython 3.12's texts (a lone surrogate in a string sentinel
  is `UnicodeEncodeError` already).

## The C# boundary

- A C# `string` is a Python `str` (`NDArray.WasPythonStr`): `NDArray x = "[1, 2, 3]"` is a 0-d `<U9`, never parsed
  numbers. `np.dtype(typeof(string))` and `np.str_` are the unsized `<U` (`<U1` once allocated); `np.array(string[])`
  discovers `<U{longest}`.
- A `byte[]` VALUE is Python `bytes` (`np.array(object[] of byte[])`, numpy.char arguments), but a `byte[][]` without a
  dtype is a NUMERIC jagged array (a uint8 matrix), and the TYPE `typeof(byte[])` names uint8: pass an `S` dtype for bytes
  objects.
- Element access: `item()` gives a `string` (`U`, `T`) or a `byte[]` (`S`); `NDArray.AsStringArray`; the indexer's writes
  are cut to the width.
- Overload traps: C# binds a literal to an `NDArray` overload before an `object` one (`np.insert(t, 1, "x")` takes the
  literal's NDArray route: box it to reach the object route); `nd.CopyTo(byte*)` binds the generic `CopyTo<T>(T*)`
  (cast to `void*`); `cond ? np.array(...) : cond2 ? new byte[] { 32 } : " "` is typed `NDArray`, so the `byte[]`
  becomes a uint8 array (cast every arm to `object`).

## Movement of `S` / `U`

- Every copy runs the numeric kernels through UNIT VIEWS `D + (chars,)` (`NDStringCopy`): a same-layout pair is one
  block move (`NDStringCopy.TryCopyDenseBlock`); a non-overlapping pair over at most two effective axes runs
  `GetElementCopyKernel` (one constant-size cpblk / initblk per element); `np.where` selects whole elements
  (`GetWhereBytesKernel`); a lone index array or prefix mask gathers whole rows on the fancy-index kernel route (never
  write an explicit trailing `:` on the unit view: it forces the general index path); a contiguous fill doubles block
  moves; a fully overwritten `U` result skips NumPy's NEEDS_INIT zero fill (`NDArray.ForOverwrite`; above a size a zeroed
  buffer is fresh OS pages, so every call would fault while writing and pay a `VirtualFree` on dispose).
- String ↔ string `astype`: ASCII `S ↔ U` with Python's Unicode error texts.
- The element readers are vector code: `NDStringCodec.ReadUcs4` (trim from the end, one range test for a unit past
  U+FFFF, a 16-unit `Vector256.Narrow` copy whose last step OVERLAPS the one before; an astral text keeps the counting
  scan and its SystemError), `ReadBytes` (vector trim), `NDUtf8.DecodeTrusted` (an ASCII text widened by
  `Ascii.ToUtf16`). The `U` writers `NDStringCodec.WidenUtf16` / `WidenBytes` copy 16 units per vector step with the same
  overlapping last step (`WriteUcs4`'s short-element path stays scalar: it folds the surrogate test into its one loop).
  A fenced test of a copy whose last step overlaps must place a SHORT value in a WIDE element at the region's start: a
  step that starts one unit early writes before the element, which a heap array's padding absorbs silently
  (`TextElementWriteTests`; the readers: `TextElementReadTests`).
- Floors, where no lever is left: a 1K call sits at the per-call NDArray floor (~1.0–1.3× NumPy); copyto / set-slice /
  fill move the same bytes as NumPy at memory speed; `AsStringArray` / `tolist` of long texts are bound by allocating
  .NET strings (two bytes a character) and the GC copying them; a 200 MB `U500` result is bound by its first-touch page
  faults, as NumPy's is.

## `StringDType` arrays

Storage is NumPy's byte for byte (`Backends/Strings/VStringAllocator.cs` = `static_string.c`): the 16-byte cell, short
strings in the cell, medium / long ones in a per-BUFFER arena, heap strings after a longer rewrite. The cells' memory
block owns the allocator, so views keep it alive. Rules a new route keeps:
- Never move a `T` cell as raw bytes (an arena offset or a heap pointer means something only to its own allocator).
  `DTypeSize` / `dtypesize` THROW for a `T` storage so a raw route fails loudly; byte arithmetic reads `ItemBytes` (16).
  Gathers run the numeric route on the CELL view and re-home each cell (`NDVStringMove.Gather` / `Rehome` /
  `RehomeMulti`); scatters write int64 marks and duplicate the marked values (`ScatterMarks`); copies go through
  `NDVStringCopy.Copy` (NumPy's overlap rules, the destination-stride walk that decides arena offsets).
- A store's value is NumPy's `PyArray_FromAny(values, dtype)` (`NDVStringMove.StoreValue`): a Python object (a C# literal
  marked `WasPythonLiteral` / `WasPythonStr` / `WasPythonBool`, null for `None`, a boxed value, the elements of a C#
  array or list) goes through `stringdtype_setitem` (the NA object, its `str()`, refused under `coerce=False`), an
  ndarray through the cast. An index of integers alone given a Python object is NumPy's `array_assign_item`: setitem
  straight into the element (`UnmanagedStorage.SetVStringItem`) with setitem's MemoryError texts; a slice assignment and
  `fill` run the copy loop ("... in string to string cast").
- Out-of-memory paths are tested by fault injection: `VStringAllocator.FailAllocationAfter(n)` makes the (n+1)-th arena
  or heap allocation of THIS thread fail (`DisarmFaultInjection` cancels; production pays one volatile read). Each
  operation raises NumPy's text for its loop and disposes its half-built result (`StringDTypeFaultTests`: the live
  allocator count returns to its baseline).
- A fill of a FRESH buffer reserves its arena once and appends in element order: only medium / long texts move the
  arena cursor, so a store may settle the texts that need no arena (short, empty, NA) first and append the rest
  afterwards in the original order; cells and arena stay NumPy's element-by-element stores byte for byte, only the
  capacity differs (`VStringAllocator.TryReserve`, `AppendFresh`). `np.array(values, T)` (`VStringArraySlice.SetRange`)
  does it in two passes over the leading run of strings (pass 1 counts a long text by its LENGTH, exact for ASCII; pass 2
  encodes each text straight into its slot with `NDUtf8.EncodeExact`), then the general store from the first value it
  cannot place; the `S` → `T` cast does it when every destination cell is zero (`ReserveForFreshBytes`). A cell holding
  a SHORT text sends a longer one to the heap (`heap_or_arena_allocate`), so never reserve for, or `AppendFresh` into, a
  destination that holds text (an append would also overwrite and leak a held heap string), and never run an exact size
  pass over all texts (a byte count per string costs what the reservation saves). Every public assignment route
  (`np.copyto`, slice / fancy / mask assignment) casts an `S` value into a fresh temporary first, so a held destination
  reaches the cast only through `NDIter.Copy` itself. Gates `StringDTypeFillTests` (the element-by-element store as the
  reference) and `StringDTypeFaultTests.Array_FailsCleanlyPartWay`.
- Heap strings are carved from pages, not allocated one by one (`VStringAllocator.HeapMalloc` / `HeapFree`): a text of at
  most `HeapPoolMaxBytes` (1,032) bytes is a block of a page of its size class (64 classes, block strides 32 to 1,040
  bytes; pages of 1–64 KiB from `SizeBucketedBufferPool`, recycled across buffers). Each block starts with an 8-byte
  header naming its page; each page has a free list and a bump pointer; a page whose last text is freed goes back to the
  pool at once, so a buffer holds memory for its live texts, not its peak; the allocator keeps ONE emptied page as a
  spare (without it, one element rewritten between two lengths made each write a pool round trip). Longer texts are
  C-allocator blocks of their own (a null page in the header). The cells stay NumPy's byte for byte; only the heap
  address differs, which the oracle masks. A cell's heap pointer is 8 bytes into a block: free it only through
  `HeapFree`, never with `NativeMemory.Free`; `ValidateHeapPages()` checks every invariant (gate
  `StringDTypeHeapPagesTests`).
- An ordering of long texts prefetches the texts ahead: a pair of random texts is ordered by its first eight bytes, so
  `<` of texts stored outside their cells reads one cache line of each text, and once those lines pass L2 every pair is
  two misses no hardware prefetcher predicts. `GetVStringCompareKernel(op, aIndex, prefetch: true)` (orderings only: `<`,
  `<=`, and `>` / `>=` as the swapped pair) emits a `prefetchnta` of each arena or heap text's first line 16 elements
  ahead (`NDVStringCells.PrefetchPair`), never a cell past the chunk. The choice (`NDVStringCells.PrefetchesTexts`): at
  least 128 arena bytes a cell on average and at least 16,384 distinct texts touched (two views of one buffer count
  once); x86 only. Never for an equality: texts of different lengths are unequal by their cells' sizes, so `==` reads no
  text and a prefetch only wastes bandwidth (`TakesPrefetchingKernel` says no and the factory throws). Hooks
  `NDVStringCells.TextPrefetchOverride` / `NDVStringComparison.LastKernel` (`[ThreadStatic]`); gate
  `StringDTypePrefetchKernelTests`.
- Allocator memory is reported through the pool's coarse accounting (`SizeBucketedBufferPool.AccountForeignAllocation` /
  `AccountForeignRelease`), never one `GC.AddMemoryPressure` per growth (that drove constant gen-2 collections). A bulk
  copy into an allocator with no arena text reserves the arena's exact total first (`TryReserve`; the arena's one
  reallocation site is `TryGrowArena`, pinned by the native-allocation chokepoint gate): only the capacity differs from
  NumPy's 125 % steps, never an offset. The pre-pass counts a text only where `heap_or_arena_allocate` puts it in the
  arena (a destination cell whose flags are 0 once the free has cleared MISSING). A fast path must give NumPy's cell for
  every input it takes: a short NON-EMPTY source into a cell that holds no arena or heap text is the source cell (two
  8-byte moves); an empty source is `pack_empty`, which keeps the destination's state (a heap or arena destination must
  go through `free`); `np.array(list, T)` writes a short ASCII text straight into its fresh cell.
- A fresh `T` result takes a recycled pooled buffer cleared up to `SizeBucketedBufferPool.PrefersWriteOnce`'s 64 MiB
  (`VStringArraySlice.Allocate`; `TakeZeroed` is calloc, whose first-touch faults every fresh array paid), and a fresh
  arena is `NativeMemory.AllocZeroed`.
- TRAPS: a short text stored over a freed arena cell keeps the old offset's high bytes past its size
  (`NpyString_newemptysize` writes only byte 15), so every reader masks; `pack_empty` leaves a never-written cell all
  zero; `string_to_bool` reads `default_string.size == 0` for a string NA (`na_object=''` is True).
- NumPy 2.4.2 defects on `T` that NumSharp does not reproduce (pinned by unit tests, kept out of the corpus):
  `np.place` and `ndarray.byteswap()` segfault (NumSharp's byteswap returns a copy, or the array itself with
  `inplace: true`: a cell has no byte order), `np.choose` / `np.put` / `np.putmask` copy cells raw, `bytes → T` skips
  UTF-8 validation, `T → complex` puts the real part into the imaginary part, a short cell copied onto itself
  (`a[...] = a[:1]`) becomes `''`, and a buffered `T` input with an allocated `T` output raises MemoryError in
  `np.nditer` (NumSharp returns the values).

## Text ↔ number

NumPy's legacy casts, byte for byte, through every route that converts: `astype`, setitem / `fill` / `full` / `ones` /
`pad` / `copyto` / `np.array`, numeric ufuncs into a text `out=` (with `where=` the out's text is parsed into the loop
dtype first, as NumPy's READWRITE buffer does), `np.add` under a string `dtype=` over numbers, numeric-`dtype=`
reductions and `np.cov` over text (parsed first).
- Number → text is the scalar's `str()` (`Backends/Strings/NDTextFormat.cs`: the float positional / scientific windows
  per type, float16 / float32 shortest digits, complex forms; an unsized `'U'` / `'S'` takes `REQUIRED_STR_LEN`: 21 for
  int64, 32 for the floats, 64 for complex). NumSharp's `char` converts as uint16 and `decimal` as float64. A Python int
  (a C# `BigInteger`) goes through `NDTextFormat.FormatPythonInt`, which refuses more than 4300 digits with CPython's
  `str()` ValueError, so a `U` / `S` / coercing `T` store of one, an unsized discovery and `StringDType(na_object=10**4300)`
  raise as in NumPy.
- Text → number is CPython 3.12's `int()` / `float()` / `complex()` (`NDTextParse.cs`, with the pinned Unicode tables of
  `Utilities/PyUnicode.cs`: decimal digits of every script, `str.isspace()`, the 4300-digit limit, strict UTF-8 for
  `S → complex`). Overflow texts are the LP64 ones (`Python int too large to convert to C long`).
- `NDTextConversion.cs` walks NumPy's raw assignment order and commits blocks of 128 (a failing element leaves the
  earlier blocks written, as NumPy does), converts only the selected runs under a mask, and quotes the NumPy scalar in an
  array cast's error (`np.str_('x')`) but the Python object in a store of a C# string (`'x'`).
- `Converts` and the other converter classes speak NumPy's text too: number → text is the scalar's `str()` and text →
  number NumPy's store of a Python str, culture-invariant, through ONE pair of helpers (`Utilities/Converts.NumPyText.cs`:
  `NumPyStr`, `ParseNumPyText<T>`, `NumPyTextTruth`) that every route reaches (the typed / object `To*` and `ToString`,
  every `ChangeType`, `FindConverter` / `Converts<T>`, `NonGenericConvert`, `TypelessConvert`, `ArrayConvert`,
  `py.complex`). A string source must never reach `CreateIntegerConverter`: its `IConvertible.ToDouble` parses in the
  CURRENT culture and truncates. An `IFormatProvider` is ignored for dtype values; a char target follows astype (numbers
  through the int32 intermediate, text as a uint16); `np.loadtxt` stores a converter's result by `PyArray_Pack`
  (`ReductionInitial.TryPack`), never `System.Convert` (which rounds). Gates `test/NumSharp.Tests/Casting/ConvertsNumPyTextTests.cs`,
  `TextNumberConversionTests`.
- The float digits are `Backends/Printing/FloatDigits.cs` (Ryu shortest digits per precision, exact half-even `%f` /
  `%e` rounding by one 64×64 → 128-bit multiply, tables generated from `BigInteger`), which also feeds scalar str / repr
  (`NDTextFormat.FormatFloatAscii`), array printing (`Dragon4`) and `np.strings.mod`. TRAP: the .NET 10 runtime's `"R"`
  writes 16 digits for 2^-25 and 2^-958 that read back as the neighbouring double, where NumPy prints 17: never format
  NumPy's digits through `"R"`.
- `None` into fixed-width text stores `'None'` cut to the width, and into a `StringDType` its NA object or `'None'` (the
  general rule for `None` in a store: `creation-manipulation.md`).

## Gates

- The string oracle: `test/oracle/gen_strings_oracle.py` → `test/NumSharp.Tests.Oracle/Strings/corpus/strings_<section>.jsonl`,
  one file per section (`--sections titer,cvtext` regenerates a subset, byte-identical to a full run; the generator warns
  past 50 MiB per section, for GitHub's 100 MiB file limit). Every case is strict. `T` values travel as values, never
  cells (`"t"` + `"values"`, `{"na": true}`), and `tobytes()` cells are compared with the heap pointers masked; NumPy's
  `T` → complex defect stays invisible by construction (the cast reads only texts whose two parts are equal).
- The soak: `test/oracle/soak_strings_oracle.py` draws fresh random cases for every computing family over `S` / `U` / `T`
  in random layouts, NA kinds and texts holding NULs, non-ASCII and arena / heap sizes; `StringOracleTests.StringSoak`
  replays the directory `NUMSHARP_STRING_ORACLE_SOAK_DIR` names (Inconclusive when unset; the corpora are not committed).
- Unit tests `test/NumSharp.Tests/Backends/Strings/*` (call order: `StringCallOrderTests`, `StringShapeDecidedTests`; storage and moves:
  `StringDTypeStorageTests`, `StringDTypeFillTests`, `StringDTypeFaultTests`, `StringDTypeHeapPagesTests`; conversions:
  `TextNumberConversionTests`); `DTypes/StringDTypeDescriptorTests.cs`, `Backends/Strings/StringDTypeProtocolTests.cs`.
- Coverage of a change is measured on the lines it ADDED (a cobertura run of the unit and oracle suites joined with
  `git diff -U0 <base>`). In a mutation pass a crashed test host counts as a kill (a corrupt cell can end it without a
  "Failed" line); `NUMSHARP_DEBUG_GUARD_PAGES=1` bypasses the buffer pool, so a junk-filled-pool test cannot see an
  unwritten byte under it: run a mutation pass with and without guard pages.

## Traps

- A lone surrogate cannot travel in `[DataRow]` or `[DynamicData]` (UTF-8 metadata, XML-serialized rows): keep such
  strings in the test body. A raw U+2028 / U+2029 / U+0085 in a C# string literal or comment is a C# LINE TERMINATOR
  (CS1010 / CS1056 far from the cause): write them as `\u2028` escapes.
- Python side: iterating a str that holds U+110000 creates it as a one-character str (CPython's SystemError); encode such
  text with `s.encode('utf-32-le', 'surrogatepass')`, never `for c in s`. `zipfile.writestr(name, bytes)` stamps the
  current time, so write reproducible `.npz` files with `ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))`. The oracle
  harness reads an expectation's `self` key as receiver identity: never name another field `self`.
- `cond ? long : BigInteger` types BOTH arms `BigInteger`.
- NumPy's `np.power` with a negative integer exponent under `where=` into a text out raises or not depending on calls
  made before it in the process: generate such cases with non-negative exponents.
- A function given `out=` returns that array: a harness that disposes results disposes the caller's buffer.
