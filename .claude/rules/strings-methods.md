---
paths:
  - "src/NumSharp.Core/APIs/np.strings*.cs"
  - "src/NumSharp.Core/Backends/Strings/{NDTextElements,NDVStringMethods,NDTextLoop,NDStringMethods,NDTranslate,NDStringMod,NDCodec,PyUnicodeData,PyUnicodeNames}*.cs"
  - "src/NumSharp.Core/Backends/Strings/NDCharObjectPath.{Translate,Mod,ModFast,Codec,CodecFast}.cs"
  - "src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Text.Methods.cs"
  - "tools/gen_unicode_{tables,names}.py"
  - "test/NumSharp.Tests/Backends/Strings/{StringMethodsLanesTests,StringFunctionsPinsTests,StringPredicatesTests,StringCaseMapTests,StringTranslateTests,StringModTests,StringCodecTests}.cs"
---

# The `np.strings` functions

`np.strings` has all 46 of NumPy's names, over `S`, `U` and `StringDType`, byte-identical to NumPy 2.4.2. The ufuncs take
`out=` / `where=` / `dtype=` in NumPy's stage order; the `_vec_string` functions run CPython 3.12's `str` / `bytes` method
per element in C order, the first element that raises ending the call. The storage and the refuse-or-compute rule:
`strings.md`; the comparisons, arithmetic and ordering: `strings-compute.md`; the numpy.char re-exports and the
chararray members: `char-chararray.md`.

## Search, strip, pad, replace, partition, slice

- `str_len`, `find` / `rfind` / `index` / `rindex` / `count`, `startswith` / `endswith`, `strip` / `lstrip` / `rstrip`,
  `center` / `ljust` / `rjust`, `zfill`, `expandtabs`, `replace`, `partition` / `rpartition`, `slice` and the six
  comparison names, with NumPy's arguments, defaults and ufunc stages in NumPy's order: the promoter's no-loop TypeError
  (`_zfill` / `_expandtabs` have no uint64 loop), `T`'s NA compatibility, the same_kind int64 input casts, the broadcast
  text. A ONE-input text ufunc (`str_len`, the predicates, the whitespace strips) names a C# literal by its default DType
  in a refusal (`Int64DType`: NumPy marks no operand of a one-input ufunc weak; `NDStringUfuncs.UnaryDispatchClass`).
- Code: `APIs/np.strings.{Search,Transform}.cs`; `S` / `U` loops `Backends/Strings/NDTextElements{,.Lanes}.cs` behind
  the IL shell `Backends/Kernels/ILKernelGenerator.Text.Methods.cs` and the driver `NDTextLoop` (trivial loop / NDIter /
  rows / masked kernel, plus the fused SIZE passes of the computed-width wrappers); `T` loops
  `NDVStringMethods{,.Short}.cs`.
- Design: an element of up to 64 units is computed from LANE MASKS (bit i of a ulong per unit: equal to a code,
  whitespace of the class, nonzero; the search family is the AND of each needle unit's mask shifted by its index), and an
  element of up to 64 bytes is moved by ONE GATHER (`vpermd` for `U`; for `S`, `vpshufb` of each broadcast 16-byte quarter
  by the index minus the quarter's base plus 0x70 saturating, which reads zero outside the quarter), so no per-element
  branch follows the data. A `T` pair of short cells takes one 16-byte vector per cell (masked to its size, code points
  = lead bytes, `pdep` for the k-th). Wider elements and hosts without AVX2 run the scalar bodies (`*Scalar`, the
  differential tests' reference). `startswith` / `endswith` over a range to the element's end take one compare
  (`NDTextElements.TailMatchWholeRange`), and the 1-D text kernels prefetch 4 KB ahead for elements of 32 bytes or more.
- `partition` over `S` / `U` returns three arrays (NumPy: field views of one structured array; a NumPy `'U0'` part is
  `U1`, `[Misaligned]`).
- NumPy defects NOT reproduced (pinned in `StringFunctionsPinsTests`): `zfill('')` reads past its output (NumPy 2.4.2
  reads uninitialized memory for an empty element's sign: oracle receivers of zfill hold no empty element); a `T`
  `replace` with a negative count other than -1 and a longer replacement writes past its buffer; a `T` `slice` with a
  step past int's range never returns; the `T` tailmatch reads past the haystack.
- Gates: oracle `strfn` / `tfn`, zero excuses; `StringMethodsLanesTests` (every lane body against its scalar body over
  every width, `FencedMemory`, short cells against the general cores, IL against managed kernels) and
  `StringFunctionsPinsTests`.

## Predicates

- `isalpha`, `isdigit`, `isspace`, `isalnum`, `islower`, `isupper`, `istitle`, `isdecimal`, `isnumeric` are NumPy's ufuncs
  (`out=` / `where=` / `dtype=` in NumPy's stage order, `_get_dtype` first; no `S` loop for `isdecimal` / `isnumeric`)
  over `S` (NumPy's ASCII rules), `U` and `T` (CPython's `Py_UNICODE_IS*` per code point, the text without its trailing
  NULs, empty is False). A `T` NA is False under a NaN-like NA object, its text under a string one, NumPy's ValueError
  under any other, and never at a masked-off position.
- The Unicode tables: `tools/gen_unicode_tables.py` → `Backends/Strings/PyUnicodeData.Tables.cs`, CPython 3.12's
  `_PyUnicode_TypeRecord` flags, decimal values and full case mappings (Unicode 15.0.0 on every .NET: .NET 10's own
  tables are Unicode 16.0), the pinned 64K-word BMP table (`flags | record << 16`) one `vpgatherdd` reads, and a SHA-256
  pin test. `PyUnicode`'s digit / space tests and `PyRepr.IsPrintable` read them (a text repr escapes a Unicode-16
  character as NumPy does, where .NET 10's tables would print it raw).
- Loops: `Backends/Strings/NDTextElements.Properties.cs` (`S` / `U` elements as lane masks; `U` runs ASCII compares until
  the CALL meets its first non-ASCII unit, then gathers: a sticky escalation whose state is the kernel's auxdata
  `KindState`; a unit past U+FFFF runs the element's scalar loop) and `NDVStringMethods.Properties.cs` (`T`: one 16-byte
  window per short cell, an arena text by ASCII chunks or a vector UTF-8 decode, two `vpshufb` per 16-byte window).
- Traps from the disassembly: a `DynamicMethod` re-tests a static class's initialization at every read of a static
  field (read the table pointer once per call into the auxdata); `Sse2.ShiftRightLogical128BitLane` with a non-constant
  count is a CALL of the software fallback; a rare call merged into the hot vector loop spills its registers (run the
  whole element scalar instead); an early exit tested after every chunk mispredicts (test the first chunk only); C#
  reduces a 32-bit shift count modulo 32 (`uint >> 32` shifts by 0).
- Gates: oracle `strprop` / `tprop`, zero excuses; `StringPredicatesTests` (the table pin, elements and UTF-8 forms
  against the code-point loop over every width, `FencedMemory`, IL against managed kernels).

## Case maps

- `upper`, `lower`, `swapcase`, `capitalize`, `title` are NumPy's `_vec_string(a, a.dtype, name)`: a fresh C-ordered result
  of the operand's own dtype (`T`: a new descriptor with its NA object and coercion), each element the `bytes` method
  (`S`: ASCII letters only, the length kept) or the `str` method (`U` / `T`: CPython 3.12's full mappings, up to three
  code points per character, `lower_ucs4`'s final sigma, title's previous-cased state of the INPUT character) of NumPy's
  scalar (`S` / `U` without trailing NULs, `T` with them); an `S` / `U` result is cut to the width
  (`upper(np.array(['ß'], 'U1'))` is `'S'`).
- Errors: a non-text operand is `string operation on non-string array`; a `U` unit past U+10FFFF CPython's `SystemError
  invalid maximum character passed to PyUnicode_New` (raised after the loop: every element's text is the same); a `T` NA
  whose object is not a str `descriptor 'upper' for 'str' objects doesn't apply to a 'float' object`; under a string NA
  object a result equal to the NA's text is stored as the NA (setitem's `na_eq_cmp`).
- Loops (`Backends/Strings/NDTextElements.Case.cs`, `NDVStringMethods.Case.cs`, IL kernels `CaseMap` / `VCaseMap`): `S`
  in 32-byte chunks as a stencil of (previous byte a letter, byte); `U` the same over 32-bit lanes until the CALL meets its
  first non-ASCII element, then per eight units one gather of the BMP flags words and one of the deltas from
  `PyUnicodeData.CaseDeltas` (`[upper | lower | title]` side by side, each lane's table chosen by its index); title
  gathers both of its mappings and selects with the previous-cased mask, carried between chunks in a register
  (`UnitsShiftedUpFrom`); a unit past U+FFFF, a multi-code-point mapping (`EXTENDED_CASE`) or U+03A3 under a lowering map
  sends the element to CPython's algorithm (`CaseMapCodePoints`); `T`: a short ASCII cell in one 16-byte vector, any
  other text decoded, mapped and re-encoded.
- Traps: a gather whose index depends on the previous chunk's carry waits for the previous chunk's gathers; a `T`
  result's arena grows by 25 % steps, each a reallocation that copies and zeroes it, so the case-map driver reserves it
  once (`VStringAllocator.TryReserve` with `NDVStringCopy.ArenaBytesForCopy`).
- Gates: oracle `strcase` / `tcase`, zero excuses; `StringCaseMapTests`.

## `translate`

- NumPy's `_vec_string(a, a.dtype, 'translate', args)`: `(table,)` for `U` (`deletechars` IGNORED),
  `[table] + _clean_args(deletechars)` for `S` / `T`; a fresh C-ordered result of the operand's dtype over the broadcast
  of a and the arguments, the first element (C order) that raises ending the call.
- `str.translate` (`U` / `T`): `table[ord(c)]`; a LookupError keeps the character, None deletes it, an int is a code
  point (`ValueError: character mapping must be in range(0x110000)`), a str replaces it, anything else `TypeError:
  character mapping must return integer, None or str`; dict keys match by Python equality (97.0 / True yes, 'a' / 97.5
  no); a str / bytes table is indexed by code point, a NumPy number keeps everything, None raises `'NoneType' object is
  not subscriptable` at the first lookup. `bytes.translate` (`S`): a 256-byte table or None (`translation table must be
  256 characters long`, raised by every call: an empty element's too, not an empty array's; an `np.bytes_` table drops
  trailing NULs, so one ending in NUL is 255 long) and a buffer delete set. `T`: a non-str NA is `descriptor 'translate'
  for 'str' objects doesn't apply to a 'NoneType' object`, `deletechars` is `str.translate() takes exactly one argument
  (2 given)`, a surrogate result `result array type does not match underlying function`, a result equal to a string NA is
  the NA. A C# `IDictionary` argument is the 0-d object array holding the dict (`VecArg.Convert`).
- Code: `APIs/np.strings.Translate.cs`; the compiled tables `Backends/Strings/NDTranslate.cs` (a dense code-point table
  sized by the call's work, 256–65,536 entries, plus the dict keys past it; route flags); the kernels
  `NDTextElements.Translate.cs` (`S` map + compaction, `U` one `vpgatherdd` per eight units, the exact path for the rest)
  and `NDVStringMethods.Translate.cs` (`T` ASCII cells by byte maps, other text by `TranslateUtf8`: ASCII on a predicted
  branch, the rest decoded four bytes at a time branch-free); the walk for arrays of tables `NDCharObjectPath.Translate.cs`.
- Traps: a branch-free UTF-8 decode serializes on every lead byte's load, so keep ASCII on a branch; CPython looks a `U`
  unit up as `PyLong_FromLong((long)c)`, a 32-bit long on Windows (a unit ≥ 2^31 is a negative key there; NumSharp models
  LP64, and the oracle keeps keyed units below 2^31); a mutation-test pin with a key at the dense table's end keeps the
  table off the gather route (the gather boundary needs a table whose keys all sit in the dense range).
- Gates: oracle `strtrans` / `ttrans`, zero excuses; `StringTranslateTests`.

## `mod`

- NumPy's `_to_bytes_or_str_array(_vec_string(a, object, '__mod__', (values,)), a)`: CPython 3.12's `%` per element in C
  order (`bytes.__mod__` for `S`, `str.__mod__` for `U` / `T`) over the broadcast of a and `np.asarray(values)`, the first
  element that raises ending the call; a `U` / `S` result as wide as the longest result (at least 1), a `T` result a
  FRESH default `StringDType` whose surrogate or unit past U+10FFFF raises "Invalid unicode code point found" AFTER the
  walk (a later formatting error wins).
- CPython's rules hold per element: a tuple's items are the arguments, any other value is one argument, a mapping
  (NumPy's numeric scalars, `np.bool`, dicts, lists, ndarrays; for bytes formatting not bytes) feeds `%(key)` and
  silences "not all arguments converted"; the parse runs as it formats, so a malformed format's error depends on the
  argument; NumPy's scalars meet the conversions (`'%r' % np.float64(1.5)` is `np.float64(1.5)`, `%d` truncates floats,
  `%x` refuses floats and `np.bool`, bytes `%s` writes a scalar's raw buffer). NumPy's `"%c"` error text on 3.12 is `%c
  requires int or char`.
- Code: `APIs/np.strings.Mod.cs`; the compiled parse `Backends/Strings/NDStringMod.cs` (`ModFormat`, once per distinct
  format element; `FastSpec` = one positional spec, no `%(key)`, no `*`); the walk, CPython's argument machinery and the
  result building `NDCharObjectPath.Mod.cs` (`ModWalkState`: `U` / `S` rows laid out from the first 64 elements, re-laid
  out for a longer one, narrowed once; `T` packed per element); the fast paths `NDCharObjectPath.ModFast.cs` (the chunk
  path writes each field straight into its row; `ModFastPathsDisabled`, a `[ThreadStatic]` hook for the differential
  tests); the float digits `Backends/Printing/FloatDigits.cs` (`strings.md`).
- Trap: a C# `string` format argument converts to a 0-d `U` array at the call site, a temporary of the CALLER's (the leak
  sweep counts it against `mod` unless the entry disposes it).
- Gates: oracle `strmod` / `tmod`, zero excuses; `StringModTests`, `FloatDigitsTests`.

## `encode` / `decode`

- NumPy's `_to_bytes_or_str_array(_vec_string(a, object, 'encode' | 'decode', _clean_args(encoding, errors)),
  np.bytes_(b'') | np.str_(''))`: the method by `a`'s dtype BEFORE any argument (an `S` operand of encode is
  `AttributeError: type object 'bytes' has no attribute 'encode'`, a `U` / `T` operand of decode the `str` one),
  arguments up to the first None, broadcast with NumPy's multi-iterator text; per element in C order the NumPy scalars (a
  one-unit `U` text past U+10FFFF: CPython's SystemError), the descriptor check (a `T` NA object that is not a str),
  Argument Clinic's checks (a str, no surrogate, no NUL), then CPython 3.12's codec (`bytes.decode` of EMPTY bytes is ''
  before any lookup). The result: empty gives `S1` / `U1` of the broadcast shape; else as wide as the longest result
  (trailing NULs count for the width); `T` encodes to `S`.
- Code: `APIs/np.strings.Codec.cs`; the codecs and handlers `Backends/Strings/NDCodec{,.Encode,.Decode,.Handlers}.cs` (on
  `PythonCodecs`: CPython's lookup, aliases, error texts; each coder's inline handlers and the generic call, CPython's
  error ranges; a codec NumSharp lacks (multi-byte CJK, `utf-7`, IDNA, the escape codecs, `mbcs`) is a
  NotSupportedException that names it); `namereplace`'s names `PyUnicodeNames{,.Data}.cs` (CPython 3.12.12, Unicode
  15.0.0, generated by `tools/gen_unicode_names.py`, pinned by a SHA-256 over every code point; a pinned BMP length memo,
  `NameLengthsPointer`, that vector code gathers from); the walk `NDCharObjectPath.Codec.cs` (`CodecWalk`, `CodecRows`:
  rows sized by an exact estimate, the first 64 elements coded, or for `namereplace` the whole chunk from the memo,
  gathered eight at a time (`NameReplacedWidest`), re-laid out, narrowed once; `NDArray.ForOverwrite` rows; the first
  element coded through the general route before any run); the fast runs `NDCharObjectPath.CodecFast.cs` (call-free inner
  runs over `U` / `S` / `T` with whole-vector row stores and an outer loop for the refused element; `namereplace` copies
  its cached names, zero-padded to 32-byte chunks behind a length byte; `CodecFastPathsDisabled`, a `[ThreadStatic]` hook;
  `ResetNameReplacementsForTests` / `PyUnicodeNames.ClearNameLengthsForTests` forget the `namereplace` caches).
- Each run declares its width (`CodecRows.FastWidthLimit`), and a re-layout for an element that fits it stops there: a
  growth policy that grows by half again can carry the rows past a fast run's width limit although no element needs it,
  and every later element then takes the per-element path.
- Traps: a loop with cold-path calls keeps its state in stack slots (split call-free inner runs from the refusal path);
  decoding into a scratch and reloading it with a wider load stalls store forwarding; a `StringDType` cell's size varies,
  so an exact read through a switch mispredicts per cell (a short cell's 16 bytes are always readable); a per-lane
  formula evaluated over all four vectors whatever the element width costs tens of instructions per element (evaluate only
  the vectors the element spans; a `vpgatherdd` with an empty mask still costs); a length a run miscounts hides when the
  longest element is the walk's first (general route) or sits in the result's last rows (exact writes): the pin sizes the
  rows one byte short with the first elements and puts the long element inside the run. A C# run entered once per NDIter
  chunk runs ONCE per call over a contiguous array, so tiered compilation keeps it, and every routine it calls without
  inlining, on tier-0 code for a short program's first calls: the runs, the element routines, the row writers and the
  coders are `[MethodImpl(AggressiveOptimization)]` and `NDCharArray.TrimmedUnits` AggressiveInlining (find what is left
  with `DOTNET_JitStdOutFile` + `DOTNET_JitDisasmSummary=1`: the Tier0 lines compiled after the first call).
- Gates: oracle `strcodec` / `tcodec`, zero excuses; `StringCodecTests`; the codec is validated by a randomized soak
  (long arrays whose elements mix content kinds so the fast runs take, refuse and resume element by element; widths
  around the runs' 32-unit limit; views over long bases; argument arrays; every codec and handler) replayed with AVX2,
  without AVX2, without hardware intrinsics and under guard pages, both frameworks.
