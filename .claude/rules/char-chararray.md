---
paths:
  - "src/NumSharp.Core/Char/**"
  - "src/NumSharp.Core/APIs/np.char*.cs"
  - "src/NumSharp.Core/Backends/Strings/NDCharObjectPath.cs"
  - "src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Text.Strip.cs"
  - "src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.MaskGather.cs"
  - "test/NumSharp.Tests/Backends/Strings/{CharModuleTests,CharArray,CharObjectPathTests,CharStringMethodsTests,ListArray}*.cs"
---

# `numpy.char` and `chararray`

The module, the class and the object path, byte-identical to NumPy 2.4.2 (the string classes and the computing rule:
`strings.md`; the functions numpy.char re-exports: `strings-methods.md`).

## The module

- `np.@char` is the MODULE object `CharModule` (`APIs/np.char.cs`, `[ModuleName("np.char")]`) with an implicit conversion
  to the `Char` dtype it used to be, so `astype(np.@char)` / `a.dtype == np.@char` / `(DType, DType)` tuple keys keep
  working (a descriptor member needs `((DType)np.@char).itemsize`).
- The six comparisons and `compare_chararrays` strip trailing NULs and ASCII whitespace (NumPy's `rstrip=True` loop: one
  `Rstrip` flag of the text comparison key, every kernel form; `ILKernelGenerator.Text.Strip.cs`: a bit-mask body for
  ≤ 64 code units, managed helpers beyond); `add` is `np.add`; `multiply` is `np.strings.multiply` with a TypeError
  re-raised as `ValueError("Can only multiply by integers")`.
- Array arguments are `object`: a `string` is `str`, a `byte[]` is `bytes`, a `byte[][]` a list of bytes; an `S` / `U`
  pair answers NumPy's `NotImplemented` as `null`; a `StringDType` operand is `comparison of non-string arrays`;
  `compare_chararrays`' `cmp` is one `object` parameter (NumPy's `s#`: a `string` or `byte[]`, anything else `a
  bytes-like object is required, not 'int'`).
- The re-exports (`np.@char.center` / `ljust` / `rjust` / `zfill` / `expandtabs`, the search family with `str_len`, the
  strips and `replace`, the nine predicates, the case maps and `translate`, `mod`, `encode` / `decode`:
  `src/NumSharp.Core/APIs/np.char.*.cs`)
  ARE `np.strings`' functions: array_likes through `CharModule.TextOperand` (a `byte[]` is Python bytes), a null OPTIONAL
  argument is NumPy's default (`OptionalOperand`), a null required one `ArgumentNullException`; the class of the result,
  its layout and every error are `np.strings`'. `np.@char.partition` / `rpartition` are NumPy's
  `np.stack(np.strings.partition(a, sep), axis=-1)`: ONE plain array, last axis 3, NumPy's stack layout; the separator is
  cast to the text's dtype first (`c.partition(1)` splits on `'1'`), and a `StringDType` text needs a `StringDType`
  separator. `replace` casts a dtype-less `old` / `new` (a string, a `byte[]`, a number, a chararray element stand-in) to
  the text's dtype. `translate`'s `deletechars` reaches only `bytes.translate`.

## `chararray` as data: `NDCharArray : NDArray`

`Char/NDCharArray{,.Strip,.Indexing,.Methods}.cs`, `APIs/np.char.array.cs`.
- `np.@char.chararray(shape, itemsize, unicode, buffer, offset, strides, order)`: a `string` buffer FILLS,
  `ReadOnlyMemory<byte>` is Python's read-only `bytes`, `byte[]` a writable `bytearray`, an `NDArray` is borrowed.
  `np.@char.array` / `asarray` follow NumPy's branches (str / bytes: `itemsize = len`, shape `len // itemsize`, Python's
  ZeroDivisionError as `DivideByZeroException`; an S / U array: a view, `itemsize // 4` for U, so U is always cast;
  numbers, bools, None and StringDType through NumPy's `tolist()` route, value-based widths; floats cast once, then
  narrowed).
- The finalize rule (`dtype.char in 'VSUbc'`, else `ValueError: Can only create a chararray from string data.`) runs
  before every allocation, and `view(dtype)` reads the OLD dtype (a `view(np.uint32)` chararray exists; its own views
  raise).
- A full integer index (0-d integer arrays count) is a READ-ONLY 0-d stand-in of Python's `rstrip()` (29 code points for
  U: `char.IsWhiteSpace` + U+001C–U+001F; 6 bytes for S), cut from a per-thread 2 KB arena whose Disposer the pieces
  share (`Char/NDCharArray.StandIn.cs`), its address taken from the strides; `item()`, `tolist()`, `take(i)` and the flat
  iterator do not strip; iteration yields stripped .NET values (1-D) or chararray rows.
- Every member re-declared with `new` keeps the class with NumPy's ownership and identities (`real`, `squeeze()`,
  `astype(copy=False)` and `to_device` return `this`); `repr` is `chararray(...)` with stripped elements;
  `NDArray.view(Type)` / `view(DType, Type)` are `view(type=)`; `T` and `copy` build the subclass directly. String mask /
  index-array gathers run directly (`DirectILKernelGenerator.MaskGather.cs`; the flat gather kernels bake ANY constant
  slab width up to 256 bytes, `FixedCopyKindFor`, numeric row gathers too).
- Divergences (`[Misaligned]`): `np.char.array(..., order=)` of string data is a chararray (NumPy: a plain ndarray, whose
  `np.asarray` drops the class; the oracle's one class excuse); an `itemsize=0` chararray (S0 / U0) is refused; a stripped
  value ending in NUL loses the NUL in the stand-in.

## `chararray` behavior: the static rule and the boundary

`Char/NDCharArray{.Operators,.Arg}.cs`, `Char/NDCharArrayBoundary.cs`.
- The OPERATORS bind at compile time (`NDMatrix`'s rule): `==` `!=` `<` `<=` `>` `>=` `+` `*` `%` on `NDCharArray`, seven
  forms each (a chararray against a chararray / `NDArray` / `object` / `string`, both orders), replaying Python's dispatch.
  A chararray on the right runs its REFLECTED method before a plain array's, so `u == c` strips. An `S` / `U` mix
  (`NotImplemented`) falls to the plain array's unstripped comparison, else Python's fallback (`==` a 0-d False, `!=` True,
  an ordering `TypeError: '<' not supported between instances of 'chararray' and 'bytes'`). `+` is `np.char.add` (a
  chararray VIEW); `*` is `np.char.asarray(np.char.multiply(...))` (a `U` copy); the C# `null` keeps `NDArray`'s reference
  test for `==` / `!=`. A chararray behind an `NDArray` reference uses `NDArray`'s operators and members, plain. The
  chararray's `%` is `asarray(mod(self, x))`; with the chararray on the RIGHT, Python's dispatch decides (`__rmod__` is
  NotImplemented): a str / bytes / element on the left formats the chararray as ONE argument object, an ndarray on the
  left runs np.remainder's TypeError, a Python number CPython's `unsupported operand type(s) for %`. CPython's `%` converts
  an ndarray argument through NumPy's slots on every path (`'%e' % np.array([1, 2])` is NumPy's `only 0-dimensional arrays
  can be converted to Python scalars`).
- `argmax` / `argmin` / `argpartition` raise `ValueError: Can only create a chararray from string data.` after NumPy's
  checks (with `out=` they compute). `partition(sep)` is the STRING partition (`np.strings.partition` stacked on a new last
  axis); its `axis=` forms are NumPy's `TypeError: chararray.partition() got an unexpected keyword argument 'axis'`, which
  `np.partition` / `np.median` / `np.quantile` / `np.percentile` raise too.
- The member return rules: rule 1 `asarray(f(self, …))` (`NDCharArray.AsArrayOfFresh` TAKES OVER a fresh `U` result,
  owning, no copy, and VIEWS an `S` one); rule 2 (strip / replace) the function's chararray view
  (`NDCharArray.FunctionResult`); rule 4 a plain int64 / bool array. The members read the STORED text
  (`np.char.array(['ab  ']).center(6, '*')` is `'*ab  *'`, `.isalpha()` False, `.islower()` True); a 0-d chararray gives a
  0-d array; `isdecimal` / `isnumeric` refuse an `S` chararray (no bytes loop); `encode` / `decode` return the function's
  PLAIN `S` / `U` result as it is.
- The np.* FUNCTIONS find a chararray at run time: `NDCharArrayBoundary.Sees(a)` at the outermost call, then a rule by
  NumPy's mechanism:
  - Kept (a method or an `np.asanyarray` view: `reshape`, `ravel`, `transpose`, `squeeze`, `sort`, `repeat`, `tile`, the
    split family, the `*_like` family where a numeric dtype is the finalize error, `np.copy(subok=True)`, `np.take` (0-d:
    the unstripped scalar), `np.flip` (0-d: the stripped element), `np.meshgrid` per input);
  - Wrapped (`__array_wrap__`, a chararray VIEW: text ufunc results, `insert` / `delete` / `diagflat` / `kron` /
    `take_along_axis` / `setdiff1d`, np.unique's values, which with a return flag merge stripped-equal runs);
  - Plain (`np.asarray` converters: the stack family, `append`, `lexsort`, the linalg / fft / emath functions);
  - IndexResult (the `np.argmax` family: the finalize error);
  - NumPy's trivial-loop copy (`check_for_trivial_loop`): a 0-d or ≤ `np.getbufsize()` 1-d chararray input cast to bool
    or StringDType is copied AS the subclass, so the logical ufuncs raise the finalize error; the same holds for every
    function with a second text input whose StringDType argument meets a short chararray (inputs in order, the first
    larger one stopping later copies); past it NumPy computes a PLAIN result, which a rule-2 member refuses
    (`NotSupportedException`) and a pad member converts with `np.char.asarray`. The marks are
    `NDCharArrayBoundary.WrappedUfunc` / `PlainUfunc` / `InsideUfunc`, the check `TrivialLoopCopyError` at NumPy's point
    (after the input casts, before the broadcast). Only the C# literal `true` (marked `WasPythonBool`) is NumPy's
    `where=True`.
- TRAPS: the boundary shares `NDMatrixBoundary`'s depth, and `PlainFor` opens a region for a chararray receiver too, so a
  function must decide its chararray rule BEFORE any `PlainFor`. A public function that passes a chararray to another
  must compute plain: the test hook `AuditInternalCallers` (a stack walk at every rule) replays the string-oracle cases
  with chararray operands and must find no library caller; the audit skips only frames named like the function, so a
  function that forwards a chararray to a DIFFERENTLY named public function (partition → `np.stack`) computes inside
  `NDCharArrayBoundary.Inside()`. A comparison operator's typed result must share the driver result's storage
  (`AsGeneric` inside `using`): an untyped wrapper left to the finalizer stranded one buffer per comparison, and
  `MakeGeneric` reports OWNDATA False.
- Divergences (`[Misaligned]`, `CharArrayBehaviorTests`): the static rule; `c == null`; `np.squeeze`'s identity; a view of
  a stand-in is read-only; `chararray + StringDType` refused by the operator (call `np.add`); numpy.ma holds a chararray's
  data plain.

## The object path: `split` / `rsplit` / `splitlines` / `join` and `NDListArray`

`APIs/np.char.split.cs`, `Char/NDCharArray.Split.cs`, `Char/NDListArray.cs`, the engine `Backends/Strings/NDCharObjectPath.cs`.
- NumPy's `_vec_string` is one NDIter walk in C order over the broadcast operands (no casting, no buffer); per element,
  CPython 3.12's `str` / `bytes` method statement for statement over 1-byte or 4-byte units, in CPython's order: the
  NumPy scalars (a ONE-unit `U` text above U+10FFFF is `SystemError: invalid maximum character passed to
  PyUnicode_New`), the descriptor check (a StringDType NA), `maxsplit`'s `__index__` / `keepends`' truth, the separator's
  type (bytes.split takes any buffer: `np.int64(5)` splits on its 8 bytes), `empty separator`, the pieces; the first
  failing element ends the call. In the object path a C# `null` IS Python's None (an object array), where the rest of the
  module raises `ArgumentNullException`.
- `split` returns an `NDListArray` (NumPy's object array of lists): one int64 list id per element (`Ids`, NumPy's 8-byte
  object pointer) into a shared, append-only `NDListHeap` (`Char/NDListHeap.cs`: each list a range of pieces, each piece
  its exact units, a per-piece `np.str_` flag: unicodeobject.c returns the element's own `np.str_` when the separator is
  longer or of a wider kind; the repr prints `np.str_('ab')`). Every structural member is the same NDArray operation on
  `Ids` (a view shares the ids, a copy copies them, the lists stay shared), so strides, flags, base and errors are NumPy's
  for the object array.
- The members that read and write the lists' VALUES (`Char/NDListArray.{Engine,Compare,Sort,Reduce,Arithmetic,Assign,TextOperands}.cs`)
  are NumPy's object loops over CPython's list semantics: `list_richcompare` (ordering a str item against a bytes item is
  CPython's TypeError), the truth members over non-emptiness, `astype` (object converts only under `'unsafe'`), the sorts
  over the lists' dense ranks (NumPy's introsort for the default kind, so equal lists land as NumPy's do; timsort for
  `'stable'`), `searchsorted` with `npy_binsearch`'s carried bounds, the reductions in NumPy's flat reduction order
  (`initial` / `where=` / `out=`), `+` / `*` / `dot` with CPython's errors, and assignment as pointer copies (a list of
  another heap is appended first; the writeability before the index).
- Indexing: basic indexing gives a view; one integer per axis in slice notation (`r["0"]`) a 0-d list array that STANDS
  FOR NumPy's Python list (`IsStandIn`: against an ndarray it is its items, `np.asarray(list)`; `==` / `!=` then run
  NDArray's operators, whose no-loop pair is NumPy's all-False constant); the `long[]` / `int[]` indexer a NEW 1-D piece
  array (NumPy: the list itself, which `item()` gives; `[Misaligned]`).
- Every member is ONE NumPy signature with all its parameters: `dtype=` of `sum` / `prod` / `cumsum` / `cumprod` / `trace`
  / `mean` / `var` / `std` / `clip` (NumPy's ufunc dtype checks, then the object loop's conversion error per class),
  `out=` of `cumsum` / `cumprod` (both `PyUFunc_Accumulate` routes' texts; NumPy 2.4.2 segfaults on a 0-d out, NumSharp
  raises the wrong-size text, `[Misaligned]`), `trace`, `conj`, `mean` / `var` / `std` (ndarray outs, `_mean` / `_var`'s
  statements), `where=` of `clip` and of `mean` in a `dtype=` loop, `mean=` of `var` / `std`, `view(dtype, type)`
  (`[Optional]` parameters, so an omitted argument and an explicit null are told apart), `reshape(shape, order, copy)`,
  `tofile(fid, sep, format)`. A NumPy result holding a non-list object (an int identity, `imag`'s zeros) is
  `NotSupportedException`; a `dtype=` loop whose NumPy result is not a list (the bool loop's `np.True_`, an empty sum's
  number) is `NotSupportedException` naming the composition `np.sum(lists.astype(dtype), ...)`; `dtype` / `data` need an
  object dtype (absent by design), and `ctypes` / `dump` / `dumps` exist on no NumSharp array.
- Speed: the members that read many lists go through `ListPointers` (raw pointers into a heap; never held across an
  append); the builders create their heap once at the counted size (`NDListHeap.Reserved`, whose appends THROW past the
  reservation: the pool's size-bucket slack would hide an under-sized heap completely); a member that selects lists
  (`max` / `min` / `clip`) shares the array's heap.
- `join` takes `seq`'s class, as wide as the longest result; over an `NDListArray` it raises NumPy 2.4.2's own
  `TypeError: currently only the no-argument instantiation is supported; use np.dtype instead.` (NumPy cannot finish
  `np.char.join(' ', np.char.split(a))`); over an EMPTY one NumPy returns an empty object array, which the one
  `join(object sep, object seq)` (an `NDArray` result) cannot hold: `NotSupportedException`.

## Gates and traps

- String oracle sections `strchar`, `strchararray`, `strcharops`, `strcharfn` (99 functions), `strcharlist` (strict),
  `strcharobj` (strict, expectation kind `lists`), `strcharmeth` (strict, with the boundary audit); unit tests
  `CharModuleTests`, `CharArrayFastPathTests`, `CharArrayBehaviorTests`, `CharObjectPathTests`, `CharStringMethodsTests`,
  `ListArrayMembersTests` / `ListArrayComputeTests` / `ListArraySignatureTests`.
- Under `DOTNET_TieredCompilation=0` CoreLib's ReadyToRun bodies are never re-optimized, and its `memmove` and span
  searches ran 3–4× slower inside a per-element loop than with `DOTNET_ReadyToRun=0`: the engine copies pieces inline and
  searches with its own vectors (`FindUnit` / `FindNeedle`); benchmark text code under the default runtime, and check
  TC=0 too.
- Two splits of the SAME text build heaps that number their lists identically, so assigning the lists of a second split
  of the same text cannot show a foreign id stored without its list being imported: assign lists split from a DIFFERENT
  text.
