---
paths:
  - "src/NumSharp.Core/IO/**"
  - "src/NumSharp.Core/APIs/np.{save,load,array2string}*.cs"
  - "src/NumSharp.Core/Backends/Printing/**"
  - "src/NumSharp.Core/Casting/NDArray.ToString.cs"
  - "test/NumSharp.Tests/IO/**"
  - "test/oracle/{gen_npy_oracle,verify_npy_interop}.py"
  - "test/oracle/interop_write.cs"
---

# File I/O and printing

## `.npy` / `.npz`

A port of NumPy 2.4.2's `numpy/lib/_format_impl.py` (NEP-01) in `IO/{NpyFormat,NpzFile,PyLiteral}.cs`, with
`APIs/np.{save,load}.cs`. The writer is byte-for-byte identical to NumPy's own `np.save`, not merely readable by it:
format versions 1.0 / 2.0 / 3.0 (2- or 4-byte header length; latin-1 or UTF-8), 64-byte `ARRAY_ALIGN` data alignment
(mmap-ready), `fortran_order` both ways (write through the transpose; read through reshape-reversed + transpose), and
big-endian files byte-swapped to native on read (per component: `>c16` swaps in 8s, `>U1` in 4s). Headers parse through
a Python-literal parser (`PyLiteral`, standing in for `ast.literal_eval`), so key order, whitespace, quoting, trailing
commas and the Python 2 `3L` suffix all work; `max_header_size` (default 10000) bounds hostile input, and `allow_pickle`
lifts it.
- `np.load` returns `object`: `NDArray` for a `.npy`, `NpzFile` for a `.npz`, dispatching on the magic bytes (NumPy's
  content-dependent return; C# has no union type). `np.load_npy` / `np.load_npz` are the typed forms, and each reports a
  directed error if handed the other kind. `NpzFile` is lazy and cached, takes `"w"` or `"w.npy"` as keys, exposes
  `.Files` stripped, supports `npz.f.weights` (NumPy's `BagObj`), and must be disposed. It is safe to share between
  threads: a cached member is returned without a lock, every read of the archive (a first load, `GetRawBytes`,
  `IsArray`'s magic check, `Dispose`) runs under one gate because a `ZipArchive` reads one stream, and two threads loading
  the same member first get one instance.

| Dtype | Maps to | Notes |
|-------|---------|-------|
| Boolean/SByte/Byte | `\|b1` / `\|i1` / `\|u1` | single-byte types take the `\|` prefix, never `<` |
| Int16…UInt64, Half, Single, Double | `<i2`…`<u8`, `<f2`, `<f4`, `<f8` | direct |
| Complex | `<c16` | `<c8` widens to `Complex` on read |
| `S` / `U` (fixed width) | `\|S<n>` / `<U<n>` | byte for byte, mmap too; a `>U` file swaps per 4-byte code point |
| `StringDType` (`T`) | `\|O` + NumPy's pickle | below |
| Char | `<U1` | WRITES `<U1` (2-byte UTF-16 widened to 4-byte UCS-4; non-BMP rejected); a `<U1` file reads back as a `U1` string array |
| Decimal | — | `NotSupportedException`: no NumPy dtype (cast to Double) |

- Object arrays (but a `T` array's pickle), structured/subarray dtypes, `datetime64`/`timedelta64`, `|V`, `<f16` and
  `<c32` are parsed, then rejected with a precise message; the zero-width `<U0`/`|S0`/`|V0` are valid NumPy dtypes and
  report "unsupported", never "invalid descriptor".
- `.npy` / `.npz` hold a `T` array as NumPy's pickle: NumPy writes the header `'descr': '|O'` and
  `pickle.dump(array, protocol=4)`, and `IO/NpyPickle.cs` writes the same bytes and reads them back, for a `T` array ONLY
  (NumSharp has no general unpickler).
  - Writer = CPython 3.12's C pickler for that object graph: a frame is committed at the first `save()` that finds it at
    64 KiB or more (EVERY save starts with that boundary check, atoms too), a frame under 4 bytes loses its header, a
    payload of 64 KiB or more goes outside the frames, `MEMOIZE` per memoized object, `BINGET` only where CPython hands
    out the SAME object (its cached strings `''` and one character below U+0100, and the na_object), `MARK … APPENDS`
    per 1000 items (a one-element list is item + `APPEND`). Elements in C logical order; `is_fortran` for an F-and-not-C
    array. It refuses an na_object with no pickle form it writes (`Half`, `decimal`).
  - Reader = the binary opcodes of protocols 2–5 over inert values (a global is two names, a `REDUCE` a record, nothing is
    ever called), then exactly NumPy's graph (`_reconstruct(ndarray, (0,), b'b')` + `(1, shape,
    _convert_to_stringdtype_kwargs(coerce[, na]), is_fortran, [elements])`); any other pickle is refused by name. The
    whole pickle sits in ONE buffer from `SizeBucketedBufferPool` (raw `NativeMemory` there fails the oracle gate
    `EveryRawNativeAllocation_IsAKnownChokepoint`), each text is an offset into it, and its UTF-8 bytes are packed
    straight into its cell (validated strictly; a string na_object compared by bytes), the arena reserved once: no string
    per element. The buffer grows only by what the stream delivers (a frame claiming 1 TB is a FormatException, never an
    allocation). An F array is filled in C order at F positions (the arena order NumPy's `_setlist_pkl` gives). A sparse
    `LONG_BINPUT` goes to a dictionary and counts towards `memo_len` as in CPython.
  - NumPy's texts: `Object arrays cannot be loaded when allow_pickle=False` (house FormatException), `Object arrays cannot
    be saved when allow_pickle=False` (ValueError, after the header), `Array can't be memory-mapped: Python objects in
    dtype.`, and `read_array`'s `UnicodeError: Unpickling a python object failed: UnicodeDecodeError('utf-8', b'\xff', 0,
    1, 'invalid start byte')` + its `encoding=` line for a text that is not UTF-8. NumPy ignores the header's shape and
    order for a pickle; so does NumSharp.
  - Gates: the `.npy` oracle's section 12 (`tpickle` byte-exact writes + reads with memory-order cells, `tpickle_raw`
    protocols 2–5 and an F source, `tnpz`), `IO/NpyOracleStringDTypeTests.cs`, `IO/NpyStringDTypePickleTests.cs`
    (several pickles in one stream, hostile lengths, the memo, refusals), and the reverse gate
    `test/oracle/verify_npy_interop.py` (NumPy reads `T` files and an `.npz` member NumSharp wrote).
- `mmap_mode` (`NpyFormat.OpenMemmap`, a port of NumPy's `open_memmap` + `numpy.memmap`): `np.load(path, mmap_mode=…)`
  returns an `NDArray` over a `MemoryMappedFile` view, zero-copy, the mapping released with the array and its views on
  dispose/GC. As NumPy behaves through `np.load`: `r`/`readonly` → a read-only view; `r+` → read-write (flushes to disk);
  `c` → copy-on-write (not flushed). The other four documented spellings validate but fail downstream as in NumPy:
  `w+` → `TypeError "object of type 'NoneType' has no len()"`, `write`/`copyonwrite`/`readwrite` → `ValueError "invalid
  mode: '<mode>b'"` (they contain the letter w, so `open_memmap` misroutes them into its create branch; NumSharp does not
  truncate the file as NumPy's `w+` does). Validation happens only on the `.npy` branch: an `.npz` ignores `mmap_mode`
  (even an invalid value). mmap needs a real path: a stream / `byte[]` npy raises `ValueError "…Memmap cannot use existing
  file handles"`. Big-endian and `<c8` files are refused (`NotSupportedException`): NumSharp byte-swaps / widens them on
  read and cannot present them as a zero-copy view; an object (`T` pickle) file is NumPy's `ValueError "Array can't be
  memory-mapped: Python objects in dtype."`. Tests: `IO/NpyMemmapTests.cs`.
- Intentional divergences: the five unsupported dtypes above are the only files NumPy loads that NumSharp refuses. NumPy
  reports `shape: (True,)` and overflowing dims through a later TypeError/OverflowError where NumSharp rejects them up
  front (both refuse; the text differs). Big-endian files byte-swap to native (NumPy keeps a byte-swapped dtype), and
  `Char` round-trips U+0000 as a NUL where NumPy's `<U` reports `''` (the bytes are identical).
- Traps:
  - `(1)` is not a tuple: only a comma makes one in Python, so `'shape': (1)` is the int 1 and NumPy rejects it
    (`PyLiteral.ParseTuple` tracks `sawComma`).
  - A header-length field is attacker-controlled: `ReadBytes` sizes its buffer from what the stream holds (seekable ⇒
    exact) and grows as bytes arrive, never from `count` (a 28-byte file can claim a 4 GB header). Pinned by
    `HostileHeaderLength_DoesNotAllocateTheClaim`.
  - `NpzFile`'s name map needs NumPy's TWO passes, not one interleaved loop: `dict(zip(stripped, full))`, then
    `.update(zip(full, full))`. For an archive holding both `a` and `a.npy` (both strip to `a`), pass 2 re-points `a` at
    the entry literally named `a`. Within a pass, later entries win (a zip may repeat a name and Python's dict keeps the
    last; .NET's `GetEntry` returns the first). Pinned by the `npz_*_names` cases.
  - Never buffer an npz member to sniff its magic: `MemoryStream`/`byte[]` cap at 2 GB. `NpyFormat.ReadArray` is strictly
    forward-only, so `StartsWithNpyMagic` sniffs on one `entry.Open()` and the reader streams from a fresh one. Pinned by
    `Npz_StreamsMembers_RatherThanBufferingThem`.
- Size limits: `.npy` and `.npz` both round-trip > 4 GiB (verified at 5 GiB across every 32-bit boundary, NumPy reading
  the result; the 5 GiB header is byte-identical to NumPy's). Array data lives in unmanaged memory and both IO paths stream
  in 256 KB chunks; Zip64 comes from .NET's `ZipArchive` on demand (NumPy forces `force_zip64=True`, gh-10776). The only
  2 GB ceilings are the `byte[]`-returning conveniences (`np.save(arr)`, `np.savez(...)` → `byte[]`,
  `NpzFile.GetRawBytes`): use the path/stream overloads above 2 GB. Tests: `NpyLargeFileTests` (`[HighMemory]`
  `[LongIndexing]`, excluded from CI; ~8 GB RAM + ~6 GB disk).
- Gates: `test/oracle/gen_npy_oracle.py` → `IO/corpus/npy_oracle.zip` (real NumPy output) replayed by `NpyOracleTests`
  under `NpyOracle` (`oracle.md`); the manual reverse-interop gate `python test/oracle/verify_npy_interop.py` (NumPy reads
  NumSharp's output, the direction byte equality cannot cover for `.npz`). `NumpyCompatibilityTests` /
  `NpyFormatVersionTests` read the hand-made fixtures in `test_compat/` relative to the working directory, which the
  csproj copies (keep that rule: without the fixtures the tests would assert nothing).

## Text I/O

`np.savetxt`, `np.loadtxt`, `np.fromstring`, `np.fromfile`, `ndarray.tofile`, Python's codecs and the `%` operator:
`text-io.md`.

## Printing

`NDArray.ToString()` is a byte-exact port of NumPy 2.4.2's array printing (`numpy/_core/arrayprint.py` + `dragon4.c`;
`Backends/Printing/{PrintOptions,Dragon4,ElementFormatters,ArrayFormatter}.cs`, `APIs/np.array2string.cs`,
`Casting/NDArray.ToString.cs`): `ToString()` / `ToString(false)` → `np.array_str` (`[0 1 2]`), `ToString(true)` →
`np.array_repr` (`array([0, 1, 2], dtype=…)`). It covers decimal-point float alignment, the maxprec/unique/fixed float
modes, the exp-format cutoffs (per dtype, native-precision ratio), nan/inf fields, complex, summarization at `threshold`
(with `…` and edgeitems), line wrapping at `linewidth`, the 0-d `str`/`repr` asymmetry (`5.0` / `5.`), and the repr
dtype/shape suffixes. The shortest digits come from Ryu in the value's own precision (`FloatDigits.Shortest`; .NET's
`"R"`, kept for reference as `ShortestR`, differs on the .NET 10 runtime for 2^-25 and 2^-958), but every rounding goes
through `ToString("F"|"E"+precision)` (which rounds the true binary value half to even), never through the shortest
string. NumSharp's `Char` dtype renders as strings (no NumPy equivalent). API: `array2string`, `array_repr`, `array_str`,
`format_float_positional`, `format_float_scientific`, `get_printoptions`, `printoptions` (an IDisposable context),
`set_printoptions`.

`np.format_float_positional` / `np.format_float_scientific` are NumPy's `dragon4.c`, statement for statement
(`Backends/Printing/Dragon4.Port.cs`, `Dragon4.BigInt.cs`). The array printer keeps the emulation of `Dragon4.cs` (exact
for the options it uses); the scalar formatters take NumPy's WHOLE option set, where only NumPy's code gives NumPy's text.
- Signatures are NumPy's: `format_float_positional(object x, int? precision = null, bool unique = true, bool fractional
  = true, string trim = "k", bool sign = false, int? pad_left = null, int? pad_right = null, int? min_digits = null)` and
  `format_float_scientific(object x, int? precision = null, bool unique = true, string trim = "k", bool sign = false,
  int? pad_left = null, int? exp_digits = null, int? min_digits = null)`.
- x: a `Half` / `float` / `double` is the np.float16 / np.float32 / np.float64 scalar, each printed with its OWN digits
  (`0.1f` → `'0.1'`); anything else is CPython's `PyFloat_AsDouble` to float64: a C# integer, `BigInteger` or `bool` is a
  Python int, correctly rounded (`int too large to convert to float`, OverflowError); a `char` is np.uint16; a `decimal`
  its float64; a 0-d NDArray is `float(a)` (its element, a 0-d text array parsed; complex: `float() argument must be a
  string or a real number, not 'complex'`); a 1-D array `only 0-dimensional arrays can be converted to Python scalars`;
  anything else `must be real number, not str` / `NoneType` / `list` / `tuple` / `bytes`.
- Check order is NumPy's: the `_none_or_positive_arg` ValueErrors (precision, pad_left, pad_right / exp_digits,
  min_digits), `precision must be greater than 0 if fractional=False`, `min_digits must be less than or equal to
  precision`, then C's: trim's converter (TypeError, `str(trim)` right-aligned in 100 characters), ``in non-unique mode
  `precision` must be supplied``, then x.
- Digits: a unique request whose cutoffs do not bind is Ryu's shortest digits (`FloatDigits.Shortest`, the fast path);
  every other request runs NumPy's Dragon4 loop over NumPy's BigInt arithmetic, ported IN PLACE with 64-bit blocks
  (allocation-free; the division normalized to bit 59 of the top block, NumPy's rule for 32-bit blocks). Two exact
  shortcuts NumPy's loop does not take: in unique mode the margins are dropped once they reach the scale, and where the
  margins no longer matter 19 digits come out of ONE Knuth single-word division. The same loop on
  `System.Numerics.BigInteger` stays as the test reference (`Dragon4.UseReferenceDigits`). Inf / nan are NumPy's
  PrintInfNan (`inf` / `-inf` / `+inf`, `nan` never signed, never padded).
- NumPy's buffer: 16384 characters (`RuntimeError: Float formatting result too large`), and NumPy's FormatScientific
  overrun reproduced (its zero padding does not count the buffer down: `format_float_scientific(1.5, precision=20000,
  unique=False)` is 16387 characters).
- Gates: the surface tier's formatter cases (its section dragon4: random bit patterns of every width under random option
  sets, x as a 0-d array of every dtype, Python ints past float64, exact digits that END inside the precision (only
  scientific trim `'-'` shows a kept zero: `'1e+22'`, not `'1.e+22'`) and powers of two under min_digits, the only
  values whose last digit the high margin alone decides after a forced continuation) and `Casting/Dragon4PortTests.cs`
  (the three digit generators held to each other over every float16 pattern and sampled float32 / float64 patterns,
  Knuth's division against BigInteger's).
