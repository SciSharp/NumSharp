---
paths:
  - "src/NumSharp.Core/IO/**"
  - "src/NumSharp.Core/APIs/np.{save,load,savetxt,loadtxt,fromstring,fromfile,array2string}*.cs"
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
| Char | `<U1` | 2-byte UTF-16 ↔ 4-byte UCS-4; non-BMP rejected (a surrogate pair) |
| Decimal | — | `NotSupportedException`: no NumPy dtype (cast to Double) |

- Object arrays, structured/subarray dtypes, `datetime64`/`timedelta64`, `|S`/`<U`n>1/`|V`, `<f16` and `<c32` are parsed,
  then rejected with a precise message; the zero-width `<U0`/`|S0`/`|V0` are valid NumPy dtypes and report "unsupported",
  never "invalid descriptor".
- `mmap_mode` (`NpyFormat.OpenMemmap`, a port of NumPy's `open_memmap` + `numpy.memmap`): `np.load(path, mmap_mode=…)`
  returns an `NDArray` over a `MemoryMappedFile` view, zero-copy, the mapping released with the array and its views on
  dispose/GC. As NumPy behaves through `np.load`: `r`/`readonly` → a read-only view; `r+` → read-write (flushes to disk);
  `c` → copy-on-write (not flushed). The other four documented spellings validate but fail downstream as in NumPy:
  `w+` → `TypeError "object of type 'NoneType' has no len()"`, `write`/`copyonwrite`/`readwrite` → `ValueError "invalid
  mode: '<mode>b'"` (they contain the letter w, so `open_memmap` misroutes them into its create branch; NumSharp does not
  truncate the file as NumPy's `w+` does). Validation happens only on the `.npy` branch: an `.npz` ignores `mmap_mode`
  (even an invalid value). mmap needs a real path: a stream / `byte[]` npy raises `ValueError "…Memmap cannot use existing
  file handles"`. Big-endian, `<U1` and `<c8` files are refused (`NotSupportedException`): NumSharp byte-swaps / widens
  them on read and cannot present them as a zero-copy view. Tests: `IO/NpyMemmapTests.cs`.
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

- `np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)` is
  a port of NumPy 2.4.2's `_npyio_impl.py::savetxt` (`APIs/np.savetxt.cs`), byte-for-byte NumPy's output.
  - A 1-D array writes one value per line (NumPy's `atleast_2d(X).T`, `ncol=1`); a 2-D array a row per line; 0-D / ≥3-D
    raise `ValueError("Expected 1D or 2D array, got {n}D array instead")`.
  - Each row is `format % tuple(row)`, where `format` is the single spec joined `ncol` times by `delimiter`, a
    caller-supplied multi-`%` string (the delimiter then ignored), or the joined list/tuple of specs, with NumPy's errors
    (`ValueError "fmt has wrong number of % formats:  …"`, `AttributeError "fmt has wrong shape.  …"`).
  - The `%` engine is `PrintfFormatter` (shared with `ndarray.tofile`; Python's `%` operator: flags `-+ 0#`, width,
    `.precision`, `d i u x X o e E f F g G s c %%`, a 2-digit minimum exponent, `nan`/`inf`, `-0.0`); the multi-column
    path is `PrintfFormatter.FormatRow`/`FormatRowInto`. Numeric specs widen to double (NumPy's `float()`/`int()`
    coercion, so int64 past 2^53 and float32 render at the double value); `%s` uses each dtype's shortest scalar repr. On
    an INTEGER conversion (`d i u x X o`) a `.precision` is Python's minimum digit count, left-zero-filled (`%.5d` →
    `00042`, `%.0d % 0` → `0`), the `0` flag composes with it (`%08.3d % 7` → `00000007`), and with `#` the zero fill lands
    between the `0x`/`0o` prefix and the digits (`%#08x` → `0x0000ff`). `%c` renders a true integer's code point in
    `[0, 0x10FFFF]` and raises `OverflowError("%c arg not in range(0x110000)")` outside it; `%c`, `%x`, `%X`, `%o` on a bool
    are the TypeError (dtype/format mismatch), while `%d`/`%i`/`%u` accept a bool. Complex wraps a single spec as
    `' (%s+%sj)'` per column, with the `+-` → `-` fix-up.
  - Targets: a filename (`.gz` → gzip; Python's text-mode `\n` → `os.linesep`, so a file matches NumPy on the same
    platform: CRLF on Windows), an open `Stream`, a `TextWriter` (both verbatim `\n`). Any layout is read in logical C
    order through `GetAtIndex`; all 15 dtypes.
  - `[Misaligned]`: `%d`/`%i`/`%u` on inf raises `System.OverflowException` (the message is NumPy's `OverflowError`; nan
    raises ValueError verbatim), and the AttributeError renders a C# `string[]` as a Python list repr.
  - Gate: `IO/SaveTxtTests.cs`. Source `Backends/Printing/PrintfFormatter.cs`.
- `np.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False,
  ndmin=0, encoding=None, max_rows=None, quotechar=None)` is a port of NumPy 2.4.2's `_npyio_impl.py::loadtxt` → `_read`
  plus its C text reader (`textreading/{tokenize.cpp,conversions.c,str_to_int.c,rows.c}`), in `APIs/np.loadtxt.cs`; it
  round-trips `np.savetxt`.
  - The C reader yields a 2-D `(rows, cols)` array; `ndmin` then squeezes size-1 axes (a single row/col → 1-D, a lone value
    → 0-d) or expands.
  - The tokenizer: `delimiter=None` splits on runs of whitespace (leading/trailing stripped; blank or comment-only lines
    give 0 fields and are skipped), an explicit single-character delimiter (`' '` included) keeps empty fields;
    `quotechar` protects delimiters and comments and doubles as an escape; a single-character comment strips to the line
    end (a multi-character comment is stripped per line, and rejected with `quotechar`). Whitespace is NumPy's
    `Py_UNICODE_ISSPACE`: `char.IsWhiteSpace` plus the four C0 information separators U+001C..U+001F
    (`IsLoadtxtWhitespace`/`TrimLoadtxtWs`).
  - The parsers: bool via int64 (`"0"`/`"1"`, `-1` → True, `"True"` → error); integers range-check (`int8 "200"` → error)
    and unsigned rejects negatives (`uint8 "-1"` → error); float uses `PyOS_string_to_double` semantics (case-insensitive
    `inf`/`infinity`/`nan`; hex, `_`, trailing junk and empty rejected; NaN → `+qNaN 0x7FF8…`); complex is `to_complex_int`
    with the imaginary unit lowercase `j` only (`a`/`aj`/`a+bj`/`(a+bj)`/`1+-2j`; `"1+2J"` → error). All 15 dtypes.
  - `converters` (a `Func<string,object>` for all columns, or an `IDictionary<int,Func<string,object>>` per column),
    `usecols` (int[], negatives from the end), `unpack` (transpose), `skiprows`/`max_rows` (blank and comment lines do not
    count toward `max_rows`), `encoding`. Inputs: a filename (`.gz` decompressed; universal newlines), a `Stream`, a
    `TextReader`, an `IEnumerable<string>` of lines. Fields are parsed from the line's char span (no per-field string).
  - Errors (NumPy's texts): `ValueError("the number of columns changed from {a} to {b} at row {n}; …")`, `ValueError("could
    not convert string '{tok}' to {dtype} at row {r}, column {c}.")` (the row 0-based, the column 1-based),
    `ValueError("invalid column index {k} at row {n} with {c} columns")`, `ValueError("Illegal value of ndmin keyword: …")`,
    `TypeError` for a multi-character or newline delimiter.
  - Divergence: the empty-input `UserWarning` is not emitted (NumSharp has no warnings module). Float-heavy input is bound
    by the correctly rounded `double.Parse` (NumPy's `strtod`). Gate: `IO/LoadTxtTests.cs`.
- `np.fromstring(string, dtype=float, count=-1, sep)` (`APIs/np.fromstring.cs`) parses a 1-D array from the numbers in a
  string with `np.fromfile`'s text item reader (`SplitTokens`/`SelectTokens`/`TokensToArray`, as NumPy shares `fromstr`):
  a whitespace-only `sep` splits on any whitespace run, a non-whitespace `sep`'s spaces are a wildcard, one trailing
  separator is tolerated, and a bad or missing item raises `ValueError("string or file could not be read to its end due
  to unmatched data")`. `count` caps the items. NumPy removed the binary mode, so an empty/null `sep` raises
  `ValueError("The binary mode of fromstring is removed, use frombuffer instead")`. The shared parser is a lenient
  superset of NumPy's: the parenthesised `"(1+2j)"` is accepted (it round-trips `tofile`; NumPy rejects it), and integer
  overflow wraps deterministically where NumPy's C parser is platform-specific (NumPy raises on a narrow-unsigned
  negative such as `uint8 "-1"` and clamps overflow through `strtol`); `loadtxt` carries strict range-checking parsers
  instead. The imaginary unit is lowercase `j` only. Gate: `IO/FromStringTests.cs`.
- Not implemented: `genfromtxt`/`fromregex` (both require structured/record dtypes, and `genfromtxt` masked arrays and
  missing values; their non-structured subset would only re-expose `loadtxt`).

## Printing

`NDArray.ToString()` is a byte-exact port of NumPy 2.4.2's array printing (`numpy/_core/arrayprint.py` + `dragon4.c`;
`Backends/Printing/{PrintOptions,Dragon4,ElementFormatters,ArrayFormatter}.cs`, `APIs/np.array2string.cs`,
`Casting/NDArray.ToString.cs`): `ToString()` / `ToString(false)` → `np.array_str` (`[0 1 2]`), `ToString(true)` →
`np.array_repr` (`array([0, 1, 2], dtype=…)`). It covers decimal-point float alignment, the maxprec/unique/fixed float
modes, the exp-format cutoffs (per dtype, native-precision ratio), nan/inf fields, complex, summarization at `threshold`
(with `…` and edgeitems), line wrapping at `linewidth`, the 0-d `str`/`repr` asymmetry (`5.0` / `5.`), and the repr
dtype/shape suffixes. Digit generation uses .NET's shortest round-trip `ToString("R")` (= Dragon4 unique), but every
rounding goes through `ToString("F"|"E"+precision)` (which rounds the true binary value half to even), never through the
shortest string. NumSharp's `Char` dtype renders as strings (no NumPy equivalent). API: `array2string`, `array_repr`,
`array_str`, `format_float_positional`, `format_float_scientific`, `get_printoptions`, `printoptions` (an IDisposable
context), `set_printoptions`.
