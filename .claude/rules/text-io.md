---
paths:
  - "src/NumSharp.Core/APIs/np.{savetxt,loadtxt,fromstring,fromfile,tofile}*.cs"
  - "src/NumSharp.Core/IO/{SaveTxtIO,ArrayFileIO}.cs"
  - "src/NumSharp.Core/IO/TextReading/**"
  - "src/NumSharp.Core/Backends/Printing/{PercentFormat,PrintfFormatter}*.cs"
  - "src/NumSharp.Core/Utilities/NDDoubleParse.cs"
  - "test/NumSharp.Tests/IO/{SaveTxt,LoadTxt,FromString,TextReaderPort,StringDTypeTextIO,TextIOCodec,PercentFormat}Tests.cs"
  - "test/NumSharp.Tests.Oracle/TextIO/**"
---

# Text I/O: `savetxt`, `loadtxt`, `fromstring`, `fromfile`, `tofile`

Every function is a port of NumPy 2.4.2, its errors and their order included, and every text encoding goes through a
port of CPython 3.12's codec layer. Gates: the committed text I/O oracle tier (`TextIO/corpus/textio_*.jsonl`, category
`TextIOOracle`, Linux NumPy, zero excuses; `oracle.md`), the string oracle's `tio` section (`StringDType`, the codecs,
the `%` format matrix through savetxt and tofile), and the unit suites named below.

## `np.savetxt`

- `np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)` is
  NumPy's `_npyio_impl.py::savetxt`, byte-for-byte NumPy's output: the overloads in `APIs/np.savetxt.cs`, the body in
  `IO/SaveTxtIO.cs`.
  - A 1-D array writes one value per line (NumPy's `atleast_2d(X).T`, `ncol=1`); a 2-D array a row per line; 0-D / ≥3-D
    raise `ValueError("Expected 1D or 2D array, got {n}D array instead")`.
  - Each row is `format % tuple(row)`, where `format` is the single spec joined `ncol` times by `delimiter`, a
    caller-supplied multi-`%` string (the delimiter then ignored), or the joined list/tuple of specs, with NumPy's errors
    (`ValueError "fmt has wrong number of % formats:  …"`, `AttributeError "fmt has wrong shape.  …"`). The row format is
    compiled once (`PercentFormat`, the `%` operator below) and run per row into the batch buffer; the elements are NUMPY
    SCALARS (`%r` is `np.float32(0.1)`, a `TypeError` names `numpy.bool`), a `StringDType` row's Python objects. Complex
    wraps a single spec as `' (%s+%sj)'` per column, with the `+-` → `-` fix-up.
  - Targets: a filename (created first; gzip for exactly `.gz`; Python's text-mode `\n` → `os.linesep`, so a file matches
    NumPy on the same platform: CRLF on Windows), an open `Stream` or a `TextWriter` (both verbatim `\n`); the encodings
    are Python's codecs (below). Any layout is read in logical C order through `GetAtIndex`; all 15 dtypes (Char is a
    `uint16` scalar whose `%c` is its character, Decimal a `float64` of its value, as `ArrayFormatter.ScalarStr` prints
    them).
  - A null string argument is Python's None, refused where NumPy's statements meet it: `fmt` at once (`ValueError("invalid
    fmt: None")`), `delimiter` only where the specs are joined with it (`AttributeError("'NoneType' object has no
    attribute 'join'")`; a multi-`%` format never reads it), `header` / `footer` at `len()` (`TypeError("object of type
    'NoneType' has no len()")`; the footer's after the rows are written), `comments` / `newline` in their concatenations
    (`TypeError("can only concatenate str (not \"NoneType\") to str")`; a real row's becomes NumPy's `Mismatch between
    array dtype ...`, a complex row's stays the plain TypeError, an empty array writes nothing and does not fail), and a
    null entry of a spec list is `str.join`'s `TypeError("sequence item {i}: expected str instance, NoneType found")`.
  - `[Misaligned]`: `%d`/`%i`/`%u` on inf raises `System.OverflowException` (the message is NumPy's `OverflowError`; nan
    raises ValueError verbatim), and the AttributeError renders a C# `string[]` as a Python list repr.
  - Gate: `IO/SaveTxtTests.cs`.

## `np.loadtxt`

- `np.loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False,
  ndmin=0, encoding=None, max_rows=None, quotechar=None)` is NumPy's `_npyio_impl.py::loadtxt` → `_read` plus its C text
  reader (`textreading/{tokenize.cpp,conversions.c,str_to_int.c,rows.c}`): the overloads in `APIs/np.loadtxt.cs`, the body
  in `IO/TextReading/LoadTxtReader.cs` (`_read`'s checks, the configuration, `read_rows`), `LoadTxtReader.Conversions.cs`
  (the field parsers of `conversions.c` / `str_to_int.c`) and `LoadTxtReader.Sinks.cs` (one sink per dtype class:
  numeric, fixed `U` / `S`, `StringDType`), one `internal static partial class LoadTxtReader`; the tokenizer is
  `LoadTxtTokenizer`, a port of `tokenize.cpp`. It round-trips `np.savetxt`.
  - The checks run in NumPy's order: the Python-level ones first (`comments` are Python objects: a non-iterable is `'int'
    object is not iterable` before the ndmin check; a null element is `str.split(None, 1)[0]`, an IndexError on a blank
    line; another object raises `must be str or None, not int` at the first line; `delimiter in comments` is Python's
    `in`), then the source, then the C reader's control-character checks.
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
    with the imaginary unit lowercase `j` only (`"1+2J"` → error). A float16 field is NumPy's `npy_to_generic`:
    `PyArray_Pack` of the field's str, i.e. Python's `float()` (underscores between digits, Unicode digits and Unicode
    whitespace accepted, which float32 / float64's own parser refuses) rounded once by `npy_double_to_half`; an ASCII
    field without `_` takes the shared double parser, any other goes through the class's `setitem` slot, and so does a
    converter's value. A `StringDType` field is stored whole; a string NA object marks an equal field missing; a
    converter's value is stored by `stringdtype_setitem`'s rules (the NA check first, then `coerce`). All 15 dtypes.
  - `converters` (a `Func<string,object>` for all columns, or an `IDictionary<int,Func<string,object>>` per column),
    `usecols` (int[], negatives from the end), `unpack` (transpose), `skiprows`/`max_rows` (blank and comment lines do not
    count toward `max_rows`), `encoding`. An unsized `U` / `S` with `max_rows=0` is `_read`'s empty chunk list:
    `ValueError("need at least one array to concatenate")`.
  - Sources: a file name follows NumPy's `DataSource` (`PyDataSource`): the name, else the name with `.bz2`, `.gz`, `.xz`
    or `.lzma` appended; the extension is Python's `os.path.splitext`, case-sensitive (`.GZ` is plain text); `.bz2`, `.xz`
    and `.lzma` are refused (`NotSupportedException`: .NET has no decompressor). The file is read through
    `LoadTxtTextFile`, a port of the read side of TextIOWrapper: `read(16384)` takes the decoded characters left over, then
    chunks of `max(8192, b2cratio * remaining)` bytes, one raw read each, `b2cratio` measured AFTER the newline translation
    `IncrementalNewlineDecoder` does inside the reader; line iteration (multiple or multi-character comments) reads
    8192-byte chunks. Those chunks are what a CPython decode error's position counts within (an invalid byte at 10,000
    reports position 10,000, at 20,000 position 3,616). A byte `Stream` is split at `0x0A` and each line decoded on its
    own (`bytes.decode`); a `TextReader` is split at `\n`; an array or a `List<string>` of lines is read by index, as
    Python's list iterator walks a list (an element a converter changes is read changed, lines a converter appends are
    read too).
  - Errors (NumPy's texts): `ValueError("the number of columns changed from {a} to {b} at row {n}; …")`, `ValueError("could
    not convert string '{tok}' to {dtype} at row {r}, column {c}.")` (the row 0-based, the column 1-based; `{dtype}` is
    `str` of the requested descriptor, `rows.c`'s `%S`, so a non-native one reads `>i4`), `ValueError("invalid column
    index {k} at row {n} with {c} columns")`, `ValueError("Illegal value of ndmin keyword: …")`, `TypeError` for a
    multi-character or newline delimiter.
  - The hot path: doubles go through `Utilities/NDDoubleParse.cs` (Clinger's fast path, then fast_float's Eisel-Lemire
    `compute_float` with the 128-bit powers of five built from exact integers, and .NET's parser only for a mantissa past
    19 digits whose truncation does not decide the rounding; correctly rounded, the bits `_Py_dg_strtod` gives; gate
    `Utilities/NDDoubleParseTests.cs`), shared with `np.fromstring` / `np.fromfile`'s scanners. The tokenizer runs
    `tokenizer_core`'s step inline in the row loop (state, buffer span and position in locals), finds a run's stop with a
    128-entry ASCII class table for the first 16 characters and the vectorized `SearchValues` search past them, and reads
    a whole quote-free line in one pass (`TokenizeUnquotedLine`), its fields left as slices of the line instead of copies.
    The sinks take a row at a time whenever there is no converter and no `usecols`; a fixed-size dtype whose source bounds
    its lines (`string[]` / `List<string>`: `LoadTxtStream.RemainingLinesHint`) allocates the result at the first row and
    writes it in place; an unsized `U` / `S` stages the fields' text and sizes the result from the widest.
  - JIT traps found here: a `stackalloc` in a method stops the JIT from inlining it even under `AggressiveInlining`; a
    P/Invoke inlined into a caller (the pool's allocation, through an inlined growth path) makes that caller set up an
    inlined call frame in its prolog on EVERY call (the growth path is `NoInlining`); a long single-call loop runs as
    on-stack-replaced code with its locals in the frame (the row loop moved into the sinks).
  - Divergences: the empty-input `UserWarning` is not emitted (NumSharp has no warnings module); a non-BMP control
    character is refused; `max_rows` buffers grow instead of being preallocated; an unsized `U` / `S` result's error rows
    count from the start (NumPy counts within 50,000-row chunks); under `encoding='bytes'` a C# converter receives a str;
    a `.gz` file's decode-error positions may differ (CPython's gzip reader hands TextIOWrapper whatever one decompression
    step produced); URLs and `DataSource`'s `destpath` fall-back are not ported. NumPy bug not reproduced: NumPy's loadtxt
    keeps the caller's `StringDType` instance, so a second call with the same instance corrupts the first result.
  - Gates: `IO/LoadTxtTests.cs`, `IO/TextReaderPortTests.cs`, `IO/StringDTypeTextIOTests.cs`, the text I/O oracle's
    loadtxt / ltfields / lttokens sections.

## Python's codecs

`IO/TextReading/TextIOCodec{,.Incremental,.Unicode}.cs` + `PythonCodecs.cs`, the tables generated from CPython 3.12 by
`tools/gen_python_codecs.py` into `PythonCodecs.Tables.cs`.
- Names: `_Py_normalize_encoding`, CPython's 326-entry alias table, then the module name. A `.` is never part of a module
  name, so `utf.8` is unknown while `iso.8859.1` is an alias. The C fast paths of `PyUnicode_Decode` /
  `PyUnicode_AsEncodedString` are honoured for one-shot calls (an `iso8859_1` spelling encodes with the `'latin-1'` text
  where the registry would pick the charmap module).
- Exact codecs: UTF-8 and `utf-8-sig`; UTF-16 / UTF-32 in both byte orders, with and without a BOM (ports of
  `PyUnicode_DecodeUTF16Stateful` / `DecodeUTF32Stateful`: their four errors, their spans and the `consumed` rule);
  Latin-1, ASCII and `undefined`; the 70 single-byte charmaps. A charmap's encode map is the inverse of its decode table
  with the LAST byte winning, which the generator verifies over the whole BMP for every codec.
- Refused by name (`NotSupportedException`): the multi-byte CJK codecs, `utf-7`, the escape codecs, IDNA and `mbcs` /
  `oem`. A bytes-to-bytes codec is CPython's `LookupError ... is not a text encoding; use codecs.decode() /
  codecs.encode() / codecs.open() ...`, the advice depending on the context.
- Defaults: a file name UTF-8 (pinned; Windows NumPy uses the ANSI code page), a stream Latin-1 for loadtxt and for
  savetxt (whose `encoding or 'latin1'` also maps `""` to Latin-1).
- BOMs: a byte stream's line decode reads a BOM per line; a file's BOM codec requires one (`UnicodeError: UTF-16 stream
  does not start with BOM`); `utf-8-sig` drops a partial BOM at the end silently, as CPython does.
- savetxt's writer (`SaveTxtWriter`): a byte stream is NumPy's `WriteWrap` (each write checked and encoded on its own, so a
  BOM codec writes a BOM before every write); a file is TextIOWrapper's encoder (`TextIOFileEncoder`: its `encodefuncs`
  chosen by codec name, so `iso8859-1` names encode as Latin-1; the BOM written once, even for an empty first write; the
  error names `'utf-16'` in the first write and `'utf-16-le'` after it; positions count the newline-TRANSLATED write,
  checked untranslated and re-checked translated only when failing, which is sound because every codec encodes `\r`).
- Trap: CPython's incremental UTF-8 decoder carries a truncated SURROGATE (`0xED 0xA0..0xBF` exactly at a chunk's end) to
  the next chunk, as it carries a valid start (`unicode_decode_utf8`'s case 2 when `consumed` is set), so the error
  reports position 0 of the NEXT decode. .NET's `Utf8.ToUtf16` calls those two bytes invalid at once, so
  `TextIOCodec.Incremental`'s tail logic defers them itself. A one-shot `bytes.decode` raises at once.
- Gate: `IO/TextIOCodecTests.cs` (exact exception types, the newline-translated position, the gzip round trip, the
  refusals).

## The `%` operator

`Backends/Printing/PercentFormat.cs` + `PrintfFormatter{,.PyObject}.cs`, a port of CPython 3.12's `PyUnicode_Format`,
shared by savetxt (`format % tuple(row)`) and tofile (`format % (item,)`, the Python object `getitem` returns for EVERY
dtype: a `float` for float16 / float32, an `int`, a `bool`, a `complex`, a `str` / `bytes`).
- The parse is `unicode_format_arg_parse`, compiled once per call. CPython parses as it formats, so each parse error is
  recorded where the parser meets it and raised when a run reaches it, after the arguments CPython takes first. ONE `h` /
  `l` / `L` is skipped (`%lld` is the unsupported `'l'`); the argument is taken BEFORE the conversion character is checked
  (`'%y' % ()` is "not enough arguments"); `%(key)` is `format requires a mapping` (a tuple); `unsupported format
  character 'y' (0x79) at index 1` shows the character only when printable ASCII and counts the index in CODE POINTS;
  `width too big` / `precision too big` digit by digit; `incomplete format`; a leftover argument is `not all arguments
  converted during string formatting`.
- A `*` width or precision takes a Python `int` only (a `bool` is one; `np.int64` is NOT an `int` subclass: `* wants
  int`); a negative width left-justifies (the most negative one pads nothing: CPython's negation overflows), a negative
  precision is 0, past `ssize_t` / `int` is CPython's OverflowError. CPython's limits: `%.2147483647e` is `%.0e` (its
  `precision + 1` wraps), `%g` without `#` strips zeros past any precision, an integer precision past `INT_MAX - 3` is
  `precision too large`, a decimal conversion of an int past 4300 digits is CPython's ValueError, a width no string can
  meet is `OutOfMemoryException` (`MemoryError`).
- Python's semantics throughout: an integer conversion's `.precision` is a minimum digit count (`%.5d` → `00042`,
  `%#08x` → `0x0000ff`); `%c` of a code point past 0x10FFFF is `OverflowError("%c arg not in range(0x110000)")`; an
  `np.bool` is not an integer (`%x` / `%c` refuse it) while a Python `bool` is (`'%x' % True` is `1`); a NumPy scalar's
  `repr` names its type, a `TypeError` quotes `numpy.float64` / `float`. A `StringDType` element is the Python object
  `getitem` returns (a `str`, or the NA object), formatted by `PrintfFormatter.PyObject` with Python's `%` rules for
  that object's type.
- tofile checks `sep` and `format` with CPython's `"s"` converter BEFORE opening the file (a NUL: `embedded null
  character`; a lone surrogate: the UTF-8 error), writes each element's text as ASCII (`UnicodeEncodeError` where NumPy
  crashes: NumPy does not check `PyUnicode_AsASCIIString`'s NULL) and the separator as its UTF-8 bytes.
- Gates: `IO/PercentFormatTests.cs` (the parse driven directly, the raw and the wrapped TypeError, Char / Decimal, NumPy's
  crash case, no file on an argument error, both writers over `FencedMemory`) and the string oracle's `%` format matrix.

## `np.fromstring` / `np.fromfile` / `ndarray.tofile`

NumPy's C readers and writers, statement for statement. Signatures `fromstring(string | byte[] string, dtype=float,
count=-1, sep, like)`, `fromfile(string file | Stream, dtype=float, count=-1, sep="", offset=0, like)`, `tofile(string |
Stream, sep="", format="%s")`. The facades (`APIs/np.{fromstring,fromfile,tofile}.cs`) are NumPy's `array_fromstring` /
`array_fromfile` / `array_tofile` (the argument conversion and the file handling); the bodies, `PyArray_FromString` /
`PyArray_FromFile` / `PyArray_ToFile`, are `IO/ArrayFileIO.cs` over the scanners of
`IO/TextReading/NumPyTextScan{,.File,.Reader}.cs`, and their dtype questions are the class slots (REFCHK
`HasReferences`, `HasScanFunc` / `HasFromStr`, the `NPY_LIST_PICKLE` flag, `copyswapn`).
- `array_from_text` (both text readers): elements and separators alternate (`fromstr_next_element` /
  `fromfile_next_element`, `fromstr_skip_separator` / `fromfile_skip_separator`, over NumPy's `swab_separator`: a
  whitespace run in `sep` is ONE wildcard space, a space is prepended when `sep` does not start with whitespace, a second
  one appended when it ends with one, so `","` is `" ,"`). The end of the text is tested BEFORE a separator's match (a
  trailing separator ends the read), the separator after the last requested element is optional, and a failed read is
  `ValueError("string or file could not be read to its end due to unmatched data")`. `fromstring` with `count ≥ 0` never
  shrinks: the result is `(count,)`, the slot of a failed scan holds what the scanner wrote (0, -1.0, -1+0j: `""` with
  `count=2` is `[-1., ?]`), the rest is NumPy's uninitialized memory (NumSharp zeroes it). The scanners report the
  position AFTER leading whitespace even when no number follows, so whitespace alone is an element: `"1, "` is `[1.,
  -1.]` (float) and `[1, 0]` (int). `fromfile` resizes to the elements read.
- `fromstring`'s scanners: CPython's `PyOS_strtol` / `PyOS_strtoul` with a 64-bit `long` (whitespace skipped before AND
  after the sign, so `"- 1"` is -1; strtoul takes no sign; overflow saturates, strtol to `LONG_MAX` for EITHER sign
  except exactly -2^63); NumPy's `NumPyOS_ascii_strtod` (every NaN form positive, `nan(...)`, `"-nan"`; `inf` /
  `infinity` in any case; CPython's decimal grammar, no hex, no `_`; no number is -1.0); `CDOUBLE_fromstr`'s quirks
  (`"1+2"` is 1+0j with the `"+2"` consumed, a bare `"j"` is -1j). A narrower integer dtype takes the C long's low bits
  (`int8 "300"` is 44), float32 / float16 the double rounded once.
- `fromfile`'s scanners: glibc's `fscanf` through a C `FILE` model (`CFileReader`: 64 KB chunks, `ungetc` (a 0xFF pushed
  back through `char` is lost), the Python file sought to `ftell` after the read, also after an error, the offset seek
  first): `%ld` / `%lu` take whitespace, a sign, then at least one digit (a sign without a digit is consumed and fails);
  strtoul negates modulo 2^64 (`uint64 "-1"` is `ULONG_MAX`), strtol clamps BY SIGN (unlike CPython's);
  `NumPyOS_ascii_ftolf`'s number-like scan (a dangling `e` or `.` is consumed); `CDOUBLE_scan`'s `ungetc` of a missing
  `j`; a separator byte at or past 0x80 never matches (`fgetc`'s int against a signed `char`).
- Platform pin: LP64 (Linux / macOS NumPy: a C `long` is 64 bits, glibc's `fscanf`). Windows NumPy (32-bit `long`, MSVC's
  `fscanf`) differs in integer parsing, the pin of the random module too (`NumPyRandom.LegacyLong`); MSVC's `fscanf` also
  consumes the character that ends a failed integer scan.
- Argument order: `string`'s `s#` (UTF-8; CPython's UnicodeEncodeError for a lone surrogate), `sep`'s `s`
  (UnicodeEncodeError, then `embedded null character`; a null `sep` is `fromfile() argument 4 must be str, not None`),
  fromstring's removed binary mode (an empty or null `sep`: `The binary mode of fromstring is removed, use frombuffer
  instead`), REFCHK (`Cannot create an object array from a string` / `Cannot read into object array` for
  `StringDType`), item size 0 (`zero-valued itemsize`), no `fromstr` / `scan` (`don't know how to read character strings
  with that array type` / `Unable to read character files of that array type`). `fromfile`'s `offset` with a separator is
  `TypeError` before the file is opened; a missing file is `FileNotFoundException("[Errno 2] No such file or directory:
  'x'")`. `tofile`'s `sep` and `format` are `s` arguments too: a null one is Python's None, `TypeError("tofile() argument
  2 must be str, not None")` (3 for the format, `sep` first) before the file is opened or the stream used.
- Binary: `fread` straight into the result's buffer (count -1: every whole item, `(size - pos) / itemsize` in C's
  truncating division, so a position past the end is `negative dimensions are not allowed` and a partial trailing item
  stays unread; a count past the end consumes the partial item); spans of 1 GB, so files past 2 GB read; a non-native
  byte order is swapped to native in place by the class's `copyswapn`; a non-seekable stream is read to its end (NumPy
  cannot read one). `tofile` writes a C-contiguous array, views at an offset included, straight from its buffer, any other
  layout through one C-order copy; text `tofile` writes through a pooled 32 KB byte buffer.
- NumPy bugs not reproduced: the uninitialized tail of a counted `fromstring`; a non-native dtype's text read (NumPy
  byte-swaps the value its scanner wrote in native order): NumSharp returns the native dtype with the right values.
  Divergences: the result of a non-native request has the native dtype (no non-native storage); an unsized `S` / `U` read
  with `count ≥ 0` is `NotSupportedException` (NumPy builds zero-width elements); `complex64` has no dtype.
- The speed floor is the file system's: `CREATE_ALWAYS` (Python's `open('wb')`) plus one `WriteFile` costs as much as the
  whole contiguous binary `tofile` on both sides. (Opening with `OPEN_ALWAYS` and truncating after the write would be 2×
  faster for a file that exists, but it is not `'wb'`: a hidden file would open instead of failing, and an interrupted
  write would leave the old tail.)
- Gates: the text I/O oracle tier (five functions), `IO/TextReaderPortTests.cs`, `IO/FromStringTests.cs`,
  `Casting/NDArray.fromfile.Test.cs`.
- The other byte-level array members: `Backends/Unmanaged/UnmanagedStorage.Convert.cs` (`ToBytes`, ndarray.tobytes'
  `PyArray_ToString`), `UnmanagedStorage.Byteswap.cs` (ndarray.byteswap's `PyArray_Byteswap`), `Utilities/NDByteOrder.cs`
  (the byte-order kernels behind `DTypeMeta.copyswapn`); np.frombuffer: `creation-manipulation.md`.

## Not implemented

`genfromtxt` / `fromregex`: both require structured/record dtypes, and `genfromtxt` masked arrays and missing values;
their non-structured subset would only re-expose `loadtxt`.
