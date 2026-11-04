# Python's codecs NumSharp refuses — what is missing and what the port costs (discovery, 2026-11-04)

> **Scope.** This is a discovery: nothing is implemented. It answers the owner's `/np-function` request "Codecs: 33 Python
> codecs for encode / decode are missing: CJK multi-byte, utf-7, idna, the escape codecs and mbcs. Task boundary: discover
> what's missing and what will it cost to port". Measured on branch `strings` at `c4f9581e` with the oracle's
> interpreter (CPython 3.12.12, NumPy 2.4.2) on Windows (ANSI code page 1252, OEM 437). The four scripts in
> [strings-codecs/](strings-codecs/) reproduce every number (§8).

## 0. The question and the answer

**What is missing.** There are **32** codec modules, not 33. They are the `encodings` modules that
`tools/gen_python_codecs.py` classifies `Unsupported`: every text codec that is not a Unicode codec, Latin-1, ASCII,
`undefined` or one of the 70 single-byte charmaps NumSharp already implements. 111 spellings reach them: the 32 module
names and 79 of the alias table's 326 entries. CPython on Linux / macOS has 30 of them, because `mbcs` and `oem` exist
on Windows only. They fall into seven families:

| Family | Modules | Aliases | What CPython runs |
|---|---|---|---|
| CJK multi-byte | 24: `gb2312` `gbk` `gb18030` `hz` · `big5` `cp950` `big5hkscs` · `shift_jis` `cp932` `euc_jp` `shift_jis_2004` `shift_jisx0213` `euc_jis_2004` `euc_jisx0213` · `euc_kr` `cp949` `johab` · `iso2022_jp` `iso2022_jp_1` `iso2022_jp_2` `iso2022_jp_2004` `iso2022_jp_3` `iso2022_jp_ext` `iso2022_kr` | 74 | C: `_multibytecodec` + `_codecs_{cn,tw,hk,jp,kr,iso2022}` |
| UTF-7 | `utf_7` | 3 | C: `PyUnicode_DecodeUTF7Stateful`, `_PyUnicode_EncodeUTF7` |
| Escape codecs | `unicode_escape`, `raw_unicode_escape` | 0 | C: `unicodeobject.c` + unicodedata's name lookup |
| Punycode | `punycode` | 0 | Python: `encodings/punycode.py` |
| IDNA | `idna` | 0 | Python: `encodings/idna.py` + `stringprep.py` + `unicodedata.ucd_3_2_0` |
| Windows code pages | `mbcs`, `oem` | 2 (`ansi`, `dbcs`) | Win32 `MultiByteToWideChar` / `WideCharToMultiByte` |
| Generic charmap | `charmap` | 0 | `codecs.charmap_encode` / `charmap_decode` with no mapping, which is Latin-1 |

**What it costs** (§6 has the table). The estimate:
- about 8,300 hand-written lines of C# at the house documentation density, porting about 7,500 lines of CPython's C and
  Python statement for statement;
- about 0.34 MB of generated tables in the assembly, about 6% of a 5.65 MB Release `NumSharp.dll` (net10.0, built
  2026-10-30 from the strings line), plus the new code's IL;
- two table generators, and about 7 K new oracle cases;
- 9–11 working sessions at the pace of P2.6 milestone 4.

The 24 CJK codecs are about 60% of all of it.

No family needs a kernel, a dtype switch or a new NDIter mode. Every codec plugs into the existing per-element
`NDCodec` dispatch under `CodecWalk` (`NDCharObjectPath.Codec.cs`), as the 81 implemented modules do. The codecs work
on code points and bytes, so dtype is already handled before the codec runs. Their bodies are table-driven,
variable-length C, and IL emission has nothing to specialize there. The one vectorizable part, all-ASCII elements of
the ASCII-transparent codecs, can reuse the existing narrow / widen runs.

**Recommended order** (§6). Batch A first, about 2 sessions of small ports on the existing error-handler machinery:
- `charmap`;
- the off-Windows `LookupError` for `mbcs` / `oem`, a divergence today (§1.4);
- `utf_7`;
- the two escape codecs;
- `punycode`.

Then batch B, the CJK codecs (4–5 sessions). They serve real data files: GBK, Shift-JIS, CP949 and Big5 exports through
`np.loadtxt` / `np.savetxt`. Then batch C, `idna` and `mbcs` / `oem` on Windows (1.5–2 sessions).

## 1. The inventory

### 1.1 The modules

`strings-codecs/inventory.py` prints the classification below. The aliases by module:
- `gb2312` 9, `gbk` 3, `gb18030` 1, `hz` 3;
- `big5` 3, `cp950` 2, `big5hkscs` 2;
- `shift_jis` 5, `cp932` 4, `euc_jp` 3, `shift_jis_2004` 3, `shift_jisx0213` 3, `euc_jis_2004` 3, `euc_jisx0213` 1;
- `euc_kr` 8, `cp949` 3, `johab` 2;
- `iso2022_jp` 3, `iso2022_kr` 3, and each other `iso2022_*` 2;
- `utf_7` 3, `mbcs` 2.

Codec state:
- ISO-2022 and HZ carry shift state in both directions.
- The JIS X 0213 and Big5-HKSCS encoders look one character ahead for combining pairs. An incremental encoder holds
  that character as pending text.
- The CJK, UTF-7, escape and code-page decoders carry an incomplete sequence across the chunks of an incremental decode.

### 1.2 Where NumSharp refuses them today

- **The generated table.** `tools/gen_python_codecs.py` writes the 32 modules into `PythonCodecs.Tables.cs` as
  `PythonCodecKind.Unsupported`. It generated the table on Windows, so `mbcs` and `oem` are listed on every OS.
- **The two lookups.** `NDCodec.Lookup` and `TextIOCodec.Lookup` throw `NotSupportedException` naming the codec.
  `NDCodec.Lookup` serves `np.strings.encode` / `decode`, `np.char.encode` / `decode` and `chararray.encode` /
  `decode`; `TextIOCodec.Lookup` serves the `encoding=` of `np.loadtxt` / `np.savetxt`.
- **The pins.**
  - `StringCodecTests` covers 6 names.
  - `TextIOCodecTests` covers 6 names, plus `big5`.
  - The leak catalogue has a `utf-7` encode entry and an `idna` decode entry.
  - `test/oracle/gen_strings_oracle.py` keeps the refused codecs out of `strcodec`, `tcodec` and the text-I/O sections.

### 1.3 The entry points that gain them

1. `np.strings.encode` / `decode` → `NDCharObjectPath.Code` → `NDCodec`. The same path serves `np.char.encode` /
   `decode` and the `chararray` members, so the new codecs need no change in the walk, the rows or the argument checks.
   The general route codes any element through `NDCodec.Encode` / `Decode` into the arena.
2. `np.loadtxt(..., encoding=)`:
   - a byte stream decodes each line with `bytes.decode`, one-shot;
   - a file name goes through TextIOWrapper's incremental decoder, chunk by chunk.
3. `np.savetxt(..., encoding=)`:
   - a byte stream encodes each write with `str.encode`, one-shot, so a stateful codec resets after every write;
   - a file name goes through TextIOWrapper's incremental encoder, whose state carries across writes and is never
     flushed at close (§2).

### 1.4 Gaps found on the way

- **Off Windows.** NumSharp answers `mbcs`, `oem`, `ansi` and `dbcs` with `NotSupportedException`. NumPy 2.4.2 on
  CPython 3.12.3 (Linux, verified in WSL) raises `LookupError: unknown encoding: mbcs`: the modules import Windows-only
  functions, and `encodings.search_function` swallows the `ImportError`. An empty `S` element still decodes to `''`
  there, as on any OS.
- **The Windows default encoding.** A file name without `encoding=` is pinned to UTF-8, a documented divergence. CPython
  uses the locale's `cp<ACP>`:
  - on a 1252 host that is the charmap codec `cp1252`, which NumSharp has;
  - on a CJK-locale host it is `cp932`, `gbk` (`cp936`), `cp949` or `cp950`, all four of them refused today.
- **Out of scope.** User-registered codecs and handlers (`codecs.register`, `codecs.register_error`) are Python-runtime
  features with no NumPy API. An unknown handler name keeps CPython's `LookupError` at the first error.

## 2. What NumPy answers (observed)

Selected lines of `strings-codecs/probe_numpy.py`. `encode` / `decode` are `np.strings.encode` / `np.strings.decode`
over a one-element array unless the line says otherwise.

| Call | NumPy 2.4.2 / CPython 3.12.12 | What it shows |
|---|---|---|
| `'Ā'.encode('charmap')` | `UnicodeEncodeError: 'latin-1' codec can't encode character '\u0100' in position 0: ordinal not in range(256)` | `charmap` is Latin-1 with Latin-1's error texts |
| `b'\x81'.decode('mbcs')` / `.decode('cp1252')` | `'\x81'` / `'charmap' codec can't decode byte 0x81 ...` | `mbcs` reads the OS table, not the `cp1252` module |
| `'Ā'.encode('mbcs')` | `'mbcs' codec can't encode characters in position 0--1: invalid character` | the strict error's range is `[0, 0)` |
| `'Ā'.encode('mbcs', 'replace')` / `'cp1252'` | `b'A'` / `b'?'` | `replace` turns Windows' best fit on |
| `'€'.encode('oem')` | `'cp1' codec can't encode characters in position 0--1 ...` | `oem` names itself `cp1` (`CP_OEMCP`) |
| `encode(['中文', 'abc'], 'gbk')` | `[b'\xd6\xd0\xce\xc4', b'abc']`, `S4` | a two-byte charset |
| `encode(['日本語abc'], 'iso2022_jp')` | `b'\x1b$BF\|K\\8l\x1b(Babc'` | designations, reset before ASCII |
| `encode(['€'], 'shift_jis')` | `'shift_jis' codec can't encode character '\u20ac' in position 0: illegal multibyte sequence` | the multi-byte reasons and error sizes |
| `encode(['中~', 'a~b'], 'hz')` | `[b'~{VP~}~~', b'a~~b']` | HZ shifts, `~` doubled, reset at the end |
| `encode(['中€'], 'hz', 'backslashreplace')` | `b'~{VP~}\\u20ac'` | the replacement is encoded by the codec, in its state |
| `encode(['\udc80'], 'gbk', 'surrogateescape')` / `'surrogatepass'` | `b'\x80'` / the original `UnicodeEncodeError` | bytes replacements are copied; `surrogatepass` is UTF-only |
| `decode([b'\x82'], 'shift_jis')` / `[b'\x82\x20']` | `... incomplete multibyte sequence` / `... illegal multibyte sequence` | `MBERR_TOOFEW` against an illegal byte |
| `decode([b'\xff'], 'gbk', 'xmlcharrefreplace')` | `TypeError: don't know how to handle UnicodeDecodeError in error callback` | the generic handlers, unchanged |
| `decode([b'\x95\x32\x82\x36'], 'gb18030')` | `'𠀀'` (U+20000) | GB18030's four-byte linear space |
| `decode([b'\xa4\xf7'], 'euc_jis_2004')` and back | `'か゚'` (two code points) and `b'\xa4\xf7'` | JIS X 0213 pairs both ways |
| `decode([b'\x88\x62'], 'big5hkscs')` | `'Ê̄'` | Big5-HKSCS pairs |
| `encode(['a+b~中\U0001F600'], 'utf-7')` | `b'a+-b+AH5OLdg93gA-'` | UTF-7 sets D / O, a surrogate pair in base64 |
| `encode(['\ud800'], 'utf-7')` / `decode([b'+2D3'], 'utf-7')` | `b'+2AA-'` / `'utf7' codec can't decode bytes in position 0-3: unterminated shift sequence` | UTF-7 never fails to encode; its decode errors name `utf7` |
| `encode(['a\\b\té€\U0001F600\ud800'], 'unicode_escape')` | `b'a\\\\b\\t\\xe9\\u20ac\\U0001f600\\ud800'` | lowercase hex, never fails |
| `decode([b'\\N{euro sign}'], 'unicode_escape')` / `b'\\N{null}'` / `b'\\N{KEYCAP NUMBER SIGN}'` | `'€'` / `'\x00'` / `unknown Unicode character name` | names case-insensitive, aliases yes, named sequences no |
| `decode([b'\\N{hangul syllable ga}'], 'unicode_escape')` / `b'\\N{CJK UNIFIED IDEOGRAPH-4e00}'` | both `unknown Unicode character name` (the upper-case spellings decode) | the computed names are case-sensitive |
| `decode([b'\\q'], 'unicode_escape')` | `'\\q'`; under `-W error` a `DeprecationWarning: invalid escape sequence '\q'` | the warning NumSharp cannot emit |
| `encode(['münchen.de', 'Bücher.example.'], 'idna')` | `[b'xn--mnchen-3ya.de', b'xn--bcher-kva.example.']` | nameprep + punycode, the trailing dot kept |
| `encode(['ü'], 'idna', 'replace')` / `encode(['اa'], 'idna')` | `UnicodeError: unsupported error handling replace` / `UnicodeError: Violation of BIDI requirement 2` | IDNA is strict-only, plain `UnicodeError` |
| `decode([b'\xff.de'], 'idna')` | `'ascii' codec can't decode byte 0xff in position 0 ...` | the position counts within the label |
| `encode(['bücher', 'abc', '中文'], 'punycode')` | `[b'bcher-kva', b'abc-', b'fiq228c']` | punycode |
| `b'99999999999999999999a'.decode('punycode')` | `UnicodeError: Invalid character U+102403d3847848f8109` | Python's unbounded integers reach the text |
| `np.loadtxt(<shift_jis file>, dtype=str, encoding='shift_jis')` | `[['名前', '値'], ['東京', '1']]` | TextIOWrapper with a multi-byte decoder |
| `np.savetxt(<file>, [['日本', '語']], fmt='%s', newline='', encoding='iso2022_jp')` | `b'\x1b$BF\|K\\\x1b(B \x1b$B8l'` | a file ends in JIS mode: TextIOWrapper never calls `encode(final=True)` |
| the same with `hz` into a `BytesIO` | `b'~{VP~} ~{ND~}'` | a stream write is a one-shot encode, reset included |

## 3. The families: CPython's code, the port, the data, the gates

### 3.1 `charmap`

- **Source.** `encodings/charmap.py` calls `codecs.charmap_encode(input, errors)` / `charmap_decode(input, errors)`
  without a mapping. `PyUnicode_EncodeCharmap` / `PyUnicode_DecodeCharmap` send a missing mapping to
  `unicode_encode_ucs1(.., 256)` / `PyUnicode_DecodeLatin1`.
- **Behaviour.** Latin-1 byte for byte, with errors that quote `'latin-1'`. `codecs.lookup` reports the name `charmap`,
  so TextIOWrapper takes the incremental encoder, which gives the same bytes.
- **Port.** A module entry with Latin-1's coders and the lookup name `charmap`.
- **Cost.** About 30 lines. The gates are a `charmap` row in the strcodec pools and one text-I/O case.

### 3.2 `mbcs` / `oem` (Windows only)

**Source.** `unicodeobject.c`, "MBCS codecs for Windows", 701 lines:
- `decode_code_page_strict` first;
- on `ERROR_NO_UNICODE_TRANSLATION`, `decode_code_page_errors`, which decodes one character at a time over up to 4 bytes;
- `encode_code_page_strict`, with `WC_NO_BEST_FIT_CHARS` and `usedDefaultChar`;
- then `encode_code_page_errors`, which encodes one code point at a time through the generic handler;
- chunks of `INT_MAX / 4`.

`PyUnicode_Decode` / `PyUnicode_AsEncodedString` also give the normalized name `mbcs` a fast path on Windows.

**Behaviour** (§2):
- the OS's tables decide every answer;
- a strict error carries CPython's `[0, 0)` range;
- `replace` lifts `WC_NO_BEST_FIT_CHARS`, so Windows' best fit applies;
- `oem` errors name `cp1`;
- an encode error handler's str must be ASCII, else "unable to encode error handler result to ASCII".

**Port options:**
- **(a)** P/Invoke kernel32 (`GetACP`, `GetOEMCP`, `MultiByteToWideChar`, `WideCharToMultiByte`) behind
  `OperatingSystem.IsWindows()`, with the C statements ported. This is exact, because it IS what CPython calls, and
  Core already imports kernel32 (`Backends/Unmanaged/Pooling/OsVirtualMemory.cs`). Off Windows the module does not
  exist (`LookupError`), as in CPython.
- **(b)** .NET's `CodePagesEncodingProvider`. Managed, but its best fit and error ranges are .NET's, not the OS's. Not
  exact.
- **(c)** Keep refusing on Windows and fix only the off-Windows error type.

**Data.** None: the tables are the OS's.

**Gates.** Unit tests that run on Windows only (the `[WindowsOnly]` category, which CI excludes elsewhere), whose
expected answers depend on the host's `GetACP()` / `GetOEMCP()` (1252 / 437 here; check the CI runner before relying on
it). An oracle case records the code page and is skipped on another host.

**Cost.** About 550 lines, 0.5 session. Risk: medium. The answers depend on the host: a Japanese-locale Windows answers
with cp932's table.

### 3.3 `utf_7`

**Source.** `unicodeobject.c`, "UTF-7 Codec", 400 lines:
- `PyUnicode_DecodeUTF7Stateful`: the shift state, decode errors through `unicode_decode_call_errorhandler_writer` with
  the codec name `utf7`, and the `restart` after an unterminated shift sequence;
- `_PyUnicode_EncodeUTF7`: never fails, ignores `errors`, and writes set O and whitespace directly.

**Behaviour:**
- Lone surrogates encode.
- A non-BMP character is a surrogate pair inside base64.
- The reasons: "unexpected special character", "ill-formed sequence", "partial character in shift sequence", "non-zero
  padding bits in shift sequence", "unterminated shift sequence".
- The incremental decoder backs off to the start of an open shift sequence (`consumed`), which TextIOWrapper's chunking
  exposes.

**Port.** An `NDCodec` coder pair, and a `BufferedIncrementalDecoder` subclass in `TextIOCodec`.

**Cost.** About 400 lines, 0.5 session.

### 3.4 `unicode_escape` / `raw_unicode_escape`

**Source.** `unicodeobject.c`: "Unicode Escape Codec" (436 lines) and "Raw Unicode Escape Codec" (223 lines), plus
unicodedata.c's `_getcode` (name → code point, about 120 lines).

**Encoders.** They never fail, and ignore `errors`.
- `unicode_escape` writes printable ASCII as it is, `\\`, `\t` `\n` `\r`, `\xhh` below U+0100, `\uhhhh`, and
  `\U00hhhhhh`, in lowercase hex.
- `raw_unicode_escape` writes code points below U+0100 as their byte.

**Decoders.** Errors go through the generic handler, with the codec names `unicodeescape` / `rawunicodeescape` and these
reasons: "\\ at end of string", "truncated \\xXX escape", "truncated \\uXXXX escape", "truncated \\UXXXXXXXX escape",
"illegal Unicode character", "malformed \\N character escape", "unknown Unicode character name", and
"\\Uxxxxxxxx out of range".

**`\N{name}`.**
- Stored names and aliases match case-insensitively (`_gethash` and `_cmpname` upper-case ASCII).
- Named sequences are refused (`with_named_seq = 0`).
- The computed names are case-sensitive: `HANGUL SYLLABLE ` and `CJK UNIFIED IDEOGRAPH-` are compared with `strncmp`,
  and the latter takes 4 or 5 upper-case hex digits of a unified ideograph.

**The warning.** An invalid escape such as `\q`, or an octal escape past 0o377, decodes and raises a
`DeprecationWarning`. Python's default filters keep it silent; `-W error` makes it an exception.

**Port.**
- An `NDCodec` coder pair for each codec.
- The reverse name lookup reuses the existing `PyUnicodeNames` blob: a case-insensitive dictionary built at the first
  `\N{`, plus CPython's computed Hangul and CJK names.
- The 474 alias targets (U+F0000..U+F01D9 to the real code points) come from `gen_unicode_names.py`, which gains them;
  the named sequences at U+F0200..U+F03CD are refused.
- NumSharp has no warnings module, so the `DeprecationWarning` becomes a `[Misaligned]` note, as `np.loadtxt`'s
  empty-input `UserWarning` is.

**Data.** About 2 KB. **Cost.** About 700 lines, 1 session.

### 3.5 `punycode` and `idna`

**Punycode.** The source is `encodings/punycode.py`, 237 lines.
- Encode never fails.
- Decode accepts `strict`, `replace` and `ignore` only; any other name is "Unsupported error handling ...".
- Its errors are plain `UnicodeError`s: "incomplete punicode string", "Invalid extended code point '%s'", "Invalid
  character U+%x".
- Python's unbounded integers reach the text: a long run of digits grows the decoded delta past 64 bits, and the
  "Invalid character U+%x" error prints the resulting code point in full (§2), so the port needs a `BigInteger`
  fallback after a `long` fast path.
- The incremental decoder decodes each chunk on its own, unbuffered, so TextIOWrapper's chunk boundaries change the
  answer, and the port must keep them.

About 300 lines, 0.5 session.

**IDNA.** The source is `encodings/idna.py` (317 lines) and `stringprep.py` (272 lines), with `ucd_3_2_0`'s NFKC from
unicodedata.c (`nfd_nfkd`, `nfc_nfkc`, `get_decomp_record`; about 330 lines of C).

**Errors.** IDNA is strict only: any other handler is a `UnicodeError`, "unsupported error handling replace" on encode
and "Unsupported ..." on decode. Its texts are plain `UnicodeError`s:
- "label empty or too long", "label too long";
- "Invalid character %r";
- "Violation of BIDI requirement 2" and "... 3";
- "Label starts with ACE prefix", "label way too long", "Invalid character in IDN label";
- the round-trip error, which prints a tuple.

**Splitting.** An ASCII input takes the fast path and splits on `.` only. Any other input splits on the four dots
U+002E, U+3002, U+FF0E and U+FF61. Decoding non-ASCII bytes raises the `ascii` codec's error at the position within
the label.

**Nameprep** has five steps:
1. Remove table B.1.
2. Map through B.2: the B.3 exceptions, `str.lower` at Unicode 15.0, and NFKC at 3.2.0, applied twice.
3. Normalize with NFKC at 3.2.0.
4. Prohibit tables C.1.2–C.9, at 3.2.0.
5. Apply the bidi rule, at 3.2.0.

**NFKC at 3.2.0.** CPython builds it from these parts:
- the CURRENT decomposition data, gated to the code points 3.2.0 assigned (measured: this is identical to
  `ucd_3_2_0.decomposition` everywhere);
- the 5 normalization corrections U+2F868, U+2F874, U+2F91F, U+2F95F and U+2F9BF;
- canonical ordering by the CURRENT combining classes;
- composition by the CURRENT pairs, with Hangul composed algorithmically;
- no quick check.

**Version.** CPython 3.13 changed `idna.py`'s errors to `UnicodeEncodeError` / `UnicodeDecodeError` with positions (a
diff of v3.12.12 against v3.13.0). The port follows 3.12.12 and needs a re-read when the oracle moves.

**Port.**
- idna's statements: the labels, ToASCII, ToUnicode, and the label buffering of the codec and of its incremental
  classes.
- stringprep's tables.
- A 3.2.0 NFKC.
- `str.lower` from the existing `PyUnicodeData` case maps, and punycode from above.

**Data** (measured by `measure_tables.py`). About 30 KB compressed, with a SHA-256 pin like the other Unicode tables:
- 1,003 ranges of 3.2.0 category × bidi class, 2 KB after zlib;
- 5,143 gated decompositions, 23 KB after zlib;
- 922 non-zero combining classes;
- 941 composition pairs;
- the 5 corrections.

**Cost.** About 900 lines, 1–1.5 sessions. Risk: medium. The behaviour is specific to the CPython version, and the
codec has few users in array code.

### 3.6 The 24 CJK multi-byte codecs

**Source**, from `Modules/cjkcodecs/`:
- `multibytecodec.c`: 2,084 lines, of which about 800 matter here (the encode / decode loops, `multibytecodec_encerror`
  / `decerror`, the incremental encoder and decoder);
- `_codecs_cn.c` 470, `_codecs_hk.c` 190, `_codecs_jp.c` 765, `_codecs_kr.c` 468, `_codecs_tw.c` 143,
  `_codecs_iso2022.c` 1,150;
- `cjkcodecs.h` 525 (macros and module glue), `alg_jisx0201.h` 65, `emu_jisx0213_2000.h` 54;
- `mappings_*.h`, the generated tables.

**Return codes.** A coder returns one of four things:
- 0;
- a positive error size: usually 1, the lead byte or the one character;
- `MBERR_TOOFEW`, an incomplete sequence: the rest of the input;
- `MBERR_TOOSMALL`: grow the buffer and retry.

The reasons are "illegal multibyte sequence" and "incomplete multibyte sequence", and errors name the codec itself.

**Handlers.**
- `strict`, `ignore` and `replace` are handled inline.
- An encode `replace` encodes `'?'` THROUGH the codec in its current state, so an ISO-2022 or HZ escape comes first, and
  falls back to a raw `'?'`. A decode `replace` writes one U+FFFD per error.
- Every other name goes through the registry. A str replacement is then encoded by the codec, strictly and in its
  current state; a bytes replacement is copied.
- `surrogatepass` refuses, since this is not a UTF codec.
- `xmlcharrefreplace` and `namereplace` on decode raise CPython's `TypeError`.

**Pairs.**
- One sequence can decode to two code points: the 49 JIS X 0213 pair entries and the 4 Big5-HKSCS ones.
- Two code points can encode to one sequence: 46 JIS X 0213 pairs and 4 Big5-HKSCS ones. The encoder looks one
  character ahead, and an incremental encoder keeps that character pending (`MBERR_TOOFEW` without `MBENC_FLUSH`).

**State.**
- ISO-2022 keeps its designations and SO / SI; HZ keeps `~{` / `~}`, plus `~~` and `~\n`.
- A one-shot encode resets at the end (`MBENC_RESET`). A TextIOWrapper file does not (§2).

**Algorithmic parts:**
- GB18030's four-byte space: 158 BMP ranges, and the supplementary planes as a linear range;
- CP949's and Johab's Hangul;
- EUC-KR's KS X 1001:1998 make-up sequences: the encoder writes 8 bytes for a CP949-extension syllable;
- CP932's user-defined area and its Windows compatibility bytes;
- the JIS X 0213:2000 emulation (`emu_jisx0213_2000.h`) used by `euc_jisx0213`, `shift_jisx0213` and the ISO-2022
  JIS X 0213:2000 designations.

**A read to keep out of the port.** CPython's HZ decoder reads the byte after a `~` (`INBYTE2`) before
`REQUIRE_INBUF(2)` checks that it exists. That is harmless on a NUL-terminated bytes object, but in NumSharp it would
read past an `S` element, and past the buffer at an array's last element. The port checks first. The answer is the
same, and the guard-page run (`NUMSHARP_DEBUG_GUARD_PAGES=1`) would catch a copied read.

**Port.**
- One `NDMultibyteCodec` framework: the loops and error calls over `NDCodecSink`, the existing generic handlers, and an
  8-byte state.
- 24 coder pairs in six files that mirror CPython's.
- `TRYMAP_ENC` / `TRYMAP_DEC` as inlined lookups over the 27 charset indexes.
- `TextIOCodec` gains a multi-byte incremental decoder, which keeps up to 8 pending bytes, and an incremental encoder
  that keeps pending text and carries its state across writes.

**Data** (measured from the interpreter by `measure_tables.py`).
- 27 charset indexes: 15 decode, 11 encode, and the JIS X 0213 pair decode map. They hold 292,388 values, 599 KB raw.
- zlib-9 of each row's values as deltas split into byte planes gives 232 KB: cn 42, hk 31, jp 71, kr 39, tw 49. hk
  compresses better plain, at 27, which makes the best mix 227 KB.
- As base64 in a `"""..."""u8` literal (the `PyUnicodeNames.Data.cs` precedent) that is 309 KB in the assembly.
- Each language is inflated at its first use, 81–184 KB each.
- Small C tables come with it: GB18030's ranges, the pair tables, and Johab's and EUC-KR's jamo tables.

**The generator** reads the `__map_*` capsules with ctypes from the oracle's interpreter: no CPython source needed, and
exact. It compresses them, and pins a SHA-256 over an exhaustive sweep of every codec:
- every 1- and 2-byte input decoded;
- `0x8F` + 2 bytes for EUC-JP / EUC-JIS-2004 / EUC-JISX0213;
- GB18030's 1.59 M four-byte inputs;
- every code point encoded.

CPython sweeps a codec in about 1 s (every code point encoded) plus 0.1 s (every 2-byte input decoded).

**Gates.**
- C# recomputes the sweeps' hashes: exact, with a small corpus.
- Scripted sequences cover the stateful codecs: escapes and shifts mixed, errors inside each mode, every handler.
- `np.strings` oracle cases per codec × handler over CJK pools.
- Text-I/O cases at TextIOWrapper's chunk boundaries: a lead byte split, an escape sequence split.

**Cost.** About 4,800 lines: framework 800, tables 300, cn 500, tw 180, hk 250, kr 550, jp 900, iso2022 1,300. The
generator is about 350 lines of Python. 4–5 sessions. Risk: medium to high. ISO-2022 is the largest single file and the
only real state machine, and the assembly grows.

## 4. Shared work (once)

- **The registry.** `gen_python_codecs.py` gets kinds for the new families, so no `Unsupported` is left, and marks
  `mbcs` / `oem` Windows-only. Off Windows such a module is CPython's "unknown encoding", which fixes §1.4. On Windows,
  `NDCodec.Lookup` / `TextIOCodec.Lookup` honour the `mbcs` fast path.
- **`TextIOCodec`.** It delegates the new kinds to `NDCodec` for its one-shot decode and encode: strict, with a bridge
  between UTF-16 chars and code points. It gains incremental classes:
  - buffered: `utf-7`, the escape codecs, `mbcs`, and IDNA's label buffering;
  - stateless per chunk: `punycode` and `charmap`;
  - multi-byte: CJK.
- **The oracle.** `CODEC_NAMES` gains the new codecs, with pools per family: CJK texts, escape sequences, UTF-7 shifts,
  domain names, punycode labels. The text-I/O sections get a codec list too. Estimated growth about 7 K cases; strcodec
  today has 13,773 cases in 9.9 MB.
- **Tests.** The refusal tests go. The leak catalogue's `utf-7` / `idna` entries then exercise success. Every new coder
  is pointer code, so it also runs under `NUMSHARP_DEBUG_GUARD_PAGES=1` over an `S` / `U` array whose last element ends
  at its buffer's end.
- **Docs.** Every text that lists the refused codecs changes:
  - the headers and XML docs of `NDCodec.cs`, `PythonCodecs.cs`, `TextIOCodec.cs`, `np.strings.Codec.cs` and
    `np.loadtxt.cs`;
  - `.claude/rules/text-io.md` and `strings-methods.md`;
  - `docs/numpy/STRING_COMPAT_MATRIX.md`;
  - the website's I/O page;
  - the release notes.
- **Cost.** About 600 lines plus the generator and test edits, 1 session.

## 5. Speed

NumPy reaches these codecs through the registry, never through a C fast path. Measured by `bench_numpy.py`, pinned to
two CPUs, best of 5 runs:

| Input | Codec | Encode | Decode |
|---|---|---|---|
| 100K × 12 CJK chars | `utf-8` (C fast path) | 697 ns / element | 653 ns / element |
|  | `gbk` | 1,099 | 1,153 |
|  | `utf-7` | 1,043 | 1,387 |
|  | `unicode_escape` | 1,048 | 1,305 |
| 100K short ASCII | `utf-8` | 634 | 509 |
|  | `gbk` / `shift_jis` / `big5` | 983 / 1,021 / 996 | 990 / 1,023 / 1,006 |
|  | `utf-7` / `unicode_escape` | 1,023 / 1,033 | 1,096 / 1,019 |
| 1 × 100K CJK chars | `utf-8` | 0.64 ms | 0.75 ms |
|  | `gbk` | 1.03 ms (10.3 ns / char) | 1.46 ms (14.6 ns / char) |
|  | `utf-7` | 1.05 ms | 2.76 ms |
|  | `unicode_escape` | 0.76 ms | 2.22 ms |

**Short elements.** NumPy spends about 1 µs per short element, almost all of it in the call. NumSharp's per-element
route, at 63.6× NumPy in geomean over its own codecs today, has a wide margin there.

**Long elements.** The 1.5× bar bites here, because NumPy pays 7.6–27.6 ns per character. CPython's multi-byte decoder
writes each character through `_PyUnicodeWriter_WriteChar`, and its encoder reads through `PyUnicode_READ` behind a
function-pointer loop. A scalar C# port that writes into the sink directly is expected to clear the bar.
**Unmeasured**: the bar has to be proven per codec once the port exists.

The DBCS codecs (all but HZ and ISO-2022) pass bytes and code points below 0x80 through unchanged, so their all-ASCII
elements can take the existing vectorized narrow / widen runs; the sweeps verify this per codec.

## 6. Cost summary and order

| Family | Modules (aliases) | CPython source ported | Generated data in the DLL | Hand-written C# | Sessions | Risk |
|---|---|---|---|---|---|---|
| `charmap` | 1 (0) | two calls with no mapping | — | ~30 | < 0.1 | none |
| `mbcs` / `oem` | 2 (2) | 701 lines C over Win32 | — (the OS's) | ~550 | 0.5 | medium: host-dependent |
| `utf_7` | 1 (3) | 400 lines C | — | ~400 | 0.5 | low |
| escape codecs | 2 (0) | 659 + ~120 lines C | ~2 KB | ~700 | 1 | low |
| `punycode` | 1 (0) | 237 lines Python | — | ~300 | 0.5 | low |
| `idna` | 1 (0) | 589 lines Python + ~330 lines C | ~30 KB | ~900 | 1–1.5 | medium |
| CJK | 24 (74) | ~4,500 lines C | ~310 KB (227–232 KB deflated) | ~4,800 | 4–5 | medium–high |
| shared work | — | — | — | ~600 | 1 | low |
| **Total** | **32 (79)** | **~7,500 lines** | **~0.34 MB** | **~8,300** | **9–11** | |

**The basis.** P2.6 milestone 4 (`1236cccc`: NDCodec, the codec walk, the names, the API) committed about 6,700
hand-written lines, 852 test lines and 16 K oracle cases 7.5 hours after milestone 3's last commit, and its speed pass
(`d314dd81`) 5 hours later. Ported table-driven codecs under exhaustive verification are slower per line, hence the
ranges.

**What it adds to run time.** Nothing at startup. Each CJK language's tables are inflated at its first use, and the
IDNA data at the first `idna` call.

**Suggested order.**
1. **Batch A** (about 2 sessions): `charmap`, the off-Windows `LookupError`, `utf_7`, the escape codecs, `punycode`. These
   are small ports on the existing error-handler machinery, and they move 5 modules and `utf_7`'s 3 aliases from refused
   to computed (`mbcs` / `oem` stay refused on Windows until batch C).
2. **Batch B** (4–5 sessions): the CJK codecs, one language at a time (cn → tw / hk → kr → jp → iso2022), each with its
   sweep pin, and the TextIO multi-byte classes with the first language. This is the batch with user value: CJK
   exports read and written through `np.loadtxt` / `np.savetxt`.
3. **Batch C** (1.5–2 sessions): `idna` (Unicode 3.2.0 nameprep), and `mbcs` / `oem` on Windows.

The value ranking is a judgement, not a measurement. A code-search survey (`docs/plans/numpy-missing-api-usage.md`'s
method) can rank the codecs by real use if wanted.

## 7. Decisions the port needs from the owner

- **D1 — `mbcs` / `oem` on Windows.** (a) P/Invoke kernel32: exact, Windows-only gates. Recommended. (b) .NET's code
  page provider: managed, inexact. (c) Keep refusing on Windows. Every option fixes the off-Windows `LookupError`.
- **D2 — CJK table storage.** (a) Deflated base64 in a `u8` literal, inflated per language at first use: +0.31 MB, the
  `PyUnicodeNames` precedent. Recommended. (b) Raw RVA data: +0.6 MB, no inflate, paged in by the OS. (c) An
  `EmbeddedResource`: +0.23 MB, read as a stream.
- **D3 — `unicode_escape`'s `DeprecationWarning`.** Not emitted, with a `[Misaligned]` note. Recommended: NumPy's
  default is silence, and NumSharp has no warnings module.
- **D4 — Scope and order.** The batches above. A separate question: once the CJK codecs exist, should a file name
  without `encoding=` follow CPython on Windows (`cp<ACP>`) instead of the pinned UTF-8?

## 8. Reproduce

Run with the oracle's interpreter (CPython 3.12, NumPy 2.4.2):

```bash
python -I docs/plans/strings-codecs/inventory.py        # §0, §1: the 32 modules, 79 aliases, what CPython runs
python -I docs/plans/strings-codecs/measure_tables.py   # §3.5, §3.6: CJK charset tables, Unicode 3.2.0 data
python -I docs/plans/strings-codecs/probe_numpy.py      # §2: NumPy's answers (mbcs / oem lines need Windows)
# §5, pinned to two CPUs (cmd):  start /b /wait /affinity 3 python -I docs/plans/strings-codecs/bench_numpy.py
```

The CPython 3.12.12 sources read for this discovery are `Modules/cjkcodecs/*`, `Objects/unicodeobject.c` (the UTF-7,
escape, MBCS and charmap sections), `Modules/unicodedata.c` and `Modules/_codecsmodule.c` from tag `v3.12.12`, and
`Lib/encodings/{idna,punycode,charmap,mbcs,oem,utf_7,unicode_escape,raw_unicode_escape}.py` and `Lib/stringprep.py`
from the installed interpreter.
