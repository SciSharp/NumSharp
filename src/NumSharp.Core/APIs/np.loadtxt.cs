using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using NumSharp.Backends.Strings;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy 2.4.2 reference: numpy/lib/_npyio_impl.py::loadtxt -> _read, and the C text reader it calls
        // (numpy/_core/src/multiarray/textreading/{readtext.c, rows.c, tokenize.cpp, stream_pyobject.c, conversions.c}).
        //
        // The structure is NumPy's: the Python layer validates the arguments and picks the input stream (a file NAME is read
        // in chunks with universal newlines; anything else is iterated line by line; multiple or multi-character comments
        // are stripped from each line first); read_rows tokenizes row by row (LoadTxtTokenizer, a port of tokenize.cpp),
        // creates the converters at the first row, converts every selected field in order into a growing result, and
        // always produces a 2-D (rows, cols) array that _ensure_ndmin_ndarray then squeezes or expands. Every error is
        // NumPy's text, raised at the point NumPy raises it.

        #region loadtxt (public overloads)

        /// <summary>
        ///     Load data from a text file into an array (NumPy's <c>np.loadtxt</c>).
        /// </summary>
        /// <param name="fname">Path to the file, found and opened as NumPy's <c>DataSource</c> does: when nothing has that
        ///     name, the name with <c>.bz2</c>, <c>.gz</c>, <c>.xz</c> or <c>.lzma</c> appended is tried, in that order; a
        ///     name whose extension is exactly <c>.gz</c> is decompressed (<c>.GZ</c> is plain text, as in NumPy). The file is
        ///     read in text mode with universal newlines (<c>"\r\n"</c> and <c>'\r'</c> end lines too), so a quoted field may
        ///     span lines.</param>
        /// <param name="dtype">The result's dtype (default float64): a numeric dtype, a fixed-width string dtype (<c>'U5'</c>,
        ///     <c>'S'</c>; an unsized class takes the longest field's width), or a <c>StringDType</c> (<c>'T'</c>,
        ///     <c>np.dtypes.StringDType(...)</c>), which stores every field's text whole.</param>
        /// <param name="comments">What starts a comment (the rest of the line is ignored): a string (a Latin-1
        ///     <c>byte[]</c> and a <see cref="char"/> are one too), a sequence of them (<c>string[]</c>, any
        ///     <see cref="System.Collections.IEnumerable"/>; each is cut off at its first occurrence, in order), or null for
        ///     none. Default <c>"#"</c>. As in NumPy, a null element of a sequence keeps each line's first
        ///     whitespace-separated word (<c>line.split(None, 1)[0]</c>), and an element of another type raises at the first
        ///     line read.</param>
        /// <param name="delimiter">The column separator, one character; null (default) splits on runs of whitespace and
        ///     ignores leading whitespace.</param>
        /// <param name="converters">Per-field parsers: a <see cref="Func{String, Object}"/> for every column, or an
        ///     <see cref="IDictionary{TKey, TValue}"/> mapping a column (negative counts from the end without
        ///     <paramref name="usecols"/>; with it, a column is matched against the <paramref name="usecols"/> entries as given)
        ///     to a parser. The parser's result is stored with the dtype's element setter (NumPy's <c>PyArray_Pack</c>).</param>
        /// <param name="skiprows">Skip this many leading lines (blank lines and comments included; quotes are not honored
        ///     while skipping).</param>
        /// <param name="usecols">Which columns to read, in this order (negatives count from the end). Null reads all.</param>
        /// <param name="unpack">Transpose the result, so its columns unpack as separate arrays.</param>
        /// <param name="ndmin">The minimum number of dimensions (0, 1 or 2); size-1 axes beyond it are squeezed.</param>
        /// <param name="encoding">The file's text encoding: any name Python's <c>codecs.lookup</c> resolves (aliases
        ///     included: <c>"utf-16le"</c>, <c>"Windows-1252"</c>, <c>"L1"</c>) for UTF-8 (<c>utf-8-sig</c> too), UTF-16 and
        ///     UTF-32 (with or without a BOM; the BOM forms require one, CPython's <c>UTF-16 stream does not start with
        ///     BOM</c>), Latin-1, ASCII and the single-byte code pages. Null (and NumPy's <c>"bytes"</c>, and
        ///     <c>"locale"</c>) is UTF-8 here, NumPy's default on Linux and macOS (Windows NumPy uses the ANSI code page).
        ///     Decoding is strict: undecodable bytes raise <see cref="UnicodeDecodeError"/>, with CPython's text and a
        ///     position counted within the chunk NumPy's TextIOWrapper decodes.</param>
        /// <param name="max_rows">Read at most this many data rows after <paramref name="skiprows"/> (blank and comment-only
        ///     lines are not counted). Null reads all.</param>
        /// <param name="quotechar">The quote character: inside a quoted field, delimiters, comments and newlines are data,
        ///     and two quotes are one literal quote. Null (default) disables quoting.</param>
        /// <returns>The array: <c>(rows, cols)</c>, then squeezed to <paramref name="ndmin"/> (one row or one column gives
        ///     a 1-D result, one value a 0-d one), transposed when <paramref name="unpack"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="fname"/> is null.</exception>
        /// <exception cref="ValueError">NumPy's texts: an invalid <paramref name="ndmin"/>, a negative
        ///     <paramref name="skiprows"/> / <paramref name="max_rows"/>, an empty comment, a field that cannot be converted
        ///     (<c>could not convert string 'x' to float64 at row 0, column 1.</c>, the cause attached), a row with another
        ///     number of columns, an invalid column index, a converter for a column that does not exist, an unquoted newline
        ///     inside one line of a line-iterated input.</exception>
        /// <exception cref="TypeError">NumPy's texts: a control character that is not one character or None, or that
        ///     matches another (<c>The values for control characters 'quotechar' and 'delimiter' are incompatible</c>), a
        ///     newline as a control character, a comment sequence holding the delimiter, converters of another kind, a null
        ///     converter in the dictionary, <paramref name="comments"/> that are not iterable (<c>'int' object is not
        ///     iterable</c>) or hold an element that is neither a string nor null (<c>must be str or None, not int</c>, at
        ///     the first line read).</exception>
        /// <exception cref="IndexError">A null comment element over an empty or whitespace-only line (<c>list index out of
        ///     range</c>, NumPy's <c>str.split(None, 1)[0]</c>).</exception>
        /// <exception cref="UnicodeDecodeError">Bytes the encoding cannot decode.</exception>
        /// <exception cref="FileNotFoundException">No file named <paramref name="fname"/>, nor any of its compressed
        ///     variants (NumPy's <c>&lt;path&gt; not found.</c>), reported before the encoding is looked up.</exception>
        /// <exception cref="LookupError">An unknown <paramref name="encoding"/> (CPython's <c>unknown encoding: name</c>),
        ///     or a codec that is not a text encoding (<c>'base64' is not a text encoding; use codecs.open() to handle
        ///     arbitrary codecs</c>).</exception>
        /// <exception cref="UnicodeError">A <c>utf-16</c> / <c>utf-32</c> file that does not start with a BOM; the
        ///     <c>undefined</c> codec.</exception>
        /// <exception cref="NotSupportedException">A <c>.bz2</c>, <c>.xz</c> or <c>.lzma</c> file (.NET has no decompressor for
        ///     them), or a Python codec NumSharp does not implement (the multi-byte CJK codecs, <c>utf-7</c>, …).</exception>
        /// <remarks>
        ///     Parity with NumPy 2.4.2's <c>np.loadtxt</c>. NumPy's warnings (an input without data, an empty line under
        ///     <paramref name="max_rows"/>) are not raised: NumSharp has no warnings channel.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
        /// </remarks>
        public static NDArray loadtxt(string fname, [Optional, DefaultParameterValue(null)] DType dtype, [Optional, DefaultParameterValue("#")] object comments,
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, string encoding = null, int? max_rows = null, string quotechar = null)
        {
            if (fname is null) throw new ArgumentNullException(nameof(fname));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
            // _datasource.open(fname, 'rt', encoding): DataSource._findfile picks the file (the name, else the name with a
            // compressed extension appended), its extension the opener (gzip for exactly '.gz'), then io.open's TextIOWrapper
            // looks the encoding up. A name that finds nothing is FileNotFoundError("<path> not found."), reported ahead of an
            // unknown encoding. The file is read in chunks (stream_python_file), unless comments must be stripped line by
            // line first (then it is iterated).
            string found = PyDataSource.FindFile(fname) ?? throw new FileNotFoundException($"{fname} not found.", fname);
            Stream raw = PyDataSource.OpenRead(found);
            TextIOCodec codec;
            try
            {
                // 'bytes' is _read's "no encoding": the locale's, pinned to UTF-8.
                codec = TextIOCodec.Resolve(encoding == "bytes" ? null : encoding, TextIOCodec.Utf8, CodecLookupContext.Open);
            }
            catch
            {
                // The open file is this call's own: a failed lookup must not leave its handle to the finalizer.
                raw.Dispose();
                throw;
            }
            var file = new LoadTxtTextFile(raw, codec);
            using LoadTxtStream stream = cfg.MultiComments != null
                ? new LoadTxtLineStream(new LoadTxtFileLineReader(file), cfg.MultiComments)
                : new LoadTxtChunkStream(file);
            return LoadTxtCore(stream, cfg, unpack);
        }

        /// <summary>
        ///     Load data from an open byte <see cref="Stream"/> (NumPy's binary file object, <c>io.BytesIO</c>): read from the
        ///     current position and left open.
        /// </summary>
        /// <param name="stream">The bytes. They are iterated line by line as Python iterates a binary file — split after
        ///     every <c>'\n'</c>, each line decoded on its own, no newline translation (a <c>'\r'</c> inside a line, outside
        ///     quotes, is NumPy's embedded-newline error).</param>
        /// <param name="encoding">The encoding of the bytes (the codecs of the file-name overload). Null (and
        ///     <c>"bytes"</c>) is Latin-1, NumPy's default for a binary file object (<c>getattr(fname, 'encoding',
        ///     'latin1')</c>): every byte is one character. NumPy decodes each line on its own as it reads it
        ///     (<c>bytes.decode</c>: a BOM codec reads a BOM per line, native little-endian without one), so an unknown
        ///     name raises <see cref="LookupError"/> at the first line, and a stream without lines never meets it.</param>
        /// <exception cref="LookupError">An unknown or non-text <paramref name="encoding"/>, once the stream has a line.</exception>
        /// <exception cref="NotSupportedException">A Python codec NumSharp does not implement, once the stream has a
        ///     line.</exception>
        /// <inheritdoc cref="loadtxt(string,DType,object,string,object,int,int[],bool,int,string,int?,string)"/>
        public static NDArray loadtxt(Stream stream, [Optional, DefaultParameterValue(null)] DType dtype, [Optional, DefaultParameterValue("#")] object comments,
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, string encoding = null, int? max_rows = null, string quotechar = null)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
            // NumPy decodes each line of a byte stream as it reads it (bytes.decode), so it looks the codec up at the FIRST
            // line: an unknown name raises LookupError there (after the control-character checks), and never for a stream
            // without lines. 'bytes' is _read's "no encoding": getattr(fname, 'encoding', 'latin1').
            TextIOCodec codec;
            Exception lookupFailure = null;
            try
            {
                codec = TextIOCodec.Resolve(encoding == "bytes" ? null : encoding, TextIOCodec.Latin1, CodecLookupContext.Decode);
            }
            catch (Exception e) when (e is LookupError || e is NotSupportedException)
            {
                codec = null;
                lookupFailure = e;
            }
            using var lines = new LoadTxtLineStream(new LoadTxtByteLineReader(stream, codec, lookupFailure), cfg.MultiComments);
            return LoadTxtCore(lines, cfg, unpack);
        }

        /// <summary>
        ///     Load data from an open <see cref="TextReader"/> (NumPy's in-memory text stream, <c>io.StringIO</c>): read from
        ///     the current position and left open.
        /// </summary>
        /// <param name="reader">The text. It is iterated line by line as Python iterates <c>io.StringIO</c> — split after
        ///     every <c>'\n'</c>, a <c>'\r'</c> left as data (outside quotes, NumPy's embedded-newline error).</param>
        /// <param name="encoding">Text needs no decoding, so only NumPy's <c>"bytes"</c> has an effect: a converter's input
        ///     is the field's Latin-1 form (a field holding a character past <c>U+00FF</c> fails its conversion with
        ///     NumPy's <see cref="UnicodeEncodeError"/> as the cause). Any other name is accepted and ignored, as NumPy
        ///     ignores it for a text stream.</param>
        /// <inheritdoc cref="loadtxt(string,DType,object,string,object,int,int[],bool,int,string,int?,string)"/>
        public static NDArray loadtxt(TextReader reader, [Optional, DefaultParameterValue(null)] DType dtype, [Optional, DefaultParameterValue("#")] object comments,
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, string encoding = null, int? max_rows = null, string quotechar = null)
        {
            if (reader is null) throw new ArgumentNullException(nameof(reader));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
            using var lines = new LoadTxtLineStream(new LoadTxtTextLineReader(reader), cfg.MultiComments);
            return LoadTxtCore(lines, cfg, unpack);
        }

        /// <summary>
        ///     Load data from a sequence of lines (NumPy's list or generator of <c>str</c>).
        /// </summary>
        /// <param name="lines">The lines: each element is one line, as given — it may end in a newline, but a newline inside
        ///     it (outside quotes) is NumPy's <c>Found an unquoted embedded newline within a single line of input.</c> error,
        ///     and a quoted field may continue into the next element.</param>
        /// <param name="encoding">As for a <see cref="TextReader"/>: only <c>"bytes"</c> has an effect (a converter's
        ///     input is the field's Latin-1 form).</param>
        /// <inheritdoc cref="loadtxt(string,DType,object,string,object,int,int[],bool,int,string,int?,string)"/>
        /// <exception cref="TypeError">A null element (<c>non-string returned while reading data</c>).</exception>
        /// <exception cref="AttributeError">A null element when multiple or multi-character comments are stripped from each
        ///     line first (NumPy's <c>_preprocess_comments</c> meets it: <c>'NoneType' object has no attribute
        ///     'split'</c>).</exception>
        public static NDArray loadtxt(IEnumerable<string> lines, [Optional, DefaultParameterValue(null)] DType dtype, [Optional, DefaultParameterValue("#")] object comments,
            string delimiter = null, object converters = null, int skiprows = 0, int[] usecols = null,
            bool unpack = false, int ndmin = 0, string encoding = null, int? max_rows = null, string quotechar = null)
        {
            if (lines is null) throw new ArgumentNullException(nameof(lines));
            var cfg = BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
            using var stream = new LoadTxtLineStream(new LoadTxtSequenceLineReader(lines), cfg.MultiComments);
            return LoadTxtCore(stream, cfg, unpack);
        }

        #endregion

        #region config (the argument checks of _read and _load_from_filelike, in NumPy's order)

        /// <summary>The resolved arguments of one <c>np.loadtxt</c> call.</summary>
        private sealed class LoadTxtConfig
        {
            /// <summary>The result's descriptor.</summary>
            public DType Descr;
            /// <summary>The descriptor's storage lane.</summary>
            public NPTypeCode Tc;
            /// <summary>The delimiter character (<see cref="LoadTxtTokenizer.None"/>: whitespace).</summary>
            public int Delimiter = LoadTxtTokenizer.None;
            /// <summary>The quote character (<see cref="LoadTxtTokenizer.None"/>: no quoting).</summary>
            public int Quote = LoadTxtTokenizer.None;
            /// <summary>The single-character comment (<see cref="LoadTxtTokenizer.None"/>: none, or stripped per line).</summary>
            public int Comment = LoadTxtTokenizer.None;
            /// <summary>The comments stripped from each line before tokenizing (NumPy's <c>_preprocess_comments</c>), or null:
            ///     Python objects in order — a string cuts the line at its first occurrence, null (<c>None</c>) is
            ///     <c>str.split(None, 1)[0]</c>, any other object raises <c>str.split</c>'s TypeError at the first line.</summary>
            public object[] MultiComments;
            /// <summary>The delimiter as given (for the control-character checks' texts).</summary>
            public string DelimiterText;
            /// <summary>The quote character as given.</summary>
            public string QuoteText;
            /// <summary>Lines skipped first.</summary>
            public int SkipRows;
            /// <summary>The maximum number of data rows (-1: all).</summary>
            public int MaxRows;
            /// <summary>The selected columns, or null for all.</summary>
            public int[] UseCols;
            /// <summary>The minimum result rank.</summary>
            public int Ndmin;
            /// <summary>The converters as given (resolved at the first data row, as NumPy resolves them).</summary>
            public object Converters;
            /// <summary>NumPy's <c>encoding='bytes'</c> (<c>python_byte_converters</c>): a converter receives the field's
            ///     Latin-1 form.</summary>
            public bool ByteConverters;
        }

        /// <summary>
        ///     Validates loadtxt's Python-level arguments in NumPy's order (<c>loadtxt</c> and <c>_read</c>, before the input
        ///     is opened) and resolves them.
        /// </summary>
        /// <param name="descr">The requested dtype, or null for float64.</param>
        /// <param name="comments">A string, a sequence (of strings, or of any Python objects), or null.</param>
        /// <param name="delimiter">The delimiter, or null for whitespace.</param>
        /// <param name="converters">The converters as given.</param>
        /// <param name="usecols">The selected columns, or null.</param>
        /// <param name="ndmin">The minimum result rank.</param>
        /// <param name="max_rows">The maximum number of data rows, or null.</param>
        /// <param name="quotechar">The quote character, or null.</param>
        /// <param name="skiprows">Leading lines to skip.</param>
        /// <param name="encoding">The encoding as given; only <c>"bytes"</c> matters here (NumPy's <c>_read</c> turns it
        ///     into byte converters and reads with the default encoding).</param>
        /// <returns>The configuration (the control characters are checked later, by <see cref="CheckLoadTxtControlCharacters"/>,
        ///     as NumPy's C reader checks them after the input is opened).</returns>
        /// <exception cref="ValueError">An invalid <paramref name="ndmin"/>, an empty comment, multiple / multi-character
        ///     comments with a quote character, a negative <paramref name="skiprows"/> / <paramref name="max_rows"/>.</exception>
        /// <exception cref="TypeError">Comments that are not iterable, or several comments one of which is the
        ///     delimiter.</exception>
        private static LoadTxtConfig BuildLoadTxtConfig(DType descr, object comments, string delimiter,
            object converters, int[] usecols, int ndmin, int? max_rows, string quotechar, int skiprows, string encoding)
        {
            descr ??= DType.Double;
            var cfg = new LoadTxtConfig
            {
                Descr = descr,
                Tc = descr.GetTypeCode(),
                DelimiterText = delimiter,
                QuoteText = quotechar,
                SkipRows = skiprows,
                UseCols = usecols,
                Ndmin = ndmin,
                Converters = converters,
                ByteConverters = encoding == "bytes",
            };

            // loadtxt normalizes the comments before it calls _read, so a non-iterable argument is reported ahead of _read's
            // checks (comments=5 with ndmin=3 is "'int' object is not iterable", not the ndmin error).
            object[] list = ResolveLoadTxtComments(comments);

            // _ensure_ndmin_ndarray_check_param.
            if (ndmin != 0 && ndmin != 1 && ndmin != 2)
                throw new ValueError($"Illegal value of ndmin keyword: {ndmin}");

            // _read: an empty comment is refused ("" in comment — only a str equals ""), one one-character str goes to the C
            // tokenizer, anything else is stripped per line; several comments must not include the delimiter (Python's `in`
            // over the tuple: None is in it when an element is None), and stripped comments cannot meet quotes. Elements that
            // are neither str nor None are kept: NumPy's str.split raises for them at the first line it reads.
            if (list != null)
            {
                foreach (object c in list)
                    if (c is string { Length: 0 })
                        throw new ValueError("comments cannot be an empty string. Use comments=None to disable comments.");
                if (list.Length == 1 && list[0] is string { Length: 1 } one)
                    cfg.Comment = one[0];
                else if (list.Length > 0)
                {
                    if (list.Length > 1 && LoadTxtCommentsContain(list, delimiter))
                        throw new TypeError($"Comment characters '{PyTupleRepr(list)}' cannot include the delimiter '{delimiter ?? "None"}'");
                    cfg.MultiComments = list;
                }
                if (cfg.MultiComments != null && quotechar != null)
                    throw new ValueError("when multiple comments or a multi-character comment is given, quotes are not supported.  In this case quotechar must be set to None.");
            }

            // _check_nonneg_int(skiplines) / (max_rows).
            if (skiprows < 0)
                throw new ValueError("argument must be nonnegative");
            if (max_rows.HasValue && max_rows.Value < 0)
                throw new ValueError("argument must be nonnegative");
            cfg.MaxRows = max_rows ?? -1;
            return cfg;
        }

        /// <summary>
        ///     <c>loadtxt</c>'s comment normalization (<c>[x.decode('latin1') if isinstance(x, bytes) else x for x in
        ///     comment]</c> after a str / bytes became a one-element list): a string, a Latin-1 <c>byte[]</c> or a
        ///     <see cref="char"/> is a one-element list, a sequence is its elements with each <c>byte[]</c> decoded as Latin-1
        ///     and each <see cref="char"/> a one-character str, null is no comments.
        /// </summary>
        /// <param name="comments">The argument.</param>
        /// <returns>The comments as Python objects (a string, null for <c>None</c>, or any other object, which NumPy keeps and
        ///     whose use as a separator fails later), or null for no comments.</returns>
        /// <exception cref="TypeError">An argument that is not iterable (<c>'int' object is not iterable</c>, Python's text for
        ///     the comprehension over it).</exception>
        private static object[] ResolveLoadTxtComments(object comments)
        {
            switch (comments)
            {
                case null:
                    return null;
                case string s:
                    return new object[] { s };
                case char ch:
                    return new object[] { ch.ToString() };
                case byte[] b:
                    return new object[] { Encoding.Latin1.GetString(b) };
                case ITuple tuple:
                {
                    // A value tuple is a Python tuple, which is iterable (C#'s ValueTuple is not an IEnumerable).
                    var items = new object[tuple.Length];
                    for (int i = 0; i < items.Length; i++)
                        items[i] = AsLoadTxtComment(tuple[i]);
                    return items;
                }
                case IEnumerable seq:
                {
                    var list = new List<object>();
                    foreach (object x in seq)
                        list.Add(AsLoadTxtComment(x));
                    return list.ToArray();
                }
                default:
                    throw new TypeError($"'{NDPyObject.TypeName(comments)}' object is not iterable");
            }
        }

        /// <summary>One element of a comment sequence after <c>loadtxt</c>'s comprehension: a <c>byte[]</c> decoded as
        ///     Latin-1, a <see cref="char"/> as its one-character str, anything else as it is.</summary>
        /// <param name="x">The element.</param>
        /// <returns>The comment as a Python object.</returns>
        private static object AsLoadTxtComment(object x) => x switch
        {
            byte[] xb => Encoding.Latin1.GetString(xb),
            char xc => xc.ToString(),
            _ => x,
        };

        /// <summary>
        ///     Python's <c>delimiter in comments</c> over the comment tuple: an element that IS the delimiter or equals it —
        ///     None is in the tuple when an element is None (whitespace splitting has the delimiter None), a str delimiter when
        ///     an equal str is (an object of another type never equals a str).
        /// </summary>
        /// <param name="comments">The comments as Python objects.</param>
        /// <param name="delimiter">The delimiter, or null for None.</param>
        /// <returns>True when the delimiter is one of the comments.</returns>
        private static bool LoadTxtCommentsContain(object[] comments, string delimiter)
        {
            foreach (object c in comments)
                if (delimiter == null ? c == null : c is string s && s == delimiter)
                    return true;
            return false;
        }

        /// <summary>Python's <c>repr()</c> of the comment tuple (<c>('#', None)</c>), for NumPy's comment / delimiter text.</summary>
        /// <param name="items">The comments as Python objects.</param>
        /// <returns>The repr.</returns>
        private static string PyTupleRepr(object[] items)
        {
            var sb = new StringBuilder("(");
            for (int i = 0; i < items.Length; i++)
            {
                if (i > 0)
                    sb.Append(", ");
                sb.Append(NDPyObject.Repr(items[i]));
            }
            if (items.Length == 1)
                sb.Append(',');
            return sb.Append(')').ToString();
        }

        /// <summary>
        ///     The C reader's control-character checks (<c>parse_control_character</c> for the delimiter, the comment and the
        ///     quote, then <c>error_if_matching_control_characters</c>), raised after the input is opened, as NumPy raises them.
        /// </summary>
        /// <param name="cfg">The configuration; its <see cref="LoadTxtConfig.Delimiter"/> and <see cref="LoadTxtConfig.Quote"/>
        ///     are set here.</param>
        /// <exception cref="TypeError">A delimiter or quote that is not exactly one character (<c>Text reading control
        ///     character must be a single unicode character or None; but got: ', '</c>), a newline control character, or two
        ///     control characters that are the same (or a whitespace comment / quote with whitespace splitting).</exception>
        private static void CheckLoadTxtControlCharacters(LoadTxtConfig cfg)
        {
            cfg.Delimiter = ParseLoadTxtControlCharacter(cfg.DelimiterText);
            cfg.Quote = ParseLoadTxtControlCharacter(cfg.QuoteText);
            int delimiter = cfg.Delimiter, quote = cfg.Quote, comment = cfg.Comment;
            const int none = LoadTxtTokenizer.None;

            string first = null, second = null;
            if (comment != none)
            {
                if (comment == '\r' || comment == '\n')
                    first = "comment";
                else if (comment == quote)
                    (first, second) = ("comment", "quotechar");
                else if (comment == delimiter)
                    (first, second) = ("comment", "delimiter");
            }
            if (first == null && quote != none)
            {
                if (quote == '\r' || quote == '\n')
                    first = "quotechar";
                else if (quote == delimiter)
                    (first, second) = ("quotechar", "delimiter");
            }
            if (first == null && delimiter != none && (delimiter == '\r' || delimiter == '\n'))
                first = "delimiter";
            // The above doesn't work with delimiter=None, which means "whitespace".
            if (first == null && delimiter == none)
            {
                if (comment != none && LoadTxtTokenizer.IsSpace((char)comment))
                    (first, second) = ("delimiter", "comment");
                else if (quote != none && LoadTxtTokenizer.IsSpace((char)quote))
                    (first, second) = ("delimiter", "quotechar");
            }
            if (first == null)
                return;
            if (second != null)
                throw new TypeError($"The values for control characters '{first}' and '{second}' are incompatible");
            throw new TypeError($"control character '{first}' cannot be a newline (`\\r` or `\\n`).");
        }

        /// <summary><c>parse_control_character</c>: null is "not set", one character is itself.</summary>
        /// <param name="s">The control character as given.</param>
        /// <returns>The character, or <see cref="LoadTxtTokenizer.None"/>.</returns>
        /// <exception cref="TypeError">Not exactly one character (NumPy's text with the value's repr, cut to 100).</exception>
        private static int ParseLoadTxtControlCharacter(string s)
        {
            if (s is null)
                return LoadTxtTokenizer.None;
            if (s.Length != 1)
                throw new TypeError("Text reading control character must be a single unicode character or None; but got: " + Repr100(s));
            return s[0];
        }

        /// <summary>Python's <c>%.100R</c>: the repr of a text, cut to 100 code points (NumPy's error texts).</summary>
        /// <param name="s">The text.</param>
        /// <returns>The cut repr.</returns>
        private static string Repr100(string s)
        {
            string repr = Backends.Printing.PyRepr.Str(s);
            // Cut at 100 CODE POINTS (a surrogate pair is one), as PyUnicode_FromFormat's precision counts characters.
            int units = 0;
            for (int kept = 0; kept < 100 && units < repr.Length; kept++)
                units += char.IsHighSurrogate(repr[units]) && units + 1 < repr.Length && char.IsLowSurrogate(repr[units + 1]) ? 2 : 1;
            return units >= repr.Length ? repr : repr.Substring(0, units);
        }

        #endregion

        #region read_rows

        /// <summary>
        ///     NumPy's <c>read_rows</c> (<c>textreading/rows.c</c>): skips the leading lines, tokenizes row by row, resolves
        ///     the converters at the first data row, converts every selected field in order through the dtype's writer, and
        ///     finishes the 2-D result; then <c>_ensure_ndmin_ndarray</c> and <c>unpack</c>.
        /// </summary>
        /// <param name="stream">The input.</param>
        /// <param name="cfg">The configuration.</param>
        /// <param name="unpack">Transpose the result.</param>
        /// <returns>The result.</returns>
        /// <exception cref="ValueError">See <see cref="loadtxt(string,DType,object,string,object,int,int[],bool,int,string,int?,string)"/>.</exception>
        /// <exception cref="TypeError">See the same.</exception>
        /// <remarks>Scoped: the 2-D <c>read_rows</c> result is an intermediate whenever <c>ndmin</c> squeezes or
        ///     <paramref name="unpack"/> transposes it (the returned view keeps its buffer alive), and a converter's
        ///     transient arrays are reclaimed with it.</remarks>
        [NDScoped]
        private static NDArray LoadTxtCore(LoadTxtStream stream, LoadTxtConfig cfg, bool unpack)
        {
            // An unsized string dtype ('U', 'S') is _read's chunked object read: `while max_rows != 0` reads chunks with the C
            // reader and casts each. With max_rows == 0 the loop never runs, so the C reader checks nothing and _read
            // concatenates an empty chunk list.
            if (cfg.MaxRows == 0 && cfg.Descr.Meta is StringDTypeMeta && cfg.Descr.itemsize == 0)
                throw new ValueError("need at least one array to concatenate");
            CheckLoadTxtControlCharacters(cfg);
            var tokenizer = new LoadTxtTokenizer(cfg.Delimiter, cfg.Quote, cfg.Comment);
            using LoadTxtSink sink = LoadTxtSink.Create(cfg.Descr, cfg.ByteConverters);

            // Fewer lines than skiplines is acceptable; a file that ends while skipping reads no rows.
            int tsResult = 0;
            for (int i = 0; i < cfg.SkipRows; i++)
            {
                tsResult = tokenizer.SkipLine(stream);
                if (tsResult != 0)
                    break;
            }

            int[] usecols = cfg.UseCols;
            int actualNumFields = usecols != null ? usecols.Length : -1;
            Func<string, object>[] convFuncs = null;
            bool started = false;
            long rowCount = 0;
            while ((cfg.MaxRows < 0 || rowCount < cfg.MaxRows) && tsResult == 0)
            {
                tsResult = tokenizer.Tokenize(stream);
                int currentNumFields = tokenizer.NumFields;
                if (currentNumFields == 0)
                    continue; // ignore an empty line (also not counted toward max_rows)

                if (!started)
                {
                    // Deferred to the first row: now the number of columns is known.
                    if (actualNumFields == -1)
                        actualNumFields = currentNumFields;
                    if (cfg.Converters != null)
                        convFuncs = CreateLoadTxtConverters(cfg.Converters, actualNumFields, usecols);
                    started = true;
                    // A source that bounds its remaining lines (an array or a list) bounds the rows: a sink of fixed-size
                    // elements then allocates the result once and fills it in place, where NumPy grows its array in blocks
                    // and shrinks it at the end (the values are the same either way).
                    long moreLines = stream.RemainingLinesHint;
                    if (moreLines >= 0 && actualNumFields > 0)
                    {
                        long bound = 1 + moreLines;
                        if (cfg.MaxRows >= 0 && cfg.MaxRows < bound)
                            bound = cfg.MaxRows;
                        sink.Reserve(bound, actualNumFields);
                    }
                }

                if (usecols == null && actualNumFields != currentNumFields)
                    throw new ValueError(
                        $"the number of columns changed from {actualNumFields} to {currentNumFields} at row {rowCount + 1}; " +
                        "use `usecols` to select a subset and avoid this error");

                if (usecols == null && convFuncs == null)
                {
                    // Every field of the row, in order, with no converter: the sink runs the row's loop itself (one call per
                    // row, the dtype's conversion inline), and reports the first field it could not store.
                    int stored = sink.StoreRow(tokenizer, actualNumFields, out Exception cause);
                    if (stored < actualNumFields)
                        throw LoadTxtConvertError(tokenizer.Field(stored), sink.DescrText, rowCount, stored, cause);
                    rowCount++;
                    continue;
                }

                for (int i = 0; i < actualNumFields; i++)
                {
                    int col = i;
                    if (usecols != null)
                    {
                        col = usecols[i];
                        // Python-like column indexing: k = -1 means the last column.
                        if (col < 0)
                            col += currentNumFields;
                        if (col < 0 || col >= currentNumFields)
                            throw new ValueError($"invalid column index {usecols[i]} at row {rowCount + 1} with {currentNumFields} columns");
                    }

                    // The main "conversion" step: parse the field's text (or run its converter) and store the result.
                    ReadOnlySpan<char> field = tokenizer.Field(col);
                    Func<string, object> conv = convFuncs?[i];
                    bool stored;
                    try
                    {
                        stored = sink.Store(field, conv);
                    }
                    catch (Exception e)
                    {
                        throw LoadTxtConvertError(field, sink.DescrText, rowCount, col, e);
                    }
                    if (!stored)
                        throw LoadTxtConvertError(field, sink.DescrText, rowCount, col, null);
                }
                rowCount++;
            }

            // No data row and no usecols: NumPy guesses one column ("we have no choice but to guess 1").
            NDArray arr = sink.Finish(rowCount, actualNumFields == -1 ? 1 : actualNumFields);
            NDArray shaped = EnsureNdmin(arr, cfg.Ndmin);
            return unpack ? shaped.T : shaped;
        }

        /// <summary>
        ///     NumPy's <c>create_conv_funcs</c>: the converter of each RESULT column. A single function serves every column;
        ///     a dictionary maps columns — without <paramref name="usecols"/> a key is a file column (negative from the end,
        ///     range-checked), with it a key must equal one of the <paramref name="usecols"/> entries AS GIVEN (the first one
        ///     that does; a key no entry equals is ignored, so <c>usecols=[-1]</c> with <c>{2: f}</c> applies nothing).
        /// </summary>
        /// <param name="converters">The converters as given.</param>
        /// <param name="numFields">The number of result columns.</param>
        /// <param name="usecols">The selected columns, or null.</param>
        /// <returns>One converter (or null) per result column.</returns>
        /// <exception cref="TypeError">Converters of another kind (<c>converters must be a dictionary mapping columns to
        ///     converter functions or a single callable.</c>), or a null converter in the dictionary.</exception>
        /// <exception cref="ValueError">A key outside the columns without <paramref name="usecols"/> (<c>converter specified
        ///     for column 5, which is invalid for the number of fields 2.</c>).</exception>
        private static Func<string, object>[] CreateLoadTxtConverters(object converters, int numFields, int[] usecols)
        {
            var conv = new Func<string, object>[numFields];
            if (converters is Func<string, object> all)
            {
                // A single converter used for all columns individually.
                for (int i = 0; i < numFields; i++)
                    conv[i] = all;
                return conv;
            }
            if (converters is not IDictionary<int, Func<string, object>> dict)
                throw new TypeError("converters must be a dictionary mapping columns to converter functions or a single callable.");

            foreach (var entry in dict)
            {
                int column = entry.Key;
                if (usecols != null)
                {
                    int i = 0;
                    for (; i < numFields; i++)
                    {
                        if (column == usecols[i])
                        {
                            column = i;
                            break;
                        }
                    }
                    if (i == numFields)
                        continue; // ignore an unused converter
                }
                else
                {
                    if (column < -numFields || column >= numFields)
                        throw new ValueError($"converter specified for column {column}, which is invalid for the number of fields {numFields}.");
                    if (column < 0)
                        column += numFields;
                }
                if (entry.Value is null)
                    throw new TypeError($"values of the converters dictionary must be callable, but the value associated with key {entry.Key} is not");
                conv[column] = entry.Value;
            }
            return conv;
        }

        /// <summary>
        ///     NumPy's conversion error (<c>rows.c</c>): <c>could not convert string %.100R to %S at row %zd, column %zd.</c> —
        ///     the field's repr cut to 100 characters, the dtype's <c>str()</c>, the 0-based data row and the 1-based column,
        ///     the original error as the cause.
        /// </summary>
        /// <param name="field">The field's text.</param>
        /// <param name="descrText">The dtype's <c>str()</c> (<c>float64</c>, <c>&lt;U5</c>, <c>StringDType()</c>).</param>
        /// <param name="row">The 0-based data row (blank and comment lines not counted).</param>
        /// <param name="col">The 0-based column in the file.</param>
        /// <param name="cause">The error the conversion raised, or null when the parser just refused the text.</param>
        /// <returns>The error to throw.</returns>
        private static ValueError LoadTxtConvertError(ReadOnlySpan<char> field, string descrText, long row, int col, Exception cause)
            => new ValueError($"could not convert string {Repr100(field.ToString())} to {descrText} at row {row}, column {col + 1}.", cause);

        /// <summary>
        ///     NumPy's <c>_ensure_ndmin_ndarray</c>: squeeze the size-1 axes of a result with more dimensions than
        ///     <paramref name="ndmin"/>, then expand one with fewer (<c>atleast_1d</c>; <c>atleast_2d(a).T</c>).
        /// </summary>
        /// <param name="a">The 2-D <c>read_rows</c> result.</param>
        /// <param name="ndmin">0, 1 or 2.</param>
        /// <returns><paramref name="a"/> ITSELF when nothing changes — NumPy's <c>PyArray_Squeeze</c> returns its
        ///     operand when no axis is 1, so a <c>(2, 3)</c> or <c>(0, 3)</c> result still owns its data — else a view.</returns>
        private static NDArray EnsureNdmin(NDArray a, int ndmin)
        {
            if (a.ndim > ndmin && Array.IndexOf(a.Shape.dimensions, 1L) >= 0)
                a = np.squeeze(a);
            if (a.ndim < ndmin)
            {
                if (ndmin == 1) a = np.atleast_1d(a);
                else if (ndmin == 2) a = np.atleast_2d(a).T;
            }
            return a;
        }

        #endregion
    }
}
