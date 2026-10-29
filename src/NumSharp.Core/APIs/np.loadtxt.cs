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
        // NumPy's text, raised at the point NumPy raises it. The overloads here convert their C# arguments and open the input;
        // _read's checks, read_rows, the converters and the sinks are the body, IO/TextReading/LoadTxtReader*.cs.

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
            var cfg = LoadTxtReader.BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
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
            return LoadTxtReader.LoadTxtCore(stream, cfg, unpack);
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
            var cfg = LoadTxtReader.BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
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
            return LoadTxtReader.LoadTxtCore(lines, cfg, unpack);
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
            var cfg = LoadTxtReader.BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
            using var lines = new LoadTxtLineStream(new LoadTxtTextLineReader(reader), cfg.MultiComments);
            return LoadTxtReader.LoadTxtCore(lines, cfg, unpack);
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
            var cfg = LoadTxtReader.BuildLoadTxtConfig(dtype, comments, delimiter, converters, usecols, ndmin, max_rows, quotechar, skiprows, encoding);
            using var stream = new LoadTxtLineStream(new LoadTxtSequenceLineReader(lines), cfg.MultiComments);
            return LoadTxtReader.LoadTxtCore(stream, cfg, unpack);
        }

        #endregion
    }
}
