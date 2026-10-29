using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using NumSharp.Backends.Printing;

namespace NumSharp.IO
{
    /// <summary>
    ///     The body of <c>np.savetxt</c>: NumPy's <c>savetxt</c> (<c>numpy/lib/_npyio_impl.py</c>) - the target opened as
    ///     NumPy opens it (a file name created or truncated, then written through a TextIOWrapper; a byte stream through
    ///     <c>WriteWrap</c>; a text writer as it is), the rank check, the row format, the header, the rows and the footer.
    /// </summary>
    /// <remarks>
    ///     The <c>np.savetxt</c> overloads check their target and call <see cref="SaveTxtToFile"/>,
    ///     <see cref="SaveTxtToStream"/>, or <see cref="SaveTxtCore"/> with <see cref="SaveTxtWriter.ForText"/>.
    /// </remarks>
    internal static partial class SaveTxtIO
    {
        #region internals

        /// <summary>
        ///     Writes to a file NAME as NumPy does: <c>open(fname, 'wt').close()</c> creates or truncates it, then
        ///     <c>_datasource.open(fname, 'wt', encoding)</c> reopens it with the opener of its extension (gzip for exactly
        ///     <c>.gz</c>) inside a TextIOWrapper, which looks the encoding up, translates every <c>'\n'</c> to the platform
        ///     line separator and encodes each write. So a bad encoding, rank or format leaves an empty file behind.
        /// </summary>
        /// <param name="fname">The path.</param>
        /// <param name="X">The array.</param>
        /// <param name="fmt">A format string or one spec per column.</param>
        /// <param name="delimiter">The column separator.</param>
        /// <param name="newline">The row terminator.</param>
        /// <param name="header">The header text.</param>
        /// <param name="footer">The footer text.</param>
        /// <param name="comments">The header / footer line prefix.</param>
        /// <param name="encoding">The file's encoding (UTF-8 when null or <c>"locale"</c>: NumPy's locale encoding on Linux
        ///     and macOS).</param>
        /// <exception cref="ArgumentNullException"><paramref name="fname"/> is null.</exception>
        /// <exception cref="NotSupportedException">A <c>.bz2</c> / <c>.xz</c> / <c>.lzma</c> name (refused before the file is
        ///     touched), or a codec NumSharp does not implement.</exception>
        /// <exception cref="LookupError">An unknown encoding (<c>'bytes'</c> and the empty name included: <c>unknown
        ///     encoding: bytes</c>), or a codec that is not a text encoding.</exception>
        /// <exception cref="UnicodeEncodeError">A header, row or footer the encoding cannot encode (the rows before it are
        ///     written).</exception>
        internal static void SaveTxtToFile(string fname, NDArray X, object fmt, string delimiter,
            string newline, string header, string footer, string comments, string encoding)
        {
            if (fname is null) throw new ArgumentNullException(nameof(fname));
            // NumPy would compress these with bz2 / lzma; NumSharp cannot, and refuses before it truncates anything.
            PyDataSource.ThrowIfUnsupportedCompression(fname);

            bool gz = PyDataSource.IsGzip(fname);
            Stream fs = new FileStream(fname, FileMode.Create, FileAccess.Write);
            try
            {
                // _datasource.open(fname, 'wt', encoding): the codec lookup happens after the truncation above.
                TextIOCodec codec = TextIOCodec.Resolve(encoding, TextIOCodec.Utf8, CodecLookupContext.Open);
                Stream outer = gz ? new GZipStream(fs, CompressionMode.Compress) : fs;
                try
                {
                    var output = SaveTxtWriter.ForFile(outer, codec.NewFileEncoder());
                    try
                    {
                        SaveTxtCore(output, X, fmt, delimiter, newline, header, footer, comments);
                    }
                    finally
                    {
                        // What was written before an error stays in the file, as Python's file object flushes it on close.
                        output.Flush();
                    }
                }
                finally
                {
                    if (gz) outer.Dispose(); // finalize the gzip trailer before the FileStream closes
                }
            }
            finally
            {
                fs.Dispose();
            }
        }

        /// <summary>
        ///     Writes to a byte stream: NumPy's <c>WriteWrap(fh, encoding or 'latin1')</c>, which encodes every write on its
        ///     own with the codec (<c>str.encode</c>; Latin-1 when none is named) and leaves the stream open.
        /// </summary>
        /// <param name="stream">The stream.</param>
        /// <param name="X">The array.</param>
        /// <param name="fmt">A format string or one spec per column.</param>
        /// <param name="delimiter">The column separator.</param>
        /// <param name="newline">The row terminator.</param>
        /// <param name="header">The header text.</param>
        /// <param name="footer">The footer text.</param>
        /// <param name="comments">The header / footer line prefix.</param>
        /// <param name="encoding">The encoding, or null / empty for Latin-1 (Python's <c>encoding or 'latin1'</c>).</param>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        /// <exception cref="LookupError">An unknown or non-text encoding, raised at the FIRST write as WriteWrap raises it
        ///     (nothing to write — a <c>(0,)</c> array without header or footer — raises nothing).</exception>
        /// <exception cref="NotSupportedException">A codec NumSharp does not implement, at the first write.</exception>
        /// <exception cref="UnicodeEncodeError">A header, row or footer the codec cannot encode (the writes before it are
        ///     in the stream).</exception>
        internal static void SaveTxtToStream(Stream stream, NDArray X, object fmt, string delimiter,
            string newline, string header, string footer, string comments, string encoding)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));

            TextIOCodec codec = null;
            Exception deferred = null;
            try
            {
                codec = TextIOCodec.Resolve(string.IsNullOrEmpty(encoding) ? null : encoding, TextIOCodec.Latin1,
                    CodecLookupContext.Encode);
            }
            catch (Exception e) when (e is LookupError || e is NotSupportedException)
            {
                // WriteWrap looks the codec up by encoding: at the first write.
                deferred = e;
            }

            var output = SaveTxtWriter.ForStream(stream, codec, deferred);
            try
            {
                SaveTxtCore(output, X, fmt, delimiter, newline, header, footer, comments);
            }
            finally
            {
                output.Flush();
            }
        }

        /// <summary>
        ///     savetxt's output: batches the Python <c>fh.write</c> calls (the header, each row, the footer) into ~32 KB chunks
        ///     and checks every write against the target's codec when it is made, so an unencodable write raises with the
        ///     writes before it already delivered — Python's <c>str.encode</c> per <c>WriteWrap.write</c> /
        ///     <c>TextIOWrapper.write</c>.
        /// </summary>
        /// <remarks>
        ///     Three targets: a <see cref="TextWriter"/> (Python's <c>StringIO</c>: takes any str, no encoding), a byte
        ///     stream (<c>WriteWrap</c>: each write encoded on its own, so a BOM codec's writes each carry a BOM and are
        ///     encoded as they come), and a file NumPy opened itself (TextIOWrapper: <c>'\n'</c> translated to the platform
        ///     line separator, then encoded by the file's encoder, a BOM codec's BOM once at the start).
        /// </remarks>
        internal sealed class SaveTxtWriter
        {
            private readonly TextWriter _text;
            private readonly Stream _bytes;
            private readonly TextIOCodec _codec;
            private readonly TextIOFileEncoder _fileEncoder;
            private readonly Exception _deferred;
            private char[] _scratch = new char[256];
            private byte[] _encoded = Array.Empty<byte>();
            private bool _anyWrite;

            /// <summary>The pending writes, appended in place by the row formatter.</summary>
            internal readonly StringBuilder Chunk = new StringBuilder(1 << 16);

            private SaveTxtWriter(TextWriter text, Stream bytes, TextIOCodec codec, TextIOFileEncoder fileEncoder, Exception deferred)
            {
                _text = text;
                _bytes = bytes;
                _codec = codec;
                _fileEncoder = fileEncoder;
                _deferred = deferred;
            }

            /// <summary>A text target: Python's <c>StringIO.write</c> takes any str, so nothing is checked or encoded.</summary>
            /// <param name="writer">The target.</param>
            /// <returns>The output.</returns>
            internal static SaveTxtWriter ForText(TextWriter writer) => new SaveTxtWriter(writer, null, null, null, null);

            /// <summary>A byte stream (<c>WriteWrap</c>): every write is checked and encoded on its own with the codec.</summary>
            /// <param name="stream">The target.</param>
            /// <param name="codec">The codec, or null when its lookup failed.</param>
            /// <param name="deferred">With a null <paramref name="codec"/>: the lookup's error, which the first write raises.</param>
            /// <returns>The output.</returns>
            internal static SaveTxtWriter ForStream(Stream stream, TextIOCodec codec, Exception deferred)
                => new SaveTxtWriter(null, stream, codec, null, deferred);

            /// <summary>A file NumPy opened itself (TextIOWrapper): writes are newline-translated, then encoded by its
            ///     encoder.</summary>
            /// <param name="stream">The file's byte stream (gzip-compressing for a <c>.gz</c> name).</param>
            /// <param name="encoder">The file's encoder.</param>
            /// <returns>The output.</returns>
            internal static SaveTxtWriter ForFile(Stream stream, TextIOFileEncoder encoder)
                => new SaveTxtWriter(null, stream, null, encoder, null);

            /// <summary>One whole write (the header or the footer).</summary>
            /// <param name="text">The text.</param>
            /// <exception cref="LookupError">The deferred codec lookup failure.</exception>
            /// <exception cref="NotSupportedException">The deferred refusal of a codec NumSharp does not implement.</exception>
            /// <exception cref="UnicodeEncodeError">The codec cannot encode the text.</exception>
            internal void Write(string text)
            {
                int start = Chunk.Length;
                Chunk.Append(text);
                Commit(start);
            }

            /// <summary>
            ///     Accepts the write that occupies <see cref="Chunk"/> from <paramref name="start"/> to its end: checks it
            ///     against the codec (positions in CPython's error count from the write's start), then flushes once the
            ///     batch is large — or, for a byte stream with a BOM codec, encodes the write right away (its own BOM first).
            /// </summary>
            /// <param name="start">Where the write begins in <see cref="Chunk"/>.</param>
            /// <exception cref="LookupError">The deferred codec lookup failure (the write is dropped).</exception>
            /// <exception cref="NotSupportedException">The deferred refusal of the codec (the write is dropped).</exception>
            /// <exception cref="UnicodeEncodeError">The codec cannot encode the write (it is dropped; the writes before it
            ///     are flushed first, as Python delivered them).</exception>
            /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
            internal void Commit(int start)
            {
                if (_deferred != null)
                {
                    Discard(start);
                    throw _deferred;
                }
                if (_bytes != null)
                {
                    int length = Chunk.Length - start;
                    if (_scratch.Length < length)
                        _scratch = new char[Math.Max(length, _scratch.Length * 2)];
                    var span = _scratch.AsSpan(0, length);
                    Chunk.CopyTo(start, span, length);
                    try
                    {
                        if (_fileEncoder != null)
                            CheckFileWrite(span);
                        else
                            _codec.CheckEncode(span);
                    }
                    catch
                    {
                        Discard(start);
                        throw;
                    }
                    _anyWrite = true;
                    if (_fileEncoder == null && _codec.WritesBomPerEncode)
                    {
                        // WriteWrap encodes each write on its own: a BOM codec's BOM leads every one of them, so this
                        // write is encoded now (the batch before it is always empty for such a codec).
                        WriteEncoded(span, bom: true);
                        Chunk.Length = start;
                        return;
                    }
                }
                if (Chunk.Length >= (1 << 15))
                    Flush();
            }

            /// <summary>
            ///     TextIOWrapper's check of one write to a file: CPython translates <c>'\n'</c> to the platform line separator
            ///     BEFORE it encodes, so an error's position counts the translated write. The untranslated write fails
            ///     exactly when the translated one does (the inserted <c>'\r'</c> is encodable by every codec), so it is
            ///     checked first and only a failing write is translated, to report the translated position.
            /// </summary>
            /// <param name="span">The write, untranslated.</param>
            /// <exception cref="UnicodeEncodeError">The write cannot be encoded.</exception>
            /// <exception cref="UnicodeError">The <c>undefined</c> codec.</exception>
            private void CheckFileWrite(ReadOnlySpan<char> span)
            {
                bool first = _fileEncoder.StartOfStream;
                try
                {
                    _fileEncoder.CheckWrite(span);
                }
                catch (UnicodeEncodeError) when (Environment.NewLine != "\n" && span.IndexOf('\n') >= 0)
                {
                    // Raises again, with the position the translated write gives.
                    _fileEncoder.Check(span.ToString().Replace("\n", Environment.NewLine), first);
                    throw;
                }
            }

            /// <summary>Drops a write that failed (from <paramref name="start"/>) and delivers the writes before it.</summary>
            /// <param name="start">Where the failed write begins.</param>
            internal void Discard(int start)
            {
                Chunk.Length = start;
                Flush();
            }

            /// <summary>
            ///     Delivers the batch to the target: as it is to a text target, encoded to a byte stream, translated to the
            ///     platform newline and encoded to a file (whose first output carries a BOM codec's BOM — even when the only
            ///     write so far was empty, as TextIOWrapper encodes an empty first write to the BOM alone).
            /// </summary>
            internal void Flush()
            {
                if (_text != null)
                {
                    if (Chunk.Length == 0)
                        return;
                    _text.Write(Chunk.ToString());
                    Chunk.Clear();
                    return;
                }
                if (Chunk.Length == 0)
                {
                    if (_fileEncoder != null && _anyWrite && _fileEncoder.BomPending)
                        WriteEncoded(ReadOnlySpan<char>.Empty, bom: true);
                    return;
                }
                string s = Chunk.ToString();
                // Python opens a filename target in text mode ('wt'), translating every '\n' to os.linesep on write (CRLF on
                // Windows). A stream target is written verbatim.
                if (_fileEncoder != null && Environment.NewLine != "\n")
                    s = s.Replace("\n", Environment.NewLine);
                WriteEncoded(s, bom: false);
                Chunk.Clear();
            }

            /// <summary>Encodes checked text with the target's codec and writes the bytes.</summary>
            /// <param name="text">The text (every write in it checked).</param>
            /// <param name="bom">For a byte stream: true to start with the codec's BOM (a BOM codec's every write). A file's
            ///     encoder decides its BOM itself.</param>
            private void WriteEncoded(ReadOnlySpan<char> text, bool bom)
            {
                TextIOCodec codec = _codec;
                int capacity = (codec ?? TextIOCodec.Utf8).MaxByteCount(text.Length) + 4;
                if (_fileEncoder != null)
                    capacity = _fileEncoder.MaxByteCount(text.Length);
                if (_encoded.Length < capacity)
                    _encoded = new byte[Math.Max(capacity, _encoded.Length * 2)];
                int n = _fileEncoder != null ? _fileEncoder.Encode(text, _encoded) : codec.Encode(text, _encoded, bom);
                _bytes.Write(_encoded, 0, n);
            }
        }

        /// <summary>
        ///     The port of savetxt's body: validate the rank, resolve the row <c>format</c>, then write the header, one line per
        ///     row (<c>format % tuple(row) + newline</c>) and the footer. Any layout is read in logical C order through
        ///     <see cref="NDArray.GetAtIndex(long)"/>.
        /// </summary>
        /// <param name="output">The batching, codec-checking writer.</param>
        /// <param name="X">The array.</param>
        /// <param name="fmt">A format string or one spec per column; null is Python's None.</param>
        /// <param name="delimiter">The column separator; null is Python's None.</param>
        /// <param name="newline">The row terminator; null is Python's None.</param>
        /// <param name="header">The header text; null is Python's None.</param>
        /// <param name="footer">The footer text; null is Python's None.</param>
        /// <param name="comments">The header / footer line prefix; null is Python's None.</param>
        /// <exception cref="ArgumentNullException"><paramref name="X"/> is null.</exception>
        /// <exception cref="ValueError">A 0-d or more than 2-d array, a format with the wrong number of specs, a NaN under
        ///     an integer spec, a null <paramref name="fmt"/> (<c>invalid fmt: None</c>).</exception>
        /// <exception cref="AttributeError">A spec list of the wrong length; a null <paramref name="delimiter"/> that NumPy
        ///     joins the specs with.</exception>
        /// <exception cref="TypeError">An element its spec cannot convert, or a null <paramref name="newline"/> once a row is
        ///     built (NumPy's <c>Mismatch between array dtype (…) and format specifier (…)</c>, the TypeError as the cause;
        ///     a complex array's row raises the plain concatenation TypeError); a null spec-list entry; a null
        ///     <paramref name="header"/> / <paramref name="footer"/> (<c>object of type 'NoneType' has no len()</c>) or a
        ///     null <paramref name="comments"/> / <paramref name="newline"/> while a non-empty header or footer is built
        ///     (<c>can only concatenate str (not "NoneType") to str</c>).</exception>
        /// <remarks>
        ///     <para>
        ///     A <c>StringDType</c> array's elements are Python objects (<c>stringdtype_getitem</c>: a plain <c>str</c>, or
        ///     the descriptor's <c>na_object</c> for an NA element), so its rows are formatted with Python's rules for those
        ///     objects (<c>'%s'</c> of NA <c>None</c> is <c>None</c>, <c>'%r'</c> of <c>'ab'</c> is <c>'ab'</c>, not
        ///     <c>np.str_('ab')</c>); every other dtype's elements are NumPy scalars.
        ///     </para>
        ///     <para>
        ///     An error in a row leaves the header and the rows before it written and nothing of that row (Python builds
        ///     the row's string before it writes it); the footer is not written.
        ///     </para>
        /// </remarks>
        internal static void SaveTxtCore(SaveTxtWriter output, NDArray X, object fmt, string delimiter,
            string newline, string header, string footer, string comments)
        {
            if (X is null) throw new ArgumentNullException(nameof(X));

            int ndim = X.ndim;
            if (ndim == 0 || ndim > 2)
                throw new ValueError($"Expected 1D or 2D array, got {ndim}D array instead");

            long nrows, ncol;
            if (ndim == 1) { nrows = X.Shape.dimensions[0]; ncol = 1; }
            else { nrows = X.Shape.dimensions[0]; ncol = X.Shape.dimensions[1]; }

            NPTypeCode tc = X.typecode;
            // NumPy: iscomplex_x = np.iscomplexobj(X), the descriptor's kind.
            bool iscomplex = X.dtype.kind == 'c';
            // A row's elements are what the class's getitem returns: Python objects for a dtype whose items hold references
            // (StringDType's str / NA object; NumPy's REFCHK, HasReferences), NumPy scalars for every other.
            bool pyObjects = X.dtype.Meta.HasReferences;

            string format = ResolveSaveTxtFormat(fmt, ncol, iscomplex, delimiter);

            // NumPy: `if len(header) > 0:` - a None header has no len(). It writes the header immediately (before the row
            // loop), so a mid-row error leaves it written; a None comments / newline fails while the header text is built.
            if (header is null)
                throw new TypeError("object of type 'NoneType' has no len()");
            if (header.Length > 0)
                output.Write(SaveTxtCommentBlock(header, comments, newline));

            // The row format is parsed once and run for every row. CPython parses it again for each row, but a parse has no
            // effect of its own: the text and the errors (in their order) are the same.
            PercentFormat program = PrintfFormatter.Compile(format);
            StringBuilder chunk = output.Chunk;
            if (iscomplex)
            {
                var args = new object[checked(2 * ncol)];
                var kinds = new NPTypeCode[args.Length];
                // number.real / number.imag of an np.complex128 are np.float64 scalars: '%r' prints np.float64(1.0), and a
                // TypeError names numpy.float64.
                for (int k = 0; k < kinds.Length; k++) kinds[k] = NPTypeCode.Double;
                var row = new StringBuilder();

                for (long r = 0; r < nrows; r++)
                {
                    long baseIdx = r * ncol;
                    for (long c = 0; c < ncol; c++)
                    {
                        var z = (System.Numerics.Complex)X.GetAtIndex(baseIdx + c);
                        args[2 * c] = z.Real;
                        args[2 * c + 1] = z.Imaginary;
                    }

                    // NumPy: s = format % tuple(row2) + newline; fh.write(s.replace('+-', '-')). The row string is built
                    // whole before it is written, so an error writes nothing of it.
                    string s;
                    try
                    {
                        row.Clear();
                        program.FormatInto(row, args, kinds);
                        // `... + newline` with a None newline: CPython's TypeError, which this branch does not wrap (the
                        // generic catch below delivers the rows before it).
                        if (newline is null)
                            throw new TypeError("can only concatenate str (not \"NoneType\") to str");
                        s = row.Append(newline).ToString().Replace("+-", "-");
                    }
                    catch (PrintfArgumentException ex)
                    {
                        // NumPy's complex branch does not wrap the TypeError of `format % tuple(row2)`: it surfaces as the
                        // plain TypeError (the formatter's internal subclass only marks the errors the real branch
                        // rewrites), after the rows before it reached the target.
                        output.Flush();
                        throw new TypeError(ex.Message, ex);
                    }
                    catch
                    {
                        // Any other error (a NaN part under %d) propagates as it is, after the rows before it.
                        output.Flush();
                        throw;
                    }
                    output.Write(s);
                }
            }
            else
            {
                var args = new object[ncol];
                var kinds = new NPTypeCode[ncol];
                // A StringDType element is the Python object getitem returns (a str, or the NA object); every other
                // dtype's element is a NumPy scalar of that dtype.
                NPTypeCode kind = pyObjects ? PercentFormat.PyObject : tc;
                for (int k = 0; k < kinds.Length; k++) kinds[k] = kind;

                for (long r = 0; r < nrows; r++)
                {
                    long baseIdx = r * ncol;
                    for (long c = 0; c < ncol; c++)
                        args[c] = X.GetAtIndex(baseIdx + c);

                    int start = chunk.Length;
                    try
                    {
                        // Append the row straight into the batch buffer — no per-row string allocation.
                        program.FormatInto(chunk, args, kinds);
                        // NumPy builds `v = format % tuple(row) + newline` inside its try: a None newline's TypeError is
                        // caught and re-raised as the Mismatch below, like a formatter's.
                        if (newline is null)
                            throw new PrintfArgumentException("can only concatenate str (not \"NoneType\") to str");
                    }
                    catch (PrintfArgumentException ex)
                    {
                        // NumPy catches the TypeError from `format % tuple(row) + newline` and re-raises this.
                        // str(X.dtype): the type name for a numeric array, the descriptor ('<U2', '|S3', 'StringDType()')
                        // for a string one.
                        output.Discard(start);
                        string dtypeText = tc == NPTypeCode.String ? X.dtype.ToString() : tc.AsNumpyDtypeName();
                        throw new TypeError(
                            $"Mismatch between array dtype ('{dtypeText}') and format specifier ('{format}')", ex);
                    }
                    catch
                    {
                        // Any other error of `format % tuple(row)` (a NaN under %d) propagates as it is; the partial row
                        // is dropped and the rows before it are delivered.
                        output.Discard(start);
                        throw;
                    }

                    chunk.Append(newline);
                    output.Commit(start);
                }
            }

            // NumPy: `if len(footer) > 0:` (a None footer has no len()), then the footer text, whose None comments / newline
            // fail while it is built. Every row was written before any of these errors, so the batch is delivered first.
            string footerText;
            try
            {
                if (footer is null)
                    throw new TypeError("object of type 'NoneType' has no len()");
                footerText = footer.Length > 0 ? SaveTxtCommentBlock(footer, comments, newline) : null;
            }
            catch
            {
                output.Flush();
                throw;
            }
            if (footerText != null)
                output.Write(footerText);
            output.Flush();
        }

        /// <summary>Builds the per-row <c>format</c> string exactly as savetxt does, with NumPy's verbatim errors.</summary>
        /// <param name="fmt">A <see cref="string"/> or a <c>string[]</c>; null is Python's None.</param>
        /// <param name="ncol">The number of columns.</param>
        /// <param name="iscomplex">A complex array (a single spec becomes <c>' (spec+specj)'</c> per column).</param>
        /// <param name="delimiter">The column separator; null is Python's None, refused only where NumPy joins with it.</param>
        /// <returns>The row format.</returns>
        /// <exception cref="AttributeError">A spec list whose length is not <paramref name="ncol"/>; a null
        ///     <paramref name="delimiter"/> that NumPy joins the specs with (<c>'NoneType' object has no attribute
        ///     'join'</c>: a spec list, or one spec repeated per column - a multi-<c>%</c> format never reads it).</exception>
        /// <exception cref="TypeError">A null entry of a spec list (<c>str.join</c>'s <c>sequence item 1: expected str
        ///     instance, NoneType found</c>).</exception>
        /// <exception cref="ValueError">A multi-spec string with the wrong number of <c>%</c>, or a null
        ///     <paramref name="fmt"/> (<c>invalid fmt: None</c>).</exception>
        private static string ResolveSaveTxtFormat(object fmt, long ncol, bool iscomplex, string delimiter)
        {
            if (fmt is string[] list)
            {
                if (list.Length != ncol)
                    throw new AttributeError($"fmt has wrong shape.  {PyListRepr(list)}");
                return PyJoin(delimiter, list);
            }

            if (fmt is string s)
            {
                int nfmt = CountPercent(s);
                if (nfmt == 1)
                {
                    // A single spec is replicated once per column (delimiter-joined); complex wraps it as
                    // ' (spec+specj)'. ncol == 0 (a (n, 0) array) yields an empty format.
                    var parts = new string[ncol];
                    string unit = iscomplex ? $" ({s}+{s}j)" : s;
                    for (long i = 0; i < ncol; i++) parts[i] = unit;
                    return PyJoin(delimiter, parts);
                }

                if (iscomplex && nfmt != 2 * ncol)
                    throw new ValueError($"fmt has wrong number of % formats:  {s}");
                if (!iscomplex && nfmt != ncol)
                    throw new ValueError($"fmt has wrong number of % formats:  {s}");

                return s; // a multi-% format string used directly (delimiter ignored)
            }

            // NumPy: raise ValueError(f'invalid fmt: {fmt!r}'). The typed overloads leave only None to reach here.
            throw new ValueError($"invalid fmt: {(fmt is null ? "None" : fmt.ToString())}");
        }

        /// <summary>
        ///     Python's <c>delimiter.join(items)</c>: a None separator has no <c>join</c> (AttributeError, raised before any
        ///     item is looked at), and a None item is refused by <c>str.join</c> (TypeError naming its index).
        /// </summary>
        /// <param name="delimiter">The separator; null is Python's None.</param>
        /// <param name="items">The items; a null one is Python's None.</param>
        /// <returns>The joined text.</returns>
        /// <exception cref="AttributeError">A null <paramref name="delimiter"/>.</exception>
        /// <exception cref="TypeError">A null item.</exception>
        private static string PyJoin(string delimiter, string[] items)
        {
            if (delimiter is null)
                throw new AttributeError("'NoneType' object has no attribute 'join'");
            for (int i = 0; i < items.Length; i++)
            {
                if (items[i] is null)
                    throw new TypeError($"sequence item {i}: expected str instance, NoneType found");
            }
            return string.Join(delimiter, items);
        }

        /// <summary>
        ///     Python's <c>left + right</c> for two str values where the right one may be None: CPython's TypeError for a None
        ///     right operand. Every concatenation of savetxt has a str on its left by the time it runs (a None
        ///     <c>comments</c> already failed in <c>'\n' + comments</c>, which NumPy evaluates first).
        /// </summary>
        /// <param name="left">The left operand (never null at its call sites).</param>
        /// <param name="right">The right operand; null is Python's None.</param>
        /// <returns>The concatenation.</returns>
        /// <exception cref="TypeError">A null <paramref name="right"/> (<c>can only concatenate str (not "NoneType") to
        ///     str</c>).</exception>
        private static string PyConcat(string left, string right)
            => right is null ? throw new TypeError("can only concatenate str (not \"NoneType\") to str") : left + right;

        /// <summary>
        ///     The text NumPy writes for a header or footer: <c>text.replace('\n', '\n' + comments)</c>, then
        ///     <c>comments + text + newline</c>, evaluated in that order - so a None <paramref name="comments"/> fails in the
        ///     first concatenation, a None <paramref name="newline"/> in the last.
        /// </summary>
        /// <param name="text">The header or footer, non-empty.</param>
        /// <param name="comments">The line prefix; null is Python's None.</param>
        /// <param name="newline">The line terminator; null is Python's None.</param>
        /// <returns>The text to write.</returns>
        /// <exception cref="TypeError">A null <paramref name="comments"/> or <paramref name="newline"/>.</exception>
        private static string SaveTxtCommentBlock(string text, string comments, string newline)
            => PyConcat(PyConcat(comments, text.Replace("\n", PyConcat("\n", comments))), newline);

        /// <summary>Python's <c>fmt.count('%')</c>.</summary>
        /// <param name="s">The format.</param>
        /// <returns>The number of <c>%</c> characters.</returns>
        private static int CountPercent(string s)
        {
            int n = 0;
            for (int i = 0; i < s.Length; i++)
                if (s[i] == '%') n++;
            return n;
        }

        /// <summary>Python's <c>str(list)</c> of strings (<c>['%d', '%.2f']</c>), for the AttributeError's text.</summary>
        /// <param name="list">The specs.</param>
        /// <returns>The repr.</returns>
        private static string PyListRepr(string[] list)
        {
            var sb = new StringBuilder("[");
            for (int i = 0; i < list.Length; i++)
            {
                if (i > 0) sb.Append(", ");
                sb.Append('\'').Append(list[i]).Append('\'');
            }
            return sb.Append(']').ToString();
        }

        #endregion
    }
}
