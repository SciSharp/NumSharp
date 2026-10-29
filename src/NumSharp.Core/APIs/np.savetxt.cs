using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using NumSharp.Backends.Printing;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy 2.4.2 reference: numpy/lib/_npyio_impl.py::savetxt.
        //   savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='',
        //           footer='', comments='# ', encoding=None)
        //
        // A 1-D array is written one value per line (NumPy's `X = atleast_2d(X).T`, ncol == 1); a 2-D
        // array is written row per line with `ncol == X.shape[1]` columns. Each row is rendered as
        // `format % tuple(row)` where `format` is either the single spec joined by `delimiter` `ncol`
        // times, a caller-supplied multi-% string (delimiter then ignored), or the list/tuple of specs
        // joined by `delimiter`. The `%`-engine is PrintfFormatter (shared with ndarray.tofile).

        #region savetxt (filename)

        /// <summary>
        ///     Save a 1-D or 2-D array to a text file.
        /// </summary>
        /// <param name="fname">
        ///     Target path. It is created (or truncated) first, as NumPy's <c>open(fname, 'wt').close()</c>; a name whose
        ///     extension is exactly <c>.gz</c> is then written gzip-compressed, as NumPy does (<c>.GZ</c> is plain text, as
        ///     in NumPy). NumPy writes <c>.bz2</c> / <c>.xz</c> / <c>.lzma</c> names compressed too; .NET has no compressor
        ///     for them, so NumSharp refuses those names before touching the file.
        /// </param>
        /// <param name="X">The 1-D or 2-D array to save (a 0-D or ≥3-D array raises <see cref="ValueError"/>).</param>
        /// <param name="fmt">
        ///     A single <c>%</c>-format spec (<c>%.18e</c>, replicated once per column), or a multi-<c>%</c>
        ///     format string applied to the whole row (in which case <paramref name="delimiter"/> is ignored).
        ///     For a complex <paramref name="X"/> a single spec becomes <c>' (%s+%sj)'</c> per column.
        /// </param>
        /// <param name="delimiter">String separating columns.</param>
        /// <param name="newline">String separating rows.</param>
        /// <param name="header">String written at the beginning of the file, each line prefixed by <paramref name="comments"/>.</param>
        /// <param name="footer">String written at the end of the file, each line prefixed by <paramref name="comments"/>.</param>
        /// <param name="comments">String prepended to <paramref name="header"/>/<paramref name="footer"/> lines.</param>
        /// <param name="encoding">The file's encoding: any name Python's <c>codecs.lookup</c> resolves for UTF-8
        ///     (<c>utf-8-sig</c> too), UTF-16 and UTF-32 (the BOM forms write their BOM once, at the start of the file),
        ///     Latin-1, ASCII and the single-byte code pages (<c>cp1252</c>, <c>iso8859_15</c>, …). Null (and
        ///     <c>"locale"</c>) is UTF-8 without a BOM — NumPy's locale encoding on Linux and macOS (Windows NumPy uses the
        ///     ANSI code page). Every write is encoded strictly, as Python's TextIOWrapper encodes it after its newline
        ///     translation: a header, row or footer the encoding cannot hold raises <see cref="UnicodeEncodeError"/> (its
        ///     position counting the translated write), with the writes before it in the file. NumPy's <c>"bytes"</c> is
        ///     no encoding here (<see cref="LookupError"/> <c>unknown encoding: bytes</c>), nor is the empty name.</param>
        /// <exception cref="ValueError">A 0-d or more than 2-d array, a format with the wrong number of specs, or a null
        ///     <paramref name="fmt"/> (Python's None: <c>invalid fmt: None</c>).</exception>
        /// <exception cref="AttributeError">A null <paramref name="delimiter"/> where NumPy joins the specs with it (one spec
        ///     repeated per column: <c>'NoneType' object has no attribute 'join'</c>; a multi-<c>%</c> format never reads
        ///     it).</exception>
        /// <exception cref="TypeError">An element its spec cannot convert, or a null <paramref name="newline"/> once a row is
        ///     built (<c>Mismatch between array dtype …</c>; for a complex array the plain <c>can only concatenate str (not
        ///     "NoneType") to str</c>); a null <paramref name="header"/> / <paramref name="footer"/> (<c>object of type
        ///     'NoneType' has no len()</c>), or a null <paramref name="comments"/> / <paramref name="newline"/> while a
        ///     non-empty header or footer is built. A footer's error comes after every row was written, as in
        ///     NumPy.</exception>
        /// <remarks>
        ///     Byte-for-byte what NumPy 2.4.2's own <c>np.savetxt</c> writes for the same array — including
        ///     the Python text-mode newline translation on a filename target: every <c>\n</c> is written as
        ///     the platform line separator (<c>\r\n</c> on Windows), so the file matches NumPy on the same
        ///     platform. The <see cref="savetxt(Stream, NDArray, string, string, string, string, string, string, string)"/>
        ///     and <see cref="TextWriter"/> overloads write <c>\n</c> verbatim, matching NumPy's file-handle path.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
        /// </remarks>
        public static void savetxt(string fname, NDArray X, string fmt = "%.18e", string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
            => SaveTxtToFile(fname, X, fmt, delimiter, newline, header, footer, comments, encoding);

        /// <summary>
        ///     Save a 1-D or 2-D array to a text file, with one <c>%</c>-format spec per column.
        /// </summary>
        /// <param name="fmt">
        ///     One format spec per column. Its length must equal the number of columns, else
        ///     <see cref="AttributeError"/> is raised. For a complex array each entry must itself contain
        ///     both the real and imaginary specs (e.g. <c>"%.3e%+.3ej"</c>).
        /// </param>
        /// <exception cref="AttributeError">A spec list of the wrong length, or a null <paramref name="delimiter"/> (Python's
        ///     None has no <c>join</c>).</exception>
        /// <exception cref="TypeError">A null entry of <paramref name="fmt"/> (<c>sequence item 1: expected str instance,
        ///     NoneType found</c>), an element its spec cannot convert, and the null header / footer / comments / newline
        ///     errors of the single-spec overload.</exception>
        /// <exception cref="ValueError">A 0-d or more than 2-d array, or a null <paramref name="fmt"/> (<c>invalid fmt:
        ///     None</c>).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(string fname, NDArray X, string[] fmt, string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
            => SaveTxtToFile(fname, X, fmt, delimiter, newline, header, footer, comments, encoding);

        #endregion

        #region savetxt (stream)

        /// <summary>
        ///     Write a 1-D or 2-D array as text to an open byte stream (NumPy's binary file object). The stream is written
        ///     from its current position and left open (the caller owns it), and rows are separated by
        ///     <paramref name="newline"/> verbatim — no platform newline translation, matching NumPy's file-handle path.
        /// </summary>
        /// <param name="stream">The byte stream.</param>
        /// <param name="X">The 1-D or 2-D array to save.</param>
        /// <param name="fmt">A single <c>%</c>-format spec, or a multi-<c>%</c> row format.</param>
        /// <param name="delimiter">String separating columns.</param>
        /// <param name="newline">String separating rows.</param>
        /// <param name="header">String written first, each line prefixed by <paramref name="comments"/>.</param>
        /// <param name="footer">String written last, each line prefixed by <paramref name="comments"/>.</param>
        /// <param name="comments">String prepended to the header / footer lines.</param>
        /// <param name="encoding">The encoding every write is encoded with, on its own (<c>str.encode</c>: the BOM forms
        ///     write a BOM before EVERY write, as NumPy's <c>WriteWrap</c> does); null or empty is Latin-1, NumPy's default
        ///     for a byte stream (<c>WriteWrap(fh, encoding or 'latin1')</c>), so a character past <c>U+00FF</c> raises
        ///     <see cref="UnicodeEncodeError"/> (<c>'latin-1' codec can't encode character '\u4e2d' in position 0: ordinal
        ///     not in range(256)</c>) with the writes before it in the stream. An unknown or non-text name (<c>"bytes"</c>
        ///     included) raises <see cref="LookupError"/> at the first write, a codec NumSharp does not implement
        ///     <see cref="NotSupportedException"/>.</param>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        /// <exception cref="ValueError">A 0-d or more than 2-d array, a format with the wrong number of specs, or a null
        ///     <paramref name="fmt"/> (<c>invalid fmt: None</c>).</exception>
        /// <exception cref="AttributeError">A null <paramref name="delimiter"/> where NumPy joins the specs with
        ///     it.</exception>
        /// <exception cref="TypeError">An element its spec cannot convert (<c>Mismatch between array dtype …</c>), and the
        ///     null newline / header / footer / comments errors of the file-name overload, at the same points.</exception>
        /// <exception cref="UnicodeEncodeError">A write the encoding cannot encode.</exception>
        /// <exception cref="LookupError">An unknown or non-text encoding, at the first write.</exception>
        /// <exception cref="NotSupportedException">A codec NumSharp does not implement, at the first write.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(Stream stream, NDArray X, string fmt = "%.18e", string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
            => SaveTxtToStream(stream, X, fmt, delimiter, newline, header, footer, comments, encoding);

        /// <summary>Write a 1-D or 2-D array as text to an open byte stream, with one <c>%</c>-format spec per column.</summary>
        /// <param name="stream">The byte stream.</param>
        /// <param name="X">The 1-D or 2-D array to save.</param>
        /// <param name="fmt">One format spec per column (else <see cref="AttributeError"/>).</param>
        /// <param name="delimiter">String separating columns.</param>
        /// <param name="newline">String separating rows.</param>
        /// <param name="header">String written first.</param>
        /// <param name="footer">String written last.</param>
        /// <param name="comments">String prepended to the header / footer lines.</param>
        /// <param name="encoding">As for the single-spec overload (null is Latin-1).</param>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        /// <exception cref="AttributeError">A spec list of the wrong length, or a null <paramref name="delimiter"/>.</exception>
        /// <exception cref="TypeError">A null entry of <paramref name="fmt"/>, an element its spec cannot convert, and the
        ///     null newline / header / footer / comments errors of the file-name overload.</exception>
        /// <exception cref="ValueError">A 0-d or more than 2-d array, or a null <paramref name="fmt"/>.</exception>
        /// <exception cref="UnicodeEncodeError">A write the encoding cannot encode.</exception>
        /// <exception cref="LookupError">An unknown or non-text encoding, at the first write.</exception>
        /// <exception cref="NotSupportedException">A codec NumSharp does not implement, at the first write.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(Stream stream, NDArray X, string[] fmt, string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
            => SaveTxtToStream(stream, X, fmt, delimiter, newline, header, footer, comments, encoding);

        #endregion

        #region savetxt (text writer)

        /// <summary>
        ///     Write a 1-D or 2-D array as text to an open <see cref="TextWriter"/> (NumPy's text file object,
        ///     <c>io.StringIO</c>). The writer is written from its current position, flushed and left open (the caller owns
        ///     it); rows are separated by <paramref name="newline"/> verbatim, with no platform newline translation.
        /// </summary>
        /// <param name="writer">The text target.</param>
        /// <param name="X">The 1-D or 2-D array to save (a 1-D array is one value per line).</param>
        /// <param name="fmt">A single <c>%</c>-format spec (repeated for every column, joined by
        ///     <paramref name="delimiter"/>), or a multi-<c>%</c> row format (<paramref name="delimiter"/> then unused).</param>
        /// <param name="delimiter">String separating columns.</param>
        /// <param name="newline">String separating rows.</param>
        /// <param name="header">String written first, each line prefixed by <paramref name="comments"/>.</param>
        /// <param name="footer">String written last, each line prefixed by <paramref name="comments"/>.</param>
        /// <param name="comments">String prepended to the header / footer lines.</param>
        /// <param name="encoding">Accepted for NumPy's signature and IGNORED, as NumPy ignores it for a text target (its
        ///     <c>WriteWrap</c> encodes only for a stream that refuses <c>str</c>): any name, an unknown one included, writes
        ///     the same text.</param>
        /// <exception cref="ArgumentNullException"><paramref name="writer"/> is null.</exception>
        /// <exception cref="ValueError">A 0-d or more than 2-d array, a format with the wrong number of specs, or a null
        ///     <paramref name="fmt"/> (<c>invalid fmt: None</c>).</exception>
        /// <exception cref="AttributeError">A null <paramref name="delimiter"/> where NumPy joins the specs with
        ///     it.</exception>
        /// <exception cref="TypeError">An element its spec cannot convert (<c>Mismatch between array dtype …</c>), and the
        ///     null newline / header / footer / comments errors of the file-name overload, at the same points.</exception>
        /// <remarks>
        ///     What reached the writer before an error stays written (NumPy writes each row as it formats it), and the
        ///     writer is flushed in every case. https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
        /// </remarks>
        public static void savetxt(TextWriter writer, NDArray X, string fmt = "%.18e", string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
        {
            if (writer is null) throw new ArgumentNullException(nameof(writer));

            // A text target takes any str (Python's StringIO.write): no codec check. What was written before an error
            // stays written.
            try
            {
                SaveTxtCore(SaveTxtWriter.ForText(writer), X, fmt, delimiter, newline, header, footer, comments);
            }
            finally
            {
                writer.Flush();
            }
        }

        /// <summary>
        ///     Write a 1-D or 2-D array to a <see cref="TextWriter"/>, with one <c>%</c>-format spec per column.
        /// </summary>
        /// <param name="writer">The text target.</param>
        /// <param name="X">The 1-D or 2-D array to save.</param>
        /// <param name="fmt">One format spec per column (else <see cref="AttributeError"/>), joined by
        ///     <paramref name="delimiter"/>.</param>
        /// <param name="delimiter">String separating columns.</param>
        /// <param name="newline">String separating rows.</param>
        /// <param name="header">String written first.</param>
        /// <param name="footer">String written last.</param>
        /// <param name="comments">String prepended to the header / footer lines.</param>
        /// <param name="encoding">Accepted for NumPy's signature and ignored, as for the single-spec overload.</param>
        /// <exception cref="ArgumentNullException"><paramref name="writer"/> is null.</exception>
        /// <exception cref="AttributeError">A spec list of the wrong length, or a null <paramref name="delimiter"/>.</exception>
        /// <exception cref="TypeError">A null entry of <paramref name="fmt"/>, an element its spec cannot convert, and the
        ///     null newline / header / footer / comments errors of the file-name overload.</exception>
        /// <exception cref="ValueError">A 0-d or more than 2-d array, or a null <paramref name="fmt"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(TextWriter writer, NDArray X, string[] fmt, string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
        {
            if (writer is null) throw new ArgumentNullException(nameof(writer));

            // A text target takes any str (Python's StringIO.write): no codec check. What was written before an error
            // stays written.
            try
            {
                SaveTxtCore(SaveTxtWriter.ForText(writer), X, fmt, delimiter, newline, header, footer, comments);
            }
            finally
            {
                writer.Flush();
            }
        }

        #endregion

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
        private static void SaveTxtToFile(string fname, NDArray X, object fmt, string delimiter,
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
        private static void SaveTxtToStream(Stream stream, NDArray X, object fmt, string delimiter,
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
        private sealed class SaveTxtWriter
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
        private static void SaveTxtCore(SaveTxtWriter output, NDArray X, object fmt, string delimiter,
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
            bool iscomplex = tc == NPTypeCode.Complex;
            bool pyObjects = X.Storage.IsVString;

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
