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
        // joined by `delimiter`. The `%`-engine is PrintfFormatter (shared with ndarray.tofile). The overloads here check
        // their target; savetxt's body - the target opened as NumPy opens it, the format, the header, the rows and the
        // footer - is IO/SaveTxtIO.cs.

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
            => SaveTxtIO.SaveTxtToFile(fname, X, fmt, delimiter, newline, header, footer, comments, encoding);

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
            => SaveTxtIO.SaveTxtToFile(fname, X, fmt, delimiter, newline, header, footer, comments, encoding);

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
            => SaveTxtIO.SaveTxtToStream(stream, X, fmt, delimiter, newline, header, footer, comments, encoding);

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
            => SaveTxtIO.SaveTxtToStream(stream, X, fmt, delimiter, newline, header, footer, comments, encoding);

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
                SaveTxtIO.SaveTxtCore(SaveTxtIO.SaveTxtWriter.ForText(writer), X, fmt, delimiter, newline, header, footer, comments);
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
                SaveTxtIO.SaveTxtCore(SaveTxtIO.SaveTxtWriter.ForText(writer), X, fmt, delimiter, newline, header, footer, comments);
            }
            finally
            {
                writer.Flush();
            }
        }

        #endregion
    }
}
