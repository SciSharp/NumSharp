using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text;
using NumSharp.Backends.Printing;

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
        ///     Target path. If it ends in <c>.gz</c> the file is written gzip-compressed, as NumPy does.
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
        /// <param name="encoding">Output encoding; <c>null</c> (default) and <c>bytes</c>/<c>utf-8</c> use UTF-8 with no BOM, <c>latin1</c> uses Latin-1.</param>
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
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(string fname, NDArray X, string[] fmt, string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
            => SaveTxtToFile(fname, X, fmt, delimiter, newline, header, footer, comments, encoding);

        #endregion

        #region savetxt (stream)

        /// <summary>
        ///     Write a 1-D or 2-D array as text to an open stream. The stream is written from its current
        ///     position and left open (the caller owns it), and rows are separated by <paramref name="newline"/>
        ///     verbatim — no platform newline translation, matching NumPy's file-handle path.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(Stream stream, NDArray X, string fmt = "%.18e", string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));

            using (var sw = new StreamWriter(stream, ResolveSaveTxtEncoding(encoding), 1 << 16, leaveOpen: true))
            {
                SaveTxtCore(sw, X, fmt, delimiter, newline, header, footer, comments, translateNewlines: false);
                sw.Flush();
            }
        }

        /// <summary>Write a 1-D or 2-D array as text to an open stream, with one <c>%</c>-format spec per column.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(Stream stream, NDArray X, string[] fmt, string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ", string encoding = null)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));

            using (var sw = new StreamWriter(stream, ResolveSaveTxtEncoding(encoding), 1 << 16, leaveOpen: true))
            {
                SaveTxtCore(sw, X, fmt, delimiter, newline, header, footer, comments, translateNewlines: false);
                sw.Flush();
            }
        }

        #endregion

        #region savetxt (text writer)

        /// <summary>
        ///     Write a 1-D or 2-D array as text to an open <see cref="TextWriter"/>. The writer is left open
        ///     and owns its encoding/newline policy; rows are separated by <paramref name="newline"/> verbatim.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(TextWriter writer, NDArray X, string fmt = "%.18e", string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ")
        {
            if (writer is null) throw new ArgumentNullException(nameof(writer));

            SaveTxtCore(writer, X, fmt, delimiter, newline, header, footer, comments, translateNewlines: false);
            writer.Flush();
        }

        /// <summary>Write a 1-D or 2-D array to a <see cref="TextWriter"/>, with one <c>%</c>-format spec per column.</summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html</remarks>
        public static void savetxt(TextWriter writer, NDArray X, string[] fmt, string delimiter = " ",
            string newline = "\n", string header = "", string footer = "", string comments = "# ")
        {
            if (writer is null) throw new ArgumentNullException(nameof(writer));

            SaveTxtCore(writer, X, fmt, delimiter, newline, header, footer, comments, translateNewlines: false);
            writer.Flush();
        }

        #endregion

        #region internals

        // Opens the file (gzip when the name ends in .gz) and drives the core with Python text-mode
        // newline translation. Matching NumPy, the file is created/truncated BEFORE X is validated, so a
        // validation error (bad ndim / fmt) leaves an empty file behind.
        private static void SaveTxtToFile(string fname, NDArray X, object fmt, string delimiter,
            string newline, string header, string footer, string comments, string encoding)
        {
            if (fname is null) throw new ArgumentNullException(nameof(fname));

            Encoding enc = ResolveSaveTxtEncoding(encoding);
            bool gz = fname.EndsWith(".gz", StringComparison.OrdinalIgnoreCase);

            Stream fs = new FileStream(fname, FileMode.Create, FileAccess.Write);
            try
            {
                Stream outer = gz ? new GZipStream(fs, CompressionMode.Compress) : fs;
                try
                {
                    using (var sw = new StreamWriter(outer, enc, 1 << 16, leaveOpen: true))
                    {
                        SaveTxtCore(sw, X, fmt, delimiter, newline, header, footer, comments, translateNewlines: true);
                        sw.Flush();
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

        // The port of savetxt's body: validate rank, resolve the row `format`, then write header, one line
        // per row, and footer — batched into ~32 KB chunks (translated to the OS line separator on flush
        // when writing a text-mode file). Any layout is read in logical C-order via GetAtIndex.
        private static void SaveTxtCore(TextWriter fh, NDArray X, object fmt, string delimiter,
            string newline, string header, string footer, string comments, bool translateNewlines)
        {
            if (X is null) throw new ArgumentNullException(nameof(X));

            int ndim = X.ndim;
            if (ndim == 0 || ndim > 2)
                throw new ValueError($"Expected 1D or 2D array, got {ndim}D array instead");

            long nrows, ncol;
            if (ndim == 1) { nrows = X.shape[0]; ncol = 1; }
            else { nrows = X.shape[0]; ncol = X.shape[1]; }

            NPTypeCode tc = X.typecode;
            bool iscomplex = tc == NPTypeCode.Complex;

            string format = ResolveSaveTxtFormat(fmt, ncol, iscomplex, delimiter);

            // NumPy writes the header immediately (before the row loop), so a mid-row format error leaves
            // the header on disk. Rows are batched; the footer follows once the rows are flushed.
            if (header.Length > 0)
                WriteSaveTxt(fh, comments + header.Replace("\n", "\n" + comments) + newline, translateNewlines);

            var chunk = new StringBuilder(1 << 16);

            if (iscomplex)
            {
                var args = new object[checked(2 * ncol)];
                var tcs = new NPTypeCode[args.Length];
                for (int k = 0; k < tcs.Length; k++) tcs[k] = NPTypeCode.Double; // real/imag are Python floats

                for (long r = 0; r < nrows; r++)
                {
                    long baseIdx = r * ncol;
                    for (long c = 0; c < ncol; c++)
                    {
                        var z = (System.Numerics.Complex)X.GetAtIndex(baseIdx + c);
                        args[2 * c] = z.Real;
                        args[2 * c + 1] = z.Imaginary;
                    }

                    // NumPy: s = format % tuple(row2) + newline; fh.write(s.replace('+-', '-'))
                    string s = (PrintfFormatter.FormatRow(format, args, tcs) + newline).Replace("+-", "-");
                    chunk.Append(s);
                    if (chunk.Length >= (1 << 15)) FlushSaveTxtChunk(fh, chunk, translateNewlines);
                }
            }
            else
            {
                var args = new object[ncol];
                var tcs = new NPTypeCode[ncol];
                for (int k = 0; k < tcs.Length; k++) tcs[k] = tc;

                for (long r = 0; r < nrows; r++)
                {
                    long baseIdx = r * ncol;
                    for (long c = 0; c < ncol; c++)
                        args[c] = X.GetAtIndex(baseIdx + c);

                    try
                    {
                        // Append the row straight into the batch buffer — no per-row string allocation.
                        PrintfFormatter.FormatRowInto(chunk, format, args, tcs);
                    }
                    catch (PrintfArgumentException ex)
                    {
                        // NumPy catches the TypeError from `format % tuple(row)` and re-raises this.
                        throw new TypeError(
                            $"Mismatch between array dtype ('{tc.AsNumpyDtypeName()}') and format specifier ('{format}')", ex);
                    }

                    chunk.Append(newline);
                    if (chunk.Length >= (1 << 15)) FlushSaveTxtChunk(fh, chunk, translateNewlines);
                }
            }

            FlushSaveTxtChunk(fh, chunk, translateNewlines);

            if (footer.Length > 0)
                WriteSaveTxt(fh, comments + footer.Replace("\n", "\n" + comments) + newline, translateNewlines);
        }

        // A single write with the same text-mode newline translation the batch flush uses.
        private static void WriteSaveTxt(TextWriter fh, string s, bool translateNewlines)
            => fh.Write(translateNewlines ? s.Replace("\n", Environment.NewLine) : s);

        private static void FlushSaveTxtChunk(TextWriter fh, StringBuilder chunk, bool translateNewlines)
        {
            if (chunk.Length == 0) return;
            string s = chunk.ToString();
            // Python opens a filename target in text mode ('wt'), translating every '\n' to os.linesep on
            // write (CRLF on Windows). A stream / file handle target is written verbatim.
            fh.Write(translateNewlines ? s.Replace("\n", Environment.NewLine) : s);
            chunk.Clear();
        }

        // Build the per-row `format` string exactly as savetxt does, with NumPy's verbatim errors.
        private static string ResolveSaveTxtFormat(object fmt, long ncol, bool iscomplex, string delimiter)
        {
            if (fmt is string[] list)
            {
                if (list.Length != ncol)
                    throw new AttributeError($"fmt has wrong shape.  {PyListRepr(list)}");
                return string.Join(delimiter, list);
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
                    return string.Join(delimiter, parts);
                }

                if (iscomplex && nfmt != 2 * ncol)
                    throw new ValueError($"fmt has wrong number of % formats:  {s}");
                if (!iscomplex && nfmt != ncol)
                    throw new ValueError($"fmt has wrong number of % formats:  {s}");

                return s; // a multi-% format string used directly (delimiter ignored)
            }

            throw new ValueError($"invalid fmt: {fmt}");
        }

        private static int CountPercent(string s)
        {
            int n = 0;
            for (int i = 0; i < s.Length; i++)
                if (s[i] == '%') n++;
            return n;
        }

        // Python's str(list) — ['%d', '%.2f'] — reused only for the AttributeError message text.
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

        private static Encoding ResolveSaveTxtEncoding(string encoding)
        {
            if (string.IsNullOrEmpty(encoding))
                return new UTF8Encoding(false);

            switch (encoding.ToLowerInvariant())
            {
                case "bytes":
                case "utf-8":
                case "utf8":
                    return new UTF8Encoding(false);
                case "latin1":
                case "latin-1":
                case "iso-8859-1":
                    return Encoding.Latin1;
                default:
                    return Encoding.GetEncoding(encoding);
            }
        }

        #endregion
    }
}
