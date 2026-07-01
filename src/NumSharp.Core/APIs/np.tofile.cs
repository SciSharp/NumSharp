using System.IO;
using System.Text;
using NumSharp.Backends.Printing;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     Write the array to a file as binary (default) or text.
        ///     <para>
        ///     Data is always written in C (row-major) order, independent of the array's own layout —
        ///     a sliced / strided / transposed / broadcast view writes its logical elements, NOT the raw
        ///     underlying buffer. The data produced can be recovered with <see cref="np.fromfile(string,System.Type)"/>.
        ///     </para>
        /// </summary>
        /// <param name="fid">A filename. The file is created (truncated if it exists).</param>
        /// <param name="sep">
        ///     Separator between items for text output. If "" (empty, the default) a binary file is
        ///     written, equivalent to <c>stream.Write(a.tobytes('C'))</c>.
        /// </param>
        /// <param name="format">
        ///     Python-style <c>%</c> format string for text output (ignored in binary mode). Each entry
        ///     is written as <c>format % item</c>. The default <c>"%s"</c> uses the element's NumPy
        ///     scalar string (e.g. <c>1.5</c>, <c>(1+2j)</c>, <c>True</c>).
        /// </param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html</remarks>
        public void tofile(string fid, string sep = "", string format = "%s")
        {
            // NumPy opens the filename with mode "wb" (create + truncate).
            using var fs = new FileStream(fid, FileMode.Create, FileAccess.Write);
            tofile(fs, sep, format);
        }

        /// <summary>
        ///     Write the array to an open <see cref="Stream"/> as binary (default) or text. The stream is
        ///     written from its current position and left open (the caller owns it), matching NumPy's
        ///     file-object <c>tofile</c>. Data is always in C (row-major) order.
        /// </summary>
        /// <param name="stream">An open, writeable stream.</param>
        /// <param name="sep">Separator between items for text output; "" (default) writes binary.</param>
        /// <param name="format">Python-style <c>%</c> format for text output (default <c>"%s"</c>).</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html</remarks>
        public void tofile(Stream stream, string sep = "", string format = "%s")
        {
            if (stream is null) throw new System.ArgumentNullException(nameof(stream));

            if (string.IsNullOrEmpty(sep))
                WriteBinary(stream);
            else
                WriteText(stream, sep, format);
        }

        // Binary: raw item bytes in C-order. Contiguous offset-0 views stream straight from their
        // buffer (no managed copy); every other layout is materialized C-contiguous first — mirroring
        // NumPy's PyArray_ToFile (whole-buffer fwrite when ISCONTIGUOUS, else a C-order element walk).
        private void WriteBinary(Stream stream)
        {
            if (size == 0 || dtypesize == 0)
                return;

            bool directable = Shape.IsContiguous && Shape.offset == 0;
            NDArray src = directable ? this : this.copy('C');

            unsafe
            {
                long len = checked((long)src.size * src.dtypesize);
                using (var ums = new UnmanagedMemoryStream((byte*)src.Storage.Address, len))
                    ums.CopyTo(stream);
            }

            // Keep the (possibly freshly-copied) source alive until the unmanaged read completes.
            System.GC.KeepAlive(src);
        }

        // Text: each element in C-order is rendered via `format % item` (default "%s" == the NumPy scalar
        // string) and joined by `sep` — no trailing separator. GetAtIndex walks logical C-order through
        // the shape's strides, so any layout is honored without a prior materialization.
        private void WriteText(Stream stream, string sep, string format)
        {
            if (size == 0)
                return;

            NPTypeCode tc = typecode;
            bool useScalarStr = string.IsNullOrEmpty(format) || format == "%s";
            long n = size;

            // ASCII-compatible, no BOM; leave the caller's stream open.
            using var sw = new StreamWriter(stream, new UTF8Encoding(false), 1 << 16, leaveOpen: true);

            // Batch elements into a StringBuilder and flush in ~32 KB chunks — one encoded Write per
            // chunk instead of two per element keeps the writer overhead off the hot path while bounding
            // memory (matters for very large arrays; a per-element StreamWriter.Write is otherwise a
            // measurable tax on top of the per-element scalar formatting).
            var sb = new StringBuilder(1 << 16);
            for (long i = 0; i < n; i++)
            {
                object v = GetAtIndex(i);
                sb.Append(useScalarStr
                    ? Backends.Printing.ArrayFormatter.ScalarStr(v, tc)
                    : PrintfFormatter.Format(format, v, tc));

                if (i != n - 1)
                    sb.Append(sep);

                if (sb.Length >= (1 << 15))
                {
                    sw.Write(sb);
                    sb.Clear();
                }
            }

            if (sb.Length > 0)
                sw.Write(sb);
        }
    }
}
