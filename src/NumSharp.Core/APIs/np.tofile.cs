using System;
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
        ///     is written as <c>format % (item,)</c>, where <c>item</c> is the Python object NumPy's <c>getitem</c>
        ///     returns (a <c>float</c> for every float dtype, an <c>int</c>, a <c>bool</c>, a <c>complex</c>, a
        ///     <c>str</c>), with CPython's rules and errors. <c>"%s"</c> (the default) and <c>""</c> write
        ///     <c>str(item)</c> (e.g. <c>1.5</c>, <c>(1+2j)</c>, <c>True</c>).
        /// </param>
        /// <remarks>
        ///     <para>
        ///     <paramref name="sep"/> and <paramref name="format"/> are checked before the file is opened, as NumPy's
        ///     argument parsing checks them: neither may hold a NUL or a lone surrogate.
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html</para>
        /// </remarks>
        /// <exception cref="ValueError"><paramref name="sep"/> or <paramref name="format"/> holds a NUL (<c>embedded null
        ///     character</c>), or a malformed format / a NaN under an integer conversion (see the stream overload).</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> or <paramref name="format"/> holds a lone surrogate,
        ///     or an element's text is not ASCII (see the stream overload).</exception>
        /// <exception cref="TypeError">A conversion the elements do not take (see the stream overload).</exception>
        /// <exception cref="OverflowException">See the stream overload.</exception>
        /// <exception cref="OSError">Binary mode over a <c>StringDType</c> array.</exception>
        public void tofile(string fid, string sep = "", string format = "%s")
        {
            // NumPy parses sep and format ("O|ss") before it opens the file: a bad one creates no file.
            CheckToFileArgument(sep);
            CheckToFileArgument(format);
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
        /// <param name="sep">Separator between items for text output; "" (default) writes binary. It is written as its
        ///     UTF-8 bytes, as NumPy writes the C string.</param>
        /// <param name="format">Python-style <c>%</c> format for text output (default <c>"%s"</c>): each element is written
        ///     as <c>format % (item,)</c> with the Python object NumPy's <c>getitem</c> returns.</param>
        /// <remarks>
        ///     <para>
        ///     Text mode is NumPy's <c>PyArray_ToFile</c>: for each element in C order, <c>str(item)</c> (format <c>""</c> or
        ///     <c>"%s"</c>) or <c>format % (item,)</c>, then the separator except after the last. <c>item</c> is a Python
        ///     <c>float</c> for every float dtype (a <c>float32</c> 0.1 writes <c>0.10000000149011612</c>), an <c>int</c>,
        ///     a <c>bool</c> (an <c>int</c> to Python's <c>%</c>: <c>'%x'</c> of <c>True</c> is <c>1</c>), a
        ///     <c>complex</c>, a <c>str</c> / <c>bytes</c>, or a <c>StringDType</c> element (a <c>str</c>, or the NA
        ///     object). NumSharp's <c>Char</c> elements are <c>int</c>s (a <c>uint16</c>) and its <c>Decimal</c> elements
        ///     <c>float</c>s.
        ///     </para>
        ///     <para>
        ///     Each element's text is written as ASCII (NumPy's <c>PyUnicode_AsASCIIString</c>); a non-ASCII text — a
        ///     string element, a <c>%c</c> past 127, non-ASCII literal text in the format — raises CPython's
        ///     <see cref="UnicodeEncodeError"/>, after the elements before it are written. (NumPy 2.4.2 does not check that
        ///     encode's result and crashes the process there.)
        ///     </para>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="sep"/> or <paramref name="format"/> holds a NUL (<c>embedded null
        ///     character</c>); a malformed format (<c>incomplete format</c>, <c>unsupported format character 'y' (0x79) at
        ///     index 1</c>); a NaN under an integer conversion.</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> or <paramref name="format"/> holds a lone surrogate
        ///     (CPython's UTF-8 error); an element's text is not ASCII.</exception>
        /// <exception cref="TypeError">CPython's <c>TypeError</c> of <c>format % (item,)</c>, unwrapped as NumPy leaves
        ///     it: <c>not enough arguments for format string</c>, <c>%x format: an integer is required, not float</c>,
        ///     <c>* wants int</c>.</exception>
        /// <exception cref="OverflowException">CPython's <c>OverflowError</c>: an infinity under an integer conversion, a
        ///     <c>%c</c> code point out of range, a <c>*</c> width past a C <c>ssize_t</c>.</exception>
        /// <exception cref="OSError">Binary mode over a <c>StringDType</c> array.</exception>
        public void tofile(Stream stream, string sep = "", string format = "%s")
        {
            if (stream is null) throw new System.ArgumentNullException(nameof(stream));
            CheckToFileArgument(sep);
            CheckToFileArgument(format);

            if (string.IsNullOrEmpty(sep))
                WriteBinary(stream);
            else
                WriteText(stream, sep, format);
        }

        // Binary: raw item bytes in C-order. Contiguous offset-0 views stream straight from their
        // buffer (no managed copy); every other layout is materialized C-contiguous first — mirroring
        // NumPy's PyArray_ToFile (whole-buffer fwrite when ISCONTIGUOUS, else a C-order element walk).
        /// <exception cref="OSError">A <c>StringDType</c> array (also an empty one): NumPy's <c>PyArray_ToFile</c> refuses
        ///     every dtype with <c>NPY_LIST_PICKLE</c> before it looks at the size, because a cell holds an arena offset or a
        ///     heap pointer that means nothing outside its own buffer.</exception>
        private void WriteBinary(Stream stream)
        {
            // Checked first, as NumPy checks it before the size: and dtypesize itself refuses a StringDType storage (the
            // phase-1 gate against reading 16-byte cells as raw bytes), so the size check must not run before this one.
            if (Storage.IsVString)
                throw new OSError("cannot write object arrays to a file in binary mode");
            if (size == 0 || dtypesize == 0)
                return;

            bool directable = Shape.IsContiguous && Shape.offset == 0;
            NDArray src = directable ? this : this.copy('C');
            try
            {
                unsafe
                {
                    long len = checked((long)src.size * src.dtypesize);
                    using (var ums = new UnmanagedMemoryStream((byte*)src.Storage.Address, len))
                        ums.CopyTo(stream);
                }

                // Keep the (possibly freshly-copied) source alive until the unmanaged read completes.
                System.GC.KeepAlive(src);
            }
            finally
            {
                // The C-order materialization is this method's own temp: release its pooled buffer now (a
                // stream write can throw — disk full — and must not strand it either). `this` is the caller's.
                if (!directable)
                    src.Dispose();
            }
        }

        /// <summary>
        ///     <c>PyArg_ParseTupleAndKeywords</c>' <c>"s"</c> converter, which <c>tofile</c> applies to <c>sep</c> and
        ///     <c>format</c>: the text must encode to UTF-8 (CPython refuses a lone surrogate) and hold no NUL (the C
        ///     string would end there).
        /// </summary>
        /// <param name="s">The argument (null or empty: nothing to check).</param>
        /// <exception cref="UnicodeEncodeError">A lone surrogate: <c>'utf-8' codec can't encode character '\ud800' in
        ///     position 0: surrogates not allowed</c>, checked first as CPython encodes before it looks for a NUL.</exception>
        /// <exception cref="ValueError"><c>embedded null character</c>.</exception>
        private static void CheckToFileArgument(string s)
        {
            if (string.IsNullOrEmpty(s))
                return;
            Backends.Strings.NDUtf8.GetByteCount(s);
            if (s.IndexOf('\0') >= 0)
                throw new ValueError("embedded null character");
        }

        /// <summary>
        ///     The Python object NumPy's <c>getitem</c> (<c>PyArray_GETITEM</c>) returns for an element, under the C#
        ///     boundary map of <see cref="Backends.Strings.NDPyObject"/>: a <c>float</c> for <c>float16</c> /
        ///     <c>float32</c> (the value widened exactly), and the value itself for a bool, an integer, a <c>float64</c>,
        ///     a complex, a <c>str</c> / <c>bytes</c> and a <c>StringDType</c> element (already Python objects).
        ///     NumSharp's <c>Char</c> is a <c>uint16</c> (an <c>int</c>) and its <c>Decimal</c> a <c>float64</c>, as they
        ///     print.
        /// </summary>
        /// <param name="v">The boxed element.</param>
        /// <param name="tc">The array's dtype.</param>
        /// <returns>The Python object.</returns>
        private static object GetItemObject(object v, NPTypeCode tc) => tc switch
        {
            NPTypeCode.Half => (double)(System.Half)v,
            NPTypeCode.Single => (double)(float)v,
            NPTypeCode.Char => (int)(char)v,
            NPTypeCode.Decimal => (double)(decimal)v,
            _ => v,
        };

        /// <summary>
        ///     NumPy's <c>PyUnicode_AsASCIIString</c> of an element's text: raises CPython's error for a character past
        ///     U+007F.
        /// </summary>
        /// <param name="item">The element's text.</param>
        /// <exception cref="UnicodeEncodeError"><c>'ascii' codec can't encode character '\xe9' in position 0: ordinal not
        ///     in range(128)</c> (or a range, for a run).</exception>
        private static void ThrowIfNotAscii(StringBuilder item)
        {
            foreach (ReadOnlyMemory<char> chunk in item.GetChunks())
            {
                if (chunk.Span.IndexOfAnyExceptInRange('\0', '\x7F') >= 0)
                {
                    // The error's positions count over the whole text: build it (the rare, failing path only).
                    Backends.Strings.NDStringCodec.ThrowIfNotAscii(item.ToString().AsSpan());
                    return;
                }
            }
        }

        // Text: each element in C-order is rendered via `format % (item,)` (default "%s" == str(item)) and joined by
        // `sep` — no trailing separator. GetAtIndex walks logical C-order through the shape's strides, so any layout is
        // honored without a prior materialization.
        private void WriteText(Stream stream, string sep, string format)
        {
            if (size == 0)
                return;

            NPTypeCode tc = typecode;
            bool useScalarStr = string.IsNullOrEmpty(format) || format == "%s";
            // The format is parsed once for every element. CPython parses it again per element, but a parse has no effect
            // of its own: the text and the errors (in their order) are the same.
            PercentFormat program = useScalarStr ? null : PrintfFormatter.Compile(format);
            long n = size;

            // ASCII-compatible, no BOM; leave the caller's stream open.
            using var sw = new StreamWriter(stream, new UTF8Encoding(false), 1 << 16, leaveOpen: true);

            // Batch elements into a StringBuilder and flush in ~32 KB chunks — one encoded Write per
            // chunk instead of two per element keeps the writer overhead off the hot path while bounding
            // memory (matters for very large arrays; a per-element StreamWriter.Write is otherwise a
            // measurable tax on top of the per-element scalar formatting).
            // NumPy's tofile pulls each element through getitem, which returns a Python float (double)
            // for float16/float32 — so the emitted text is that widened double's, not the value rendered
            // at the array's narrower native precision (e.g. float32 0.1 -> "0.10000000149011612", not
            // "0.1"). Widen Half/Single to double here to match; every other dtype already renders at the
            // width getitem hands back (int, bool, complex128, char).
            bool widenToDouble = tc == NPTypeCode.Half || tc == NPTypeCode.Single;
            NPTypeCode renderTc = widenToDouble ? NPTypeCode.Double : tc;

            // NumPy writes each element's text through PyUnicode_AsASCIIString, so a text that is not ASCII cannot be
            // written: CPython raises UnicodeEncodeError there (NumPy 2.4.2 does not check the NULL it gets back and
            // crashes the process). A string element, a '%c' past 127 and non-ASCII literal text in the format all make
            // such text, so every element is checked (one vectorized scan of a short text).
            bool text = tc == NPTypeCode.String;
            // A StringDType element is the Python OBJECT getitem returns: a plain str, or the descriptor's na_object (None,
            // a float NaN, a string, an int ...), whose str() / `format % obj` are CPython's rules for that object's type.
            bool pyObjects = Storage.IsVString;

            var sb = new StringBuilder(1 << 16);
            // NumPy's `format % (item,)`: a one-item tuple of the Python object getitem returned. The tuple and the
            // element's text buffer are reused for every element.
            var args = new object[1];
            var kinds = new[] { PercentFormat.PyObject };
            var item = new StringBuilder(64);
            try
            {
                for (long i = 0; i < n; i++)
                {
                    object v = GetAtIndex(i);
                    if (useScalarStr)
                    {
                        if (widenToDouble)
                            v = tc == NPTypeCode.Half ? (double)(System.Half)v : (double)(float)v;
                        // getitem of a string element is the Python str / bytes object: str() of a bytes value is its b'...'
                        // repr (NOT the np.bytes_(...) a 0-d array prints), of a str value the text itself.
                        string str = pyObjects
                            ? Backends.Strings.NDPyObject.Str(v)
                            : text
                                ? (v is byte[] bytes ? Backends.Printing.PyRepr.Bytes(bytes) : (string)v)
                                : Backends.Printing.ArrayFormatter.ScalarStr(v, renderTc);
                        Backends.Strings.NDStringCodec.ThrowIfNotAscii(str.AsSpan());
                        sb.Append(str);
                    }
                    else
                    {
                        item.Clear();
                        args[0] = pyObjects ? v : GetItemObject(v, tc);
                        program.FormatInto(item, args, kinds);
                        ThrowIfNotAscii(item);
                        sb.Append(item);
                    }

                    if (i != n - 1)
                        sb.Append(sep);

                    if (sb.Length >= (1 << 15))
                    {
                        sw.Write(sb);
                        sb.Clear();
                    }
                }
            }
            catch (PrintfArgumentException e)
            {
                // CPython's `format % item` raises a plain TypeError, which NumPy's tofile does not wrap. The formatter's
                // internal subclass only marks the errors np.savetxt rewrites into its Mismatch text, so it surfaces here as
                // the TypeError itself (the formatter's exception kept as the inner one).
                throw new TypeError(e.Message, e);
            }
            finally
            {
                // NumPy fwrites element by element (each element, then its separator), so an element that fails leaves every
                // element before it — and the separator after each — in the file. The batch flushes here on success and on
                // failure alike; the failing element's own text never reached the batch.
                if (sb.Length > 0)
                    sw.Write(sb);
            }
        }
    }
}
