using System;
using System.Buffers;
using System.IO;
using System.Text;
using NumSharp.Backends.Printing;

namespace NumSharp.IO
{
    /// <summary>
    ///     NumPy's raw array files - the bodies of <c>np.fromfile</c>, <c>np.fromstring</c> and <c>ndarray.tofile</c>, below
    ///     their argument conversion (the facades, NumPy's <c>array_fromfile</c> / <c>array_fromstring</c> in
    ///     <c>multiarraymodule.c</c> and <c>array_tofile</c> in <c>methods.c</c>): <c>PyArray_FromFile</c> and
    ///     <c>PyArray_FromString</c> (<c>ctors.c</c>), <c>PyArray_ToFile</c> (<c>convert.c</c>).
    /// </summary>
    /// <remarks>
    ///     The files carry no header: the dtype, the shape and the byte order are the caller's to know, which is why NumPy
    ///     calls them a convenience for quick storage rather than an archive format (the <c>.npy</c> format is
    ///     <see cref="NpyFormat"/>). The text modes' element scanners are <see cref="NumPyTextScan"/>'s.
    /// </remarks>
    internal static class ArrayFileIO
    {
        // ---- np.fromfile: array_fromfile's offset seek and default dtype, then PyArray_FromFile ----------------------

        /// <summary>
        ///     The body of <c>array_fromfile</c> after the file is open: the offset seek (<c>npy_fseek(fp, offset,
        ///     SEEK_CUR)</c>), then <c>PyArray_FromFile</c>'s dtype checks and its binary or text read.
        /// </summary>
        /// <param name="stream">The open stream.</param>
        /// <param name="dtype">The dtype (null: float64).</param>
        /// <param name="count">Elements to read; negative reads to the end.</param>
        /// <param name="sep">The separator's UTF-8 bytes (empty: binary).</param>
        /// <param name="offset">Bytes to skip first (0 for text).</param>
        /// <returns>The array.</returns>
        /// <exception cref="OSError">The offset lands before the start of the stream.</exception>
        /// <exception cref="ValueError">NumPy's refusals and read errors (see the public overloads).</exception>
        /// <exception cref="NotSupportedException">An unsized string dtype with a count of 0 or more.</exception>
        internal static NDArray FromFile(Stream stream, DType dtype, long count, byte[] sep, long offset)
        {
            // The seek happens before PyArray_FromFile looks at the dtype, so a refused dtype still moves the stream.
            if (offset != 0)
                SeekCurrent(stream, offset);
            dtype ??= np.float64;
            // PyArray_FromFile refuses a dtype whose items hold references (a StringDType cell points into its buffer's
            // allocator: no bytes of a file can be one) before it reads anything.
            if (dtype.Meta is VStringDTypeMeta)
                throw new ValueError("Cannot read into object array");
            // An item size of 0 (an unsized string class) reads nothing and builds `count` zero-width elements.
            if (dtype.itemsize == 0)
            {
                if (count < 0)
                    throw new ValueError("negative dimensions are not allowed");
                throw new NotSupportedException(
                    $"np.fromfile with the unsized string dtype {dtype.ToString(true)}: NumPy builds a zero-width (itemsize 0) " +
                    "array here, which NumSharp's fixed-width string buffers cannot hold. Give a width: 'S1', '<U5'.");
            }
            if (sep.Length == 0)
                return FromFileBinary(stream, dtype, count);
            // The string classes have no element scanner (their scanfunc is NULL).
            if (Backends.Strings.NDStringStorage.IsString(dtype))
                throw new ValueError("Unable to read character files of that array type");
            using var fp = new CFileReader(stream);
            try
            {
                return NumPyTextScan.FromFile(fp, dtype, count, sep);
            }
            finally
            {
                // npy_PyFile_DupClose2 seeks the Python file to where the C stream stopped, also after an error.
                fp.RestorePosition();
            }
        }

        /// <summary>
        ///     <c>npy_fseek(fp, offset, SEEK_CUR)</c>: moves the stream by <paramref name="offset"/> bytes (past its end is
        ///     allowed). A non-seekable stream skips forward by reading.
        /// </summary>
        /// <param name="stream">The stream.</param>
        /// <param name="offset">The byte offset from the current position.</param>
        /// <exception cref="OSError">The target position is before the start (<c>[Errno 22] Invalid argument</c>), or a
        ///     non-seekable stream cannot move back.</exception>
        private static void SeekCurrent(Stream stream, long offset)
        {
            if (stream.CanSeek)
            {
                if (stream.Position + offset < 0)
                    throw new OSError("[Errno 22] Invalid argument");
                stream.Seek(offset, SeekOrigin.Current);
                return;
            }
            if (offset < 0)
                throw new OSError("[Errno 29] Illegal seek");
            byte[] discard = ArrayPool<byte>.Shared.Rent((int)Math.Min(offset, 1 << 16));
            try
            {
                long left = offset;
                while (left > 0)
                {
                    int n = stream.Read(discard, 0, (int)Math.Min(discard.Length, left));
                    if (n <= 0)
                        break;
                    left -= n;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(discard);
            }
        }

        /// <summary>
        ///     NumPy's <c>array_fromfile_binary</c>: with a negative count, every whole item between the position and the end
        ///     (<c>(size - position) / itemsize</c> in C's truncating division, so a position past the end is a negative count:
        ///     <c>negative dimensions are not allowed</c>); otherwise <c>fread</c> of <paramref name="count"/> items, which also
        ///     consumes a trailing partial item. The bytes become the elements of a fresh array; a non-native byte order is
        ///     swapped to native.
        /// </summary>
        /// <param name="stream">The stream at the read position.</param>
        /// <param name="dtype">The dtype (sized, no references).</param>
        /// <param name="count">Elements to read; negative: all.</param>
        /// <returns>The 1-D array of the whole items read.</returns>
        /// <exception cref="ValueError">A negative computed count, or a count no array can hold.</exception>
        private static unsafe NDArray FromFileBinary(Stream stream, DType dtype, long count)
        {
            int itemsize = dtype.itemsize;
            long want;
            if (count < 0)
            {
                if (!stream.CanSeek)
                    want = long.MaxValue; // NumPy cannot read such a stream; NumSharp reads it to its end
                else
                {
                    long num = (stream.Length - stream.Position) / itemsize;
                    if (num < 0)
                        throw new ValueError("negative dimensions are not allowed");
                    want = num * itemsize;
                }
            }
            else
            {
                AllocationGuard.CheckElementCount(count, itemsize);
                want = count * itemsize;
            }

            var native = Backends.Strings.NDStringStorage.IsString(dtype)
                ? Backends.Strings.NDStringStorage.AllocationDescr(dtype)
                : (DType)dtype.GetTypeCode();
            NDArray result = stream.CanSeek
                ? ReadSeekable(stream, native, want, itemsize)
                : ReadUnseekable(stream, native, want, itemsize);
            if (!dtype.isnative && itemsize > 1)
                result.byteswap(inplace: true); // big-endian items into host order, per unit
            return result;
        }

        /// <summary>
        ///     <c>fread</c> of a seekable stream: the remaining length is known, so the result is allocated for the whole items
        ///     first and the stream is read straight into its buffer (NumPy's <c>fread</c> into the array's data); a trailing
        ///     partial item is consumed past them, as <c>fread</c> consumes it.
        /// </summary>
        /// <param name="stream">The stream at the read position.</param>
        /// <param name="native">The result's (native) descriptor.</param>
        /// <param name="want">The bytes requested.</param>
        /// <param name="itemsize">The item size (positive).</param>
        /// <returns>The array of the whole items read.</returns>
        private static unsafe NDArray ReadSeekable(Stream stream, DType native, long want, int itemsize)
        {
            long remaining = Math.Max(0, stream.Length - stream.Position);
            long take = Math.Min(want, remaining);
            long whole = take - take % itemsize;
            var result = new NDArray(native, Shape.Vector(whole / itemsize), false);
            long got;
            try
            {
                got = whole == 0 ? 0 : ReadInto(stream, (byte*)result.Storage.Address, whole);
            }
            catch
            {
                // A failed read (an I/O error) returns the result's buffer now rather than leaving it to the finalizer.
                result.Dispose();
                throw;
            }
            if (got == whole)
            {
                long extra = take - whole;
                if (extra > 0)
                    stream.Seek(extra, SeekOrigin.Current);
                return result;
            }
            // The stream ended before its reported length (it shrank while read): the whole items that arrived.
            using (result)
            {
                long items = got / itemsize;
                var shorter = new NDArray(native, Shape.Vector(items), false);
                long bytes = items * itemsize;
                if (bytes > 0)
                    Buffer.MemoryCopy((byte*)result.Storage.Address, (byte*)shorter.Storage.Address, bytes, bytes);
                return shorter;
            }
        }

        /// <summary>
        ///     <c>fread</c> of a stream with no length (NumPy cannot read one; NumSharp reads it): the bytes up to
        ///     <paramref name="want"/> or the end are staged in a pooled buffer that doubles, then the whole items become the
        ///     result. Every byte read counts as consumed, a partial trailing item included.
        /// </summary>
        /// <param name="stream">The stream.</param>
        /// <param name="native">The result's (native) descriptor.</param>
        /// <param name="want">The bytes requested (<see cref="long.MaxValue"/>: to the end).</param>
        /// <param name="itemsize">The item size (positive).</param>
        /// <returns>The array of the whole items read.</returns>
        private static unsafe NDArray ReadUnseekable(Stream stream, DType native, long want, int itemsize)
        {
            byte* buffer = null;
            long capacity = 0, total = 0;
            try
            {
                while (total < want)
                {
                    if (total == capacity)
                    {
                        long grown = Math.Max(capacity * 2, 1 << 16);
                        var fresh = (byte*)Backends.Unmanaged.Pooling.SizeBucketedBufferPool.Take(grown);
                        if (buffer != null)
                        {
                            Buffer.MemoryCopy(buffer, fresh, grown, total);
                            Backends.Unmanaged.Pooling.SizeBucketedBufferPool.Return((IntPtr)buffer, capacity);
                        }
                        buffer = fresh;
                        capacity = grown;
                    }
                    int n = stream.Read(new Span<byte>(buffer + total, (int)Math.Min(Math.Min(capacity - total, want - total), int.MaxValue)));
                    if (n <= 0)
                        break;
                    total += n;
                }
                long items = total / itemsize;
                var result = new NDArray(native, Shape.Vector(items), false);
                long bytes = items * itemsize;
                if (bytes > 0)
                    Buffer.MemoryCopy(buffer, (byte*)result.Storage.Address, bytes, bytes);
                return result;
            }
            finally
            {
                // Returned with the size it was taken with (the pool's rule), also when the read raises.
                if (buffer != null)
                    Backends.Unmanaged.Pooling.SizeBucketedBufferPool.Return((IntPtr)buffer, capacity);
            }
        }

        /// <summary>Reads <paramref name="count"/> bytes into unmanaged memory, or fewer when the stream ends first, in
        ///     spans of at most a gigabyte (a span's length is an <see cref="int"/>; the file itself has no such limit).</summary>
        /// <param name="s">The stream.</param>
        /// <param name="dst">The destination.</param>
        /// <param name="count">The byte count.</param>
        /// <returns>The bytes read.</returns>
        private static unsafe long ReadInto(Stream s, byte* dst, long count)
        {
            long off = 0;
            while (off < count)
            {
                int n = s.Read(new Span<byte>(dst + off, (int)Math.Min(count - off, 1 << 30)));
                if (n <= 0)
                    break;
                off += n;
            }
            return off;
        }

        // ---- CPython's "s" argument converter (array_fromfile / array_fromstring / array_tofile) ----------------

        /// <summary>
        ///     <c>PyArg_ParseTupleAndKeywords</c>' <c>"s"</c> converter (NumPy's <c>sep</c> of <c>fromfile</c> /
        ///     <c>fromstring</c>, <c>tofile</c>'s <c>sep</c> / <c>format</c>): the text's UTF-8 bytes, refused for a lone
        ///     surrogate (CPython encodes first) and then for a NUL (the C string would end there).
        /// </summary>
        /// <param name="s">The argument (not null).</param>
        /// <returns>The UTF-8 bytes.</returns>
        /// <exception cref="UnicodeEncodeError">A lone surrogate (<c>'utf-8' codec can't encode character '\ud800' in position
        ///     0: surrogates not allowed</c>).</exception>
        /// <exception cref="ValueError"><c>embedded null character</c>.</exception>
        internal static byte[] CStringArgument(string s)
        {
            byte[] bytes = Utf8Strict(s);
            if (Array.IndexOf(bytes, (byte)0) >= 0)
                throw new ValueError("embedded null character");
            return bytes;
        }

        /// <summary>CPython's <c>PyUnicode_AsUTF8AndSize</c>: the UTF-8 bytes of a <c>str</c>, refused for a lone surrogate.</summary>
        /// <param name="s">The text.</param>
        /// <returns>The bytes.</returns>
        /// <exception cref="UnicodeEncodeError">A lone surrogate, with CPython's text and position.</exception>
        internal static byte[] Utf8Strict(string s)
        {
            int n = Backends.Strings.NDUtf8.GetByteCount(s);
            var bytes = new byte[n];
            if (n > 0)
                Backends.Strings.NDUtf8.Encode(s, bytes);
            return bytes;
        }

        // ---- np.fromstring: PyArray_FromString ---------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_FromString</c> (<c>ctors.c</c>) below <c>array_fromstring</c>'s argument conversion: the dtype
        ///     checks, then the text read of <c>array_from_text</c> (<see cref="NumPyTextScan.FromString"/>).
        /// </summary>
        /// <param name="data">The text's bytes (a C string: a scan stops at an embedded NUL).</param>
        /// <param name="dtype">The dtype; null is NumPy's default float64.</param>
        /// <param name="count">Elements to read; negative reads all of them.</param>
        /// <param name="sep">The separator's UTF-8 bytes (not empty: the caller refuses the removed binary mode).</param>
        /// <returns>The 1-D array.</returns>
        /// <exception cref="ValueError">A <c>StringDType</c> (<c>Cannot create an object array from a string</c>); an unsized
        ///     string dtype (<c>zero-valued itemsize</c>); a sized string dtype (<c>don't know how to read character strings
        ///     with that array type</c>); a count no array can hold (<c>array is too big</c>); text that does not read to its
        ///     end (<c>string or file could not be read to its end due to unmatched data</c>).</exception>
        internal static NDArray FromString(byte[] data, DType dtype, long count, byte[] sep)
        {
            dtype ??= np.float64;
            // A dtype whose items hold references (StringDType: a cell points into its buffer's allocator) cannot be filled
            // from text, whatever the text or the count.
            if (dtype.Meta is VStringDTypeMeta)
                throw new ValueError("Cannot create an object array from a string");
            // A fixed-width string dtype: an unsized class first, then any dtype without a text element reader.
            if (Backends.Strings.NDStringStorage.IsString(dtype))
                throw new ValueError(dtype.itemsize == 0
                    ? "zero-valued itemsize"
                    : "don't know how to read character strings with that array type");
            return NumPyTextScan.FromString(data, dtype, count, sep);
        }

        // ---- ndarray.tofile: PyArray_ToFile -------------------------------------------------------------------------

        /// <summary>
        ///     NumPy's <c>PyArray_ToFile</c> (<c>convert.c</c>) below <c>array_tofile</c>'s argument conversion: binary mode
        ///     for an empty separator, text mode otherwise.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="stream">The destination, written from its position and left open.</param>
        /// <param name="sep">The separator (not null: the caller's <c>"s"</c> converter refuses <c>None</c>); empty is
        ///     binary mode.</param>
        /// <param name="format">The <c>%</c> format of text mode (not null).</param>
        /// <exception cref="OSError">Binary mode over a <c>StringDType</c> array.</exception>
        /// <exception cref="ValueError">A malformed format; a NaN under an integer conversion.</exception>
        /// <exception cref="TypeError">CPython's <c>TypeError</c> of <c>format % (item,)</c>.</exception>
        /// <exception cref="OverflowException">CPython's <c>OverflowError</c> of <c>format % (item,)</c>.</exception>
        /// <exception cref="UnicodeEncodeError">An element's text is not ASCII.</exception>
        internal static void ToFile(NDArray a, Stream stream, string sep, string format)
        {
            if (sep.Length == 0)
                WriteBinary(a, stream);
            else
                WriteText(a, stream, sep, format);
        }

        /// <summary>
        ///     <c>PyArray_ToFile</c>'s binary mode: the raw item bytes in C order. A C-contiguous array (a view included) is
        ///     written straight from its buffer (no managed copy); every other layout is materialized C-contiguous first —
        ///     NumPy's whole-buffer <c>fwrite</c> when <c>ISCONTIGUOUS</c>, else a C-order element walk.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="stream">The destination, written from its position.</param>
        /// <exception cref="OSError">A <c>StringDType</c> array (also an empty one): NumPy's <c>PyArray_ToFile</c> refuses
        ///     every dtype with <c>NPY_LIST_PICKLE</c> before it looks at the size, because a cell holds an arena offset or a
        ///     heap pointer that means nothing outside its own buffer.</exception>
        private static void WriteBinary(NDArray a, Stream stream)
        {
            // Checked first, as NumPy checks it before the size: and dtypesize itself refuses a StringDType storage (the
            // phase-1 gate against reading 16-byte cells as raw bytes), so the size check must not run before this one.
            if (a.Storage.IsVString)
                throw new OSError("cannot write object arrays to a file in binary mode");
            if (a.size == 0 || a.dtypesize == 0)
                return;

            // A C-contiguous array is written from its own buffer, starting at its first element (a contiguous slice
            // re-seats the address and keeps offset 0, a contiguous view of a strided parent keeps the base address and a
            // non-zero offset: address + offset * itemsize covers both); any other layout is materialized in C order first.
            bool directable = a.Shape.IsContiguous;
            NDArray src = directable ? a : a.copy('C');
            try
            {
                unsafe
                {
                    int itemSize = src.dtypesize;
                    byte* p = (byte*)src.Storage.Address + src.Shape.offset * (long)itemSize;
                    long len = checked((long)src.size * itemSize);
                    // NumPy's single fwrite of the buffer: spans written straight from it (a stream passes a span larger
                    // than its own buffer to the OS directly), at most a gigabyte each because a span's length is an int.
                    while (len > 0)
                    {
                        int n = (int)Math.Min(len, 1 << 30);
                        stream.Write(new ReadOnlySpan<byte>(p, n));
                        p += n;
                        len -= n;
                    }
                }

                // Keep the (possibly freshly-copied) source alive until the unmanaged writes complete.
                System.GC.KeepAlive(src);
            }
            finally
            {
                // The C-order materialization is this method's own temp: release its pooled buffer now (a
                // stream write can throw — disk full — and must not strand it either). `a` is the caller's.
                if (!directable)
                    src.Dispose();
            }
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

        /// <summary>
        ///     <c>PyArray_ToFile</c>'s text mode: for each element in C order, its text (<c>str(item)</c> or
        ///     <c>format % (item,)</c> of the Python object <c>getitem</c> returns) as ASCII, then the separator's UTF-8 bytes
        ///     except after the last element.
        /// </summary>
        /// <param name="a">The array.</param>
        /// <param name="stream">The destination (left open, flushed at the end).</param>
        /// <param name="sep">The separator (non-empty; checked by the caller).</param>
        /// <param name="format">The <c>%</c> format, or <c>""</c> / <c>"%s"</c> for <c>str(item)</c>.</param>
        /// <exception cref="UnicodeEncodeError">An element's text is not ASCII.</exception>
        /// <exception cref="TypeError">CPython's <c>TypeError</c> of <c>format % (item,)</c>.</exception>
        /// <remarks>
        ///     The bytes go straight into a pooled 32 KiB buffer that is written whenever it fills: the element texts are
        ///     ASCII (each is checked), so each character is its own byte, and the separator is encoded once. A
        ///     <see cref="StreamWriter"/> with a 64 K-character buffer and a builder of the same size cost more per call than
        ///     a thousand elements take to format. An element that fails leaves the elements before it, each with its
        ///     separator, in the stream — NumPy fwrites element by element — because the buffer is written on failure too.
        /// </remarks>
        private static void WriteText(NDArray a, Stream stream, string sep, string format)
        {
            if (a.size == 0)
                return;

            NPTypeCode tc = a.typecode;
            bool useScalarStr = string.IsNullOrEmpty(format) || format == "%s";
            // The format is parsed once for every element. CPython parses it again per element, but a parse has no effect
            // of its own: the text and the errors (in their order) are the same.
            PercentFormat program = useScalarStr ? null : PrintfFormatter.Compile(format);
            long n = a.size;

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
            bool pyObjects = a.Storage.IsVString;

            // The separator is written as its UTF-8 bytes (NumPy fwrites the C string).
            byte[] sepBytes = Encoding.UTF8.GetBytes(sep);
            var output = new AsciiOutput(stream);
            // NumPy's `format % (item,)`: a one-item tuple of the Python object getitem returned. The tuple and the
            // element's text buffer are reused for every element.
            var args = new object[1];
            var kinds = new[] { PercentFormat.PyObject };
            var item = new StringBuilder(64);
            try
            {
                for (long i = 0; i < n; i++)
                {
                    object v = a.GetAtIndex(i);
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
                        output.Ascii(str.AsSpan());
                    }
                    else
                    {
                        item.Clear();
                        args[0] = pyObjects ? v : GetItemObject(v, tc);
                        program.FormatInto(item, args, kinds);
                        ThrowIfNotAscii(item);
                        foreach (ReadOnlyMemory<char> chunk in item.GetChunks())
                            output.Ascii(chunk.Span);
                    }

                    if (i != n - 1)
                        output.Bytes(sepBytes);
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
                // On success and on failure alike: every element written before a failing one stays (the failing
                // element's own text never reached the buffer), then the stream is flushed as NumPy's file is.
                output.Close();
            }
        }

        /// <summary>
        ///     The byte buffer <see cref="WriteText"/> writes through: ASCII text one byte per character and raw bytes,
        ///     collected in a pooled 32 KiB array that is written to the stream whenever it fills.
        /// </summary>
        /// <remarks>A class rather than locals because the element loop and the error paths share it; <see cref="Close"/>
        ///     returns the array to the pool (idempotent).</remarks>
        private sealed class AsciiOutput
        {
            private readonly Stream _stream;
            private byte[] _buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(1 << 15);
            private int _used;

            /// <summary>Creates the buffer over a stream.</summary>
            /// <param name="stream">The destination.</param>
            internal AsciiOutput(Stream stream) => _stream = stream;

            /// <summary>Appends text whose characters are all ASCII (checked by the caller): one byte each.</summary>
            /// <param name="text">The text.</param>
            internal void Ascii(ReadOnlySpan<char> text)
            {
                while (text.Length > 0)
                {
                    if (_used == _buffer.Length)
                        Drain();
                    int n = Math.Min(text.Length, _buffer.Length - _used);
                    Span<byte> dst = _buffer.AsSpan(_used, n);
                    for (int i = 0; i < n; i++)
                        dst[i] = (byte)text[i];
                    _used += n;
                    text = text.Slice(n);
                }
            }

            /// <summary>Appends raw bytes (the separator's UTF-8).</summary>
            /// <param name="bytes">The bytes.</param>
            internal void Bytes(ReadOnlySpan<byte> bytes)
            {
                while (bytes.Length > 0)
                {
                    if (_used == _buffer.Length)
                        Drain();
                    int n = Math.Min(bytes.Length, _buffer.Length - _used);
                    bytes.Slice(0, n).CopyTo(_buffer.AsSpan(_used, n));
                    _used += n;
                    bytes = bytes.Slice(n);
                }
            }

            /// <summary>Writes the buffered bytes to the stream.</summary>
            private void Drain()
            {
                _stream.Write(_buffer, 0, _used);
                _used = 0;
            }

            /// <summary>Writes what is buffered, flushes the stream and returns the array to the pool.</summary>
            internal void Close()
            {
                if (_buffer == null)
                    return;
                try
                {
                    if (_used > 0)
                        Drain();
                    _stream.Flush();
                }
                finally
                {
                    System.Buffers.ArrayPool<byte>.Shared.Return(_buffer);
                    _buffer = null;
                }
            }
        }
    }
}
