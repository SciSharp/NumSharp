using System;
using System.Buffers;
using System.Globalization;
using System.IO;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using NumSharp.IO;

namespace NumSharp
{
    public static partial class np
    {
        private static readonly CultureInfo CI = CultureInfo.InvariantCulture;

        // NumPy 2.4.2 reference: numpy/_core/src/multiarray/multiarraymodule.c::array_fromfile ->
        // ctors.c::PyArray_FromFile (array_fromfile_binary / array_from_text with fromfile_next_element and
        // fromfile_skip_separator) -> arraytypes.c.src's <TYPE>_scan, the C library's fscanf (glibc's, ported) and
        // numpyos.c's NumPyOS_ascii_ftolf. The text scanners live in IO/TextReading/NumPyTextScan*.cs.

        /// <summary>
        ///     Construct an array from data in a text or binary file. Efficient for binary data of a known dtype, and parses
        ///     simply-formatted text files. Data written with <see cref="NDArray.tofile(string,string,string)"/> can be read
        ///     back with this function.
        /// </summary>
        /// <param name="file">A file name.</param>
        /// <param name="dtype">Element type (default float64). Binary: its item size; text: the scanner NumPy's dtype has
        ///     (signed / unsigned integers through glibc's <c>fscanf</c>, floats and bools through NumPy's own float reader,
        ///     complex through <c>CDOUBLE_scan</c>). A non-native byte order (<c>'&gt;i4'</c>) reads big-endian binary data
        ///     into a native array.</param>
        /// <param name="count">Elements to read; negative (default -1) reads the whole file.</param>
        /// <param name="sep">Separator between items of a text file. Empty (default) reads binary. A space of it matches any
        ///     run of whitespace (none included), and whitespace around the separator is always allowed.</param>
        /// <param name="offset">Bytes to skip from the file's current position first. Binary files only.</param>
        /// <param name="like">Reference array for NumPy's array-function dispatch: accepted for signature parity, no
        ///     effect.</param>
        /// <remarks>
        ///     <para>
        ///     NumPy's checks, in its order: <paramref name="sep"/>'s conversion (a lone surrogate is CPython's
        ///     <see cref="UnicodeEncodeError"/>, a NUL <c>embedded null character</c>), an <paramref name="offset"/> with a
        ///     separator, the file, the offset seek, then the dtype (a <c>StringDType</c> cannot be read, an unsized string dtype
        ///     builds <paramref name="count"/> zero-width elements, a string dtype has no text scanner).
        ///     </para>
        ///     <para>
        ///     A text read stops at the end of the file, after <paramref name="count"/> elements (the separator after the last
        ///     one is then optional), or at a separator that does not match; an element that does not scan is NumPy's
        ///     <see cref="ValueError"/>. The scanners consume what NumPy's do: <c>"1.5e,2"</c> reads 1.5 and 2 (the dangling
        ///     <c>e</c> is consumed), <c>"1, "</c> is an error for a float and not for an integer.
        ///     </para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html
        /// </remarks>
        /// <returns>The 1-D array of the elements read.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="file"/> is null.</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> holds a lone surrogate.</exception>
        /// <exception cref="TypeError"><paramref name="sep"/> is null (NumPy's <c>sep=None</c>: <c>fromfile() argument 4 must be
        ///     str, not None</c>), or an <paramref name="offset"/> with a text separator (<c>'offset' argument only permitted
        ///     for binary files</c>), checked before the file is opened.</exception>
        /// <exception cref="FileNotFoundException">No such file (Python's <c>[Errno 2] No such file or directory: 'name'</c>).</exception>
        /// <exception cref="OSError">An offset before the start of the file (<c>[Errno 22] Invalid argument</c>).</exception>
        /// <exception cref="ValueError"><paramref name="sep"/> holds a NUL; a <c>StringDType</c> (<c>Cannot read into object
        ///     array</c>); an unsized string dtype with a negative count, or a count past the end of a binary file
        ///     (<c>negative dimensions are not allowed</c>); a string dtype with a separator (<c>Unable to read character files
        ///     of that array type</c>); text that does not scan to its end (<c>string or file could not be read to its end due
        ///     to unmatched data</c>); a count no array can hold (<c>array is too big</c>).</exception>
        /// <exception cref="NotSupportedException">An unsized string dtype with a count of 0 or more: NumPy builds zero-width
        ///     elements, which NumSharp's string buffers cannot hold.</exception>
        public static NDArray fromfile(string file, DType dtype = null, long count = -1, string sep = "", long offset = 0,
            NDArray like = null)
        {
            if (file is null) throw new ArgumentNullException(nameof(file));
            // PyArg_ParseTupleAndKeywords converts sep ("s") before the body looks at the file.
            byte[] sepBytes = FromFileSep(sep);
            _ = like;
            // array_fromfile tests offset-with-a-separator before it opens the file, so a missing file reports this first.
            if (offset != 0 && sepBytes.Length != 0)
                throw new TypeError("'offset' argument only permitted for binary files");
            FileStream fs;
            try
            {
                fs = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            }
            catch (FileNotFoundException e)
            {
                throw new FileNotFoundException(
                    $"[Errno 2] No such file or directory: {Backends.Printing.PyRepr.Str(file)}", file, e);
            }
            using (fs)
                return FromFileCore(fs, dtype, count, sepBytes, offset);
        }

        /// <summary>
        ///     Construct an array from data in an open <see cref="Stream"/> (NumPy's file-object form), from its current
        ///     position. The stream stays open and ends where NumPy leaves the Python file: after the bytes the read consumed
        ///     (a binary read's partial trailing item included, a text scanner's look-ahead byte excluded), also when the read
        ///     raises.
        /// </summary>
        /// <param name="stream">The stream. A non-seekable one is read as NumPy cannot (it needs a seekable file
        ///     descriptor): a binary count of -1 reads to the end, and a text read's look-ahead stays consumed.</param>
        /// <param name="dtype">Element type (default float64).</param>
        /// <param name="count">Elements to read; negative (default -1) reads to the end.</param>
        /// <param name="sep">Separator between items of a text file. Empty (default) reads binary.</param>
        /// <param name="offset">Bytes to skip from the stream's position first. Binary only.</param>
        /// <param name="like">Accepted for NumPy's signature; no effect.</param>
        /// <returns>The 1-D array of the elements read.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
        /// <exception cref="UnicodeEncodeError"><paramref name="sep"/> holds a lone surrogate.</exception>
        /// <exception cref="TypeError"><paramref name="sep"/> is null, or an <paramref name="offset"/> with a text
        ///     separator.</exception>
        /// <exception cref="OSError">An offset before the start of the stream (<c>[Errno 22] Invalid argument</c>).</exception>
        /// <exception cref="ValueError">As <see cref="fromfile(string,DType,long,string,long,NDArray)"/>.</exception>
        /// <exception cref="NotSupportedException">An unsized string dtype with a count of 0 or more.</exception>
        public static NDArray fromfile(Stream stream, DType dtype = null, long count = -1, string sep = "", long offset = 0,
            NDArray like = null)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));
            byte[] sepBytes = FromFileSep(sep);
            _ = like;
            if (offset != 0 && sepBytes.Length != 0)
                throw new TypeError("'offset' argument only permitted for binary files");
            return FromFileCore(stream, dtype, count, sepBytes, offset);
        }

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
        private static NDArray FromFileCore(Stream stream, DType dtype, long count, byte[] sep, long offset)
        {
            // The seek happens before PyArray_FromFile looks at the dtype, so a refused dtype still moves the stream.
            if (offset != 0)
                SeekCurrent(stream, offset);
            dtype ??= float64;
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

        /// <summary>
        ///     <c>np.fromfile</c>'s <c>sep</c> through its <c>"s"</c> converter: <c>""</c> is NumPy's default (binary), and a
        ///     null is an explicit <c>None</c>, which the <c>"s"</c> format refuses.
        /// </summary>
        /// <param name="sep">The separator argument.</param>
        /// <returns>Its UTF-8 bytes.</returns>
        /// <exception cref="TypeError">A null separator (<c>fromfile() argument 4 must be str, not None</c>).</exception>
        /// <exception cref="UnicodeEncodeError">A lone surrogate.</exception>
        /// <exception cref="ValueError">A NUL (<c>embedded null character</c>).</exception>
        private static byte[] FromFileSep(string sep)
            => sep is null ? throw new TypeError("fromfile() argument 4 must be str, not None") : CStringArgument(sep);

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
    }
}
