using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The casts between a NEP 55 <c>StringDType</c> array and every other dtype — the strided loops of NumPy 2.4.2's
    ///     <c>numpy/_core/src/multiarray/stringdtype/casts.cpp</c>: <c>U</c> / <c>S</c> → <c>StringDType</c> (encode / validate
    ///     the text), <c>StringDType</c> → <c>U</c> / <c>S</c> (decode, truncate to the width, ASCII check), bool ↔
    ///     <c>StringDType</c>, and numbers ↔ <c>StringDType</c> (a number's NumPy scalar <c>str()</c>; a text parsed by CPython's
    ///     <c>int()</c> / <c>float()</c> / <c>complex()</c>). <c>StringDType</c> → <c>StringDType</c> is the copy core
    ///     (<see cref="NDVStringCopy"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Order and partial writes.</b> NumPy assigns through <c>raw_array_assign_array</c> (the axes sorted by the
    ///     DESTINATION's strides, <see cref="RawArrayIter.PrepareThree"/>) and calls the cast's strided loop once per innermost
    ///     run. These loops are ArrayMethods, not wrapped legacy casts, so they write element by element: an error leaves the
    ///     elements before it (in that order) converted, and names the first failing element.
    ///     </para>
    ///     <para>
    ///     <b>Missing data.</b> An NA element of a descriptor with an NA object reads, for a TEXT target, as the NA object's
    ///     <c>str()</c> (<c>'None'</c>, <c>'nan'</c>) — or as the string NA itself — and for a NUMERIC target raises
    ///     <c>Arrays with missing data cannot be converted to a non-nullable type</c>; a bool target takes NumPy's own rule
    ///     (a NaN-like NA is True, any other NA is False, a string NA is <c>len(na) == 0</c>). In the other direction only a
    ///     float64 NaN becomes an NA element (NumPy compares a <c>np.float64</c> scalar with a float NaN NA object; a float32 /
    ///     float16 / complex NaN is written as its text).
    ///     </para>
    ///     <para>
    ///     <b>Locking.</b> The <c>StringDType</c> side's allocator is locked once for the whole cast (NumPy's
    ///     <c>NpyString_acquire_allocator</c> per loop call).
    ///     </para>
    /// </remarks>
    internal static unsafe class NDVStringCasting
    {
        /// <summary>NumPy's text when an NA element meets a numeric target (<c>load_non_nullable_string</c>).</summary>
        internal const string MissingDataText = "Arrays with missing data cannot be converted to a non-nullable type";

        /// <summary>
        ///     <c>dst[...] = src</c> where exactly one of the two storages holds <c>StringDType</c> text: the source broadcasts
        ///     to the destination and every element is converted by NumPy's StringDType cast loop for the pair.
        /// </summary>
        /// <param name="dst">The destination (writeable).</param>
        /// <param name="src">The source, broadcastable to <paramref name="dst"/>.</param>
        /// <exception cref="ArgumentNullException">An argument is null.</exception>
        /// <exception cref="ArgumentException">Both or neither storage holds <c>StringDType</c> text (the caller's routing).</exception>
        /// <exception cref="ValueError"><paramref name="dst"/> is read-only, a text does not parse into a numeric
        ///     destination, or an NA element meets a numeric destination.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="src"/> does not broadcast to <paramref name="dst"/>.</exception>
        /// <exception cref="OverflowException">An integer text outside the destination (NumPy's StringDType texts).</exception>
        /// <exception cref="TypeError">A <c>U</c> source holds an invalid code point (<c>Invalid unicode code point found</c>).</exception>
        /// <exception cref="UnicodeDecodeError">An <c>S</c> source element is not UTF-8.</exception>
        /// <exception cref="UnicodeEncodeError">A text is not ASCII and the destination is <c>S</c>.</exception>
        /// <exception cref="RuntimeError">A bool source byte is neither 0 nor 1 (<c>invalid value encountered in bool to
        ///     string cast</c>).</exception>
        /// <exception cref="NotSupportedException">The other dtype has no StringDType cast in NumSharp (no storage).</exception>
        internal static void Copy(UnmanagedStorage dst, UnmanagedStorage src)
        {
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst.IsVString == src.IsVString)
                throw new ArgumentException("A StringDType cast needs exactly one StringDType storage.");
            // NumPy's text for a read-only destination (NumSharpException.ThrowIfNotWriteable's).
            if (!dst.Shape.IsWriteable)
                throw new ValueError("assignment destination is read-only");
            // np.broadcast_to's validation (its error names the arrays' own shapes), as every NumSharp copy.
            if (!SameDimensions(src.Shape, dst.Shape))
                np.broadcast_to(src.Shape, dst.Shape);
            if (dst.Shape.size == 0)
                return;

            if (dst.IsVString)
                IntoVString(dst, src);
            else
                FromVString(dst, src);
        }

        // ---------------------------------------------------------------------------------------------
        // Into StringDType
        // ---------------------------------------------------------------------------------------------

        /// <summary>Converts every source element into the <c>StringDType</c> destination (its allocator locked once).</summary>
        /// <param name="dst">The <c>StringDType</c> destination.</param>
        /// <param name="src">The <c>U</c> / <c>S</c> / bool / numeric source.</param>
        /// <exception cref="Exception">See <see cref="Copy"/>.</exception>
        private static void IntoVString(UnmanagedStorage dst, UnmanagedStorage src)
        {
            var slice = dst.VStringSlice;
            var allocator = slice.Allocator;
            RunAction run;
            if (src.IsFixedString)
            {
                var fixedSlice = src.StringSlice;
                int chars = fixedSlice.Chars;
                // One scratch for the whole walk: the widest element's UTF-8 form (4 bytes per code point at most).
                byte[] scratch = new byte[Math.Max(4 * chars, 1)];
                run = fixedSlice.IsUnicode
                    ? (s, ss, d, ds, n) => UnicodeRun(s, ss, chars, d, ds, n, allocator, scratch)
                    : (s, ss, d, ds, n) => BytesRun(s, ss, chars, d, ds, n, allocator);
            }
            else if (src.TypeCode == NPTypeCode.Boolean)
            {
                run = (s, ss, d, ds, n) => BoolRun(s, ss, d, ds, n, allocator);
            }
            else
            {
                NPTypeCode from = src.TypeCode;
                // Only a float64 NaN compares equal to a float NaN NA object (NumPy builds a np.float64 scalar, a Python float
                // subclass; float32 / float16 / complex scalars are not, and NaN != NaN for them).
                bool nanToNA = slice.Descr.VString.HasNanNA && from == NPTypeCode.Double;
                run = (s, ss, d, ds, n) => NumberRun(s, ss, from, d, ds, n, allocator, nanToNA);
            }

            allocator.Acquire();
            try
            {
                Walk(dst, src, run);
            }
            finally
            {
                allocator.Release();
            }
        }

        /// <summary>
        ///     <c>unicode_to_string</c>: each <c>U</c> element (trailing NUL code points dropped) encoded as UTF-8 and packed; a
        ///     surrogate or a unit above U+10FFFF is NumPy's <c>TypeError</c>.
        /// </summary>
        /// <param name="s">The first source element.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="chars">The source width in code points.</param>
        /// <param name="d">The first destination cell.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="allocator">The destination's allocator (locked).</param>
        /// <param name="scratch">Room for one element's UTF-8 form.</param>
        /// <exception cref="TypeError"><c>Invalid unicode code point found</c>.</exception>
        /// <exception cref="OutOfMemoryException"><c>Failed to allocate string in unicode to string cast</c>.</exception>
        private static void UnicodeRun(byte* s, long ss, int chars, byte* d, long ds, long n, VStringAllocator allocator, byte[] scratch)
        {
            fixed (byte* buf = scratch)
            {
                for (long i = 0; i < n; i++, s += ss, d += ds)
                {
                    uint* u = (uint*)s;
                    int len = chars;
                    while (len > 0 && u[len - 1] == 0)
                        len--;
                    int size = 0;
                    for (int k = 0; k < len; k++)
                    {
                        uint cp = u[k];
                        // num_utf8_bytes_for_codepoint: surrogates and values past U+10FFFF have no UTF-8 form.
                        if (cp > NDStringCodec.MaxCodePoint || (cp >= 0xD800 && cp <= 0xDFFF))
                            throw new TypeError("Invalid unicode code point found");
                        size += cp <= 0x7F ? 1 : cp <= 0x7FF ? 2 : cp <= 0xFFFF ? 3 : 4;
                    }
                    int w = 0;
                    for (int k = 0; k < len; k++)
                        w += EncodeCodePoint(u[k], buf + w);
                    if (allocator.Pack(d, buf, size) < 0)
                        throw new OutOfMemoryException("Failed to allocate string in unicode to string cast");
                }
            }
        }

        /// <summary>
        ///     <c>bytes_to_string</c>: each <c>S</c> element (trailing NUL bytes dropped) validated as strict UTF-8 and packed as
        ///     it is.
        /// </summary>
        /// <param name="s">The first source element.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="chars">The source width in bytes.</param>
        /// <param name="d">The first destination cell.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="allocator">The destination's allocator (locked).</param>
        /// <exception cref="UnicodeDecodeError">CPython's text for the element's first bad UTF-8 sequence.</exception>
        /// <exception cref="OutOfMemoryException"><c>Failed to allocate string in bytes to string cast</c>.</exception>
        private static void BytesRun(byte* s, long ss, int chars, byte* d, long ds, long n, VStringAllocator allocator)
        {
            for (long i = 0; i < n; i++, s += ss, d += ds)
            {
                int len = chars;
                while (len > 0 && s[len - 1] == 0)
                    len--;
                NDUtf8.ThrowIfInvalid(new ReadOnlySpan<byte>(s, len));
                if (allocator.Pack(d, s, len) < 0)
                    throw new OutOfMemoryException("Failed to allocate string in bytes to string cast");
            }
        }

        /// <summary><c>bool_to_string</c>: <c>True</c> / <c>False</c>; a byte that is neither 0 nor 1 is NumPy's RuntimeError.</summary>
        /// <param name="s">The first bool.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="d">The first destination cell.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="allocator">The destination's allocator (locked).</param>
        /// <exception cref="RuntimeError"><c>invalid value encountered in bool to string cast</c>.</exception>
        /// <exception cref="OutOfMemoryException"><c>Failed to pack string in bool to string cast</c>.</exception>
        private static void BoolRun(byte* s, long ss, byte* d, long ds, long n, VStringAllocator allocator)
        {
            byte* text = stackalloc byte[5];
            for (long i = 0; i < n; i++, s += ss, d += ds)
            {
                int size;
                if (*s == 1)
                {
                    text[0] = (byte)'T'; text[1] = (byte)'r'; text[2] = (byte)'u'; text[3] = (byte)'e';
                    size = 4;
                }
                else if (*s == 0)
                {
                    text[0] = (byte)'F'; text[1] = (byte)'a'; text[2] = (byte)'l'; text[3] = (byte)'s'; text[4] = (byte)'e';
                    size = 5;
                }
                else
                {
                    throw new RuntimeError("invalid value encountered in bool to string cast");
                }
                if (allocator.Pack(d, text, size) < 0)
                    throw new OutOfMemoryException("Failed to pack string in bool to string cast");
            }
        }

        /// <summary>
        ///     <c>type_to_string</c> / <c>float_to_string</c>: each number's NumPy scalar <c>str()</c> (the formatter of the
        ///     legacy casts, <see cref="NDTextFormat"/>) packed as UTF-8; a float64 NaN becomes an NA element when the
        ///     descriptor's NA object is a float NaN.
        /// </summary>
        /// <param name="s">The first number.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="from">The numeric source dtype.</param>
        /// <param name="d">The first destination cell.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="allocator">The destination's allocator (locked).</param>
        /// <param name="nanToNA">True when a NaN is stored as an NA element (a float64 source, a NaN NA object).</param>
        /// <exception cref="OutOfMemoryException">NumPy's string <c>MemoryError</c> texts.</exception>
        private static void NumberRun(byte* s, long ss, NPTypeCode from, byte* d, long ds, long n, VStringAllocator allocator, bool nanToNA)
        {
            var kind = NDTextFormat.KindOf(from);
            var canonType = NDTextFormat.CanonicalType(kind);
            int canonSize = canonType.SizeOf();
            Span<char> chars = stackalloc char[NDTextFormat.MaxTextLength];
            byte* text = stackalloc byte[NDTextFormat.MaxTextLength];
            // A block of canonical values: the house cast kernel converts the source to its family's canonical type.
            const int Block = NDTextConversion.BlockSize;
            byte* block = stackalloc byte[Block * 16];
            var cast = canonType == from ? null : CastLoop(from, canonType);
            for (long done = 0; done < n; done += Block)
            {
                int k = (int)Math.Min(Block, n - done);
                byte* values;
                long vstride;
                if (cast is null)
                {
                    values = s + done * ss;
                    vstride = ss;
                }
                else
                {
                    cast(s + done * ss, ss, block, canonSize, k);
                    values = block;
                    vstride = canonSize;
                }
                byte* cell = d + done * ds;
                for (int j = 0; j < k; j++, values += vstride, cell += ds)
                {
                    if (nanToNA && double.IsNaN(*(double*)values))
                    {
                        if (allocator.PackNull(cell) < 0)
                            throw new OutOfMemoryException("Failed to pack null string during float to string cast");
                        continue;
                    }
                    int len = NDTextFormat.Format(kind, values, chars);
                    for (int c = 0; c < len; c++)
                        text[c] = (byte)chars[c]; // every number's str() is ASCII
                    if (allocator.Pack(cell, text, len) < 0)
                        throw new OutOfMemoryException("Failed to pack string while converting from python string");
                }
            }
        }

        // ---------------------------------------------------------------------------------------------
        // From StringDType
        // ---------------------------------------------------------------------------------------------

        /// <summary>Converts every <c>StringDType</c> source element into the destination (the source's allocator locked once).</summary>
        /// <param name="dst">The <c>U</c> / <c>S</c> / bool / numeric destination.</param>
        /// <param name="src">The <c>StringDType</c> source.</param>
        /// <exception cref="Exception">See <see cref="Copy"/>.</exception>
        private static void FromVString(UnmanagedStorage dst, UnmanagedStorage src)
        {
            var slice = src.VStringSlice;
            var allocator = slice.Allocator;
            var p = slice.Descr.VString;
            // load_nullable_string: an NA element reads as the NA object's str() when the descriptor has a non-string NA
            // object, else as the default string (the string NA itself; '' without an NA object).
            byte[] naText = NDUtf8.Encode(p.HasNAObject && !p.HasStringNA ? p.NAName : p.DefaultString);
            RunAction run;
            if (dst.IsFixedString)
            {
                var fixedSlice = dst.StringSlice;
                int chars = fixedSlice.Chars;
                run = fixedSlice.IsUnicode
                    ? (s, ss, d, ds, n) => ToUnicodeRun(s, ss, d, ds, n, chars, allocator, naText)
                    : (s, ss, d, ds, n) => ToBytesRun(s, ss, d, ds, n, chars, allocator, naText);
            }
            else if (dst.TypeCode == NPTypeCode.Boolean)
            {
                // string_to_bool's NA rule, decided once for the descriptor.
                byte naTruth = p.HasNAObject && !p.HasStringNA
                    ? (p.HasNanNA ? (byte)1 : (byte)0)
                    : (byte)(p.DefaultString.Length == 0 ? 1 : 0);
                run = (s, ss, d, ds, n) => ToBoolRun(s, ss, d, ds, n, allocator, naTruth);
            }
            else
            {
                NPTypeCode to = dst.TypeCode;
                run = (s, ss, d, ds, n) => ToNumberRun(s, ss, d, ds, n, to, allocator);
            }

            allocator.Acquire();
            try
            {
                Walk(dst, src, run);
            }
            finally
            {
                allocator.Release();
            }
        }

        /// <summary>Loads a cell's text (the allocator is locked), raising NumPy's text for a corrupt cell.</summary>
        /// <param name="cell">The cell.</param>
        /// <param name="allocator">Its allocator.</param>
        /// <param name="buf">Receives the text's first byte.</param>
        /// <param name="size">Receives its size.</param>
        /// <param name="context">The operation NumPy names (<c>in string to unicode cast</c>).</param>
        /// <returns>True for an NA element.</returns>
        /// <exception cref="OutOfMemoryException"><c>Failed to load string in {context}</c>.</exception>
        private static bool Load(byte* cell, VStringAllocator allocator, out byte* buf, out long size, string context)
        {
            int isNull = allocator.Load(cell, out buf, out size);
            if (isNull < 0)
                throw new OutOfMemoryException($"Failed to load string in {context}");
            return isNull == 1;
        }

        /// <summary>
        ///     <c>string_to_unicode</c>: each element decoded into at most <paramref name="chars"/> code points (a longer text is
        ///     cut at a code point), the rest NUL-filled; an NA element is <paramref name="naText"/>.
        /// </summary>
        /// <param name="s">The first source cell.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="d">The first <c>U</c> element.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="chars">The destination width in code points.</param>
        /// <param name="allocator">The source's allocator (locked).</param>
        /// <param name="naText">The UTF-8 text an NA element reads as.</param>
        /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
        private static void ToUnicodeRun(byte* s, long ss, byte* d, long ds, long n, int chars, VStringAllocator allocator, byte[] naText)
        {
            fixed (byte* na = naText)
            {
                for (long i = 0; i < n; i++, s += ss, d += ds)
                {
                    byte* buf;
                    long size;
                    if (Load(s, allocator, out buf, out size, "in string to unicode cast"))
                    {
                        buf = na;
                        size = naText.Length;
                    }
                    uint* u = (uint*)d;
                    int k = 0;
                    long pos = 0;
                    while (k < chars && pos < size)
                    {
                        System.Text.Rune.DecodeFromUtf8(new ReadOnlySpan<byte>(buf + pos, (int)Math.Min(size - pos, 4)), out var rune, out int used);
                        u[k++] = (uint)rune.Value;
                        pos += used;
                    }
                    for (; k < chars; k++)
                        u[k] = 0;
                }
            }
        }

        /// <summary>
        ///     <c>string_to_bytes</c>: each element's UTF-8 text checked to be ASCII (the first byte above 127 is CPython's
        ///     <c>'ascii' codec can't encode character …</c>), then its first <paramref name="chars"/> bytes, NUL-padded; an NA
        ///     element is <paramref name="naText"/>.
        /// </summary>
        /// <param name="s">The first source cell.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="d">The first <c>S</c> element.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="chars">The destination width in bytes.</param>
        /// <param name="allocator">The source's allocator (locked).</param>
        /// <param name="naText">The UTF-8 text an NA element reads as.</param>
        /// <exception cref="UnicodeEncodeError">A text that is not ASCII.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
        private static void ToBytesRun(byte* s, long ss, byte* d, long ds, long n, int chars, VStringAllocator allocator, byte[] naText)
        {
            fixed (byte* na = naText)
            {
                for (long i = 0; i < n; i++, s += ss, d += ds)
                {
                    byte* buf;
                    long size;
                    if (Load(s, allocator, out buf, out size, "in string to bytes cast"))
                    {
                        buf = na;
                        size = naText.Length;
                    }
                    for (long b = 0; b < size; b++)
                    {
                        if (buf[b] > 127)
                        {
                            // NumPy raises UnicodeEncodeError(ascii, str, i, i + 1): every byte before i is ASCII, so the byte
                            // index is also the code-point index of the character CPython quotes.
                            System.Text.Rune.DecodeFromUtf8(new ReadOnlySpan<byte>(buf + b, (int)Math.Min(size - b, 4)), out var rune, out _);
                            throw NDStringCodec.EncodeErrorFor((uint)rune.Value, b, b + 1);
                        }
                    }
                    long copy = Math.Min(size, chars);
                    Buffer.MemoryCopy(buf, d, chars, copy);
                    for (long b = copy; b < chars; b++)
                        d[b] = 0;
                }
            }
        }

        /// <summary><c>string_to_bool</c>: <c>''</c> is False, any other text True, an NA element <paramref name="naTruth"/>.</summary>
        /// <param name="s">The first source cell.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="d">The first bool.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="allocator">The source's allocator (locked).</param>
        /// <param name="naTruth">The truth value of an NA element (1 or 0).</param>
        /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
        private static void ToBoolRun(byte* s, long ss, byte* d, long ds, long n, VStringAllocator allocator, byte naTruth)
        {
            for (long i = 0; i < n; i++, s += ss, d += ds)
            {
                if (Load(s, allocator, out _, out long size, "string to bool cast"))
                    *d = naTruth;
                else
                    *d = size == 0 ? (byte)0 : (byte)1;
            }
        }

        /// <summary>
        ///     <c>string_to_int</c> / <c>string_to_float</c> / <c>string_to_complex_float</c>: each element parsed
        ///     (<see cref="NDTextParse.ParseVStringElement"/>) and cast from its family's canonical value into the destination;
        ///     an NA element is NumPy's <c>ValueError</c>.
        /// </summary>
        /// <param name="s">The first source cell.</param>
        /// <param name="ss">The source byte stride.</param>
        /// <param name="d">The first number.</param>
        /// <param name="ds">The destination byte stride.</param>
        /// <param name="n">The number of elements.</param>
        /// <param name="to">The numeric destination dtype.</param>
        /// <param name="allocator">The source's allocator (locked).</param>
        /// <exception cref="ValueError">An NA element, or a text that is not a number of the destination's kind.</exception>
        /// <exception cref="OverflowException">An integer text outside the destination.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt cell.</exception>
        private static void ToNumberRun(byte* s, long ss, byte* d, long ds, long n, NPTypeCode to, VStringAllocator allocator)
        {
            var canonType = NDTextParse.CanonicalType(NDTextParse.FamilyOf(to));
            var cast = canonType == to ? null : CastLoop(canonType, to);
            int canonSize = canonType.SizeOf();
            byte* canon = stackalloc byte[16];
            for (long i = 0; i < n; i++, s += ss, d += ds)
            {
                if (Load(s, allocator, out byte* buf, out long size, "string to number cast"))
                    throw new ValueError(MissingDataText);
                if (size > int.MaxValue)
                    throw new OutOfMemoryException($"A StringDType element of {size} bytes is longer than a .NET span can be.");
                NDTextParse.ParseVStringElement(new ReadOnlySpan<byte>(buf, (int)size), to, canon);
                // Written as it goes (an ArrayMethod loop, no commit blocks): the cast from the canonical value is exact for
                // a range-checked integer and rounds once for float32 / float16 (npy_double_to_half).
                if (cast is null)
                    Buffer.MemoryCopy(canon, d, canonSize, canonSize);
                else
                    cast(canon, canonSize, d, ds, 1);
            }
        }

        // ---------------------------------------------------------------------------------------------
        // The walk
        // ---------------------------------------------------------------------------------------------

        /// <summary>One innermost run: source pointer and byte stride, destination pointer and byte stride, length.</summary>
        /// <param name="src">The run's first source element.</param>
        /// <param name="srcStride">The source byte stride.</param>
        /// <param name="dst">The run's first destination element.</param>
        /// <param name="dstStride">The destination byte stride.</param>
        /// <param name="count">The run length.</param>
        internal delegate void RunAction(byte* src, long srcStride, byte* dst, long dstStride, long count);

        /// <summary>
        ///     NumPy's raw assignment walk: the axes sorted by the destination's strides (<see cref="RawArrayIter.PrepareThree"/>),
        ///     the source broadcast (a higher-rank source's extra leading axes are extent 1 and are dropped), one
        ///     <paramref name="run"/> call per innermost run.
        /// </summary>
        /// <param name="dst">The destination.</param>
        /// <param name="src">The source.</param>
        /// <param name="run">The per-run converter.</param>
        /// <remarks>
        ///     The walk order is observable for <c>StringDType</c> destinations: each duplicated arena string is appended to the
        ///     destination's arena in walk order, so the cells' offsets (<c>tobytes()</c>) are NumPy's only when the walk is
        ///     NumPy's — which is why the <c>StringDType</c> copy core uses this walk too (<see cref="NDVStringCopy"/>).
        /// </remarks>
        internal static void Walk(UnmanagedStorage dst, UnmanagedStorage src, RunAction run)
        {
            var ds = dst.Shape;
            int ndim = ds.NDim;
            int rank = Math.Max(ndim, 1);
            long* shape = stackalloc long[rank];
            long* sa = stackalloc long[rank];
            long* sb = stackalloc long[rank];
            long* sc = stackalloc long[rank];
            long* oShape = stackalloc long[rank];
            long* oa = stackalloc long[rank];
            long* ob = stackalloc long[rank];
            long* oc = stackalloc long[rank];

            int dItem = dst.ItemBytes, sItem = src.ItemBytes;
            for (int k = 0; k < ndim; k++)
            {
                shape[k] = ds.dimensions[k];
                sa[k] = ds.strides[k] * dItem;
            }
            var ss = src.Shape;
            int lead = ndim - ss.NDim;
            for (int k = 0; k < ndim; k++)
            {
                int sd = k - lead;
                sb[k] = sd < 0 || ss.dimensions[sd] == 1 ? 0 : ss.strides[sd] * sItem;
                sc[k] = sb[k]; // the third operand repeats the source (it coalesces exactly when the source does)
            }
            byte* a = dst.Address + ds.offset * dItem;
            byte* b = src.Address + ss.offset * sItem;
            byte* c = b;

            int nAxes = RawArrayIter.PrepareThree(ndim, shape, ref a, sa, ref b, sb, ref c, sc, oShape, oa, ob, oc);
            if (oShape[0] == 0)
                return;

            long* coord = stackalloc long[nAxes];
            for (int k = 0; k < nAxes; k++)
                coord[k] = 0;
            long inner = oShape[0];
            while (true)
            {
                run(b, ob[0], a, oa[0], inner);
                int axis = 1;
                for (; axis < nAxes; axis++)
                {
                    a += oa[axis];
                    b += ob[axis];
                    if (++coord[axis] < oShape[axis])
                        break;
                    a -= oa[axis] * oShape[axis];
                    b -= ob[axis] * oShape[axis];
                    coord[axis] = 0;
                }
                if (axis >= nAxes)
                    break;
            }
            GC.KeepAlive(dst);
            GC.KeepAlive(src);
        }

        // ---------------------------------------------------------------------------------------------
        // Helpers
        // ---------------------------------------------------------------------------------------------

        /// <summary>The house cast kernel between two numeric dtypes (the NumPy-exact cast table), or a scalar loop where
        ///     dynamic code is unavailable.</summary>
        /// <param name="from">The source dtype.</param>
        /// <param name="to">The destination dtype.</param>
        /// <returns>The kernel.</returns>
        private static DirectILKernelGenerator.InnerCastLoop CastLoop(NPTypeCode from, NPTypeCode to)
            => DirectILKernelGenerator.TryGetInnerCastKernel(from, to) ?? ((sp, sst, dp, dst, count) =>
            {
                byte* sb = (byte*)sp, db = (byte*)dp;
                for (long i = 0; i < count; i++, sb += sst, db += dst)
                    NDIterCasting.ConvertValue(sb, db, from, to);
            });

        /// <summary>Encodes one valid code point as UTF-8 (<c>ucs4_code_to_utf8_char</c>).</summary>
        /// <param name="cp">The code point (not a surrogate, at most U+10FFFF).</param>
        /// <param name="dst">Receives 1 to 4 bytes.</param>
        /// <returns>The number of bytes written.</returns>
        private static int EncodeCodePoint(uint cp, byte* dst)
        {
            if (cp <= 0x7F)
            {
                dst[0] = (byte)cp;
                return 1;
            }
            if (cp <= 0x7FF)
            {
                dst[0] = (byte)(0xC0 | (cp >> 6));
                dst[1] = (byte)(0x80 | (cp & 0x3F));
                return 2;
            }
            if (cp <= 0xFFFF)
            {
                dst[0] = (byte)(0xE0 | (cp >> 12));
                dst[1] = (byte)(0x80 | ((cp >> 6) & 0x3F));
                dst[2] = (byte)(0x80 | (cp & 0x3F));
                return 3;
            }
            dst[0] = (byte)(0xF0 | (cp >> 18));
            dst[1] = (byte)(0x80 | ((cp >> 12) & 0x3F));
            dst[2] = (byte)(0x80 | ((cp >> 6) & 0x3F));
            dst[3] = (byte)(0x80 | (cp & 0x3F));
            return 4;
        }

        /// <summary>Whether two shapes have identical dimensions.</summary>
        /// <param name="a">The first shape.</param>
        /// <param name="b">The second shape.</param>
        /// <returns>True for the same rank and extents.</returns>
        private static bool SameDimensions(Shape a, Shape b)
        {
            if (a.NDim != b.NDim)
                return false;
            for (int i = 0; i < a.NDim; i++)
                if (a.dimensions[i] != b.dimensions[i])
                    return false;
            return true;
        }
    }
}
