using System;
using System.Collections;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Printing;
using NumSharp.Utilities;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The element-level helpers of the fixed-width string lane that sit between the codec
    ///     (<see cref="NDStringCodec"/>, one value ↔ one element's bytes) and the storage (<see cref="UnmanagedStorage"/>):
    ///     which C# values a string element accepts, how one value is replicated across a buffer, and which descriptor an
    ///     allocation uses.
    /// </summary>
    /// <remarks>
    ///     <b>Accepted values</b> follow NumPy's <c>STRING_setitem</c> / <c>UNICODE_setitem</c>: text (<see cref="string"/>,
    ///     <see cref="char"/>), bytes (<c>byte[]</c>, <see cref="ReadOnlyMemory{T}"/> of <see cref="byte"/>), a 0-d string
    ///     array (NumPy's "convert to scalar and retry"), and Python's <c>None</c> (C# <see langword="null"/>), which NumPy
    ///     stores as its <c>str()</c> — the text <c>"None"</c>. A sequence raises NumPy's
    ///     <c>setting an array element with a sequence</c>. A number stores as its <c>str()</c> (<c>a[0] = 5</c> gives
    ///     <c>'5'</c>), truncated to the width: the C# integers, <see cref="bool"/>, <see cref="float"/>, <see cref="double"/>
    ///     and <see cref="System.Numerics.Complex"/> are Python scalars (a <see cref="float"/> is a Python float, so it prints
    ///     its double value: <c>0.1f</c> is <c>'0.10000000149011612'</c>), <see cref="Half"/> and <see cref="decimal"/> are
    ///     NumPy scalars (<c>np.float16(0.1)</c> is <c>'0.1'</c>), a numeric 0-d array prints as its own dtype's scalar.
    /// </remarks>
    internal static unsafe class NDStringStorage
    {
        /// <summary>
        ///     The descriptor an ARRAY of <paramref name="descr"/> is allocated with — NumPy's
        ///     <c>PyArray_NewFromDescr_int</c> rule: an unsized string request (<c>S0</c>, <c>U0</c>, <c>np.str_</c>)
        ///     becomes ONE character (<c>S1</c>/<c>U1</c>), and a non-native byte order is canonicalised (NumSharp's
        ///     buffers are host-endian).
        /// </summary>
        /// <param name="descr">A string descriptor.</param>
        /// <returns>A sized, native descriptor of the same class.</returns>
        /// <exception cref="ArgumentException"><paramref name="descr"/> is not a string descriptor.</exception>
        public static DType AllocationDescr(DType descr)
        {
            if (descr?.Meta is not StringDTypeMeta meta)
                throw new ArgumentException($"{descr?.ToString(true) ?? "null"} is not a fixed-width string dtype.", nameof(descr));
            if (descr.itemsize == 0)
                return meta.DefaultDescr();
            return descr.isnative ? descr : meta.EnsureCanonical(descr);
        }

        /// <summary>Whether <paramref name="descr"/> is a fixed-width string descriptor (<c>S</c> or <c>U</c>).</summary>
        /// <param name="descr">Any descriptor (null is false).</param>
        /// <returns>True for the string classes.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsString(DType descr) => descr?.Meta is StringDTypeMeta;

        /// <summary>
        ///     Stores one C# value into one string element — the dispatch in front of the codec, NumPy's <c>setitem</c>.
        /// </summary>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="unicode">True for a <c>U</c> element (UCS-4 units), false for <c>S</c> (bytes).</param>
        /// <param name="chars">The element's width in characters.</param>
        /// <param name="value">The value (see the class remarks for what is accepted).</param>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        /// <exception cref="ValueError">A sequence (<c>setting an array element with a sequence</c>).</exception>
        /// <exception cref="NotSupportedException">A value that is neither text, bytes, None, a number nor a 0-d array (a C#
        ///     object has no Python <c>str()</c> to store).</exception>
        public static void WriteElement(byte* dst, bool unicode, int chars, object value)
        {
            switch (value)
            {
                case string s:
                    WriteText(dst, unicode, chars, s.AsSpan());
                    return;
                case char c:
                    // A C# char is one UTF-16 unit — Python's one-character str.
                    WriteText(dst, unicode, chars, new ReadOnlySpan<char>(&c, 1));
                    return;
                case null:
                    // NumPy stores str(None) for None (`a[0] = None` gives 'None'), in both string classes.
                    WriteText(dst, unicode, chars, "None".AsSpan());
                    return;
                case byte[] b:
                    WriteBinary(dst, unicode, chars, b);
                    return;
                case ReadOnlyMemory<byte> rom:
                    WriteBinary(dst, unicode, chars, rom.Span);
                    return;
                case Memory<byte> mem:
                    WriteBinary(dst, unicode, chars, mem.Span);
                    return;
                case NDArray nd when nd.ndim == 0:
                    // NumPy's convert_to_scalar_and_retry: a 0-d array stores as its element — a number as ITS dtype's
                    // scalar str() (a float32 0-d array prints float32's shortest digits).
                    if (nd.typecode != NPTypeCode.String)
                    {
                        NDTextConversion.FormatRun((byte*)nd.Address + nd.Shape.offset * (long)nd.typecode.SizeOf(), 0,
                            nd.typecode, dst, 0, unicode, chars, 1);
                        GC.KeepAlive(nd);
                        return;
                    }
                    WriteElement(dst, unicode, chars, nd.Storage.GetAtIndex(0));
                    return;
                case IEnumerable:
                    // PySequence_NoString_Check: a list / array / NDArray of rank >= 1 is not one element.
                    throw new ValueError("setting an array element with a sequence");
                default:
                    WriteNumber(dst, unicode, chars, value);
                    return;
            }
        }

        /// <summary>
        ///     The length of the text <see cref="WriteElement"/> stores for a number (its <c>str()</c>, before truncation) —
        ///     the width an UNSIZED string dtype discovers from it (<c>np.array([1, 22], dtype='U')</c> is <c>&lt;U2</c>).
        /// </summary>
        /// <param name="value">The number (boxed).</param>
        /// <returns>The number of characters.</returns>
        /// <exception cref="NotSupportedException"><paramref name="value"/> is not a number NumSharp knows.</exception>
        internal static int StrLength(object value)
        {
            if (value is BigInteger big)
                return big.ToString(CultureInfo.InvariantCulture).Length;
            Span<char> buf = stackalloc char[NDTextFormat.MaxTextLength];
            return FormatNumber(value, buf);
        }

        /// <summary>
        ///     Stores a number as its Python / NumPy <c>str()</c> (see the class remarks for which C# type is which), truncated
        ///     to the element width and NUL-padded — <c>UNICODE_setitem</c> / <c>STRING_setitem</c> calling
        ///     <c>PyObject_Str</c>.
        /// </summary>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="unicode">True for <c>U</c>.</param>
        /// <param name="chars">The element width in characters.</param>
        /// <param name="value">The number (boxed).</param>
        /// <exception cref="NotSupportedException"><paramref name="value"/> is not a number NumSharp knows.</exception>
        private static void WriteNumber(byte* dst, bool unicode, int chars, object value)
        {
            if (value is BigInteger big)
            {
                // A Python int of any size: its decimal digits (a value past 64 bits can exceed MaxTextLength).
                NDTextFormat.Store(big.ToString(CultureInfo.InvariantCulture).AsSpan(), dst, unicode, chars);
                return;
            }
            Span<char> buf = stackalloc char[NDTextFormat.MaxTextLength];
            int n = FormatNumber(value, buf);
            NDTextFormat.Store(buf.Slice(0, n), dst, unicode, chars);
        }

        /// <summary>
        ///     The <c>str()</c> of a number (not a <see cref="BigInteger"/>) into <paramref name="buf"/>: the Python scalars by
        ///     Python's rules, <see cref="Half"/> and <see cref="decimal"/> by their NumPy scalar's (see the class remarks).
        /// </summary>
        /// <param name="value">The number (boxed).</param>
        /// <param name="buf">Receives the text; room for <see cref="NDTextFormat.MaxTextLength"/> characters.</param>
        /// <returns>The text's length.</returns>
        /// <exception cref="NotSupportedException"><paramref name="value"/> is not a number NumSharp knows.</exception>
        private static int FormatNumber(object value, Span<char> buf)
        {
            switch (value)
            {
                case bool b:
                    return NDTextFormat.FormatBool(b, buf);
                case sbyte or short or int or long:
                    return NDTextFormat.FormatInt64(Convert.ToInt64(value, CultureInfo.InvariantCulture), buf);
                case byte or ushort or uint or ulong:
                    return NDTextFormat.FormatUInt64(Convert.ToUInt64(value, CultureInfo.InvariantCulture), buf);
                case float f:
                    // A Python float: the float32 value widened, printed by the double's repr.
                    return NDTextFormat.FormatFloat(f, FloatKind.Double, buf);
                case double d:
                    return NDTextFormat.FormatFloat(d, FloatKind.Double, buf);
                case Half h:
                    // np.float16: its own shortest digits and positional window.
                    return NDTextFormat.FormatFloat((double)h, FloatKind.Half, buf);
                case decimal m:
                    // NumSharp's decimal prints as a float64 scalar (ArrayFormatter.ScalarStr's rule).
                    return NDTextFormat.FormatFloat(Converts.ToDouble(m), FloatKind.Double, buf);
                case Complex c:
                    return NDTextFormat.FormatComplex(c, buf);
                default:
                    throw new NotSupportedException(
                        $"Cannot store a {value.GetType().Name} value in a fixed-width string array: a string element takes text " +
                        "(string, char, byte[]), a number (stored as its str()), None, or a 0-d array.");
            }
        }

        /// <summary>Writes text into one element (ASCII-encoding for <c>S</c>, UCS-4 for <c>U</c>).</summary>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="unicode">True for <c>U</c>.</param>
        /// <param name="chars">The element width in characters.</param>
        /// <param name="text">The text.</param>
        /// <exception cref="UnicodeEncodeError">Non-ASCII text into <c>S</c>.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteText(byte* dst, bool unicode, int chars, ReadOnlySpan<char> text)
        {
            if (unicode)
                NDStringCodec.WriteUcs4(text, (uint*)dst, chars);
            else
                NDStringCodec.WriteAscii(text, dst, chars);
        }

        /// <summary>Writes bytes into one element (verbatim for <c>S</c>, ASCII-decoded for <c>U</c>).</summary>
        /// <param name="dst">The element's first byte.</param>
        /// <param name="unicode">True for <c>U</c>.</param>
        /// <param name="chars">The element width in characters.</param>
        /// <param name="bytes">The bytes.</param>
        /// <exception cref="UnicodeDecodeError">Non-ASCII bytes into <c>U</c>.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WriteBinary(byte* dst, bool unicode, int chars, ReadOnlySpan<byte> bytes)
        {
            if (unicode)
                NDStringCodec.WriteUcs4FromAscii(bytes, (uint*)dst, chars);
            else
                NDStringCodec.WriteBytes(bytes, dst, chars);
        }

        /// <summary>
        ///     Copies the first <paramref name="itemsize"/> bytes of <paramref name="buffer"/> into every following element
        ///     by doubling block copies (1, 2, 4, … elements per copy), so a fill of N elements is O(log N) memcpy calls.
        /// </summary>
        /// <param name="buffer">The buffer; element 0 already holds the value.</param>
        /// <param name="itemsize">Bytes per element.</param>
        /// <param name="count">Total elements in the buffer.</param>
        public static void ReplicateFirstElement(byte* buffer, int itemsize, long count)
        {
            long total = count * itemsize;
            long filled = itemsize;
            while (filled < total)
            {
                long chunk = Math.Min(filled, total - filled);
                Buffer.MemoryCopy(buffer, buffer + filled, chunk, chunk);
                filled += chunk;
            }
        }
    }
}
