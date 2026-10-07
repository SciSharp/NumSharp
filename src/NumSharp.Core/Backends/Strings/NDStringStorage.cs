using System;
using System.Collections;
using System.Runtime.CompilerServices;

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
    ///     <c>setting an array element with a sequence</c>. Numbers (<c>str(5)</c> → <c>'5'</c>) are NumPy's number→text
    ///     conversion, which belongs to the computational phase; they raise <see cref="NotSupportedException"/> naming that.
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
        /// <exception cref="StringComputationNotSupportedException">A number or other non-text value (number→text is the computational phase).</exception>
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
                    // NumPy's convert_to_scalar_and_retry: a 0-d array stores as its element.
                    if (nd.typecode != NPTypeCode.String)
                        throw NumberToText(nd.dtype.ToString());
                    WriteElement(dst, unicode, chars, nd.Storage.GetAtIndex(0));
                    return;
                case IEnumerable:
                    // PySequence_NoString_Check: a list / array / NDArray of rank >= 1 is not one element.
                    throw new ValueError("setting an array element with a sequence");
                default:
                    throw NumberToText(value.GetType().Name);
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

        /// <summary>The refusal for a value NumPy would store as its <c>str()</c> — number→text is not part of this phase.</summary>
        /// <param name="what">The value's type name.</param>
        /// <returns>The exception (not thrown).</returns>
        internal static StringComputationNotSupportedException NumberToText(string what)
            => new StringComputationNotSupportedException(
                $"Cannot store a {what} value in a fixed-width string array: converting numbers to text (NumPy's str(value)) " +
                "is not part of NumSharp's string storage phase yet; pass the text itself (a string or byte[]).");

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
