using System;
using System.Buffers;
using System.Text;
using System.Text.Unicode;

namespace NumSharp.Backends.Strings
{
    /// <summary>
    ///     The UTF-8 codec of NEP 55 <c>StringDType</c> text: NumPy stores every <c>StringDType</c> element as UTF-8, so
    ///     each C# <see cref="string"/> crossing into such an array is ENCODED (strictly — a lone surrogate is CPython's
    ///     <c>UnicodeEncodeError</c>) and each <c>bytes</c> value is DECODED (strictly — an invalid sequence is CPython's
    ///     <c>UnicodeDecodeError</c>), with CPython 3.12's error texts byte for byte.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Fast path first, CPython's diagnosis second.</b> Validation and transcoding run on .NET's vectorized UTF-8
    ///     routines (<see cref="Utf8.IsValid(ReadOnlySpan{byte})"/>, <see cref="Utf8.FromUtf16"/>); only when they report
    ///     invalid data does this class walk the input the way CPython's decoder / encoder does, to name the first bad byte
    ///     range and the reason exactly as Python would (<c>invalid start byte</c>, <c>invalid continuation byte</c>,
    ///     <c>unexpected end of data</c>, <c>surrogates not allowed</c>). Valid input never pays for the diagnosis.
    ///     </para>
    ///     <para>
    ///     <b>Code points, not UTF-16 units.</b> A Python <c>str</c> is a sequence of code points, so an encode error's
    ///     position counts code points: a valid surrogate pair is ONE position, a lone surrogate is one position of its
    ///     own, and consecutive lone surrogates are reported as one range (<c>characters in position 0-1</c>), as CPython's
    ///     <c>strict</c> handler reports them.
    ///     </para>
    /// </remarks>
    internal static class NDUtf8
    {
        /// <summary>
        ///     The UTF-8 byte length of a C# string, refusing a lone surrogate the way Python's <c>str.encode('utf-8')</c> does.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <returns>The number of UTF-8 bytes.</returns>
        /// <exception cref="UnicodeEncodeError">The text holds a lone surrogate:
        ///     <c>'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed</c>.</exception>
        /// <exception cref="OverflowException">The UTF-8 form is longer than <see cref="int.MaxValue"/> bytes (a C# string
        ///     of over ~715 million three-byte characters).</exception>
        public static int GetByteCount(ReadOnlySpan<char> s)
        {
            // Pure ASCII (the common case) is one byte per unit; the vectorized check settles it.
            if (Ascii.IsValid(s))
                return s.Length;
            long count = 0;
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (c < 0x80)
                    count += 1;
                else if (c < 0x800)
                    count += 2;
                else if (char.IsHighSurrogate(c) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                {
                    count += 4;
                    i++;
                }
                else if (char.IsSurrogate(c))
                    throw EncodeError(s, i);
                else
                    count += 3;
            }
            if (count > int.MaxValue)
                throw new OverflowException("The UTF-8 form of the text is longer than 2 GB.");
            return (int)count;
        }

        /// <summary>
        ///     Encodes a C# string as UTF-8 into <paramref name="dst"/> (which must hold <see cref="GetByteCount"/> bytes).
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="dst">The destination, exactly <see cref="GetByteCount"/> bytes long or longer.</param>
        /// <returns>The number of bytes written.</returns>
        /// <exception cref="UnicodeEncodeError">The text holds a lone surrogate (CPython's text, see
        ///     <see cref="GetByteCount"/>).</exception>
        /// <exception cref="ArgumentException"><paramref name="dst"/> is too short.</exception>
        public static int Encode(ReadOnlySpan<char> s, Span<byte> dst)
        {
            var status = Utf8.FromUtf16(s, dst, out int read, out int written, replaceInvalidSequences: false);
            switch (status)
            {
                case OperationStatus.Done:
                    return written;
                case OperationStatus.InvalidData:
                    // FromUtf16 stopped AT the lone surrogate: report it (and the run that follows) as CPython does.
                    throw EncodeError(s, read);
                default:
                    throw new ArgumentException("The destination is too short for the UTF-8 form of the text.", nameof(dst));
            }
        }

        /// <summary>
        ///     Encodes a C# string as a new UTF-8 byte array (strict).
        /// </summary>
        /// <param name="s">The text.</param>
        /// <returns>The UTF-8 bytes.</returns>
        /// <exception cref="UnicodeEncodeError">The text holds a lone surrogate.</exception>
        public static byte[] Encode(string s)
        {
            var bytes = new byte[GetByteCount(s)];
            Encode(s, bytes);
            return bytes;
        }

        /// <summary>
        ///     Decodes UTF-8 bytes this library wrote itself (a <c>StringDType</c> cell's payload) into a C# string — no
        ///     validation, because a cell only ever holds text that was encoded strictly.
        /// </summary>
        /// <param name="utf8">Valid UTF-8.</param>
        /// <returns>The text.</returns>
        public static string DecodeTrusted(ReadOnlySpan<byte> utf8) => utf8.IsEmpty ? string.Empty : Encoding.UTF8.GetString(utf8);

        /// <summary>
        ///     Decodes external bytes (a Python <c>bytes</c> value, an <c>S</c> element) as strict UTF-8 — Python's
        ///     <c>bytes.decode('utf-8')</c>.
        /// </summary>
        /// <param name="bytes">The bytes.</param>
        /// <returns>The text.</returns>
        /// <exception cref="UnicodeDecodeError">The bytes are not valid UTF-8:
        ///     <c>'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte</c>.</exception>
        public static string DecodeStrict(ReadOnlySpan<byte> bytes)
        {
            ThrowIfInvalid(bytes);
            return DecodeTrusted(bytes);
        }

        /// <summary>
        ///     Validates external bytes as UTF-8 and raises CPython's <c>UnicodeDecodeError</c> for the first bad sequence.
        /// </summary>
        /// <param name="bytes">The bytes.</param>
        /// <exception cref="UnicodeDecodeError">The bytes are not valid UTF-8 (CPython's range and reason).</exception>
        public static void ThrowIfInvalid(ReadOnlySpan<byte> bytes)
        {
            if (Utf8.IsValid(bytes))
                return;
            FindDecodeError(bytes, out int start, out int end, out string reason);
            throw DecodeError(bytes, start, end, reason);
        }

        /// <summary>
        ///     CPython's UTF-8 decode error for a range of input bytes: one byte reads <c>byte 0x80 in position 0</c>, a range
        ///     <c>bytes in position 0-1</c>.
        /// </summary>
        /// <param name="bytes">The input.</param>
        /// <param name="start">The first byte of the bad range.</param>
        /// <param name="end">One past the last byte of the bad range.</param>
        /// <param name="reason">CPython's reason text.</param>
        /// <returns>The exception to throw.</returns>
        internal static UnicodeDecodeError DecodeError(ReadOnlySpan<byte> bytes, int start, int end, string reason)
            => new UnicodeDecodeError(end - start == 1
                ? $"'utf-8' codec can't decode byte 0x{bytes[start]:x2} in position {start}: {reason}"
                : $"'utf-8' codec can't decode bytes in position {start}-{end - 1}: {reason}");

        /// <summary>
        ///     CPython 3.12's strict UTF-8 decoder diagnosis (<c>stringlib/codecs.h</c>, <c>utf8_decode</c>,
        ///     non-incremental): the FIRST error's byte range and reason. Only called for input already known to be invalid.
        /// </summary>
        /// <param name="b">The bytes (known invalid).</param>
        /// <param name="start">Receives the first byte of the bad range.</param>
        /// <param name="end">Receives one past its last byte.</param>
        /// <param name="reason">Receives CPython's reason.</param>
        /// <remarks>
        ///     A byte that cannot start a sequence (0x80..0xC1, 0xF5..0xFF) is <c>invalid start byte</c> over that byte; a
        ///     continuation that is missing, out of range, or makes an overlong form, a surrogate (<c>ED A0..</c>) or a value
        ///     past U+10FFFF (<c>F4 90..</c>) is <c>invalid continuation byte</c> over the bytes BEFORE the bad one; a
        ///     sequence cut off by the end of the value is <c>unexpected end of data</c> over the bytes to the end — unless an
        ///     early byte already proves it invalid, which is reported as a continuation error.
        /// </remarks>
        private static void FindDecodeError(ReadOnlySpan<byte> b, out int start, out int end, out string reason)
        {
            int i = 0, n = b.Length;
            while (i < n)
            {
                uint ch = b[i];
                if (ch < 0x80)
                {
                    i++;
                    continue;
                }
                if (ch < 0xE0)
                {
                    if (ch < 0xC2)
                    {
                        Set(i, i + 1, "invalid start byte", out start, out end, out reason);
                        return;
                    }
                    if (n - i < 2)
                    {
                        Set(i, n, "unexpected end of data", out start, out end, out reason);
                        return;
                    }
                    if (!IsContinuation(b[i + 1]))
                    {
                        Set(i, i + 1, "invalid continuation byte", out start, out end, out reason);
                        return;
                    }
                    i += 2;
                    continue;
                }
                if (ch < 0xF0)
                {
                    if (n - i < 3)
                    {
                        if (n - i >= 2)
                        {
                            uint t2 = b[i + 1];
                            if (!IsContinuation(t2) || (t2 < 0xA0 ? ch == 0xE0 : ch == 0xED))
                            {
                                Set(i, i + 1, "invalid continuation byte", out start, out end, out reason);
                                return;
                            }
                        }
                        Set(i, n, "unexpected end of data", out start, out end, out reason);
                        return;
                    }
                    uint c2 = b[i + 1], c3 = b[i + 2];
                    if (!IsContinuation(c2) || (ch == 0xE0 && c2 < 0xA0) || (ch == 0xED && c2 >= 0xA0))
                    {
                        Set(i, i + 1, "invalid continuation byte", out start, out end, out reason);
                        return;
                    }
                    if (!IsContinuation(c3))
                    {
                        Set(i, i + 2, "invalid continuation byte", out start, out end, out reason);
                        return;
                    }
                    i += 3;
                    continue;
                }
                if (ch < 0xF5)
                {
                    if (n - i < 4)
                    {
                        if (n - i >= 2)
                        {
                            uint t2 = b[i + 1];
                            if (!IsContinuation(t2) || (t2 < 0x90 ? ch == 0xF0 : ch == 0xF4))
                            {
                                Set(i, i + 1, "invalid continuation byte", out start, out end, out reason);
                                return;
                            }
                            if (n - i >= 3 && !IsContinuation(b[i + 2]))
                            {
                                Set(i, i + 2, "invalid continuation byte", out start, out end, out reason);
                                return;
                            }
                        }
                        Set(i, n, "unexpected end of data", out start, out end, out reason);
                        return;
                    }
                    uint c2 = b[i + 1], c3 = b[i + 2], c4 = b[i + 3];
                    if (!IsContinuation(c2) || (ch == 0xF0 && c2 < 0x90) || (ch == 0xF4 && c2 >= 0x90))
                    {
                        Set(i, i + 1, "invalid continuation byte", out start, out end, out reason);
                        return;
                    }
                    if (!IsContinuation(c3))
                    {
                        Set(i, i + 2, "invalid continuation byte", out start, out end, out reason);
                        return;
                    }
                    if (!IsContinuation(c4))
                    {
                        Set(i, i + 3, "invalid continuation byte", out start, out end, out reason);
                        return;
                    }
                    i += 4;
                    continue;
                }
                Set(i, i + 1, "invalid start byte", out start, out end, out reason);
                return;
            }
            // Unreachable for invalid input; report the whole value rather than nothing if .NET and CPython ever disagree.
            Set(0, n, "invalid data", out start, out end, out reason);

            static bool IsContinuation(uint c) => (c & 0xC0) == 0x80;

            static void Set(int s, int e, string r, out int start, out int end, out string reason)
            {
                start = s;
                end = e;
                reason = r;
            }
        }

        /// <summary>
        ///     CPython's UTF-8 encode error at the lone surrogate starting at UTF-16 index <paramref name="unitIndex"/>: the
        ///     strict handler reports the whole run of consecutive lone surrogates, positions counted in code points.
        /// </summary>
        /// <param name="s">The text.</param>
        /// <param name="unitIndex">The UTF-16 index of the first lone surrogate.</param>
        /// <returns>The exception to throw.</returns>
        private static UnicodeEncodeError EncodeError(ReadOnlySpan<char> s, int unitIndex)
        {
            // The code-point position of unitIndex: every valid pair before it counts once.
            int position = 0;
            for (int i = 0; i < unitIndex; i++, position++)
                if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
                    i++;
            // The run of consecutive LONE surrogates (a valid pair would end it — it is a real character).
            int endUnit = unitIndex;
            while (endUnit < s.Length && char.IsSurrogate(s[endUnit])
                   && !(char.IsHighSurrogate(s[endUnit]) && endUnit + 1 < s.Length && char.IsLowSurrogate(s[endUnit + 1])))
                endUnit++;
            int runLength = endUnit - unitIndex;
            if (runLength == 1)
                return new UnicodeEncodeError($"'utf-8' codec can't encode character '\\u{(int)s[unitIndex]:x4}' in position {position}: surrogates not allowed");
            return new UnicodeEncodeError($"'utf-8' codec can't encode characters in position {position}-{position + runLength - 1}: surrogates not allowed");
        }
    }
}
