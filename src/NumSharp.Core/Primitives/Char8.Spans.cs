// Span-level primitives for ReadOnlySpan<Char8> / Span<Char8>.
// Zero-copy wrappers over ReadOnlySpan<byte> operations.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp
{
    public static class Char8SpanExtensions
    {
        // ========================================================================
        // Search
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int IndexOf(this ReadOnlySpan<Char8> span, Char8 value)
            => MemoryMarshal.Cast<Char8, byte>(span).IndexOf(value.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int LastIndexOf(this ReadOnlySpan<Char8> span, Char8 value)
            => MemoryMarshal.Cast<Char8, byte>(span).LastIndexOf(value.Value);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int IndexOf(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> value)
            => MemoryMarshal.Cast<Char8, byte>(span).IndexOf(MemoryMarshal.Cast<Char8, byte>(value));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int LastIndexOf(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> value)
            => MemoryMarshal.Cast<Char8, byte>(span).LastIndexOf(MemoryMarshal.Cast<Char8, byte>(value));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Contains(this ReadOnlySpan<Char8> span, Char8 value)
            => MemoryMarshal.Cast<Char8, byte>(span).IndexOf(value.Value) >= 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool Contains(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> value)
            => span.IndexOf(value) >= 0;

        public static int IndexOfAny(this ReadOnlySpan<Char8> span, Char8 a, Char8 b)
            => MemoryMarshal.Cast<Char8, byte>(span).IndexOfAny(a.Value, b.Value);

        public static int IndexOfAny(this ReadOnlySpan<Char8> span, Char8 a, Char8 b, Char8 c)
            => MemoryMarshal.Cast<Char8, byte>(span).IndexOfAny(a.Value, b.Value, c.Value);

        public static int IndexOfAny(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> values)
            => MemoryMarshal.Cast<Char8, byte>(span).IndexOfAny(MemoryMarshal.Cast<Char8, byte>(values));

        public static int Count(this ReadOnlySpan<Char8> span, Char8 value)
        {
            var bytes = MemoryMarshal.Cast<Char8, byte>(span);
            byte target = value.Value;
            int count = 0;
            for (int i = 0; i < bytes.Length; i++)
                if (bytes[i] == target) count++;
            return count;
        }

        public static int Count(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> value)
        {
            if (value.Length == 0) return span.Length + 1;   // Python: b.count(b'') == len(b) + 1
            int count = 0, from = 0;
            while (true)
            {
                int idx = span.Slice(from).IndexOf(value);
                if (idx < 0) break;
                count++;
                from += idx + value.Length;
                if (from > span.Length) break;
            }
            return count;
        }

        // ========================================================================
        // Equality / comparison
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SequenceEqual(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> other)
            => MemoryMarshal.Cast<Char8, byte>(span).SequenceEqual(MemoryMarshal.Cast<Char8, byte>(other));

        /// <summary>ASCII case-insensitive equality. Non-ASCII bytes compare as-is.</summary>
        public static bool EqualsIgnoreCaseAscii(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> other)
        {
            if (span.Length != other.Length) return false;
            var a = MemoryMarshal.Cast<Char8, byte>(span);
            var b = MemoryMarshal.Cast<Char8, byte>(other);
            for (int i = 0; i < a.Length; i++)
            {
                byte ba = a[i], bb = b[i];
                if (ba == bb) continue;
                // ASCII letters: flipping bit 5 maps 'A'↔'a'
                if ((uint)((ba | 0x20) - 'a') <= ('z' - 'a') &&
                    (uint)((bb | 0x20) - 'a') <= ('z' - 'a') &&
                    (ba | 0x20) == (bb | 0x20))
                    continue;
                return false;
            }
            return true;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool StartsWith(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> value)
            => MemoryMarshal.Cast<Char8, byte>(span).StartsWith(MemoryMarshal.Cast<Char8, byte>(value));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool EndsWith(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> value)
            => MemoryMarshal.Cast<Char8, byte>(span).EndsWith(MemoryMarshal.Cast<Char8, byte>(value));

        public static bool StartsWith(this ReadOnlySpan<Char8> span, Char8 value)
            => span.Length > 0 && span[0].Value == value.Value;

        public static bool EndsWith(this ReadOnlySpan<Char8> span, Char8 value)
            => span.Length > 0 && span[span.Length - 1].Value == value.Value;

        public static int CompareTo(this ReadOnlySpan<Char8> span, ReadOnlySpan<Char8> other)
        {
            int min = span.Length < other.Length ? span.Length : other.Length;
            var a = MemoryMarshal.Cast<Char8, byte>(span);
            var b = MemoryMarshal.Cast<Char8, byte>(other);
            for (int i = 0; i < min; i++)
            {
                int diff = a[i] - b[i];
                if (diff != 0) return diff < 0 ? -1 : 1;
            }
            return span.Length.CompareTo(other.Length);
        }

        // ========================================================================
        // String interop without materialization
        // ========================================================================

        /// <summary>Compares this span to a string, assuming Latin-1 decoding of the bytes.</summary>
        public static bool EqualsString(this ReadOnlySpan<Char8> span, string other)
        {
            if (other is null) return false;
            if (span.Length != other.Length) return false;
            for (int i = 0; i < span.Length; i++)
                if ((char)span[i].Value != other[i]) return false;
            return true;
        }

        public static bool StartsWithString(this ReadOnlySpan<Char8> span, string prefix)
        {
            if (prefix is null) return false;
            if (prefix.Length > span.Length) return false;
            for (int i = 0; i < prefix.Length; i++)
                if ((char)span[i].Value != prefix[i]) return false;
            return true;
        }

        public static bool EndsWithString(this ReadOnlySpan<Char8> span, string suffix)
        {
            if (suffix is null) return false;
            if (suffix.Length > span.Length) return false;
            int offset = span.Length - suffix.Length;
            for (int i = 0; i < suffix.Length; i++)
                if ((char)span[offset + i].Value != suffix[i]) return false;
            return true;
        }
    }

    public readonly partial struct Char8
    {
        // ========================================================================
        // UTF-8 byte classification (lets callers detect UTF-8 structure even
        // though Char8 itself doesn't encode a full UTF-8 scalar)
        // ========================================================================

        /// <summary>True for ASCII bytes (0x00..0x7F) — single-byte UTF-8 sequences.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsUtf8SingleByte(Char8 c) => c.m_value <= 0x7F;

        /// <summary>True for UTF-8 continuation bytes (0x80..0xBF).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsUtf8ContinuationByte(Char8 c) => (c.m_value & 0xC0) == 0x80;

        /// <summary>True for UTF-8 lead bytes of multi-byte sequences (0xC2..0xF4).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsUtf8LeadByte(Char8 c) => (uint)(c.m_value - 0xC2) <= (0xF4 - 0xC2);

        /// <summary>True for bytes that are never valid in UTF-8 (0xC0, 0xC1, 0xF5..0xFF).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsUtf8Invalid(Char8 c) => c.m_value == 0xC0 || c.m_value == 0xC1 || c.m_value >= 0xF5;

        /// <summary>
        /// Returns the number of bytes in the UTF-8 sequence whose lead byte is <paramref name="c"/>.
        /// Returns 1 for ASCII, 2/3/4 for valid multi-byte leads, 0 for continuation or invalid bytes.
        /// </summary>
        public static int GetUtf8SequenceLength(Char8 c)
        {
            byte b = c.m_value;
            if (b <= 0x7F) return 1;
            if (b < 0xC2) return 0;              // continuation or invalid
            if (b < 0xE0) return 2;              // 0xC2..0xDF → 2 bytes
            if (b < 0xF0) return 3;              // 0xE0..0xEF → 3 bytes
            if (b < 0xF5) return 4;              // 0xF0..0xF4 → 4 bytes
            return 0;                             // 0xF5..0xFF → invalid
        }
    }
}
