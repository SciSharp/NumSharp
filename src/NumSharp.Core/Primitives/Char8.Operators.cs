// Mixed-type operators, no-throw conversions, span reinterpret helpers.

using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NumSharp
{
    public readonly partial struct Char8
    {
        // ========================================================================
        // Char8 <-> char comparison operators
        // (widens Char8 to char via Latin-1)
        // ========================================================================

        public static bool operator ==(Char8 left, char right) => (char)left.m_value == right;
        public static bool operator !=(Char8 left, char right) => (char)left.m_value != right;
        public static bool operator <(Char8 left, char right) => (char)left.m_value < right;
        public static bool operator >(Char8 left, char right) => (char)left.m_value > right;
        public static bool operator <=(Char8 left, char right) => (char)left.m_value <= right;
        public static bool operator >=(Char8 left, char right) => (char)left.m_value >= right;

        public static bool operator ==(char left, Char8 right) => left == (char)right.m_value;
        public static bool operator !=(char left, Char8 right) => left != (char)right.m_value;
        public static bool operator <(char left, Char8 right) => left < (char)right.m_value;
        public static bool operator >(char left, Char8 right) => left > (char)right.m_value;
        public static bool operator <=(char left, Char8 right) => left <= (char)right.m_value;
        public static bool operator >=(char left, Char8 right) => left >= (char)right.m_value;

        // ========================================================================
        // Char8 <-> byte comparison operators
        // ========================================================================

        public static bool operator ==(Char8 left, byte right) => left.m_value == right;
        public static bool operator !=(Char8 left, byte right) => left.m_value != right;
        public static bool operator <(Char8 left, byte right) => left.m_value < right;
        public static bool operator >(Char8 left, byte right) => left.m_value > right;
        public static bool operator <=(Char8 left, byte right) => left.m_value <= right;
        public static bool operator >=(Char8 left, byte right) => left.m_value >= right;

        public static bool operator ==(byte left, Char8 right) => left == right.m_value;
        public static bool operator !=(byte left, Char8 right) => left != right.m_value;
        public static bool operator <(byte left, Char8 right) => left < right.m_value;
        public static bool operator >(byte left, Char8 right) => left > right.m_value;
        public static bool operator <=(byte left, Char8 right) => left <= right.m_value;
        public static bool operator >=(byte left, Char8 right) => left >= right.m_value;

        // ========================================================================
        // Char8 <-> int comparison operators
        // ========================================================================

        public static bool operator ==(Char8 left, int right) => left.m_value == right;
        public static bool operator !=(Char8 left, int right) => left.m_value != right;
        public static bool operator <(Char8 left, int right) => left.m_value < right;
        public static bool operator >(Char8 left, int right) => left.m_value > right;
        public static bool operator <=(Char8 left, int right) => left.m_value <= right;
        public static bool operator >=(Char8 left, int right) => left.m_value >= right;

        public static bool operator ==(int left, Char8 right) => left == right.m_value;
        public static bool operator !=(int left, Char8 right) => left != right.m_value;
        public static bool operator <(int left, Char8 right) => left < right.m_value;
        public static bool operator >(int left, Char8 right) => left > right.m_value;
        public static bool operator <=(int left, Char8 right) => left <= right.m_value;
        public static bool operator >=(int left, Char8 right) => left >= right.m_value;

        // ========================================================================
        // Arithmetic with int and byte
        // ========================================================================

        /// <summary>Adds an integer offset, wrapping at byte boundary.</summary>
        public static Char8 operator +(Char8 left, int right) => new Char8((byte)(left.m_value + right));
        public static Char8 operator +(int left, Char8 right) => new Char8((byte)(left + right.m_value));
        public static Char8 operator -(Char8 left, int right) => new Char8((byte)(left.m_value - right));

        public static Char8 operator +(Char8 left, byte right) => new Char8((byte)(left.m_value + right));
        public static Char8 operator +(byte left, Char8 right) => new Char8((byte)(left + right.m_value));
        public static Char8 operator -(Char8 left, byte right) => new Char8((byte)(left.m_value - right));

        // ========================================================================
        // Equals overloads for mixed-type equality (avoid box in Equals(object))
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(char other) => (char)m_value == other;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(byte other) => m_value == other;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Equals(int other) => m_value == other;

        // ========================================================================
        // No-throw conversions
        // ========================================================================

        /// <summary>Tries to narrow a <see cref="char"/> to a <see cref="Char8"/>. Returns false if the char is outside Latin-1.</summary>
        public static bool TryFromChar(char c, out Char8 result)
        {
            if ((uint)c > 0xFF) { result = default; return false; }
            result = new Char8((byte)c);
            return true;
        }

        /// <summary>Tries to narrow an <see cref="int"/> to a <see cref="Char8"/>. Returns false if outside [0, 255].</summary>
        public static bool TryFromInt32(int v, out Char8 result)
        {
            if ((uint)v > 0xFF) { result = default; return false; }
            result = new Char8((byte)v);
            return true;
        }

        // ========================================================================
        // Deconstruct
        // ========================================================================

        /// <summary>Deconstructs to the underlying byte. Enables pattern matching and assignment like <c>var (b) = char8;</c>.</summary>
        public void Deconstruct(out byte value) => value = m_value;

        // ========================================================================
        // Span reinterpret helpers (zero-copy via MemoryMarshal.Cast)
        // ========================================================================

        /// <summary>Reinterprets a <see cref="ReadOnlySpan{Char8}"/> as <see cref="ReadOnlySpan{Byte}"/>. Zero-copy.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ReadOnlySpan<byte> AsBytes(ReadOnlySpan<Char8> chars)
            => MemoryMarshal.Cast<Char8, byte>(chars);

        /// <summary>Reinterprets a <see cref="Span{Char8}"/> as <see cref="Span{Byte}"/>. Zero-copy.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<byte> AsBytes(Span<Char8> chars)
            => MemoryMarshal.Cast<Char8, byte>(chars);

        /// <summary>Reinterprets a <see cref="ReadOnlySpan{Byte}"/> as <see cref="ReadOnlySpan{Char8}"/>. Zero-copy.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ReadOnlySpan<Char8> AsChar8s(ReadOnlySpan<byte> bytes)
            => MemoryMarshal.Cast<byte, Char8>(bytes);

        /// <summary>Reinterprets a <see cref="Span{Byte}"/> as <see cref="Span{Char8}"/>. Zero-copy.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Span<Char8> AsChar8s(Span<byte> bytes)
            => MemoryMarshal.Cast<byte, Char8>(bytes);

        // ========================================================================
        // Formatting
        // ========================================================================

        /// <summary>Returns the hex representation <c>"0xNN"</c>.</summary>
        public string ToHex() => "0x" + m_value.ToString("X2");

        /// <summary>Returns the Python-style escaped representation — printable ASCII is returned as-is, recognized escapes use their literal form, all others use <c>\xNN</c>.</summary>
        public string ToEscaped()
        {
            return m_value switch
            {
                (byte)'\\' => "\\\\",
                (byte)'\'' => "\\'",
                (byte)'\"' => "\\\"",
                (byte)'\n' => "\\n",
                (byte)'\r' => "\\r",
                (byte)'\t' => "\\t",
                (byte)'\b' => "\\b",
                (byte)'\f' => "\\f",
                (byte)'\0' => "\\0",
                var b when b >= 0x20 && b <= 0x7E => ((char)b).ToString(),
                _ => "\\x" + m_value.ToString("x2")
            };
        }
    }
}
