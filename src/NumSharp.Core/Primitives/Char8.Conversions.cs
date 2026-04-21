// Conversions to and from all NumSharp-supported primitive dtypes.

using System;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public readonly partial struct Char8
    {
        // ========================================================================
        // Char8 -> other dtypes (widens or converts)
        // ========================================================================

        /// <summary>Returns <c>true</c> if the byte is non-zero (C convention).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool ToBoolean() => m_value != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public byte ToByte() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public sbyte ToSByte() => checked((sbyte)m_value);

        /// <summary>Returns the underlying byte as a <see cref="short"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public short ToInt16() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ushort ToUInt16() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int ToInt32() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public uint ToUInt32() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public long ToInt64() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ulong ToUInt64() => m_value;

        /// <summary>Widens to <see cref="char"/> via Latin-1 (0xE9 → 'é').</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public char ToChar() => (char)m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float ToSingle() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double ToDouble() => m_value;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public decimal ToDecimal() => m_value;

        // ========================================================================
        // FromXxx static factories (narrowing with overflow check)
        // ========================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromBoolean(bool b) => new Char8(b ? (byte)1 : (byte)0);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromByte(byte b) => new Char8(b);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromSByte(sbyte b)
        {
            if (b < 0) throw new OverflowException("Negative sbyte cannot be converted to Char8.");
            return new Char8((byte)b);
        }

        public static Char8 FromInt16(short v)
        {
            if ((uint)v > 0xFF) throw new OverflowException("Int16 value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromUInt16(ushort v)
        {
            if (v > 0xFF) throw new OverflowException("UInt16 value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        public static Char8 FromInt32(int v)
        {
            if ((uint)v > 0xFF) throw new OverflowException("Int32 value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        public static Char8 FromUInt32(uint v)
        {
            if (v > 0xFF) throw new OverflowException("UInt32 value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        public static Char8 FromInt64(long v)
        {
            if ((ulong)v > 0xFF) throw new OverflowException("Int64 value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        public static Char8 FromUInt64(ulong v)
        {
            if (v > 0xFF) throw new OverflowException("UInt64 value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        /// <summary>Narrows a <see cref="char"/> to <see cref="Char8"/>. Throws if the char is outside Latin-1 (> 0xFF).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromChar(char c)
        {
            if ((uint)c > 0xFF) throw new OverflowException("Char value " + (int)c + " exceeds Char8 max (0xFF).");
            return new Char8((byte)c);
        }

        public static Char8 FromSingle(float v)
        {
            if (float.IsNaN(v) || v < 0 || v > 255) throw new OverflowException("Single value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        public static Char8 FromDouble(double v)
        {
            if (double.IsNaN(v) || v < 0 || v > 255) throw new OverflowException("Double value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        public static Char8 FromDecimal(decimal v)
        {
            if (v < 0 || v > 255) throw new OverflowException("Decimal value out of Char8 range [0, 255].");
            return new Char8((byte)v);
        }

        // ========================================================================
        // Saturating / truncating variants (no-throw, always succeed)
        // ========================================================================

        /// <summary>Saturates the input to [0, 255] — negative becomes 0, > 255 becomes 255, NaN becomes 0.</summary>
        public static Char8 FromInt32Saturating(int v) => new Char8((byte)(v < 0 ? 0 : v > 255 ? 255 : v));

        /// <inheritdoc cref="FromInt32Saturating(int)"/>
        public static Char8 FromInt64Saturating(long v) => new Char8((byte)(v < 0 ? 0 : v > 255 ? 255 : v));

        /// <inheritdoc cref="FromInt32Saturating(int)"/>
        public static Char8 FromDoubleSaturating(double v)
        {
            if (double.IsNaN(v)) return new Char8(0);
            if (v < 0) return new Char8(0);
            if (v > 255) return new Char8(255);
            return new Char8((byte)v);
        }

        /// <summary>Truncates to 8 bits by masking (always succeeds).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromInt16Truncating(short v) => new Char8((byte)v);

        /// <inheritdoc cref="FromInt16Truncating(short)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromUInt16Truncating(ushort v) => new Char8((byte)v);

        /// <inheritdoc cref="FromInt16Truncating(short)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromUInt32Truncating(uint v) => new Char8((byte)v);

        /// <inheritdoc cref="FromInt16Truncating(short)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromInt64Truncating(long v) => new Char8((byte)v);

        /// <inheritdoc cref="FromInt16Truncating(short)"/>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char8 FromUInt64Truncating(ulong v) => new Char8((byte)v);

        // ========================================================================
        // Element-wise array conversions (useful for NDArray storage interop)
        // ========================================================================

        public static bool[] ToBooleanArray(ReadOnlySpan<Char8> src)
        {
            var r = new bool[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].m_value != 0;
            return r;
        }

        public static short[] ToInt16Array(ReadOnlySpan<Char8> src)
        {
            var r = new short[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].m_value;
            return r;
        }

        public static int[] ToInt32Array(ReadOnlySpan<Char8> src)
        {
            var r = new int[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].m_value;
            return r;
        }

        public static long[] ToInt64Array(ReadOnlySpan<Char8> src)
        {
            var r = new long[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].m_value;
            return r;
        }

        public static float[] ToSingleArray(ReadOnlySpan<Char8> src)
        {
            var r = new float[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].m_value;
            return r;
        }

        public static double[] ToDoubleArray(ReadOnlySpan<Char8> src)
        {
            var r = new double[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].m_value;
            return r;
        }

        public static char[] ToCharArray(ReadOnlySpan<Char8> src)
        {
            var r = new char[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = (char)src[i].m_value;
            return r;
        }

        public static Char8[] FromInt32Array(ReadOnlySpan<int> src, bool truncating = false)
        {
            var r = new Char8[src.Length];
            if (truncating)
            {
                for (int i = 0; i < src.Length; i++) r[i] = new Char8((byte)src[i]);
            }
            else
            {
                for (int i = 0; i < src.Length; i++)
                {
                    int v = src[i];
                    if ((uint)v > 0xFF) throw new OverflowException($"int[{i}]={v} out of Char8 range [0, 255].");
                    r[i] = new Char8((byte)v);
                }
            }
            return r;
        }

        public static Char8[] FromDoubleArray(ReadOnlySpan<double> src, bool saturating = false)
        {
            var r = new Char8[src.Length];
            if (saturating)
            {
                for (int i = 0; i < src.Length; i++) r[i] = FromDoubleSaturating(src[i]);
            }
            else
            {
                for (int i = 0; i < src.Length; i++) r[i] = FromDouble(src[i]);
            }
            return r;
        }
    }
}
