// Char8 primitive conversions — parallel to Converts.Native.cs for all 12 NumSharp dtypes
// (bool, byte, sbyte, char, int16/32/64, uint16/32/64, single, double, decimal) + string + object.
//
// Semantics match NumSharp's existing Converts.* primitives (throw on overflow/NaN). For
// saturating / truncating alternatives, use Char8.FromXxxSaturating / FromXxxTruncating.

using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    public static partial class Converts
    {
        // ====================================================================
        // Char8 -> other primitives (always safe — byte value widens)
        // ====================================================================

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(Char8 value) => value.Value != 0;

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(Char8 value)
        {
            if (value.Value > sbyte.MaxValue) throw new OverflowException("Overflow_SByte");
            return (sbyte)value.Value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(Char8 value) => (char)value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(Char8 value) => value.Value;

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(Char8 value) => value.Value;

        /// <summary>Returns a 1-character string (Latin-1 decode of the byte).</summary>
        [MethodImpl(OptimizeAndInline)]
        public static string ToString(Char8 value) => new string((char)value.Value, 1);

        // ====================================================================
        // Other primitives -> Char8 (throws on out-of-range)
        // ====================================================================

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(bool value) => new Char8(value ? (byte)1 : (byte)0);

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(byte value) => new Char8(value);

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(sbyte value)
        {
            if (value < 0) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(char value)
        {
            if ((uint)value > 0xFF) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(short value)
        {
            if ((uint)value > 0xFF) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(ushort value)
        {
            if (value > 0xFF) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(int value)
        {
            if ((uint)value > 0xFF) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(uint value)
        {
            if (value > 0xFF) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(long value)
        {
            if ((ulong)value > 0xFF) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(ulong value)
        {
            if (value > 0xFF) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(float value)
        {
            if (float.IsNaN(value) || value < 0 || value > 255) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(double value)
        {
            if (double.IsNaN(value) || value < 0 || value > 255) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(decimal value)
        {
            if (value < 0 || value > 255) throw new OverflowException("Overflow_Char8");
            return new Char8((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Char8 ToChar8(Char8 value) => value;

        /// <summary>Parses a one-character string as Char8 (Latin-1 decoded). Throws on empty, multi-char, or non-Latin-1.</summary>
        public static Char8 ToChar8(string value)
        {
            if (value == null) throw new ArgumentNullException(nameof(value));
            if (value.Length != 1) throw new FormatException("String must be exactly one character.");
            return ToChar8(value[0]);
        }

        // ====================================================================
        // Object / IConvertible dispatchers
        // ====================================================================

        /// <summary>Converts any IConvertible-supporting value to Char8. Dispatches on <see cref="IConvertible.GetTypeCode"/>.</summary>
        public static Char8 ToChar8(object value)
        {
            if (value == null) return default;
            if (value is Char8 c) return c;
            if (value is IConvertible ic) return ToChar8(ic, null);
            throw new InvalidCastException("Cannot convert object to Char8: value is not IConvertible.");
        }

        public static Char8 ToChar8(object value, IFormatProvider provider)
        {
            if (value == null) return default;
            if (value is Char8 c) return c;
            if (value is IConvertible ic) return ToChar8(ic, provider);
            throw new InvalidCastException("Cannot convert object to Char8: value is not IConvertible.");
        }

        private static Char8 ToChar8(IConvertible value, IFormatProvider provider)
        {
            return value.GetTypeCode() switch
            {
                TypeCode.Boolean => ToChar8(value.ToBoolean(provider)),
                TypeCode.Byte    => ToChar8(value.ToByte(provider)),
                TypeCode.SByte   => ToChar8(value.ToSByte(provider)),
                TypeCode.Char    => ToChar8(value.ToChar(provider)),
                TypeCode.Int16   => ToChar8(value.ToInt16(provider)),
                TypeCode.UInt16  => ToChar8(value.ToUInt16(provider)),
                TypeCode.Int32   => ToChar8(value.ToInt32(provider)),
                TypeCode.UInt32  => ToChar8(value.ToUInt32(provider)),
                TypeCode.Int64   => ToChar8(value.ToInt64(provider)),
                TypeCode.UInt64  => ToChar8(value.ToUInt64(provider)),
                TypeCode.Single  => ToChar8(value.ToSingle(provider)),
                TypeCode.Double  => ToChar8(value.ToDouble(provider)),
                TypeCode.Decimal => ToChar8(value.ToDecimal(provider)),
                TypeCode.String  => ToChar8(value.ToString(provider)),
                _ => throw new InvalidCastException($"Cannot convert {value.GetTypeCode()} to Char8.")
            };
        }

        // ====================================================================
        // Generic dispatcher — ToChar8<T>
        // ====================================================================

        /// <summary>
        /// Converts any NumSharp-supported primitive value to <see cref="Char8"/>.
        /// Dispatches on <see cref="InfoOf{T}.NPTypeCode"/>.
        /// </summary>
        [MethodImpl(Optimize)]
        public static Char8 ToChar8<T>(T value) where T : struct
        {
            // Char8 itself bypasses the generic dispatch (NPTypeCode.Empty for Char8)
            if (typeof(T) == typeof(Char8)) return Unsafe.As<T, Char8>(ref value);

            switch (InfoOf<T>.NPTypeCode)
            {
                case NPTypeCode.Boolean: return ToChar8(Unsafe.As<T, bool>(ref value));
                case NPTypeCode.Byte:    return ToChar8(Unsafe.As<T, byte>(ref value));
                case NPTypeCode.Int16:   return ToChar8(Unsafe.As<T, short>(ref value));
                case NPTypeCode.UInt16:  return ToChar8(Unsafe.As<T, ushort>(ref value));
                case NPTypeCode.Int32:   return ToChar8(Unsafe.As<T, int>(ref value));
                case NPTypeCode.UInt32:  return ToChar8(Unsafe.As<T, uint>(ref value));
                case NPTypeCode.Int64:   return ToChar8(Unsafe.As<T, long>(ref value));
                case NPTypeCode.UInt64:  return ToChar8(Unsafe.As<T, ulong>(ref value));
                case NPTypeCode.Char:    return ToChar8(Unsafe.As<T, char>(ref value));
                case NPTypeCode.Double:  return ToChar8(Unsafe.As<T, double>(ref value));
                case NPTypeCode.Single:  return ToChar8(Unsafe.As<T, float>(ref value));
                case NPTypeCode.Decimal: return ToChar8(Unsafe.As<T, decimal>(ref value));
                default:
                    // Fallback for Empty (incl. Char8) or unsupported T
                    return ToChar8((object)value);
            }
        }

        /// <summary>Converts a <see cref="Char8"/> to a NumSharp-supported primitive by target type code.</summary>
        [MethodImpl(Optimize)]
        public static object ToObject(Char8 value, NPTypeCode typeCode)
        {
            return typeCode switch
            {
                NPTypeCode.Boolean => (object)ToBoolean(value),
                NPTypeCode.Byte    => (object)ToByte(value),
                NPTypeCode.Int16   => (object)ToInt16(value),
                NPTypeCode.UInt16  => (object)ToUInt16(value),
                NPTypeCode.Int32   => (object)ToInt32(value),
                NPTypeCode.UInt32  => (object)ToUInt32(value),
                NPTypeCode.Int64   => (object)ToInt64(value),
                NPTypeCode.UInt64  => (object)ToUInt64(value),
                NPTypeCode.Char    => (object)ToChar(value),
                NPTypeCode.Double  => (object)ToDouble(value),
                NPTypeCode.Single  => (object)ToSingle(value),
                NPTypeCode.Decimal => (object)ToDecimal(value),
                NPTypeCode.String  => (object)ToString(value),
                _ => throw new NotSupportedException($"Cannot convert Char8 to {typeCode}.")
            };
        }

        // ====================================================================
        // Bulk array conversions (for NDArray storage interop)
        // ====================================================================

        /// <summary>Converts a <c>byte[]</c> to <c>Char8[]</c> (zero-copy reinterpret would require MemoryMarshal; this one copies).</summary>
        public static Char8[] ToChar8Array(byte[] src)
        {
            if (src == null) return null;
            var r = new Char8[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = new Char8(src[i]);
            return r;
        }

        /// <summary>Converts a <c>Char8[]</c> to <c>byte[]</c>.</summary>
        public static byte[] ToByteArray(Char8[] src)
        {
            if (src == null) return null;
            var r = new byte[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].Value;
            return r;
        }

        public static int[] ToInt32Array(Char8[] src)
        {
            if (src == null) return null;
            var r = new int[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].Value;
            return r;
        }

        public static double[] ToDoubleArray(Char8[] src)
        {
            if (src == null) return null;
            var r = new double[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = src[i].Value;
            return r;
        }

        public static Char8[] ToChar8ArrayFromInt32(int[] src)
        {
            if (src == null) return null;
            var r = new Char8[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = ToChar8(src[i]);
            return r;
        }

        public static Char8[] ToChar8ArrayFromDouble(double[] src)
        {
            if (src == null) return null;
            var r = new Char8[src.Length];
            for (int i = 0; i < src.Length; i++) r[i] = ToChar8(src[i]);
            return r;
        }
    }
}
