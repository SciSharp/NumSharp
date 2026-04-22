using System;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Security;
using System.Threading;

namespace NumSharp.Utilities
{
    [SuppressMessage("ReSharper", "MergeConditionalExpression")]
    [SuppressMessage("ReSharper", "JoinDeclarationAndInitializer")]
    public static partial class Converts
    {
        internal static readonly char[] base64Table = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/', '='};
        private const int base64LineBreakPosition = 76;

        public static readonly object DBNull = System.DBNull.Value;

        // Returns the type code for the given object. If the argument is null,
        // the result is TypeCode.Empty. If the argument is not a value (i.e. if
        // the object does not implement IConvertible), the result is TypeCode.Object.
        // Otherwise, the result is the type code of the object, as determined by
        // the object's implementation of IConvertible.
        [Pure]
        [MethodImpl(OptimizeAndInline)]
        public static TypeCode GetTypeCode(object value)
        {
            if (value == null) return TypeCode.Empty;
            if (value is IConvertible temp)
            {
                return temp.GetTypeCode();
            }

            return TypeCode.Object;
        }

        // Returns true if the given object is a database null. This operation
        // corresponds to "value.GetTypeCode() == TypeCode.DBNull".
        [Pure]
        [MethodImpl(OptimizeAndInline)]
        public static bool IsDBNull(object value)
        {
            if (value == System.DBNull.Value) return true;
            return value is IConvertible convertible && convertible.GetTypeCode() == TypeCode.DBNull;
        }

        // Converts the given object to the given type. In general, this method is
        // equivalent to calling the Value.ToXXX(value) method for the given
        // typeCode and boxing the result.
        //
        // The method first checks if the given object implements IConvertible. If not,
        // the only permitted conversion is from a null to TypeCode.Empty, the
        // result of which is null.
        //
        // If the object does implement IConvertible, a check is made to see if the
        // object already has the given type code, in which case the object is
        // simply returned. Otherwise, the appropriate ToXXX() is invoked on the
        // object's implementation of IConvertible.
        [MethodImpl(OptimizeAndInline)]
        public static object ChangeType(object value, TypeCode typeCode)
        {
            return ChangeType(value, typeCode, Thread.CurrentThread.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static object ChangeType(object value, TypeCode typeCode, IFormatProvider provider)
        {
            if (value == null && (typeCode == TypeCode.Empty || typeCode == TypeCode.String || typeCode == TypeCode.Object))
            {
                return null;
            }


            // Route numeric/bool/char conversions through the NumPy-aware object dispatchers
            // (Converts.ToXxx) so Half/Complex/char sources work and truncation/wrap/NaN match NumPy.
            // Raw IConvertible is preserved only for DateTime (not a NumPy dtype).
            switch (typeCode)
            {
                case TypeCode.Boolean:
                    return Converts.ToBoolean(value);
                case TypeCode.Char:
                    return Converts.ToChar(value);
                case TypeCode.SByte:
                    return Converts.ToSByte(value);
                case TypeCode.Byte:
                    return Converts.ToByte(value);
                case TypeCode.Int16:
                    return Converts.ToInt16(value);
                case TypeCode.UInt16:
                    return Converts.ToUInt16(value);
                case TypeCode.Int32:
                    return Converts.ToInt32(value);
                case TypeCode.UInt32:
                    return Converts.ToUInt32(value);
                case TypeCode.Int64:
                    return Converts.ToInt64(value);
                case TypeCode.UInt64:
                    return Converts.ToUInt64(value);
                case TypeCode.Single:
                    return Converts.ToSingle(value);
                case TypeCode.Double:
                    return Converts.ToDouble(value);
                case TypeCode.Decimal:
                    return Converts.ToDecimal(value);
                case TypeCode.DateTime:
                    return ToDateTime(value, provider);
                case TypeCode.String:
                    // Half/Complex don't implement IConvertible; IFormattable covers every supported type.
                    return value is IFormattable f ? f.ToString(null, provider) : value.ToString();
                case TypeCode.Object:
                    return value;
                case TypeCode.DBNull:
                    throw new InvalidCastException(("InvalidCast_DBNull"));
                case TypeCode.Empty:
                    throw new InvalidCastException(("InvalidCast_Empty"));
                default:
                    throw new ArgumentException(("Arg_UnknownTypeCode"));
            }
        }

        // Conversions to Boolean
        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(object value)
        {
            if (value == null) return false;
            return value switch
            {
                bool b => b,
                double d => ToBoolean(d),
                float f => ToBoolean(f),
                Half h => ToBoolean(h),
                Complex c => ToBoolean(c),
                decimal m => ToBoolean(m),
                long l => ToBoolean(l),
                ulong ul => ToBoolean(ul),
                int i => ToBoolean(i),
                uint u => ToBoolean(u),
                short s => ToBoolean(s),
                ushort us => ToBoolean(us),
                sbyte sb => ToBoolean(sb),
                byte by => ToBoolean(by),
                char ch => ToBoolean(ch),
                DateTime64 d64 => ToBoolean(d64),
                DateTime dt => ToBoolean(dt),
                TimeSpan ts => ToBoolean(ts),
                _ => ((IConvertible)value).ToBoolean(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(object value, IFormatProvider provider)
        {
            return ToBoolean(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(bool value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(sbyte value)
        {
            return value != 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(char value)
        {
            // Char is a 16-bit unsigned integer in NumSharp; treat like ushort.
            return value != (char)0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(byte value)
        {
            return value != 0;
        }


        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(short value)
        {
            return value != 0;
        }


        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(ushort value)
        {
            return value != 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(int value)
        {
            return value != 0;
        }


        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(uint value)
        {
            return value != 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(long value)
        {
            return value != 0;
        }


        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(ulong value)
        {
            return value != 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(string value)
        {
            if (value == null)
                return false;
            return bool.Parse(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(string value, IFormatProvider provider)
        {
            if (value == null)
                return false;
            return bool.Parse(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(float value)
        {
            return value != 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(double value)
        {
            return value != 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(decimal value)
        {
            return value != 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(Half value)
        {
            return value != (Half)0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(System.Numerics.Complex value)
        {
            return value != System.Numerics.Complex.Zero;
        }

        // DateTime/TimeSpan are not NumPy dtypes, but we provide conversions mirroring
        // NumPy's datetime64/timedelta64 semantics: both are stored as int64 (Ticks).
        // bool(dt/ts) = (Ticks != 0) mirrors NumPy's bool(datetime64/timedelta64).
        // NaT equivalents: TimeSpan.MinValue (Ticks == long.MinValue, exact parity);
        // DateTime.MinValue (Ticks == 0) for overflows/NaN where .NET DateTime cannot
        // represent the full int64 range.

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(DateTime value)
        {
            return value.Ticks != 0L;
        }

        [MethodImpl(OptimizeAndInline)]
        public static bool ToBoolean(TimeSpan value)
        {
            return value.Ticks != 0L;
        }

        // Conversions to Char


        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(object value)
        {
            if (value == null) return (char)0;
            return value switch
            {
                char c => c,
                byte b => (char)b,
                sbyte sb => unchecked((char)sb),
                short s => unchecked((char)s),
                ushort us => (char)us,
                int i => unchecked((char)i),
                uint u => unchecked((char)u),
                long l => unchecked((char)l),
                ulong ul => unchecked((char)ul),
                float f => ToChar(f),
                double d => ToChar(d),
                Half h => ToChar(h),
                Complex cx => ToChar(cx),
                decimal m => ToChar(m),
                bool bo => ToChar(bo),
                DateTime64 d64 => ToChar(d64),
                DateTime dt => ToChar(dt),
                TimeSpan tsv => ToChar(tsv),
                _ => ((IConvertible)value).ToChar(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(object value, IFormatProvider provider)
        {
            return ToChar(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(bool value)
        {
            // NumPy bool -> integer: true=1, false=0
            return value ? (char)1 : (char)0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(char value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(sbyte value)
        {
            return unchecked((char)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(byte value)
        {
            return (char)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(short value)
        {
            return unchecked((char)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(ushort value)
        {
            return (char)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(int value)
        {
            return unchecked((char)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(uint value)
        {
            return unchecked((char)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(long value)
        {
            return unchecked((char)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(ulong value)
        {
            return unchecked((char)value);
        }

        //
        // @VariantSwitch
        // Remove FormatExceptions;
        //
        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(string value)
        {
            return ToChar(value, null);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(string value, IFormatProvider provider)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            Contract.EndContractBlock();

            if (value.Length != 1)
                throw new FormatException(("ResId.Format_NeedSingleChar"));

            return value[0];
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(float value)
        {
            return ToChar((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(double value)
        {
            // NumPy uses int32 as intermediate for small int types. Route through ToInt32 so
            // fractional values inside int32 range (e.g. 2147483647.4) correctly truncate and
            // wrap, while values outside int32 range collapse to int.MinValue whose low 16
            // bits are 0 (NumPy's NaT-propagation convention for small ints). char is a
            // 16-bit unsigned integer in NumSharp, so wrap to ushort then reinterpret as char.
            return unchecked((char)(ushort)ToInt32(value));
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(decimal value)
        {
            // Truncate toward zero, wrap via int32 intermediate (matches NumPy uint16 pattern)
            var truncated = decimal.Truncate(value);
            if (truncated < int.MinValue || truncated > int.MaxValue)
            {
                return (char)0;
            }
            return unchecked((char)(ushort)(int)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(Half value)
        {
            // NumPy behavior: NaN/Inf -> 0 for small integer types (char is 16-bit unsigned)
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return (char)0;
            }
            // Half always fits in int32; truncate toward zero then wrap to char (ushort)
            return unchecked((char)(ushort)(int)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(System.Numerics.Complex value)
        {
            // NumPy: complex -> integer takes real part only
            return ToChar(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(DateTime value)
        {
            return ToChar(value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static char ToChar(TimeSpan value)
        {
            return ToChar(value.Ticks);
        }

        // Conversions to SByte


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(object value)
        {
            if (value == null) return 0;
            return value switch
            {
                sbyte sb => sb,
                byte b => unchecked((sbyte)b),
                short s => unchecked((sbyte)s),
                ushort us => unchecked((sbyte)us),
                int i => unchecked((sbyte)i),
                uint u => unchecked((sbyte)u),
                long l => unchecked((sbyte)l),
                ulong ul => unchecked((sbyte)ul),
                float f => ToSByte(f),
                double d => ToSByte(d),
                Half h => ToSByte(h),
                Complex cx => ToSByte(cx),  // NumPy: discard imaginary
                decimal m => ToSByte(m),
                bool bo => bo ? (sbyte)1 : (sbyte)0,
                char c => unchecked((sbyte)c),
                DateTime64 d64 => ToSByte(d64),
                DateTime dt => ToSByte(dt),
                TimeSpan ts => ToSByte(ts),
                _ => ((IConvertible)value).ToSByte(null)
            };
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(object value, IFormatProvider provider)
        {
            return ToSByte(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(bool value)
        {
            return value ? (sbyte)1 : (sbyte)0;
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(sbyte value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(char value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(byte value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(short value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(ushort value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(int value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(uint value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(long value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(ulong value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((sbyte)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(float value)
        {
            return ToSByte((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(double value)
        {
            // NumPy uses int32 as intermediate for small int types. Route through ToInt32 so
            // fractional values inside int32 range (e.g. 2147483647.4) correctly truncate and
            // wrap (-> -1), while values outside int32 range collapse to int.MinValue whose
            // low byte is 0 (NumPy's NaT-propagation convention for small ints).
            return unchecked((sbyte)ToInt32(value));
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(decimal value)
        {
            // NumPy parity: truncate toward zero, wrap via int32 intermediate.
            // Decimal values outside int32 range return 0 (matches float->int8 NaN/overflow pattern).
            var truncated = decimal.Truncate(value);
            if (truncated < int.MinValue || truncated > int.MaxValue)
            {
                return 0;
            }
            return unchecked((sbyte)(int)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(Half value)
        {
            // NumPy behavior: NaN/Inf -> 0 for int8
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return 0;
            }
            // NumPy uses int32 as intermediate - Half always fits in int32
            return unchecked((sbyte)(int)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(System.Numerics.Complex value)
        {
            return ToSByte(value.Real);
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(string value)
        {
            if (value == null)
                return 0;
            return sbyte.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(string value, IFormatProvider provider)
        {
            return sbyte.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(DateTime value)
        {
            return unchecked((sbyte)value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static sbyte ToSByte(TimeSpan value)
        {
            return unchecked((sbyte)value.Ticks);
        }

        // Conversions to Byte

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(object value)
        {
            if (value == null) return 0;
            return value switch
            {
                byte b => b,
                sbyte sb => unchecked((byte)sb),
                short s => unchecked((byte)s),
                ushort us => unchecked((byte)us),
                int i => unchecked((byte)i),
                uint u => unchecked((byte)u),
                long l => unchecked((byte)l),
                ulong ul => unchecked((byte)ul),
                float f => ToByte(f),
                double d => ToByte(d),
                Half h => ToByte(h),
                Complex c => ToByte(c),  // NumPy: discard imaginary
                decimal m => ToByte(m),
                bool bo => bo ? (byte)1 : (byte)0,
                char c => unchecked((byte)c),
                DateTime64 d64 => ToByte(d64),
                DateTime dt => ToByte(dt),
                TimeSpan ts => ToByte(ts),
                _ => ((IConvertible)value).ToByte(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(object value, IFormatProvider provider)
        {
            return ToByte(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(bool value)
        {
            return value ? (byte)1 : (byte)0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(byte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(char value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(sbyte value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(short value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(ushort value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(int value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(uint value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(long value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(ulong value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((byte)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(float value)
        {
            return ToByte((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(double value)
        {
            // NumPy uses int32 as intermediate for small int types. Route through ToInt32 so
            // fractional values inside int32 range (e.g. 2147483647.4) correctly truncate and
            // wrap (-> 255), while values outside int32 range collapse to int.MinValue whose
            // low byte is 0 (NumPy's NaT-propagation convention for small ints).
            return unchecked((byte)ToInt32(value));
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(decimal value)
        {
            // NumPy parity: truncate toward zero, wrap via int32 intermediate.
            // Negative values wrap (e.g. -1m -> 255). Values outside int32 range return 0.
            var truncated = decimal.Truncate(value);
            if (truncated < int.MinValue || truncated > int.MaxValue)
            {
                return 0;
            }
            return unchecked((byte)(int)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(Half value)
        {
            // NumPy behavior: NaN/Inf -> 0 for uint8
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return 0;
            }
            // NumPy uses int32 as intermediate - Half always fits in int32
            return unchecked((byte)(int)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(System.Numerics.Complex value)
        {
            return ToByte(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(string value)
        {
            if (value == null)
                return 0;
            return byte.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return byte.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(DateTime value)
        {
            return unchecked((byte)value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static byte ToByte(TimeSpan value)
        {
            return unchecked((byte)value.Ticks);
        }

        // Conversions to Int16

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(object value)
        {
            if (value == null) return 0;
            return value switch
            {
                short s => s,
                ushort us => unchecked((short)us),
                int i => unchecked((short)i),
                uint u => unchecked((short)u),
                long l => unchecked((short)l),
                ulong ul => unchecked((short)ul),
                sbyte sb => sb,
                byte b => b,
                float f => ToInt16(f),
                double d => ToInt16(d),
                Half h => ToInt16(h),
                Complex cx => ToInt16(cx),  // NumPy: discard imaginary
                decimal m => ToInt16(m),
                bool bo => bo ? (short)1 : (short)0,
                char c => unchecked((short)c),
                DateTime64 d64 => ToInt16(d64),
                DateTime dt => ToInt16(dt),
                TimeSpan ts => ToInt16(ts),
                _ => ((IConvertible)value).ToInt16(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(object value, IFormatProvider provider)
        {
            return ToInt16(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(bool value)
        {
            return value ? (short)1 : (short)0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(char value)
        {
            return unchecked((short)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(sbyte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(byte value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(ushort value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((short)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(int value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((short)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(uint value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((short)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(short value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(long value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((short)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(ulong value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((short)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(float value)
        {
            return ToInt16((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(double value)
        {
            // NumPy uses int32 as intermediate for small int types. Route through ToInt32 so
            // fractional values inside int32 range (e.g. 2147483647.4) correctly truncate and
            // wrap (-> -1), while values outside int32 range collapse to int.MinValue whose
            // low 16 bits are 0 (NumPy's NaT-propagation convention for small ints).
            return unchecked((short)ToInt32(value));
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(decimal value)
        {
            // NumPy parity: truncate toward zero, wrap via int32 intermediate.
            var truncated = decimal.Truncate(value);
            if (truncated < int.MinValue || truncated > int.MaxValue)
            {
                return 0;
            }
            return unchecked((short)(int)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(Half value)
        {
            // NumPy behavior: NaN/Inf -> 0 for int16
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return 0;
            }
            // NumPy uses int32 as intermediate - Half always fits in int32
            return unchecked((short)(int)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(System.Numerics.Complex value)
        {
            return ToInt16(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(string value)
        {
            if (value == null)
                return 0;
            return short.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return short.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(DateTime value)
        {
            return unchecked((short)value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static short ToInt16(TimeSpan value)
        {
            return unchecked((short)value.Ticks);
        }

        // Conversions to UInt16


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(object value)
        {
            if (value == null) return 0;
            return value switch
            {
                ushort us => us,
                short s => unchecked((ushort)s),
                int i => unchecked((ushort)i),
                uint u => unchecked((ushort)u),
                long l => unchecked((ushort)l),
                ulong ul => unchecked((ushort)ul),
                sbyte sb => unchecked((ushort)sb),
                byte b => b,
                float f => ToUInt16(f),
                double d => ToUInt16(d),
                Half h => ToUInt16(h),
                Complex cx => ToUInt16(cx),  // NumPy: discard imaginary
                decimal m => ToUInt16(m),
                bool bo => bo ? (ushort)1 : (ushort)0,
                char c => c,
                DateTime64 d64 => ToUInt16(d64),
                DateTime dt => ToUInt16(dt),
                TimeSpan ts => ToUInt16(ts),
                _ => ((IConvertible)value).ToUInt16(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(object value, IFormatProvider provider)
        {
            return ToUInt16(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(bool value)
        {
            return value ? (ushort)1 : (ushort)0;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(char value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(sbyte value)
        {
            return unchecked((ushort)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(byte value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(short value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((ushort)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(int value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((ushort)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(ushort value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(uint value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((ushort)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(long value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((ushort)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(ulong value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((ushort)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(float value)
        {
            return ToUInt16((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(double value)
        {
            // NumPy uses int32 as intermediate for small int types. Route through ToInt32 so
            // fractional values inside int32 range (e.g. 2147483647.4) correctly truncate and
            // wrap (-> 65535), while values outside int32 range collapse to int.MinValue whose
            // low 16 bits are 0 (NumPy's NaT-propagation convention for small ints).
            return unchecked((ushort)ToInt32(value));
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(decimal value)
        {
            // NumPy parity: truncate toward zero, wrap via int32 intermediate.
            // Negative values wrap (e.g. -1m -> 65535).
            var truncated = decimal.Truncate(value);
            if (truncated < int.MinValue || truncated > int.MaxValue)
            {
                return 0;
            }
            return unchecked((ushort)(int)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(Half value)
        {
            // NumPy behavior: NaN/Inf -> 0 for uint16
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return 0;
            }
            // NumPy uses int32 as intermediate - Half always fits in int32
            return unchecked((ushort)(int)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(System.Numerics.Complex value)
        {
            return ToUInt16(value.Real);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(string value)
        {
            if (value == null)
                return 0;
            return ushort.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return ushort.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(DateTime value)
        {
            return unchecked((ushort)value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ushort ToUInt16(TimeSpan value)
        {
            return unchecked((ushort)value.Ticks);
        }

        // Conversions to Int32

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(object value)
        {
            if (value == null) return 0;
            return value switch
            {
                int i => i,
                uint u => unchecked((int)u),
                long l => unchecked((int)l),
                ulong ul => unchecked((int)ul),
                short s => s,
                ushort us => us,
                sbyte sb => sb,
                byte b => b,
                float f => ToInt32(f),
                double d => ToInt32(d),
                Half h => ToInt32(h),
                Complex c => ToInt32(c),  // NumPy: discard imaginary
                decimal m => ToInt32(m),
                bool bo => bo ? 1 : 0,
                char c => c,
                DateTime64 d64 => ToInt32(d64),
                DateTime dt => ToInt32(dt),
                TimeSpan ts => ToInt32(ts),
                _ => ((IConvertible)value).ToInt32(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(object value, IFormatProvider provider)
        {
            return ToInt32(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(bool value)
        {
            return value ? 1 : 0;
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(char value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(sbyte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(byte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(short value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(ushort value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(uint value)
        {
            return unchecked((int)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(int value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(long value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((int)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(ulong value)
        {
            // NumPy: integer-to-integer uses wrapping (modulo arithmetic)
            return unchecked((int)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(float value)
        {
            return ToInt32((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(double value)
        {
            // NumPy: truncate toward zero FIRST, then overflow-check the truncated integer.
            // NaN/Inf/overflow -> int32.MinValue. Comparing `value > int.MaxValue` directly
            // breaks for fractional values like 2147483647.4 which NumPy truncates to
            // 2147483647 (in-range), but the naive comparison rejects as overflow.
            if (double.IsNaN(value) || double.IsInfinity(value)) return int.MinValue;
            double t = Math.Truncate(value);
            if (t < int.MinValue || t > int.MaxValue) return int.MinValue;
            return (int)t;
        }

        [System.Security.SecuritySafeCritical] // auto-generated
        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(decimal value)
        {
            // NumPy parity: truncate toward zero. Values outside int32 range -> int32.MinValue
            // (matches NumPy float->int32 overflow behavior).
            var truncated = decimal.Truncate(value);
            if (truncated < int.MinValue || truncated > int.MaxValue)
            {
                return int.MinValue;
            }
            return (int)truncated;
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(Half value)
        {
            // NumPy behavior: special values -> int.MinValue for int32
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return int.MinValue;
            }
            return (int)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(System.Numerics.Complex value)
        {
            return ToInt32(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(string value)
        {
            if (value == null)
                return 0;
            return int.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return int.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(DateTime value)
        {
            return unchecked((int)value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToInt32(TimeSpan value)
        {
            return unchecked((int)value.Ticks);
        }

        // Conversions to UInt32


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(object value)
        {
            if (value == null) return 0;
            // Type dispatch for NumPy-compatible unchecked wrapping
            return value switch
            {
                uint u => u,
                int i => unchecked((uint)i),
                long l => unchecked((uint)l),
                ulong ul => unchecked((uint)ul),
                short s => unchecked((uint)s),
                ushort us => us,
                sbyte sb => unchecked((uint)sb),
                byte b => b,
                float f => ToUInt32(f),
                double d => ToUInt32(d),
                Half h => ToUInt32(h),
                Complex cx => ToUInt32(cx),  // NumPy: discard imaginary
                decimal m => ToUInt32(m),
                bool bo => bo ? 1u : 0u,
                char c => c,
                DateTime64 d64 => ToUInt32(d64),
                DateTime dt => ToUInt32(dt),
                TimeSpan ts => ToUInt32(ts),
                _ => ((IConvertible)value).ToUInt32(null)
            };
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(object value, IFormatProvider provider)
        {
            return ToUInt32(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(bool value)
        {
            return value ? 1u : 0u;
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(char value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(sbyte value)
        {
            return unchecked((uint)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(byte value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(short value)
        {
            return unchecked((uint)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(ushort value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(int value)
        {
            return unchecked((uint)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(uint value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(long value)
        {
            return unchecked((uint)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(ulong value)
        {
            return unchecked((uint)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(float value)
        {
            return ToUInt32((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(double value)
        {
            // NumPy behavior: NaN/Inf -> 0 for uint32
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return 0;
            }
            // Out-of-int64-range values: NumPy's int64 overflow returns int64.MinValue,
            // and unchecked((uint)int64.MinValue) == 0. Use exclusive upper bound 2^63
            // (since (double)long.MaxValue rounds to 2^63 and is itself overflow).
            if (value < (double)long.MinValue || value >= 9223372036854775808.0)
            {
                return 0;
            }
            // NumPy: truncate toward zero, then wrap modularly to uint
            return unchecked((uint)(long)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(decimal value)
        {
            // NumPy parity: truncate toward zero. Negative values wrap via int64 intermediate.
            // Values outside int64 range return 0.
            var truncated = decimal.Truncate(value);
            if (truncated < long.MinValue || truncated > long.MaxValue)
            {
                return 0;
            }
            return unchecked((uint)(long)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(Half value)
        {
            // NumPy behavior: NaN/Inf -> 0 for uint32
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return 0;
            }
            // NumPy: truncate toward zero, then wrap modularly
            return unchecked((uint)(long)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(System.Numerics.Complex value)
        {
            return ToUInt32(value.Real);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(string value)
        {
            if (value == null)
                return 0;
            return uint.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return uint.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(DateTime value)
        {
            return unchecked((uint)value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static uint ToUInt32(TimeSpan value)
        {
            return unchecked((uint)value.Ticks);
        }

        // Conversions to Int64

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(object value)
        {
            if (value == null) return 0;
            return value switch
            {
                long l => l,
                ulong ul => unchecked((long)ul),
                int i => i,
                uint u => u,
                short s => s,
                ushort us => us,
                sbyte sb => sb,
                byte b => b,
                float f => ToInt64(f),
                double d => ToInt64(d),
                Half h => ToInt64(h),
                Complex cx => ToInt64(cx),  // NumPy: discard imaginary
                decimal m => ToInt64(m),
                bool bo => bo ? 1L : 0L,
                char c => c,
                DateTime64 d64 => ToInt64(d64),
                DateTime dt => ToInt64(dt),
                TimeSpan ts => ToInt64(ts),
                _ => ((IConvertible)value).ToInt64(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(object value, IFormatProvider provider)
        {
            return ToInt64(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(bool value)
        {
            return value ? 1L : 0L;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(char value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(sbyte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(byte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(short value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(ushort value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(int value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(uint value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(ulong value)
        {
            return unchecked((long)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(long value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(float value)
        {
            return ToInt64((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(double value)
        {
            // NumPy behavior: truncation toward zero for normal values
            // For special values (inf, -inf, nan, overflow): returns long.MinValue
            // NOTE: `value > long.MaxValue` isn't safe — (double)long.MaxValue rounds UP
            // to 2^63 (same bit pattern as (double)(long.MaxValue+1)) so the check misses
            // values that NumPy treats as overflow. Use exclusive upper bound at 2^63.
            if (double.IsNaN(value) || double.IsInfinity(value)
                || value < (double)long.MinValue
                || value >= 9223372036854775808.0)   // 2^63, smallest double > long.MaxValue
            {
                return long.MinValue;  // NumPy returns int64.min for all special/overflow cases
            }
            return (long)value;  // C# cast truncates toward zero
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(decimal value)
        {
            // NumPy parity: truncate toward zero. Values outside int64 range -> int64.MinValue.
            var truncated = decimal.Truncate(value);
            if (truncated < long.MinValue || truncated > long.MaxValue)
            {
                return long.MinValue;
            }
            return (long)truncated;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(Half value)
        {
            // NumPy behavior: special values -> long.MinValue
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return long.MinValue;
            }
            return (long)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(System.Numerics.Complex value)
        {
            return ToInt64(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(string value)
        {
            if (value == null)
                return 0;
            return long.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return long.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(DateTime value)
        {
            return value.Ticks;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long ToInt64(TimeSpan value)
        {
            return value.Ticks;
        }

        // Conversions to UInt64


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(object value)
        {
            if (value == null) return 0;
            return value switch
            {
                ulong ul => ul,
                long l => unchecked((ulong)l),
                uint u => u,
                int i => unchecked((ulong)i),
                ushort us => us,
                short s => unchecked((ulong)s),
                byte b => b,
                sbyte sb => unchecked((ulong)sb),
                float f => ToUInt64(f),
                double d => ToUInt64(d),
                Half h => ToUInt64(h),
                Complex cx => ToUInt64(cx),  // NumPy: discard imaginary
                decimal m => ToUInt64(m),
                bool bo => bo ? 1UL : 0UL,
                char c => c,
                DateTime64 d64 => ToUInt64(d64),
                DateTime dt => ToUInt64(dt),
                TimeSpan ts => ToUInt64(ts),
                _ => ((IConvertible)value).ToUInt64(null)
            };
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(object value, IFormatProvider provider)
        {
            return ToUInt64(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(bool value)
        {
            return value ? 1ul : 0ul;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(char value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(sbyte value)
        {
            return unchecked((ulong)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(byte value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(short value)
        {
            return unchecked((ulong)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(ushort value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(int value)
        {
            return unchecked((ulong)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(uint value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(long value)
        {
            return unchecked((ulong)value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(ulong value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(float value)
        {
            return ToUInt64((double)value);
        }

        // NumPy special value for uint64 overflow: 2^63 = 9223372036854775808
        private const ulong NumPyUInt64Overflow = 9223372036854775808UL;

        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(double value)
        {
            // NumPy behavior: NaN/Inf -> 2^63 for uint64
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return NumPyUInt64Overflow;
            }
            // Precision note: (double)long.MaxValue rounds to 2^63 (out of long range);
            // (double)ulong.MaxValue rounds to 2^64 (out of ulong range). Both bounds must
            // be exclusive or NumPy parity breaks.
            //   value < -2^63               -> overflow (NaT sentinel)
            //   value in [-2^63, 2^63)      -> cast via signed long, unchecked wrap
            //   value in [2^63, 2^64)       -> direct ulong cast (upper half)
            //   value >= 2^64               -> overflow (NaT sentinel)
            const double twoPow63 = 9223372036854775808.0;           // 2^63  (= NaT / overflow marker)
            const double twoPow64 = 18446744073709551616.0;           // 2^64  (= (double)ulong.MaxValue after rounding)
            if (value < (double)long.MinValue || value >= twoPow64)
            {
                return NumPyUInt64Overflow;
            }
            if (value >= twoPow63)
            {
                return (ulong)value;
            }
            // NumPy: truncate toward zero, then wrap modularly to ulong.
            // For -1.0: truncate to -1, wrap to 2^64-1. For -3.7: truncate to -3, wrap to 2^64-3.
            return unchecked((ulong)(long)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(decimal value)
        {
            // NumPy parity: truncate toward zero, wrap via int64 intermediate for negatives.
            // Positive values within ulong range convert directly. Values outside range return 0.
            var truncated = decimal.Truncate(value);
            if (truncated < long.MinValue)
            {
                return 0;
            }
            if (truncated < 0m)
            {
                return unchecked((ulong)(long)truncated);
            }
            if (truncated > (decimal)ulong.MaxValue)
            {
                return 0;
            }
            return (ulong)truncated;
        }

        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(Half value)
        {
            // NumPy behavior: NaN/Inf -> 2^63 for uint64
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return NumPyUInt64Overflow;
            }
            // NumPy: truncate toward zero, then wrap modularly
            // Half range is small enough to always fit in long
            return unchecked((ulong)(long)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(System.Numerics.Complex value)
        {
            return ToUInt64(value.Real);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(string value)
        {
            if (value == null)
                return 0;
            return ulong.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return ulong.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(DateTime value)
        {
            return unchecked((ulong)value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static ulong ToUInt64(TimeSpan value)
        {
            return unchecked((ulong)value.Ticks);
        }

        // Conversions to Single

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(object value)
        {
            if (value == null) return 0f;
            return value switch
            {
                float f => f,
                double d => ToSingle(d),
                Half h => ToSingle(h),
                Complex c => ToSingle(c),
                decimal m => ToSingle(m),
                long l => ToSingle(l),
                ulong ul => ToSingle(ul),
                int i => ToSingle(i),
                uint u => ToSingle(u),
                short s => ToSingle(s),
                ushort us => ToSingle(us),
                sbyte sb => ToSingle(sb),
                byte by => ToSingle(by),
                char ch => ToSingle(ch),
                bool bo => bo ? 1f : 0f,
                DateTime64 d64 => ToSingle(d64),
                DateTime dt => ToSingle(dt),
                TimeSpan ts => ToSingle(ts),
                _ => ((IConvertible)value).ToSingle(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(object value, IFormatProvider provider)
        {
            return ToSingle(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(sbyte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(byte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(char value)
        {
            return (float)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(short value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(ushort value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(int value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(uint value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(long value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(ulong value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(float value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(double value)
        {
            return (float)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(decimal value)
        {
            return (float)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(Half value)
        {
            return (float)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(System.Numerics.Complex value)
        {
            return (float)value.Real;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(string value)
        {
            if (value == null)
                return 0;
            return float.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return float.Parse(value, NumberStyles.Float | NumberStyles.AllowThousands, provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(bool value)
        {
            return value ? 1f : 0f;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(DateTime value)
        {
            return (float)value.Ticks;
        }

        [MethodImpl(OptimizeAndInline)]
        public static float ToSingle(TimeSpan value)
        {
            return (float)value.Ticks;
        }

        // Conversions to Double

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(object value)
        {
            if (value == null) return 0d;
            return value switch
            {
                double d => d,
                float f => ToDouble(f),
                Half h => ToDouble(h),
                Complex c => c.Real, // NumPy: discard imaginary
                decimal m => ToDouble(m),
                long l => ToDouble(l),
                ulong ul => ToDouble(ul),
                int i => ToDouble(i),
                uint u => ToDouble(u),
                short s => ToDouble(s),
                ushort us => ToDouble(us),
                sbyte sb => ToDouble(sb),
                byte by => ToDouble(by),
                char ch => ToDouble(ch),
                bool bo => bo ? 1d : 0d,
                DateTime64 d64 => ToDouble(d64),
                DateTime dt => ToDouble(dt),
                TimeSpan ts => ToDouble(ts),
                _ => ((IConvertible)value).ToDouble(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(object value, IFormatProvider provider)
        {
            return ToDouble(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(sbyte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(byte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(short value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(char value)
        {
            return (double)value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(ushort value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(int value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(uint value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(long value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(ulong value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(float value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(double value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(decimal value)
        {
            return (double)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(Half value)
        {
            return (double)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(System.Numerics.Complex value)
        {
            return value.Real;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(string value)
        {
            if (value == null)
                return 0;
            return double.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return double.Parse(value, NumberStyles.Float | NumberStyles.AllowThousands, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(bool value)
        {
            return value ? 1d : 0d;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(DateTime value)
        {
            return (double)value.Ticks;
        }

        [MethodImpl(OptimizeAndInline)]
        public static double ToDouble(TimeSpan value)
        {
            return (double)value.Ticks;
        }

        // Conversions to Decimal

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(object value)
        {
            if (value == null) return 0m;
            return value switch
            {
                decimal m => m,
                double d => ToDecimal(d),
                float f => ToDecimal(f),
                Half h => ToDecimal(h),
                Complex cx => ToDecimal(cx),
                long l => l,
                ulong ul => ul,
                int i => i,
                uint u => u,
                short s => s,
                ushort us => us,
                sbyte sb => sb,
                byte b => b,
                char c => c,
                bool bo => bo ? 1m : 0m,
                DateTime64 d64 => ToDecimal(d64),
                DateTime dt => ToDecimal(dt),
                TimeSpan ts => ToDecimal(ts),
                _ => ((IConvertible)value).ToDecimal(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(object value, IFormatProvider provider)
        {
            return ToDecimal(value);
        }


        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(sbyte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(byte value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(char value)
        {
            return (decimal)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(short value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(ushort value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(int value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(uint value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(long value)
        {
            return value;
        }


        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(ulong value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(float value)
        {
            return ToDecimal((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(double value)
        {
            // NaN/Inf and out-of-range values return 0 (consistent with small-integer NaN handling).
            // Decimal cannot represent NaN/Inf and cast would throw OverflowException.
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                return 0m;
            }
            if (value < (double)decimal.MinValue || value > (double)decimal.MaxValue)
            {
                return 0m;
            }
            return (decimal)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(Half value)
        {
            // Half range (~±65504) fits comfortably in decimal, but Half.NaN/Inf would throw
            if (Half.IsNaN(value) || Half.IsInfinity(value))
            {
                return 0m;
            }
            return (decimal)(double)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(System.Numerics.Complex value)
        {
            // Discard imaginary part, route through double->decimal for NaN/Inf safety
            return ToDecimal(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(string value)
        {
            if (value == null)
                return 0m;
            return decimal.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0m;
            return decimal.Parse(value, NumberStyles.Number, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(decimal value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(bool value)
        {
            return value ? 1m : 0m;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(DateTime value)
        {
            return (decimal)value.Ticks;
        }

        [MethodImpl(OptimizeAndInline)]
        public static decimal ToDecimal(TimeSpan value)
        {
            return (decimal)value.Ticks;
        }

        // Conversions to Half (float16)
        // Note: Half doesn't implement IConvertible, so all conversions go through double

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(object value)
        {
            if (value == null) return default;
            return value switch
            {
                Half h => h,
                double d => ToHalf(d),
                float f => ToHalf(f),
                Complex c => ToHalf(c),
                decimal m => ToHalf(m),
                long l => ToHalf(l),
                ulong ul => ToHalf(ul),
                int i => ToHalf(i),
                uint u => ToHalf(u),
                short s => ToHalf(s),
                ushort us => ToHalf(us),
                sbyte sb => ToHalf(sb),
                byte by => ToHalf(by),
                char ch => ToHalf(ch),
                bool bo => ToHalf(bo),
                DateTime64 d64 => ToHalf(d64),
                DateTime dt => ToHalf(dt),
                TimeSpan ts => ToHalf(ts),
                _ => (Half)((IConvertible)value).ToDouble(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(object value, IFormatProvider provider)
        {
            return ToHalf(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(bool value)
        {
            return (Half)(value ? 1 : 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(Half value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(sbyte value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(byte value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(char value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(short value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(ushort value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(int value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(uint value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(long value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(ulong value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(float value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(double value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(decimal value)
        {
            return (Half)value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(System.Numerics.Complex value)
        {
            // NumPy: complex -> float16 uses the real part
            return (Half)value.Real;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(string value)
        {
            if (value == null)
                return default;
            return Half.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(string value, IFormatProvider provider)
        {
            if (value == null)
                return default;
            return Half.Parse(value, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(DateTime value)
        {
            return (Half)(double)value.Ticks;
        }

        [MethodImpl(OptimizeAndInline)]
        public static Half ToHalf(TimeSpan value)
        {
            return (Half)(double)value.Ticks;
        }

        // Conversions to Complex (complex128)
        // Note: Complex and Half don't implement IConvertible

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(object value)
        {
            if (value == null) return default;
            return value switch
            {
                Complex c => c,
                Half h => ToComplex(h),
                double d => ToComplex(d),
                float f => ToComplex(f),
                decimal m => ToComplex(m),
                long l => ToComplex(l),
                ulong ul => ToComplex(ul),
                int i => ToComplex(i),
                uint u => ToComplex(u),
                short s => ToComplex(s),
                ushort us => ToComplex(us),
                sbyte sb => ToComplex(sb),
                byte by => ToComplex(by),
                char ch => ToComplex(ch),
                bool bo => ToComplex(bo),
                DateTime64 d64 => ToComplex(d64),
                DateTime dt => ToComplex(dt),
                TimeSpan ts => ToComplex(ts),
                _ => new Complex(((IConvertible)value).ToDouble(null), 0)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(object value, IFormatProvider provider)
        {
            return ToComplex(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(bool value)
        {
            return new System.Numerics.Complex(value ? 1.0 : 0.0, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(System.Numerics.Complex value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(Half value)
        {
            return new System.Numerics.Complex((double)value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(sbyte value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(byte value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(char value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(short value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(ushort value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(int value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(uint value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(long value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(ulong value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(float value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(double value)
        {
            return new System.Numerics.Complex(value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(decimal value)
        {
            return new System.Numerics.Complex((double)value, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(DateTime value)
        {
            return new System.Numerics.Complex((double)value.Ticks, 0);
        }

        [MethodImpl(OptimizeAndInline)]
        public static System.Numerics.Complex ToComplex(TimeSpan value)
        {
            return new System.Numerics.Complex((double)value.Ticks, 0);
        }

        // Conversions to DateTime
        //
        // NumPy-parity semantics: numeric values are interpreted as DateTime.Ticks
        // (mirrors NumPy datetime64 which stores the raw int64 count of units since epoch).
        // .NET DateTime only permits ticks in [0, DateTime.MaxValue.Ticks]; out-of-range
        // or invalid (NaN/Inf) values collapse to DateTime.MinValue (our NaT-equivalent).

        // DateTime.MaxValue.Ticks (3155378975999999999) as double loses precision at the
        // top of the range, so we keep the upper bound as a double constant for comparison.
        private const double DateTimeMaxTicksAsDouble = 3.1553789759999999e18;

        [MethodImpl(OptimizeAndInline)]
        private static DateTime TicksToDateTime(long ticks)
        {
            // Clamp to valid DateTime range. Out-of-range -> DateTime.MinValue (NaT-like).
            if ((ulong)ticks > (ulong)DateTime.MaxValue.Ticks)
                return DateTime.MinValue;
            return new DateTime(ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(DateTime value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(object value)
        {
            if (value == null) return DateTime.MinValue;
            return value switch
            {
                DateTime dt => dt,
                TimeSpan ts => TicksToDateTime(ts.Ticks),
                bool b => TicksToDateTime(b ? 1L : 0L),
                sbyte sb => TicksToDateTime(sb),
                byte by => TicksToDateTime(by),
                short s => TicksToDateTime(s),
                ushort us => TicksToDateTime(us),
                int i => TicksToDateTime(i),
                uint u => TicksToDateTime(u),
                long l => TicksToDateTime(l),
                ulong ul => TicksToDateTime(unchecked((long)ul)),
                char c => TicksToDateTime(c),
                float f => ToDateTime(f),
                double d => ToDateTime(d),
                Half h => ToDateTime(h),
                Complex cx => ToDateTime(cx),
                decimal m => ToDateTime(m),
                string str => ToDateTime(str),
                _ => ((IConvertible)value).ToDateTime(null)
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(object value, IFormatProvider provider)
        {
            if (value == null) return DateTime.MinValue;
            if (value is string s) return ToDateTime(s, provider);
            return ToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(string value)
        {
            if (value == null)
                return DateTime.MinValue;
            return DateTime.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(string value, IFormatProvider provider)
        {
            if (value == null)
                return DateTime.MinValue;
            return DateTime.Parse(value, provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(sbyte value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(byte value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(short value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(ushort value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(int value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(uint value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(long value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(ulong value)
        {
            return TicksToDateTime(unchecked((long)value));
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(bool value)
        {
            // NumPy: bool -> integer (true=1, false=0), then reinterpret as ticks.
            return value ? new DateTime(1L) : DateTime.MinValue;
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(char value)
        {
            return TicksToDateTime(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(float value)
        {
            return ToDateTime((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(double value)
        {
            // NumPy: NaN/Inf -> NaT, which we map to DateTime.MinValue.
            // Out-of-DateTime-range also collapses to MinValue (best we can do).
            if (double.IsNaN(value) || double.IsInfinity(value)) return DateTime.MinValue;
            if (value < 0d || value > DateTimeMaxTicksAsDouble) return DateTime.MinValue;
            // (double)DateTime.MaxValue.Ticks rounds UP by precision loss, so even values
            // inside the upper bound can cast to a long that exceeds MaxValue.Ticks.
            // Route through TicksToDateTime which clamps again after the cast.
            return TicksToDateTime((long)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(Half value)
        {
            if (Half.IsNaN(value) || Half.IsInfinity(value)) return DateTime.MinValue;
            return ToDateTime((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(System.Numerics.Complex value)
        {
            // NumPy: complex -> scalar uses the real part.
            return ToDateTime(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(decimal value)
        {
            var truncated = decimal.Truncate(value);
            if (truncated < 0m || truncated > (decimal)DateTime.MaxValue.Ticks)
                return DateTime.MinValue;
            return new DateTime((long)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static DateTime ToDateTime(TimeSpan value)
        {
            return TicksToDateTime(value.Ticks);
        }

        // Conversions to TimeSpan
        //
        // NumPy-parity semantics: numeric values are interpreted as TimeSpan.Ticks.
        // .NET TimeSpan covers the full int64 range, so NaT (long.MinValue) maps exactly
        // to TimeSpan.MinValue — perfect parity with NumPy timedelta64 NaT.
        // NaN/Inf/out-of-range values collapse to TimeSpan.MinValue.

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(TimeSpan value)
        {
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(DateTime value)
        {
            return new TimeSpan(value.Ticks);
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(object value)
        {
            if (value == null) return TimeSpan.Zero;
            return value switch
            {
                TimeSpan ts => ts,
                DateTime64 d64 => new TimeSpan(d64.Ticks),
                DateTime dt => new TimeSpan(dt.Ticks),
                bool b => b ? new TimeSpan(1L) : TimeSpan.Zero,
                sbyte sb => new TimeSpan(sb),
                byte by => new TimeSpan(by),
                short s => new TimeSpan(s),
                ushort us => new TimeSpan(us),
                int i => new TimeSpan(i),
                uint u => new TimeSpan(u),
                long l => new TimeSpan(l),
                ulong ul => new TimeSpan(unchecked((long)ul)),
                char c => new TimeSpan(c),
                float f => ToTimeSpan(f),
                double d => ToTimeSpan(d),
                Half h => ToTimeSpan(h),
                Complex cx => ToTimeSpan(cx),
                decimal m => ToTimeSpan(m),
                string str => ToTimeSpan(str),
                _ => TimeSpan.Zero
            };
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(bool value)
        {
            return value ? new TimeSpan(1L) : TimeSpan.Zero;
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(sbyte value) => new TimeSpan(value);
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(byte value) => new TimeSpan(value);
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(short value) => new TimeSpan(value);
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(ushort value) => new TimeSpan(value);
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(int value) => new TimeSpan(value);
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(uint value) => new TimeSpan(value);
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(long value) => new TimeSpan(value);
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(ulong value) => new TimeSpan(unchecked((long)value));
        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(char value) => new TimeSpan(value);

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(float value)
        {
            return ToTimeSpan((double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(double value)
        {
            // NumPy: NaN/Inf -> NaT = int64.MinValue = TimeSpan.MinValue.Ticks (exact parity).
            // Precision note: (double)long.MaxValue rounds UP to 2^63, which is out of long
            // range. Use exclusive upper bound at 2^63 so boundary values overflow to NaT.
            if (double.IsNaN(value) || double.IsInfinity(value)) return TimeSpan.MinValue;
            if (value < (double)long.MinValue || value >= 9223372036854775808.0) return TimeSpan.MinValue;
            return new TimeSpan((long)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(Half value)
        {
            if (Half.IsNaN(value) || Half.IsInfinity(value)) return TimeSpan.MinValue;
            return new TimeSpan((long)(double)value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(System.Numerics.Complex value)
        {
            return ToTimeSpan(value.Real);
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(decimal value)
        {
            var truncated = decimal.Truncate(value);
            if (truncated < long.MinValue || truncated > long.MaxValue)
                return TimeSpan.MinValue;
            return new TimeSpan((long)truncated);
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(string value)
        {
            if (value == null) return TimeSpan.Zero;
            return TimeSpan.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static TimeSpan ToTimeSpan(string value, IFormatProvider provider)
        {
            if (value == null) return TimeSpan.Zero;
            return TimeSpan.Parse(value, provider);
        }

        // Conversions to String

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(object value)
        {
            return ToString(value, null);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(object value, IFormatProvider provider)
        {
            switch (value)
            {
                case IConvertible ic:
                    return ic.ToString(provider);
                case IFormattable formattable:
                    return formattable.ToString(null, provider);
                default:
                    return value == null ? string.Empty : value.ToString();
            }
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(bool value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString();
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(bool value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(char value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return char.ToString(value);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(char value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(sbyte value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(sbyte value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(byte value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(byte value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(short value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(short value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(ushort value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(ushort value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(int value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(int value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(uint value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(uint value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(long value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(long value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(ulong value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(OptimizeAndInline)]
        public static string ToString(ulong value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(float value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(float value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(double value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(double value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(decimal value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(decimal value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(DateTime value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.InvariantCulture);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(DateTime value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(string value)
        {
            Contract.Ensures(Contract.Result<string>() == value); // We were always skipping the null check here.
            return value;
        }

        [MethodImpl(OptimizeAndInline)]
        public static string ToString(string value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() == value); // We were always skipping the null check here.
            return value; // avoid the null check
        }

        [MethodImpl(OptimizeAndInline)]
        public static int ToBase64CharArray(byte[] inArray, int offsetIn, int length, char[] outArray, int offsetOut)
        {
            Contract.Ensures(Contract.Result<int>() >= 0);
            Contract.Ensures(Contract.Result<int>() <= outArray.Length);
            Contract.EndContractBlock();

            return ToBase64CharArray(inArray, offsetIn, length, outArray, offsetOut, Base64FormattingOptions.None);
        }

        [System.Security.SecuritySafeCritical] // auto-generated
        [System.Runtime.InteropServices.ComVisible(false)]
        [MethodImpl(OptimizeAndInline)]
        public static unsafe int ToBase64CharArray(byte[] inArray, int offsetIn, int length, char[] outArray, int offsetOut, Base64FormattingOptions options)
        {
            //Do data verfication
            if (inArray == null)
                throw new ArgumentNullException(nameof(inArray));
            if (outArray == null)
                throw new ArgumentNullException(nameof(outArray));
            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), ("ArgumentOutOfRange_Index"));
            if (offsetIn < 0)
                throw new ArgumentOutOfRangeException(nameof(offsetIn), ("ArgumentOutOfRange_GenericPositive"));
            if (offsetOut < 0)
                throw new ArgumentOutOfRangeException(nameof(offsetOut), ("ArgumentOutOfRange_GenericPositive"));

            if (options < Base64FormattingOptions.None || options > Base64FormattingOptions.InsertLineBreaks)
                throw new ArgumentNullException();

            Contract.Ensures(Contract.Result<int>() >= 0);
            Contract.Ensures(Contract.Result<int>() <= outArray.Length);
            Contract.EndContractBlock();

            int retVal;

            int inArrayLength;
            int outArrayLength;
            int numElementsToCopy;

            inArrayLength = inArray.Length;

            if (offsetIn > inArrayLength - length)
                throw new ArgumentOutOfRangeException(nameof(offsetIn), ("ArgumentOutOfRange_OffsetLength"));

            if (inArrayLength == 0)
                return 0;

            bool insertLineBreaks = (options == Base64FormattingOptions.InsertLineBreaks);
            //This is the maximally required length that must be available in the char array
            outArrayLength = outArray.Length;

            // Length of the char buffer required
            numElementsToCopy = ToBase64_CalculateAndValidateOutputLength(length, insertLineBreaks);

            if (offsetOut > outArrayLength - numElementsToCopy)
                throw new ArgumentOutOfRangeException(nameof(offsetOut), ("ArgumentOutOfRange_OffsetOut"));

            fixed (char* outChars = &outArray[offsetOut])
            {
                fixed (byte* inData = inArray)
                {
                    retVal = ConvertToBase64Array(outChars, inData, offsetIn, length, insertLineBreaks);
                }
            }

            return retVal;
        }

        [System.Security.SecurityCritical] // auto-generated
        private static unsafe int ConvertToBase64Array(char* outChars, byte* inData, int offset, int length, bool insertLineBreaks)
        {
            int lengthmod3 = length % 3;
            int calcLength = offset + (length - lengthmod3);
            int j = 0;
            int charcount = 0;
            //Convert three bytes at a time to base64 notation.  This will consume 4 chars.
            int i;

            // get a pointer to the base64Table to avoid unnecessary range checking
            fixed (char* base64 = base64Table)
            {
                for (i = offset; i < calcLength; i += 3)
                {
                    if (insertLineBreaks)
                    {
                        if (charcount == base64LineBreakPosition)
                        {
                            outChars[j++] = '\r';
                            outChars[j++] = '\n';
                            charcount = 0;
                        }

                        charcount += 4;
                    }

                    outChars[j] = base64[(inData[i] & 0xfc) >> 2];
                    outChars[j + 1] = base64[((inData[i] & 0x03) << 4) | ((inData[i + 1] & 0xf0) >> 4)];
                    outChars[j + 2] = base64[((inData[i + 1] & 0x0f) << 2) | ((inData[i + 2] & 0xc0) >> 6)];
                    outChars[j + 3] = base64[(inData[i + 2] & 0x3f)];
                    j += 4;
                }

                //Where we left off before
                i = calcLength;

                if (insertLineBreaks && (lengthmod3 != 0) && (charcount == base64LineBreakPosition))
                {
                    outChars[j++] = '\r';
                    outChars[j++] = '\n';
                }

                switch (lengthmod3)
                {
                    case 2: //One character padding needed
                        outChars[j] = base64[(inData[i] & 0xfc) >> 2];
                        outChars[j + 1] = base64[((inData[i] & 0x03) << 4) | ((inData[i + 1] & 0xf0) >> 4)];
                        outChars[j + 2] = base64[(inData[i + 1] & 0x0f) << 2];
                        outChars[j + 3] = base64[64]; //Pad
                        j += 4;
                        break;
                    case 1: // Two character padding needed
                        outChars[j] = base64[(inData[i] & 0xfc) >> 2];
                        outChars[j + 1] = base64[(inData[i] & 0x03) << 4];
                        outChars[j + 2] = base64[64]; //Pad
                        outChars[j + 3] = base64[64]; //Pad
                        j += 4;
                        break;
                }
            }

            return j;
        }

        private static int ToBase64_CalculateAndValidateOutputLength(int inputLength, bool insertLineBreaks)
        {
            long outlen = ((long)inputLength) / 3 * 4; // the base length - we want integer division here. 
            outlen += ((inputLength % 3) != 0) ? 4 : 0; // at most 4 more chars for the remainder

            if (outlen == 0)
                return 0;

            if (insertLineBreaks)
            {
                long newLines = outlen / base64LineBreakPosition;
                if ((outlen % base64LineBreakPosition) == 0)
                {
                    --newLines;
                }

                outlen += newLines * 2; // the number of line break chars we'll add, "\r\n"
            }

            // If we overflow an int then we cannot allocate enough
            // memory to output the value so throw
            if (outlen > int.MaxValue)
                throw new OutOfMemoryException();

            return (int)outlen;
        }


        /// <summary>
        /// Converts the specified string, which encodes binary data as Base64 digits, to the equivalent byte array.
        /// </summary>
        /// <param name="s">The string to convert</param>
        /// <returns>The array of bytes represented by the specifed Base64 string.</returns>
        [SecuritySafeCritical]
        [MethodImpl(OptimizeAndInline)]
        public static byte[] FromBase64String(string s)
        {
            // "s" is an unfortunate parameter name, but we need to keep it for backward compat.

            if (s == null)
                throw new ArgumentNullException(nameof(s));

            Contract.EndContractBlock();

            unsafe
            {
                fixed (char* sPtr = s)
                {
                    return FromBase64CharPtr(sPtr, s.Length);
                }
            }
        }


        /// <summary>
        /// Converts the specified range of a Char array, which encodes binary data as Base64 digits, to the equivalent byte array.     
        /// </summary>
        /// <param name="inArray">Chars representing Base64 encoding characters</param>
        /// <param name="offset">A position within the input array.</param>
        /// <param name="length">Number of element to convert.</param>
        /// <returns>The array of bytes represented by the specified Base64 encoding characters.</returns>
        [SecuritySafeCritical]
        [MethodImpl(OptimizeAndInline)]
        public static byte[] FromBase64CharArray(char[] inArray, int offset, int length)
        {
            if (inArray == null)
                throw new ArgumentNullException(nameof(inArray));

#if FEATURE_LEGACYNETCF
            Contract.EndContractBlock();
 
            // throw FormatException, to ensure compatibility with Mango Apps.
            if (CompatibilitySwitches.IsAppEarlierThanWindowsPhone8) {
                if(inArray.Length == 0) {
                     throw new FormatException();
                }
            }
#endif

            if (length < 0)
                throw new ArgumentOutOfRangeException(nameof(length), ("ArgumentOutOfRange_Index"));

            if (offset < 0)
                throw new ArgumentOutOfRangeException(nameof(offset), ("ArgumentOutOfRange_GenericPositive"));

            if (offset > inArray.Length - length)
                throw new ArgumentOutOfRangeException(nameof(offset), ("ArgumentOutOfRange_OffsetLength"));

#if !FEATURE_LEGACYNETCF // Our compat hack above breaks CCRewrite's rules on valid contracts.
            Contract.EndContractBlock();
#endif

            unsafe
            {
                fixed (char* inArrayPtr = inArray)
                {
                    return FromBase64CharPtr(inArrayPtr + offset, length);
                }
            }
        }


        /// <summary>
        /// Convert Base64 encoding characters to bytes:
        ///  - Compute result length exactly by actually walking the input;
        ///  - Allocate new result array based on computation;
        ///  - Decode input into the new array;
        /// </summary>
        /// <param name="inputPtr">Pointer to the first input char</param>
        /// <param name="inputLength">Number of input chars</param>
        /// <returns></returns>
        [SecurityCritical]
        private static unsafe byte[] FromBase64CharPtr(char* inputPtr, int inputLength)
        {
            // The validity of parameters much be checked by callers, thus we are Critical here.

            Contract.Assert(0 <= inputLength);

            // We need to get rid of any trailing white spaces.
            // Otherwise we would be rejecting input such as "abc= ":
            while (inputLength > 0)
            {
                int lastChar = inputPtr[inputLength - 1];
                if (lastChar != ' ' && lastChar != '\n' && lastChar != '\r' && lastChar != '\t')
                    break;
                inputLength--;
            }

            // Compute the output length:
            int resultLength = FromBase64_ComputeResultLength(inputPtr, inputLength);

            Contract.Assert(0 <= resultLength);

            // resultLength can be zero. We will still enter FromBase64_Decode and process the input.
            // It may either simply write no bytes (e.g. input = " ") or throw (e.g. input = "ab").

            // Create result byte blob:
            byte[] decodedBytes = new byte[resultLength];

            // Convert Base64 chars into bytes:
            int actualResultLength;
            fixed (byte* decodedBytesPtr = decodedBytes)
                actualResultLength = FromBase64_Decode(inputPtr, inputLength, decodedBytesPtr, resultLength);

            // Note that actualResultLength can differ from resultLength if the caller is modifying the array
            // as it is being converted. Silently ignore the failure.
            // Consider throwing exception in an non in-place release.

            // We are done:
            return decodedBytes;
        }


        /// <summary>
        /// Decode characters representing a Base64 encoding into bytes:
        /// Walk the input. Every time 4 chars are read, convert them to the 3 corresponding output bytes.
        /// This method is a bit lengthy on purpose. We are trying to avoid jumps to helpers in the loop
        /// to aid performance.
        /// </summary>
        /// <param name="inputPtr">Pointer to first input char</param>
        /// <param name="inputLength">Number of input chars</param>
        /// <param name="destPtr">Pointer to location for teh first result byte</param>
        /// <param name="destLength">Max length of the preallocated result buffer</param>
        /// <returns>If the result buffer was not large enough to write all result bytes, return -1;
        /// Otherwise return the number of result bytes actually produced.</returns>
        [SecurityCritical]
        private static unsafe int FromBase64_Decode(char* startInputPtr, int inputLength, byte* startDestPtr, int destLength)
        {
            // You may find this method weird to look at. It’s written for performance, not aesthetics.
            // You will find unrolled loops label jumps and bit manipulations.

            const uint intA = (uint)'A';
            const uint inta = (uint)'a';
            const uint int0 = (uint)'0';
            const uint intEq = (uint)'=';
            const uint intPlus = (uint)'+';
            const uint intSlash = (uint)'/';
            const uint intSpace = (uint)' ';
            const uint intTab = (uint)'\t';
            const uint intNLn = (uint)'\n';
            const uint intCRt = (uint)'\r';
            const uint intAtoZ = (uint)('Z' - 'A'); // = ('z' - 'a')
            const uint int0to9 = (uint)('9' - '0');

            char* inputPtr = startInputPtr;
            byte* destPtr = startDestPtr;

            // Pointers to the end of input and output:
            char* endInputPtr = inputPtr + inputLength;
            byte* endDestPtr = destPtr + destLength;

            // Current char code/value:
            uint currCode;

            // This 4-byte integer will contain the 4 codes of the current 4-char group.
            // Eeach char codes for 6 bits = 24 bits.
            // The remaining byte will be FF, we use it as a marker when 4 chars have been processed.            
            uint currBlockCodes = 0x000000FFu;

            unchecked
            {
                while (true)
                {
                    // break when done:
                    if (inputPtr >= endInputPtr)
                        goto _AllInputConsumed;

                    // Get current char:
                    currCode = *inputPtr;
                    inputPtr++;

                    // Determine current char code:

                    if (currCode - intA <= intAtoZ)
                        currCode -= intA;

                    else if (currCode - inta <= intAtoZ)
                        currCode -= (inta - 26u);

                    else if (currCode - int0 <= int0to9)
                        currCode -= (int0 - 52u);

                    else
                    {
                        // Use the slower switch for less common cases:
                        switch (currCode)
                        {
                            // Significant chars:
                            case intPlus:
                                currCode = 62u;
                                break;

                            case intSlash:
                                currCode = 63u;
                                break;

                            // Legal no-value chars (we ignore these):
                            case intCRt:
                            case intNLn:
                            case intSpace:
                            case intTab:
                                continue;

                            // The equality char is only legal at the end of the input.
                            // Jump after the loop to make it easier for the JIT register predictor to do a good job for the loop itself:
                            case intEq:
                                goto _EqualityCharEncountered;

                            // Other chars are illegal:
                            default:
                                throw new FormatException(("Format_BadBase64Char"));
                        }
                    }

                    // Ok, we got the code. Save it:
                    currBlockCodes = (currBlockCodes << 6) | currCode;

                    // Last bit in currBlockCodes will be on after in shifted right 4 times:
                    if ((currBlockCodes & 0x80000000u) != 0u)
                    {
                        if ((int)(endDestPtr - destPtr) < 3)
                            return -1;

                        *(destPtr) = (byte)(currBlockCodes >> 16);
                        *(destPtr + 1) = (byte)(currBlockCodes >> 8);
                        *(destPtr + 2) = (byte)(currBlockCodes);
                        destPtr += 3;

                        currBlockCodes = 0x000000FFu;
                    }
                }
            } // unchecked while

            // 'd be nice to have an assert that we never get here, but CS0162: Unreachable code detected.
            // Contract.Assert(false, "We only leave the above loop by jumping; should never get here.");

            // We jump here out of the loop if we hit an '=':
            _EqualityCharEncountered:

            Contract.Assert(currCode == intEq);

            // Recall that inputPtr is now one position past where '=' was read.
            // '=' can only be at the last input pos:
            if (inputPtr == endInputPtr)
            {
                // Code is zero for trailing '=':
                currBlockCodes <<= 6;

                // The '=' did not complete a 4-group. The input must be bad:
                if ((currBlockCodes & 0x80000000u) == 0u)
                    throw new FormatException(("Format_BadBase64CharArrayLength"));

                if ((int)(endDestPtr - destPtr) < 2) // Autch! We underestimated the output length!
                    return -1;

                // We are good, store bytes form this past group. We had a single "=", so we take two bytes:
                *(destPtr++) = (byte)(currBlockCodes >> 16);
                *(destPtr++) = (byte)(currBlockCodes >> 8);

                currBlockCodes = 0x000000FFu;
            }
            else
            {
                // '=' can also be at the pre-last position iff the last is also a '=' excluding the white spaces:

                // We need to get rid of any intermediate white spaces.
                // Otherwise we would be rejecting input such as "abc= =":
                while (inputPtr < (endInputPtr - 1))
                {
                    int lastChar = *(inputPtr);
                    if (lastChar != ' ' && lastChar != '\n' && lastChar != '\r' && lastChar != '\t')
                        break;
                    inputPtr++;
                }

                if (inputPtr == (endInputPtr - 1) && *(inputPtr) == '=')
                {
                    // Code is zero for each of the two '=':
                    currBlockCodes <<= 12;

                    // The '=' did not complete a 4-group. The input must be bad:
                    if ((currBlockCodes & 0x80000000u) == 0u)
                        throw new FormatException(("Format_BadBase64CharArrayLength"));

                    if ((int)(endDestPtr - destPtr) < 1) // Autch! We underestimated the output length!
                        return -1;

                    // We are good, store bytes form this past group. We had a "==", so we take only one byte:
                    *(destPtr++) = (byte)(currBlockCodes >> 16);

                    currBlockCodes = 0x000000FFu;
                }
                else // '=' is not ok at places other than the end:
                    throw new FormatException(("Format_BadBase64Char"));
            }

            // We get here either from above or by jumping out of the loop:
            _AllInputConsumed:

            // The last block of chars has less than 4 items
            if (currBlockCodes != 0x000000FFu)
                throw new FormatException(("Format_BadBase64CharArrayLength"));

            // Return how many bytes were actually recovered:
            return (int)(destPtr - startDestPtr);
        } // Int32 FromBase64_Decode(...)


        /// <summary>
        /// Compute the number of bytes encoded in the specified Base 64 char array:
        /// Walk the entire input counting white spaces and padding chars, then compute result length
        /// based on 3 bytes per 4 chars.
        /// </summary>
        [SecurityCritical]
        private static unsafe int FromBase64_ComputeResultLength(char* inputPtr, int inputLength)
        {
            const uint intEq = (uint)'=';
            const uint intSpace = (uint)' ';

            Contract.Assert(0 <= inputLength);

            char* inputEndPtr = inputPtr + inputLength;
            int usefulInputLength = inputLength;
            int padding = 0;

            while (inputPtr < inputEndPtr)
            {
                uint c = *inputPtr;
                inputPtr++;

                // We want to be as fast as possible and filter out spaces with as few comparisons as possible.
                // We end up accepting a number of illegal chars as legal white-space chars.
                // This is ok: as soon as we hit them during actual decode we will recognise them as illegal and throw.
                if (c <= intSpace)
                    usefulInputLength--;

                else if (c == intEq)
                {
                    usefulInputLength--;
                    padding++;
                }
            }

            Contract.Assert(0 <= usefulInputLength);

            // For legal input, we can assume that 0 <= padding < 3. But it may be more for illegal input.
            // We will notice it at decode when we see a '=' at the wrong place.
            Contract.Assert(0 <= padding);

            // Perf: reuse the variable that stored the number of '=' to store the number of bytes encoded by the
            // last group that contains the '=':
            if (padding != 0)
            {
                if (padding == 1)
                    padding = 2;
                else if (padding == 2)
                    padding = 1;
                else
                    throw new FormatException(("Format_BadBase64Char"));
            }

            // Done:
            return (usefulInputLength / 4) * 3 + padding;
        }
    }
}
