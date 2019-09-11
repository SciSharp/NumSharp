using System;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics.Contracts;
using System.Globalization;
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static object ChangeType(object value, TypeCode typeCode)
        {
            return ChangeType(value, typeCode, Thread.CurrentThread.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static object ChangeType(object value, TypeCode typeCode, IFormatProvider provider)
        {
            if (value == null && (typeCode == TypeCode.Empty || typeCode == TypeCode.String || typeCode == TypeCode.Object))
            {
                return null;
            }


            // This line is invalid for things like Enums that return a TypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetTypeCode() == typeCode) return value;
            switch (typeCode)
            {
                case TypeCode.Boolean:
                    return ((IConvertible)value).ToBoolean(provider);
                case TypeCode.Char:
                    return ((IConvertible)value).ToChar(provider);
                case TypeCode.SByte:
                    return ((IConvertible)value).ToSByte(provider);
                case TypeCode.Byte:
                    return ((IConvertible)value).ToByte(provider);
                case TypeCode.Int16:
                    return ((IConvertible)value).ToInt16(provider);
                case TypeCode.UInt16:
                    return ((IConvertible)value).ToUInt16(provider);
                case TypeCode.Int32:
                    return ((IConvertible)value).ToInt32(provider);
                case TypeCode.UInt32:
                    return ((IConvertible)value).ToUInt32(provider);
                case TypeCode.Int64:
                    return ((IConvertible)value).ToInt64(provider);
                case TypeCode.UInt64:
                    return ((IConvertible)value).ToUInt64(provider);
                case TypeCode.Single:
                    return ((IConvertible)value).ToSingle(provider);
                case TypeCode.Double:
                    return ((IConvertible)value).ToDouble(provider);
                case TypeCode.Decimal:
                    return ((IConvertible)value).ToDecimal(provider);
                case TypeCode.DateTime:
                    return ((IConvertible)value).ToDateTime(provider);
                case TypeCode.String:
                    return ((IConvertible)value).ToString(provider);
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(object value)
        {
            return value != null && ((IConvertible)value).ToBoolean(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(object value, IFormatProvider provider)
        {
            return value != null && ((IConvertible)value).ToBoolean(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(bool value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(sbyte value)
        {
            return value != 0;
        }

        // To be consistent with IConvertible in the base data types else we get different semantics
        // with widening operations. Without this operator this widen succeeds,with this API the widening throws.
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(char value)
        {
            return ((IConvertible)value).ToBoolean(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(byte value)
        {
            return value != 0;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(short value)
        {
            return value != 0;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(ushort value)
        {
            return value != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(int value)
        {
            return value != 0;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(uint value)
        {
            return value != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(long value)
        {
            return value != 0;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(ulong value)
        {
            return value != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(string value)
        {
            if (value == null)
                return false;
            return bool.Parse(value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(string value, IFormatProvider provider)
        {
            if (value == null)
                return false;
            return bool.Parse(value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(float value)
        {
            return value != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(double value)
        {
            return value != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(decimal value)
        {
            return value != 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static bool ToBoolean(DateTime value)
        {
            return ((IConvertible)value).ToBoolean(null);
        }

        // Disallowed conversions to Boolean
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static bool ToBoolean(TimeSpan value)

        // Conversions to Char


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(object value)
        {
            return value == null ? (char)0 : ((IConvertible)value).ToChar(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(object value, IFormatProvider provider)
        {
            return value == null ? (char)0 : ((IConvertible)value).ToChar(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(bool value)
        {
            return ((IConvertible)value).ToChar(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(char value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(sbyte value)
        {
            if (value < 0) throw new OverflowException(("Overflow_Char"));
            Contract.EndContractBlock();
            return (char)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(byte value)
        {
            return (char)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(short value)
        {
            if (value < 0) throw new OverflowException(("Overflow_Char"));
            Contract.EndContractBlock();
            return (char)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(ushort value)
        {
            return (char)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(int value)
        {
            if (value < 0 || value > char.MaxValue) throw new OverflowException(("Overflow_Char"));
            Contract.EndContractBlock();
            return (char)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(uint value)
        {
            if (value > char.MaxValue) throw new OverflowException(("Overflow_Char"));
            Contract.EndContractBlock();
            return (char)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(long value)
        {
            if (value < 0 || value > char.MaxValue) throw new OverflowException(("Overflow_Char"));
            Contract.EndContractBlock();
            return (char)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(ulong value)
        {
            if (value > char.MaxValue) throw new OverflowException(("Overflow_Char"));
            Contract.EndContractBlock();
            return (char)value;
        }

        //
        // @VariantSwitch
        // Remove FormatExceptions;
        //
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(string value)
        {
            return ToChar(value, null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(string value, IFormatProvider provider)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            Contract.EndContractBlock();

            if (value.Length != 1)
                throw new FormatException(("ResId.Format_NeedSingleChar"));

            return value[0];
        }

        // To be consistent with IConvertible in the base data types else we get different semantics
        // with widening operations. Without this operator this widen succeeds,with this API the widening throws.
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(float value)
        {
            return ((IConvertible)value).ToChar(null);
        }

        // To be consistent with IConvertible in the base data types else we get different semantics
        // with widening operations. Without this operator this widen succeeds,with this API the widening throws.
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(double value)
        {
            return ((IConvertible)value).ToChar(null);
        }

        // To be consistent with IConvertible in the base data types else we get different semantics
        // with widening operations. Without this operator this widen succeeds,with this API the widening throws.
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(decimal value)
        {
            return ((IConvertible)value).ToChar(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static char ToChar(DateTime value)
        {
            return ((IConvertible)value).ToChar(null);
        }


        // Disallowed conversions to Char
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static char ToChar(TimeSpan value)

        // Conversions to SByte


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(object value)
        {
            return value == null ? (sbyte)0 : ((IConvertible)value).ToSByte(null);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(object value, IFormatProvider provider)
        {
            return value == null ? (sbyte)0 : ((IConvertible)value).ToSByte(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(bool value)
        {
            return value ? (sbyte)1 : (sbyte)0;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(sbyte value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(char value)
        {
            if (value > sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(byte value)
        {
            if (value > sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(short value)
        {
            if (value < sbyte.MinValue || value > sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(ushort value)
        {
            if (value > sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(int value)
        {
            if (value < sbyte.MinValue || value > sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(uint value)
        {
            if (value > sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(long value)
        {
            if (value < sbyte.MinValue || value > sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(ulong value)
        {
            if (value > (ulong)sbyte.MaxValue) throw new OverflowException(("Overflow_SByte"));
            Contract.EndContractBlock();
            return (sbyte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(float value)
        {
            return ToSByte((double)value);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(double value)
        {
            return ToSByte(ToInt32(value));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(decimal value)
        {
            return decimal.ToSByte(decimal.Round(value, 0));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(string value)
        {
            if (value == null)
                return 0;
            return sbyte.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(string value, IFormatProvider provider)
        {
            return sbyte.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static sbyte ToSByte(DateTime value)
        {
            return ((IConvertible)value).ToSByte(null);
        }

        // Disallowed conversions to SByte
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static sbyte ToSByte(TimeSpan value)

        // Conversions to Byte

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(object value)
        {
            return value == null ? (byte)0 : ((IConvertible)value).ToByte(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(object value, IFormatProvider provider)
        {
            return value == null ? (byte)0 : ((IConvertible)value).ToByte(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(bool value)
        {
            return value ? (byte)1 : (byte)0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(byte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(char value)
        {
            if (value > byte.MaxValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(sbyte value)
        {
            if (value < byte.MinValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(short value)
        {
            if (value < byte.MinValue || value > byte.MaxValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(ushort value)
        {
            if (value > byte.MaxValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(int value)
        {
            if (value < byte.MinValue || value > byte.MaxValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(uint value)
        {
            if (value > byte.MaxValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(long value)
        {
            if (value < byte.MinValue || value > byte.MaxValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(ulong value)
        {
            if (value > byte.MaxValue) throw new OverflowException(("Overflow_Byte"));
            Contract.EndContractBlock();
            return (byte)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(float value)
        {
            return ToByte((double)value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(double value)
        {
            return ToByte(ToInt32(value));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(decimal value)
        {
            return decimal.ToByte(decimal.Round(value, 0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(string value)
        {
            if (value == null)
                return 0;
            return byte.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return byte.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static byte ToByte(DateTime value)
        {
            return ((IConvertible)value).ToByte(null);
        }


        // Disallowed conversions to Byte
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static byte ToByte(TimeSpan value)

        // Conversions to Int16

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(object value)
        {
            return value == null ? (short)0 : ((IConvertible)value).ToInt16(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(object value, IFormatProvider provider)
        {
            return value == null ? (short)0 : ((IConvertible)value).ToInt16(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(bool value)
        {
            return value ? (short)1 : (short)0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(char value)
        {
            if (value > short.MaxValue) throw new OverflowException(("Overflow_Int16"));
            Contract.EndContractBlock();
            return (short)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(sbyte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(byte value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(ushort value)
        {
            if (value > short.MaxValue) throw new OverflowException(("Overflow_Int16"));
            Contract.EndContractBlock();
            return (short)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(int value)
        {
            if (value < short.MinValue || value > short.MaxValue) throw new OverflowException(("Overflow_Int16"));
            Contract.EndContractBlock();
            return (short)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(uint value)
        {
            if (value > short.MaxValue) throw new OverflowException(("Overflow_Int16"));
            Contract.EndContractBlock();
            return (short)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(short value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(long value)
        {
            if (value < short.MinValue || value > short.MaxValue) throw new OverflowException(("Overflow_Int16"));
            Contract.EndContractBlock();
            return (short)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(ulong value)
        {
            if (value > (ulong)short.MaxValue) throw new OverflowException(("Overflow_Int16"));
            Contract.EndContractBlock();
            return (short)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(float value)
        {
            return ToInt16((double)value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(double value)
        {
            return ToInt16(ToInt32(value));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(decimal value)
        {
            return decimal.ToInt16(decimal.Round(value, 0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(string value)
        {
            if (value == null)
                return 0;
            return short.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return short.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static short ToInt16(DateTime value)
        {
            return ((IConvertible)value).ToInt16(null);
        }


        // Disallowed conversions to Int16
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static short ToInt16(TimeSpan value)

        // Conversions to UInt16


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(object value)
        {
            return value == null ? (ushort)0 : ((IConvertible)value).ToUInt16(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(object value, IFormatProvider provider)
        {
            return value == null ? (ushort)0 : ((IConvertible)value).ToUInt16(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(bool value)
        {
            return value ? (ushort)1 : (ushort)0;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(char value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(sbyte value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt16"));
            Contract.EndContractBlock();
            return (ushort)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(byte value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(short value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt16"));
            Contract.EndContractBlock();
            return (ushort)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(int value)
        {
            if (value < 0 || value > ushort.MaxValue) throw new OverflowException(("Overflow_UInt16"));
            Contract.EndContractBlock();
            return (ushort)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(ushort value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(uint value)
        {
            if (value > ushort.MaxValue) throw new OverflowException(("Overflow_UInt16"));
            Contract.EndContractBlock();
            return (ushort)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(long value)
        {
            if (value < 0 || value > ushort.MaxValue) throw new OverflowException(("Overflow_UInt16"));
            Contract.EndContractBlock();
            return (ushort)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(ulong value)
        {
            if (value > ushort.MaxValue) throw new OverflowException(("Overflow_UInt16"));
            Contract.EndContractBlock();
            return (ushort)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(float value)
        {
            return ToUInt16((double)value);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(double value)
        {
            return ToUInt16(ToInt32(value));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(decimal value)
        {
            return decimal.ToUInt16(decimal.Round(value, 0));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(string value)
        {
            if (value == null)
                return 0;
            return ushort.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return ushort.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ushort ToUInt16(DateTime value)
        {
            return ((IConvertible)value).ToUInt16(null);
        }

        // Disallowed conversions to UInt16
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static ushort ToUInt16(TimeSpan value)

        // Conversions to Int32

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(object value)
        {
            return value == null ? 0 : ((IConvertible)value).ToInt32(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(object value, IFormatProvider provider)
        {
            return value == null ? 0 : ((IConvertible)value).ToInt32(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(bool value)
        {
            return value ? 1 : 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(char value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(sbyte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(byte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(short value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(ushort value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(uint value)
        {
            if (value > int.MaxValue) throw new OverflowException(("Overflow_Int32"));
            Contract.EndContractBlock();
            return (int)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(int value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(long value)
        {
            if (value < int.MinValue || value > int.MaxValue) throw new OverflowException(("Overflow_Int32"));
            Contract.EndContractBlock();
            return (int)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(ulong value)
        {
            if (value > int.MaxValue) throw new OverflowException(("Overflow_Int32"));
            Contract.EndContractBlock();
            return (int)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(float value)
        {
            return ToInt32((double)value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(double value)
        {
            if (value >= 0)
            {
                if (value < 2147483647.5)
                {
                    int result = (int)value;
                    double dif = value - result;
                    if (dif > 0.5 || dif == 0.5 && (result & 1) != 0) result++;
                    return result;
                }
            }
            else
            {
                if (value >= -2147483648.5)
                {
                    int result = (int)value;
                    double dif = value - result;
                    if (dif < -0.5 || dif == -0.5 && (result & 1) != 0) result--;
                    return result;
                }
            }

            throw new OverflowException(("Overflow_Int32"));
        }

        [System.Security.SecuritySafeCritical] // auto-generated
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(decimal value)
        {
            return Converts.ToInt32(value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(string value)
        {
            if (value == null)
                return 0;
            return int.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return int.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToInt32(DateTime value)
        {
            return ((IConvertible)value).ToInt32(null);
        }


        // Disallowed conversions to Int32
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static int ToInt32(TimeSpan value)

        // Conversions to UInt32


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(object value)
        {
            return value == null ? 0 : ((IConvertible)value).ToUInt32(null);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(object value, IFormatProvider provider)
        {
            return value == null ? 0 : ((IConvertible)value).ToUInt32(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(bool value)
        {
            return value ? 1u : 0u;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(char value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(sbyte value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt32"));
            Contract.EndContractBlock();
            return (uint)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(byte value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(short value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt32"));
            Contract.EndContractBlock();
            return (uint)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(ushort value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(int value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt32"));
            Contract.EndContractBlock();
            return (uint)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(uint value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(long value)
        {
            if (value < 0 || value > uint.MaxValue) throw new OverflowException(("Overflow_UInt32"));
            Contract.EndContractBlock();
            return (uint)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(ulong value)
        {
            if (value > uint.MaxValue) throw new OverflowException(("Overflow_UInt32"));
            Contract.EndContractBlock();
            return (uint)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(float value)
        {
            return ToUInt32((double)value);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(double value)
        {
            if (value >= -0.5 && value < 4294967295.5)
            {
                uint result = (uint)value;
                double dif = value - result;
                if (dif > 0.5 || dif == 0.5 && (result & 1) != 0) result++;
                return result;
            }

            throw new OverflowException(("Overflow_UInt32"));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(decimal value)
        {
            return decimal.ToUInt32(decimal.Round(value, 0));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(string value)
        {
            if (value == null)
                return 0;
            return uint.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return uint.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static uint ToUInt32(DateTime value)
        {
            return ((IConvertible)value).ToUInt32(null);
        }

        // Disallowed conversions to UInt32
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static uint ToUInt32(TimeSpan value)

        // Conversions to Int64

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(object value)
        {
            return value == null ? 0 : ((IConvertible)value).ToInt64(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(object value, IFormatProvider provider)
        {
            return value == null ? 0 : ((IConvertible)value).ToInt64(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(bool value)
        {
            return value ? 1L : 0L;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(char value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(sbyte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(byte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(short value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(ushort value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(int value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(uint value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(ulong value)
        {
            if (value > long.MaxValue) throw new OverflowException(("Overflow_Int64"));
            Contract.EndContractBlock();
            return (long)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(long value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(float value)
        {
            return ToInt64((double)value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(double value)
        {
            return checked((long)Math.Round(value));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(decimal value)
        {
            return decimal.ToInt64(decimal.Round(value, 0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(string value)
        {
            if (value == null)
                return 0;
            return long.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return long.Parse(value, NumberStyles.Integer, provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static long ToInt64(DateTime value)
        {
            return ((IConvertible)value).ToInt64(null);
        }

        // Disallowed conversions to Int64
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static long ToInt64(TimeSpan value)

        // Conversions to UInt64


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(object value)
        {
            return value == null ? 0 : ((IConvertible)value).ToUInt64(null);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(object value, IFormatProvider provider)
        {
            return value == null ? 0 : ((IConvertible)value).ToUInt64(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(bool value)
        {
            return value ? 1ul : 0ul;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(char value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(sbyte value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt64"));
            Contract.EndContractBlock();
            return (ulong)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(byte value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(short value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt64"));
            Contract.EndContractBlock();
            return (ulong)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(ushort value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(int value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt64"));
            Contract.EndContractBlock();
            return (ulong)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(uint value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(long value)
        {
            if (value < 0) throw new OverflowException(("Overflow_UInt64"));
            Contract.EndContractBlock();
            return (ulong)value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(ulong value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(float value)
        {
            return ToUInt64((double)value);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(double value)
        {
            return checked((ulong)Math.Round(value));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(decimal value)
        {
            return decimal.ToUInt64(decimal.Round(value, 0));
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(string value)
        {
            if (value == null)
                return 0;
            return ulong.Parse(value, CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return ulong.Parse(value, NumberStyles.Integer, provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static ulong ToUInt64(DateTime value)
        {
            return ((IConvertible)value).ToUInt64(null);
        }

        // Disallowed conversions to UInt64
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static ulong ToUInt64(TimeSpan value)

        // Conversions to Single

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(object value)
        {
            return value == null ? 0 : ((IConvertible)value).ToSingle(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(object value, IFormatProvider provider)
        {
            return value == null ? 0 : ((IConvertible)value).ToSingle(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(sbyte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(byte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(char value)
        {
            return ((IConvertible)value).ToSingle(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(short value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(ushort value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(int value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(uint value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(long value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(ulong value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(float value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(double value)
        {
            return (float)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(decimal value)
        {
            return (float)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(string value)
        {
            if (value == null)
                return 0;
            return float.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return float.Parse(value, NumberStyles.Float | NumberStyles.AllowThousands, provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(bool value)
        {
            return value ? 1f : 0f;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static float ToSingle(DateTime value)
        {
            return ((IConvertible)value).ToSingle(null);
        }

        // Disallowed conversions to Single
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static float ToSingle(TimeSpan value)

        // Conversions to Double

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(object value)
        {
            return value == null ? 0 : ((IConvertible)value).ToDouble(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(object value, IFormatProvider provider)
        {
            return value == null ? 0 : ((IConvertible)value).ToDouble(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(sbyte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(byte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(short value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(char value)
        {
            return ((IConvertible)value).ToDouble(null);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(ushort value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(int value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(uint value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(long value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(ulong value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(float value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(double value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(decimal value)
        {
            return (double)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(string value)
        {
            if (value == null)
                return 0;
            return double.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0;
            return double.Parse(value, NumberStyles.Float | NumberStyles.AllowThousands, provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(bool value)
        {
            return value ? 1d : 0d;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static double ToDouble(DateTime value)
        {
            return ((IConvertible)value).ToDouble(null);
        }

        // Disallowed conversions to Double
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static double ToDouble(TimeSpan value)

        // Conversions to Decimal

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(object value)
        {
            return value == null ? 0 : ((IConvertible)value).ToDecimal(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(object value, IFormatProvider provider)
        {
            return value == null ? 0 : ((IConvertible)value).ToDecimal(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(sbyte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(byte value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(char value)
        {
            return ((IConvertible)value).ToDecimal(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(short value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(ushort value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(int value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(uint value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(long value)
        {
            return value;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(ulong value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(float value)
        {
            return (decimal)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(double value)
        {
            return (decimal)value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(string value)
        {
            if (value == null)
                return 0m;
            return decimal.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(string value, IFormatProvider provider)
        {
            if (value == null)
                return 0m;
            return decimal.Parse(value, NumberStyles.Number, provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(decimal value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(bool value)
        {
            return value ? 1m : 0m;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static decimal ToDecimal(DateTime value)
        {
            return ((IConvertible)value).ToDecimal(null);
        }

        // Disallowed conversions to Decimal
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static decimal ToDecimal(TimeSpan value)

        // Conversions to DateTime

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(DateTime value)
        {
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(object value)
        {
            return value == null ? DateTime.MinValue : ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(object value, IFormatProvider provider)
        {
            return value == null ? DateTime.MinValue : ((IConvertible)value).ToDateTime(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(string value)
        {
            if (value == null)
                return new DateTime(0);
            return DateTime.Parse(value, CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(string value, IFormatProvider provider)
        {
            if (value == null)
                return new DateTime(0);
            return DateTime.Parse(value, provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(sbyte value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(byte value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(short value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(ushort value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(int value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(uint value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(long value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(ulong value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(bool value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(char value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(float value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(double value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static DateTime ToDateTime(decimal value)
        {
            return ((IConvertible)value).ToDateTime(null);
        }

        // Disallowed conversions to DateTime
        // [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions) 512)] public static DateTime ToDateTime(TimeSpan value)

        // Conversions to String

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(object value)
        {
            return ToString(value, null);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
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

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(bool value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(bool value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(char value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return char.ToString(value);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(char value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(sbyte value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(sbyte value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(byte value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(byte value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(short value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(short value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(ushort value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(ushort value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(int value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(int value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(uint value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(uint value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(long value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(long value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(ulong value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(ulong value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(float value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(float value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(double value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(double value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(decimal value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.CurrentCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(decimal value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(DateTime value)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(CultureInfo.InvariantCulture);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(DateTime value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() != null);
            return value.ToString(provider);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(string value)
        {
            Contract.Ensures(Contract.Result<string>() == value); // We were always skipping the null check here.
            return value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static string ToString(string value, IFormatProvider provider)
        {
            Contract.Ensures(Contract.Result<string>() == value); // We were always skipping the null check here.
            return value; // avoid the null check
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
        public static int ToBase64CharArray(byte[] inArray, int offsetIn, int length, char[] outArray, int offsetOut)
        {
            Contract.Ensures(Contract.Result<int>() >= 0);
            Contract.Ensures(Contract.Result<int>() <= outArray.Length);
            Contract.EndContractBlock();

            return ToBase64CharArray(inArray, offsetIn, length, outArray, offsetOut, Base64FormattingOptions.None);
        }

        [System.Security.SecuritySafeCritical] // auto-generated
        [System.Runtime.InteropServices.ComVisible(false)]
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
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
        [MethodImpl(MethodImplOptions.AggressiveInlining | (MethodImplOptions)512)]
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
