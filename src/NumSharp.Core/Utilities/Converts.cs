using System;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides various methods related to <see cref="System.Convert"/>.
    /// </summary>
    public static partial class Converts
    {
        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl((MethodImplOptions)512)]
        public static TOut ChangeType<TOut>(Object value)
        {
            if (value == null)
                return default;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;
            switch (InfoOf<TOut>.NPTypeCode)
            {
                case NPTypeCode.Boolean:
                    return (TOut)(object)((IConvertible)value).ToBoolean(CultureInfo.InvariantCulture);
                case NPTypeCode.Char:
                    return (TOut)(object)((IConvertible)value).ToChar(CultureInfo.InvariantCulture);
                case NPTypeCode.Byte:
                    return (TOut)(object)((IConvertible)value).ToByte(CultureInfo.InvariantCulture);
                case NPTypeCode.Int16:
                    return (TOut)(object)((IConvertible)value).ToInt16(CultureInfo.InvariantCulture);
                case NPTypeCode.UInt16:
                    return (TOut)(object)((IConvertible)value).ToUInt16(CultureInfo.InvariantCulture);
                case NPTypeCode.Int32:
                    return (TOut)(object)((IConvertible)value).ToInt32(CultureInfo.InvariantCulture);
                case NPTypeCode.UInt32:
                    return (TOut)(object)((IConvertible)value).ToUInt32(CultureInfo.InvariantCulture);
                case NPTypeCode.Int64:
                    return (TOut)(object)((IConvertible)value).ToInt64(CultureInfo.InvariantCulture);
                case NPTypeCode.UInt64:
                    return (TOut)(object)((IConvertible)value).ToUInt64(CultureInfo.InvariantCulture);
                case NPTypeCode.Single:
                    return (TOut)(object)((IConvertible)value).ToSingle(CultureInfo.InvariantCulture);
                case NPTypeCode.Double:
                    return (TOut)(object)((IConvertible)value).ToDouble(CultureInfo.InvariantCulture);
                case NPTypeCode.Decimal:
                    return (TOut)(object)((IConvertible)value).ToDecimal(CultureInfo.InvariantCulture);
                case NPTypeCode.String:
                    return (TOut)(object)((IConvertible)value).ToString(CultureInfo.InvariantCulture);
                case NPTypeCode.Empty:
                    throw new InvalidCastException("InvalidCast_Empty");
                default:
                    throw new ArgumentException("Arg_UnknownNPTypeCode");
            }
        }


        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl((MethodImplOptions)512)]
        public static Object ChangeType(Object value, NPTypeCode typeCode)
        {
            if (value == null && (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String))
                return null;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;
            switch (typeCode)
            {
                case NPTypeCode.Boolean:
                    return ((IConvertible)value).ToBoolean(CultureInfo.InvariantCulture);
                case NPTypeCode.Char:
                    return ((IConvertible)value).ToChar(CultureInfo.InvariantCulture);
                case NPTypeCode.Byte:
                    return ((IConvertible)value).ToByte(CultureInfo.InvariantCulture);
                case NPTypeCode.Int16:
                    return ((IConvertible)value).ToInt16(CultureInfo.InvariantCulture);
                case NPTypeCode.UInt16:
                    return ((IConvertible)value).ToUInt16(CultureInfo.InvariantCulture);
                case NPTypeCode.Int32:
                    return ((IConvertible)value).ToInt32(CultureInfo.InvariantCulture);
                case NPTypeCode.UInt32:
                    return ((IConvertible)value).ToUInt32(CultureInfo.InvariantCulture);
                case NPTypeCode.Int64:
                    return ((IConvertible)value).ToInt64(CultureInfo.InvariantCulture);
                case NPTypeCode.UInt64:
                    return ((IConvertible)value).ToUInt64(CultureInfo.InvariantCulture);
                case NPTypeCode.Single:
                    return ((IConvertible)value).ToSingle(CultureInfo.InvariantCulture);
                case NPTypeCode.Double:
                    return ((IConvertible)value).ToDouble(CultureInfo.InvariantCulture);
                case NPTypeCode.Decimal:
                    return ((IConvertible)value).ToDecimal(CultureInfo.InvariantCulture);
                case NPTypeCode.String:
                    return ((IConvertible)value).ToString(CultureInfo.InvariantCulture);
                case NPTypeCode.Empty:
                    throw new InvalidCastException("InvalidCast_Empty");
                default:
                    throw new ArgumentException("Arg_UnknownNPTypeCode");
            }
        }

        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl((MethodImplOptions)512)]
        public static Object ChangeType<T>(T value, NPTypeCode typeCode) where T : IConvertible
        {
            if (value == null && (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String))
                return null;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;

            #if _REGEN
            switch (typeCode)
            {
	            %foreach supported_dtypes, supported_dtypes_lowercase%
	            case NPTypeCode.#1:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                	    %foreach supported_dtypes, supported_dtypes_lowercase%
	                    case NPTypeCode.#101: return Converts.To#1(Unsafe.As<T, #102>(ref value));
	                    %
                        default: throw new NotSupportedException();
                    }
	            %
	            default:
		            throw new NotSupportedException();
            }
            #else

            switch (typeCode)
            {
                case NPTypeCode.Boolean:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToBoolean(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToBoolean(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToBoolean(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToBoolean(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToBoolean(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToBoolean(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToBoolean(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToBoolean(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToBoolean(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToBoolean(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToBoolean(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToBoolean(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Byte:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToByte(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToByte(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToByte(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToByte(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToByte(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToByte(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToByte(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToByte(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToByte(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToByte(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToByte(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToByte(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Int16:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToInt16(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToInt16(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToInt16(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToInt16(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToInt16(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToInt16(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToInt16(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToInt16(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToInt16(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToInt16(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToInt16(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToInt16(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.UInt16:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToUInt16(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToUInt16(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToUInt16(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToUInt16(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToUInt16(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToUInt16(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToUInt16(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToUInt16(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToUInt16(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToUInt16(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToUInt16(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToUInt16(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Int32:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToInt32(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToInt32(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToInt32(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToInt32(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToInt32(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToInt32(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToInt32(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToInt32(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToInt32(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToInt32(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToInt32(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToInt32(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.UInt32:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToUInt32(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToUInt32(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToUInt32(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToUInt32(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToUInt32(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToUInt32(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToUInt32(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToUInt32(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToUInt32(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToUInt32(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToUInt32(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToUInt32(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Int64:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToInt64(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToInt64(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToInt64(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToInt64(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToInt64(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToInt64(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToInt64(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToInt64(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToInt64(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToInt64(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToInt64(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToInt64(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.UInt64:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToUInt64(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToUInt64(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToUInt64(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToUInt64(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToUInt64(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToUInt64(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToUInt64(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToUInt64(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToUInt64(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToUInt64(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToUInt64(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToUInt64(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Char:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToChar(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToChar(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToChar(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToChar(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToChar(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToChar(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToChar(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToChar(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToChar(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToChar(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToChar(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToChar(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Double:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToDouble(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToDouble(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToDouble(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToDouble(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToDouble(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToDouble(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToDouble(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToDouble(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToDouble(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToDouble(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToDouble(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToDouble(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Single:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToSingle(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToSingle(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToSingle(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToSingle(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToSingle(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToSingle(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToSingle(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToSingle(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToSingle(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToSingle(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToSingle(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToSingle(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                case NPTypeCode.Decimal:
                    switch (InfoOf<T>.NPTypeCode)
                    {
                        case NPTypeCode.Boolean: return Converts.ToDecimal(Unsafe.As<T, bool>(ref value));
                        case NPTypeCode.Byte:    return Converts.ToDecimal(Unsafe.As<T, byte>(ref value));
                        case NPTypeCode.Int16:   return Converts.ToDecimal(Unsafe.As<T, short>(ref value));
                        case NPTypeCode.UInt16:  return Converts.ToDecimal(Unsafe.As<T, ushort>(ref value));
                        case NPTypeCode.Int32:   return Converts.ToDecimal(Unsafe.As<T, int>(ref value));
                        case NPTypeCode.UInt32:  return Converts.ToDecimal(Unsafe.As<T, uint>(ref value));
                        case NPTypeCode.Int64:   return Converts.ToDecimal(Unsafe.As<T, long>(ref value));
                        case NPTypeCode.UInt64:  return Converts.ToDecimal(Unsafe.As<T, ulong>(ref value));
                        case NPTypeCode.Char:    return Converts.ToDecimal(Unsafe.As<T, char>(ref value));
                        case NPTypeCode.Double:  return Converts.ToDecimal(Unsafe.As<T, double>(ref value));
                        case NPTypeCode.Single:  return Converts.ToDecimal(Unsafe.As<T, float>(ref value));
                        case NPTypeCode.Decimal: return Converts.ToDecimal(Unsafe.As<T, decimal>(ref value));
                        default:
                            throw new NotSupportedException();
                    }
                default:
                    throw new NotSupportedException();
            }
            #endif
        }


        /// <summary>Returns an object of the specified type whose value is equivalent to the specified object.</summary>
        /// <param name="value">An object that implements the <see cref="T:System.IConvertible"></see> interface.</param>
        /// <param name="typeCode">The type of object to return.</param>
        /// <returns>An object whose underlying type is <paramref name="typeCode">typeCode</paramref> and whose value is equivalent to <paramref name="value">value</paramref>.
        /// -or-
        /// A null reference (Nothing in Visual Basic), if <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> is <see cref="System.TypeCode.Empty"></see>, <see cref="System.TypeCode.String"></see>, or <see cref="System.TypeCode.Object"></see>.</returns>
        /// <exception cref="T:System.InvalidCastException">This conversion is not supported.
        /// -or-
        /// <paramref name="value">value</paramref> is null and <paramref name="typeCode">typeCode</paramref> specifies a value type.
        /// -or-
        /// <paramref name="value">value</paramref> does not implement the <see cref="System.IConvertible"></see> interface.</exception>
        /// <exception cref="T:System.FormatException"><paramref name="value">value</paramref> is not in a format recognized by the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.OverflowException"><paramref name="value">value</paramref> represents a number that is out of the range of the <paramref name="typeCode">typeCode</paramref> type.</exception>
        /// <exception cref="T:System.ArgumentException"><paramref name="typeCode">typeCode</paramref> is invalid.</exception>
        [MethodImpl((MethodImplOptions)512)]
        public static Object ChangeType(Object value, NPTypeCode typeCode, IFormatProvider provider)
        {
            if (value == null && (typeCode == NPTypeCode.Empty || typeCode == NPTypeCode.String))
                return null;

            // This line is invalid for things like Enums that return a NPTypeCode
            // of Int32, but the object can't actually be cast to an Int32.
            //            if (v.GetNPTypeCode() == NPTypeCode) return value;
            switch (typeCode)
            {
                case NPTypeCode.Boolean:
                    return ((IConvertible)value).ToBoolean(provider);
                case NPTypeCode.Char:
                    return ((IConvertible)value).ToChar(provider);
                case NPTypeCode.Byte:
                    return ((IConvertible)value).ToByte(provider);
                case NPTypeCode.Int16:
                    return ((IConvertible)value).ToInt16(provider);
                case NPTypeCode.UInt16:
                    return ((IConvertible)value).ToUInt16(provider);
                case NPTypeCode.Int32:
                    return ((IConvertible)value).ToInt32(provider);
                case NPTypeCode.UInt32:
                    return ((IConvertible)value).ToUInt32(provider);
                case NPTypeCode.Int64:
                    return ((IConvertible)value).ToInt64(provider);
                case NPTypeCode.UInt64:
                    return ((IConvertible)value).ToUInt64(provider);
                case NPTypeCode.Single:
                    return ((IConvertible)value).ToSingle(provider);
                case NPTypeCode.Double:
                    return ((IConvertible)value).ToDouble(provider);
                case NPTypeCode.Decimal:
                    return ((IConvertible)value).ToDecimal(provider);
                case NPTypeCode.String:
                    return ((IConvertible)value).ToString(provider);
                case NPTypeCode.Empty:
                    throw new InvalidCastException("InvalidCast_Empty");
                default:
                    throw new ArgumentException("Arg_UnknownNPTypeCode");
            }
        }


        /// <summary>
        ///     Finds the conversion method from <see cref="Convert"/> based on <typeparamref name="TIn"/> and <typeparamref name="TOut"/>.
        /// </summary>
        /// <typeparam name="TIn">The type that is expected as input and to be converted from</typeparam>
        /// <typeparam name="TOut">The type we expect to convert to.</typeparam>
        public static Func<TIn, TOut> FindConverter<TIn, TOut>()
        {
            #if _REGEN
#region Compute
            //#n is input, #10n is output
		    switch (InfoOf<TIn>.NPTypeCode)
		    {
			    %foreach supported_dtypes,supported_dtypes_lowercase%
			    case NPTypeCode.#1:
			    {
				    switch (InfoOf<TOut>.NPTypeCode)
		            {
			            %foreach supported_dtypes,supported_dtypes_lowercase%
			            case NPTypeCode.#101:
			            {
				            Func<#2, #102> ret = Converts.55To#101;
                            return (Func<TIn, TOut>) (object) ret;
			            }
			            %
			            default:
                        {
                            var tout = typeof(TOut);
                            return @in => (TOut)Convert.ChangeType(@in, tout);
                        }
		            }
			    }
			    %
			    default:
                {
                    var tout = typeof(TOut);
                    return @in => (TOut)Convert.ChangeType(@in, tout);
                }
		    }
#endregion
            #else

            #region Compute

            //#n is input, #10n is output
            switch (InfoOf<TIn>.NPTypeCode)
            {
                case NPTypeCode.Boolean:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<bool, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<bool, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<bool, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<bool, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<bool, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<bool, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<bool, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<bool, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<bool, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<bool, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<bool, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<bool, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Byte:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<byte, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<byte, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<byte, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<byte, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<byte, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<byte, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<byte, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<byte, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<byte, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<byte, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<byte, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<byte, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Int16:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<short, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<short, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<short, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<short, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<short, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<short, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<short, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<short, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<short, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<short, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<short, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<short, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.UInt16:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<ushort, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<ushort, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<ushort, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<ushort, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<ushort, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<ushort, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<ushort, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<ushort, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<ushort, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<ushort, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<ushort, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<ushort, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Int32:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<int, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<int, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<int, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<int, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<int, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<int, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<int, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<int, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<int, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<int, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<int, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<int, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.UInt32:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<uint, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<uint, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<uint, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<uint, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<uint, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<uint, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<uint, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<uint, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<uint, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<uint, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<uint, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<uint, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Int64:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<long, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<long, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<long, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<long, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<long, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<long, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<long, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<long, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<long, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<long, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<long, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<long, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.UInt64:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<ulong, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<ulong, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<ulong, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<ulong, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<ulong, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<ulong, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<ulong, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<ulong, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<ulong, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<ulong, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<ulong, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<ulong, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Char:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<char, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<char, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<char, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<char, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<char, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<char, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<char, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<char, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<char, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<char, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<char, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<char, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Double:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<double, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<double, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<double, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<double, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<double, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<double, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<double, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<double, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<double, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<double, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<double, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<double, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Single:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<float, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<float, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<float, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<float, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<float, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<float, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<float, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<float, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<float, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<float, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<float, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<float, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                case NPTypeCode.Decimal:
                    {
                        switch (InfoOf<TOut>.NPTypeCode)
                        {
                            case NPTypeCode.Boolean:
                                {
                                    Func<decimal, bool> ret = Converts.ToBoolean;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Byte:
                                {
                                    Func<decimal, byte> ret = Converts.ToByte;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int16:
                                {
                                    Func<decimal, short> ret = Converts.ToInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt16:
                                {
                                    Func<decimal, ushort> ret = Converts.ToUInt16;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int32:
                                {
                                    Func<decimal, int> ret = Converts.ToInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt32:
                                {
                                    Func<decimal, uint> ret = Converts.ToUInt32;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Int64:
                                {
                                    Func<decimal, long> ret = Converts.ToInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.UInt64:
                                {
                                    Func<decimal, ulong> ret = Converts.ToUInt64;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Char:
                                {
                                    Func<decimal, char> ret = Converts.ToChar;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Double:
                                {
                                    Func<decimal, double> ret = Converts.ToDouble;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Single:
                                {
                                    Func<decimal, float> ret = Converts.ToSingle;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            case NPTypeCode.Decimal:
                                {
                                    Func<decimal, decimal> ret = Converts.ToDecimal;
                                    return (Func<TIn, TOut>)(object)ret;
                                }
                            default:
                                {
                                    var tout = typeof(TOut);
                                    return @in => (TOut)Convert.ChangeType(@in, tout);
                                }
                        }
                    }
                default:
                    {
                        var tout = typeof(TOut);
                        return @in => (TOut)Convert.ChangeType(@in, tout);
                    }
            }

            #endregion

            #endif
        }

        #region ToScalar

        #if _REGEN
#region Compute

		%foreach supported_dtypes,supported_dtypes_lowercase%
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static #2 To#1(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.#1 ? nd.GetAtIndex<#2>(0) : Converts.To#1(nd.GetAtIndex(0));
        }
		%
			    
#endregion
        #else

        #region Compute

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool ToBoolean(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Boolean ? nd.GetAtIndex<bool>(0) : Converts.ToBoolean(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ToByte(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Byte ? nd.GetAtIndex<byte>(0) : Converts.ToByte(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static short ToInt16(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int16 ? nd.GetAtIndex<short>(0) : Converts.ToInt16(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ushort ToUInt16(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.UInt16 ? nd.GetAtIndex<ushort>(0) : Converts.ToUInt16(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ToInt32(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int32 ? nd.GetAtIndex<int>(0) : Converts.ToInt32(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint ToUInt32(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.UInt32 ? nd.GetAtIndex<uint>(0) : Converts.ToUInt32(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long ToInt64(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Int64 ? nd.GetAtIndex<long>(0) : Converts.ToInt64(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong ToUInt64(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.UInt64 ? nd.GetAtIndex<ulong>(0) : Converts.ToUInt64(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static char ToChar(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Char ? nd.GetAtIndex<char>(0) : Converts.ToChar(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double ToDouble(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Double ? nd.GetAtIndex<double>(0) : Converts.ToDouble(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ToSingle(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Single ? nd.GetAtIndex<float>(0) : Converts.ToSingle(nd.GetAtIndex(0));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static decimal ToDecimal(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return nd.typecode == NPTypeCode.Decimal ? nd.GetAtIndex<decimal>(0) : Converts.ToDecimal(nd.GetAtIndex(0));
        }

        #endregion

        #endif

        #endregion
    }
}
