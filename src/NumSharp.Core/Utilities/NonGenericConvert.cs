using System;
using System.Runtime.CompilerServices;

namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides a way to convert boxed object from known time to specific type.
    /// </summary>
    public static class NonGenericConvert
    {
#if _REGEN
        %foreach forevery(supported_primitives, supported_primitives, true)%
        /// <summary>
        ///     Convert from #1 to #2 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="#1"/> and then converted to <see cref="#2"/></param>
        /// <returns>#2</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static #2 From#1To#2(object input)
        {
            return Converts.To#2((#1)input);
        }

        %
#else

        /// <summary>
        ///     Convert from Boolean to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromBooleanToByte(object input)
        {
            return Converts.ToByte((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromBooleanToInt16(object input)
        {
            return Converts.ToInt16((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromBooleanToUInt16(object input)
        {
            return Converts.ToUInt16((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromBooleanToInt32(object input)
        {
            return Converts.ToInt32((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromBooleanToUInt32(object input)
        {
            return Converts.ToUInt32((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromBooleanToInt64(object input)
        {
            return Converts.ToInt64((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromBooleanToUInt64(object input)
        {
            return Converts.ToUInt64((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromBooleanToChar(object input)
        {
            return Converts.ToChar((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromBooleanToDouble(object input)
        {
            return Converts.ToDouble((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromBooleanToSingle(object input)
        {
            return Converts.ToSingle((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromBooleanToDecimal(object input)
        {
            return Converts.ToDecimal((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromBooleanToString(object input)
        {
            return Converts.ToString((Boolean)input);
        }

        /// <summary>
        ///     Convert from Byte to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromByteToBoolean(object input)
        {
            return Converts.ToBoolean((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromByteToInt16(object input)
        {
            return Converts.ToInt16((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromByteToUInt16(object input)
        {
            return Converts.ToUInt16((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromByteToInt32(object input)
        {
            return Converts.ToInt32((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromByteToUInt32(object input)
        {
            return Converts.ToUInt32((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromByteToInt64(object input)
        {
            return Converts.ToInt64((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromByteToUInt64(object input)
        {
            return Converts.ToUInt64((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromByteToChar(object input)
        {
            return Converts.ToChar((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromByteToDouble(object input)
        {
            return Converts.ToDouble((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromByteToSingle(object input)
        {
            return Converts.ToSingle((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromByteToDecimal(object input)
        {
            return Converts.ToDecimal((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromByteToString(object input)
        {
            return Converts.ToString((Byte)input);
        }

        /// <summary>
        ///     Convert from Int16 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromInt16ToBoolean(object input)
        {
            return Converts.ToBoolean((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromInt16ToByte(object input)
        {
            return Converts.ToByte((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromInt16ToUInt16(object input)
        {
            return Converts.ToUInt16((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromInt16ToInt32(object input)
        {
            return Converts.ToInt32((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromInt16ToUInt32(object input)
        {
            return Converts.ToUInt32((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromInt16ToInt64(object input)
        {
            return Converts.ToInt64((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromInt16ToUInt64(object input)
        {
            return Converts.ToUInt64((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromInt16ToChar(object input)
        {
            return Converts.ToChar((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromInt16ToDouble(object input)
        {
            return Converts.ToDouble((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromInt16ToSingle(object input)
        {
            return Converts.ToSingle((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromInt16ToDecimal(object input)
        {
            return Converts.ToDecimal((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromInt16ToString(object input)
        {
            return Converts.ToString((Int16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromUInt16ToBoolean(object input)
        {
            return Converts.ToBoolean((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromUInt16ToByte(object input)
        {
            return Converts.ToByte((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromUInt16ToInt16(object input)
        {
            return Converts.ToInt16((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromUInt16ToInt32(object input)
        {
            return Converts.ToInt32((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromUInt16ToUInt32(object input)
        {
            return Converts.ToUInt32((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromUInt16ToInt64(object input)
        {
            return Converts.ToInt64((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromUInt16ToUInt64(object input)
        {
            return Converts.ToUInt64((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromUInt16ToChar(object input)
        {
            return Converts.ToChar((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromUInt16ToDouble(object input)
        {
            return Converts.ToDouble((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromUInt16ToSingle(object input)
        {
            return Converts.ToSingle((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromUInt16ToDecimal(object input)
        {
            return Converts.ToDecimal((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromUInt16ToString(object input)
        {
            return Converts.ToString((UInt16)input);
        }

        /// <summary>
        ///     Convert from Int32 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromInt32ToBoolean(object input)
        {
            return Converts.ToBoolean((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromInt32ToByte(object input)
        {
            return Converts.ToByte((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromInt32ToInt16(object input)
        {
            return Converts.ToInt16((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromInt32ToUInt16(object input)
        {
            return Converts.ToUInt16((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromInt32ToUInt32(object input)
        {
            return Converts.ToUInt32((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromInt32ToInt64(object input)
        {
            return Converts.ToInt64((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromInt32ToUInt64(object input)
        {
            return Converts.ToUInt64((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromInt32ToChar(object input)
        {
            return Converts.ToChar((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromInt32ToDouble(object input)
        {
            return Converts.ToDouble((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromInt32ToSingle(object input)
        {
            return Converts.ToSingle((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromInt32ToDecimal(object input)
        {
            return Converts.ToDecimal((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromInt32ToString(object input)
        {
            return Converts.ToString((Int32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromUInt32ToBoolean(object input)
        {
            return Converts.ToBoolean((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromUInt32ToByte(object input)
        {
            return Converts.ToByte((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromUInt32ToInt16(object input)
        {
            return Converts.ToInt16((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromUInt32ToUInt16(object input)
        {
            return Converts.ToUInt16((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromUInt32ToInt32(object input)
        {
            return Converts.ToInt32((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromUInt32ToInt64(object input)
        {
            return Converts.ToInt64((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromUInt32ToUInt64(object input)
        {
            return Converts.ToUInt64((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromUInt32ToChar(object input)
        {
            return Converts.ToChar((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromUInt32ToDouble(object input)
        {
            return Converts.ToDouble((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromUInt32ToSingle(object input)
        {
            return Converts.ToSingle((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromUInt32ToDecimal(object input)
        {
            return Converts.ToDecimal((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromUInt32ToString(object input)
        {
            return Converts.ToString((UInt32)input);
        }

        /// <summary>
        ///     Convert from Int64 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromInt64ToBoolean(object input)
        {
            return Converts.ToBoolean((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromInt64ToByte(object input)
        {
            return Converts.ToByte((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromInt64ToInt16(object input)
        {
            return Converts.ToInt16((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromInt64ToUInt16(object input)
        {
            return Converts.ToUInt16((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromInt64ToInt32(object input)
        {
            return Converts.ToInt32((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromInt64ToUInt32(object input)
        {
            return Converts.ToUInt32((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromInt64ToUInt64(object input)
        {
            return Converts.ToUInt64((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromInt64ToChar(object input)
        {
            return Converts.ToChar((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromInt64ToDouble(object input)
        {
            return Converts.ToDouble((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromInt64ToSingle(object input)
        {
            return Converts.ToSingle((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromInt64ToDecimal(object input)
        {
            return Converts.ToDecimal((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromInt64ToString(object input)
        {
            return Converts.ToString((Int64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromUInt64ToBoolean(object input)
        {
            return Converts.ToBoolean((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromUInt64ToByte(object input)
        {
            return Converts.ToByte((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromUInt64ToInt16(object input)
        {
            return Converts.ToInt16((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromUInt64ToUInt16(object input)
        {
            return Converts.ToUInt16((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromUInt64ToInt32(object input)
        {
            return Converts.ToInt32((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromUInt64ToUInt32(object input)
        {
            return Converts.ToUInt32((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromUInt64ToInt64(object input)
        {
            return Converts.ToInt64((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromUInt64ToChar(object input)
        {
            return Converts.ToChar((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromUInt64ToDouble(object input)
        {
            return Converts.ToDouble((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromUInt64ToSingle(object input)
        {
            return Converts.ToSingle((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromUInt64ToDecimal(object input)
        {
            return Converts.ToDecimal((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromUInt64ToString(object input)
        {
            return Converts.ToString((UInt64)input);
        }

        /// <summary>
        ///     Convert from Char to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromCharToBoolean(object input)
        {
            return Converts.ToBoolean((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromCharToByte(object input)
        {
            return Converts.ToByte((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromCharToInt16(object input)
        {
            return Converts.ToInt16((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromCharToUInt16(object input)
        {
            return Converts.ToUInt16((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromCharToInt32(object input)
        {
            return Converts.ToInt32((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromCharToUInt32(object input)
        {
            return Converts.ToUInt32((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromCharToInt64(object input)
        {
            return Converts.ToInt64((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromCharToUInt64(object input)
        {
            return Converts.ToUInt64((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromCharToDouble(object input)
        {
            return Converts.ToDouble((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromCharToSingle(object input)
        {
            return Converts.ToSingle((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromCharToDecimal(object input)
        {
            return Converts.ToDecimal((Char)input);
        }

        /// <summary>
        ///     Convert from Char to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromCharToString(object input)
        {
            return Converts.ToString((Char)input);
        }

        /// <summary>
        ///     Convert from Double to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromDoubleToBoolean(object input)
        {
            return Converts.ToBoolean((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromDoubleToByte(object input)
        {
            return Converts.ToByte((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromDoubleToInt16(object input)
        {
            return Converts.ToInt16((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromDoubleToUInt16(object input)
        {
            return Converts.ToUInt16((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromDoubleToInt32(object input)
        {
            return Converts.ToInt32((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromDoubleToUInt32(object input)
        {
            return Converts.ToUInt32((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromDoubleToInt64(object input)
        {
            return Converts.ToInt64((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromDoubleToUInt64(object input)
        {
            return Converts.ToUInt64((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromDoubleToChar(object input)
        {
            return Converts.ToChar((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromDoubleToSingle(object input)
        {
            return Converts.ToSingle((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromDoubleToDecimal(object input)
        {
            return Converts.ToDecimal((Double)input);
        }

        /// <summary>
        ///     Convert from Double to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromDoubleToString(object input)
        {
            return Converts.ToString((Double)input);
        }

        /// <summary>
        ///     Convert from Single to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromSingleToBoolean(object input)
        {
            return Converts.ToBoolean((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromSingleToByte(object input)
        {
            return Converts.ToByte((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromSingleToInt16(object input)
        {
            return Converts.ToInt16((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromSingleToUInt16(object input)
        {
            return Converts.ToUInt16((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromSingleToInt32(object input)
        {
            return Converts.ToInt32((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromSingleToUInt32(object input)
        {
            return Converts.ToUInt32((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromSingleToInt64(object input)
        {
            return Converts.ToInt64((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromSingleToUInt64(object input)
        {
            return Converts.ToUInt64((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromSingleToChar(object input)
        {
            return Converts.ToChar((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromSingleToDouble(object input)
        {
            return Converts.ToDouble((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromSingleToDecimal(object input)
        {
            return Converts.ToDecimal((Single)input);
        }

        /// <summary>
        ///     Convert from Single to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromSingleToString(object input)
        {
            return Converts.ToString((Single)input);
        }

        /// <summary>
        ///     Convert from Decimal to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromDecimalToBoolean(object input)
        {
            return Converts.ToBoolean((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromDecimalToByte(object input)
        {
            return Converts.ToByte((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromDecimalToInt16(object input)
        {
            return Converts.ToInt16((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromDecimalToUInt16(object input)
        {
            return Converts.ToUInt16((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromDecimalToInt32(object input)
        {
            return Converts.ToInt32((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromDecimalToUInt32(object input)
        {
            return Converts.ToUInt32((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromDecimalToInt64(object input)
        {
            return Converts.ToInt64((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromDecimalToUInt64(object input)
        {
            return Converts.ToUInt64((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromDecimalToChar(object input)
        {
            return Converts.ToChar((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromDecimalToDouble(object input)
        {
            return Converts.ToDouble((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromDecimalToSingle(object input)
        {
            return Converts.ToSingle((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromDecimalToString(object input)
        {
            return Converts.ToString((Decimal)input);
        }

        /// <summary>
        ///     Convert from String to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromStringToBoolean(object input)
        {
            return Converts.ToBoolean((String)input);
        }

        /// <summary>
        ///     Convert from String to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromStringToByte(object input)
        {
            return Converts.ToByte((String)input);
        }

        /// <summary>
        ///     Convert from String to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromStringToInt16(object input)
        {
            return Converts.ToInt16((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromStringToUInt16(object input)
        {
            return Converts.ToUInt16((String)input);
        }

        /// <summary>
        ///     Convert from String to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromStringToInt32(object input)
        {
            return Converts.ToInt32((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromStringToUInt32(object input)
        {
            return Converts.ToUInt32((String)input);
        }

        /// <summary>
        ///     Convert from String to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromStringToInt64(object input)
        {
            return Converts.ToInt64((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromStringToUInt64(object input)
        {
            return Converts.ToUInt64((String)input);
        }

        /// <summary>
        ///     Convert from String to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromStringToChar(object input)
        {
            return Converts.ToChar((String)input);
        }

        /// <summary>
        ///     Convert from String to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromStringToDouble(object input)
        {
            return Converts.ToDouble((String)input);
        }

        /// <summary>
        ///     Convert from String to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromStringToSingle(object input)
        {
            return Converts.ToSingle((String)input);
        }

        /// <summary>
        ///     Convert from String to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromStringToDecimal(object input)
        {
            return Converts.ToDecimal((String)input);
        }
#endif
    }
}
