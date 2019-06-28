using System;
using System.Runtime.CompilerServices;

#if _REGEN_GLOBAL
    %supportedTypes_Primitives = ["Boolean","Byte","Int16","UInt16","Int32","UInt32","Int64","UInt64","Char","Double","Single","Decimal","String"]
    %supportTypesLower_Primitives = ["bool","byte","short","ushort","int","uint","long","ulong","char","double","float","decimal","string"]
#endif
namespace NumSharp.Utilities
{
    /// <summary>
    ///     Provides a way to convert boxed object from known time to specific type.
    /// </summary>
    public static class NonGenericConvert
    {
#if _REGEN
        %foreach forevery(supportedTypes_Primitives, supportedTypes_Primitives, true)%
        /// <summary>
        ///     Convert from #1 to #2 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="#1"/> and then converted to <see cref="#2"/></param>
        /// <returns>#2</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static #2 From#1To#2(object input)
        {
            return Convert.To#2((#1)input);
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
            return Convert.ToByte((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromBooleanToInt16(object input)
        {
            return Convert.ToInt16((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromBooleanToUInt16(object input)
        {
            return Convert.ToUInt16((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromBooleanToInt32(object input)
        {
            return Convert.ToInt32((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromBooleanToUInt32(object input)
        {
            return Convert.ToUInt32((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromBooleanToInt64(object input)
        {
            return Convert.ToInt64((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromBooleanToUInt64(object input)
        {
            return Convert.ToUInt64((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromBooleanToChar(object input)
        {
            return Convert.ToChar((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromBooleanToDouble(object input)
        {
            return Convert.ToDouble((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromBooleanToSingle(object input)
        {
            return Convert.ToSingle((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromBooleanToDecimal(object input)
        {
            return Convert.ToDecimal((Boolean)input);
        }

        /// <summary>
        ///     Convert from Boolean to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Boolean"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromBooleanToString(object input)
        {
            return Convert.ToString((Boolean)input);
        }

        /// <summary>
        ///     Convert from Byte to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromByteToBoolean(object input)
        {
            return Convert.ToBoolean((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromByteToInt16(object input)
        {
            return Convert.ToInt16((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromByteToUInt16(object input)
        {
            return Convert.ToUInt16((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromByteToInt32(object input)
        {
            return Convert.ToInt32((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromByteToUInt32(object input)
        {
            return Convert.ToUInt32((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromByteToInt64(object input)
        {
            return Convert.ToInt64((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromByteToUInt64(object input)
        {
            return Convert.ToUInt64((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromByteToChar(object input)
        {
            return Convert.ToChar((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromByteToDouble(object input)
        {
            return Convert.ToDouble((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromByteToSingle(object input)
        {
            return Convert.ToSingle((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromByteToDecimal(object input)
        {
            return Convert.ToDecimal((Byte)input);
        }

        /// <summary>
        ///     Convert from Byte to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Byte"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromByteToString(object input)
        {
            return Convert.ToString((Byte)input);
        }

        /// <summary>
        ///     Convert from Int16 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromInt16ToBoolean(object input)
        {
            return Convert.ToBoolean((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromInt16ToByte(object input)
        {
            return Convert.ToByte((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromInt16ToUInt16(object input)
        {
            return Convert.ToUInt16((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromInt16ToInt32(object input)
        {
            return Convert.ToInt32((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromInt16ToUInt32(object input)
        {
            return Convert.ToUInt32((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromInt16ToInt64(object input)
        {
            return Convert.ToInt64((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromInt16ToUInt64(object input)
        {
            return Convert.ToUInt64((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromInt16ToChar(object input)
        {
            return Convert.ToChar((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromInt16ToDouble(object input)
        {
            return Convert.ToDouble((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromInt16ToSingle(object input)
        {
            return Convert.ToSingle((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromInt16ToDecimal(object input)
        {
            return Convert.ToDecimal((Int16)input);
        }

        /// <summary>
        ///     Convert from Int16 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int16"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromInt16ToString(object input)
        {
            return Convert.ToString((Int16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromUInt16ToBoolean(object input)
        {
            return Convert.ToBoolean((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromUInt16ToByte(object input)
        {
            return Convert.ToByte((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromUInt16ToInt16(object input)
        {
            return Convert.ToInt16((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromUInt16ToInt32(object input)
        {
            return Convert.ToInt32((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromUInt16ToUInt32(object input)
        {
            return Convert.ToUInt32((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromUInt16ToInt64(object input)
        {
            return Convert.ToInt64((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromUInt16ToUInt64(object input)
        {
            return Convert.ToUInt64((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromUInt16ToChar(object input)
        {
            return Convert.ToChar((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromUInt16ToDouble(object input)
        {
            return Convert.ToDouble((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromUInt16ToSingle(object input)
        {
            return Convert.ToSingle((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromUInt16ToDecimal(object input)
        {
            return Convert.ToDecimal((UInt16)input);
        }

        /// <summary>
        ///     Convert from UInt16 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt16"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromUInt16ToString(object input)
        {
            return Convert.ToString((UInt16)input);
        }

        /// <summary>
        ///     Convert from Int32 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromInt32ToBoolean(object input)
        {
            return Convert.ToBoolean((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromInt32ToByte(object input)
        {
            return Convert.ToByte((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromInt32ToInt16(object input)
        {
            return Convert.ToInt16((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromInt32ToUInt16(object input)
        {
            return Convert.ToUInt16((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromInt32ToUInt32(object input)
        {
            return Convert.ToUInt32((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromInt32ToInt64(object input)
        {
            return Convert.ToInt64((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromInt32ToUInt64(object input)
        {
            return Convert.ToUInt64((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromInt32ToChar(object input)
        {
            return Convert.ToChar((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromInt32ToDouble(object input)
        {
            return Convert.ToDouble((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromInt32ToSingle(object input)
        {
            return Convert.ToSingle((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromInt32ToDecimal(object input)
        {
            return Convert.ToDecimal((Int32)input);
        }

        /// <summary>
        ///     Convert from Int32 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int32"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromInt32ToString(object input)
        {
            return Convert.ToString((Int32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromUInt32ToBoolean(object input)
        {
            return Convert.ToBoolean((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromUInt32ToByte(object input)
        {
            return Convert.ToByte((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromUInt32ToInt16(object input)
        {
            return Convert.ToInt16((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromUInt32ToUInt16(object input)
        {
            return Convert.ToUInt16((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromUInt32ToInt32(object input)
        {
            return Convert.ToInt32((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromUInt32ToInt64(object input)
        {
            return Convert.ToInt64((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromUInt32ToUInt64(object input)
        {
            return Convert.ToUInt64((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromUInt32ToChar(object input)
        {
            return Convert.ToChar((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromUInt32ToDouble(object input)
        {
            return Convert.ToDouble((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromUInt32ToSingle(object input)
        {
            return Convert.ToSingle((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromUInt32ToDecimal(object input)
        {
            return Convert.ToDecimal((UInt32)input);
        }

        /// <summary>
        ///     Convert from UInt32 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt32"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromUInt32ToString(object input)
        {
            return Convert.ToString((UInt32)input);
        }

        /// <summary>
        ///     Convert from Int64 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromInt64ToBoolean(object input)
        {
            return Convert.ToBoolean((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromInt64ToByte(object input)
        {
            return Convert.ToByte((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromInt64ToInt16(object input)
        {
            return Convert.ToInt16((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromInt64ToUInt16(object input)
        {
            return Convert.ToUInt16((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromInt64ToInt32(object input)
        {
            return Convert.ToInt32((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromInt64ToUInt32(object input)
        {
            return Convert.ToUInt32((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromInt64ToUInt64(object input)
        {
            return Convert.ToUInt64((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromInt64ToChar(object input)
        {
            return Convert.ToChar((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromInt64ToDouble(object input)
        {
            return Convert.ToDouble((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromInt64ToSingle(object input)
        {
            return Convert.ToSingle((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromInt64ToDecimal(object input)
        {
            return Convert.ToDecimal((Int64)input);
        }

        /// <summary>
        ///     Convert from Int64 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Int64"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromInt64ToString(object input)
        {
            return Convert.ToString((Int64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromUInt64ToBoolean(object input)
        {
            return Convert.ToBoolean((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromUInt64ToByte(object input)
        {
            return Convert.ToByte((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromUInt64ToInt16(object input)
        {
            return Convert.ToInt16((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromUInt64ToUInt16(object input)
        {
            return Convert.ToUInt16((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromUInt64ToInt32(object input)
        {
            return Convert.ToInt32((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromUInt64ToUInt32(object input)
        {
            return Convert.ToUInt32((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromUInt64ToInt64(object input)
        {
            return Convert.ToInt64((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromUInt64ToChar(object input)
        {
            return Convert.ToChar((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromUInt64ToDouble(object input)
        {
            return Convert.ToDouble((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromUInt64ToSingle(object input)
        {
            return Convert.ToSingle((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromUInt64ToDecimal(object input)
        {
            return Convert.ToDecimal((UInt64)input);
        }

        /// <summary>
        ///     Convert from UInt64 to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="UInt64"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromUInt64ToString(object input)
        {
            return Convert.ToString((UInt64)input);
        }

        /// <summary>
        ///     Convert from Char to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromCharToBoolean(object input)
        {
            return Convert.ToBoolean((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromCharToByte(object input)
        {
            return Convert.ToByte((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromCharToInt16(object input)
        {
            return Convert.ToInt16((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromCharToUInt16(object input)
        {
            return Convert.ToUInt16((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromCharToInt32(object input)
        {
            return Convert.ToInt32((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromCharToUInt32(object input)
        {
            return Convert.ToUInt32((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromCharToInt64(object input)
        {
            return Convert.ToInt64((Char)input);
        }

        /// <summary>
        ///     Convert from Char to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromCharToUInt64(object input)
        {
            return Convert.ToUInt64((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromCharToDouble(object input)
        {
            return Convert.ToDouble((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromCharToSingle(object input)
        {
            return Convert.ToSingle((Char)input);
        }

        /// <summary>
        ///     Convert from Char to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromCharToDecimal(object input)
        {
            return Convert.ToDecimal((Char)input);
        }

        /// <summary>
        ///     Convert from Char to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Char"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromCharToString(object input)
        {
            return Convert.ToString((Char)input);
        }

        /// <summary>
        ///     Convert from Double to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromDoubleToBoolean(object input)
        {
            return Convert.ToBoolean((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromDoubleToByte(object input)
        {
            return Convert.ToByte((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromDoubleToInt16(object input)
        {
            return Convert.ToInt16((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromDoubleToUInt16(object input)
        {
            return Convert.ToUInt16((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromDoubleToInt32(object input)
        {
            return Convert.ToInt32((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromDoubleToUInt32(object input)
        {
            return Convert.ToUInt32((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromDoubleToInt64(object input)
        {
            return Convert.ToInt64((Double)input);
        }

        /// <summary>
        ///     Convert from Double to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromDoubleToUInt64(object input)
        {
            return Convert.ToUInt64((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromDoubleToChar(object input)
        {
            return Convert.ToChar((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromDoubleToSingle(object input)
        {
            return Convert.ToSingle((Double)input);
        }

        /// <summary>
        ///     Convert from Double to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromDoubleToDecimal(object input)
        {
            return Convert.ToDecimal((Double)input);
        }

        /// <summary>
        ///     Convert from Double to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Double"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromDoubleToString(object input)
        {
            return Convert.ToString((Double)input);
        }

        /// <summary>
        ///     Convert from Single to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromSingleToBoolean(object input)
        {
            return Convert.ToBoolean((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromSingleToByte(object input)
        {
            return Convert.ToByte((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromSingleToInt16(object input)
        {
            return Convert.ToInt16((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromSingleToUInt16(object input)
        {
            return Convert.ToUInt16((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromSingleToInt32(object input)
        {
            return Convert.ToInt32((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromSingleToUInt32(object input)
        {
            return Convert.ToUInt32((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromSingleToInt64(object input)
        {
            return Convert.ToInt64((Single)input);
        }

        /// <summary>
        ///     Convert from Single to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromSingleToUInt64(object input)
        {
            return Convert.ToUInt64((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromSingleToChar(object input)
        {
            return Convert.ToChar((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromSingleToDouble(object input)
        {
            return Convert.ToDouble((Single)input);
        }

        /// <summary>
        ///     Convert from Single to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromSingleToDecimal(object input)
        {
            return Convert.ToDecimal((Single)input);
        }

        /// <summary>
        ///     Convert from Single to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Single"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromSingleToString(object input)
        {
            return Convert.ToString((Single)input);
        }

        /// <summary>
        ///     Convert from Decimal to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromDecimalToBoolean(object input)
        {
            return Convert.ToBoolean((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromDecimalToByte(object input)
        {
            return Convert.ToByte((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromDecimalToInt16(object input)
        {
            return Convert.ToInt16((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromDecimalToUInt16(object input)
        {
            return Convert.ToUInt16((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromDecimalToInt32(object input)
        {
            return Convert.ToInt32((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromDecimalToUInt32(object input)
        {
            return Convert.ToUInt32((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromDecimalToInt64(object input)
        {
            return Convert.ToInt64((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromDecimalToUInt64(object input)
        {
            return Convert.ToUInt64((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromDecimalToChar(object input)
        {
            return Convert.ToChar((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromDecimalToDouble(object input)
        {
            return Convert.ToDouble((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromDecimalToSingle(object input)
        {
            return Convert.ToSingle((Decimal)input);
        }

        /// <summary>
        ///     Convert from Decimal to String when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="Decimal"/> and then converted to <see cref="String"/></param>
        /// <returns>String</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static String FromDecimalToString(object input)
        {
            return Convert.ToString((Decimal)input);
        }

        /// <summary>
        ///     Convert from String to Boolean when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Boolean"/></param>
        /// <returns>Boolean</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Boolean FromStringToBoolean(object input)
        {
            return Convert.ToBoolean((String)input);
        }

        /// <summary>
        ///     Convert from String to Byte when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Byte"/></param>
        /// <returns>Byte</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Byte FromStringToByte(object input)
        {
            return Convert.ToByte((String)input);
        }

        /// <summary>
        ///     Convert from String to Int16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int16"/></param>
        /// <returns>Int16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int16 FromStringToInt16(object input)
        {
            return Convert.ToInt16((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt16 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt16"/></param>
        /// <returns>UInt16</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt16 FromStringToUInt16(object input)
        {
            return Convert.ToUInt16((String)input);
        }

        /// <summary>
        ///     Convert from String to Int32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int32"/></param>
        /// <returns>Int32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int32 FromStringToInt32(object input)
        {
            return Convert.ToInt32((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt32 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt32"/></param>
        /// <returns>UInt32</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt32 FromStringToUInt32(object input)
        {
            return Convert.ToUInt32((String)input);
        }

        /// <summary>
        ///     Convert from String to Int64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Int64"/></param>
        /// <returns>Int64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Int64 FromStringToInt64(object input)
        {
            return Convert.ToInt64((String)input);
        }

        /// <summary>
        ///     Convert from String to UInt64 when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="UInt64"/></param>
        /// <returns>UInt64</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static UInt64 FromStringToUInt64(object input)
        {
            return Convert.ToUInt64((String)input);
        }

        /// <summary>
        ///     Convert from String to Char when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Char"/></param>
        /// <returns>Char</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Char FromStringToChar(object input)
        {
            return Convert.ToChar((String)input);
        }

        /// <summary>
        ///     Convert from String to Double when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Double"/></param>
        /// <returns>Double</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Double FromStringToDouble(object input)
        {
            return Convert.ToDouble((String)input);
        }

        /// <summary>
        ///     Convert from String to Single when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Single"/></param>
        /// <returns>Single</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Single FromStringToSingle(object input)
        {
            return Convert.ToSingle((String)input);
        }

        /// <summary>
        ///     Convert from String to Decimal when input is a boxed non-generic <see cref="object"/>.
        /// </summary>
        /// <param name="input">The object that will be casted to <see cref="String"/> and then converted to <see cref="Decimal"/></param>
        /// <returns>Decimal</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Decimal FromStringToDecimal(object input)
        {
            return Convert.ToDecimal((String)input);
        }
#endif
    }
}
